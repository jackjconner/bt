"""End-to-end production pipeline over the synthetic datasets.

Ties the per-module production features together on one coherent, schema-valid
dataset: generate → load (with validation) → signal research → model CV →
portfolio construction → backtest with costs → performance analytics →
profiling. This is the integration that proves the modules compose; the
parameterized scaling experiment in ``main.py`` is kept separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from analysis import BacktestAnalyzerImpl, sharpe
from backtest import ProductionBacktestConfig, ProductionBacktestEngine, SignalFrame
from etl import DatasetLoader, to_matrix
from etl.datasets import GenSpec, write_all
from etl.source import to_float
from models import (
    ModelConfig,
    RidgeModel,
    WalkForwardSplitter,
    build_panel,
    walk_forward_cv,
)
from portfolio import (
    build_from_long,
    constraints_from_polars,
    ledoit_wolf_cov,
    mean_variance,
    tracking_error,
)
from profiling import capture_environment, fit_scaling, run_trials
from signals import ic_horizon_curve, ic_series_v2, neutralize_sector
from strategy import StrategySpec


@dataclass(frozen=True)
class PipelineSummary:
    ic_raw: float
    ic_neutralized: float
    horizon_ic: dict[int, float]
    wf_mean_ic: float
    wf_mean_r2: float
    opt_converged: bool
    opt_gross: float
    factor_vol: float
    tracking_error: float
    gross_sharpe: float
    net_sharpe: float
    cost_drag: float
    n_scaling_fits: int
    backtest_p50_s: float


def _returns_from_prices(prices: pl.DataFrame) -> pl.DataFrame:
    """Percent returns implied by the price panel — guarantees the returns
    align to the same session axis as every other dataset (the raw
    ``generate_returns`` uses a calendar-day axis)."""
    return (
        prices.sort(["id", "date"])
        .with_columns(
            ((pl.col("close") / pl.col("close").shift(1).over("id") - 1.0) * 100.0)
            .fill_null(0.0)
            .alias("return")
        )
        .select("date", "id", "return")
    )


def run_production_pipeline(
    spec: GenSpec,
    workdir: Path,
    strategy: StrategySpec | None = None,
) -> PipelineSummary:
    """Run the end-to-end production pipeline.

    When ``strategy`` is ``None`` the pipeline uses the same hardcoded
    defaults that existed before ``StrategySpec`` was introduced — output is
    byte-for-byte identical to the pre-spec behaviour.  Pass a
    ``StrategySpec`` to control every tunable knob explicitly.
    """
    # Resolve knobs: when no strategy is supplied, fall through to the exact
    # literals that were hardcoded before this parameter existed.
    if strategy is not None:
        wf_n_splits = strategy.wf_n_splits
        wf_embargo = strategy.wf_embargo_periods
        opt_risk_aversion = strategy.opt_risk_aversion
        opt_max_iter = strategy.opt_max_iter
        bt_max_weight = strategy.bt_max_weight
        bt_enable_umask = strategy.bt_enable_universe_mask
        bt_enable_costs = strategy.bt_enable_costs
        bt_enable_slippage = strategy.bt_enable_slippage
        profile_trials = strategy.profile_trials
        profile_warmup = strategy.profile_warmup
        bt_profile_trials = strategy.bt_profile_trials
        bt_profile_warmup = strategy.bt_profile_warmup
        neutralize_sectors = strategy.neutralize_sectors
        horizon_map = strategy.horizon_map
    else:
        wf_n_splits = 4
        wf_embargo = 5
        opt_risk_aversion = 1.0
        opt_max_iter = 3000
        bt_max_weight = 0.1
        bt_enable_umask = True
        bt_enable_costs = True
        bt_enable_slippage = True
        profile_trials = 3
        profile_warmup = 1
        bt_profile_trials = 3
        bt_profile_warmup = 1
        neutralize_sectors = True
        horizon_map = {1: "fwd_ret_1", 5: "fwd_ret_5", 21: "fwd_ret_21", 63: "fwd_ret_63"}

    write_all(workdir, spec)
    loader = DatasetLoader(workdir, spec)

    prices = loader.load("prices")
    returns = _returns_from_prices(prices)
    tcosts = loader.load("transaction_costs")
    umask = loader.load("universe_mask")
    sec_master = loader.load("security_master")
    fwd = loader.load("forward_returns")

    momentum = (
        loader.load("alpha_signals")
        .filter(pl.col("signal_name") == "momentum")
        .select("date", "id", "signal")
    )

    # --- signals: IC, neutralization, horizon decay ---------------------- #
    ic_raw = to_float(ic_series_v2(momentum, fwd, return_col="fwd_ret_1")["ic"].mean())
    if neutralize_sectors:
        neutral = neutralize_sector(momentum, sec_master)
        ic_neut = to_float(ic_series_v2(neutral, fwd, return_col="fwd_ret_1")["ic"].mean())
    else:
        ic_neut = ic_raw
    curve = ic_horizon_curve(momentum, fwd, horizon_map)
    horizon_ic = {p.horizon: float(p.mean_ic) for p in curve.points}

    # --- models: purged walk-forward CV on the feature panel ------------- #
    panel = build_panel(
        loader.load("feature_panel"),
        fwd,
        "fwd_ret_1",
        weights=loader.load("sample_weights"),
    )
    wf = walk_forward_cv(
        panel,
        WalkForwardSplitter(n_splits=wf_n_splits, embargo_periods=wf_embargo),
        lambda alpha: RidgeModel(ModelConfig(n_features=spec.n_features, alpha=alpha)),
    )

    # --- portfolio: factor risk model + mean-variance optimize ----------- #
    as_of = cast(date, prices["date"].max())
    risk_model = build_from_long(
        loader.load("factor_loadings"),
        loader.load("factor_covariance"),
        loader.load("specific_risk"),
        as_of,
    )
    R, _ = to_matrix(returns, "return")
    cov = ledoit_wolf_cov(R)
    raw_alpha = momentum.filter(pl.col("date") == as_of).sort("id")["signal"].to_numpy()
    alpha_vec = (raw_alpha - raw_alpha.mean()) / (raw_alpha.std() + 1e-9)
    cspec = constraints_from_polars(
        loader.load("position_constraints"),
        loader.load("group_constraints"),
        sec_master,
        spec.n_assets,
        long_only=False,
        net_exposure=1.0,
    )
    opt = mean_variance(
        alpha_vec, cov, cspec, risk_aversion=opt_risk_aversion, max_iter=opt_max_iter
    )
    factor_vol = float(np.sqrt(max(risk_model.portfolio_variance(opt.weights), 0.0)))
    bench_w = np.full(spec.n_assets, 1.0 / spec.n_assets)
    te = tracking_error(opt.weights, bench_w, cov)

    # --- backtest: gross vs net of costs --------------------------------- #
    signals = SignalFrame(df=momentum, is_categorical=False)
    gross_cfg = ProductionBacktestConfig(
        n_assets=spec.n_assets,
        n_dates=spec.n_dates,
        enable_universe_mask=bt_enable_umask,
        max_weight=bt_max_weight,
    )
    net_cfg = ProductionBacktestConfig(
        n_assets=spec.n_assets,
        n_dates=spec.n_dates,
        enable_universe_mask=bt_enable_umask,
        enable_costs=bt_enable_costs,
        enable_slippage=bt_enable_slippage,
        max_weight=bt_max_weight,
    )
    gross = ProductionBacktestEngine(gross_cfg).run(
        returns, signals, prices=prices, transaction_costs=tcosts, universe_mask=umask
    )
    net = ProductionBacktestEngine(net_cfg).run(
        returns, signals, prices=prices, transaction_costs=tcosts, universe_mask=umask
    )

    analyzer = BacktestAnalyzerImpl()
    gross_a = analyzer.analyze(gross)
    net_a = analyzer.analyze(net)
    gross_sharpe = sharpe(gross_a.returns_series)
    net_sharpe = sharpe(net_a.returns_series)
    cost_drag = float(gross.nav_history["nav"][-1] - net.nav_history["nav"][-1])

    # --- profiling: trials + scaling fit --------------------------------- #
    capture_environment("pipeline_run", trials=profile_trials, warmup_trials=profile_warmup)
    trial = run_trials(
        "backtest",
        lambda: ProductionBacktestEngine(net_cfg).run(
            returns, signals, prices=prices, transaction_costs=tcosts, universe_mask=umask
        ),
        lambda r: {"nav": r.nav_history},
        n_trials=bt_profile_trials,
        warmup=bt_profile_warmup,
    )
    fits = fit_scaling(loader.load("stage_measurements"), run_id="run_0000")

    return PipelineSummary(
        ic_raw=ic_raw,
        ic_neutralized=ic_neut,
        horizon_ic=horizon_ic,
        wf_mean_ic=float(wf.mean_ic),
        wf_mean_r2=float(wf.mean_r2),
        opt_converged=bool(opt.converged),
        opt_gross=float(np.abs(opt.weights).sum()),
        factor_vol=factor_vol,
        tracking_error=float(te),
        gross_sharpe=gross_sharpe,
        net_sharpe=net_sharpe,
        cost_drag=cost_drag,
        n_scaling_fits=len(fits),
        backtest_p50_s=float(trial.elapsed_p50),
    )


def run_from_spec(strategy: StrategySpec, workdir: Path) -> PipelineSummary:
    """Convenience wrapper: run the pipeline fully from a ``StrategySpec``.

    Equivalent to ``run_production_pipeline(strategy.gen, workdir, strategy)``.
    """
    return run_production_pipeline(strategy.gen, workdir, strategy)


def print_pipeline_summary(s: PipelineSummary) -> None:
    print(f"\n{'=' * 78}")
    print("PRODUCTION PIPELINE")
    print(f"{'=' * 78}")
    print(f"  signal IC (raw / sector-neutral):   {s.ic_raw:+.4f} / {s.ic_neutralized:+.4f}")
    decay = "  ".join(f"{h}d={ic:+.3f}" for h, ic in sorted(s.horizon_ic.items()))
    print(f"  IC horizon decay:                   {decay}")
    print(f"  walk-forward CV (mean IC / R²):     {s.wf_mean_ic:+.4f} / {s.wf_mean_r2:+.4f}")
    print(f"  optimizer (converged / gross):      {s.opt_converged} / {s.opt_gross:.2f}")
    print(f"  factor-model vol / tracking error:  {s.factor_vol:.4f} / {s.tracking_error:.4f}")
    print(f"  Sharpe gross / net:                 {s.gross_sharpe:+.3f} / {s.net_sharpe:+.3f}")
    print(f"  cost drag (final NAV gross-net):    {s.cost_drag:,.0f}")
    print(
        f"  scaling fits / backtest p50:        "
        f"{s.n_scaling_fits} / {s.backtest_p50_s * 1000:.1f} ms"
    )


__all__ = [
    "PipelineSummary",
    "print_pipeline_summary",
    "run_from_spec",
    "run_production_pipeline",
]
