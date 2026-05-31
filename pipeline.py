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

from analysis import BacktestAnalyzerImpl, cagr, max_drawdown, returns_from_nav, sharpe
from backtest import ProductionBacktestConfig, ProductionBacktestEngine, SignalFrame
from etl import DatasetLoader, to_matrix
from etl.datasets import GenSpec, write_all
from etl.source import to_float
from models import (
    ModelConfig,
    RidgeModel,
    WalkForwardConfig,
    WalkForwardSplitter,
    WFResult,
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


def run_production_pipeline(spec: GenSpec, workdir: Path) -> PipelineSummary:
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
    neutral = neutralize_sector(momentum, sec_master)
    ic_neut = to_float(ic_series_v2(neutral, fwd, return_col="fwd_ret_1")["ic"].mean())
    curve = ic_horizon_curve(
        momentum,
        fwd,
        {1: "fwd_ret_1", 5: "fwd_ret_5", 21: "fwd_ret_21", 63: "fwd_ret_63"},
    )
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
        WalkForwardSplitter(n_splits=4, embargo_periods=5),
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
    opt = mean_variance(alpha_vec, cov, cspec, risk_aversion=1.0, max_iter=3000)
    factor_vol = float(np.sqrt(max(risk_model.portfolio_variance(opt.weights), 0.0)))
    bench_w = np.full(spec.n_assets, 1.0 / spec.n_assets)
    te = tracking_error(opt.weights, bench_w, cov)

    # --- backtest: gross vs net of costs --------------------------------- #
    signals = SignalFrame(df=momentum, is_categorical=False)
    gross_cfg = ProductionBacktestConfig(
        n_assets=spec.n_assets,
        n_dates=spec.n_dates,
        enable_universe_mask=True,
        max_weight=0.1,
    )
    net_cfg = ProductionBacktestConfig(
        n_assets=spec.n_assets,
        n_dates=spec.n_dates,
        enable_universe_mask=True,
        enable_costs=True,
        enable_slippage=True,
        max_weight=0.1,
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
    capture_environment("pipeline_run", trials=3, warmup_trials=1)
    trial = run_trials(
        "backtest",
        lambda: ProductionBacktestEngine(net_cfg).run(
            returns, signals, prices=prices, transaction_costs=tcosts, universe_mask=umask
        ),
        lambda r: {"nav": r.nav_history},
        n_trials=3,
        warmup=1,
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


@dataclass(frozen=True)
class WalkForwardBacktestSummary:
    """Results from ``run_walk_forward_backtest``.

    Attributes
    ----------
    n_folds:
        Number of completed CV folds that contributed to the OOS NAV.
    oos_sharpe:
        Annualized Sharpe ratio of the stitched OOS model-prediction NAV.
    oos_cagr:
        Compound annual growth rate of the stitched OOS NAV (fractional).
    oos_max_drawdown:
        Maximum peak-to-trough drawdown of the stitched OOS NAV (negative).
    mean_fold_ic:
        Mean cross-sectional rank IC across all OOS test-fold dates.
    naive_sharpe:
        Sharpe of the naive raw-signal backtest over the same OOS window.
    naive_cagr:
        CAGR of the naive raw-signal backtest over the same OOS window.
    model_vs_naive_sharpe:
        ``oos_sharpe - naive_sharpe``: positive means the CV model added value
        over the raw signal in a live backtest.
    """

    n_folds: int
    oos_sharpe: float
    oos_cagr: float
    oos_max_drawdown: float
    mean_fold_ic: float
    naive_sharpe: float
    naive_cagr: float
    model_vs_naive_sharpe: float


def _stitch_nav_history(fold_navs: list[pl.DataFrame]) -> pl.DataFrame:
    """Concatenate per-fold NAV histories into a single continuous series.

    Each fold's NAV starts at 1_000_000 (engine default).  We rescale each
    subsequent fold so it begins at the prior fold's ending NAV, producing a
    seamless equity curve.  The first fold is kept as-is.

    Parameters
    ----------
    fold_navs:
        Per-fold ``nav_history`` DataFrames, each with ``(date, nav)`` columns,
        in chronological order.

    Returns
    -------
    pl.DataFrame
        Combined ``(date, nav)`` frame sorted by date with no date gaps between
        fold boundaries (folds are non-overlapping by construction).
    """
    if not fold_navs:
        return pl.DataFrame(
            {"date": pl.Series([], dtype=pl.Date), "nav": pl.Series([], dtype=pl.Float64)}
        )

    stitched = fold_navs[0]
    for nxt in fold_navs[1:]:
        # Scale factor: carry the terminal NAV of the prior fold to the start.
        last_nav = to_float(stitched["nav"].last())
        first_nav = to_float(nxt["nav"].first())
        scale = 1.0 if first_nav == 0.0 else last_nav / first_nav
        scaled = nxt.with_columns((pl.col("nav") * scale).alias("nav"))
        stitched = pl.concat([stitched, scaled])

    return stitched.sort("date")


def _predictions_panel_to_signal_frame(pred_df: pl.DataFrame) -> SignalFrame:
    """Convert a ``predictions_panel`` slice to a ``SignalFrame``.

    Renames ``prediction → signal`` so the backtest engine sees the expected
    column name.
    """
    df = pred_df.select(
        pl.col("date"),
        pl.col("id"),
        pl.col("prediction").alias("signal"),
    )
    return SignalFrame(df=df, is_categorical=False)


def _run_backtest_on_window(
    returns: pl.DataFrame,
    signals: SignalFrame,
    prices: pl.DataFrame,
    tcosts: pl.DataFrame,
    umask: pl.DataFrame,
    n_assets: int,
) -> pl.DataFrame:
    """Run a minimal ProductionBacktestEngine on a date-windowed slice.

    Returns the ``nav_history`` DataFrame for the given window.
    """
    # Determine n_dates from the returns slice.
    n_dates = returns["date"].n_unique()
    cfg = ProductionBacktestConfig(
        n_assets=n_assets,
        n_dates=n_dates,
        enable_universe_mask=True,
        max_weight=0.1,
    )
    result = ProductionBacktestEngine(cfg).run(
        returns,
        signals,
        prices=prices,
        transaction_costs=tcosts,
        universe_mask=umask,
    )
    return result.nav_history


def run_walk_forward_backtest(spec: GenSpec, workdir: Path) -> WalkForwardBacktestSummary:
    """Walk-forward backtest that stitches per-fold OOS model predictions.

    For each CV fold the model's OOS ``predictions_panel`` is converted to a
    ``SignalFrame`` and used to drive a ``ProductionBacktestEngine`` over that
    fold's test window.  The per-fold NAV histories are stitched into one
    continuous OOS equity curve.  A naive-signal baseline (same raw alpha signal
    used without model refinement, run over the concatenated OOS window) is
    computed for comparison.

    Parameters
    ----------
    spec:
        Dataset generation spec (controls n_assets, n_dates, seed, etc.).
    workdir:
        Directory where datasets are written (same as ``run_production_pipeline``).

    Returns
    -------
    WalkForwardBacktestSummary
    """
    write_all(workdir, spec)
    loader = DatasetLoader(workdir, spec)

    returns = _returns_from_prices(loader.load("prices"))
    prices = loader.load("prices")
    tcosts = loader.load("transaction_costs")
    umask = loader.load("universe_mask")
    fwd = loader.load("forward_returns")
    momentum = (
        loader.load("alpha_signals")
        .filter(pl.col("signal_name") == "momentum")
        .select("date", "id", "signal")
    )

    # Build panel and run walk-forward CV to get keyed OOS predictions.
    panel = build_panel(
        loader.load("feature_panel"),
        fwd,
        "fwd_ret_1",
        weights=loader.load("sample_weights"),
    )
    wf_cfg = WalkForwardConfig(alpha_grid=[0.01, 0.1, 1.0, 10.0], scale_features=True)
    wf: WFResult = walk_forward_cv(
        panel,
        WalkForwardSplitter(n_splits=4, embargo_periods=5),
        lambda alpha: RidgeModel(ModelConfig(n_features=spec.n_features, alpha=alpha)),
        wf_cfg,
    )

    if wf.predictions_panel is None or len(wf.fold_results) == 0:
        return WalkForwardBacktestSummary(
            n_folds=0,
            oos_sharpe=0.0,
            oos_cagr=0.0,
            oos_max_drawdown=0.0,
            mean_fold_ic=0.0,
            naive_sharpe=0.0,
            naive_cagr=0.0,
            model_vs_naive_sharpe=0.0,
        )

    pred_panel = wf.predictions_panel
    # Unique fold indices, in chronological order (folds were appended in order).
    fold_indices = sorted(pred_panel["fold"].unique().to_list())
    n_folds = len(fold_indices)

    # All dates that appear in the OOS window, for filtering the naive signal later.
    oos_dates: set[date] = set(pred_panel["date"].unique().to_list())

    fold_model_navs: list[pl.DataFrame] = []

    for fi in fold_indices:
        fold_preds = pred_panel.filter(pl.col("fold") == fi)
        fold_date_list = sorted(fold_preds["date"].unique().to_list())
        d_min = fold_date_list[0]
        d_max = fold_date_list[-1]

        # Slice returns/prices/costs to this fold's date range.
        ret_fold = returns.filter(pl.col("date").is_between(d_min, d_max))
        prices_fold = prices.filter(pl.col("date").is_between(d_min, d_max))
        tcosts_fold = tcosts.filter(pl.col("date").is_between(d_min, d_max))
        umask_fold = umask.filter(pl.col("date").is_between(d_min, d_max))

        # Model signal from OOS predictions — fill missing assets with 0.
        # The predictions_panel may not cover every asset on every date if
        # the panel had NaN rows dropped; inner-join ensures alignment.
        model_sf = _predictions_panel_to_signal_frame(fold_preds)

        # Align returns universe to the assets in this fold's signals.
        sig_ids = set(fold_preds["id"].unique().to_list())
        ret_fold_aligned = ret_fold.filter(pl.col("id").is_in(list(sig_ids)))

        nav_df = _run_backtest_on_window(
            ret_fold_aligned,
            model_sf,
            prices_fold.filter(pl.col("id").is_in(list(sig_ids))),
            tcosts_fold.filter(pl.col("id").is_in(list(sig_ids))),
            umask_fold.filter(pl.col("id").is_in(list(sig_ids))),
            len(sig_ids),
        )
        fold_model_navs.append(nav_df)

    stitched_nav = _stitch_nav_history(fold_model_navs)
    stitched_rets = returns_from_nav(stitched_nav)
    oos_sharpe = sharpe(stitched_rets)
    oos_cagr = cagr(stitched_nav)
    oos_mdd = max_drawdown(stitched_nav)

    # --- Naive baseline: run raw momentum signal over the full OOS window --- #
    # Collect full OOS date range across all folds.
    oos_date_list = sorted(oos_dates)
    oos_d_min = oos_date_list[0]
    oos_d_max = oos_date_list[-1]

    # Use the same asset universe as in the OOS predictions.
    oos_ids = set(pred_panel["id"].unique().to_list())
    naive_ret = returns.filter(
        pl.col("date").is_between(oos_d_min, oos_d_max) & pl.col("id").is_in(list(oos_ids))
    )
    naive_sf = SignalFrame(
        df=momentum.filter(
            pl.col("date").is_between(oos_d_min, oos_d_max) & pl.col("id").is_in(list(oos_ids))
        ),
        is_categorical=False,
    )
    naive_prices = prices.filter(
        pl.col("date").is_between(oos_d_min, oos_d_max) & pl.col("id").is_in(list(oos_ids))
    )
    naive_tcosts = tcosts.filter(
        pl.col("date").is_between(oos_d_min, oos_d_max) & pl.col("id").is_in(list(oos_ids))
    )
    naive_umask = umask.filter(
        pl.col("date").is_between(oos_d_min, oos_d_max) & pl.col("id").is_in(list(oos_ids))
    )
    naive_n_dates = naive_ret["date"].n_unique()
    naive_cfg = ProductionBacktestConfig(
        n_assets=len(oos_ids),
        n_dates=naive_n_dates,
        enable_universe_mask=True,
        max_weight=0.1,
    )
    naive_result = ProductionBacktestEngine(naive_cfg).run(
        naive_ret,
        naive_sf,
        prices=naive_prices,
        transaction_costs=naive_tcosts,
        universe_mask=naive_umask,
    )
    naive_rets = returns_from_nav(naive_result.nav_history)
    naive_sharpe_val = sharpe(naive_rets)
    naive_cagr_val = cagr(naive_result.nav_history)

    return WalkForwardBacktestSummary(
        n_folds=n_folds,
        oos_sharpe=oos_sharpe,
        oos_cagr=oos_cagr,
        oos_max_drawdown=oos_mdd,
        mean_fold_ic=float(wf.mean_ic),
        naive_sharpe=naive_sharpe_val,
        naive_cagr=naive_cagr_val,
        model_vs_naive_sharpe=oos_sharpe - naive_sharpe_val,
    )


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
    "WalkForwardBacktestSummary",
    "print_pipeline_summary",
    "run_production_pipeline",
    "run_walk_forward_backtest",
]
