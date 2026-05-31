"""One profiling benchmark per distinct component, each exercising the
component's production path on schema-valid synthetic data.

``setup`` loads the inputs (untimed); ``run`` is the timed production call.
The benchmarks are deliberately the same production paths the integration
tests assert on, so a regression in cost and a regression in behaviour are
caught by the same wiring.
"""

from __future__ import annotations

import polars as pl

from analysis import BacktestAnalyzerImpl, alpha, beta, information_ratio, two_way_turnover
from backtest import (
    BacktestResult,
    ProductionBacktestConfig,
    ProductionBacktestEngine,
    SignalFrame,
)
from etl import adjust_prices, check, to_masked_matrix, to_matrix
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
)
from profiling import check_regressions, fit_scaling
from signals import ic_horizon_curve, ic_series_v2, neutralize_sector, quantile_spread

from .spec import BenchmarkContext, ComponentBenchmark, no_frames


def returns_from_prices(prices: pl.DataFrame) -> pl.DataFrame:
    """Session-aligned percent returns implied by the price panel."""
    return (
        prices.sort(["id", "date"])
        .with_columns(
            ((pl.col("close") / pl.col("close").shift(1).over("id") - 1.0) * 100.0)
            .fill_null(0.0)
            .alias("return")
        )
        .select("date", "id", "return")
    )


def _momentum(ctx: BenchmarkContext) -> pl.DataFrame:
    return (
        ctx.loader.load("alpha_signals")
        .filter(pl.col("signal_name") == "momentum")
        .select("date", "id", "signal")
    )


# --- etl ------------------------------------------------------------------- #
def _etl_setup(ctx: BenchmarkContext) -> dict:
    return {
        "prices": ctx.loader.load("prices"),
        "corporate_actions": ctx.loader.load("corporate_actions"),
    }


def _etl_run(inp: dict) -> dict:
    adjusted = adjust_prices(inp["prices"], inp["corporate_actions"])
    mat, mask, _, _ = to_masked_matrix(inp["prices"], "close")
    report = check(inp["prices"], "close")
    return {"adjusted": adjusted.prices, "matrix": mat, "mask": mask, "ok": report.ok}


def _etl_frames(out: dict) -> dict[str, object]:
    return {"adjusted": out["adjusted"], "matrix": out["matrix"]}


# --- signals --------------------------------------------------------------- #
def _signals_setup(ctx: BenchmarkContext) -> dict:
    return {
        "signal": _momentum(ctx),
        "fwd": ctx.loader.load("forward_returns"),
        "sec": ctx.loader.load("security_master"),
    }


def _signals_run(inp: dict) -> dict:
    ic = ic_series_v2(inp["signal"], inp["fwd"], return_col="fwd_ret_1")
    neutral = neutralize_sector(inp["signal"], inp["sec"])
    curve = ic_horizon_curve(
        inp["signal"],
        inp["fwd"],
        {1: "fwd_ret_1", 5: "fwd_ret_5", 21: "fwd_ret_21", 63: "fwd_ret_63"},
    )
    spread = quantile_spread(inp["signal"], inp["fwd"], return_col="fwd_ret_1")
    return {"ic": ic, "neutral": neutral, "curve": curve, "spread": spread}


def _signals_frames(out: dict) -> dict[str, object]:
    return {"ic": out["ic"], "neutral": out["neutral"]}


# --- models ---------------------------------------------------------------- #
def _models_setup(ctx: BenchmarkContext) -> dict:
    panel = build_panel(
        ctx.loader.load("feature_panel"),
        ctx.loader.load("forward_returns"),
        "fwd_ret_1",
        weights=ctx.loader.load("sample_weights"),
    )
    return {"panel": panel, "n_features": ctx.spec.n_features}


def _models_run(inp: dict) -> object:
    return walk_forward_cv(
        inp["panel"],
        WalkForwardSplitter(n_splits=4, embargo_periods=5),
        lambda a: RidgeModel(ModelConfig(n_features=inp["n_features"], alpha=a)),
    )


# --- analysis -------------------------------------------------------------- #
def _analysis_setup(ctx: BenchmarkContext) -> dict:
    prices = ctx.loader.load("prices")
    returns = returns_from_prices(prices)
    signals = SignalFrame(df=_momentum(ctx), is_categorical=False)
    cfg = ProductionBacktestConfig(
        n_assets=ctx.spec.n_assets, n_dates=ctx.spec.n_dates, max_weight=0.1
    )
    result = ProductionBacktestEngine(cfg).run(returns, signals)
    bench = (
        ctx.loader.load("benchmark_returns")
        .filter(pl.col("benchmark_id") == "BMK0")
        .with_columns((pl.col("return") / 100.0).alias("return_1d"))
        .select("date", "return_1d")
    )
    return {"result": result, "bench": bench}


def _analysis_run(inp: dict) -> dict:
    analysis = BacktestAnalyzerImpl().analyze(inp["result"])
    rets = analysis.returns_series
    return {
        "analysis": analysis,
        "alpha": alpha(rets, inp["bench"]),
        "beta": beta(rets, inp["bench"]),
        "ir": information_ratio(rets, inp["bench"]),
        "turnover": two_way_turnover(inp["result"].trade_log),
    }


def _analysis_frames(out: dict) -> dict[str, object]:
    return {"returns": out["analysis"].returns_series}


# --- portfolio ------------------------------------------------------------- #
def _portfolio_setup(ctx: BenchmarkContext) -> dict:
    prices = ctx.loader.load("prices")
    returns = returns_from_prices(prices)
    R, _ = to_matrix(returns, "return")
    as_of = prices["date"].max()
    raw_alpha = _momentum(ctx).filter(pl.col("date") == as_of).sort("id")["signal"].to_numpy()
    return {
        "R": R,
        "alpha": (raw_alpha - raw_alpha.mean()) / (raw_alpha.std() + 1e-9),
        "loadings": ctx.loader.load("factor_loadings"),
        "fcov": ctx.loader.load("factor_covariance"),
        "specific": ctx.loader.load("specific_risk"),
        "as_of": as_of,
        "cspec": constraints_from_polars(
            ctx.loader.load("position_constraints"),
            ctx.loader.load("group_constraints"),
            ctx.loader.load("security_master"),
            ctx.spec.n_assets,
            long_only=False,
            net_exposure=1.0,
        ),
    }


def _portfolio_run(inp: dict) -> dict:
    risk_model = build_from_long(inp["loadings"], inp["fcov"], inp["specific"], inp["as_of"])
    cov = ledoit_wolf_cov(inp["R"])
    opt = mean_variance(
        inp["alpha"], cov, inp["cspec"], risk_aversion=1.0, max_iter=3000, solver="osqp"
    )
    var = risk_model.portfolio_variance(opt.weights)
    return {"weights": opt.weights, "variance": var, "converged": opt.converged}


# --- backtest -------------------------------------------------------------- #
def _backtest_setup(ctx: BenchmarkContext) -> dict:
    prices = ctx.loader.load("prices")
    return {
        "returns": returns_from_prices(prices),
        "signals": SignalFrame(df=_momentum(ctx), is_categorical=False),
        "prices": prices,
        "tcosts": ctx.loader.load("transaction_costs"),
        "umask": ctx.loader.load("universe_mask"),
        "cfg": ProductionBacktestConfig(
            n_assets=ctx.spec.n_assets,
            n_dates=ctx.spec.n_dates,
            enable_costs=True,
            enable_slippage=True,
            enable_universe_mask=True,
            max_weight=0.1,
        ),
    }


def _backtest_run(inp: dict) -> BacktestResult:
    return ProductionBacktestEngine(inp["cfg"]).run(
        inp["returns"],
        inp["signals"],
        prices=inp["prices"],
        transaction_costs=inp["tcosts"],
        universe_mask=inp["umask"],
    )


def _backtest_frames(out: BacktestResult) -> dict[str, object]:
    return {"nav": out.nav_history, "trades": out.trade_log}


# --- profiling (telemetry analytics over its own schema) ------------------- #
def _profiling_setup(ctx: BenchmarkContext) -> dict:
    return {
        "measurements": ctx.loader.load("stage_measurements"),
        "baselines": ctx.loader.load("stage_baselines"),
        "thresholds": ctx.loader.load("regression_thresholds"),
    }


def _profiling_run(inp: dict) -> dict:
    fits = fit_scaling(inp["measurements"], run_id="harness_self")
    current = (
        inp["measurements"]
        .group_by("stage")
        .agg(
            pl.col("elapsed_s").median(),
            pl.col("result_mb").median(),
            pl.col("peak_rss_mb").median(),
        )
    )
    report = check_regressions(current, inp["baselines"], inp["thresholds"])
    return {"n_fits": len(fits), "passed": report.passed}


def build_components() -> list[ComponentBenchmark]:
    return [
        ComponentBenchmark("etl", _etl_setup, _etl_run, _etl_frames),
        ComponentBenchmark("signals", _signals_setup, _signals_run, _signals_frames),
        ComponentBenchmark("models", _models_setup, _models_run, no_frames),
        ComponentBenchmark("analysis", _analysis_setup, _analysis_run, _analysis_frames),
        ComponentBenchmark("portfolio", _portfolio_setup, _portfolio_run, no_frames),
        ComponentBenchmark("backtest", _backtest_setup, _backtest_run, _backtest_frames),
        ComponentBenchmark("profiling", _profiling_setup, _profiling_run, no_frames),
    ]


__all__ = ["build_components", "returns_from_prices"]
