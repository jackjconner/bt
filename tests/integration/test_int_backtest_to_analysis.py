"""Contract: backtest result → analysis (absolute + benchmark-relative).

`BacktestResult.nav_history` / `trade_log` are the inputs the analysis module
turns into performance and benchmark-relative metrics. This verifies the
hand-off produces finite, well-formed analytics.
"""

from __future__ import annotations

import math

import polars as pl

from analysis import (
    BacktestAnalyzerImpl,
    alpha,
    beta,
    information_ratio,
    two_way_turnover,
)
from backtest import ProductionBacktestConfig, ProductionBacktestEngine, SignalFrame
from harness.components import returns_from_prices


def _result(synth):
    loader, spec = synth.loader, synth.spec
    prices = loader.load("prices")
    momentum = (
        loader.load("alpha_signals")
        .filter(pl.col("signal_name") == "momentum")
        .select("date", "id", "signal")
    )
    cfg = ProductionBacktestConfig(n_assets=spec.n_assets, n_dates=spec.n_dates, max_weight=0.1)
    return ProductionBacktestEngine(cfg).run(
        returns_from_prices(prices), SignalFrame(df=momentum, is_categorical=False)
    )


def test_analysis_consumes_backtest_result(synth) -> None:
    result = _result(synth)
    analysis = BacktestAnalyzerImpl().analyze(result)

    assert analysis.returns_series.height == result.nav_history.height - 1
    for v in (
        analysis.sharpe,
        analysis.cagr,
        analysis.sortino,
        analysis.annualized_vol,
        analysis.max_drawdown,
    ):
        assert math.isfinite(v)
    assert analysis.max_drawdown <= 0.0


def test_benchmark_relative_metrics(synth) -> None:
    result = _result(synth)
    analysis = BacktestAnalyzerImpl().analyze(result)
    bench = (
        synth.loader.load("benchmark_returns")
        .filter(pl.col("benchmark_id") == "BMK0")
        .with_columns((pl.col("return") / 100.0).alias("return_1d"))
        .select("date", "return_1d")
    )
    a = alpha(analysis.returns_series, bench)
    b = beta(analysis.returns_series, bench)
    ir = information_ratio(analysis.returns_series, bench)
    assert all(math.isfinite(x) for x in (a, b, ir))


def test_benchmark_beta_against_itself_is_one(synth) -> None:
    bench = (
        synth.loader.load("benchmark_returns")
        .filter(pl.col("benchmark_id") == "BMK0")
        .with_columns((pl.col("return") / 100.0).alias("return_1d"))
        .select("date", "return_1d")
    )
    assert abs(beta(bench, bench) - 1.0) < 1e-9


def test_turnover_from_trade_log(synth) -> None:
    result = _result(synth)
    turnover = two_way_turnover(result.trade_log)
    assert turnover.height > 0
