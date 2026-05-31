"""Equivalence tests for the fused single-pass metrics engine.

The fused engine (``analysis.engine``) reimplements the per-metric suite in one
pass over the data. These tests pin it to the scalar source-of-truth path: every
field of ``analyze_fused`` must equal the result of composing ``returns_from_nav``
+ ``sharpe`` + ``max_drawdown`` + ``risk.cagr`` + ``risk.sortino``, and every
field of ``benchmark_metrics_fused`` must equal the matching scalar
``benchmark.*`` function — at ``rf=0`` (the production and harness call) the match
is exact (``==``), and for non-zero risk-free rates within floating-point
tolerance.

A test here fails if a future edit to the engine moves any reported number away
from the contract the rest of the system codes against.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from analysis.benchmark import alpha, beta, information_ratio, tracking_error
from analysis.engine import analyze_fused, benchmark_metrics_fused
from analysis.metrics import BacktestAnalyzerImpl, max_drawdown, returns_from_nav, sharpe
from analysis.risk import cagr, sortino
from backtest.engine import BacktestResult
from etl.source import session_axis, to_float


def _nav(n: int = 300, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0006, 0.012, n)
    nav_vals = 1_000_000.0 * np.cumprod(1.0 + r)
    dates = session_axis(n).to_list()
    return pl.DataFrame({"date": dates, "nav": nav_vals.tolist()})


def _bench(returns: pl.DataFrame, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = returns.height
    b = rng.normal(0.0004, 0.010, n)
    return returns.select("date").with_columns(pl.Series("return_1d", b))


# ---------------------------------------------------------------------------
# analyze_fused == scalar composition
# ---------------------------------------------------------------------------


class TestAnalyzeFusedEquivalence:
    """analyze_fused reproduces the metric-by-metric path bit-for-bit at rf=0."""

    def test_all_scalars_bit_identical_at_zero_rf(self):
        nav = _nav()
        rets = returns_from_nav(nav)
        f = analyze_fused(nav, 0.0)

        assert f.sharpe == sharpe(rets, 0.0)
        assert f.max_drawdown == max_drawdown(nav)
        assert f.cagr == cagr(nav)
        assert f.sortino == sortino(rets, 0.0)
        assert f.annualized_vol == to_float(rets["return_1d"].std() or 0.0) * (252**0.5)
        assert f.annualized_return == to_float(rets["return_1d"].mean() or 0.0) * 252

    def test_returns_and_drawdown_series_equal(self):
        nav = _nav()
        rets = returns_from_nav(nav)
        dd = nav.with_columns(
            (pl.col("nav") / pl.col("nav").cum_max() - 1.0).alias("drawdown")
        ).select("date", "drawdown")
        f = analyze_fused(nav, 0.0)
        assert rets.equals(f.returns_series)
        assert dd.equals(f.drawdown_series)

    def test_nonzero_rf_matches_within_tolerance(self):
        nav = _nav()
        rets = returns_from_nav(nav)
        f = analyze_fused(nav, 0.03)
        assert f.sharpe == pytest.approx(sharpe(rets, 0.03), rel=1e-12, abs=1e-12)
        assert f.sortino == pytest.approx(sortino(rets, 0.03 / 252), rel=1e-12, abs=1e-12)

    def test_analyzer_impl_delegates_to_fused(self):
        nav = _nav()
        result = BacktestResult(
            nav_history=nav,
            trade_log=pl.DataFrame(
                {"date": [], "id": [], "quantity": []},
                schema={"date": pl.Date, "id": pl.Int64, "quantity": pl.Float64},
            ),
            final_positions=np.zeros(3),
        )
        a = BacktestAnalyzerImpl().analyze(result)
        f = analyze_fused(nav, 0.0)
        assert a.sharpe == f.sharpe
        assert a.max_drawdown == f.max_drawdown
        assert a.cagr == f.cagr
        assert a.sortino == f.sortino
        assert a.annualized_return == f.annualized_return
        assert a.annualized_vol == f.annualized_vol


# ---------------------------------------------------------------------------
# benchmark_metrics_fused == scalar benchmark functions
# ---------------------------------------------------------------------------


class TestBenchmarkMetricsFusedEquivalence:
    """benchmark_metrics_fused reproduces the four scalar benchmark functions."""

    def test_matches_scalar_functions(self):
        nav = _nav()
        rets = returns_from_nav(nav)
        bench = _bench(rets)
        m = benchmark_metrics_fused(rets, bench)
        assert m.beta == pytest.approx(beta(rets, bench), rel=1e-12, abs=1e-12)
        assert m.alpha == pytest.approx(alpha(rets, bench), rel=1e-12, abs=1e-12)
        assert m.tracking_error == pytest.approx(tracking_error(rets, bench), rel=1e-12, abs=1e-12)
        assert m.information_ratio == pytest.approx(
            information_ratio(rets, bench), rel=1e-12, abs=1e-12
        )

    def test_beta_against_itself_is_one(self):
        nav = _nav()
        rets = returns_from_nav(nav)
        m = benchmark_metrics_fused(rets, rets)
        assert m.beta == pytest.approx(1.0, abs=1e-12)
        assert m.tracking_error == pytest.approx(0.0, abs=1e-12)
        assert m.information_ratio == 0.0

    def test_alpha_with_risk_free(self):
        nav = _nav()
        rets = returns_from_nav(nav)
        bench = _bench(rets)
        m = benchmark_metrics_fused(rets, bench, rf=0.0001)
        assert m.alpha == pytest.approx(alpha(rets, bench, 0.0001), rel=1e-12, abs=1e-12)

    def test_inner_join_on_disjoint_dates(self):
        nav = _nav()
        rets = returns_from_nav(nav)
        bench = _bench(rets).head(rets.height // 2)
        m = benchmark_metrics_fused(rets, bench)
        sliced = rets.join(bench.select("date"), on="date", how="inner")
        assert m.beta == pytest.approx(beta(sliced, bench), rel=1e-12, abs=1e-12)
