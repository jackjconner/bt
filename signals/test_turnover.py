"""Tests for turnover-aware signal scoring."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from etl.datasets import GenSpec, generate
from signals.turnover import TurnoverResult, rank_stability, signal_autocorr, turnover_score

SPEC = GenSpec(n_assets=40, n_dates=50, seed=17)


def _signals(name: str = "momentum") -> pl.DataFrame:
    df = generate("alpha_signals", SPEC)
    return df.filter(pl.col("signal_name") == name).select("date", "id", "signal")


def _constant_signal() -> pl.DataFrame:
    """A signal that never changes — autocorr should be 1."""
    from etl.datasets import _panel

    grid = _panel(SPEC)
    rng = np.random.default_rng(0)
    static = rng.normal(0.0, 1.0, SPEC.n_assets)
    values = np.tile(static, SPEC.n_dates)
    return grid.with_columns(pl.Series("signal", values))


def _random_signal() -> pl.DataFrame:
    """An iid-normal signal — autocorr should be near 0."""
    from etl.datasets import _panel

    grid = _panel(SPEC)
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 1.0, SPEC.n_assets * SPEC.n_dates)
    return grid.with_columns(pl.Series("signal", values))


# ---------------------------------------------------------------------------
# signal_autocorr
# ---------------------------------------------------------------------------


def test_signal_autocorr_constant_is_one():
    sig = _constant_signal()
    ac = signal_autocorr(sig)
    assert ac == pytest.approx(1.0, abs=1e-6)


def test_signal_autocorr_random_is_near_zero():
    sig = _random_signal()
    ac = signal_autocorr(sig)
    # Random signal: expected autocorr ~0; should be well below 0.5
    assert abs(ac) < 0.3


def test_signal_autocorr_realistic_signal():
    sig = _signals()
    ac = signal_autocorr(sig)
    assert -1.0 <= ac <= 1.0
    assert np.isfinite(ac)


# ---------------------------------------------------------------------------
# rank_stability
# ---------------------------------------------------------------------------


def test_rank_stability_constant_is_one():
    sig = _constant_signal()
    rs = rank_stability(sig)
    assert rs == pytest.approx(1.0, abs=1e-6)


def test_rank_stability_random_is_low():
    sig = _random_signal()
    rs = rank_stability(sig)
    # Random: ~1/n_quantiles of extreme assets stay; should be < 0.5
    assert rs < 0.5


def test_rank_stability_range():
    sig = _signals()
    rs = rank_stability(sig)
    assert 0.0 <= rs <= 1.0


# ---------------------------------------------------------------------------
# turnover_score
# ---------------------------------------------------------------------------


def test_turnover_score_returns_turnover_result():
    sig = _signals()
    result = turnover_score(sig, ic_ir_gross=0.5)
    assert isinstance(result, TurnoverResult)


def test_turnover_score_net_leq_gross():
    sig = _random_signal()
    result = turnover_score(sig, ic_ir_gross=1.0, cost_bps=10.0)
    # High turnover signal → cost drag should reduce net IC-IR below gross
    assert result.ic_ir_net <= result.ic_ir_gross


def test_turnover_score_constant_signal_no_drag():
    """Constant signal has zero turnover — net IC-IR should equal gross."""
    sig = _constant_signal()
    result = turnover_score(sig, ic_ir_gross=1.0, cost_bps=10.0)
    assert result.ic_ir_net == pytest.approx(result.ic_ir_gross, abs=0.01)


def test_turnover_score_n_dates():
    sig = _signals()
    result = turnover_score(sig, ic_ir_gross=0.5)
    assert result.n_dates == SPEC.n_dates
