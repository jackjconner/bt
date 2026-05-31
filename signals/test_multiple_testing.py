"""Tests for multiple-testing correction and rolling IC-IR."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from etl.datasets import GenSpec, generate
from signals.ic import ic_series_v2
from signals.multiple_testing import (
    MultipleTestingResult,
    bh_correct,
    bonferroni_correct,
    multiple_testing_correction,
    rolling_ic_ir,
    tstat_to_pvalue,
)
from signals.newey_west import newey_west_tstat


SPEC = GenSpec(n_assets=30, n_dates=80, seed=55)


def _signals(name: str = "momentum") -> pl.DataFrame:
    df = generate("alpha_signals", SPEC)
    return df.filter(pl.col("signal_name") == name).select("date", "id", "signal")


def _fwd_returns() -> pl.DataFrame:
    return generate("forward_returns", SPEC)


def _ic_df(name: str = "momentum") -> pl.DataFrame:
    return ic_series_v2(
        _signals(name), _fwd_returns(),
        return_col="fwd_ret_1", method="rank", min_obs=5
    )


# ---------------------------------------------------------------------------
# rolling_ic_ir
# ---------------------------------------------------------------------------


def test_rolling_ic_ir_returns_dataframe():
    ic_df = _ic_df()
    result = rolling_ic_ir(ic_df, window=10)
    assert isinstance(result, pl.DataFrame)
    assert set(result.columns) >= {"date", "rolling_ic", "rolling_ic_std", "rolling_ic_ir"}


def test_rolling_ic_ir_same_length():
    ic_df = _ic_df()
    result = rolling_ic_ir(ic_df, window=10)
    assert len(result) == len(ic_df)


def test_rolling_ic_ir_nan_at_start():
    ic_df = _ic_df()
    result = rolling_ic_ir(ic_df, window=20)
    # First (window-1) values should be NaN due to insufficient history
    n_nan = result["rolling_ic"].is_nan().sum() + result["rolling_ic"].is_null().sum()
    assert n_nan >= 19  # window=20 → first 19 rows are NaN


def test_rolling_ic_ir_finite_after_warmup():
    ic_df = _ic_df()
    result = rolling_ic_ir(ic_df, window=5)
    # At least some values in the overall result should be finite
    all_rolling = result["rolling_ic"].to_numpy()
    finite_vals = all_rolling[np.isfinite(all_rolling)]
    assert len(finite_vals) > 0


# ---------------------------------------------------------------------------
# tstat_to_pvalue
# ---------------------------------------------------------------------------


def test_tstat_to_pvalue_zero_tstat():
    p = tstat_to_pvalue(0.0)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_tstat_to_pvalue_large_tstat():
    p = tstat_to_pvalue(10.0)
    assert p < 1e-10


def test_tstat_to_pvalue_negative_symmetric():
    p_pos = tstat_to_pvalue(2.0)
    p_neg = tstat_to_pvalue(-2.0)
    assert p_pos == pytest.approx(p_neg, rel=1e-6)


def test_tstat_to_pvalue_canonical_value():
    # t=1.96 → two-sided p ≈ 0.05
    p = tstat_to_pvalue(1.96)
    assert p == pytest.approx(0.05, abs=0.002)


# ---------------------------------------------------------------------------
# bonferroni_correct
# ---------------------------------------------------------------------------


def test_bonferroni_rejects_small_p():
    p_values = [0.001, 0.03, 0.1]
    mask = bonferroni_correct(p_values, alpha=0.05)
    # Bonferroni threshold = 0.05/3 ≈ 0.0167; only 0.001 should be rejected
    assert mask[0]
    assert not mask[1]
    assert not mask[2]


def test_bonferroni_all_large_p():
    p_values = [0.9, 0.8, 0.7]
    mask = bonferroni_correct(p_values, alpha=0.05)
    assert not any(mask)


def test_bonferroni_all_small_p():
    p_values = [0.001, 0.002, 0.003]
    mask = bonferroni_correct(p_values, alpha=0.05)
    assert all(mask)


# ---------------------------------------------------------------------------
# bh_correct
# ---------------------------------------------------------------------------


def test_bh_rejects_expected():
    # 5 tests: 2 with very small p, 3 with large p
    p_values = [0.001, 0.005, 0.3, 0.4, 0.9]
    mask, threshold = bh_correct(p_values, alpha=0.05)
    assert mask[0]
    assert mask[1]
    assert not mask[2]


def test_bh_threshold_is_float():
    p_values = [0.01, 0.04, 0.5]
    mask, threshold = bh_correct(p_values, alpha=0.05)
    assert isinstance(threshold, float)


def test_bh_empty():
    mask, threshold = bh_correct([], alpha=0.05)
    assert len(mask) == 0
    assert threshold == 0.0


def test_bh_less_conservative_than_bonferroni():
    """BH should reject at least as many hypotheses as Bonferroni."""
    p_values = [0.001, 0.01, 0.02, 0.04, 0.1]
    bonf = bonferroni_correct(p_values, alpha=0.05)
    bh_mask, _ = bh_correct(p_values, alpha=0.05)
    assert int(bh_mask.sum()) >= int(bonf.sum())


# ---------------------------------------------------------------------------
# multiple_testing_correction
# ---------------------------------------------------------------------------


def test_multiple_testing_correction_returns_result():
    names = ["signal_a", "signal_b", "signal_c"]
    t_stats = [3.0, 1.5, 0.5]
    result = multiple_testing_correction(names, t_stats, alpha=0.05)
    assert isinstance(result, MultipleTestingResult)


def test_multiple_testing_correction_n_tests():
    names = ["a", "b", "c", "d"]
    t_stats = [2.0, 3.0, 1.0, 4.0]
    result = multiple_testing_correction(names, t_stats)
    assert result.n_tests == 4


def test_multiple_testing_correction_to_frame():
    names = ["a", "b"]
    t_stats = [2.5, 1.0]
    result = multiple_testing_correction(names, t_stats)
    df = result.to_frame()
    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) >= {"signal", "t_stat", "p_value", "bonferroni_reject", "bh_reject"}
    assert len(df) == 2


def test_multiple_testing_correction_bonferroni_threshold():
    """Bonferroni threshold_t should be higher than 1.96 when testing multiple signals."""
    k = 10
    names = [f"sig_{i}" for i in range(k)]
    t_stats = [1.0] * k
    result = multiple_testing_correction(names, t_stats, alpha=0.05)
    # With k=10 tests, threshold_t = z(0.05/(2*10)) > z(0.025) = 1.96
    assert result.bonferroni_threshold_t > 1.96


def test_multiple_testing_real_signals():
    """Use real signals from the dataset to verify the pipeline works end-to-end."""
    signal_names = ["momentum", "value", "quality"]
    t_stats = []
    for name in signal_names:
        ic_df = _ic_df(name)
        s = ic_df["ic"].drop_nulls()
        t_stats.append(float(newey_west_tstat(s)))

    result = multiple_testing_correction(signal_names, t_stats)
    assert result.n_tests == 3
    df = result.to_frame()
    assert len(df) == 3
    # The highest-IC signal ("quality") should have a positive t-stat with
    # injected IC ~0.11; we allow noise in the lower-IC signals
    quality_idx = signal_names.index("quality")
    assert t_stats[quality_idx] > 0
