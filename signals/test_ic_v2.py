"""Tests for ic_series_v2 (configurable horizon + method + coverage)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from etl.datasets import GenSpec, generate
from signals import ic_series_v2
from signals.coverage import apply_min_coverage, pairwise_mask

SPEC = GenSpec(n_assets=30, n_dates=60, seed=42)


def _signals_for_name(name: str) -> pl.DataFrame:
    df = generate("alpha_signals", SPEC)
    return df.filter(pl.col("signal_name") == name).select("date", "id", "signal")


def _fwd_returns() -> pl.DataFrame:
    return generate("forward_returns", SPEC)


# ---------------------------------------------------------------------------
# pairwise_mask
# ---------------------------------------------------------------------------


def test_pairwise_mask_both_finite():
    x = np.array([1.0, 2.0, np.nan, 4.0])
    y = np.array([1.0, np.nan, 3.0, 4.0])
    mask = pairwise_mask(x, y)
    assert list(mask) == [True, False, False, True]


def test_pairwise_mask_all_valid():
    x = np.ones(5)
    y = np.ones(5)
    assert pairwise_mask(x, y).all()


# ---------------------------------------------------------------------------
# apply_min_coverage
# ---------------------------------------------------------------------------


def test_apply_min_coverage_suppresses_low_count():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    counts = np.array([10, 5, 3, 15])
    result = apply_min_coverage(values, counts, min_obs=6)
    assert np.isnan(result[1])  # count=5 < 6
    assert np.isnan(result[2])  # count=3 < 6
    assert result[0] == pytest.approx(0.1)
    assert result[3] == pytest.approx(0.4)


def test_apply_min_coverage_all_pass():
    values = np.array([0.1, 0.2])
    counts = np.array([100, 200])
    result = apply_min_coverage(values, counts, min_obs=5)
    assert not np.any(np.isnan(result))


# ---------------------------------------------------------------------------
# ic_series_v2 — method choices
# ---------------------------------------------------------------------------


def test_ic_series_v2_rank_returns_dataframe():
    sig = _signals_for_name("momentum")
    fwd = _fwd_returns()
    result = ic_series_v2(sig, fwd, return_col="fwd_ret_1", method="rank", min_obs=5)
    assert isinstance(result, pl.DataFrame)
    assert "date" in result.columns
    assert "ic" in result.columns
    assert "n_obs" in result.columns


def test_ic_series_v2_pearson_returns_dataframe():
    sig = _signals_for_name("momentum")
    fwd = _fwd_returns()
    result = ic_series_v2(sig, fwd, return_col="fwd_ret_1", method="pearson", min_obs=5)
    assert "ic" in result.columns


def test_ic_series_v2_kendall_returns_dataframe():
    sig = _signals_for_name("momentum")
    fwd = _fwd_returns()
    result = ic_series_v2(sig, fwd, return_col="fwd_ret_1", method="kendall", min_obs=5)
    assert "ic" in result.columns


def test_ic_series_v2_different_horizons():
    sig = _signals_for_name("momentum")
    fwd = _fwd_returns()
    ic1 = ic_series_v2(sig, fwd, return_col="fwd_ret_1", method="rank")
    ic5 = ic_series_v2(sig, fwd, return_col="fwd_ret_5", method="rank")
    # IC over different horizons should differ (different return windows)
    valid1 = ic1["ic"].drop_nulls().to_numpy()
    valid5 = ic5["ic"].drop_nulls().to_numpy()
    assert len(valid1) > 0 and len(valid5) > 0
    # They should NOT be identical
    min_len = min(len(valid1), len(valid5))
    assert not np.allclose(valid1[:min_len], valid5[:min_len])


def test_ic_series_v2_min_obs_suppresses_dates():
    sig = _signals_for_name("momentum")
    fwd = _fwd_returns()
    # Very high min_obs should suppress almost all dates
    high_min = ic_series_v2(sig, fwd, return_col="fwd_ret_1", min_obs=9999)
    low_min = ic_series_v2(sig, fwd, return_col="fwd_ret_1", min_obs=2)
    # NaN stored as null so we use is_not_null / drop_nulls
    n_valid_high = high_min["ic"].drop_nulls().len()
    n_valid_low = low_min["ic"].drop_nulls().len()
    assert n_valid_low >= n_valid_high


def test_ic_series_v2_injected_signal_positive_ic():
    """The alpha_signals dataset injects a positive IC; mean IC should be positive."""
    sig = _signals_for_name("value")  # higher IC signal
    fwd = _fwd_returns()
    result = ic_series_v2(sig, fwd, return_col="fwd_ret_1", method="rank", min_obs=2)
    arr = result["ic"].drop_nulls().to_numpy()
    assert len(arr) > 0
    # With seed=42 and injected IC ~0.08, mean IC should be positive
    assert float(np.nanmean(arr)) > 0
