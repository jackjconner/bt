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


# ---------------------------------------------------------------------------
# _spearman_ic_rows fast path 3 — row-homogeneous NaN (tail forward returns)
# ---------------------------------------------------------------------------


def test_spearman_ic_rows_row_homogeneous_matches_loop():
    """Fast path 3 (row-homogeneous NaN) produces bit-identical IC to the fallback.

    Constructs S with no NaN and R with NaN in the last N rows (mimicking
    forward returns that are missing for the most-recent dates), then checks
    that the optimized code path returns the same IC values as the per-date
    loop reference.
    """
    from signals.ic import _spearman_ic_rows

    rng = np.random.default_rng(99)
    n_dates, n_assets = 200, 40
    S = rng.standard_normal((n_dates, n_assets))

    # R with NaN only in last 5 rows (all assets NaN simultaneously).
    R_full = rng.standard_normal((n_dates, n_assets))
    R_tail_nan = R_full.copy()
    R_tail_nan[-5:] = np.nan

    ic_full, n_full = _spearman_ic_rows(S, R_full, min_obs=2)
    ic_tail, n_tail = _spearman_ic_rows(S, R_tail_nan, min_obs=2)

    # Non-tail dates: IC should be identical to the full (no-NaN) case.
    np.testing.assert_array_equal(ic_full[:-5], ic_tail[:-5])
    np.testing.assert_array_equal(n_full[:-5], n_tail[:-5])
    # Tail dates: NaN (all assets missing).
    assert np.all(np.isnan(ic_tail[-5:]))
    assert np.all(n_tail[-5:] == 0)


def test_spearman_ic_rows_row_homogeneous_vs_per_date_loop():
    """Fast path 3 values are numerically identical to a plain per-date loop."""
    from scipy import stats

    from signals.ic import _spearman_ic_rows

    rng = np.random.default_rng(42)
    n_dates, n_assets = 100, 20
    S = rng.standard_normal((n_dates, n_assets))
    R = rng.standard_normal((n_dates, n_assets))
    R[-3:] = np.nan  # trigger fast path 3

    ic_fast, _ = _spearman_ic_rows(S, R, min_obs=2)

    # Reference: per-date loop
    ic_ref = np.full(n_dates, np.nan)
    for t in range(n_dates - 3):
        rx = stats.rankdata(S[t])
        ry = stats.rankdata(R[t])
        rx_c = rx - rx.mean()
        ry_c = ry - ry.mean()
        denom = float(np.sqrt((rx_c**2).sum() * (ry_c**2).sum()))
        ic_ref[t] = float((rx_c * ry_c).sum() / denom) if denom > 0 else np.nan

    np.testing.assert_allclose(ic_fast[:-3], ic_ref[:-3], rtol=1e-10)
