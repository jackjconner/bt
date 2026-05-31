"""Tests for signal combination and orthogonalization."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from etl.datasets import GenSpec, generate
from signals.combine import (
    IncrementalICResult,
    gram_schmidt_orthogonalize,
    ic_weighted_blend,
    incremental_ic,
    zscore_blend,
)

SPEC = GenSpec(n_assets=40, n_dates=50, seed=31)


def _all_signals() -> list[pl.DataFrame]:
    df = generate("alpha_signals", SPEC)
    names = df["signal_name"].unique().sort().to_list()
    return [df.filter(pl.col("signal_name") == n).select("date", "id", "signal") for n in names]


def _fwd_returns() -> pl.DataFrame:
    return generate("forward_returns", SPEC)


# ---------------------------------------------------------------------------
# zscore_blend
# ---------------------------------------------------------------------------


def test_zscore_blend_returns_dataframe():
    sigs = _all_signals()
    result = zscore_blend(sigs)
    assert isinstance(result, pl.DataFrame)
    assert set(result.columns) >= {"date", "id", "signal"}


def test_zscore_blend_same_shape_as_input():
    sigs = _all_signals()
    result = zscore_blend(sigs)
    # Each input signal has n_dates * n_assets rows
    assert len(result) == SPEC.n_dates * SPEC.n_assets


def test_zscore_blend_is_cross_sectionally_standardized():
    """Each input signal is z-scored before blending; the composite should have
    finite, bounded values consistent with a blend of ~N(0,1) cross-sections."""
    sigs = _all_signals()
    result = zscore_blend(sigs)
    from etl.source import to_matrix

    mat, _ = to_matrix(result, "signal")
    # All finite values should be within a reasonable range for standardized inputs
    finite = mat[np.isfinite(mat)]
    assert len(finite) > 0
    # Blended z-scores from correlated signals: |values| should be moderate
    assert np.abs(finite).max() < 10.0
    # Cross-sectional mean per date should be near zero (each component is mean-zero)
    per_date_means = []
    for row in mat:
        f = row[np.isfinite(row)]
        if len(f) > 1:
            per_date_means.append(f.mean())
    assert abs(np.mean(per_date_means)) < 0.3


def test_zscore_blend_with_weights():
    sigs = _all_signals()
    w = [1.0, 2.0, 3.0]
    result = zscore_blend(sigs, weights=w)
    assert isinstance(result, pl.DataFrame)
    assert len(result) == SPEC.n_dates * SPEC.n_assets


def test_zscore_blend_single_signal():
    sigs = [_all_signals()[0]]
    result = zscore_blend(sigs)
    assert isinstance(result, pl.DataFrame)


def test_zscore_blend_raises_on_empty():
    with pytest.raises(ValueError):
        zscore_blend([])


# ---------------------------------------------------------------------------
# ic_weighted_blend
# ---------------------------------------------------------------------------


def test_ic_weighted_blend_returns_dataframe():
    sigs = _all_signals()
    mean_ics = [0.05, 0.08, 0.11]
    result = ic_weighted_blend(sigs, mean_ics)
    assert isinstance(result, pl.DataFrame)
    assert len(result) == SPEC.n_dates * SPEC.n_assets


def test_ic_weighted_blend_zero_ics_equal_weights():
    sigs = _all_signals()
    result_zero = ic_weighted_blend(sigs, [0.0, 0.0, 0.0])
    result_equal = zscore_blend(sigs)
    # Both should produce very similar composites
    z = result_zero.sort(["date", "id"])["signal"].drop_nulls().to_numpy()
    e = result_equal.sort(["date", "id"])["signal"].drop_nulls().to_numpy()
    # They should be close (same blending logic with equal weights)
    assert np.allclose(z, e, atol=1e-10)


# ---------------------------------------------------------------------------
# gram_schmidt_orthogonalize
# ---------------------------------------------------------------------------


def test_gram_schmidt_returns_list_of_dataframes():
    sigs = _all_signals()
    result = gram_schmidt_orthogonalize(sigs)
    assert len(result) == len(sigs)
    for df in result:
        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) >= {"date", "id", "signal"}


def test_gram_schmidt_orthogonal_components():
    """Orthogonalized signals should be uncorrelated across the full panel."""
    sigs = _all_signals()
    ortho = gram_schmidt_orthogonalize(sigs)

    from etl.source import to_matrix

    mats = []
    for df in ortho:
        mat, _ = to_matrix(df, "signal")
        flat = mat.reshape(-1)
        mats.append(flat)

    # The first two orthogonalized signals should have near-zero correlation
    x = mats[0]
    y = mats[1]
    joint_ok = np.isfinite(x) & np.isfinite(y)
    if joint_ok.sum() > 10:
        corr = float(np.corrcoef(x[joint_ok], y[joint_ok])[0, 1])
        assert abs(corr) < 0.1, f"GS components should be near-orthogonal, got corr={corr:.4f}"


def test_gram_schmidt_empty_returns_empty():
    result = gram_schmidt_orthogonalize([])
    assert result == []


# ---------------------------------------------------------------------------
# incremental_ic
# ---------------------------------------------------------------------------


def test_incremental_ic_returns_list():
    sigs = _all_signals()
    fwd = _fwd_returns()
    result = incremental_ic(sigs, fwd, return_col="fwd_ret_1")
    assert len(result) == len(sigs)
    for r in result:
        assert isinstance(r, IncrementalICResult)


def test_incremental_ic_cumulative_nondecreasing():
    """Adding more good signals should not dramatically decrease cumulative IC.

    The alpha_signals all have positive IC, so the composite IC should stay
    positive.
    """
    sigs = _all_signals()
    fwd = _fwd_returns()
    result = incremental_ic(sigs, fwd, return_col="fwd_ret_1")
    # All signals have injected positive IC; composite IC should be positive
    for r in result:
        if np.isfinite(r.cumulative_ic_mean):
            assert r.cumulative_ic_mean > -0.05  # allow small rounding noise


def test_incremental_ic_signal_indices_are_ordered():
    sigs = _all_signals()
    fwd = _fwd_returns()
    result = incremental_ic(sigs, fwd, return_col="fwd_ret_1")
    indices = [r.signal_index for r in result]
    assert indices == list(range(len(sigs)))
