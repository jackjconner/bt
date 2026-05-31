"""Tests for the lazy/streaming Polars Spearman IC engine.

The contract of ``engine="lazy"`` is that it is *bit-identical* to the
incumbent ``engine="matrix"`` path for the rank method — same per-date IC,
same date axis, same n_obs — while never pivoting to a dense matrix.  These
tests pin that equivalence over the data shapes the harness exercises:
clean panels, panels with trailing-horizon NaN forward returns, sparse
(irregular-NaN) panels, and the min_obs suppression boundary.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from etl.datasets import GenSpec, generate
from signals import ic_horizon_curve, ic_series_v2, spearman_ic_lazy

SPEC = GenSpec(n_assets=40, n_dates=80, seed=7)
HORIZONS = {1: "fwd_ret_1", 5: "fwd_ret_5", 21: "fwd_ret_21", 63: "fwd_ret_63"}


def _momentum() -> pl.DataFrame:
    return (
        generate("alpha_signals", SPEC)
        .filter(pl.col("signal_name") == "momentum")
        .select("date", "id", "signal")
    )


def _fwd() -> pl.DataFrame:
    return generate("forward_returns", SPEC)


def _assert_ic_frames_identical(a: pl.DataFrame, b: pl.DataFrame) -> None:
    assert a.columns == b.columns
    assert a.schema == b.schema
    assert a["date"].to_list() == b["date"].to_list()
    assert a["n_obs"].to_list() == b["n_obs"].to_list()
    av = a["ic"].fill_null(np.nan).to_numpy()
    bv = b["ic"].fill_null(np.nan).to_numpy()
    # nan positions coincide
    np.testing.assert_array_equal(np.isnan(av), np.isnan(bv))
    np.testing.assert_array_equal(av[~np.isnan(av)], bv[~np.isnan(bv)])


# ---------------------------------------------------------------------------
# ic_series_v2: lazy == matrix, bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ret_col", ["fwd_ret_1", "fwd_ret_5", "fwd_ret_21", "fwd_ret_63"])
@pytest.mark.parametrize("min_obs", [2, 10, 30])
def test_lazy_matches_matrix_ic_series(ret_col: str, min_obs: int) -> None:
    sig, fwd = _momentum(), _fwd()
    lazy = ic_series_v2(sig, fwd, return_col=ret_col, min_obs=min_obs, engine="lazy")
    matrix = ic_series_v2(sig, fwd, return_col=ret_col, min_obs=min_obs, engine="matrix")
    _assert_ic_frames_identical(lazy, matrix)


def test_lazy_is_the_default_engine() -> None:
    sig, fwd = _momentum(), _fwd()
    default = ic_series_v2(sig, fwd, return_col="fwd_ret_1")
    lazy = ic_series_v2(sig, fwd, return_col="fwd_ret_1", engine="lazy")
    _assert_ic_frames_identical(default, lazy)


def test_lazy_pearson_kendall_unaffected_by_engine() -> None:
    """engine is a no-op for non-rank methods — same result either way."""
    sig, fwd = _momentum(), _fwd()
    for method in ("pearson", "kendall"):
        a = ic_series_v2(sig, fwd, return_col="fwd_ret_1", method=method, engine="lazy")
        b = ic_series_v2(sig, fwd, return_col="fwd_ret_1", method=method, engine="matrix")
        _assert_ic_frames_identical(a, b)


# ---------------------------------------------------------------------------
# spearman_ic_lazy direct: NaN handling and the date axis
# ---------------------------------------------------------------------------


def test_lazy_emits_intersection_date_axis_with_tail_nan() -> None:
    """A fully-NaN trailing horizon date is emitted with null IC and n_obs=0."""
    sig, fwd = _momentum(), _fwd()
    out = spearman_ic_lazy(sig, fwd, signal_col="signal", return_col="fwd_ret_63", min_obs=2)
    # fwd_ret_63 is NaN on the last 63 dates; those dates remain in the axis
    # (intersection of sig & fwd dates) with a null IC.
    assert out["date"].to_list() == sorted(out["date"].to_list())
    tail = out.tail(63)
    assert tail["ic"].null_count() == 63
    assert tail["n_obs"].to_list() == [0] * 63


def test_lazy_irregular_nan_matches_matrix() -> None:
    """Scattered NaN (not just the tail) still matches the matrix path."""
    sig, fwd = _momentum(), _fwd()
    rng = np.random.default_rng(123)
    drop = rng.random(fwd.height) < 0.2
    fwd_holed = fwd.with_columns(
        pl.when(pl.Series(drop)).then(None).otherwise(pl.col("fwd_ret_1")).alias("fwd_ret_1")
    )
    lazy = ic_series_v2(sig, fwd_holed, return_col="fwd_ret_1", min_obs=2, engine="lazy")
    matrix = ic_series_v2(sig, fwd_holed, return_col="fwd_ret_1", min_obs=2, engine="matrix")
    _assert_ic_frames_identical(lazy, matrix)


def test_lazy_constant_signal_yields_null_ic() -> None:
    """A date where the signal has zero rank-variance produces a null IC."""
    sig, fwd = _momentum(), _fwd()
    const = sig.with_columns(pl.lit(1.0).alias("signal"))
    out = spearman_ic_lazy(const, fwd, signal_col="signal", return_col="fwd_ret_1", min_obs=2)
    assert out["ic"].null_count() == out.height


# ---------------------------------------------------------------------------
# ic_horizon_curve: lazy == matrix across the whole curve
# ---------------------------------------------------------------------------


def test_horizon_curve_lazy_matches_matrix() -> None:
    sig, fwd = _momentum(), _fwd()
    lazy = ic_horizon_curve(sig, fwd, HORIZONS, min_obs=5, engine="lazy")
    matrix = ic_horizon_curve(sig, fwd, HORIZONS, min_obs=5, engine="matrix")
    assert lazy.horizons() == matrix.horizons()
    for pl_pt, pm_pt in zip(lazy.points, matrix.points, strict=True):
        assert pl_pt.n_dates == pm_pt.n_dates
        np.testing.assert_allclose(pl_pt.mean_ic, pm_pt.mean_ic, rtol=0, atol=0)
        np.testing.assert_allclose(pl_pt.ic_ir, pm_pt.ic_ir, rtol=0, atol=0)
        np.testing.assert_allclose(pl_pt.t_stat, pm_pt.t_stat, rtol=0, atol=0)


def test_horizon_curve_lazy_is_default() -> None:
    sig, fwd = _momentum(), _fwd()
    default = ic_horizon_curve(sig, fwd, HORIZONS, min_obs=5)
    lazy = ic_horizon_curve(sig, fwd, HORIZONS, min_obs=5, engine="lazy")
    for pd_pt, pl_pt in zip(default.points, lazy.points, strict=True):
        np.testing.assert_array_equal(pd_pt.mean_ic, pl_pt.mean_ic)
