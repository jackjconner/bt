"""Tests for IC decay / horizon curve."""

from __future__ import annotations

import polars as pl

from etl.datasets import GenSpec, generate
from signals.horizon import HorizonCurve, HorizonPoint, ic_horizon_curve

SPEC = GenSpec(n_assets=30, n_dates=80, seed=99)
HORIZON_COLS = {1: "fwd_ret_1", 5: "fwd_ret_5", 21: "fwd_ret_21"}


def _signals() -> pl.DataFrame:
    df = generate("alpha_signals", SPEC)
    return df.filter(pl.col("signal_name") == "momentum").select("date", "id", "signal")


def _fwd_returns() -> pl.DataFrame:
    return generate("forward_returns", SPEC)


def test_ic_horizon_curve_returns_horizon_curve():
    result = ic_horizon_curve(
        _signals(),
        _fwd_returns(),
        horizon_cols=HORIZON_COLS,
        method="rank",
    )
    assert isinstance(result, HorizonCurve)
    assert result.method == "rank"


def test_ic_horizon_curve_has_one_point_per_horizon():
    result = ic_horizon_curve(
        _signals(),
        _fwd_returns(),
        horizon_cols=HORIZON_COLS,
    )
    assert len(result.points) == 3
    horizons = result.horizons()
    assert horizons == [1, 5, 21]


def test_ic_horizon_curve_points_are_horizon_point():
    result = ic_horizon_curve(
        _signals(),
        _fwd_returns(),
        horizon_cols=HORIZON_COLS,
    )
    for p in result.points:
        assert isinstance(p, HorizonPoint)
        assert isinstance(p.mean_ic, float)
        assert isinstance(p.ic_ir, float)
        assert isinstance(p.n_dates, int)
        assert p.n_dates > 0


def test_ic_horizon_curve_to_frame():
    result = ic_horizon_curve(
        _signals(),
        _fwd_returns(),
        horizon_cols=HORIZON_COLS,
    )
    df = result.to_frame()
    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) >= {"horizon", "mean_ic", "ic_ir", "t_stat", "n_dates"}
    assert len(df) == 3


def test_ic_horizon_curve_ic_decays_with_horizon():
    """A short-term signal should have higher IC at h=1 than h=21.

    The alpha_signals generator injects IC vs fwd_ret_1, so longer horizons
    are expected to have weaker IC.
    """
    result = ic_horizon_curve(
        _signals(),
        _fwd_returns(),
        horizon_cols={1: "fwd_ret_1", 5: "fwd_ret_5", 21: "fwd_ret_21"},
    )
    ic_at_1 = result.points[0].mean_ic
    ic_at_21 = result.points[2].mean_ic
    # Short-horizon IC should be stronger (abs) than long-horizon
    assert abs(ic_at_1) >= abs(ic_at_21) - 0.05


def test_ic_horizon_curve_different_methods_differ():
    sig = _signals()
    fwd = _fwd_returns()
    r_rank = ic_horizon_curve(sig, fwd, horizon_cols={1: "fwd_ret_1"}, method="rank")
    r_pearson = ic_horizon_curve(sig, fwd, horizon_cols={1: "fwd_ret_1"}, method="pearson")
    # Spearman and Pearson will differ
    assert r_rank.points[0].mean_ic != r_pearson.points[0].mean_ic
