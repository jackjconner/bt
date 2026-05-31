"""Tests for scaling.py — log-log slope fitting."""

from __future__ import annotations

import numpy as np
import polars as pl

from profiling.scaling import fit_scaling, fits_to_dataframe


def _linear_measurements(n_points: int = 6) -> pl.DataFrame:
    """Synthetic measurements where elapsed_s scales linearly with n_assets.

    elapsed_s = 0.001 * n_assets  →  log-log slope ≈ 1.0
    """
    n_assets_vals = np.array([100, 200, 400, 800, 1600, 3200], dtype=float)[:n_points]
    elapsed = 0.001 * n_assets_vals
    return pl.DataFrame(
        {
            "stage": pl.Series(["etl.batch"] * n_points, dtype=pl.Categorical),
            "param_point_id": list(range(n_points)),
            "n_assets": n_assets_vals.astype(int).tolist(),
            "n_dates": [252] * n_points,
            "n_features": [20] * n_points,
            "n_factors": [5] * n_points,
            "elapsed_s": elapsed.tolist(),
            "peak_rss_mb": (10.0 * n_assets_vals).tolist(),
        }
    )


def _quadratic_measurements(n_points: int = 6) -> pl.DataFrame:
    """elapsed_s = n_assets^2 * 1e-6  →  log-log slope ≈ 2.0."""
    n_assets_vals = np.array([100, 200, 400, 800, 1600, 3200], dtype=float)[:n_points]
    elapsed = (n_assets_vals**2) * 1e-6
    return pl.DataFrame(
        {
            "stage": pl.Series(["backtest"] * n_points, dtype=pl.Categorical),
            "param_point_id": list(range(n_points)),
            "n_assets": n_assets_vals.astype(int).tolist(),
            "n_dates": [252] * n_points,
            "n_features": [20] * n_points,
            "n_factors": [5] * n_points,
            "elapsed_s": elapsed.tolist(),
            "peak_rss_mb": (100.0 * n_assets_vals).tolist(),
        }
    )


def test_linear_slope_approx_one() -> None:
    """A perfectly linear relationship must yield slope ≈ 1.0."""
    fits = fit_scaling(_linear_measurements(), run_id="test")
    n_assets_fits = [
        f
        for f in fits
        if f.scaling_dim == "n_assets" and f.stage == "etl.batch" and f.metric == "elapsed_s"
    ]
    assert len(n_assets_fits) == 1
    fit = n_assets_fits[0]
    assert abs(fit.log_log_slope - 1.0) < 0.01
    assert fit.r_squared > 0.999


def test_quadratic_slope_approx_two() -> None:
    """A quadratic relationship must yield slope ≈ 2.0."""
    fits = fit_scaling(_quadratic_measurements(), run_id="test")
    n_assets_fits = [
        f
        for f in fits
        if f.scaling_dim == "n_assets" and f.stage == "backtest" and f.metric == "elapsed_s"
    ]
    assert len(n_assets_fits) == 1
    fit = n_assets_fits[0]
    assert abs(fit.log_log_slope - 2.0) < 0.02


def test_r_squared_near_one_for_perfect_data() -> None:
    fits = fit_scaling(_linear_measurements(), run_id="test")
    for f in fits:
        if f.scaling_dim == "n_assets":
            assert f.r_squared > 0.99


def test_min_points_filter() -> None:
    """With only 2 data points (below min_points=3) no fit should be returned."""
    fits = fit_scaling(_linear_measurements(n_points=2), run_id="test", min_points=3)
    assert len(fits) == 0


def test_fits_to_dataframe_schema() -> None:
    """fits_to_dataframe must produce the expected column set."""
    fits = fit_scaling(_linear_measurements(), run_id="test")
    df = fits_to_dataframe(fits)
    expected_cols = {
        "run_id",
        "stage",
        "metric",
        "scaling_dim",
        "log_log_slope",
        "intercept",
        "r_squared",
        "n_points",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == len(fits)


def test_fits_to_dataframe_empty() -> None:
    df = fits_to_dataframe([])
    assert len(df) == 0
    assert "log_log_slope" in df.columns


def test_multiple_stages_and_metrics() -> None:
    combined = pl.concat([_linear_measurements(), _quadratic_measurements()])
    fits = fit_scaling(combined, run_id="test", metrics=("elapsed_s", "peak_rss_mb"))
    stages = {f.stage for f in fits}
    assert "etl.batch" in stages
    assert "backtest" in stages
    metrics = {f.metric for f in fits}
    assert "elapsed_s" in metrics
    assert "peak_rss_mb" in metrics


def test_n_points_correct() -> None:
    fits = fit_scaling(_linear_measurements(6), run_id="test")
    n_assets_fits = [f for f in fits if f.scaling_dim == "n_assets"]
    assert all(f.n_points == 6 for f in n_assets_fits)
