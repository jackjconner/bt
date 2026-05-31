"""Tests for scaling.py — log-log slope fitting."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from profiling.scaling import ScalingFit, fit_scaling, fits_to_dataframe, stage_metric_r_squared


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


def _anchored_two_axis_measurements() -> pl.DataFrame:
    """Anchored grid varying n_assets and n_dates on separate axes.

    elapsed_s = 1e-9 * n_assets^2 * n_dates  — quadratic in assets, linear in
    dates. With confounder control, the n_assets fit (held at the modal n_dates)
    must recover slope ≈ 2 and the n_dates fit (held at the modal n_assets) ≈ 1.
    A naive pooled fit would blend the date-sweep points into the n_assets=100
    column and corrupt both slopes.
    """
    # baseline + n_assets sweep @ 252d, then n_dates sweep @ 100 assets
    points = [
        (50, 252),
        (100, 252),  # baseline (modal n_assets=100, modal n_dates=252)
        (200, 252),
        (100, 756),
        (100, 1260),
        (100, 2016),
        (100, 5040),
    ]
    n_assets = [a for a, _ in points]
    n_dates = [d for _, d in points]
    elapsed = [1e-9 * a**2 * d for a, d in points]
    return pl.DataFrame(
        {
            "stage": pl.Series(["signals"] * len(points), dtype=pl.Categorical),
            "param_point_id": list(range(len(points))),
            "n_assets": n_assets,
            "n_dates": n_dates,
            "n_features": [10] * len(points),
            "n_factors": [4] * len(points),
            "elapsed_s": elapsed,
            "peak_rss_mb": [1.0] * len(points),
        }
    )


def _memory_measurements(n_points: int = 4) -> pl.DataFrame:
    """elapsed/result/traced each scale linearly with n_assets (slope ≈ 1)."""
    n_assets_vals = np.array([100, 200, 400, 800], dtype=float)[:n_points]
    return pl.DataFrame(
        {
            "stage": pl.Series(["models"] * n_points, dtype=pl.Categorical),
            "param_point_id": list(range(n_points)),
            "n_assets": n_assets_vals.astype(int).tolist(),
            "n_dates": [252] * n_points,
            "n_features": [10] * n_points,
            "n_factors": [4] * n_points,
            "elapsed_s": (0.001 * n_assets_vals).tolist(),
            "result_mb": (0.5 * n_assets_vals).tolist(),
            "peak_traced_mb": (2.0 * n_assets_vals).tolist(),
            "peak_rss_mb": (250.0 + 0.01 * n_assets_vals).tolist(),
        }
    )


def test_fits_memory_metrics_by_default() -> None:
    """result_mb and peak_traced_mb are fit by default when present."""
    fits = fit_scaling(_memory_measurements(), run_id="test")
    fit_keys = {(f.metric, f.scaling_dim) for f in fits}
    assert ("result_mb", "n_assets") in fit_keys
    assert ("peak_traced_mb", "n_assets") in fit_keys
    result_fit = next(f for f in fits if f.metric == "result_mb" and f.scaling_dim == "n_assets")
    assert result_fit.log_log_slope == pytest.approx(1.0, abs=0.05)


def _fit(stage: str, metric: str, dim: str, r_squared: float) -> ScalingFit:
    return ScalingFit(
        run_id="test",
        stage=stage,
        metric=metric,
        scaling_dim=dim,
        log_log_slope=1.0,
        intercept=0.0,
        r_squared=r_squared,
        n_points=4,
    )


def test_stage_metric_r_squared_takes_max_across_dims() -> None:
    """Per (stage, metric) confidence is the best r² across scaling dims."""
    fits = [
        _fit("etl.batch", "elapsed_s", "n_assets", 0.40),
        _fit("etl.batch", "elapsed_s", "n_dates", 0.95),
        _fit("etl.batch", "result_mb", "n_assets", 0.10),
    ]
    conf = stage_metric_r_squared(fits)
    assert conf[("etl.batch", "elapsed_s")] == pytest.approx(0.95)
    assert conf[("etl.batch", "result_mb")] == pytest.approx(0.10)


def test_stage_metric_r_squared_empty() -> None:
    assert stage_metric_r_squared([]) == {}


def test_confounder_control_isolates_each_axis() -> None:
    """On a two-axis grid each dim's slope is fit holding the others at baseline."""
    fits = fit_scaling(_anchored_two_axis_measurements(), run_id="test")
    by_dim = {f.scaling_dim: f for f in fits if f.metric == "elapsed_s"}
    # n_assets sweep is at the modal n_dates (252): elapsed ∝ n_assets^2
    assert by_dim["n_assets"].log_log_slope == pytest.approx(2.0, abs=0.05)
    assert by_dim["n_assets"].n_points == 3
    # n_dates sweep is at the modal n_assets (100): elapsed ∝ n_dates^1
    assert by_dim["n_dates"].log_log_slope == pytest.approx(1.0, abs=0.05)
    assert by_dim["n_dates"].n_points == 5
