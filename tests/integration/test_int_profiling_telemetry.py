"""Contract: telemetry schema → profiling analytics (scaling + regression).

`stage_measurements` / `stage_baselines` / `regression_thresholds` are the
inputs the profiling component turns into scaling-curve fits and a
regression verdict. This verifies that hand-off on the synthetic telemetry.
"""

from __future__ import annotations

import math

import polars as pl

from profiling import check_regressions, fit_scaling


def test_scaling_fit_over_telemetry(synth) -> None:
    measurements = synth.loader.load("stage_measurements")
    fits = fit_scaling(measurements, run_id="int_test")
    assert len(fits) > 0
    for f in fits:
        assert math.isfinite(f.log_log_slope)
        assert 0.0 <= f.r_squared <= 1.0
        assert f.n_points >= 3


def test_regression_check_over_telemetry(synth) -> None:
    measurements = synth.loader.load("stage_measurements")
    current = measurements.group_by("stage").agg(
        pl.col("elapsed_s").median(),
        pl.col("result_mb").median(),
        pl.col("peak_rss_mb").median(),
    )
    report = check_regressions(
        current,
        synth.loader.load("stage_baselines"),
        synth.loader.load("regression_thresholds"),
    )
    # report is well-formed; passed is consistent with the violation list
    assert report.passed == (len(report.violations) == 0)
