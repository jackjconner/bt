"""Tests for output.py — JSON and Parquet structured output."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from profiling.environment import capture_environment
from profiling.output import write_json, write_measurements_parquet
from profiling.regression import RegressionReport, RegressionViolation
from profiling.scaling import ScalingFit
from profiling.trials import TrialMeasurement, TrialResult


def _make_trial_result(stage: str = "etl.batch", n: int = 3) -> TrialResult:
    measurements = tuple(
        TrialMeasurement(
            trial_idx=i,
            elapsed_s=0.1 * (i + 1),
            result_mb=10.0,
            rss_delta_mb=2.0,
            peak_rss_mb=500.0,
            peak_traced_mb=5.0,
        )
        for i in range(n)
    )
    import numpy as np

    elapsed = np.array([m.elapsed_s for m in measurements])
    return TrialResult(
        stage=stage,
        n_trials=n,
        elapsed_min=float(elapsed.min()),
        elapsed_p50=float(np.percentile(elapsed, 50)),
        elapsed_p90=float(np.percentile(elapsed, 90)),
        elapsed_p95=float(np.percentile(elapsed, 95)),
        elapsed_stddev=float(elapsed.std(ddof=1) if n > 1 else 0.0),
        result_mb=10.0,
        rss_delta_mb=2.0,
        peak_rss_mb=500.0,
        peak_traced_mb=5.0,
        trials=measurements,
    )


def test_write_json_creates_file(tmp_path: Path) -> None:
    env = capture_environment("run-json-001")
    out = tmp_path / "report.json"
    write_json(out, env, [_make_trial_result()])
    assert out.exists()


def test_write_json_valid_structure(tmp_path: Path) -> None:
    env = capture_environment("run-json-002")
    out = tmp_path / "report.json"
    write_json(out, env, [_make_trial_result("etl.batch"), _make_trial_result("backtest")])

    doc = json.loads(out.read_text())
    assert "run" in doc
    assert doc["run"]["run_id"] == "run-json-002"
    assert len(doc["stages"]) == 2
    assert doc["stages"][0]["stage"] == "etl.batch"
    # Both sets of percentiles present
    assert "elapsed_p50" in doc["stages"][0]
    assert "elapsed_p90" in doc["stages"][0]


def test_write_json_with_scaling_fits(tmp_path: Path) -> None:
    env = capture_environment("run-json-003")
    fits = [
        ScalingFit(
            run_id="run-json-003",
            stage="etl.batch",
            metric="elapsed_s",
            scaling_dim="n_assets",
            log_log_slope=1.05,
            intercept=-7.0,
            r_squared=0.999,
            n_points=6,
        )
    ]
    out = tmp_path / "report.json"
    write_json(out, env, [_make_trial_result()], scaling_fits=fits)
    doc = json.loads(out.read_text())
    assert len(doc["scaling_fits"]) == 1
    assert abs(doc["scaling_fits"][0]["log_log_slope"] - 1.05) < 1e-4


def test_write_json_with_regression_report(tmp_path: Path) -> None:
    env = capture_environment("run-json-004")
    violation = RegressionViolation(
        stage="etl.batch",
        metric="elapsed_s",
        baseline_value=1.0,
        current_value=1.5,
        pct_increase=0.5,
        abs_increase=0.5,
        triggered_by="pct",
    )
    report = RegressionReport(passed=False, violations=(violation,))
    out = tmp_path / "report.json"
    write_json(out, env, [_make_trial_result()], regression_report=report)
    doc = json.loads(out.read_text())
    assert doc["regression"]["passed"] is False
    assert len(doc["regression"]["violations"]) == 1


def test_write_measurements_parquet(tmp_path: Path) -> None:
    params = {"n_assets": 100, "n_dates": 252, "n_features": 20, "n_factors": 5}
    out = tmp_path / "meas.parquet"
    write_measurements_parquet(
        out,
        run_id="run-pq-001",
        param_point_id=0,
        params=params,
        trial_results=[_make_trial_result("etl.batch", 3), _make_trial_result("backtest", 3)],
    )
    assert out.exists()
    df = pl.read_parquet(out)
    assert len(df) == 6  # 3 trials × 2 stages
    assert set(df["stage"].cast(pl.String).to_list()) == {"etl.batch", "backtest"}
    assert df["n_assets"].to_list() == [100] * 6


def test_write_json_creates_parent_dirs(tmp_path: Path) -> None:
    env = capture_environment("r")
    out = tmp_path / "nested" / "deep" / "report.json"
    write_json(out, env, [])
    assert out.exists()
