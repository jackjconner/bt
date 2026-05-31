"""Tests for storage.py — Parquet persistence of runs and measurements."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from profiling.environment import capture_environment
from profiling.storage import read_measurements, read_runs, write_run
from profiling.trials import TrialMeasurement


def _make_measurements(n: int = 3) -> list[TrialMeasurement]:
    return [
        TrialMeasurement(
            trial_idx=i,
            elapsed_s=0.1 * (i + 1),
            result_mb=10.0,
            rss_delta_mb=2.0,
            peak_rss_mb=500.0,
            peak_traced_mb=5.0,
        )
        for i in range(n)
    ]


def test_write_then_read_runs(tmp_path: Path) -> None:
    env = capture_environment("run-001", trials=3, warmup_trials=1)
    write_run(tmp_path, env, [])

    runs = read_runs(tmp_path)
    assert len(runs) == 1
    assert runs["run_id"][0] == "run-001"
    assert runs["trials"][0] == 3
    assert runs["warmup_trials"][0] == 1


def test_write_then_read_measurements(tmp_path: Path) -> None:
    env = capture_environment("run-002", trials=3)
    params = {"n_assets": 100, "n_dates": 252, "n_features": 20, "n_factors": 5}
    measurements = _make_measurements(3)
    trial_results = [(0, params, "etl.batch", measurements)]

    write_run(tmp_path, env, trial_results)

    meas = read_measurements(tmp_path)
    assert len(meas) == 3
    assert set(meas["stage"].cast(pl.String).to_list()) == {"etl.batch"}
    assert meas["run_id"].to_list() == ["run-002"] * 3


def test_append_accumulates_rows(tmp_path: Path) -> None:
    """Two calls to write_run must accumulate, not overwrite."""
    params = {"n_assets": 100, "n_dates": 252, "n_features": 20, "n_factors": 5}

    env1 = capture_environment("run-A", trials=2)
    write_run(tmp_path, env1, [(0, params, "etl.batch", _make_measurements(2))])

    env2 = capture_environment("run-B", trials=2)
    write_run(tmp_path, env2, [(0, params, "etl.batch", _make_measurements(2))])

    runs = read_runs(tmp_path)
    assert len(runs) == 2
    assert set(runs["run_id"].to_list()) == {"run-A", "run-B"}

    meas = read_measurements(tmp_path)
    assert len(meas) == 4


def test_read_empty_store_returns_empty_frames(tmp_path: Path) -> None:
    runs = read_runs(tmp_path)
    meas = read_measurements(tmp_path)
    assert len(runs) == 0
    assert len(meas) == 0


def test_measurements_params_stored_correctly(tmp_path: Path) -> None:
    env = capture_environment("run-003")
    params = {"n_assets": 500, "n_dates": 504, "n_features": 50, "n_factors": 10}
    write_run(tmp_path, env, [(1, params, "backtest", _make_measurements(1))])

    meas = read_measurements(tmp_path)
    row = meas.row(0, named=True)
    assert row["n_assets"] == 500
    assert row["n_dates"] == 504
    assert row["param_point_id"] == 1
