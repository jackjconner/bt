"""Persistent metrics storage: write and read Parquet files.

Two tables are persisted:
  - ``profiling_runs`` — one row per run (environment metadata).
  - ``stage_measurements`` — one row per (run, param_point, stage, trial).

The Parquet layout is append-friendly: each call to ``write_run`` produces two
files under ``store_dir / {table_name}.parquet``.  When a file already exists
the new rows are concatenated with the existing data and the file is
re-written.  This is intentionally simple; for large histories a partitioned
layout (Hive by date) would be preferable, but simplicity beats premature
optimisation here because profiling runs are infrequent.

Column dtypes follow the ``profiling_runs`` / ``stage_measurements`` schemas in
etl.datasets exactly so the stored Parquet can be read back by the synthetic-
data tooling and compared against baselines.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .environment import RunEnvironment
from .trials import TrialMeasurement

# Canonical dtypes matching etl.datasets schemas so stored files are schema-
# compatible with the synthetic generator's output.
_RUNS_SCHEMA: dict[str, type[pl.DataType]] = {
    "run_id": pl.String,
    "run_ts": pl.Date,
    "git_sha": pl.String,
    "git_dirty": pl.Boolean,
    "hostname": pl.String,
    "cpu_model": pl.String,
    "n_cores": pl.Int64,
    "total_ram_mb": pl.Float64,
    "python_version": pl.String,
    "polars_version": pl.String,
    "numpy_version": pl.String,
    "blas_threads": pl.Int64,
    "trials": pl.Int64,
    "warmup_trials": pl.Int64,
}

_MEASUREMENTS_SCHEMA: dict[str, type[pl.DataType]] = {
    "run_id": pl.String,
    "param_point_id": pl.Int64,
    "n_assets": pl.Int64,
    "n_dates": pl.Int64,
    "n_features": pl.Int64,
    "n_factors": pl.Int64,
    "stage": pl.Categorical,
    "trial_idx": pl.Int64,
    "elapsed_s": pl.Float64,
    "result_mb": pl.Float64,
    "rss_delta_mb": pl.Float64,
    "peak_rss_mb": pl.Float64,
    "peak_traced_mb": pl.Float64,
}


def _cast_schema(df: pl.DataFrame, schema: dict[str, type[pl.DataType]]) -> pl.DataFrame:
    return df.with_columns(
        pl.col(name).cast(dtype) for name, dtype in schema.items() if name in df.columns
    )


def _upsert_parquet(path: Path, new_rows: pl.DataFrame) -> None:
    """Append new_rows to the Parquet at path, creating it if absent."""
    if path.exists():
        existing = pl.read_parquet(path)
        combined = pl.concat([existing, new_rows], how="diagonal_relaxed")
    else:
        combined = new_rows
    combined.write_parquet(path)


def write_run(
    store_dir: Path,
    env: RunEnvironment,
    trial_results: list[tuple[int, dict[str, int], str, list[TrialMeasurement]]],
) -> None:
    """Persist one profiling run to Parquet.

    Args:
        store_dir: Directory where Parquet files are stored.  Created if absent.
        env: The ``RunEnvironment`` captured by ``environment.capture_environment``.
        trial_results: List of ``(param_point_id, params, stage, measurements)``.
            ``params`` must contain the keys ``n_assets``, ``n_dates``,
            ``n_features``, ``n_factors`` that appear in ``stage_measurements``.

    Files written:
        ``store_dir/profiling_runs.parquet``
        ``store_dir/stage_measurements.parquet``
    """
    store_dir.mkdir(parents=True, exist_ok=True)

    # --- profiling_runs row ---
    run_row: dict[str, object] = {
        "run_id": env.run_id,
        "run_ts": env.run_ts,
        "git_sha": env.git_sha,
        "git_dirty": env.git_dirty,
        "hostname": env.hostname,
        "cpu_model": env.cpu_model,
        "n_cores": env.n_cores,
        "total_ram_mb": env.total_ram_mb,
        "python_version": env.python_version,
        "polars_version": env.polars_version,
        "numpy_version": env.numpy_version,
        "blas_threads": env.blas_threads,
        "trials": env.trials,
        "warmup_trials": env.warmup_trials,
    }
    runs_df = _cast_schema(pl.DataFrame([run_row]), _RUNS_SCHEMA)
    _upsert_parquet(store_dir / "profiling_runs.parquet", runs_df)

    # --- stage_measurements rows ---
    meas_rows: list[dict[str, object]] = []
    for param_point_id, params, stage, measurements in trial_results:
        for m in measurements:
            meas_rows.append(
                {
                    "run_id": env.run_id,
                    "param_point_id": param_point_id,
                    "n_assets": params.get("n_assets", 0),
                    "n_dates": params.get("n_dates", 0),
                    "n_features": params.get("n_features", 0),
                    "n_factors": params.get("n_factors", 0),
                    "stage": stage,
                    "trial_idx": m.trial_idx,
                    "elapsed_s": m.elapsed_s,
                    "result_mb": m.result_mb,
                    "rss_delta_mb": m.rss_delta_mb,
                    "peak_rss_mb": m.peak_rss_mb,
                    "peak_traced_mb": m.peak_traced_mb,
                }
            )

    if meas_rows:
        meas_df = _cast_schema(pl.DataFrame(meas_rows), _MEASUREMENTS_SCHEMA)
        _upsert_parquet(store_dir / "stage_measurements.parquet", meas_df)


def read_runs(store_dir: Path) -> pl.DataFrame:
    """Read all stored profiling-run metadata rows."""
    path = store_dir / "profiling_runs.parquet"
    if not path.exists():
        return pl.DataFrame(schema={k: v for k, v in _RUNS_SCHEMA.items()})
    return pl.read_parquet(path)


def read_measurements(store_dir: Path) -> pl.DataFrame:
    """Read all stored per-trial stage measurements."""
    path = store_dir / "stage_measurements.parquet"
    if not path.exists():
        return pl.DataFrame(schema={k: v for k, v in _MEASUREMENTS_SCHEMA.items()})
    return pl.read_parquet(path)
