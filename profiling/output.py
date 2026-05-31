"""Structured output: JSON and Parquet serialisation of profiling results.

The existing ``print_report`` emits a human-readable table to stdout.  This
module adds machine-readable outputs so dashboards, CI artefact collectors,
and offline analysis tools can consume the same data without screen-scraping.

Two formats:
  - JSON: a single dict keyed by section (run, stages, scaling_fits,
    regressions) written to a ``.json`` file.  Dates are ISO-8601 strings;
    floats are rounded to 6 significant figures to avoid spurious precision.
  - Parquet: the same ``stage_measurements`` schema written per-run so it can
    be read directly into a DataFrame for charting or further SQL queries.

Neither format is a replacement for the storage.py append-log (which is the
authoritative historical store); these are point-in-time snapshot artefacts
suitable for CI upload.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .environment import RunEnvironment
from .regression import RegressionReport
from .scaling import ScalingFit
from .trials import TrialResult


def _round(v: float, sig: int = 6) -> float:
    """Round to sig significant figures, keeping the value JSON-serialisable."""
    import math

    if v == 0 or not math.isfinite(v):
        return v
    magnitude = math.floor(math.log10(abs(v)))
    factor = 10 ** (sig - 1 - magnitude)
    return round(v * factor) / factor


def _trial_result_to_dict(tr: TrialResult) -> dict:
    return {
        "stage": tr.stage,
        "n_trials": tr.n_trials,
        "elapsed_min": _round(tr.elapsed_min),
        "elapsed_p50": _round(tr.elapsed_p50),
        "elapsed_p90": _round(tr.elapsed_p90),
        "elapsed_p95": _round(tr.elapsed_p95),
        "elapsed_stddev": _round(tr.elapsed_stddev),
        "result_mb": _round(tr.result_mb),
        "rss_delta_mb": _round(tr.rss_delta_mb),
        "peak_rss_mb": _round(tr.peak_rss_mb),
        "peak_traced_mb": _round(tr.peak_traced_mb),
    }


def write_json(
    path: Path,
    env: RunEnvironment,
    trial_results: list[TrialResult],
    scaling_fits: list[ScalingFit] | None = None,
    regression_report: RegressionReport | None = None,
) -> None:
    """Write a structured JSON snapshot of the profiling run.

    Args:
        path: Output ``.json`` file path.  Parent directories are created.
        env: Environment metadata from ``capture_environment``.
        trial_results: Per-stage ``TrialResult`` records.
        scaling_fits: Optional list of ``ScalingFit`` records.
        regression_report: Optional regression check result.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    doc: dict = {
        "run": {
            "run_id": env.run_id,
            "run_ts": env.run_ts.isoformat(),
            "git_sha": env.git_sha,
            "git_dirty": env.git_dirty,
            "hostname": env.hostname,
            "cpu_model": env.cpu_model,
            "n_cores": env.n_cores,
            "total_ram_mb": _round(env.total_ram_mb),
            "python_version": env.python_version,
            "polars_version": env.polars_version,
            "numpy_version": env.numpy_version,
            "blas_threads": env.blas_threads,
            "trials": env.trials,
            "warmup_trials": env.warmup_trials,
        },
        "stages": [_trial_result_to_dict(tr) for tr in trial_results],
        "scaling_fits": (
            [
                {
                    "stage": f.stage,
                    "metric": f.metric,
                    "scaling_dim": f.scaling_dim,
                    "log_log_slope": _round(f.log_log_slope),
                    "intercept": _round(f.intercept),
                    "r_squared": _round(f.r_squared),
                    "n_points": f.n_points,
                }
                for f in scaling_fits
            ]
            if scaling_fits is not None
            else []
        ),
        "regression": (
            {
                "passed": regression_report.passed,
                "scaling_fit_confidence_ok": regression_report.scaling_fit_confidence_ok,
                "excluded_low_confidence": [
                    [stage, metric] for stage, metric in regression_report.excluded_low_confidence
                ],
                "violations": [
                    {
                        "stage": v.stage,
                        "metric": v.metric,
                        "baseline_value": _round(v.baseline_value),
                        "current_value": _round(v.current_value),
                        "pct_increase": _round(v.pct_increase),
                        "abs_increase": _round(v.abs_increase),
                        "triggered_by": v.triggered_by,
                    }
                    for v in regression_report.violations
                ],
            }
            if regression_report is not None
            else None
        ),
    }

    path.write_text(json.dumps(doc, indent=2))


def write_measurements_parquet(
    path: Path,
    run_id: str,
    param_point_id: int,
    params: dict[str, int],
    trial_results: list[TrialResult],
) -> None:
    """Write per-trial measurements to a Parquet file (``stage_measurements`` schema).

    Args:
        path: Output ``.parquet`` file path.  Parent directories are created.
        run_id: Run identifier (same as env.run_id).
        param_point_id: Integer index of the param grid point.
        params: Dict with keys ``n_assets``, ``n_dates``, ``n_features``,
            ``n_factors``.
        trial_results: List of ``TrialResult`` from ``run_trials``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for tr in trial_results:
        for m in tr.trials:
            rows.append(
                {
                    "run_id": run_id,
                    "param_point_id": param_point_id,
                    "n_assets": params.get("n_assets", 0),
                    "n_dates": params.get("n_dates", 0),
                    "n_features": params.get("n_features", 0),
                    "n_factors": params.get("n_factors", 0),
                    "stage": tr.stage,
                    "trial_idx": m.trial_idx,
                    "elapsed_s": m.elapsed_s,
                    "result_mb": m.result_mb,
                    "rss_delta_mb": m.rss_delta_mb,
                    "peak_rss_mb": m.peak_rss_mb,
                    "peak_traced_mb": m.peak_traced_mb,
                }
            )

    if not rows:
        return

    df = pl.DataFrame(rows).with_columns(
        pl.col("param_point_id").cast(pl.Int64),
        pl.col("n_assets").cast(pl.Int64),
        pl.col("n_dates").cast(pl.Int64),
        pl.col("n_features").cast(pl.Int64),
        pl.col("n_factors").cast(pl.Int64),
        pl.col("stage").cast(pl.Categorical),
        pl.col("trial_idx").cast(pl.Int64),
    )
    df.write_parquet(path)
