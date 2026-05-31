"""Three-table append-log for agentic profiling history.

Tables (Parquet, append-friendly):
  improvement_runs     — one row per run: what the agent did and why
  component_snapshots  — one row per (run, component): how each component performed
  agent_annotations    — one row per run: what the agent learned

The "e2e" component in snapshots is synthetic: p50/p90/min are sums across
all real component rows; stddev is sqrt(sum of variances) (assumes independence);
memory stats are sums. Scaling slope/r2 are null for e2e.

Delta fields (vs_prev_p50_pct, vs_baseline_p50_pct) are computed at write time
by reading the existing history before appending the new run:
  - prev    = most recent run already in history (by run_ts)
  - baseline = oldest run in history (by run_ts)
"""

from __future__ import annotations

import datetime
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from profiling.regression import RegressionReport
from profiling.scaling import ScalingFit
from profiling.trials import TrialResult

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentContext:
    """What the agent was trying to do this run."""

    agent_id: str
    goal: str
    strategy: str
    target_component: str | None = None  # None = full-pipeline sweep
    parent_run_id: str | None = None


@dataclass(frozen=True)
class AgentAnnotation:
    """What the agent learned and what the next agent should know."""

    hypothesis: str
    outcome: str
    lessons: str
    next_target: str
    confidence: float  # 0.0–1.0
    improvement_type: str  # algorithmic|vectorization|caching|memory|io|other


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_RUNS_SCHEMA: dict[str, type] = {
    "run_id": pl.String,
    "run_ts": pl.Datetime,
    "git_sha": pl.String,
    "branch": pl.String,
    "commit_message": pl.String,
    "agent_id": pl.String,
    "target_component": pl.String,
    "goal": pl.String,
    "strategy": pl.String,
    "e2e_p50_s": pl.Float64,
    "e2e_vs_prev_pct": pl.Float64,
    "regression_passed": pl.Boolean,
    "n_violations": pl.Int64,
    "parent_run_id": pl.String,
}

_SNAPSHOTS_SCHEMA: dict[str, type] = {
    "run_id": pl.String,
    "component": pl.Categorical,
    "p50_ms": pl.Float64,
    "p90_ms": pl.Float64,
    "min_ms": pl.Float64,
    "stddev_ms": pl.Float64,
    "peak_rss_mb": pl.Float64,
    "result_mb": pl.Float64,
    "scaling_slope": pl.Float64,
    "scaling_r2": pl.Float64,
    "vs_prev_p50_pct": pl.Float64,
    "vs_baseline_p50_pct": pl.Float64,
}

_ANNOTATIONS_SCHEMA: dict[str, type] = {
    "run_id": pl.String,
    "hypothesis": pl.String,
    "outcome": pl.String,
    "lessons": pl.String,
    "next_target": pl.String,
    "confidence": pl.Float64,
    "improvement_type": pl.Categorical,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _git_commit_message() -> str:
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def _cast_schema(df: pl.DataFrame, schema: dict[str, type]) -> pl.DataFrame:
    return df.with_columns(
        pl.col(name).cast(dtype) for name, dtype in schema.items() if name in df.columns
    )


def _upsert_parquet(path: Path, new_rows: pl.DataFrame) -> None:
    if path.exists():
        existing = pl.read_parquet(path)
        combined = pl.concat([existing, new_rows], how="diagonal_relaxed")
    else:
        combined = new_rows
    combined.write_parquet(path)


def _delta_pct(
    component: str, current_p50_ms: float, ref: pl.DataFrame | None
) -> float | None:
    if ref is None or ref.is_empty():
        return None
    rows = ref.filter(ref["component"].cast(pl.String) == component)
    if rows.is_empty():
        return None
    ref_p50 = rows["p50_ms"][0]
    if ref_p50 is None or float(ref_p50) == 0.0:
        return None
    return (current_p50_ms - float(ref_p50)) / float(ref_p50) * 100.0


def _build_snapshots(
    run_id: str,
    stats: list[TrialResult],
    scaling_fits: list[ScalingFit],
    prev_snaps: pl.DataFrame | None,
    baseline_snaps: pl.DataFrame | None,
) -> pl.DataFrame:
    # Index elapsed_s fits by stage; first match wins.
    fit_by_stage: dict[str, ScalingFit] = {}
    for f in scaling_fits:
        if f.metric == "elapsed_s" and f.stage not in fit_by_stage:
            fit_by_stage[f.stage] = f

    # Aggregate multiple param-point results per stage (median across points).
    by_stage: dict[str, list[TrialResult]] = defaultdict(list)
    for tr in stats:
        by_stage[tr.stage].append(tr)

    rows: list[dict] = []
    e2e_p50_ms = 0.0
    e2e_p90_ms = 0.0
    e2e_min_ms = 0.0
    e2e_stddev_var = 0.0
    e2e_peak_rss_mb = 0.0
    e2e_result_mb = 0.0

    for stage in sorted(by_stage):
        trs = by_stage[stage]
        p50_ms = float(np.median([tr.elapsed_p50 * 1e3 for tr in trs]))
        p90_ms = float(np.median([tr.elapsed_p90 * 1e3 for tr in trs]))
        min_ms = float(np.median([tr.elapsed_min * 1e3 for tr in trs]))
        stddev_ms = float(np.median([tr.elapsed_stddev * 1e3 for tr in trs]))
        peak_rss_mb = float(np.median([tr.peak_rss_mb for tr in trs]))
        result_mb = float(np.median([tr.result_mb for tr in trs]))

        fit = fit_by_stage.get(stage)
        rows.append(
            {
                "run_id": run_id,
                "component": stage,
                "p50_ms": p50_ms,
                "p90_ms": p90_ms,
                "min_ms": min_ms,
                "stddev_ms": stddev_ms,
                "peak_rss_mb": peak_rss_mb,
                "result_mb": result_mb,
                "scaling_slope": fit.log_log_slope if fit else None,
                "scaling_r2": fit.r_squared if fit else None,
                "vs_prev_p50_pct": _delta_pct(stage, p50_ms, prev_snaps),
                "vs_baseline_p50_pct": _delta_pct(stage, p50_ms, baseline_snaps),
            }
        )

        e2e_p50_ms += p50_ms
        e2e_p90_ms += p90_ms
        e2e_min_ms += min_ms
        e2e_stddev_var += stddev_ms**2
        e2e_peak_rss_mb += peak_rss_mb
        e2e_result_mb += result_mb

    rows.append(
        {
            "run_id": run_id,
            "component": "e2e",
            "p50_ms": e2e_p50_ms,
            "p90_ms": e2e_p90_ms,
            "min_ms": e2e_min_ms,
            "stddev_ms": math.sqrt(e2e_stddev_var) if e2e_stddev_var > 0.0 else 0.0,
            "peak_rss_mb": e2e_peak_rss_mb,
            "result_mb": e2e_result_mb,
            "scaling_slope": None,
            "scaling_r2": None,
            "vs_prev_p50_pct": _delta_pct("e2e", e2e_p50_ms, prev_snaps),
            "vs_baseline_p50_pct": _delta_pct("e2e", e2e_p50_ms, baseline_snaps),
        }
    )

    return _cast_schema(pl.DataFrame(rows), _SNAPSHOTS_SCHEMA)


def _load_reference_snapshots(
    history_dir: Path,
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    runs = read_improvement_runs(history_dir)
    snaps = read_component_snapshots(history_dir)

    if runs.is_empty() or snaps.is_empty():
        return None, None

    sorted_runs = runs.sort("run_ts")
    prev_run_id = sorted_runs["run_id"][-1]  # most recent
    baseline_run_id = sorted_runs["run_id"][0]  # oldest

    prev_snaps = snaps.filter(snaps["run_id"] == prev_run_id)
    baseline_snaps = snaps.filter(snaps["run_id"] == baseline_run_id)
    return prev_snaps, baseline_snaps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_history_run(
    history_dir: Path,
    run_id: str,
    run_ts: datetime.datetime,
    git_sha: str,
    agent_ctx: AgentContext,
    stats: list[TrialResult],
    scaling_fits: list[ScalingFit],
    regression: RegressionReport | None = None,
    annotation: AgentAnnotation | None = None,
) -> None:
    """Append one agentic profiling run to the three history tables.

    Args:
        history_dir: Directory for the three Parquet files.  Created if absent.
        run_id: Unique identifier for this run.
        run_ts: Wall-clock timestamp of the run (used to order prev/baseline).
        git_sha: Short git SHA at run time (from ``RunEnvironment.git_sha``).
        agent_ctx: What the agent was trying to do.
        stats: Per-component ``TrialResult`` list from ``run_harness``.
            Multiple entries per component are allowed (one per param point);
            they are aggregated by median before storage.
        scaling_fits: Scaling-curve fits from ``fit_scaling``; the elapsed_s
            fit per component is stored in ``component_snapshots``.
        regression: Regression check result; if None, ``regression_passed``
            is recorded as True with zero violations.
        annotation: Optional agent notes; if None the annotations table is not
            written for this run.
    """
    history_dir.mkdir(parents=True, exist_ok=True)

    # Read reference data BEFORE writing so deltas compare against prior runs.
    prev_snaps, baseline_snaps = _load_reference_snapshots(history_dir)

    snaps_df = _build_snapshots(run_id, stats, scaling_fits, prev_snaps, baseline_snaps)

    e2e_row = snaps_df.filter(snaps_df["component"].cast(pl.String) == "e2e")
    e2e_p50_s = float(e2e_row["p50_ms"][0]) / 1000.0
    e2e_vs_prev_raw = e2e_row["vs_prev_p50_pct"][0]
    e2e_vs_prev_pct = float(e2e_vs_prev_raw) if e2e_vs_prev_raw is not None else None

    run_row: dict[str, object] = {
        "run_id": run_id,
        "run_ts": run_ts,
        "git_sha": git_sha,
        "branch": _git_branch(),
        "commit_message": _git_commit_message(),
        "agent_id": agent_ctx.agent_id,
        "target_component": agent_ctx.target_component,
        "goal": agent_ctx.goal,
        "strategy": agent_ctx.strategy,
        "e2e_p50_s": e2e_p50_s,
        "e2e_vs_prev_pct": e2e_vs_prev_pct,
        "regression_passed": regression.passed if regression is not None else True,
        "n_violations": len(regression.violations) if regression is not None else 0,
        "parent_run_id": agent_ctx.parent_run_id,
    }
    runs_df = _cast_schema(pl.DataFrame([run_row]), _RUNS_SCHEMA)

    _upsert_parquet(history_dir / "improvement_runs.parquet", runs_df)
    _upsert_parquet(history_dir / "component_snapshots.parquet", snaps_df)

    if annotation is not None:
        ann_row: dict[str, object] = {
            "run_id": run_id,
            "hypothesis": annotation.hypothesis,
            "outcome": annotation.outcome,
            "lessons": annotation.lessons,
            "next_target": annotation.next_target,
            "confidence": annotation.confidence,
            "improvement_type": annotation.improvement_type,
        }
        ann_df = _cast_schema(pl.DataFrame([ann_row]), _ANNOTATIONS_SCHEMA)
        _upsert_parquet(history_dir / "agent_annotations.parquet", ann_df)


def read_improvement_runs(history_dir: Path) -> pl.DataFrame:
    """Read all improvement-run metadata rows."""
    path = history_dir / "improvement_runs.parquet"
    if not path.exists():
        return pl.DataFrame(schema=_RUNS_SCHEMA)
    return pl.read_parquet(path)


def read_component_snapshots(history_dir: Path) -> pl.DataFrame:
    """Read all per-component snapshot rows."""
    path = history_dir / "component_snapshots.parquet"
    if not path.exists():
        return pl.DataFrame(schema=_SNAPSHOTS_SCHEMA)
    return pl.read_parquet(path)


def read_agent_annotations(history_dir: Path) -> pl.DataFrame:
    """Read all agent annotation rows."""
    path = history_dir / "agent_annotations.parquet"
    if not path.exists():
        return pl.DataFrame(schema=_ANNOTATIONS_SCHEMA)
    return pl.read_parquet(path)


__all__ = [
    "AgentAnnotation",
    "AgentContext",
    "read_agent_annotations",
    "read_component_snapshots",
    "read_improvement_runs",
    "write_history_run",
]
