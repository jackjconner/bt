"""Within-stage flame-graph capture and a parquet index that points at the blobs.

The rest of ``profiling`` answers *which stage is slow and how it scales* with
scalar rows (``stage_measurements``).  This module answers the orthogonal
*where inside the stage does the time / memory go* with sampled call-stack
profiles.  A flame graph is a tree of stack samples, not a row of scalars, so it
cannot live in the columnar tables — it follows the snapshot-artifact pattern
(see ``output.py``) instead: the blob lands on disk, and a small index table
(``profile_artifacts.parquet``) records one row per blob so a consumer can look
up "the CPU profile for build_panel in run X" without scanning the tree.

Two in-process profilers, both unprivileged (no ptrace/root) so they run inside
an agent loop:

  - **pyinstrument** (CPU) — samples the Python stack on a timer.  We store a
    ``.speedscope.json.gz`` flame graph (open at speedscope.app) and a
    ``.calltree.txt`` readout (agent-greppable: the indented tree shows the hot
    path with per-frame time).  Native time (Polars' Rust, NumPy's C) is
    attributed to the Python call site that entered it — you see *which*
    operation is hot, not the breakdown *inside* the native call.  Seeing into
    Rust frames needs py-spy ``--native`` or perf, both of which require ptrace
    privileges this loop does not assume.
  - **memray** (memory) — an in-process tracker that *does* see native
    allocations (``memory.py`` notes tracemalloc misses Polars' Rust and NumPy's
    C buffers).  We store its binary capture (``.bin``) — every memray reporter
    (``memray flamegraph|tree|summary``) replays from it — plus a ``.summary.txt``
    top-allocators readout rendered once for agents.

Storage layout — the blob tree is a *sibling* of the scalar parquet store so the
small, commit-friendly tables stay cheap to sync while the large blobs are
pruned independently::

    <profiles_dir>/
      profile_artifacts.parquet          # the index (this module owns it)
      <run_id>/
        build_panel.0.cpu.speedscope.json.gz
        build_panel.0.cpu.calltree.txt
        build_panel.0.mem.memray.bin
        build_panel.0.mem.summary.txt

Retention: ``prune_profiles`` keeps the latest N runs and never deletes a run
whose artifacts are flagged ``on_regression`` — so the evidence for a caught
regression survives even after it ages past the window.
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import polars as pl

from .storage import _upsert_parquet

T = TypeVar("T")

_INDEX_FILENAME = "profile_artifacts.parquet"

# Schema for the artifact index.  ``path`` is stored relative to profiles_dir so
# the whole tree can be relocated without rewriting rows.
_ARTIFACTS_SCHEMA: dict[str, type[pl.DataType]] = {
    "run_id": pl.String,
    "param_point_id": pl.Int64,
    "stage": pl.Categorical,
    "profiler": pl.Categorical,  # "pyinstrument" | "memray"
    "kind": pl.Categorical,  # "cpu" | "memory"
    "fmt": pl.Categorical,  # "speedscope" | "calltree" | "memray-bin" | "summary"
    "path": pl.String,  # relative to profiles_dir
    "size_bytes": pl.Int64,
    "on_regression": pl.Boolean,
    "git_sha": pl.String,
}


@dataclass(frozen=True)
class ProfileArtifact:
    """One captured profile blob — one row in ``profile_artifacts``."""

    run_id: str
    param_point_id: int
    stage: str
    profiler: str
    kind: str
    fmt: str
    path: str  # relative to profiles_dir
    size_bytes: int
    on_regression: bool
    git_sha: str


def _slug(stage: str) -> str:
    """Filesystem-safe stem for a stage label."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in stage)


def _artifact_for(
    blob: Path,
    *,
    profiles_dir: Path,
    run_id: str,
    param_point_id: int,
    stage: str,
    profiler: str,
    kind: str,
    fmt: str,
    on_regression: bool,
    git_sha: str,
) -> ProfileArtifact:
    return ProfileArtifact(
        run_id=run_id,
        param_point_id=param_point_id,
        stage=stage,
        profiler=profiler,
        kind=kind,
        fmt=fmt,
        path=str(blob.relative_to(profiles_dir)),
        size_bytes=blob.stat().st_size,
        on_regression=on_regression,
        git_sha=git_sha,
    )


def capture_cpu[T](
    stage: str,
    fn: Callable[[], T],
    *,
    profiles_dir: Path,
    run_id: str,
    param_point_id: int,
    git_sha: str,
    on_regression: bool = False,
    interval_s: float = 0.001,
) -> tuple[T, list[ProfileArtifact]]:
    """Run ``fn`` under pyinstrument and store a flame graph plus a call tree.

    pyinstrument samples the Python stack from *inside* the process on a timer,
    so it needs no ptrace and runs unprivileged.  Time spent in native code
    (Polars' Rust, NumPy's C) is attributed to the Python call site that entered
    it — you see *which operation* is hot, not the breakdown inside the native
    call.  For native allocation frames on the memory side, see
    ``capture_memory``.

    Two artifacts:
      - ``.cpu.speedscope.json.gz`` — a flame graph; open it at speedscope.app.
      - ``.cpu.calltree.txt`` — the indented call tree with per-frame time, the
        agent-readable readout (analogous to memray's summary).

    Args:
        stage: Stage label; used in the filename and the index row.
        fn: Zero-argument callable doing the work to profile.
        profiles_dir: Sibling blob tree root (NOT the scalar parquet store).
        run_id / param_point_id / git_sha: Index keys, matching
            ``stage_measurements``.
        on_regression: Flag the artifacts as regression evidence so
            ``prune_profiles`` keeps them past the retention window.
        interval_s: Sampling interval in seconds; smaller resolves shorter
            frames at higher overhead.

    Returns:
        ``(fn's return value, [flame-graph artifact, call-tree artifact])``.
    """
    from pyinstrument import Profiler
    from pyinstrument.renderers import SpeedscopeRenderer

    run_dir = profiles_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_slug(stage)}.{param_point_id}.cpu"
    speedscope_path = run_dir / f"{stem}.speedscope.json.gz"
    calltree_path = run_dir / f"{stem}.calltree.txt"

    prof = Profiler(interval=interval_s, async_mode="disabled")
    prof.start()
    try:
        result = fn()
    finally:
        prof.stop()

    # Speedscope JSON is compact and renders to a flame graph; gzip it since the
    # frame table is repetitive.  The text call tree stays plain for grepping.
    with gzip.open(speedscope_path, "wt", encoding="utf-8") as f:
        f.write(prof.output(SpeedscopeRenderer()))
    calltree_path.write_text(prof.output_text(unicode=True, color=False))

    common = {
        "profiles_dir": profiles_dir,
        "run_id": run_id,
        "param_point_id": param_point_id,
        "stage": stage,
        "profiler": "pyinstrument",
        "kind": "cpu",
        "on_regression": on_regression,
        "git_sha": git_sha,
    }
    return result, [
        _artifact_for(speedscope_path, fmt="speedscope", **common),
        _artifact_for(calltree_path, fmt="calltree", **common),
    ]


def capture_memory[T](
    stage: str,
    fn: Callable[[], T],
    *,
    profiles_dir: Path,
    run_id: str,
    param_point_id: int,
    git_sha: str,
    on_regression: bool = False,
    native_traces: bool = True,
) -> tuple[T, list[ProfileArtifact]]:
    """Run ``fn`` under memray and store the binary capture plus a text summary.

    memray tracks allocations from inside the process and, with
    ``native_traces``, attributes them through Rust/C frames — catching exactly
    the Polars/NumPy buffers ``tracemalloc`` cannot see.  Two artifacts:

      - ``.memray.bin`` — the canonical capture; every memray reporter
        (``memray flamegraph|tree <bin>``) replays from it on demand.
      - ``.summary.txt`` — a top-allocators table rendered once via
        ``memray summary`` so an agent can read where the memory went without
        having memray installed or parsing the binary.

    Returns:
        ``(fn's return value, [bin artifact, summary artifact])``.

    Raises:
        RuntimeError: memray is not installed, or summary rendering failed.
    """
    try:
        import memray
    except ImportError as e:
        raise RuntimeError("memray not installed — run `uv add memray` to enable it") from e

    run_dir = profiles_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bin_path = run_dir / f"{_slug(stage)}.{param_point_id}.mem.memray.bin"
    summary_path = run_dir / f"{_slug(stage)}.{param_point_id}.mem.summary.txt"
    # memray refuses to overwrite an existing destination.
    if bin_path.exists():
        bin_path.unlink()

    with memray.Tracker(bin_path, native_traces=native_traces):
        result = fn()

    # `memray summary` writes a top-allocators table to stdout; capture it so the
    # readout survives without re-running the workload.
    summary = subprocess.run(
        ["memray", "summary", str(bin_path)],
        capture_output=True,
        text=True,
    )
    if summary.returncode != 0:
        raise RuntimeError(f"memray summary exited {summary.returncode}: {summary.stderr.strip()}")
    summary_path.write_text(summary.stdout)

    common = {
        "profiles_dir": profiles_dir,
        "run_id": run_id,
        "param_point_id": param_point_id,
        "stage": stage,
        "profiler": "memray",
        "kind": "memory",
        "on_regression": on_regression,
        "git_sha": git_sha,
    }
    return result, [
        _artifact_for(bin_path, fmt="memray-bin", **common),
        _artifact_for(summary_path, fmt="summary", **common),
    ]


def _index_path(profiles_dir: Path) -> Path:
    return profiles_dir / _INDEX_FILENAME


def write_artifacts(profiles_dir: Path, artifacts: list[ProfileArtifact]) -> None:
    """Append artifact rows to ``profile_artifacts.parquet`` (upsert).

    Mirrors ``storage.write_run``: build a schema-cast frame and concat it onto
    the existing index.  Insertion order is preserved, which ``prune_profiles``
    relies on to identify the latest runs.
    """
    if not artifacts:
        return
    profiles_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_id": a.run_id,
            "param_point_id": a.param_point_id,
            "stage": a.stage,
            "profiler": a.profiler,
            "kind": a.kind,
            "fmt": a.fmt,
            "path": a.path,
            "size_bytes": a.size_bytes,
            "on_regression": a.on_regression,
            "git_sha": a.git_sha,
        }
        for a in artifacts
    ]
    df = pl.DataFrame(rows).with_columns(
        pl.col(name).cast(dtype) for name, dtype in _ARTIFACTS_SCHEMA.items()
    )
    _upsert_parquet(_index_path(profiles_dir), df)


def read_artifacts(profiles_dir: Path) -> pl.DataFrame:
    """Read the artifact index, or an empty schema-typed frame if absent."""
    path = _index_path(profiles_dir)
    if not path.exists():
        return pl.DataFrame(schema=dict(_ARTIFACTS_SCHEMA.items()))
    return pl.read_parquet(path)


def prune_profiles(profiles_dir: Path, keep_last_n: int) -> list[str]:
    """Delete blob dirs for old runs, keeping the latest N and all regressions.

    A run is kept if it is among the most recent ``keep_last_n`` runs (by index
    insertion order) OR any of its artifacts is flagged ``on_regression``.
    Everything else has its blob directory removed and its index rows dropped.

    Returns:
        The run_ids that were pruned.
    """
    if keep_last_n < 0:
        raise ValueError("keep_last_n must be >= 0")

    index = read_artifacts(profiles_dir)
    if index.is_empty():
        return []

    ordered_runs = index["run_id"].unique(maintain_order=True).to_list()
    protected = set(index.filter(pl.col("on_regression"))["run_id"].unique().to_list())
    keep = set(ordered_runs[len(ordered_runs) - keep_last_n :]) | protected
    to_prune = [r for r in ordered_runs if r not in keep]
    if not to_prune:
        return []

    for run_id in to_prune:
        run_dir = profiles_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)

    remaining = index.filter(~pl.col("run_id").is_in(to_prune))
    remaining.write_parquet(_index_path(profiles_dir))
    return to_prune
