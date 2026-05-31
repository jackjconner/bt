# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv run main.py                 # run entry point
uv run python -c "..."         # one-off script in the venv
uv add <package>               # add dependency (updates pyproject.toml + uv.lock)
```

## Profiling

Scalar stage metrics (timing, memory, scaling, regression) live in `profiling/`.
For **within-stage flame graphs**, use `profiling/flamegraph.py` — both capture
functions are in-process and unprivileged (no ptrace/root), so they run inside
an agent loop:

```python
from profiling import capture_both, write_artifacts, prune_profiles

# CPU (pyinstrument → speedscope + .calltree.txt) AND memory (memray → .bin +
# .summary.txt) from ONE fn execution — 4 artifacts. Prefer this over separate
# capture_cpu / capture_memory passes (which re-run fn); use those only when you
# need a clean single-profiler read free of the other's overhead.
result, arts = capture_both("build_panel", lambda: build_panel(...),
                            profiles_dir=pdir, run_id=rid, param_point_id=0, git_sha=sha)

write_artifacts(pdir, arts)           # → profile_artifacts.parquet index, keyed like stage_measurements
prune_profiles(pdir, keep_last_n=5)   # keeps latest N runs + any flagged on_regression=True
```

- Blobs land in a sibling `profiles/<run_id>/` tree; query `profile_artifacts.parquet`
  (via `read_artifacts`) to find the path for a given run/stage, then read the
  `.calltree.txt` / `.summary.txt` directly.
- pyinstrument attributes native (Rust/C) CPU time to the Python call site, not
  the frames inside it. Seeing into Polars' Rust needs py-spy `--native` (ptrace,
  intentionally not used); memray already gives native frames for memory.

**Wired-in capture (opt-in, off by default):**

- `run_harness(..., profiles_dir=<dir>)` — captures every component at every grid
  point in a *separate* pass (never wraps the timed loop), indexes them, prunes
  to the latest N runs (regressions retained). A unique per-run id is derived
  from the timestamp so retention works despite the fixed `run_id`.
- `BT_FLAMEGRAPHS=1 uv run main.py` — flips on both the harness pass and a
  whole-pipeline `full_pipeline` capture into `.oversight/profiles/`. The
  whole-pipeline flame graph already contains each component's subtree, so one
  capture decomposes the full run. Off by default → the plain run pays no cost.

## Rules

- do not make decisions on architecture, design, or workaround without explicitly consulting me
- do not add fallbacks without explicitly consulting me
- do not go beyound the scope of the ask unless specified
