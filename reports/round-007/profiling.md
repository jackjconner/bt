---
round: 7
component: profiling
pr: 50
date: "2026-05-31"
metric: "new capability (additive, default-off); pipeline golden untouched"
verdict: accepted
headline_delta: "r²-confidence gating for regression detection — exclude noisy scaling fits from verdicts"
---

# profiling · round 007

> **✓ Merged.** Landed as **PR #50** (`improve/profiling-fit-confidence`, commit
> `416642d`) on `main`. A **`feature`** round — additive, default-off; existing
> `check_regressions` behavior byte-identical unless a caller opts in. Re-validated
> post-merge: **1284 passed**, evalgate **17/0**.

Use the scaling-fit **r²** the profiler already computes (but never consumed) to
**gate regression verdicts**, so noisy grids stop raising false alarms.

## What it adds

- `check_regressions(...)` gains `scaling_fits` and `min_r_squared=None`. When set,
  any `(stage, metric)` whose best fit r² is below the floor is **excluded** from
  verdicts; when `None` (default, what the harness uses), verdicts are byte-identical.
- `RegressionReport` gains `scaling_fit_confidence_ok` + `excluded_low_confidence`.
- New pure helper `profiling.scaling.stage_metric_r_squared(fits)` (max r² per
  `(stage, metric)`), exported; `output.write_json` surfaces the new fields.

A `(stage, metric)` with **no** fit is never excluded — absence of a fit is not
evidence of noise.

## How it's golden-safe

profiling is observability — off the pipeline-number path. The new gate is
default-off and the timed hot loop is untouched.

## Gates

- **lint** — clean; no suppressions.
- **test** — `343 passed, 1 skipped` (+7).
- **bench** — no regression (post-run analysis).
- **eval** — 17/0; `n_scaling_fits=36` unchanged.

## System impact

Regression detection now distinguishes a real slowdown from log-log fit noise on a
sparse/ill-conditioned grid. +195/−6, 7 files, all in lane.
