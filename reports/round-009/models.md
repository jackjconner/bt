---
round: 9
component: models
pr: 67
date: "2026-05-31"
metric: "models p50_ms — vectorize per-date rank-IC (drop per-date spearmanr loop)"
verdict: accepted
headline_delta: "~2.8× on the n_dates-heavy walk-forward path; golden held"
---

# models · round 009

> **✓ Merged.** Landed as **PR #67** (`improve/models-perf`, commit `840e5ac`) on
> `main`. An **exploit (performance)** round. Re-validated post-merge in the batch:
> full suite **1291 passed**, evalgate **17/17** (golden held).

models is the **dominant cost** in the harness (≈3.5 s at the largest grid point).
This round took the biggest bite out of it.

## What it addressed

**Hotspot:** profiling `walk_forward_cv` at the grid extreme showed
`_score_fold → rank_ic_series → scipy.stats.spearmanr` at **86%** of stage time
(1.033 s / 1.201 s) — thousands of per-date `spearmanr` calls in a Python loop. The
batched ridge core (a prior round, PR #41) was already negligible.

## How it decided

Spearman ρ = Pearson on average-tie ranks, and in production every test date carries
the **full cross-section** (verified equal-size). So the per-date Python loop
collapses to one batched `rankdata(axis=1)` + a vectorized Pearson over the panel.
The ragged-group case (unequal sizes) keeps the original per-date loop as the
fallback path and the equivalence oracle. Diff: two functions in `models/scoring.py`
+ 5 equivalence tests.

## Pre/post profile (models `elapsed_s` p50, seed 0, same host, bench lock)

| grid point | before | after | speedup |
|---|---|---|---|
| 100a×252d | 178.1 | 64.8 | 2.75× |
| 200a×252d | 235.7 | 117.4 | 2.01× |
| 500a×252d | 427.7 | 294.5 | 1.45× |
| 100a×756d | 526.2 | 183.8 | 2.86× |
| 100a×1260d | 851.5 | 303.4 | 2.81× |

The n_dates-heavy points see ~2.8× (rank-IC scales with n_dates). Scaling shape
unchanged; the curve shifts down.

## System impact

Golden **held** — 17/17 PASS. `wf_mean_ic` and `wf_mean_r2` each moved exactly
**1.11e-16** (rel ~1.5e-16): at/under the documented FP-noise floor, pure
summation-order between scipy's internal Pearson and the vectorized form. The other
15 fields are bit-identical. No golden re-save (within tolerance).

Lane-pure: only `models/scoring.py` + `models/test_scoring.py`. `rank_ic_series`
public API unchanged; both WF engines route through it, so both benefit. Clean revert.

## Suggested next steps

The batched ridge core is already tuned; the next models cost after this is the
fold-assembly / scoring glue. Diminishing returns — models is no longer the runaway
dominant stage it was.
