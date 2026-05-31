---
round: 5
component: models
pr: 41
date: "2026-05-31"
metric: "models walk_forward_cv p50 ms"
verdict: pending
headline_delta: "walk-forward 1.26–2.41×; wf_mean_ic byte-identical"
---

# models · round 005

> **⏳ Pending review.** Open as **PR #41** (`explore/models-gram-wf`), **not yet
> merged**. Gates below ran on the branch; verdict flips to `accepted` on merge.

Replace the per-fold sklearn `Ridge` refit loop with a **batched numpy-core
walk-forward engine**: compute each date block's weighted moments once, assemble
every fold's standardized Gram as a difference of cumulative block sums, and
solve all alphas against one Cholesky-ready Gram — **1.26–2.41× faster**, with
`wf_mean_ic` byte-identical.

## What it addressed

**Component:** models. **Metric:** the `walk_forward_cv` call (models harness
`elapsed_s` p50). The incumbent refits sklearn `Ridge` from scratch for every
alpha and every expanding/rolling fold (and inner-CV sub-window), repaying the
same standardization and Gram assembly thousands of times.

## How it decided

A ridge fit is a closed-form solve against a **standardized Gram matrix**. The
Gram over any fold window is a **difference of cumulative per-date-block
moments** — so compute block moments once, prefix-sum them, and every fold's Gram
is an O(k²) subtraction rather than a re-walk. All alphas share the Gram and
solve via one Cholesky. Closed-form weighted ridge matches sklearn to ~1e-16.

## Pre/post profile

Controlled bench (loop vs batched engine, identical panels, median of 7,
bench-locked):

| panel (assets × dates) | before | after | speedup |
|:-----------------------|:-------|:------|:--------|
| 100 × 252  | 46.81 ms | 29.59 ms | 1.58× |
| 500 × 252  | 135.54 ms | 56.14 ms | **2.41×** |
| 2000 × 252 | 455.00 ms | 249.99 ms | 1.82× |
| 100 × 1260 | 229.00 ms | 143.93 ms | 1.59× |
| 100 × 5040 | 952.27 ms | 754.68 ms | 1.26× |

## System impact

`evalgate` vs golden — **17 PASS / 0 FAIL**. `wf_mean_ic` byte-identical
(0.00e+00); `wf_mean_r2` at machine epsilon (1.11e-16). No accuracy number moved.
**pytest:** 1175 passed, 1 skipped (+18 equivalence/dispatch tests). Additive:
wired via `WalkForwardConfig.engine="auto"` — dispatches to the batched engine
for closed-form-ridge factories, non-ridge models keep the loop. +739/−0, 4
files, all in `models/`.

## Suggested next steps

1. The batched engine only fires for closed-form ridge. If other linear model
   factories become hot, the same block-moment / cumulative-Gram trick extends to
   any GLS-shaped estimator — worth generalizing the dispatch.
