---
round: 5
component: signals
pr: 42
date: "2026-05-31"
metric: "signals component p50 ms (lazy rank-IC)"
verdict: accepted
headline_delta: "signals p50 −48% (median); IC bit-identical"
---

# signals · round 005

> **✓ Merged.** Landed as **PR #42** (`explore/signals-lazy-rankcov`) on `main`.
> Gates ran on the branch and were re-validated post-merge (full suite **1215
> passed**; evalgate **17/17**).

Replace the dense pivot + `scipy.rankdata` Spearman-IC path with a **fully-lazy /
streaming Polars rank-IC engine** (`spearman_ic_lazy`), made the default
(`engine="lazy"`) for `ic_series_v2` and `ic_horizon_curve`; the incumbent is
retained as `engine="matrix"`. **−48% signals p50**, IC bit-identical.

## What it addressed

**Component:** signals. **Metric:** signals component p50_ms — the
cross-sectional IC computation across the horizon grid. The incumbent pivoted to
a dense per-date matrix and called `scipy.stats.rankdata` per date; the dense
materialization dominated at scale.

## How it decided

Spearman IC is **Pearson correlation on within-date ranks**. Polars computes
within-group ranks lazily (`.rank("average")`, whose tie handling matches
`scipy.rankdata`), so the whole IC series is one long-format `group_by`/window
query with **no dense matrix and no scipy** — and it streams. The equality is
principled, not coincidental, and is pinned bit-for-bit by tests.

## Pre/post profile

signals component p50_ms, same harness grid + seed + box, via `scripts/bench`:

| aggregate | before | after | delta |
|:----------|:-------|:------|:------|
| median p50 | 248.64 ms | 128.66 ms | **−48.3%** |
| sum across grid | 3845 ms | 2443 ms | −36.5% |

Every one of 11 grid points improved (−14.7% to −56.8%; the win grows with panel
size). No other stage regressed.

## System impact

`evalgate` vs golden — **17 PASS / 0 FAIL**. All signals fields (`ic_raw`,
`ic_neutralized`, `horizon_ic[1/5/21/63]`) **bit-identical** (abs_delta
0.00e+00); golden holds within tolerance everywhere else. **pytest:** 1176
passed, 1 skipped (+19 new `signals/test_lazy_ic.py` pinning lazy == matrix
bit-for-bit across horizons, min_obs, scattered/tail NaN, constant-signal, and
the full horizon curve). Additive: `engine="lazy"` default, `engine="matrix"`
incumbent retained; touches only the 5 `signals/` files.

## Suggested next steps

1. With the dense pivot gone from the IC path, the next signals hot spot is the
   neutralization / sector-demean step — profile the full grid to confirm before
   picking the next target.
