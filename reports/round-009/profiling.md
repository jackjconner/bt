---
round: 9
component: profiling
pr: 64
date: "2026-05-31"
metric: "profiling p50_ms — vectorize fit_scaling per-stage loops in NumPy"
verdict: accepted
headline_delta: "101.9→38.3 ms (−62%, 2.66×); n_scaling_fits=36 held"
---

# profiling · round 009

> **✓ Merged.** Landed as **PR #64** (`improve/profiling-perf`, commit `9244950`) on
> `main`. An **exploit (performance)** round. Re-validated post-merge: full suite
> **1291 passed**, evalgate **17/17** (`n_scaling_fits=36` held).

profiling is **fixed overhead**, not data-scaling (≈101 ms flat, r² near zero across
both dims). The win here is reducing the constant cost.

## What it addressed

**Hotspot:** `profiling/scaling.py::fit_scaling` — the single hot function on
profiling's flat path. It ran **144 inner iterations** (9 stages × 4 metrics × 4
dims), each rebuilding the same per-stage Polars filter chain and recomputing each
dim's `.mode()`.

## How it decided

The confounder-control subset and modes depend only on `(stage, dim)` — **not** the
metric. Hoisted them to once-per-stage, pulled each stage's columns to NumPy once, and
built the control mask in NumPy over the Polars-computed modes. Polars `Series.mode()`
is retained so tie-break semantics are byte-preserved; `(metric, dim)` append order is
kept so the returned fit list is unchanged.

## Pre/post profile (largest grid point, bench lock, seed 0)

| metric | before | after | delta |
|---|---|---|---|
| profiling component p50 | 101.9 ms | 38.3 ms | **−62% (2.66×)** |
| `fit_scaling` in isolation | ~50–65 ms | ~7.5 ms | ~7× |

No other stage regressed. Equivalence proven by sorted diff of `fit_scaling` output
vs a clean `main` process: identical 36-fit set and values.

## System impact

Golden **held** — 17/17 PASS, **`n_scaling_fits = 36` PASS**, all 16 accuracy fields
byte-stable (only the pre-existing 1e-16 BLAS noise on `wf_mean_r2`/`factor_vol`).
profiling is off the production number path; only *how fast* the fits are computed
changed, not *what* is computed.

Lane-pure: only `profiling/scaling.py` + `profiling/test_scaling.py`. Clean revert.

## Suggested next steps

`run_trials` per-trial setup is the next fixed-overhead target. The across-stage
`scaling_fits` row order is already non-deterministic on `main` (hash-order, not part
of any golden) — unchanged by this round.
