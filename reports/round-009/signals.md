---
round: 9
component: signals
pr: 66
date: "2026-05-31"
metric: "signals p50_ms — batch per-date sector OLS into one lstsq solve"
verdict: accepted
headline_delta: "−24% on the date-extreme grid point; n_dates scaling 0.92→0.87; golden held"
---

# signals · round 009

> **✓ Merged.** Landed as **PR #66** (`improve/signals-perf`, commit `fb2c3c9`) on
> `main`. An **exploit (performance)** round. Re-validated post-merge: full suite
> **1291 passed**, evalgate **17/17** (golden held, signals fields bit-identical).

## What it addressed

**Hotspot:** profiling `_signals_run` at the date-extreme (100a×5040d) showed the
cost was **not** the IC engine (already tuned in PRs #42/#17/#30) but
`neutralize_sector → _ols_residual` at **54%** of stage time — a
`for t in range(n_dates)` loop of tiny per-date `np.linalg.lstsq` solves.

## How it decided

Sector dummies are **time-invariant**, so the regressor matrix is shared across every
date → the whole panel becomes **one batched `lstsq(X, Sᵀ)`** instead of `n_dates`
separate solves. Added `_ols_residual_batched`; kept the per-date `_ols_residual` as
the NaN-in-signal fallback and equivalence oracle. One file + two tests (~90 lines).

## Pre/post profile (signals `elapsed_s`, seed 0, same host)

| grid point | before | after | delta |
|---|---|---|---|
| date-extreme (100a×5040d) | 349.8 | 265.3 | **−24.2%** |
| asset-extreme (3000a×252d) | 395.2 | 412.6 | ±noise |

Date-extreme is where the sector loop dominates; at 252 dates it's a minor share, so
the asset-extreme is unchanged within run-to-run noise. **Scaling `n_dates^0.92 →
^0.87`.** No other stage regressed.

## System impact

Golden **held** — `ic_neutralized`, `ic_raw`, all `horizon_ic[*]` show
`abs_delta = 0.00e+00`. (Multi-RHS lstsq differs from single-RHS only at ~1e-15 in
residuals, which are then rank-transformed into Spearman IC → no observable change,
far inside the 1e-6 tolerance.) No golden re-save.

Lane-pure: only `signals/neutralize.py` + `signals/test_neutralize.py`. Clean revert.

## Suggested next steps

The IC engine is already tuned; sector OLS is now batched. Remaining signals cost is
spread across the ranking / horizon passes — no single dominant hotspot left.
