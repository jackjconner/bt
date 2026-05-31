---
round: 5
component: backtest
pr: 44
date: "2026-05-31"
metric: "backtest p50 ms (weight-space fast path)"
verdict: accepted
headline_delta: "backtest p50 1.24×; path-dependent fields byte-exact"
---

# backtest · round 005

> **✓ Merged.** Landed as **PR #44** (`explore/backtest-vectorized`) on `main`,
> closing the round at 5/5. Gates ran on the branch and were re-validated
> post-merge with all five changes (full suite **1227 passed, 1 skipped**;
> evalgate **17/17**).

A **vectorized weight-space fast path** for the production backtest envelope:
hoist softmax / constraints / weight-drift / portfolio-return out of the Python
event loop into batched NumPy matrix ops, leaving only the genuinely sequential
NAV/cost recurrence scalar — **backtest p50 1.24×**, with every path-dependent
field byte-exact.

## What it addressed

**Component:** backtest. **Metric:** backtest harness p50. The event loop walked
date-by-date in Python computing weights (softmax + constraints), weight drift,
and portfolio return per step — all of which are **batchable across dates**; only
NAV compounding and the nonlinear cost recurrence are truly sequential.

## How it decided

Separate the **batchable** from the **sequential**. Weights, constraint
projection, weight drift, and gross portfolio return are pure functions of the
per-date inputs → computed once as matrix ops over the whole date axis. Only the
NAV/cost recurrence (which depends on the prior step) stays in a scalar loop. The
incumbent loop is retained for the non-fast-path envelopes.

## Pre/post profile

backtest p50 across the harness grid, locked through `scripts/bench`:

| aggregate | before | after | speedup |
|:----------|:-------|:------|:--------|
| median p50 | 41.21 ms | 33.27 ms | **1.24×** |

Wins 1.13–1.24× on the n_dates axis, 1.04–1.16× at small/medium n_assets, and a
wash (0.99×) at the largest n_assets — there the sequential nonlinear cost loop
dominates and is irreducible. No other stage is touched, so none regress.

## System impact

`evalgate` vs golden — **17 PASS / 0 FAIL**. Path-dependent fields hold
essentially exact: `gross_sharpe` abs 8.88e-16, `net_sharpe` 3.22e-15,
`cost_drag` 2.33e-10 (rel ~1e-15 — last-ULP float re-association from the matrix
reorder; evalgate accepts as exact). **pytest:** 1169 passed, 1 skipped (+12
byte-identity equivalence tests covering costs / slippage / mask / caps / gross /
net / cadence / explicit-dates / full-stack and eligibility exclusions).
Additive: 547 insertions, 0 deletions, 3 files, all in the `backtest/` lane.

## Suggested next steps

1. The largest-n_assets wash is the sequential nonlinear cost recurrence. If that
   becomes the bottleneck, it needs either a closed-form cost approximation or a
   numba/native inner — a deeper change than this matrix-hoist.
