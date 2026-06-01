---
round: 9
component: backtest
pr: 65
date: "2026-05-31"
metric: "backtest p50_ms — vectorize trade-log assembly in the weight-space fast path"
verdict: accepted
headline_delta: "1.4–1.8× across the grid; n_assets scaling 0.93→0.46; golden held"
---

# backtest · round 009

> **✓ Merged.** Landed as **PR #65** (`improve/backtest-perf`, commit `a6a5562`) on
> `main`. An **exploit (performance)** round. Re-validated post-merge: full suite
> **1291 passed**, evalgate **17/17** (golden held).

backtest is the **2nd-largest cost** in the harness. This round flattened its
per-asset scaling.

## What it addressed

**Hotspot:** at high `n_assets`, building the 756k-row `trade_log` from Python lists
via Polars `new_from_any_values` type inference consumed 0.057 s of 0.084 s
(3000a×252d) — a per-bar `list.extend` of `date` objects.

## How it decided

Replaced the per-bar list accumulation with NumPy (`repeat`/`tile`/`concatenate`) +
typed Polars series — a pure repacking of *identical* values, no math change. The
trade-log build alone went **~63 ms → ~1 ms (57×)**. Deliberately did **not** touch
the per-bar `compute_transaction_costs` / `compute_slippage` loop: the commission
`max(·, nav)` floor and sqrt-slippage are nonlinear in path-dependent NAV, so
hoisting them would change the FP op-tree and risk moving `cost_drag` / `net_sharpe`.
That headroom remains for a future round but isn't a golden-safe simple hoist.

## Pre/post profile (backtest `elapsed_s`, seed 0, same host)

| grid point | before | after | speedup |
|---|---|---|---|
| 3000a×252d | 186 | 131 | 1.42× |
| 100a×5040d | 178 | 100 | 1.78× |

Every grid point improved 1.27–1.78×. **Scaling `n_assets^0.93 → ^0.46`** (the
per-asset trade-log cost is gone); `n_dates^0.99 → ^0.94`. No other stage regressed.

## System impact

Golden **held** — 17/17 PASS. All accuracy fields (`gross_sharpe`, `net_sharpe`,
`cost_drag`, IC/wf/factor) `abs_delta = 0.00e+00`; only the timing field
`backtest_p50_s` moved (field-tol exempt). Byte-identity proven directly:
`DataFrame.equals` vs `main` on `trade_log`/`nav_history`/`cash_history` across 5
grid points → all `True`.

Lane-pure: only `backtest/{engine_pro,vectorized,test_engine_pro}.py`. Clean revert.

## Suggested next steps

The per-bar cost/slippage loop (n_dates axis) is the remaining hot path — but it's
nonlinear in NAV, so a future round must prove FP-equivalence carefully (or accept a
justified tie).
