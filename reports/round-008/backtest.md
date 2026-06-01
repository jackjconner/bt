---
round: 8
component: backtest
pr: 60
date: "2026-05-31"
metric: "activation — short-gating now on in production; golden byte-identical (long-only book)"
verdict: accepted
headline_delta: "activate short-availability gating + financing on the production path (correctness insurance, zero current delta)"
---

# backtest · round 008

> **✓ Merged.** Landed as **PR #60** (`activate/backtest-short-gating`, commit
> `a2ef663`) on `main`. A **feature-activation** round — it *flips on* the
> default-off capability that round 007 (#53) shipped dormant. Re-validated
> post-merge: **1285 passed**, evalgate **17/17 byte-identical**.

The first round under the new **per-feature default-state policy** (`DECISIONS.md`):
take a dormant flag-off capability and deliberately make it the production default.

## What it activates

Round 007 added short-availability gating + financing behind
`enable_short_availability_gating` (default off). This round turns it **on for the
production pipeline** — wiring the existing etl `borrow_rates` dataset into the
three production backtest `.run()` calls and setting the flags on `gross_cfg` /
`net_cfg` in `run_production_pipeline`. Real shorts are borrow-constrained
(shortability / loan availability) and pay borrow cost; the production path now
models that.

## The expected golden move that wasn't — and why that's correct

This was teed up as a deliberate **golden-moving** correctness round (case (c) of
the default-state policy). It turned out to be **case (b): golden unchanged**, and
the worker proved why rather than forcing a move:

- The production book is **long-only by construction** — targets are
  `_softmax(signal_row)`, all-positive, under `max_weight=0.1`, no `min_weight`.
  No shorts are ever taken.
- Short-availability gating only touches negative weights; borrow/financing accrue
  zero when short MV is 0 and gross is 1.0.
- So gating **binds on nothing** in the production spec → `net_sharpe`,
  `gross_sharpe`, `cost_drag` are **byte-identical** (`0.00e+00`). A direct NAV
  probe shows max \|Δ\| 3.5e-10 (rel ~3.7e-16) — pure fp reordering from the
  always-zero accrual branch, the same class as the pre-existing BLAS noise.
- The data *could* bind (the production `borrow_rates` has 2 498 / 25 200
  non-shortable rows, finite `loan_availability`, nonzero borrow) — it just isn't
  exercised by a long-only book. So the activation is **correctness insurance**: the
  moment a strategy shorts, it is priced correctly, with **no config change and zero
  disturbance to today's numbers.**

## Design call

Activated at the **production construction site** (`pipeline.py`), not the
`ProductionBacktestConfig` dataclass default. Flipping the global default would make
every caller that omits `borrow_rates` (benchmark harness, slice/naive helpers)
hard-`ValueError` — a wide blast radius for no production benefit. The engine default
stays `False`; only the production pipeline opts in. Callers can still force it off
(tested).

## Gates

- **lint** — clean (ruff + ty); no suppressions.
- **test** — `370 passed, 1 skipped` (backtest + integration); `test_pipeline.py`
  +1 (`test_production_backtest_runs_with_short_gating_on` pins the long-only no-op
  invariant + that callers can force gating off). Full suite post-merge **1285
  passed**.
- **bench** — no regression.
- **eval** — **17/17 byte-identical**; golden not re-saved (nothing moved).

## System impact

The production backtest is now institutionally honest about shorts without touching
the long-only golden. F-007 moves from *built (dormant)* to *active*. Models
fold-diagnostics and etl quality-flags were deliberately left dormant this pass
(no consumer / contract change respectively).
