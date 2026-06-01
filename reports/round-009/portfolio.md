---
round: 9
component: portfolio
pr: 63
date: "2026-05-31"
metric: "portfolio p50_ms — assemble OSQP constraint matrix from COO triplets"
verdict: accepted
headline_delta: "−7…−27% (constraint build ~2.5×); golden byte-identical"
---

# portfolio · round 009

> **✓ Merged.** Landed as **PR #63** (`improve/portfolio-perf`, commit `d7d5fa8`) on
> `main`. An **exploit (performance)** round. Re-validated post-merge: full suite
> **1291 passed**, evalgate **17/17** (golden byte-identical).

portfolio is **already tiny** (≈8 ms). This round found the one genuine, golden-safe
win and correctly left the irreducible solve alone.

## What it addressed

**Hotspot:** the dominant cost is the OSQP ADMM solve (~575 iters to 1e-8 at n=3000)
— **intrinsic and not golden-safely reducible** (tested: eps 1e-8→1e-4 saves only
~22% iters but moves weights ~4e-6, risking the golden; w0 warm-start gave zero
benefit). The one genuine win was **constraint-matrix assembly**: it built one sparse
matrix per block (`sp.eye`/`hstack`/`vstack`) when every block is a shifted identity
or dense row whose nonzeros are known in closed form.

## How it decided

Rewrote assembly to emit **COO triplets straight into one CSC**. The matrix `A` is
bit-identical to the old assembly (proven over 32 spec configs incl.
sector/gross/band/no-cost + factor-equality rows), so OSQP returns byte-identical
weights (`max|Δw| = 0.0`).

## Pre/post profile (full `_portfolio_run`, seed 0, same host)

| n_assets | before | after | delta |
|---|---|---|---|
| 100 | 3.27 ms | 2.39 ms | **−27%** |
| 500 | 7.82 ms | 6.79 ms | −13% |
| 1000 | 15.03 ms | 13.93 ms | −7% |
| 3000 | 56.16 ms | 51.99 ms | −7% |

Constraint build alone at n=3000: **2.58 → 1.03 ms (~2.5×)**. The win is largest at
small n (where the grid p50 lives); at large n the untouched ADMM solve dominates.
Net diff −91 lines (removed three now-dead block-builder helpers + their tests).

## System impact

Golden **byte-identical** — `opt_gross`, `tracking_error` exactly 0 delta;
`factor_vol` at the known 1e-16 noise floor; 17/17 PASS.

Lane-pure: only `portfolio/optimizer.py` + `portfolio/test_helpers.py`. Clean revert.

## Suggested next steps

The ADMM solve is the floor; further portfolio perf needs a solver-level change
(different conic backend / problem reformulation) — an **explore**-round target, not
an exploit one, since it would move the numbers.
