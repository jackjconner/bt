---
round: 9
component: analysis
pr: 68
date: "2026-05-31"
metric: "analysis p50_ms — ordered group-by for turnover (skip hash + redundant sort)"
verdict: accepted
headline_delta: "two_way_turnover 1.44→1.13 ms (−21%) at long-history extreme; golden held"
---

# analysis · round 009

> **✓ Merged.** Landed as **PR #68** (`improve/analysis-perf`, commit `405b48d`) on
> `main`. An **exploit (performance)** round. Re-validated post-merge: full suite
> **1291 passed**, evalgate **17/17** (golden held).

analysis is the **smallest stage** (≈3 ms). This round found a real, golden-safe win
in its dominant sub-call rather than forcing a micro-opt.

## What it addressed

**Hotspot:** profiling `_analysis_run` at the largest grid point (n_dates=5040):
`analyze` 0.23 ms, `alpha+beta+ir` 0.49 ms, **`two_way_turnover` 1.49 ms**
(dominant). The turnover functions did `group_by("date").agg().sort("date")` — a hash
group-by **plus a redundant final sort** — despite the production trade_log already
being emitted in strict date order.

## How it decided

Switched both `one_way_turnover` / `two_way_turnover` to
`sort("date").group_by("date", maintain_order=True).agg()` — a sorted-key group-by
that skips the hash table. The leading sort makes the output order-independent of the
input (added a test that shuffles the trade_log and asserts identical date-sorted
results). Deliberately rejected the faster `set_sorted("date")` variant (~0.49 ms): it
bakes an unsafe sortedness assumption into a public function.

## Pre/post profile (A/B on identical trade_log, seed 0)

| grid point | before | after | delta |
|---|---|---|---|
| n_dates=5040 (504k rows) | 1.437 ms | 1.128 ms | **−21%** |
| n_dates=252 modal (25k rows) | 0.193 ms | 0.188 ms | ±noise |

The win scales with trade count → material at the long-history grid points, neutral
at the modal point. No other stage touched.

## System impact

Golden **held** — 16/17 fields byte-identical or ~1e-16, including all
analysis-adjacent outputs (`tracking_error`, `gross_sharpe`, `net_sharpe`,
`cost_drag` = 0.00 delta). No golden re-save.

Lane-pure: only `analysis/turnover.py` + `analysis/test_analysis_production.py`.
Clean revert.

## Suggested next steps

Noted (cross-lane, **not** actioned): `alpha`/`beta`/`information_ratio` each call
`_align` → 3 redundant inner-joins of identical data (~0.49 ms). The fix is wiring
`harness/components.py` to the existing `benchmark_metrics_fused` (one join) — a
harness call-site change, a candidate for a future harness/profiling round.
