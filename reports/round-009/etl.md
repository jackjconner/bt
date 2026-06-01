---
round: 9
component: etl
pr: 62
date: "2026-05-31"
metric: "etl p50_ms — scatter-fill masked matrix instead of pivot"
verdict: accepted
headline_delta: "−35% at 3000 assets (to_masked_matrix ~4×); golden held"
---

# etl · round 009

> **✓ Merged.** Landed as **PR #62** (`improve/etl-perf`, commit `4354679`) on
> `main`. An **exploit (performance)** round. Re-validated post-merge: full suite
> **1291 passed**, evalgate **17/17** (golden held).

## What it addressed

**Hotspot:** profiling *contradicted* the context pack — at the wide grid extreme
(3000 assets) the largest etl sub-call was `to_masked_matrix` (~40 ms of ~96 ms), not
`adjust_prices`, and its **`pivot` dominated (~34 ms)** because pivot cost scales with
the number of **asset columns**.

## How it decided

Replaced the pivot with a `rank("dense")`-driven NumPy **scatter** — one columnar
pass, no sort, no wide-column allocation — so cost now scales with **observations**,
not columns. Output is byte-identical to the pivot (verified cell-by-cell across the
grid).

## Pre/post profile (full `_etl_run`, p50, seed 0, 16-core host)

| grid point | before | after | delta |
|---|---|---|---|
| 3000a×252d | 95.8 | 61.4 | **−35%** |
| 2000a×252d | 63.7 | 45.9 | −30% |
| 1000a×756d | 75.8 | 60.0 | −20% |
| 100a×5040d | 45.0 | 41.4 | −9% |

`to_masked_matrix` alone at 3000a×252d: **37 → 9 ms (~4×)**. Improves at every grid
point; no stage flagged by `check_regressions`.

## System impact

Golden **held** — 17/17 PASS. `to_masked_matrix` is not on the golden pipeline path,
and its output is byte-identical, so the consumed price panel is unchanged.

**One documented behavior change**, confined to an *invalid-input* case: duplicate
`(date, id)` keys now tiebreak to "first row in input order" rather than "first in
`(date, id)`-sorted order". Duplicate keys are a hard invariant violation that
`etl.quality.check` flags and that never occur in production / golden / bench panels;
pinned by a test and called out in the PR's cons. Not ADR-worthy (no behavior change
on valid input).

Lane-pure: only `etl/masked_pivot.py` + `etl/test_masked_pivot.py` (+68/−15, +3 tests).
Clean revert.

## Suggested next steps

`adjust_prices` and the schema-validation pass are the next etl costs; both scale
near-linearly and have no obvious redundant work after this round.
