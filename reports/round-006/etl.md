---
round: 6
component: etl
pr: 47
date: "2026-05-31"
metric: "dead code removed (lines); golden byte-identical"
verdict: accepted
headline_delta: "remove dead _adjust_single_asset oracle — −193 lines, byte-identical"
---

# etl · round 006

> **✓ Merged.** Landed as **PR #47** (`consolidate/etl-drop-single-asset`) on
> `main` — the loop's first **`consolidate`** round (it *removes* a superseded
> path rather than adding one). Re-validated post-merge: pytest **1211 passed**
> (−16 oracle tests), golden byte-identical.

Delete the dead `_adjust_single_asset` oracle and make `_adjust_vectorized`
(round-005, PR #39) the **sole** corporate-action adjuster — **−193 net lines**,
public surface untouched, golden **byte-identical**.

## What it addressed

**Round type:** consolidate (the second half of the two-phase
add-then-consolidate flow). Round 005 vectorized `adjust_prices` but, under the
then-additive discipline, **kept** the old per-asset loop as a test oracle.
`adjust_prices` has delegated unconditionally to `_adjust_vectorized` ever since;
the loop survived only because the rule forbade deletion. With backwards-compat
dropped (see `DECISIONS.md`), it's dead weight to remove.

## How it decided

The orchestrator scouted all five round-005 shadows before dispatching. Four were
**live** and kept (signals `engine="matrix"` is the non-`rank` backend; models
`"loop"` is the non-ridge auto-dispatch fallback; backtest's scalar loop handles
non-`weight_space_eligible` configs; analysis's scalar fns are still called by
`report.py`). Only etl's `_adjust_single_asset` had **no production caller** —
referenced solely by `etl/test_adjust.py`. That made it the round's single clean
target.

## What was removed

- `_adjust_single_asset` (the per-asset loop) + the three scalar helpers used
  only by it (`_split_factor`, `_dividend_factor`, `_apply_factor`).
- The `_legacy_oracle` / `_random_panel` test scaffolding and the
  vectorized-vs-oracle **equivalence** test.
- Docstring/comment references to the "retained below" legacy path.
- Kept `_build_adj_log` (still live).

## System impact

**Correctness preserved, not reduced.** The equivalence cross-check became a
**direct** fixture test — `test_vectorized_back_adjusts_known_split_and_dividend_panel`
asserts `_adjust_vectorized` produces hand-computed back-adjusted closes (and
audit-log factors) on a two-asset split+dividend panel. Same guarantee, one
implementation.

- **Golden:** **byte-identical** — direct branch-vs-`main` `PipelineSummary` diff
  reports `Files are identical`. Removing dead code moved nothing.
- **Tests:** `402 passed, 1 skipped`; `test_adjust.py` 33 → 17 functions (16
  oracle/helper unit tests removed, 1 direct fixture added).
- **Profiling:** no regression — the hot path `_adjust_vectorized` is untouched
  (`etl elapsed_s ∝ n_assets^0.78`).
- **Public surface:** unchanged — `etl/__init__.py` and `_protocol.py` diffs are
  empty; `adjust_prices` / `AdjustmentResult` / `_adjust_vectorized` signatures
  and `__all__` untouched.

## Suggested next steps

1. The `consolidate` bar is "identical to pre-change `main`," not "golden at
   `--tolerance 0`" — the golden carries pre-existing epsilon fp-noise (1e-16)
   that `--tolerance 0` flags spuriously. Fold a `gate consolidate-check`
   (branch-vs-main `PipelineSummary` diff) into `scripts/gate` so the bar is
   measured directly.
