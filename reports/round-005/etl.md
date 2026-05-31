---
round: 5
component: etl
pr: 39
date: "2026-05-31"
metric: "etl p50 ms — adjust_prices"
verdict: pending
headline_delta: "adjust_prices −46% to −94% (~18× at 5040 dates); golden untouched"
---

# etl · round 005

> **⏳ Pending review.** Open as **PR #39** (`explore/etl-arrow-fuse`), **not yet
> merged**. Gates below ran on the branch; verdict flips to `accepted` on merge.

Replace the per-asset *partition → Python loop → concat* corporate-action
adjuster with a **whole-panel vectorized pass** — one `join_asof` plus a
segment-reset reverse-cumulative-product in log space — so `adjust_prices`
scales with the **number of actions, not the number of assets**: **−46% to −94%
(≈18× at the 5040-date point)**, golden untouched.

## What it addressed

**Component:** etl. **Metric:** etl benchmark p50 (`adjust_prices`, the dominant
cost in the timed etl path). The incumbent partitioned the panel by asset, ran a
Python loop applying back-adjustment factors per asset, and concatenated — cost
grew with #assets × #dates even though corporate actions are sparse.

## How it decided

Back-adjustment is a **suffix product of split/dividend factors**. Computed in
log space it becomes a segment-reset reverse cumulative *sum*, and the
factor-to-date attachment is a single `join_asof`. Both are whole-panel Polars
ops — no per-asset partition, no Python loop. The legacy `_adjust_single_asset`
is retained as a test oracle.

## Pre/post profile

etl p50, full grid seed 0, via `scripts/bench`:

| grid point | before | after | speedup |
|:-----------|:-------|:------|:--------|
| pt6 — 3000 assets | 226.47 ms | 121.30 ms | **−46% (~1.9×)** |
| pt10 — 100 × 5040 dates | 1103.36 ms | 62.57 ms | **−94% (~18×)** |

All 11 grid points improve, none regress (reproduced across two runs). Scaling
flattened to `n_assets^0.78` / `n_dates^0.77` — both sub-linear.

## System impact

`evalgate` vs golden — **17 PASS / 0 FAIL**. All 16 accuracy fields 0.00e+00
delta (`adjust_prices` is on the etl benchmark path, not the pipeline golden
path). Only `backtest_p50_s` moved (wall-clock, exempt). **pytest:** 1164 passed,
1 skipped (+7 vectorized-path tests, incl. an oracle cross-check vs the legacy
single-asset adjuster). Additive API only: `adjust_prices` signature,
`AdjustmentResult`, `__all__`, `_protocol.py` all unchanged; new code is a
private `_adjust_vectorized`. Touched only `etl/adjust.py` + `etl/test_adjust.py`.

## Suggested next steps

1. A benign polars `UserWarning` ("Sortedness of columns cannot be checked …")
   fires from the `join_asof`; the result is verified numerically identical to
   the legacy path and is documented in a comment (suppressing it would need a
   disallowed filter marker). If the warning becomes noisy, pre-sort the `by`
   key explicitly to silence it without a marker.
