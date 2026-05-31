---
round: 7
component: signals
pr: 49
date: "2026-05-31"
metric: "new capability (additive); golden 17 fields byte-identical"
verdict: accepted
headline_delta: "signal_pair_correlation — cross-signal rank-correlation matrix + diversification ratio"
---

# signals · round 007

> **✓ Merged.** Landed as **PR #49** (`improve/signals-pair-correlation`, commit
> `2e317f8`) on `main`. A **`feature`** round — a new additive capability behind a
> stable API; the production golden held byte-identical. Re-validated post-merge:
> full suite **1284 passed**, evalgate **17/0**.

Add a **pair-wise signal rank-correlation + diversification** capability so a book
of alpha signals can be screened for redundancy *without* running a backtest.

## What it adds

`signal_pair_correlation(...)` → frozen `SignalCorrelationResult` carrying the
symmetric pair-wise **Spearman rank-correlation matrix** (unit diagonal, bounded
`[-1, 1]`), a **diversification ratio** `(1ᵀC1)/n²` in `[0, 1]` (lower = more
diversified), and the **mean absolute off-diagonal correlation**. Spearman =
Pearson on within-date ranks via Polars `rank()` "average" tie default (matches the
IC engine / scipy), pairwise-complete per-date masking.

## How it's golden-safe

Pure new surface: a new module + two exported symbols, no edit to the IC engine.
`ic_raw`, `ic_neutralized`, `horizon_ic[1/5/21/63]` untouched; nothing reaches
`PipelineSummary`.

## Gates

- **lint** — clean (ruff + ty); no suppressions.
- **test** — `142 passed` for `signals/` (+23), integration green.
- **bench** — no regression; off the timed hot path (`n_dates^0.93`).
- **eval** — 17/17 within tolerance; protected golden fields `0.00e+00`.

## System impact

Breadth for alpha research: signal selection / orthogonalization can now be informed
by measured cross-signal redundancy. Read-only, additive, no contract change.
