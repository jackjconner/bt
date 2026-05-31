---
round: 7
component: portfolio
pr: 51
date: "2026-05-31"
metric: "new capability (additive, API-only); golden 17 fields byte-identical"
verdict: accepted
headline_delta: "FactorRiskModel.factor_risk_breakdown — total/factor/specific variance + per-factor attribution"
---

# portfolio · round 007

> **✓ Merged.** Landed as **PR #51** (`improve/portfolio-risk-decomp`, commit
> `f745f7f`) on `main`. A **`feature`** round — additive, read-only; weights /
> objective / constraints unchanged, production golden byte-identical. Re-validated
> post-merge: **1284 passed**, evalgate **17/0**.

Add **per-factor risk decomposition / attribution** so a portfolio's variance can be
split into the factor and specific sources that drive it.

## What it adds

`FactorRiskModel.factor_risk_breakdown(weights)` → frozen `FactorRiskBreakdown` with
`total_variance`, `factor_variance`, `specific_variance`, a per-factor
`factor_contrib` vector, and `factor_fraction` / `specific_fraction`. Built **on top
of** the existing `factor_component_contrib(w)` (no duplication — asserted by an
array-equality test) and routed through the factored Σ = B·F·Bᵀ + D form, so it never
materializes the dense `.cov` and stays off the optimizer hot path.

## How it's golden-safe

Read-only decomposition; `_protocol.py` untouched. Surfaced as an **API method
only** — not wired into `PipelineSummary` this round (most golden-conservative), so
no new field appears.

## Gates

- **lint** — clean; no suppressions.
- **test** — `408 passed, 1 skipped` (+9, incl. sum-to-portfolio-variance + analytic
  factored-form match).
- **bench** — no regression (`n_assets^0.53`).
- **eval** — 17/0, byte-identical.

## System impact

Unblocks downstream risk-attribution / tear-sheet capabilities (analysis can later
consume the breakdown) without touching the optimizer. Pairs naturally with the new
analysis drawdown-recovery table this same round. 3 files, all in lane.
