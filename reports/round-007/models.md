---
round: 7
component: models
pr: 52
date: "2026-05-31"
metric: "new capability (additive, flag-off default); golden byte-identical"
verdict: accepted
headline_delta: "per-fold IC dispersion + hit-rate diagnostics behind fold_ic_dispersion_enabled"
---

# models · round 007

> **✓ Merged.** Landed as **PR #52** (`improve/models-fold-diagnostics`, commit
> `7ff645b`) on `main`. A **`feature`** round — additive behind a default-off flag;
> flag-off `wf_mean_ic` / `wf_mean_r2` byte-identical, no new fields. Re-validated
> post-merge: **1284 passed**, evalgate **17/0** (`wf_mean_ic` `0.00e+00`).

Add **per-fold IC dispersion + hit-rate diagnostics** to walk-forward CV, so
fold-level stability (a correctness/robustness signal) is measurable without a re-run.

## What it adds

- `WalkForwardConfig.fold_ic_dispersion_enabled: bool = False`.
- `FoldResult.fold_ic_std` / `fold_hit_rate` and `WFResult.fold_diagnostics`
  (additive, default `None`).
- When on, `_attach_fold_diagnostics` derives `fold_ic_std` (population std of
  per-date IC) and `fold_hit_rate` (fraction of dates with IC > 0) **from the
  existing `ic_values`** — no IC recompute. Both CV engines (generic loop + batched
  ridge) thread the flag through one shared assembler, keeping them in lockstep.

## How it's golden-safe

Flag defaults off; the flag-off hot path is the exact pre-change code (zero added
work). New fields populated only when explicitly enabled.

## Gates

- **lint** — clean; no suppressions.
- **test** — `437 passed, 1 skipped` (+54 model tests: flag-toggle byte-identity,
  flag-on bounds, auto-vs-loop agreement).
- **bench** — no regression (flag-off adds zero work).
- **eval** — 17/0; `wf_mean_ic` bit-identical, `wf_mean_r2` at 1.11e-16 FP-noise.

## System impact

Fold-level diagnostics are the raw material for regime-stability auditing — a step
toward the Correctness pillar. +189/−1, 3 files, all in lane.
