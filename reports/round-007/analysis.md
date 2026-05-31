---
round: 7
component: analysis
pr: 55
date: "2026-05-31"
metric: "new capability (additive); golden 16 numeric fields byte-identical"
verdict: accepted
headline_delta: "drawdown_recovery — per-event drawdown duration + time-to-recovery from the NAV path"
---

# analysis · round 007

> **✓ Merged.** Landed as **PR #55** (`improve/analysis-drawdown-recovery`, commit
> `50b5281`) on `main`. A **`feature`** round — additive; existing analysis scalars
> byte-identical. Re-validated post-merge: **1284 passed**, evalgate **17/0**.

Add **drawdown duration & recovery analytics** — *how long was I down, and how long
to climb back* — the risk-observability view a tear sheet needs. (Turnover, rolling
metrics, and periodic tables already exist in `analysis`; this fills the remaining
drawdown-event gap.)

## What it adds

`drawdown_recovery(nav) -> pl.DataFrame`, exported in `analysis/__init__.py`. It
isolates each drawdown **event** off the same `nav / nav.cum_max() - 1` series the
rest of `analysis` reports, one row per event: `peak_date`, `trough_date`,
`recovery_date` (null if unrecovered), `depth`, `drawdown_days` (peak→trough),
`recovery_days` (trough→recovery), `peak_to_recovery_days`. Durations in sessions,
matching the engine's session axis.

## How it's golden-safe

Additive only — a new function reusing the existing drawdown series; no
`PipelineSummary` field, not on the harness hot path. The depth of the deepest event
equals the existing `max_drawdown` (asserted by a test).

## Gates

- **lint** — clean; no suppressions.
- **test** — `390 passed, 1 skipped` (+8: monotonic→empty, single full-recovery,
  multiple events, unrecovered-tail→null, depth==`max_drawdown`, recovery-on-prior-
  peak, empty NAV).
- **bench** — no regression (`n_dates^0.19`, off the timed path).
- **eval** — all 16 numeric fields byte-identical.

## System impact

Drawdown-recovery events are core tear-sheet / risk-report material and pair
naturally with portfolio's new factor-risk breakdown this same round. Additive, in
lane.
