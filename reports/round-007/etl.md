---
round: 7
component: etl
pr: 54
date: "2026-05-31"
metric: "new capability (additive, flag-off default); consumed frame byte-identical (flag off)"
verdict: accepted
headline_delta: "optional data-quality flag columns behind include_quality_flags (default off)"
---

# etl · round 007

> **✓ Merged.** Landed as **PR #54** (`improve/etl-quality-flags`, commits
> `3ebad50` + `6b3ee98`) on `main`. A **`feature`** round — additive behind a
> default-off flag; with the flag off the price panel consumed by
> backtest/signals/analysis is shape- and value-identical. Re-validated post-merge:
> **1284 passed**, evalgate **17/0**.

Attach **data-quality flags as optional output columns** so downstream consumers can
gate or weight by data confidence without re-running the checks — Robustness +
Observability, the two pillars etl most directly serves.

## What it adds

`annotate_quality_flags(df, value_col, *, expected_dates, expected_ids,
spike_z_threshold)` **reuses** the existing `check(...)` logic and projects each
`QualityReport` finding onto the panel as a boolean column: `is_duplicate_key`,
`is_frozen_series`, `sparse_coverage`, `outlier_flagged`, plus per-row `price_stale`
(value unchanged from the same id's prior obs; first obs defaults not-stale).
`QUALITY_FLAG_COLUMNS` is the single source of column set + order. `adjust_prices`
gains `include_quality_flags` (default off → returns the existing object unchanged).

**Scope honesty:** the brief floated `is_halted` / `is_delisted` too — the worker
deliberately **did not** ship those, because `check()` doesn't receive the
universe/security-master inputs they need and synthesizing them would violate "reuse
the logic, don't reimplement." The shipped set maps 1:1 onto `check()`'s real
outputs — the additive discipline working as intended.

## How it's golden-safe

Flag off returns the frame exactly as today — proven on a 40×120 panel:
`adjust_prices(...).prices` with the flag off is `frame_equal` to before. With it on,
row count and every original column are byte-identical and only the 5 flag columns
are appended after them.

## Gates

- **lint** — clean; **no suppression markers** (an early draft's `# noqa: FBT003`
  was removed — the `fill_null(value=False)` keyword form satisfies `ty` honestly).
- **test** — `416 passed, 1 skipped` (+15).
- **bench** — no regression (`n_assets^0.78`); flag-off adds zero work.
- **eval** — 17/0; the residual 1e-16 ULP jitter reproduces on clean `main`.

## System impact

Downstream tradability masking / low-confidence weighting / quality observability all
become possible from a single flag, with no contract change when it's off. 2 commits,
all in the etl lane.
