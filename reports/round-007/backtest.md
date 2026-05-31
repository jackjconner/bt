---
round: 7
component: backtest
pr: 53
date: "2026-05-31"
metric: "new capability (additive, flag-off default); golden byte-identical (flag off)"
verdict: accepted
headline_delta: "short-availability gating + financing behind enable_short_availability_gating (default off)"
---

# backtest · round 007

> **✓ Merged.** Landed as **PR #53** (`improve/backtest-short-gating`, commit
> `57c4bd7`) on `main`. A **`feature`** round — additive behind a default-off flag;
> with the flag off the production golden (`gross_sharpe`, `net_sharpe`, `cost_drag`,
> NAV path) is **byte-identical**. Re-validated post-merge: **1284 passed**, evalgate
> **17/0**.

Add **short-availability gating + financing costs** — the first realism step toward
modelling shorts the way a prime broker actually constrains them — behind
`enable_short_availability_gating` (default **OFF**, per the round decision to hold
the golden now and flip it on deliberately later).

## What it adds (when ON)

Forbids shorts on `shortable=False` names, caps each short's market value at
`loan_availability` (dollar→weight via NAV), and charges a daily per-asset **borrow
cost** on surviving shorts. New `apply_short_availability_cap` in `constraints.py`,
wired into the rebalance point and `_accrue_daily_costs`. Consumes the **existing**
`etl` `BORROW_RATES` dataset (`shortable` / `loan_availability` / `borrow_rate_bps`)
— **no cross-lane API request**. Enabling without `borrow_rates` raises `ValueError`
(a boundary assert, not a silent no-op). Excluded from the vectorized fast path
(nav-dependent).

## How it's golden-safe — and how it was proven

Flag defaults off. The proof is the careful part: the committed golden carries
pre-existing ~1e-16 last-ULP noise on 5 fields that **reproduces on clean `main`**.
To show byte-identity is *this change's* property, the worker saved a fresh golden
from clean `main` and ran the feature branch at `--tolerance 0` → every numeric field
`0.00e+00`. Flag-ON behavior is proven by dedicated tests.

## Gates

- **lint** — clean; no suppressions.
- **test** — `370 passed, 1 skipped` (backtest unit `91 passed`, +6).
- **bench** — no regression (`n_assets^0.92, n_dates^1.02`).
- **eval** (flag off) — 17/0; clean-main golden proof `0.00e+00` on every field.

## System impact

Realism for short books without disturbing the current golden: the capability ships
dormant, to be flipped on in a later deliberately golden-moving round. 4 files, all
in lane.
