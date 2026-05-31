---
round: 5
component: analysis
pr: 38
date: "2026-05-31"
metric: "analysis benchmark-suite µs (4 joins → 1)"
verdict: accepted
headline_delta: "benchmark suite 2.78× (4 joins → 1); golden bit-identical"
---

# analysis · round 005

> **✓ Merged.** Landed as **PR #38** (`explore/analysis-fused-engine`) on `main`.
> Gates ran on the branch and were re-validated post-merge with the other
> round-005 changes (full suite **1215 passed, 1 skipped**; evalgate **17/17**).

Fuse the analysis metrics suite into a single-pass engine (`analyze_fused` +
`benchmark_metrics_fused`) so each metric cluster computes from **shared
moments** instead of re-walking the returns frame and re-joining per metric —
**2.78× on the benchmark suite** (four joins collapsed to one), with every
reported number held bit-identical to the golden.

## What it addressed

**Component:** analysis (first explore round on it — no prior perf work).
**Metric:** the benchmark-metrics suite (alpha / beta / information-ratio /
turnover) and the core `analyze` walk over the returns + trade frames. Each
metric independently re-derived the same moments and re-joined the benchmark
series, so the suite paid four passes/joins for one logical pass.

## How it decided

The structural reading: the suite's metrics are all functions of a **small set
of shared moments** (returns mean/var, cross-moments with the benchmark, trade
aggregates). Compute those once, then every scalar metric is an arithmetic
combination — no per-metric walk, no per-metric join. `analyze` keeps its public
signature and delegates to the fused path.

## Pre/post profile

Isolated micro-benchmarks (n_dates = 756):

| path | before | after | speedup |
|:-----|:-------|:------|:--------|
| benchmark suite (4 scalar fns) | 470.4 µs | 169.1 µs | **2.78×** (4 joins → 1) |
| `analyze` | 236.8 µs | 221.4 µs | 1.07× |

Harness `analysis` component p50: 1.816 → 1.743 ms (−4%, noisy — the harness
also times un-fused alpha/beta/IR/turnover paths it calls directly). No other
stage regressed.

## System impact

`evalgate` vs the golden — **17 PASS / 0 FAIL**. The fields that flow through the
rewritten `analyze` (`gross_sharpe`, `net_sharpe`) are **bit-identical**
(abs_delta 0.00e+00) — the analytics-hold-exactly bar is met. **pytest:** 1165
passed, 1 skipped (baseline 1157+1; +8 equivalence tests). Additive API only:
three new exported symbols (`analyze_fused`, `benchmark_metrics_fused`,
`BenchmarkMetrics`), no signatures changed.

## Suggested next steps

1. The bigger 2.78× win is currently only reachable through the **new API** — the
   harness still calls the un-fused scalar functions directly. Wiring the harness
   to `benchmark_metrics_fused` is cross-lane → an `API_REQUESTS.md` entry
   (analysis → harness) for a later round.
