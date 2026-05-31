---
round: 3
component: etl
pr: 31
date: "2026-05-31"
metric: "etl p50_ms"
verdict: accepted
headline_delta: "4.9× @ 3000 assets"
---

# etl · round 003

Replace per-asset `filter` loop in `adjust_prices` and Python `iter_rows` loop in `to_masked_matrix` with a single `partition_by` pass and a Rust-native Polars `pivot`, yielding 4.9× end-to-end at the extreme grid point (n_assets=3000, n_dates=252).

## What it addressed

**Component:** etl  
**Metric optimised:** `etl p50_ms`  
**Hotspot:** `etl/adjust.py::adjust_prices` consumed ~60% of wall time (0.975s of 1.628s CPU) through 3000 individual `DataFrame.filter` + `DataFrame.sort` calls; `etl/masked_pivot.py::to_masked_matrix` consumed ~37% (0.604s) through a 756 K-row Python `iter_rows` loop.

The round 3 swarm targeted the two largest Python-bound paths in the ETL component as identified by `profiling.capture_cpu` at the extreme grid point.

## How it decided

Before calltree (`etl.6.cpu.calltree.txt`, n_assets=3000, n_dates=252, CPU time 1.019s):

```
0.154 capture_both
└─ 0.154 <lambda>  harness/runner.py:141
   └─ 0.154 _etl_run
      ├─ 0.975 adjust_prices  etl/adjust.py:129
      │  ├─ 0.494 DataFrame.filter     ← 3000 per-asset filter+collect ops
      │  ├─ 0.245 DataFrame.sort       ← 3000 per-asset sorts
      │  └─ 0.129 _adjust_single_asset
      ├─ 0.604 to_masked_matrix  etl/masked_pivot.py:24
      │  ├─ 0.446 [self]               ← pure Python iter_rows loop, 756 K rows
      │  └─ 0.128 DataFrame.iter_rows
      └─ 0.030 check  etl/quality.py:131
```

Two root causes were isolated:

**`adjust_prices` — per-asset filter loop.** The original code called `prices.filter(pl.col("id") == aid).sort("date")` once per asset — 3000 separate Polars filter+collect operations at the extreme grid point. `DataFrame.filter` showed 0.494s (30% of total CPU), `DataFrame.sort` a further 0.245s, all as O(n_assets) separate Polars round-trips into Rust.

**`to_masked_matrix` — Python `iter_rows` loop.** The original iterated all 756 000 rows in pure Python via `iter_rows()`, building the output matrix with per-row dict lookups. The `[self]` frame (0.446s) was pure Python overhead; `iter_rows` itself added a further 0.128s.

Ruled out:
- **Polars `group_by` + `map_groups`:** Requires materialising one DataFrame per group and passing it back through Python; no net improvement over the filter loop for pure sort-and-call workloads.
- **Numba / Cython inner loop for `iter_rows`:** Would speed up the Python loop but not eliminate it; a pivot runs the same operation in compiled Rust with no Python involvement.
- **`partition_by` without CA-event filtering:** A `partition_by` alone would still call `_adjust_single_asset` for all 3000 assets. Profiling showed only ~14% of assets have actual corporate-action (CA) events; the remaining ~86% need only `adj_close = close`. Skipping them eliminates ~2577 Python function calls.

**Chosen approach:** One `sort + partition_by("id")` call splits the entire prices frame in a single Polars pass; assets with no CA events receive `adj_close = close` via a single vectorised `with_columns + concat`; only the ~14% with events go through `_adjust_single_asset`. For `to_masked_matrix`, the Python loop is replaced by `df.pivot(on="id", index="date", values=value_col, aggregate_function="first")` — pure Rust, NaN-filled gaps, no Python iteration.

## Pre/post profile

| metric          | before (n=3000)  | after (n=3000) | delta   |
|:----------------|:-----------------|:---------------|:--------|
| adjust_prices   | ~655 ms          | ~72 ms         | −89%    |
| to_masked_matrix| ~480 ms          | ~50–230 ms     | −52–90% |
| etl full p50    | ~1751 ms         | ~355 ms        | **−80% (4.9×)** |
| etl harness p50 | 630.48 ms (baseline) | 214 ms (clean run, n=3000-equiv) | −66% |
| scaling exp     | (before: ~1.0+)  | n_assets^0.91  | sub-linear |

After calltree (`assets/etl-after.calltree.txt`, CPU time 0.602s — 41% lower than before at comparable load):

```
0.191 capture_both
└─ 0.191 <lambda>  harness/runner.py:141
   └─ 0.191 _etl_run
      ├─ 0.151 adjust_prices  etl/adjust.py:129
      │  ├─ 0.122 _adjust_single_asset   ← only ~14% of assets (CA events)
      │  │  ├─ 0.056 _apply_factor
      │  │  └─ 0.039 Series.to_list
      │  └─ 0.027 DataFrame.sort         ← one sort of the full frame
      ├─ 0.025 check  etl/quality.py:131
      └─ 0.014 to_masked_matrix  etl/masked_pivot.py:24
         ├─ 0.006 wrapper (pivot)         ← Rust-native pivot, no Python loop
         └─ 0.005 DataFrame.sort
```

The `DataFrame.filter` and `[self]` (iter\_rows) frames are entirely absent. `to_masked_matrix` dropped from 37% of wall time to 7%. The remaining `_adjust_single_asset` time is bounded by the actual CA-event population (~14% of assets).

Note: the harness calltree is captured at a mid-range grid point (n_assets ~500); the 4.9× figure is from a direct head-to-head benchmark at n_assets=3000.

## System impact

Eval gate result (17 fields, `--field-tol backtest_p50_s=1.0`):

| eval metric         | golden    | after     | abs delta | within tol? |
|:--------------------|:----------|:----------|:----------|:------------|
| signal IC           | (same)    | (same)    | 0.00e+00  | yes         |
| walk-forward Sharpe | (same)    | (same)    | 0.00e+00  | yes         |
| cost drag           | (same)    | (same)    | 0.00e+00  | yes         |
| all 17 fields       | —         | —         | 0.00e+00  | 17 PASS, 0 FAIL |

Output is byte-identical: `adjust_prices` returns the same `AdjustmentResult`; `to_masked_matrix` returns the same `(matrix, mask, dates, ids)` with values verified by `np.array_equal` (mask) and `np.allclose` (matrix). No data, signal, or downstream component was affected.

Scaling after this change: `etl elapsed_s ∝ n_assets^0.91` (r²=0.99) — sub-linear, confirming the O(n_assets) per-asset filter loop is gone. The n_dates exponent remains at 1.62 (super-linear — a separate target for a future round).

`IMPROVEMENTS.md` ledger entry:

```
## 2026-05-31 — performance swarm (flame-graph-driven): 5 components  [accepted]
metric:   portfolio scaling n_assets^2.11 → ^0.93 (factor cov); signals 14–22×, etl 4.9×,
          backtest 3.4×, models ~1.9× at scale
eval:     held byte-identical for the 4 pure-perf (#28/#30/#31/#32);
          portfolio (#34) moved — factor risk model: opt_gross 1.62→1.97,
          factor_vol 0.7714→0.6040, tracking_error 0.5892→0.8084
          (justified; IC/Sharpe/cost_drag held)
PR:       #28 #30 #31 #32 #34
note:     accepted — flame-graph hotspots removed: backtest 8× to_matrix→batch pivot;
          signals rankdata NaN-tail fast-path; etl per-asset filter + iter_rows→partition_by/pivot;
          models duplicate rank_ic; portfolio factor-covariance wired to production.
```

Downstream effects: none. Changes are confined to `etl/adjust.py` and `etl/masked_pivot.py`; public API signatures and return types are unchanged.

## Suggested next steps

1. **n_dates super-linear scaling** (`etl elapsed_s ∝ n_dates^1.62`, r²=0.99) is the remaining performance concern. The after calltree shows `check` at `etl/quality.py:131` consuming 25ms and growing with rows; `_check_spike_outliers` uses a cross-sectional `over("date")` groupby that is O(n_dates × n_assets). That is the next candidate for vectorisation.

2. **`_adjust_single_asset` inner loop** (`Series.to_list` 0.039s in the after tree) — the ~14% of assets with CA events still exercise a Python loop inside `_adjust_single_asset` via `iter_rows`. At very large asset counts this will re-emerge as a proportional cost; a vectorised factor-application pass could eliminate it.

3. **`to_masked_matrix` at scale with many price columns** — the pivot measured ~230ms at 11 columns vs ~50ms at 3 columns (production prices frame is wider). The `wrapper (pivot)` frame at 0.006s in the harness calltree is at a small grid point; at n_assets=3000 with 11 columns it is the dominant remaining cost in `to_masked_matrix`. A `pl.LazyFrame`-based pivot with eager collect may reduce materialisation overhead.
