---
round: 3
component: backtest
pr: 28
date: "2026-05-31"
metric: "backtest p50_ms"
verdict: accepted
headline_delta: "3.4× @ 3000 assets"
---

# backtest · round 003

Batch-pivoting `_preprocess_inputs` removes 7 redundant Polars sort+pivot operations
per `run()` call, cutting wall time at n_assets=3000 from 817 ms to 239 ms (3.4×) while
leaving all financial outputs byte-identical.

## What it addressed

**Component:** backtest  
**Metric optimised:** `backtest p50_ms` — wall-time per `ProductionBacktestEngine.run()` call  
**Hotspot:** `_preprocess_inputs` → `_to_matrix_or_none` (Polars sort+pivot+to_numpy) called
once per value column, accounting for 63% of total runtime at n_assets=3000 (1.63 s of 2.58 s
CPU time in the before flame graph).

Round 3 proposal (from `.oversight/round_state.json`):

> Scale to production breadth — kill portfolio's n^1.89 time / n^1.97 memory
> (factor-model covariance) + flame-graph hotspot fixes across components.
> Swarm: each component flame-graphs and fixes its top hotspot at large n.

For backtest the assigned target was: find and fix the top within-stage hotspot at the
large grid points (n_assets=3000 and n_dates=5040), without moving any financial metric.

## How it decided

The worker captured a pyinstrument CPU flame graph at n_assets=3000, n_dates=252 (grid
point 6) before making any changes. The calltree read:

```
2.58s ProductionBacktestEngine.run  backtest/engine_pro.py
├─ 1.63s _preprocess_inputs                 ← 63% of total
│  ├─ 1.43s _to_matrix_or_none × 7          (each: DataFrame.sort + pivot + to_numpy)
│  └─ 0.20s _to_bool_matrix_or_none         (sort + cast-to-Float64 + pivot + to_bool)
├─ 0.44s to_matrix  etl/source.py           (returns + signals, called directly in run())
├─ 0.33s _assemble_result
└─ 0.11s _execute_rebalance                 (actual simulation)
```

At n_dates=5040 (100 assets, 20 yr), `_preprocess_inputs` consumed 43% (1.33 s of 3.06 s),
with `_execute_rebalance` rising to 18% as the date-loop work became visible.

**Ruled out:**

- **Caching between `run()` calls**: the public API takes `pl.DataFrame` objects by reference;
  caching would require the caller to hold a handle and add a cache-invalidation layer. The
  optimisation had to work inside a single `run()` call to preserve the API contract.
- **Replacing `to_matrix` for returns/signals**: those two calls each operate on a single value
  column and are invoked directly in `run()`, not in `_preprocess_inputs`. Batching them would
  require restructuring the `run()` public API, which was out of scope.
- **Polars lazy `select+collect`**: explored as a way to fuse the sort with the reshape, but
  `to_numpy()` on a lazy frame forces collect anyway; the extra lazy-frame overhead was not
  justified.

**Chosen approach:** A new `_df_to_multi_matrix` helper sorts by `(date, id)` **once**, calls
`to_numpy()` on all requested columns in one shot, and reshapes the contiguous block to
`(n_dates, n_assets, n_cols)`. Slicing `[..., col_idx]` extracts each column matrix.
This replaces N individual sort+pivot+to_numpy calls (one per column) with a single
sort+reshape pass per input DataFrame. `_to_bool_matrix_or_none` was also updated to use
the same sort+reshape pattern instead of the cast-to-Float64+pivot path.

The reshape is valid because all datasets produced by `write_all` are complete
`(date × id)` grids with no missing cells — the same invariant the original `pivot`
relied on, verified in the worker's correctness checks before implementation.

## Pre/post profile

| metric          | before          | after           | delta      |
|:----------------|:----------------|:----------------|:-----------|
| p50_ms (n_assets=3000, n_dates=252) | 817 ms | 239 ms | −71% (3.4×) |
| p50_ms (n_assets=100, n_dates=5040) | 278 ms | 181 ms | −35% (1.5×) |
| scaling exponent (n_assets) | ~1.6–1.7 (SMALL_GRID est.) | 0.93 (FULL_GRID, r²=1.00) | near-linear |
| peak_rss_mb (n_assets=3000) | — | 660 MB | — |

After flame graph (grid point 10, n_assets=100, n_dates=5040 — the largest date-sweep
point; `assets/backtest-after.calltree.txt`):

```
0.233s ProductionBacktestEngine.run  backtest/engine_pro.py
├─ 0.119s _execute_rebalance
│  └─ 0.095s _rebalance_weight_space
│     ├─ 0.044s compute_transaction_costs   ← new top hotspot (19%)
│     └─ 0.036s compute_slippage            (15%)
├─ 0.035s _assemble_result
│  └─ 0.035s DataFrame.__init__  polars
├─ 0.022s _compute_target_weights
│  └─ 0.017s _softmax
├─ 0.019s to_matrix  etl/source.py          (returns + signals)
├─ 0.009s _preprocess_inputs                ← was 63%, now 4%
│  └─ 0.007s _df_to_multi_matrix
│     └─ 0.005s DataFrame.sort
└─ 0.011s _mark_to_market
```

At n_assets=3000 (grid point 6, Duration: 0.200 s), `_preprocess_inputs` is 0.009 s (4%),
down from 1.63 s (63%) before. The simulation loop — `_execute_rebalance` → `_rebalance_weight_space`
→ `compute_transaction_costs` + `compute_slippage` — is now the dominant cost.

Only `backtest/engine_pro.py` was changed (+91 / −22 lines); no other file was touched.

## System impact

Eval gate (`scripts/eval --field-tol backtest_p50_s=1.0`): **ALL 17 PASS**.

All financial metrics are byte-identical (delta = 0.00e+00):

| eval metric             | golden  | after   | delta     | within tol? |
|:------------------------|:--------|:--------|:----------|:------------|
| signal IC (raw)         | +0.0482 | +0.0482 | 0.000     | yes         |
| walk-forward Sharpe     | +0.7597 | +0.7597 | 0.000     | yes         |
| cost drag               | 185,626 | 185,626 | 0         | yes         |
| backtest_p50_s          | 0.639 s | 0.080 s | −87%      | yes (exempt) |

Production scaling (clean harness run, all round-3 changes merged):

```
backtest  elapsed_s  ∝ n_assets^0.93  (r²=1.00)
backtest  elapsed_s  ∝ n_dates^0.99   (r²=1.00)
```

`IMPROVEMENTS.md` ledger entry:

```
## 2026-05-31 — performance swarm (flame-graph-driven): 5 components  [accepted]
metric:   ... backtest 3.4× ...
eval:     held byte-identical for the 4 pure-perf (#28/#30/#31/#32)
PR:       #28 #30 #31 #32 #34
note:     accepted — flame-graph hotspots removed: backtest 8× to_matrix→batch pivot; ...
```

Downstream effects: none. The change is internal to `ProductionBacktestEngine._preprocess_inputs`;
the public `run()` signature, return type (`BacktestResult`), and all downstream consumers
(pipeline, eval, harness) are unaffected.

## Suggested next steps

The after calltree shows the simulation loop has replaced preprocessing as the dominant cost:

1. **`compute_transaction_costs` / `compute_slippage`** are each 15–19% of runtime at the
   n_dates=5040 point (0.044 s and 0.036 s respectively at n_assets=3000). Both functions
   perform per-bar vectorised numpy operations — profiling their inner loops with a
   dedicated capture could reveal further gains (e.g., precomputing sparse cost masks).

2. **`_assemble_result` → `DataFrame.__init__`** (15% at n_assets=100 / n_dates=5040,
   0.035 s) builds the result `pl.DataFrame` from Python lists. Replacing the list-of-scalars
   pattern with pre-allocated numpy arrays handed directly to Polars constructors would
   reduce allocation pressure here.

3. **`to_matrix` for returns/signals** (8% at n_assets=3000, 0.019 s) still uses the
   single-column sort+pivot path. If both could be batched together (the engine always
   requests both in the same `run()` call), one sort pass would serve both.

4. **Scaling exponent n_assets^0.93** is now near-linear but just above 1.0 at the
   largest date-sweep point (`n_dates^0.99`). The `_execute_rebalance` date-loop is O(n_dates)
   by construction; eliminating the per-date Python overhead (e.g., vectorising the
   rebalance loop with a full-period numpy broadcast) is the next order-of-magnitude target.
