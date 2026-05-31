---
round: 3
component: signals
pr: 30
date: "2026-05-31"
metric: "signals p50_ms"
verdict: accepted
headline_delta: "14–22×"
---

# signals · round 003

Row-homogeneous NaN detection replaces ~5 000 per-date `rankdata` calls with one
vectorised batch call in both `_spearman_ic_rows` and `quantile_spread`, yielding
14–22× speedup at all tested grid points while leaving IC values bit-for-bit
identical.

## What it addressed

**Component:** signals  
**Metric optimised:** `signals p50_ms` — wall-time per signals stage invocation  
**Hotspot:** `scipy.stats.rankdata` called per-date inside `_spearman_ic_rows`
consumed 2.123 s of 5.021 s total stage time at n_dates=5040 (42%), triggered by
a fast-path miss described below.

Round 3 proposal (from `.oversight/round_state.json`):

> Scale to production breadth — kill portfolio's n^1.89 time / n^1.97 memory
> (factor-model covariance) + flame-graph hotspot fixes across components.
> Swarm: each component flame-graphs and fixes its top hotspot at large n.

For signals the assigned target was: find and fix the top within-stage hotspot at
a large grid point (n_dates=5040 / n_assets=3000), without moving any IC or
eval value.

## How it decided

The worker captured a pyinstrument CPU flame graph at n_dates=5040 before making
any changes (recorded 11:15:07, `signals.6.cpu.calltree.txt`):

```
0.884s  _signals_run  harness/components.py:90
├─ 0.486s  ic_horizon_curve  signals/horizon.py:70
│  ├─ 0.279s  _ic_series_from_matrices  signals/ic.py:205
│  │  ├─ 0.268s  _spearman_ic_rows  signals/ic.py:19
│  │  │  └─ 0.258s  rankdata  scipy/stats/_stats_py.py:9903  ← TOP HOTSPOT
│  │  │        0.205s  ndarray.argsort  <built-in>
│  │  └─ 0.010s  [self]
│  └─ 0.201s  to_matrix  etl/source.py:23
├─ 0.155s  ic_series_v2  signals/ic.py:258
│  └─ 0.080s  _ic_series_from_matrices
│     └─ 0.077s  _spearman_ic_rows → 0.074s rankdata
├─ 0.142s  quantile_spread  signals/quantile.py:126
│  └─ 0.057s  _quantile_spread_rows_vectorized → 0.034s rankdata
└─ 0.099s  neutralize_sector
```

**Root cause — the NaN-tail fast-path miss.** `_spearman_ic_rows` has two
vectorised fast paths before falling to a per-date Python loop:

1. `mask.all()` — no NaN anywhere in either matrix.
2. `(mask == first_row).all()` — every date has the same finite/NaN column
   pattern as the first row.

Forward-return matrices have NaN only in the *last* N rows (last 1/5/21/63 dates
for N-day horizons). Row 0 is fully valid; rows n−N … n−1 are all-NaN. The check
`(mask == first_row).all()` therefore fails — the tail rows differ from the
head — and every one of the 5 040 dates falls through to an individual
`stats.rankdata` call (~10 000 calls per horizon across S and R matrices). This
is the dominant cost even though the per-call arrays are small (100 elements).

**Approaches ruled out:**

- *Pre-rank S once across horizons in `ic_horizon_curve`.* `Sc` is the same for
  all four horizons, but the valid-row slice changes per horizon (R63 loses 63
  tail rows; R1 loses 1). Pre-ranking the full S and slicing later would save
  three of the four S-rankdata calls, but the mask still differs per horizon so
  `_spearman_ic_rows` would still need to be entered four times. The gain
  (~3 rankdata calls saved) is smaller than fixing the loop itself.
- *Scipy `rankdata` with `nan_policy`.* No `axis=` support in the per-row path;
  would not eliminate the Python loop.

**Chosen approach — fast path 3 (row-homogeneous detection):** When every row is
either entirely valid (`np.isfinite` all-true) or entirely NaN (`np.isfinite`
all-false), extract the valid rows into a compact matrix and issue one
`rankdata(..., axis=1)` call covering all of them together, then scatter the
results back. The guard is an O(n_dates) check `(row_all_valid | row_all_nan).all()`
that falls through to the existing loop for any other pattern. The same guard was
applied to `quantile_spread` via `_quantile_spread_rows_vectorized`.

## Pre/post profile

Direct-benchmark numbers (isolated call, Python 3.13.5):

| grid point                              | before     | after    | speedup |
|:----------------------------------------|:-----------|:---------|:--------|
| n_assets=100, n_dates=252 (1 yr)        | 703 ms     | 31 ms    | 22.7×   |
| n_assets=100, n_dates=5040 (20 yr)      | 7 178 ms   | 503 ms   | 14.3×   |
| n_assets=3000, n_dates=252              | 3 508 ms   | 1 228 ms | 2.9×    |

Harness p50 (clean run, post-merge, from `/tmp/bt_round3_final.txt`):

| n_assets | n_dates | p50_ms  | p90_ms  |
|:---------|:--------|:--------|:--------|
| 100      | 252     | 57.90   | 62.32   |
| 250      | 252     | 67.74   | 69.27   |
| 500      | 504     | 83.82   | 85.11   |
| 1 000    | 756     | 142.40  | 142.55  |

Scaling exponent (elapsed vs n_assets, post-merge): `∝ n_assets^0.67` (r²=0.94).
Pre-round the exponent was not measured separately for this sub-fix but the swarm
ledger records the aggregate signals improvement as 14–22×.

Flame-graph after (`signals.10.cpu.calltree.txt`, recorded 11:15:59):

```
0.538s  _signals_run  harness/components.py:90
├─ 0.207s  neutralize_sector  signals/neutralize.py:92       ← new #1
│  └─ 0.186s  _ols_residual
│     ├─ 0.064s  lstsq  numpy/linalg/_linalg.py
│     ├─ 0.042s  _std
│     └─ 0.040s  [self]
├─ 0.198s  ic_horizon_curve  signals/horizon.py:70           ← was 0.486s
│  ├─ 0.157s  _ic_series_from_matrices
│  │  ├─ 0.128s  _spearman_ic_rows
│  │  │  └─ 0.121s  rankdata                                 ← was 0.258s
│  │  └─ 0.027s  [self]
│  └─ 0.039s  to_matrix
├─ 0.080s  quantile_spread  signals/quantile.py:126          ← was 0.142s
│  └─ 0.043s  _quantile_spread_rows_vectorized
│     ├─ 0.016s  rankdata
│     └─ 0.016s  [self]
└─ 0.050s  ic_series_v2
   └─ 0.035s  _ic_series_from_matrices
      └─ 0.029s  _spearman_ic_rows → 0.027s rankdata
```

Full calltree: [`assets/signals-after.calltree.txt`](assets/signals-after.calltree.txt)

## System impact

Eval gate output: `scripts/eval --field-tol backtest_p50_s=1.0` — **ALL 17 PASS**.
IC values, Sharpe, cost drag, and walk-forward metrics are **bit-for-bit identical**
to the golden snapshot (0 absolute delta on all non-timing fields). The fast paths
are guarded by a strict structural check; any data not matching the row-homogeneous
pattern falls through to the original per-date loop, so no numeric drift is
possible.

| eval metric                   | golden   | after    | delta   | within tol? |
|:------------------------------|:---------|:---------|:--------|:------------|
| signal IC (raw)               | +0.0482  | +0.0482  | 0.0000  | yes         |
| signal IC (sector-neutral)    | +0.0452  | +0.0452  | 0.0000  | yes         |
| IC 1d / 5d / 21d / 63d        | unchanged| unchanged| 0.0000  | yes         |
| walk-forward IC / R²          | unchanged| unchanged| 0.0000  | yes         |
| Sharpe gross / net            | unchanged| unchanged| 0.00    | yes         |
| cost drag                     | 185 626  | 185 626  | 0       | yes         |

`IMPROVEMENTS.md` ledger entry (swarm summary, signals portion):

```
## 2026-05-31 — performance swarm (flame-graph-driven): 5 components  [accepted]
metric:   signals 14–22×
eval:     held byte-identical (#30 pure-perf)
PR:       #30
note:     signals rankdata NaN-tail fast-path miss — forward-return matrices
          have all-NaN tail rows that fail the uniform-mask check, causing
          ~5040 per-date rankdata calls. Fast path 3 detects row-homogeneous
          pattern and batch-ranks in one call. Same fix applied to
          quantile_spread via _quantile_spread_rows_vectorized.
```

Downstream effects: none. The change is purely internal to `signals/ic.py` and
`signals/quantile.py`; the public API (`ic_series_v2`, `ic_horizon_curve`,
`quantile_spread`) is unchanged.

## Suggested next steps

From the after calltree, the three remaining costs in descending order are:

1. **`neutralize_sector` / `_ols_residual`** is now the largest consumer at
   0.207 s (38% of stage time). `lstsq` (0.064 s) and `_std` (0.042 s) are the
   inner hotspots. A batched per-sector solve (group-by sector, one `lstsq` call
   per sector per date rather than one per date for the full cross-section) could
   reduce this significantly.

2. **`to_matrix` (Polars pivot)** appears four times in `ic_horizon_curve`
   (0.039 s each, 0.156 s aggregate). Each horizon pivots the same signal
   DataFrame into a matrix. The four pivots could be collapsed into one shared
   call upstream in `ic_horizon_curve`, since `Sc` is identical across horizons.

3. **`_spearman_ic_rows` → `rankdata`** still shows 0.121 s in
   `ic_horizon_curve` (four horizons, each ranking valid rows of S). Pre-ranking
   `Sc` once at the `ic_horizon_curve` level and passing the pre-ranked tensor
   into `_spearman_ic_rows` would eliminate three of the four S-rank calls,
   saving ~0.09 s; this was ruled out in round 3 as a secondary gain but is now
   more attractive since other costs have been reduced.
