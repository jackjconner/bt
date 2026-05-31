---
round: 3
component: models
pr: 32
date: "2026-05-31"
metric: "models p50_ms"
verdict: accepted
headline_delta: "~1.9× @ 20yr (9.1s → 4.7s)"
---

# models · round 003

Eliminated a duplicate `rank_ic_series` call inside `_score_fold`, removing
the redundant O(n_test_dates × n_assets) Spearman ρ computation that ran
every fold — yielding ~1.9× wall-clock improvement at 20-year history depth.

## What it addressed

**Component:** models (`models/walk_forward.py`)
**Metric optimised:** `models p50_ms` (walk-forward CV wall time)
**Hotspot:** `_score_fold` consumed ~72% of CPU time at `n_dates=5040,
n_assets=100` (6.62 s out of 8.3 s), split equally across two calls to
`rank_ic_series` — one implicit via `rank_ic_score`, one explicit.

Round 3 targeted `models` as the next bottleneck after signals was optimised
in PR #17.  The harness showed models at ~190 ms p50 at the medium grid and
rising steeply with history length; the flame graph revealed the cause
immediately.

## How it decided

A `profiling.capture_cpu` run at `n_dates=5040, n_assets=100` produced the
following before calltree:

```
walk_forward_cv  (9.15s CPU total)
└─ _score_fold          6.620s  (72%)
   ├─ rank_ic_series    3.529s  ← called from rank_ic_score
   └─ rank_ic_series    3.091s  ← called again directly
```

`_score_fold` called `rank_ic_score(y_te, preds, grp_te)` to get `test_ic`
(which internally calls `rank_ic_series` and returns `.mean()`), and then
called `rank_ic_series(y_te, preds, grp_te)` a second time to obtain
`ic_values` for per-fold storage — **computing the same cross-sectional
Spearman ρ twice per fold**.

**Ruled out:**

- Caching / memoisation of `rank_ic_series`: would work but adds indirection
  and implicit state; the duplication is a straightforward code bug.
- Optimising `_spearman` itself: the call is inherently O(n_assets log
  n_assets) per date; no algorithmic shortcut available at this layer.
- Replacing scipy `spearmanr` with a hand-rolled rank correlation: valid
  future work but orthogonal to removing the duplicate call.
- Vectorising across folds: correct direction for a subsequent round; blocked
  by the current fold-serial structure.

**Chosen approach:** Drop the `rank_ic_score` call; call `rank_ic_series`
once and derive `test_ic = float(ic_values.mean())`.  Identical arithmetic —
same numbers, half the IC-scoring work per fold.  Diff is 1 file, 8
insertions, 2 deletions.

## Pre/post profile

The PR benchmarked 10 trials at `n_dates=5040, 2-warmup`:

| metric        | before     | after      | delta    |
|:--------------|:-----------|:-----------|:---------|
| p50 (direct)  | 9.084 s    | 4.704 s    | −48%     |
| min (direct)  | 5.940 s    | 3.355 s    | −44%     |
| scaling exp   | n_dates^1.0 | n_dates^1.0 | 0.00    |
| n_assets exp  | n_assets^?  | n_assets^0.65 | —     |

The clean harness run (`harness-20260531T111426-076075`) at the largest grid
point (`n_dates=5040, n_assets=533`) confirms:

| grid point       | p50_ms (after) |
|:-----------------|:---------------|
| n_dates=5040 (largest) | 3858 ms  |

Before harness p50 at 20yr (from PR pyinstrument capture): ~9 100 ms.
Improvement ratio at scale: **~1.9×**.

After calltree (`assets/models-after.calltree.txt`, grid index 10,
`n_dates=5040`):

```
walk_forward_cv  (CPU 7.258s)
└─ _score_fold          1.046s
   └─ rank_ic_series    1.045s  ← single call
      ├─ _spearman      0.865s
      │  └─ spearmanr   0.855s  [scipy/numpy]
      └─ [self]         0.169s
├─ _best_alpha          0.163s
│  └─ RidgeModel.fit    0.156s
├─ _scale_fold          0.060s
├─ _fit_fold            0.043s
└─ _build_fold_panel_df 0.042s
```

## System impact

Evaluation gate (`scripts/eval --field-tol backtest_p50_s=1.0`) reported all
17 fields within tolerance, with zero numerical delta on the IC and R² fields:

| eval metric   | golden      | after       | abs_delta | within tol? |
|:--------------|:------------|:------------|:----------|:------------|
| wf_mean_ic    | +0.759711   | +0.759711   | 0.00e+00  | yes         |
| wf_mean_r2    | +0.606416   | +0.606416   | 0.00e+00  | yes         |
| ic_raw        | +0.0482054  | +0.0482054  | 0.00e+00  | yes         |
| (14 others)   | —           | —           | 0.00e+00  | yes         |

The change is a pure deduplication: `mean(rank_ic_series(...))` vs
`rank_ic_score(...)` produce byte-identical floating-point results because
`rank_ic_score` is defined as exactly that expression.

Clean run scaling fit: `elapsed_s ∝ n_dates^0.97 (r²=1.00)` — history scaling
is unchanged (still approximately linear), as expected for an O(n)-per-fold
fix.  Asset scaling: `n_assets^0.65 (r²=0.97)` — sub-linear, reflecting the
alpha-grid search dominating at small date ranges.

`IMPROVEMENTS.md` ledger entry (from the round-3 swarm record):

```
## 2026-05-31 — performance swarm (flame-graph-driven): 5 components  [accepted]
metric:   ... models ~1.9× at scale ...
eval:     held byte-identical for the 4 pure-perf (#28/#30/#31/#32) ...
PR:       #28 #30 #31 #32 #34
note:     accepted — ... models duplicate rank_ic ...
```

Downstream effects: none.  `rank_ic_score` remains exported from
`models/__init__.py` via `scoring.py`; only the private import in
`walk_forward.py` was removed.  Public API (`WFResult`, `FoldResult`,
`walk_forward_cv` signature) is unchanged.

## Suggested next steps

With `_score_fold` no longer calling `rank_ic_series` twice, the after
calltree at the 20-year grid point shows a new leader: `_spearman` (scipy
`spearmanr`) accounts for 0.865 s / 7.26 s CPU — the entire remaining
`_score_fold` cost traces to a single per-fold Spearman call.

1. **`_spearman` / scipy `spearmanr`** is the next hotspot (12% of CPU in
   the after profile at the large grid point, and ~100% of `_score_fold`
   cost).  Options: replace with a vectorised numpy rank-correlation
   across all test dates in one call, or exploit the fact that assets are
   already ranked by `build_panel` to skip one `argsort`.

2. **`_best_alpha` / `RidgeModel.fit`** (0.163 s, second-largest child in
   the after calltree) runs the full alpha grid for every fold.  If the
   optimal alpha is stable across folds — which it often is — early-exit or
   warm-starting the grid search could cut this materially.

3. **`_build_fold_panel_df`** (0.042 s) constructs a Polars `DataFrame` from
   Python lists each fold.  Switching to pre-allocated arrays or reusing a
   schema object might reduce the per-fold allocation overhead.

4. **History scaling** (`n_dates^0.97`) is unchanged and still approximately
   linear — models is now the slowest component at the 20-year grid point
   (~3.9 s p50).  Fusing folds or parallelising across splits are the natural
   next architectural levers once the within-fold hotspots above are addressed.
