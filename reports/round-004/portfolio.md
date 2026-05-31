---
round: 4
component: portfolio
pr: 36
date: "2026-05-31"
metric: "portfolio risk-model build ms / peak_mb"
verdict: accepted
headline_delta: "build 18-112x faster; peak mem n^2 -> flat"
---

# portfolio · round 004

Lazily materialize the dense factor covariance Σ and vectorize `build_from_long`,
removing the only O(n²) object from the risk-model build/optimizer hot path —
**build 18–112× faster and peak memory growth eliminated** (flat ~150 MB vs
climbing to 376 MB at 3000 assets), with the optimizer's numbers byte-identical.

> **Provenance.** This is the salvaged output of the loop's first `explore` round
> (a K=3 divergent tournament on `FactorRiskModel.build`). The round itself was
> abandoned after a tooling incident — the Workflow runner spawned unsupervised
> worktree copies that exhausted tmpfs — but one worker's rewrite was sound. It
> was reviewed and re-gated from scratch and shipped as **PR #36** (merged into
> `main`). See `DECISIONS.md` (heavy temp off tmpfs) and `SPIKES.md`.

## What it addressed

**Component:** portfolio
**Metric optimised:** `FactorRiskModel.build` / `build_from_long` wall time + peak RSS
**Hotspot:** post-Round-3, the factor-covariance assembly was ~54% of portfolio
time and carried the component's residual **super-linear memory** (≈ n^1.74). The
build formed the dense Σ = B·F·Bᵀ + D — an n×n object — and `build_from_long`
scattered the long frames into matrices with a per-row Python `iter_rows` loop.

## How it decided

The structural reading: **the dense Σ is the only n²-scaling object in the model,
and nothing on the live path reads it.** The optimizer consumes B / F_cov / D
directly, and `portfolio_variance` / `marginal_contrib` are exact in factor space:

```
Σ w = B (F (Bᵀ w)) + specific_var ⊙ w        # O(n·k + k²), never forms Σ
wᵀ Σ w = factor_variance(w) + specific_variance(w)
```

So Σ never needs to exist on the build path — only risk-decomposition reports and
PSD checks genuinely want the dense matrix.

The explore tournament ran three divergent strategies on this target:

- `lazy-polars` — fuse the whole build into one `LazyFrame` query.
- `numpy-sparse` — assemble Σ = BFBᵀ + D with `scipy.sparse` structured ops.
- `wide-layout` — pivot long→wide once and go matrix-native. **← won**

The winner doesn't fight to compute Σ faster — it **stops computing it at all** on
the hot path (lazy cached `.cov` property), and replaces the `iter_rows` scatter
with vectorized numpy (`np.unique`/`return_inverse` for B, `searchsorted` for
F_cov and specific_var). The dense path (`B @ F @ Bᵀ` plus a `np.diag` second n×n
temporary) is the thing it removed.

## Pre/post profile

Focused benchmark: `build_from_long` → `portfolio_variance` → `marginal_contrib`
at k=10 factors, peak RSS via `getrusage` (captures numpy buffers tracemalloc
cannot), median of 5 builds per point.

| n_assets | build ms before | build ms after | speedup | peak MB before | peak MB after |
|:---------|:----------------|:---------------|:--------|:---------------|:--------------|
| 500      | 3.34            | 0.19           | 18×     | 158            | 147           |
| 1000     | 8.52            | 0.31           | 28×     | 182            | 147           |
| 2000     | 22.3            | 0.43           | 52×     | 276            | 149           |
| 3000     | 49.3            | 0.44           | 112×    | 376            | 151           |

- **Time:** build scaling exponent ≈ **1.50 → 0.47** — super-linear to sub-linear.
  The gap *widens* with n (18× → 112×) because the removed work is the n² term.
- **Memory:** peak RSS grew **+218 MB** across 500→3000 before; **+4 MB** after.
  The n² growth is gone — the build is now flat in n (the ~150 MB floor is the
  interpreter + polars + numpy baseline, not the model).

(No flame graph asset: this was profiled with a targeted micro-benchmark during
PR salvage rather than the full `BT_FLAMEGRAPHS=1` harness; the whole-grid
flame-graph capture is the adjudication step before merge.)

## System impact

`python -m evalgate` vs the golden `PipelineSummary` — **16/17 fields hold the
golden exactly**; every substantive number is byte-identical or fp-noise:

| eval metric        | golden     | after      | delta     | within tol? |
|:-------------------|:-----------|:-----------|:----------|:------------|
| factor_vol         | 0.603993   | 0.603993   | 1.1e-16   | yes         |
| tracking_error     | 0.808416   | 0.808416   | 0.0       | yes         |
| opt_gross          | 1.96836    | 1.96836    | 0.0       | yes         |
| gross_sharpe       | 1.52787    | 1.52787    | 0.0       | yes         |
| net_sharpe         | -0.843769  | -0.843769  | 0.0       | yes         |
| cost_drag          | 185626     | 185626     | 0.0       | yes         |
| signal IC (raw)    | 0.0482054  | 0.0482054  | 0.0       | yes         |
| backtest_p50_s     | 0.0769     | 0.0743     | -3.5%     | timing*     |

\* `backtest_p50_s` is a wall-clock timing field (inherently noisy, moved
*faster*); exempt with `--field-tol backtest_p50_s=0.5`. In the benchmark above,
`portfolio_variance` and `marginal_contrib` were byte-identical before↔after at
every n — Σ semantics are exactly preserved.

Correctness gates: `ty` ✓ · `ruff` ✓ · `pytest -q portfolio/ tests/integration/`
→ **401 passed**. Downstream: the optimizer reads B / F_cov / D unchanged, so the
constructed portfolio is identical.

`IMPROVEMENTS.md` entry (merged, post-merge re-validation: pytest 401 passed, golden held):

```
## 2026-05-31 — portfolio: lazy-materialize Σ + vectorize build_from_long  [accepted]
type:     explore (salvaged winner of the abandoned tournament)
metric:   risk-model build @2k assets — 22.3 ms → 0.43 ms (52×); peak n² growth → flat
eval:     golden held (16/17 byte-identical; backtest_p50_s timing only, faster)
PR:       #36
note:     wide-layout / lazy-Σ strategy won; numpy-sparse and lazy-polars spiked (SPIKES.md).
```

## Suggested next steps

1. The dense Σ is still O(n²) **when explicitly accessed** (`.cov` for risk
   reports / PSD checks). If those become hot, compute decompositions block-wise
   in factor space rather than realizing the full matrix.
2. With the build no longer the bottleneck, the next portfolio hot path is
   `ledoit_wolf_cov` / the OSQP solve — profile the full grid (`BT_FLAMEGRAPHS=1
   uv run main.py`) at adjudication to confirm and pick Round 5's target.
3. Confirm the memory win on the production harness grid (this report used a
   component micro-benchmark); the whole-pipeline `peak_traced_mb` scaling fit
   should drop from ^1.74 toward linear.
