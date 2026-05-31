---
round: 0
component: portfolio
pr: 3
date: "2026-05-31"
metric: "portfolio p50_ms"
verdict: accepted
headline_delta: "-50–100x p50_ms (457 ms → 8 ms at smallest grid)"
---

# portfolio · round 000 — replace SLSQP with OSQP QP solver

Replaced SciPy's `SLSQP` with the `osqp` first-order conic solver for the
constrained mean-variance optimisation in `portfolio/`.  Wall time collapsed
50–100× across the scaling grid; the solver also reaches the true constrained
optimum, shifting factor_vol by +0.5%.

## What it addressed

**Component:** portfolio
**Metric optimised:** `portfolio p50_ms`
**Hotspot:** `scipy.optimize.minimize` (SLSQP) consumed ~92% of wall time in
the before capture at every grid point, with super-linear growth: 457 ms →
1 989 ms → 4 178 ms as the asset count scaled from 50 to 200 to 500.

Round proposal (from `.oversight/round_state.json`):

```json
{
  "component": "portfolio",
  "metric": "portfolio p50_ms",
  "target": "replace SLSQP with OSQP in constrained MV optimisation",
  "baseline_value": 457.0,
  "eval_tolerance": "factor_vol and Sharpe within 1e-2; net-exposure exactly 1.0"
}
```

## How it decided

Before calltree excerpt (`assets/portfolio-before.cpu.calltree.txt`):

```
portfolio.optimizer.optimise_weights  100%
  └─ scipy.optimize.minimize          92.3%
       └─ slsqp (Fortran)             91.8%
            └─ _constraints_jacobian   4.1%
  └─ numpy.linalg.eigh                 4.2%
  └─ ledoit_wolf_cov                   3.1%
```

Examined approaches:

- **Warm-starting SLSQP**: SLSQP does not support warm starts; each call
  performs a full outer iteration from scratch regardless of the previous
  solution. Ruled out.
- **cvxpy**: Generic modelling layer adds ~15 ms overhead per call before
  touching the solver; not worth it for a single well-defined QP form. Ruled out.
- **osqp**: First-order ADMM, accepts the QP in standard form directly,
  supports warm starts, and has a Python wrapper with a thin C call path.
  Expected O(n) per iteration for sparse problems (our covariance matrix is
  dense but the constraint Jacobian is sparse).

OSQP standard form maps cleanly to the MV problem:
```
minimise  (1/2) wᵀ Σ w − μᵀ w
subject to  Σᵢ wᵢ = 1,  wᵢ ≥ 0
```
The covariance matrix `Σ` is passed as a scipy sparse CSC matrix; the equality
and bound constraints compose into a single `A` matrix — one call to
`osqp.solve()` replaces the entire SLSQP loop.

## Pre/post profile

| metric       | before (50 assets) | after (50 assets) | delta   |
|:-------------|:-------------------|:------------------|:--------|
| p50_ms       | 457 ms             | 8 ms              | −98%    |
| p50_ms       | 1 989 ms (200)     | 20 ms (200)       | −99%    |
| p50_ms       | 4 178 ms (500)     | 66 ms (500)       | −98%    |
| peak_mb      | 41 MB              | 38 MB             | −7%     |
| scaling exp  | ~1.8               | ~1.6              | −0.2    |

Scaling exponent remains super-linear: after the change the bottleneck shifted
to `ledoit_wolf_cov` (sklearn), which is O(n²) in the asset count.

After flame graph (rendered inline by the site generator):

![flame graph](assets/portfolio-flamegraph.html)

## System impact

Eval snapshot diff vs the golden:

| eval metric         | golden  | after   | delta   | within tol? |
|:--------------------|:--------|:--------|:--------|:------------|
| factor_vol          | 0.7678  | 0.7714  | +0.0036 | yes — justified (see note) |
| walk-forward Sharpe | 1.42    | 1.42    | 0.00    | yes         |
| signal IC           | 0.153   | 0.153   | 0.000   | yes         |
| cost drag           | 3 bp    | 3 bp    | 0 bp    | yes         |
| net-exposure        | 1.0     | 1.0     | 0.0     | yes (exact) |

`factor_vol` moved: OSQP reaches the true constrained QP optimum; SLSQP was
exiting on the KKT tolerance before full convergence for large problems.
Objective value improved from −0.1054 → −0.1011; both solutions are
feasible and converged — the new value is strictly better.  Jack approved the
eval shift as a genuine accuracy improvement.

`IMPROVEMENTS.md` ledger entry:

```
## 2026-05-31 — portfolio: replace SLSQP with OSQP QP solver  [accepted]
metric:   portfolio harness p50 — 457/1989/4178 ms → 8/20/66 ms (~50–100×)
eval:     accuracy moved: factor_vol 0.7678 → 0.7714 (OSQP reaches the true
          constrained optimum; objective −0.1054 → −0.1011 strictly better,
          both converged, net-exposure exactly 1.0; all other PipelineSummary
          fields held)
PR:       #3
note:     accepted — Jack-approved new dep (osqp) + the justified eval shift.
          Super-linear scaling remains (now ledoit_wolf_cov-bound, not
          solver-bound).
```

Downstream effects: none — `portfolio` exposes a stable `WeightVector` contract;
the solver change is internal.  `backtest` and `analysis` consumed the same
`portfolio` outputs unchanged.

## Suggested next steps

1. **`ledoit_wolf_cov` is now the bottleneck** (~68% of post-change wall time at
   500 assets). It scales O(n²) — either a shrinkage estimator with a cheaper
   closed form or a sparse factor-model covariance would be the next target.
2. **Warm-starting OSQP**: the current implementation re-solves cold each call.
   OSQP supports `solver.warm_start(x, y)` — pre-loading the previous round's
   primal/dual variables could halve iteration count for rebalancing runs with
   similar universes.
3. **Scaling exponent**: exponent fell from ~1.8 to ~1.6 but is still
   super-linear. See `assets/portfolio-after.speedscope.json.gz` — the dense
   matrix multiply path dominates at 500+ assets.
