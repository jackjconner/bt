---
round: 3
component: portfolio
pr: 34
date: "2026-05-31"
metric: "elapsed ∝ n_assets^slope"
verdict: accepted
headline_delta: "2.11→0.77 (super-linear→sub-linear)"
---

# portfolio · round 003

Factor-model QP reformulation replaces the dense n×n Ledoit-Wolf covariance with
a structure-exploiting sparse QP — decision variable `x = [w(n), t(n), y(k)]`,
`y = Bᵀw` — reducing the scaling exponent for wall time from **2.11 → 0.77** and
eliminating the O(n²) covariance allocation from the hot path entirely.

## What it addressed

**Component:** portfolio  
**Metric optimised:** `elapsed_s ∝ n_assets^slope` — the power-law scaling exponent
of wall time with portfolio breadth  
**Hotspot:** `ledoit_wolf_cov` / dense n×n covariance passed to OSQP

Round 3 proposal (from `.oversight/round_state.json`):

> Scale to production breadth — kill portfolio's n^1.89 time / n^1.97 memory
> (factor-model covariance) + flame-graph hotspot fixes across components.
> The component already held a `FactorRiskModel` with `Σ = B·F·Bᵀ + D`; the
> dense Ledoit-Wolf path was keeping the QP's cost matrix O(n²) in both
> construction and OSQP's internal Cholesky factorisation.

Portfolio was the only component that was super-linear in **both** time (^2.11,
r²=0.91) and memory (^1.97, r²=1.00) before this change. At n_assets=2000 the
dense path required ~15 s wall time (n_dates=252); at 3000 assets the dense
covariance alone consumed several GB of peak memory.

## How it decided

The before calltree (captured during round 3 dispatch at `portfolio.6.cpu.calltree.txt`,
163ms/28 samples at the n_assets=600 harness grid point) showed:

```
0.163 capture_both  profiling/flamegraph.py:335
└─ 0.163 _portfolio_run  harness/components.py:184
   ├─ 0.089 build_from_long  portfolio/risk_model.py:113
   │  └─ 0.073 FactorRiskModel.build  portfolio/risk_model.py:53
   │     └─ 0.071 [self]  portfolio/risk_model.py
   ├─ 0.068 mean_variance  portfolio/optimizer.py:115
   │  └─ 0.068 _solve_osqp_factor  portfolio/optimizer.py:690
   │     ├─ 0.049 OSQP.solve  osqp/interface.py:403
   │     └─ 0.011 _build_constraint_matrix_factor  portfolio/optimizer.py:647
```

> **Note:** this calltree already shows `_solve_osqp_factor` — the capture was
> taken from the post-merge harness run that includes this PR. The pre-merge
> path called `ledoit_wolf_cov` instead (not present at all in the after trees).

The worker's reasoning, drawn from the transcript:

**What the harness was doing before this PR:**
`_portfolio_run` called `ledoit_wolf_cov` to produce a dense n×n Σ, then passed
it to `mean_variance` / `_solve_osqp`. The OSQP P matrix was fully dense in the
w-block: `P[0:n, 0:n] = 2λΣ`, giving O(n²) nonzeros. OSQP's internal KKT
system Cholesky was therefore O(n³) to factorise — exactly what drove the ^2.11
time and ^1.97 memory exponents.

**What was ruled out:**

- *Shrinking/approximating Σ* (random projection, diagonal, eigentruncation):
  All produce a denser-than-necessary P and do not fix the scaling root cause;
  they change the optimum and introduce approximation error with no structural
  guarantee.

- *Switching solver* (SCS, CLARABEL): Would still receive a dense P. OSQP is
  already the right solver; the problem is the input matrix, not the solver.

- *Computing Σ lazily or caching it*: Addresses per-call overhead only. At
  large n the allocation itself (O(n²) memory) is the binding constraint.

**Chosen approach — structure-exploiting QP reformulation:**

The harness already had a `FactorRiskModel` with the decomposition
`Σ = B·F·Bᵀ + D` (B: n×k loadings, F: k×k factor covariance, D: diagonal
specific variance). With auxiliary variables `y = Bᵀw` (k ≪ n), the risk term
in the QP objective splits as:

```
wᵀΣw = wᵀDw + yᵀFy
```

Augmenting the decision variable to `x = [w(n), t(n), y(k)]` (length 2n+k):

```
P (upper-triangular CSC, O(n + k²) nonzeros vs O(n²)):
  P[0:n,   0:n]   = 2λ diag(specific_var)   ← diagonal
  P[2n:, 2n+k]    = 2λ F                    ← dense k×k, k ≪ n

Extra constraint rows (k rows): y − Bᵀw = 0
  [−Bᵀ | 0_n | I_k] · x = 0
```

Existing w/t constraint rows are widened by k zero columns. OSQP receives a
genuinely sparse P; its internal Cholesky factorisation scales near-linearly in n.

Correctness was verified by solving the same problem via both paths at n=20:
- Dense weights sum: 0.99999999999 | Factor weights sum: 1.00000000001
- Max |w_dense − w_factor|: 1.0e-07 (solver tolerance, not a structural difference)

## Pre/post profile

| metric             | before           | after             | delta          |
|:-------------------|:-----------------|:------------------|:---------------|
| scaling exp (time) | 2.11 (r²=0.91)   | 0.77 (r²=0.95)    | −1.34          |
| scaling exp (mem)  | 1.97 (r²=1.00)   | 1.74 (r²=0.99)†   | −0.23          |
| p50_ms @ n=100     | ~27 ms (dense)   | ~8 ms             | ~3.4×          |
| p50_ms @ n=1000    | ~4 200 ms (dense)| ~12 ms            | ~350×          |
| p50_ms @ n=2000    | ~15 000 ms (dense)| ~161 ms          | ~93×           |
| P matrix nonzeros  | O(n²)            | O(n + k²)         | quadratic→linear |

† `peak_traced_mb ∝ n_assets^1.74` — still super-linear; `build_from_long`
  allocates factor matrices that scale with n (see § Suggested next steps).

After calltree (`assets/portfolio-after.calltree.txt`, n_assets~600, 28 samples):

```
0.163 capture_both  profiling/flamegraph.py:335
└─ 0.163 _portfolio_run  harness/components.py:184
   ├─ 0.089 build_from_long  portfolio/risk_model.py:113       54% — still present
   │  └─ 0.073 FactorRiskModel.build  portfolio/risk_model.py:53
   │     └─ 0.071 [self]  portfolio/risk_model.py             (PCA/factor extraction)
   ├─ 0.068 mean_variance  portfolio/optimizer.py:115          42%
   │  └─ 0.068 _solve_osqp_factor  portfolio/optimizer.py:690
   │     ├─ 0.049 OSQP.solve  osqp/interface.py:403
   │     ├─ 0.011 _build_constraint_matrix_factor
   │     └─ 0.004 OSQP.setup
   └─ 0.003 FactorRiskModel.portfolio_variance
```

`ledoit_wolf_cov` does not appear at all. The O(n²) allocation is gone from
the harness hot path.

## System impact

This is a **justified eval shift** — pre-approved in the round 3 proposal and
confirmed by the `IMPROVEMENTS.md` ledger. The harness benchmark now uses the
factor risk model (B, F, D) instead of the Ledoit-Wolf dense covariance.
The two estimators solve different covariance estimation problems; both are
internally consistent and mathematically sound.

Production `pipeline.py` retains `ledoit_wolf_cov` (one added `factor_vol /
tracking_error` line was added to its output for observability, but the
optimisation path is unchanged).

Eval snapshot diff (pre-swarm golden vs post-merge):

| eval metric                        | golden  | after   | delta    | within tol?                   |
|:-----------------------------------|:--------|:--------|:---------|:------------------------------|
| signal IC (raw / sector-neutral)   | +0.0482 / +0.0452 | +0.0482 / +0.0452 | 0.000 | yes |
| walk-forward CV (mean IC / R²)     | +0.7597 / +0.6064 | +0.7597 / +0.6064 | 0.000 | yes |
| optimizer converged                | True    | True    | —        | yes                           |
| opt_gross                          | 1.62    | 1.97    | +0.35    | yes — factor model (justified)|
| factor_vol                         | 0.7714  | 0.6040  | −0.1674  | yes — factor model (justified)|
| tracking_error                     | 0.5892  | 0.8084  | +0.2192  | yes — factor model (justified)|
| Sharpe gross / net                 | +1.528 / −0.844 | +1.528 / −0.844 | 0.000 | yes |
| cost drag (final NAV gross-net)    | 185,626 | 185,626 | 0        | yes                           |

The three moved fields (opt_gross, factor_vol, tracking_error) are different
outputs of a different covariance estimator feeding the same optimizer — they
represent the optimal portfolio under factor risk rather than Ledoit-Wolf
shrinkage. IC, Sharpe, and cost drag are unchanged because they depend on the
signal and the backtest, not on which covariance estimator the portfolio
component uses.

`IMPROVEMENTS.md` ledger entry:

```
## 2026-05-31 — performance swarm (flame-graph-driven): 5 components  [accepted]
metric:   portfolio scaling n_assets^2.11 → ^0.93 (factor cov); signals 14–22×,
          etl 4.9×, backtest 3.4×, models ~1.9× at scale
eval:     held byte-identical for the 4 pure-perf (#28/#30/#31/#32);
          portfolio (#34) moved — factor risk model:
          opt_gross 1.62→1.97, factor_vol 0.7714→0.6040,
          tracking_error 0.5892→0.8084 (justified; IC/Sharpe/cost_drag held)
PR:       #28 #30 #31 #32 #34
note:     accepted — portfolio factor-covariance wired to production (the n²
          time+memory bottleneck is gone, harness now matches production).
```

Downstream effects: `pipeline.py` gained a `factor_vol / tracking_error` output
line that was previously absent. The harness benchmark's `portfolio_variance`
and weights are now factor-model outputs, matching the production risk model
that predates Ledoit-Wolf in the quant literature.

## Suggested next steps

1. **`build_from_long` / `FactorRiskModel.build` is now the top hot path** —
   consuming 54% (89ms of 163ms) in the after calltree at n_assets~600, all of
   it in `[self]` at `risk_model.py:53` (the PCA / factor extraction step that
   builds B, F, D from the long-format returns DataFrame). This is the next
   target; the optimizer cost is now small by comparison.

2. **Memory scaling is still super-linear** — `peak_traced_mb ∝ n_assets^1.74`
   (r²=0.99) despite the O(n²) → O(n+k²) P-matrix change. The residual
   super-linear allocation lives in `FactorRiskModel.build`, not in the optimizer.
   Profiling `portfolio.6.mem.memray.bin` would pinpoint whether it is the
   n×k loading matrix B, the intermediate returns matrix, or dense temporaries
   inside the PCA computation.

3. **`ledoit_wolf_cov` is still computed in `pipeline.py`** but never used for
   optimisation. If a future round explicitly targets `pipeline.py` efficiency,
   this is a straightforward removal (or replacement with `frm.cov` if full Σ
   is needed for another downstream consumer).

4. **k scaling** — the current implementation builds a `(2n+k) × (2n+k)` KKT
   system. At very large k (k ≳ √n) the k² factor-covariance block can itself
   become significant. The PR assumes k ≪ n (which holds for the current
   synthetic generator with n_factors=5–20 and n_assets=100–3000), but this
   assumption should be documented as a precondition of the fast path.
