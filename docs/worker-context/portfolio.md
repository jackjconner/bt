# portfolio — build portfolios with a real factor risk model and a constrained optimizer

## Files
- `optimizer.py` — `mean_variance` → `OptimizeResult` (mean-variance QP; SLSQP or OSQP solver, analytic gradient, cost/turnover penalty, no-trade band; factor-risk-model path bypasses dense cov). Largest file.
- `risk_model.py` — `FactorRiskModel` (Σ = B·F·Bᵀ + D, lazy cached `.cov`), `build_from_long`; `portfolio_variance`/`factor_variance`/`specific_variance`, marginal/component contribs.
- `covariance.py` — `sample_cov`, `ewma_cov`, `ledoit_wolf_cov` (shrinkage).
- `constraints.py` — `ConstraintSpec`, `from_polars` (exported as `constraints_from_polars`): long-only/short, per-asset bounds, net, gross, sector min/max.
- `tracking.py` — `tracking_error` (√((w−b)ᵀΣ(w−b))), `information_ratio`.
- `risk_metrics.py` — `parametric_var`/`parametric_cvar`, `var_cvar_table`.
- `schemes.py` — `equal_weight`, `inverse_vol`, `cap_weight`, `optimized_weight`, `apply_no_trade_band`, `turnover`, `transaction_cost`, `RebalanceResult`, `Scheme`.
- `risk.py` — POC `rolling_vol`, `var_historical`, `drawdown_series`.
- `factors.py` — POC `compute_exposures`, `random_loadings`, `FactorExposure`. `holdings.py` — `HoldingsFrame`.

## Public API (additive-only contract — do not break)
`__all__`: `ConstraintSpec`, `FactorExposure`, `FactorRiskModel`, `HoldingsFrame`, `OptimizeResult`, `PortfolioAnalyzer`, `RebalanceResult`, `Scheme`, `apply_no_trade_band`, `build_from_long`, `cap_weight`, `compute_exposures`, `constraints_from_polars`, `drawdown_series`, `equal_weight`, `ewma_cov`, `information_ratio`, `inverse_vol`, `ledoit_wolf_cov`, `mean_variance`, `optimized_weight`, `parametric_cvar`, `parametric_var`, `random_loadings`, `rolling_vol`, `sample_cov`, `tracking_error`, `transaction_cost`, `turnover`, `var_cvar_table`, `var_historical`.
Protocol (`_protocol.py`): `PortfolioAnalyzer.compute_exposures(self, holdings: HoldingsFrame) -> FactorExposure`; `.rolling_vol(self, holdings, returns, window) -> pl.DataFrame`.
Key sigs: `mean_variance(alpha, cov, spec, risk_aversion=1.0, w0=None, cost_per_unit=None, cost_scale=1.0, no_trade_band=0.0, max_iter=500, solver="slsqp", factor_risk_model=None) -> OptimizeResult`; `build_from_long(factor_loadings, factor_covariance, specific_risk, as_of_date) -> FactorRiskModel`.

## Harness entry / hot path
`harness/components.py::_portfolio_run` (component-benchmark path). Timed call: `build_from_long(...)` then `mean_variance(alpha, dummy_cov, cspec, solver="osqp", factor_risk_model=frm, max_iter=3000)`. The QP solve dominates; the harness deliberately uses the factor-model path (no dense n×n cov) so the gate measures production. NOTE: harness passes a dummy 1×1 cov — it is ignored when `factor_risk_model` is supplied.

## Data contract
Consumes `factor_loadings` (date,id,factor_id,loading), `factor_covariance` (date,factor_i,factor_j,cov), `specific_risk` (date,id,specific_var), `position_constraints`, `group_constraints`, `security_master`; alpha = z-scored momentum at `as_of`. Scales with `n_assets`, `n_factors`.

## Recently optimized (don't re-attempt — see IMPROVEMENTS.md)
- SLSQP → OSQP QP solver (`solver="osqp"`): ~50–100× faster, reaches true constrained optimum. PR #3.
- Factor-covariance path wired to production + lazy-materialized Σ: `mean_variance(factor_risk_model=frm)` bypasses the dense n×n build; scaling n_assets^2.11 → ^0.93, build 18–112× faster, peak RSS n² → flat. PRs #34 / #36.
