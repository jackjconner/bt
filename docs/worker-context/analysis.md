# analysis — turn a BacktestResult into performance, risk, and benchmark-relative analytics

## Files
- `engine.py` — `analyze_fused`, `benchmark_metrics_fused`, `BenchmarkMetrics` (single-pass fused metrics engine from shared moments; the perf-critical path).
- `metrics.py` — `AnalysisResult`, `BacktestAnalyzerImpl` (`.analyze`), `returns_from_nav`, `sharpe`, `max_drawdown` (POC + analyzer entry).
- `benchmark.py` — `alpha`, `beta`, `r_squared`, `tracking_error`, `information_ratio`, `up_capture`/`down_capture`, `active_returns`, `relative_drawdown`, `benchmark_returns_to_fractional`.
- `risk.py` — `cagr`, `annualized_return_calendar`, `sortino`, `calmar`, `var_historical`/`cvar_historical`, `skewness`, `excess_kurtosis`, `hit_rate`, `best_day`/`worst_day`.
- `turnover.py` — `one_way_turnover`/`two_way_turnover`, `net_nav`, `reconstruct_weights`, `gross_exposure`/`net_exposure`, `top_n_weight`, `effective_n`.
- `rolling.py` — `rolling_sharpe`/`rolling_vol`/`rolling_beta`/`rolling_max_drawdown`.
- `periodic.py` — `monthly_returns`/`monthly_returns_wide`/`quarterly_returns`/`annual_returns`.
- `attribution.py` — `factor_attribution`/`FactorAttributionResult`, `sector_attribution`.
- `report.py` — `analyze_attribution`, `AttributionReport`, `BrinsonDecomposition` (largest file).

## Public API (additive-only contract — do not break)
`__all__`: `AnalysisResult`, `AttributionReport`, `BacktestAnalyzer`, `BacktestAnalyzerImpl`, `BenchmarkMetrics`, `BrinsonDecomposition`, `FactorAttributionResult`, `active_returns`, `alpha`, `analyze_attribution`, `analyze_fused`, `annual_returns`, `annualized_return_calendar`, `benchmark_metrics_fused`, `benchmark_returns_to_fractional`, `best_day`, `beta`, `cagr`, `calmar`, `cvar_historical`, `down_capture`, `effective_n`, `excess_kurtosis`, `factor_attribution`, `gross_exposure`, `hit_rate`, `information_ratio`, `max_drawdown`, `monthly_returns`, `monthly_returns_wide`, `net_exposure`, `net_nav`, `one_way_turnover`, `quarterly_returns`, `r_squared`, `reconstruct_weights`, `relative_drawdown`, `returns_from_nav`, `rolling_beta`, `rolling_max_drawdown`, `rolling_sharpe`, `rolling_vol`, `sector_attribution`, `sharpe`, `skewness`, `sortino`, `top_n_weight`, `tracking_error`, `two_way_turnover`, `up_capture`, `var_historical`, `worst_day`.
Protocol (`_protocol.py`): `BacktestAnalyzer.analyze(self, result: BacktestResult) -> AnalysisResult`.

## Harness entry / hot path
`harness/components.py::_analysis_run` (component-benchmark path). Setup runs a `ProductionBacktestEngine` to produce a `BacktestResult`; the timed call is `BacktestAnalyzerImpl().analyze(result)` + `alpha`/`beta`/`information_ratio`/`two_way_turnover`. The fused `analyze` over the NAV/returns series dominates.

## Data contract
Consumes `BacktestResult` (nav_history, returns, trade_log), `benchmark_returns` (filtered to `BMK0`, scaled to `return_1d`). Scales with `n_dates` (series length); `two_way_turnover` scales with trade count.

## Recently optimized (don't re-attempt — see IMPROVEMENTS.md)
- Fused single-pass metrics engine: `analyze_fused` + `benchmark_metrics_fused` compute from shared moments instead of re-walking/re-joining per metric (2.78x suite, 4 joins to 1, sharpe bit-identical). PR #38. The suite win is only reachable through the new fused API.
