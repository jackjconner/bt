# signals — alpha research: score predictive content of signals vs forward returns

## Files
- `ic.py` — `ic_series`/`rolling_ic`/`ICEvaluator` (POC) + `ic_series_v2`, `ICMethod`, `ICEngine` (production cross-sectional IC; horizon column + method choice + coverage masking).
- `lazy_ic.py` — `spearman_ic_lazy`: long-format group_by rank-IC engine (default path for `ic_series_v2(engine="lazy")`).
- `horizon.py` — `ic_horizon_curve` → `HorizonCurve`/`HorizonPoint` (IC(h) decay over a horizon grid).
- `quantile.py` — `quantile_spread` → `QuantileResult` (decile bucket returns, top-minus-bottom spread, monotonicity).
- `neutralize.py` — `neutralize_sector`/`neutralize_factors`, `evaluate_neutralization` → `NeutralizationResult` (per-date OLS residualization).
- `coverage.py` — `pairwise_mask`, `apply_min_coverage`.
- `turnover.py` — `signal_autocorr`, `rank_stability`, `turnover_score` → `TurnoverResult`.
- `combine.py` — `zscore_blend`, `ic_weighted_blend`, `gram_schmidt_orthogonalize`, `incremental_ic`.
- `multiple_testing.py` — `rolling_ic_ir`, Bonferroni/BH t-stat deflation, `tstat_to_pvalue`.
- `regime.py` — `detect_regimes`, `regime_conditional_ic` → `RegimeConditionalICResult`.
- `newey_west.py` — `newey_west_tstat`, `default_lags`.

## Public API (additive-only contract — do not break)
`__all__`: `HorizonCurve`, `HorizonPoint`, `ICEngine`, `ICEvaluator`, `ICMethod`, `ICResult`, `IncrementalICResult`, `MultipleTestingResult`, `NeutralizationResult`, `QuantileResult`, `RegimeConditionalICResult`, `SignalEvaluator`, `TurnoverResult`, `apply_min_coverage`, `bh_correct`, `bonferroni_correct`, `default_lags`, `detect_regimes`, `evaluate_neutralization`, `gram_schmidt_orthogonalize`, `ic_horizon_curve`, `ic_series`, `ic_series_v2`, `ic_weighted_blend`, `incremental_ic`, `multiple_testing_correction`, `neutralize_factors`, `neutralize_sector`, `newey_west_tstat`, `pairwise_mask`, `quantile_spread`, `rank_stability`, `regime_conditional_ic`, `rolling_ic`, `rolling_ic_ir`, `signal_autocorr`, `spearman_ic_lazy`, `tstat_to_pvalue`, `turnover_score`, `zscore_blend`.
Protocol (`_protocol.py`): `SignalEvaluator.evaluate(self, signals: SignalFrame, returns: pl.DataFrame) -> ICResult`.
Key sig: `ic_series_v2(signals, forward_returns, *, signal_col="signal", return_col, method="rank", min_obs=10) -> pl.DataFrame`.

## Harness entry / hot path
`harness/components.py::_signals_run` (component-benchmark path). Timed call runs `ic_series_v2` + `neutralize_sector` + `ic_horizon_curve` (4 horizons) + `quantile_spread`. The cross-sectional IC compute (`ic_series_v2`/`spearman_ic_lazy`) dominates.

## Data contract
Consumes `alpha_signals` (filtered to `signal_name=="momentum"` → date,id,signal), `forward_returns` (`fwd_ret_1/5/21/63`), `security_master` (sector). Scales with `n_assets`, `n_dates`.

## Recently optimized (don't re-attempt — see IMPROVEMENTS.md)
- Lazy/streaming Polars Spearman-IC engine: long-format group_by rank-IC replaced dense pivot + scipy.rankdata; default `engine="lazy"` (−48%, bit-identical). PR #42.
- Earlier vectorization: rankdata-vectorized per-date Spearman loop, single horizon-grid pivot, Polars-native sector neutralization (−36 to −39%). PR #17 / #30.
