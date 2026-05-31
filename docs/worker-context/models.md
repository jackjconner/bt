# models — cross-sectional return models with leakage-safe cross-validation

## Files
- `walk_forward.py` — `walk_forward_cv`, `WalkForwardConfig`, `WFResult`, `FoldResult`, `FoldScaler` (per-fold train-only standardization, inner alpha grid, R²+IC; `engine="auto"` dispatches to batched core for ridge).
- `wf_batched.py` — `walk_forward_cv_batched`, `is_ridge_factory` (numpy-core engine: cumulative block moments + one Cholesky for all alphas).
- `panel.py` — `build_panel` → `PanelArrays` (align long feature panel + forward-return target on (date,id), NaN-mask), `date_ordinals`.
- `splitters.py` — `PurgedEmbargoCVSplitter`, `WalkForwardSplitter`, `RollingWindowSplitter`, `splits_from_calendar` (purge + embargo, never train on future).
- `ridge.py` — `RidgeModel`, `ModelConfig`, `ModelResult` (POC + the production fitter the harness uses).
- `models_zoo.py` — `LassoModel`/`LassoConfig`, `ElasticNetModel`/`ElasticNetConfig` behind `FinancialModel`.
- `boosting.py` — `GradientBoostModel`, `GradientBoostConfig`.
- `scoring.py` — `rank_ic_score`, `rank_ic_series`, `ic_stats`, `held_out_r2`.
- `persistence.py` — `save_artifact`/`load_artifact`/`predict_from_artifact`/`artifact_from_fold` → `ModelArtifact`.
- `cross_val.py` — POC `cv_loop`, `CVConfig`, `CVResult` (sklearn KFold — leaks; kept intact).
- `compare.py` — `compare_models`/`ModelComparison`. `leakage.py` — `audit_leakage`/`LeakageReport`/`CheckResult`.

## Public API (additive-only contract — do not break)
`__all__`: `CVConfig`, `CVResult`, `CheckResult`, `ElasticNetConfig`, `ElasticNetModel`, `FinancialModel`, `FoldResult`, `FoldScaler`, `GradientBoostConfig`, `GradientBoostModel`, `LassoConfig`, `LassoModel`, `LeakageReport`, `ModelArtifact`, `ModelComparison`, `ModelConfig`, `ModelResult`, `PanelArrays`, `PurgedEmbargoCVSplitter`, `RidgeModel`, `RollingWindowSplitter`, `WFResult`, `WalkForwardConfig`, `WalkForwardSplitter`, `artifact_from_fold`, `audit_leakage`, `build_panel`, `compare_models`, `cv_loop`, `date_ordinals`, `held_out_r2`, `ic_stats`, `is_ridge_factory`, `load_artifact`, `predict_from_artifact`, `rank_ic_score`, `rank_ic_series`, `save_artifact`, `splits_from_calendar`, `walk_forward_cv`, `walk_forward_cv_batched`.
Protocol (`_protocol.py`): `FinancialModel.fit(self, X, y, sample_weight=None) -> ModelResult`; `.predict(self, X) -> np.ndarray`.
Key sigs: `build_panel(features, target, target_col, *, weights=None, feature_cols=None) -> PanelArrays`; `walk_forward_cv(panel, splitter, model_factory, config=WalkForwardConfig()) -> WFResult`.

## Harness entry / hot path
`harness/components.py::_models_run` (component-benchmark path). Timed call: `walk_forward_cv(panel, WalkForwardSplitter(n_splits=4, embargo_periods=5), lambda a: RidgeModel(...))`. The per-fold fit/scale loop (now the batched ridge core) dominates.

## Data contract
Consumes `feature_panel` (date,id,feature_0..N), `forward_returns` (`fwd_ret_1` target), `sample_weights`. Scales with `n_assets`, `n_dates`, `n_features`.

## Recently optimized (don't re-attempt — see IMPROVEMENTS.md)
- Batched numpy-core walk-forward engine: per-fold standardized Gram as a difference of cumulative block moments, one Cholesky solves all alphas; wired via `WalkForwardConfig.engine="auto"` (2.41× at scale, byte-identical IC). PR #41.
- Earlier round removed duplicate `rank_ic` work (~1.9× at scale). PR #32.
