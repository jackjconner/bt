# profiling — measure cost (time + memory) of the other components; baselines, scaling curves, regression verdicts

## Files
- `flamegraph.py` — `capture_both`/`capture_cpu`/`capture_memory` (in-process pyinstrument + memray), `ProfileArtifact`, `write_artifacts`/`read_artifacts`, `prune_profiles`. Largest file.
- `trials.py` — `run_trials` (warmup-discarded repeated trials → min/p50/p90/p95/stddev + tracemalloc peak) → `TrialResult`/`TrialStats`/`TrialMeasurement`.
- `scaling.py` — `fit_scaling` (log-log OLS slope per stage×metric×dim) → `ScalingFit`, `fits_to_dataframe`.
- `regression.py` — `check_regressions` (pct OR abs thresholds vs baselines) → `RegressionReport`/`RegressionViolation`.
- `storage.py` — `write_run`/`read_runs`/`read_measurements` (append-friendly parquet, schema-compatible with profiling_runs/stage_measurements).
- `environment.py` — `capture_environment` (git sha/dirty, host, cpu, RAM, lib versions, BLAS threads) → `RunEnvironment`.
- `output.py` — `write_json`, `write_measurements_parquet`.
- `report.py` — POC `collect_stage`, `print_report`, `StageProfile`, `ScalingResult`. `timer.py` — `StageTimer`, `TimingResult`. `memory.py` — `rss_mb`, `obj_size_mb`, `frames_size_mb`, `snapshot`, `MemSnapshot`.

## Public API (additive-only contract — do not break)
`__all__`: `MemSnapshot`, `ProfileArtifact`, `RegressionReport`, `RegressionViolation`, `RunEnvironment`, `ScalingFit`, `ScalingResult`, `StageProfile`, `StageTimer`, `TimingResult`, `TrialMeasurement`, `TrialResult`, `TrialStats`, `capture_both`, `capture_cpu`, `capture_environment`, `capture_memory`, `check_regressions`, `collect_stage`, `fit_scaling`, `fits_to_dataframe`, `frames_size_mb`, `obj_size_mb`, `print_report`, `prune_profiles`, `read_artifacts`, `read_measurements`, `read_runs`, `rss_mb`, `run_trials`, `snapshot`, `write_artifacts`, `write_json`, `write_measurements_parquet`, `write_run`.
No `_protocol.py` (the package has no Protocol — the stable surface is `report.py`/`trials.py` types above).
Key sigs: `run_trials(stage, fn, frames_after_fn, n_trials=5, warmup=1) -> TrialResult`; `fit_scaling(measurements, run_id, metrics=("elapsed_s","result_mb","peak_traced_mb","peak_rss_mb"), scaling_dims=("n_assets","n_dates","n_features","n_factors"), min_points=3) -> list[ScalingFit]`; `check_regressions(current_metrics, baselines, thresholds) -> RegressionReport`.

## Harness entry / hot path
`harness/components.py::_profiling_run` (component-benchmark path; profiling is dogfooded — it analyzes its own telemetry schema). Timed call: `fit_scaling(measurements, run_id="harness_self")` + a Polars group_by aggregation + `check_regressions(current, baselines, thresholds)`. `fit_scaling` (per-stage log-log OLS) dominates.

## Data contract
Consumes `stage_measurements` (stage,elapsed_s,result_mb,peak_rss_mb,n_assets,…), `stage_baselines`, `regression_thresholds`. Also reads/writes `profiling_runs`, `scaling_fits`, `cpu_profile_frames`. Scales with row count of the measurements frame.

## Recently optimized (don't re-attempt — see IMPROVEMENTS.md)
- No dedicated profiling perf round has shipped (profiling is the measurement infra, not a perf target). Treat IMPROVEMENTS.md as authoritative; check it before targeting this component.
