"""Drive component benchmarks across a GenSpec grid and produce a report.

For each param point in the grid the runner writes a full synthetic dataset,
then for each component: builds inputs (untimed) and runs repeated timed trials
via ``profiling.run_trials``.  Per-trial rows are persisted to Parquet through
``profiling.write_run`` (schema-compatible with ``stage_measurements``), then
log-log scaling curves are fit over the grid and — when baselines/thresholds
are supplied — a regression check is run against them.

This is the profiling module dogfooding itself: the harness measures every
other component using exactly the persistence/scaling/regression machinery the
profiling component ships.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from etl import DatasetLoader
from etl.datasets import GenSpec, write_all
from profiling import (
    RegressionReport,
    ScalingFit,
    TrialResult,
    capture_environment,
    check_regressions,
    fit_scaling,
    read_measurements,
    run_trials,
    write_run,
)

from .components import build_components
from .history import AgentAnnotation, AgentContext, write_history_run
from .spec import BenchmarkContext, ComponentBenchmark


@dataclass(frozen=True)
class HarnessReport:
    run_id: str
    stats: list[TrialResult]  # one per (component, param point)
    measurements: pl.DataFrame  # persisted stage_measurements rows
    scaling_fits: list[ScalingFit]
    regression: RegressionReport | None = None
    grid: tuple[GenSpec, ...] = field(default_factory=tuple)
    agent_ctx: AgentContext | None = None


def _params(spec: GenSpec) -> dict[str, int]:
    return {
        "n_assets": spec.n_assets,
        "n_dates": spec.n_dates,
        "n_features": spec.n_features,
        "n_factors": spec.n_factors,
    }


def run_harness(
    grid: list[GenSpec],
    store_dir: Path,
    *,
    components: list[ComponentBenchmark] | None = None,
    n_trials: int = 5,
    warmup: int = 1,
    run_id: str = "harness",
    baselines: pl.DataFrame | None = None,
    thresholds: pl.DataFrame | None = None,
    agent_ctx: AgentContext | None = None,
    annotation: AgentAnnotation | None = None,
    history_dir: Path | None = None,
) -> HarnessReport:
    components = components or build_components()
    store_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.datetime.now()
    env = capture_environment(run_id, trials=n_trials, warmup_trials=warmup)

    stats: list[TrialResult] = []
    trial_results: list[tuple[int, dict[str, int], str, list]] = []

    for pp, spec in enumerate(grid):
        data_dir = store_dir / f"data_pp{pp}"
        write_all(data_dir, spec)
        ctx = BenchmarkContext(
            spec=spec, loader=DatasetLoader(data_dir, spec), workdir=data_dir
        )
        params = _params(spec)

        for comp in components:
            inputs = comp.setup(ctx)
            tr = run_trials(
                comp.name,
                lambda c=comp, i=inputs: c.run(i),
                lambda out, c=comp: c.frames(out),
                n_trials=n_trials,
                warmup=warmup,
            )
            stats.append(tr)
            trial_results.append((pp, params, comp.name, list(tr.trials)))

    write_run(store_dir, env, trial_results)
    measurements = read_measurements(store_dir)
    fits = fit_scaling(measurements, run_id=env.run_id)

    regression: RegressionReport | None = None
    if baselines is not None and thresholds is not None:
        current = measurements.group_by("stage").agg(
            pl.col("elapsed_s").median(),
            pl.col("result_mb").median(),
            pl.col("peak_rss_mb").median(),
        )
        regression = check_regressions(current, baselines, thresholds)

    if history_dir is not None and agent_ctx is not None:
        write_history_run(
            history_dir=history_dir,
            run_id=env.run_id,
            run_ts=run_ts,
            git_sha=env.git_sha,
            agent_ctx=agent_ctx,
            stats=stats,
            scaling_fits=fits,
            regression=regression,
            annotation=annotation,
        )

    return HarnessReport(
        run_id=env.run_id,
        stats=stats,
        measurements=measurements,
        scaling_fits=fits,
        regression=regression,
        grid=tuple(grid),
        agent_ctx=agent_ctx,
    )


def print_harness_report(report: HarnessReport) -> None:
    print(f"\n{'=' * 78}")
    print(f"COMPONENT BENCHMARK HARNESS  (run_id={report.run_id})")
    print(f"{'=' * 78}")
    print(
        f"{'component':<14} {'p50_ms':>9} {'p90_ms':>9} "
        f"{'min_ms':>9} {'peak_mb':>9} {'result_mb':>10}"
    )
    print(f"{'─' * 14} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 10}")
    for s in report.stats:
        print(
            f"{s.stage:<14} {s.elapsed_p50 * 1e3:>9.2f} {s.elapsed_p90 * 1e3:>9.2f} "
            f"{s.elapsed_min * 1e3:>9.2f} {s.peak_rss_mb:>9.0f} {s.result_mb:>10.2f}"
        )

    if report.scaling_fits:
        print("\n  scaling curves (metric ∝ dim^slope):")
        for f in report.scaling_fits:
            flag = "  <-- super-linear" if f.log_log_slope >= 1.5 else ""
            print(
                f"    {f.stage:<12} {f.metric:<12} ∝ {f.scaling_dim}"
                f"^{f.log_log_slope:.2f}  (r²={f.r_squared:.2f}){flag}"
            )

    if report.regression is not None:
        status = "PASS" if report.regression.passed else "FAIL"
        print(
            f"\n  regression vs baseline: {status} ({len(report.regression.violations)} violations)"
        )
        for v in report.regression.violations:
            print(
                f"    {v.stage}/{v.metric}: {v.current_value:.4g} vs {v.baseline_value:.4g} "
                f"(+{v.pct_increase * 100:.0f}%, by {v.triggered_by})"
            )


__all__ = ["HarnessReport", "print_harness_report", "run_harness"]
