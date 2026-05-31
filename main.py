from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from analysis import BacktestAnalyzerImpl
from backtest import BacktestConfig, BacktestEngine, SignalFrame
from etl import BatchLoader, ETLConfig, StreamLoader, write_parquet
from etl.datasets import GenSpec
from harness import AgentContext, print_harness_report, run_harness
from models import CVConfig, ModelConfig, cv_loop
from pipeline import print_pipeline_summary, run_production_pipeline
from portfolio import (
    HoldingsFrame,
    compute_exposures,
    random_loadings,
    rolling_vol,
)
from profiling import ScalingResult, collect_stage, print_report
from signals import ICEvaluator

PARAM_GRID: list[dict[str, int]] = [
    {"n_assets": 100, "n_dates": 252, "n_features": 20, "n_factors": 5},
    {"n_assets": 250, "n_dates": 252, "n_features": 30, "n_factors": 10},
    {"n_assets": 500, "n_dates": 504, "n_features": 50, "n_factors": 10},
    {"n_assets": 1000, "n_dates": 756, "n_features": 50, "n_factors": 20},
]

ROLLING_WINDOW = 63


def run_experiment(params: dict[str, int], workdir: Path) -> ScalingResult:
    na, nd = params["n_assets"], params["n_dates"]
    nf, nk = params["n_features"], params["n_factors"]
    profiles = []

    src = workdir / f"src_{na}_{nd}.parquet"
    write_parquet(src, na, nd, seed=0)
    cfg = ETLConfig(n_assets=na, n_dates=nd, source_path=src)

    returns, p = collect_stage(
        "etl.batch", lambda: BatchLoader(cfg).load(), lambda df: {"returns": df}
    )
    profiles.append(p)

    _, p = collect_stage("etl.stream", lambda: StreamLoader(cfg).load(), lambda df: {"out": df})
    profiles.append(p)

    signals = SignalFrame.random_continuous(na, nd, seed=1)

    bt_result, p = collect_stage(
        "backtest",
        lambda: BacktestEngine(BacktestConfig(n_assets=na, n_dates=nd)).run(returns, signals),
        lambda r: {"nav": r.nav_history, "trades": r.trade_log},
    )
    profiles.append(p)

    _, p = collect_stage(
        "analysis",
        lambda: BacktestAnalyzerImpl().analyze(bt_result),
        lambda a: {"returns": a.returns_series, "drawdown": a.drawdown_series},
    )
    profiles.append(p)

    holdings, p = collect_stage(
        "portfolio.holdings",
        lambda: HoldingsFrame.from_signals(signals),
        lambda h: {"holdings": h.df},
    )
    profiles.append(p)

    loadings = random_loadings(na, nk, seed=2)
    _, p = collect_stage(
        "portfolio.exposures",
        lambda: compute_exposures(holdings, loadings),
        lambda fe: {"loadings": fe.loadings, "exposures": fe.exposures},
    )
    profiles.append(p)

    _, p = collect_stage(
        "portfolio.rolling_vol",
        lambda: rolling_vol(holdings, returns, ROLLING_WINDOW),
        lambda df: {"vol": df},
    )
    profiles.append(p)

    _, p = collect_stage(
        "signals.ic",
        lambda: ICEvaluator().evaluate(signals, returns),
        lambda r: {"ic": r.ic_series},
    )
    profiles.append(p)

    rng = np.random.default_rng(3)
    X = rng.normal(0.0, 1.0, (nd, nf))
    y = rng.normal(0.0, 1.0, nd)
    _, p = collect_stage(
        "models.cv",
        lambda: cv_loop(ModelConfig(n_features=nf), X, y, CVConfig(n_splits=5)),
        lambda r: {},
    )
    profiles.append(p)

    return ScalingResult(params=params, stages=profiles)


PIPELINE_SPEC = GenSpec(n_assets=100, n_dates=252, n_features=20, n_factors=5, seed=0)

HARNESS_GRID = [
    GenSpec(n_assets=50, n_dates=126, n_features=10, n_factors=4, seed=0),
    GenSpec(n_assets=100, n_dates=126, n_features=10, n_factors=4, seed=0),
    GenSpec(n_assets=200, n_dates=126, n_features=10, n_factors=4, seed=0),
]


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="bt_scaling_"))
    try:
        for params in PARAM_GRID:
            result = run_experiment(params, workdir)
            print_report(result)

        pipe_dir = Path(tempfile.mkdtemp(prefix="bt_pipeline_"))
        try:
            print_pipeline_summary(run_production_pipeline(PIPELINE_SPEC, pipe_dir))
        finally:
            shutil.rmtree(pipe_dir, ignore_errors=True)

        history_dir = Path.cwd() / ".oversight" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        agent_ctx = AgentContext(
            agent_id="main",
            goal="manual profiling sweep across all components",
            strategy="run full HARNESS_GRID, record baseline measurements",
        )
        harness_dir = Path(tempfile.mkdtemp(prefix="bt_harness_"))
        try:
            print_harness_report(
                run_harness(
                    HARNESS_GRID,
                    harness_dir,
                    n_trials=3,
                    warmup=1,
                    history_dir=history_dir,
                    agent_ctx=agent_ctx,
                )
            )
        finally:
            shutil.rmtree(harness_dir, ignore_errors=True)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
