"""Tests for history.py — three-table agentic profiling history."""

from __future__ import annotations

import datetime
from pathlib import Path

from harness.history import (
    AgentAnnotation,
    AgentContext,
    read_agent_annotations,
    read_component_snapshots,
    read_improvement_runs,
    write_history_run,
)
from profiling.trials import TrialResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMPONENTS = ("etl", "signals", "models", "analysis", "portfolio", "backtest")


def _trial_result(stage: str, p50_s: float) -> TrialResult:
    return TrialResult(
        stage=stage,
        n_trials=3,
        elapsed_min=p50_s * 0.9,
        elapsed_p50=p50_s,
        elapsed_p90=p50_s * 1.1,
        elapsed_p95=p50_s * 1.15,
        elapsed_stddev=p50_s * 0.05,
        result_mb=10.0,
        rss_delta_mb=2.0,
        peak_rss_mb=500.0,
        peak_traced_mb=1.0,
        trials=(),
    )


def _stats(p50_by_comp: dict[str, float]) -> list[TrialResult]:
    return [_trial_result(comp, p50) for comp, p50 in p50_by_comp.items()]


def _ctx(target: str | None = "etl") -> AgentContext:
    return AgentContext(
        agent_id="test-agent",
        goal="reduce etl p50",
        strategy="vectorize loops",
        target_component=target,
    )


def _write(
    tmp_path: Path,
    run_id: str,
    p50_by_comp: dict[str, float],
    run_ts: datetime.datetime | None = None,
    annotation: AgentAnnotation | None = None,
) -> None:
    write_history_run(
        history_dir=tmp_path,
        run_id=run_id,
        run_ts=run_ts or datetime.datetime(2026, 5, 30, 12, 0, 0),
        git_sha="deadbeef",
        agent_ctx=_ctx(),
        stats=_stats(p50_by_comp),
        scaling_fits=[],
        annotation=annotation,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_store_returns_empty_frames(tmp_path: Path) -> None:
    runs = read_improvement_runs(tmp_path)
    snaps = read_component_snapshots(tmp_path)
    anns = read_agent_annotations(tmp_path)
    assert len(runs) == 0
    assert len(snaps) == 0
    assert len(anns) == 0


def test_first_write_creates_all_tables(tmp_path: Path) -> None:
    p50 = {c: 0.1 for c in _COMPONENTS}
    _write(tmp_path, "run-1", p50)

    assert (tmp_path / "improvement_runs.parquet").exists()
    assert (tmp_path / "component_snapshots.parquet").exists()


def test_first_write_one_run_row(tmp_path: Path) -> None:
    _write(tmp_path, "run-1", {c: 0.1 for c in _COMPONENTS})
    runs = read_improvement_runs(tmp_path)
    assert len(runs) == 1
    assert runs["run_id"][0] == "run-1"


def test_first_write_snapshot_count(tmp_path: Path) -> None:
    # One row per component + one e2e row.
    _write(tmp_path, "run-1", {c: 0.1 for c in _COMPONENTS})
    snaps = read_component_snapshots(tmp_path)
    assert len(snaps) == len(_COMPONENTS) + 1


def test_first_write_has_null_deltas(tmp_path: Path) -> None:
    _write(tmp_path, "run-1", {c: 0.1 for c in _COMPONENTS})
    snaps = read_component_snapshots(tmp_path)
    assert snaps["vs_prev_p50_pct"].is_null().all()
    assert snaps["vs_baseline_p50_pct"].is_null().all()


def test_p50_stored_as_ms(tmp_path: Path) -> None:
    _write(tmp_path, "run-1", {"etl": 0.1})  # 0.1 s → 100 ms
    snaps = read_component_snapshots(tmp_path)
    etl_row = snaps.filter(snaps["component"].cast(str) == "etl")
    assert abs(etl_row["p50_ms"][0] - 100.0) < 1e-9


def test_e2e_row_sums_component_p50s(tmp_path: Path) -> None:
    p50_by_comp = {"etl": 0.1, "signals": 0.08, "models": 0.3}
    expected_e2e_ms = sum(v * 1e3 for v in p50_by_comp.values())
    _write(tmp_path, "run-1", p50_by_comp)

    snaps = read_component_snapshots(tmp_path)
    e2e = snaps.filter(snaps["component"].cast(str) == "e2e")
    assert abs(e2e["p50_ms"][0] - expected_e2e_ms) < 1e-6


def test_e2e_p50_s_in_runs_table(tmp_path: Path) -> None:
    p50_by_comp = {"etl": 0.1, "signals": 0.08}
    _write(tmp_path, "run-1", p50_by_comp)
    runs = read_improvement_runs(tmp_path)
    expected = 0.1 + 0.08  # seconds
    assert abs(runs["e2e_p50_s"][0] - expected) < 1e-9


def test_second_write_prev_delta(tmp_path: Path) -> None:
    # Run 1: etl p50 = 100 ms.  Run 2: etl p50 = 80 ms → -20 %.
    _write(
        tmp_path, "run-1", {"etl": 0.1}, run_ts=datetime.datetime(2026, 5, 28, 10, 0, 0)
    )
    _write(
        tmp_path,
        "run-2",
        {"etl": 0.08},
        run_ts=datetime.datetime(2026, 5, 29, 10, 0, 0),
    )

    snaps = read_component_snapshots(tmp_path)
    run2_etl = snaps.filter(
        (snaps["run_id"] == "run-2") & (snaps["component"].cast(str) == "etl")
    )
    assert abs(run2_etl["vs_prev_p50_pct"][0] - (-20.0)) < 1e-6


def test_second_write_e2e_vs_prev_in_runs(tmp_path: Path) -> None:
    _write(
        tmp_path, "run-1", {"etl": 0.1}, run_ts=datetime.datetime(2026, 5, 28, 10, 0, 0)
    )
    _write(
        tmp_path,
        "run-2",
        {"etl": 0.08},
        run_ts=datetime.datetime(2026, 5, 29, 10, 0, 0),
    )

    runs = read_improvement_runs(tmp_path)
    run2 = runs.filter(runs["run_id"] == "run-2")
    assert abs(run2["e2e_vs_prev_pct"][0] - (-20.0)) < 1e-6


def test_third_write_baseline_is_oldest_run(tmp_path: Path) -> None:
    # Run 1: 100 ms, Run 2: 80 ms, Run 3: 70 ms
    # Run 3 vs_baseline should use Run 1 (oldest).
    _write(
        tmp_path, "run-1", {"etl": 0.1}, run_ts=datetime.datetime(2026, 5, 27, 10, 0, 0)
    )
    _write(
        tmp_path,
        "run-2",
        {"etl": 0.08},
        run_ts=datetime.datetime(2026, 5, 28, 10, 0, 0),
    )
    _write(
        tmp_path,
        "run-3",
        {"etl": 0.07},
        run_ts=datetime.datetime(2026, 5, 29, 10, 0, 0),
    )

    snaps = read_component_snapshots(tmp_path)
    run3_etl = snaps.filter(
        (snaps["run_id"] == "run-3") & (snaps["component"].cast(str) == "etl")
    )
    # vs_prev: (70 - 80) / 80 * 100 = -12.5
    assert abs(run3_etl["vs_prev_p50_pct"][0] - (-12.5)) < 1e-6
    # vs_baseline: (70 - 100) / 100 * 100 = -30.0
    assert abs(run3_etl["vs_baseline_p50_pct"][0] - (-30.0)) < 1e-6


def test_append_accumulates_run_rows(tmp_path: Path) -> None:
    _write(
        tmp_path, "run-1", {"etl": 0.1}, run_ts=datetime.datetime(2026, 5, 28, 10, 0, 0)
    )
    _write(
        tmp_path,
        "run-2",
        {"etl": 0.09},
        run_ts=datetime.datetime(2026, 5, 29, 10, 0, 0),
    )

    runs = read_improvement_runs(tmp_path)
    assert len(runs) == 2
    assert set(runs["run_id"].to_list()) == {"run-1", "run-2"}


def test_append_accumulates_snapshot_rows(tmp_path: Path) -> None:
    _write(
        tmp_path, "run-1", {"etl": 0.1}, run_ts=datetime.datetime(2026, 5, 28, 10, 0, 0)
    )
    _write(
        tmp_path,
        "run-2",
        {"etl": 0.09},
        run_ts=datetime.datetime(2026, 5, 29, 10, 0, 0),
    )

    snaps = read_component_snapshots(tmp_path)
    # Each write: 1 component row + 1 e2e = 2 rows; two writes = 4.
    assert len(snaps) == 4


def test_annotation_not_written_when_absent(tmp_path: Path) -> None:
    _write(tmp_path, "run-1", {"etl": 0.1})
    assert not (tmp_path / "agent_annotations.parquet").exists()


def test_annotation_written_when_provided(tmp_path: Path) -> None:
    ann = AgentAnnotation(
        hypothesis="vectorizing loops halves etl",
        outcome="etl dropped 22%, no regressions",
        lessons="adjust_prices was the bottleneck",
        next_target="signals: neutralize_sector copies per date",
        confidence=0.85,
        improvement_type="vectorization",
    )
    _write(tmp_path, "run-1", {"etl": 0.1}, annotation=ann)

    anns = read_agent_annotations(tmp_path)
    assert len(anns) == 1
    assert anns["run_id"][0] == "run-1"
    assert anns["hypothesis"][0] == "vectorizing loops halves etl"
    assert abs(anns["confidence"][0] - 0.85) < 1e-9


def test_agent_context_stored_in_runs(tmp_path: Path) -> None:
    write_history_run(
        history_dir=tmp_path,
        run_id="run-x",
        run_ts=datetime.datetime(2026, 5, 30, 12, 0, 0),
        git_sha="abc",
        agent_ctx=AgentContext(
            agent_id="etl-agent-7",
            goal="cut peak_rss",
            strategy="stream instead of materialise",
            target_component="etl",
            parent_run_id="run-w",
        ),
        stats=[_trial_result("etl", 0.1)],
        scaling_fits=[],
    )
    runs = read_improvement_runs(tmp_path)
    row = runs.row(0, named=True)
    assert row["agent_id"] == "etl-agent-7"
    assert row["goal"] == "cut peak_rss"
    assert row["target_component"] == "etl"
    assert row["parent_run_id"] == "run-w"


def test_scaling_fit_stored_in_snapshots(tmp_path: Path) -> None:
    from profiling.scaling import ScalingFit

    fits = [
        ScalingFit(
            run_id="run-1",
            stage="etl",
            metric="elapsed_s",
            scaling_dim="n_assets",
            log_log_slope=1.2,
            intercept=-2.5,
            r_squared=0.95,
            n_points=4,
        )
    ]
    write_history_run(
        history_dir=tmp_path,
        run_id="run-1",
        run_ts=datetime.datetime(2026, 5, 30, 12, 0, 0),
        git_sha="abc",
        agent_ctx=_ctx(),
        stats=[_trial_result("etl", 0.1)],
        scaling_fits=fits,
    )
    snaps = read_component_snapshots(tmp_path)
    etl = snaps.filter(snaps["component"].cast(str) == "etl")
    assert abs(etl["scaling_slope"][0] - 1.2) < 1e-9
    assert abs(etl["scaling_r2"][0] - 0.95) < 1e-9


def test_e2e_has_null_scaling(tmp_path: Path) -> None:
    _write(tmp_path, "run-1", {"etl": 0.1})
    snaps = read_component_snapshots(tmp_path)
    e2e = snaps.filter(snaps["component"].cast(str) == "e2e")
    assert e2e["scaling_slope"].is_null().all()
    assert e2e["scaling_r2"].is_null().all()
