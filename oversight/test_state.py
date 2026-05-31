"""Tests for state.py — the live round contract + its CLI write surface."""

from __future__ import annotations

from pathlib import Path

from oversight.state import (
    GATE_ORDER,
    AgentLane,
    GoldenSummary,
    ProfilingRow,
    Proposal,
    RoundState,
    main,
    read_round_state,
    round_state_from_dict,
    round_state_to_json,
    write_round_state,
)


def _full_state() -> RoundState:
    return RoundState(
        round_number=7,
        phase="adjudicating",
        dispatched=2,
        proposal=Proposal(
            target="Break the optimizer's super-linear scaling on n_assets.",
            component="portfolio",
            metric="elapsed_s p50",
            grid_point="n_assets=200",
            baseline_value=1.84,
            golden_value=1.137,
            eval_tolerance="1e-6 rel",
            dedup_clean=True,
            rationale="cache the Cholesky across rebalances",
        ),
        golden=GoldenSummary(ic_raw=0.0517, net_sharpe=1.137, cost_drag=0.18),
        lanes=(
            AgentLane(
                component="portfolio",
                slug="chol-cache",
                branch="improve/portfolio-chol-cache",
                pr_number=42,
                title="Cache factor-covariance Cholesky",
                status="accepted",
                lint="pass",
                types="pass",
                correctness="pass",
                profiling="pass",
                evaluation="pass",
                profiling_rows=(ProfilingRow("portfolio @ n=200", "1.84s", "0.71s", "-61%"),),
                eval_delta="Δ 4e-9 · held",
                headline_delta="-61%",
            ),
        ),
    )


def test_roundtrip_preserves_state() -> None:
    state = _full_state()
    restored = round_state_from_dict(__import__("json").loads(round_state_to_json(state)))
    assert restored == state


def test_write_then_read(tmp_path: Path) -> None:
    path = tmp_path / "round_state.json"
    state = _full_state()
    write_round_state(path, state)
    assert read_round_state(path) == state


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert read_round_state(tmp_path / "nope.json") is None


def test_empty_file_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "round_state.json"
    path.write_text("   \n")
    assert read_round_state(path) is None


def test_lane_lookup_and_gates() -> None:
    state = _full_state()
    lane = state.lane("portfolio")
    assert lane is not None
    assert lane.gates() == dict.fromkeys(GATE_ORDER, "pass")
    assert state.lane("signals") is None


def test_from_dict_tolerates_extra_keys() -> None:
    state = round_state_from_dict({"round_number": 3, "phase": "proposing", "stray": "ignored"})
    assert state.round_number == 3
    assert state.lanes == ()


def test_cli_set_target_then_phase_then_gate(tmp_path: Path) -> None:
    path = str(tmp_path / "round_state.json")
    main(
        [
            "--path",
            path,
            "set-target",
            "--round",
            "7",
            "--component",
            "portfolio",
            "--target",
            "speed up the solve",
            "--metric",
            "elapsed_s p50",
            "--baseline",
            "1.84",
        ]
    )
    main(["--path", path, "set-phase", "--phase", "adjudicating", "--dispatched", "2"])
    main(
        [
            "--path",
            path,
            "set-lane",
            "--component",
            "portfolio",
            "--pr",
            "42",
            "--branch",
            "improve/portfolio-chol-cache",
            "--status",
            "accepted",
        ]
    )
    main(
        [
            "--path",
            path,
            "mark-gate",
            "--component",
            "portfolio",
            "--gate",
            "evaluation",
            "--verdict",
            "pass",
        ]
    )

    state = read_round_state(Path(path))
    assert state is not None
    assert state.round_number == 7
    assert state.phase == "adjudicating"
    assert state.dispatched == 2
    assert state.proposal is not None
    assert state.proposal.baseline_value == 1.84
    lane = state.lane("portfolio")
    assert lane is not None
    assert lane.pr_number == 42
    assert lane.status == "accepted"
    assert lane.evaluation == "pass"


def test_cli_mark_gate_creates_lane_when_absent(tmp_path: Path) -> None:
    path = str(tmp_path / "round_state.json")
    main(
        [
            "--path",
            path,
            "mark-gate",
            "--component",
            "signals",
            "--gate",
            "lint",
            "--verdict",
            "running",
        ]
    )
    state = read_round_state(Path(path))
    assert state is not None
    lane = state.lane("signals")
    assert lane is not None
    assert lane.lint == "running"
