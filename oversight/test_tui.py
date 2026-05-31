"""Tests for tui.py — the renderer produces a frame without crashing.

These exercise ``render`` over both empty and populated decks by capturing the
rich output to a string console (no live terminal, no network).
"""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from oversight.read_model import load
from oversight.state import (
    AgentLane,
    GoldenSummary,
    ProfilingRow,
    Proposal,
    RoundState,
    write_round_state,
)
from oversight.tui import render


def _capture(renderable: object) -> str:
    console = Console(width=160, record=True, file=io.StringIO())
    console.print(renderable)
    return console.export_text()


def test_render_empty_state_mentions_awaiting(tmp_path: Path) -> None:
    deck = load(tmp_path, tmp_path / "history")
    out = _capture(render(deck))
    assert "Awaiting" in out


def _populated(root: Path) -> None:
    state = RoundState(
        round_number=7,
        phase="adjudicating",
        dispatched=2,
        proposal=Proposal(
            target="Break the optimizer's super-linear scaling.",
            component="portfolio",
            metric="elapsed_s p50",
            grid_point="n_assets=200",
            baseline_value=1.84,
            eval_tolerance="1e-6 rel",
            rationale="cache the Cholesky",
        ),
        golden=GoldenSummary(net_sharpe=1.137),
        lanes=(
            AgentLane(
                component="portfolio",
                branch="improve/portfolio-chol-cache",
                pr_number=42,
                title="Cache Cholesky",
                status="accepted",
                lint="pass",
                types="pass",
                correctness="pass",
                profiling="pass",
                evaluation="pass",
                profiling_rows=(ProfilingRow("portfolio @ n=200", "1.84s", "0.71s", "-61%"),),
                headline_delta="-61%",
            ),
            AgentLane(
                component="signals",
                branch="improve/signals-ic-vectorize",
                pr_number=43,
                title="Vectorize IC",
                status="flagged",
                lint="pass",
                types="pass",
                correctness="pass",
                profiling="pass",
                evaluation="fail",
            ),
        ),
    )
    write_round_state(root / ".oversight" / "round_state.json", state)


def test_render_populated_includes_sections(tmp_path: Path) -> None:
    _populated(tmp_path)
    deck = load(tmp_path, tmp_path / "history")
    out = _capture(render(deck))
    assert "the round" in out
    assert "the gauntlet" in out
    assert "the fan-out" in out
    assert "serial merge" in out
    assert "the ledgers" in out


def test_render_populated_shows_lane_detail(tmp_path: Path) -> None:
    _populated(tmp_path)
    deck = load(tmp_path, tmp_path / "history")
    out = _capture(render(deck))
    assert "portfolio" in out
    assert "PR #42" in out
    assert "-61%" in out
    # the flagged signals lane and its failed gate are visible
    assert "signals" in out


def test_render_returns_renderable_for_both_states(tmp_path: Path) -> None:
    empty = load(tmp_path, tmp_path / "history")
    assert render(empty) is not None
    _populated(tmp_path)
    populated = load(tmp_path, tmp_path / "history")
    assert render(populated) is not None
