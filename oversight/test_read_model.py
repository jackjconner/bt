"""Tests for read_model.py — the DeckState aggregator (empty + populated)."""

from __future__ import annotations

import datetime
from pathlib import Path

from harness.history import AgentContext, write_history_run
from oversight.read_model import load
from oversight.state import (
    AgentLane,
    GoldenSummary,
    ProfilingRow,
    Proposal,
    RoundState,
    write_round_state,
)
from profiling.trials import TrialResult


def _trial(stage: str, p50_s: float) -> TrialResult:
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


def _seed_history(history_dir: Path) -> None:
    write_history_run(
        history_dir=history_dir,
        run_id="run_0001",
        run_ts=datetime.datetime(2026, 5, 20, 12, 0, 0),
        git_sha="abc1234",
        agent_ctx=AgentContext(agent_id="a1", goal="g", strategy="s", target_component="portfolio"),
        stats=[_trial("portfolio", 1.84), _trial("backtest", 0.5)],
        scaling_fits=[],
    )


def _seed_round_state(root: Path) -> None:
    state = RoundState(
        round_number=7,
        phase="adjudicating",
        dispatched=2,
        proposal=Proposal(
            target="Break the optimizer's super-linear scaling.",
            component="portfolio",
            metric="elapsed_s p50",
            baseline_value=1.84,
            eval_tolerance="1e-6 rel",
        ),
        golden=GoldenSummary(ic_raw=0.0517, net_sharpe=1.137),
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
            ),
        ),
    )
    write_round_state(root / ".oversight" / "round_state.json", state)


_IMPROVEMENTS = """# Improvements log

```
## <date> — <component>: <target>  [accepted | rejected]
```

---

## 2026-05-10 — etl: PIT as-of join  [accepted]
metric:   load p50 — 0.6s → 0.45s (-24%)
PR:       #34
note:     golden held

## 2026-05-14 — analysis: rolling metrics  [rejected]
metric:   n/a
note:     eval moved Sortino w/o justification
"""


def test_empty_state_is_awaiting(tmp_path: Path) -> None:
    deck = load(tmp_path, tmp_path / "history")
    assert deck.awaiting is True
    assert deck.lanes == ()
    assert deck.proposal is None
    # the DAG still renders all seven components, all idle
    assert len(deck.nodes) == 7
    assert all(node.state == "idle" for node in deck.nodes)


def test_populated_round_state_drives_the_round(tmp_path: Path) -> None:
    _seed_round_state(tmp_path)
    deck = load(tmp_path, tmp_path / "history")
    assert deck.awaiting is False
    assert deck.round_number == 7
    assert deck.phase == "adjudicating"
    assert deck.proposal is not None
    assert deck.proposal.component == "portfolio"
    assert deck.golden_net_sharpe == 1.137
    assert deck.eval_tolerance == "1e-6 rel"
    lane = next(lane for lane in deck.lanes if lane.component == "portfolio")
    assert lane.pr_number == 42
    # portfolio node is active in the DAG
    portfolio_node = next(n for n in deck.nodes if n.name == "portfolio")
    assert portfolio_node.state == "active"


def test_ledgers_feed_the_ratchet_and_cumulative(tmp_path: Path) -> None:
    (tmp_path / "IMPROVEMENTS.md").write_text(_IMPROVEMENTS)
    deck = load(tmp_path, tmp_path / "history")
    assert deck.cumulative_landed == 1  # one accepted, one rejected
    assert len(deck.improvements) == 2
    verdicts = [tooth.verdict for tooth in deck.ratchet]
    assert "landed" in verdicts
    assert "rejected" in verdicts


def test_history_feeds_trends(tmp_path: Path) -> None:
    history_dir = tmp_path / "history"
    _seed_history(history_dir)
    deck = load(tmp_path, history_dir)
    comps = {row.component for row in deck.trends}
    assert "portfolio" in comps
    assert "e2e" in comps
    portfolio = next(row for row in deck.trends if row.component == "portfolio")
    assert portfolio.p50_ms is not None
    assert portfolio.p50_ms > 0.0


def test_no_history_yields_empty_trends(tmp_path: Path) -> None:
    deck = load(tmp_path, tmp_path / "history")
    assert deck.trends == ()
