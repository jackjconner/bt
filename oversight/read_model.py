"""DeckState aggregator — one snapshot the renderer binds to.

``load(root, history_dir)`` merges the four sources into a single immutable
``DeckState``:

  1. live ``RoundState`` (``.oversight/round_state.json``) — *preferred* for the
     rich round detail (proposal, golden, per-gate lane verdicts);
  2. git + ``gh`` derived lanes (branches, worktrees, open PRs + parsed bodies);
  3. the two parsed ledgers (``IMPROVEMENTS.md`` / ``API_REQUESTS.md``);
  4. the parquet history (``harness.history`` readers) for trend/ratchet.

Empty-safe: with nothing present it returns a DeckState whose ``awaiting`` flag
is set, so the TUI renders an "awaiting first round" deck instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.history import read_component_snapshots, read_improvement_runs
from oversight import gitgh, ledgers
from oversight.state import (
    GATE_ORDER,
    AgentLane,
    GateVerdict,
    GoldenSummary,
    LaneStatus,
    ProfilingRow,
    Proposal,
    RoundState,
    read_round_state,
)

# The seven components in dependency order, plus the synthetic e2e roll-up.
COMPONENTS: tuple[str, ...] = (
    "etl",
    "signals",
    "models",
    "analysis",
    "portfolio",
    "backtest",
    "profiling",
)


@dataclass(frozen=True)
class RatchetTooth:
    label: str
    verdict: str  # landed | rejected | active | pending


@dataclass(frozen=True)
class ComponentNode:
    name: str
    state: str  # idle | active | queued
    meta: str


@dataclass(frozen=True)
class TrendRow:
    component: str
    p50_ms: float | None
    vs_baseline_pct: float | None


@dataclass(frozen=True)
class DeckState:
    awaiting: bool
    round_number: int
    phase: str
    cumulative_landed: int
    golden_net_sharpe: float | None
    ratchet: tuple[RatchetTooth, ...]
    nodes: tuple[ComponentNode, ...]
    proposal: Proposal | None
    golden: GoldenSummary | None
    dispatched: int
    open_prs: int
    active_lane: AgentLane | None
    lanes: tuple[AgentLane, ...]
    improvements: tuple[ledgers.ImprovementEntry, ...]
    requests: tuple[ledgers.RequestEntry, ...]
    trends: tuple[TrendRow, ...]
    eval_tolerance: str


# ---------------------------------------------------------------------------
# Lane derivation — RoundState preferred, git/gh fills the gaps
# ---------------------------------------------------------------------------


def _lane_from_pr(pr: gitgh.PullRequest) -> AgentLane:
    def verdict(gate: str) -> GateVerdict:
        if gate not in pr.gates:
            return "pending"
        return "pass" if pr.gates[gate] else "fail"

    rows = tuple(ProfilingRow(r.stage, r.before, r.after, r.delta) for r in pr.prof_rows)
    all_known = [pr.gates[g] for g in GATE_ORDER if g in pr.gates]
    status: LaneStatus
    if all_known and all(all_known):
        status = "accepted"
    elif all_known and not all(all_known):
        status = "flagged"
    else:
        status = "gates"
    return AgentLane(
        component=pr.component or "unknown",
        branch=pr.branch,
        pr_number=pr.number,
        title=pr.title,
        status=status,
        lint=verdict("lint"),
        types=verdict("types"),
        correctness=verdict("correctness"),
        profiling=verdict("profiling"),
        evaluation=verdict("evaluation"),
        profiling_rows=rows,
    )


def _merge_lanes(state: RoundState | None, prs: list[gitgh.PullRequest]) -> tuple[AgentLane, ...]:
    """RoundState lanes win; PRs without a live lane are appended."""
    if state is not None and state.lanes:
        live_components = {lane.component for lane in state.lanes}
        derived = tuple(
            _lane_from_pr(pr) for pr in prs if (pr.component or "unknown") not in live_components
        )
        return (*state.lanes, *derived)
    return tuple(_lane_from_pr(pr) for pr in prs)


def _active_lane(lanes: tuple[AgentLane, ...]) -> AgentLane | None:
    """The lane the gauntlet focuses on: a working/gates lane, else the first."""
    for lane in lanes:
        if lane.status in {"working", "gates"}:
            return lane
    return lanes[0] if lanes else None


# ---------------------------------------------------------------------------
# Ratchet + DAG + trends
# ---------------------------------------------------------------------------


def _ratchet(
    improvements: tuple[ledgers.ImprovementEntry, ...],
    state: RoundState | None,
) -> tuple[RatchetTooth, ...]:
    teeth: list[RatchetTooth] = []
    round_no = 0
    for entry in improvements:
        round_no += 1
        verdict = "landed" if entry.verdict == "accepted" else "rejected"
        teeth.append(RatchetTooth(label=f"{round_no:03d}", verdict=verdict))
    if state is not None and state.phase != "done":
        round_no = max(round_no + 1, state.round_number)
        teeth.append(RatchetTooth(label=f"{round_no:03d}", verdict="active"))
    # pad a couple of pending teeth so the strip reads as a forward ratchet
    for i in range(1, 3):
        teeth.append(RatchetTooth(label=f"{round_no + i:03d}", verdict="pending"))
    return tuple(teeth)


def _nodes(
    state: RoundState | None,
    lanes: tuple[AgentLane, ...],
    requests: tuple[ledgers.RequestEntry, ...],
) -> tuple[ComponentNode, ...]:
    lane_by_comp = {lane.component: lane for lane in lanes}
    open_request_comps = {r.producer for r in requests if r.status == "open"}
    active_comp = state.proposal.component if state is not None and state.proposal else None

    nodes: list[ComponentNode] = []
    for comp in COMPONENTS:
        lane = lane_by_comp.get(comp)
        if lane is not None:
            pr = f"PR #{lane.pr_number}" if lane.pr_number is not None else "working"
            if lane.status == "flagged":
                nodes.append(ComponentNode(comp, "queued", f"{pr} · flagged"))
            elif comp == active_comp or lane.status in {"working", "gates"}:
                nodes.append(ComponentNode(comp, "active", f"{pr} · in gauntlet"))
            else:
                nodes.append(ComponentNode(comp, "queued", f"{pr} · {lane.status}"))
        else:
            meta = "stable · req open" if comp in open_request_comps else "stable"
            nodes.append(ComponentNode(comp, "idle", meta))
    return tuple(nodes)


def _trends(history_dir: Path) -> tuple[TrendRow, ...]:
    runs = read_improvement_runs(history_dir)
    snaps = read_component_snapshots(history_dir)
    if runs.is_empty() or snaps.is_empty():
        return ()
    latest_run = runs.sort("run_ts")["run_id"][-1]
    latest = snaps.filter(snaps["run_id"] == latest_run)
    rows: list[TrendRow] = []
    for row in latest.iter_rows(named=True):
        rows.append(
            TrendRow(
                component=str(row["component"]),
                p50_ms=_opt_f(row.get("p50_ms")),
                vs_baseline_pct=_opt_f(row.get("vs_baseline_p50_pct")),
            )
        )
    return tuple(rows)


def _opt_f(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _golden_net_sharpe(state: RoundState | None) -> float | None:
    if state is not None and state.golden is not None:
        return state.golden.net_sharpe
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load(root: Path, history_dir: Path) -> DeckState:
    state = read_round_state(root / ".oversight" / "round_state.json")

    improvements = tuple(ledgers.read_improvements(root / "IMPROVEMENTS.md"))
    requests = tuple(ledgers.read_requests(root / "API_REQUESTS.md"))

    prs = gitgh.open_pull_requests(root)
    lanes = _merge_lanes(state, prs)

    cumulative_landed = sum(1 for e in improvements if e.verdict == "accepted")
    trends = _trends(history_dir)

    awaiting = state is None and not lanes and not improvements and not trends

    proposal = state.proposal if state is not None else None
    golden = state.golden if state is not None else None
    round_number = state.round_number if state is not None else (len(improvements) + 1)
    phase = state.phase if state is not None else "proposing"
    dispatched = state.dispatched if state is not None else len(lanes)
    eval_tolerance = proposal.eval_tolerance if proposal is not None else ""

    return DeckState(
        awaiting=awaiting,
        round_number=round_number,
        phase=phase,
        cumulative_landed=cumulative_landed,
        golden_net_sharpe=_golden_net_sharpe(state),
        ratchet=_ratchet(improvements, state),
        nodes=_nodes(state, lanes, requests),
        proposal=proposal,
        golden=golden,
        dispatched=dispatched,
        open_prs=len(prs),
        active_lane=_active_lane(lanes),
        lanes=lanes,
        improvements=improvements,
        requests=requests,
        trends=trends,
        eval_tolerance=eval_tolerance,
    )


__all__ = [
    "COMPONENTS",
    "ComponentNode",
    "DeckState",
    "RatchetTooth",
    "TrendRow",
    "load",
]
