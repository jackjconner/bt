"""The live round contract — data source #2 for the oversight deck.

The improvement-orchestrator emits and updates ``.oversight/round_state.json``
as a round progresses; the TUI reads it on a refresh interval. This module is
the typed schema for that file plus a small CLI the orchestrator drives:

    python -m oversight.state set-target  --round 7 --component portfolio ...
    python -m oversight.state set-phase   --phase merging
    python -m oversight.state set-lane    --component portfolio --pr 42 ...
    python -m oversight.state mark-gate   --component portfolio --gate evaluation --verdict pass

A missing/empty file is *not* an error — ``read_round_state`` returns ``None``
so the deck renders its "awaiting first round" state.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

# ---------------------------------------------------------------------------
# Vocabularies (kept narrow so the renderer can map them to the signal palette)
# ---------------------------------------------------------------------------

GateName = Literal["lint", "types", "correctness", "profiling", "evaluation"]
GateVerdict = Literal["pending", "running", "pass", "fail"]
LaneStatus = Literal["working", "gates", "accepted", "flagged", "merged"]
RoundPhase = Literal["proposing", "dispatched", "adjudicating", "merging", "revalidating", "done"]

GATE_ORDER: tuple[GateName, ...] = get_args(GateName)


# ---------------------------------------------------------------------------
# Nested frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Proposal:
    """The single thing the head agent is addressing this round."""

    target: str
    component: str
    metric: str
    grid_point: str = ""
    baseline_value: float | None = None
    golden_value: float | None = None
    eval_tolerance: str = ""
    dedup_clean: bool = True
    rationale: str = ""


@dataclass(frozen=True)
class GoldenSummary:
    """The round's golden — the ``PipelineSummary`` fields captured before any
    worker touches the tree. Eval gates diff against these."""

    ic_raw: float | None = None
    ic_neutralized: float | None = None
    wf_mean_ic: float | None = None
    wf_mean_r2: float | None = None
    gross_sharpe: float | None = None
    net_sharpe: float | None = None
    cost_drag: float | None = None
    tracking_error: float | None = None
    backtest_p50_s: float | None = None


@dataclass(frozen=True)
class ProfilingRow:
    """One before→after profiling line for a lane (e.g. 'portfolio @ n=200')."""

    stage: str
    before: str
    after: str
    delta: str


@dataclass(frozen=True)
class AgentLane:
    """One dispatched worker's lane: its branch, PR, gate verdicts, and deltas."""

    component: str
    slug: str = ""
    branch: str = ""
    worktree: str = ""
    pr_number: int | None = None
    title: str = ""
    status: LaneStatus = "working"
    lint: GateVerdict = "pending"
    types: GateVerdict = "pending"
    correctness: GateVerdict = "pending"
    profiling: GateVerdict = "pending"
    evaluation: GateVerdict = "pending"
    profiling_rows: tuple[ProfilingRow, ...] = ()
    eval_delta: str = ""
    headline_delta: str = ""
    note: str = ""

    def gate(self, name: GateName) -> GateVerdict:
        return getattr(self, name)

    def gates(self) -> dict[GateName, GateVerdict]:
        return {name: self.gate(name) for name in GATE_ORDER}


@dataclass(frozen=True)
class RoundState:
    """The whole live round, as the orchestrator sees it."""

    round_number: int
    phase: RoundPhase = "proposing"
    dispatched: int = 0
    proposal: Proposal | None = None
    golden: GoldenSummary | None = None
    lanes: tuple[AgentLane, ...] = ()

    def lane(self, component: str) -> AgentLane | None:
        for lane in self.lanes:
            if lane.component == component:
                return lane
        return None


# ---------------------------------------------------------------------------
# JSON (de)serialization
# ---------------------------------------------------------------------------


def _to_jsonable(state: RoundState) -> dict[str, object]:
    return dataclasses.asdict(state)


def round_state_to_json(state: RoundState) -> str:
    return json.dumps(_to_jsonable(state), indent=2, sort_keys=False)


def _str(d: dict[str, object], key: str, default: str = "") -> str:
    v = d.get(key)
    return v if isinstance(v, str) else default


def _int(d: dict[str, object], key: str, default: int = 0) -> int:
    v = d.get(key)
    return v if isinstance(v, int) and not isinstance(v, bool) else default


def _opt_int(d: dict[str, object], key: str) -> int | None:
    v = d.get(key)
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _opt_float(d: dict[str, object], key: str) -> float | None:
    v = d.get(key)
    if isinstance(v, bool):
        return None
    return float(v) if isinstance(v, (int, float)) else None


def _bool(d: dict[str, object], key: str, default: bool) -> bool:
    v = d.get(key)
    return v if isinstance(v, bool) else default


def _gate(d: dict[str, object], key: str) -> GateVerdict:
    v = d.get(key)
    if v == "running":
        return "running"
    if v == "pass":
        return "pass"
    if v == "fail":
        return "fail"
    return "pending"


def _proposal_from(raw: object) -> Proposal | None:
    d = _as_str_dict(raw)
    if d is None:
        return None
    return Proposal(
        target=_str(d, "target"),
        component=_str(d, "component"),
        metric=_str(d, "metric"),
        grid_point=_str(d, "grid_point"),
        baseline_value=_opt_float(d, "baseline_value"),
        golden_value=_opt_float(d, "golden_value"),
        eval_tolerance=_str(d, "eval_tolerance"),
        dedup_clean=_bool(d, "dedup_clean", True),
        rationale=_str(d, "rationale"),
    )


def _golden_from(raw: object) -> GoldenSummary | None:
    d = _as_str_dict(raw)
    if d is None:
        return None
    return GoldenSummary(
        ic_raw=_opt_float(d, "ic_raw"),
        ic_neutralized=_opt_float(d, "ic_neutralized"),
        wf_mean_ic=_opt_float(d, "wf_mean_ic"),
        wf_mean_r2=_opt_float(d, "wf_mean_r2"),
        gross_sharpe=_opt_float(d, "gross_sharpe"),
        net_sharpe=_opt_float(d, "net_sharpe"),
        cost_drag=_opt_float(d, "cost_drag"),
        tracking_error=_opt_float(d, "tracking_error"),
        backtest_p50_s=_opt_float(d, "backtest_p50_s"),
    )


def _as_str_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return {str(k): v for k, v in value.items()}


def _dict_list(d: dict[str, object], key: str) -> list[dict[str, object]]:
    v = d.get(key)
    if not isinstance(v, list):
        return []
    out: list[dict[str, object]] = []
    for item in v:
        coerced = _as_str_dict(item)
        if coerced is not None:
            out.append(coerced)
    return out


def _lane_status(d: dict[str, object]) -> LaneStatus:
    v = d.get("status")
    if v == "gates":
        return "gates"
    if v == "accepted":
        return "accepted"
    if v == "flagged":
        return "flagged"
    if v == "merged":
        return "merged"
    return "working"


def _lane_from(d: dict[str, object]) -> AgentLane:
    rows = tuple(
        ProfilingRow(
            stage=_str(r, "stage"),
            before=_str(r, "before"),
            after=_str(r, "after"),
            delta=_str(r, "delta"),
        )
        for r in _dict_list(d, "profiling_rows")
    )
    return AgentLane(
        component=_str(d, "component"),
        slug=_str(d, "slug"),
        branch=_str(d, "branch"),
        worktree=_str(d, "worktree"),
        pr_number=_opt_int(d, "pr_number"),
        title=_str(d, "title"),
        status=_lane_status(d),
        lint=_gate(d, "lint"),
        types=_gate(d, "types"),
        correctness=_gate(d, "correctness"),
        profiling=_gate(d, "profiling"),
        evaluation=_gate(d, "evaluation"),
        profiling_rows=rows,
        eval_delta=_str(d, "eval_delta"),
        headline_delta=_str(d, "headline_delta"),
        note=_str(d, "note"),
    )


def _round_phase(d: dict[str, object]) -> RoundPhase:
    v = d.get("phase")
    if v == "dispatched":
        return "dispatched"
    if v == "adjudicating":
        return "adjudicating"
    if v == "merging":
        return "merging"
    if v == "revalidating":
        return "revalidating"
    if v == "done":
        return "done"
    return "proposing"


def round_state_from_dict(d: dict[str, object]) -> RoundState:
    lanes = tuple(_lane_from(item) for item in _dict_list(d, "lanes"))
    return RoundState(
        round_number=_int(d, "round_number"),
        phase=_round_phase(d),
        dispatched=_int(d, "dispatched"),
        proposal=_proposal_from(d.get("proposal")),
        golden=_golden_from(d.get("golden")),
        lanes=lanes,
    )


def read_round_state(path: Path) -> RoundState | None:
    """Read ``round_state.json``; ``None`` if missing or empty (not an error)."""
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    return round_state_from_dict(json.loads(text))


def write_round_state(path: Path, state: RoundState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(round_state_to_json(state))


# ---------------------------------------------------------------------------
# CLI — the orchestrator's write surface
# ---------------------------------------------------------------------------

_DEFAULT_PATH = Path(".oversight/round_state.json")


def _or(value: str | None, fallback: str) -> str:
    return value if value is not None else fallback


def _load_or_init(path: Path, round_number: int | None) -> RoundState:
    state = read_round_state(path)
    if state is not None:
        return state
    return RoundState(round_number=round_number if round_number is not None else 0)


def _replace_lane(state: RoundState, lane: AgentLane) -> RoundState:
    existing = [lane_item for lane_item in state.lanes if lane_item.component != lane.component]
    merged = (*existing, lane)
    ordered = tuple(sorted(merged, key=lambda lane_item: lane_item.component))
    return dataclasses.replace(state, lanes=ordered)


def _cmd_set_target(args: argparse.Namespace) -> None:
    path = Path(args.path)
    state = _load_or_init(path, args.round)
    proposal = Proposal(
        target=args.target,
        component=args.component,
        metric=args.metric,
        grid_point=args.grid_point,
        baseline_value=args.baseline,
        golden_value=args.golden,
        eval_tolerance=args.tolerance,
        dedup_clean=not args.dedup_dirty,
        rationale=args.rationale,
    )
    state = dataclasses.replace(state, round_number=args.round, proposal=proposal)
    write_round_state(path, state)


def _cmd_set_phase(args: argparse.Namespace) -> None:
    path = Path(args.path)
    state = _load_or_init(path, None)
    state = dataclasses.replace(state, phase=args.phase)
    if args.dispatched is not None:
        state = dataclasses.replace(state, dispatched=args.dispatched)
    write_round_state(path, state)


def _cmd_set_lane(args: argparse.Namespace) -> None:
    path = Path(args.path)
    state = _load_or_init(path, None)
    prior = state.lane(args.component)
    base = prior if prior is not None else AgentLane(component=args.component)
    status: LaneStatus = args.status if args.status is not None else base.status
    pr_number: int | None = args.pr if args.pr is not None else base.pr_number
    lane = dataclasses.replace(
        base,
        slug=_or(args.slug, base.slug),
        branch=_or(args.branch, base.branch),
        worktree=_or(args.worktree, base.worktree),
        pr_number=pr_number,
        title=_or(args.title, base.title),
        status=status,
        headline_delta=_or(args.headline_delta, base.headline_delta),
        eval_delta=_or(args.eval_delta, base.eval_delta),
        note=_or(args.note, base.note),
    )
    state = _replace_lane(state, lane)
    write_round_state(path, state)


def _cmd_mark_gate(args: argparse.Namespace) -> None:
    path = Path(args.path)
    state = _load_or_init(path, None)
    prior = state.lane(args.component)
    base = prior if prior is not None else AgentLane(component=args.component)
    gate: GateName = args.gate
    verdict: GateVerdict = args.verdict
    gates = base.gates()
    gates[gate] = verdict
    lane = dataclasses.replace(base, **gates)
    state = _replace_lane(state, lane)
    write_round_state(path, state)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m oversight.state")
    parser.add_argument("--path", default=str(_DEFAULT_PATH), help="round_state.json path")
    sub = parser.add_subparsers(dest="command", required=True)

    p_target = sub.add_parser("set-target", help="set the round proposal")
    p_target.add_argument("--round", type=int, required=True)
    p_target.add_argument("--component", required=True)
    p_target.add_argument("--target", required=True)
    p_target.add_argument("--metric", required=True)
    p_target.add_argument("--grid-point", default="")
    p_target.add_argument("--baseline", type=float, default=None)
    p_target.add_argument("--golden", type=float, default=None)
    p_target.add_argument("--tolerance", default="")
    p_target.add_argument("--rationale", default="")
    p_target.add_argument("--dedup-dirty", action="store_true")
    p_target.set_defaults(func=_cmd_set_target)

    p_phase = sub.add_parser("set-phase", help="advance the round phase")
    p_phase.add_argument("--phase", required=True, choices=get_args(RoundPhase))
    p_phase.add_argument("--dispatched", type=int, default=None)
    p_phase.set_defaults(func=_cmd_set_phase)

    p_lane = sub.add_parser("set-lane", help="create or update a worker lane")
    p_lane.add_argument("--component", required=True)
    p_lane.add_argument("--slug", default=None)
    p_lane.add_argument("--branch", default=None)
    p_lane.add_argument("--worktree", default=None)
    p_lane.add_argument("--pr", type=int, default=None)
    p_lane.add_argument("--title", default=None)
    p_lane.add_argument("--status", default=None, choices=get_args(LaneStatus))
    p_lane.add_argument("--headline-delta", default=None)
    p_lane.add_argument("--eval-delta", default=None)
    p_lane.add_argument("--note", default=None)
    p_lane.set_defaults(func=_cmd_set_lane)

    p_gate = sub.add_parser("mark-gate", help="set one gate verdict on a lane")
    p_gate.add_argument("--component", required=True)
    p_gate.add_argument("--gate", required=True, choices=get_args(GateName))
    p_gate.add_argument("--verdict", required=True, choices=get_args(GateVerdict))
    p_gate.set_defaults(func=_cmd_mark_gate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()


__all__ = [
    "GATE_ORDER",
    "AgentLane",
    "GateName",
    "GateVerdict",
    "GoldenSummary",
    "LaneStatus",
    "ProfilingRow",
    "Proposal",
    "RoundPhase",
    "RoundState",
    "read_round_state",
    "round_state_from_dict",
    "round_state_to_json",
    "write_round_state",
]
