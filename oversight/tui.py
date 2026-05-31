"""rich.Live dashboard — the overseer deck as a TUI.

Echoes the sections and signal palette of ``design/oversight-ui.html``:

  masthead + ratchet · left rail (component DAG + round vitals) · main column
  (the round / proposal · the gauntlet / 5 gates · the fan-out / PR lanes ·
  serial merge + re-validation · the two ledgers).

Palette: green = pass/improve, amber = running/awaiting, oxide(red) = fail,
cyan = metric/data, violet = docs/meta.

Re-reads ``DeckState`` on an interval (default 2s) so the deck tracks the live
round. ``q`` quits. ``--once`` renders a single frame and exits (smoke test).
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import tty
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from oversight.read_model import ComponentNode, DeckState, RatchetTooth, load
from oversight.state import GATE_ORDER, AgentLane, GateName, GateVerdict

# Signal palette (mirrors the mockup's CSS variables).
GREEN = "#76e0a0"
AMBER = "#f0b44b"
OXIDE = "#e0654f"
CYAN = "#6fc6d6"
VIOLET = "#b79be0"
BONE = "#e9e1cf"
DIM = "#b3a98f"
FAINT = "#7c735d"

_GATE_STYLE: dict[GateVerdict, str] = {
    "pass": GREEN,
    "running": AMBER,
    "fail": OXIDE,
    "pending": FAINT,
}
_GATE_GLYPH: dict[GateVerdict, str] = {
    "pass": "✓",
    "running": "◐",
    "fail": "✗",
    "pending": "·",
}
_STATUS_STYLE: dict[str, str] = {
    "accepted": GREEN,
    "merged": GREEN,
    "flagged": OXIDE,
    "working": AMBER,
    "gates": AMBER,
}
_NODE_STYLE: dict[str, str] = {"active": AMBER, "queued": CYAN, "idle": FAINT}


def _eyebrow(text: str) -> Text:
    return Text(text.upper(), style=f"{FAINT}")


# ---------------------------------------------------------------------------
# Masthead + ratchet
# ---------------------------------------------------------------------------

_PHASE_LABEL: dict[str, str] = {
    "proposing": "PROPOSING TARGET",
    "dispatched": "WORKERS DISPATCHED",
    "adjudicating": "AWAITING ADJUDICATION",
    "merging": "SERIAL MERGE",
    "revalidating": "RE-VALIDATING MAIN",
    "done": "ROUND COMPLETE",
}


def _ratchet_text(teeth: tuple[RatchetTooth, ...]) -> Text:
    out = Text()
    glyphs = {
        "landed": ("▰", GREEN),
        "rejected": ("▱", OXIDE),
        "active": ("▰", AMBER),
        "pending": ("▱", FAINT),
    }
    for tooth in teeth:
        glyph, color = glyphs.get(tooth.verdict, ("▱", FAINT))
        out.append(f"{glyph} ", style=color)
    out.append("→ one tooth / round", style=GREEN)
    return out


def _masthead(deck: DeckState) -> Panel:
    title = Text()
    title.append("bt ", style=f"bold {BONE}")
    title.append("improvement loop", style=f"italic {AMBER}")

    phase = _PHASE_LABEL.get(deck.phase, deck.phase.upper())
    phase_color = AMBER if deck.phase in {"adjudicating", "merging", "revalidating"} else DIM

    stats = Text()
    stats.append("  cumulative landed ", style=FAINT)
    stats.append(f"+{deck.cumulative_landed}", style=f"bold {BONE}")
    sharpe = deck.golden_net_sharpe
    stats.append("    net sharpe · golden ", style=FAINT)
    stats.append(f"{sharpe:.3f}" if sharpe is not None else "—", style=GREEN)

    top = Table.grid(expand=True)
    top.add_column(ratio=1)
    top.add_column(justify="right")
    top.add_row(
        Group(
            title,
            Text("seven components · improved behind their APIs · every change gated", style=FAINT),
        ),
        Group(Text(f"● {phase}", style=phase_color), stats),
    )

    ratchet_line = Table.grid(expand=True)
    ratchet_line.add_column(justify="left", width=10)
    ratchet_line.add_column(ratio=1)
    ratchet_line.add_row(Text("RATCHET", style=FAINT), _ratchet_text(deck.ratchet))

    return Panel(
        Group(top, Text(""), ratchet_line),
        border_style=FAINT,
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Left rail — DAG + vitals
# ---------------------------------------------------------------------------


def _dag(nodes: tuple[ComponentNode, ...]) -> Panel:
    flow = Text("data → signals → models → portfolio → backtest → analysis\n", style=FAINT)
    flow.append("  ↑ etl ingests        ↑ profiling measures it all", style=FAINT)

    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=2)
    table.add_column(ratio=1)
    table.add_column(justify="right")
    for node in nodes:
        color = _NODE_STYLE.get(node.state, FAINT)
        glyph = "●" if node.state == "active" else ("◆" if node.state == "queued" else "■")
        name_style = color if node.state != "idle" else DIM
        table.add_row(
            Text(glyph, style=color),
            Text(node.name, style=name_style),
            Text(node.meta, style=FAINT),
        )
    return Panel(
        Group(flow, Text(""), table),
        title=Text("i · the machine", style=DIM),
        border_style=FAINT,
        title_align="left",
    )


def _vitals(deck: DeckState) -> Panel:
    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(justify="right")

    def row(key: str, value: Text | str) -> None:
        table.add_row(
            Text(key, style=FAINT), value if isinstance(value, Text) else Text(value, style=BONE)
        )

    prop = deck.proposal
    row("round", Text(f"{deck.round_number:03d}", style=f"bold {BONE}"))
    if prop is not None:
        metric = Text(prop.component + " ", style=BONE)
        metric.append(prop.metric, style=CYAN)
        row("target metric", metric)
        if prop.grid_point:
            row("@ grid point", prop.grid_point)
        if prop.baseline_value is not None:
            row("baseline (golden)", Text(f"{prop.baseline_value:.3g}", style=f"bold {BONE}"))
        if prop.eval_tolerance:
            row("eval tolerance", Text(prop.eval_tolerance, style=CYAN))
    row("agents dispatched", Text(str(deck.dispatched), style=BONE))
    row("PRs open", Text(str(deck.open_prs), style=BONE))
    return Panel(
        table, title=Text("ii · round vitals", style=DIM), border_style=FAINT, title_align="left"
    )


def _rail(deck: DeckState) -> Group:
    return Group(_dag(deck.nodes), _vitals(deck))


# ---------------------------------------------------------------------------
# Main — the round (proposal)
# ---------------------------------------------------------------------------


def _proposal(deck: DeckState) -> Panel:
    prop = deck.proposal
    if prop is None:
        body: RenderableType = Text("Awaiting the head agent's proposal for this round.", style=DIM)
        return Panel(
            body, title=Text("01 · the round", style=DIM), border_style=AMBER, title_align="left"
        )

    head = Group(
        _eyebrow("target · single thing this round"),
        Text(prop.target, style=f"bold {BONE}"),
        Text(prop.rationale, style=DIM) if prop.rationale else Text(""),
    )

    grid = Table.grid(expand=True, padding=(0, 2))
    for _ in range(4):
        grid.add_column(ratio=1)

    def cell(label: str, value: Text) -> Group:
        return Group(_eyebrow(label), value)

    before = (
        Text(f"{prop.baseline_value:.3g}", style=BONE)
        if prop.baseline_value is not None
        else Text("—", style=FAINT)
    )
    dedup = Text("clean ✓", style=GREEN) if prop.dedup_clean else Text("re-attempt ✗", style=OXIDE)
    grid.add_row(
        cell("optimizes", Text(prop.metric, style=CYAN)),
        cell("before", before),
        cell("eval golden", Text(prop.eval_tolerance or "held", style=BONE)),
        cell("dedup check", dedup),
    )
    return Panel(
        Group(head, Text(""), grid),
        title=Text("01 · the round", style=DIM),
        border_style=AMBER,
        title_align="left",
    )


# ---------------------------------------------------------------------------
# Main — the gauntlet (5 gates for the active lane)
# ---------------------------------------------------------------------------

_GATE_BAR: dict[GateName, str] = {
    "lint": "ruff clean · per-commit",
    "types": "ty strict · 0 suppressions",
    "correctness": "unit + integration",
    "profiling": "target improved",
    "evaluation": "golden diff in tol",
}


def _gauntlet(deck: DeckState) -> Panel:
    lane = deck.active_lane
    if lane is None:
        return Panel(
            Text("No PR in the gauntlet yet.", style=DIM),
            title=Text("02 · the gauntlet", style=DIM),
            border_style=FAINT,
            title_align="left",
        )

    head = Table.grid(expand=True)
    head.add_column(ratio=1)
    head.add_column(justify="right")
    pr_label = f"PR #{lane.pr_number}" if lane.pr_number is not None else "(no PR yet)"
    passed = sum(1 for g in GATE_ORDER if lane.gate(g) == "pass")
    head.add_row(
        Group(Text(pr_label, style=f"bold {BONE}"), Text(lane.branch, style=CYAN)),
        Text(f"{passed} / {len(GATE_ORDER)} gates", style=_STATUS_STYLE.get(lane.status, AMBER)),
    )

    gates = Table.grid(expand=True, padding=(0, 1))
    for _ in GATE_ORDER:
        gates.add_column(ratio=1)
    cells: list[RenderableType] = []
    for name in GATE_ORDER:
        verdict = lane.gate(name)
        color = _GATE_STYLE[verdict]
        cell = Group(
            Text(f"{_GATE_GLYPH[verdict]} {name}", style=color),
            Text(_GATE_BAR[name], style=FAINT),
            Text(verdict, style=color),
        )
        cells.append(Panel(cell, border_style=color, padding=(0, 1)))
    gates.add_row(*cells)

    return Panel(
        Group(head, Text(""), gates),
        title=Text("02 · the gauntlet", style=DIM),
        border_style=FAINT,
        title_align="left",
    )


# ---------------------------------------------------------------------------
# Main — the fan-out (PR lanes)
# ---------------------------------------------------------------------------


def _lane_card(lane: AgentLane) -> Panel:
    status_color = _STATUS_STYLE.get(lane.status, AMBER)

    head = Table.grid(expand=True)
    head.add_column(ratio=1)
    head.add_column(justify="right")
    title = Text()
    title.append(f"{lane.component}  ", style=f"bold {BONE}")
    if lane.title:
        title.append(lane.title, style=DIM)
    delta = Text(lane.headline_delta, style=GREEN if lane.headline_delta.startswith("-") else DIM)
    head.add_row(title, delta)

    sub = Text()
    if lane.pr_number is not None:
        sub.append(f"PR #{lane.pr_number}  ", style=FAINT)
    if lane.branch:
        sub.append(f"⌥ {lane.branch}  ", style=CYAN)
    sub.append(lane.status, style=status_color)

    mg = Text()
    for name in GATE_ORDER:
        v = lane.gate(name)
        mg.append(f"{_GATE_GLYPH[v]} {name}  ", style=_GATE_STYLE[v])

    parts: list[RenderableType] = [head, sub, mg]

    if lane.profiling_rows:
        prof = Table(box=None, expand=True, pad_edge=False)
        prof.add_column("stage · p50", style=DIM)
        prof.add_column("before", justify="right", style=BONE)
        prof.add_column("after", justify="right", style=BONE)
        prof.add_column("Δ", justify="right")
        for r in lane.profiling_rows:
            d_color = GREEN if r.delta.strip().startswith(("-", "−")) else OXIDE
            prof.add_row(r.stage, r.before, r.after, Text(r.delta, style=d_color))
        parts.append(prof)

    if lane.note:
        parts.append(Text(lane.note, style=DIM))

    return Panel(Group(*parts), border_style=status_color, padding=(0, 1))


def _fanout(deck: DeckState) -> Panel:
    if not deck.lanes:
        return Panel(
            Text("No worker lanes dispatched yet.", style=DIM),
            title=Text("03 · the fan-out", style=DIM),
            border_style=FAINT,
            title_align="left",
        )
    cards = Group(*[_lane_card(lane) for lane in deck.lanes])
    return Panel(
        cards, title=Text("03 · the fan-out", style=DIM), border_style=FAINT, title_align="left"
    )


# ---------------------------------------------------------------------------
# Main — serial merge + re-validation
# ---------------------------------------------------------------------------


def _merge_queue(deck: DeckState) -> Panel:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=2)
    table.add_column(ratio=1)
    table.add_column(justify="right")

    ranked = sorted(
        deck.lanes,
        key=lambda lane: 0 if lane.status in {"accepted", "merged"} else 1,
    )
    n = 0
    any_ready = False
    for lane in ranked:
        n += 1
        ready = lane.status in {"accepted", "merged"} and not any_ready
        if lane.status == "flagged":
            tag = Text("held", style=OXIDE)
            sub = "gate failed · will not enter queue"
        elif ready:
            tag = Text("next ▸", style=GREEN)
            sub = "gates green · ready to merge"
            any_ready = True
        else:
            tag = Text(lane.status, style=DIM)
            sub = lane.status
        marker_color = GREEN if ready else FAINT
        table.add_row(
            Text(str(n), style=marker_color),
            Group(
                Text(
                    f"{lane.component} · PR #{lane.pr_number}"
                    if lane.pr_number
                    else lane.component,
                    style=BONE,
                ),
                Text(sub, style=FAINT),
            ),
            tag,
        )
    if n == 0:
        table.add_row(Text("·", style=FAINT), Text("queue empty", style=DIM), Text(""))

    note = Text(
        "After a merge, the gauntlet re-runs on post-merge main; only green clears the next merge.",
        style=FAINT,
    )
    return Panel(
        Group(table, Text(""), note),
        title=Text("merge queue", style=DIM),
        border_style=FAINT,
        title_align="left",
    )


def _revalidation(deck: DeckState) -> Panel:
    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(justify="right")
    if deck.trends:
        for row in deck.trends:
            if row.component == "e2e":
                continue
            delta = row.vs_baseline_pct
            if delta is None:
                val = Text(f"{row.p50_ms:.0f}ms" if row.p50_ms is not None else "—", style=BONE)
            else:
                color = GREEN if delta <= 0 else OXIDE
                val = Text(f"{delta:+.0f}% vs baseline", style=color)
            table.add_row(Text(row.component, style=DIM), val)
    else:
        table.add_row(Text("no harness history yet", style=DIM), Text("trends empty", style=FAINT))

    docs = Text(
        "¶ docs agent reconciles README · WORKING_NOTES · DECISIONS after a merge stands.",
        style=VIOLET,
    )
    return Panel(
        Group(table, Text(""), docs),
        title=Text("post-merge re-validation", style=DIM),
        border_style=FAINT,
        title_align="left",
    )


def _merge_section(deck: DeckState) -> Panel:
    cols = Columns([_merge_queue(deck), _revalidation(deck)], expand=True, equal=True)
    return Panel(
        cols, title=Text("04 · serial merge", style=DIM), border_style=FAINT, title_align="left"
    )


# ---------------------------------------------------------------------------
# Main — the ledgers
# ---------------------------------------------------------------------------


def _improvements_panel(deck: DeckState) -> Panel:
    if not deck.improvements:
        body: RenderableType = Text("No rounds recorded yet.", style=DIM)
    else:
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(width=5)
        table.add_column(ratio=1)
        table.add_column(justify="right")
        for i, entry in enumerate(deck.improvements, start=1):
            color = GREEN if entry.verdict == "accepted" else OXIDE
            comp = Text(f"{entry.component} · {entry.target}".strip(" ·"), style=BONE)
            table.add_row(Text(f"{i:03d}", style=FAINT), comp, Text(entry.verdict, style=color))
            if entry.metric:
                table.add_row(Text(""), Text(entry.metric, style=FAINT), Text(""))
        body = table
    return Panel(
        body, title=Text("IMPROVEMENTS.md", style=DIM), border_style=FAINT, title_align="left"
    )


def _requests_panel(deck: DeckState) -> Panel:
    if not deck.requests:
        body: RenderableType = Text("No open inter-agent requests.", style=DIM)
    else:
        rows: list[RenderableType] = []
        for req in deck.requests:
            color = {"open": AMBER, "accepted": CYAN, "done": GREEN}.get(req.status, FAINT)
            flow = Text()
            flow.append(req.requester, style=BONE)
            flow.append(" ←needs— ", style=AMBER)
            flow.append(req.producer, style=BONE)
            flow.append(f"   [{req.status}]", style=color)
            rows.append(Group(flow, Text(req.why, style=FAINT)))
        body = Group(*rows)
    return Panel(
        body, title=Text("API_REQUESTS.md", style=DIM), border_style=FAINT, title_align="left"
    )


def _ledgers(deck: DeckState) -> Panel:
    cols = Columns([_improvements_panel(deck), _requests_panel(deck)], expand=True, equal=True)
    return Panel(
        cols, title=Text("05 · the ledgers", style=DIM), border_style=FAINT, title_align="left"
    )


# ---------------------------------------------------------------------------
# Deck assembly
# ---------------------------------------------------------------------------


def render(deck: DeckState) -> RenderableType:
    """Assemble the whole deck as a flowing renderable.

    A flowing ``Group`` (rather than a fixed-height ``Layout``) is used for the
    body so the full deck renders top-to-bottom — in a tall terminal it fills
    the screen, and in a short one it scrolls naturally instead of cropping.
    """
    if deck.awaiting:
        awaiting = Align.center(
            Group(
                Text("bt · overseer deck", style=f"bold {BONE}"),
                Text(""),
                Text("Awaiting the first round.", style=AMBER),
                Text("Seed .oversight/round_state.json or run the improvement loop.", style=FAINT),
            ),
            vertical="middle",
        )
        return Panel(awaiting, border_style=FAINT)

    main = Group(
        _proposal(deck),
        _gauntlet(deck),
        _fanout(deck),
        _merge_section(deck),
        _ledgers(deck),
    )

    body = Table.grid(expand=True, padding=(0, 1))
    body.add_column(width=37)
    body.add_column(ratio=1)
    body.add_row(_rail(deck), main)

    return Group(_masthead(deck), body)


# ---------------------------------------------------------------------------
# Input handling + the live loop
# ---------------------------------------------------------------------------


@contextmanager
def _raw_mode(stream: object) -> Iterator[None]:
    """Best-effort cbreak mode so a bare ``q`` quits; a no-op without a tty."""
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _quit_requested() -> bool:
    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return False
    ch = sys.stdin.read(1)
    return ch in {"q", "Q"}


def run(root: Path, history_dir: Path, interval: float, console: Console) -> None:
    deck = load(root, history_dir)
    with (
        _raw_mode(sys.stdin),
        Live(render(deck), console=console, screen=True, refresh_per_second=8) as live,
    ):
        import time

        next_reload = time.monotonic()
        while True:
            if _quit_requested():
                break
            now = time.monotonic()
            if now >= next_reload:
                deck = load(root, history_dir)
                live.update(render(deck))
                next_reload = now + interval
            time.sleep(0.05)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m oversight.tui")
    parser.add_argument("--root", default=".", help="repo root to read from")
    parser.add_argument("--history-dir", default=".oversight/history", help="parquet history dir")
    parser.add_argument("--interval", type=float, default=2.0, help="refresh seconds")
    parser.add_argument("--once", action="store_true", help="render one frame and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    root = Path(args.root)
    history_dir = Path(args.history_dir)
    console = Console()
    if args.once:
        console.print(render(load(root, history_dir)))
        return
    run(root, history_dir, args.interval, console)


if __name__ == "__main__":
    main()


__all__ = ["main", "render", "run"]
