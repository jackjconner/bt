"""Oversight deck for bt's component-improvement loop.

A live, auto-refreshing terminal dashboard of one improvement round: the
proposal, the five-gate gauntlet, the fan-out PR lanes, the serial merge queue,
and the two ledgers. Reads from a live ``round_state.json`` (preferred), git +
``gh``, the markdown ledgers, and the parquet history. See ``oversight.tui``.
"""

from __future__ import annotations

from oversight.read_model import DeckState, load
from oversight.state import (
    AgentLane,
    GoldenSummary,
    ProfilingRow,
    Proposal,
    RoundState,
    read_round_state,
    write_round_state,
)

__all__ = [
    "AgentLane",
    "DeckState",
    "GoldenSummary",
    "ProfilingRow",
    "Proposal",
    "RoundState",
    "load",
    "read_round_state",
    "write_round_state",
]
