from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import polars as pl

if TYPE_CHECKING:
    from .engine import BacktestResult
    from .signals import SignalFrame


class BacktestRunner(Protocol):
    def run(self, returns: pl.DataFrame, signals: SignalFrame) -> BacktestResult: ...
