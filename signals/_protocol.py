from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import polars as pl

if TYPE_CHECKING:
    from backtest.signals import SignalFrame

    from .ic import ICResult


class SignalEvaluator(Protocol):
    def evaluate(self, signals: SignalFrame, returns: pl.DataFrame) -> ICResult: ...
