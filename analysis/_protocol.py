from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from backtest.engine import BacktestResult

    from .metrics import AnalysisResult


class BacktestAnalyzer(Protocol):
    def analyze(self, result: BacktestResult) -> AnalysisResult: ...
