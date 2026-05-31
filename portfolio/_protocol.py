from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import polars as pl

if TYPE_CHECKING:
    from .factors import FactorExposure
    from .holdings import HoldingsFrame


class PortfolioAnalyzer(Protocol):
    def compute_exposures(self, holdings: HoldingsFrame) -> FactorExposure: ...
    def rolling_vol(
        self, holdings: HoldingsFrame, returns: pl.DataFrame, window: int
    ) -> pl.DataFrame: ...
