from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from backtest.signals import SignalFrame
from etl.source import to_matrix


@dataclass(frozen=True)
class HoldingsFrame:
    """Long-format holdings: (date, id, weight). O(n_dates * n_assets)."""

    df: pl.DataFrame

    def to_wide(self) -> np.ndarray:
        """Dense (n_dates, n_assets) weight matrix — the peak allocation."""
        mat, _ = to_matrix(self.df, "weight")
        return mat

    @staticmethod
    def from_signals(signals: SignalFrame) -> HoldingsFrame:
        """Cross-sectional softmax of the signal each date → weights summing to 1.

        Vectorized over the whole panel in Polars: O(n_dates * n_assets).
        """
        df = (
            signals.df.with_columns(
                (pl.col("signal") - pl.col("signal").max().over("date")).exp().alias("_e")
            )
            .with_columns((pl.col("_e") / pl.col("_e").sum().over("date")).alias("weight"))
            .select("date", "id", "weight")
        )
        return HoldingsFrame(df=df)
