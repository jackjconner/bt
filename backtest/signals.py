from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from etl.source import date_axis


@dataclass(frozen=True)
class SignalFrame:
    """Long-format signal: (date, id, signal). O(n_dates * n_assets)."""

    df: pl.DataFrame
    is_categorical: bool = False

    @staticmethod
    def random_continuous(
        n_assets: int, n_dates: int, start: str = "2000-01-01", seed: int | None = None
    ) -> SignalFrame:
        dates = date_axis(n_dates, start)
        ids = pl.int_range(0, n_assets, eager=True)
        grid = pl.DataFrame({"date": dates}).join(pl.DataFrame({"id": ids}), how="cross")
        rng = np.random.default_rng(seed)
        df = grid.with_columns(pl.Series("signal", rng.normal(0.0, 1.0, len(grid))))
        return SignalFrame(df=df, is_categorical=False)

    @staticmethod
    def random_binary(
        n_assets: int, n_dates: int, start: str = "2000-01-01", seed: int | None = None
    ) -> SignalFrame:
        dates = date_axis(n_dates, start)
        ids = pl.int_range(0, n_assets, eager=True)
        grid = pl.DataFrame({"date": dates}).join(pl.DataFrame({"id": ids}), how="cross")
        rng = np.random.default_rng(seed)
        df = grid.with_columns(pl.Series("signal", rng.integers(0, 2, len(grid)).astype(float)))
        return SignalFrame(df=df, is_categorical=True)
