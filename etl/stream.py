from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import polars as pl

from .batch import ETLConfig


@dataclass(frozen=True)
class StreamLoader:
    """Processes the dataset through the streaming engine.

    The streaming engine bounds the in-flight working set to a morsel at a
    time rather than the whole file. The final collect still materializes
    O(n_assets * n_dates) rows, but the *peak* during scanning/filtering is
    far below the batch path — that gap is the measurement target.
    """

    config: ETLConfig

    def as_lazy(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.config.source_path)

    def load(self) -> pl.DataFrame:
        return cast(pl.DataFrame, self.as_lazy().collect(engine="streaming"))
