from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl


@dataclass(frozen=True)
class ETLConfig:
    n_assets: int
    n_dates: int
    source_path: Path
    batch_size: int = 100_000


@dataclass(frozen=True)
class BatchLoader:
    """Materializes the entire dataset into memory in one shot.

    Peak working set is O(n_assets * n_dates) — the full file is read and
    held before any consumer sees a row.
    """

    config: ETLConfig

    def as_lazy(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.config.source_path)

    def load(self) -> pl.DataFrame:
        return cast(pl.DataFrame, self.as_lazy().collect(engine="in-memory"))
