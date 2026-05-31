from __future__ import annotations

from typing import Protocol

import polars as pl


class Loader(Protocol):
    def load(self) -> pl.DataFrame: ...
