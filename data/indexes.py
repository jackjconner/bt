from __future__ import annotations

from dataclasses import dataclass
import polars as pl

from .constituents import constituents

_REGISTERED_INDEXES: dict[str, Index] = {}


@dataclass(frozen=True)
class Index:
    name: str
    constituents: pl.DataFrame
    freq: str = "1d"


sp_500_index = Index("sp500", constituents(500))
russell_1000_index = Index("russell_1000", constituents(1000))
russell_3000_index = Index("russell_3000", constituents(3000))

_REGISTERED_INDEXES["sp500"] = sp_500_index
_REGISTERED_INDEXES["russell_1000"] = russell_1000_index
_REGISTERED_INDEXES["russell_3000"] = russell_3000_index


def get_index(name: str) -> Index:
    return _REGISTERED_INDEXES[name]
