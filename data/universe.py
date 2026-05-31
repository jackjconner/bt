from __future__ import annotations

from dataclasses import dataclass
import polars as pl

from .constituents import constituents

_REGISTERED_UNIVERSES: dict[str, Universe] = {}


@dataclass(frozen=True)
class Universe:
    name: str
    constituents: pl.DataFrame
    freq: str = "1d"


def universe(name: str, constituents: pl.DataFrame) -> Universe:
    return Universe(name, constituents)


sp_500_universe = universe("sp500", constituents(500))
russell_1000_universe = universe("russell_1000", constituents(1000))
russell_3000_universe = universe("russell_3000", constituents(3000))
msci_acwi_universe = universe("msci_acwi", constituents(2000))
msci_acwi_imi_universe = universe("msci_acwi_imi", constituents(8000))

_REGISTERED_UNIVERSES["sp500"] = sp_500_universe
_REGISTERED_UNIVERSES["russell_1000"] = russell_1000_universe
_REGISTERED_UNIVERSES["russell_3000"] = russell_3000_universe
_REGISTERED_UNIVERSES["msci_acwi"] = msci_acwi_universe
_REGISTERED_UNIVERSES["msci_acwi_imi"] = msci_acwi_imi_universe


def registered_universes() -> list[Universe]:
    return list(_REGISTERED_UNIVERSES.values())


def get_universe(name: str) -> Universe:
    return _REGISTERED_UNIVERSES[name]
