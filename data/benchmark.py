from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .constituents import constituents
from .dates import START_DATE, END_DATE
from .returns import returns

_REGISTERED_BENCHMARKS: dict[str, Benchmark] = {}


@dataclass(frozen=True)
class Benchmark:
    name: str
    constituents: pl.DataFrame
    returns: pl.DataFrame


sp_500 = Benchmark(
    "sp500",
    constituents(500),
    returns(pl.int_range(0, 500, eager=True), START_DATE, END_DATE),
)
russell_1000 = Benchmark(
    "russell_1000",
    constituents(1000),
    returns(pl.int_range(0, 1000, eager=True), START_DATE, END_DATE),
)
russell_3000 = Benchmark(
    "russell_3000",
    constituents(3000),
    returns(pl.int_range(0, 3000, eager=True), START_DATE, END_DATE),
)

_REGISTERED_BENCHMARKS["sp500"] = sp_500
_REGISTERED_BENCHMARKS["russell_1000"] = russell_1000
_REGISTERED_BENCHMARKS["russell_3000"] = russell_3000


def get_benchmark(name: str) -> Benchmark:
    return _REGISTERED_BENCHMARKS[name]
