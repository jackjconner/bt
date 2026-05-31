from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from .benchmark import Benchmark, get_benchmark
from .constituents import constituents
from .indexes import Index, get_index
from .returns import returns
from .universe import Universe, get_universe


@dataclass(frozen=True)
class DataConfig:
    name: str = "default"
    universe: str = "sp500"
    index: str = "sp500"
    benchmark: str = "sp500"
    start_date: str = "2000-01-01"
    end_date: str = "2022-12-31"


@dataclass(frozen=True)
class Data:
    config: DataConfig
    universe: Universe
    index: Index
    benchmark: Benchmark
    constituents: pl.DataFrame
    returns: pl.DataFrame

    def memory_report(self) -> None:
        datasets = {
            "constituents": self.constituents,
            "returns": self.returns,
            "universe.constituents": self.universe.constituents,
            "index.constituents": self.index.constituents,
            "benchmark.constituents": self.benchmark.constituents,
            "benchmark.returns": self.benchmark.returns,
        }
        total = 0.0
        for name, df in datasets.items():
            mb = df.estimated_size("mb")
            total += mb
            print(f"{name:<30} {mb:.2f} MB")
        print(f"{'total':<30} {total:.2f} MB")


def get_constituents(universe_name: str, start: str, end: str) -> pl.DataFrame:
    n = get_universe(universe_name).constituents["id"].n_unique()
    return constituents(n, start, end)


def get_returns(universe_name: str, start: str, end: str) -> pl.DataFrame:
    n = get_universe(universe_name).constituents["id"].n_unique()
    return returns(pl.int_range(0, n, eager=True), start, end)


def get_data(config: DataConfig = DataConfig()) -> Data:
    return Data(
        config,
        universe=get_universe(config.universe),
        index=get_index(config.index),
        benchmark=get_benchmark(config.benchmark),
        constituents=get_constituents(
            config.universe, config.start_date, config.end_date
        ),
        returns=get_returns(config.universe, config.start_date, config.end_date),
    )
