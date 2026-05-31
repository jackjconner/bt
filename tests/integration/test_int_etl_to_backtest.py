"""Contract: etl loader output → production backtest input.

The backtest engine consumes a (date, id, return) frame plus the price /
cost / mask panels the etl loader produces. This asserts those shapes line up
and the engine runs a full path against them.
"""

from __future__ import annotations

import dataclasses
import math

import polars as pl

from backtest import ProductionBacktestConfig, ProductionBacktestEngine, SignalFrame
from harness.components import returns_from_prices


def _momentum(loader) -> pl.DataFrame:
    return (
        loader.load("alpha_signals")
        .filter(pl.col("signal_name") == "momentum")
        .select("date", "id", "signal")
    )


def test_loader_panels_drive_production_backtest(synth) -> None:
    loader, spec = synth.loader, synth.spec

    prices = loader.load("prices")  # schema-validated at load
    returns = returns_from_prices(prices)

    cfg = ProductionBacktestConfig(
        n_assets=spec.n_assets,
        n_dates=spec.n_dates,
        enable_costs=True,
        enable_slippage=True,
        enable_universe_mask=True,
        max_weight=0.1,
    )
    result = ProductionBacktestEngine(cfg).run(
        returns,
        SignalFrame(df=_momentum(loader), is_categorical=False),
        prices=prices,
        transaction_costs=loader.load("transaction_costs"),
        universe_mask=loader.load("universe_mask"),
    )

    assert result.nav_history.height == spec.n_dates
    assert set(result.nav_history.columns) == {"date", "nav"}
    assert math.isfinite(result.nav_history["nav"][-1])
    assert result.nav_history["nav"][-1] > 0.0


def test_costs_reduce_terminal_nav(synth) -> None:
    loader, spec = synth.loader, synth.spec
    prices = loader.load("prices")
    returns = returns_from_prices(prices)
    sig = SignalFrame(df=_momentum(loader), is_categorical=False)
    tcosts = loader.load("transaction_costs")
    umask = loader.load("universe_mask")
    base_cfg = ProductionBacktestConfig(
        n_assets=spec.n_assets,
        n_dates=spec.n_dates,
        enable_universe_mask=True,
        max_weight=0.1,
    )
    net_cfg = dataclasses.replace(base_cfg, enable_costs=True, enable_slippage=True)
    gross = ProductionBacktestEngine(base_cfg).run(
        returns, sig, prices=prices, transaction_costs=tcosts, universe_mask=umask
    )
    net = ProductionBacktestEngine(net_cfg).run(
        returns, sig, prices=prices, transaction_costs=tcosts, universe_mask=umask
    )

    assert net.nav_history["nav"][-1] < gross.nav_history["nav"][-1]
