from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from etl.source import to_matrix

from .signals import SignalFrame


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass(frozen=True)
class BacktestConfig:
    n_assets: int
    n_dates: int
    initial_cash: float = 1_000_000.0
    rebalance_every: int = 1


@dataclass
class PortfolioState:
    """Mutable state threaded through the event loop. O(n_assets)."""

    date: date
    positions: np.ndarray
    cash: float
    nav: float


@dataclass(frozen=True)
class BacktestResult:
    nav_history: pl.DataFrame  # date, nav            — O(n_dates)
    trade_log: pl.DataFrame  # date, id, quantity   — O(n_rebalances * n_assets)
    final_positions: np.ndarray  # (n_assets,)          — O(n_assets)
    # Production fields — default to empty so legacy construction sites still work.
    fill_log: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Date),
                "id": pl.Series([], dtype=pl.Int64),
                "shares": pl.Series([], dtype=pl.Float64),
                "fill_price": pl.Series([], dtype=pl.Float64),
                "cost": pl.Series([], dtype=pl.Float64),
                "slippage": pl.Series([], dtype=pl.Float64),
            }
        )
    )
    cash_history: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(
            {"date": pl.Series([], dtype=pl.Date), "cash": pl.Series([], dtype=pl.Float64)}
        )
    )


@dataclass(frozen=True)
class BacktestEngine:
    config: BacktestConfig

    def run(self, returns: pl.DataFrame, signals: SignalFrame) -> BacktestResult:
        R, dates = to_matrix(returns, "return")
        S, _ = to_matrix(signals.df, "signal")
        n_dates, n_assets = R.shape

        positions = np.zeros(n_assets)
        nav = self.config.initial_cash

        nav_hist: list[float] = []
        trade_dates: list[date] = []
        trade_ids: list[int] = []
        trade_qty: list[float] = []
        asset_ids = np.arange(n_assets)

        for t in range(n_dates):
            if t % self.config.rebalance_every == 0:
                target = _softmax(S[t])
                deltas = target - positions
                trade_dates.extend([dates[t]] * n_assets)
                trade_ids.extend(asset_ids.tolist())
                trade_qty.extend((deltas * nav).tolist())
                positions = target

            r = R[t] / 100.0
            port_ret = float(positions @ r)
            nav *= 1.0 + port_ret
            nav_hist.append(nav)

            drifted = positions * (1.0 + r)
            total = drifted.sum()
            if total > 0:
                positions = drifted / total

        return BacktestResult(
            nav_history=pl.DataFrame({"date": dates, "nav": nav_hist}),
            trade_log=pl.DataFrame({"date": trade_dates, "id": trade_ids, "quantity": trade_qty}),
            final_positions=positions,
        )
