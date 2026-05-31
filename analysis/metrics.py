from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from backtest.engine import BacktestResult

TRADING_DAYS = 252


@dataclass(frozen=True)
class AnalysisResult:
    returns_series: pl.DataFrame    # date, return_1d   — O(n_dates)
    drawdown_series: pl.DataFrame   # date, drawdown    — O(n_dates)
    sharpe: float
    max_drawdown: float
    annualized_return: float
    annualized_vol: float
    # Enriched fields added in the production build — default to sentinel
    # values so existing call-sites that construct AnalysisResult directly
    # (e.g. tests) are not broken.
    cagr: float = field(default=0.0)
    sortino: float = field(default=0.0)


def returns_from_nav(nav: pl.DataFrame) -> pl.DataFrame:
    return (
        nav.with_columns((pl.col("nav") / pl.col("nav").shift(1) - 1.0).alias("return_1d"))
        .drop_nulls()
        .select("date", "return_1d")
    )


def sharpe(
    returns: pl.DataFrame,
    rf: float | pl.Series = 0.0,
) -> float:
    """Annualized Sharpe ratio.

    `rf` may be a scalar annual rate (divided internally by TRADING_DAYS) or
    a Polars Series of fractional *daily* rates aligned to `returns` by
    position — the latter supports a time-varying risk-free rate from
    `risk_free_rate.daily_rate`.
    """
    r = returns["return_1d"]
    if isinstance(rf, pl.Series):
        excess = r - rf
    else:
        excess = r - rf / TRADING_DAYS
    std = excess.std()
    if std is None or std == 0:
        return 0.0
    return float(excess.mean() / std * (TRADING_DAYS ** 0.5))


def max_drawdown(nav: pl.DataFrame) -> float:
    dd = nav.select((pl.col("nav") / pl.col("nav").cum_max() - 1.0).alias("dd"))
    return float(dd["dd"].min())


@dataclass(frozen=True)
class BacktestAnalyzerImpl:
    risk_free: float = 0.0

    def analyze(self, result: BacktestResult) -> AnalysisResult:
        nav = result.nav_history
        rets = returns_from_nav(nav)
        dd = nav.with_columns(
            (pl.col("nav") / pl.col("nav").cum_max() - 1.0).alias("drawdown")
        ).select("date", "drawdown")

        ann_vol = float(rets["return_1d"].std() or 0.0) * (TRADING_DAYS ** 0.5)
        ann_ret = float(rets["return_1d"].mean() or 0.0) * TRADING_DAYS

        from .risk import cagr as _cagr, sortino as _sortino

        return AnalysisResult(
            returns_series=rets,
            drawdown_series=dd,
            sharpe=sharpe(rets, self.risk_free),
            max_drawdown=max_drawdown(nav),
            annualized_return=ann_ret,
            annualized_vol=ann_vol,
            cagr=_cagr(nav),
            sortino=_sortino(rets, self.risk_free / TRADING_DAYS),
        )
