from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from etl.source import to_float

if TYPE_CHECKING:
    from backtest.engine import BacktestResult

TRADING_DAYS = 252


@dataclass(frozen=True)
class AnalysisResult:
    returns_series: pl.DataFrame  # date, return_1d   — O(n_dates)
    drawdown_series: pl.DataFrame  # date, drawdown    — O(n_dates)
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
    excess = r - rf if isinstance(rf, pl.Series) else r - rf / TRADING_DAYS
    std = excess.std()
    if std is None or std == 0:
        return 0.0
    return to_float(excess.mean()) / to_float(std) * (TRADING_DAYS**0.5)


def max_drawdown(nav: pl.DataFrame) -> float:
    dd = nav.select((pl.col("nav") / pl.col("nav").cum_max() - 1.0).alias("dd"))
    return to_float(dd["dd"].min())


@dataclass(frozen=True)
class BacktestAnalyzerImpl:
    risk_free: float = 0.0

    def analyze(self, result: BacktestResult) -> AnalysisResult:
        """Compute the full ``AnalysisResult`` for a backtest.

        Delegates to ``analysis.engine.analyze_fused``, which derives the
        returns and drawdown series and every reported scalar (sharpe,
        max-drawdown, annualized return/vol, CAGR, Sortino) from a single set of
        fused passes over the NAV frame. The result is bit-for-bit identical to
        composing the per-metric helpers (``returns_from_nav`` + ``sharpe`` +
        ``max_drawdown`` + ``risk.cagr`` + ``risk.sortino``) at ``risk_free=0``,
        which is the production and harness call.
        """
        from .engine import analyze_fused

        return analyze_fused(result.nav_history, self.risk_free)
