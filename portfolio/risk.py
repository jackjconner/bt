from __future__ import annotations

import numpy as np
import polars as pl

from etl.source import to_matrix

from .holdings import HoldingsFrame


def rolling_vol(holdings: HoldingsFrame, returns: pl.DataFrame, window: int) -> pl.DataFrame:
    """Rolling portfolio volatility: sqrt(wᵀ Σ w) over a trailing window.

    The per-step asset covariance Σ is (n_assets, n_assets), so this is
    O(n_assets²) memory and CPU per date — the quadratic hotspot. Output is
    O(n_dates).
    """
    W = holdings.to_wide()
    R, dates = to_matrix(returns, "return")
    n_dates, _ = R.shape

    out_dates: list = []
    vols: list[float] = []
    for t in range(window, n_dates):
        cov = np.cov(R[t - window : t].T)
        w = W[t]
        var = float(w @ cov @ w)
        out_dates.append(dates[t])
        vols.append(float(np.sqrt(max(var, 0.0))))

    return pl.DataFrame({"date": out_dates, "portfolio_vol": vols})


def var_historical(
    holdings: HoldingsFrame,
    returns: pl.DataFrame,
    window: int,
    confidence: float = 0.95,
) -> pl.DataFrame:
    """Historical VaR from the trailing window of portfolio returns. O(n_dates)."""
    W = holdings.to_wide()
    R, dates = to_matrix(returns, "return")
    n_dates, _ = R.shape
    q = (1.0 - confidence) * 100.0

    out_dates: list = []
    vars_: list[float] = []
    for t in range(window, n_dates):
        port_rets = R[t - window : t] @ W[t]
        out_dates.append(dates[t])
        vars_.append(float(-np.percentile(port_rets, q)))

    return pl.DataFrame({"date": out_dates, "var": vars_})


def drawdown_series(nav: pl.DataFrame) -> pl.DataFrame:
    """Drawdown from running peak. O(n_dates)."""
    return nav.with_columns(
        (pl.col("nav") / pl.col("nav").cum_max() - 1.0).alias("drawdown")
    ).select("date", "drawdown")
