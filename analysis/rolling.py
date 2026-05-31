"""Rolling window metrics.

Why this file: rolling analytics require a different computation pattern
(stride over time) from point-in-time summary statistics. Keeping them
separate avoids bloating metrics.py and makes the window logic easy to
audit in isolation.

All functions return Polars DataFrames with a `date` column and one or more
metric columns. Missing leading windows are represented as `null` so callers
can decide on fill policy.

Benchmark `returns_bmk` frames must have `(date, return_1d)` in fractional
units before being passed here.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

TRADING_DAYS = 252


def _rolling_numpy(
    dates: list,
    values: np.ndarray,
    window: int,
    fn,
) -> list:
    """Apply `fn(window_array) -> float` with a min-period of `window`."""
    out: list[float | None] = []
    for i in range(len(values)):
        if i + 1 < window:
            out.append(None)
        else:
            out.append(fn(values[i - window + 1 : i + 1]))
    return out


# ---------------------------------------------------------------------------
# Rolling Sharpe
# ---------------------------------------------------------------------------


def rolling_sharpe(
    returns: pl.DataFrame,
    window: int = 63,
    rf: float = 0.0,
) -> pl.DataFrame:
    """Rolling annualized Sharpe ratio.

    `window` is in trading days. Leading rows with fewer than `window`
    observations are null. `rf` is a constant daily rate; for a time-varying
    rate use `rolling_sharpe_rf`.
    """
    r = returns["return_1d"].to_numpy()
    daily_rf = rf

    def _sharpe(arr: np.ndarray) -> float:
        exc = arr - daily_rf
        std = exc.std()
        if std == 0:
            return 0.0
        return float(exc.mean() / std * math.sqrt(TRADING_DAYS))

    vals = _rolling_numpy(returns["date"].to_list(), r, window, _sharpe)
    return pl.DataFrame({"date": returns["date"], "rolling_sharpe": vals})


# ---------------------------------------------------------------------------
# Rolling volatility
# ---------------------------------------------------------------------------


def rolling_vol(
    returns: pl.DataFrame,
    window: int = 63,
) -> pl.DataFrame:
    """Rolling annualized volatility (standard deviation).

    Returns `(date, rolling_vol)`.
    """
    r = returns["return_1d"].to_numpy()

    def _vol(arr: np.ndarray) -> float:
        return float(arr.std() * math.sqrt(TRADING_DAYS))

    vals = _rolling_numpy(returns["date"].to_list(), r, window, _vol)
    return pl.DataFrame({"date": returns["date"], "rolling_vol": vals})


# ---------------------------------------------------------------------------
# Rolling beta
# ---------------------------------------------------------------------------


def rolling_beta(
    returns: pl.DataFrame,
    benchmark: pl.DataFrame,
    window: int = 63,
) -> pl.DataFrame:
    """Rolling beta of strategy returns on benchmark returns.

    Inner-joins on date first, then applies a rolling window. Returns
    `(date, rolling_beta)` on the intersection date range.
    """
    joined = returns.join(benchmark, on="date", how="inner", suffix="_bmk").sort("date")
    r = joined["return_1d"].to_numpy()
    b = joined["return_1d_bmk"].to_numpy()

    def _beta(idx: int) -> float | None:
        if idx + 1 < window:
            return None
        ri = r[idx - window + 1 : idx + 1]
        bi = b[idx - window + 1 : idx + 1]
        # ddof=0 throughout so numerator and denominator share the same divisor
        b_var = float(bi.var(ddof=0))
        if b_var == 0:
            return 0.0
        cov = float(((ri - ri.mean()) * (bi - bi.mean())).mean())
        return cov / b_var

    vals = [_beta(i) for i in range(len(r))]
    return joined.select("date").with_columns(pl.Series("rolling_beta", vals))


# ---------------------------------------------------------------------------
# Rolling drawdown
# ---------------------------------------------------------------------------


def rolling_max_drawdown(
    returns: pl.DataFrame,
    window: int = 63,
) -> pl.DataFrame:
    """Maximum drawdown within each rolling window.

    Builds a local NAV index for each window and records the worst peak-to-
    trough. Returns `(date, rolling_max_drawdown)`. Leading rows are null.
    """
    r = returns["return_1d"].to_numpy()

    def _mdd(arr: np.ndarray) -> float:
        nav = np.cumprod(1.0 + arr)
        running_max = np.maximum.accumulate(nav)
        dd = nav / running_max - 1.0
        return float(dd.min())

    vals = _rolling_numpy(returns["date"].to_list(), r, window, _mdd)
    return pl.DataFrame({"date": returns["date"], "rolling_max_drawdown": vals})
