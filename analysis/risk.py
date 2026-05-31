"""Downside and distributional risk metrics.

Why a separate file: Sharpe lives in metrics.py for backward compatibility.
These metrics require a target/threshold return concept (Sortino, Calmar)
or pure distributional statistics (VaR, CVaR, skew, kurtosis) that are
conceptually distinct from the raw annualization in metrics.py.

All functions consume a `returns` DataFrame with a `return_1d` column of
*fractional* daily returns (not percent). Risk-free series, when required,
must also be fractional daily (matching `risk_free_rate.daily_rate`).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from etl.source import to_float

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _excess(returns: pl.DataFrame, rf: pl.Series | float = 0.0) -> np.ndarray:
    """Return excess daily returns as a numpy array.

    `rf` may be a scalar daily rate or a Polars Series aligned to `returns`
    by position. The Series path supports a time-varying risk-free rate from
    `risk_free_rate.daily_rate`.
    """
    r = returns["return_1d"].to_numpy()
    if isinstance(rf, pl.Series):
        return r - rf.to_numpy()
    return r - float(rf)


# ---------------------------------------------------------------------------
# CAGR / geometric annualized return
# ---------------------------------------------------------------------------


def cagr(nav: pl.DataFrame, n_sessions: int | None = None) -> float:
    """Geometric (compound) annualized return.

    Uses the ratio of terminal to initial NAV so the result is independent of
    the arithmetic mean approximation. `n_sessions` overrides the default of
    using the row count as the session count — pass the actual trading-calendar
    session count when the NAV series spans a calendar range that includes
    non-sessions, though in practice the backtest engine already runs on a
    session axis.

    Returns fractional annualized growth, e.g. 0.12 for 12 %.
    """
    nav_vals = nav["nav"]
    n = n_sessions if n_sessions is not None else len(nav_vals)
    if n <= 0:
        return 0.0
    start = to_float(nav_vals.first())
    end = to_float(nav_vals.last())
    if start <= 0:
        return 0.0
    return (end / start) ** (TRADING_DAYS / n) - 1.0


def annualized_return_calendar(
    returns: pl.DataFrame,
    calendar: pl.DataFrame,
) -> float:
    """CAGR computed against the actual session count in `calendar`.

    `calendar` is the `trading_calendar` dataset filtered to the period of
    interest — only rows where `is_session = True` are counted. This avoids
    the subtle 252-vs-actual-sessions mismatch that builds up over multi-year
    periods with holiday schedules.
    """
    sessions = int(calendar.filter(pl.col("is_session")).height)
    r = returns["return_1d"].to_numpy()
    terminal_growth = float(np.prod(1.0 + r))
    if sessions <= 0 or terminal_growth <= 0:
        return 0.0
    return terminal_growth ** (TRADING_DAYS / sessions) - 1.0


# ---------------------------------------------------------------------------
# Sortino
# ---------------------------------------------------------------------------


def sortino(
    returns: pl.DataFrame,
    rf: pl.Series | float = 0.0,
    target: float = 0.0,
) -> float:
    """Sortino ratio: mean excess return / downside deviation.

    Uses the same annualization factor as Sharpe (sqrt(252)) so the two are
    directly comparable in magnitude when returns are symmetric.

    `target` is a daily threshold below which returns count as losses;
    defaults to 0.0 (plain downside volatility). Providing a daily rf rate
    as `target` gives the classic downside-relative-to-rf Sortino.
    """
    excess = _excess(returns, rf)
    mean_excess = float(excess.mean())
    downside = excess - target
    downside_sq = downside[downside < 0] ** 2
    dd_vol = math.sqrt(float(downside_sq.mean())) if len(downside_sq) > 0 else 0.0
    if dd_vol == 0.0:
        return 0.0
    return mean_excess / dd_vol * math.sqrt(TRADING_DAYS)


# ---------------------------------------------------------------------------
# Calmar
# ---------------------------------------------------------------------------


def calmar(nav: pl.DataFrame, n_sessions: int | None = None) -> float:
    """Calmar ratio: CAGR / |max drawdown|.

    Returns 0.0 when max drawdown is zero (perfect upward path).
    """
    from .metrics import max_drawdown  # avoid circular at module level

    g = cagr(nav, n_sessions)
    mdd = abs(max_drawdown(nav))
    if mdd == 0.0:
        return 0.0
    return g / mdd


# ---------------------------------------------------------------------------
# VaR / CVaR (historical)
# ---------------------------------------------------------------------------


def var_historical(returns: pl.DataFrame, confidence: float = 0.95) -> float:
    """Historical (empirical) Value-at-Risk at `confidence` level.

    Returns a negative number representing the loss at the given percentile,
    e.g. -0.02 means a 2 % loss is not exceeded on `confidence` of days.
    Convention: sign is negative so the caller can compare directly to a
    drawdown series.
    """
    r = returns["return_1d"].to_numpy()
    return float(np.percentile(r, (1.0 - confidence) * 100.0))


def cvar_historical(returns: pl.DataFrame, confidence: float = 0.95) -> float:
    """Historical Conditional VaR (Expected Shortfall).

    Mean of returns that fall below the VaR threshold — the average loss in
    the worst `(1-confidence)` fraction of days.
    """
    r = returns["return_1d"].to_numpy()
    cutoff = np.percentile(r, (1.0 - confidence) * 100.0)
    tail = r[r <= cutoff]
    if len(tail) == 0:
        return float(cutoff)
    return float(tail.mean())


# ---------------------------------------------------------------------------
# Distributional statistics
# ---------------------------------------------------------------------------


def skewness(returns: pl.DataFrame) -> float:
    """Pearson skewness of daily returns (positive = right tail, fat upside)."""
    r = returns["return_1d"].to_numpy()
    if r.std() == 0:
        return 0.0
    return float(((r - r.mean()) ** 3).mean() / r.std() ** 3)


def excess_kurtosis(returns: pl.DataFrame) -> float:
    """Excess kurtosis of daily returns (normal = 0; fat tails > 0)."""
    r = returns["return_1d"].to_numpy()
    if r.std() == 0:
        return 0.0
    return float(((r - r.mean()) ** 4).mean() / r.std() ** 4) - 3.0


# ---------------------------------------------------------------------------
# Hit-rate, best/worst day
# ---------------------------------------------------------------------------


def hit_rate(returns: pl.DataFrame) -> float:
    """Fraction of days with positive return."""
    r = returns["return_1d"]
    return float((r > 0).sum()) / len(r)


def best_day(returns: pl.DataFrame) -> float:
    """Largest single-day return (fractional)."""
    return to_float(returns["return_1d"].max())


def worst_day(returns: pl.DataFrame) -> float:
    """Smallest (most negative) single-day return (fractional)."""
    return to_float(returns["return_1d"].min())
