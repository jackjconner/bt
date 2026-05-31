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

import datetime as dt
import math

import numpy as np
import polars as pl

from etl.schema import DataTypeLike
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


# ---------------------------------------------------------------------------
# Drawdown duration & recovery analytics
# ---------------------------------------------------------------------------

_DRAWDOWN_RECOVERY_SCHEMA: dict[str, DataTypeLike] = {
    "peak_date": pl.Date,
    "trough_date": pl.Date,
    "recovery_date": pl.Date,
    "depth": pl.Float64,
    "drawdown_days": pl.Int64,
    "recovery_days": pl.Int64,
    "peak_to_recovery_days": pl.Int64,
}


def drawdown_recovery(nav: pl.DataFrame) -> pl.DataFrame:
    """Per-event drawdown duration & recovery table from the NAV/equity path.

    A *drawdown event* runs from a peak (a session at a running high, so
    drawdown == 0) through the subsequent trough (the most negative point) to
    the recovery (the first session that regains the prior peak, i.e. drawdown
    returns to 0). Events are isolated off the same drawdown series the rest of
    ``analysis`` reports — ``nav / nav.cum_max() - 1`` — so the deepest event's
    ``depth`` equals :func:`analysis.metrics.max_drawdown` exactly.

    One row per event, in chronological order, with columns:

    - ``peak_date`` — date of the running high the event drops from.
    - ``trough_date`` — date of the most negative drawdown within the event.
    - ``recovery_date`` — first date the event regains the prior peak;
      ``None`` for an unrecovered tail drawdown still open at series end.
    - ``depth`` — drawdown at the trough (negative fraction, e.g. ``-0.20``).
    - ``drawdown_days`` — sessions from peak to trough (peak->trough).
    - ``recovery_days`` — sessions from trough to recovery; ``None`` if
      unrecovered.
    - ``peak_to_recovery_days`` — sessions from peak to recovery (the full
      underwater span); ``None`` if unrecovered.

    Durations are counted in *sessions* (index steps), matching the
    session-axis convention the engine runs on. A monotonically rising or flat
    NAV yields an empty (but correctly typed) frame.
    """
    if nav.height == 0:
        return pl.DataFrame(schema=_DRAWDOWN_RECOVERY_SCHEMA)

    dd = nav.select(
        pl.col("date"),
        (pl.col("nav") / pl.col("nav").cum_max() - 1.0).alias("drawdown"),
    )
    dates = dd["date"].to_list()
    drawdown = dd["drawdown"].to_numpy()

    peak_dates: list[dt.date] = []
    trough_dates: list[dt.date] = []
    recovery_dates: list[dt.date | None] = []
    depths: list[float] = []
    drawdown_days: list[int] = []
    recovery_days: list[int | None] = []
    peak_to_recovery_days: list[int | None] = []

    n = len(drawdown)
    i = 0
    while i < n:
        if drawdown[i] >= 0.0:
            i += 1
            continue
        # Event opened at i; the peak is the immediately preceding session,
        # which by construction sat at a running high (drawdown == 0).
        peak_idx = i - 1
        trough_idx = i
        j = i
        while j < n and drawdown[j] < 0.0:
            if drawdown[j] < drawdown[trough_idx]:
                trough_idx = j
            j += 1
        # j is either past the end (unrecovered) or the first recovered session.
        recovered = j < n
        recovery_idx = j if recovered else None

        peak_dates.append(dates[peak_idx])
        trough_dates.append(dates[trough_idx])
        depths.append(float(drawdown[trough_idx]))
        drawdown_days.append(trough_idx - peak_idx)
        if recovery_idx is not None:
            recovery_dates.append(dates[recovery_idx])
            recovery_days.append(recovery_idx - trough_idx)
            peak_to_recovery_days.append(recovery_idx - peak_idx)
        else:
            recovery_dates.append(None)
            recovery_days.append(None)
            peak_to_recovery_days.append(None)
        i = j

    return pl.DataFrame(
        {
            "peak_date": peak_dates,
            "trough_date": trough_dates,
            "recovery_date": recovery_dates,
            "depth": depths,
            "drawdown_days": drawdown_days,
            "recovery_days": recovery_days,
            "peak_to_recovery_days": peak_to_recovery_days,
        },
        schema=_DRAWDOWN_RECOVERY_SCHEMA,
    )
