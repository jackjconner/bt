"""Benchmark-relative analytics.

Why this file: benchmark comparison is a distinct concern from absolute risk
metrics. All functions require a benchmark return series aligned to the
strategy by date. The benchmark series comes from `benchmark_returns` where
`return` is in PERCENT units; callers must divide by 100 before passing here
(consistent with how the backtest engine converts `R / 100`).

Convention throughout: returns are *fractional* daily (e.g. 0.01 = 1 %).
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

TRADING_DAYS = 252


def _align(
    returns: pl.DataFrame,
    benchmark: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner-join strategy and benchmark on date; return aligned numpy arrays.

    `returns` must have `(date, return_1d)`.
    `benchmark` must have `(date, return_1d)` — caller renames the column.
    Joining on date is essential: the two series may not have identical index
    ranges when the benchmark covers a wider calendar.
    """
    joined = returns.join(benchmark, on="date", how="inner", suffix="_bmk")
    r = joined["return_1d"].to_numpy()
    b = joined["return_1d_bmk"].to_numpy()
    return r, b


# ---------------------------------------------------------------------------
# Beta / alpha (OLS)
# ---------------------------------------------------------------------------


def beta(returns: pl.DataFrame, benchmark: pl.DataFrame) -> float:
    """Market beta: OLS slope of strategy daily returns on benchmark returns.

    A benchmark regressed against itself gives beta = 1.0 exactly. Uses
    population variance (ddof=0) throughout so the covariance matrix numerator
    and denominator share the same divisor and the ratio is exact.
    `benchmark` frame must carry a `return_1d` column.
    """
    r, b = _align(returns, benchmark)
    b_var = float(b.var(ddof=0))
    if b_var == 0:
        return 0.0
    b_mean, r_mean = b.mean(), r.mean()
    cov = float(((r - r_mean) * (b - b_mean)).mean())
    return cov / b_var


def alpha(
    returns: pl.DataFrame,
    benchmark: pl.DataFrame,
    rf: pl.Series | float = 0.0,
) -> float:
    """Jensen's alpha: annualized intercept after stripping market beta.

    alpha = mean(r_excess) - beta * mean(b_excess), annualized by *252.
    Uses population-variance OLS (ddof=0) for consistency with `beta()`.
    The scalar/Series `rf` is a fractional daily risk-free rate; it is
    aligned to `returns` by position when a Series is supplied.
    """
    r, b = _align(returns, benchmark)
    rf_arr = rf.to_numpy()[: len(r)] if isinstance(rf, pl.Series) else np.full(len(r), float(rf))

    r_exc = r - rf_arr
    b_exc = b - rf_arr[: len(b)]

    b_var = float(b_exc.var(ddof=0))
    if b_var == 0:
        return 0.0

    b_mean = float(b_exc.mean())
    r_mean = float(r_exc.mean())
    cov = float(((r_exc - r_mean) * (b_exc - b_mean)).mean())
    b_val = cov / b_var
    a_daily = r_mean - b_val * b_mean
    return a_daily * TRADING_DAYS


def r_squared(returns: pl.DataFrame, benchmark: pl.DataFrame) -> float:
    """R² of the strategy-on-benchmark OLS regression.

    Measures how much of strategy variance is explained by benchmark
    co-movement; high R² with low alpha suggests a closet index fund.
    """
    r, b = _align(returns, benchmark)
    if r.var() == 0 or b.var() == 0:
        return 0.0
    corr = float(np.corrcoef(r, b)[0, 1])
    return corr**2


# ---------------------------------------------------------------------------
# Tracking error / information ratio
# ---------------------------------------------------------------------------


def tracking_error(returns: pl.DataFrame, benchmark: pl.DataFrame) -> float:
    """Annualized standard deviation of active returns (strategy minus benchmark)."""
    r, b = _align(returns, benchmark)
    active = r - b
    return float(active.std()) * math.sqrt(TRADING_DAYS)


def information_ratio(returns: pl.DataFrame, benchmark: pl.DataFrame) -> float:
    """IR: annualized mean active return divided by tracking error.

    The sign convention matches standard usage: positive means the strategy
    outperforms on a risk-adjusted basis.
    """
    r, b = _align(returns, benchmark)
    active = r - b
    te = float(active.std()) * math.sqrt(TRADING_DAYS)
    if te == 0:
        return 0.0
    return float(active.mean()) * TRADING_DAYS / te


# ---------------------------------------------------------------------------
# Up / down capture
# ---------------------------------------------------------------------------


def up_capture(returns: pl.DataFrame, benchmark: pl.DataFrame) -> float:
    """Up-market capture ratio.

    Mean strategy return on days the benchmark is positive, divided by mean
    benchmark return on those same days. Values > 1 indicate the strategy
    amplifies positive benchmark moves.
    """
    r, b = _align(returns, benchmark)
    mask = b > 0
    if mask.sum() == 0:
        return 0.0
    return float(r[mask].mean()) / float(b[mask].mean())


def down_capture(returns: pl.DataFrame, benchmark: pl.DataFrame) -> float:
    """Down-market capture ratio.

    Mean strategy return on days the benchmark is negative, divided by mean
    benchmark return on those same days. Values < 1 indicate the strategy
    loses less on down days (desirable).
    """
    r, b = _align(returns, benchmark)
    mask = b < 0
    if mask.sum() == 0:
        return 0.0
    return float(r[mask].mean()) / float(b[mask].mean())


# ---------------------------------------------------------------------------
# Active returns / relative drawdown
# ---------------------------------------------------------------------------


def active_returns(returns: pl.DataFrame, benchmark: pl.DataFrame) -> pl.DataFrame:
    """Date-aligned series of (strategy return − benchmark return).

    Returns a DataFrame with columns `(date, active_return)` on the
    inner-join date range. Used downstream for drawdown and rolling metrics.
    """
    joined = returns.join(benchmark, on="date", how="inner", suffix="_bmk")
    return joined.select(
        pl.col("date"),
        (pl.col("return_1d") - pl.col("return_1d_bmk")).alias("active_return"),
    )


def relative_drawdown(returns: pl.DataFrame, benchmark: pl.DataFrame) -> pl.DataFrame:
    """Drawdown of the cumulative active-return index.

    Builds a notional portfolio whose daily return is (strategy − benchmark);
    the drawdown of that series captures underperformance streaks. Returns a
    DataFrame with `(date, rel_drawdown)`.
    """
    ar = active_returns(returns, benchmark)
    active = ar["active_return"].to_numpy()
    cum = np.cumprod(1.0 + active)
    running_max = np.maximum.accumulate(cum)
    dd = cum / running_max - 1.0
    return ar.select("date").with_columns(pl.Series("rel_drawdown", dd))


def benchmark_returns_to_fractional(benchmark: pl.DataFrame) -> pl.DataFrame:
    """Convert benchmark `return` column from PERCENT to fractional.

    `benchmark_returns` from the dataset layer stores returns in percent
    (matching the engine's `R / 100` convention). Call this helper once before
    passing to any function in this module so the unit contract is satisfied.

    Input columns: `(date, benchmark_id, return)`.
    Output columns: `(date, return_1d)` — one benchmark only; caller must
    filter to the desired `benchmark_id` first.
    """
    return benchmark.with_columns((pl.col("return") / 100.0).alias("return_1d")).select(
        "date", "return_1d"
    )
