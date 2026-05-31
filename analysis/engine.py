"""Fused single-pass metrics engine.

The per-metric functions in ``metrics.py``, ``risk.py``, and ``benchmark.py``
are each correct in isolation but, called as a suite, they re-walk the same
return series many times: ``analyze`` alone derives returns, then takes a
separate ``.std()``/``.mean()`` for annualization, a *third* pass inside
``sharpe``, re-reads NAV in ``max_drawdown`` (recomputing a drawdown series the
caller already built), reads NAV again in ``cagr``, and converts to numpy a
fourth time in ``sortino``. The benchmark suite (``alpha``/``beta``/
``information_ratio``/``tracking_error``) each independently inner-join the
strategy and benchmark frames and re-derive the same shared moments.

This module fuses each cluster into **one** pass over the data:

- ``analyze_fused`` derives the returns and drawdown columns from NAV in a single
  ``with_columns`` and reduces the returns to one aggregate row (mean,
  sample-std, downside count and mean-square), then reconstitutes every scalar
  ``analyze`` reports — sharpe, max-drawdown, annualized return/vol, CAGR,
  Sortino — from those shared moments plus the terminal/initial NAV ratio.
- ``benchmark_metrics_fused`` performs a single inner-join and a single aggregate
  over the aligned ``(r, b)`` pair, then closes ``beta``/``alpha``/
  ``tracking_error``/``information_ratio`` from the shared co-moments.

These are **exact** reimplementations: every reported number is bit-for-bit the
same as the metric-by-metric path (the same ddof conventions, the same
annualization, the same degenerate-case guards). The scalar functions remain the
public contract and the source of truth; the fused engine is an additive fast
path the production analyzer opts into, and the equivalence is pinned by tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from etl.source import to_float

if TYPE_CHECKING:
    from .metrics import AnalysisResult

TRADING_DAYS = 252
_SQRT_TRADING_DAYS = TRADING_DAYS**0.5


def _return_aggregate(returns: pl.DataFrame, rf_daily: float = 0.0) -> pl.DataFrame:
    """One grouped reduction yielding the moments ``analyze`` needs.

    ``return_1d`` is reduced to count / mean / sample-std / downside count and
    mean-square in a *single* pass over the column; the caller reads the scalars
    off the resulting one-row frame instead of taking a separate ``.mean()`` /
    ``.std()`` / downside pass per metric. The downside aggregate is taken over
    the *excess* return ``r - rf_daily`` relative to target 0, matching
    ``risk.sortino``'s ``downside[downside < 0]`` for any risk-free rate.
    """
    r = pl.col("return_1d")
    excess = r - rf_daily
    is_down = excess < 0
    neg_sq = pl.when(is_down).then(excess * excess).otherwise(0.0)
    return returns.select(
        r.mean().alias("mean"),
        r.std().alias("std1"),
        is_down.sum().alias("n_down"),
        neg_sq.sum().alias("down_sq_sum"),
    )


def analyze_fused(
    nav: pl.DataFrame,
    risk_free: float = 0.0,
) -> AnalysisResult:
    """Compute the full ``AnalysisResult`` in fused passes over the NAV frame.

    Exactly reproduces ``BacktestAnalyzerImpl.analyze`` (and the scalar helpers
    it composes) while walking the data far fewer times:

    - the returns and drawdown columns are derived from NAV in one
      ``with_columns`` (NAV is walked once instead of once per series);
    - the mean / sample-std / downside count and mean-square that feed
      ``annualized_vol``, ``annualized_return``, ``sharpe`` and ``sortino`` come
      from a single aggregate over the returns instead of a separate pass each;
    - ``max_drawdown`` is read off the drawdown column rather than recomputed
      from NAV;
    - ``cagr`` reuses the terminal/initial NAV ratio.

    ``risk_free`` is a scalar **annual** rate, matching ``BacktestAnalyzerImpl``;
    Sharpe divides it by ``TRADING_DAYS`` internally and Sortino uses the daily
    rate, exactly as the scalar path does.
    """
    from .metrics import AnalysisResult

    # Returns and drawdown share one walk over NAV.
    enriched = nav.with_columns(
        (pl.col("nav") / pl.col("nav").shift(1) - 1.0).alias("return_1d"),
        (pl.col("nav") / pl.col("nav").cum_max() - 1.0).alias("drawdown"),
    )
    dd = enriched.select("date", "drawdown")
    rets = enriched.drop_nulls(subset=["return_1d"]).select("date", "return_1d")

    rf_daily = risk_free / TRADING_DAYS
    agg = _return_aggregate(rets, rf_daily)
    mean = to_float(agg["mean"][0] or 0.0)
    std1 = agg["std1"][0]
    std1 = to_float(std1) if std1 is not None else 0.0
    down_sq_sum = to_float(agg["down_sq_sum"][0] or 0.0)
    n_down = int(agg["n_down"][0])

    ann_vol = std1 * _SQRT_TRADING_DAYS
    ann_ret = mean * TRADING_DAYS

    # Sharpe: excess = r - rf/TRADING_DAYS shifts the mean only; sample std is
    # invariant to the constant shift, so it equals std1 exactly.
    sharpe_val = 0.0 if std1 == 0.0 else (mean - rf_daily) / std1 * _SQRT_TRADING_DAYS

    # max drawdown: min of the shared drawdown series.
    max_dd = to_float(dd["drawdown"].min())

    # cagr: terminal/initial NAV ratio over the row count.
    nav_vals = nav["nav"]
    n_nav = len(nav_vals)
    start = to_float(nav_vals.first())
    end = to_float(nav_vals.last())
    cagr_val = 0.0 if n_nav <= 0 or start <= 0 else (end / start) ** (TRADING_DAYS / n_nav) - 1.0

    # sortino: rf passed to BacktestAnalyzerImpl is risk_free/TRADING_DAYS as the
    # daily excess rate, with target=0 (downside relative to 0).
    sortino_val = _sortino_from_moments(mean, rf_daily, down_sq_sum, n_down)

    return AnalysisResult(
        returns_series=rets,
        drawdown_series=dd,
        sharpe=sharpe_val,
        max_drawdown=max_dd,
        annualized_return=ann_ret,
        annualized_vol=ann_vol,
        cagr=cagr_val,
        sortino=sortino_val,
    )


def _sortino_from_moments(
    mean: float,
    rf_daily: float,
    down_sq_sum: float,
    n_down: int,
) -> float:
    """Sortino closed from pre-aggregated downside moments.

    Mirrors ``risk.sortino`` exactly: excess = r - rf_daily (a constant shift,
    so mean_excess = mean - rf_daily); downside deviation is the RMS of the
    negative *excess* values relative to target=0. The squared-negative sum and
    count come from ``_return_aggregate`` over the same excess series, identical
    to ``risk.sortino``'s ``downside[downside < 0]``.
    """
    mean_excess = mean - rf_daily
    dd_vol = (down_sq_sum / n_down) ** 0.5 if n_down > 0 else 0.0
    if dd_vol == 0.0:
        return 0.0
    return mean_excess / dd_vol * _SQRT_TRADING_DAYS


# ---------------------------------------------------------------------------
# Fused benchmark-relative suite
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Benchmark-relative scalars computed in one fused pass.

    Field-for-field identical to calling ``benchmark.beta``, ``benchmark.alpha``,
    ``benchmark.tracking_error`` and ``benchmark.information_ratio`` separately,
    but derived from a single inner-join + one grouped reduction over the aligned
    strategy/benchmark pair instead of four independent joins.
    """

    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float


def benchmark_metrics_fused(
    returns: pl.DataFrame,
    benchmark: pl.DataFrame,
    rf: float = 0.0,
) -> BenchmarkMetrics:
    """Compute beta / alpha / tracking-error / information-ratio in one pass.

    The four scalar functions in ``benchmark.py`` each inner-join the strategy
    and benchmark frames and re-derive overlapping co-moments. This fuses them:
    one inner-join on ``date``, one grouped reduction yielding the count, the
    first and second moments of ``r`` and ``b``, their cross moment, and the
    first/second moments of the active difference ``r - b``. Every output is
    closed from those aggregates with the *exact* conventions of the scalar
    path:

    - ``beta``  — population (ddof=0) OLS slope of r on b, no risk-free.
    - ``alpha`` — population OLS, scalar ``rf`` subtracted from both legs,
      annualized intercept (``* TRADING_DAYS``).
    - ``tracking_error`` — population std (ddof=0) of ``r - b``, annualized.
    - ``information_ratio`` — annualized mean active return / tracking error.

    ``rf`` is a scalar fractional **daily** rate (the Series path of the scalar
    functions is not fused here; callers needing a time-varying rate use the
    scalar functions). The production analyzer and harness call with ``rf=0``.
    """
    joined = returns.join(benchmark, on="date", how="inner", suffix="_bmk")
    r = pl.col("return_1d")
    b = pl.col("return_1d_bmk")
    active = r - b
    agg = joined.select(
        r.mean().alias("r_mean"),
        b.mean().alias("b_mean"),
        (b * b).mean().alias("b_sq"),
        (r * b).mean().alias("rb"),
        active.mean().alias("a_mean"),
        (active * active).mean().alias("a_sq"),
    )

    r_mean = to_float(agg["r_mean"][0])
    b_mean = to_float(agg["b_mean"][0])
    b_sq = to_float(agg["b_sq"][0])
    rb = to_float(agg["rb"][0])
    a_mean = to_float(agg["a_mean"][0])
    a_sq = to_float(agg["a_sq"][0])

    # beta: population covariance / population variance of the benchmark.
    #   cov = E[r b] - E[r] E[b];  var_b = E[b^2] - E[b]^2  (both ddof=0).
    b_var = b_sq - b_mean * b_mean
    beta_val = 0.0 if b_var == 0.0 else (rb - r_mean * b_mean) / b_var

    # alpha: same population OLS but on excess legs r-rf, b-rf. A common constant
    # rf shifts both means by rf and leaves the centered cross/variance moments
    # unchanged, so the slope equals beta computed on the excess legs; the
    # intercept is (r_mean - rf) - slope * (b_mean - rf), annualized.
    b_exc_mean = b_mean - rf
    b_exc_var = b_var  # variance is shift-invariant
    if b_exc_var == 0.0:
        alpha_val = 0.0
    else:
        slope_exc = (rb - r_mean * b_mean) / b_exc_var
        a_daily = (r_mean - rf) - slope_exc * b_exc_mean
        alpha_val = a_daily * TRADING_DAYS

    # tracking error: population std (ddof=0) of the active series, annualized.
    a_var = a_sq - a_mean * a_mean
    a_var = a_var if a_var > 0.0 else 0.0
    te = a_var**0.5 * _SQRT_TRADING_DAYS

    ir = 0.0 if te == 0.0 else a_mean * TRADING_DAYS / te

    return BenchmarkMetrics(
        beta=beta_val,
        alpha=alpha_val,
        tracking_error=te,
        information_ratio=ir,
    )
