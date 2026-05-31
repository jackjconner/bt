"""Turnover-aware signal scoring.

A signal that changes its cross-sectional ranking every day has high
turnover and will eat its own IC in transaction costs.  This module measures:

1. **Signal autocorrelation**: the Pearson correlation of each asset's signal
   rank today vs its rank yesterday, averaged across assets and dates.  A value
   near 1 means the signal barely turns over; near 0 means it reshuffles every
   day.

2. **Rank stability**: fraction of assets whose top-/bottom-quintile
   membership is unchanged from one day to the next.  Complements
   autocorrelation with an interpretable bucket-level view.

3. **IC-IR net of turnover drag**: adjusts the IC-IR downward by an estimate
   of how much of the gross IC is consumed by transaction costs.  The model is:
       net_IC = gross_IC − 2 * cost_bps * (1 − autocorr)
   where ``cost_bps`` is the assumed one-way round-trip cost in the same units
   as the forward return (typically percent), and ``(1 − autocorr)`` is the
   fraction of the book that turns over each period.

The autocorrelation-based drag is a linear approximation; it gives a useful
sanity-check rather than a precise cost model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from etl.source import to_matrix


@dataclass(frozen=True)
class TurnoverResult:
    """Signal turnover statistics."""

    autocorr: float
    """Mean lag-1 cross-sectional rank autocorrelation (0 = full turnover, 1 = static)."""

    rank_stability: float
    """Fraction of assets keeping their top/bottom quintile membership day-to-day."""

    ic_ir_gross: float
    """Raw IC-IR before cost adjustment."""

    ic_ir_net: float
    """IC-IR after subtracting estimated turnover cost drag."""

    estimated_cost_drag: float
    """Estimated per-period IC drag from turnover costs."""

    n_dates: int


def signal_autocorr(
    signals: pl.DataFrame,
    *,
    signal_col: str = "signal",
) -> float:
    """Compute mean lag-1 cross-sectional rank autocorrelation of the signal.

    Ranks the signal cross-sectionally each date, then computes the Pearson
    correlation of rank[t] vs rank[t-1] across assets.  Averages across all
    consecutive date pairs.

    A high value (> 0.9) indicates low daily turnover; a low value (< 0.5)
    indicates the signal reshuffles most of its ordering each day.
    """
    from scipy import stats as scipy_stats

    S, _dates = to_matrix(signals.select("date", "id", signal_col), signal_col)
    n_dates = S.shape[0]

    corrs: list[float] = []
    for t in range(1, n_dates):
        x = S[t - 1]
        y = S[t]
        # Pairwise-complete finite values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            continue
        rx = scipy_stats.rankdata(x[mask])
        ry = scipy_stats.rankdata(y[mask])
        c = float(scipy_stats.pearsonr(rx, ry).statistic)
        corrs.append(c)

    return float(np.mean(corrs)) if corrs else float("nan")


def rank_stability(
    signals: pl.DataFrame,
    *,
    signal_col: str = "signal",
    n_quantiles: int = 5,
) -> float:
    """Fraction of assets retaining their extreme-quantile bucket day-to-day.

    Only assets in the top or bottom quantile bucket on day t are considered.
    Returns the fraction that remain in the *same* extreme bucket on day t+1.
    """
    from scipy import stats as scipy_stats

    S, _dates = to_matrix(signals.select("date", "id", signal_col), signal_col)
    n_dates, _n_assets = S.shape

    stable_counts: list[float] = []
    for t in range(1, n_dates):
        x0 = S[t - 1]
        x1 = S[t]
        mask = np.isfinite(x0) & np.isfinite(x1)
        if mask.sum() < n_quantiles * 2:
            continue
        xm0, xm1 = x0[mask], x1[mask]
        n = len(xm0)

        r0 = np.ceil(scipy_stats.rankdata(xm0) / n * n_quantiles).astype(int)
        r1 = np.ceil(scipy_stats.rankdata(xm1) / n * n_quantiles).astype(int)
        r0 = np.clip(r0, 1, n_quantiles)
        r1 = np.clip(r1, 1, n_quantiles)

        extreme_mask = (r0 == 1) | (r0 == n_quantiles)
        if extreme_mask.sum() == 0:
            continue
        same = ((r1 == r0) & extreme_mask).sum()
        stable_counts.append(same / extreme_mask.sum())

    return float(np.mean(stable_counts)) if stable_counts else float("nan")


def turnover_score(
    signals: pl.DataFrame,
    ic_ir_gross: float,
    *,
    signal_col: str = "signal",
    cost_bps: float = 10.0,
    n_quantiles: int = 5,
) -> TurnoverResult:
    """Compute autocorrelation, rank stability, and net IC-IR.

    Parameters
    ----------
    signals:
        Long-format (date, id, signal_col).
    ic_ir_gross:
        The gross IC-IR (before cost adjustment) from the IC analysis.
    signal_col:
        Column in ``signals`` holding the signal value.
    cost_bps:
        Assumed one-way round-trip transaction cost in the same percent units
        as forward returns.  Default 10 bps = 0.10%.
    n_quantiles:
        Number of quantile buckets for rank-stability computation.

    Returns
    -------
    TurnoverResult
    """
    S, _dates = to_matrix(signals.select("date", "id", signal_col), signal_col)
    n_dates = S.shape[0]

    autocorr = signal_autocorr(signals, signal_col=signal_col)
    rs = rank_stability(signals, signal_col=signal_col, n_quantiles=n_quantiles)

    # Linear drag model: each period we turn over (1-autocorr) of the book,
    # costing 2 * cost_bps (buy + sell legs).  The drag is on gross IC per
    # period, expressed in the same units.
    cost_pct = cost_bps / 100.0
    estimated_drag = 2.0 * cost_pct * (1.0 - autocorr) if np.isfinite(autocorr) else 0.0
    ic_ir_net = ic_ir_gross - estimated_drag

    return TurnoverResult(
        autocorr=autocorr,
        rank_stability=rs,
        ic_ir_gross=ic_ir_gross,
        ic_ir_net=ic_ir_net,
        estimated_cost_drag=estimated_drag,
        n_dates=n_dates,
    )
