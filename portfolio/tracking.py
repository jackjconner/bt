"""Ex-ante tracking error and active-weight metrics.

Ex-ante tracking error (TE) is the expected annualized volatility of the
active return (portfolio return minus benchmark return). Given covariance Σ:

    TE = sqrt((w − b)ᵀ Σ (w − b))

For w == b the active weight vector is zero, so TE is exactly 0.

Annualization: multiply the per-period TE by sqrt(periods_per_year).
The default 252 assumes a daily covariance matrix.
"""

from __future__ import annotations

import numpy as np


def tracking_error(
    weights: np.ndarray,
    benchmark: np.ndarray,
    cov: np.ndarray,
) -> float:
    """Ex-ante tracking error: sqrt((w−b)ᵀ Σ (w−b)).

    Args:
        weights:   (n_assets,) portfolio weight vector.
        benchmark: (n_assets,) benchmark weight vector.
        cov:       (n_assets, n_assets) asset covariance matrix.

    Returns:
        Non-negative tracking error in the same units as the covariance.
        If cov is a daily covariance, the result is daily TE; multiply by
        sqrt(252) to annualise.
    """
    active = weights - benchmark
    var = float(active @ cov @ active)
    return float(np.sqrt(max(var, 0.0)))


def information_ratio(
    active_returns: np.ndarray,
    periods_per_year: float = 252.0,
) -> float:
    """Realised information ratio from a series of active returns.

    IR = annualised mean active return / annualised active return vol.
    Returns NaN when there are fewer than 2 observations.

    Args:
        active_returns:    1-D array of (portfolio − benchmark) return per period.
        periods_per_year:  Trading periods per year for annualisation.
    """
    if len(active_returns) < 2:
        return float("nan")
    mu = np.mean(active_returns)
    sigma = np.std(active_returns, ddof=1)
    if sigma < 1e-15:
        return float("nan")
    return float(mu / sigma * np.sqrt(periods_per_year))
