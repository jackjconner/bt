"""Covariance estimators: sample, EWMA, and Ledoit-Wolf shrinkage.

Why multiple estimators:
- Sample covariance is ill-conditioned when n_assets approaches or exceeds
  the window length; eigenvalues are noisy and inversion amplifies errors.
- EWMA downweights stale observations, which captures volatility clustering
  (returns are heteroskedastic) at the cost of a single tuning parameter.
- Ledoit-Wolf shrinks the sample matrix analytically toward a scaled identity,
  minimising Frobenius-norm estimation error without extra tuning.
"""

from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def sample_cov(returns: np.ndarray) -> np.ndarray:
    """Standard sample covariance. (n_assets, n_assets). O(T * n²)."""
    return np.cov(returns.T, ddof=1)


def ewma_cov(returns: np.ndarray, halflife: float) -> np.ndarray:
    """Exponentially weighted covariance matrix.

    Each observation is weighted by λ^(T-1-t) where λ = 0.5^(1/halflife).
    The mean is subtracted using the same weights before computing the outer
    products — this keeps the estimator consistent with the weighted variance.

    Args:
        returns: (T, n_assets) return matrix.
        halflife: Decay half-life in periods. Smaller → more weight on recent.
    """
    t, _n = returns.shape
    lam = 0.5 ** (1.0 / halflife)
    # weights in chronological order: oldest = lam^(T-1), newest = 1
    w = lam ** np.arange(t - 1, -1, -1, dtype=float)
    w /= w.sum()

    mu = (w[:, None] * returns).sum(axis=0)
    excess = returns - mu
    # weighted outer-product sum: Σ w_t * (r_t - μ)(r_t - μ)ᵀ
    return (w[:, None] * excess).T @ excess


def ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage estimator.

    Shrinks the sample covariance toward a scaled identity matrix using the
    analytically optimal shrinkage coefficient (Oracle Approximating Shrinkage,
    Chen-Ledoit-Wolf formula). Relies on sklearn's well-tested implementation.

    The resulting matrix is guaranteed positive-definite for T > 1.
    """
    lw = LedoitWolf(assume_centered=False)
    lw.fit(returns)
    return lw.covariance_
