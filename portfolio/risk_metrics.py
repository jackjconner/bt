"""Parametric (Gaussian) VaR and CVaR alongside the existing historical VaR.

Parametric VaR assumes portfolio returns are normally distributed with mean μ
and standard deviation σ derived from the covariance matrix. This is fast and
produces smooth risk estimates across time but underestimates tail risk when
returns are fat-tailed.

CVaR (Conditional Value at Risk / Expected Shortfall) is the expected loss
given that we are in the tail beyond VaR. For a Gaussian it has the closed form:

    CVaR_α = −(μ − σ · φ(z_α) / (1 − α))

where φ is the standard normal PDF, z_α = Φ⁻¹(1 − α), and Φ is the CDF.

Horizon scaling: for an h-period horizon under i.i.d. returns,
    μ_h = h · μ_1,   σ_h = sqrt(h) · σ_1.

Multi-horizon VaR/CVaR tables are useful for risk reporting (1d, 5d, 21d).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def parametric_var(
    weights: np.ndarray,
    cov: np.ndarray,
    confidence: float = 0.95,
    mean: np.ndarray | None = None,
    horizon: int = 1,
) -> float:
    """Gaussian parametric VaR at given confidence and horizon.

    Args:
        weights:    (n_assets,) portfolio weight vector.
        cov:        (n_assets, n_assets) per-period covariance.
        confidence: 1 − α, e.g. 0.95 for 95% VaR.
        mean:       (n_assets,) expected return vector. Defaults to zero.
        horizon:    Number of periods. Scales μ linearly and σ by sqrt.

    Returns:
        VaR as a positive loss (i.e. the threshold below which losses exceed
        the VaR level). Multiply by portfolio value for dollar VaR.
    """
    mu_p = 0.0 if mean is None else float(weights @ mean)

    var_p = float(weights @ cov @ weights)
    sigma_p = np.sqrt(max(var_p, 0.0))

    mu_h = horizon * mu_p
    sigma_h = np.sqrt(horizon) * sigma_p

    z = norm.ppf(1.0 - confidence)
    # portfolio return quantile at the loss tail
    ret_quantile = mu_h + z * sigma_h
    return float(-ret_quantile)  # positive = loss


def parametric_cvar(
    weights: np.ndarray,
    cov: np.ndarray,
    confidence: float = 0.95,
    mean: np.ndarray | None = None,
    horizon: int = 1,
) -> float:
    """Gaussian parametric CVaR (Expected Shortfall) at given confidence.

    For a normal N(μ_h, σ_h²), the expected shortfall at level α is:
        ES_α = −μ_h + σ_h · φ(Φ⁻¹(1−α)) / (1−α)

    This is always ≥ parametric_var at the same confidence.

    Args:
        weights:    (n_assets,) portfolio weight vector.
        cov:        (n_assets, n_assets) per-period covariance.
        confidence: 1 − α, e.g. 0.95.
        mean:       (n_assets,) expected return vector. Defaults to zero.
        horizon:    Number of periods.

    Returns:
        CVaR as a positive expected loss in the tail.
    """
    mu_p = 0.0 if mean is None else float(weights @ mean)

    var_p = float(weights @ cov @ weights)
    sigma_p = np.sqrt(max(var_p, 0.0))

    mu_h = horizon * mu_p
    sigma_h = np.sqrt(horizon) * sigma_p

    alpha = 1.0 - confidence  # tail probability
    z_alpha = norm.ppf(alpha)  # negative quantile
    phi_z = norm.pdf(z_alpha)  # PDF at that point
    # ES = −μ_h + σ_h · φ(z_α) / α
    es = -mu_h + sigma_h * phi_z / alpha
    return float(es)


def var_cvar_table(
    weights: np.ndarray,
    cov: np.ndarray,
    confidences: tuple[float, ...] = (0.95, 0.99),
    horizons: tuple[int, ...] = (1, 5, 21),
    mean: np.ndarray | None = None,
) -> list[dict]:
    """Compute parametric VaR and CVaR for a grid of confidence × horizon.

    Returns a list of dicts with keys: confidence, horizon, var, cvar.
    Useful for risk reporting and sanity checks.
    """
    rows = []
    for c in confidences:
        for h in horizons:
            rows.append(
                {
                    "confidence": c,
                    "horizon": h,
                    "var": parametric_var(weights, cov, c, mean, h),
                    "cvar": parametric_cvar(weights, cov, c, mean, h),
                }
            )
    return rows
