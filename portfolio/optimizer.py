"""Mean-variance portfolio optimizer.

Objective (maximise):
    αᵀ w  −  λ · wᵀ Σ w  −  κ · Σ_i tc_i · |w_i − w0_i|

where α is the expected alpha vector, Σ is the asset covariance, λ is the
risk-aversion coefficient, κ scales a per-asset transaction cost penalty, and
w0 is the current (pre-rebalance) weight vector.

Implementation uses scipy.optimize.minimize with SLSQP (Sequential Least
Squares Programming), which handles quadratic objectives and non-linear
constraints efficiently. SLSQP requires the gradient (jac) for speed and
accuracy; we provide analytic gradients for the quadratic terms.

No-trade bands: if `no_trade_band` > 0, weights within the band of w0 are
treated as a feasible solution during a pre-check; only assets outside the
band enter the cost penalty. This models the common "don't trade small drifts"
rebalance rule. The full optimization still runs; the band widens via the cost
penalty.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .constraints import ConstraintSpec


@dataclass(frozen=True)
class OptimizeResult:
    """Outcome of a single mean-variance optimization.

    Attributes:
        weights:    (n_assets,) optimal weight vector.
        converged:  Whether SLSQP reported convergence (status == 0).
        message:    Solver message (useful for diagnosing infeasibility).
        obj_value:  Final objective (negated, so higher = better alpha-λvar).
        n_iter:     Number of solver iterations consumed.
    """

    weights: np.ndarray
    converged: bool
    message: str
    obj_value: float
    n_iter: int


def _equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def mean_variance(
    alpha: np.ndarray,
    cov: np.ndarray,
    spec: ConstraintSpec,
    risk_aversion: float = 1.0,
    w0: np.ndarray | None = None,
    cost_per_unit: np.ndarray | None = None,
    cost_scale: float = 1.0,
    no_trade_band: float = 0.0,
    max_iter: int = 500,
) -> OptimizeResult:
    """Solve the mean-variance allocation problem via SLSQP.

    Args:
        alpha:        (n_assets,) expected return / alpha vector.
        cov:          (n_assets, n_assets) asset covariance matrix.
        spec:         Constraint specification (bounds + linear constraints).
        risk_aversion: λ — scales the variance penalty relative to alpha.
        w0:           (n_assets,) current weights (default: equal-weight).
                      Used for the turnover/cost penalty term.
        cost_per_unit: Per-asset cost per unit of turnover (bps or fraction).
                       Shape (n_assets,). If None and cost_scale > 0, uniform.
        cost_scale:   κ — overall cost penalty multiplier. Set to 0 to disable.
        no_trade_band: Half-width of the no-trade band around w0. Positions
                       within ±no_trade_band of w0 incur no cost penalty.
        max_iter:     Maximum SLSQP iterations.

    Returns:
        OptimizeResult with the optimal weights and solver metadata.
    """
    n = spec.n_assets
    if w0 is None:
        w0 = _equal_weight(n)
    if cost_per_unit is None:
        cost_per_unit = np.ones(n)

    cp = np.asarray(cost_per_unit, dtype=float)

    # Precompute 2Σ for the gradient of the variance term
    two_cov = 2.0 * cov

    def objective(w: np.ndarray) -> float:
        var = float(w @ cov @ w)
        alpha_ret = float(alpha @ w)
        if cost_scale > 0.0:
            delta = np.abs(w - w0)
            if no_trade_band > 0.0:
                delta = np.maximum(delta - no_trade_band, 0.0)
            cost = cost_scale * float(cp @ delta)
        else:
            cost = 0.0
        # Negate because minimize() minimises; we maximise alpha − λ·var − cost
        return -(alpha_ret - risk_aversion * var - cost)

    def jac(w: np.ndarray) -> np.ndarray:
        grad_alpha = alpha
        grad_var = two_cov @ w
        if cost_scale > 0.0:
            delta = w - w0
            if no_trade_band > 0.0:
                # gradient of max(|delta| - band, 0): sign(delta) where |delta|>band
                sign_delta = np.sign(delta)
                in_band = np.abs(delta) <= no_trade_band
                sign_delta[in_band] = 0.0
            else:
                sign_delta = np.sign(delta)
            grad_cost = cost_scale * cp * sign_delta
        else:
            grad_cost = 0.0
        return -(grad_alpha - risk_aversion * grad_var - grad_cost)

    bounds = spec.per_asset_bounds()
    constraints = spec.scipy_constraints()

    # Warm start: project equal-weight toward per-asset bounds
    w_init = np.clip(w0, [b[0] for b in bounds], [b[1] for b in bounds])
    # Renormalize to satisfy net_exposure equality
    s = w_init.sum()
    if abs(s) > 1e-12:
        w_init = w_init * spec.net_exposure / s
    else:
        w_init = _equal_weight(n) * spec.net_exposure

    res = minimize(
        objective,
        w_init,
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": max_iter, "ftol": 1e-9},
    )

    return OptimizeResult(
        weights=res.x,
        converged=res.status == 0,
        message=res.message,
        obj_value=-float(res.fun),
        n_iter=res.nit,
    )
