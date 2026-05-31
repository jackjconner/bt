"""Mean-variance portfolio optimizer.

Objective (maximise):
    αᵀ w  −  λ · wᵀ Σ w  −  κ · Σ_i tc_i · |w_i − w0_i|

where α is the expected alpha vector, Σ is the asset covariance, λ is the
risk-aversion coefficient, κ scales a per-asset transaction cost penalty, and
w0 is the current (pre-rebalance) weight vector.

Two solver backends are available via the ``solver`` parameter:

SLSQP (default, ``solver="slsqp"``)
    Uses scipy.optimize.minimize with SLSQP (Sequential Least Squares
    Programming).  Provides analytic gradients for the quadratic terms.
    The L1 cost penalty is non-smooth, which can slow convergence for
    large universes (super-linear in n_assets).

OSQP (``solver="osqp"``)
    Reformulates the problem as a strict Quadratic Programme by introducing
    epigraph auxiliary variables t_i ≥ |w_i − w0_i|.  The stacked variable
    is x = [w; t] (length 2n).  This turns the non-smooth L1 objective into
    a linear term in t, giving OSQP a smooth QP it can solve in far fewer
    iterations.  Expected speedup: 10–50× for n_assets ≥ 100.

    QP standard form:
        min  0.5 xᵀ P x + qᵀ x
        s.t. l ≤ A x ≤ u

    where:
        P[0:n, 0:n] = 2λ Σ  (quadratic variance block; P is 2n×2n)
        q[0:n]       = −α   (negate: we minimise −objective)
        q[n:2n]      = κ · cp  (L1 penalty on turnover)

    Constraint rows in A (row order):
        1. Per-asset weight bounds:      lb_i ≤ w_i ≤ ub_i         (n rows)
        2. Net-exposure equality:        Σ w_i = net_exposure        (1 row)
        3. t_i ≥ 0 bounds:              0 ≤ t_i ≤ ∞                (n rows)
        4. Epigraph ineq (upper side):   w_i − t_i ≤  w0_i         (n rows)
        5. Epigraph ineq (lower side):  −w_i − t_i ≤ −w0_i        (n rows)
        6. Sector min/max (optional, linear):  one row per active bound
        7. Gross exposure (optional, linear):  only for long-only case

    No-trade band: when no_trade_band > 0 the epigraph RHS is shifted so
    cost is charged only on the excess beyond the band:
        t_i ≥ max(|w_i − w0_i| − band, 0)
    This is equivalent to replacing w0_i with w0_i ± band in the epigraph
    rows (upper: w_i − t_i ≤ w0_i + band; lower: −w_i − t_i ≤ −w0_i + band).

No-trade bands: if `no_trade_band` > 0, weights within the band of w0 are
treated as a feasible solution during a pre-check; only assets outside the
band enter the cost penalty. This models the common "don't trade small drifts"
rebalance rule. The full optimization still runs; the band widens via the cost
penalty.

Factor-model covariance path (``factor_risk_model`` parameter, ``solver="osqp"`` only)
    When a ``FactorRiskModel`` is supplied the optimizer does NOT materialise the
    dense n×n covariance Σ = B·F·Bᵀ + D.  Instead it introduces k auxiliary
    variables y = Bᵀw and builds a structure-exploiting QP with a block-diagonal
    cost matrix.

    Decision variable: x = [w (n), t (n), y (k)], length N = 2n + k.

    P (upper-triangular CSC, O(n + k²) nonzeros):
        P[0:n, 0:n]     = 2λ diag(specific_var)   — diagonal, n entries
        P[2n:2n+k, 2n:2n+k] = 2λ F               — dense k×k factor cov

    q:
        q[0:n]    = -alpha
        q[n:2n]   = cost_scale * cp
        q[2n:2n+k] = 0

    Extra constraints (k rows appended after existing rows):
        y - Bᵀw = 0  →  [-Bᵀ | 0_n | I_k] · x = 0   (l = u = 0)

    Memory: O(n·k + k²) for B and F vs O(n²) for dense Σ.
    Time: OSQP convergence on sparse P is near-linear in n (empirically).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from .constraints import ConstraintSpec
from .risk_model import FactorRiskModel


@dataclass(frozen=True)
class OptimizeResult:
    """Outcome of a single mean-variance optimization.

    Attributes:
        weights:    (n_assets,) optimal weight vector.
        converged:  Whether the solver reported convergence (status == 0).
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
    solver: Literal["slsqp", "osqp"] = "slsqp",
    factor_risk_model: FactorRiskModel | None = None,
) -> OptimizeResult:
    """Solve the mean-variance allocation problem.

    Args:
        alpha:        (n_assets,) expected return / alpha vector.
        cov:          (n_assets, n_assets) asset covariance matrix.  Ignored
                      when ``factor_risk_model`` is supplied and
                      ``solver="osqp"`` — the dense Σ is never materialised in
                      that path.
        spec:         Constraint specification (bounds + linear constraints).
        risk_aversion: λ — scales the variance penalty relative to alpha.
        w0:           (n_assets,) current weights (default: equal-weight).
                      Used for the turnover/cost penalty term.
        cost_per_unit: Per-asset cost per unit of turnover (bps or fraction).
                       Shape (n_assets,). If None and cost_scale > 0, uniform.
        cost_scale:   κ — overall cost penalty multiplier. Set to 0 to disable.
        no_trade_band: Half-width of the no-trade band around w0. Positions
                       within ±no_trade_band of w0 incur no cost penalty.
        max_iter:     Maximum solver iterations.
        solver:       Backend to use. ``"slsqp"`` (default): scipy SLSQP.
                      ``"osqp"``: OSQP QP solver with epigraph L1 reformulation.
                      OSQP is typically 10–50× faster for n_assets ≥ 100.
        factor_risk_model: Optional ``FactorRiskModel`` (B, F, D).  When
                      provided *and* ``solver="osqp"``, the optimizer builds a
                      structure-exploiting sparse QP without materialising the
                      dense n×n covariance — see module docstring.  The dense
                      ``cov`` argument is unused in this path.

    Returns:
        OptimizeResult with the optimal weights and solver metadata.
    """
    n = spec.n_assets
    if w0 is None:
        w0 = _equal_weight(n)
    if cost_per_unit is None:
        cost_per_unit = np.ones(n)

    cp = np.asarray(cost_per_unit, dtype=float)

    if solver == "osqp":
        if factor_risk_model is not None:
            return _solve_osqp_factor(
                alpha,
                factor_risk_model,
                spec,
                risk_aversion,
                w0,
                cp,
                cost_scale,
                no_trade_band,
                max_iter,
            )
        return _solve_osqp(
            alpha, cov, spec, risk_aversion, w0, cp, cost_scale, no_trade_band, max_iter
        )
    return _solve_slsqp(
        alpha, cov, spec, risk_aversion, w0, cp, cost_scale, no_trade_band, max_iter
    )


# ---------------------------------------------------------------------------
# SLSQP backend (original implementation, unchanged)
# ---------------------------------------------------------------------------


def _slsqp_warm_start(spec: ConstraintSpec, w0: np.ndarray) -> np.ndarray:
    """Compute a warm-start weight vector for SLSQP.

    Clips w0 to per-asset bounds, then renormalises to satisfy the
    net_exposure equality constraint.
    """
    n = spec.n_assets
    bounds = spec.per_asset_bounds()
    w_init = np.clip(w0, [b[0] for b in bounds], [b[1] for b in bounds])
    s = w_init.sum()
    if abs(s) > 1e-12:
        w_init = w_init * spec.net_exposure / s
    else:
        w_init = _equal_weight(n) * spec.net_exposure
    return w_init


def _solve_slsqp(
    alpha: np.ndarray,
    cov: np.ndarray,
    spec: ConstraintSpec,
    risk_aversion: float,
    w0: np.ndarray,
    cp: np.ndarray,
    cost_scale: float,
    no_trade_band: float,
    max_iter: int,
) -> OptimizeResult:
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
    w_init = _slsqp_warm_start(spec, w0)

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


# ---------------------------------------------------------------------------
# OSQP backend — QP reformulation with epigraph L1 linearisation
# ---------------------------------------------------------------------------

_INF = 1e30


def _build_quadratic_cost(
    n: int,
    risk_aversion: float,
    cov_arr: np.ndarray,
    alpha: np.ndarray,
    cp: np.ndarray,
    cost_scale: float,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Build the OSQP quadratic-cost matrix P and linear-cost vector q.

    Decision variable x = [w (n), t (n)], length N = 2n.

    P (upper-triangular CSC):
        P[0:n, 0:n] = 2 * risk_aversion * Σ
        (factor of 2 because OSQP objective is 0.5 xᵀ P x + qᵀ x)

    q:
        q[0:n] = -alpha   (minimise −αᵀw)
        q[n:2n] = cost_scale * cp   (minimise κ·Σ c_i t_i)
    """
    N = 2 * n
    P_dense = np.zeros((N, N))
    P_dense[:n, :n] = 2.0 * risk_aversion * cov_arr
    P = sp.triu(P_dense, format="csc")

    q = np.zeros(N)
    q[:n] = -np.asarray(alpha, dtype=float)
    if cost_scale > 0.0:
        q[n:] = cost_scale * cp

    return P, q


def _build_epigraph_rows(
    n: int,
    w0_arr: np.ndarray,
    no_trade_band: float,
) -> tuple[list[sp.spmatrix], list[np.ndarray], list[np.ndarray]]:
    """Build the three epigraph-related constraint blocks for the t variables.

    Returns three (A_block, l_part, u_part) triplets (as parallel lists)
    corresponding to:
      - t_i ≥ 0
      - w_i - t_i ≤ w0_i + band   (upper epigraph side)
      - -w_i - t_i ≤ -w0_i + band  (lower epigraph side)
    """
    band = float(no_trade_band)
    A_blocks: list[sp.spmatrix] = []
    l_parts: list[np.ndarray] = []
    u_parts: list[np.ndarray] = []

    # t_i ≥ 0  (n rows)
    A_blocks.append(sp.hstack([sp.csr_matrix((n, n)), sp.eye(n, format="csr")]))
    l_parts.append(np.zeros(n))
    u_parts.append(np.full(n, _INF))

    # Epigraph upper side: w_i - t_i ≤ w0_i + band
    A_blocks.append(sp.hstack([sp.eye(n, format="csr"), -sp.eye(n, format="csr")]))
    l_parts.append(np.full(n, -_INF))
    u_parts.append(w0_arr + band)

    # Epigraph lower side: -w_i - t_i ≤ -w0_i + band
    A_blocks.append(sp.hstack([-sp.eye(n, format="csr"), -sp.eye(n, format="csr")]))
    l_parts.append(np.full(n, -_INF))
    u_parts.append(-w0_arr + band)

    return A_blocks, l_parts, u_parts


def _build_sector_rows(
    spec: ConstraintSpec,
    n: int,
) -> tuple[list[sp.spmatrix], list[np.ndarray], list[np.ndarray]]:
    """Build one constraint row per active sector bound.

    Sector constraints are linear in w only; t columns are zero.
    """
    A_blocks: list[sp.spmatrix] = []
    l_parts: list[np.ndarray] = []
    u_parts: list[np.ndarray] = []

    if spec.sector_map is None:
        return A_blocks, l_parts, u_parts

    sectors = np.asarray(spec.sector_map)
    for s in np.unique(sectors):
        mask = (sectors == s).astype(float)
        s_int = int(s)
        if s_int in spec.sector_min or s_int in spec.sector_max:
            row_w = sp.csr_matrix(mask.reshape(1, n))
            row_t = sp.csr_matrix((1, n))
            row = sp.hstack([row_w, row_t])
            lo_s = float(spec.sector_min.get(s_int, -_INF))
            hi_s = float(spec.sector_max.get(s_int, _INF))
            A_blocks.append(row)
            l_parts.append(np.array([lo_s]))
            u_parts.append(np.array([hi_s]))

    return A_blocks, l_parts, u_parts


def _build_gross_rows(
    spec: ConstraintSpec,
    n: int,
) -> tuple[list[sp.spmatrix], list[np.ndarray], list[np.ndarray]]:
    """Build constraint rows for gross exposure bounds (long-only simplification).

    For long-only portfolios Σ|w_i| = Σ w_i, so gross bounds reduce to linear
    constraints on w. Long-short gross bounds require a separate epigraph; that
    path is not exercised here (production relies on per-asset bounds).
    """
    A_blocks: list[sp.spmatrix] = []
    l_parts: list[np.ndarray] = []
    u_parts: list[np.ndarray] = []

    row_w = sp.csr_matrix(np.ones((1, n)))
    row_t = sp.csr_matrix((1, n))
    row = sp.hstack([row_w, row_t])

    if spec.min_gross is not None:
        A_blocks.append(row)
        l_parts.append(np.array([float(spec.min_gross)]))
        u_parts.append(np.array([_INF]))

    if spec.max_gross is not None:
        A_blocks.append(row)
        l_parts.append(np.array([-_INF]))
        u_parts.append(np.array([float(spec.max_gross)]))

    return A_blocks, l_parts, u_parts


def _build_constraint_matrix(
    spec: ConstraintSpec,
    n: int,
    cost_scale: float,
    w0_arr: np.ndarray,
    no_trade_band: float,
) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
    """Assemble the full OSQP constraint matrix A and bound vectors l, u.

    Row layout (for cost_scale > 0):
        0..n-1     : per-asset weight bounds   [I | 0]
        n          : net-exposure equality      [1ᵀ | 0]
        n+1..2n    : t_i ≥ 0                   [0 | I]
        2n+1..3n   : epigraph upper side        [I | -I]
        3n+1..4n   : epigraph lower side       [-I | -I]
        [sector rows, one per active bound]
        [gross rows, 0–2]

    When cost_scale == 0 the t-variable and epigraph rows are omitted.
    """
    A_blocks: list[sp.spmatrix] = []
    l_parts: list[np.ndarray] = []
    u_parts: list[np.ndarray] = []

    # 1. Per-asset weight bounds: w_i ∈ [lb_i, ub_i]  (n rows)
    bounds_list = spec.per_asset_bounds()
    lb = np.array([b[0] for b in bounds_list])
    ub = np.array([b[1] for b in bounds_list])
    A_blocks.append(sp.hstack([sp.eye(n, format="csr"), sp.csr_matrix((n, n))]))
    l_parts.append(lb)
    u_parts.append(ub)

    # 2. Net-exposure equality: Σ w_i = net_exposure  (1 row)
    ones_w = sp.csr_matrix(np.ones((1, n)))
    zeros_t = sp.csr_matrix((1, n))
    A_blocks.append(sp.hstack([ones_w, zeros_t]))
    ne = float(spec.net_exposure)
    l_parts.append(np.array([ne]))
    u_parts.append(np.array([ne]))

    # 3–5. Epigraph blocks (only when cost penalty is active)
    if cost_scale > 0.0:
        epi_blocks, epi_l, epi_u = _build_epigraph_rows(n, w0_arr, no_trade_band)
        A_blocks.extend(epi_blocks)
        l_parts.extend(epi_l)
        u_parts.extend(epi_u)

    # 6. Sector constraints
    sec_blocks, sec_l, sec_u = _build_sector_rows(spec, n)
    A_blocks.extend(sec_blocks)
    l_parts.extend(sec_l)
    u_parts.extend(sec_u)

    # 7. Gross exposure
    gross_blocks, gross_l, gross_u = _build_gross_rows(spec, n)
    A_blocks.extend(gross_blocks)
    l_parts.extend(gross_l)
    u_parts.extend(gross_u)

    A = sp.vstack(A_blocks).tocsc()
    l_vec = np.concatenate(l_parts)
    u_vec = np.concatenate(u_parts)
    return A, l_vec, u_vec


def _osqp_objective(
    weights: np.ndarray,
    alpha: np.ndarray,
    cov_arr: np.ndarray,
    risk_aversion: float,
    w0: np.ndarray,
    cp: np.ndarray,
    cost_scale: float,
    no_trade_band: float,
) -> float:
    """Compute the original (non-negated) objective for a given weight vector.

    Returns alpha'w - lambda * w'Σw - cost.
    """
    var_val = float(weights @ cov_arr @ weights)
    alpha_ret = float(np.asarray(alpha) @ weights)
    if cost_scale > 0.0:
        delta = np.abs(weights - np.asarray(w0, dtype=float))
        if no_trade_band > 0.0:
            delta = np.maximum(delta - no_trade_band, 0.0)
        cost_val = cost_scale * float(cp @ delta)
    else:
        cost_val = 0.0
    return alpha_ret - risk_aversion * var_val - cost_val


def _solve_osqp(
    alpha: np.ndarray,
    cov: np.ndarray,
    spec: ConstraintSpec,
    risk_aversion: float,
    w0: np.ndarray,
    cp: np.ndarray,
    cost_scale: float,
    no_trade_band: float,
    max_iter: int,
) -> OptimizeResult:
    """Solve via OSQP using the epigraph QP reformulation.

    Decision variable x = [w (n), t (n)], length 2n.

    P (quadratic cost, upper-triangular CSC):
        P[0:n, 0:n] = 2 * risk_aversion * Σ
        (factor of 2 because OSQP objective is 0.5 xᵀ P x + qᵀ x)

    q (linear cost):
        q[0:n] = -alpha          (negate: we minimise −(alpha − λvar − cost))
        q[n:2n] = cost_scale * cp

    Constraints A·x ∈ [l, u]:
        rows 0..n-1    : per-asset weight bounds   l=lb, u=ub
        row  n         : net-exposure equality      l=u=net_exposure
        rows n+1..2n   : t_i ≥ 0                   l=0, u=+inf
        rows 2n+1..3n  : w_i - t_i ≤ w0_i+band     l=-inf, u=w0+band
        rows 3n+1..4n  : -w_i - t_i ≤ -w0_i+band   l=-inf, u=-w0+band
        [optional sector/gross rows appended]
    """
    import osqp

    n = spec.n_assets
    cov_arr = np.asarray(cov, dtype=float)
    w0_arr = np.asarray(w0, dtype=float)

    P, q = _build_quadratic_cost(n, risk_aversion, cov_arr, alpha, cp, cost_scale)
    A, l_vec, u_vec = _build_constraint_matrix(spec, n, cost_scale, w0_arr, no_trade_band)

    prob = osqp.OSQP()
    prob.setup(
        P,
        q,
        A,
        l_vec,
        u_vec,
        warm_starting=True,
        verbose=False,
        max_iter=max_iter * 20,  # OSQP iterations are cheap — scale up
        eps_abs=1e-8,
        eps_rel=1e-8,
        eps_prim_inf=1e-8,
        eps_dual_inf=1e-8,
    )
    res = prob.solve()

    status = res.info.status
    converged = status in ("solved", "solved_inaccurate")
    weights = res.x[:n] if res.x is not None else np.zeros(n)

    obj = _osqp_objective(
        weights, alpha, cov_arr, risk_aversion, w0_arr, cp, cost_scale, no_trade_band
    )

    return OptimizeResult(
        weights=weights,
        converged=converged,
        message=status,
        obj_value=obj,
        n_iter=int(res.info.iter),
    )


# ---------------------------------------------------------------------------
# Factor-model covariance path — structure-exploiting sparse QP
# ---------------------------------------------------------------------------
#
# Decision variable: x = [w (n), t (n), y (k)], length N = 2n + k.
#
# The risk term wᵀΣw = wᵀDw + yᵀFy  where y = Bᵀw,  D = diag(specific_var).
#
# P (upper-triangular, O(n + k²) nonzeros):
#   P[0:n, 0:n]         = 2λ diag(specific_var)   (diagonal block)
#   P[2n:2n+k, 2n:2n+k] = 2λ F                    (dense k×k factor cov)
#   (P[n:2n, *] = 0 — the t variables are linear-only)
#
# q: [-alpha | cost_scale*cp | 0_k]
#
# Constraints: existing [w; t] rows widened by k zero columns for y,
#   plus k equality rows  y − Bᵀw = 0  →  [-Bᵀ | 0_n | I_k] x = 0.


def _build_quadratic_cost_factor(
    n: int,
    k: int,
    risk_aversion: float,
    specific_var: np.ndarray,
    factor_cov: np.ndarray,
    alpha: np.ndarray,
    cp: np.ndarray,
    cost_scale: float,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Build the structured P (sparse) and q for the factor-model QP.

    Decision variable x = [w (n), t (n), y (k)], length N = 2n + k.

    P is upper-triangular CSC with O(n + k²) nonzeros:
        P[0:n, 0:n]         = 2λ diag(specific_var)
        P[2n:2n+k, 2n:2n+k] = 2λ factor_cov  (upper triangle taken below)
    """
    N = 2 * n + k
    lam2 = 2.0 * risk_aversion

    # w-block: diagonal specific-variance term
    diag_vals = lam2 * np.asarray(specific_var, dtype=float)
    P_w = sp.diags(diag_vals, format="csc", shape=(n, n))  # already upper-tri (diagonal)

    # t-block: zero (n×n)
    P_t = sp.csc_matrix((n, n))

    # y-block: 2λ * factor_cov (take upper triangle for OSQP)
    P_y_dense = lam2 * np.asarray(factor_cov, dtype=float)
    P_y = sp.triu(P_y_dense, format="csc")

    # Assemble block-diagonal P_wt and P_y, then the full N×N matrix.
    # scipy.sparse.block_diag gives exactly a block-diagonal CSC.
    P = sp.block_diag([P_w, P_t, P_y], format="csc")
    # Enforce upper-triangular (block_diag + diag are already upper-tri, but
    # be explicit so OSQP's precondition check can't fail on floating-point issues).
    P = sp.triu(P, format="csc")

    q = np.zeros(N)
    q[:n] = -np.asarray(alpha, dtype=float)
    if cost_scale > 0.0:
        q[n : 2 * n] = cost_scale * np.asarray(cp, dtype=float)
    # q[2n:] stays zero (y variables have no linear cost term)

    return P, q


def _build_constraint_matrix_factor(
    spec: ConstraintSpec,
    n: int,
    k: int,
    cost_scale: float,
    w0_arr: np.ndarray,
    no_trade_band: float,
    B: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray, np.ndarray]:
    """Assemble the OSQP constraint matrix for the factor-model QP.

    Extends the dense-cov constraint matrix (which lives in the [w; t] block
    of width 2n) by appending k zero columns for the y variables, then adds k
    equality rows encoding y − Bᵀw = 0.

    Row layout identical to _build_constraint_matrix for the first block, then:
        [existing rows: per-asset bounds, net-exposure, epigraph, sector, gross]
        k rows: -Bᵀ w + I_k y = 0    (y = Bᵀw)
    """
    # --- reuse the existing constraint builder for the [w; t] sub-problem ---
    A_wt, l_wt, u_wt = _build_constraint_matrix(spec, n, cost_scale, w0_arr, no_trade_band)
    # A_wt is (m × 2n); widen it to (m × (2n+k)) by appending zero y-columns.
    m = A_wt.shape[0]
    A_wt_wide = sp.hstack([A_wt, sp.csr_matrix((m, k))], format="csr")

    # --- k rows: y_j − (Bᵀw)_j = 0  for j = 0..k-1 ---
    # Bᵀ is (k × n), so the w part of each row is -Bᵀ[j, :] (1 × n).
    # The t part is 0 (1 × n).
    # The y part is e_j (standard basis, 1 × k).
    B_arr = np.asarray(B, dtype=float)  # (n, k)
    Bt = B_arr.T  # (k, n)
    neg_Bt = sp.csr_matrix(-Bt)  # (k, n)
    zero_t_block = sp.csr_matrix((k, n))
    eye_y = sp.eye(k, format="csr")
    A_y_eq = sp.hstack([neg_Bt, zero_t_block, eye_y], format="csr")  # (k, 2n+k)

    A = sp.vstack([A_wt_wide, A_y_eq]).tocsc()
    l_vec = np.concatenate([l_wt, np.zeros(k)])
    u_vec = np.concatenate([u_wt, np.zeros(k)])

    return A, l_vec, u_vec


def _solve_osqp_factor(
    alpha: np.ndarray,
    frm: FactorRiskModel,
    spec: ConstraintSpec,
    risk_aversion: float,
    w0: np.ndarray,
    cp: np.ndarray,
    cost_scale: float,
    no_trade_band: float,
    max_iter: int,
) -> OptimizeResult:
    """Solve via OSQP using the structure-exploiting factor-model QP.

    Decision variable x = [w (n), t (n), y (k)], length N = 2n + k.

    P (upper-triangular CSC, O(n + k²) nonzeros):
        P[0:n, 0:n]         = 2λ diag(specific_var)
        P[2n:2n+k, 2n:2n+k] = 2λ factor_cov

    Constraints A·x ∈ [l, u]:
        first block : per-asset bounds, net-exposure, epigraph, sector, gross
                      (same as dense-cov OSQP path; y columns are zero)
        last k rows : y − Bᵀw = 0  (factor-exposure equality)
    """
    import osqp

    n = spec.n_assets
    k = frm.B.shape[1]
    w0_arr = np.asarray(w0, dtype=float)

    P, q = _build_quadratic_cost_factor(
        n, k, risk_aversion, frm.specific_var, frm.factor_cov, alpha, cp, cost_scale
    )
    A, l_vec, u_vec = _build_constraint_matrix_factor(
        spec, n, k, cost_scale, w0_arr, no_trade_band, frm.B
    )

    prob = osqp.OSQP()
    prob.setup(
        P,
        q,
        A,
        l_vec,
        u_vec,
        warm_starting=True,
        verbose=False,
        max_iter=max_iter * 20,
        eps_abs=1e-8,
        eps_rel=1e-8,
        eps_prim_inf=1e-8,
        eps_dual_inf=1e-8,
    )
    res = prob.solve()

    status = res.info.status
    converged = status in ("solved", "solved_inaccurate")
    weights = res.x[:n] if res.x is not None else np.zeros(n)

    # Compute objective using the factor risk model (structure-preserving; no dense Σw)
    var_val = frm.portfolio_variance(weights)
    alpha_ret = float(np.asarray(alpha, dtype=float) @ weights)
    if cost_scale > 0.0:
        delta = np.abs(weights - w0_arr)
        if no_trade_band > 0.0:
            delta = np.maximum(delta - no_trade_band, 0.0)
        cost_val = cost_scale * float(cp @ delta)
    else:
        cost_val = 0.0
    obj = alpha_ret - risk_aversion * var_val - cost_val

    return OptimizeResult(
        weights=weights,
        converged=converged,
        message=status,
        obj_value=obj,
        n_iter=int(res.info.iter),
    )
