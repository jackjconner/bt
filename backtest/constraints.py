"""Portfolio and position constraint helpers.

Constraints are applied to the *raw* target weights produced by the signal
before they are translated into trades.  This keeps constraint logic separate
from both signal generation and execution mechanics.

Supported constraints
---------------------
1. **Universe mask** — assets flagged as non-tradeable receive zero weight and
   any existing position is liquidated by setting their target to 0.
2. **Per-name weight caps** — ``[min_weight, max_weight]`` per asset, e.g.
   long-only with 10% cap.
3. **Gross exposure cap** — ``sum(|w_i|) ≤ max_gross``; if violated, weights
   are rescaled uniformly.
4. **Net exposure cap** — ``|sum(w_i)| ≤ max_net``; excess is trimmed from the
   dominant side.

All constraints compose: they are applied in the order above so masking takes
precedence, then per-name caps, then portfolio-level caps.  The result is a
weight vector that respects all constraints but is not guaranteed to sum to
exactly 1 (cash position absorbs the remainder when fully-invested constraint
is not active).

Design note: we deliberately avoid a full QP optimizer here.  The heuristic
rescaling is O(n_assets) and sufficient for the production path; a proper
optimizer belongs in the ``portfolio`` module.
"""

from __future__ import annotations

import numpy as np


def apply_universe_mask(
    weights: np.ndarray,
    tradable: np.ndarray,
) -> np.ndarray:
    """Zero-out weights for non-tradeable assets.

    Parameters
    ----------
    weights:
        Raw target weights (n_assets,); may be modified in-place.
    tradable:
        Boolean mask (n_assets,); ``True`` = asset can be traded today.

    Returns
    -------
    np.ndarray
        Weights with non-tradeable assets zeroed.
    """
    result = weights.copy()
    result[~tradable] = 0.0
    return result


def apply_weight_caps(
    weights: np.ndarray,
    min_weight: np.ndarray,
    max_weight: np.ndarray,
) -> np.ndarray:
    """Clip weights to per-asset [min_weight, max_weight] bounds.

    For long-only portfolios pass ``min_weight=0`` and ``max_weight=cap``.
    After clipping, weights are **not** renormalized; the caller decides
    whether to renormalize or leave the residual as cash.
    """
    return np.clip(weights, min_weight, max_weight)


def apply_gross_exposure_cap(
    weights: np.ndarray,
    max_gross: float,
) -> np.ndarray:
    """Rescale weights proportionally so gross exposure ≤ max_gross.

    If current gross is already within the cap, weights are returned
    unchanged.  ``max_gross=1.0`` is fully-invested long-only;
    ``max_gross=2.0`` allows 2× leverage.
    """
    gross = float(np.abs(weights).sum())
    if gross <= max_gross or gross == 0.0:
        return weights
    return weights * (max_gross / gross)


def apply_net_exposure_cap(
    weights: np.ndarray,
    max_net: float,
) -> np.ndarray:
    """Trim net exposure (sum of weights) to ±max_net.

    Excess net is removed proportionally from the dominant-sign side.
    ``max_net=1.0`` is standard; ``max_net=0.0`` would force dollar-neutral.
    """
    net = float(weights.sum())
    if abs(net) <= max_net:
        return weights
    excess = abs(net) - max_net
    sign = 1.0 if net > 0.0 else -1.0
    # Reduce dominant-side positions proportionally
    dominant_mask = (np.sign(weights) == sign)
    dominant_sum = float(np.abs(weights[dominant_mask]).sum())
    if dominant_sum == 0.0:
        return weights
    result = weights.copy()
    result[dominant_mask] -= sign * excess * np.abs(weights[dominant_mask]) / dominant_sum
    return result


def apply_all_constraints(
    weights: np.ndarray,
    *,
    tradable: np.ndarray | None = None,
    min_weight: np.ndarray | None = None,
    max_weight: np.ndarray | None = None,
    max_gross: float | None = None,
    max_net: float | None = None,
) -> np.ndarray:
    """Apply all active constraints in canonical order.

    Parameters that are ``None`` are skipped (constraint inactive).
    """
    w = weights
    if tradable is not None:
        w = apply_universe_mask(w, tradable)
    if min_weight is not None and max_weight is not None:
        w = apply_weight_caps(w, min_weight, max_weight)
    if max_gross is not None:
        w = apply_gross_exposure_cap(w, max_gross)
    if max_net is not None:
        w = apply_net_exposure_cap(w, max_net)
    return w
