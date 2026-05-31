"""Configurable weighting schemes and turnover/cost-aware rebalancing.

Schemes:
  equal_weight    — 1/N; simplest baseline, no data required.
  inverse_vol     — 1/σ_i (rescaled to sum to 1); tilts away from volatile names.
  cap_weight      — proportional to market cap; the passive benchmark.
  optimized       — delegates to the mean-variance optimizer.

Turnover / no-trade band:
  `apply_no_trade_band` clips small weight changes to zero when the drift from
  the previous weight is within the band threshold. This models the practical
  rule "don't trade if the cost of trading exceeds the benefit of correcting
  the drift."

  The complementary approach (cost penalty in the optimizer) is handled by
  optimizer.mean_variance via `cost_per_unit` and `no_trade_band`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .constraints import ConstraintSpec


class Scheme(Enum):
    EQUAL = auto()
    INVERSE_VOL = auto()
    CAP_WEIGHT = auto()
    OPTIMIZED = auto()


def equal_weight(n_assets: int) -> np.ndarray:
    """Equal weight: 1/N for all assets."""
    return np.full(n_assets, 1.0 / n_assets)


def inverse_vol(
    vols: np.ndarray,
    min_vol: float = 1e-8,
) -> np.ndarray:
    """Inverse-volatility weights, normalized to sum to 1.

    Args:
        vols:    (n_assets,) per-asset volatility (any positive scale).
        min_vol: Floor to prevent division by near-zero vol.

    Returns:
        (n_assets,) weight vector summing to 1.
    """
    iv = 1.0 / np.maximum(vols, min_vol)
    return iv / iv.sum()


def cap_weight(market_caps: np.ndarray, min_cap: float = 0.0) -> np.ndarray:
    """Market-cap weights, normalized to sum to 1.

    Args:
        market_caps: (n_assets,) market capitalizations (positive).
        min_cap:     Minimum cap floor (excludes near-zero / negative entries).

    Returns:
        (n_assets,) weight vector summing to 1.
    """
    mc = np.maximum(market_caps, min_cap)
    total = mc.sum()
    if total < 1e-15:
        return equal_weight(len(market_caps))
    return mc / total


def optimized_weight(
    alpha: np.ndarray,
    cov: np.ndarray,
    spec: ConstraintSpec,
    risk_aversion: float = 1.0,
    w0: np.ndarray | None = None,
    cost_per_unit: np.ndarray | None = None,
    cost_scale: float = 0.0,
    no_trade_band: float = 0.0,
) -> np.ndarray:
    """Mean-variance optimized weights (thin wrapper around optimizer.mean_variance).

    Returns the weight vector from OptimizeResult.
    """
    from .optimizer import mean_variance

    result = mean_variance(
        alpha=alpha,
        cov=cov,
        spec=spec,
        risk_aversion=risk_aversion,
        w0=w0,
        cost_per_unit=cost_per_unit,
        cost_scale=cost_scale,
        no_trade_band=no_trade_band,
    )
    return result.weights


def apply_no_trade_band(
    w_new: np.ndarray,
    w_prev: np.ndarray,
    band: float,
) -> np.ndarray:
    """Suppress trades where the weight change is within ±band.

    For each asset i, if |w_new_i − w_prev_i| ≤ band, keep w_prev_i.
    The resulting weights generally do not sum to exactly 1; the caller
    is responsible for renormalizing if required.

    This is the post-optimization no-trade rule. The optimizer itself also
    supports a cost-penalty-based band via its `no_trade_band` parameter.

    Args:
        w_new:  (n_assets,) target weights from the optimizer.
        w_prev: (n_assets,) current portfolio weights.
        band:   Minimum absolute change required to trigger a trade.

    Returns:
        (n_assets,) weights after applying the no-trade band.
    """
    drift = np.abs(w_new - w_prev)
    return np.where(drift <= band, w_prev, w_new)


def turnover(w_new: np.ndarray, w_prev: np.ndarray) -> float:
    """One-way turnover: Σ_i |w_new_i − w_prev_i| / 2.

    Divides by 2 because a sell of asset A and buy of asset B each appear
    as a change, but only one side of the trade occurred. The convention
    matches standard industry reporting (one-way turnover).
    """
    return float(np.abs(w_new - w_prev).sum() / 2.0)


def transaction_cost(
    w_new: np.ndarray,
    w_prev: np.ndarray,
    cost_per_unit: np.ndarray | float = 0.001,
) -> float:
    """Total round-trip transaction cost for a rebalance.

    Args:
        w_new:         (n_assets,) target weights.
        w_prev:        (n_assets,) current weights.
        cost_per_unit: Per-asset cost per unit of weight change (fraction).
                       Scalar or (n_assets,).

    Returns:
        Total cost as a fraction of portfolio value.
    """
    return float((np.abs(w_new - w_prev) * cost_per_unit).sum())


@dataclass(frozen=True)
class RebalanceResult:
    """Output of a single rebalance step.

    Attributes:
        weights:          (n_assets,) post-rebalance weights.
        weights_pre_band: (n_assets,) weights before no-trade band is applied.
        one_way_turnover: Fraction of portfolio traded (one-way).
        total_cost:       Estimated total transaction cost.
    """

    weights: np.ndarray
    weights_pre_band: np.ndarray
    one_way_turnover: float
    total_cost: float
