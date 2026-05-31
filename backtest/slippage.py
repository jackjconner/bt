"""Market-impact slippage model.

We implement a square-root impact model, the industry-standard workhorse:

    impact_bps(i) = impact_coef(i) * sqrt(|trade_size_i| / adv_i)

where ``adv_i`` is the 20-day average dollar volume for asset *i* and
``trade_size_i`` is the absolute dollar value traded.  The formula models
permanent + temporary impact together as a single one-way cost.

The square-root scaling comes from empirical market microstructure research
(e.g. Almgren-Chriss 2000): doubling trade size roughly 41% more than doubles
impact, reflecting the order-book depth decay.

For assets with zero ADV (illiquid/halted) the impact is set to a configurable
penalty so the engine can still run without division-by-zero.

Slippage is expressed as a positive dollar cost (always reduces NAV).
"""

from __future__ import annotations

import numpy as np

_FALLBACK_IMPACT_BPS = 500.0  # 5% impact for zero-ADV assets — punitive


def compute_slippage(
    trade_value: np.ndarray,
    adv: np.ndarray,
    impact_coef: np.ndarray,
    fallback_impact_bps: float = _FALLBACK_IMPACT_BPS,
) -> float:
    """Return total slippage cost in dollars for a vector of trades.

    Parameters
    ----------
    trade_value:
        Signed dollar notional per asset (n_assets,).
    adv:
        20-day average dollar volume per asset (n_assets,); must be ≥ 0.
    impact_coef:
        Per-asset impact coefficient (n_assets,); dimensionless, calibrated so
        that a 1% participation rate incurs ``impact_coef * sqrt(0.01)`` bps.
    fallback_impact_bps:
        Impact applied when ``adv[i] == 0`` (illiquid / halted assets).

    Returns
    -------
    float
        Total slippage cost in dollars (always non-negative).
    """
    abs_notional = np.abs(trade_value)
    safe_adv = np.where(adv > 0.0, adv, np.nan)
    participation = abs_notional / safe_adv
    sqrt_impact_bps = impact_coef * np.sqrt(participation)
    # assets with zero ADV get the fallback penalty
    sqrt_impact_bps = np.where(
        adv > 0.0,
        sqrt_impact_bps,
        np.where(abs_notional > 0.0, fallback_impact_bps, 0.0),
    )
    # impact_bps * notional / 1e4 = dollar cost
    return float((abs_notional * sqrt_impact_bps / 1e4).sum())


def fill_price_with_slippage(
    mid_price: float,
    trade_value: float,
    adv: float,
    impact_coef: float,
    fallback_impact_bps: float = _FALLBACK_IMPACT_BPS,
) -> float:
    """Return the slippage-adjusted fill price for a single asset.

    Buys pay above mid; sells receive below mid.  Sign convention: positive
    ``trade_value`` = buy, negative = sell.
    """
    if adv > 0.0:
        participation = abs(trade_value) / adv
        impact_bps = impact_coef * (participation**0.5)
    else:
        impact_bps = fallback_impact_bps if trade_value != 0.0 else 0.0
    direction = 1.0 if trade_value >= 0.0 else -1.0
    return mid_price * (1.0 + direction * impact_bps / 1e4)
