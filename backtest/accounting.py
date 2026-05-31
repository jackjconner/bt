"""Price-based share-level accounting helpers.

The default engine runs in *weight space*: it tracks portfolio weights and
propagates NAV by ``positions @ (R/100)``.  This is mathematically clean but
doesn't tell you how many shares you own, what your cash balance is, or what
your fill price was.

The production engine runs in *share space*:
- ``shares[i]``  — how many shares of asset *i* the portfolio holds.
- ``cash``       — the undeployed/margin cash balance.
- NAV            = ``shares @ prices + cash``

Functions here handle:
1. Converting a target-weight vector into share deltas (how many shares to buy/sell).
2. Computing NAV from shares + prices + cash.
3. Marking the portfolio to market when prices update.

The fill price fed into share-quantity calculation already incorporates
slippage (see ``slippage.fill_price_with_slippage``), so accounting itself
is mechanical once prices are set.
"""

from __future__ import annotations

import numpy as np


def nav_from_shares(
    shares: np.ndarray,
    prices: np.ndarray,
    cash: float,
) -> float:
    """Compute portfolio NAV as mark-to-market equity + cash.

    Parameters
    ----------
    shares:
        Share counts per asset (n_assets,); may be negative for shorts.
    prices:
        Current mid (or close) price per asset (n_assets,).
    cash:
        Cash balance (positive = long cash, negative = margin debt).

    Returns
    -------
    float
        Total portfolio value in dollars.
    """
    return float(shares @ prices) + cash


def target_weights_to_share_deltas(
    target_weights: np.ndarray,
    current_shares: np.ndarray,
    fill_prices: np.ndarray,
    nav: float,
) -> np.ndarray:
    """Compute the share delta needed to reach ``target_weights`` at fill prices.

    Parameters
    ----------
    target_weights:
        Desired portfolio weights summing to ≤ 1 (or > 1 for leverage).
    current_shares:
        Current share holdings (n_assets,).
    fill_prices:
        Execution prices including slippage (n_assets,).
    nav:
        Current NAV used to translate weights → dollar notional.

    Returns
    -------
    np.ndarray
        Share delta (n_assets,); positive = buy, negative = sell.
        Zero for assets with zero fill price (untradeable).
    """
    safe_prices = np.where(fill_prices > 0.0, fill_prices, np.nan)
    target_shares = np.where(
        fill_prices > 0.0,
        target_weights * nav / safe_prices,
        current_shares,  # can't trade → keep current
    )
    return target_shares - current_shares


def execute_trades(
    current_shares: np.ndarray,
    share_deltas: np.ndarray,
    fill_prices: np.ndarray,
    cash: float,
    total_cost: float,
) -> tuple[np.ndarray, float]:
    """Apply share deltas, debit cash, and charge total cost.

    Parameters
    ----------
    current_shares:
        Pre-trade share counts (n_assets,).
    share_deltas:
        Shares to trade per asset (n_assets,).
    fill_prices:
        Fill price per asset; used to compute cash impact.
    cash:
        Pre-trade cash balance.
    total_cost:
        Total transaction + slippage cost in dollars (always subtracted).

    Returns
    -------
    tuple[np.ndarray, float]
        (new_shares, new_cash) after settlement.
    """
    new_shares = current_shares + share_deltas
    # Cash decreases by the cost of buys and increases by the proceeds of sells,
    # then decreases again by the friction cost.
    cash_impact = -(share_deltas * fill_prices).sum()
    new_cash = cash + cash_impact - total_cost
    return new_shares, float(new_cash)


def weights_from_shares(
    shares: np.ndarray,
    prices: np.ndarray,
    nav: float,
) -> np.ndarray:
    """Compute portfolio weights from share/price/NAV.  NAV must be > 0."""
    if nav <= 0.0:
        return np.zeros_like(shares)
    return shares * prices / nav
