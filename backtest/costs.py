"""Transaction-cost and financing-cost helpers.

All cost functions consume numpy arrays already aligned to the (n_assets,)
layout used by the engine loop.  They return a scalar total-cost in dollar
terms so the caller can subtract from NAV (or cash) directly.

Convention: ``trade_value[i]`` is the **signed** dollar notional traded for
asset *i* (positive = buy, negative = sell).  All bps values come from the
``transaction_costs`` dataset; borrow bps from ``borrow_rates``.

Formula per asset *i*:
    commission   = max(|notional_i| * commission_bps/1e4, min_commission)
    spread_cost  = |notional_i| * half_spread_bps/1e4  (one-way half-spread)
    exchange_fee = |notional_i| * exchange_fee_bps/1e4
    total_i      = commission + spread_cost + exchange_fee

Borrow cost per asset per day:
    borrow_i = max(0, -weight_i) * nav * borrow_rate_bps/1e4 / 252

These are kept separate because transaction costs fire on trade events while
borrow costs accrue daily on open short positions.
"""

from __future__ import annotations

import numpy as np


def compute_transaction_costs(
    trade_value: np.ndarray,
    commission_bps: np.ndarray,
    half_spread_bps: np.ndarray,
    exchange_fee_bps: np.ndarray,
    min_commission: np.ndarray,
) -> float:
    """Return total transaction cost in dollars for a vector of trades.

    Parameters
    ----------
    trade_value:
        Signed dollar notional per asset (n_assets,).
    commission_bps, half_spread_bps, exchange_fee_bps, min_commission:
        Per-asset cost parameters aligned to ``trade_value``; all (n_assets,).

    Returns
    -------
    float
        Total cost in dollars (always non-negative).
    """
    abs_notional = np.abs(trade_value)
    commission = np.maximum(abs_notional * commission_bps / 1e4, min_commission)
    # commission applies only to assets that were actually traded
    commission = np.where(abs_notional > 0.0, commission, 0.0)
    spread = abs_notional * half_spread_bps / 1e4
    exchange = abs_notional * exchange_fee_bps / 1e4
    return float((commission + spread + exchange).sum())


def compute_borrow_cost(
    weights: np.ndarray,
    nav: float,
    borrow_rate_bps: np.ndarray,
) -> float:
    """Daily borrow cost charged against NAV for short positions.

    Parameters
    ----------
    weights:
        Current portfolio weights (n_assets,); negatives are shorts.
    nav:
        Current NAV in dollars.
    borrow_rate_bps:
        Per-asset annualized borrow rate in bps (n_assets,).

    Returns
    -------
    float
        Total daily borrow cost in dollars (always non-negative).
    """
    short_weights = np.maximum(-weights, 0.0)
    return float((short_weights * nav * borrow_rate_bps / 1e4 / 252.0).sum())


def compute_cash_interest(cash: float, annual_rate: float) -> float:
    """Daily interest credit/charge on the cash balance.

    Positive cash earns interest; negative cash (margin) is charged.
    Both use the same *annual_rate* as a simplification (typical for
    overnight repo / prime brokerage pricing).

    Returns
    -------
    float
        Dollar interest accrual for one day (positive = credit, negative = charge).
    """
    return cash * annual_rate / 252.0


def compute_financing_cost(
    weights: np.ndarray,
    nav: float,
    borrow_rate_annual: float,
    funding_rate_annual: float,
    dt: float,
) -> float:
    """Daily financing drag for a leveraged or short portfolio.

    Two components are charged:

    1. **Short borrow** — cost of borrowing shares to maintain short positions:
       ``sum(max(0, -w_i)) * nav * borrow_rate_annual * dt``

    2. **Leverage funding** — cost of margin debt above 1× gross exposure:
       ``max(gross_leverage - 1, 0) * nav * funding_rate_annual * dt``

    A long-only, unleveraged portfolio (gross == 1) incurs ~zero cost from both
    terms (long-only has no shorts; gross == 1 means no excess leverage).

    Parameters
    ----------
    weights:
        Current portfolio weights (n_assets,); negatives indicate shorts.
    nav:
        Current NAV in dollars.
    borrow_rate_annual:
        Annualized borrow rate applied to the aggregate short market value
        (fraction, e.g. ``0.005`` for 50 bps).
    funding_rate_annual:
        Annualized funding rate applied to leveraged exposure above 1× gross
        (fraction, e.g. ``0.02`` for 200 bps).
    dt:
        Period length as a fraction of a year (e.g. ``1/252`` for daily).

    Returns
    -------
    float
        Total financing cost in dollars for this period (always non-negative).
    """
    short_mv = float(np.maximum(-weights, 0.0).sum()) * nav
    gross_leverage = float(np.abs(weights).sum())
    excess_leverage = max(gross_leverage - 1.0, 0.0) * nav
    return (short_mv * borrow_rate_annual + excess_leverage * funding_rate_annual) * dt
