"""Corporate-action adjustment helpers.

Supported action types (matching the ``action_type`` column in the
``corporate_actions`` dataset):

- ``split``:          shares multiply by ``split_ratio``; price divides by
                      ``split_ratio``; NAV is unchanged.
- ``cash_dividend``:  cash per share is credited to the cash ledger; share
                      count / weight is unchanged (portfolio NAV increases by
                      the dividend received, consistent with total-return).
- ``special_dividend``: same accounting as ``cash_dividend``.

Unsupported types (spinoff, delisting) are ignored: the engine should handle
delistings via the universe mask, and spinoffs require new-id mapping that is
outside the scope of the current model.

All functions operate on the engine's internal representation:
- ``shares``:  (n_assets,) float array of share counts.
- ``prices``:  (n_assets,) float array of current mid prices.
- ``cash``:    scalar cash balance.

The caller passes only the actions for the **current** ex-date so the inner
loop stays O(n_actions_today), not O(n_actions_total).
"""

from __future__ import annotations

import numpy as np


def apply_corporate_actions(
    shares: np.ndarray,
    prices: np.ndarray,
    cash: float,
    action_ids: list[int],
    action_types: list[str],
    split_ratios: list[float | None],
    cash_amounts: list[float | None],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply today's corporate actions in-place and return updated state.

    Parameters
    ----------
    shares:
        Current share counts per asset (n_assets,); mutated by splits.
    prices:
        Current prices per asset (n_assets,); mutated by splits.
    cash:
        Current cash balance; incremented by dividends.
    action_ids:
        Asset indices for each action (parallel with the other lists).
    action_types:
        Action type string per action.
    split_ratios:
        Split ratio for ``split`` actions; ``None`` otherwise.
    cash_amounts:
        Cash-per-share amount for dividend actions; ``None`` otherwise.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        Updated (shares, prices, cash).  ``shares`` and ``prices`` are
        modified in-place and also returned for convenience.
    """
    for idx, atype, ratio, amount in zip(
        action_ids, action_types, split_ratios, cash_amounts, strict=False
    ):
        if idx < 0 or idx >= len(shares):
            continue
        if atype == "split" and ratio is not None and ratio > 0.0:
            # Shares multiply; price divides — NAV invariant.
            shares[idx] *= ratio
            prices[idx] /= ratio
        elif atype in ("cash_dividend", "special_dividend") and amount is not None and amount > 0.0:
            # Dividend credited to cash ledger; total-return accounting.
            cash += float(shares[idx]) * amount
    return shares, prices, cash


def build_action_index(
    corporate_actions: pl.DataFrame,  # noqa: F821  — polars import at call-site
) -> dict:
    """Pre-index corporate actions by ex_date for O(1) per-step lookup.

    Returns a dict mapping ``date → list[dict]`` where each dict has keys
    ``id``, ``action_type``, ``split_ratio``, ``cash_amount``.
    """
    index: dict = {}
    for row in corporate_actions.iter_rows(named=True):
        d = row["ex_date"]
        if d not in index:
            index[d] = []
        index[d].append(
            {
                "id": int(row["id"]),
                "action_type": str(row["action_type"]),
                "split_ratio": row["split_ratio"],
                "cash_amount": row["cash_amount"],
            }
        )
    return index
