"""Corporate-action adjustment: back-adjusted, split/dividend-adjusted prices.

Feature 7 from the plan: given raw OHLCV prices and a ``corporate_actions``
frame, compute a *back-adjusted total-return price* series (dividend-inclusive,
split-corrected) that gives continuous percentage returns consistent with the
engine's ``R / 100.0`` accounting.

Methodology
-----------
Back-adjustment means we walk from the *most recent* date backwards and apply
a cumulative adjustment factor to all historical prices whenever a corporate
action's ex-date is reached.  This preserves the latest price level and adjusts
history, so the most recent price equals the raw price.

For a **split** with ratio ``r`` (e.g. 2.0 for a 2-for-1 split), all prices
*before* ex_date are divided by ``r``.

For a **cash dividend** with amount ``d``, the adjustment factor applied to
prior prices is ``close_{ex_date-1} / (close_{ex_date-1} + d)`` — i.e. the
pre-ex-date close is deflated by the dividend yield.  This converts prices to
total-return.

We only handle ``split`` and ``cash_dividend``/``special_dividend`` action
types; ``spinoff`` and ``delisting`` are noted in the returned log but their
pricing adjustment is complex and out of scope here.

Public API
----------
``adjust_prices(prices, corporate_actions, *, close_col)``
    Returns the adjusted price DataFrame (same schema as input with a new
    ``adj_close`` column) and an adjustment-log DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class AdjustmentResult:
    """Adjusted prices and an audit log of all applied factors."""

    prices: pl.DataFrame       # original columns + adj_close
    adj_log: pl.DataFrame      # ex_date, id, action_type, factor


def adjust_prices(
    prices: pl.DataFrame,
    corporate_actions: pl.DataFrame,
    *,
    close_col: str = "close",
) -> AdjustmentResult:
    """Back-adjust ``close_col`` for splits and cash dividends.

    Parameters
    ----------
    prices:
        Long-format OHLCV frame with columns ``date`` (Date), ``id`` (Int64),
        and at least ``close_col``.
    corporate_actions:
        Long-format corporate-actions frame with columns ``ex_date`` (Date),
        ``id`` (Int64), ``action_type`` (Categorical/String), ``split_ratio``
        (Float64, nullable), ``cash_amount`` (Float64, nullable).
    close_col:
        Name of the price column to adjust.  Other OHLC columns are scaled by
        the same factor so ratios (high/low spread) are preserved.

    Returns
    -------
    AdjustmentResult
        ``prices`` is the input frame extended with ``adj_close``.  The
        adjusted column back-adjusts so the most recent close equals the raw
        close.  ``adj_log`` records each applied factor for audit purposes.
    """
    # Collect all relevant corporate actions (splits and dividends only).
    ca = corporate_actions.filter(
        pl.col("action_type").cast(pl.String).is_in(
            ["split", "cash_dividend", "special_dividend"]
        )
    ).sort("ex_date")

    # Work per asset: sort dates descending, apply factors in reverse-ex order.
    ids = sorted(prices["id"].unique().to_list())
    all_prices_by_id = {
        aid: prices.filter(pl.col("id") == aid).sort("date")
        for aid in ids
    }
    ca_by_id = {
        aid: ca.filter(pl.col("id") == aid).sort("ex_date")
        for aid in ids
    }

    adj_frames: list[pl.DataFrame] = []
    log_rows: list[dict] = []

    for aid in ids:
        pdf = all_prices_by_id[aid]
        if pdf.is_empty():
            continue
        dates = pdf["date"].to_list()
        closes = pdf[close_col].to_numpy().copy()
        factor = np.ones(len(dates))  # cumulative adjustment factor per row

        actions = ca_by_id.get(aid)
        if actions is not None and not actions.is_empty():
            for action_row in actions.iter_rows(named=True):
                ex_date = action_row["ex_date"]
                atype = str(action_row["action_type"])
                # Find the index of the last price *before* ex_date.
                prior_indices = [i for i, d in enumerate(dates) if d < ex_date]
                if not prior_indices:
                    continue
                last_prior = prior_indices[-1]
                pre_close = closes[last_prior]

                if atype == "split":
                    ratio = action_row.get("split_ratio")
                    if ratio is None or np.isnan(ratio) or ratio <= 0:
                        continue
                    # Divide all prices strictly before ex_date by the ratio.
                    f = 1.0 / ratio
                    for i in range(last_prior + 1):
                        factor[i] *= f
                    log_rows.append({"ex_date": ex_date, "id": aid, "action_type": atype, "factor": f})

                elif atype in ("cash_dividend", "special_dividend"):
                    cash = action_row.get("cash_amount")
                    if cash is None or np.isnan(cash) or cash <= 0 or pre_close <= 0:
                        continue
                    # pre-ex-date prices are deflated by the dividend yield
                    f = pre_close / (pre_close + cash)
                    for i in range(last_prior + 1):
                        factor[i] *= f
                    log_rows.append({"ex_date": ex_date, "id": aid, "action_type": atype, "factor": f})

        adj_close = closes * factor
        adj_frames.append(pdf.with_columns(pl.Series("adj_close", adj_close)))

    if adj_frames:
        result_prices = pl.concat(adj_frames).sort("date", "id")
    else:
        result_prices = prices.with_columns(
            pl.col(close_col).alias("adj_close")
        )

    if log_rows:
        adj_log = pl.DataFrame(log_rows).with_columns(
            pl.col("id").cast(pl.Int64),
            pl.col("action_type").cast(pl.Categorical),
        )
    else:
        adj_log = pl.DataFrame(
            schema={
                "ex_date": pl.Date,
                "id": pl.Int64,
                "action_type": pl.Categorical,
                "factor": pl.Float64,
            }
        )

    return AdjustmentResult(prices=result_prices, adj_log=adj_log)
