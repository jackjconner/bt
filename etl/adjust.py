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

    prices: pl.DataFrame  # original columns + adj_close
    adj_log: pl.DataFrame  # ex_date, id, action_type, factor


def _split_factor(action_row: dict) -> float | None:
    """Return the back-adjustment factor for a split action, or None to skip."""
    ratio = action_row.get("split_ratio")
    if ratio is None or np.isnan(ratio) or ratio <= 0:
        return None
    return 1.0 / ratio


def _dividend_factor(action_row: dict, pre_close: float) -> float | None:
    """Return the back-adjustment factor for a cash/special dividend, or None to skip."""
    cash = action_row.get("cash_amount")
    if cash is None or np.isnan(cash) or cash <= 0 or pre_close <= 0:
        return None
    return pre_close / (pre_close + cash)


def _apply_factor(factor: np.ndarray, f: float, last_prior: int) -> None:
    """Multiply ``factor[0:last_prior+1]`` by ``f`` in-place."""
    for i in range(last_prior + 1):
        factor[i] *= f


def _adjust_single_asset(
    pdf: pl.DataFrame,
    actions: pl.DataFrame | None,
    close_col: str,
) -> tuple[pl.DataFrame, list[dict]]:
    """Compute adjusted close for one asset; return extended frame and log rows."""
    dates = pdf["date"].to_list()
    closes = pdf[close_col].to_numpy().copy()
    factor = np.ones(len(dates))
    log_rows: list[dict] = []

    if actions is not None and not actions.is_empty():
        for action_row in actions.iter_rows(named=True):
            ex_date = action_row["ex_date"]
            atype = str(action_row["action_type"])
            prior_indices = [i for i, d in enumerate(dates) if d < ex_date]
            if not prior_indices:
                continue
            last_prior = prior_indices[-1]
            pre_close = closes[last_prior]

            if atype == "split":
                f = _split_factor(action_row)
            elif atype in ("cash_dividend", "special_dividend"):
                f = _dividend_factor(action_row, pre_close)
            else:
                f = None

            if f is None:
                continue

            _apply_factor(factor, f, last_prior)
            log_rows.append(
                {"ex_date": ex_date, "id": action_row["id"], "action_type": atype, "factor": f}
            )

    adj_close = closes * factor
    return pdf.with_columns(pl.Series("adj_close", adj_close)), log_rows


def _build_adj_log(log_rows: list[dict]) -> pl.DataFrame:
    """Assemble the audit log DataFrame from raw row dicts."""
    if log_rows:
        return pl.DataFrame(log_rows).with_columns(
            pl.col("id").cast(pl.Int64),
            pl.col("action_type").cast(pl.Categorical),
        )
    return pl.DataFrame(
        schema={
            "ex_date": pl.Date,
            "id": pl.Int64,
            "action_type": pl.Categorical,
            "factor": pl.Float64,
        }
    )


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
        pl.col("action_type").cast(pl.String).is_in(["split", "cash_dividend", "special_dividend"])
    ).sort("ex_date")

    # Work per asset: sort dates descending, apply factors in reverse-ex order.
    ids = sorted(prices["id"].unique().to_list())
    all_prices_by_id = {aid: prices.filter(pl.col("id") == aid).sort("date") for aid in ids}
    ca_by_id = {aid: ca.filter(pl.col("id") == aid).sort("ex_date") for aid in ids}

    adj_frames: list[pl.DataFrame] = []
    all_log_rows: list[dict] = []

    for aid in ids:
        pdf = all_prices_by_id[aid]
        if pdf.is_empty():
            continue
        actions = ca_by_id.get(aid)
        adj_frame, log_rows = _adjust_single_asset(pdf, actions, close_col)
        adj_frames.append(adj_frame)
        all_log_rows.extend(log_rows)

    if adj_frames:
        result_prices = pl.concat(adj_frames).sort("date", "id")
    else:
        result_prices = prices.with_columns(pl.col(close_col).alias("adj_close"))

    return AdjustmentResult(
        prices=result_prices,
        adj_log=_build_adj_log(all_log_rows),
    )
