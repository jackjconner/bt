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

from .quality import annotate_quality_flags


@dataclass(frozen=True)
class AdjustmentResult:
    """Adjusted prices and an audit log of all applied factors."""

    prices: pl.DataFrame  # original columns + adj_close
    adj_log: pl.DataFrame  # ex_date, id, action_type, factor


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
    include_quality_flags: bool = False,
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
    include_quality_flags:
        When ``True``, append the per-row data-quality flag columns produced by
        :func:`etl.quality.annotate_quality_flags` (computed on the *raw*
        ``close_col``) to ``AdjustmentResult.prices``.  Off by default: the
        returned frame is then byte-identical to the no-flag path — existing
        consumers see no new columns and no reshaping.

    Returns
    -------
    AdjustmentResult
        ``prices`` is the input frame extended with ``adj_close`` (and, when
        ``include_quality_flags`` is set, the quality-flag columns).  The
        adjusted column back-adjusts so the most recent close equals the raw
        close.  ``adj_log`` records each applied factor for audit purposes.

    Notes
    -----
    The adjustment is computed by ``_adjust_vectorized``: a single columnar
    pass over the whole panel — no per-asset partition into Python frames and
    no ``pl.concat`` of N per-asset blocks.  Per-asset action factors are found
    with one ``join_asof`` (raw pre-ex close), and the back-adjustment factor at
    every ``(id, date)`` cell is a single segmented reverse-cumulative-product
    over the sorted long frame.  This scales with the *number of actions* rather
    than the *number of assets*.
    """
    result = _adjust_vectorized(prices, corporate_actions, close_col)
    if not include_quality_flags:
        return result
    flagged = annotate_quality_flags(result.prices, close_col)
    return AdjustmentResult(prices=flagged, adj_log=result.adj_log)


def _adjust_vectorized(
    prices: pl.DataFrame,
    corporate_actions: pl.DataFrame,
    close_col: str,
) -> AdjustmentResult:
    """Whole-panel vectorized back-adjustment (see ``adjust_prices`` Notes).

    The factor applied to a ``(id, t)`` cell is the product of every action
    factor for that id whose ``ex_date > t`` (back-adjustment: an action shifts
    all strictly-prior history).  Computing that for every cell is a per-asset
    reverse cumulative product.  We place each action factor at the first
    in-segment row ``>= ex_date`` (so it back-applies to all rows before it),
    take a *segment-reset* reverse cumulative sum in log space (underflow-proof
    even when many splits multiply), exponentiate, and scale the close.  An
    action whose ``ex_date`` falls after an asset's last session applies to the
    whole segment (a per-segment tail factor); one whose ``ex_date`` is at or
    before the first session has no prior row and is dropped (no history to
    back-adjust).
    """
    ca = corporate_actions.filter(
        pl.col("action_type").cast(pl.String).is_in(["split", "cash_dividend", "special_dividend"])
    )

    prices_sorted = prices.sort(["id", "date"])
    closes = prices_sorted[close_col].to_numpy()

    if ca.is_empty():
        return AdjustmentResult(
            prices=prices_sorted.with_columns(pl.col(close_col).alias("adj_close")).sort(
                "date", "id"
            ),
            adj_log=_build_adj_log([]),
        )

    ids = prices_sorted["id"].to_numpy()
    dates = prices_sorted["date"].to_numpy()
    n = len(ids)

    # Contiguous per-id segments in the (id, date)-sorted frame.
    uniq_ids, seg_start = np.unique(ids, return_index=True)
    seg_end = np.empty_like(seg_start)
    seg_end[:-1] = seg_start[1:]
    seg_end[-1] = n
    pos_of_id = {int(v): k for k, v in enumerate(uniq_ids.tolist())}

    # Raw close on the last session strictly before each action's ex_date.
    # (polars can't verify right-side sortedness under a ``by`` group, so it
    # emits a benign UserWarning here; both frames are sorted on the asof key.)
    pre = prices_sorted.select("id", "date", pl.col(close_col).alias("_pre")).sort("date")
    matched = ca.sort("ex_date").join_asof(
        pre,
        left_on="ex_date",
        right_on="date",
        by="id",
        strategy="backward",
        allow_exact_matches=False,
    )
    at = pl.col("action_type").cast(pl.String)
    split_f = pl.when(
        (at == "split") & pl.col("split_ratio").is_not_null() & (pl.col("split_ratio") > 0)
    ).then(1.0 / pl.col("split_ratio"))
    div_f = pl.when(
        at.is_in(["cash_dividend", "special_dividend"])
        & pl.col("cash_amount").is_not_null()
        & (pl.col("cash_amount") > 0)
        & pl.col("_pre").is_not_null()
        & (pl.col("_pre") > 0)
    ).then(pl.col("_pre") / (pl.col("_pre") + pl.col("cash_amount")))
    matched = matched.with_columns(split_f.otherwise(div_f).alias("factor")).filter(
        pl.col("factor").is_not_null()
    )

    if matched.is_empty():
        return AdjustmentResult(
            prices=prices_sorted.with_columns(pl.col(close_col).alias("adj_close")).sort(
                "date", "id"
            ),
            adj_log=_build_adj_log([]),
        )

    a_id = matched["id"].to_numpy()
    a_ex = matched["ex_date"].to_numpy()
    a_factor = matched["factor"].to_numpy()
    k_arr = np.fromiter((pos_of_id[int(v)] for v in a_id.tolist()), dtype=np.int64, count=len(a_id))
    s_arr = seg_start[k_arr]
    e_arr = seg_end[k_arr]

    # First in-segment row index whose date >= ex_date (where the factor lands).
    gpos = np.empty(len(a_id), dtype=np.int64)
    for i in range(len(a_id)):
        s, e = int(s_arr[i]), int(e_arr[i])
        gpos[i] = s + int(np.searchsorted(dates[s:e], np.datetime64(a_ex[i]), side="left"))

    is_tail = gpos >= e_arr  # ex_date after the asset's last session → whole segment
    has_prior = gpos > s_arr  # at least one in-segment row strictly before ex_date

    log_factor = np.zeros(n)  # per-row log back-adjustment factor (exp → multiplier)
    in_seg = ~is_tail
    np.add.at(log_factor, gpos[in_seg], np.log(a_factor[in_seg]))
    seg_tail_log = np.zeros(len(uniq_ids))
    np.add.at(seg_tail_log, k_arr[is_tail], np.log(a_factor[is_tail]))

    # Segment-reset reverse cumulative sum: row j gets the sum of placed logs at
    # positions > j within its own segment.  Global reverse cumsum minus the
    # value at the segment end isolates each segment (log space ⇒ no underflow).
    rev = np.zeros(n + 1)
    rev[:-1] = np.cumsum(log_factor[::-1])[::-1]
    seg_end_row = np.repeat(seg_end, seg_end - seg_start)
    tail_row = np.repeat(seg_tail_log, seg_end - seg_start)
    factor = np.exp((rev[1:] - rev[seg_end_row]) + tail_row)

    result_prices = prices_sorted.with_columns(pl.Series("adj_close", closes * factor)).sort(
        "date", "id"
    )

    # Audit log: one row per applied factor.  An action with no prior session
    # (ex_date at/before the first row) changes nothing and is omitted.
    applied = is_tail | has_prior
    adj_log = (
        matched.select("ex_date", "id", "action_type", "factor")
        .filter(pl.Series(applied))
        .with_columns(pl.col("action_type").cast(pl.Categorical))
    )

    return AdjustmentResult(prices=result_prices, adj_log=adj_log)
