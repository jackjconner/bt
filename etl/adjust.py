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

    Notes
    -----
    The adjustment is computed by ``_adjust_vectorized``: a single columnar
    pass over the whole panel — no per-asset partition into Python frames and
    no ``pl.concat`` of N per-asset blocks.  Per-asset action factors are found
    with one ``join_asof`` (raw pre-ex close), and the back-adjustment factor at
    every ``(id, date)`` cell is a single segmented reverse-cumulative-product
    over the sorted long frame.  This is numerically identical to the legacy
    per-asset loop (``_adjust_single_asset``, retained below) but scales with
    the *number of actions* rather than the *number of assets*.
    """
    return _adjust_vectorized(prices, corporate_actions, close_col)


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
    before the first session has no prior row and is dropped — matching the
    legacy ``prior_indices`` guard.
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
    # (ex_date at/before the first row) changes nothing and is omitted, matching
    # the legacy per-asset path's ``prior_indices`` guard.
    applied = is_tail | has_prior
    adj_log = (
        matched.select("ex_date", "id", "action_type", "factor")
        .filter(pl.Series(applied))
        .with_columns(pl.col("action_type").cast(pl.Categorical))
    )

    return AdjustmentResult(prices=result_prices, adj_log=adj_log)
