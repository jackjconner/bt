"""Survivorship-bias-free universe resolution.

Feature 4 from the plan: given the ``universe_mask`` panel (with per-date
``in_universe``, ``tradable``, ``listed``, ``halted`` flags) and/or the
``security_master`` (with ``listing_date``/``delisting_date`` intervals),
produce a date×id boolean membership array aligned to an arbitrary date/id
grid.

Survivorship bias arises when only currently-active assets are included: dead
names are silently excluded, inflating historical returns (you always held the
winners that survived).  This module keeps the full universe including names
that were later delisted.

Public API
----------
``resolve_universe(universe_mask, dates, ids, *, flag)``
    Build a ``(n_dates, n_ids)`` boolean matrix from the long-format
    ``universe_mask`` frame.  Dates/ids not found in the source are filled
    with ``False`` (unknown = not investable).

``apply_security_master(mask_df, security_master, dates)``
    Overlay listing/delisting intervals from ``security_master`` onto a
    ``universe_mask`` frame: an asset is not listed before its ``listing_date``
    or after its ``delisting_date``.  Returns the adjusted frame.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl


def resolve_universe(
    universe_mask: pl.DataFrame,
    dates: list[date],
    ids: list[int],
    *,
    flag: str = "in_universe",
) -> np.ndarray:
    """Pivot ``universe_mask`` to a dense ``(n_dates, n_ids)`` boolean array.

    Missing ``(date, id)`` combinations are treated as ``False`` — an asset
    that has no record for a given session is presumed not to be in the universe.
    This is the conservative, survivorship-safe default.

    Parameters
    ----------
    universe_mask:
        Long-format frame with columns ``date`` (Date), ``id`` (Int64), and at
        least ``flag``.
    dates:
        Ordered list of session dates that form the row axis.
    ids:
        Ordered list of asset ids that form the column axis.
    flag:
        Boolean column to pivot.  Defaults to ``"in_universe"``; pass
        ``"tradable"`` for the tradable sub-universe.

    Returns
    -------
    np.ndarray shape (len(dates), len(ids)), dtype bool
    """
    if flag not in universe_mask.columns:
        raise ValueError(f"Column {flag!r} not found in universe_mask")

    # Build a reference grid so we can left-join and fill unknowns with False.
    date_index = {d: i for i, d in enumerate(dates)}
    id_index = {v: j for j, v in enumerate(ids)}

    mask = np.zeros((len(dates), len(ids)), dtype=bool)
    for row in universe_mask.select("date", "id", flag).iter_rows():
        d, asset_id, val = row
        di = date_index.get(d)
        ji = id_index.get(asset_id)
        if di is not None and ji is not None and val:
            mask[di, ji] = True
    return mask


def apply_security_master(
    universe_mask: pl.DataFrame,
    security_master: pl.DataFrame,
    dates: list[date],
) -> pl.DataFrame:
    """Set ``in_universe``/``listed`` to False outside listing/delisting windows.

    An asset's record on a date before its ``listing_date`` or after its
    ``delisting_date`` (when present) is forced to ``in_universe=False`` and
    ``listed=False``.  All other columns are unchanged.

    This is an *overlay*: if ``universe_mask`` already has a row for an asset
    on a date when it isn't listed the row is simply updated.  Rows for assets
    absent from ``security_master`` are left untouched (permissive for assets
    whose metadata is missing).

    Parameters
    ----------
    universe_mask:
        Long-format frame (``date``, ``id``, ``in_universe``, ``listed``, …).
    security_master:
        Per-asset static frame with ``id``, ``listing_date``,
        ``delisting_date`` (nullable).
    dates:
        Full session axis — used only for type consistency, not filtering.
    """
    sm = security_master.select("id", "listing_date", "delisting_date")
    joined = universe_mask.join(sm, on="id", how="left")

    in_window = (pl.col("date") >= pl.col("listing_date")) & (
        pl.col("delisting_date").is_null() | (pl.col("date") <= pl.col("delisting_date"))
    )
    return joined.with_columns(
        pl.when(in_window)
        .then(pl.col("in_universe"))
        .otherwise(pl.lit(False))
        .alias("in_universe"),
        pl.when(in_window).then(pl.col("listed")).otherwise(pl.lit(False)).alias("listed"),
    ).drop("listing_date", "delisting_date")
