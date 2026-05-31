"""Point-in-time (PIT) as-of join helper.

Feature 3 from the plan: given a dataset that carries both a ``knowledge_date``
(when the data was *available* to a model) and an ``effective_date`` /
``report_date`` (when the underlying event occurred), return only the rows a
model would have seen by a given ``as_of`` date.  This prevents look-ahead
bias that arises from using data before it was published.

The canonical use-case is ``fundamentals``: a quarterly report filed on
2024-02-15 may have a report_date of 2023-12-31 but a knowledge_date of
2024-02-15 (the filing date).  A model running on 2024-01-31 must not see it.

Design
------
``as_of_slice(df, as_of)``
    Thin filter: keep rows where ``knowledge_date <= as_of``.  Returns the
    entire history up to that point.

``latest_as_of(df, as_of, *, by)``
    "Most recent known snapshot": within each group defined by ``by``, keep
    the row with the largest ``knowledge_date`` that is <= ``as_of``.  This
    gives a *current view* rather than the full history.

Both functions operate on in-memory ``pl.DataFrame`` to stay composable with
the rest of the ETL; callers can push the ``as_of`` filter into ``scan_parquet``
themselves via ``DatasetLoader.scan`` if they need lazy evaluation.
"""

from __future__ import annotations

from datetime import date

import polars as pl


def as_of_slice(
    df: pl.DataFrame,
    as_of: date,
    *,
    knowledge_col: str = "knowledge_date",
) -> pl.DataFrame:
    """Return rows known on or before *as_of*.

    Parameters
    ----------
    df:
        Long-format frame with a ``knowledge_col`` (Date) column.
    as_of:
        The point-in-time cutoff.  Rows with ``knowledge_date > as_of`` are
        excluded (they hadn't been published yet).
    knowledge_col:
        Column name holding the publication/availability date.  Defaults to
        ``"knowledge_date"`` (the ``fundamentals`` schema).
    """
    if knowledge_col not in df.columns:
        raise ValueError(f"Column {knowledge_col!r} not found in frame")
    return df.filter(pl.col(knowledge_col) <= as_of)


def latest_as_of(
    df: pl.DataFrame,
    as_of: date,
    *,
    by: list[str],
    knowledge_col: str = "knowledge_date",
) -> pl.DataFrame:
    """Return the most-recently-known row per group as of *as_of*.

    Within each combination of ``by`` keys (e.g. ``["id"]`` for fundamentals),
    selects the single row whose ``knowledge_col`` is the maximum value that
    does not exceed *as_of*.  Groups with no known row by *as_of* are dropped.

    Parameters
    ----------
    df:
        Long-format frame with ``knowledge_col`` and all ``by`` columns.
    as_of:
        Point-in-time cutoff.
    by:
        Grouping keys (e.g. ``["id"]`` or ``["id", "report_date"]``).
    knowledge_col:
        Publication-date column.

    Returns
    -------
    pl.DataFrame
        One row per unique combination of ``by`` keys.
    """
    if knowledge_col not in df.columns:
        raise ValueError(f"Column {knowledge_col!r} not found in frame")
    sliced = as_of_slice(df, as_of, knowledge_col=knowledge_col)
    if sliced.is_empty():
        return sliced
    return sliced.sort(knowledge_col).group_by(by, maintain_order=False).last()
