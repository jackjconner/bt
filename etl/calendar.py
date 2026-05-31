"""Trading-calendar alignment helpers.

Feature 8 from the plan: the existing ``date_axis`` in ``source.py`` produces
naive calendar days (including weekends), which breaks forward-return horizons,
annualization constants, and rebalancing cadence.  ``session_axis`` introduced
business days as an improvement, but the authoritative session set comes from
the ``trading_calendar`` dataset (which also captures exchange holidays and
half-days).

``align_to_calendar(df, calendar)``
    Reindex a long-format panel so it contains exactly one row per ``(session,
    id)`` pair from the trading calendar.  Missing sessions are filled with
    ``null``; extra (non-session) dates are dropped.

``sessions_between(calendar, start, end, *, exchange)``
    Return the sorted list of trading sessions for an exchange between two
    dates.

``fill_sessions(df, calendar, *, method, value_col)``
    Forward-fill (or backward-fill, or zero-fill) a panel to every trading
    session so there are no gaps.

All functions accept the ``trading_calendar`` long frame from ``datasets.py``
and work on the ``(date, id)`` long format used throughout the codebase.
"""

from __future__ import annotations

from datetime import date

import polars as pl


def sessions_between(
    calendar: pl.DataFrame,
    start: date,
    end: date,
    *,
    exchange: str = "XNYS",
) -> list[date]:
    """Return sorted trading sessions for *exchange* in ``[start, end]``.

    Parameters
    ----------
    calendar:
        ``trading_calendar`` long frame with ``date``, ``exchange``,
        ``is_session`` columns.
    start, end:
        Inclusive date bounds.
    exchange:
        Exchange code to filter on.
    """
    result = (
        calendar.filter(
            (pl.col("exchange").cast(pl.String) == exchange)
            & pl.col("is_session")
            & (pl.col("date") >= start)
            & (pl.col("date") <= end)
        )
        .sort("date")["date"]
        .to_list()
    )
    return result


def align_to_calendar(
    df: pl.DataFrame,
    calendar: pl.DataFrame,
    ids: list[int],
    *,
    exchange: str = "XNYS",
) -> pl.DataFrame:
    """Reindex ``df`` to the exact sessions in ``calendar``.

    Every ``(session, id)`` pair from the calendar × ``ids`` cross-product is
    present in the output.  Rows in ``df`` that fall on non-session dates are
    dropped; sessions missing from ``df`` receive ``null`` values.

    Parameters
    ----------
    df:
        Long-format frame with ``date`` and ``id`` columns.
    calendar:
        ``trading_calendar`` frame.
    ids:
        Universe of asset ids to include in every session row.
    exchange:
        Exchange code.

    Returns
    -------
    pl.DataFrame
        Left-joined: every (session, id) pair present, extra rows dropped.
    """
    session_df = (
        calendar.filter(
            (pl.col("exchange").cast(pl.String) == exchange) & pl.col("is_session")
        )
        .select("date")
        .unique()
        .sort("date")
    )
    id_df = pl.DataFrame({"id": pl.Series(ids, dtype=pl.Int64)})
    grid = session_df.join(id_df, how="cross")

    return grid.join(df, on=["date", "id"], how="left")


def fill_sessions(
    df: pl.DataFrame,
    calendar: pl.DataFrame,
    ids: list[int],
    *,
    exchange: str = "XNYS",
    method: str = "forward",
    value_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Align to calendar sessions then fill gaps in value columns.

    Parameters
    ----------
    df:
        Long-format panel with ``date`` and ``id``.
    calendar:
        ``trading_calendar`` frame.
    ids:
        Asset ids to include.
    exchange:
        Exchange code.
    method:
        ``"forward"`` (last-observation-carried-forward), ``"backward"``, or
        ``"zero"`` (fill with 0.0).
    value_cols:
        Columns to fill.  When ``None`` all non-key columns are filled.

    Returns
    -------
    pl.DataFrame
        Panel aligned to calendar sessions with gaps filled.
    """
    aligned = align_to_calendar(df, calendar, ids, exchange=exchange)

    if value_cols is None:
        value_cols = [c for c in aligned.columns if c not in ("date", "id")]

    if method == "forward":
        # sort so fill_null(strategy="forward") walks time correctly per id
        aligned = aligned.sort("id", "date")
        aligned = aligned.with_columns(
            pl.col(c).forward_fill().over("id") for c in value_cols
        )
    elif method == "backward":
        aligned = aligned.sort("id", "date")
        aligned = aligned.with_columns(
            pl.col(c).backward_fill().over("id") for c in value_cols
        )
    elif method == "zero":
        aligned = aligned.with_columns(
            pl.col(c).fill_null(0.0) for c in value_cols
        )
    else:
        raise ValueError(f"Unknown fill method {method!r}; expected 'forward', 'backward', or 'zero'")

    return aligned.sort("date", "id")
