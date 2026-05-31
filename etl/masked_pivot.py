"""Explicit missing-data policy via masked pivot.

Feature 6 from the plan: the existing ``to_matrix`` pivot in ``source.py``
silently produces ``NaN`` for any ``(date, id)`` pair absent from the long
frame.  Downstream code then operates on NaN-contaminated matrices without
knowing which cells are structurally missing vs genuinely zero.

``to_masked_matrix`` returns *both* the dense matrix and a boolean validity
mask so callers can make an explicit choice (fill, drop, impute) rather than
having the NaN propagate silently through matrix algebra.

The mask convention: ``mask[t, j] = True`` means cell ``(t, j)`` has a real
observed value; ``False`` means it was absent in the source frame.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl


def to_masked_matrix(
    df: pl.DataFrame,
    value_col: str,
) -> tuple[np.ndarray, np.ndarray, list[date], list[int]]:
    """Pivot long ``(date, id, value)`` to a dense matrix plus a validity mask.

    Unlike ``source.to_matrix``, this function:
    - Builds the full cross-product of observed dates × ids so the output is
      always dense (no ragged columns).
    - Returns a boolean ``mask`` alongside the value array so callers know
      which cells were actually observed.
    - Replaces missing values with ``0.0`` in the matrix (not NaN) to avoid
      silent NaN propagation; callers should gate on the mask before using
      any cell.

    Parameters
    ----------
    df:
        Long-format frame with columns ``date`` (Date), ``id`` (Int64), and
        ``value_col`` (Float64 or compatible).

    Returns
    -------
    matrix : np.ndarray, shape (n_dates, n_ids), float64
        Dense value array; missing cells are 0.0.
    mask : np.ndarray, shape (n_dates, n_ids), bool
        ``True`` where the corresponding matrix cell has a real value.
    dates : list[date]
        Row axis, ascending.
    ids : list[int]
        Column axis, ascending.
    """
    # Determine the full axis from what's observed.
    dates: list[date] = sorted(df["date"].unique().to_list())
    ids: list[int] = sorted(df["id"].unique().to_list())

    # Polars pivot produces the dense wide frame in Rust — no Python loop over
    # rows.  Columns are labelled by the string representation of each id value,
    # in the order they appear after sorting by (date, id).  We sort before
    # pivoting so the id column order is ascending (matching `ids`).
    wide = (
        df.sort(["date", "id"])
        .pivot(on="id", index="date", values=value_col, aggregate_function="first")
        .sort("date")
    )

    # Extract the numeric block; missing (date, id) pairs become NaN in the
    # pivot output.  Build the mask before zeroing them out.
    mat_np = wide.drop("date").to_numpy()
    mask = ~np.isnan(mat_np)
    np.nan_to_num(mat_np, copy=False, nan=0.0)

    return mat_np, mask, dates, ids
