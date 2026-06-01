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

    Notes
    -----
    The dense block is filled by scatter, not by a Polars ``pivot``.  A
    ``rank("dense")`` over ``date`` and over ``id`` yields, for every row, its
    ascending-axis row/column index directly (no sort, no per-id wide column
    materialisation); a single flat assignment into a pre-zeroed buffer places
    each value.  This costs one columnar pass plus one numpy scatter instead of
    the pivot's column-by-column wide build, and scales with the *number of
    observations* rather than the *number of asset columns* — the dominant cost
    when the panel is wide.  The output is identical to the pivot path:
    missing cells are ``0.0`` with ``mask`` ``False``; explicit ``NaN`` values
    are likewise treated as missing.  A duplicate ``(date, id)`` key (a hard
    invariant violation that :func:`etl.quality.check` flags) resolves to the
    first such row in input order.
    """
    dates: list[date] = sorted(df["date"].unique().to_list())
    ids: list[int] = sorted(df["id"].unique().to_list())
    n_d, n_i = len(dates), len(ids)

    # Dense rank → 0-based index into the ascending-unique axis, in one Rust
    # pass per column (no sort of the long frame, no wide-column allocation).
    coded = df.select(
        pl.col("date").rank("dense").sub(1).alias("_row"),
        pl.col("id").rank("dense").sub(1).alias("_col"),
        pl.col(value_col).alias("_val"),
    )
    row = coded["_row"].to_numpy()
    col = coded["_col"].to_numpy()
    values = coded["_val"].to_numpy()
    flat = row * n_i + col

    mat = np.zeros(n_d * n_i, dtype=np.float64)
    mask = np.zeros(n_d * n_i, dtype=bool)
    # Reverse assignment so the first row of any duplicate key wins (matches the
    # pivot's "first" aggregate); for the unique-key panels this contract
    # guarantees, order is irrelevant.
    mat[flat[::-1]] = values[::-1]
    # An explicit NaN value is "missing": mark a cell observed only where a real
    # (non-NaN) value landed in it.
    mask[flat[~np.isnan(values)]] = True
    np.nan_to_num(mat, copy=False, nan=0.0)

    return mat.reshape(n_d, n_i), mask.reshape(n_d, n_i), dates, ids
