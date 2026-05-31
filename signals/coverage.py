"""NaN / coverage-aware masking for cross-sectional signal scoring.

Every IC computation must handle real panels with missing data.  The
functions here implement pairwise-complete masking per date (only assets
that have both a signal value *and* a return value contribute) and a
minimum-coverage threshold that suppresses the IC estimate for sparse
dates where the estimate would be unreliable.

Design note: masking is kept separate from the correlation step so the
same helpers can be reused by horizon.py, quantile.py, and neutralize.py
without coupling those modules to each other.
"""

from __future__ import annotations

import numpy as np


def pairwise_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask of positions where both x and y are finite (non-NaN/Inf).

    Returns a 1-D bool array of length len(x) == len(y).
    """
    return np.isfinite(x) & np.isfinite(y)


def apply_min_coverage(
    values: np.ndarray,
    counts: np.ndarray,
    min_obs: int,
) -> np.ndarray:
    """Replace IC values at dates with fewer than min_obs paired observations
    with NaN.

    Parameters
    ----------
    values:
        1-D array of per-date IC values (may already contain NaN).
    counts:
        1-D integer array — number of valid pairs used at each date.
    min_obs:
        Minimum number of non-NaN paired observations required to keep an IC
        estimate.  Dates below this threshold are set to NaN so they do not
        distort mean IC or t-stat calculations.

    Returns
    -------
    values array with low-coverage dates zeroed out to NaN.
    """
    out = values.copy()
    out[counts < min_obs] = np.nan
    return out
