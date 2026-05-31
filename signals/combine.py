"""Signal combination and orthogonalization.

When multiple signals are available, naive averaging dilutes idiosyncratic
information.  This module provides three combination strategies:

1. **Z-score blend** — standardize each signal cross-sectionally (mean 0, std
   1) then take a simple or weighted average.  Handles scale differences but
   does not remove redundancy.

2. **IC-weighted blend** — weight each signal's contribution by its historical
   mean IC.  Signals with better track records get more weight.  The final
   composite is z-scored cross-sectionally.

3. **Gram-Schmidt orthogonalization** — orthogonalize the signal matrix so
   each component is uncorrelated with all previous ones (in the full panel
   sense, i.e. across all dates and assets stacked).  The first signal is kept
   as-is; each subsequent signal is residualized against all previous ones.
   This is the ``incremental IC`` decomposition: the marginal contribution of
   each signal once the previous signals are already in the book.

Additionally, ``incremental_ic`` evaluates the gain from adding each signal
one at a time to a baseline composite, measuring how much IC improves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from etl.source import to_matrix

from .ic import ICMethod, ic_series_v2


def _zscore_cross_section(mat: np.ndarray) -> np.ndarray:
    """Z-score each row (date) independently, ignoring NaN.

    Returns a matrix of the same shape with NaN preserved where the input had
    NaN.  Dates with zero std are left as NaN.
    """
    out = np.full_like(mat, np.nan)
    for t in range(mat.shape[0]):
        row = mat[t]
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        mu = row[mask].mean()
        sd = row[mask].std()
        if sd > 0:
            out[t, mask] = (row[mask] - mu) / sd
    return out


def _long_to_matrix(
    signals_list: list[pl.DataFrame], signal_col: str
) -> tuple[list[np.ndarray], list, list[int]]:
    """Convert a list of signal DataFrames to aligned matrices.

    Returns (matrices, common_dates, common_ids) where all matrices are
    (n_dates, n_assets) with the same date/id alignment.
    """
    # Find common dates and ids across all signal frames
    all_dates: list[set] = []
    all_ids: list[set] = []
    mats: list[tuple[np.ndarray, list, list[int]]] = []

    for df in signals_list:
        mat, dates = to_matrix(df.select("date", "id", signal_col), signal_col)
        ids = sorted(df["id"].unique().to_list())
        all_dates.append(set(dates))
        all_ids.append(set(ids))
        mats.append((mat, dates, ids))

    common_dates = sorted(set.intersection(*all_dates))
    common_ids = sorted(set.intersection(*all_ids))

    aligned: list[np.ndarray] = []
    for mat, dates, ids in mats:
        d_idx = {d: i for i, d in enumerate(dates)}
        id_idx = {v: i for i, v in enumerate(ids)}
        rows = [d_idx[d] for d in common_dates]
        cols = [id_idx[aid] for aid in common_ids]
        aligned.append(mat[np.ix_(rows, cols)])

    return aligned, common_dates, common_ids


def _matrix_to_long(
    mat: np.ndarray,
    dates: list,
    ids: list[int],
    signal_col: str = "signal",
) -> pl.DataFrame:
    """Convert (n_dates, n_assets) matrix back to long format (date, id, signal)."""
    rows = []
    for ti, d in enumerate(dates):
        for ai, aid in enumerate(ids):
            rows.append({"date": d, "id": aid, signal_col: float(mat[ti, ai])})
    return pl.DataFrame(rows).with_columns(
        pl.col("id").cast(pl.Int64),
        pl.col(signal_col).cast(pl.Float64),
    )


def zscore_blend(
    signals_list: list[pl.DataFrame],
    weights: list[float] | None = None,
    *,
    signal_col: str = "signal",
) -> pl.DataFrame:
    """Weighted average of z-scored signals.

    Each signal is standardized cross-sectionally before blending.  Equal
    weights are used when ``weights`` is None.

    Parameters
    ----------
    signals_list:
        One DataFrame per signal, each long-format (date, id, signal_col).
    weights:
        Per-signal blend weights (need not sum to 1; they are normalized).
        If None, equal weights are used.
    signal_col:
        The signal column name (same in all input frames).

    Returns
    -------
    Long-format (date, id, signal) of the blended composite.
    """
    if not signals_list:
        raise ValueError("signals_list must be non-empty")
    n = len(signals_list)
    w = np.array(weights, dtype=float) if weights is not None else np.ones(n)
    w = w / w.sum()

    aligned, dates, ids = _long_to_matrix(signals_list, signal_col)
    z_mats = [_zscore_cross_section(m) for m in aligned]

    composite = np.zeros_like(z_mats[0])
    total_weight = np.zeros_like(z_mats[0])
    for i, zm in enumerate(z_mats):
        finite_mask = np.isfinite(zm)
        composite[finite_mask] += w[i] * zm[finite_mask]
        total_weight[finite_mask] += w[i]

    # Avoid division by zero; set to NaN where no signal contributed
    with np.errstate(invalid="ignore"):
        out = np.where(total_weight > 0, composite / total_weight, np.nan)

    return _matrix_to_long(out, dates, ids, signal_col)


def ic_weighted_blend(
    signals_list: list[pl.DataFrame],
    mean_ics: list[float],
    *,
    signal_col: str = "signal",
) -> pl.DataFrame:
    """IC-weighted blend: signal weights proportional to their mean IC.

    Signals with positive mean IC get positive weight; negative IC signals are
    either excluded (if all ICs are non-positive) or get negative weight so
    their direction is flipped.  Weights are normalized by the sum of absolute
    values of non-zero ICs.

    Parameters
    ----------
    signals_list:
        One DataFrame per signal, each long-format (date, id, signal_col).
    mean_ics:
        Per-signal historical mean IC.  Must be the same length as
        ``signals_list``.
    signal_col:
        The signal column name (same in all input frames).

    Returns
    -------
    Long-format (date, id, signal) of the IC-weighted composite.
    """
    ic_arr = np.array(mean_ics, dtype=float)
    total = np.abs(ic_arr).sum()
    # Fallback to equal weights when all ICs are zero
    weights = np.ones(len(signals_list)) / len(signals_list) if total == 0 else ic_arr / total
    return zscore_blend(signals_list, weights=weights.tolist(), signal_col=signal_col)


def gram_schmidt_orthogonalize(
    signals_list: list[pl.DataFrame],
    *,
    signal_col: str = "signal",
) -> list[pl.DataFrame]:
    """Gram-Schmidt orthogonalization of signal matrices.

    The first signal is returned unchanged (z-scored cross-sectionally).
    Each subsequent signal is residualized against all previous orthogonalized
    signals, then z-scored, so the resulting set is mutually uncorrelated
    across the full (stacked) panel.

    The orthogonalization is done in the flattened (date * asset) space:
    stack all (date, asset) pairs into a single vector and orthogonalize
    there.  NaN cells are excluded from the projection but receive NaN in
    the orthogonalized output.

    Returns
    -------
    List of orthogonalized signal DataFrames, same length as input.
    """
    if not signals_list:
        return []

    aligned, dates, ids = _long_to_matrix(signals_list, signal_col)

    # Z-score each signal and flatten to 1-D for the GS procedure
    z_mats = [_zscore_cross_section(m) for m in aligned]
    shape = z_mats[0].shape

    ortho_flat: list[np.ndarray] = []
    results: list[pl.DataFrame] = []

    for _i, zm in enumerate(z_mats):
        flat = zm.reshape(-1).copy()
        # Project out all previous orthogonalized components
        for prev in ortho_flat:
            # Only project where both current and previous are finite
            joint_ok = np.isfinite(flat) & np.isfinite(prev)
            if joint_ok.sum() < 2:
                continue
            dot_pp = float(prev[joint_ok] @ prev[joint_ok])
            if dot_pp == 0:
                continue
            proj_coef = float(prev[joint_ok] @ flat[joint_ok]) / dot_pp
            flat[joint_ok] -= proj_coef * prev[joint_ok]

        # Re-z-score the residual cross-sectionally
        residual_mat = flat.reshape(shape)
        z_resid = _zscore_cross_section(residual_mat)
        ortho_flat.append(z_resid.reshape(-1))
        results.append(_matrix_to_long(z_resid, dates, ids, signal_col))

    return results


@dataclass(frozen=True)
class IncrementalICResult:
    """Incremental IC contribution of each signal."""

    signal_index: int
    cumulative_ic_mean: float
    """Mean IC of the composite that includes signals 0..signal_index."""
    incremental_ic_mean: float
    """Mean IC added by including this signal (cumulative[i] − cumulative[i-1])."""


def incremental_ic(
    signals_list: list[pl.DataFrame],
    forward_returns: pl.DataFrame,
    *,
    signal_col: str = "signal",
    return_col: str,
    method: ICMethod = "rank",
    min_obs: int = 10,
) -> list[IncrementalICResult]:
    """Measure the marginal IC contribution of adding each signal to the composite.

    Builds cumulative composites: composite_1 = signal_0; composite_2 =
    blend(signal_0, signal_1); … and records the mean IC at each step.

    Returns one IncrementalICResult per signal in signals_list.
    """
    prev_mean_ic = 0.0
    results: list[IncrementalICResult] = []

    for i in range(1, len(signals_list) + 1):
        subset = signals_list[:i]
        composite = zscore_blend(subset, signal_col=signal_col)
        ic_df = ic_series_v2(
            composite,
            forward_returns,
            signal_col=signal_col,
            return_col=return_col,
            method=method,
            min_obs=min_obs,
        )
        arr = ic_df["ic"].drop_nulls().to_numpy()
        cum_mean = float(np.nanmean(arr)) if len(arr) > 0 else float("nan")
        results.append(
            IncrementalICResult(
                signal_index=i - 1,
                cumulative_ic_mean=cum_mean,
                incremental_ic_mean=cum_mean - prev_mean_ic,
            )
        )
        prev_mean_ic = cum_mean

    return results
