from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from scipy import stats

from backtest.signals import SignalFrame
from etl.source import to_float, to_matrix

from .coverage import apply_min_coverage, pairwise_mask
from .newey_west import newey_west_tstat

ICMethod = Literal["rank", "pearson", "kendall"]


def _spearman_ic_rows(
    S: np.ndarray,
    R: np.ndarray,
    min_obs: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-row Spearman IC between signal matrix S and return matrix R.

    S and R are (n_dates, n_assets) dense matrices (may contain NaN).
    Returns (ic_array, n_valid_array) both of length n_dates.

    Fast paths (in order):
    1. No NaN anywhere — fully vectorized with scipy.stats.rankdata(axis=1).
    2. Uniform NaN column pattern across all rows — extract valid columns once,
       fully vectorized.
    3. Row-homogeneous pattern — each row is either entirely valid (all assets
       finite) or entirely NaN (no valid assets).  Common when forward returns
       are missing only at the tail (last N dates).  Extract the valid-row
       sub-matrix and batch-rank it, skipping the all-NaN rows.
    Fallback: per-date loop for irregular NaN patterns.

    Tie-breaking follows scipy.stats.rankdata default ("average"), which is the
    same method scipy.stats.spearmanr uses internally, so results are numerically
    identical to the previous per-date spearmanr calls.
    """
    n_dates, n_assets = S.shape
    mask = np.isfinite(S) & np.isfinite(R)
    n_valid = mask.sum(axis=1)

    def _vectorized_pearson_on_ranks(Rx: np.ndarray, Ry: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            rx_c = Rx - Rx.mean(axis=1, keepdims=True)
            ry_c = Ry - Ry.mean(axis=1, keepdims=True)
            numer = (rx_c * ry_c).sum(axis=1)
            denom = np.sqrt((rx_c**2).sum(axis=1) * (ry_c**2).sum(axis=1))
            return np.where(denom > 0, numer / denom, np.nan)

    if mask.all():
        # Fast path 1: no NaN anywhere — fully vectorized.
        Rx = stats.rankdata(S, axis=1)
        Ry = stats.rankdata(R, axis=1)
        ics = _vectorized_pearson_on_ranks(Rx, Ry)
        ics = np.where(n_valid >= min_obs, ics, np.nan)
        return ics, n_valid.astype(int)

    first_row = mask[0]
    if (mask == first_row).all():
        # Fast path 2: uniform NaN pattern across all rows — extract valid columns
        # and batch-rank only those, then vectorized Pearson.
        Sv = S[:, first_row]
        Rv = R[:, first_row]
        if Sv.shape[1] == 0:
            # No valid columns at all — every date is NaN.
            return np.full(n_dates, np.nan), n_valid.astype(int)
        Rx = stats.rankdata(Sv, axis=1)
        Ry = stats.rankdata(Rv, axis=1)
        ics = _vectorized_pearson_on_ranks(Rx, Ry)
        ics = np.where(n_valid >= min_obs, ics, np.nan)
        return ics, n_valid.astype(int)

    # Fast path 3: row-homogeneous pattern — each row is either entirely valid
    # or entirely NaN (zero valid assets).  Typical when forward returns carry
    # NaN only in the last N dates (one NaN per asset × N tail dates).
    # Extract the fully-valid rows, batch-rank them, and scatter back.
    row_all_valid = n_valid == n_assets
    row_all_nan = n_valid == 0
    if (row_all_valid | row_all_nan).all():
        valid_rows = np.where(row_all_valid)[0]
        ics = np.full(n_dates, np.nan)
        if len(valid_rows) > 0:
            Sv = S[valid_rows]
            Rv = R[valid_rows]
            Rx = stats.rankdata(Sv, axis=1)
            Ry = stats.rankdata(Rv, axis=1)
            sub_ics = _vectorized_pearson_on_ranks(Rx, Ry)
            sub_n = n_valid[valid_rows]
            ics[valid_rows] = np.where(sub_n >= min_obs, sub_ics, np.nan)
        return ics, n_valid.astype(int)

    # Fallback: per-date loop for irregular NaN patterns.
    ics = np.full(n_dates, np.nan)
    for t in range(n_dates):
        if n_valid[t] < min_obs:
            continue
        row_mask = mask[t]
        xm = S[t, row_mask]
        ym = R[t, row_mask]
        rx = stats.rankdata(xm)
        ry = stats.rankdata(ym)
        rx_c = rx - rx.mean()
        ry_c = ry - ry.mean()
        denom = float(np.sqrt((rx_c**2).sum() * (ry_c**2).sum()))
        ics[t] = float((rx_c * ry_c).sum() / denom) if denom > 0 else np.nan
    return ics, n_valid.astype(int)


@dataclass(frozen=True)
class ICResult:
    ic_series: pl.DataFrame  # date, ic   — O(n_dates)
    mean_ic: float
    ic_ir: float  # mean_ic / std_ic
    t_stat: float  # Newey-West adjusted


def ic_series(signals: SignalFrame, returns: pl.DataFrame) -> pl.DataFrame:
    """IC of signal_t against forward return_{t+1}, one value per date.

    Continuous signals → Spearman rank correlation.
    Categorical/binary signals → point-biserial (Pearson on the 0/1 signal).
    Working set is O(n_assets) per date; output is O(n_dates).
    """
    S, _ = to_matrix(signals.df, "signal")
    R, dates = to_matrix(returns, "return")
    n_dates, _ = S.shape

    out_dates: list = []
    ics: list[float] = []
    for t in range(n_dates - 1):
        x = S[t]
        y = R[t + 1]
        if signals.is_categorical:
            ic = float(stats.pearsonr(x, y).statistic)
        else:
            ic = float(stats.spearmanr(x, y).statistic)
        out_dates.append(dates[t])
        ics.append(ic)

    return pl.DataFrame({"date": out_dates, "ic": ics})


def rolling_ic(ic: pl.DataFrame, window: int) -> pl.DataFrame:
    """Trailing-window mean IC. O(n_dates)."""
    return ic.with_columns(
        pl.col("ic").rolling_mean(window_size=window).alias("rolling_ic")
    ).select("date", "rolling_ic")


@dataclass(frozen=True)
class ICEvaluator:
    def evaluate(self, signals: SignalFrame, returns: pl.DataFrame) -> ICResult:
        ic = ic_series(signals, returns)
        s = ic["ic"]
        mean_ic = to_float(s.mean())
        std_ic = to_float(s.std() or 0.0)
        return ICResult(
            ic_series=ic,
            mean_ic=mean_ic,
            ic_ir=(mean_ic / std_ic if std_ic else 0.0),
            t_stat=newey_west_tstat(s),
        )


# ---------------------------------------------------------------------------
# Configurable horizon + method IC
# ---------------------------------------------------------------------------


def _cross_sectional_ic(
    x: np.ndarray,
    y: np.ndarray,
    method: ICMethod,
) -> tuple[float, int]:
    """Compute one cross-sectional IC value with pairwise-complete masking.

    Returns (ic, n_valid) where n_valid is the number of paired observations
    used.  Returns (nan, 0) when fewer than 2 valid pairs exist.

    Spearman / Kendall are monotone-rank correlations and measure whether the
    *ordering* of the signal predicts the *ordering* of returns, which is what
    a long-short book cares about.  Pearson measures linear co-movement and is
    appropriate when both signal and return are expected to be linearly related
    (e.g. a beta-adjusted expected-return estimate).
    """
    mask = pairwise_mask(x, y)
    n = int(mask.sum())
    if n < 2:
        return float("nan"), 0
    xm, ym = x[mask], y[mask]
    if method == "rank":
        ic = float(stats.spearmanr(xm, ym).statistic)
    elif method == "pearson":
        ic = float(stats.pearsonr(xm, ym).statistic)
    else:  # kendall
        ic = float(stats.kendalltau(xm, ym).statistic)
    return ic, n


def _ic_series_from_matrices(
    S: np.ndarray,
    s_dates: list,
    R: np.ndarray,
    r_dates: list,
    method: ICMethod,
    min_obs: int,
) -> pl.DataFrame:
    """Compute ic_series_v2 output given pre-computed dense matrices.

    This internal helper is called by ``ic_series_v2`` and by
    ``ic_horizon_curve`` (which pre-builds the signals matrix once and passes
    different return matrices for each horizon, avoiding redundant pivots).

    For the "rank" method the implementation uses the vectorized
    ``_spearman_ic_rows`` path; other methods fall back to the per-date loop
    over ``_cross_sectional_ic``.
    """
    s_set = {d: i for i, d in enumerate(s_dates)}
    r_set = {d: i for i, d in enumerate(r_dates)}
    common = sorted(set(s_set) & set(r_set))

    if not common:
        return pl.DataFrame({"date": [], "ic": [], "n_obs": []}).with_columns(
            pl.col("ic").cast(pl.Float64),
            pl.col("n_obs").cast(pl.Int32),
        )

    s_idx = [s_set[d] for d in common]
    r_idx = [r_set[d] for d in common]
    Sc = S[s_idx]
    Rc = R[r_idx]

    if method == "rank":
        ic_arr, ns_arr = _spearman_ic_rows(Sc, Rc, min_obs=min_obs)
        # apply_min_coverage already applied inside _spearman_ic_rows via the
        # min_obs guard, so ic_arr entries below threshold are already NaN.
        # We still call apply_min_coverage for the fallback path correctness but
        # it is a no-op when _spearman_ic_rows handled it.
    else:
        ics: list[float] = []
        ns: list[int] = []
        for t in range(len(common)):
            ic, n = _cross_sectional_ic(Sc[t], Rc[t], method)
            ics.append(ic)
            ns.append(n)
        ns_arr = np.array(ns, dtype=int)
        ic_arr = apply_min_coverage(np.array(ics), ns_arr, min_obs)

    ic_list: list[float | None] = [None if np.isnan(v) else float(v) for v in ic_arr]
    return pl.DataFrame({"date": common, "ic": ic_list, "n_obs": ns_arr.tolist()})


def ic_series_v2(
    signals: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    signal_col: str = "signal",
    return_col: str,
    method: ICMethod = "rank",
    min_obs: int = 10,
) -> pl.DataFrame:
    """Per-date IC of a signal against a chosen forward-return column.

    Unlike the original ``ic_series``, this function:
    - Accepts raw long-format DataFrames (date, id, signal) and
      (date, id, <return_col>) rather than a ``SignalFrame``.
    - Takes an explicit ``return_col`` so any forward horizon can be used.
    - Supports ``method`` ∈ {"rank", "pearson", "kendall"} explicitly.
    - Applies pairwise-complete masking and ``min_obs`` suppression.

    Returns a DataFrame with columns ``(date, ic, n_obs)``.
    """
    S, s_dates = to_matrix(signals.select("date", "id", signal_col), signal_col)
    R, r_dates = to_matrix(forward_returns.select("date", "id", return_col), return_col)
    return _ic_series_from_matrices(S, s_dates, R, r_dates, method, min_obs)
