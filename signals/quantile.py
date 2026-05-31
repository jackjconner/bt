"""Quantile / decile spread analysis and monotonicity scoring.

A long-short book is built by going long the top quantile and short the
bottom quantile of the signal cross-section.  The spread between those
buckets, averaged over time, is the most direct measure of what the signal
would *earn* net of factor exposures (before costs).

Monotonicity score: if the signal predicts returns linearly, average return
should rise monotonically from bucket 1 to bucket n_quantiles.  We measure
this with the Spearman rank correlation between bucket rank and mean return
across the n_quantiles buckets — a perfect step-up gives +1, a purely
inverted ordering gives -1.  Values near 0 indicate a non-monotone
relationship (e.g., only the tails work), which is still useful but harder
to express in a linear long-short book.

NaN / coverage: assets missing either signal or forward return are excluded
from bucket assignment for that date; the denominator of each bucket mean is
the number of valid assets in that bucket on that date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

from etl.source import to_matrix


@dataclass(frozen=True)
class QuantileResult:
    """Results from quantile-spread analysis."""

    bucket_returns: pl.DataFrame
    """Mean forward return per (bucket, date) — long format (bucket, date, mean_ret)."""

    mean_by_bucket: pl.DataFrame
    """Aggregate mean return per bucket across all dates — (bucket, mean_ret)."""

    spread: float
    """Top-bucket mean return minus bottom-bucket mean return (top − bottom).

    A positive spread means the top quantile outperformed the bottom on average.
    """

    spread_ir: float
    """Spread IR: mean of the per-date (top − bottom) return series divided by
    its standard deviation.  Analogous to IC-IR; captures consistency."""

    monotonicity_score: float
    """Spearman r between bucket rank (1..n_quantiles) and mean return per bucket.
    +1 = perfectly monotone rising, −1 = perfectly monotone falling."""

    n_quantiles: int


def _quantile_spread_rows_vectorized(
    Sc: np.ndarray,
    Rc: np.ndarray,
    common: list,
    n_quantiles: int,
    min_obs: int,
) -> tuple[list, list[float]]:
    """Vectorized bucket computation for row-homogeneous arrays.

    Each row in Sc / Rc is either entirely finite (all n_assets valid) or
    entirely NaN — typical when forward returns have NaN only at the tail.
    Processes the fully-valid rows with a single batch ``rankdata`` call and
    NumPy bin-sums instead of a per-date Python loop.

    Returns (rows, spreads) matching the per-date loop's output.
    """
    _, n_assets = Sc.shape
    mask = np.isfinite(Sc) & np.isfinite(Rc)
    n_valid = mask.sum(axis=1)
    valid_rows = np.where(n_valid == n_assets)[0]

    if len(valid_rows) == 0:
        return [], []

    Sv = Sc[valid_rows]  # (n_valid_dates, n_assets)
    Rv = Rc[valid_rows]

    # One batch rankdata call replaces n_valid_dates individual calls.
    Rx = stats.rankdata(Sv, axis=1)

    # Map ranks to 1-based buckets in [1, n_quantiles].
    buckets = np.ceil(Rx / n_assets * n_quantiles).astype(np.intp)
    np.clip(buckets, 1, n_quantiles, out=buckets)

    # Per-bucket sum and count using a broadcasting mask over all buckets at once.
    b_range = np.arange(1, n_quantiles + 1, dtype=np.intp)  # (n_quantiles,)
    b_mask = buckets[:, :, np.newaxis] == b_range  # (n_vd, n_assets, n_quantiles)
    bucket_counts = b_mask.sum(axis=1)  # (n_vd, n_quantiles)
    bucket_sums = (Rv[:, :, np.newaxis] * b_mask).sum(axis=1)  # (n_vd, n_quantiles)

    with np.errstate(invalid="ignore"):
        bucket_means = np.where(bucket_counts > 0, bucket_sums / bucket_counts, np.nan)

    # Build spreads array (top-bucket mean − bottom-bucket mean).
    top_ret = bucket_means[:, n_quantiles - 1]
    bot_ret = bucket_means[:, 0]
    both_finite = np.isfinite(top_ret) & np.isfinite(bot_ret)
    spreads = (top_ret[both_finite] - bot_ret[both_finite]).tolist()

    # Build the row list matching the per-date loop's dict structure.
    # Tile dates and bucket indices for fast construction.
    valid_dates = [common[i] for i in valid_rows]
    rows: list[dict] = []
    for i, d in enumerate(valid_dates):
        for b in range(n_quantiles):
            rows.append(
                {
                    "date": d,
                    "bucket": b + 1,
                    "mean_ret": float(bucket_means[i, b]),
                    "n_assets": int(bucket_counts[i, b]),
                }
            )

    return rows, spreads


def quantile_spread(
    signals: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    signal_col: str = "signal",
    return_col: str,
    n_quantiles: int = 5,
    min_obs: int = 10,
) -> QuantileResult:
    """Bucket assets by signal rank each date and track forward-return per bucket.

    Parameters
    ----------
    signals:
        Long-format (date, id, signal_col).
    forward_returns:
        Long-format (date, id, return_col).
    signal_col:
        Column in ``signals`` containing the numeric signal value.
    return_col:
        Column in ``forward_returns`` containing the forward return.
    n_quantiles:
        Number of equal-width quantile buckets (5 = quintiles, 10 = deciles).
    min_obs:
        Minimum valid assets per date to include that date.

    Returns
    -------
    QuantileResult
    """
    S, s_dates = to_matrix(signals.select("date", "id", signal_col), signal_col)
    R, r_dates = to_matrix(forward_returns.select("date", "id", return_col), return_col)

    s_map = {d: i for i, d in enumerate(s_dates)}
    r_map = {d: i for i, d in enumerate(r_dates)}
    common = sorted(set(s_map) & set(r_map))

    if not common:
        bucket_returns = pl.DataFrame(
            {"date": [], "bucket": [], "mean_ret": [], "n_assets": []}
        ).with_columns(
            pl.col("bucket").cast(pl.Int32),
            pl.col("mean_ret").cast(pl.Float64),
            pl.col("n_assets").cast(pl.Int32),
        )
        return QuantileResult(
            bucket_returns=bucket_returns,
            mean_by_bucket=pl.DataFrame({"bucket": [], "mean_ret": []}).with_columns(
                pl.col("bucket").cast(pl.Int32), pl.col("mean_ret").cast(pl.Float64)
            ),
            spread=float("nan"),
            spread_ir=0.0,
            monotonicity_score=float("nan"),
            n_quantiles=n_quantiles,
        )

    s_idx = [s_map[d] for d in common]
    r_idx = [r_map[d] for d in common]
    Sc = S[s_idx]
    Rc = R[r_idx]

    n_dates, n_assets = Sc.shape
    mask = np.isfinite(Sc) & np.isfinite(Rc)
    n_valid = mask.sum(axis=1)
    row_all_valid = n_valid == n_assets
    row_all_nan = n_valid == 0

    rows: list[dict]
    spreads: list[float]

    if (row_all_valid | row_all_nan).all():
        # Fast path: row-homogeneous — each date is either fully valid or fully
        # NaN.  Use a single batch rankdata call plus NumPy bin-sums instead of
        # a per-date Python loop over rankdata + dict building.
        rows, spreads = _quantile_spread_rows_vectorized(Sc, Rc, common, n_quantiles, min_obs)
    else:
        # Fallback: irregular NaN patterns — per-date loop.
        rows = []
        spreads = []
        for ti in range(n_dates):
            x = Sc[ti]
            y = Rc[ti]
            m = mask[ti]
            if m.sum() < min_obs:
                continue
            xm, ym = x[m], y[m]

            # Assign quantile buckets via rank-based cut (equal-count)
            ranks = stats.rankdata(xm, method="average")
            # Map ranks to 1..n_quantiles buckets (lower rank → lower bucket)
            buckets = np.ceil(ranks / len(ranks) * n_quantiles).astype(int)
            buckets = np.clip(buckets, 1, n_quantiles)

            date_spreads: dict[int, list[float]] = {b: [] for b in range(1, n_quantiles + 1)}
            for b, ret in zip(buckets, ym, strict=False):
                date_spreads[b].append(ret)

            dn = date_spreads[n_quantiles]
            top_ret = float(np.mean(dn)) if dn else float("nan")
            bot_ret = float(np.mean(date_spreads[1])) if date_spreads[1] else float("nan")

            for b in range(1, n_quantiles + 1):
                vals = date_spreads[b]
                rows.append(
                    {
                        "date": common[ti],
                        "bucket": b,
                        "mean_ret": float(np.mean(vals)) if vals else float("nan"),
                        "n_assets": len(vals),
                    }
                )

            if np.isfinite(top_ret) and np.isfinite(bot_ret):
                spreads.append(top_ret - bot_ret)

    if rows:
        bucket_returns = pl.DataFrame(rows).with_columns(pl.col("bucket").cast(pl.Int32))
    else:
        bucket_returns = pl.DataFrame(
            {"date": [], "bucket": [], "mean_ret": [], "n_assets": []}
        ).with_columns(
            pl.col("bucket").cast(pl.Int32),
            pl.col("mean_ret").cast(pl.Float64),
            pl.col("n_assets").cast(pl.Int32),
        )

    # Aggregate across dates
    mean_by_bucket = (
        bucket_returns.filter(pl.col("mean_ret").is_finite())
        .group_by("bucket")
        .agg(pl.col("mean_ret").mean().alias("mean_ret"))
        .sort("bucket")
    )

    spread_arr = np.array(spreads)
    spread_val = float(np.nanmean(spread_arr)) if len(spread_arr) > 0 else float("nan")
    spread_ir_val = (
        float(np.nanmean(spread_arr) / np.nanstd(spread_arr))
        if len(spread_arr) > 1 and np.nanstd(spread_arr) > 0
        else 0.0
    )

    # Monotonicity: Spearman r between bucket rank and mean return
    mbr = mean_by_bucket.sort("bucket")
    if len(mbr) >= 3:
        mono = float(
            stats.spearmanr(
                mbr["bucket"].to_numpy(),
                mbr["mean_ret"].to_numpy(),
            ).statistic
        )
    else:
        mono = float("nan")

    return QuantileResult(
        bucket_returns=bucket_returns,
        mean_by_bucket=mean_by_bucket,
        spread=spread_val,
        spread_ir=spread_ir_val,
        monotonicity_score=mono,
        n_quantiles=n_quantiles,
    )
