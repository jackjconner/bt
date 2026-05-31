"""Lazy / streaming Polars formulation of cross-sectional rank IC.

The incumbent ``ic.py`` path computes Spearman IC by pivoting the long
``(date, id, value)`` frames into dense ``(n_dates, n_assets)`` numpy
matrices and ranking each row with ``scipy.stats.rankdata``.  Two costs
dominate that path at scale: the repeated ``df.sort(...).pivot(...)``
allocations (one per signal, one per return horizon) and the per-row
``rankdata`` calls.

This module computes the *same* per-date Spearman IC without ever
materialising a dense matrix.  Spearman IC on one date is Pearson
correlation of the within-date ranks, so it can be expressed entirely as
Polars window + group-by aggregations over the long frame:

1. inner-join signal to a chosen return column, keeping only rows where
   *both* are finite (pairwise-complete masking, per date);
2. rank the signal and the return *within each date* with
   ``pl.col(...).rank().over("date")`` — Polars' rank default is the
   "average" tie method, identical to ``scipy.stats.rankdata``;
3. reduce each date to (n_obs, centred-rank covariance, the two centred-rank
   variances) and form ``cov / sqrt(vs * vr)``.

The ranks never leave Polars' native (Rust) engine, the join replaces the
pivot, and the whole computation runs lazily so the optimiser fuses the
filter / rank / aggregate.  The result is **bit-identical** to the matrix
path (verified in ``test_lazy_ic.py``), so it is the default for the
``"rank"`` method; ``engine="matrix"`` selects the incumbent path.

Pairwise masking note: for a horizon whose forward return is NaN on the
trailing dates, those rows drop out of the join, so the signal is ranked
over exactly the assets that survive the mask — matching the matrix path's
per-date ``pairwise_mask``.  Dates with no surviving rows are re-attached as
null-IC rows so the output date axis matches the matrix path exactly.
"""

from __future__ import annotations

from typing import cast

import polars as pl


def _finite(col: str) -> pl.Expr:
    """Rows where ``col`` is a finite, non-null number."""
    c = pl.col(col)
    return c.is_not_null() & c.is_finite()


def spearman_ic_lazy(
    signals: pl.DataFrame,
    returns: pl.DataFrame,
    *,
    signal_col: str,
    return_col: str,
    min_obs: int,
) -> pl.DataFrame:
    """Per-date Spearman IC of ``signal_col`` against ``return_col``.

    Returns a ``(date, ic, n_obs)`` DataFrame whose date axis is the
    intersection of the signal and return date axes (a date present on both
    sides but with no pairwise-complete asset is emitted with a null IC and
    ``n_obs = 0``), sorted ascending — matching ``_ic_series_from_matrices``.
    """
    sig = signals.select("date", "id", signal_col).lazy()
    ret = returns.select("date", "id", return_col).lazy()

    # Full date intersection (incl. dates that survive no pairwise mask), so the
    # output axis matches the matrix path which iterates over `common` dates.
    common_dates = (
        sig.select("date").unique().join(ret.select("date").unique(), on="date", how="inner")
    )

    paired = (
        sig.join(ret, on=["date", "id"], how="inner")
        .filter(_finite(signal_col) & _finite(return_col))
        .with_columns(
            pl.col(signal_col).rank().over("date").alias("_rs"),
            pl.col(return_col).rank().over("date").alias("_rr"),
        )
    )

    rs_c = pl.col("_rs") - pl.col("_rs").mean()
    rr_c = pl.col("_rr") - pl.col("_rr").mean()
    agg = (
        paired.group_by("date")
        .agg(
            pl.len().alias("n_obs"),
            (rs_c * rr_c).sum().alias("_cov"),
            (rs_c**2).sum().alias("_vs"),
            (rr_c**2).sum().alias("_vr"),
        )
        .with_columns(
            pl.when((pl.col("n_obs") >= min_obs) & (pl.col("_vs") > 0) & (pl.col("_vr") > 0))
            .then(pl.col("_cov") / (pl.col("_vs") * pl.col("_vr")).sqrt())
            .otherwise(None)
            .alias("ic")
        )
        .select("date", "ic", "n_obs")
    )

    out = (
        common_dates.join(agg, on="date", how="left")
        .with_columns(
            pl.col("ic").cast(pl.Float64),
            pl.col("n_obs").fill_null(0).cast(pl.Int64),
        )
        .sort("date")
        .collect(engine="in-memory")
    )
    # collect() is typed as `InProcessQuery | DataFrame` for the background-mode
    # overload; narrow it like etl.loader does — same cast, not a blanket ignore.
    return cast(pl.DataFrame, out)
