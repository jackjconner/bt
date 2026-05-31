"""Periodic return tables: monthly, quarterly, annual.

Why this file: periodizing requires grouping by calendar period (year, month,
quarter) and compounding daily returns — a different operation from rolling
windows or cross-sectional metrics. Keeping it separate makes the logic easy
to extend (e.g. weekly or custom periods) without touching other files.

Returns compound to avoid the arithmetic-mean distortion that compounds over
multi-day periods. A month with days [+1%, -1%, +1%] has a compounded return
of (1.01)(0.99)(1.01) - 1, not the arithmetic sum.

All functions accept a `returns` DataFrame with `(date, return_1d)` in
fractional units and return wide-format DataFrames suitable for display.
"""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Monthly returns table
# ---------------------------------------------------------------------------


def monthly_returns(returns: pl.DataFrame) -> pl.DataFrame:
    """Compound monthly returns.

    Returns a long DataFrame `(year, month, monthly_return)` sorted by year
    then month. Callers can pivot on `month` to get the classic heatmap shape.
    """
    with_ym = returns.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    )
    return (
        with_ym.group_by(["year", "month"])
        .agg(((1.0 + pl.col("return_1d")).product() - 1.0).alias("monthly_return"))
        .sort(["year", "month"])
    )


def monthly_returns_wide(returns: pl.DataFrame) -> pl.DataFrame:
    """Monthly returns in wide format: rows = years, columns = months 1–12.

    Missing months (e.g. first/last partial year) appear as null. This is the
    standard presentation for a tear-sheet return heatmap.
    """
    long = monthly_returns(returns)
    wide = long.pivot(on="month", index="year", values="monthly_return").sort("year")
    # Ensure columns 1–12 exist even if the data doesn't span all months
    for m in range(1, 13):
        if str(m) not in wide.columns:
            wide = wide.with_columns(pl.lit(None).cast(pl.Float64).alias(str(m)))
    col_order = ["year"] + [str(m) for m in range(1, 13)]
    existing = [c for c in col_order if c in wide.columns]
    return wide.select(existing)


# ---------------------------------------------------------------------------
# Quarterly returns table
# ---------------------------------------------------------------------------


def quarterly_returns(returns: pl.DataFrame) -> pl.DataFrame:
    """Compound quarterly returns.

    Returns `(year, quarter, quarterly_return)` where `quarter` is 1–4.
    """
    with_yq = returns.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.quarter().alias("quarter"),
    )
    return (
        with_yq.group_by(["year", "quarter"])
        .agg(((1.0 + pl.col("return_1d")).product() - 1.0).alias("quarterly_return"))
        .sort(["year", "quarter"])
    )


# ---------------------------------------------------------------------------
# Annual returns table
# ---------------------------------------------------------------------------


def annual_returns(returns: pl.DataFrame) -> pl.DataFrame:
    """Compound annual returns.

    Returns `(year, annual_return)`. Partial years (first/last) are included
    but their return is the compound of only the available sessions.
    """
    with_y = returns.with_columns(pl.col("date").dt.year().alias("year"))
    return (
        with_y.group_by("year")
        .agg(((1.0 + pl.col("return_1d")).product() - 1.0).alias("annual_return"))
        .sort("year")
    )
