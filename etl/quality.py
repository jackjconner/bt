"""Data-quality checks for ETL panels.

Feature 5 from the plan: a single ``check(df, …)`` call returns a structured
``QualityReport`` that flags:

1. Duplicate ``(date, id)`` keys — a hard invariant violation.
2. Missing sessions vs the trading calendar — gaps that could silently produce
   stale forward-fills or mis-aligned matrices.
3. Frozen / zero-variance series — any asset whose value never changes over
   the observation window is almost certainly a data error (flat price feed).
4. Return-spike outliers — cross-sectional z-scores with |z| above a threshold
   indicate bad ticks or corporate-action contamination.

The report is a ``dataclass`` of plain Polars DataFrames so it's easy to
log, serialize, or assert on in tests.  No side-effects: the function never
modifies the input frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl


@dataclass(frozen=True)
class QualityReport:
    """Structured result of a data-quality scan.

    Each attribute is a DataFrame whose schema is documented inline.  An
    attribute is empty (zero rows) when no issues of that kind were found.
    Callers can check ``report.ok`` to gate downstream processing.
    """

    # (date, id) rows that appear more than once.
    duplicate_keys: pl.DataFrame  # columns: date, id, n_dupes

    # (date, id) pairs that are in the expected cross-product but absent from df.
    missing_sessions: pl.DataFrame  # columns: date, id

    # Assets whose value column has variance == 0 over the whole window.
    frozen_series: pl.DataFrame  # columns: id, n_obs, value_constant

    # (date, id) rows whose value is a cross-sectional outlier.
    spike_outliers: pl.DataFrame  # columns: date, id, value, z_score

    @property
    def ok(self) -> bool:
        """True when no issues were found."""
        return all(
            f.is_empty()
            for f in (
                self.duplicate_keys,
                self.missing_sessions,
                self.frozen_series,
                self.spike_outliers,
            )
        )

    def summary(self) -> str:
        lines = [
            f"duplicate_keys   : {self.duplicate_keys.height}",
            f"missing_sessions : {self.missing_sessions.height}",
            f"frozen_series    : {self.frozen_series.height}",
            f"spike_outliers   : {self.spike_outliers.height}",
        ]
        return "\n".join(lines)


def check(
    df: pl.DataFrame,
    value_col: str,
    *,
    expected_dates: list[date] | None = None,
    expected_ids: list[int] | None = None,
    spike_z_threshold: float = 10.0,
) -> QualityReport:
    """Run all data-quality checks on a long-format ``(date, id, value)`` frame.

    Parameters
    ----------
    df:
        Long-format frame.  Must have ``date`` (Date) and ``id`` (Int64)
        columns plus ``value_col``.
    value_col:
        Name of the numeric column to inspect for frozen series and spikes.
    expected_dates:
        Full set of trading sessions the panel *should* cover.  When ``None``
        the missing-sessions check is skipped.
    expected_ids:
        Full set of asset ids the panel *should* cover.  When ``None`` the
        missing-sessions check is skipped.
    spike_z_threshold:
        Cross-sectional |z-score| above which a value is flagged as an
        outlier.  Default 10 — conservative enough to catch bad ticks while
        not flagging legitimate extreme moves.

    Returns
    -------
    QualityReport
    """
    # ------------------------------------------------------------------ #
    # 1. Duplicate (date, id) keys
    # ------------------------------------------------------------------ #
    dup = (
        df.select("date", "id")
        .group_by("date", "id")
        .len()
        .filter(pl.col("len") > 1)
        .rename({"len": "n_dupes"})
        .sort("date", "id")
    )

    # ------------------------------------------------------------------ #
    # 2. Missing sessions
    # ------------------------------------------------------------------ #
    if expected_dates is not None and expected_ids is not None:
        expected = pl.DataFrame({"date": pl.Series(expected_dates, dtype=pl.Date)}).join(
            pl.DataFrame({"id": pl.Series(expected_ids, dtype=pl.Int64)}),
            how="cross",
        )
        present = df.select("date", "id").unique()
        missing = expected.join(present, on=["date", "id"], how="anti").sort("date", "id")
    else:
        missing = pl.DataFrame(schema={"date": pl.Date, "id": pl.Int64})

    # ------------------------------------------------------------------ #
    # 3. Frozen / zero-variance series
    # ------------------------------------------------------------------ #
    # std == 0 across all observations for a given id
    stats = (
        df.group_by("id")
        .agg(
            pl.col(value_col).std().alias("_std"),
            pl.col(value_col).count().alias("n_obs"),
            pl.col(value_col).first().alias("value_constant"),
        )
        .filter(pl.col("_std").is_null() | (pl.col("_std") == 0.0))
        .drop("_std")
        .sort("id")
    )

    # ------------------------------------------------------------------ #
    # 4. Return-spike outliers (cross-sectional z-score per date)
    # ------------------------------------------------------------------ #
    with_z = df.with_columns(
        (
            (pl.col(value_col) - pl.col(value_col).mean().over("date"))
            / (pl.col(value_col).std().over("date") + 1e-12)
        ).alias("_z")
    )
    spikes = (
        with_z.filter(pl.col("_z").abs() > spike_z_threshold)
        .select("date", "id", pl.col(value_col).alias("value"), pl.col("_z").alias("z_score"))
        .sort("date", "id")
    )

    return QualityReport(
        duplicate_keys=dup,
        missing_sessions=missing,
        frozen_series=stats,
        spike_outliers=spikes,
    )
