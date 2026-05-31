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


def _check_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """Return (date, id, n_dupes) for any key that appears more than once."""
    return (
        df.select("date", "id")
        .group_by("date", "id")
        .len()
        .filter(pl.col("len") > 1)
        .rename({"len": "n_dupes"})
        .sort("date", "id")
    )


def _check_missing_sessions(
    df: pl.DataFrame,
    expected_dates: list[date] | None,
    expected_ids: list[int] | None,
) -> pl.DataFrame:
    """Return (date, id) pairs from the expected cross-product absent in df."""
    if expected_dates is None or expected_ids is None:
        return pl.DataFrame(schema={"date": pl.Date, "id": pl.Int64})
    expected = pl.DataFrame({"date": pl.Series(expected_dates, dtype=pl.Date)}).join(
        pl.DataFrame({"id": pl.Series(expected_ids, dtype=pl.Int64)}),
        how="cross",
    )
    present = df.select("date", "id").unique()
    return expected.join(present, on=["date", "id"], how="anti").sort("date", "id")


def _check_frozen_series(df: pl.DataFrame, value_col: str) -> pl.DataFrame:
    """Return assets whose value column has std == 0 (or only one observation)."""
    return (
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


def _check_spike_outliers(
    df: pl.DataFrame, value_col: str, spike_z_threshold: float
) -> pl.DataFrame:
    """Return (date, id, value, z_score) for cross-sectional outliers above threshold."""
    with_z = df.with_columns(
        (
            (pl.col(value_col) - pl.col(value_col).mean().over("date"))
            / (pl.col(value_col).std().over("date") + 1e-12)
        ).alias("_z")
    )
    return (
        with_z.filter(pl.col("_z").abs() > spike_z_threshold)
        .select("date", "id", pl.col(value_col).alias("value"), pl.col("_z").alias("z_score"))
        .sort("date", "id")
    )


QUALITY_FLAG_COLUMNS: tuple[str, ...] = (
    "is_duplicate_key",
    "is_frozen_series",
    "sparse_coverage",
    "price_stale",
    "outlier_flagged",
)


def annotate_quality_flags(
    df: pl.DataFrame,
    value_col: str,
    *,
    expected_dates: list[date] | None = None,
    expected_ids: list[int] | None = None,
    spike_z_threshold: float = 10.0,
) -> pl.DataFrame:
    """Append per-row boolean data-quality flags to a ``(date, id, value)`` frame.

    This is the row-level view of :func:`check`: it runs the very same
    :class:`QualityReport` scan and projects each finding back onto the panel
    as a boolean column, so a downstream consumer can filter or weight rows by
    quality without re-deriving the checks.  The original columns are returned
    unchanged and in their original order; the flag columns are appended after
    them.  Row count and row order are preserved.

    Flags (all ``Boolean``, never null):

    ``is_duplicate_key``
        The row's ``(date, id)`` key appears more than once in ``df``
        (``QualityReport.duplicate_keys``).
    ``is_frozen_series``
        The row's ``id`` has zero variance over the window — a flat/frozen
        feed (``QualityReport.frozen_series``).
    ``sparse_coverage``
        The row's ``id`` is missing at least one expected session
        (``QualityReport.missing_sessions``).  Always ``False`` when
        ``expected_dates``/``expected_ids`` are not supplied (the
        missing-session check is skipped).
    ``price_stale``
        ``value_col`` equals the same ``id``'s value on its previous
        observation (sorted by ``date``).  The first observation of each ``id``
        has no prior and defaults to ``False`` (not stale).
    ``outlier_flagged``
        The row is a cross-sectional ``value_col`` outlier with
        ``|z| > spike_z_threshold`` (``QualityReport.spike_outliers``).

    Parameters mirror :func:`check`.  Returns a new frame; ``df`` is not
    mutated.
    """
    report = check(
        df,
        value_col,
        expected_dates=expected_dates,
        expected_ids=expected_ids,
        spike_z_threshold=spike_z_threshold,
    )

    dup_keys = report.duplicate_keys.select("date", "id").with_columns(
        pl.lit(True).alias("is_duplicate_key")
    )
    frozen_ids = report.frozen_series.select("id").with_columns(
        pl.lit(True).alias("is_frozen_series")
    )
    sparse_ids = (
        report.missing_sessions.select("id")
        .unique()
        .with_columns(pl.lit(True).alias("sparse_coverage"))
    )
    outliers = report.spike_outliers.select("date", "id").with_columns(
        pl.lit(True).alias("outlier_flagged")
    )

    # price_stale: value unchanged from the same id's previous observation.
    # First observation per id has no prior → not stale.
    annotated = (
        df.with_row_index("_orig_order")
        .sort("id", "date", maintain_order=True)
        .with_columns(
            (pl.col(value_col) == pl.col(value_col).shift(1).over("id"))
            .fill_null(value=False)
            .alias("price_stale")
        )
        .sort("_orig_order")
        .drop("_orig_order")
    )

    annotated = (
        annotated.join(dup_keys, on=["date", "id"], how="left")
        .join(frozen_ids, on="id", how="left")
        .join(sparse_ids, on="id", how="left")
        .join(outliers, on=["date", "id"], how="left")
    )

    annotated = annotated.with_columns(
        pl.col("is_duplicate_key").fill_null(value=False),
        pl.col("is_frozen_series").fill_null(value=False),
        pl.col("sparse_coverage").fill_null(value=False),
        pl.col("outlier_flagged").fill_null(value=False),
    )

    # Original columns unchanged and first, flags appended in documented order.
    return annotated.select(*df.columns, *QUALITY_FLAG_COLUMNS)


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
    return QualityReport(
        duplicate_keys=_check_duplicates(df),
        missing_sessions=_check_missing_sessions(df, expected_dates, expected_ids),
        frozen_series=_check_frozen_series(df, value_col),
        spike_outliers=_check_spike_outliers(df, value_col, spike_z_threshold),
    )
