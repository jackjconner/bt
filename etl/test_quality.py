"""Tests for data-quality checks."""

from __future__ import annotations

from datetime import date

import polars as pl

from .quality import check


def _clean_frame(n_ids: int = 20) -> pl.DataFrame:
    """Use enough assets so the cross-sectional z-score is well-defined."""
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    rows = []
    for d in dates:
        for i in range(n_ids):
            rows.append({"date": d, "id": i, "value": float(i + 1) + 0.01 * (d.day)})
    return pl.DataFrame(rows, schema={"date": pl.Date, "id": pl.Int64, "value": pl.Float64})


def test_clean_frame_no_issues():
    df = _clean_frame()
    report = check(df, "value")
    assert report.ok
    assert report.duplicate_keys.is_empty()
    assert report.frozen_series.is_empty()
    assert report.spike_outliers.is_empty()


def test_duplicate_key_detected():
    df = _clean_frame()
    duped = pl.concat([df, df.head(1)])
    report = check(duped, "value")
    assert not report.duplicate_keys.is_empty()
    assert report.duplicate_keys["n_dupes"][0] == 2


def test_frozen_series_detected():
    df = _clean_frame()
    # Make asset 1 all the same value
    frozen = df.with_columns(
        pl.when(pl.col("id") == 1).then(pl.lit(5.0)).otherwise(pl.col("value")).alias("value")
    )
    report = check(frozen, "value")
    assert not report.frozen_series.is_empty()
    assert 1 in report.frozen_series["id"].to_list()


def test_spike_outlier_detected():
    df = _clean_frame()
    # Use a threshold of 4.0 — with 20 assets a 1000x spike reliably exceeds it.
    spike = df.with_columns(
        pl.when((pl.col("id") == 0) & (pl.col("date") == date(2020, 1, 2)))
        .then(pl.lit(1000.0))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    report = check(spike, "value", spike_z_threshold=4.0)
    assert not report.spike_outliers.is_empty()
    row = report.spike_outliers.filter(
        (pl.col("id") == 0) & (pl.col("date") == date(2020, 1, 2))
    )
    assert row.height == 1
    assert abs(row["z_score"][0]) > 4.0


def test_missing_sessions_detected():
    df = _clean_frame()
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    ids = [0, 1, 2]
    # Remove asset 2 from date 2020-01-03
    df_gap = df.filter(
        ~((pl.col("id") == 2) & (pl.col("date") == date(2020, 1, 3)))
    )
    report = check(df_gap, "value", expected_dates=dates, expected_ids=ids)
    assert not report.missing_sessions.is_empty()
    missing_row = report.missing_sessions.filter(
        (pl.col("id") == 2) & (pl.col("date") == date(2020, 1, 3))
    )
    assert missing_row.height == 1


def test_missing_sessions_skipped_when_no_expected():
    df = _clean_frame()
    report = check(df, "value")
    assert report.missing_sessions.is_empty()


def test_report_summary_runs():
    df = _clean_frame()
    report = check(df, "value")
    s = report.summary()
    assert "duplicate_keys" in s
    assert "frozen_series" in s


def test_spike_threshold_respected():
    df = _clean_frame()
    # Mild outlier should not be flagged with default threshold 10
    mild = df.with_columns(
        pl.when((pl.col("id") == 0) & (pl.col("date") == date(2020, 1, 2)))
        .then(pl.lit(5.0))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    report = check(mild, "value", spike_z_threshold=10.0)
    assert report.spike_outliers.is_empty()
