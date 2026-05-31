"""Tests for data-quality checks."""

from __future__ import annotations

from datetime import date

import polars as pl

from .quality import (
    QUALITY_FLAG_COLUMNS,
    _check_duplicates,
    _check_frozen_series,
    _check_missing_sessions,
    _check_spike_outliers,
    annotate_quality_flags,
    check,
)


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
    row = report.spike_outliers.filter((pl.col("id") == 0) & (pl.col("date") == date(2020, 1, 2)))
    assert row.height == 1
    assert abs(row["z_score"][0]) > 4.0


def test_missing_sessions_detected():
    df = _clean_frame()
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    ids = [0, 1, 2]
    # Remove asset 2 from date 2020-01-03
    df_gap = df.filter(~((pl.col("id") == 2) & (pl.col("date") == date(2020, 1, 3))))
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


# ------------------------------------------------------------------ #
# Unit tests for extracted helpers
# ------------------------------------------------------------------ #


def test_check_duplicates_no_dupes():
    df = _clean_frame()
    dup = _check_duplicates(df)
    assert dup.is_empty()


def test_check_duplicates_finds_dupe():
    df = _clean_frame()
    duped = pl.concat([df, df.head(1)])
    dup = _check_duplicates(duped)
    assert dup.height >= 1
    assert dup["n_dupes"][0] == 2


def test_check_duplicates_schema():
    df = _clean_frame()
    dup = _check_duplicates(df)
    assert "date" in dup.columns
    assert "id" in dup.columns
    assert "n_dupes" in dup.columns


def test_check_missing_sessions_no_expected_returns_empty():
    df = _clean_frame()
    missing = _check_missing_sessions(df, None, None)
    assert missing.is_empty()
    assert "date" in missing.columns
    assert "id" in missing.columns


def test_check_missing_sessions_partial_expected():
    """Only one id listed in expected_ids — gaps for unlisted ids not flagged."""
    df = _clean_frame(n_ids=3)
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    # Remove id=0 from one date
    df_gap = df.filter(~((pl.col("id") == 0) & (pl.col("date") == date(2020, 1, 3))))
    missing = _check_missing_sessions(df_gap, dates, [0, 1, 2])
    assert missing.height == 1
    assert missing["id"][0] == 0
    assert missing["date"][0] == date(2020, 1, 3)


def test_check_missing_sessions_no_gap():
    df = _clean_frame(n_ids=3)
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    missing = _check_missing_sessions(df, dates, [0, 1, 2])
    assert missing.is_empty()


def test_check_frozen_series_clean():
    df = _clean_frame()
    frozen = _check_frozen_series(df, "value")
    assert frozen.is_empty()


def test_check_frozen_series_detects_flat():
    df = _clean_frame()
    flat = df.with_columns(
        pl.when(pl.col("id") == 5).then(pl.lit(42.0)).otherwise(pl.col("value")).alias("value")
    )
    frozen = _check_frozen_series(flat, "value")
    assert 5 in frozen["id"].to_list()


def test_check_frozen_series_single_obs():
    """An asset with only one observation has std=null → should be flagged as frozen."""
    single = pl.DataFrame(
        {
            "date": pl.Series([date(2020, 1, 2)], dtype=pl.Date),
            "id": pl.Series([99], dtype=pl.Int64),
            "value": pl.Series([7.0], dtype=pl.Float64),
        }
    )
    frozen = _check_frozen_series(single, "value")
    assert 99 in frozen["id"].to_list()


def test_check_spike_outliers_no_spikes():
    df = _clean_frame()
    spikes = _check_spike_outliers(df, "value", spike_z_threshold=10.0)
    assert spikes.is_empty()


def test_check_spike_outliers_finds_spike():
    # With 20 assets and values 1-20, a 1000x spike on asset 0 gives z ≈ 4.25
    # (sample std is inflated by the spike itself).  Use threshold 4.0 to flag it.
    df = _clean_frame()
    spiked = df.with_columns(
        pl.when((pl.col("id") == 0) & (pl.col("date") == date(2020, 1, 2)))
        .then(pl.lit(9999.0))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    spikes = _check_spike_outliers(spiked, "value", spike_z_threshold=4.0)
    assert spikes.height >= 1
    flagged = spikes.filter((pl.col("id") == 0) & (pl.col("date") == date(2020, 1, 2)))
    assert flagged.height == 1
    assert abs(flagged["z_score"][0]) > 4.0


def test_check_spike_outliers_threshold_gates():
    """A modest outlier must not be flagged under a high threshold."""
    df = _clean_frame()
    slightly_off = df.with_columns(
        pl.when((pl.col("id") == 0) & (pl.col("date") == date(2020, 1, 2)))
        .then(pl.lit(3.0))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    spikes = _check_spike_outliers(slightly_off, "value", spike_z_threshold=100.0)
    assert spikes.is_empty()


def test_check_spike_outliers_schema():
    df = _clean_frame()
    spikes = _check_spike_outliers(df, "value", spike_z_threshold=10.0)
    assert "date" in spikes.columns
    assert "id" in spikes.columns
    assert "value" in spikes.columns
    assert "z_score" in spikes.columns


# ------------------------------------------------------------------ #
# annotate_quality_flags — row-level projection of the QualityReport
# ------------------------------------------------------------------ #


def test_annotate_preserves_rows_and_original_columns():
    df = _clean_frame()
    out = annotate_quality_flags(df, "value")
    # Row count preserved.
    assert out.height == df.height
    # Every original column preserved, unchanged, and in original order first.
    assert out.columns[: len(df.columns)] == df.columns
    for col in df.columns:
        assert out[col].to_list() == df[col].to_list()
    # Flag columns appended in documented order, all Boolean.
    assert out.columns[len(df.columns) :] == list(QUALITY_FLAG_COLUMNS)
    for flag in QUALITY_FLAG_COLUMNS:
        assert out.schema[flag] == pl.Boolean


def test_annotate_does_not_mutate_input():
    df = _clean_frame()
    before = df.clone()
    annotate_quality_flags(df, "value")
    assert df.equals(before)


def test_annotate_clean_frame_no_flags_fire():
    df = _clean_frame()
    out = annotate_quality_flags(df, "value")
    for flag in QUALITY_FLAG_COLUMNS:
        assert not out[flag].any(), f"{flag} fired on a clean frame"


def test_annotate_outlier_flag_fires_on_spike():
    df = _clean_frame()
    spike = df.with_columns(
        pl.when((pl.col("id") == 0) & (pl.col("date") == date(2020, 1, 2)))
        .then(pl.lit(1000.0))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    out = annotate_quality_flags(spike, "value", spike_z_threshold=4.0)
    flagged = out.filter(pl.col("outlier_flagged"))
    assert flagged.height == 1
    assert flagged["id"][0] == 0
    assert flagged["date"][0] == date(2020, 1, 2)


def test_annotate_frozen_flag_fires_on_flat_series():
    df = _clean_frame()
    frozen = df.with_columns(
        pl.when(pl.col("id") == 5).then(pl.lit(42.0)).otherwise(pl.col("value")).alias("value")
    )
    out = annotate_quality_flags(frozen, "value")
    # Every row of the frozen id is flagged; no other id is.
    assert out.filter(pl.col("is_frozen_series"))["id"].unique().to_list() == [5]
    assert out.filter(pl.col("id") == 5)["is_frozen_series"].all()


def test_annotate_duplicate_key_flag_fires():
    df = _clean_frame()
    duped = pl.concat([df, df.head(1)])
    out = annotate_quality_flags(duped, "value")
    dup_rows = out.filter(pl.col("is_duplicate_key"))
    # The duplicated (date, id) key is flagged on both copies.
    first = df.head(1)
    assert (
        dup_rows.filter(
            (pl.col("date") == first["date"][0]) & (pl.col("id") == first["id"][0])
        ).height
        == 2
    )


def test_annotate_sparse_coverage_flag_fires():
    df = _clean_frame(n_ids=3)
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    df_gap = df.filter(~((pl.col("id") == 2) & (pl.col("date") == date(2020, 1, 3))))
    out = annotate_quality_flags(df_gap, "value", expected_dates=dates, expected_ids=[0, 1, 2])
    # id 2 has a missing session → all its surviving rows are sparse-flagged.
    assert out.filter(pl.col("sparse_coverage"))["id"].unique().to_list() == [2]
    assert out.filter(pl.col("id") == 2)["sparse_coverage"].all()


def test_annotate_sparse_coverage_false_without_expected():
    df = _clean_frame()
    out = annotate_quality_flags(df, "value")
    assert not out["sparse_coverage"].any()


def test_annotate_price_stale_fires_on_repeat():
    # id 0: 10, 10 (second is stale), 11 (not). id 1: distinct values.
    df = pl.DataFrame(
        {
            "date": pl.Series(
                [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)] * 2, dtype=pl.Date
            ),
            "id": pl.Series([0, 0, 0, 1, 1, 1], dtype=pl.Int64),
            "value": pl.Series([10.0, 10.0, 11.0, 20.0, 21.0, 22.0], dtype=pl.Float64),
        }
    )
    out = annotate_quality_flags(df, "value")
    stale = out.filter(pl.col("price_stale"))
    assert stale.height == 1
    assert stale["id"][0] == 0
    assert stale["date"][0] == date(2020, 1, 3)


def test_annotate_price_stale_first_row_not_stale():
    """The first observation of each id has no prior obs → defaults to not-stale."""
    df = pl.DataFrame(
        {
            "date": pl.Series([date(2020, 1, 2), date(2020, 1, 2)], dtype=pl.Date),
            "id": pl.Series([0, 1], dtype=pl.Int64),
            "value": pl.Series([10.0, 10.0], dtype=pl.Float64),
        }
    )
    out = annotate_quality_flags(df, "value")
    assert not out["price_stale"].any()


def test_annotate_preserves_row_order_with_unsorted_input():
    """Flags must align to rows even when input is not (id, date)-sorted."""
    df = pl.DataFrame(
        {
            "date": pl.Series(
                [date(2020, 1, 6), date(2020, 1, 2), date(2020, 1, 3)], dtype=pl.Date
            ),
            "id": pl.Series([0, 0, 0], dtype=pl.Int64),
            "value": pl.Series([11.0, 10.0, 10.0], dtype=pl.Float64),
        }
    )
    out = annotate_quality_flags(df, "value")
    # Original row order preserved.
    assert out["date"].to_list() == df["date"].to_list()
    # Stale fires only on Jan-3 (prior obs Jan-2 had the same 10.0), not on the
    # Jan-6 row that physically precedes it in the frame.
    stale = out.filter(pl.col("price_stale"))
    assert stale.height == 1
    assert stale["date"][0] == date(2020, 1, 3)
