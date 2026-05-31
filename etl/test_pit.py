"""Tests for point-in-time as-of join helpers."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from .pit import as_of_slice, latest_as_of


def _fundamentals() -> pl.DataFrame:
    """Small synthetic fundamentals frame with 3 knowledge dates for 2 assets."""
    return pl.DataFrame(
        {
            "report_date": [
                date(2023, 3, 31),
                date(2023, 3, 31),
                date(2023, 6, 30),
                date(2023, 6, 30),
                date(2023, 9, 30),
                date(2023, 9, 30),
            ],
            "knowledge_date": [
                date(2023, 5, 15),
                date(2023, 5, 15),
                date(2023, 8, 14),
                date(2023, 8, 14),
                date(2023, 11, 14),
                date(2023, 11, 14),
            ],
            "id": [0, 1, 0, 1, 0, 1],
            "revenue": [100.0, 200.0, 110.0, 210.0, 120.0, 220.0],
        },
        schema={
            "report_date": pl.Date,
            "knowledge_date": pl.Date,
            "id": pl.Int64,
            "revenue": pl.Float64,
        },
    )


def test_as_of_slice_excludes_future_knowledge():
    df = _fundamentals()
    # As of May 14 — before any rows are published
    result = as_of_slice(df, date(2023, 5, 14))
    assert result.is_empty()


def test_as_of_slice_includes_rows_on_cutoff():
    df = _fundamentals()
    result = as_of_slice(df, date(2023, 5, 15))
    assert result.height == 2
    assert result["revenue"].to_list() == [100.0, 200.0]


def test_as_of_slice_after_all_dates_returns_all():
    df = _fundamentals()
    result = as_of_slice(df, date(2024, 1, 1))
    assert result.height == 6


def test_as_of_slice_missing_column_raises():
    df = _fundamentals().drop("knowledge_date")
    with pytest.raises(ValueError, match="knowledge_date"):
        as_of_slice(df, date(2023, 5, 15))


def test_latest_as_of_returns_one_row_per_group():
    df = _fundamentals()
    # By Aug 15 two rounds of filings exist; latest_as_of picks the most recent
    result = latest_as_of(df, date(2023, 8, 15), by=["id"])
    assert result.height == 2
    # Asset 0: revenue should be 110.0 (Q2 filing, knowledge 2023-08-14)
    row0 = result.filter(pl.col("id") == 0)
    assert row0["revenue"][0] == 110.0


def test_latest_as_of_before_first_filing_returns_empty():
    df = _fundamentals()
    result = latest_as_of(df, date(2023, 4, 1), by=["id"])
    assert result.is_empty()


def test_latest_as_of_missing_column_raises():
    df = _fundamentals().drop("knowledge_date")
    with pytest.raises(ValueError, match="knowledge_date"):
        latest_as_of(df, date(2023, 5, 15), by=["id"])


def test_as_of_prevents_lookahead():
    """Running at 2023-08-13 must not see the Q2 filing (published Aug 14)."""
    df = _fundamentals()
    result = latest_as_of(df, date(2023, 8, 13), by=["id"])
    # Only Q1 filing visible
    assert result.height == 2
    row0 = result.filter(pl.col("id") == 0)
    assert row0["revenue"][0] == 100.0
