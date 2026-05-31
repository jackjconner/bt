"""Tests for trading-calendar alignment helpers."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from .calendar import align_to_calendar, fill_sessions, sessions_between


def _calendar() -> pl.DataFrame:
    """Tiny 5-session calendar for XNYS."""
    sessions = [
        date(2020, 1, 2),
        date(2020, 1, 3),
        date(2020, 1, 6),
        date(2020, 1, 7),
        date(2020, 1, 8),
    ]
    return pl.DataFrame(
        {
            "date": pl.Series(sessions, dtype=pl.Date),
            "exchange": pl.Series(["XNYS"] * 5, dtype=pl.Categorical),
            "is_session": [True] * 5,
            "is_half_day": [False] * 5,
            "session_open": ["09:30"] * 5,
            "session_close": ["16:00"] * 5,
        }
    )


def _panel() -> pl.DataFrame:
    """3 sessions × 2 assets, plus one extra weekend date that should be dropped."""
    rows = []
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)]  # Jan 4 = weekend
    for d in dates:
        for i in [0, 1]:
            rows.append({"date": d, "id": i, "value": float(i + 1)})
    return pl.DataFrame(rows, schema={"date": pl.Date, "id": pl.Int64, "value": pl.Float64})


def test_sessions_between_returns_correct_dates():
    cal = _calendar()
    result = sessions_between(cal, date(2020, 1, 2), date(2020, 1, 7))
    assert len(result) == 4
    assert result[0] == date(2020, 1, 2)
    assert result[-1] == date(2020, 1, 7)


def test_sessions_between_respects_bounds():
    cal = _calendar()
    result = sessions_between(cal, date(2020, 1, 5), date(2020, 1, 6))
    assert result == [date(2020, 1, 6)]


def test_align_drops_non_session_dates():
    cal = _calendar()
    panel = _panel()
    aligned = align_to_calendar(panel, cal, ids=[0, 1])
    # Jan 4 is not a session — should not appear
    assert date(2020, 1, 4) not in aligned["date"].to_list()


def test_align_fills_missing_sessions_with_null():
    cal = _calendar()
    panel = _panel()
    aligned = align_to_calendar(panel, cal, ids=[0, 1])
    # Jan 6, 7, 8 are sessions not in panel — value should be null
    jan6 = aligned.filter(pl.col("date") == date(2020, 1, 6))
    assert jan6.height == 2  # 2 assets
    assert jan6["value"].null_count() == 2


def test_align_preserves_present_values():
    cal = _calendar()
    panel = _panel()
    aligned = align_to_calendar(panel, cal, ids=[0, 1])
    jan2 = aligned.filter((pl.col("date") == date(2020, 1, 2)) & (pl.col("id") == 0))
    assert jan2["value"][0] == 1.0


def test_fill_sessions_forward():
    cal = _calendar()
    panel = _panel()
    filled = fill_sessions(panel, cal, ids=[0, 1], method="forward")
    # After forward-fill, Jan 6+ should have values carried from Jan 3
    jan6 = filled.filter((pl.col("date") == date(2020, 1, 6)) & (pl.col("id") == 0))
    assert jan6["value"][0] == 1.0  # carried from Jan 3


def test_fill_sessions_zero():
    cal = _calendar()
    panel = _panel()
    filled = fill_sessions(panel, cal, ids=[0, 1], method="zero")
    jan6 = filled.filter((pl.col("date") == date(2020, 1, 6)))
    assert (jan6["value"] == 0.0).all()


def test_fill_sessions_unknown_method_raises():
    cal = _calendar()
    panel = _panel()
    with pytest.raises(ValueError, match="Unknown fill method"):
        fill_sessions(panel, cal, ids=[0, 1], method="interpolate")


def test_align_output_sorted_by_date_id():
    cal = _calendar()
    panel = _panel()
    aligned = align_to_calendar(panel, cal, ids=[0, 1])
    dates = aligned["date"].to_list()
    ids = aligned["id"].to_list()
    pairs = list(zip(dates, ids))
    assert pairs == sorted(pairs)
