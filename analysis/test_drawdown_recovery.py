"""Tests for per-event drawdown duration & recovery analytics.

``drawdown_recovery(nav)`` isolates each drawdown *event* from the NAV/equity
path and reports, per event: peak date, trough date, recovery date (null if the
event has not yet recovered by the end of the series), drawdown depth, drawdown
duration (peak->trough sessions) and time-to-recovery (trough->recovery and
peak->recovery sessions).

It builds on the same drawdown series the rest of ``analysis`` reports
(``nav / nav.cum_max() - 1``), so its events line up exactly with
``max_drawdown`` / ``AnalysisResult.drawdown_series``.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from analysis.risk import drawdown_recovery


def _nav(values: list[float]) -> pl.DataFrame:
    start = dt.date(2020, 1, 1)
    dates = [start + dt.timedelta(days=i) for i in range(len(values))]
    return pl.DataFrame({"date": dates, "nav": [float(v) for v in values]})


def test_monotonic_up_has_no_events() -> None:
    nav = _nav([100, 101, 102, 103, 104])
    events = drawdown_recovery(nav)
    assert events.height == 0
    # Schema is still well-formed so downstream consumers can rely on it.
    assert events.columns == [
        "peak_date",
        "trough_date",
        "recovery_date",
        "depth",
        "drawdown_days",
        "recovery_days",
        "peak_to_recovery_days",
    ]


def test_flat_series_has_no_events() -> None:
    # A flat NAV never dips below its running peak.
    nav = _nav([100, 100, 100, 100])
    assert drawdown_recovery(nav).height == 0


def test_single_drawdown_with_full_recovery() -> None:
    # peak=100 (day 1), dip to 80 (day 3, trough), back to a new high 110 (day 5).
    nav = _nav([90, 100, 90, 80, 95, 110])
    events = drawdown_recovery(nav)
    assert events.height == 1
    row = events.row(0, named=True)
    assert row["peak_date"] == dt.date(2020, 1, 2)  # day index 1, nav=100
    assert row["trough_date"] == dt.date(2020, 1, 4)  # day index 3, nav=80
    assert row["recovery_date"] == dt.date(2020, 1, 6)  # day index 5, nav=110 > 100
    assert row["recovery_date"] > row["trough_date"]
    assert abs(row["depth"] - (-0.20)) < 1e-12  # 80/100 - 1
    assert row["drawdown_days"] == 2  # day1 -> day3
    assert row["recovery_days"] == 2  # day3 -> day5
    assert row["peak_to_recovery_days"] == 4  # day1 -> day5


def test_multiple_drawdowns() -> None:
    # Event A: peak 100 (d1) -> trough 90 (d2) -> recover 101 (d3).
    # Event B: peak 101 (d3) -> trough 80 (d5) -> recover 105 (d6).
    nav = _nav([100, 90, 101, 90, 80, 105])
    events = drawdown_recovery(nav)
    assert events.height == 2

    a = events.row(0, named=True)
    assert a["peak_date"] == dt.date(2020, 1, 1)
    assert a["trough_date"] == dt.date(2020, 1, 2)
    assert a["recovery_date"] == dt.date(2020, 1, 3)
    assert abs(a["depth"] - (-0.10)) < 1e-12

    b = events.row(1, named=True)
    assert b["peak_date"] == dt.date(2020, 1, 3)
    assert b["trough_date"] == dt.date(2020, 1, 5)
    assert b["recovery_date"] == dt.date(2020, 1, 6)
    assert abs(b["depth"] - (80.0 / 101.0 - 1.0)) < 1e-12


def test_unrecovered_tail_drawdown_has_null_recovery() -> None:
    # Final drawdown never gets back above its prior peak before the series ends.
    nav = _nav([100, 120, 110, 90, 95])
    events = drawdown_recovery(nav)
    assert events.height == 1
    row = events.row(0, named=True)
    assert row["peak_date"] == dt.date(2020, 1, 2)  # nav=120
    assert row["trough_date"] == dt.date(2020, 1, 4)  # nav=90
    assert row["recovery_date"] is None
    assert row["recovery_days"] is None
    assert row["peak_to_recovery_days"] is None
    assert abs(row["depth"] - (90.0 / 120.0 - 1.0)) < 1e-12
    assert row["drawdown_days"] == 2


def test_depth_matches_max_drawdown() -> None:
    from analysis.metrics import max_drawdown
    from etl.source import to_float

    nav = _nav([100, 90, 101, 90, 80, 105])
    events = drawdown_recovery(nav)
    deepest = to_float(events["depth"].min())
    assert abs(deepest - max_drawdown(nav)) < 1e-12


def test_recovery_on_exact_prior_peak_counts_as_recovered() -> None:
    # Returning to exactly the prior peak (drawdown == 0) is a recovery.
    nav = _nav([100, 80, 100, 100])
    events = drawdown_recovery(nav)
    assert events.height == 1
    row = events.row(0, named=True)
    assert row["recovery_date"] == dt.date(2020, 1, 3)
    assert row["trough_date"] == dt.date(2020, 1, 2)


def test_empty_nav_returns_empty() -> None:
    nav = pl.DataFrame({"date": [], "nav": []}, schema={"date": pl.Date, "nav": pl.Float64})
    assert drawdown_recovery(nav).height == 0
