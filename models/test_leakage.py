"""Unit tests for models.leakage — extracted helpers and check functions.

Covers:
  - _build_next_period_returns: shift semantics and null at last date
  - _max_relative_error: pure numeric helper
  - _check_fwd_ret_horizon: passes on correct data, fails on look-ahead data
  - _check_feature_target_alignment: passes / fails with a known-bad panel
  - _check_embargo_invariant: passes / fails depending on fold gap
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from .leakage import (
    CheckResult,
    _build_next_period_returns,
    _check_embargo_invariant,
    _check_feature_target_alignment,
    _check_fwd_ret_horizon,
    _max_relative_error,
    audit_leakage,
)
from .panel import PanelArrays, build_panel
from .splitters import WalkForwardSplitter

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _make_daily_returns(n_dates: int = 10, n_assets: int = 4, seed: int = 0) -> pl.DataFrame:
    """Deterministic daily return frame (date, id, return)."""
    rng = np.random.default_rng(seed)
    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        for aid in range(n_assets):
            rows.append({"date": d, "id": aid, "return": float(rng.normal(0.0, 0.02))})
    return pl.DataFrame(rows).with_columns(pl.col("id").cast(pl.Int64))


def _make_forward_returns_from_daily(daily: pl.DataFrame) -> pl.DataFrame:
    """Correctly compute fwd_ret_1 at date T = return at T+1 (no look-ahead)."""
    next_ret = _build_next_period_returns(daily)
    return next_ret.select("date", "id", pl.col("return_next_date").alias("fwd_ret_1")).drop_nulls()


def _make_panel_from_daily(daily: pl.DataFrame, n_features: int = 3, seed: int = 42) -> PanelArrays:
    """Build a panel whose target exactly equals return[T+1] (correct alignment)."""
    rng = np.random.default_rng(seed)
    dates_list = sorted(daily["date"].unique().to_list())
    ids_list = sorted(daily["id"].unique().to_list())

    feat_rows = []
    for d in dates_list:
        for aid in ids_list:
            row: dict = {"date": d, "id": aid}
            for f in range(n_features):
                row[f"feat_{f}"] = float(rng.normal())
            feat_rows.append(row)
    feat_df = pl.DataFrame(feat_rows).with_columns(pl.col("id").cast(pl.Int64))
    fwd_df = _make_forward_returns_from_daily(daily)
    return build_panel(feat_df, fwd_df, "fwd_ret_1")


# --------------------------------------------------------------------------- #
# _build_next_period_returns
# --------------------------------------------------------------------------- #


class TestBuildNextPeriodReturns:
    def test_schema(self):
        daily = _make_daily_returns(5, 2)
        result = _build_next_period_returns(daily)
        assert "date" in result.columns
        assert "id" in result.columns
        assert "return_next_date" in result.columns

    def test_last_date_per_id_is_null(self):
        """The last trading date for each id must have a null return_next_date."""
        daily = _make_daily_returns(5, 3)
        result = _build_next_period_returns(daily)
        max_date = daily["date"].max()
        last_rows = result.filter(pl.col("date") == max_date)
        assert last_rows["return_next_date"].null_count() == len(last_rows)

    def test_next_date_value_matches_actual_return(self):
        """return_next_date[T] must equal return[T+1] for interior dates."""
        # Build a tiny deterministic frame: 3 dates, 1 asset, known values
        dates = [date(2022, 1, 3), date(2022, 1, 4), date(2022, 1, 5)]
        returns = [0.01, 0.02, 0.03]
        daily = pl.DataFrame({"date": dates, "id": [0, 0, 0], "return": returns}).with_columns(
            pl.col("id").cast(pl.Int64)
        )

        result = _build_next_period_returns(daily).sort("date")
        # date 2022-01-03: next return = 0.02
        row_jan3 = result.filter(pl.col("date") == date(2022, 1, 3))
        assert abs(row_jan3["return_next_date"][0] - 0.02) < 1e-10
        # date 2022-01-04: next return = 0.03
        row_jan4 = result.filter(pl.col("date") == date(2022, 1, 4))
        assert abs(row_jan4["return_next_date"][0] - 0.03) < 1e-10

    def test_per_id_independence(self):
        """Each id's shift must be independent — no cross-id contamination."""
        dates = [date(2022, 1, 3), date(2022, 1, 4)]
        daily = pl.DataFrame(
            {
                "date": dates * 2,
                "id": [0, 0, 1, 1],
                "return": [0.10, 0.20, 0.30, 0.40],
            }
        ).with_columns(pl.col("id").cast(pl.Int64))
        result = _build_next_period_returns(daily).sort(["id", "date"])
        id0_jan3 = result.filter((pl.col("id") == 0) & (pl.col("date") == date(2022, 1, 3)))
        id1_jan3 = result.filter((pl.col("id") == 1) & (pl.col("date") == date(2022, 1, 3)))
        assert abs(id0_jan3["return_next_date"][0] - 0.20) < 1e-10
        assert abs(id1_jan3["return_next_date"][0] - 0.40) < 1e-10


# --------------------------------------------------------------------------- #
# _max_relative_error
# --------------------------------------------------------------------------- #


class TestMaxRelativeError:
    def test_identical_arrays_is_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _max_relative_error(a, a) < 1e-10

    def test_known_relative_error(self):
        a = np.array([1.1])
        b = np.array([1.0])
        # abs(1.1 - 1.0) / (1.0 + 1e-8) ≈ 0.1
        err = _max_relative_error(a, b)
        assert abs(err - 0.1) < 1e-6

    def test_max_is_maximum_not_mean(self):
        a = np.array([1.01, 2.5])
        b = np.array([1.0, 2.0])
        # second element: abs(2.5 - 2.0) / (2.0 + 1e-8) ≈ 0.25
        err = _max_relative_error(a, b)
        assert abs(err - 0.25) < 1e-6


# --------------------------------------------------------------------------- #
# _check_fwd_ret_horizon
# --------------------------------------------------------------------------- #


class TestCheckFwdRetHorizon:
    def test_passes_on_correct_data(self):
        daily = _make_daily_returns(15, 4)
        fwd = _make_forward_returns_from_daily(daily)
        result = _check_fwd_ret_horizon(fwd, daily)
        assert isinstance(result, CheckResult)
        assert result.passed, result.detail

    def test_fails_when_fwd_ret_column_missing(self):
        daily = _make_daily_returns(5, 2)
        bad_fwd = daily.select("date", "id")  # no fwd_ret_1
        result = _check_fwd_ret_horizon(bad_fwd, daily)
        assert not result.passed
        assert "not found" in result.detail

    def test_fails_on_same_day_look_ahead(self):
        """When fwd_ret_1 is the SAME-DAY return (look-ahead), the check must fail."""
        daily = _make_daily_returns(15, 4)
        # Deliberately build a 'wrong' forward return: fwd_ret_1[T] = return[T]
        bad_fwd = daily.rename({"return": "fwd_ret_1"}).select("date", "id", "fwd_ret_1")
        result = _check_fwd_ret_horizon(bad_fwd, daily)
        assert not result.passed


# --------------------------------------------------------------------------- #
# _check_feature_target_alignment
# --------------------------------------------------------------------------- #


class TestCheckFeatureTargetAlignment:
    def test_passes_on_correctly_aligned_panel(self):
        daily = _make_daily_returns(20, 5)
        panel = _make_panel_from_daily(daily)
        result = _check_feature_target_alignment(panel, daily)
        assert result.passed, result.detail

    def test_fails_when_return_column_missing(self):
        daily = _make_daily_returns(10, 3)
        panel = _make_panel_from_daily(daily)
        bad_daily = daily.rename({"return": "daily_ret"})
        result = _check_feature_target_alignment(panel, bad_daily)
        assert not result.passed
        assert "not found" in result.detail

    def test_fails_on_misaligned_panel(self):
        """Panel whose y equals return[T] (not T+1) must trigger failure."""
        daily = _make_daily_returns(20, 5)
        # Build a panel with the WRONG target (same-day return, not next-day)
        dates_list = sorted(daily["date"].unique().to_list())
        ids_list = sorted(daily["id"].unique().to_list())
        rng = np.random.default_rng(7)
        feat_rows = []
        for d in dates_list:
            for aid in ids_list:
                row: dict = {"date": d, "id": aid, "feat_0": float(rng.normal())}
                feat_rows.append(row)
        feat_df = pl.DataFrame(feat_rows).with_columns(pl.col("id").cast(pl.Int64))
        # Use same-day return as target (look-ahead violation)
        same_day_target = daily.rename({"return": "fwd_ret_1"}).select("date", "id", "fwd_ret_1")
        panel = build_panel(feat_df, same_day_target, "fwd_ret_1")
        result = _check_feature_target_alignment(panel, daily)
        # This should fail: y matches return[T], not return[T+1]
        assert not result.passed


# --------------------------------------------------------------------------- #
# _check_embargo_invariant
# --------------------------------------------------------------------------- #


class TestCheckEmbargoInvariant:
    def _make_panel(self, n_dates: int = 60, n_assets: int = 5) -> PanelArrays:
        daily = _make_daily_returns(n_dates, n_assets)
        return _make_panel_from_daily(daily)

    def test_passes_when_gap_satisfied(self):
        panel = self._make_panel()
        splitter = WalkForwardSplitter(n_splits=4, min_train_periods=10, embargo_periods=0)
        result = _check_embargo_invariant(panel, splitter, embargo_periods=0)
        assert result.passed, result.detail

    def test_fails_when_gap_violated(self):
        """Requesting embargo=5 days but the splitter has embargo=0 must fail."""
        panel = self._make_panel()
        splitter = WalkForwardSplitter(n_splits=4, min_train_periods=10, embargo_periods=0)
        # Check expects gap > 5 calendar days but the splitter has 0 embargo —
        # ordinals are date.toordinal() values (calendar days), consecutive dates
        # differ by 1 so the gap between max_train and min_test is 1 day.
        result = _check_embargo_invariant(panel, splitter, embargo_periods=5)
        assert not result.passed
        assert "violated" in result.detail


# --------------------------------------------------------------------------- #
# audit_leakage — end-to-end smoke test
# --------------------------------------------------------------------------- #


def test_audit_leakage_all_pass():
    """Full audit must pass on correctly-constructed synthetic data."""
    daily = _make_daily_returns(30, 5)
    fwd = _make_forward_returns_from_daily(daily)
    panel = _make_panel_from_daily(daily)
    splitter = WalkForwardSplitter(n_splits=3, min_train_periods=8, embargo_periods=0)
    report = audit_leakage(
        forward_returns=fwd,
        daily_returns=daily,
        panel=panel,
        splitter=splitter,
        embargo_periods=0,
    )
    assert report.all_passed, [c.detail for c in report.checks if not c.passed]
    assert len(report.checks) == 3
