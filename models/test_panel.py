"""Tests for models.panel — long (date, id) panel → aligned numpy arrays."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .panel import (
    _attach_weights,
    _drop_null_rows,
    _extract_arrays,
    _join_features_target,
    build_panel,
    date_ordinals,
)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _make_features(
    n_dates: int = 10, n_assets: int = 5, n_features: int = 3, seed: int = 0
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    from datetime import date, timedelta

    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        for aid in range(n_assets):
            row: dict[str, Any] = {"date": d, "id": aid}
            for f in range(n_features):
                row[f"feat_{f}"] = float(rng.normal())
            rows.append(row)
    return pl.DataFrame(rows).with_columns(pl.col("id").cast(pl.Int64))


def _make_targets(n_dates: int = 10, n_assets: int = 5, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    from datetime import date, timedelta

    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        for aid in range(n_assets):
            rows.append({"date": d, "id": aid, "fwd_ret_1": float(rng.normal())})
    return pl.DataFrame(rows).with_columns(pl.col("id").cast(pl.Int64))


# --------------------------------------------------------------------------- #
# date_ordinals
# --------------------------------------------------------------------------- #


def test_date_ordinals_monotone():
    """date_ordinals must be strictly increasing for a sorted date series."""
    from datetime import date

    dates = pl.Series([date(2020, 1, i + 1) for i in range(5)])
    ords = date_ordinals(dates)
    assert list(ords) == sorted(ords), "ordinals are not sorted"
    assert len(set(ords)) == len(ords), "ordinals are not unique"


# --------------------------------------------------------------------------- #
# build_panel
# --------------------------------------------------------------------------- #


def test_build_panel_shape():
    """Output arrays must all have the same length = n_dates * n_assets (no NaN)."""
    n_dates, n_assets, n_features = 10, 5, 3
    feats = _make_features(n_dates, n_assets, n_features)
    tgts = _make_targets(n_dates, n_assets)
    panel = build_panel(feats, tgts, "fwd_ret_1")
    n = n_dates * n_assets
    assert panel.X.shape == (n, n_features)
    assert panel.y.shape == (n,)
    assert panel.groups.shape == (n,)
    assert panel.weights.shape == (n,)
    assert panel.dates.shape == (n,)
    assert panel.ids.shape == (n,)


def test_build_panel_nan_rows_dropped():
    """Rows with NaN in any feature or in the target must be removed."""
    feats = _make_features(10, 5, 3)
    tgts = _make_targets(10, 5)

    # inject a NaN into feat_0 for a specific row
    feats_w_nan = feats.with_columns(
        pl.when((pl.col("id") == 0) & (pl.col("date") == feats["date"][0]))
        .then(None)
        .otherwise(pl.col("feat_0"))
        .alias("feat_0")
    )
    panel = build_panel(feats_w_nan, tgts, "fwd_ret_1")
    # one row was dropped
    assert len(panel.y) == 10 * 5 - 1
    # no NaNs remain in X
    assert not np.any(np.isnan(panel.X))


def test_build_panel_target_nan_dropped():
    """Rows with NaN in the target must be removed even if features are clean."""
    feats = _make_features(10, 5, 3)
    tgts = _make_targets(10, 5)
    tgts_w_nan = tgts.with_columns(
        pl.when((pl.col("id") == 1) & (pl.col("date") == tgts["date"][0]))
        .then(None)
        .otherwise(pl.col("fwd_ret_1"))
        .alias("fwd_ret_1")
    )
    panel = build_panel(feats, tgts_w_nan, "fwd_ret_1")
    assert len(panel.y) == 10 * 5 - 1


def test_build_panel_groups_same_per_date():
    """All samples on the same date must share the same group ordinal."""
    feats = _make_features(10, 5, 2)
    tgts = _make_targets(10, 5)
    panel = build_panel(feats, tgts, "fwd_ret_1")
    # for each date, all group values should be the same
    for d in np.unique(panel.dates):
        mask = panel.dates == d
        assert len(np.unique(panel.groups[mask])) == 1


def test_build_panel_uniform_weights_when_none():
    """Without weights, all sample weights should be 1.0."""
    feats = _make_features(5, 3, 2)
    tgts = _make_targets(5, 3)
    panel = build_panel(feats, tgts, "fwd_ret_1")
    np.testing.assert_array_equal(panel.weights, np.ones(len(panel.y)))


def test_build_panel_custom_weights():
    """Provided weights should appear in panel.weights."""
    from datetime import date, timedelta

    n_dates, n_assets = 5, 3
    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        for aid in range(n_assets):
            rows.append({"date": d, "id": aid, "weight": float(aid + 1)})
    w_df = pl.DataFrame(rows).with_columns(pl.col("id").cast(pl.Int64))
    feats = _make_features(n_dates, n_assets, 2)
    tgts = _make_targets(n_dates, n_assets)
    panel = build_panel(feats, tgts, "fwd_ret_1", weights=w_df)
    # weights should not all be 1.0 (some assets have weight > 1)
    assert not np.all(panel.weights == 1.0)


def test_build_panel_feature_names():
    """feature_names must match the columns selected from features."""
    feats = _make_features(5, 3, 4)
    tgts = _make_targets(5, 3)
    panel = build_panel(feats, tgts, "fwd_ret_1", feature_cols=["feat_0", "feat_2"])
    assert panel.feature_names == ("feat_0", "feat_2")
    assert panel.X.shape[1] == 2


# --------------------------------------------------------------------------- #
# Extracted helper unit tests
# --------------------------------------------------------------------------- #


class TestJoinFeaturesTarget:
    def test_inner_join_keeps_only_shared_rows(self):
        """Rows in features but not in target (trailing dates) must be dropped."""
        feats = _make_features(10, 5, 2)
        tgts = _make_targets(8, 5)  # two fewer dates
        feature_cols = ["feat_0", "feat_1"]
        joined = _join_features_target(feats, tgts, "fwd_ret_1", feature_cols)
        assert len(joined) == 8 * 5

    def test_all_columns_present(self):
        feats = _make_features(5, 3, 2)
        tgts = _make_targets(5, 3)
        feature_cols = ["feat_0", "feat_1"]
        joined = _join_features_target(feats, tgts, "fwd_ret_1", feature_cols)
        for col in ["date", "id", "feat_0", "feat_1", "fwd_ret_1"]:
            assert col in joined.columns

    def test_respects_feature_cols_subset(self):
        feats = _make_features(5, 3, 4)
        tgts = _make_targets(5, 3)
        joined = _join_features_target(feats, tgts, "fwd_ret_1", ["feat_0"])
        assert "feat_0" in joined.columns
        assert "feat_1" not in joined.columns


class TestAttachWeights:
    def test_no_weights_gives_all_ones(self):
        feats = _make_features(5, 3, 2)
        tgts = _make_targets(5, 3)
        joined = _join_features_target(feats, tgts, "fwd_ret_1", ["feat_0", "feat_1"])
        result = _attach_weights(joined, None)
        assert "weight" in result.columns
        np.testing.assert_array_equal(result["weight"].to_numpy(), np.ones(len(result)))

    def test_custom_weights_appear_in_output(self):
        from datetime import date, timedelta

        n_dates, n_assets = 5, 3
        start = date(2020, 1, 2)
        dates = [start + timedelta(days=i) for i in range(n_dates)]
        rows = []
        for d in dates:
            for aid in range(n_assets):
                rows.append({"date": d, "id": aid, "weight": float(aid + 2)})
        w_df = pl.DataFrame(rows).with_columns(pl.col("id").cast(pl.Int64))

        feats = _make_features(n_dates, n_assets, 2)
        tgts = _make_targets(n_dates, n_assets)
        joined = _join_features_target(feats, tgts, "fwd_ret_1", ["feat_0", "feat_1"])
        result = _attach_weights(joined, w_df)
        assert not np.all(result["weight"].to_numpy() == 1.0)

    def test_missing_weight_rows_filled_with_one(self):
        """If a (date, id) pair is missing from weights, its weight defaults to 1.0."""
        from datetime import date, timedelta

        n_dates, n_assets = 5, 3
        start = date(2020, 1, 2)
        dates = [start + timedelta(days=i) for i in range(n_dates)]
        # Only supply weights for id=0
        rows = [{"date": d, "id": 0, "weight": 99.0} for d in dates]
        w_df = pl.DataFrame(rows).with_columns(pl.col("id").cast(pl.Int64))

        feats = _make_features(n_dates, n_assets, 2)
        tgts = _make_targets(n_dates, n_assets)
        joined = _join_features_target(feats, tgts, "fwd_ret_1", ["feat_0", "feat_1"])
        result = _attach_weights(joined, w_df)
        # id=1 and id=2 should have weight=1.0
        non_zero_ids = result.filter(pl.col("id") != 0)
        np.testing.assert_array_equal(non_zero_ids["weight"].to_numpy(), np.ones(len(non_zero_ids)))


class TestDropNullRows:
    def test_no_nulls_unchanged(self):
        feats = _make_features(5, 3, 2)
        tgts = _make_targets(5, 3)
        joined = _join_features_target(feats, tgts, "fwd_ret_1", ["feat_0", "feat_1"])
        joined = _attach_weights(joined, None)
        result = _drop_null_rows(joined, ["feat_0", "feat_1"], "fwd_ret_1")
        assert len(result) == len(joined)

    def test_feature_null_row_dropped(self):
        feats = _make_features(10, 5, 2)
        tgts = _make_targets(10, 5)
        # inject null into feat_0 for (id=0, first date)
        feats_w_null = feats.with_columns(
            pl.when((pl.col("id") == 0) & (pl.col("date") == feats["date"][0]))
            .then(None)
            .otherwise(pl.col("feat_0"))
            .alias("feat_0")
        )
        joined = _join_features_target(feats_w_null, tgts, "fwd_ret_1", ["feat_0", "feat_1"])
        joined = _attach_weights(joined, None).sort(["date", "id"])
        result = _drop_null_rows(joined, ["feat_0", "feat_1"], "fwd_ret_1")
        assert len(result) == 10 * 5 - 1

    def test_target_null_row_dropped(self):
        feats = _make_features(10, 5, 2)
        tgts = _make_targets(10, 5)
        tgts_w_null = tgts.with_columns(
            pl.when((pl.col("id") == 2) & (pl.col("date") == tgts["date"][0]))
            .then(None)
            .otherwise(pl.col("fwd_ret_1"))
            .alias("fwd_ret_1")
        )
        joined = _join_features_target(feats, tgts_w_null, "fwd_ret_1", ["feat_0", "feat_1"])
        joined = _attach_weights(joined, None).sort(["date", "id"])
        result = _drop_null_rows(joined, ["feat_0", "feat_1"], "fwd_ret_1")
        assert len(result) == 10 * 5 - 1


class TestExtractArrays:
    def test_shapes_match(self):
        feats = _make_features(8, 4, 3)
        tgts = _make_targets(8, 4)
        feature_cols = ["feat_0", "feat_1", "feat_2"]
        joined = _join_features_target(feats, tgts, "fwd_ret_1", feature_cols)
        joined = _attach_weights(joined, None).sort(["date", "id"])
        joined = _drop_null_rows(joined, feature_cols, "fwd_ret_1")
        X, y, w, dates_arr, ids_arr = _extract_arrays(joined, feature_cols, "fwd_ret_1")
        n = len(joined)
        assert X.shape == (n, 3)
        assert y.shape == (n,)
        assert w.shape == (n,)
        assert dates_arr.shape == (n,)
        assert ids_arr.shape == (n,)

    def test_dtypes(self):
        feats = _make_features(5, 3, 2)
        tgts = _make_targets(5, 3)
        joined = _join_features_target(feats, tgts, "fwd_ret_1", ["feat_0", "feat_1"])
        joined = _attach_weights(joined, None).sort(["date", "id"])
        joined = _drop_null_rows(joined, ["feat_0", "feat_1"], "fwd_ret_1")
        X, y, w, _dates_arr, ids_arr = _extract_arrays(joined, ["feat_0", "feat_1"], "fwd_ret_1")
        assert X.dtype == np.float64
        assert y.dtype == np.float64
        assert w.dtype == np.float64
        assert ids_arr.dtype == np.int64
