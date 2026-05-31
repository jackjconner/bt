"""Tests for models.panel — long (date, id) panel → aligned numpy arrays."""
from __future__ import annotations

import numpy as np
import polars as pl

from .panel import build_panel, date_ordinals


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _make_features(n_dates: int = 10, n_assets: int = 5, n_features: int = 3, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    from datetime import date, timedelta
    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_dates)]
    rows = []
    for d in dates:
        for aid in range(n_assets):
            row = {"date": d, "id": aid}
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
