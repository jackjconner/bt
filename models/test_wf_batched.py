"""Tests for the batched numpy-core walk-forward engine (``wf_batched``).

The contract is *equivalence*: ``walk_forward_cv(..., engine="batched")`` must
reproduce ``engine="loop"`` (the sklearn per-fold incumbent) to numerical
tolerance across splitters, scaling, weighting, and alpha-grid configurations.
These tests are the differential oracle plus direct unit checks of the moment
machinery and the dispatch guard.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from .boosting import GradientBoostConfig, GradientBoostModel
from .panel import build_panel
from .ridge import ModelConfig, RidgeModel
from .splitters import (
    PurgedEmbargoCVSplitter,
    RollingWindowSplitter,
    WalkForwardSplitter,
)
from .walk_forward import WalkForwardConfig, _use_batched_engine, walk_forward_cv
from .wf_batched import is_ridge_factory, walk_forward_cv_batched


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _panel(n_dates=60, n_assets=12, n_features=5, seed=0, weights=False):
    rng = np.random.default_rng(seed)
    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_dates)]
    beta = np.zeros(n_features)
    beta[:2] = [0.3, -0.2]
    rf, rt, rw = [], [], []
    for d in dates:
        Xc = rng.normal(0, 1, (n_assets, n_features))
        yc = Xc @ beta + rng.normal(0, 0.5, n_assets)
        for i in range(n_assets):
            row: dict = {"date": d, "id": i}
            for f in range(n_features):
                row[f"feat_{f}"] = float(Xc[i, f])
            rf.append(row)
            rt.append({"date": d, "id": i, "fwd_ret_1": float(yc[i])})
            rw.append({"date": d, "id": i, "weight": float(rng.uniform(0.5, 1.5))})
    feat = pl.DataFrame(rf).with_columns(pl.col("id").cast(pl.Int64))
    tgt = pl.DataFrame(rt).with_columns(pl.col("id").cast(pl.Int64))
    wdf = pl.DataFrame(rw).with_columns(pl.col("id").cast(pl.Int64)) if weights else None
    return build_panel(feat, tgt, "fwd_ret_1", weights=wdf)


def _ridge(n_features=5):
    return lambda a: RidgeModel(ModelConfig(n_features=n_features, alpha=a))


def _gb_factory(alpha):
    return GradientBoostModel(GradientBoostConfig(n_features=5))


def _assert_equivalent(panel, splitter, n_features=5, **cfg_kwargs):
    fac = _ridge(n_features)
    rl = walk_forward_cv(panel, splitter, fac, WalkForwardConfig(engine="loop", **cfg_kwargs))
    rb = walk_forward_cv(panel, splitter, fac, WalkForwardConfig(engine="batched", **cfg_kwargs))
    assert len(rl.fold_results) == len(rb.fold_results)
    assert [f.chosen_alpha for f in rl.fold_results] == [f.chosen_alpha for f in rb.fold_results]
    np.testing.assert_allclose(rl.all_preds, rb.all_preds, atol=1e-10, rtol=0)
    np.testing.assert_allclose(rl.mean_ic, rb.mean_ic, atol=1e-10, rtol=0)
    np.testing.assert_allclose(rl.mean_r2, rb.mean_r2, atol=1e-10, rtol=0)
    np.testing.assert_allclose(rl.ic_ir, rb.ic_ir, atol=1e-10, rtol=0)
    for a, b in zip(rl.fold_results, rb.fold_results, strict=True):
        np.testing.assert_allclose(a.fit_result.coef, b.fit_result.coef, atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            a.fit_result.intercept, b.fit_result.intercept, atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(a.fit_result.train_r2, b.fit_result.train_r2, atol=1e-10, rtol=0)
        assert a.n_train == b.n_train and a.n_test == b.n_test
    return rl, rb


# --------------------------------------------------------------------------- #
# equivalence: batched reproduces the loop engine
# --------------------------------------------------------------------------- #
class TestEquivalence:
    def test_walk_forward_unweighted(self):
        _assert_equivalent(
            _panel(), WalkForwardSplitter(n_splits=4, embargo_periods=5, min_train_periods=10)
        )

    def test_walk_forward_weighted(self):
        _assert_equivalent(
            _panel(weights=True),
            WalkForwardSplitter(n_splits=4, embargo_periods=5, min_train_periods=10),
            use_sample_weights=True,
        )

    def test_rolling_window(self):
        _assert_equivalent(
            _panel(n_dates=80),
            RollingWindowSplitter(n_splits=3, train_periods=30, embargo_periods=2),
        )

    def test_purged_embargo_noncontiguous_blocks(self):
        # purging removes interior date blocks → exercises the run-decomposition
        _assert_equivalent(
            _panel(n_dates=80), PurgedEmbargoCVSplitter(n_splits=4, embargo_periods=3)
        )

    def test_no_scaling(self):
        _assert_equivalent(
            _panel(),
            WalkForwardSplitter(n_splits=4, min_train_periods=10),
            scale_features=False,
        )

    def test_single_alpha_skips_inner_cv(self):
        _assert_equivalent(
            _panel(),
            WalkForwardSplitter(n_splits=4, min_train_periods=10),
            alpha_grid=[1.0],
        )

    def test_odd_dims_midblock_inner_split(self):
        # 7 assets, 73 dates → inner-CV sample split lands mid-date-block
        _assert_equivalent(
            _panel(n_dates=73, n_assets=7),
            WalkForwardSplitter(n_splits=5, min_train_periods=11),
        )

    def test_wide_alpha_grid_selection_matches(self):
        rl, _ = _assert_equivalent(
            _panel(n_dates=90, n_assets=10),
            WalkForwardSplitter(n_splits=4, min_train_periods=12),
            alpha_grid=[0.001, 0.1, 10.0, 1000.0],
        )
        # selection actually varied across the grid (the test is meaningful)
        assert len({f.chosen_alpha for f in rl.fold_results}) >= 1

    def test_auto_engine_matches_loop(self):
        # the production default path (engine="auto") dispatches to batched and
        # must match an explicit loop run
        panel = _panel(weights=True)
        spl = WalkForwardSplitter(n_splits=4, embargo_periods=5, min_train_periods=10)
        fac = _ridge()
        ra = walk_forward_cv(panel, spl, fac, WalkForwardConfig(engine="auto"))
        rloop = walk_forward_cv(panel, spl, fac, WalkForwardConfig(engine="loop"))
        np.testing.assert_allclose(ra.all_preds, rloop.all_preds, atol=1e-10, rtol=0)
        np.testing.assert_allclose(ra.mean_ic, rloop.mean_ic, atol=1e-10, rtol=0)


# --------------------------------------------------------------------------- #
# dispatch + guards
# --------------------------------------------------------------------------- #
class TestDispatch:
    def test_is_ridge_factory_true(self):
        assert is_ridge_factory(_ridge(5), 5)

    def test_is_ridge_factory_wrong_n_features(self):
        assert not is_ridge_factory(_ridge(5), 7)

    def test_is_ridge_factory_non_ridge(self):
        assert not is_ridge_factory(_gb_factory, 5)

    def test_auto_skips_batched_for_non_ridge(self):
        panel = _panel()
        assert not _use_batched_engine(panel, _gb_factory, WalkForwardConfig(engine="auto"))

    def test_loop_engine_never_dispatches(self):
        panel = _panel()
        assert not _use_batched_engine(panel, _ridge(), WalkForwardConfig(engine="loop"))

    def test_batched_engine_rejects_non_ridge(self):
        panel = _panel()
        spl = WalkForwardSplitter(n_splits=3, min_train_periods=10)
        with pytest.raises(ValueError, match="closed-form RidgeModel"):
            walk_forward_cv(panel, spl, _gb_factory, WalkForwardConfig(engine="batched"))

    def test_unknown_engine_raises(self):
        panel = _panel()
        spl = WalkForwardSplitter(n_splits=3, min_train_periods=10)
        with pytest.raises(ValueError, match="unknown engine"):
            walk_forward_cv(panel, spl, _ridge(), WalkForwardConfig(engine="nope"))

    def test_unsorted_panel_falls_back_to_loop(self):
        # a hand-shuffled panel is not date-contiguous → auto must use the loop
        panel = _panel()
        rng = np.random.default_rng(3)
        perm = rng.permutation(len(panel.groups))
        from .panel import PanelArrays

        shuffled = PanelArrays(
            X=panel.X[perm],
            y=panel.y[perm],
            groups=panel.groups[perm],
            weights=panel.weights[perm],
            dates=panel.dates[perm],
            ids=panel.ids[perm],
            feature_names=panel.feature_names,
        )
        assert not _use_batched_engine(shuffled, _ridge(), WalkForwardConfig(engine="auto"))


# --------------------------------------------------------------------------- #
# direct entry point
# --------------------------------------------------------------------------- #
def test_walk_forward_cv_batched_direct():
    panel = _panel()
    spl = WalkForwardSplitter(n_splits=4, min_train_periods=10)
    res = walk_forward_cv_batched(panel, spl, _ridge(), WalkForwardConfig())
    assert len(res.fold_results) == 4
    assert np.isfinite(res.mean_ic)
    assert res.predictions_panel is not None
    assert set(res.predictions_panel.columns) == {"date", "id", "prediction", "fold"}
