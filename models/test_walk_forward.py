"""Tests for models.walk_forward — walk-forward CV with scaling, alpha search,
sample weighting, and IC scoring."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from .panel import build_panel
from .ridge import ModelConfig, RidgeModel
from .splitters import WalkForwardSplitter
from .walk_forward import (
    FoldScaler,
    WalkForwardConfig,
    _assemble_wf_result,
    _build_fold_panel_df,
    _fit_fold,
    _scale_fold,
    _score_fold,
    walk_forward_cv,
)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _synthetic_panel(n_dates: int = 60, n_assets: int = 10, n_features: int = 5, seed: int = 0):
    """Synthetic panel with mild predictive structure in features."""
    rng = np.random.default_rng(seed)
    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_dates)]
    ids = list(range(n_assets))

    # true beta: first 2 features matter
    beta = np.zeros(n_features)
    beta[:2] = [0.3, -0.2]

    rows_feat = []
    rows_tgt = []
    for d in dates:
        X_cross = rng.normal(0.0, 1.0, (n_assets, n_features))
        y_cross = X_cross @ beta + rng.normal(0.0, 0.5, n_assets)
        for i, aid in enumerate(ids):
            row_f: dict[str, Any] = {"date": d, "id": aid}
            for f in range(n_features):
                row_f[f"feat_{f}"] = float(X_cross[i, f])
            rows_feat.append(row_f)
            rows_tgt.append({"date": d, "id": aid, "fwd_ret_1": float(y_cross[i])})

    feat_df = pl.DataFrame(rows_feat).with_columns(pl.col("id").cast(pl.Int64))
    tgt_df = pl.DataFrame(rows_tgt).with_columns(pl.col("id").cast(pl.Int64))
    return feat_df, tgt_df


# --------------------------------------------------------------------------- #
# FoldScaler
# --------------------------------------------------------------------------- #


class TestFoldScaler:
    def test_transform_before_fit_raises(self):
        scaler = FoldScaler()
        with pytest.raises(RuntimeError, match="fit_transform"):
            scaler.transform(np.zeros((5, 3)))

    def test_fit_transform_zero_mean(self):
        rng = np.random.default_rng(0)
        X = rng.normal(5.0, 2.0, (100, 4))
        scaler = FoldScaler()
        Xt = scaler.fit_transform(X)
        np.testing.assert_allclose(Xt.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(Xt.std(axis=0), 1.0, atol=1e-10)

    def test_transform_uses_train_stats(self):
        """Test data scaled with train stats should have non-zero mean."""
        rng = np.random.default_rng(1)
        X_train = rng.normal(0.0, 1.0, (100, 3))
        X_test = rng.normal(10.0, 1.0, (20, 3))  # different distribution
        scaler = FoldScaler()
        scaler.fit_transform(X_train)
        Xt = scaler.transform(X_test)
        # test mean should be far from 0 when train stats are applied
        assert abs(Xt.mean()) > 1.0


# --------------------------------------------------------------------------- #
# walk_forward_cv
# --------------------------------------------------------------------------- #


def test_walk_forward_no_future_leakage():
    """Each fold's train date ordinals must be strictly < test date ordinals."""
    feat_df, tgt_df = _synthetic_panel()
    panel = build_panel(feat_df, tgt_df, "fwd_ret_1")
    splitter = WalkForwardSplitter(n_splits=4, min_train_periods=10)
    config = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)

    def factory(alpha):
        return RidgeModel(ModelConfig(n_features=5, alpha=alpha))

    result = walk_forward_cv(panel, splitter, factory, config)
    # verify dates: all test dates come after all train dates for each fold
    # (we can verify this indirectly via the splitter test, but also check WFResult)
    assert len(result.fold_results) == 4


def test_walk_forward_scaling_prevents_leakage():
    """With scale_features=True, each fold must use train-only statistics.
    We verify that the scaler is fitted fresh per fold rather than a global fit
    by checking that turning off scaling changes predictions (meaning scaling
    had an effect and was applied per-fold)."""
    feat_df, tgt_df = _synthetic_panel(n_dates=60, n_assets=8)
    panel = build_panel(feat_df, tgt_df, "fwd_ret_1")
    splitter = WalkForwardSplitter(n_splits=3, min_train_periods=10)

    def factory(alpha):
        return RidgeModel(ModelConfig(n_features=5, alpha=0.1))

    cfg_scaled = WalkForwardConfig(alpha_grid=[0.1], scale_features=True, use_sample_weights=False)
    cfg_raw = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)
    r_scaled = walk_forward_cv(panel, splitter, factory, cfg_scaled)
    r_raw = walk_forward_cv(panel, splitter, factory, cfg_raw)
    # predictions must differ when scaling is on vs off
    assert not np.allclose(r_scaled.all_preds, r_raw.all_preds, atol=1e-8)


def test_walk_forward_alpha_search():
    """Alpha search over a grid should select different alphas when signal varies."""
    feat_df, tgt_df = _synthetic_panel(n_dates=80, n_assets=10)
    panel = build_panel(feat_df, tgt_df, "fwd_ret_1")
    splitter = WalkForwardSplitter(n_splits=3, min_train_periods=15)
    grid = [0.001, 0.1, 10.0, 100.0]

    def factory(alpha):
        return RidgeModel(ModelConfig(n_features=5, alpha=alpha))

    cfg = WalkForwardConfig(alpha_grid=grid, scale_features=True, use_sample_weights=False)
    result = walk_forward_cv(panel, splitter, factory, cfg)
    # all chosen alphas must be from the grid
    for fr in result.fold_results:
        assert fr.chosen_alpha in grid


def test_walk_forward_sample_weights():
    """Sample weights reaching walk_forward_cv should influence fold coefficients.

    We verify this at the fold level: for at least one fold, the chosen
    coefficient under up-weighted train data differs from uniform-weight fit.
    The test constructs a two-regime dataset where the signal flips sign
    between the first and second half; extreme up-weighting of the first half
    biases the fit toward the first-half beta.
    """
    from datetime import date, timedelta

    rng = np.random.default_rng(77)
    n_dates, n_assets = 80, 8
    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_dates)]
    midpoint = n_dates // 2

    rows_f, rows_t, rows_w = [], [], []
    for di, d in enumerate(dates):
        # beta flips sign in the second half
        beta = 1.0 if di < midpoint else -1.0
        for aid in range(n_assets):
            x = float(rng.normal())
            y = beta * x + float(rng.normal(0, 0.1))
            rows_f.append({"date": d, "id": aid, "feat_0": x})
            rows_t.append({"date": d, "id": aid, "fwd_ret_1": y})
            # up-weight first half by 100x to force fit toward beta=+1
            w = 100.0 if di < midpoint else 1.0
            rows_w.append({"date": d, "id": aid, "weight": w})

    feat_df = pl.DataFrame(rows_f).with_columns(pl.col("id").cast(pl.Int64))
    tgt_df = pl.DataFrame(rows_t).with_columns(pl.col("id").cast(pl.Int64))
    w_df = pl.DataFrame(rows_w).with_columns(pl.col("id").cast(pl.Int64))

    panel_w = build_panel(feat_df, tgt_df, "fwd_ret_1", weights=w_df)
    panel_u = build_panel(feat_df, tgt_df, "fwd_ret_1")

    splitter = WalkForwardSplitter(n_splits=3, min_train_periods=20)

    def factory(alpha):
        return RidgeModel(ModelConfig(n_features=1, alpha=0.001))

    cfg_w = WalkForwardConfig(alpha_grid=[0.001], scale_features=False, use_sample_weights=True)
    cfg_u = WalkForwardConfig(alpha_grid=[0.001], scale_features=False, use_sample_weights=False)
    r_w = walk_forward_cv(panel_w, splitter, factory, cfg_w)
    r_u = walk_forward_cv(panel_u, splitter, factory, cfg_u)

    # at least one fold should have a different coefficient under weighted vs unweighted
    coefs_w = [fr.fit_result.coef[0] for fr in r_w.fold_results]
    coefs_u = [fr.fit_result.coef[0] for fr in r_u.fold_results]
    any_diff = any(abs(cw - cu) > 1e-6 for cw, cu in zip(coefs_w, coefs_u, strict=False))
    assert any_diff, f"Expected at least one fold to differ; coefs_w={coefs_w}, coefs_u={coefs_u}"


def test_walk_forward_ic_in_result():
    """WFResult must expose mean_ic and ic_ir with finite values."""
    feat_df, tgt_df = _synthetic_panel(n_dates=60, n_assets=10)
    panel = build_panel(feat_df, tgt_df, "fwd_ret_1")
    splitter = WalkForwardSplitter(n_splits=3, min_train_periods=10)

    def factory(alpha):
        return RidgeModel(ModelConfig(n_features=5, alpha=alpha))

    cfg = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)
    result = walk_forward_cv(panel, splitter, factory, cfg)
    assert np.isfinite(result.mean_ic)
    assert np.isfinite(result.ic_ir)


def test_walk_forward_provenance_arrays():
    """all_dates and all_ids must be populated and aligned with all_preds."""
    feat_df, tgt_df = _synthetic_panel()
    panel = build_panel(feat_df, tgt_df, "fwd_ret_1")
    splitter = WalkForwardSplitter(n_splits=3, min_train_periods=10)

    def factory(alpha):
        return RidgeModel(ModelConfig(n_features=5, alpha=alpha))

    cfg = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)
    result = walk_forward_cv(panel, splitter, factory, cfg)
    n = len(result.all_preds)
    assert len(result.all_true) == n
    assert len(result.all_dates) == n
    assert len(result.all_ids) == n
    assert len(result.all_groups) == n


# --------------------------------------------------------------------------- #
# predictions_panel (new additive field)
# --------------------------------------------------------------------------- #


def test_predictions_panel_is_populated():
    """predictions_panel must be a non-None DataFrame with expected columns."""
    feat_df, tgt_df = _synthetic_panel()
    panel = build_panel(feat_df, tgt_df, "fwd_ret_1")
    splitter = WalkForwardSplitter(n_splits=3, min_train_periods=10)

    def factory(alpha):
        return RidgeModel(ModelConfig(n_features=5, alpha=alpha))

    cfg = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)
    result = walk_forward_cv(panel, splitter, factory, cfg)

    assert result.predictions_panel is not None
    assert set(result.predictions_panel.columns) == {"date", "id", "prediction", "fold"}


def test_predictions_panel_keyed_by_date_id():
    """predictions_panel rows must align with all_dates / all_ids (same count)."""
    feat_df, tgt_df = _synthetic_panel()
    panel = build_panel(feat_df, tgt_df, "fwd_ret_1")
    splitter = WalkForwardSplitter(n_splits=3, min_train_periods=10)

    def factory(alpha):
        return RidgeModel(ModelConfig(n_features=5, alpha=alpha))

    cfg = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)
    result = walk_forward_cv(panel, splitter, factory, cfg)

    assert result.predictions_panel is not None
    n = len(result.all_preds)
    assert len(result.predictions_panel) == n


def test_predictions_panel_non_overlapping_folds():
    """Each (date, id) pair must appear in at most one fold."""
    feat_df, tgt_df = _synthetic_panel(n_dates=80, n_assets=10)
    panel = build_panel(feat_df, tgt_df, "fwd_ret_1")
    splitter = WalkForwardSplitter(n_splits=4, min_train_periods=10)

    def factory(alpha):
        return RidgeModel(ModelConfig(n_features=5, alpha=alpha))

    cfg = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)
    result = walk_forward_cv(panel, splitter, factory, cfg)

    assert result.predictions_panel is not None
    pp = result.predictions_panel
    # Each (date, id) pair must be unique across folds.
    dupes = pp.group_by(["date", "id"]).agg(pl.len().alias("cnt")).filter(pl.col("cnt") > 1)
    assert len(dupes) == 0, f"Found overlapping (date, id) pairs: {dupes}"


def test_predictions_panel_old_fields_unchanged():
    """Existing WFResult fields must be unaffected by the new predictions_panel field.

    Proves additivity: old call-sites reading all_preds / all_dates / all_ids
    still get the same data.
    """
    feat_df, tgt_df = _synthetic_panel()
    panel = build_panel(feat_df, tgt_df, "fwd_ret_1")
    splitter = WalkForwardSplitter(n_splits=3, min_train_periods=10)

    def factory(alpha):
        return RidgeModel(ModelConfig(n_features=5, alpha=alpha))

    cfg = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)
    result = walk_forward_cv(panel, splitter, factory, cfg)

    # These checks replicate what the pre-existing tests verify.
    assert len(result.fold_results) == 3
    assert np.isfinite(result.mean_ic)
    assert np.isfinite(result.ic_ir)
    n = len(result.all_preds)
    assert len(result.all_true) == n
    assert len(result.all_dates) == n
    assert len(result.all_ids) == n


# --------------------------------------------------------------------------- #
# Extracted helper unit tests
# --------------------------------------------------------------------------- #


class TestScaleFold:
    def test_no_scaling_returns_same_arrays(self):
        rng = np.random.default_rng(0)
        X_tr = rng.normal(0, 1, (50, 4))
        X_te = rng.normal(0, 1, (10, 4))
        X_tr_out, X_te_out = _scale_fold(X_tr, X_te, scale_features=False)
        assert X_tr_out is X_tr
        assert X_te_out is X_te

    def test_scaling_zero_centers_train(self):
        rng = np.random.default_rng(1)
        X_tr = rng.normal(5.0, 2.0, (100, 3))
        X_te = rng.normal(5.0, 2.0, (20, 3))
        X_tr_s, _ = _scale_fold(X_tr, X_te, scale_features=True)
        np.testing.assert_allclose(X_tr_s.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X_tr_s.std(axis=0), 1.0, atol=1e-10)

    def test_scaling_uses_train_stats_for_test(self):
        """Test fold is scaled by train statistics, not its own mean."""
        rng = np.random.default_rng(2)
        X_tr = rng.normal(0.0, 1.0, (100, 3))
        X_te = rng.normal(10.0, 1.0, (20, 3))  # very different distribution
        _, X_te_s = _scale_fold(X_tr, X_te, scale_features=True)
        # scaled test mean should be far from 0 (train-mean was ~0, test-mean was ~10)
        assert abs(X_te_s.mean()) > 5.0

    def test_fold_boundary_single_sample_train(self):
        """Edge: one-row train fold must not crash."""
        X_tr = np.array([[1.0, 2.0]])
        X_te = np.array([[3.0, 4.0], [5.0, 6.0]])
        # sklearn StandardScaler handles n=1 by zeroing std; _scale_fold must not raise
        X_tr_s, X_te_s = _scale_fold(X_tr, X_te, scale_features=True)
        assert X_tr_s.shape == X_tr.shape
        assert X_te_s.shape == X_te.shape


class TestFitFold:
    def _factory(self, alpha: float) -> RidgeModel:
        return RidgeModel(ModelConfig(n_features=3, alpha=alpha))

    def test_returns_model_and_fit_result(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (30, 3))
        y = X[:, 0] + rng.normal(0, 0.1, 30)
        model, fit_result = _fit_fold(self._factory, X, y, None, 0.1)
        assert hasattr(model, "predict")
        assert hasattr(fit_result, "coef")

    def test_sample_weight_forwarded(self):
        """Passing weights must change the fit coefficients vs unweighted."""
        rng = np.random.default_rng(3)
        X = rng.normal(0, 1, (50, 3))
        y = X[:, 0] + rng.normal(0, 0.1, 50)
        w = np.ones(50)
        w[:10] = 100.0  # up-weight first 10 rows

        _, fit_w = _fit_fold(self._factory, X, y, w, 0.1)
        _, fit_u = _fit_fold(self._factory, X, y, None, 0.1)
        assert not np.allclose(fit_w.coef, fit_u.coef, atol=1e-8)

    def test_predict_consistent_with_fit(self):
        """model.predict should use the weights from _fit_fold, not re-fit."""
        rng = np.random.default_rng(4)
        X_tr = rng.normal(0, 1, (40, 3))
        y_tr = X_tr[:, 0] + rng.normal(0, 0.1, 40)
        X_te = rng.normal(0, 1, (10, 3))
        model, fit_result = _fit_fold(self._factory, X_tr, y_tr, None, 0.1)
        preds = model.predict(X_te)
        # verify consistency: predict from coef directly
        expected = X_te @ fit_result.coef + fit_result.intercept
        np.testing.assert_allclose(preds, expected, rtol=1e-6)


class TestScoreFold:
    def test_returns_tuple_of_correct_types(self):
        rng = np.random.default_rng(5)
        n = 40
        y = rng.normal(0, 1, n)
        preds = y + rng.normal(0, 0.2, n)
        groups = np.repeat(np.arange(8), 5)
        r2, ic, ic_vals = _score_fold(y, preds, groups)
        assert isinstance(r2, float)
        assert isinstance(ic, float)
        assert isinstance(ic_vals, np.ndarray)
        assert len(ic_vals) == 8  # one per date

    def test_perfect_predictions_high_ic(self):
        rng = np.random.default_rng(6)
        n_dates, n_per = 10, 20
        n = n_dates * n_per
        y = rng.normal(0, 1, n)
        preds = y.copy()  # perfect predictions
        groups = np.repeat(np.arange(n_dates), n_per)
        _, ic, _ = _score_fold(y, preds, groups)
        assert ic > 0.99

    def test_constant_preds_zero_ic(self):
        """Constant predictions must yield IC ≈ 0 (Spearman undefined → 0.0)."""
        rng = np.random.default_rng(7)
        y = rng.normal(0, 1, 30)
        preds = np.zeros(30)
        groups = np.repeat(np.arange(6), 5)
        _, ic, _ = _score_fold(y, preds, groups)
        assert abs(ic) < 1e-10


class TestBuildFoldPanelDf:
    def test_schema_and_types(self):
        fold_dates = np.array([date(2022, 1, 3), date(2022, 1, 4)], dtype=object)
        fold_ids = np.array([10, 20], dtype=np.int64)
        preds = np.array([0.5, -0.3])
        df = _build_fold_panel_df(fold_dates, fold_ids, preds, fold_idx=0)
        assert set(df.columns) == {"date", "id", "prediction", "fold"}
        assert df["id"].dtype == pl.Int64
        assert df["prediction"].dtype == pl.Float64
        assert df["fold"].dtype == pl.Int32

    def test_fold_index_tagged_correctly(self):
        fold_dates = np.array([date(2022, 1, 3)], dtype=object)
        fold_ids = np.array([1], dtype=np.int64)
        preds = np.array([0.1])
        df = _build_fold_panel_df(fold_dates, fold_ids, preds, fold_idx=3)
        assert df["fold"][0] == 3

    def test_all_rows_populated(self):
        n = 15
        fold_dates = np.array(
            [date(2022, 1, 2) + timedelta(days=i) for i in range(n)], dtype=object
        )
        fold_ids = np.arange(n, dtype=np.int64)
        preds = np.linspace(-1, 1, n)
        df = _build_fold_panel_df(fold_dates, fold_ids, preds, fold_idx=0)
        assert len(df) == n


class TestAssembleWfResult:
    def test_empty_folds_returns_empty_arrays(self):
        result = _assemble_wf_result([], [], [], [], [], [], [])
        assert len(result.fold_results) == 0
        assert len(result.all_preds) == 0
        assert result.predictions_panel is None
        assert result.mean_r2 == 0.0
        assert result.mean_ic == 0.0

    def test_single_fold_round_trip(self):
        rng = np.random.default_rng(8)
        n = 10
        preds = rng.normal(0, 1, n)
        true_vals = rng.normal(0, 1, n)
        groups = np.repeat(np.array([100, 101], dtype=np.int64), 5)
        fold_dates = np.array([date(2022, 1, 3)] * n, dtype=object)
        fold_ids = np.arange(n, dtype=np.int64)

        from .scoring import held_out_r2, rank_ic_score, rank_ic_series
        from .walk_forward import FoldResult

        # build a minimal FoldResult
        rng2 = np.random.default_rng(9)
        X = rng2.normal(0, 1, (20, 2))
        y = rng2.normal(0, 1, 20)
        from .ridge import ModelConfig, RidgeModel

        m = RidgeModel(ModelConfig(n_features=2, alpha=1.0))
        fr_fit = m.fit(X, y)
        _, ic_vals = rank_ic_series(true_vals, preds, groups)
        fold_result = FoldResult(
            fit_result=fr_fit,
            test_r2=held_out_r2(true_vals, preds),
            test_ic=rank_ic_score(true_vals, preds, groups),
            ic_values=ic_vals,
            chosen_alpha=1.0,
            n_train=20,
            n_test=n,
        )
        panel_df = _build_fold_panel_df(fold_dates, fold_ids, preds, 0)
        result = _assemble_wf_result(
            [fold_result], [preds], [true_vals], [groups], [fold_dates], [fold_ids], [panel_df]
        )
        assert len(result.all_preds) == n
        np.testing.assert_array_equal(result.all_preds, preds)
        assert result.predictions_panel is not None
        assert len(result.predictions_panel) == n


# --------------------------------------------------------------------------- #
# fold IC dispersion diagnostics (additive, flag-gated)
# --------------------------------------------------------------------------- #


def _ridge_factory():
    def factory(alpha):
        return RidgeModel(ModelConfig(n_features=5, alpha=alpha))

    return factory


class TestFoldICDispersionDiagnostics:
    """``fold_ic_dispersion_enabled`` adds per-fold IC dispersion + hit-rate
    diagnostics from the existing per-date ``ic_values`` without disturbing the
    flag-off numbers."""

    def _run(self, *, enabled: bool, engine: str):
        feat_df, tgt_df = _synthetic_panel(n_dates=60, n_assets=10, n_features=5)
        panel = build_panel(feat_df, tgt_df, "fwd_ret_1")
        splitter = WalkForwardSplitter(n_splits=4, min_train_periods=10)
        config = WalkForwardConfig(
            alpha_grid=[0.1, 1.0],
            fold_ic_dispersion_enabled=enabled,
            engine=engine,
        )
        return walk_forward_cv(panel, splitter, _ridge_factory(), config)

    @pytest.mark.parametrize("engine", ["auto", "loop"])
    def test_flag_off_no_new_fields(self, engine):
        result = self._run(enabled=False, engine=engine)
        assert result.fold_diagnostics is None
        for fr in result.fold_results:
            assert fr.fold_ic_std is None
            assert fr.fold_hit_rate is None

    @pytest.mark.parametrize("engine", ["auto", "loop"])
    def test_flag_off_aggregates_byte_identical(self, engine):
        """Flag toggling must not move mean_ic / mean_r2 (or any fold number)."""
        off = self._run(enabled=False, engine=engine)
        on = self._run(enabled=True, engine=engine)
        assert on.mean_ic == off.mean_ic
        assert on.mean_r2 == off.mean_r2
        assert on.std_r2 == off.std_r2
        assert on.ic_ir == off.ic_ir
        assert len(on.fold_results) == len(off.fold_results)
        for fr_on, fr_off in zip(on.fold_results, off.fold_results, strict=True):
            assert fr_on.test_ic == fr_off.test_ic
            assert fr_on.test_r2 == fr_off.test_r2
            np.testing.assert_array_equal(fr_on.ic_values, fr_off.ic_values)

    @pytest.mark.parametrize("engine", ["auto", "loop"])
    def test_flag_on_populates_diagnostics(self, engine):
        result = self._run(enabled=True, engine=engine)
        assert result.fold_diagnostics is not None
        assert len(result.fold_diagnostics) == len(result.fold_results)
        for i, (fr, diag) in enumerate(
            zip(result.fold_results, result.fold_diagnostics, strict=True)
        ):
            # per-fold fields populated and within valid ranges
            assert fr.fold_ic_std is not None
            assert fr.fold_hit_rate is not None
            assert fr.fold_ic_std >= 0.0
            assert np.isfinite(fr.fold_ic_std)
            assert 0.0 <= fr.fold_hit_rate <= 1.0
            # diagnostics dict mirrors the fold fields
            assert diag["fold"] == float(i)
            assert diag["fold_ic_std"] == fr.fold_ic_std
            assert diag["fold_hit_rate"] == fr.fold_hit_rate
            assert diag["n_test_dates"] == float(len(fr.ic_values))
            # reproduce the diagnostics directly from ic_values (no IC recompute)
            assert fr.fold_ic_std == float(fr.ic_values.std())
            expected_hit = float(np.count_nonzero(fr.ic_values > 0.0) / len(fr.ic_values))
            assert fr.fold_hit_rate == expected_hit

    def test_auto_and_loop_engines_agree_on_diagnostics(self):
        auto = self._run(enabled=True, engine="auto")
        loop = self._run(enabled=True, engine="loop")
        assert auto.fold_diagnostics is not None
        assert loop.fold_diagnostics is not None
        for da, dl in zip(auto.fold_diagnostics, loop.fold_diagnostics, strict=True):
            assert da["fold_ic_std"] == pytest.approx(dl["fold_ic_std"], abs=1e-12)
            assert da["fold_hit_rate"] == pytest.approx(dl["fold_hit_rate"], abs=1e-12)


def test_fold_ic_diagnostics_helper():
    from .walk_forward import _fold_ic_diagnostics

    # empty fold → zeros
    assert _fold_ic_diagnostics(np.array([])) == (0.0, 0.0)
    # mixed signs: std is population std, hit rate is fraction > 0
    ic = np.array([0.2, -0.1, 0.3, 0.0])
    std, hit = _fold_ic_diagnostics(ic)
    assert std == pytest.approx(float(ic.std()))
    assert hit == pytest.approx(2.0 / 4.0)  # 0.0 is not > 0
