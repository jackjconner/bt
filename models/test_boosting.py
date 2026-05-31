"""Tests for models.boosting — GradientBoostModel."""

from __future__ import annotations

import numpy as np
import pytest

from .boosting import GradientBoostConfig, GradientBoostModel
from .ridge import ModelConfig, ModelResult, RidgeModel


def _xy(n_samples: int = 120, n_features: int = 5, seed: int = 0):
    """Synthetic linear dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n_samples, n_features))
    beta = rng.normal(0.0, 1.0, n_features)
    y = X @ beta + rng.normal(0.0, 0.1, n_samples)
    return X, y


def _xy_nonlinear(n_samples: int = 300, n_features: int = 5, seed: int = 0):
    """Synthetic nonlinear dataset: y depends on X[0]^2 and X[1]*X[2]."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n_samples, n_features))
    y = X[:, 0] ** 2 + X[:, 1] * X[:, 2] + rng.normal(0.0, 0.05, n_samples)
    return X, y


# --------------------------------------------------------------------------- #
# Basic protocol conformance
# --------------------------------------------------------------------------- #


class TestGradientBoostModel:
    def test_fit_returns_model_result(self):
        X, y = _xy()
        model = GradientBoostModel(GradientBoostConfig(n_features=5))
        result = model.fit(X, y)
        assert isinstance(result, ModelResult)

    def test_coef_shape(self):
        """coef must be a zero-vector with length == n_features (n_features from X)."""
        X, y = _xy(n_features=7)
        model = GradientBoostModel(GradientBoostConfig(n_features=7))
        result = model.fit(X, y)
        assert result.coef.shape == (7,)
        assert np.all(result.coef == 0.0)

    def test_intercept_finite(self):
        X, y = _xy()
        model = GradientBoostModel(GradientBoostConfig(n_features=5))
        result = model.fit(X, y)
        assert np.isfinite(result.intercept)

    def test_train_r2_in_range(self):
        X, y = _xy()
        model = GradientBoostModel(GradientBoostConfig(n_features=5))
        result = model.fit(X, y)
        # boosted trees typically overfit training data (R² close to 1)
        assert result.train_r2 > 0.0

    def test_predict_before_fit_raises(self):
        model = GradientBoostModel(GradientBoostConfig(n_features=5))
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.zeros((3, 5)))

    def test_predict_shape(self):
        X, y = _xy()
        model = GradientBoostModel(GradientBoostConfig(n_features=5))
        model.fit(X, y)
        preds = model.predict(X[:20])
        assert preds.shape == (20,)

    def test_predict_dtype_float(self):
        X, y = _xy()
        model = GradientBoostModel(GradientBoostConfig(n_features=5))
        model.fit(X, y)
        preds = model.predict(X)
        assert np.issubdtype(preds.dtype, np.floating)

    def test_sample_weight_changes_fit(self):
        """Non-uniform sample weights should produce different predictions."""
        X, y = _xy(n_samples=200)
        w = np.ones(200)
        w[:100] = 10.0
        m1 = GradientBoostModel(GradientBoostConfig(n_features=5))
        m2 = GradientBoostModel(GradientBoostConfig(n_features=5))
        m1.fit(X, y)
        m2.fit(X, y, sample_weight=w)
        assert not np.allclose(m1.predict(X), m2.predict(X), atol=1e-8)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_gradient_boost_deterministic():
    """Same input must yield identical predictions across two independent fits."""
    X, y = _xy(n_samples=150)
    config = GradientBoostConfig(n_features=5, random_state=0)
    m1 = GradientBoostModel(config)
    m2 = GradientBoostModel(config)
    m1.fit(X, y)
    m2.fit(X, y)
    np.testing.assert_array_equal(m1.predict(X), m2.predict(X))


# --------------------------------------------------------------------------- #
# Protocol conformance (parametrized alongside linear models)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model",
    [
        RidgeModel(ModelConfig(n_features=5)),
        GradientBoostModel(GradientBoostConfig(n_features=5)),
    ],
)
def test_protocol_fit_predict(model):
    """GradientBoostModel must satisfy the same FinancialModel protocol shape as RidgeModel."""
    X, y = _xy(n_features=5)
    result = model.fit(X, y)
    assert hasattr(result, "coef")
    assert hasattr(result, "intercept")
    assert hasattr(result, "train_r2")
    preds = model.predict(X)
    assert preds.shape == (len(y),)


# --------------------------------------------------------------------------- #
# Nonlinear advantage
# --------------------------------------------------------------------------- #


def test_gradient_boost_beats_ridge_on_nonlinear():
    """On a nonlinear target GradientBoostModel should achieve a higher train R²
    than RidgeModel, demonstrating that the tree model captures the nonlinearity."""
    X, y = _xy_nonlinear(n_samples=400)
    ridge = RidgeModel(ModelConfig(n_features=5, alpha=0.1))
    boost = GradientBoostModel(GradientBoostConfig(n_features=5, max_iter=300, min_samples_leaf=5))
    r_ridge = ridge.fit(X, y)
    r_boost = boost.fit(X, y)
    assert r_boost.train_r2 > r_ridge.train_r2, (
        f"Expected boost R²={r_boost.train_r2:.4f} > ridge R²={r_ridge.train_r2:.4f}"
    )


# --------------------------------------------------------------------------- #
# Integration: works inside walk_forward_cv
# --------------------------------------------------------------------------- #


def test_gradient_boost_in_walk_forward():
    """GradientBoostModel must run end-to-end inside walk_forward_cv."""
    from datetime import date, timedelta
    from typing import Any

    import polars as pl

    from .panel import build_panel
    from .splitters import WalkForwardSplitter
    from .walk_forward import WalkForwardConfig, walk_forward_cv

    rng = np.random.default_rng(7)
    n_dates, n_assets, n_features = 60, 10, 5
    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_dates)]
    beta = np.array([0.3, -0.2, 0.0, 0.0, 0.0])

    rows_feat: list[dict[str, Any]] = []
    rows_tgt: list[dict[str, Any]] = []
    for d in dates:
        X_cross = rng.normal(0.0, 1.0, (n_assets, n_features))
        y_cross = X_cross @ beta + rng.normal(0.0, 0.5, n_assets)
        for aid in range(n_assets):
            row: dict[str, Any] = {"date": d, "id": aid}
            for f in range(n_features):
                row[f"feat_{f}"] = float(X_cross[aid, f])
            rows_feat.append(row)
            rows_tgt.append({"date": d, "id": aid, "fwd_ret_1": float(y_cross[aid])})

    feat_df = pl.DataFrame(rows_feat).with_columns(pl.col("id").cast(pl.Int64))
    tgt_df = pl.DataFrame(rows_tgt).with_columns(pl.col("id").cast(pl.Int64))
    panel = build_panel(feat_df, tgt_df, "fwd_ret_1")

    splitter = WalkForwardSplitter(n_splits=3, min_train_periods=10)
    # alpha argument is intentionally ignored by the factory
    config = WalkForwardConfig(alpha_grid=[1.0], scale_features=False, use_sample_weights=False)

    def factory(_alpha: float) -> GradientBoostModel:
        return GradientBoostModel(GradientBoostConfig(n_features=n_features))

    result = walk_forward_cv(panel, splitter, factory, config)
    assert len(result.fold_results) == 3
    assert np.isfinite(result.mean_ic)
    assert len(result.all_preds) > 0
