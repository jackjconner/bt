"""Tests for models.models_zoo — LassoModel and ElasticNetModel."""

from __future__ import annotations

import numpy as np
import pytest

from .models_zoo import ElasticNetConfig, ElasticNetModel, LassoConfig, LassoModel
from .ridge import ModelConfig, ModelResult, RidgeModel


def _xy(n_samples: int = 120, n_features: int = 10, seed: int = 0, sparse: bool = False):
    """Synthetic dataset.  When sparse=True, only the first 3 features matter."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n_samples, n_features))
    if sparse:
        beta = np.zeros(n_features)
        beta[:3] = rng.normal(0.0, 1.0, 3)
    else:
        beta = rng.normal(0.0, 1.0, n_features)
    y = X @ beta + rng.normal(0.0, 0.1, n_samples)
    return X, y


# --------------------------------------------------------------------------- #
# LassoModel
# --------------------------------------------------------------------------- #


class TestLassoModel:
    def test_fit_returns_model_result(self):
        X, y = _xy()
        model = LassoModel(LassoConfig(n_features=10, alpha=0.01))
        result = model.fit(X, y)
        assert isinstance(result, ModelResult)
        assert result.coef.shape == (10,)
        assert np.isfinite(result.intercept)
        assert 0.0 <= result.train_r2 <= 1.0

    def test_predict_before_fit_raises(self):
        model = LassoModel(LassoConfig(n_features=5, alpha=0.1))
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.zeros((3, 5)))

    def test_lasso_zeros_coefficients(self):
        """With a sparse true signal and large alpha, Lasso should zero most coefs."""
        X, y = _xy(n_samples=200, n_features=20, sparse=True)
        model = LassoModel(LassoConfig(n_features=20, alpha=0.5))
        result = model.fit(X, y)
        n_zero = np.sum(np.abs(result.coef) < 1e-8)
        assert n_zero > 10, f"Expected Lasso to zero >10 coefs, only zeroed {n_zero}"

    def test_sample_weight_changes_fit(self):
        """Fitting with non-uniform sample weights should produce different coefs."""
        X, y = _xy()
        w = np.ones(120)
        w[:60] = 5.0  # up-weight first half
        model_uw = LassoModel(LassoConfig(n_features=10, alpha=0.01))
        model_w = LassoModel(LassoConfig(n_features=10, alpha=0.01))
        r_uw = model_uw.fit(X, y)
        r_w = model_w.fit(X, y, sample_weight=w)
        # coefs should differ when weights are non-uniform
        assert not np.allclose(r_uw.coef, r_w.coef, atol=1e-6)

    def test_predict_shape(self):
        X, y = _xy()
        model = LassoModel(LassoConfig(n_features=10, alpha=0.1))
        model.fit(X, y)
        preds = model.predict(X[:20])
        assert preds.shape == (20,)


# --------------------------------------------------------------------------- #
# ElasticNetModel
# --------------------------------------------------------------------------- #


class TestElasticNetModel:
    def test_fit_returns_model_result(self):
        X, y = _xy()
        model = ElasticNetModel(ElasticNetConfig(n_features=10, alpha=0.01, l1_ratio=0.5))
        result = model.fit(X, y)
        assert isinstance(result, ModelResult)
        assert result.coef.shape == (10,)
        assert np.isfinite(result.intercept)

    def test_predict_before_fit_raises(self):
        model = ElasticNetModel(ElasticNetConfig(n_features=5))
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(np.zeros((3, 5)))

    def test_l1_ratio_1_approaches_lasso(self):
        """l1_ratio=1.0 is pure Lasso; should zero some coefficients when alpha is large."""
        X, y = _xy(n_samples=200, n_features=20, sparse=True)
        model = ElasticNetModel(ElasticNetConfig(n_features=20, alpha=0.5, l1_ratio=1.0))
        result = model.fit(X, y)
        n_zero = np.sum(np.abs(result.coef) < 1e-8)
        assert n_zero > 5

    def test_l1_ratio_0_approaches_ridge(self):
        """l1_ratio=0.0 is pure Ridge; expect no zero coefficients on a dense signal."""
        X, y = _xy(n_samples=200, n_features=10, sparse=False)
        model = ElasticNetModel(ElasticNetConfig(n_features=10, alpha=0.001, l1_ratio=0.0))
        result = model.fit(X, y)
        n_nonzero = np.sum(np.abs(result.coef) > 1e-8)
        assert n_nonzero == 10, "Ridge-mode EN should keep all coefs non-zero"

    def test_sample_weight_changes_fit(self):
        X, y = _xy()
        w = np.ones(120)
        w[:60] = 5.0
        m1 = ElasticNetModel(ElasticNetConfig(n_features=10, alpha=0.01))
        m2 = ElasticNetModel(ElasticNetConfig(n_features=10, alpha=0.01))
        r1 = m1.fit(X, y)
        r2 = m2.fit(X, y, sample_weight=w)
        assert not np.allclose(r1.coef, r2.coef, atol=1e-6)

    def test_predict_shape(self):
        X, y = _xy()
        model = ElasticNetModel(ElasticNetConfig(n_features=10))
        model.fit(X, y)
        assert model.predict(X[:15]).shape == (15,)


# --------------------------------------------------------------------------- #
# Protocol conformance (all three models share the same interface)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model",
    [
        RidgeModel(ModelConfig(n_features=5)),
        LassoModel(LassoConfig(n_features=5, alpha=0.01)),
        ElasticNetModel(ElasticNetConfig(n_features=5, alpha=0.01)),
    ],
)
def test_protocol_fit_predict(model):
    """Every model type must satisfy the FinancialModel protocol shape."""
    X, y = _xy(n_features=5)
    result = model.fit(X, y)
    assert hasattr(result, "coef")
    assert hasattr(result, "intercept")
    assert hasattr(result, "train_r2")
    preds = model.predict(X)
    assert preds.shape == (len(y),)
