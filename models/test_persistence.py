"""Tests for models.persistence — save/load ModelArtifact + predict_from_artifact."""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import numpy as np

from .persistence import (
    artifact_from_fold,
    load_artifact,
    predict_from_artifact,
    save_artifact,
)
from .ridge import ModelConfig, RidgeModel

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _fit_ridge(n: int = 80, nf: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n, nf))
    y = X @ rng.normal(0.0, 1.0, nf) + rng.normal(0.0, 0.1, n)
    model = RidgeModel(ModelConfig(n_features=nf, alpha=1.0))
    result = model.fit(X, y)
    return result, X, y


# --------------------------------------------------------------------------- #
# round-trip save / load
# --------------------------------------------------------------------------- #


class TestSaveLoadArtifact:
    def test_round_trip_coef(self):
        result, _X, _ = _fit_ridge()
        artifact = artifact_from_fold(
            result,
            feature_names=tuple(f"f{i}" for i in range(5)),
            alpha=1.0,
            model_type="RidgeModel",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "model"
            save_artifact(artifact, base)
            loaded = load_artifact(base)
        np.testing.assert_array_equal(loaded.coef, artifact.coef)
        assert loaded.intercept == artifact.intercept
        assert loaded.train_r2 == artifact.train_r2

    def test_round_trip_scaler(self):
        from sklearn.preprocessing import StandardScaler

        result, X, _ = _fit_ridge()
        scaler = StandardScaler()
        scaler.fit(X)
        artifact = artifact_from_fold(
            result,
            feature_names=tuple(f"f{i}" for i in range(5)),
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            alpha=1.0,
            model_type="RidgeModel",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "model"
            save_artifact(artifact, base)
            loaded = load_artifact(base)
        np.testing.assert_array_almost_equal(loaded.scaler_mean, scaler.mean_)
        np.testing.assert_array_almost_equal(loaded.scaler_scale, scaler.scale_)

    def test_round_trip_metadata(self):
        result, _, _ = _fit_ridge()
        artifact = artifact_from_fold(
            result,
            feature_names=("a", "b", "c", "d", "e"),
            train_start=date(2020, 1, 1),
            train_end=date(2021, 12, 31),
            alpha=2.0,
            model_type="RidgeModel",
            extra={"note": "test"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "model"
            save_artifact(artifact, base)
            loaded = load_artifact(base)
        assert loaded.feature_names == ("a", "b", "c", "d", "e")
        assert loaded.train_start == date(2020, 1, 1)
        assert loaded.train_end == date(2021, 12, 31)
        assert loaded.alpha == 2.0
        assert loaded.model_type == "RidgeModel"
        assert loaded.extra["note"] == "test"

    def test_round_trip_no_scaler(self):
        result, _, _ = _fit_ridge()
        artifact = artifact_from_fold(
            result,
            feature_names=tuple(f"f{i}" for i in range(5)),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "model"
            save_artifact(artifact, base)
            loaded = load_artifact(base)
        assert loaded.scaler_mean is None
        assert loaded.scaler_scale is None


# --------------------------------------------------------------------------- #
# predict_from_artifact
# --------------------------------------------------------------------------- #


class TestPredictFromArtifact:
    def test_clean_data_matches_linear_score(self):
        """Predictions without scaling should equal X @ coef + intercept."""
        result, X, _ = _fit_ridge(nf=5)
        artifact = artifact_from_fold(
            result,
            feature_names=tuple(f"f{i}" for i in range(5)),
        )
        preds = predict_from_artifact(artifact, X)
        expected = X @ result.coef + result.intercept
        np.testing.assert_allclose(preds, expected, rtol=1e-10)

    def test_nan_rows_get_nan_prediction(self):
        """Rows with any NaN feature should produce NaN predictions."""
        result, X, _ = _fit_ridge(nf=5)
        artifact = artifact_from_fold(result, feature_names=tuple(f"f{i}" for i in range(5)))
        X_nan = X.copy()
        X_nan[2, 1] = np.nan
        X_nan[5, :] = np.nan
        preds = predict_from_artifact(artifact, X_nan)
        assert np.isnan(preds[2])
        assert np.isnan(preds[5])
        # other rows should be finite
        valid_mask = ~np.isnan(preds)
        assert valid_mask.sum() == len(X) - 2
        assert np.all(np.isfinite(preds[valid_mask]))

    def test_return_mask_flag(self):
        result, X, _ = _fit_ridge(nf=5)
        artifact = artifact_from_fold(result, feature_names=tuple(f"f{i}" for i in range(5)))
        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        _preds, mask = predict_from_artifact(artifact, X_nan, return_mask=True)
        assert mask.shape == (len(X),)
        assert mask.dtype == bool
        assert not mask[0]
        assert mask[1:].all()

    def test_scaling_applied(self):
        """With scaler metadata, predictions should differ from unscaled."""
        from sklearn.preprocessing import StandardScaler

        result, X, _ = _fit_ridge(nf=5)
        scaler = StandardScaler()
        scaler.fit(X)
        artifact_scaled = artifact_from_fold(
            result,
            feature_names=tuple(f"f{i}" for i in range(5)),
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
        )
        artifact_raw = artifact_from_fold(result, feature_names=tuple(f"f{i}" for i in range(5)))
        p_scaled = predict_from_artifact(artifact_scaled, X)
        p_raw = predict_from_artifact(artifact_raw, X)
        assert not np.allclose(p_scaled, p_raw, atol=1e-8)
