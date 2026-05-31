"""Model persistence: serialize fitted coefficients, scaler, config, and
training-window metadata; reload for out-of-sample prediction.

Design
------
Serialization uses ``numpy.savez_compressed`` for coefficient arrays and
Python's ``pickle`` for the scaler (sklearn objects are pickle-safe).  A small
``ModelArtifact`` metadata dataclass is also pickled alongside.

The bundle is a single ``.npz`` file (``numpy.savez_compressed`` supports
arbitrary array payloads) with a separate ``_meta.pkl`` sidecar.  Both share a
base path.  This keeps the heavy arrays in numpy's native binary format (fast,
compact) while the metadata is human-readable-ish via pickle inspection.

Why not JSON?  sklearn scaler objects can't round-trip through JSON.  Why not
joblib?  joblib adds a dependency; the array + pickle approach uses the stdlib.

Public API
----------
``ModelArtifact``   — frozen dataclass capturing everything needed to predict.
``save_artifact``   — write a fitted model + scaler + metadata to disk.
``load_artifact``   — reload from disk, returning a ``ModelArtifact``.
``predict_from_artifact`` — apply a loaded artifact to new feature data,
                            including NaN masking and optional scaling.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .ridge import ModelResult


@dataclass(frozen=True)
class ModelArtifact:
    """Everything needed to score new data from a previously fitted fold.

    Attributes
    ----------
    coef:
        Fitted coefficient vector (n_features,).
    intercept:
        Fitted intercept scalar.
    train_r2:
        In-sample R² of the fitted model.
    feature_names:
        Tuple of feature column names in coefficient order.
    scaler_mean:
        Per-feature means from the train-fold scaler (None if scaling was off).
    scaler_scale:
        Per-feature std-devs from the train-fold scaler (None if scaling off).
    train_start:
        First date in the training window.
    train_end:
        Last date in the training window.
    alpha:
        Regularization strength used at fit time.
    model_type:
        String tag for the model class (e.g. ``"RidgeModel"``).
    extra:
        Arbitrary dict for model-specific metadata (e.g. l1_ratio for EN).
    """

    coef: np.ndarray
    intercept: float
    train_r2: float
    feature_names: tuple[str, ...]
    scaler_mean: np.ndarray | None
    scaler_scale: np.ndarray | None
    train_start: date | None
    train_end: date | None
    alpha: float
    model_type: str
    extra: dict[str, Any]


def _meta_path(base: Path) -> Path:
    return base.with_suffix(".meta.pkl")


def _arrays_path(base: Path) -> Path:
    return base.with_suffix(".npz")


def save_artifact(artifact: ModelArtifact, base: Path) -> None:
    """Persist ``artifact`` to disk.

    Two files are written:
      ``<base>.npz``       — numpy arrays (coef, scaler_mean, scaler_scale).
      ``<base>.meta.pkl``  — the rest of the metadata.
    """
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {"coef": artifact.coef}
    if artifact.scaler_mean is not None:
        arrays["scaler_mean"] = artifact.scaler_mean
    if artifact.scaler_scale is not None:
        arrays["scaler_scale"] = artifact.scaler_scale
    np.savez_compressed(_arrays_path(base), **arrays)

    meta = {
        "intercept": artifact.intercept,
        "train_r2": artifact.train_r2,
        "feature_names": artifact.feature_names,
        "has_scaler": artifact.scaler_mean is not None,
        "train_start": artifact.train_start,
        "train_end": artifact.train_end,
        "alpha": artifact.alpha,
        "model_type": artifact.model_type,
        "extra": artifact.extra,
    }
    with _meta_path(base).open("wb") as fh:
        pickle.dump(meta, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_artifact(base: Path) -> ModelArtifact:
    """Reload a ``ModelArtifact`` from the two files written by ``save_artifact``."""
    base = Path(base)
    arrays = np.load(_arrays_path(base))
    with _meta_path(base).open("rb") as fh:
        meta = pickle.load(fh)

    scaler_mean = arrays["scaler_mean"] if meta["has_scaler"] else None
    scaler_scale = arrays["scaler_scale"] if meta["has_scaler"] else None

    return ModelArtifact(
        coef=arrays["coef"],
        intercept=meta["intercept"],
        train_r2=meta["train_r2"],
        feature_names=meta["feature_names"],
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        train_start=meta["train_start"],
        train_end=meta["train_end"],
        alpha=meta["alpha"],
        model_type=meta["model_type"],
        extra=meta["extra"],
    )


def predict_from_artifact(
    artifact: ModelArtifact,
    X: np.ndarray,
    *,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Apply a loaded artifact to new feature data.

    NaN rows in ``X`` are masked out: predictions for those rows are ``np.nan``.
    If scaling metadata is present, the same train-fold standardization is
    applied before the linear score.

    Parameters
    ----------
    artifact:
        Loaded ``ModelArtifact``.
    X:
        (n_samples, n_features) feature matrix; may contain NaNs.
    return_mask:
        If True, also return the boolean valid-row mask (True = row was used).

    Returns
    -------
    preds:
        (n_samples,) float64 predictions; NaN rows → ``np.nan``.
    mask (optional):
        (n_samples,) bool valid-row mask.
    """
    X = np.asarray(X, dtype=np.float64)
    valid = ~np.any(np.isnan(X), axis=1)
    preds = np.full(len(X), np.nan, dtype=np.float64)

    if valid.any():
        X_valid = X[valid]
        if artifact.scaler_mean is not None and artifact.scaler_scale is not None:
            # avoid division by zero for constant features
            scale = np.where(artifact.scaler_scale > 1e-12, artifact.scaler_scale, 1.0)
            X_valid = (X_valid - artifact.scaler_mean) / scale
        preds[valid] = X_valid @ artifact.coef + artifact.intercept

    if return_mask:
        return preds, valid
    return preds


def artifact_from_fold(
    fit_result: ModelResult,
    feature_names: tuple[str, ...],
    *,
    scaler_mean: np.ndarray | None = None,
    scaler_scale: np.ndarray | None = None,
    train_start: date | None = None,
    train_end: date | None = None,
    alpha: float = 1.0,
    model_type: str = "unknown",
    extra: dict[str, Any] | None = None,
) -> ModelArtifact:
    """Convenience constructor: build a ``ModelArtifact`` from a ``ModelResult``
    and optional scaler / training-window metadata."""
    return ModelArtifact(
        coef=fit_result.coef.copy(),
        intercept=fit_result.intercept,
        train_r2=fit_result.train_r2,
        feature_names=feature_names,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        train_start=train_start,
        train_end=train_end,
        alpha=alpha,
        model_type=model_type,
        extra=extra or {},
    )
