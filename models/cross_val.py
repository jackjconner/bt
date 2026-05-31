from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold

from .ridge import ModelConfig, ModelResult, RidgeModel


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    shuffle: bool = False  # False → walk-forward-style contiguous folds


@dataclass(frozen=True)
class CVResult:
    fold_results: list[ModelResult]
    fold_r2s: list[float]
    mean_r2: float
    std_r2: float


_DEFAULT_CV_CONFIG = CVConfig()


def cv_loop(
    model_config: ModelConfig,
    X: np.ndarray,
    y: np.ndarray,
    cv_config: CVConfig = _DEFAULT_CV_CONFIG,
) -> CVResult:
    """K-fold CV. Peak memory is one fold's train matrix: O(n_samples * n_features).
    CPU is O(K * n_samples * n_features^2)."""
    kf = KFold(n_splits=cv_config.n_splits, shuffle=cv_config.shuffle)
    fold_results: list[ModelResult] = []
    fold_r2s: list[float] = []

    for train_idx, test_idx in kf.split(X):
        model = RidgeModel(model_config)
        fit_result = model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        y_test = y[test_idx]
        ss_res = float(((y_test - preds) ** 2).sum())
        ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
        test_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        fold_r2s.append(test_r2)
        fold_results.append(fit_result)

    arr = np.array(fold_r2s)
    return CVResult(
        fold_results=fold_results,
        fold_r2s=fold_r2s,
        mean_r2=float(arr.mean()),
        std_r2=float(arr.std()),
    )
