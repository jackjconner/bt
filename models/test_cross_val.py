from __future__ import annotations

import numpy as np

from .cross_val import CVConfig, cv_loop
from .ridge import ModelConfig, RidgeModel


def _xy(n_samples: int, n_features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (n_samples, n_features))
    beta = rng.normal(0.0, 1.0, n_features)
    y = X @ beta + rng.normal(0.0, 0.1, n_samples)
    return X, y


def test_fold_train_r2_is_train_not_test() -> None:
    """fold_results[i].train_r2 must be the train-set R^2, recomputed here
    independently of cv_loop. The held-out R^2 lives in fold_r2s, and on a
    signal-bearing dataset the two differ — so a regression that puts the
    test R^2 into train_r2 (the original bug) fails this."""
    X, y = _xy(120, 5, seed=0)
    cfg = ModelConfig(n_features=5)
    result = cv_loop(cfg, X, y, CVConfig(n_splits=4))

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=4, shuffle=False)
    for (train_idx, _), fold in zip(kf.split(X), result.fold_results, strict=True):
        expected_train_r2 = RidgeModel(cfg).fit(X[train_idx], y[train_idx]).train_r2
        assert fold.train_r2 == expected_train_r2


def test_fold_r2s_are_holdout() -> None:
    """fold_r2s are the held-out scores, recomputable independently, and
    distinct from the in-sample train_r2 of the same fold."""
    from sklearn.model_selection import KFold

    X, y = _xy(120, 5, seed=1)
    cfg = ModelConfig(n_features=5)
    result = cv_loop(cfg, X, y, CVConfig(n_splits=4))

    kf = KFold(n_splits=4, shuffle=False)
    for (train_idx, test_idx), fold, test_r2 in zip(
        kf.split(X), result.fold_results, result.fold_r2s, strict=True
    ):
        model = RidgeModel(cfg)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        y_test = y[test_idx]
        ss_res = float(((y_test - preds) ** 2).sum())
        ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
        expected_test_r2 = 1.0 - ss_res / ss_tot
        assert test_r2 == expected_test_r2
        assert test_r2 != fold.train_r2
