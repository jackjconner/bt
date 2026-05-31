from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class ModelConfig:
    n_features: int
    alpha: float = 1.0


@dataclass(frozen=True)
class ModelResult:
    coef: np.ndarray    # (n_features,)  — O(n_features)
    intercept: float
    train_r2: float


class RidgeModel:
    """Stateful estimator: `fit` trains and retains the fitted Ridge for
    `predict`, sklearn-style. Not frozen because the fit is the state."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._model: Ridge | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> ModelResult:
        """Closed-form ridge. CPU is O(n_samples * n_features^2); the Gram
        matrix path costs O(n_features^2) memory when n_features > n_samples.
        ``sample_weight`` is forwarded to sklearn's Ridge so recency/vol/
        liquidity weighting can be applied without subclassing.
        """
        model = Ridge(alpha=self.config.alpha)
        if sample_weight is not None:
            model.fit(X, y, sample_weight=sample_weight)
        else:
            model.fit(X, y)
        self._model = model
        return ModelResult(
            coef=model.coef_,
            intercept=float(model.intercept_),
            train_r2=float(model.score(X, y, sample_weight=sample_weight)),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit must be called before predict")
        return self._model.predict(X)
