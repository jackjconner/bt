from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class RidgeModel:
    config: ModelConfig
    _model: Ridge = field(compare=False, repr=False, default=None)

    def _fresh(self) -> Ridge:
        return Ridge(alpha=self.config.alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> ModelResult:
        """Closed-form ridge. CPU is O(n_samples * n_features^2); the Gram
        matrix path costs O(n_features^2) memory when n_features > n_samples.
        """
        model = self._fresh()
        model.fit(X, y)
        object.__setattr__(self, "_model", model)
        return ModelResult(
            coef=model.coef_,
            intercept=float(model.intercept_),
            train_r2=float(model.score(X, y)),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit must be called before predict")
        return self._model.predict(X)
