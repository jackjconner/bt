"""Gradient-boosted regressor behind the ``FinancialModel`` protocol.

``GradientBoostModel`` wraps sklearn's ``HistGradientBoostingRegressor``, which
uses a histogram-based algorithm (similar to LightGBM) that handles large
datasets efficiently and natively supports missing values.

Why gradient boosting for financial features
---------------------------------------------
Linear models impose a global linearity assumption.  Asset-return signals can
exhibit threshold or interaction effects (e.g. momentum only works in low-vol
regimes) that gradient-boosted trees capture without manual feature engineering.
``HistGradientBoostingRegressor`` is a practical choice: it requires no
preprocessing (no scaling needed), handles missing values internally, and its
histogram approximation keeps wall-time tractable for panel datasets.

``sample_weight`` is forwarded to sklearn when provided, so recency/vol/
liquidity weighting from ``walk_forward_cv`` flows through unchanged.

Determinism
-----------
``random_state`` is always fixed so repeated calls with identical data produce
identical predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .ridge import ModelResult


@dataclass(frozen=True)
class GradientBoostConfig:
    """Configuration for ``GradientBoostModel``.

    Parameters
    ----------
    n_features:
        Expected number of input features (informational; not enforced).
    learning_rate:
        Shrinkage applied to each tree's contribution.  Smaller values reduce
        overfitting but require more trees.
    max_iter:
        Maximum number of boosting iterations (trees).
    max_depth:
        Maximum depth of each individual tree.  ``None`` means unlimited.
    min_samples_leaf:
        Minimum number of samples in a leaf node.  Higher values regularize.
    l2_regularization:
        L2 regularization term on leaf values (analogous to Ridge alpha).
    random_state:
        Fixed seed for reproducibility.
    """

    n_features: int
    learning_rate: float = 0.05
    max_iter: int = 200
    max_depth: int | None = 4
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    random_state: int = 42


class GradientBoostModel:
    """Gradient-boosted regressor; same ``fit``/``predict`` shape as ``RidgeModel``.

    Wraps ``HistGradientBoostingRegressor`` behind the ``FinancialModel`` protocol
    so it can be used as a drop-in replacement anywhere ``RidgeModel`` is accepted,
    including inside ``walk_forward_cv``.

    ``ModelResult.coef`` is populated as a zero-vector of length ``n_features``
    because gradient-boosted trees do not have a single linear coefficient vector.
    ``ModelResult.intercept`` is set to the baseline prediction (``_raw_predict``
    on a zero input) so the result field is informative rather than meaningless.
    ``ModelResult.train_r2`` is the standard R² on the training set.
    """

    def __init__(self, config: GradientBoostConfig) -> None:
        self.config = config
        self._model: HistGradientBoostingRegressor | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> ModelResult:
        """Fit the boosted model; ``sample_weight`` is forwarded when provided."""
        model = HistGradientBoostingRegressor(
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            random_state=self.config.random_state,
        )
        if sample_weight is not None:
            model.fit(X, y, sample_weight=sample_weight)
        else:
            model.fit(X, y)
        self._model = model
        train_r2 = float(model.score(X, y, sample_weight=sample_weight))
        # Trees have no linear coefficients; use a zero vector so ModelResult
        # is structurally identical to linear-model results.
        coef = np.zeros(X.shape[1], dtype=np.float64)
        # Use mean prediction on zero-input as a proxy for the intercept.
        intercept = float(model.predict(np.zeros((1, X.shape[1])))[0])
        return ModelResult(coef=coef, intercept=intercept, train_r2=train_r2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit must be called before predict")
        return self._model.predict(X)
