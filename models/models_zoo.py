"""Additional linear model types behind the ``FinancialModel`` protocol.

``RidgeModel`` (in ``ridge.py``) is the reference implementation; this module
provides ``LassoModel`` and ``ElasticNetModel`` with identical external shapes.
All three share the same ``ModelConfig`` + ``ModelResult`` types.

Lasso vs Ridge for financial features
--------------------------------------
Ridge distributes regularization across correlated features; Lasso induces
sparsity and zeros out coefficients whose signal is dominated by other features.
When ``n_features`` is large and only a handful are truly predictive, Lasso's
automatic feature selection often out-of-samples better.  ElasticNet bridges the
two: the ``l1_ratio`` controls the mix.

Both models support ``sample_weight`` in ``fit``, forwarded to sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso

from .ridge import ModelResult


@dataclass(frozen=True)
class LassoConfig:
    """Configuration for ``LassoModel``.

    Parameters
    ----------
    n_features:
        Expected number of input features (informational; not enforced).
    alpha:
        Regularization strength.  Larger → more coefficients zeroed out.
    max_iter:
        Maximum coordinate-descent iterations.  Increase for high-dimensional,
        correlated feature matrices where convergence is slow.
    """

    n_features: int
    alpha: float = 1.0
    max_iter: int = 2000


@dataclass(frozen=True)
class ElasticNetConfig:
    """Configuration for ``ElasticNetModel``.

    Parameters
    ----------
    n_features:
        Expected number of input features (informational; not enforced).
    alpha:
        Overall regularization strength.
    l1_ratio:
        Mix between L1 and L2: 0.0 = pure Ridge, 1.0 = pure Lasso.  A value
        of 0.5 is a reasonable starting point; increase toward 1.0 for sparser
        solutions.
    max_iter:
        Maximum coordinate-descent iterations.
    """

    n_features: int
    alpha: float = 1.0
    l1_ratio: float = 0.5
    max_iter: int = 2000


class LassoModel:
    """Lasso regression; same ``fit``/``predict`` shape as ``RidgeModel``.

    The sklearn Lasso uses coordinate descent, so convergence can be slow for
    ill-conditioned or highly correlated feature matrices.  Features should be
    standardized before fitting (use ``walk_forward.FoldScaler``).
    """

    def __init__(self, config: LassoConfig) -> None:
        self.config = config
        self._model: Lasso | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> ModelResult:
        """Fit Lasso; ``sample_weight`` is forwarded to sklearn if provided."""
        model = Lasso(
            alpha=self.config.alpha,
            max_iter=self.config.max_iter,
            fit_intercept=True,
        )
        kw = {} if sample_weight is None else {"sample_weight": sample_weight}
        model.fit(X, y, **kw)
        self._model = model
        train_r2 = float(model.score(X, y, sample_weight=sample_weight))
        return ModelResult(
            coef=model.coef_,
            intercept=float(model.intercept_),
            train_r2=train_r2,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit must be called before predict")
        return self._model.predict(X)


class ElasticNetModel:
    """ElasticNet regression; same ``fit``/``predict`` shape as ``RidgeModel``.

    Combines L1 (Lasso-style sparsity) and L2 (Ridge-style coefficient shrinkage).
    Well-suited for correlated feature sets where Lasso arbitrarily picks one
    correlated feature while Ridge spreads weight across all.
    """

    def __init__(self, config: ElasticNetConfig) -> None:
        self.config = config
        self._model: ElasticNet | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> ModelResult:
        """Fit ElasticNet; ``sample_weight`` is forwarded to sklearn if provided."""
        model = ElasticNet(
            alpha=self.config.alpha,
            l1_ratio=self.config.l1_ratio,
            max_iter=self.config.max_iter,
            fit_intercept=True,
        )
        kw = {} if sample_weight is None else {"sample_weight": sample_weight}
        model.fit(X, y, **kw)
        self._model = model
        train_r2 = float(model.score(X, y, sample_weight=sample_weight))
        return ModelResult(
            coef=model.coef_,
            intercept=float(model.intercept_),
            train_r2=train_r2,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit must be called before predict")
        return self._model.predict(X)
