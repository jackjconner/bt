from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .ridge import ModelResult


class FinancialModel(Protocol):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> ModelResult: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
