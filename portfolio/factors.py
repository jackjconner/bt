from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .holdings import HoldingsFrame


@dataclass(frozen=True)
class FactorExposure:
    loadings: np.ndarray    # (n_assets, n_factors)  — O(n_assets * n_factors)
    exposures: np.ndarray   # (n_dates, n_factors)   — O(n_dates * n_factors)


def random_loadings(n_assets: int, n_factors: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (n_assets, n_factors))


def compute_exposures(holdings: HoldingsFrame, loadings: np.ndarray) -> FactorExposure:
    """W @ L: (n_dates, n_assets) @ (n_assets, n_factors) → (n_dates, n_factors).

    CPU is O(n_dates * n_assets * n_factors); the W materialization via
    to_wide() is the O(n_dates * n_assets) memory peak.
    """
    weights = holdings.to_wide()
    exposures = weights @ loadings
    return FactorExposure(loadings=loadings, exposures=exposures)
