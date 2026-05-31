"""Structured factor risk model and risk decomposition.

The model builds a full-rank asset covariance from three ingredients:
  Σ = B · F_cov · Bᵀ + D
where
  B      (n_assets, n_factors) — cross-sectional factor loadings
  F_cov  (n_factors, n_factors) — factor return covariance
  D      diag(specific_var)    — idiosyncratic / specific variance

Decomposition:
  Factor variance:   wᵀ · (B F_cov Bᵀ) · w
  Specific variance: wᵀ · D · w
  Total portfolio variance: their sum

Marginal contribution to risk (MCR) per asset i:
  MCR_i = (Σ w)_i         (unnormalised gradient of portfolio variance)

Component contribution to risk (CCR) per asset i:
  CCR_i = w_i · MCR_i     (sum = portfolio variance)

All inputs are plain numpy arrays; the Polars data wrangling lives in the
callers. Using numpy throughout avoids repeated DataFrame allocations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorRiskModel:
    """Assembled factor risk model for a single cross-section.

    Attributes:
        B:          (n_assets, n_factors) loading matrix.
        factor_cov: (n_factors, n_factors) factor covariance matrix.
        specific_var: (n_assets,) idiosyncratic variances (diagonal of D).
        cov:        (n_assets, n_assets) full asset covariance Σ = B F_cov Bᵀ + D.
    """

    B: np.ndarray  # (n_assets, n_factors)
    factor_cov: np.ndarray  # (n_factors, n_factors)
    specific_var: np.ndarray  # (n_assets,)
    cov: np.ndarray  # (n_assets, n_assets)

    @classmethod
    def build(
        cls,
        B: np.ndarray,
        factor_cov: np.ndarray,
        specific_var: np.ndarray,
    ) -> FactorRiskModel:
        """Construct Σ = B · F_cov · Bᵀ + D from raw ingredients.

        Args:
            B:            (n_assets, n_factors).
            factor_cov:   (n_factors, n_factors). Must be positive semi-definite.
            specific_var: (n_assets,) non-negative idiosyncratic variances.
        """
        factor_part = B @ factor_cov @ B.T
        D = np.diag(specific_var)
        cov = factor_part + D
        return cls(B=B, factor_cov=factor_cov, specific_var=specific_var, cov=cov)

    def portfolio_variance(self, w: np.ndarray) -> float:
        """Total portfolio variance wᵀ Σ w."""
        return float(w @ self.cov @ w)

    def factor_variance(self, w: np.ndarray) -> float:
        """Variance explained by systematic factor exposures."""
        Bw = self.B.T @ w
        return float(Bw @ self.factor_cov @ Bw)

    def specific_variance(self, w: np.ndarray) -> float:
        """Variance from idiosyncratic (stock-specific) risk."""
        return float((w**2) @ self.specific_var)

    def marginal_contrib(self, w: np.ndarray) -> np.ndarray:
        """Per-asset marginal contribution to portfolio variance.

        MCR_i = ∂(wᵀ Σ w)/∂w_i = 2 (Σ w)_i

        Returned without the factor of 2 so that CCR sums to portfolio
        variance (the 2 cancels in the w_i · MCR_i form used by convention).
        """
        return self.cov @ w

    def component_contrib(self, w: np.ndarray) -> np.ndarray:
        """Per-asset component contribution: CCR_i = w_i · (Σ w)_i.

        Σ_i CCR_i = portfolio_variance(w). Useful for risk budgeting.
        """
        return w * self.marginal_contrib(w)

    def factor_component_contrib(self, w: np.ndarray) -> np.ndarray:
        """Per-factor component contribution to portfolio variance.

        Returns (n_factors,) array where entry k is the contribution of
        factor k, decomposing factor_variance(w) by factor.
        """
        Bw = self.B.T @ w  # (n_factors,)
        F = self.factor_cov  # (n_factors, n_factors)
        return Bw * (F @ Bw)  # element-wise; sums to factor_variance


def build_from_long(
    factor_loadings: object,  # pl.DataFrame (date, id, factor_id, loading)
    factor_covariance: object,  # pl.DataFrame (date, factor_i, factor_j, cov)
    specific_risk: object,  # pl.DataFrame (date, id, specific_var)
    as_of_date: object,  # date
) -> FactorRiskModel:
    """Extract a single-date FactorRiskModel from long-format Polars frames.

    Filters each DataFrame to `as_of_date` and pivots to arrays. Assets and
    factors are sorted by integer id so the matrices are consistently aligned.

    Args:
        factor_loadings:  long (date, id, factor_id, loading).
        factor_covariance: long (date, factor_i, factor_j, cov).
        specific_risk:    long (date, id, specific_var).
        as_of_date:       the date to extract.
    """
    import polars as pl

    date_ = as_of_date

    # --- B: (n_assets, n_factors) ---
    bl = factor_loadings.filter(pl.col("date") == date_).sort(["id", "factor_id"])
    assets = sorted(bl["id"].unique().to_list())
    factors = sorted(bl["factor_id"].unique().to_list())
    na, nk = len(assets), len(factors)
    asset_idx = {a: i for i, a in enumerate(assets)}
    factor_idx = {f: i for i, f in enumerate(factors)}
    B = np.zeros((na, nk))
    for row in bl.iter_rows(named=True):
        B[asset_idx[row["id"]], factor_idx[row["factor_id"]]] = row["loading"]

    # --- F_cov: (n_factors, n_factors) ---
    fc = factor_covariance.filter(pl.col("date") == date_).sort(["factor_i", "factor_j"])
    F_cov = np.zeros((nk, nk))
    for row in fc.iter_rows(named=True):
        i, j = factor_idx[row["factor_i"]], factor_idx[row["factor_j"]]
        F_cov[i, j] = row["cov"]

    # --- specific_var: (n_assets,) ---
    sr = specific_risk.filter(pl.col("date") == date_).sort("id")
    spec_map = dict(zip(sr["id"].to_list(), sr["specific_var"].to_list(), strict=False))
    specific_var = np.array([spec_map[a] for a in assets])

    return FactorRiskModel.build(B=B, factor_cov=F_cov, specific_var=specific_var)
