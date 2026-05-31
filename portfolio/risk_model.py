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

Structure over density
----------------------
The full asset covariance Σ is (n_assets × n_assets) and is the only object
in this module that scales as n²; everything the downstream optimizer needs —
``portfolio_variance``, ``factor_variance``, ``marginal_contrib`` — is a
*structured* product of the (n×k) loadings B, the (k×k) factor covariance and
the length-n specific variance, all of which scale linearly in n for fixed k.

So Σ is never materialised on the build path: ``cov`` is a lazily-computed,
cached property.  Code that genuinely needs the dense matrix (risk
decomposition reports, eigenvalue / PSD checks) touches ``.cov`` and pays the
n² cost on first access only; the optimizer hot path never does.  Marginal and
component contributions likewise route through the factored form
  Σ w = B (F (Bᵀ w)) + specific_var ⊙ w
which costs O(n·k + k²) instead of forming and multiplying the dense Σ.

All linear-algebra inputs are plain numpy arrays; the Polars→matrix pivot in
``build_from_long`` is done once, vectorised, with no per-row Python loop and
without ever holding a dense long frame alongside the matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl


@dataclass(frozen=True)
class FactorRiskModel:
    """Assembled factor risk model for a single cross-section.

    Attributes:
        B:          (n_assets, n_factors) loading matrix.
        factor_cov: (n_factors, n_factors) factor covariance matrix.
        specific_var: (n_assets,) idiosyncratic variances (diagonal of D).
        cov:        (n_assets, n_assets) full asset covariance Σ = B F_cov Bᵀ + D.
                    Lazily materialised and cached on first access — the build
                    path and the optimizer never trigger it.
    """

    B: np.ndarray  # (n_assets, n_factors)
    factor_cov: np.ndarray  # (n_factors, n_factors)
    specific_var: np.ndarray  # (n_assets,)
    # Cache slot for the dense Σ.  A single-element list rather than an
    # attribute so the frozen dataclass can populate it without __setattr__
    # gymnastics.  Excluded from init/repr/eq/hash: two models with equal
    # ingredients compare equal regardless of whether either has realised .cov.
    _cov_cache: list[np.ndarray] = field(
        default_factory=list, init=False, repr=False, compare=False, hash=False
    )

    @classmethod
    def build(
        cls,
        B: np.ndarray,
        factor_cov: np.ndarray,
        specific_var: np.ndarray,
    ) -> FactorRiskModel:
        """Construct the model from raw ingredients.

        Σ = B · F_cov · Bᵀ + D is *not* formed here — it is realised lazily on
        first access to ``.cov``.  Build is therefore O(1) in allocation beyond
        retaining the (linear-in-n) ingredients.

        Args:
            B:            (n_assets, n_factors).
            factor_cov:   (n_factors, n_factors). Must be positive semi-definite.
            specific_var: (n_assets,) non-negative idiosyncratic variances.
        """
        return cls(
            B=np.asarray(B, dtype=float),
            factor_cov=np.asarray(factor_cov, dtype=float),
            specific_var=np.asarray(specific_var, dtype=float),
        )

    @property
    def cov(self) -> np.ndarray:
        """Dense Σ = B F_cov Bᵀ + D, materialised and cached on first access.

        This is the only n² object in the model.  Tests, PSD/eigenvalue checks,
        and risk-decomposition reports that need the explicit matrix read it
        here; the optimizer and the variance/contribution helpers below stay on
        the factored path and never trigger this allocation.
        """
        if not self._cov_cache:
            cov = self.B @ self.factor_cov @ self.B.T
            # Add D in place along the diagonal — avoids a second n² allocation
            # that np.diag(specific_var) would incur.
            diag = np.einsum("ii->i", cov)
            diag += self.specific_var
            self._cov_cache.append(cov)
        return self._cov_cache[0]

    def portfolio_variance(self, w: np.ndarray) -> float:
        """Total portfolio variance wᵀ Σ w (factored: O(n·k + k²), no dense Σ)."""
        return self.factor_variance(w) + self.specific_variance(w)

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

        Computed via the factored form Σ w = B (F (Bᵀ w)) + specific_var ⊙ w,
        which is O(n·k + k²) and never forms the dense Σ.
        """
        Bw = self.B.T @ w  # (k,)
        return self.B @ (self.factor_cov @ Bw) + self.specific_var * w

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
    factor_loadings: pl.DataFrame,  # (date, id, factor_id, loading)
    factor_covariance: pl.DataFrame,  # (date, factor_i, factor_j, cov)
    specific_risk: pl.DataFrame,  # (date, id, specific_var)
    as_of_date: date,
) -> FactorRiskModel:
    """Extract a single-date FactorRiskModel from long-format Polars frames.

    Filters each frame to ``as_of_date`` and pivots to dense matrices in one
    vectorised pass per frame — no per-row Python loop, no dense long frame
    held alongside the matrices.  Assets and factors are sorted by integer id
    so B, F_cov and specific_var are consistently aligned, matching the
    ordering the optimizer expects.

    Args:
        factor_loadings:  long (date, id, factor_id, loading).
        factor_covariance: long (date, factor_i, factor_j, cov).
        specific_risk:    long (date, id, specific_var).
        as_of_date:       the date to extract.
    """
    import polars as pl

    date_ = as_of_date

    # --- B: (n_assets, n_factors) -----------------------------------------
    # Pull only the columns we need for this date as contiguous numpy, then
    # scatter into a zero matrix by dense rank of (id, factor_id).  The whole
    # pivot is three numpy arrays + np.unique; the long rows are never widened
    # into a dense intermediate frame.
    bl = factor_loadings.filter(pl.col("date") == date_)
    ids = bl["id"].to_numpy()
    fids = bl["factor_id"].to_numpy()
    loadings = bl["loading"].to_numpy()

    assets, asset_pos = np.unique(ids, return_inverse=True)
    factors, factor_pos = np.unique(fids, return_inverse=True)
    na, nk = int(assets.shape[0]), int(factors.shape[0])

    B = np.zeros((na, nk), dtype=float)
    B[asset_pos, factor_pos] = loadings

    # --- F_cov: (n_factors, n_factors) ------------------------------------
    # factor_i / factor_j index into the same factor universe as B's columns;
    # searchsorted maps each factor id to its column index in sorted `factors`.
    fc = factor_covariance.filter(pl.col("date") == date_)
    fi_pos = np.searchsorted(factors, fc["factor_i"].to_numpy())
    fj_pos = np.searchsorted(factors, fc["factor_j"].to_numpy())
    F_cov = np.zeros((nk, nk), dtype=float)
    F_cov[fi_pos, fj_pos] = fc["cov"].to_numpy()

    # --- specific_var: (n_assets,) ----------------------------------------
    sr = specific_risk.filter(pl.col("date") == date_)
    s_pos = np.searchsorted(assets, sr["id"].to_numpy())
    specific_var = np.zeros(na, dtype=float)
    specific_var[s_pos] = sr["specific_var"].to_numpy()

    return FactorRiskModel.build(B=B, factor_cov=F_cov, specific_var=specific_var)
