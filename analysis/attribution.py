"""Factor and sector return attribution.

Why this file: attribution is the most data-hungry analysis — it needs factor
returns, factor loadings, and optionally a sector map. Keeping it isolated
prevents the lighter modules (risk.py, benchmark.py) from pulling in those
dependencies.

Approach: OLS regression of strategy daily returns on contemporaneous factor
returns. The regression coefficients are the factor exposures; the intercept is
the factor-unexplained (idiosyncratic) return component.

    r_t = alpha + sum_k(beta_k * f_k_t) + epsilon_t

Factor returns `f` from `factor_returns` are in PERCENT; caller must divide by
100 before passing here (same convention as daily returns).

Sector attribution uses the `security_master` sector map and a weight panel
from `turnover.reconstruct_weights` to compute the per-sector contribution to
total return on each date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Factor attribution via OLS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorAttributionResult:
    """Output of a single factor regression.

    `factor_exposures`: dict mapping factor_id (int) → beta coefficient.
    `alpha_annualized`: annualized daily intercept (fractional).
    `r_squared`: fraction of strategy variance explained by the factors.
    `residual_returns`: date-indexed DataFrame `(date, residual)` — the
        idiosyncratic return series after stripping factor contributions.
    """

    factor_exposures: dict[int, float]
    alpha_annualized: float
    r_squared: float
    residual_returns: pl.DataFrame


def factor_attribution(
    returns: pl.DataFrame,
    factor_returns: pl.DataFrame,
) -> FactorAttributionResult:
    """OLS regression of strategy returns on factor returns.

    `returns`: `(date, return_1d)` in fractional units.
    `factor_returns`: long `(date, factor_id, return)` in fractional units
        (caller divides percent returns by 100).

    The regression is pooled across all dates in the inner-join intersection.
    Factor returns are pivoted to wide format `(date, f_0, f_1, ...)` before
    forming the design matrix.

    Returns a `FactorAttributionResult` with per-factor betas, annualized
    alpha, R², and the residual return series.
    """
    TRADING_DAYS = 252

    # Pivot factor returns to (date, factor_0, factor_1, ...)
    wide_factors = factor_returns.sort(["date", "factor_id"]).pivot(
        on="factor_id", index="date", values="return"
    )

    joined = returns.join(wide_factors, on="date", how="inner")
    factor_cols = [c for c in wide_factors.columns if c != "date"]
    if not factor_cols:
        raise ValueError("factor_returns has no factor columns after pivot")

    y = joined["return_1d"].to_numpy()
    X_raw = joined.select(factor_cols).to_numpy()
    n = len(y)

    # Prepend intercept column
    X = np.column_stack([np.ones(n), X_raw])

    # OLS via normal equations; use lstsq for numerical stability
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    intercept = float(coeffs[0])
    betas = coeffs[1:]

    y_hat = X @ coeffs
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    residuals = y - y_hat

    factor_ids = [int(c) for c in factor_cols]
    exposures = {fid: float(b) for fid, b in zip(factor_ids, betas, strict=False)}

    resid_df = joined.select("date").with_columns(pl.Series("residual", residuals))

    return FactorAttributionResult(
        factor_exposures=exposures,
        alpha_annualized=intercept * TRADING_DAYS,
        r_squared=float(r2),
        residual_returns=resid_df,
    )


# ---------------------------------------------------------------------------
# Sector attribution
# ---------------------------------------------------------------------------


def sector_attribution(
    returns: pl.DataFrame,
    weights: pl.DataFrame,
    security_master: pl.DataFrame,
    asset_returns: pl.DataFrame,
) -> pl.DataFrame:
    """Per-sector contribution to portfolio return.

    For each date:
        sector_contribution[s] = sum_i(w_i * r_i)  for all i in sector s

    `weights`: `(date, id, weight)` from `turnover.reconstruct_weights`.
    `security_master`: `(id, sector, ...)` from the dataset layer.
    `asset_returns`: long `(date, id, return_1d)` in fractional units — the
        per-asset daily returns used by the backtest engine.

    Returns a long DataFrame `(date, sector, sector_contribution)`. Days
    without weight data are excluded (rebalance-date-only weights are
    forward-filled before calling this function in practice, but this function
    operates on whatever dates are in `weights`).
    """
    sector_map = security_master.select("id", "sector")

    # Attach sector to weights
    w_with_sector = weights.join(sector_map, on="id", how="left")

    # Attach asset return to weights
    w_with_ret = w_with_sector.join(
        asset_returns.select("date", "id", "return_1d"),
        on=["date", "id"],
        how="left",
    ).with_columns(pl.col("return_1d").fill_null(0.0))

    # Contribution = weight * return
    contrib = w_with_ret.with_columns(
        (pl.col("weight") * pl.col("return_1d")).alias("contribution")
    )

    return (
        contrib.group_by(["date", "sector"])
        .agg(pl.col("contribution").sum().alias("sector_contribution"))
        .sort(["date", "sector"])
    )
