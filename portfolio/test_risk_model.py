"""Tests for the factor risk model."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from portfolio.risk_model import (
    FactorRiskBreakdown,
    FactorRiskModel,
    build_from_long,
)


@pytest.fixture
def small_model():
    rng = np.random.default_rng(0)
    n_assets, n_factors = 10, 3
    B = rng.normal(0.0, 1.0, (n_assets, n_factors))
    A = rng.normal(0.0, 1.0, (n_factors, n_factors))
    F = A @ A.T / n_factors + np.eye(n_factors)
    specific_var = np.abs(rng.normal(1.0, 0.3, n_assets))
    return FactorRiskModel.build(B=B, factor_cov=F, specific_var=specific_var)


@pytest.fixture
def weights(small_model):
    n = small_model.B.shape[0]
    return np.ones(n) / n


class TestFactorRiskModel:
    def test_cov_shape(self, small_model):
        n = small_model.B.shape[0]
        assert small_model.cov.shape == (n, n)

    def test_cov_symmetric(self, small_model):
        np.testing.assert_allclose(small_model.cov, small_model.cov.T, atol=1e-12)

    def test_cov_positive_definite(self, small_model):
        eigvals = np.linalg.eigvalsh(small_model.cov)
        assert eigvals.min() > 0.0, f"min eigenvalue {eigvals.min()}"

    def test_portfolio_variance_positive(self, small_model, weights):
        pvar = small_model.portfolio_variance(weights)
        assert pvar > 0.0

    def test_variance_decomposition_sums(self, small_model, weights):
        """Factor + specific variance should sum to total portfolio variance."""
        total = small_model.portfolio_variance(weights)
        fac = small_model.factor_variance(weights)
        spec = small_model.specific_variance(weights)
        assert abs(total - (fac + spec)) < 1e-10

    def test_component_contrib_sums_to_variance(self, small_model, weights):
        ccr = small_model.component_contrib(weights)
        total = small_model.portfolio_variance(weights)
        assert abs(ccr.sum() - total) < 1e-10

    def test_factor_component_contrib_sums(self, small_model, weights):
        fcc = small_model.factor_component_contrib(weights)
        fac_var = small_model.factor_variance(weights)
        assert abs(fcc.sum() - fac_var) < 1e-10

    def test_factor_risk_breakdown_total_equals_components(self, small_model, weights):
        """factor + specific variance must sum to total variance."""
        bd = small_model.factor_risk_breakdown(weights)
        assert isinstance(bd, FactorRiskBreakdown)
        assert abs(bd.total_variance - (bd.factor_variance + bd.specific_variance)) < 1e-12

    def test_factor_risk_breakdown_matches_scalar_helpers(self, small_model, weights):
        """Breakdown fields must match the lower-level scalar helpers exactly."""
        bd = small_model.factor_risk_breakdown(weights)
        assert bd.total_variance == small_model.portfolio_variance(weights)
        assert bd.factor_variance == small_model.factor_variance(weights)
        assert bd.specific_variance == small_model.specific_variance(weights)

    def test_factor_risk_breakdown_per_factor_sums_to_factor_variance(self, small_model, weights):
        """Per-factor contributions must sum to the factor variance."""
        bd = small_model.factor_risk_breakdown(weights)
        n_factors = small_model.B.shape[1]
        assert bd.factor_contrib.shape == (n_factors,)
        assert abs(bd.factor_contrib.sum() - bd.factor_variance) < 1e-12

    def test_factor_risk_breakdown_reuses_factor_component_contrib(self, small_model, weights):
        """Per-factor contributions must equal factor_component_contrib (no dup)."""
        bd = small_model.factor_risk_breakdown(weights)
        np.testing.assert_array_equal(
            bd.factor_contrib, small_model.factor_component_contrib(weights)
        )

    def test_factor_risk_breakdown_matches_dense_analytic_form(self, small_model, weights):
        """Breakdown must match the explicit factored (B F Bᵀ + D) decomposition."""
        w = weights
        B, F, sv = small_model.B, small_model.factor_cov, small_model.specific_var
        factor_var_dense = float(w @ (B @ F @ B.T) @ w)
        specific_var_dense = float(w @ np.diag(sv) @ w)
        bd = small_model.factor_risk_breakdown(w)
        assert abs(bd.factor_variance - factor_var_dense) < 1e-10
        assert abs(bd.specific_variance - specific_var_dense) < 1e-10
        assert abs(bd.total_variance - (factor_var_dense + specific_var_dense)) < 1e-10

    def test_factor_risk_breakdown_fractions_sum_to_one(self, small_model, weights):
        """factor_fraction + specific_fraction must sum to 1 for nonzero variance."""
        bd = small_model.factor_risk_breakdown(weights)
        assert abs(bd.factor_fraction + bd.specific_fraction - 1.0) < 1e-12

    def test_factor_risk_breakdown_zero_weights(self, small_model):
        """All-zero weights give zero variance and zero (not NaN) fractions."""
        n = small_model.B.shape[0]
        bd = small_model.factor_risk_breakdown(np.zeros(n))
        assert bd.total_variance == 0.0
        assert bd.factor_fraction == 0.0
        assert bd.specific_fraction == 0.0

    def test_zero_specific_risk_variance_is_factor_only(self):
        rng = np.random.default_rng(1)
        n, k = 5, 2
        B = rng.normal(0.0, 1.0, (n, k))
        A = rng.normal(0.0, 1.0, (k, k))
        F = A @ A.T + np.eye(k)
        model = FactorRiskModel.build(B, F, np.zeros(n))
        w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        assert abs(model.specific_variance(w)) < 1e-15
        assert abs(model.portfolio_variance(w) - model.factor_variance(w)) < 1e-12


class TestBuildFromLong:
    def test_roundtrip(self):
        """build_from_long should recover the same structure as direct .build()."""
        rng = np.random.default_rng(7)
        n_assets, n_factors = 4, 2
        d = date(2024, 1, 2)

        # Create long-format Polars frames for a single date
        ids = list(range(n_assets))
        fids = list(range(n_factors))

        # factor_loadings
        rows_fl = []
        for i in ids:
            for f in fids:
                rows_fl.append({"date": d, "id": i, "factor_id": f, "loading": rng.normal()})
        fl = pl.DataFrame(rows_fl).with_columns(
            pl.col("id").cast(pl.Int64),
            pl.col("factor_id").cast(pl.Int64),
        )

        # factor_covariance (full symmetric matrix)
        raw = rng.normal(0.0, 1.0, (n_factors, n_factors))
        F = raw @ raw.T + np.eye(n_factors)
        rows_fc = []
        for i in fids:
            for j in fids:
                rows_fc.append({"date": d, "factor_i": i, "factor_j": j, "cov": F[i, j]})
        fc = pl.DataFrame(rows_fc).with_columns(
            pl.col("factor_i").cast(pl.Int64),
            pl.col("factor_j").cast(pl.Int64),
        )

        # specific_risk
        sv = np.abs(rng.normal(1.0, 0.3, n_assets))
        sr = pl.DataFrame(
            {"date": [d] * n_assets, "id": ids, "specific_var": sv.tolist()}
        ).with_columns(pl.col("id").cast(pl.Int64))

        model = build_from_long(fl, fc, sr, d)

        assert model.cov.shape == (n_assets, n_assets)
        # covariance should be positive definite
        eigvals = np.linalg.eigvalsh(model.cov)
        assert eigvals.min() > 0.0
