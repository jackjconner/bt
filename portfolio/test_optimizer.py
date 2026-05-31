"""Tests for the mean-variance optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from portfolio.constraints import ConstraintSpec
from portfolio.optimizer import mean_variance


def make_diagonal_cov(vols: np.ndarray) -> np.ndarray:
    """Diagonal covariance from a vol vector — no off-diagonal risk."""
    return np.diag(vols**2)


@pytest.fixture
def simple_spec():
    """5-asset long-only, budget=1, equal per-name cap of 0.4."""
    return ConstraintSpec(
        n_assets=5,
        long_only=True,
        min_weight=0.0,
        max_weight=0.4,
        net_exposure=1.0,
    )


class TestMeanVarianceBasic:
    def test_weights_sum_to_one(self, simple_spec):
        rng = np.random.default_rng(0)
        alpha = rng.normal(0.0, 1.0, 5)
        cov = make_diagonal_cov(np.ones(5))
        res = mean_variance(alpha, cov, simple_spec)
        assert abs(res.weights.sum() - 1.0) < 1e-6

    def test_all_weights_nonneg(self, simple_spec):
        rng = np.random.default_rng(1)
        alpha = rng.normal(0.0, 1.0, 5)
        cov = make_diagonal_cov(np.ones(5))
        res = mean_variance(alpha, cov, simple_spec)
        assert np.all(res.weights >= -1e-8)

    def test_max_weight_bound_respected(self, simple_spec):
        rng = np.random.default_rng(2)
        alpha = rng.normal(0.0, 1.0, 5)
        cov = make_diagonal_cov(np.ones(5))
        res = mean_variance(alpha, cov, simple_spec)
        # 0.4 cap + numerical tolerance
        assert np.all(res.weights <= 0.4 + 1e-6)

    def test_high_alpha_asset_gets_max_weight(self, simple_spec):
        """If one asset has much higher alpha, it should hit its cap."""
        alpha = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        cov = make_diagonal_cov(np.ones(5))
        res = mean_variance(alpha, cov, simple_spec, risk_aversion=0.1)
        # Asset 0 should be at or near the 0.4 cap
        assert res.weights[0] >= 0.35

    def test_converged(self, simple_spec):
        rng = np.random.default_rng(3)
        alpha = rng.normal(0.0, 1.0, 5)
        cov = make_diagonal_cov(np.ones(5))
        res = mean_variance(alpha, cov, simple_spec)
        assert res.converged

    def test_equal_alpha_roughly_equal_weights(self, simple_spec):
        """Zero alpha → minimum variance → weights should be roughly equal."""
        alpha = np.zeros(5)
        cov = make_diagonal_cov(np.ones(5))
        res = mean_variance(alpha, cov, simple_spec, risk_aversion=1.0)
        np.testing.assert_allclose(res.weights, 0.2, atol=0.05)


class TestMeanVarianceLongShort:
    def test_net_exposure_zero(self):
        """Dollar-neutral (Σw = 0) long-short portfolio."""
        spec = ConstraintSpec(
            n_assets=6,
            long_only=False,
            min_weight=-0.3,
            max_weight=0.3,
            net_exposure=0.0,
        )
        alpha = np.array([1.0, 0.8, 0.5, -0.5, -0.8, -1.0])
        cov = make_diagonal_cov(np.ones(6))
        res = mean_variance(alpha, cov, spec)
        assert abs(res.weights.sum()) < 1e-5

    def test_long_short_positive_alpha_positive_weight(self):
        """Top-alpha assets get positive weights, bottom-alpha get negative."""
        spec = ConstraintSpec(
            n_assets=4,
            long_only=False,
            min_weight=-0.5,
            max_weight=0.5,
            net_exposure=0.0,
        )
        alpha = np.array([2.0, 1.0, -1.0, -2.0])
        cov = make_diagonal_cov(np.ones(4))
        res = mean_variance(alpha, cov, spec, risk_aversion=0.01)
        assert res.weights[0] > 0.0
        assert res.weights[3] < 0.0


class TestMeanVarianceCostPenalty:
    def test_cost_reduces_turnover(self):
        """With a large cost penalty, weights should stay close to w0."""
        n = 5
        spec = ConstraintSpec(n_assets=n, long_only=True, max_weight=1.0)
        rng = np.random.default_rng(99)
        w0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        alpha = rng.normal(0.0, 1.0, n)
        cov = make_diagonal_cov(np.ones(n))

        res_cheap = mean_variance(alpha, cov, spec, risk_aversion=1.0, w0=w0, cost_scale=0.0)
        res_costly = mean_variance(alpha, cov, spec, risk_aversion=1.0, w0=w0, cost_scale=50.0)

        to_cheap = np.abs(res_cheap.weights - w0).sum()
        to_costly = np.abs(res_costly.weights - w0).sum()
        assert to_costly < to_cheap


class TestMeanVarianceSectorConstraints:
    def test_sector_max_respected(self):
        """Sector exposure should not exceed the per-sector cap."""
        n = 6
        # Sectors: 0,1,2 → sector A (label 0), 3,4,5 → sector B (label 1)
        sector_map = np.array([0, 0, 0, 1, 1, 1])
        spec = ConstraintSpec(
            n_assets=n,
            long_only=True,
            min_weight=0.0,
            max_weight=1.0,
            net_exposure=1.0,
            sector_map=sector_map,
            sector_min={0: 0.0, 1: 0.0},
            sector_max={0: 0.5, 1: 0.5},
        )
        alpha = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0])  # all alpha in sector A
        cov = make_diagonal_cov(np.ones(n))
        res = mean_variance(alpha, cov, spec, risk_aversion=0.01)
        sector_a_exp = res.weights[:3].sum()
        sector_b_exp = res.weights[3:].sum()
        assert sector_a_exp <= 0.5 + 1e-5
        assert sector_b_exp <= 0.5 + 1e-5


# ---------------------------------------------------------------------------
# OSQP-specific tests
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_spec_osqp():
    """5-asset long-only, budget=1, equal per-name cap of 0.4 — for OSQP tests."""
    return ConstraintSpec(
        n_assets=5,
        long_only=True,
        min_weight=0.0,
        max_weight=0.4,
        net_exposure=1.0,
    )


class TestOSQPBasic:
    """OSQP path mirrors the SLSQP path — same constraint coverage."""

    def test_weights_sum_to_one(self, simple_spec_osqp):
        rng = np.random.default_rng(0)
        alpha = rng.normal(0.0, 1.0, 5)
        cov = make_diagonal_cov(np.ones(5))
        res = mean_variance(alpha, cov, simple_spec_osqp, solver="osqp")
        assert abs(res.weights.sum() - 1.0) < 1e-5

    def test_all_weights_nonneg(self, simple_spec_osqp):
        rng = np.random.default_rng(1)
        alpha = rng.normal(0.0, 1.0, 5)
        cov = make_diagonal_cov(np.ones(5))
        res = mean_variance(alpha, cov, simple_spec_osqp, solver="osqp")
        assert np.all(res.weights >= -1e-6)

    def test_max_weight_bound_respected(self, simple_spec_osqp):
        rng = np.random.default_rng(2)
        alpha = rng.normal(0.0, 1.0, 5)
        cov = make_diagonal_cov(np.ones(5))
        res = mean_variance(alpha, cov, simple_spec_osqp, solver="osqp")
        assert np.all(res.weights <= 0.4 + 1e-5)

    def test_converged(self, simple_spec_osqp):
        rng = np.random.default_rng(3)
        alpha = rng.normal(0.0, 1.0, 5)
        cov = make_diagonal_cov(np.ones(5))
        res = mean_variance(alpha, cov, simple_spec_osqp, solver="osqp")
        assert res.converged

    def test_high_alpha_asset_gets_max_weight(self, simple_spec_osqp):
        """If one asset has much higher alpha, it should hit its cap."""
        alpha = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        cov = make_diagonal_cov(np.ones(5))
        res = mean_variance(alpha, cov, simple_spec_osqp, risk_aversion=0.1, solver="osqp")
        assert res.weights[0] >= 0.35

    def test_equal_alpha_roughly_equal_weights(self, simple_spec_osqp):
        """Zero alpha → minimum variance → weights should be roughly equal."""
        alpha = np.zeros(5)
        cov = make_diagonal_cov(np.ones(5))
        res = mean_variance(alpha, cov, simple_spec_osqp, risk_aversion=1.0, solver="osqp")
        np.testing.assert_allclose(res.weights, 0.2, atol=0.05)


class TestOSQPObjectiveDominates:
    """OSQP should find an optimum at least as good as SLSQP on the same problem.

    The QP reformulation with exact L1 linearisation is a globally optimal
    solve (modulo solver tolerance), so its objective should be ≥ SLSQP's.
    """

    def test_osqp_objective_geq_slsqp(self):
        """OSQP objective ≥ SLSQP objective on a cost-penalised problem."""
        n = 10
        spec = ConstraintSpec(n_assets=n, long_only=True, max_weight=0.3, net_exposure=1.0)
        rng = np.random.default_rng(7)
        alpha = rng.normal(0.0, 1.0, n)
        # Dense covariance: random PSD matrix
        A = rng.normal(0.0, 1.0, (n, n))
        cov = A @ A.T / n + np.eye(n) * 0.1
        w0 = np.full(n, 1.0 / n)
        res_slsqp = mean_variance(
            alpha, cov, spec, risk_aversion=1.0, w0=w0, cost_scale=5.0, solver="slsqp"
        )
        res_osqp = mean_variance(
            alpha, cov, spec, risk_aversion=1.0, w0=w0, cost_scale=5.0, solver="osqp"
        )
        # Allow a tiny tolerance for solver precision differences
        assert res_osqp.obj_value >= res_slsqp.obj_value - 1e-5

    def test_osqp_constraints_satisfied(self):
        """OSQP solution satisfies all constraints."""
        n = 8
        spec = ConstraintSpec(
            n_assets=n,
            long_only=False,
            min_weight=-0.2,
            max_weight=0.4,
            net_exposure=1.0,
        )
        rng = np.random.default_rng(17)
        alpha = rng.normal(0.0, 1.0, n)
        A = rng.normal(0.0, 1.0, (n, n))
        cov = A @ A.T / n + np.eye(n) * 0.1
        w0 = np.full(n, 1.0 / n)
        res = mean_variance(
            alpha, cov, spec, risk_aversion=1.0, w0=w0, cost_scale=2.0, solver="osqp"
        )
        assert res.converged
        assert abs(res.weights.sum() - 1.0) < 1e-4
        bounds = spec.per_asset_bounds()
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        assert np.all(res.weights >= lo - 1e-4)
        assert np.all(res.weights <= hi + 1e-4)


class TestOSQPLongShort:
    def test_net_exposure_zero(self):
        """Dollar-neutral OSQP solve."""
        spec = ConstraintSpec(
            n_assets=6,
            long_only=False,
            min_weight=-0.3,
            max_weight=0.3,
            net_exposure=0.0,
        )
        alpha = np.array([1.0, 0.8, 0.5, -0.5, -0.8, -1.0])
        cov = make_diagonal_cov(np.ones(6))
        res = mean_variance(alpha, cov, spec, solver="osqp")
        assert abs(res.weights.sum()) < 1e-4

    def test_long_short_signs(self):
        spec = ConstraintSpec(
            n_assets=4,
            long_only=False,
            min_weight=-0.5,
            max_weight=0.5,
            net_exposure=0.0,
        )
        alpha = np.array([2.0, 1.0, -1.0, -2.0])
        cov = make_diagonal_cov(np.ones(4))
        res = mean_variance(alpha, cov, spec, risk_aversion=0.01, solver="osqp")
        assert res.weights[0] > 0.0
        assert res.weights[3] < 0.0


class TestOSQPCostPenalty:
    def test_cost_reduces_turnover(self):
        """High cost penalty keeps OSQP solution close to w0."""
        n = 5
        spec = ConstraintSpec(n_assets=n, long_only=True, max_weight=1.0)
        rng = np.random.default_rng(99)
        w0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        alpha = rng.normal(0.0, 1.0, n)
        cov = make_diagonal_cov(np.ones(n))
        res_cheap = mean_variance(
            alpha, cov, spec, risk_aversion=1.0, w0=w0, cost_scale=0.0, solver="osqp"
        )
        res_costly = mean_variance(
            alpha, cov, spec, risk_aversion=1.0, w0=w0, cost_scale=50.0, solver="osqp"
        )
        to_cheap = np.abs(res_cheap.weights - w0).sum()
        to_costly = np.abs(res_costly.weights - w0).sum()
        assert to_costly < to_cheap


class TestOSQPSectorConstraints:
    def test_sector_max_respected(self):
        """OSQP honours sector-level exposure caps."""
        n = 6
        sector_map = np.array([0, 0, 0, 1, 1, 1])
        spec = ConstraintSpec(
            n_assets=n,
            long_only=True,
            min_weight=0.0,
            max_weight=1.0,
            net_exposure=1.0,
            sector_map=sector_map,
            sector_min={0: 0.0, 1: 0.0},
            sector_max={0: 0.5, 1: 0.5},
        )
        alpha = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0])
        cov = make_diagonal_cov(np.ones(n))
        res = mean_variance(alpha, cov, spec, risk_aversion=0.01, solver="osqp")
        assert res.weights[:3].sum() <= 0.5 + 1e-4
        assert res.weights[3:].sum() <= 0.5 + 1e-4
