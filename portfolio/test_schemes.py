"""Tests for weighting schemes and turnover utilities."""
from __future__ import annotations

import numpy as np
import pytest

from portfolio.constraints import ConstraintSpec
from portfolio.schemes import (
    apply_no_trade_band,
    cap_weight,
    equal_weight,
    inverse_vol,
    optimized_weight,
    transaction_cost,
    turnover,
)


@pytest.fixture
def rng():
    return np.random.default_rng(30)


class TestEqualWeight:
    def test_sums_to_one(self):
        w = equal_weight(10)
        assert abs(w.sum() - 1.0) < 1e-12

    def test_all_equal(self):
        w = equal_weight(7)
        np.testing.assert_allclose(w, 1.0 / 7)

    def test_shape(self):
        assert equal_weight(5).shape == (5,)


class TestInverseVol:
    def test_sums_to_one(self, rng):
        vols = rng.uniform(0.01, 0.05, 8)
        w = inverse_vol(vols)
        assert abs(w.sum() - 1.0) < 1e-12

    def test_lower_vol_higher_weight(self):
        vols = np.array([0.1, 0.2])
        w = inverse_vol(vols)
        assert w[0] > w[1]

    def test_handles_zero_vol(self):
        vols = np.array([0.0, 0.01, 0.02])
        w = inverse_vol(vols)
        assert np.all(np.isfinite(w))
        assert abs(w.sum() - 1.0) < 1e-12


class TestCapWeight:
    def test_sums_to_one(self, rng):
        caps = rng.uniform(1e6, 1e10, 10)
        w = cap_weight(caps)
        assert abs(w.sum() - 1.0) < 1e-12

    def test_proportional(self):
        caps = np.array([1.0, 2.0, 3.0])
        w = cap_weight(caps)
        np.testing.assert_allclose(w, caps / caps.sum())

    def test_zero_caps_fallback(self):
        caps = np.zeros(5)
        w = cap_weight(caps)
        np.testing.assert_allclose(w, 0.2)


class TestOptimizedWeight:
    def test_sums_to_one(self):
        n = 5
        spec = ConstraintSpec(n_assets=n, long_only=True, max_weight=0.5)
        alpha = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
        cov = np.diag(np.ones(n))
        w = optimized_weight(alpha, cov, spec, risk_aversion=1.0)
        assert abs(w.sum() - 1.0) < 1e-5

    def test_non_negative_long_only(self):
        n = 4
        spec = ConstraintSpec(n_assets=n, long_only=True, max_weight=1.0)
        alpha = np.ones(n)
        cov = np.diag(np.ones(n))
        w = optimized_weight(alpha, cov, spec)
        assert np.all(w >= -1e-8)


class TestNoTradeBand:
    def test_no_change_within_band(self):
        w_new = np.array([0.21, 0.19, 0.20, 0.20, 0.20])
        w_prev = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
        band = 0.02
        result = apply_no_trade_band(w_new, w_prev, band)
        # all drifts ≤ 0.01 < 0.02, so no trades
        np.testing.assert_allclose(result, w_prev)

    def test_large_drift_triggers_trade(self):
        w_new = np.array([0.40, 0.19, 0.19, 0.19, 0.19])
        w_prev = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
        band = 0.05
        result = apply_no_trade_band(w_new, w_prev, band)
        # asset 0 drifted 0.20 > 0.05 → takes new weight
        assert result[0] == pytest.approx(0.40)
        # assets 1-4 drifted 0.01 < 0.05 → stays at prev
        np.testing.assert_allclose(result[1:], w_prev[1:])


class TestTurnover:
    def test_zero_when_unchanged(self):
        w = np.array([0.25, 0.25, 0.25, 0.25])
        assert turnover(w, w) == pytest.approx(0.0)

    def test_full_rebalance(self):
        """Switching from all-asset-0 to all-asset-1 is 100% one-way turnover."""
        w_prev = np.array([1.0, 0.0])
        w_new = np.array([0.0, 1.0])
        assert turnover(w_new, w_prev) == pytest.approx(1.0)

    def test_partial(self):
        w_prev = np.array([0.5, 0.5])
        w_new = np.array([0.6, 0.4])
        assert turnover(w_new, w_prev) == pytest.approx(0.1)


class TestTransactionCost:
    def test_zero_when_unchanged(self):
        w = np.array([0.25, 0.25, 0.25, 0.25])
        assert transaction_cost(w, w) == pytest.approx(0.0)

    def test_scales_with_cost_per_unit(self):
        w_prev = np.array([0.5, 0.5])
        w_new = np.array([0.6, 0.4])
        cost_low = transaction_cost(w_new, w_prev, cost_per_unit=0.001)
        cost_high = transaction_cost(w_new, w_prev, cost_per_unit=0.01)
        assert cost_high == pytest.approx(10 * cost_low)
