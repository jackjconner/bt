"""Tests for tracking error and information ratio."""

from __future__ import annotations

import numpy as np
import pytest

from portfolio.tracking import information_ratio, tracking_error


@pytest.fixture
def rng():
    return np.random.default_rng(10)


@pytest.fixture
def small_cov(rng):
    n = 8
    A = rng.normal(0, 1, (n, n))
    return A @ A.T / n + np.eye(n)


@pytest.fixture
def equal_weights():
    n = 8
    return np.full(n, 1.0 / n)


class TestTrackingError:
    def test_zero_when_w_equals_b(self, small_cov, equal_weights):
        """Tracking error is zero when portfolio == benchmark."""
        te = tracking_error(equal_weights, equal_weights, small_cov)
        assert abs(te) < 1e-12

    def test_positive_when_w_differs(self, small_cov, rng):
        n = small_cov.shape[0]
        w = rng.dirichlet(np.ones(n))
        b = rng.dirichlet(np.ones(n))
        te = tracking_error(w, b, small_cov)
        assert te > 0.0

    def test_non_negative(self, small_cov, rng):
        n = small_cov.shape[0]
        for _ in range(10):
            w = rng.dirichlet(np.ones(n))
            b = rng.dirichlet(np.ones(n))
            assert tracking_error(w, b, small_cov) >= 0.0

    def test_scales_with_active_weight(self, rng):
        """Larger active tilt → larger tracking error."""
        n = 5
        A = rng.normal(0, 1, (n, n))
        cov = A @ A.T / n + np.eye(n)
        b = np.full(n, 0.2)
        w_small = b.copy()
        w_small[0] += 0.05
        w_small[1] -= 0.05
        w_large = b.copy()
        w_large[0] += 0.20
        w_large[1] -= 0.20
        te_small = tracking_error(w_small, b, cov)
        te_large = tracking_error(w_large, b, cov)
        assert te_large > te_small


class TestInformationRatio:
    def test_positive_with_positive_mean(self):
        rng = np.random.default_rng(999)
        active = rng.normal(0.05, 1.0, 500)
        ir = information_ratio(active)
        assert ir > 0.0

    def test_nan_on_too_few_obs(self):
        assert np.isnan(information_ratio(np.array([0.1])))

    def test_nan_on_zero_vol(self):
        assert np.isnan(information_ratio(np.ones(100)))

    def test_sign(self):
        rng = np.random.default_rng(998)
        active_neg = rng.normal(-0.05, 1.0, 500)
        assert information_ratio(active_neg) < 0.0
