"""Tests for covariance estimators."""
from __future__ import annotations

import numpy as np
import pytest

from portfolio.covariance import ewma_cov, ledoit_wolf_cov, sample_cov


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def returns_wide(rng):
    # 100 periods, 20 assets — overdetermined, sample cov is full-rank
    return rng.normal(0.0, 1.0, (100, 20))


@pytest.fixture
def returns_thin(rng):
    # 15 periods, 20 assets — underdetermined, rank-deficient sample cov
    return rng.normal(0.0, 1.0, (15, 20))


class TestSampleCov:
    def test_shape(self, returns_wide):
        c = sample_cov(returns_wide)
        n = returns_wide.shape[1]
        assert c.shape == (n, n)

    def test_symmetric(self, returns_wide):
        c = sample_cov(returns_wide)
        np.testing.assert_allclose(c, c.T, atol=1e-12)

    def test_positive_semidefinite_wide(self, returns_wide):
        c = sample_cov(returns_wide)
        eigvals = np.linalg.eigvalsh(c)
        assert eigvals.min() >= -1e-10


class TestEwmaCov:
    def test_shape(self, returns_wide):
        c = ewma_cov(returns_wide, halflife=30.0)
        n = returns_wide.shape[1]
        assert c.shape == (n, n)

    def test_symmetric(self, returns_wide):
        c = ewma_cov(returns_wide, halflife=30.0)
        np.testing.assert_allclose(c, c.T, atol=1e-12)

    def test_positive_semidefinite(self, returns_wide):
        c = ewma_cov(returns_wide, halflife=30.0)
        eigvals = np.linalg.eigvalsh(c)
        assert eigvals.min() >= -1e-10

    def test_short_halflife_weights_recent_more(self, rng):
        # Inject a vol spike in the last 5 periods.
        R = rng.normal(0.0, 1.0, (60, 5))
        R[-5:] *= 5.0  # big spike at the end
        c_short = ewma_cov(R, halflife=5.0)
        c_long = ewma_cov(R, halflife=100.0)
        # short halflife → recent spike dominates → higher diagonal entries
        assert c_short[0, 0] > c_long[0, 0]


class TestLedoitWolfCov:
    def test_shape(self, returns_thin):
        c = ledoit_wolf_cov(returns_thin)
        n = returns_thin.shape[1]
        assert c.shape == (n, n)

    def test_positive_definite(self, returns_thin):
        # Ledoit-Wolf must be PD even when T < n_assets
        c = ledoit_wolf_cov(returns_thin)
        eigvals = np.linalg.eigvalsh(c)
        assert eigvals.min() > 0.0

    def test_shrinks_toward_scaled_identity(self, returns_thin):
        """Off-diagonal entries should be smaller than in sample cov (in mean abs)."""
        n = returns_thin.shape[1]
        c_lw = ledoit_wolf_cov(returns_thin)
        c_s = sample_cov(returns_thin)
        # compare mean absolute off-diagonal magnitude
        mask = ~np.eye(n, dtype=bool)
        assert np.abs(c_lw[mask]).mean() <= np.abs(c_s[mask]).mean()
