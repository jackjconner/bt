"""Tests for parametric VaR and CVaR."""

from __future__ import annotations

import numpy as np
import pytest

from portfolio.risk_metrics import parametric_cvar, parametric_var, var_cvar_table


@pytest.fixture
def rng():
    return np.random.default_rng(20)


@pytest.fixture
def equal_weights():
    n = 5
    return np.full(n, 1.0 / n)


@pytest.fixture
def diag_cov():
    vols = np.array([0.01, 0.02, 0.015, 0.025, 0.012])
    return np.diag(vols**2)


class TestParametricVar:
    def test_positive(self, equal_weights, diag_cov):
        v = parametric_var(equal_weights, diag_cov, confidence=0.95)
        assert v >= 0.0

    def test_higher_confidence_higher_var(self, equal_weights, diag_cov):
        v95 = parametric_var(equal_weights, diag_cov, confidence=0.95)
        v99 = parametric_var(equal_weights, diag_cov, confidence=0.99)
        assert v99 > v95

    def test_longer_horizon_higher_var(self, equal_weights, diag_cov):
        v1 = parametric_var(equal_weights, diag_cov, horizon=1)
        v5 = parametric_var(equal_weights, diag_cov, horizon=5)
        assert v5 > v1

    def test_zero_cov_zero_var(self):
        w = np.array([1.0])
        cov = np.array([[0.0]])
        v = parametric_var(w, cov, confidence=0.95)
        assert abs(v) < 1e-12


class TestParametricCVaR:
    def test_cvar_geq_var(self, equal_weights, diag_cov):
        v = parametric_var(equal_weights, diag_cov, confidence=0.95)
        cv = parametric_cvar(equal_weights, diag_cov, confidence=0.95)
        assert cv >= v - 1e-10

    def test_positive(self, equal_weights, diag_cov):
        cv = parametric_cvar(equal_weights, diag_cov, confidence=0.95)
        assert cv >= 0.0

    def test_higher_confidence_higher_cvar(self, equal_weights, diag_cov):
        cv95 = parametric_cvar(equal_weights, diag_cov, confidence=0.95)
        cv99 = parametric_cvar(equal_weights, diag_cov, confidence=0.99)
        assert cv99 > cv95


class TestVarCVarTable:
    def test_returns_correct_grid(self, equal_weights, diag_cov):
        rows = var_cvar_table(
            equal_weights,
            diag_cov,
            confidences=(0.95, 0.99),
            horizons=(1, 5),
        )
        assert len(rows) == 4
        confs = {r["confidence"] for r in rows}
        horizons = {r["horizon"] for r in rows}
        assert confs == {0.95, 0.99}
        assert horizons == {1, 5}

    def test_all_cvar_geq_var(self, equal_weights, diag_cov):
        rows = var_cvar_table(equal_weights, diag_cov)
        for r in rows:
            assert r["cvar"] >= r["var"] - 1e-10
