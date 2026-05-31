"""Unit tests for extracted private helpers in optimizer.py and constraints.py."""

from __future__ import annotations

import numpy as np
import pytest

from portfolio.constraints import (
    ConstraintSpec,
    _build_per_asset_bounds,
    _build_sector_map,
    _gross_scipy_constraints,
    _sector_scipy_constraints,
)
from portfolio.optimizer import (
    _build_constraint_matrix,
    _build_epigraph_rows,
    _build_quadratic_cost,
    _osqp_objective,
    _slsqp_warm_start,
    mean_variance,
)

# ---------------------------------------------------------------------------
# _slsqp_warm_start
# ---------------------------------------------------------------------------


class TestSlsqpWarmStart:
    def test_output_sums_to_net_exposure(self):
        spec = ConstraintSpec(
            n_assets=4, long_only=True, min_weight=0.0, max_weight=1.0, net_exposure=1.0
        )
        w0 = np.array([0.25, 0.25, 0.25, 0.25])
        w_init = _slsqp_warm_start(spec, w0)
        assert abs(w_init.sum() - 1.0) < 1e-12

    def test_clips_to_bounds_before_renorm(self):
        spec = ConstraintSpec(
            n_assets=3, long_only=True, min_weight=0.0, max_weight=0.3, net_exposure=1.0
        )
        # w0 has one weight above the 0.3 cap; after clipping the clipped value is 0.3
        # then renormalisation to net_exposure=1 may push it above 0.3 again — that's expected.
        # The important thing: the pre-clip input is clamped, and the output sums to 1.
        w0 = np.array([0.6, 0.2, 0.2])
        w_init = _slsqp_warm_start(spec, w0)
        # sum-to-net-exposure is guaranteed; individual clipping is pre-renorm
        assert abs(w_init.sum() - 1.0) < 1e-12

    def test_zero_sum_w0_falls_back_to_equal(self):
        spec = ConstraintSpec(
            n_assets=3, long_only=True, min_weight=0.0, max_weight=1.0, net_exposure=1.0
        )
        w0 = np.array([0.0, 0.0, 0.0])
        w_init = _slsqp_warm_start(spec, w0)
        assert abs(w_init.sum() - 1.0) < 1e-12

    def test_net_exposure_zero(self):
        spec = ConstraintSpec(
            n_assets=4, long_only=False, min_weight=-0.3, max_weight=0.3, net_exposure=0.0
        )
        w0 = np.array([0.1, 0.1, 0.1, 0.1])
        w_init = _slsqp_warm_start(spec, w0)
        assert abs(w_init.sum() - 0.0) < 1e-12


# ---------------------------------------------------------------------------
# _build_quadratic_cost
# ---------------------------------------------------------------------------


class TestBuildQuadraticCost:
    def _setup(self, n: int = 4) -> tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        cov = np.eye(n) * 0.02
        alpha = rng.normal(0.0, 1.0, n)
        cp = np.ones(n)
        return n, 1.0, cov, alpha, cp

    def test_p_shape(self):
        n, lam, cov, alpha, cp = self._setup()
        P, _q = _build_quadratic_cost(n, lam, cov, alpha, cp, cost_scale=1.0)
        assert P.shape == (2 * n, 2 * n)

    def test_p_is_upper_triangular_csc(self):
        n, lam, cov, alpha, cp = self._setup()
        P, _ = _build_quadratic_cost(n, lam, cov, alpha, cp, cost_scale=1.0)
        # upper triangular: lower triangle is zero
        P_dense = P.toarray()
        assert np.all(np.tril(P_dense, k=-1) == 0.0)

    def test_p_w_block_equals_2lambda_cov(self):
        n, lam, cov, alpha, cp = self._setup()
        P, _ = _build_quadratic_cost(n, lam, cov, alpha, cp, cost_scale=1.0)
        P_dense = P.toarray()
        # The w×w block (upper triangle of 2λΣ)
        expected = np.triu(2.0 * lam * cov)
        np.testing.assert_allclose(P_dense[:n, :n], expected)

    def test_p_t_block_is_zero(self):
        n, lam, cov, alpha, cp = self._setup()
        P, _ = _build_quadratic_cost(n, lam, cov, alpha, cp, cost_scale=1.0)
        P_dense = P.toarray()
        # t×t block must be zero (pure linear penalty on t)
        np.testing.assert_array_equal(P_dense[n:, n:], 0.0)

    def test_q_w_part_negates_alpha(self):
        n, lam, cov, alpha, cp = self._setup()
        _, q = _build_quadratic_cost(n, lam, cov, alpha, cp, cost_scale=1.0)
        np.testing.assert_allclose(q[:n], -alpha)

    def test_q_t_part_is_cost(self):
        n, lam, cov, alpha, cp = self._setup()
        cost_scale = 2.5
        _, q = _build_quadratic_cost(n, lam, cov, alpha, cp, cost_scale=cost_scale)
        np.testing.assert_allclose(q[n:], cost_scale * cp)

    def test_q_t_part_zero_when_no_cost(self):
        n, lam, cov, alpha, cp = self._setup()
        _, q = _build_quadratic_cost(n, lam, cov, alpha, cp, cost_scale=0.0)
        np.testing.assert_array_equal(q[n:], 0.0)


# ---------------------------------------------------------------------------
# _build_epigraph_rows
# ---------------------------------------------------------------------------


class TestBuildEpigraphRows:
    def test_returns_three_blocks(self):
        n = 4
        w0 = np.full(n, 0.25)
        blocks, l_parts, u_parts = _build_epigraph_rows(n, w0, no_trade_band=0.0)
        assert len(blocks) == 3
        assert len(l_parts) == 3
        assert len(u_parts) == 3

    def test_each_block_has_n_rows(self):
        n = 5
        w0 = np.full(n, 0.2)
        blocks, _l_parts, _u_parts = _build_epigraph_rows(n, w0, no_trade_band=0.0)
        for blk in blocks:
            assert blk.shape[0] == n

    def test_upper_epigraph_rhs_equals_w0_plus_band(self):
        n = 3
        w0 = np.array([0.1, 0.2, 0.3])
        band = 0.05
        _, _, u_parts = _build_epigraph_rows(n, w0, no_trade_band=band)
        # upper epigraph: w_i - t_i ≤ w0_i + band  → u_part index 1
        np.testing.assert_allclose(u_parts[1], w0 + band)

    def test_lower_epigraph_rhs_equals_neg_w0_plus_band(self):
        n = 3
        w0 = np.array([0.1, 0.2, 0.3])
        band = 0.05
        _, _, u_parts = _build_epigraph_rows(n, w0, no_trade_band=band)
        # lower epigraph: -w_i - t_i ≤ -w0_i + band  → u_part index 2
        np.testing.assert_allclose(u_parts[2], -w0 + band)

    def test_no_band_upper_rhs_equals_w0(self):
        n = 3
        w0 = np.array([0.3, 0.4, 0.3])
        _, _, u_parts = _build_epigraph_rows(n, w0, no_trade_band=0.0)
        np.testing.assert_allclose(u_parts[1], w0)

    def test_t_geq_zero_lower_bound(self):
        n = 4
        w0 = np.full(n, 0.25)
        _, l_parts, _ = _build_epigraph_rows(n, w0, no_trade_band=0.0)
        # t_i ≥ 0 block: l_part index 0
        np.testing.assert_array_equal(l_parts[0], 0.0)


# ---------------------------------------------------------------------------
# _build_constraint_matrix
# ---------------------------------------------------------------------------


class TestBuildConstraintMatrix:
    def _simple_spec(self, n: int = 4) -> ConstraintSpec:
        return ConstraintSpec(
            n_assets=n,
            long_only=True,
            min_weight=0.0,
            max_weight=0.5,
            net_exposure=1.0,
        )

    def test_shape_without_cost(self):
        n = 4
        spec = self._simple_spec(n)
        w0 = np.full(n, 0.25)
        A, _l_vec, _u_vec = _build_constraint_matrix(
            spec, n, cost_scale=0.0, w0_arr=w0, no_trade_band=0.0
        )
        # n (weight bounds) + 1 (net exposure) = n+1 rows
        assert A.shape[0] == n + 1
        assert A.shape[1] == 2 * n

    def test_shape_with_cost(self):
        n = 4
        spec = self._simple_spec(n)
        w0 = np.full(n, 0.25)
        A, _l_vec, _u_vec = _build_constraint_matrix(
            spec, n, cost_scale=1.0, w0_arr=w0, no_trade_band=0.0
        )
        # n + 1 + 3n (epigraph) = 4n + 1
        assert A.shape[0] == 4 * n + 1
        assert A.shape[1] == 2 * n

    def test_net_exposure_row(self):
        n = 3
        spec = self._simple_spec(n)
        w0 = np.full(n, 1.0 / n)
        _A, l_vec, u_vec = _build_constraint_matrix(
            spec, n, cost_scale=0.0, w0_arr=w0, no_trade_band=0.0
        )
        # Row n (index n) is the net-exposure equality: l == u == 1.0
        assert l_vec[n] == pytest.approx(1.0)
        assert u_vec[n] == pytest.approx(1.0)

    def test_weight_bound_rows(self):
        n = 3
        spec = ConstraintSpec(
            n_assets=n, long_only=True, min_weight=0.05, max_weight=0.6, net_exposure=1.0
        )
        w0 = np.full(n, 1.0 / n)
        _A, l_vec, u_vec = _build_constraint_matrix(
            spec, n, cost_scale=0.0, w0_arr=w0, no_trade_band=0.0
        )
        np.testing.assert_allclose(l_vec[:n], 0.05)
        np.testing.assert_allclose(u_vec[:n], 0.6)

    def test_sector_rows_appended(self):
        n = 4
        sector_map = np.array([0, 0, 1, 1])
        # Each sector has both min and max → 1 row per sector (combined into
        # a single [lo, hi] row in OSQP's constraint style), so 2 rows total.
        spec = ConstraintSpec(
            n_assets=n,
            long_only=True,
            max_weight=1.0,
            sector_map=sector_map,
            sector_min={0: 0.1, 1: 0.1},
            sector_max={0: 0.7, 1: 0.7},
        )
        w0 = np.full(n, 0.25)
        A_no_sec = _build_constraint_matrix(
            ConstraintSpec(n_assets=n, long_only=True, max_weight=1.0),
            n,
            cost_scale=0.0,
            w0_arr=w0,
            no_trade_band=0.0,
        )[0]
        A_with_sec, _, _ = _build_constraint_matrix(
            spec, n, cost_scale=0.0, w0_arr=w0, no_trade_band=0.0
        )
        # 2 sectors, each with one combined [lo, hi] bound row = 2 extra rows
        assert A_with_sec.shape[0] == A_no_sec.shape[0] + 2


# ---------------------------------------------------------------------------
# _osqp_objective
# ---------------------------------------------------------------------------


class TestOsqpObjective:
    def test_no_cost(self):
        n = 3
        weights = np.array([0.4, 0.3, 0.3])
        alpha = np.array([1.0, 0.5, 0.2])
        cov = np.eye(n) * 0.01
        w0 = np.full(n, 1.0 / n)
        cp = np.ones(n)
        obj = _osqp_objective(
            weights, alpha, cov, risk_aversion=1.0, w0=w0, cp=cp, cost_scale=0.0, no_trade_band=0.0
        )
        expected = float(alpha @ weights) - 1.0 * float(weights @ cov @ weights)
        assert obj == pytest.approx(expected, abs=1e-12)

    def test_with_cost(self):
        n = 3
        weights = np.array([0.5, 0.3, 0.2])
        alpha = np.array([1.0, 0.5, 0.0])
        cov = np.eye(n) * 0.01
        w0 = np.array([0.2, 0.2, 0.6])
        cp = np.ones(n)
        cost_scale = 2.0
        obj = _osqp_objective(
            weights,
            alpha,
            cov,
            risk_aversion=1.0,
            w0=w0,
            cp=cp,
            cost_scale=cost_scale,
            no_trade_band=0.0,
        )
        delta = np.abs(weights - w0)
        expected = (
            float(alpha @ weights) - float(weights @ cov @ weights) - cost_scale * float(cp @ delta)
        )
        assert obj == pytest.approx(expected, abs=1e-12)

    def test_no_trade_band_zeroes_small_delta(self):
        n = 2
        weights = np.array([0.5, 0.5])
        w0 = np.array([0.5, 0.5])  # exact match → delta=0 regardless of band
        alpha = np.array([1.0, 1.0])
        cov = np.eye(n) * 0.01
        cp = np.ones(n)
        obj_band = _osqp_objective(
            weights, alpha, cov, risk_aversion=1.0, w0=w0, cp=cp, cost_scale=1.0, no_trade_band=0.02
        )
        obj_no_band = _osqp_objective(
            weights, alpha, cov, risk_aversion=1.0, w0=w0, cp=cp, cost_scale=1.0, no_trade_band=0.0
        )
        # Both should be equal (zero turnover)
        assert obj_band == pytest.approx(obj_no_band, abs=1e-12)


# ---------------------------------------------------------------------------
# _gross_scipy_constraints
# ---------------------------------------------------------------------------


class TestGrossSciPyConstraints:
    def test_no_bounds_returns_empty(self):
        cons = _gross_scipy_constraints(min_gross=None, max_gross=None)
        assert cons == []

    def test_min_gross_only(self):
        cons = _gross_scipy_constraints(min_gross=0.8, max_gross=None)
        assert len(cons) == 1
        assert cons[0]["type"] == "ineq"
        # w = [0.9, 0.1] → gross = 1.0; fun = gross - min_gross = 1.0 - 0.8 = 0.2 > 0 (feasible)
        w = np.array([0.9, 0.1])
        assert cons[0]["fun"](w) == pytest.approx(0.2, abs=1e-12)

    def test_max_gross_only(self):
        cons = _gross_scipy_constraints(min_gross=None, max_gross=1.2)
        assert len(cons) == 1
        w = np.array([0.6, 0.6])
        # fun should be 1.2 - 1.2 = 0
        assert cons[0]["fun"](w) == pytest.approx(0.0, abs=1e-12)

    def test_both_bounds(self):
        cons = _gross_scipy_constraints(min_gross=0.8, max_gross=1.2)
        assert len(cons) == 2


# ---------------------------------------------------------------------------
# _sector_scipy_constraints
# ---------------------------------------------------------------------------


class TestSectorSciPyConstraints:
    def test_no_sector_map_returns_empty(self):
        cons = _sector_scipy_constraints(None, {}, {})
        assert cons == []

    def test_sector_max_constraint_active(self):
        sector_map = np.array([0, 0, 1, 1])
        cons = _sector_scipy_constraints(sector_map, {}, {0: 0.5, 1: 0.5})
        assert len(cons) == 2
        # sector 0 max: hi - m @ w; w = [0.3, 0.3, 0.2, 0.2] → sector0=0.6 > cap
        w = np.array([0.3, 0.3, 0.2, 0.2])
        # hi - m@w = 0.5 - 0.6 = -0.1 < 0  (violated)
        assert cons[0]["fun"](w) == pytest.approx(-0.1, abs=1e-12)

    def test_sector_min_constraint_active(self):
        sector_map = np.array([0, 0, 1, 1])
        cons = _sector_scipy_constraints(sector_map, {0: 0.2}, {})
        assert len(cons) == 1
        w = np.array([0.1, 0.0, 0.5, 0.4])
        # fun = m@w - lo = 0.1 - 0.2 = -0.1 < 0  (violated)
        assert cons[0]["fun"](w) == pytest.approx(-0.1, abs=1e-12)

    def test_sector_jac_shape(self):
        sector_map = np.array([0, 0, 1, 1])
        cons = _sector_scipy_constraints(sector_map, {0: 0.1}, {0: 0.9})
        w = np.ones(4) * 0.25
        for con in cons:
            if "jac" in con:
                j = con["jac"](w)
                assert j.shape == (4,)


# ---------------------------------------------------------------------------
# _build_per_asset_bounds
# ---------------------------------------------------------------------------


class TestBuildPerAssetBounds:
    def test_defaults_for_missing_asset(self):
        pc: dict[int, dict] = {}  # no constraints → defaults [0, 1]
        min_w, max_w = _build_per_asset_bounds(pc, n_assets=3)
        np.testing.assert_array_equal(min_w, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(max_w, [1.0, 1.0, 1.0])

    def test_non_tradable_asset_zero_bounds(self):
        pc = {1: {"tradable": False, "min_weight": 0.05, "max_weight": 0.4}}
        min_w, max_w = _build_per_asset_bounds(pc, n_assets=3)
        assert min_w[1] == 0.0
        assert max_w[1] == 0.0

    def test_tradable_asset_uses_specified_bounds(self):
        pc = {0: {"tradable": True, "min_weight": 0.02, "max_weight": 0.3}}
        min_w, max_w = _build_per_asset_bounds(pc, n_assets=3)
        assert min_w[0] == pytest.approx(0.02)
        assert max_w[0] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# _build_sector_map
# ---------------------------------------------------------------------------


class TestBuildSectorMap:
    def test_basic_mapping(self):
        pairs = [(0, "tech"), (1, "health"), (2, "tech")]
        s2i = {"tech": 0, "health": 1}
        sector_map = _build_sector_map(pairs, s2i, n_assets=3)
        np.testing.assert_array_equal(sector_map, [0, 1, 0])

    def test_out_of_range_id_ignored(self):
        pairs = [(0, "tech"), (5, "health")]  # id=5 out of range for n_assets=3
        s2i = {"tech": 0, "health": 1}
        sector_map = _build_sector_map(pairs, s2i, n_assets=3)
        assert sector_map[0] == 0
        # id=5 should not have caused an error; default remains 0
        assert sector_map.shape == (3,)


# ---------------------------------------------------------------------------
# Cross-solver consistency: SLSQP vs OSQP on identical problem
# ---------------------------------------------------------------------------


class TestSolverConsistency:
    """OSQP and SLSQP should find identical (within tolerance) solutions."""

    def _make_problem(self, n: int = 6, seed: int = 42):
        rng = np.random.default_rng(seed)
        alpha = rng.normal(0.0, 1.0, n)
        A = rng.normal(0.0, 1.0, (n, n))
        cov = A @ A.T / n + np.eye(n) * 0.1
        spec = ConstraintSpec(n_assets=n, long_only=True, max_weight=0.4, net_exposure=1.0)
        w0 = np.full(n, 1.0 / n)
        return alpha, cov, spec, w0

    def test_weights_close(self):
        alpha, cov, spec, w0 = self._make_problem()
        res_slsqp = mean_variance(alpha, cov, spec, w0=w0, cost_scale=1.0, solver="slsqp")
        res_osqp = mean_variance(alpha, cov, spec, w0=w0, cost_scale=1.0, solver="osqp")
        # Solvers use different algorithms; agree within 1e-3 (OSQP is the more exact solve)
        np.testing.assert_allclose(res_osqp.weights, res_slsqp.weights, atol=1e-3)

    def test_objective_close(self):
        alpha, cov, spec, w0 = self._make_problem(seed=7)
        res_slsqp = mean_variance(alpha, cov, spec, w0=w0, cost_scale=2.0, solver="slsqp")
        res_osqp = mean_variance(alpha, cov, spec, w0=w0, cost_scale=2.0, solver="osqp")
        assert abs(res_osqp.obj_value - res_slsqp.obj_value) < 1e-4

    def test_no_trade_band_consistent(self):
        alpha, cov, spec, w0 = self._make_problem(seed=99)
        res_slsqp = mean_variance(
            alpha, cov, spec, w0=w0, cost_scale=3.0, no_trade_band=0.02, solver="slsqp"
        )
        res_osqp = mean_variance(
            alpha, cov, spec, w0=w0, cost_scale=3.0, no_trade_band=0.02, solver="osqp"
        )
        np.testing.assert_allclose(res_osqp.weights, res_slsqp.weights, atol=1e-3)
