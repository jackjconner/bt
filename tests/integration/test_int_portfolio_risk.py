"""Contract: factor datasets → factor risk model + constrained optimizer.

`factor_loadings` / `factor_covariance` / `specific_risk` build the risk model
Σ = B·F·Bᵀ + D; `position_constraints` / `group_constraints` / `security_master`
build the constraint set. This verifies the optimizer honours those bounds and
the risk model returns a non-negative variance.
"""

from __future__ import annotations

import numpy as np

from etl import to_matrix
from harness.components import returns_from_prices
from portfolio import (
    build_from_long,
    constraints_from_polars,
    ledoit_wolf_cov,
    mean_variance,
    tracking_error,
)


def test_factor_risk_model_and_optimizer(synth) -> None:
    loader, spec = synth.loader, synth.spec
    prices = loader.load("prices")
    R, _ = to_matrix(returns_from_prices(prices), "return")
    cov = ledoit_wolf_cov(R)
    assert cov.shape == (spec.n_assets, spec.n_assets)

    risk_model = build_from_long(
        loader.load("factor_loadings"),
        loader.load("factor_covariance"),
        loader.load("specific_risk"),
        prices["date"].max(),
    )

    cspec = constraints_from_polars(
        loader.load("position_constraints"),
        loader.load("group_constraints"),
        loader.load("security_master"),
        spec.n_assets,
        long_only=False,
        net_exposure=1.0,
    )
    rng = np.random.default_rng(0)
    alpha_vec = rng.normal(0.0, 1.0, spec.n_assets)
    opt = mean_variance(alpha_vec, cov, cspec, risk_aversion=1.0, max_iter=3000)

    assert opt.converged
    assert abs(opt.weights.sum() - 1.0) < 1e-4          # net-exposure budget
    bounds = cspec.per_asset_bounds()
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    assert np.all(opt.weights >= lo - 1e-6)
    assert np.all(opt.weights <= hi + 1e-6)

    assert risk_model.portfolio_variance(opt.weights) >= 0.0


def test_tracking_error_zero_against_self(synth) -> None:
    loader, spec = synth.loader, synth.spec
    R, _ = to_matrix(returns_from_prices(loader.load("prices")), "return")
    cov = ledoit_wolf_cov(R)
    w = np.full(spec.n_assets, 1.0 / spec.n_assets)
    assert tracking_error(w, w, cov) < 1e-9
