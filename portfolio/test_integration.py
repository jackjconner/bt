"""Integration tests: wire real synthetic datasets through the portfolio stack.

These tests use `etl.datasets.generate` to produce small but realistic
data and exercise the full call chain: long → numpy → optimizer / risk model.
"""

from __future__ import annotations

import numpy as np
import pytest

from etl.datasets import GenSpec, generate
from portfolio.optimizer import mean_variance
from portfolio.risk_model import build_from_long
from portfolio.schemes import cap_weight, equal_weight, inverse_vol, turnover
from portfolio.tracking import tracking_error

SPEC = GenSpec(n_assets=10, n_dates=30, n_factors=3, seed=42)


@pytest.fixture(scope="module")
def datasets():
    names = [
        "factor_loadings",
        "factor_covariance",
        "specific_risk",
        "benchmark_weights",
        "security_master",
        "position_constraints",
        "group_constraints",
        "shares_outstanding",
    ]
    return {n: generate(n, SPEC) for n in names}


def test_factor_risk_model_from_synthetic(datasets):
    """build_from_long produces a positive-definite covariance matrix."""
    fl = datasets["factor_loadings"]
    fc = datasets["factor_covariance"]
    sr = datasets["specific_risk"]
    date_ = fl["date"].unique().sort()[0]

    model = build_from_long(fl, fc, sr, date_)
    eigvals = np.linalg.eigvalsh(model.cov)
    assert eigvals.min() > 0.0
    assert model.cov.shape[0] == SPEC.n_assets


def test_optimizer_budget_with_factor_cov(datasets):
    """Optimizer respects the budget constraint when using a real factor cov."""
    fl = datasets["factor_loadings"]
    fc = datasets["factor_covariance"]
    sr = datasets["specific_risk"]
    date_ = fl["date"].unique().sort()[0]

    model = build_from_long(fl, fc, sr, date_)
    from portfolio.constraints import ConstraintSpec

    spec = ConstraintSpec(
        n_assets=SPEC.n_assets,
        long_only=True,
        min_weight=0.0,
        max_weight=1.0,
        net_exposure=1.0,
    )

    rng = np.random.default_rng(0)
    alpha = rng.normal(0.0, 1.0, SPEC.n_assets)
    res = mean_variance(alpha, model.cov, spec, risk_aversion=1.0)

    assert abs(res.weights.sum() - 1.0) < 1e-5, f"budget: {res.weights.sum()}"
    assert np.all(res.weights >= -1e-7)


def test_tracking_error_benchmark_match(datasets):
    """Tracking error against benchmark weights is zero when w == b."""
    bw = datasets["benchmark_weights"]
    fl = datasets["factor_loadings"]
    fc = datasets["factor_covariance"]
    sr = datasets["specific_risk"]
    date_ = fl["date"].unique().sort()[0]

    model = build_from_long(fl, fc, sr, date_)

    # Extract benchmark weights for this date
    bw_date = bw.filter(bw["date"] == date_).sort("id")
    b = bw_date["benchmark_weight"].to_numpy()
    b = b / b.sum()  # normalize

    te = tracking_error(b, b, model.cov)
    assert abs(te) < 1e-12


def test_cap_weight_from_market_cap(datasets):
    """Cap-weight scheme sums to 1 and gives higher weight to larger caps."""
    so = datasets["shares_outstanding"]
    date_ = so["date"].unique().sort()[0]
    mc = so.filter(so["date"] == date_).sort("id")["market_cap"].to_numpy()
    w = cap_weight(mc)
    assert abs(w.sum() - 1.0) < 1e-12
    max_idx = mc.argmax()
    assert w[max_idx] == w.max()


def test_weighting_scheme_turnover_comparison():
    """Equal-weight turnover vs inverse-vol: equal should change less
    on a flat vol day (when both start equal-weight)."""
    n = 10
    w_prev = equal_weight(n)
    vols = np.ones(n)  # flat vols → inverse-vol == equal-weight
    w_iv = inverse_vol(vols)
    w_eq = equal_weight(n)
    assert turnover(w_eq, w_prev) == pytest.approx(0.0)
    assert turnover(w_iv, w_prev) == pytest.approx(0.0)
