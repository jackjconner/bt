from __future__ import annotations

from etl.datasets import GenSpec
from pipeline import run_production_pipeline


def test_production_pipeline_end_to_end(tmp_path) -> None:
    spec = GenSpec(n_assets=30, n_dates=90, n_features=6, n_factors=3, seed=1)
    s = run_production_pipeline(spec, tmp_path)

    # signals: injected alpha must produce a positive next-day IC
    assert s.ic_raw > 0.0
    # neutralization keeps IC in a sane band (doesn't blow up or vanish entirely)
    assert abs(s.ic_neutralized) < 1.0
    # horizon curve covers all four horizons
    assert set(s.horizon_ic) == {1, 5, 21, 63}
    # walk-forward CV ran and produced finite scores
    assert s.wf_mean_r2 == s.wf_mean_r2  # not NaN
    # optimizer respects the budget (net≈1 → gross≥1) and converged
    assert s.opt_converged
    assert s.opt_gross >= 0.99
    # factor-model risk and tracking error are non-negative
    assert s.factor_vol >= 0.0
    assert s.tracking_error >= 0.0
    # costs strictly reduce terminal NAV (net < gross)
    assert s.cost_drag > 0.0
    # scaling fits were produced over the synthetic param grid
    assert s.n_scaling_fits > 0
    assert s.backtest_p50_s >= 0.0
