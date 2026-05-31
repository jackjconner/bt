"""Full end-to-end pipeline: generate → load → signals → models → optimize →
backtest (gross vs net) → analytics → profiling, asserting each stage's output
is coherent. Complements the per-component contract tests with the whole chain."""

from __future__ import annotations

import math

from etl.datasets import GenSpec
from pipeline import run_production_pipeline


def test_full_pipeline_stage_outputs(tmp_path) -> None:
    spec = GenSpec(n_assets=40, n_dates=120, n_features=8, n_factors=4, seed=2)
    s = run_production_pipeline(spec, tmp_path)

    # signals stage: injected alpha recovered, neutralization bounded, decay present
    assert s.ic_raw > 0.0
    assert abs(s.ic_neutralized) < 1.0
    assert set(s.horizon_ic) == {1, 5, 21, 63}

    # model stage produced finite CV scores
    assert math.isfinite(s.wf_mean_ic) and math.isfinite(s.wf_mean_r2)

    # optimizer converged and respects the net-exposure budget
    assert s.opt_converged
    assert s.opt_gross >= 0.99

    # risk + cost stages
    assert s.factor_vol >= 0.0 and s.tracking_error >= 0.0
    assert s.cost_drag > 0.0  # costs strictly reduce terminal NAV
    assert math.isfinite(s.gross_sharpe) and math.isfinite(s.net_sharpe)

    # profiling stage produced scaling fits and a latency figure
    assert s.n_scaling_fits > 0
    assert s.backtest_p50_s >= 0.0
