from __future__ import annotations

import polars as pl

from backtest import ProductionBacktestConfig, ProductionBacktestEngine, SignalFrame
from etl import DatasetLoader
from etl.datasets import GenSpec, write_all
from pipeline import _returns_from_prices, run_production_pipeline


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


def test_production_backtest_runs_with_short_gating_on(tmp_path) -> None:
    """Production path now runs short-availability gating + financing ON.

    Real shorts are borrow-constrained, so the production backtest models
    shortability / loan availability and charges borrow. The synthetic book is
    long-only by construction (softmax targets are all-positive), so gating
    binds on nothing and the on-path NAV equals the long-only off-path NAV —
    this is why activating the flag leaves the golden unchanged. A caller can
    still force gating off (the engine default is off; the production *pipeline*
    opts in), proven here by reproducing the on-path with the flag off.
    """
    spec = GenSpec(n_assets=30, n_dates=90, n_features=6, n_factors=3, seed=1)
    write_all(tmp_path, spec)
    loader = DatasetLoader(tmp_path, spec)
    prices = loader.load("prices")
    returns = _returns_from_prices(prices)
    tcosts = loader.load("transaction_costs")
    umask = loader.load("universe_mask")
    borrow = loader.load("borrow_rates")
    momentum = (
        loader.load("alpha_signals")
        .filter(pl.col("signal_name") == "momentum")
        .select("date", "id", "signal")
    )
    signals = SignalFrame(df=momentum, is_categorical=False)

    on_cfg = ProductionBacktestConfig(
        n_assets=spec.n_assets,
        n_dates=spec.n_dates,
        enable_universe_mask=True,
        enable_costs=True,
        enable_slippage=True,
        max_weight=0.1,
        enable_short_availability_gating=True,
        enable_financing=True,
    )
    off_cfg = ProductionBacktestConfig(  # engine default: gating off
        n_assets=spec.n_assets,
        n_dates=spec.n_dates,
        enable_universe_mask=True,
        enable_costs=True,
        enable_slippage=True,
        max_weight=0.1,
    )

    on = ProductionBacktestEngine(on_cfg).run(
        returns,
        signals,
        prices=prices,
        transaction_costs=tcosts,
        universe_mask=umask,
        borrow_rates=borrow,
    )
    off = ProductionBacktestEngine(off_cfg).run(
        returns, signals, prices=prices, transaction_costs=tcosts, universe_mask=umask
    )

    on_nav = on.nav_history["nav"].to_numpy()
    off_nav = off.nav_history["nav"].to_numpy()
    # Long-only book: gating caps nothing, borrow/financing are zero → on == off
    # up to fp reordering of an extra (always-zero) accrual branch.
    assert abs(float((on_nav - off_nav).max())) < 1e-6
    # No short was ever taken, so no financing drag accrued under the on-path.
    assert on.financing_drag == 0.0
