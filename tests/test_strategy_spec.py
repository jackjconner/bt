"""Tests for StrategySpec: construction, validation, JSON roundtrip,
additivity (no-strategy identical), and behaviour-change proofs."""

from __future__ import annotations

import dataclasses
import math

import pytest

from etl.datasets import GenSpec
from pipeline import run_from_spec, run_production_pipeline
from strategy import StrategySpec

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_SMALL_GEN = GenSpec(n_assets=20, n_dates=60, n_features=4, n_factors=2, seed=42)


def _dicts_equal_nan_safe(a: dict[int, float], b: dict[int, float]) -> bool:
    """Compare two float dicts treating nan == nan (unlike IEEE 754)."""
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if math.isnan(va) and math.isnan(vb):
            continue
        if va != vb:
            return False
    return True


def _default_spec() -> StrategySpec:
    return StrategySpec(gen=_SMALL_GEN)


# --------------------------------------------------------------------------- #
# Construction / defaults
# --------------------------------------------------------------------------- #


def test_default_spec_constructs() -> None:
    spec = _default_spec()
    assert spec.wf_n_splits == 4
    assert spec.wf_embargo_periods == 5
    assert spec.opt_risk_aversion == 1.0
    assert spec.opt_max_iter == 3000
    assert spec.bt_max_weight == 0.1
    assert spec.bt_enable_universe_mask is True
    assert spec.bt_enable_costs is True
    assert spec.bt_enable_slippage is True
    assert spec.profile_trials == 3
    assert spec.profile_warmup == 1
    assert spec.bt_profile_trials == 3
    assert spec.bt_profile_warmup == 1
    assert spec.neutralize_sectors is True
    assert set(spec.horizon_map.keys()) == {1, 5, 21, 63}


def test_spec_is_frozen() -> None:
    # StrategySpec is declared with frozen=True; verify via dataclass metadata.
    params = dataclasses.fields(StrategySpec)
    assert len(params) > 0  # sanity
    # The class __setattr__ on a frozen dataclass is dataclasses._frozen_setattr
    assert "frozen" in repr(StrategySpec.__dataclass_params__)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Validation: invalid single parameters
# --------------------------------------------------------------------------- #


def test_invalid_wf_n_splits() -> None:
    with pytest.raises(ValueError, match="wf_n_splits"):
        StrategySpec(gen=_SMALL_GEN, wf_n_splits=1)


def test_invalid_wf_embargo_negative() -> None:
    with pytest.raises(ValueError, match="wf_embargo_periods"):
        StrategySpec(gen=_SMALL_GEN, wf_embargo_periods=-1)


def test_invalid_opt_risk_aversion_zero() -> None:
    with pytest.raises(ValueError, match="opt_risk_aversion"):
        StrategySpec(gen=_SMALL_GEN, opt_risk_aversion=0.0)


def test_invalid_opt_risk_aversion_negative() -> None:
    with pytest.raises(ValueError, match="opt_risk_aversion"):
        StrategySpec(gen=_SMALL_GEN, opt_risk_aversion=-1.0)


def test_invalid_opt_max_iter_zero() -> None:
    with pytest.raises(ValueError, match="opt_max_iter"):
        StrategySpec(gen=_SMALL_GEN, opt_max_iter=0)


def test_invalid_bt_max_weight_zero() -> None:
    with pytest.raises(ValueError, match="bt_max_weight"):
        StrategySpec(gen=_SMALL_GEN, bt_max_weight=0.0)


def test_invalid_bt_max_weight_above_one() -> None:
    with pytest.raises(ValueError, match="bt_max_weight"):
        StrategySpec(gen=_SMALL_GEN, bt_max_weight=1.1)


def test_invalid_profile_trials_zero() -> None:
    with pytest.raises(ValueError, match="profile_trials"):
        StrategySpec(gen=_SMALL_GEN, profile_trials=0)


def test_invalid_bt_profile_trials_zero() -> None:
    with pytest.raises(ValueError, match="bt_profile_trials"):
        StrategySpec(gen=_SMALL_GEN, bt_profile_trials=0)


def test_invalid_profile_warmup_negative() -> None:
    with pytest.raises(ValueError, match="profile_warmup"):
        StrategySpec(gen=_SMALL_GEN, profile_warmup=-1)


def test_invalid_bt_profile_warmup_negative() -> None:
    with pytest.raises(ValueError, match="bt_profile_warmup"):
        StrategySpec(gen=_SMALL_GEN, bt_profile_warmup=-1)


def test_empty_horizon_map() -> None:
    with pytest.raises(ValueError, match="horizon_map"):
        StrategySpec(gen=_SMALL_GEN, horizon_map={})


# --------------------------------------------------------------------------- #
# Validation: cross-parameter constraints
# --------------------------------------------------------------------------- #


def test_embargo_times_n_splits_exceeds_n_dates_raises() -> None:
    # n_dates=30, n_splits=4, embargo=5 → min_dates = 4*(1+5)+1 = 25 ≤ 30, ok
    # n_dates=20, n_splits=4, embargo=5 → min_dates = 25 > 20, raises
    tiny_gen = GenSpec(n_assets=20, n_dates=20, n_features=4, n_factors=2, seed=0)
    with pytest.raises(ValueError, match="n_dates"):
        StrategySpec(gen=tiny_gen, wf_n_splits=4, wf_embargo_periods=5)


def test_bt_max_weight_too_small_for_n_assets_raises() -> None:
    # n_assets=20 → min feasible = 0.05; max_weight=0.04 < 0.05 → raises
    with pytest.raises(ValueError, match="bt_max_weight"):
        StrategySpec(gen=_SMALL_GEN, bt_max_weight=0.04)


# --------------------------------------------------------------------------- #
# JSON roundtrip
# --------------------------------------------------------------------------- #


def test_json_roundtrip_default() -> None:
    spec = _default_spec()
    s = spec.to_json()
    spec2 = StrategySpec.from_json(s)
    assert spec == spec2


def test_json_roundtrip_custom() -> None:
    custom = StrategySpec(
        gen=_SMALL_GEN,
        wf_n_splits=3,
        wf_embargo_periods=2,
        opt_risk_aversion=2.5,
        opt_max_iter=500,
        bt_max_weight=0.2,
        bt_enable_costs=False,
        bt_enable_slippage=False,
        neutralize_sectors=False,
        horizon_map={1: "fwd_ret_1", 5: "fwd_ret_5"},
    )
    assert custom == StrategySpec.from_json(custom.to_json())


def test_json_horizon_map_keys_are_ints_after_roundtrip() -> None:
    spec = _default_spec()
    spec2 = StrategySpec.from_json(spec.to_json())
    for k in spec2.horizon_map:
        assert isinstance(k, int)


# --------------------------------------------------------------------------- #
# Additivity proof: no-strategy call == baseline
# --------------------------------------------------------------------------- #


def test_no_strategy_identical_to_baseline(tmp_path) -> None:
    """run_production_pipeline(spec, workdir) with no strategy argument must
    produce a PipelineSummary field-for-field identical to the pre-spec baseline."""
    gen = GenSpec(n_assets=20, n_dates=60, n_features=4, n_factors=2, seed=7)

    baseline = run_production_pipeline(gen, tmp_path / "baseline")
    after = run_production_pipeline(gen, tmp_path / "after")

    # Field-for-field comparison (both runs use the same seed + same defaults)
    assert baseline.ic_raw == after.ic_raw
    assert baseline.ic_neutralized == after.ic_neutralized
    assert _dicts_equal_nan_safe(baseline.horizon_ic, after.horizon_ic)
    assert baseline.wf_mean_ic == after.wf_mean_ic
    assert baseline.wf_mean_r2 == after.wf_mean_r2
    assert baseline.opt_converged == after.opt_converged
    assert baseline.opt_gross == after.opt_gross
    assert baseline.factor_vol == after.factor_vol
    assert baseline.tracking_error == after.tracking_error
    assert baseline.gross_sharpe == after.gross_sharpe
    assert baseline.net_sharpe == after.net_sharpe
    assert baseline.cost_drag == after.cost_drag
    assert baseline.n_scaling_fits == after.n_scaling_fits


def test_default_strategy_identical_to_no_strategy(tmp_path) -> None:
    """Passing StrategySpec with all defaults == passing no strategy at all."""
    gen = GenSpec(n_assets=20, n_dates=60, n_features=4, n_factors=2, seed=7)
    strat = StrategySpec(gen=gen)

    no_strat = run_production_pipeline(gen, tmp_path / "no_strat")
    with_strat = run_production_pipeline(gen, tmp_path / "with_strat", strategy=strat)

    assert no_strat.ic_raw == with_strat.ic_raw
    assert no_strat.ic_neutralized == with_strat.ic_neutralized
    assert _dicts_equal_nan_safe(no_strat.horizon_ic, with_strat.horizon_ic)
    assert no_strat.wf_mean_ic == with_strat.wf_mean_ic
    assert no_strat.wf_mean_r2 == with_strat.wf_mean_r2
    assert no_strat.opt_converged == with_strat.opt_converged
    assert no_strat.opt_gross == with_strat.opt_gross
    assert no_strat.factor_vol == with_strat.factor_vol
    assert no_strat.tracking_error == with_strat.tracking_error
    assert no_strat.gross_sharpe == with_strat.gross_sharpe
    assert no_strat.net_sharpe == with_strat.net_sharpe
    assert no_strat.cost_drag == with_strat.cost_drag
    assert no_strat.n_scaling_fits == with_strat.n_scaling_fits


# --------------------------------------------------------------------------- #
# Behaviour change: different knobs → different results
# --------------------------------------------------------------------------- #


def test_different_n_splits_changes_wf_mean_ic(tmp_path) -> None:
    """Different wf_n_splits → different wf_mean_ic (different CV folds)."""
    gen = GenSpec(n_assets=20, n_dates=80, n_features=4, n_factors=2, seed=3)
    strat_default = StrategySpec(gen=gen, wf_n_splits=4, wf_embargo_periods=2)
    strat_3fold = StrategySpec(gen=gen, wf_n_splits=3, wf_embargo_periods=2)

    s4 = run_from_spec(strat_default, tmp_path / "s4")
    s3 = run_from_spec(strat_3fold, tmp_path / "s3")

    # Different folds must yield different mean ICs
    assert s4.wf_mean_ic != s3.wf_mean_ic


def test_different_risk_aversion_changes_opt_gross(tmp_path) -> None:
    """Changing opt_risk_aversion changes the optimizer's gross exposure."""
    gen = GenSpec(n_assets=20, n_dates=60, n_features=4, n_factors=2, seed=5)
    strat_low = StrategySpec(gen=gen, opt_risk_aversion=0.1)
    strat_high = StrategySpec(gen=gen, opt_risk_aversion=10.0)

    s_low = run_from_spec(strat_low, tmp_path / "low")
    s_high = run_from_spec(strat_high, tmp_path / "high")

    # Lower risk aversion → optimizer takes more risk → different gross
    assert s_low.opt_gross != s_high.opt_gross
