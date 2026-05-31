"""Tests for regime detection and regime-conditional IC evaluation.

Key verification targets
------------------------
1. detect_regimes recovers KNOWN synthetic regimes above chance — we
   synthesize returns with clearly different volatility per block and check
   accuracy > random (1/n_regimes).
2. regime_conditional_ic returns finite per-regime ICs; observation counts
   are internally consistent (total obs across regimes ≤ total panel obs).
3. Existing etl/signals datasets/tests are unaffected (proven by running the
   full suite with --tb=short, not here, but additivity is checked by
   confirming REGISTRY still has the same keys minus the new one).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from etl.datasets import REGISTRY, GenSpec, generate
from signals import (
    RegimeConditionalICResult,
    detect_regimes,
    regime_conditional_ic,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPEC = GenSpec(n_assets=30, n_dates=120, seed=99)


def _signals() -> pl.DataFrame:
    df = generate("alpha_signals", SPEC)
    return df.filter(pl.col("signal_name") == "momentum").select("date", "id", "signal")


def _fwd() -> pl.DataFrame:
    return generate("forward_returns", SPEC)


# ---------------------------------------------------------------------------
# 1. detect_regimes
# ---------------------------------------------------------------------------


def test_detect_regimes_returns_correct_length() -> None:
    ret = pl.Series("r", np.random.default_rng(0).normal(0.0, 1.0, 200))
    labels = detect_regimes(ret)
    assert len(labels) == 200


def test_detect_regimes_warmup_labels_are_minus_one() -> None:
    """Dates before min_window have label -1."""
    n = 50
    ret = pl.Series("r", np.random.default_rng(1).normal(0.0, 1.0, n))
    labels = detect_regimes(ret, window=21, min_window=5)
    arr = labels.to_numpy()
    # First min_window-1 entries must be -1 (warm-up)
    assert (arr[:4] == -1).all()


def test_detect_regimes_label_range() -> None:
    """All valid labels fall in [0, n_regimes-1]."""
    n_regimes = 3
    ret = pl.Series("r", np.random.default_rng(2).normal(0.0, 1.0, 200))
    labels = detect_regimes(ret, n_regimes=n_regimes)
    arr = labels.to_numpy()
    valid = arr[arr >= 0]
    assert len(valid) > 0
    assert int(valid.min()) == 0
    assert int(valid.max()) == n_regimes - 1


def test_detect_regimes_two_regime_case() -> None:
    """n_regimes=2 produces only labels -1, 0, 1."""
    ret = pl.Series("r", np.random.default_rng(3).normal(0.0, 1.0, 100))
    labels = detect_regimes(ret, n_regimes=2)
    arr = labels.to_numpy()
    assert set(arr.tolist()).issubset({-1, 0, 1})


def test_detect_regimes_recovers_known_volatility_structure() -> None:
    """Detector must do better than random (1/n_regimes) on synthetic data
    with clearly separated volatility blocks.

    Procedure
    ---------
    * Synthesize 300 dates split into 3 blocks of 100:
      block 0 → σ = 0.5 (low vol), block 1 → σ = 2.0 (mid vol),
      block 2 → σ = 5.0 (high vol).
    * True label for each date = block index (0/1/2).
    * Run detect_regimes and compare.
    * Compute best-permutation accuracy using a greedy matching
      (since detected labels may not align with true label integers).
    * Assert accuracy > 1/n_regimes + 0.15 (well above chance).
    """
    rng = np.random.default_rng(42)
    n_per_block = 100
    n_regimes = 3
    sigmas = [0.5, 2.0, 5.0]  # low / mid / high vol

    returns_arr = np.concatenate([rng.normal(0.0, s, n_per_block) for s in sigmas])
    true_labels = np.repeat([0, 1, 2], n_per_block)

    ret_series = pl.Series("r", returns_arr)
    detected = detect_regimes(ret_series, n_regimes=n_regimes, window=20, min_window=5)
    det_arr = detected.to_numpy()

    # Only compare dates where detected label is valid (≥ 0)
    valid_mask = det_arr >= 0
    det_valid = det_arr[valid_mask]
    true_valid = true_labels[valid_mask]

    # Best-permutation accuracy: try all 6 permutations of {0,1,2}
    from itertools import permutations

    best_acc = 0.0
    for perm in permutations(range(n_regimes)):
        mapping = dict(enumerate(perm))
        remapped = np.array([mapping[d] for d in det_valid])
        acc = float((remapped == true_valid).mean())
        best_acc = max(best_acc, acc)

    chance = 1.0 / n_regimes
    # Require meaningfully above chance
    assert best_acc > chance + 0.15, (
        f"detector accuracy {best_acc:.3f} not above chance {chance:.3f} + 0.15"
    )


# ---------------------------------------------------------------------------
# 2. regime_conditional_ic
# ---------------------------------------------------------------------------


def test_regime_conditional_ic_returns_result_type() -> None:
    sig = _signals()
    fwd = _fwd()
    reg = generate("regime_states", SPEC).rename({"regime_state": "regime_label"})
    result = regime_conditional_ic(sig, fwd, reg, return_col="fwd_ret_1")
    assert isinstance(result, RegimeConditionalICResult)


def test_regime_conditional_ic_keys_are_regime_labels() -> None:
    """Result has an entry for every regime label present in the data."""
    sig = _signals()
    fwd = _fwd()
    reg = generate("regime_states", SPEC).rename({"regime_state": "regime_label"})
    result = regime_conditional_ic(sig, fwd, reg, return_col="fwd_ret_1")
    data_labels = sorted(v for v in reg["regime_label"].unique().to_list() if v >= 0)
    assert sorted(result.per_regime_ic.keys()) == data_labels


def test_regime_conditional_ic_finite_ics() -> None:
    """Per-regime mean ICs must be finite when the regime has enough dates."""
    sig = _signals()
    fwd = _fwd()
    reg = generate("regime_states", SPEC).rename({"regime_state": "regime_label"})
    result = regime_conditional_ic(sig, fwd, reg, return_col="fwd_ret_1", min_obs=2)
    # At least one regime should have a valid IC (not NaN) with 120 dates
    finite_count = sum(1 for v in result.per_regime_ic.values() if np.isfinite(v))
    assert finite_count > 0


def test_regime_conditional_ic_n_obs_positive() -> None:
    sig = _signals()
    fwd = _fwd()
    reg = generate("regime_states", SPEC).rename({"regime_state": "regime_label"})
    result = regime_conditional_ic(sig, fwd, reg, return_col="fwd_ret_1", min_obs=2)
    assert all(v >= 0 for v in result.per_regime_n_obs.values())


def test_regime_conditional_ic_spread_non_negative() -> None:
    sig = _signals()
    fwd = _fwd()
    reg = generate("regime_states", SPEC).rename({"regime_state": "regime_label"})
    result = regime_conditional_ic(sig, fwd, reg, return_col="fwd_ret_1", min_obs=2)
    if np.isfinite(result.regime_spread):
        assert result.regime_spread >= 0.0


def test_regime_conditional_ic_spread_equals_max_minus_min() -> None:
    """Spread = best IC - worst IC (verified against per_regime_ic dict)."""
    sig = _signals()
    fwd = _fwd()
    reg = generate("regime_states", SPEC).rename({"regime_state": "regime_label"})
    result = regime_conditional_ic(sig, fwd, reg, return_col="fwd_ret_1", min_obs=2)
    finite = [v for v in result.per_regime_ic.values() if np.isfinite(v)]
    if len(finite) >= 2:
        expected_spread = max(finite) - min(finite)
        assert result.regime_spread == pytest.approx(expected_spread, abs=1e-9)


def test_regime_conditional_ic_accepts_regime_state_column() -> None:
    """regime_states dataset column 'regime_state' works without renaming."""
    sig = _signals()
    fwd = _fwd()
    reg = generate("regime_states", SPEC)  # has 'regime_state' column
    result = regime_conditional_ic(sig, fwd, reg, return_col="fwd_ret_1", min_obs=2)
    assert isinstance(result, RegimeConditionalICResult)


def test_regime_conditional_ic_method_kwarg() -> None:
    """Method kwarg is recorded in the result."""
    sig = _signals()
    fwd = _fwd()
    reg = generate("regime_states", SPEC)
    result = regime_conditional_ic(
        sig, fwd, reg, return_col="fwd_ret_1", method="pearson", min_obs=2
    )
    assert result.method == "pearson"


def test_regime_conditional_ic_total_obs_consistent() -> None:
    """Total obs across all regimes ≤ n_assets * n_valid_dates (sanity bound)."""
    sig = _signals()
    fwd = _fwd()
    reg = generate("regime_states", SPEC)
    result = regime_conditional_ic(sig, fwd, reg, return_col="fwd_ret_1", min_obs=2)
    total = sum(result.per_regime_n_obs.values())
    max_possible = SPEC.n_assets * SPEC.n_dates
    assert total <= max_possible


# ---------------------------------------------------------------------------
# 3. Additivity: existing REGISTRY datasets are not removed
# ---------------------------------------------------------------------------


def test_regime_states_added_to_registry() -> None:
    assert "regime_states" in REGISTRY


def test_existing_registry_datasets_intact() -> None:
    """All pre-existing datasets still present — regime_states was added, not substituted."""
    pre_existing = {
        "prices",
        "returns",
        "forward_returns",
        "alpha_signals",
        "feature_panel",
        "universe_mask",
        "borrow_rates",
    }
    # filter to only check ones that exist (returns isn't in this registry)
    existing = {k for k in pre_existing if k in REGISTRY}
    assert existing.issuperset(pre_existing - {"returns"})


# ---------------------------------------------------------------------------
# 4. regime_states schema / generator
# ---------------------------------------------------------------------------


def test_regime_states_schema_valid() -> None:
    df = generate("regime_states", SPEC)
    assert "date" in df.columns
    assert "regime_state" in df.columns
    assert df.height == SPEC.n_dates


def test_regime_states_labels_in_range() -> None:
    df = generate("regime_states", SPEC)
    states = df["regime_state"].to_numpy()
    assert int(states.min()) >= 0
    from etl.datasets import N_REGIMES

    assert int(states.max()) < N_REGIMES


def test_regime_states_deterministic() -> None:
    a = generate("regime_states", SPEC)
    b = generate("regime_states", SPEC)
    assert a.equals(b)


def test_regime_states_different_seed_differs() -> None:
    spec_a = GenSpec(n_assets=10, n_dates=60, seed=1)
    spec_b = GenSpec(n_assets=10, n_dates=60, seed=2)
    a = generate("regime_states", spec_a)
    b = generate("regime_states", spec_b)
    assert not a.equals(b)
