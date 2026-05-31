"""Contract: signal panel → IC research → model panel → walk-forward CV.

`alpha_signals` and `feature_panel` carry an injected correlation with
`forward_returns`; this verifies the signal→model hand-off recovers that
structure end-to-end (positive IC, finite CV scores) and that the panel
builder aligns features and target on (date, id).
"""

from __future__ import annotations

import math

import polars as pl

from etl.source import to_float
from models import (
    ModelConfig,
    RidgeModel,
    WalkForwardSplitter,
    build_panel,
    walk_forward_cv,
)
from signals import ic_horizon_curve, ic_series_v2, neutralize_sector


def test_signal_panel_has_recoverable_ic(synth) -> None:
    loader = synth.loader
    momentum = (
        loader.load("alpha_signals")
        .filter(pl.col("signal_name") == "momentum")
        .select("date", "id", "signal")
    )
    fwd = loader.load("forward_returns")
    ic = ic_series_v2(momentum, fwd, return_col="fwd_ret_1")
    assert to_float(ic["ic"].mean()) > 0.0

    neutral = neutralize_sector(momentum, loader.load("security_master"))
    ic_n = ic_series_v2(neutral, fwd, return_col="fwd_ret_1")
    assert abs(to_float(ic_n["ic"].mean())) < 1.0

    curve = ic_horizon_curve(
        momentum, fwd, {1: "fwd_ret_1", 5: "fwd_ret_5", 21: "fwd_ret_21", 63: "fwd_ret_63"}
    )
    assert {p.horizon for p in curve.points} == {1, 5, 21, 63}


def test_feature_panel_feeds_walk_forward_cv(synth) -> None:
    loader, spec = synth.loader, synth.spec
    panel = build_panel(
        loader.load("feature_panel"),
        loader.load("forward_returns"),
        "fwd_ret_1",
        weights=loader.load("sample_weights"),
    )
    assert panel.X.shape[0] == panel.y.shape[0] == panel.groups.shape[0]
    assert panel.X.shape[1] == spec.n_features

    wf = walk_forward_cv(
        panel,
        WalkForwardSplitter(n_splits=4, embargo_periods=5),
        lambda a: RidgeModel(ModelConfig(n_features=spec.n_features, alpha=a)),
    )
    assert math.isfinite(wf.mean_r2)
    assert math.isfinite(wf.mean_ic)
    assert len(wf.all_preds) == len(wf.all_true)


def test_walk_forward_never_trains_on_future(synth) -> None:
    """The splitter's train indices must all precede its test indices in time."""
    loader = synth.loader
    panel = build_panel(loader.load("feature_panel"), loader.load("forward_returns"), "fwd_ret_1")
    splitter = WalkForwardSplitter(n_splits=4, embargo_periods=5)
    for train_idx, test_idx in splitter.split(panel.X, groups=panel.groups):
        if len(train_idx) and len(test_idx):
            assert panel.groups[train_idx].max() < panel.groups[test_idx].min()
