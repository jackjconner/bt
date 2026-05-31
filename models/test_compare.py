"""Tests for models.compare — compare_models and ModelComparison."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl

from .boosting import GradientBoostConfig, GradientBoostModel
from .compare import ModelComparison, compare_models
from .panel import build_panel
from .ridge import ModelConfig, RidgeModel
from .splitters import WalkForwardSplitter
from .walk_forward import WalkForwardConfig

# --------------------------------------------------------------------------- #
# Synthetic panel helpers
# --------------------------------------------------------------------------- #


def _linear_panel(n_dates: int = 80, n_assets: int = 10, n_features: int = 5, seed: int = 0):
    """Panel with a mild linear signal (first two features matter)."""
    rng = np.random.default_rng(seed)
    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_dates)]
    beta = np.zeros(n_features)
    beta[:2] = [0.4, -0.3]

    rows_feat: list[dict[str, Any]] = []
    rows_tgt: list[dict[str, Any]] = []
    for d in dates:
        X_cross = rng.normal(0.0, 1.0, (n_assets, n_features))
        y_cross = X_cross @ beta + rng.normal(0.0, 0.5, n_assets)
        for aid in range(n_assets):
            row: dict[str, Any] = {"date": d, "id": aid}
            for f in range(n_features):
                row[f"feat_{f}"] = float(X_cross[aid, f])
            rows_feat.append(row)
            rows_tgt.append({"date": d, "id": aid, "fwd_ret_1": float(y_cross[aid])})

    feat_df = pl.DataFrame(rows_feat).with_columns(pl.col("id").cast(pl.Int64))
    tgt_df = pl.DataFrame(rows_tgt).with_columns(pl.col("id").cast(pl.Int64))
    return build_panel(feat_df, tgt_df, "fwd_ret_1")


def _nonlinear_panel(n_dates: int = 100, n_assets: int = 12, n_features: int = 5, seed: int = 1):
    """Panel where the true signal is nonlinear (X[0]^2 + X[1]*X[2])."""
    rng = np.random.default_rng(seed)
    start = date(2020, 1, 2)
    dates = [start + timedelta(days=i) for i in range(n_dates)]

    rows_feat: list[dict[str, Any]] = []
    rows_tgt: list[dict[str, Any]] = []
    for d in dates:
        X_cross = rng.normal(0.0, 1.0, (n_assets, n_features))
        y_cross = (
            X_cross[:, 0] ** 2 + X_cross[:, 1] * X_cross[:, 2] + rng.normal(0.0, 0.1, n_assets)
        )
        for aid in range(n_assets):
            row: dict[str, Any] = {"date": d, "id": aid}
            for f in range(n_features):
                row[f"feat_{f}"] = float(X_cross[aid, f])
            rows_feat.append(row)
            rows_tgt.append({"date": d, "id": aid, "fwd_ret_1": float(y_cross[aid])})

    feat_df = pl.DataFrame(rows_feat).with_columns(pl.col("id").cast(pl.Int64))
    tgt_df = pl.DataFrame(rows_tgt).with_columns(pl.col("id").cast(pl.Int64))
    return build_panel(feat_df, tgt_df, "fwd_ret_1")


def _make_factories(n_features: int):
    def ridge_factory(alpha: float) -> RidgeModel:
        return RidgeModel(ModelConfig(n_features=n_features, alpha=alpha))

    def boost_factory(_alpha: float) -> GradientBoostModel:
        return GradientBoostModel(GradientBoostConfig(n_features=n_features))

    return {"ridge": ridge_factory, "boost": boost_factory}


# --------------------------------------------------------------------------- #
# ModelComparison shape and types
# --------------------------------------------------------------------------- #


class TestModelComparison:
    def test_returns_model_comparison(self):
        panel = _linear_panel()
        splitter = WalkForwardSplitter(n_splits=3, min_train_periods=15)
        config = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)
        result = compare_models(_make_factories(5), panel, splitter, config)
        assert isinstance(result, ModelComparison)

    def test_results_keys_match_input(self):
        panel = _linear_panel()
        splitter = WalkForwardSplitter(n_splits=3, min_train_periods=15)
        config = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)
        result = compare_models(_make_factories(5), panel, splitter, config)
        assert set(result.results.keys()) == {"ridge", "boost"}

    def test_ranking_covers_all_models(self):
        panel = _linear_panel()
        splitter = WalkForwardSplitter(n_splits=3, min_train_periods=15)
        config = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)
        result = compare_models(_make_factories(5), panel, splitter, config)
        ranked_names = [name for name, _ in result.ranking]
        assert set(ranked_names) == {"ridge", "boost"}

    def test_ranking_descending_by_ic(self):
        """Ranking must be sorted by mean OOS IC descending."""
        panel = _linear_panel()
        splitter = WalkForwardSplitter(n_splits=3, min_train_periods=15)
        config = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)
        result = compare_models(_make_factories(5), panel, splitter, config)
        ics = [ic for _, ic in result.ranking]
        assert ics == sorted(ics, reverse=True)

    def test_best_is_first_in_ranking(self):
        """best must equal the first entry in ranking."""
        panel = _linear_panel()
        splitter = WalkForwardSplitter(n_splits=3, min_train_periods=15)
        config = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)
        result = compare_models(_make_factories(5), panel, splitter, config)
        assert result.best == result.ranking[0][0]

    def test_per_model_ic_finite(self):
        """All per-model OOS ICs must be finite."""
        panel = _linear_panel()
        splitter = WalkForwardSplitter(n_splits=3, min_train_periods=15)
        config = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)
        result = compare_models(_make_factories(5), panel, splitter, config)
        for _name, ic in result.ranking:
            assert np.isfinite(ic)

    def test_frozen_dataclass(self):
        """ModelComparison must be declared as a frozen dataclass."""
        import dataclasses

        assert dataclasses.is_dataclass(ModelComparison)
        assert ModelComparison.__dataclass_params__.frozen  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Single-model edge case
# --------------------------------------------------------------------------- #


def test_single_model_comparison():
    """compare_models with a single model must still return a valid ranking."""
    panel = _linear_panel()
    splitter = WalkForwardSplitter(n_splits=3, min_train_periods=15)
    config = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)

    def ridge_factory(alpha: float) -> RidgeModel:
        return RidgeModel(ModelConfig(n_features=5, alpha=alpha))

    result = compare_models({"ridge": ridge_factory}, panel, splitter, config)
    assert len(result.ranking) == 1
    assert result.best == "ridge"


# --------------------------------------------------------------------------- #
# Nonlinear data: boost should not rank worse than ridge (sanity check)
# --------------------------------------------------------------------------- #


def test_boost_not_worse_on_nonlinear():
    """On data with a nonlinear true signal, GradientBoostModel OOS IC should
    be >= Ridge OOS IC.  This is a sanity check, not a hard guarantee."""
    panel = _nonlinear_panel()
    splitter = WalkForwardSplitter(n_splits=4, min_train_periods=15)
    config = WalkForwardConfig(alpha_grid=[0.1], scale_features=False, use_sample_weights=False)

    def ridge_factory(alpha: float) -> RidgeModel:
        return RidgeModel(ModelConfig(n_features=5, alpha=alpha))

    def boost_factory(_alpha: float) -> GradientBoostModel:
        return GradientBoostModel(
            GradientBoostConfig(n_features=5, max_iter=300, min_samples_leaf=5)
        )

    result = compare_models(
        {"ridge": ridge_factory, "boost": boost_factory}, panel, splitter, config
    )
    boost_ic = result.results["boost"].mean_ic
    ridge_ic = result.results["ridge"].mean_ic
    assert boost_ic >= ridge_ic, (
        f"Expected boost IC={boost_ic:.4f} >= ridge IC={ridge_ic:.4f} on nonlinear data"
    )
