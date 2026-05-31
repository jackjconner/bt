"""Tests for models.scoring — rank IC and R² scoring."""

from __future__ import annotations

import numpy as np
import pytest

from .scoring import held_out_r2, ic_stats, rank_ic_score, rank_ic_series


def _groups(n_dates: int, n_assets: int) -> np.ndarray:
    return np.repeat(np.arange(n_dates, dtype=np.int64), n_assets)


def test_rank_ic_perfect_positive():
    """Predictions identical to targets should give IC = 1.0 on every date."""
    rng = np.random.default_rng(42)
    n_dates, n_assets = 20, 10
    y = rng.normal(0.0, 1.0, n_dates * n_assets)
    groups = _groups(n_dates, n_assets)
    _, ic_vals = rank_ic_series(y, y, groups)
    np.testing.assert_allclose(ic_vals, 1.0, atol=1e-10)


def test_rank_ic_perfect_negative():
    """Predictions that are the reverse of targets should give IC = -1.0."""
    rng = np.random.default_rng(43)
    n_dates, n_assets = 20, 10
    y = rng.normal(0.0, 1.0, n_dates * n_assets)
    groups = _groups(n_dates, n_assets)
    _, ic_vals = rank_ic_series(y, -y, groups)
    np.testing.assert_allclose(ic_vals, -1.0, atol=1e-10)


def test_rank_ic_score_positive_for_correlated_signal():
    """A signal with injected positive correlation should produce a positive mean IC."""
    rng = np.random.default_rng(44)
    n_dates, n_assets = 50, 20
    y_true = rng.normal(0.0, 1.0, n_dates * n_assets)
    # signal correlated 0.5 with truth
    signal = 0.5 * y_true + np.sqrt(1 - 0.25) * rng.normal(0.0, 1.0, n_dates * n_assets)
    groups = _groups(n_dates, n_assets)
    ic = rank_ic_score(y_true, signal, groups)
    assert ic > 0.0, f"Expected positive IC but got {ic}"


def test_rank_ic_score_near_zero_for_noise():
    """A purely random prediction should produce an IC near zero (within noise)."""
    rng = np.random.default_rng(45)
    n_dates, n_assets = 100, 30
    y_true = rng.normal(0.0, 1.0, n_dates * n_assets)
    noise = rng.normal(0.0, 1.0, n_dates * n_assets)
    groups = _groups(n_dates, n_assets)
    ic = rank_ic_score(y_true, noise, groups)
    # 3-sigma bound for mean of 100 ICs with n=30 per date
    assert abs(ic) < 0.15, f"IC={ic} seems too large for pure noise"


def test_rank_ic_series_length():
    """rank_ic_series must return one IC value per unique date."""
    n_dates, n_assets = 15, 8
    y = np.zeros(n_dates * n_assets)
    groups = _groups(n_dates, n_assets)
    unique_grps, ic_vals = rank_ic_series(y, y, groups)
    assert len(unique_grps) == n_dates
    assert len(ic_vals) == n_dates


def test_rank_ic_degenerate_date_returns_zero():
    """A date with fewer than 3 assets should return IC = 0.0 without error."""
    # only 2 assets on date 0
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([0.5, 1.5])
    groups = np.array([0, 0], dtype=np.int64)
    _, ic_vals = rank_ic_series(y_true, y_pred, groups)
    assert ic_vals[0] == 0.0


def test_ic_stats_keys():
    """ic_stats should return expected keys."""
    ic_vals = np.array([0.05, 0.10, 0.03, -0.02, 0.08])
    stats = ic_stats(ic_vals)
    for key in ("mean_ic", "std_ic", "ic_ir", "t_stat", "n_dates"):
        assert key in stats


def test_ic_stats_values():
    ic_vals = np.array([0.1, 0.2, 0.3])
    stats = ic_stats(ic_vals)
    assert abs(stats["mean_ic"] - 0.2) < 1e-10
    assert stats["n_dates"] == 3
    expected_ir = 0.2 / np.std(ic_vals, ddof=1)
    assert abs(stats["ic_ir"] - expected_ir) < 1e-10


def test_held_out_r2_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert held_out_r2(y, y) == pytest.approx(1.0)


def test_held_out_r2_zero_variance():
    y = np.array([1.0, 1.0, 1.0])
    preds = np.array([2.0, 2.0, 2.0])
    # zero variance target → returns 0.0 without division error
    assert held_out_r2(y, preds) == 0.0
