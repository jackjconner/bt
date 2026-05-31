"""Tests for quantile spread analysis and monotonicity scoring."""

from __future__ import annotations

import numpy as np
import polars as pl

from etl.datasets import GenSpec, generate
from signals.quantile import QuantileResult, quantile_spread

SPEC = GenSpec(n_assets=50, n_dates=60, seed=7)


def _signals_for(name: str) -> pl.DataFrame:
    df = generate("alpha_signals", SPEC)
    return df.filter(pl.col("signal_name") == name).select("date", "id", "signal")


def _fwd_returns() -> pl.DataFrame:
    return generate("forward_returns", SPEC)


def test_quantile_spread_returns_quantile_result():
    result = quantile_spread(
        _signals_for("momentum"),
        _fwd_returns(),
        return_col="fwd_ret_1",
        n_quantiles=5,
        min_obs=5,
    )
    assert isinstance(result, QuantileResult)


def test_quantile_spread_bucket_returns_has_correct_buckets():
    result = quantile_spread(
        _signals_for("momentum"),
        _fwd_returns(),
        return_col="fwd_ret_1",
        n_quantiles=5,
        min_obs=5,
    )
    buckets = result.mean_by_bucket["bucket"].to_list()
    assert set(buckets) == {1, 2, 3, 4, 5}


def test_quantile_spread_monotonicity_positive_for_predictive_signal():
    """A signal with injected positive IC should produce positive monotonicity."""
    result = quantile_spread(
        _signals_for("quality"),  # highest IC signal (IC ~0.11)
        _fwd_returns(),
        return_col="fwd_ret_1",
        n_quantiles=5,
        min_obs=5,
    )
    # Monotonicity score: +1 = perfectly rising, should be positive for a good signal
    assert result.monotonicity_score > 0


def test_quantile_spread_positive_spread_for_predictive_signal():
    """Top-minus-bottom spread should be positive for a signal with positive IC."""
    result = quantile_spread(
        _signals_for("quality"),
        _fwd_returns(),
        return_col="fwd_ret_1",
        n_quantiles=5,
        min_obs=5,
    )
    assert result.spread > 0


def test_quantile_spread_n_quantiles_respected():
    for nq in [3, 5, 10]:
        result = quantile_spread(
            _signals_for("momentum"),
            _fwd_returns(),
            return_col="fwd_ret_1",
            n_quantiles=nq,
            min_obs=3,
        )
        assert result.n_quantiles == nq
        buckets = result.mean_by_bucket["bucket"].to_list()
        assert set(buckets) == set(range(1, nq + 1))


def test_quantile_spread_spread_ir_finite():
    result = quantile_spread(
        _signals_for("momentum"),
        _fwd_returns(),
        return_col="fwd_ret_1",
    )
    assert np.isfinite(result.spread_ir)


def test_quantile_spread_bucket_returns_columns():
    result = quantile_spread(
        _signals_for("momentum"),
        _fwd_returns(),
        return_col="fwd_ret_1",
    )
    assert set(result.bucket_returns.columns) >= {"date", "bucket", "mean_ret", "n_assets"}


def test_quantile_spread_min_obs_reduces_dates():
    low = quantile_spread(
        _signals_for("momentum"),
        _fwd_returns(),
        return_col="fwd_ret_1",
        min_obs=2,
    )
    high = quantile_spread(
        _signals_for("momentum"),
        _fwd_returns(),
        return_col="fwd_ret_1",
        min_obs=999,
    )
    # Higher min_obs means fewer qualifying dates => fewer rows in bucket_returns
    assert len(low.bucket_returns) >= len(high.bucket_returns)
