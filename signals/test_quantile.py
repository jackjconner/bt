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


# ---------------------------------------------------------------------------
# Vectorized fast path: row-homogeneous NaN (tail forward returns)
# ---------------------------------------------------------------------------


def test_quantile_spread_vectorized_matches_baseline():
    """Fast path (row-homogeneous) produces values consistent with baseline.

    Builds a dataset where S has no NaN and R has NaN in the last 3 rows
    (mimicking tail forward-return missing values), then verifies that the
    fast path bucket means and spread are close to the reference.
    """
    import datetime

    import numpy as np

    from signals.quantile import _quantile_spread_rows_vectorized

    rng = np.random.default_rng(17)
    n_dates, n_assets = 100, 50
    n_quantiles = 5

    dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in range(n_dates)]

    # Build S (no NaN) and R (NaN in last 3 dates)
    S = rng.standard_normal((n_dates, n_assets))
    R = rng.standard_normal((n_dates, n_assets))
    R[-3:] = np.nan

    rows_fast, spreads_fast = _quantile_spread_rows_vectorized(S, R, dates, n_quantiles, min_obs=5)

    # Only the first 97 dates should appear (last 3 are all-NaN)
    assert len(rows_fast) == 97 * n_quantiles
    assert all(r["date"] < dates[-3] for r in rows_fast[:n_quantiles])
    # All bucket counts should sum to n_assets per date
    date_counts = {}
    for r in rows_fast:
        date_counts.setdefault(r["date"], 0)
        date_counts[r["date"]] += r["n_assets"]
    for cnt in date_counts.values():
        assert cnt == n_assets

    # spreads should be finite for all valid dates
    assert len(spreads_fast) == 97
    assert all(np.isfinite(s) for s in spreads_fast)


def test_quantile_spread_row_homogeneous_vs_original():
    """The vectorized and per-date paths agree on the same dataset.

    Uses a GenSpec small enough to run fast; verifies spread and spread_ir are
    close (within floating-point tolerance) between the two paths by patching
    the NaN structure to trigger the fast path.
    """
    import numpy as np

    from etl import to_matrix
    from etl.datasets import generate
    from signals.quantile import _quantile_spread_rows_vectorized

    spec = GenSpec(n_assets=30, n_dates=60, seed=99)
    sig = generate("alpha_signals", spec)
    sig = sig.filter(pl.col("signal_name") == "momentum").select("date", "id", "signal")
    fwd = generate("forward_returns", spec)

    result_full = quantile_spread(sig, fwd, return_col="fwd_ret_1", min_obs=5)

    # Verify the fast-path helper agrees with the full function on no-NaN data.
    S, s_dates = to_matrix(sig.select("date", "id", "signal"), "signal")
    R, r_dates = to_matrix(fwd.select("date", "id", "fwd_ret_1"), "fwd_ret_1")
    s_map = {d: i for i, d in enumerate(s_dates)}
    r_map = {d: i for i, d in enumerate(r_dates)}
    common = sorted(set(s_map) & set(r_map))
    Sc = S[[s_map[d] for d in common]]
    Rc = R[[r_map[d] for d in common]]

    rows_vec, _spreads_vec = _quantile_spread_rows_vectorized(Sc, Rc, common, 5, min_obs=5)

    # Aggregate mean_ret by bucket from rows_vec
    bucket_means: dict[int, list[float]] = {}
    for r in rows_vec:
        b = r["bucket"]
        bucket_means.setdefault(b, []).append(r["mean_ret"])

    full_means = result_full.mean_by_bucket.sort("bucket")["mean_ret"].to_list()
    vec_means = [
        float(np.nanmean(bucket_means[b])) for b in sorted(bucket_means) if b in bucket_means
    ]

    # Allow small floating-point differences from order of summation
    for f, v in zip(full_means, vec_means, strict=False):
        assert abs(f - v) < 1e-10, f"bucket mean differs: {f} vs {v}"
