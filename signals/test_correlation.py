"""Tests for cross-signal pair-wise rank-correlation and diversification."""

from __future__ import annotations

import datetime

import numpy as np
import polars as pl
import pytest
from scipy import stats

from etl.datasets import GenSpec, generate
from signals.correlation import (
    SignalCorrelationResult,
    signal_pair_correlation,
)

SPEC = GenSpec(n_assets=50, n_dates=60, seed=7)
NAMES = ["momentum", "value", "quality"]


def _signal(name: str, spec: GenSpec = SPEC) -> pl.DataFrame:
    df = generate("alpha_signals", spec)
    return df.filter(pl.col("signal_name") == name).select("date", "id", "signal")


def _book(spec: GenSpec = SPEC) -> dict[str, pl.DataFrame]:
    return {name: _signal(name, spec) for name in NAMES}


# ---------------------------------------------------------------------------
# Shape / type contract
# ---------------------------------------------------------------------------


def test_returns_signal_correlation_result():
    result = signal_pair_correlation(_book(), min_obs=5)
    assert isinstance(result, SignalCorrelationResult)
    assert result.n_signals == 3
    assert result.signal_names == NAMES


def test_matrix_is_square_with_signal_column():
    result = signal_pair_correlation(_book(), min_obs=5)
    m = result.correlation_matrix
    assert m.height == 3
    assert m.columns == ["signal", *NAMES]
    assert m["signal"].to_list() == NAMES


def test_diagonal_is_unit():
    result = signal_pair_correlation(_book(), min_obs=5)
    m = result.correlation_matrix
    for name in NAMES:
        diag = m.filter(pl.col("signal") == name)[name].item()
        assert diag == 1.0


def test_matrix_is_symmetric():
    result = signal_pair_correlation(_book(), min_obs=5)
    arr = result.correlation_matrix.drop("signal").to_numpy()
    assert np.allclose(arr, arr.T, equal_nan=True)


def test_entries_bounded_in_unit_interval():
    result = signal_pair_correlation(_book(), min_obs=5)
    arr = result.correlation_matrix.drop("signal").to_numpy()
    finite = arr[np.isfinite(arr)]
    assert np.all(finite >= -1.0 - 1e-12)
    assert np.all(finite <= 1.0 + 1e-12)


def test_mean_abs_correlation_in_range():
    result = signal_pair_correlation(_book(), min_obs=5)
    assert 0.0 <= result.mean_abs_correlation <= 1.0


def test_redundancy_ratio_in_range():
    result = signal_pair_correlation(_book(), min_obs=5)
    # Redundancy in [0, 1]; lower = more diversified.  Near-uncorrelated signals
    # sit around 1/n and may dip slightly below it when correlations are negative.
    assert 0.0 - 1e-12 <= result.diversification_ratio <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# Numerical correctness vs scipy on within-date ranks
# ---------------------------------------------------------------------------


def test_pair_corr_matches_scipy_pooled_within_date_ranks():
    """Off-diagonal entry equals Pearson on within-date scipy ranks, pooled."""
    book = _book()
    result = signal_pair_correlation(book, min_obs=5)

    a = book["momentum"]
    b = book["value"]
    joined = a.rename({"signal": "sa"}).join(
        b.rename({"signal": "sb"}), on=["date", "id"], how="inner"
    )
    joined = joined.filter(
        pl.col("sa").is_not_null()
        & pl.col("sa").is_finite()
        & pl.col("sb").is_not_null()
        & pl.col("sb").is_finite()
    )
    ra_parts: list[np.ndarray] = []
    rb_parts: list[np.ndarray] = []
    for (_d,), grp in joined.group_by("date", maintain_order=True):
        ra_parts.append(stats.rankdata(grp["sa"].to_numpy(), method="average"))
        rb_parts.append(stats.rankdata(grp["sb"].to_numpy(), method="average"))
    ra = np.concatenate(ra_parts)
    rb = np.concatenate(rb_parts)
    expected = float(np.corrcoef(ra, rb)[0, 1])

    got = result.correlation_matrix.filter(pl.col("signal") == "momentum")["value"].item()
    assert abs(got - expected) < 1e-10


def test_self_correlation_via_duplicate_is_one():
    sig = _signal("momentum")
    result = signal_pair_correlation({"a": sig, "b": sig}, min_obs=5)
    off = result.correlation_matrix.filter(pl.col("signal") == "a")["b"].item()
    assert abs(off - 1.0) < 1e-10
    # Identical signals => no diversification => redundancy == 1.
    assert abs(result.diversification_ratio - 1.0) < 1e-10
    assert abs(result.mean_abs_correlation - 1.0) < 1e-10


def test_negated_signal_correlates_minus_one():
    sig = _signal("momentum")
    neg = sig.with_columns((-pl.col("signal")).alias("signal"))
    result = signal_pair_correlation({"a": sig, "b": neg}, min_obs=5)
    off = result.correlation_matrix.filter(pl.col("signal") == "a")["b"].item()
    assert abs(off - (-1.0)) < 1e-10


def test_anticorrelated_pair_redundancy_near_zero():
    """A perfectly anti-correlated pair fully cancels: redundancy -> 0."""
    sig = _signal("momentum")
    neg = sig.with_columns((-pl.col("signal")).alias("signal"))
    result = signal_pair_correlation({"a": sig, "b": neg}, min_obs=5)
    # (1ᵀ C 1)/n² = (2 + 2*(-1))/4 = 0 for a 2-signal book with corr -1.
    assert abs(result.diversification_ratio) < 1e-10
    assert result.diversification_ratio < 1.0 / 2


# ---------------------------------------------------------------------------
# Diversification behaviour
# ---------------------------------------------------------------------------


def test_uncorrelated_book_more_diversified_than_redundant_book():
    """A book of near-duplicate signals is more redundant than an independent one."""
    base = _signal("momentum")
    near_dup = base.with_columns((pl.col("signal") * 1.0 + 1e-6 * pl.col("id")).alias("signal"))
    redundant = signal_pair_correlation({"a": base, "b": near_dup}, min_obs=5)

    independent = signal_pair_correlation(
        {"a": _signal("momentum"), "b": _signal("value")}, min_obs=5
    )
    assert redundant.diversification_ratio > independent.diversification_ratio


def test_redundancy_decreases_as_signals_added_independently():
    one = signal_pair_correlation({"momentum": _signal("momentum")}, min_obs=5)
    assert np.isnan(one.diversification_ratio)
    three = signal_pair_correlation(_book(), min_obs=5)
    # Three independent-ish signals should not be maximally redundant.
    assert three.diversification_ratio < 1.0


# ---------------------------------------------------------------------------
# Edge cases: single signal, NaN tails, multi-horizon, input forms
# ---------------------------------------------------------------------------


def test_single_signal_yields_nan_scalars():
    result = signal_pair_correlation({"momentum": _signal("momentum")}, min_obs=5)
    assert result.n_signals == 1
    assert result.correlation_matrix["momentum"].item() == 1.0
    assert np.isnan(result.mean_abs_correlation)
    assert np.isnan(result.diversification_ratio)


def test_nan_tails_handled_via_pairwise_mask():
    """Cells where either signal is NaN drop out; the pair still scores."""
    a = _signal("momentum")
    b = _signal("value")
    # Inject NaN into the trailing dates of b (mimicking a horizon tail).
    last_dates = sorted(b["date"].unique().to_list())[-5:]
    b_nan = b.with_columns(
        pl.when(pl.col("date").is_in(last_dates))
        .then(None)
        .otherwise(pl.col("signal"))
        .alias("signal")
    )
    result = signal_pair_correlation({"a": a, "b": b_nan}, min_obs=5)
    off = result.correlation_matrix.filter(pl.col("signal") == "a")["b"].item()
    assert np.isfinite(off)
    assert -1.0 <= off <= 1.0


def test_insufficient_coverage_gives_nan_entry():
    a = _signal("momentum")
    b = _signal("value")
    result = signal_pair_correlation({"a": a, "b": b}, min_obs=10**9)
    off = result.correlation_matrix.filter(pl.col("signal") == "a")["b"].item()
    assert np.isnan(off)
    # Mean abs correlation over no finite pair is nan.
    assert np.isnan(result.mean_abs_correlation)


def test_multi_horizon_distinct_books_score_independently():
    """The function is horizon-agnostic: caller passes whatever set it scores."""
    short = signal_pair_correlation(
        {"momentum": _signal("momentum"), "value": _signal("value")}, min_obs=5
    )
    full = signal_pair_correlation(_book(), min_obs=5)
    assert short.n_signals == 2
    assert full.n_signals == 3
    # The shared (momentum, value) pair is identical across the two books.
    mv_short = short.correlation_matrix.filter(pl.col("signal") == "momentum")["value"].item()
    mv_full = full.correlation_matrix.filter(pl.col("signal") == "momentum")["value"].item()
    assert abs(mv_short - mv_full) < 1e-12


def test_list_input_with_default_names():
    result = signal_pair_correlation([_signal("momentum"), _signal("value")], min_obs=5)
    assert result.signal_names == ["signal_0", "signal_1"]


def test_list_input_with_explicit_names():
    result = signal_pair_correlation(
        [_signal("momentum"), _signal("value")],
        names=["mom", "val"],
        min_obs=5,
    )
    assert result.signal_names == ["mom", "val"]
    assert result.correlation_matrix.columns == ["signal", "mom", "val"]


def test_names_length_mismatch_raises():
    with pytest.raises(ValueError, match="names must match"):
        signal_pair_correlation([_signal("momentum")], names=["a", "b"])


def test_empty_signals_raises():
    with pytest.raises(ValueError, match="non-empty"):
        signal_pair_correlation([])


def test_custom_signal_col():
    a = _signal("momentum").rename({"signal": "alpha"})
    b = _signal("value").rename({"signal": "alpha"})
    result = signal_pair_correlation({"a": a, "b": b}, signal_col="alpha", min_obs=5)
    off = result.correlation_matrix.filter(pl.col("signal") == "a")["b"].item()
    assert np.isfinite(off)


# ---------------------------------------------------------------------------
# Hand-built deterministic check
# ---------------------------------------------------------------------------


def test_known_perfect_rank_agreement():
    """Two signals that rank assets identically every date correlate +1."""
    dates = [datetime.date(2021, 1, 1) + datetime.timedelta(days=i) for i in range(4)]
    ids = list(range(6))
    rows_a = []
    rows_b = []
    rng = np.random.default_rng(3)
    for d in dates:
        vals = rng.standard_normal(len(ids))
        for aid, v in zip(ids, vals, strict=True):
            rows_a.append({"date": d, "id": aid, "signal": float(v)})
            # b is a strictly increasing transform of a => identical ranks.
            rows_b.append({"date": d, "id": aid, "signal": float(np.exp(v))})
    a = pl.DataFrame(rows_a)
    b = pl.DataFrame(rows_b)
    result = signal_pair_correlation({"a": a, "b": b}, min_obs=3)
    off = result.correlation_matrix.filter(pl.col("signal") == "a")["b"].item()
    assert abs(off - 1.0) < 1e-12
