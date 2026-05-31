"""Cross-component INVARIANT test suite.

Parametrized over a grid of GenSpecs, each test asserts a structural contract
that must hold regardless of data shape or random seed.  These are the pipeline
guarantees that were previously only checked against the single shared fixture.

GenSpec grid
------------
- ``standard``   : n_assets=40, n_dates=120, n_features=8, n_factors=4, seed=3
                   (same as the shared session fixture — baseline sanity)
- ``wide``        : n_assets=80, n_dates=60,  n_features=16, n_factors=6, seed=7
                   (more assets, fewer dates — stress portfolio / optimizer paths)
- ``tall``        : n_assets=20, n_dates=200, n_features=6,  n_factors=3, seed=11
                   (long time-series — stress splitter fold arithmetic)
- ``edge_small``  : n_assets=5,  n_dates=50,  n_features=4,  n_factors=2, seed=42
                   (tiny universe — exercises edge cases in IC, splitter, optimizer)
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from backtest import BacktestConfig, BacktestEngine, SignalFrame
from etl.datasets import (
    REGISTRY,
    GenSpec,
    _dates,
    _returns_matrix,
    gen_alpha_signals,
    generate,
)
from models import (
    PurgedEmbargoCVSplitter,
    WalkForwardSplitter,
    build_panel,
)
from portfolio import ConstraintSpec, mean_variance
from signals.ic import ic_series_v2

# ---------------------------------------------------------------------------
# GenSpec grid
# ---------------------------------------------------------------------------

_SPECS: list[tuple[str, GenSpec]] = [
    ("standard", GenSpec(n_assets=40, n_dates=120, n_features=8, n_factors=4, seed=3)),
    ("wide", GenSpec(n_assets=80, n_dates=60, n_features=16, n_factors=6, seed=7)),
    ("tall", GenSpec(n_assets=20, n_dates=200, n_features=6, n_factors=3, seed=11)),
    ("edge_small", GenSpec(n_assets=5, n_dates=50, n_features=4, n_factors=2, seed=42)),
]

_IDS = [label for label, _ in _SPECS]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Inject ``spec_id`` + ``spec`` for every test that declares them."""
    if "spec" in metafunc.fixturenames:
        metafunc.parametrize(
            ("spec_id", "spec"),
            [(label, s) for label, s in _SPECS],
            ids=_IDS,
        )


# ---------------------------------------------------------------------------
# INVARIANT 1 — Return / price reconciliation
#
# close[t] / close[t-1] - 1  ==  return[t] / 100   (within float tol)
#
# The generator computes close as base * cumprod(1 + R) where R is the
# _returns_matrix / 100.  We verify the link is exact (not just approximate).
# ---------------------------------------------------------------------------


def test_return_price_reconciliation(spec_id: str, spec: GenSpec) -> None:
    """close[t]/close[t-1] - 1 == return[t]/100 for all (date, id) pairs."""
    prices = generate("prices", spec)

    # Implied return from consecutive closes, per asset, sorted (id, date)
    implied = (
        prices.sort(["id", "date"])
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("id") - 1.0).alias("implied_ret")
        )
        .filter(pl.col("implied_ret").is_not_null())
    )

    # _returns_matrix(spec) is (n_dates, n_assets) in percent units;
    # close[t] = base[a] * prod(1 + R[:t+1, a] / 100), so implied_ret[t, a] = R[t, a] / 100
    # for t >= 1.  Skip t=0 (no prior close to compute implied return).
    R = _returns_matrix(spec)  # (n_dates, n_assets) — percent returns
    # Flatten to (na, nd-1) then reshape to 1-D, matching implied sorted by (id, date)
    expected_long = R[1:, :].T.reshape(-1) / 100.0  # id-major order, skip t=0

    actual = implied.sort(["id", "date"])["implied_ret"].to_numpy()

    np.testing.assert_allclose(
        actual,
        expected_long,
        rtol=1e-9,
        atol=1e-12,
        err_msg=f"[{spec_id}] price/return reconciliation failed",
    )


# ---------------------------------------------------------------------------
# INVARIANT 2 — Session axis
#
# All (date, id) panels live on a business-day axis: no Saturday or Sunday.
# polars dt.weekday(): Mon=1, Tue=2, ..., Fri=5, Sat=6, Sun=7.
# ---------------------------------------------------------------------------

_DATE_BEARING_DATASETS: list[str] = [
    "prices",
    "universe_mask",
    "borrow_rates",
    "transaction_costs",
    "specific_risk",
    "factor_loadings",
    "factor_returns",
    "feature_panel",
    "forward_returns",
    "alpha_signals",
    "sample_weights",
    "risk_free_rate",
    "benchmark_returns",
    "benchmark_weights",
    "trading_calendar",
]


@pytest.mark.parametrize("dataset_name", _DATE_BEARING_DATASETS)
def test_session_axis_no_weekends(spec_id: str, spec: GenSpec, dataset_name: str) -> None:
    """Generated panels must only contain Mon–Fri dates (no weekends)."""
    if dataset_name not in REGISTRY:
        pytest.skip(f"dataset {dataset_name!r} not in REGISTRY")
    df = generate(dataset_name, spec)
    date_col = "date" if "date" in df.columns else "ex_date"
    weekdays = df[date_col].dt.weekday()
    n_weekend = int((weekdays > 5).sum())
    assert n_weekend == 0, f"[{spec_id}] {dataset_name}: found {n_weekend} weekend dates"


# ---------------------------------------------------------------------------
# INVARIANT 3 — Schema validity
#
# Every dataset in REGISTRY validates against its declared schema without
# error for every GenSpec in the grid.  The feature_panel schema depends on
# n_features, so this catches shape-dependent schema regressions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_name", list(REGISTRY.keys()))
def test_schema_validity(spec_id: str, spec: GenSpec, dataset_name: str) -> None:
    """Every REGISTRY dataset validates cleanly for each GenSpec in the grid."""
    # generate() calls schema.validate() when validate=True (the default)
    df = generate(dataset_name, spec, validate=True)
    assert df.height > 0, f"[{spec_id}] {dataset_name}: generated an empty DataFrame"


# ---------------------------------------------------------------------------
# INVARIANT 4 — Splitter invariants
#
# For valid (n_splits, embargo) combos, walk-forward folds satisfy:
#   - Train and test index sets are disjoint.
#   - All train ordinals < min(test ordinals) - embargo_periods.
#   - Test windows across folds are non-overlapping (ordinal-level).
# ---------------------------------------------------------------------------


def _panel_groups(spec: GenSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_dummy, groups) for the feature_panel + fwd_ret_1 panel."""
    panel = build_panel(
        generate("feature_panel", spec, validate=False),
        generate("forward_returns", spec, validate=False),
        "fwd_ret_1",
    )
    return np.zeros((len(panel.groups), 1)), panel.groups


_SPLITTER_CASES: list[tuple[str, int, int]] = [
    ("wf_2_0", 2, 0),
    ("wf_3_2", 3, 2),
    ("wf_4_5", 4, 5),
    ("purged_3_0", 3, 0),
    ("purged_4_3", 4, 3),
]


@pytest.mark.parametrize(
    ("splitter_id", "n_splits", "embargo"),
    [(s, n, e) for s, n, e in _SPLITTER_CASES],
    ids=[s for s, _, _ in _SPLITTER_CASES],
)
def test_splitter_no_overlap_and_temporal_order(
    spec_id: str,
    spec: GenSpec,
    splitter_id: str,
    n_splits: int,
    embargo: int,
) -> None:
    """Walk-forward folds: no train/test overlap; all train before test minus embargo."""
    X_dummy, groups = _panel_groups(spec)
    n_unique = len(np.unique(groups))

    if splitter_id.startswith("wf"):
        min_train = max(1, n_splits)
        if n_unique < min_train + n_splits * (1 + embargo):
            pytest.skip(
                f"[{spec_id}] too few unique dates ({n_unique}) for "
                f"n_splits={n_splits}, embargo={embargo}"
            )
        splitter = WalkForwardSplitter(
            n_splits=n_splits,
            embargo_periods=embargo,
            min_train_periods=min_train,
        )
    else:
        if n_unique < n_splits * 2:
            pytest.skip(
                f"[{spec_id}] too few unique dates ({n_unique}) for "
                f"PurgedEmbargoCV n_splits={n_splits}"
            )
        splitter = PurgedEmbargoCVSplitter(n_splits=n_splits, embargo_periods=embargo)

    fold_test_ordinals: list[np.ndarray] = []
    for train_idx, test_idx in splitter.split(X_dummy, groups=groups):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        train_ord = groups[train_idx]
        test_ord = groups[test_idx]

        # Train and test index sets must be disjoint
        overlap = np.intersect1d(train_idx, test_idx)
        assert len(overlap) == 0, (
            f"[{spec_id}/{splitter_id}] train/test index overlap: {len(overlap)} rows"
        )

        # All train ordinals must be strictly before test_start minus embargo
        effective_test_start = int(test_ord.min()) - embargo
        assert int(train_ord.max()) < effective_test_start, (
            f"[{spec_id}/{splitter_id}] train bleeds into test window: "
            f"max_train={train_ord.max()}, effective_test_start={effective_test_start}"
        )

        fold_test_ordinals.append(test_ord)

    # Test windows across folds must be non-overlapping
    for i, ti in enumerate(fold_test_ordinals):
        for j, tj in enumerate(fold_test_ordinals):
            if i >= j:
                continue
            shared = np.intersect1d(ti, tj)
            assert len(shared) == 0, (
                f"[{spec_id}/{splitter_id}] fold {i} and fold {j} share {len(shared)} test ordinals"
            )


# ---------------------------------------------------------------------------
# INVARIANT 5 — Signal IC bounds
#
# The Spearman rank IC of any signal against fwd_ret_1 must be in [-1, 1]
# and finite on every date where enough observations exist.
# ---------------------------------------------------------------------------


def test_signal_ic_bounds(spec_id: str, spec: GenSpec) -> None:
    """IC values ∈ [-1, 1] and finite for all valid dates."""
    signals = generate("alpha_signals", spec, validate=False)
    fwd = generate("forward_returns", spec, validate=False)

    first_name = signals["signal_name"].cast(pl.String).unique().sort()[0]
    sig_single = signals.filter(pl.col("signal_name").cast(pl.String) == first_name).select(
        "date", "id", "signal"
    )

    ic_df = ic_series_v2(sig_single, fwd, return_col="fwd_ret_1", min_obs=2)

    valid = ic_df.filter(pl.col("ic").is_not_null())
    if valid.height == 0:
        pytest.skip(f"[{spec_id}] no valid IC rows — universe too small for min_obs=2")

    ic_vals = valid["ic"].to_numpy()
    assert np.all(np.isfinite(ic_vals)), (
        f"[{spec_id}] non-finite IC values: {ic_vals[~np.isfinite(ic_vals)]}"
    )
    assert np.all(ic_vals >= -1.0 - 1e-9), f"[{spec_id}] IC below -1: min={ic_vals.min()}"
    assert np.all(ic_vals <= 1.0 + 1e-9), f"[{spec_id}] IC above +1: max={ic_vals.max()}"


# ---------------------------------------------------------------------------
# INVARIANT 6 — Optimizer invariants
#
# mean_variance weights must:
#   (a) sum to net_exposure within solver tolerance
#   (b) respect per-asset position bounds
# ---------------------------------------------------------------------------

_OPT_CASES: list[tuple[str, float, bool, float, float]] = [
    ("long_only_1.0", 1.0, True, 0.0, 0.5),
    ("long_short_0.0", 0.0, False, -0.2, 0.2),
    ("long_short_1.0", 1.0, False, -0.1, 0.3),
    ("tight_bounds", 1.0, True, 0.0, 0.1),
]


@pytest.mark.parametrize(
    ("opt_id", "net_exp", "long_only", "min_w", "max_w"),
    [(o, n, lo, mi, mx) for o, n, lo, mi, mx in _OPT_CASES],
    ids=[o for o, *_ in _OPT_CASES],
)
def test_optimizer_invariants(
    spec_id: str,
    spec: GenSpec,
    opt_id: str,
    net_exp: float,
    long_only: bool,
    min_w: float,
    max_w: float,
) -> None:
    """Optimizer weights sum to net_exposure and respect position bounds."""
    n = spec.n_assets
    rng = np.random.default_rng(spec.seed + 100)
    alpha_vec = rng.normal(0.0, 1.0, n)

    # Diagonal covariance avoids the full factor-model setup while still being PSD
    vol = rng.uniform(0.1, 0.3, n)
    cov = np.diag(vol**2)

    cspec = ConstraintSpec(
        n_assets=n,
        long_only=long_only,
        min_weight=min_w,
        max_weight=max_w,
        net_exposure=net_exp,
    )

    # Skip infeasible configurations rather than expecting solver failure
    max_achievable = n * max_w
    min_achievable = n * (max(min_w, 0.0) if long_only else min_w)
    if net_exp > max_achievable + 1e-6 or net_exp < min_achievable - 1e-6:
        pytest.skip(
            f"[{spec_id}/{opt_id}] net_exp={net_exp} infeasible: "
            f"bounds [{min_achievable:.3f}, {max_achievable:.3f}]"
        )

    result = mean_variance(alpha_vec, cov, cspec, risk_aversion=1.0, max_iter=1000)

    weight_sum = float(result.weights.sum())
    assert abs(weight_sum - net_exp) < 1e-4, (
        f"[{spec_id}/{opt_id}] weights.sum()={weight_sum:.6f} != net_exp={net_exp}"
    )

    bounds = cspec.per_asset_bounds()
    lo_arr = np.array([b[0] for b in bounds])
    hi_arr = np.array([b[1] for b in bounds])
    assert np.all(result.weights >= lo_arr - 1e-6), (
        f"[{spec_id}/{opt_id}] weight below lower bound; "
        f"max violation: {(lo_arr - result.weights).max():.2e}"
    )
    assert np.all(result.weights <= hi_arr + 1e-6), (
        f"[{spec_id}/{opt_id}] weight above upper bound; "
        f"max violation: {(result.weights - hi_arr).max():.2e}"
    )


# ---------------------------------------------------------------------------
# INVARIANT 7 — Panel alignment (no look-ahead via build_panel)
#
# For fwd_ret_1 at date t, the realization window uses t+1, so the last
# session date always has a null forward return and must be dropped by
# build_panel.  After the inner join + null-strip, no sample may have its
# feature_date equal to the last session date.  Additionally, all panel.y
# values must be finite (null/NaN rows were stripped).
# ---------------------------------------------------------------------------


def test_panel_alignment_no_lookahead(spec_id: str, spec: GenSpec) -> None:
    """build_panel strips null-target rows; last session date absent from panel."""
    features = generate("feature_panel", spec, validate=False)
    fwd = generate("forward_returns", spec, validate=False)

    panel = build_panel(features, fwd, "fwd_ret_1")

    all_dates = sorted(features["date"].unique().to_list())
    last_date = all_dates[-1]

    panel_dates = set(panel.dates.tolist())
    assert last_date not in panel_dates, (
        f"[{spec_id}] last date {last_date} appears in panel — look-ahead possible"
    )

    assert np.all(np.isfinite(panel.y)), (
        f"[{spec_id}] panel.y contains non-finite values after null-strip"
    )


# ---------------------------------------------------------------------------
# INVARIANT 8 — Backtest sanity (NAV finite and strictly positive)
#
# BacktestResult.nav_history["nav"] must be finite and > 0 throughout the
# entire run.  Uses BacktestEngine (simple, no costs) to avoid production-
# engine setup complexity while still exercising the core accounting loop.
# ---------------------------------------------------------------------------


def test_backtest_nav_finite_positive(spec_id: str, spec: GenSpec) -> None:
    """NAV is finite and strictly positive on every bar."""
    R_mat = _returns_matrix(spec)  # (n_dates, n_assets) percent returns
    nd, na = R_mat.shape
    dates_series = _dates(spec)

    date_col: list = []
    id_col: list[int] = []
    ret_col: list[float] = []
    for t in range(nd):
        for a in range(na):
            date_col.append(dates_series[t])
            id_col.append(a)
            ret_col.append(float(R_mat[t, a]))
    returns_df = pl.DataFrame({"date": date_col, "id": id_col, "return": ret_col})

    signals_df = gen_alpha_signals(spec)
    first_name = signals_df["signal_name"].cast(pl.String).unique().sort()[0]
    sig_df = signals_df.filter(pl.col("signal_name").cast(pl.String) == first_name).select(
        "date", "id", "signal"
    )

    cfg = BacktestConfig(n_assets=na, n_dates=nd, initial_cash=1_000_000.0)
    result = BacktestEngine(cfg).run(returns_df, SignalFrame(df=sig_df, is_categorical=False))

    nav_arr = result.nav_history["nav"].to_numpy()

    assert np.all(np.isfinite(nav_arr)), f"[{spec_id}] NAV contains non-finite values"
    assert np.all(nav_arr > 0.0), (
        f"[{spec_id}] NAV contains non-positive values; min={nav_arr.min():.6f}"
    )
    assert result.nav_history.height == nd, (
        f"[{spec_id}] nav_history height={result.nav_history.height} != n_dates={nd}"
    )
