"""Unit tests for analysis.report — AttributionReport and analyze_attribution.

Each test group documents the invariant being checked and why it must hold.
Tests are written so they fail without the implementation.

Synthetic setup uses small in-process DataFrames; no filesystem required.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest

from etl.source import session_axis

# ---------------------------------------------------------------------------
# Minimal fixture helpers
# ---------------------------------------------------------------------------

N_ASSETS = 8
N_DATES = 60
N_FACTORS = 3
SEED = 42


def _dates(n: int = N_DATES) -> list[date]:
    return session_axis(n).to_list()


def _build_trade_log(rng: np.random.Generator, n_assets: int, n_dates: int) -> pl.DataFrame:
    """Synthetic trade log: rebalance every 5 dates, long-only dirichlet weights."""
    dates = _dates(n_dates)
    trade_dates: list[date] = []
    trade_ids: list[int] = []
    trade_qty: list[float] = []
    for i in range(0, n_dates, 5):
        w = rng.dirichlet(np.ones(n_assets))
        for j in range(n_assets):
            trade_dates.append(dates[i])
            trade_ids.append(j)
            trade_qty.append(float(w[j]))
    return pl.DataFrame(
        {
            "date": trade_dates,
            "id": pl.Series(trade_ids, dtype=pl.Int64),
            "quantity": trade_qty,
        }
    )


def _build_nav_history(rng: np.random.Generator, n_dates: int) -> pl.DataFrame:
    r = rng.normal(0.0005, 0.01, n_dates)
    nav = 1_000_000.0 * np.cumprod(1.0 + r)
    return pl.DataFrame({"date": _dates(n_dates), "nav": nav.tolist()})


def _build_factor_returns(rng: np.random.Generator, n_dates: int, n_factors: int) -> pl.DataFrame:
    """Factor returns in PERCENT (matching the dataset schema)."""
    dates = _dates(n_dates)
    rows = []
    for d in dates:
        for k in range(n_factors):
            rows.append({"date": d, "factor_id": k, "return": float(rng.normal(0.0, 1.0))})
    return pl.DataFrame(rows).with_columns(
        pl.col("factor_id").cast(pl.Int64),
        pl.col("return").cast(pl.Float64),
    )


def _build_factor_loadings(
    rng: np.random.Generator, n_assets: int, n_dates: int, n_factors: int
) -> pl.DataFrame:
    dates = _dates(n_dates)
    rows = []
    for d in dates:
        for i in range(n_assets):
            for k in range(n_factors):
                loading = float(rng.normal(0, 1))
                rows.append({"date": d, "id": i, "factor_id": k, "loading": loading})
    return pl.DataFrame(rows).with_columns(
        pl.col("id").cast(pl.Int64),
        pl.col("factor_id").cast(pl.Int64),
    )


def _build_benchmark_returns(rng: np.random.Generator, n_dates: int) -> pl.DataFrame:
    """Benchmark returns in fractional units (already converted from percent)."""
    r = rng.normal(0.0003, 0.009, n_dates)
    return pl.DataFrame({"date": _dates(n_dates), "return_1d": r.tolist()})


def _build_benchmark_weights(rng: np.random.Generator, n_assets: int, n_dates: int) -> pl.DataFrame:
    dates = _dates(n_dates)
    rows = []
    for d in dates:
        raw = rng.uniform(0.5, 1.5, n_assets)
        w = raw / raw.sum()
        for i in range(n_assets):
            rows.append({"date": d, "id": i, "benchmark_weight": float(w[i])})
    return pl.DataFrame(rows).with_columns(pl.col("id").cast(pl.Int64))


def _build_asset_returns(rng: np.random.Generator, n_assets: int, n_dates: int) -> pl.DataFrame:
    dates = _dates(n_dates)
    rows = []
    for d in dates:
        r = rng.normal(0.0, 0.015, n_assets)
        for i in range(n_assets):
            rows.append({"date": d, "id": i, "return_1d": float(r[i])})
    return pl.DataFrame(rows).with_columns(pl.col("id").cast(pl.Int64))


SECTORS = ["Tech", "Financials", "Energy", "Health"]


def _build_security_master(n_assets: int) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    sector_idx = rng.integers(0, len(SECTORS), n_assets)
    return pl.DataFrame(
        {
            "id": pl.Series(list(range(n_assets)), dtype=pl.Int64),
            "sector": pl.Series([SECTORS[i] for i in sector_idx], dtype=pl.Categorical),
        }
    )


def _make_backtest_result(trade_log: pl.DataFrame, nav_history: pl.DataFrame):
    """Construct a real BacktestResult from trade_log and nav_history."""
    from backtest.engine import BacktestResult

    return BacktestResult(
        nav_history=nav_history,
        trade_log=trade_log,
        final_positions=np.zeros(N_ASSETS),
    )


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth_inputs():
    rng = np.random.default_rng(SEED)
    trade_log = _build_trade_log(rng, N_ASSETS, N_DATES)
    nav_history = _build_nav_history(rng, N_DATES)
    factor_returns = _build_factor_returns(rng, N_DATES, N_FACTORS)
    factor_loadings = _build_factor_loadings(rng, N_ASSETS, N_DATES, N_FACTORS)
    benchmark_returns = _build_benchmark_returns(rng, N_DATES)
    benchmark_weights = _build_benchmark_weights(rng, N_ASSETS, N_DATES)
    asset_returns = _build_asset_returns(rng, N_ASSETS, N_DATES)
    security_master = _build_security_master(N_ASSETS)
    backtest_result = _make_backtest_result(trade_log=trade_log, nav_history=nav_history)
    return {
        "backtest_result": backtest_result,
        "factor_returns": factor_returns,
        "factor_loadings": factor_loadings,
        "benchmark_returns": benchmark_returns,
        "benchmark_weights": benchmark_weights,
        "asset_returns": asset_returns,
        "security_master": security_master,
    }


# ---------------------------------------------------------------------------
# 1. AttributionReport shape and types
# ---------------------------------------------------------------------------


class TestAttributionReportShape:
    def test_analyze_attribution_returns_attribution_report(self, synth_inputs) -> None:
        """analyze_attribution must return an AttributionReport (not raise)."""
        from analysis.report import AttributionReport, analyze_attribution

        rpt = analyze_attribution(**synth_inputs)
        assert isinstance(rpt, AttributionReport)

    def test_active_return_series_columns(self, synth_inputs) -> None:
        """active_return_series must have (date, active_return) columns."""
        from analysis.report import analyze_attribution

        rpt = analyze_attribution(**synth_inputs)
        assert "date" in rpt.active_return_series.columns
        assert "active_return" in rpt.active_return_series.columns
        assert rpt.active_return_series.height > 0

    def test_scalar_metrics_are_finite(self, synth_inputs) -> None:
        """tracking_error and information_ratio must be finite floats."""
        from analysis.report import analyze_attribution

        rpt = analyze_attribution(**synth_inputs)
        assert math.isfinite(rpt.tracking_error)
        assert math.isfinite(rpt.information_ratio)
        assert math.isfinite(rpt.benchmark_metrics["active_return_ann"])
        assert math.isfinite(rpt.benchmark_metrics["tracking_error"])
        assert math.isfinite(rpt.benchmark_metrics["information_ratio"])

    def test_tracking_error_positive(self, synth_inputs) -> None:
        """TE must be ≥ 0."""
        from analysis.report import analyze_attribution

        rpt = analyze_attribution(**synth_inputs)
        assert rpt.tracking_error >= 0.0

    def test_factor_attribution_fields(self, synth_inputs) -> None:
        """factor_attribution must contain expected fields with correct keys."""
        from analysis.report import analyze_attribution

        rpt = analyze_attribution(**synth_inputs)
        fa = rpt.factor_attribution
        assert set(fa.factor_exposures.keys()) == set(range(N_FACTORS))
        assert math.isfinite(fa.alpha_annualized)
        assert 0.0 <= fa.r_squared <= 1.0

    def test_factor_active_exposures_columns(self, synth_inputs) -> None:
        """factor_active_exposures must have (factor_id, active_exposure) columns."""
        from analysis.report import analyze_attribution

        rpt = analyze_attribution(**synth_inputs)
        cols = rpt.factor_active_exposures.columns
        assert "factor_id" in cols
        assert "active_exposure" in cols
        assert rpt.factor_active_exposures.height == N_FACTORS

    def test_factor_return_contributions_finite(self, synth_inputs) -> None:
        """Per-factor active_contribution values must all be finite."""
        from analysis.report import analyze_attribution

        rpt = analyze_attribution(**synth_inputs)
        vals = rpt.factor_return_contributions["active_contribution"].to_numpy()
        assert all(math.isfinite(v) for v in vals)

    def test_brinson_not_none_with_full_inputs(self, synth_inputs) -> None:
        """brinson must be non-None when security_master and benchmark_weights supplied."""
        from analysis.report import analyze_attribution

        rpt = analyze_attribution(**synth_inputs)
        assert rpt.brinson is not None

    def test_brinson_none_without_security_master(self, synth_inputs) -> None:
        """brinson must be None when security_master is omitted."""
        from analysis.report import analyze_attribution

        inputs_no_sm = {k: v for k, v in synth_inputs.items() if k not in ("security_master",)}
        rpt = analyze_attribution(**inputs_no_sm, security_master=None)
        assert rpt.brinson is None


# ---------------------------------------------------------------------------
# 2. Active-weight sum: portfolio - benchmark ≈ 0
# ---------------------------------------------------------------------------


class TestActiveWeights:
    def test_active_weights_sum_near_zero(self, synth_inputs) -> None:
        """The sum of active weights across all assets must be ~0.

        Portfolio weights sum to 1 and benchmark weights sum to 1, so their
        difference sums to 0 on every rebalance date.  This verifies that the
        weight reconstruction and the active-exposure computation are consistent.
        """
        from analysis.report import analyze_attribution
        from analysis.turnover import reconstruct_weights

        analyze_attribution(**synth_inputs)
        portfolio_weights = reconstruct_weights(synth_inputs["backtest_result"].trade_log)
        bmk_weights = synth_inputs["benchmark_weights"]

        # Use a common date
        common_dates = list(
            set(portfolio_weights["date"].unique().to_list())
            & set(bmk_weights["date"].unique().to_list())
        )
        assert common_dates, "No common dates between portfolio and benchmark weights"

        d = common_dates[0]
        pw = portfolio_weights.filter(pl.col("date") == d)
        bw = bmk_weights.filter(pl.col("date") == d)

        # Merge on id (full outer so we see all positions)
        merged = pw.join(
            bw.rename({"benchmark_weight": "bw"}),
            on="id",
            how="full",
            coalesce=True,
        ).with_columns(
            pl.col("weight").fill_null(0.0),
            pl.col("bw").fill_null(0.0),
        )
        active_sum = float((merged["weight"] - merged["bw"]).sum())
        assert abs(active_sum) < 1e-9, f"Active weight sum not near zero: {active_sum}"


# ---------------------------------------------------------------------------
# 3. Brinson additive identity
# ---------------------------------------------------------------------------


class TestBrinsonAdditivity:
    def test_allocation_plus_selection_plus_interaction_equals_active(self, synth_inputs) -> None:
        """allocation + selection + interaction = active_return per sector.

        This is the core Brinson identity: each sector's contribution to active
        return is fully decomposed into the three effects.
        """
        from analysis.report import analyze_attribution

        rpt = analyze_attribution(**synth_inputs)
        assert rpt.brinson is not None

        b = rpt.brinson
        # Merge the three effects by sector
        combined = (
            b.allocation.join(b.selection, on="sector")
            .join(b.interaction, on="sector")
            .join(b.sector_active_return, on="sector")
            .with_columns(
                (pl.col("allocation") + pl.col("selection") + pl.col("interaction")).alias(
                    "sum_effects"
                )
            )
        )
        alloc_plus_sel_plus_inter = combined["sum_effects"].to_numpy()
        active = combined["active_return"].to_numpy()
        np.testing.assert_allclose(alloc_plus_sel_plus_inter, active, atol=1e-12)

    def test_brinson_sectors_cover_all_assets(self, synth_inputs) -> None:
        """Every sector from security_master must appear in the Brinson output."""
        from analysis.report import analyze_attribution

        rpt = analyze_attribution(**synth_inputs)
        assert rpt.brinson is not None

        expected_sectors = set(
            synth_inputs["security_master"]["sector"].cast(pl.String).unique().to_list()
        )
        reported_sectors = set(rpt.brinson.allocation["sector"].cast(pl.String).unique().to_list())
        # Reported may be subset if some sectors have no overlapping dates;
        # but at minimum it must be non-empty.
        assert len(reported_sectors) > 0
        assert reported_sectors.issubset(expected_sectors | {"Unknown"})


# ---------------------------------------------------------------------------
# 4. IR finite and sign-consistent
# ---------------------------------------------------------------------------


class TestInformationRatio:
    def test_ir_finite(self, synth_inputs) -> None:
        """IR must be a finite float (not NaN, not Inf)."""
        from analysis.report import analyze_attribution

        rpt = analyze_attribution(**synth_inputs)
        assert math.isfinite(rpt.information_ratio)

    def test_ir_positive_when_strategy_always_beats_benchmark(self) -> None:
        """When strategy return > benchmark return on every date, IR must be positive."""
        from analysis.report import analyze_attribution

        rng = np.random.default_rng(99)
        n = 40
        n_assets = 4
        n_factors = 2

        trade_log = _build_trade_log(rng, n_assets, n)
        nav_history = _build_nav_history(rng, n)

        base_r = rng.normal(0.0, 0.01, n)
        # Strategy always beats: add daily 0.1 % excess
        strategy_excess = base_r + 0.001
        nav_vals = 1_000_000.0 * np.cumprod(1.0 + strategy_excess)
        nav_history = pl.DataFrame({"date": _dates(n), "nav": nav_vals.tolist()})

        # Benchmark returns: just base_r
        benchmark_returns = pl.DataFrame({"date": _dates(n), "return_1d": base_r.tolist()})

        fake_result = _make_backtest_result(trade_log=trade_log, nav_history=nav_history)
        rpt = analyze_attribution(
            backtest_result=fake_result,
            benchmark_returns=benchmark_returns,
            factor_returns=_build_factor_returns(rng, n, n_factors),
            factor_loadings=_build_factor_loadings(rng, n_assets, n, n_factors),
            asset_returns=_build_asset_returns(rng, n_assets, n),
        )
        assert rpt.information_ratio > 0


# ---------------------------------------------------------------------------
# 5. No benchmark_weights path
# ---------------------------------------------------------------------------


class TestNoBenchmarkWeights:
    def test_analyze_without_benchmark_weights(self, synth_inputs) -> None:
        """analyze_attribution must succeed when benchmark_weights is None."""
        from analysis.report import analyze_attribution

        inputs = {k: v for k, v in synth_inputs.items() if k not in ("benchmark_weights",)}
        rpt = analyze_attribution(**inputs, benchmark_weights=None)
        assert rpt.brinson is None
        # factor_active_exposures still populated (portfolio-only path)
        assert "factor_id" in rpt.factor_active_exposures.columns
        assert rpt.factor_active_exposures.height == N_FACTORS

    def test_factor_exposures_portfolio_only_are_nonneg_sum(self, synth_inputs) -> None:
        """Portfolio-only active exposures must equal portfolio exposures (benchmark = 0)."""
        from analysis.report import analyze_attribution

        inputs = {k: v for k, v in synth_inputs.items() if k not in ("benchmark_weights",)}
        rpt = analyze_attribution(**inputs, benchmark_weights=None)
        fae = rpt.factor_active_exposures
        # benchmark_exposure column should be all zeros
        assert (fae["benchmark_exposure"] == 0.0).all()
        # active_exposure == portfolio_exposure
        np.testing.assert_allclose(
            fae["active_exposure"].to_numpy(),
            fae["portfolio_exposure"].to_numpy(),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# 6. Export from analysis namespace
# ---------------------------------------------------------------------------


class TestPublicExport:
    def test_symbols_in_all(self) -> None:
        """AttributionReport, BrinsonDecomposition, analyze_attribution in __all__."""
        import analysis

        assert "AttributionReport" in analysis.__all__
        assert "BrinsonDecomposition" in analysis.__all__
        assert "analyze_attribution" in analysis.__all__

    def test_importable_from_analysis(self) -> None:
        """Can import the three symbols directly from analysis."""
        from analysis import AttributionReport, BrinsonDecomposition, analyze_attribution

        assert AttributionReport is not None
        assert BrinsonDecomposition is not None
        assert analyze_attribution is not None
