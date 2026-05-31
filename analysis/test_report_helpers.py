"""Unit tests for the extracted private helpers in analysis.report.

These tests pin the behavior of each sub-function extracted from the two large
functions (_brinson_decomposition, analyze_attribution) so regressions are
caught at the helper level, not only through end-to-end tests.

Each class targets one extracted helper and documents the invariant being
checked and why it must hold.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Minimal DataFrame builders
# ---------------------------------------------------------------------------


def _make_sector_map(ids: list[int], sectors: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": pl.Series(ids, dtype=pl.Int64),
            "sector": pl.Series(sectors, dtype=pl.Categorical),
        }
    )


def _make_portfolio_weights(
    dates: list[date], ids: list[int], weights: list[float]
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": dates,
            "id": pl.Series(ids, dtype=pl.Int64),
            "weight": pl.Series(weights, dtype=pl.Float64),
        }
    )


def _make_benchmark_weights(
    dates: list[date], ids: list[int], weights: list[float]
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": dates,
            "id": pl.Series(ids, dtype=pl.Int64),
            "benchmark_weight": pl.Series(weights, dtype=pl.Float64),
        }
    )


def _make_asset_returns(dates: list[date], ids: list[int], returns: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": dates,
            "id": pl.Series(ids, dtype=pl.Int64),
            "return_1d": pl.Series(returns, dtype=pl.Float64),
        }
    )


D1 = date(2020, 1, 2)
D2 = date(2020, 1, 3)


# ---------------------------------------------------------------------------
# _empty_brinson_decomposition
# ---------------------------------------------------------------------------


class TestEmptyBrinsonDecomposition:
    def test_returns_brinson_decomposition(self) -> None:
        """_empty_brinson_decomposition must return a BrinsonDecomposition."""
        from analysis.report import BrinsonDecomposition, _empty_brinson_decomposition

        result = _empty_brinson_decomposition()
        assert isinstance(result, BrinsonDecomposition)

    def test_all_fields_empty(self) -> None:
        """Every frame in the empty decomposition must have zero rows."""
        from analysis.report import _empty_brinson_decomposition

        result = _empty_brinson_decomposition()
        assert result.allocation.height == 0
        assert result.selection.height == 0
        assert result.interaction.height == 0
        assert result.sector_active_return.height == 0

    def test_schema_has_sector_column(self) -> None:
        """Each empty frame must still carry the 'sector' column."""
        from analysis.report import _empty_brinson_decomposition

        result = _empty_brinson_decomposition()
        all_frames = (
            result.allocation,
            result.selection,
            result.interaction,
            result.sector_active_return,
        )
        for df in all_frames:
            assert "sector" in df.columns


# ---------------------------------------------------------------------------
# _attach_sector_with_returns
# ---------------------------------------------------------------------------


class TestAttachSectorWithReturns:
    def test_sector_attached_correctly(self) -> None:
        """Sector column must be populated from the sector_map join."""
        from analysis.report import _attach_sector_with_returns

        pw = _make_portfolio_weights([D1, D1], [0, 1], [0.6, 0.4])
        sector_map = _make_sector_map([0, 1], ["Tech", "Energy"])
        ar = _make_asset_returns([D1, D1], [0, 1], [0.01, -0.02])

        result = _attach_sector_with_returns(pw, sector_map, ar, "weight")
        sectors = result["sector"].cast(pl.String).to_list()
        assert "Tech" in sectors
        assert "Energy" in sectors

    def test_unknown_sector_for_missing_id(self) -> None:
        """Assets not found in sector_map must get sector = 'Unknown'."""
        from analysis.report import _attach_sector_with_returns

        pw = _make_portfolio_weights([D1], [99], [1.0])
        sector_map = _make_sector_map([0, 1], ["Tech", "Energy"])
        ar = _make_asset_returns([D1], [99], [0.005])

        result = _attach_sector_with_returns(pw, sector_map, ar, "weight")
        assert result["sector"].cast(pl.String)[0] == "Unknown"

    def test_missing_return_filled_zero(self) -> None:
        """An asset with no return entry must have return_1d = 0.0."""
        from analysis.report import _attach_sector_with_returns

        pw = _make_portfolio_weights([D1], [0], [1.0])
        sector_map = _make_sector_map([0], ["Tech"])
        # Use a date that exists but a different id so the join misses
        ar = _make_asset_returns([D1], [99], [0.005])

        result = _attach_sector_with_returns(pw, sector_map, ar, "weight")
        assert result["return_1d"][0] == pytest.approx(0.0)

    def test_return_values_preserved(self) -> None:
        """Attached return_1d must match the input asset_returns exactly."""
        from analysis.report import _attach_sector_with_returns

        pw = _make_portfolio_weights([D1, D1], [0, 1], [0.5, 0.5])
        sector_map = _make_sector_map([0, 1], ["Tech", "Tech"])
        ar = _make_asset_returns([D1, D1], [0, 1], [0.03, -0.01])

        result = _attach_sector_with_returns(pw, sector_map, ar, "weight")
        # Sort by id for stable comparison
        result_sorted = result.sort("id")
        np.testing.assert_allclose(result_sorted["return_1d"].to_numpy(), [0.03, -0.01], atol=1e-12)


# ---------------------------------------------------------------------------
# _sector_weights_and_returns
# ---------------------------------------------------------------------------


class TestSectorWeightsAndReturns:
    def _build_portfolio_with_sector(self) -> pl.DataFrame:
        """Two assets in 'Tech', one in 'Energy' on a single date."""
        return pl.DataFrame(
            {
                "date": [D1, D1, D1],
                "id": pl.Series([0, 1, 2], dtype=pl.Int64),
                "sector": pl.Series(["Tech", "Tech", "Energy"], dtype=pl.Categorical),
                "weight": [0.4, 0.3, 0.3],
                "return_1d": [0.02, -0.01, 0.04],
            }
        )

    def test_sector_weight_sums(self) -> None:
        """w_{prefix}_s must equal the sum of asset weights within each sector."""
        from analysis.report import _sector_weights_and_returns

        df = self._build_portfolio_with_sector()
        result = _sector_weights_and_returns(df, "weight", "p")
        tech = result.filter(pl.col("sector").cast(pl.String) == "Tech")
        energy = result.filter(pl.col("sector").cast(pl.String) == "Energy")
        assert tech["w_p_s"][0] == pytest.approx(0.7, abs=1e-12)
        assert energy["w_p_s"][0] == pytest.approx(0.3, abs=1e-12)

    def test_sector_return_is_weighted_average(self) -> None:
        """R_{prefix}_s must equal sum(w_i * r_i) / w_s for each sector."""
        from analysis.report import _sector_weights_and_returns

        df = self._build_portfolio_with_sector()
        result = _sector_weights_and_returns(df, "weight", "p")
        # Tech: (0.4*0.02 + 0.3*(-0.01)) / 0.7
        expected_tech = (0.4 * 0.02 + 0.3 * (-0.01)) / 0.7
        tech = result.filter(pl.col("sector").cast(pl.String) == "Tech")
        assert tech["R_p_s"][0] == pytest.approx(expected_tech, abs=1e-12)

    def test_output_columns_present(self) -> None:
        """Output must have (date, sector, w_p_s, R_p_s) columns."""
        from analysis.report import _sector_weights_and_returns

        df = self._build_portfolio_with_sector()
        result = _sector_weights_and_returns(df, "weight", "p")
        assert "date" in result.columns
        assert "sector" in result.columns
        assert "w_p_s" in result.columns
        assert "R_p_s" in result.columns

    def test_raw_column_dropped(self) -> None:
        """The intermediate R_raw column must not appear in the output."""
        from analysis.report import _sector_weights_and_returns

        df = self._build_portfolio_with_sector()
        result = _sector_weights_and_returns(df, "weight", "p")
        assert "R_p_s_raw" not in result.columns


# ---------------------------------------------------------------------------
# _compute_brinson_effects
# ---------------------------------------------------------------------------


class TestComputeBrinsonEffects:
    def _build_sector_df(self) -> pl.DataFrame:
        """Minimal sector_df with one date and two sectors."""
        return pl.DataFrame(
            {
                "date": [D1, D1],
                "sector": pl.Series(["Tech", "Energy"], dtype=pl.Categorical),
                "w_p_s": [0.6, 0.4],
                "R_p_s": [0.02, 0.01],
                "w_b_s": [0.5, 0.5],
                "R_b_s": [0.015, 0.012],
                "R_b": [0.0135, 0.0135],  # total benchmark return (same for all sectors on date)
            }
        )

    def test_allocation_formula(self) -> None:
        """allocation = (w_p_s - w_b_s) * (R_b_s - R_b)."""
        from analysis.report import _compute_brinson_effects

        df = self._build_sector_df()
        result = _compute_brinson_effects(df)
        tech = result.filter(pl.col("sector").cast(pl.String) == "Tech")
        expected = (0.6 - 0.5) * (0.015 - 0.0135)
        assert tech["allocation"][0] == pytest.approx(expected, abs=1e-12)

    def test_selection_formula(self) -> None:
        """selection = w_b_s * (R_p_s - R_b_s)."""
        from analysis.report import _compute_brinson_effects

        df = self._build_sector_df()
        result = _compute_brinson_effects(df)
        tech = result.filter(pl.col("sector").cast(pl.String) == "Tech")
        expected = 0.5 * (0.02 - 0.015)
        assert tech["selection"][0] == pytest.approx(expected, abs=1e-12)

    def test_interaction_formula(self) -> None:
        """interaction = (w_p_s - w_b_s) * (R_p_s - R_b_s)."""
        from analysis.report import _compute_brinson_effects

        df = self._build_sector_df()
        result = _compute_brinson_effects(df)
        tech = result.filter(pl.col("sector").cast(pl.String) == "Tech")
        expected = (0.6 - 0.5) * (0.02 - 0.015)
        assert tech["interaction"][0] == pytest.approx(expected, abs=1e-12)

    def test_active_return_equals_sum_of_effects(self) -> None:
        """active_return must equal allocation + selection + interaction exactly."""
        from analysis.report import _compute_brinson_effects

        df = self._build_sector_df()
        result = _compute_brinson_effects(df)
        result = result.with_columns(
            (pl.col("allocation") + pl.col("selection") + pl.col("interaction")).alias(
                "expected_active"
            )
        )
        np.testing.assert_allclose(
            result["active_return"].to_numpy(),
            result["expected_active"].to_numpy(),
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# _aggregate_sector_effects
# ---------------------------------------------------------------------------


class TestAggregateSectorEffects:
    def _build_two_date_df(self) -> pl.DataFrame:
        """'Tech' appears on two dates; values should be summed."""
        return pl.DataFrame(
            {
                "date": [D1, D2, D1],
                "sector": pl.Series(["Tech", "Tech", "Energy"], dtype=pl.Categorical),
                "allocation": [0.001, 0.002, 0.0005],
                "selection": [0.003, 0.004, 0.0],
                "interaction": [0.0001, 0.0002, 0.0],
                "active_return": [0.0041, 0.0062, 0.0005],
            }
        )

    def test_sums_over_dates(self) -> None:
        """Each effect must be summed across dates for the same sector."""
        from analysis.report import _aggregate_sector_effects

        df = self._build_two_date_df()
        result = _aggregate_sector_effects(df)
        tech = result.filter(pl.col("sector").cast(pl.String) == "Tech")
        assert tech["allocation"][0] == pytest.approx(0.003, abs=1e-12)
        assert tech["selection"][0] == pytest.approx(0.007, abs=1e-12)

    def test_one_row_per_sector(self) -> None:
        """Output must have exactly one row per sector."""
        from analysis.report import _aggregate_sector_effects

        df = self._build_two_date_df()
        result = _aggregate_sector_effects(df)
        assert result.height == 2

    def test_output_columns(self) -> None:
        """Output must contain (sector, allocation, selection, interaction, active_return)."""
        from analysis.report import _aggregate_sector_effects

        df = self._build_two_date_df()
        result = _aggregate_sector_effects(df)
        for col in ("sector", "allocation", "selection", "interaction", "active_return"):
            assert col in result.columns


# ---------------------------------------------------------------------------
# _brinson_decomposition — single-sector edge case
# ---------------------------------------------------------------------------


class TestBrinsonDecompositionSingleSector:
    def test_single_sector_additive_identity(self) -> None:
        """With one sector, allocation+selection+interaction == active_return exactly."""
        from analysis.report import BrinsonDecomposition, _brinson_decomposition

        d = D1
        # Portfolio: one asset, full weight in "Tech"
        pw = _make_portfolio_weights([d], [0], [1.0])
        bw = _make_benchmark_weights([d], [0], [1.0])
        ar = _make_asset_returns([d], [0], [0.05])
        sm = _make_sector_map([0], ["Tech"])

        result = _brinson_decomposition(pw, bw, ar, sm)
        assert isinstance(result, BrinsonDecomposition)

        combined = (
            result.allocation.join(result.selection, on="sector")
            .join(result.interaction, on="sector")
            .join(result.sector_active_return, on="sector")
        )
        expected_active = (
            combined["allocation"] + combined["selection"] + combined["interaction"]
        ).to_numpy()
        np.testing.assert_allclose(
            combined["active_return"].to_numpy(), expected_active, atol=1e-12
        )

    def test_no_overlap_returns_empty(self) -> None:
        """When dates don't overlap across panels, result must be empty."""
        from analysis.report import _brinson_decomposition

        pw = _make_portfolio_weights([D1], [0], [1.0])
        bw = _make_benchmark_weights([D2], [0], [1.0])  # different date
        ar = _make_asset_returns([D1], [0], [0.01])
        sm = _make_sector_map([0], ["Tech"])

        result = _brinson_decomposition(pw, bw, ar, sm)
        assert result.allocation.height == 0


# ---------------------------------------------------------------------------
# _portfolio_weights_and_strategy_returns
# ---------------------------------------------------------------------------


class TestPortfolioWeightsAndStrategyReturns:
    def test_returns_two_dataframes(self) -> None:
        """Must return (portfolio_weights, strategy_returns) as a 2-tuple."""
        from analysis.report import _portfolio_weights_and_strategy_returns
        from backtest.engine import BacktestResult

        trade_log = pl.DataFrame(
            {
                "date": [D1, D1],
                "id": pl.Series([0, 1], dtype=pl.Int64),
                "quantity": [0.6, 0.4],
            }
        )
        nav_history = pl.DataFrame({"date": [D1, D2], "nav": [1_000_000.0, 1_010_000.0]})
        br = BacktestResult(
            nav_history=nav_history,
            trade_log=trade_log,
            final_positions=np.zeros(2),
        )

        pw, sr = _portfolio_weights_and_strategy_returns(br)
        assert "date" in pw.columns
        assert "weight" in pw.columns
        assert "date" in sr.columns
        assert "return_1d" in sr.columns

    def test_weights_sum_to_one(self) -> None:
        """Portfolio weights on a rebalance date must sum to 1.0."""
        from analysis.report import _portfolio_weights_and_strategy_returns
        from backtest.engine import BacktestResult

        trade_log = pl.DataFrame(
            {
                "date": [D1, D1, D1],
                "id": pl.Series([0, 1, 2], dtype=pl.Int64),
                "quantity": [0.5, 0.3, 0.2],
            }
        )
        nav_history = pl.DataFrame({"date": [D1, D2], "nav": [1_000_000.0, 1_005_000.0]})
        br = BacktestResult(
            nav_history=nav_history,
            trade_log=trade_log,
            final_positions=np.zeros(3),
        )

        pw, _ = _portfolio_weights_and_strategy_returns(br)
        weight_sum = float(pw.filter(pl.col("date") == D1)["weight"].sum())
        assert weight_sum == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# _benchmark_relative_scalars
# ---------------------------------------------------------------------------


class TestBenchmarkRelativeScalars:
    def _make_returns(self, vals: list[float]) -> pl.DataFrame:
        from etl.source import session_axis

        dates = session_axis(len(vals)).to_list()
        return pl.DataFrame({"date": dates, "return_1d": vals})

    def test_returns_four_elements(self) -> None:
        """Must return a 4-tuple (ar_series, te, ir, benchmark_metrics)."""
        from analysis.report import _benchmark_relative_scalars

        r = self._make_returns([0.01, 0.02, -0.01, 0.005])
        b = self._make_returns([0.005, 0.015, -0.005, 0.002])
        result = _benchmark_relative_scalars(r, b)
        assert len(result) == 4

    def test_ar_series_columns(self) -> None:
        """active_return_series must have (date, active_return) columns."""
        from analysis.report import _benchmark_relative_scalars

        r = self._make_returns([0.01, 0.02, -0.01])
        b = self._make_returns([0.005, 0.015, -0.005])
        ar_series, _, _, _ = _benchmark_relative_scalars(r, b)
        assert "date" in ar_series.columns
        assert "active_return" in ar_series.columns

    def test_te_non_negative(self) -> None:
        """Tracking error must be >= 0."""
        from analysis.report import _benchmark_relative_scalars

        rng = np.random.default_rng(7)
        r = self._make_returns(rng.normal(0.001, 0.01, 60).tolist())
        b = self._make_returns(rng.normal(0.0008, 0.009, 60).tolist())
        _, te, _, _ = _benchmark_relative_scalars(r, b)
        assert te >= 0.0

    def test_benchmark_metrics_keys(self) -> None:
        """benchmark_metrics must contain the three canonical keys."""
        from analysis.report import _benchmark_relative_scalars

        r = self._make_returns([0.01, 0.02, -0.01, 0.005, 0.003])
        b = self._make_returns([0.005, 0.015, -0.005, 0.002, 0.001])
        _, _, _, bm = _benchmark_relative_scalars(r, b)
        assert "active_return_ann" in bm
        assert "tracking_error" in bm
        assert "information_ratio" in bm
        assert all(math.isfinite(v) for v in bm.values())

    def test_positive_ir_when_strategy_always_beats(self) -> None:
        """IR must be positive when strategy outperforms benchmark on every date."""
        from analysis.report import _benchmark_relative_scalars

        rng = np.random.default_rng(11)
        base = rng.normal(0.0, 0.01, 60)
        r = self._make_returns((base + 0.002).tolist())
        b = self._make_returns(base.tolist())
        _, _, ir, _ = _benchmark_relative_scalars(r, b)
        assert ir > 0.0
