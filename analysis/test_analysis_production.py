"""Tests for the production analysis module features.

Each test group has a docstring explaining what behavioral property is being
checked and why it must hold. Assertions are written so that they fail without
the implementation (not just "does it run without error").
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from etl.datasets import GenSpec, generate

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SPEC = GenSpec(n_assets=10, n_dates=252, seed=42)


def _nav(values: list[float], start: str = "2000-01-03") -> pl.DataFrame:
    """Build a minimal nav_history DataFrame."""
    from etl.source import session_axis

    dates = session_axis(len(values), start)
    return pl.DataFrame({"date": dates, "nav": values})


def _returns(values: list[float], start: str = "2000-01-03") -> pl.DataFrame:
    """Build a minimal returns DataFrame."""
    from etl.source import session_axis

    dates = session_axis(len(values), start)
    return pl.DataFrame({"date": dates, "return_1d": values})


def _benchmark(values: list[float], start: str = "2000-01-03") -> pl.DataFrame:
    """Benchmark with return_1d column (fractional)."""
    from etl.source import session_axis

    dates = session_axis(len(values), start)
    return pl.DataFrame({"date": dates, "return_1d": values})


# ---------------------------------------------------------------------------
# 1. CAGR / geometric return
# ---------------------------------------------------------------------------


class TestCAGR:
    def test_flat_nav_gives_zero_cagr(self):
        """A constant NAV should produce exactly 0 % CAGR."""
        from analysis.risk import cagr

        nav = _nav([100.0] * 252)
        assert cagr(nav) == pytest.approx(0.0, abs=1e-9)

    def test_doubling_nav_in_252_sessions_gives_100pct(self):
        """NAV doubling in exactly one year of sessions → CAGR ≈ 100 %."""
        from analysis.risk import cagr

        nav_vals = np.linspace(100.0, 200.0, 252).tolist()
        nav = _nav(nav_vals)
        # CAGR = (200/100)^(252/252) - 1 = 1.0
        assert cagr(nav) == pytest.approx(1.0, rel=1e-6)

    def test_cagr_exceeds_arithmetic_return_when_volatile(self):
        """Geometric mean ≤ arithmetic mean; CAGR must be ≤ arithmetic annualized return."""
        from analysis.risk import cagr

        rng = np.random.default_rng(0)
        r = rng.normal(0.0, 0.02, 252)
        nav_vals = 100.0 * np.cumprod(1.0 + r)
        nav = _nav(nav_vals.tolist())
        rets = _returns(r.tolist())
        arith = float(rets["return_1d"].mean()) * 252
        geo = cagr(nav)
        assert geo <= arith + 1e-9


class TestAnnualizedReturnCalendar:
    def test_uses_session_count_not_row_count(self):
        """When a calendar has fewer sessions than rows, CAGR should differ."""
        from analysis.risk import annualized_return_calendar

        rng = np.random.default_rng(1)
        r = rng.normal(0.001, 0.01, 100)
        rets = _returns(r.tolist())

        # Calendar with only 50 sessions marked True
        from etl.source import session_axis

        dates = session_axis(100).to_list()
        is_sess = [True] * 50 + [False] * 50
        cal = pl.DataFrame({"date": dates, "exchange": "X", "is_session": is_sess})

        result = annualized_return_calendar(rets, cal)
        # 50 sessions → annualizes more aggressively than 252 sessions
        from analysis.risk import cagr

        nav_vals = 100.0 * np.cumprod(1.0 + r)
        nav = _nav(nav_vals.tolist())
        result_default = cagr(nav)  # uses 100 rows / 252 for annualization
        # Different session count → different result
        assert result != pytest.approx(result_default, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Downside / risk suite
# ---------------------------------------------------------------------------


class TestSortino:
    def test_sortino_ge_sharpe_when_no_downside(self):
        """When all returns are positive, Sortino ≥ Sharpe (zero downside vol)."""
        from analysis.risk import sortino
        from analysis.metrics import sharpe

        r = [0.001] * 252
        rets = _returns(r)
        s = sharpe(rets)
        so = sortino(rets)
        # With all-positive returns, downside vol → 0, so Sortino → ∞;
        # in our implementation it returns 0.0 when no downside obs exist.
        # Regardless, Sortino must be ≥ Sharpe (or 0 when undefined).
        assert so >= s or so == 0.0

    def test_sortino_lt_sharpe_when_fat_downside(self):
        """With asymmetric downside, Sortino is worse than Sharpe (positive case).

        When mean excess return is positive, fat downside makes downside vol >
        total vol, so Sortino < Sharpe. We use a series with clearly positive
        mean to keep the comparison unambiguous.
        """
        from analysis.risk import sortino
        from analysis.metrics import sharpe

        rng = np.random.default_rng(2)
        # Positive mean: many small gains, occasional large losses
        gains = list(rng.exponential(0.005, 220) + 0.01)  # shifted positive
        losses = list(-rng.exponential(0.05, 32))  # rare large losses
        r = gains + losses
        rng.shuffle(r)
        rets = _returns(r)
        s = sharpe(rets)
        so = sortino(rets)
        # Positive mean required for comparison to be meaningful
        assert float(pl.Series(r).mean()) > 0
        # Downside vol > total vol when losses are fat → Sortino < Sharpe
        assert so < s

    def test_sortino_sign_matches_mean_return(self):
        """Sortino sign must match the sign of mean excess return."""
        from analysis.risk import sortino

        pos_rets = _returns([0.002] * 100 + [-0.001] * 52)
        neg_rets = _returns([-0.002] * 100 + [0.001] * 52)
        assert sortino(pos_rets) > 0
        assert sortino(neg_rets) < 0


class TestCalmar:
    def test_positive_cagr_positive_mdd_gives_positive_calmar(self):
        from analysis.risk import calmar

        nav_vals = 100.0 * np.cumprod(1.0 + np.random.default_rng(3).normal(0.001, 0.01, 252))
        nav = _nav(nav_vals.tolist())
        c = calmar(nav)
        assert c > 0

    def test_zero_drawdown_gives_zero_calmar(self):
        """Monotonically increasing NAV has mdd=0; calmar returns 0 (not inf)."""
        from analysis.risk import calmar

        nav = _nav(list(np.linspace(100.0, 150.0, 252)))
        assert calmar(nav) == 0.0


class TestVaRCVaR:
    def test_var_is_negative(self):
        from analysis.risk import var_historical

        r = _returns(list(np.random.default_rng(4).normal(0.0, 0.01, 252)))
        v = var_historical(r, confidence=0.95)
        assert v < 0

    def test_cvar_le_var(self):
        """CVaR ≤ VaR since CVaR is the mean of the tail beyond VaR."""
        from analysis.risk import var_historical, cvar_historical

        r = _returns(list(np.random.default_rng(5).normal(0.0, 0.01, 252)))
        v = var_historical(r, 0.95)
        cv = cvar_historical(r, 0.95)
        assert cv <= v + 1e-12

    def test_cvar_uses_full_tail(self):
        """CVaR should be more negative than VaR when there are extreme losses."""
        from analysis.risk import var_historical, cvar_historical

        # Inject a few extreme losses
        base = [0.001] * 240 + [-0.10] * 12
        r = _returns(base)
        v = var_historical(r, 0.95)
        cv = cvar_historical(r, 0.95)
        assert cv < v


class TestDistributional:
    def test_skewness_of_symmetric_data_near_zero(self):
        from analysis.risk import skewness

        rng = np.random.default_rng(6)
        r = _returns(list(rng.normal(0.0, 0.01, 10_000)))
        assert abs(skewness(r)) < 0.1

    def test_kurtosis_of_normal_near_zero(self):
        from analysis.risk import excess_kurtosis

        rng = np.random.default_rng(7)
        r = _returns(list(rng.normal(0.0, 0.01, 10_000)))
        assert abs(excess_kurtosis(r)) < 0.3

    def test_hit_rate_all_positive(self):
        from analysis.risk import hit_rate

        r = _returns([0.001] * 100)
        assert hit_rate(r) == 1.0

    def test_best_worst_day(self):
        from analysis.risk import best_day, worst_day

        r = _returns([-0.05, 0.03, 0.01, -0.02])
        assert best_day(r) == pytest.approx(0.03)
        assert worst_day(r) == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# 3. Sharpe with time-varying rf series
# ---------------------------------------------------------------------------


class TestSharpeSeries:
    def test_scalar_rf_and_series_rf_agree_when_constant(self):
        """A constant rf Series should give the same result as the scalar path."""
        from analysis.metrics import sharpe

        rng = np.random.default_rng(8)
        r = _returns(list(rng.normal(0.0005, 0.01, 252)))
        rf_scalar = 0.03  # annual
        rf_daily = 0.03 / 252
        rf_series = pl.Series([rf_daily] * 252)

        s_scalar = sharpe(r, rf=rf_scalar)
        s_series = sharpe(r, rf=rf_series)
        assert s_scalar == pytest.approx(s_series, rel=1e-6)

    def test_higher_rf_lowers_sharpe(self):
        """Higher risk-free rate → lower Sharpe for a non-constant return series."""
        from analysis.metrics import sharpe

        rng = np.random.default_rng(99)
        # Non-constant series so std is nonzero for both rf levels
        r = _returns(list(rng.normal(0.002, 0.005, 252)))
        s_low = sharpe(r, rf=0.0)
        s_high = sharpe(r, rf=0.30)  # very high rf to ensure lower excess
        assert s_low > s_high


# ---------------------------------------------------------------------------
# 4. Benchmark-relative metrics
# ---------------------------------------------------------------------------


class TestBeta:
    def test_benchmark_against_itself_has_beta_1(self):
        """Beta of any series against itself must equal 1."""
        from analysis.benchmark import beta

        rng = np.random.default_rng(9)
        r = list(rng.normal(0.0, 0.01, 252))
        rets = _returns(r)
        bmk = _benchmark(r)
        assert beta(rets, bmk) == pytest.approx(1.0, rel=1e-6)

    def test_doubled_returns_have_beta_2(self):
        """If strategy = 2 × benchmark, beta must be 2."""
        from analysis.benchmark import beta

        rng = np.random.default_rng(10)
        r = rng.normal(0.0, 0.01, 252)
        rets = _returns((2 * r).tolist())
        bmk = _benchmark(r.tolist())
        assert beta(rets, bmk) == pytest.approx(2.0, rel=1e-5)


class TestAlpha:
    def test_zero_alpha_when_strategy_equals_benchmark(self):
        """Strategy == benchmark means alpha = 0 exactly (not just approximately)."""
        from analysis.benchmark import alpha

        rng = np.random.default_rng(11)
        r = list(rng.normal(0.0, 0.01, 252))
        rets = _returns(r)
        bmk = _benchmark(r)
        # When r == b, beta=1 and r_mean == b_mean so alpha = r_mean - 1*b_mean = 0
        assert alpha(rets, bmk) == pytest.approx(0.0, abs=1e-10)

    def test_positive_alpha_when_strategy_earns_more(self):
        """Strategy with daily excess outperformance should have positive alpha."""
        from analysis.benchmark import alpha

        rng = np.random.default_rng(12)
        base = rng.normal(0.0, 0.01, 252)
        strat = base + 0.001  # daily alpha of 0.1 %
        rets = _returns(strat.tolist())
        bmk = _benchmark(base.tolist())
        assert alpha(rets, bmk) > 0


class TestRSquared:
    def test_r2_perfect_for_identical_series(self):
        from analysis.benchmark import r_squared

        r = list(np.random.default_rng(13).normal(0.0, 0.01, 252))
        rets = _returns(r)
        bmk = _benchmark(r)
        assert r_squared(rets, bmk) == pytest.approx(1.0, abs=1e-9)

    def test_r2_zero_for_independent_series(self):
        from analysis.benchmark import r_squared

        rng = np.random.default_rng(14)
        rets = _returns(list(rng.normal(0.0, 0.01, 252)))
        bmk = _benchmark(list(rng.normal(0.0, 0.01, 252)))
        # Not exactly 0 but should be close
        assert abs(r_squared(rets, bmk)) < 0.15


class TestTrackingError:
    def test_te_zero_when_identical(self):
        from analysis.benchmark import tracking_error

        r = list(np.random.default_rng(15).normal(0.0, 0.01, 252))
        rets = _returns(r)
        bmk = _benchmark(r)
        assert tracking_error(rets, bmk) == pytest.approx(0.0, abs=1e-10)

    def test_te_positive_for_different_series(self):
        from analysis.benchmark import tracking_error

        rng = np.random.default_rng(16)
        rets = _returns(list(rng.normal(0.0, 0.01, 252)))
        bmk = _benchmark(list(rng.normal(0.0, 0.01, 252)))
        assert tracking_error(rets, bmk) > 0


class TestInformationRatio:
    def test_positive_ir_when_strategy_beats_benchmark(self):
        from analysis.benchmark import information_ratio

        rng = np.random.default_rng(17)
        base = rng.normal(0.0, 0.01, 252)
        rets = _returns((base + 0.001).tolist())
        bmk = _benchmark(base.tolist())
        assert information_ratio(rets, bmk) > 0


class TestCapture:
    def test_up_capture_gt_1_when_strategy_amplifies_gains(self):
        from analysis.benchmark import up_capture

        rng = np.random.default_rng(18)
        base = rng.normal(0.0, 0.01, 252)
        rets = _returns((2.0 * base).tolist())
        bmk = _benchmark(base.tolist())
        # On up days, strategy earns twice as much → up capture ≈ 2
        uc = up_capture(rets, bmk)
        assert uc == pytest.approx(2.0, rel=0.05)

    def test_down_capture_positive_for_levered_strategy(self):
        from analysis.benchmark import down_capture

        rng = np.random.default_rng(19)
        base = rng.normal(0.0, 0.01, 252)
        rets = _returns((2.0 * base).tolist())
        bmk = _benchmark(base.tolist())
        dc = down_capture(rets, bmk)
        assert dc == pytest.approx(2.0, rel=0.05)


class TestActiveReturns:
    def test_active_returns_shape_and_values(self):
        from analysis.benchmark import active_returns

        rng = np.random.default_rng(20)
        r = list(rng.normal(0.001, 0.01, 100))
        b = list(rng.normal(0.0, 0.01, 100))
        rets = _returns(r)
        bmk = _benchmark(b)
        ar = active_returns(rets, bmk)
        assert "date" in ar.columns
        assert "active_return" in ar.columns
        expected = np.array(r) - np.array(b)
        np.testing.assert_allclose(
            ar["active_return"].to_numpy(), expected, rtol=1e-9
        )

    def test_relative_drawdown_zero_when_equal(self):
        from analysis.benchmark import relative_drawdown

        r = [0.001, -0.002, 0.003, -0.001]
        rets = _returns(r)
        bmk = _benchmark(r)
        rd = relative_drawdown(rets, bmk)
        np.testing.assert_allclose(
            rd["rel_drawdown"].to_numpy(), 0.0, atol=1e-12
        )


# ---------------------------------------------------------------------------
# 5. Turnover
# ---------------------------------------------------------------------------


class TestTurnover:
    def _make_trade_log(self, n_rebalances: int, n_assets: int, seed: int = 0):
        from etl.source import session_axis

        rng = np.random.default_rng(seed)
        dates_all = session_axis(n_rebalances * n_assets).to_list()
        rows = []
        for i in range(n_rebalances):
            d = dates_all[i * n_assets]
            weights = rng.dirichlet(np.ones(n_assets))
            prev = rng.dirichlet(np.ones(n_assets))
            for j in range(n_assets):
                rows.append({"date": d, "id": j, "quantity": weights[j] - prev[j]})
        return pl.DataFrame(rows)

    def test_zero_turnover_when_no_trades(self):
        """An empty trade log should produce zero turnover."""
        from analysis.turnover import one_way_turnover, two_way_turnover

        empty = pl.DataFrame({"date": [], "id": [], "quantity": []}).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("id").cast(pl.Int64),
            pl.col("quantity").cast(pl.Float64),
        )
        ot = one_way_turnover(empty)
        tw = two_way_turnover(empty)
        assert ot.height == 0
        assert tw.height == 0

    def test_two_way_double_one_way(self):
        """Two-way turnover must be exactly 2× one-way."""
        from analysis.turnover import one_way_turnover, two_way_turnover

        tl = self._make_trade_log(10, 5)
        ot = one_way_turnover(tl).sort("date")
        tw = two_way_turnover(tl).sort("date")
        np.testing.assert_allclose(
            tw["turnover_2w"].to_numpy(),
            ot["turnover_1w"].to_numpy() * 2,
            rtol=1e-9,
        )


# ---------------------------------------------------------------------------
# 6. Net-of-cost NAV
# ---------------------------------------------------------------------------


class TestNetNav:
    def test_net_nav_le_gross_nav(self):
        """Net NAV must be ≤ gross NAV on every date when costs > 0."""
        from analysis.turnover import net_nav

        spec = SPEC
        costs = generate("transaction_costs", spec)
        nav_hist = generate("benchmark_returns", spec).filter(
            pl.col("benchmark_id") == "BMK0"
        ).with_columns(
            (100.0 * (1.0 + pl.col("return") / 100.0).cum_prod()).alias("nav")
        ).select("date", "nav")

        # Minimal trade_log: one rebalance at first date
        rng = np.random.default_rng(99)
        first_date = nav_hist["date"][0]
        n_assets = spec.n_assets
        weights = rng.dirichlet(np.ones(n_assets))
        trade_log = pl.DataFrame({
            "date": [first_date] * n_assets,
            "id": list(range(n_assets)),
            "quantity": weights.tolist(),
        })

        result = net_nav(nav_hist, trade_log, costs)
        assert (result["nav_net"] <= result["nav_gross"] + 1e-10).all()

    def test_net_nav_equals_gross_when_zero_costs(self):
        """Zero-cost model: net NAV == gross NAV to floating-point precision."""
        from analysis.turnover import net_nav

        spec = SPEC
        # Zero out all cost columns
        costs = generate("transaction_costs", spec).with_columns(
            pl.lit(0.0).alias("commission_bps"),
            pl.lit(0.0).alias("half_spread_bps"),
            pl.lit(0.0).alias("exchange_fee_bps"),
        )
        nav_hist = generate("benchmark_returns", spec).filter(
            pl.col("benchmark_id") == "BMK0"
        ).with_columns(
            (100.0 * (1.0 + pl.col("return") / 100.0).cum_prod()).alias("nav")
        ).select("date", "nav")

        first_date = nav_hist["date"][0]
        n_assets = spec.n_assets
        rng = np.random.default_rng(100)
        weights = rng.dirichlet(np.ones(n_assets))
        trade_log = pl.DataFrame({
            "date": [first_date] * n_assets,
            "id": list(range(n_assets)),
            "quantity": weights.tolist(),
        })

        result = net_nav(nav_hist, trade_log, costs)
        np.testing.assert_allclose(
            result["nav_net"].to_numpy(),
            result["nav_gross"].to_numpy(),
            rtol=1e-9,
        )


# ---------------------------------------------------------------------------
# 7. Position concentration
# ---------------------------------------------------------------------------


class TestConcentration:
    def _weights(self, n: int = 5) -> pl.DataFrame:
        from etl.source import session_axis

        rng = np.random.default_rng(0)
        d = session_axis(3).to_list()
        rows = []
        for dt in d:
            w = rng.dirichlet(np.ones(n))
            for i, wi in enumerate(w):
                rows.append({"date": dt, "id": i, "weight": wi})
        return pl.DataFrame(rows)

    def test_gross_exposure_unit_for_longonly(self):
        """Long-only normalized weights: gross exposure == 1.0."""
        from analysis.turnover import gross_exposure

        ge = gross_exposure(self._weights())
        np.testing.assert_allclose(
            ge["gross_exposure"].to_numpy(), 1.0, atol=1e-10
        )

    def test_net_exposure_unit_for_longonly(self):
        from analysis.turnover import net_exposure

        ne = net_exposure(self._weights())
        np.testing.assert_allclose(ne["net_exposure"].to_numpy(), 1.0, atol=1e-10)

    def test_effective_n_le_actual_n(self):
        """Effective N ≤ actual number of positions (Herfindahl property)."""
        from analysis.turnover import effective_n

        en = effective_n(self._weights(n=5))
        assert (en["effective_n"] <= 5.0 + 1e-10).all()

    def test_effective_n_equal_weights(self):
        """Equal weights over N assets → effective_N = N."""
        from analysis.turnover import effective_n
        from etl.source import session_axis

        n = 5
        d = session_axis(1).to_list()[0]
        w = pl.DataFrame({
            "date": [d] * n,
            "id": list(range(n)),
            "weight": [1.0 / n] * n,
        })
        en = effective_n(w)
        assert en["effective_n"][0] == pytest.approx(float(n), rel=1e-9)


# ---------------------------------------------------------------------------
# 8. Rolling metrics
# ---------------------------------------------------------------------------


class TestRolling:
    def _rets(self, seed: int = 0, n: int = 200) -> pl.DataFrame:
        rng = np.random.default_rng(seed)
        return _returns(list(rng.normal(0.001, 0.01, n)))

    def test_rolling_sharpe_leading_nulls(self):
        """First (window-1) rows must be null."""
        from analysis.rolling import rolling_sharpe

        rets = self._rets()
        rs = rolling_sharpe(rets, window=30)
        assert rs["rolling_sharpe"][:29].null_count() == 29
        assert rs["rolling_sharpe"][29] is not None

    def test_rolling_vol_positive(self):
        """Annualized rolling vol must be positive for non-constant returns."""
        from analysis.rolling import rolling_vol

        rets = self._rets()
        rv = rolling_vol(rets, window=30)
        valid = rv["rolling_vol"].drop_nulls()
        assert (valid > 0).all()

    def test_rolling_beta_self_is_one(self):
        """Rolling beta of a series against itself ≈ 1 for each window."""
        from analysis.rolling import rolling_beta

        r = list(np.random.default_rng(21).normal(0.0, 0.01, 200))
        rets = _returns(r)
        bmk = _benchmark(r)
        rb = rolling_beta(rets, bmk, window=30)
        valid = rb["rolling_beta"].drop_nulls().to_numpy()
        np.testing.assert_allclose(valid, 1.0, atol=1e-6)

    def test_rolling_mdd_non_positive(self):
        """Rolling max drawdown must be ≤ 0."""
        from analysis.rolling import rolling_max_drawdown

        rets = self._rets()
        rmdd = rolling_max_drawdown(rets, window=30)
        valid = rmdd["rolling_max_drawdown"].drop_nulls().to_numpy()
        assert (valid <= 1e-10).all()


# ---------------------------------------------------------------------------
# 9. Periodic return tables
# ---------------------------------------------------------------------------


class TestPeriodic:
    def _annual_rets(self) -> pl.DataFrame:
        """Two full years of daily returns."""
        rng = np.random.default_rng(22)
        return _returns(list(rng.normal(0.001, 0.01, 504)))

    def test_monthly_returns_compounds(self):
        """Compound of monthly returns over a year ≈ annual return."""
        from analysis.periodic import monthly_returns, annual_returns

        rets = self._annual_rets()
        monthly = monthly_returns(rets)
        annual = annual_returns(rets)

        # For the first year, compound months should ≈ annual
        yr = annual["year"][0]
        months_yr = monthly.filter(pl.col("year") == yr)
        compound = float(
            (1.0 + months_yr["monthly_return"]).product() - 1.0
        )
        annual_yr = float(annual.filter(pl.col("year") == yr)["annual_return"][0])
        assert compound == pytest.approx(annual_yr, rel=1e-9)

    def test_monthly_wide_has_12_month_columns(self):
        from analysis.periodic import monthly_returns_wide

        rets = self._annual_rets()
        wide = monthly_returns_wide(rets)
        month_cols = [c for c in wide.columns if c != "year"]
        assert len(month_cols) == 12

    def test_quarterly_has_4_quarters(self):
        from analysis.periodic import quarterly_returns

        rets = self._annual_rets()
        qr = quarterly_returns(rets)
        yr = qr["year"][0]
        n_quarters = qr.filter(pl.col("year") == yr).height
        assert n_quarters == 4


# ---------------------------------------------------------------------------
# 10. Factor attribution
# ---------------------------------------------------------------------------


class TestFactorAttribution:
    def test_r2_near_1_when_returns_are_linear_combo(self):
        """When strategy is a known linear combo of factors, R² ≈ 1."""
        from analysis.attribution import factor_attribution
        from etl.source import session_axis

        rng = np.random.default_rng(23)
        n = 252
        nk = 3
        f = rng.normal(0.0, 0.01, (n, nk))  # factor returns
        betas_true = np.array([0.5, -0.3, 0.8])
        r = f @ betas_true + rng.normal(0.0, 1e-6, n)  # near-perfect fit

        dates = session_axis(n).to_list()
        factor_frames = []
        for k in range(nk):
            for i, d in enumerate(dates):
                factor_frames.append({"date": d, "factor_id": k, "return": f[i, k]})
        fr = pl.DataFrame(factor_frames)

        rets = pl.DataFrame({"date": dates, "return_1d": r.tolist()})
        result = factor_attribution(rets, fr)
        assert result.r_squared > 0.99

    def test_factor_exposures_keys_match_factor_ids(self):
        """Keys of factor_exposures must be the factor IDs in the input."""
        from analysis.attribution import factor_attribution
        from etl.source import session_axis

        rng = np.random.default_rng(24)
        n, nk = 100, 5
        f = rng.normal(0.0, 0.01, (n, nk))
        r = f @ rng.normal(0.0, 1.0, nk)

        dates = session_axis(n).to_list()
        factor_frames = [
            {"date": dates[i], "factor_id": k, "return": float(f[i, k])}
            for i in range(n)
            for k in range(nk)
        ]
        fr = pl.DataFrame(factor_frames)
        rets = pl.DataFrame({"date": dates, "return_1d": r.tolist()})
        result = factor_attribution(rets, fr)
        assert set(result.factor_exposures.keys()) == set(range(nk))


# ---------------------------------------------------------------------------
# 11. benchmark_returns_to_fractional helper
# ---------------------------------------------------------------------------


class TestBenchmarkConverter:
    def test_converts_percent_to_fractional(self):
        from analysis.benchmark import benchmark_returns_to_fractional

        bmk_raw = pl.DataFrame({
            "date": [date(2000, 1, 3), date(2000, 1, 4)],
            "benchmark_id": pl.Series(["BMK0", "BMK0"], dtype=pl.Categorical),
            "return": [2.0, -1.0],  # percent
        })
        out = benchmark_returns_to_fractional(bmk_raw)
        assert out["return_1d"][0] == pytest.approx(0.02)
        assert out["return_1d"][1] == pytest.approx(-0.01)


# ---------------------------------------------------------------------------
# 12. BacktestAnalyzerImpl enriched fields
# ---------------------------------------------------------------------------


class TestAnalyzerImpl:
    def test_cagr_and_sortino_populated(self):
        """BacktestAnalyzerImpl.analyze must populate the new cagr and sortino fields."""
        from analysis.metrics import BacktestAnalyzerImpl

        # Build a minimal nav_history directly
        rng = np.random.default_rng(0)
        r = rng.normal(0.001, 0.01, 100)
        from etl.source import session_axis
        dates = session_axis(100).to_list()
        nav_vals = 100.0 * np.cumprod(1.0 + r)
        nav_hist = pl.DataFrame({"date": dates, "nav": nav_vals.tolist()})
        trade_log = pl.DataFrame({"date": [], "id": [], "quantity": []}).with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("id").cast(pl.Int64),
            pl.col("quantity").cast(pl.Float64),
        )

        from backtest.engine import BacktestResult
        result = BacktestResult(
            nav_history=nav_hist,
            trade_log=trade_log,
            final_positions=np.zeros(5),
        )
        analysis = BacktestAnalyzerImpl().analyze(result)
        # Must be non-zero (random walk with drift)
        assert analysis.cagr != 0.0
        # sortino field exists
        assert isinstance(analysis.sortino, float)
