"""Tests for corporate-action price adjustment."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from .adjust import (
    _adjust_single_asset,
    _apply_factor,
    _build_adj_log,
    _dividend_factor,
    _split_factor,
    adjust_prices,
)


def _prices(close_vals: list[float], asset_id: int = 0) -> pl.DataFrame:
    n = len(close_vals)
    base_date = date(2020, 1, 2)
    from datetime import timedelta

    dates = [base_date + timedelta(days=i) for i in range(n)]
    return pl.DataFrame(
        {
            "date": pl.Series(dates, dtype=pl.Date),
            "id": pl.Series([asset_id] * n, dtype=pl.Int64),
            "close": pl.Series(close_vals, dtype=pl.Float64),
        }
    )


def _no_actions() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ex_date": pl.Date,
            "id": pl.Int64,
            "action_type": pl.Categorical,
            "split_ratio": pl.Float64,
            "cash_amount": pl.Float64,
            "currency": pl.Categorical,
            "new_id": pl.Int64,
        }
    )


def test_no_actions_adj_close_equals_close():
    prices = _prices([10.0, 11.0, 12.0])
    result = adjust_prices(prices, _no_actions())
    np.testing.assert_array_equal(result.prices["adj_close"].to_numpy(), [10.0, 11.0, 12.0])


def test_split_2for1_halves_prior_prices():
    """2-for-1 split on day 2 (ex_date=Jan 3): day 0 and 1 close should be halved."""
    prices = _prices([10.0, 10.0, 5.0])
    ca = pl.DataFrame(
        {
            "ex_date": pl.Series([date(2020, 1, 4)], dtype=pl.Date),
            "id": pl.Series([0], dtype=pl.Int64),
            "action_type": pl.Series(["split"], dtype=pl.Categorical),
            "split_ratio": pl.Series([2.0], dtype=pl.Float64),
            "cash_amount": pl.Series([None], dtype=pl.Float64),
            "currency": pl.Series(["USD"], dtype=pl.Categorical),
            "new_id": pl.Series([None], dtype=pl.Int64),
        }
    )
    result = adjust_prices(prices, ca)
    adj = result.prices.sort("date")["adj_close"].to_numpy()
    # Before ex_date (Jan 2 and Jan 3): halved
    assert abs(adj[0] - 5.0) < 1e-9
    assert abs(adj[1] - 5.0) < 1e-9
    # On/after ex_date (Jan 4): unchanged
    assert abs(adj[2] - 5.0) < 1e-9


def test_cash_dividend_deflates_prior_prices():
    """$1 dividend on day 2 (ex_date=Jan 4), pre-ex close = 10.0.
    Factor = 10/(10+1) ≈ 0.909; prior closes multiply by that factor."""
    prices = _prices([10.0, 10.0, 9.0])  # price drops by ~dividend on ex-date
    ca = pl.DataFrame(
        {
            "ex_date": pl.Series([date(2020, 1, 4)], dtype=pl.Date),
            "id": pl.Series([0], dtype=pl.Int64),
            "action_type": pl.Series(["cash_dividend"], dtype=pl.Categorical),
            "split_ratio": pl.Series([None], dtype=pl.Float64),
            "cash_amount": pl.Series([1.0], dtype=pl.Float64),
            "currency": pl.Series(["USD"], dtype=pl.Categorical),
            "new_id": pl.Series([None], dtype=pl.Int64),
        }
    )
    result = adjust_prices(prices, ca)
    adj = result.prices.sort("date")["adj_close"].to_numpy()
    expected_factor = 10.0 / (10.0 + 1.0)
    assert abs(adj[0] - 10.0 * expected_factor) < 1e-9
    assert abs(adj[1] - 10.0 * expected_factor) < 1e-9
    # On/after ex_date: unchanged
    assert abs(adj[2] - 9.0) < 1e-9


def test_adj_log_recorded():
    prices = _prices([10.0, 10.0, 5.0])
    ca = pl.DataFrame(
        {
            "ex_date": pl.Series([date(2020, 1, 4)], dtype=pl.Date),
            "id": pl.Series([0], dtype=pl.Int64),
            "action_type": pl.Series(["split"], dtype=pl.Categorical),
            "split_ratio": pl.Series([2.0], dtype=pl.Float64),
            "cash_amount": pl.Series([None], dtype=pl.Float64),
            "currency": pl.Series(["USD"], dtype=pl.Categorical),
            "new_id": pl.Series([None], dtype=pl.Int64),
        }
    )
    result = adjust_prices(prices, ca)
    assert result.adj_log.height == 1
    assert abs(result.adj_log["factor"][0] - 0.5) < 1e-9


def test_no_actions_empty_log():
    prices = _prices([10.0, 11.0])
    result = adjust_prices(prices, _no_actions())
    assert result.adj_log.is_empty()


def test_action_before_any_price_no_effect():
    """An ex_date before all price dates should not apply any factor."""
    prices = _prices([10.0, 11.0])
    ca = pl.DataFrame(
        {
            "ex_date": pl.Series([date(2019, 12, 31)], dtype=pl.Date),
            "id": pl.Series([0], dtype=pl.Int64),
            "action_type": pl.Series(["split"], dtype=pl.Categorical),
            "split_ratio": pl.Series([2.0], dtype=pl.Float64),
            "cash_amount": pl.Series([None], dtype=pl.Float64),
            "currency": pl.Series(["USD"], dtype=pl.Categorical),
            "new_id": pl.Series([None], dtype=pl.Int64),
        }
    )
    result = adjust_prices(prices, ca)
    # No prior rows exist before the first price, so nothing changes
    adj = result.prices.sort("date")["adj_close"].to_numpy()
    np.testing.assert_array_equal(adj, [10.0, 11.0])


def test_multiple_assets_independent():
    """Splits for asset 0 must not affect asset 1."""
    p0 = _prices([10.0, 10.0, 5.0], asset_id=0)
    p1 = _prices([20.0, 20.0, 20.0], asset_id=1)
    prices = pl.concat([p0, p1])
    ca = pl.DataFrame(
        {
            "ex_date": pl.Series([date(2020, 1, 4)], dtype=pl.Date),
            "id": pl.Series([0], dtype=pl.Int64),
            "action_type": pl.Series(["split"], dtype=pl.Categorical),
            "split_ratio": pl.Series([2.0], dtype=pl.Float64),
            "cash_amount": pl.Series([None], dtype=pl.Float64),
            "currency": pl.Series(["USD"], dtype=pl.Categorical),
            "new_id": pl.Series([None], dtype=pl.Int64),
        }
    )
    result = adjust_prices(prices, ca)
    a1 = result.prices.filter(pl.col("id") == 1).sort("date")["adj_close"].to_numpy()
    np.testing.assert_array_equal(a1, [20.0, 20.0, 20.0])


# ------------------------------------------------------------------ #
# Unit tests for extracted helpers
# ------------------------------------------------------------------ #


def test_split_factor_2for1():
    row = {"split_ratio": 2.0}
    f = _split_factor(row)
    assert f is not None
    assert abs(f - 0.5) < 1e-12


def test_split_factor_3for1():
    row = {"split_ratio": 3.0}
    f = _split_factor(row)
    assert f is not None
    assert abs(f - 1.0 / 3.0) < 1e-12


def test_split_factor_none_ratio_returns_none():
    assert _split_factor({"split_ratio": None}) is None


def test_split_factor_zero_ratio_returns_none():
    assert _split_factor({"split_ratio": 0.0}) is None


def test_split_factor_nan_ratio_returns_none():
    assert _split_factor({"split_ratio": float("nan")}) is None


def test_dividend_factor_basic():
    """$2 dividend with pre-close $10: factor = 10/(10+2) = 5/6."""
    row = {"cash_amount": 2.0}
    f = _dividend_factor(row, pre_close=10.0)
    assert f is not None
    assert abs(f - 10.0 / 12.0) < 1e-12


def test_dividend_factor_none_cash_returns_none():
    assert _dividend_factor({"cash_amount": None}, pre_close=10.0) is None


def test_dividend_factor_zero_cash_returns_none():
    assert _dividend_factor({"cash_amount": 0.0}, pre_close=10.0) is None


def test_dividend_factor_zero_pre_close_returns_none():
    assert _dividend_factor({"cash_amount": 1.0}, pre_close=0.0) is None


def test_dividend_factor_nan_cash_returns_none():
    assert _dividend_factor({"cash_amount": float("nan")}, pre_close=10.0) is None


def test_apply_factor_partial():
    """_apply_factor multiplies only indices 0..last_prior inclusive."""
    factor = np.ones(5)
    _apply_factor(factor, 0.5, 2)
    np.testing.assert_array_almost_equal(factor, [0.5, 0.5, 0.5, 1.0, 1.0])


def test_apply_factor_all():
    factor = np.ones(3)
    _apply_factor(factor, 2.0, 2)
    np.testing.assert_array_almost_equal(factor, [2.0, 2.0, 2.0])


def test_apply_factor_only_first():
    factor = np.ones(4)
    _apply_factor(factor, 0.25, 0)
    np.testing.assert_array_almost_equal(factor, [0.25, 1.0, 1.0, 1.0])


def test_build_adj_log_empty():
    log = _build_adj_log([])
    assert log.is_empty()
    assert log.schema["id"] == pl.Int64
    assert log.schema["action_type"] == pl.Categorical
    assert log.schema["factor"] == pl.Float64


def test_build_adj_log_with_rows():
    rows = [
        {"ex_date": date(2020, 1, 3), "id": 0, "action_type": "split", "factor": 0.5},
        {"ex_date": date(2020, 2, 1), "id": 1, "action_type": "cash_dividend", "factor": 0.9},
    ]
    log = _build_adj_log(rows)
    assert log.height == 2
    assert log.schema["id"] == pl.Int64
    assert log.schema["action_type"] == pl.Categorical
    assert abs(log["factor"][0] - 0.5) < 1e-12


def test_adjust_single_asset_no_actions():
    pdf = _prices([10.0, 11.0, 12.0])
    frame, log_rows = _adjust_single_asset(pdf, None, "close")
    adj = frame.sort("date")["adj_close"].to_numpy()
    np.testing.assert_array_equal(adj, [10.0, 11.0, 12.0])
    assert log_rows == []


def test_adjust_single_asset_empty_actions():
    pdf = _prices([10.0, 11.0, 12.0])
    empty_ca = pl.DataFrame(
        schema={
            "ex_date": pl.Date,
            "id": pl.Int64,
            "action_type": pl.Categorical,
            "split_ratio": pl.Float64,
            "cash_amount": pl.Float64,
        }
    )
    frame, log_rows = _adjust_single_asset(pdf, empty_ca, "close")
    adj = frame.sort("date")["adj_close"].to_numpy()
    np.testing.assert_array_equal(adj, [10.0, 11.0, 12.0])
    assert log_rows == []


def test_adjust_single_asset_split():
    """2-for-1 split ex_date Jan 4: first two closes halved."""
    pdf = _prices([10.0, 10.0, 5.0])
    ca = pl.DataFrame(
        {
            "ex_date": pl.Series([date(2020, 1, 4)], dtype=pl.Date),
            "id": pl.Series([0], dtype=pl.Int64),
            "action_type": pl.Series(["split"], dtype=pl.Categorical),
            "split_ratio": pl.Series([2.0], dtype=pl.Float64),
            "cash_amount": pl.Series([None], dtype=pl.Float64),
        }
    )
    frame, log_rows = _adjust_single_asset(pdf, ca, "close")
    adj = frame.sort("date")["adj_close"].to_numpy()
    assert abs(adj[0] - 5.0) < 1e-9
    assert abs(adj[1] - 5.0) < 1e-9
    assert abs(adj[2] - 5.0) < 1e-9
    assert len(log_rows) == 1
    assert abs(log_rows[0]["factor"] - 0.5) < 1e-12


def test_special_dividend_treated_as_cash_dividend():
    """special_dividend should be handled identically to cash_dividend."""
    prices = _prices([10.0, 10.0, 9.5])
    ca = pl.DataFrame(
        {
            "ex_date": pl.Series([date(2020, 1, 4)], dtype=pl.Date),
            "id": pl.Series([0], dtype=pl.Int64),
            "action_type": pl.Series(["special_dividend"], dtype=pl.Categorical),
            "split_ratio": pl.Series([None], dtype=pl.Float64),
            "cash_amount": pl.Series([0.5], dtype=pl.Float64),
            "currency": pl.Series(["USD"], dtype=pl.Categorical),
            "new_id": pl.Series([None], dtype=pl.Int64),
        }
    )
    result = adjust_prices(prices, ca)
    adj = result.prices.sort("date")["adj_close"].to_numpy()
    expected_factor = 10.0 / (10.0 + 0.5)
    assert abs(adj[0] - 10.0 * expected_factor) < 1e-9
    assert abs(adj[1] - 10.0 * expected_factor) < 1e-9
    # On/after ex_date: unchanged
    assert abs(adj[2] - 9.5) < 1e-9
