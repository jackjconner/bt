"""Tests for corporate-action price adjustment."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from .adjust import adjust_prices


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
