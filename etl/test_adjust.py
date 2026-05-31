"""Tests for corporate-action price adjustment."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from .adjust import (
    _adjust_vectorized,
    _build_adj_log,
    adjust_prices,
)
from .quality import QUALITY_FLAG_COLUMNS, annotate_quality_flags


def _prices(close_vals: list[float], asset_id: int = 0) -> pl.DataFrame:
    n = len(close_vals)
    base_date = date(2020, 1, 2)
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


# ------------------------------------------------------------------ #
# Vectorized whole-panel path (_adjust_vectorized, used by adjust_prices)
# ------------------------------------------------------------------ #


def _ca_full(rows: list[dict]) -> pl.DataFrame:
    """Build a corporate_actions frame with the full registry schema."""
    if not rows:
        return _no_actions()
    return pl.DataFrame(
        {
            "ex_date": pl.Series([r["ex_date"] for r in rows], dtype=pl.Date),
            "id": pl.Series([r["id"] for r in rows], dtype=pl.Int64),
            "action_type": pl.Series([r["action_type"] for r in rows], dtype=pl.Categorical),
            "split_ratio": pl.Series([r.get("split_ratio") for r in rows], dtype=pl.Float64),
            "cash_amount": pl.Series([r.get("cash_amount") for r in rows], dtype=pl.Float64),
            "currency": pl.Series(["USD"] * len(rows), dtype=pl.Categorical),
            "new_id": pl.Series([None] * len(rows), dtype=pl.Int64),
        }
    )


def test_vectorized_back_adjusts_known_split_and_dividend_panel():
    """Direct correctness fixture: whole-panel vectorized adjust on a two-asset
    panel carrying a split (id 0) and a cash dividend (id 1), checked cell-for-
    cell against hand-computed back-adjusted closes.

    Asset 0: closes [10, 12, 6, 7] on Jan 2..5, 2-for-1 split ex_date Jan 4.
      Factor 1/2 applies to every session strictly before Jan 4 (Jan 2, Jan 3);
      Jan 4 and Jan 5 are on/after ex_date and unchanged.
    Asset 1: closes [20, 22, 21, 23] on Jan 2..5, $2 cash dividend ex_date Jan 4.
      Pre-ex close is the Jan 3 close (22.0); factor = 22/(22+2) = 11/12 applies
      to Jan 2 and Jan 3; Jan 4 and Jan 5 unchanged.
    """
    p0 = _prices([10.0, 12.0, 6.0, 7.0], asset_id=0)
    p1 = _prices([20.0, 22.0, 21.0, 23.0], asset_id=1)
    prices = pl.concat([p0, p1])
    ca = _ca_full(
        [
            {"ex_date": date(2020, 1, 4), "id": 0, "action_type": "split", "split_ratio": 2.0},
            {
                "ex_date": date(2020, 1, 4),
                "id": 1,
                "action_type": "cash_dividend",
                "cash_amount": 2.0,
            },
        ]
    )
    result = _adjust_vectorized(prices, ca, "close")
    adj = result.prices.sort("id", "date")

    a0 = adj.filter(pl.col("id") == 0)["adj_close"].to_numpy()
    np.testing.assert_allclose(a0, [5.0, 6.0, 6.0, 7.0], rtol=0, atol=1e-9)

    div_f = 22.0 / (22.0 + 2.0)
    a1 = adj.filter(pl.col("id") == 1)["adj_close"].to_numpy()
    np.testing.assert_allclose(a1, [20.0 * div_f, 22.0 * div_f, 21.0, 23.0], rtol=0, atol=1e-9)

    # Both actions back-adjust prior history, so both are recorded in the log.
    assert result.adj_log.height == 2
    factors = dict(
        zip(
            result.adj_log["id"].to_list(),
            result.adj_log["factor"].to_list(),
            strict=True,
        )
    )
    assert abs(factors[0] - 0.5) < 1e-12
    assert abs(factors[1] - div_f) < 1e-12


def test_vectorized_two_splits_same_asset_compound():
    """Two splits on one asset compound multiplicatively on prior history."""
    prices = _prices([12.0, 12.0, 6.0, 6.0, 2.0])  # dates Jan 2..6
    ca = _ca_full(
        [
            {"ex_date": date(2020, 1, 4), "id": 0, "action_type": "split", "split_ratio": 2.0},
            {"ex_date": date(2020, 1, 6), "id": 0, "action_type": "split", "split_ratio": 3.0},
        ]
    )
    adj = adjust_prices(prices, ca).prices.sort("date")["adj_close"].to_numpy()
    # Before Jan 4: both splits apply → factor 1/6.
    assert abs(adj[0] - 12.0 / 6.0) < 1e-9
    assert abs(adj[1] - 12.0 / 6.0) < 1e-9
    # Jan 4 and Jan 5 (>= first split, < second): only the 3-for-1 applies → 1/3.
    assert abs(adj[2] - 6.0 / 3.0) < 1e-9
    assert abs(adj[3] - 6.0 / 3.0) < 1e-9
    # Jan 6 (>= both ex_dates): unchanged.
    assert abs(adj[4] - 2.0) < 1e-9


def test_vectorized_tail_action_after_last_session():
    """An ex_date after the asset's last session back-adjusts the whole segment."""
    prices = _prices([10.0, 10.0, 10.0])  # Jan 2..4
    ca = _ca_full(
        [{"ex_date": date(2020, 2, 1), "id": 0, "action_type": "split", "split_ratio": 2.0}]
    )
    adj = adjust_prices(prices, ca).prices.sort("date")["adj_close"].to_numpy()
    np.testing.assert_allclose(adj, [5.0, 5.0, 5.0], rtol=0, atol=1e-9)


def test_vectorized_tail_action_not_logged_when_no_factor_change_is_logged():
    """A tail action with a real factor IS recorded in the audit log."""
    prices = _prices([10.0, 10.0])
    ca = _ca_full(
        [{"ex_date": date(2020, 3, 1), "id": 0, "action_type": "split", "split_ratio": 2.0}]
    )
    result = adjust_prices(prices, ca)
    assert result.adj_log.height == 1
    assert abs(result.adj_log["factor"][0] - 0.5) < 1e-12


def test_vectorized_action_before_first_session_not_applied_or_logged():
    """ex_date before the first price has no prior row: no change, no log entry."""
    prices = _prices([10.0, 11.0])  # Jan 2, Jan 3
    ca = _ca_full(
        [{"ex_date": date(2019, 1, 1), "id": 0, "action_type": "split", "split_ratio": 2.0}]
    )
    result = adjust_prices(prices, ca)
    np.testing.assert_array_equal(result.prices.sort("date")["adj_close"].to_numpy(), [10.0, 11.0])
    assert result.adj_log.is_empty()


def test_vectorized_asset_without_actions_untouched_amid_others():
    """An asset with no actions is unaffected by a split on a neighbour id."""
    p0 = _prices([10.0, 10.0, 5.0], asset_id=0)
    p1 = _prices([20.0, 21.0, 22.0], asset_id=1)
    prices = pl.concat([p0, p1])
    ca = _ca_full(
        [{"ex_date": date(2020, 1, 4), "id": 0, "action_type": "split", "split_ratio": 2.0}]
    )
    result = adjust_prices(prices, ca)
    a1 = result.prices.filter(pl.col("id") == 1).sort("date")["adj_close"].to_numpy()
    np.testing.assert_array_equal(a1, [20.0, 21.0, 22.0])


def test_vectorized_log_schema_matches_builder():
    """adj_log carries the documented dtypes regardless of which path built it."""
    prices = _prices([10.0, 10.0, 5.0])
    ca = _ca_full(
        [{"ex_date": date(2020, 1, 4), "id": 0, "action_type": "split", "split_ratio": 2.0}]
    )
    log = adjust_prices(prices, ca).adj_log
    empty = _build_adj_log([])
    assert log.schema["id"] == empty.schema["id"]
    assert log.schema["action_type"] == empty.schema["action_type"]
    assert log.schema["factor"] == empty.schema["factor"]


# ------------------------------------------------------------------ #
# include_quality_flags — additive, flag-off byte-identical
# ------------------------------------------------------------------ #


def test_flag_off_is_byte_identical_to_default():
    """With the flag off (and at its default) the returned frame is unchanged."""
    p0 = _prices([10.0, 10.0, 5.0], asset_id=0)
    p1 = _prices([20.0, 21.0, 22.0], asset_id=1)
    prices = pl.concat([p0, p1])
    ca = _ca_full(
        [{"ex_date": date(2020, 1, 4), "id": 0, "action_type": "split", "split_ratio": 2.0}]
    )
    default = adjust_prices(prices, ca)
    explicit_off = adjust_prices(prices, ca, include_quality_flags=False)
    assert default.prices.equals(explicit_off.prices)
    assert default.prices.columns == explicit_off.prices.columns
    assert default.adj_log.equals(explicit_off.adj_log)


def test_flag_on_appends_columns_without_disturbing_existing():
    """Flag on: original columns + adj_close are untouched; flags appended after."""
    p0 = _prices([10.0, 10.0, 5.0], asset_id=0)
    p1 = _prices([20.0, 21.0, 22.0], asset_id=1)
    prices = pl.concat([p0, p1])
    ca = _ca_full(
        [{"ex_date": date(2020, 1, 4), "id": 0, "action_type": "split", "split_ratio": 2.0}]
    )
    off = adjust_prices(prices, ca).prices
    on = adjust_prices(prices, ca, include_quality_flags=True).prices

    # Same row count.
    assert on.height == off.height
    # Original columns are a prefix of the flagged frame, value-identical.
    assert on.columns[: len(off.columns)] == off.columns
    assert on.select(off.columns).equals(off)
    # Flags appended in documented order.
    assert on.columns[len(off.columns) :] == list(QUALITY_FLAG_COLUMNS)


def test_flag_on_each_flag_fires_on_its_synthetic_condition():
    """A panel engineered to trip every flag: each fires on the right rows."""
    base = date(2020, 1, 2)
    dates = [base + timedelta(days=i) for i in range(3)]
    n_ids = 20  # enough assets for the cross-sectional z of the spike to clear 4.0
    rows = []
    for d in dates:
        for i in range(n_ids):
            rows.append({"date": d, "id": i, "close": float(i + 1) + 0.01 * d.day})
    df = pl.DataFrame(rows, schema={"date": pl.Date, "id": pl.Int64, "close": pl.Float64})

    # id 5: frozen flat series → is_frozen_series + price_stale on its 2nd/3rd rows.
    df = df.with_columns(
        pl.when(pl.col("id") == 5).then(pl.lit(42.0)).otherwise(pl.col("close")).alias("close")
    )
    # id 0 on day 0: huge spike → outlier_flagged.
    df = df.with_columns(
        pl.when((pl.col("id") == 0) & (pl.col("date") == dates[0]))
        .then(pl.lit(9999.0))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    # Drop id 1 on day 1 → sparse_coverage for id 1.
    df = df.filter(~((pl.col("id") == 1) & (pl.col("date") == dates[1])))
    # Duplicate one (date, id) key → is_duplicate_key.
    dup_row = df.filter((pl.col("id") == 2) & (pl.col("date") == dates[0]))
    df = pl.concat([df, dup_row])

    # adjust_prices threads include_quality_flags through (default thresholds):
    # row count preserved and the flag columns are appended.
    threaded = adjust_prices(df, _no_actions(), include_quality_flags=True).prices
    assert threaded.height == df.height
    assert threaded.columns[-len(QUALITY_FLAG_COLUMNS) :] == list(QUALITY_FLAG_COLUMNS)

    # Each flag's firing is asserted via the underlying annotate_quality_flags
    # with the tuned spike threshold + expected grid the default path omits.
    out = annotate_quality_flags(
        adjust_prices(df, _no_actions()).prices,
        "close",
        expected_dates=dates,
        expected_ids=list(range(n_ids)),
        spike_z_threshold=4.0,
    )

    assert out.filter(pl.col("is_frozen_series"))["id"].unique().to_list() == [5]
    # id 5 is flat → its repeat rows are stale (the duplicated id-2 row also
    # equals its prior obs, so it legitimately stale-matches too).
    assert 5 in out.filter(pl.col("price_stale"))["id"].unique().to_list()
    assert out.filter(pl.col("id") == 5).filter(pl.col("price_stale")).height >= 1
    assert (
        out.filter(pl.col("outlier_flagged"))
        .filter((pl.col("id") == 0) & (pl.col("date") == dates[0]))
        .height
        >= 1
    )
    assert out.filter(pl.col("sparse_coverage"))["id"].unique().to_list() == [1]
    assert out.filter(pl.col("is_duplicate_key"))["id"].unique().to_list() == [2]
