"""Tests for survivorship-bias-free universe resolution."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from .universe import apply_security_master, resolve_universe


def _mask_df() -> pl.DataFrame:
    """3 dates × 3 assets, some holes."""
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    ids = [0, 1, 2]
    rows = []
    for d in dates:
        for i in ids:
            rows.append(
                {
                    "date": d,
                    "id": i,
                    "in_universe": True,
                    "tradable": True,
                    "halted": False,
                    "listed": True,
                }
            )
    df = pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "id": pl.Int64,
            "in_universe": pl.Boolean,
            "tradable": pl.Boolean,
            "halted": pl.Boolean,
            "listed": pl.Boolean,
        },
    )
    # Asset 2 is not in universe on date 2
    return df.with_columns(
        pl.when((pl.col("id") == 2) & (pl.col("date") == date(2020, 1, 3)))
        .then(pl.lit(False))
        .otherwise(pl.col("in_universe"))
        .alias("in_universe")
    )


def test_resolve_universe_shape():
    df = _mask_df()
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    ids = [0, 1, 2]
    mask = resolve_universe(df, dates, ids)
    assert mask.shape == (3, 3)
    assert mask.dtype == bool


def test_resolve_universe_missing_cell_is_false():
    df = _mask_df()
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    ids = [0, 1, 2]
    mask = resolve_universe(df, dates, ids)
    # (date index 1, id index 2) should be False
    assert not mask[1, 2]
    # All other cells are True
    assert mask[0, 0] and mask[0, 1] and mask[0, 2]


def test_resolve_universe_unknown_flag_raises():
    df = _mask_df()
    with pytest.raises(ValueError, match="no_such_flag"):
        resolve_universe(df, [], [], flag="no_such_flag")


def test_resolve_universe_tradable_flag():
    df = _mask_df().with_columns(
        pl.when(pl.col("id") == 1)
        .then(pl.lit(False))
        .otherwise(pl.col("tradable"))
        .alias("tradable")
    )
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    ids = [0, 1, 2]
    mask = resolve_universe(df, dates, ids, flag="tradable")
    # Asset 1 should be False in all rows
    assert not mask[:, 1].any()


def test_resolve_universe_extra_date_not_in_expected_ignored():
    df = _mask_df()
    dates = [date(2020, 1, 2)]  # only request first date
    ids = [0, 1, 2]
    mask = resolve_universe(df, dates, ids)
    assert mask.shape == (1, 3)


def test_apply_security_master_forces_out_of_window_to_false():
    df = _mask_df()
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    sm = pl.DataFrame(
        {
            "id": pl.Series([0, 1, 2], dtype=pl.Int64),
            "listing_date": pl.Series([date(2020, 1, 2)] * 3, dtype=pl.Date),
            # Asset 0 delists after Jan 2
            "delisting_date": pl.Series([date(2020, 1, 2), None, None], dtype=pl.Date),
        }
    )
    result = apply_security_master(df, sm, dates)
    # Asset 0 should only be in_universe on Jan 2
    asset0 = result.filter(pl.col("id") == 0)
    on_jan2 = asset0.filter(pl.col("date") == date(2020, 1, 2))["in_universe"][0]
    after_jan2 = asset0.filter(pl.col("date") > date(2020, 1, 2))["in_universe"]
    assert on_jan2
    assert not after_jan2.any()


def test_apply_security_master_preserves_other_columns():
    df = _mask_df()
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    sm = pl.DataFrame(
        {
            "id": pl.Series([0, 1, 2], dtype=pl.Int64),
            "listing_date": pl.Series([date(2020, 1, 1)] * 3, dtype=pl.Date),
            "delisting_date": pl.Series([None, None, None], dtype=pl.Date),
        }
    )
    result = apply_security_master(df, sm, dates)
    assert "halted" in result.columns
    assert "tradable" in result.columns
