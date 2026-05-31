from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import SupportsFloat, cast

import numpy as np
import polars as pl


def to_float(x: object) -> float:
    """``float()`` of a scalar whose static type is an over-broad union.

    polars types its Series aggregations (``.min``/``.mean``/``.std``/…) as a
    union that includes temporal types, because the same method works on date
    and duration columns. On the numeric columns we feed them that union is a
    false positive; the ``cast`` narrows it for the type checker — it asserts
    one type, it does not hide other errors the way a blanket ignore would.
    """
    return float(cast(SupportsFloat, x))


def to_matrix(df: pl.DataFrame, value_col: str) -> tuple[np.ndarray, list[date]]:
    """Pivot long (date, id, value) → dense (n_dates, n_assets) matrix.

    Sorting by (date, id) before the pivot guarantees ascending-id column
    order, so matrices built from different long frames stay aligned. The
    pivot is itself an O(n_dates * n_assets) allocation.
    """
    wide = df.sort(["date", "id"]).pivot(on="id", index="date", values=value_col)
    dates = wide["date"].to_list()
    mat = wide.drop("date").to_numpy()
    return mat, dates


def date_axis(n_dates: int, start: str = "2000-01-01") -> pl.Series:
    start_d = date.fromisoformat(start)
    end_d = start_d + timedelta(days=n_dates - 1)
    return pl.date_range(start_d, end_d, interval="1d", eager=True)


def session_axis(n_dates: int, start: str = "2000-01-01") -> pl.Series:
    """Trading-day axis: `n_dates` consecutive weekday (Mon–Fri) sessions.

    `date_axis` walks calendar days, which silently puts weekends in the panel
    and makes "next day" forward returns and 252-day annualization
    inconsistent. Production panels use this business-day axis instead. (Public
    holidays are modeled separately via the `trading_calendar` dataset.)
    """
    start_d = date.fromisoformat(start)
    span = pl.date_range(
        start_d, start_d + timedelta(days=2 * n_dates + 7), interval="1d", eager=True
    )
    sessions = span.filter(span.dt.weekday() <= 5)
    return sessions.head(n_dates)


def generate_returns(
    n_assets: int,
    n_dates: int,
    start: str = "2000-01-01",
    seed: int | None = None,
) -> pl.DataFrame:
    """Long-format synthetic returns: (date, id, return). O(n_assets * n_dates) rows."""
    dates = date_axis(n_dates, start)
    ids = pl.int_range(0, n_assets, eager=True)
    df = pl.DataFrame({"date": dates}).join(pl.DataFrame({"id": ids}), how="cross")
    rng = np.random.default_rng(seed)
    return df.with_columns(pl.Series("return", rng.normal(0.0, 3.0, len(df))))


def write_parquet(
    path: Path,
    n_assets: int,
    n_dates: int,
    start: str = "2000-01-01",
    seed: int | None = None,
) -> Path:
    """Materialize a synthetic source to disk, modeling an on-disk/S3 dataset."""
    generate_returns(n_assets, n_dates, start, seed).write_parquet(path)
    return path
