from __future__ import annotations
from datetime import date

import numpy as np
import polars as pl


def returns(
    ids: pl.Series, start: str | date, end: str | date | None = None
) -> pl.DataFrame:
    start = date.fromisoformat(start) if isinstance(start, str) else start
    end = date.fromisoformat(end) if isinstance(end, str) else (end or date.today())
    date_range = pl.date_range(start, end, interval="1d", eager=True)
    df = pl.DataFrame({"date": date_range}).join(
        pl.DataFrame({"id": ids}), how="cross"
    )
    return df.with_columns(
        pl.Series("return", np.random.normal(0.0, 3.0, len(df)))
    )


def cumulative_returns() -> pl.Expr:
    return pl.col("return_1d").log().cum_sum().exp().alias("cumulative_return_1d")


def rolling_return(n: int, column: str = "return_1d") -> pl.Expr:
    return (
        pl.col(column)
        .log()
        .sum()
        .exp()
        .rolling(index_column="date", period=f"{n}d")
        .alias(f"rolling_{column.split('return_')[0]}{n}d")
    )
