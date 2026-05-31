from functools import cache
from datetime import date

import polars as pl

START_DATE = "2020-01-01"
END_DATE = "2022-12-31"


@cache
def dates(start: str = "2020-01-01", end: str | None = END_DATE) -> pl.DataFrame:
    start_date = date.fromisoformat(start)
    if end is None:
        end_date = date.today()
    else:
        end_date = date.fromisoformat(end)

    return pl.DataFrame(
        {
            "date": pl.date_range(
                start_date,
                end_date,
                interval="1d",
                closed="left",
                eager=True,
            )
        }
    )
