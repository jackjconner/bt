from functools import cache
import polars as pl

from .dates import dates, START_DATE, END_DATE

N_CONSTITUENTS = 500


@cache
def constituents(n: int = N_CONSTITUENTS, start: str = START_DATE, end: str = END_DATE) -> pl.DataFrame:
    return dates(start, end).join(
        pl.DataFrame({"id": pl.int_range(0, n, eager=True)}),
        how="cross",
    )
