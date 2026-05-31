from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

from backtest.signals import SignalFrame
from etl.source import to_matrix

from .newey_west import newey_west_tstat


@dataclass(frozen=True)
class ICResult:
    ic_series: pl.DataFrame   # date, ic   — O(n_dates)
    mean_ic: float
    ic_ir: float              # mean_ic / std_ic
    t_stat: float             # Newey-West adjusted


def ic_series(signals: SignalFrame, returns: pl.DataFrame) -> pl.DataFrame:
    """IC of signal_t against forward return_{t+1}, one value per date.

    Continuous signals → Spearman rank correlation.
    Categorical/binary signals → point-biserial (Pearson on the 0/1 signal).
    Working set is O(n_assets) per date; output is O(n_dates).
    """
    S, _ = to_matrix(signals.df, "signal")
    R, dates = to_matrix(returns, "return")
    n_dates, _ = S.shape

    out_dates: list = []
    ics: list[float] = []
    for t in range(n_dates - 1):
        x = S[t]
        y = R[t + 1]
        if signals.is_categorical:
            ic = float(stats.pearsonr(x, y).statistic)
        else:
            ic = float(stats.spearmanr(x, y).statistic)
        out_dates.append(dates[t])
        ics.append(ic)

    return pl.DataFrame({"date": out_dates, "ic": ics})


def rolling_ic(ic: pl.DataFrame, window: int) -> pl.DataFrame:
    """Trailing-window mean IC. O(n_dates)."""
    return ic.with_columns(
        pl.col("ic").rolling_mean(window_size=window).alias("rolling_ic")
    ).select("date", "rolling_ic")


@dataclass(frozen=True)
class ICEvaluator:
    def evaluate(self, signals: SignalFrame, returns: pl.DataFrame) -> ICResult:
        ic = ic_series(signals, returns)
        s = ic["ic"]
        mean_ic = float(s.mean())
        std_ic = float(s.std() or 0.0)
        return ICResult(
            ic_series=ic,
            mean_ic=mean_ic,
            ic_ir=(mean_ic / std_ic if std_ic else 0.0),
            t_stat=newey_west_tstat(s),
        )
