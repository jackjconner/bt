from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from scipy import stats

from backtest.signals import SignalFrame
from etl.source import to_float, to_matrix

from .coverage import apply_min_coverage, pairwise_mask
from .newey_west import newey_west_tstat

ICMethod = Literal["rank", "pearson", "kendall"]


@dataclass(frozen=True)
class ICResult:
    ic_series: pl.DataFrame  # date, ic   — O(n_dates)
    mean_ic: float
    ic_ir: float  # mean_ic / std_ic
    t_stat: float  # Newey-West adjusted


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
        mean_ic = to_float(s.mean())
        std_ic = to_float(s.std() or 0.0)
        return ICResult(
            ic_series=ic,
            mean_ic=mean_ic,
            ic_ir=(mean_ic / std_ic if std_ic else 0.0),
            t_stat=newey_west_tstat(s),
        )


# ---------------------------------------------------------------------------
# Configurable horizon + method IC
# ---------------------------------------------------------------------------


def _cross_sectional_ic(
    x: np.ndarray,
    y: np.ndarray,
    method: ICMethod,
) -> tuple[float, int]:
    """Compute one cross-sectional IC value with pairwise-complete masking.

    Returns (ic, n_valid) where n_valid is the number of paired observations
    used.  Returns (nan, 0) when fewer than 2 valid pairs exist.

    Spearman / Kendall are monotone-rank correlations and measure whether the
    *ordering* of the signal predicts the *ordering* of returns, which is what
    a long-short book cares about.  Pearson measures linear co-movement and is
    appropriate when both signal and return are expected to be linearly related
    (e.g. a beta-adjusted expected-return estimate).
    """
    mask = pairwise_mask(x, y)
    n = int(mask.sum())
    if n < 2:
        return float("nan"), 0
    xm, ym = x[mask], y[mask]
    if method == "rank":
        ic = float(stats.spearmanr(xm, ym).statistic)
    elif method == "pearson":
        ic = float(stats.pearsonr(xm, ym).statistic)
    else:  # kendall
        ic = float(stats.kendalltau(xm, ym).statistic)
    return ic, n


def ic_series_v2(
    signals: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    signal_col: str = "signal",
    return_col: str,
    method: ICMethod = "rank",
    min_obs: int = 10,
) -> pl.DataFrame:
    """Per-date IC of a signal against a chosen forward-return column.

    Unlike the original ``ic_series``, this function:
    - Accepts raw long-format DataFrames (date, id, signal) and
      (date, id, <return_col>) rather than a ``SignalFrame``.
    - Takes an explicit ``return_col`` so any forward horizon can be used.
    - Supports ``method`` ∈ {"rank", "pearson", "kendall"} explicitly.
    - Applies pairwise-complete masking and ``min_obs`` suppression.

    Returns a DataFrame with columns ``(date, ic, n_obs)``.
    """
    S, s_dates = to_matrix(signals.select("date", "id", signal_col), signal_col)
    R, r_dates = to_matrix(forward_returns.select("date", "id", return_col), return_col)

    # Align on dates that appear in both
    s_set = {d: i for i, d in enumerate(s_dates)}
    r_set = {d: i for i, d in enumerate(r_dates)}
    common = sorted(set(s_set) & set(r_set))

    out_dates: list = []
    ics: list[float] = []
    ns: list[int] = []
    for d in common:
        x = S[s_set[d]]
        y = R[r_set[d]]
        ic, n = _cross_sectional_ic(x, y, method)
        out_dates.append(d)
        ics.append(ic)
        ns.append(n)

    ic_arr = apply_min_coverage(np.array(ics), np.array(ns), min_obs)
    # Convert float NaN → Polars null so drop_nulls() works correctly.
    # Polars stores NaN and null as distinct concepts; NaN is kept by drop_nulls().
    ic_list: list[float | None] = [None if np.isnan(v) else float(v) for v in ic_arr]
    return pl.DataFrame({"date": out_dates, "ic": ic_list, "n_obs": ns})
