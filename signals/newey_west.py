from __future__ import annotations

import numpy as np
import polars as pl


def default_lags(n_obs: int) -> int:
    """Data-driven lag rule: floor(4 * (T/100)^(2/9))."""
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


def newey_west_tstat(ic_series: pl.Series, n_lags: int | None = None) -> float:
    """t-stat for the mean IC using a Bartlett-kernel HAC variance.

    Corrects the standard error of the mean for serial correlation in the
    rolling IC series. O(n_obs * n_lags).
    """
    x = ic_series.to_numpy()
    n = len(x)
    if n_lags is None:
        n_lags = default_lags(n)

    mu = float(x.mean())
    e = x - mu
    var = float(e @ e) / n  # gamma_0
    for lag in range(1, n_lags + 1):
        weight = 1.0 - lag / (n_lags + 1)
        cov = float(e[lag:] @ e[:-lag]) / n
        var += 2.0 * weight * cov

    se = np.sqrt(var / n)
    if se == 0:
        return 0.0
    return mu / se
