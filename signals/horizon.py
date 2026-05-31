"""IC decay / horizon curve.

``ic_horizon_curve`` computes mean IC (and IC-IR) over a grid of forward-return
horizons, producing an IC(h) decay curve.  The shape of the curve tells you:

- where the signal's predictive power peaks (optimal holding period),
- how quickly it decays (guides rebalance frequency),
- whether it is driven by short-term microstructure or longer-term fundamentals.

Design: forward returns for every horizon must already be materialised in the
``forward_returns`` DataFrame so we avoid repeating compound-return arithmetic.
The ``forward_returns`` dataset in ``etl.datasets`` provides columns
``fwd_ret_1``, ``fwd_ret_5``, ``fwd_ret_21``, ``fwd_ret_63``; callers supply
the column names they want to sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from etl.source import to_matrix

from .ic import ICEngine, ICMethod, _ic_series_from_matrices
from .lazy_ic import spearman_ic_lazy
from .newey_west import newey_west_tstat


@dataclass(frozen=True)
class HorizonPoint:
    """Summary statistics for a single horizon."""

    horizon: int  # trading-day horizon label (e.g. 1, 5, 21, 63)
    mean_ic: float
    ic_ir: float
    t_stat: float
    n_dates: int  # number of dates with valid IC estimates


@dataclass(frozen=True)
class HorizonCurve:
    """IC decay curve across a grid of horizons."""

    points: tuple[HorizonPoint, ...]  # sorted by ascending horizon
    method: ICMethod

    def horizons(self) -> list[int]:
        return [p.horizon for p in self.points]

    def mean_ics(self) -> list[float]:
        return [p.mean_ic for p in self.points]

    def ic_irs(self) -> list[float]:
        return [p.ic_ir for p in self.points]

    def to_frame(self) -> pl.DataFrame:
        """Long-format summary DataFrame for all horizons."""
        return pl.DataFrame(
            {
                "horizon": [p.horizon for p in self.points],
                "mean_ic": [p.mean_ic for p in self.points],
                "ic_ir": [p.ic_ir for p in self.points],
                "t_stat": [p.t_stat for p in self.points],
                "n_dates": [p.n_dates for p in self.points],
            }
        )


def ic_horizon_curve(
    signals: pl.DataFrame,
    forward_returns: pl.DataFrame,
    horizon_cols: dict[int, str],
    *,
    signal_col: str = "signal",
    method: ICMethod = "rank",
    min_obs: int = 10,
    engine: ICEngine = "lazy",
) -> HorizonCurve:
    """Compute mean IC over a grid of forward-return horizons.

    Parameters
    ----------
    signals:
        Long-format (date, id, signal_col).
    forward_returns:
        Long-format (date, id, fwd_ret_1, fwd_ret_5, ...).
    horizon_cols:
        Mapping from integer horizon label to the column name in
        ``forward_returns``, e.g. ``{1: "fwd_ret_1", 5: "fwd_ret_5"}``.
    signal_col:
        Name of the signal column in ``signals``.
    method:
        IC correlation method — "rank" (Spearman), "pearson", or "kendall".
    min_obs:
        Minimum paired observations per date to retain an IC estimate.
    engine:
        Spearman backend for ``method="rank"``: ``"lazy"`` (default) streams
        the per-date IC in long-format Polars without pivoting to a dense
        matrix; ``"matrix"`` uses the incumbent pivot + ``scipy.rankdata``
        path.  Bit-identical results; ``"lazy"`` is faster across the grid.
        Ignored for the pearson / kendall methods (always matrix).

    Returns
    -------
    HorizonCurve with one HorizonPoint per entry in ``horizon_cols``.
    """
    use_lazy = method == "rank" and engine == "lazy"
    # Matrix path pre-computes the signal pivot once; each horizon reuses it.
    if not use_lazy:
        S, s_dates = to_matrix(signals.select("date", "id", signal_col), signal_col)

    points = []
    for h, col in sorted(horizon_cols.items()):
        if use_lazy:
            ic_df = spearman_ic_lazy(
                signals,
                forward_returns,
                signal_col=signal_col,
                return_col=col,
                min_obs=min_obs,
            )
        else:
            R, r_dates = to_matrix(forward_returns.select("date", "id", col), col)
            ic_df = _ic_series_from_matrices(S, s_dates, R, r_dates, method, min_obs)
        s = ic_df["ic"].drop_nulls()
        arr = s.to_numpy()
        if len(arr) == 0:
            points.append(
                HorizonPoint(horizon=h, mean_ic=float("nan"), ic_ir=0.0, t_stat=0.0, n_dates=0)
            )
            continue
        mean_ic = float(np.nanmean(arr))
        std_ic = float(np.nanstd(arr))
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
        t_stat = float(newey_west_tstat(s))
        points.append(
            HorizonPoint(
                horizon=h,
                mean_ic=mean_ic,
                ic_ir=ic_ir,
                t_stat=t_stat,
                n_dates=len(arr),
            )
        )
    return HorizonCurve(points=tuple(points), method=method)
