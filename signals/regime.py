"""Market-regime detection and regime-conditional signal evaluation.

Two public APIs:

``detect_regimes(returns, n_regimes, window)``
    Infers a per-date regime label from a univariate returns series (e.g.
    market index or cross-sectional mean return) using rolling realized
    volatility quantile-binning.  No ground truth is required — fully
    unsupervised and dependency-free (only numpy/polars).

``regime_conditional_ic(signals, forward_returns, regimes, ...)``
    Slices the panel by regime label and computes signal IC within each
    regime, reusing ``ic_series_v2`` from this package.  Returns a frozen
    dataclass with per-regime IC, observation counts, and the spread between
    the best and worst regime.

Design choices:
- No new dependencies: rolling vol + np.digitize is sufficient.
- The regime label is an integer starting at 0 (ascending realized-vol order
  so label 0 = low-vol, label n_regimes-1 = high-vol).
- The result dataclass is frozen so it can be passed as a value object
  without defensive copying.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .ic import ICMethod, ic_series_v2


@dataclass(frozen=True)
class RegimeConditionalICResult:
    """Per-regime IC summary from ``regime_conditional_ic``.

    Attributes
    ----------
    per_regime_ic:
        Mean IC within each regime, keyed by integer regime label.
    per_regime_n_obs:
        Total cross-sectional observations used within each regime
        (sum of n_obs across all dates in that regime).
    regime_spread:
        IC of the best regime minus IC of the worst regime.  A large
        positive spread means the signal works in some regimes but not
        others.
    method:
        IC correlation method used.
    """

    per_regime_ic: dict[int, float]
    per_regime_n_obs: dict[int, int]
    regime_spread: float
    method: ICMethod


def detect_regimes(
    returns: pl.Series,
    *,
    n_regimes: int = 3,
    window: int = 21,
    min_window: int = 5,
) -> pl.Series:
    """Assign a per-date regime label from a univariate returns series.

    Computes trailing realized volatility (annualized daily std over
    ``window`` sessions) then bins dates into ``n_regimes`` quantile-equal
    buckets (ascending vol order).  Dates with fewer than ``min_window``
    valid observations in their trailing window receive label -1 (warm-up).

    Parameters
    ----------
    returns:
        Univariate float series of per-date returns (NOT percent — or
        consistently in any unit, since we only compare magnitudes).
    n_regimes:
        Number of volatility regimes to create (default 3: low/mid/high).
    window:
        Rolling window length (trading days) for realized vol.
    min_window:
        Minimum observations required in a rolling window to emit a valid
        vol estimate.  Dates below this get label -1.

    Returns
    -------
    Polars Int64 Series of regime labels (same length as ``returns``).
    Label -1 = warm-up / insufficient data; 0 … n_regimes-1 = regimes
    in ascending realized-volatility order.
    """
    arr = returns.to_numpy().astype(float)
    n = len(arr)
    rvol = np.full(n, np.nan)
    for t in range(n):
        start = max(0, t - window + 1)
        window_data = arr[start : t + 1]
        if len(window_data) >= min_window:
            rvol[t] = float(np.std(window_data, ddof=1))

    # Compute quantile edges on all valid vol values
    valid_mask = np.isfinite(rvol)
    labels = np.full(n, -1, dtype=np.int64)
    if valid_mask.sum() >= n_regimes:
        edges = np.quantile(rvol[valid_mask], np.linspace(0.0, 1.0, n_regimes + 1))
        # np.digitize bins: 1-indexed → subtract 1, clip to [0, n_regimes-1]
        raw = np.digitize(rvol[valid_mask], edges[1:-1])  # 0 … n_regimes-1
        labels[valid_mask] = raw.astype(np.int64)

    return pl.Series("regime_label", labels, dtype=pl.Int64)


def regime_conditional_ic(
    signals: pl.DataFrame,
    forward_returns: pl.DataFrame,
    regimes: pl.DataFrame,
    *,
    signal_col: str = "signal",
    return_col: str,
    method: ICMethod = "rank",
    min_obs: int = 10,
) -> RegimeConditionalICResult:
    """Compute signal IC within each market regime.

    Slices ``signals`` and ``forward_returns`` to dates belonging to each
    regime label, then calls ``ic_series_v2`` on each slice.  Regime -1
    (warm-up) is skipped.

    Parameters
    ----------
    signals:
        Long-format (date, id, signal_col).
    forward_returns:
        Long-format (date, id, return_col).
    regimes:
        Per-date regime labels with columns (date, regime_label).  Typically
        produced by ``detect_regimes`` joined to a date axis, or loaded from
        the ``regime_states`` dataset (column ``regime_state`` renamed).
    signal_col:
        Name of the signal column in ``signals``.
    return_col:
        Name of the return column in ``forward_returns``.
    method:
        IC correlation method — "rank" (Spearman), "pearson", or "kendall".
    min_obs:
        Minimum paired observations per date to retain an IC estimate (passed
        through to ``ic_series_v2``).

    Returns
    -------
    RegimeConditionalICResult with per-regime mean IC, observation counts,
    and the best-minus-worst regime IC spread.
    """
    label_col = "regime_label" if "regime_label" in regimes.columns else "regime_state"
    regime_labels = regimes.select("date", pl.col(label_col).alias("_regime"))

    # Merge regime labels into signals to get a filtered date set per regime
    sig_with_regime = signals.join(regime_labels, on="date", how="left")
    fwd_with_regime = forward_returns.join(regime_labels, on="date", how="left")

    unique_labels = sorted(
        v for v in sig_with_regime["_regime"].unique().to_list() if v is not None and v >= 0
    )

    per_regime_ic: dict[int, float] = {}
    per_regime_n_obs: dict[int, int] = {}

    for label in unique_labels:
        sig_slice = sig_with_regime.filter(pl.col("_regime") == label).drop("_regime")
        fwd_slice = fwd_with_regime.filter(pl.col("_regime") == label).drop("_regime")

        if sig_slice.is_empty() or fwd_slice.is_empty():
            per_regime_ic[label] = float("nan")
            per_regime_n_obs[label] = 0
            continue

        ic_df = ic_series_v2(
            sig_slice,
            fwd_slice,
            signal_col=signal_col,
            return_col=return_col,
            method=method,
            min_obs=min_obs,
        )

        valid = ic_df["ic"].drop_nulls().to_numpy()
        per_regime_ic[label] = float(np.nanmean(valid)) if len(valid) > 0 else float("nan")
        total_obs = int(ic_df["n_obs"].sum())
        per_regime_n_obs[label] = total_obs

    # Regime spread: best IC minus worst IC (ignoring NaN)
    finite_ics = [v for v in per_regime_ic.values() if np.isfinite(v)]
    if len(finite_ics) >= 2:
        regime_spread = float(max(finite_ics) - min(finite_ics))
    elif len(finite_ics) == 1:
        regime_spread = 0.0
    else:
        regime_spread = float("nan")

    return RegimeConditionalICResult(
        per_regime_ic=per_regime_ic,
        per_regime_n_obs=per_regime_n_obs,
        regime_spread=regime_spread,
        method=method,
    )
