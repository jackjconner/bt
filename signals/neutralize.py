"""Cross-sectional neutralization of signals.

A raw signal often loads on incidental tilts (sector, size, beta) that are
not the intended source of alpha.  Neutralization residualises the signal
against those tilts so the IC measures only what the signal adds *beyond*
the common factors.

Two neutralization modes are supported:

1. **Sector dummies** — OLS of signal on one-hot sector dummies, keep residual.
   Removes any cross-sector mean difference in the signal.

2. **Factor loadings** — OLS of signal on a (date, id, factor) matrix, keep
   residual.  Removes the portion of the signal that can be explained by the
   provided factors (e.g. size, value, beta from a risk model).

Both modes can be combined; the residual signal is then both sector-neutral and
factor-neutral.  The function also returns a ``NeutralizationResult`` that
reports raw IC vs neutralized IC so the caller can see how much of the original
IC survived neutralization.

Implementation: per-date cross-sectional OLS via the normal equations.  This is
O(n_assets * n_factors²) per date, which is negligible for typical panel sizes.
NaN / coverage: assets missing signal *or* any regressor are excluded from that
date's regression; assets with a valid residual are kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from etl.source import to_matrix

from .ic import ICMethod, ic_series_v2


@dataclass(frozen=True)
class NeutralizationResult:
    """Comparison of raw IC vs neutralized IC."""

    raw_ic_mean: float
    neutralized_ic_mean: float

    raw_ic_ir: float
    neutralized_ic_ir: float

    raw_t_stat: float
    neutralized_t_stat: float

    neutralized_signals: pl.DataFrame
    """Long-format (date, id, signal) of the residualized signal."""


def _ols_residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cross-sectional OLS residual for one date, with missing-data handling.

    y: (n,) signal values, may contain NaN.
    X: (n, k) regressor matrix, may contain NaN rows.

    Returns: (n,) residual array.  Rows where y or any X column is NaN get
    NaN in the residual (they were not used in the fit).
    """
    n = len(y)
    out = np.full(n, np.nan)

    # Pairwise-complete: row is valid only if y and all X columns are finite
    row_ok = np.isfinite(y)
    for j in range(X.shape[1]):
        row_ok = row_ok & np.isfinite(X[:, j])

    if row_ok.sum() < X.shape[1] + 1:
        # Not enough observations to fit; return NaN residuals
        return out

    yv, Xv = y[row_ok], X[row_ok]
    # OLS: beta = (X'X)^-1 X'y via least-squares
    beta, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid_v = yv - Xv @ beta
    # Z-score residuals cross-sectionally to keep signal scale consistent.
    # Use a tolerance so near-zero std (perfect fit or degenerate case) does
    # not amplify floating-point noise through division.
    std = resid_v.std()
    scale = np.abs(yv).mean() if np.abs(yv).mean() > 0 else 1.0
    if std > 1e-8 * scale:
        resid_v = resid_v / std
    else:
        resid_v = np.zeros_like(resid_v)
    out[row_ok] = resid_v
    return out


def neutralize_sector(
    signals: pl.DataFrame,
    security_master: pl.DataFrame,
    *,
    signal_col: str = "signal",
    sector_col: str = "sector",
) -> pl.DataFrame:
    """Residualize signal against sector dummies, one date at a time.

    Returns long-format (date, id, signal) where the signal is the
    cross-sectional OLS residual after projecting out sector means.

    Parameters
    ----------
    signals:
        Long-format (date, id, signal_col).
    security_master:
        Static asset reference with (id, sector_col).  Sector is assumed
        time-invariant here (no time dimension in the security master).
    signal_col:
        Column in ``signals`` to neutralize.
    sector_col:
        Categorical sector column in ``security_master``.
    """
    # Attach sector to signals
    sm = security_master.select("id", sector_col)
    joined = signals.join(sm, on="id", how="left")

    # Enumerate sectors as integer codes for dummy construction
    sectors = joined[sector_col].unique().sort().to_list()
    sector_to_idx = {s: i for i, s in enumerate(sectors)}
    n_sectors = len(sectors)

    # pivot to wide for vectorised processing
    S, s_dates = to_matrix(joined.select("date", "id", signal_col), signal_col)
    ids = sorted(joined["id"].unique().to_list())

    # Build (n_dates, n_assets, n_sectors) dummy array — static per asset
    id_to_col = {v: i for i, v in enumerate(ids)}
    sector_arr = np.zeros((len(ids), n_sectors))
    for row in sm.iter_rows(named=True):
        col_idx = id_to_col.get(row["id"])
        if col_idx is not None and row[sector_col] is not None:
            sector_arr[col_idx, sector_to_idx[row[sector_col]]] = 1.0

    out_rows: list[dict] = []
    for t, d in enumerate(s_dates):
        y = S[t]
        resid = _ols_residual(y, sector_arr)
        for asset_col, asset_id in enumerate(ids):
            out_rows.append({"date": d, "id": asset_id, signal_col: resid[asset_col]})

    return pl.DataFrame(out_rows).with_columns(
        pl.col("id").cast(pl.Int64),
        pl.col(signal_col).cast(pl.Float64),
    )


def neutralize_factors(
    signals: pl.DataFrame,
    factor_loadings: pl.DataFrame,
    *,
    signal_col: str = "signal",
    factor_id_col: str = "factor_id",
    loading_col: str = "loading",
) -> pl.DataFrame:
    """Residualize signal against factor loadings, one date at a time.

    Returns long-format (date, id, signal) where the signal is the residual
    after projecting out the factor-loading space.

    Parameters
    ----------
    signals:
        Long-format (date, id, signal_col).
    factor_loadings:
        Long-format (date, id, factor_id, loading).  Time-varying loadings
        are supported; the loadings are joined to signals on (date, id).
    """
    # Pivot factor loadings: (date, id) → vector of factor loadings
    factor_ids = sorted(factor_loadings[factor_id_col].unique().to_list())
    n_factors = len(factor_ids)
    fid_to_col = {fid: i for i, fid in enumerate(factor_ids)}

    # Build dict: (date, id) → loadings array
    loading_map: dict[tuple, np.ndarray] = {}
    for row in factor_loadings.iter_rows(named=True):
        key = (row["date"], row["id"])
        if key not in loading_map:
            loading_map[key] = np.zeros(n_factors)
        col = fid_to_col.get(row[factor_id_col])
        if col is not None:
            loading_map[key][col] = row[loading_col]

    S, s_dates = to_matrix(signals.select("date", "id", signal_col), signal_col)
    ids = sorted(signals["id"].unique().to_list())
    id_to_col = {v: i for i, v in enumerate(ids)}

    out_rows: list[dict] = []
    for t, d in enumerate(s_dates):
        y = S[t]
        # Build factor matrix for this date: (n_assets, n_factors)
        X = np.full((len(ids), n_factors), np.nan)
        for aid in ids:
            ci = id_to_col[aid]
            key = (d, aid)
            if key in loading_map:
                X[ci] = loading_map[key]

        resid = _ols_residual(y, X)
        for asset_col, asset_id in enumerate(ids):
            out_rows.append({"date": d, "id": asset_id, signal_col: resid[asset_col]})

    return pl.DataFrame(out_rows).with_columns(
        pl.col("id").cast(pl.Int64),
        pl.col(signal_col).cast(pl.Float64),
    )


def evaluate_neutralization(
    raw_signals: pl.DataFrame,
    neutralized_signals: pl.DataFrame,
    forward_returns: pl.DataFrame,
    *,
    signal_col: str = "signal",
    return_col: str,
    method: ICMethod = "rank",
    min_obs: int = 10,
) -> NeutralizationResult:
    """Compute raw vs neutralized IC and return a NeutralizationResult.

    Useful for reporting how much IC survives neutralization — if the IC drops
    dramatically, the original signal was primarily a sector or factor bet
    rather than a genuine idiosyncratic alpha.
    """
    from .newey_west import newey_west_tstat

    def _summarize(df: pl.DataFrame) -> tuple[float, float, float]:
        ic_df = ic_series_v2(
            df, forward_returns,
            signal_col=signal_col, return_col=return_col,
            method=method, min_obs=min_obs,
        )
        s = ic_df["ic"].drop_nulls()
        arr = s.to_numpy()
        if len(arr) == 0:
            return float("nan"), 0.0, 0.0
        mean_ic = float(np.nanmean(arr))
        std_ic = float(np.nanstd(arr))
        ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
        t_stat = float(newey_west_tstat(s))
        return mean_ic, ic_ir, t_stat

    raw_mean, raw_ir, raw_t = _summarize(raw_signals)
    neu_mean, neu_ir, neu_t = _summarize(neutralized_signals)

    return NeutralizationResult(
        raw_ic_mean=raw_mean,
        neutralized_ic_mean=neu_mean,
        raw_ic_ir=raw_ir,
        neutralized_ic_ir=neu_ir,
        raw_t_stat=raw_t,
        neutralized_t_stat=neu_t,
        neutralized_signals=neutralized_signals,
    )
