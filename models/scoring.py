"""Cross-sectional and aggregate scoring metrics for financial models.

R² is near-zero for return-predicting models even when the signal is economically
large — a cross-sectional rank-IC (Spearman ρ between predicted rank and realised
rank on each date) is the right held-out metric.  A per-date IC series also lets
you examine regime stability.

Public API
----------
``rank_ic_score``     — scalar mean IC over all dates in the provided arrays.
``rank_ic_series``    — per-date IC Series with the date as index.
``ic_stats``          — dict of mean, std, IR (mean/std), and t-stat.
``held_out_r2``       — standard R² helper (already in cross_val.py but
                        re-exported here for completeness).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ between two 1-D arrays; returns 0.0 when degenerate."""
    if len(x) < 3:
        return 0.0
    rho, _ = stats.spearmanr(x, y)
    return float(rho) if np.isfinite(rho) else 0.0


def _pearson_on_ranks(Rx: np.ndarray, Ry: np.ndarray) -> np.ndarray:
    """Per-row Pearson ρ of two ``(n_dates, k)`` rank matrices.

    Spearman ρ is exactly Pearson on (average-tie) ranks, so this reproduces
    ``scipy.stats.spearmanr`` to machine precision.  Rows whose rank variance is
    zero (denominator 0 — a constant cross-section) yield 0.0, mirroring the
    ``_spearman`` non-finite → 0.0 contract.
    """
    rx_c = Rx - Rx.mean(axis=1, keepdims=True)
    ry_c = Ry - Ry.mean(axis=1, keepdims=True)
    numer = (rx_c * ry_c).sum(axis=1)
    denom = np.sqrt((rx_c**2).sum(axis=1) * (ry_c**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0.0, numer / denom, 0.0)


def rank_ic_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-date cross-sectional rank IC.

    Parameters
    ----------
    y_true:
        Realised returns, shape (n_samples,).
    y_pred:
        Model predictions, shape (n_samples,).
    groups:
        Date ordinals, shape (n_samples,).  All samples sharing the same
        ordinal belong to the same cross-section.

    Returns
    -------
    unique_groups:
        Sorted unique date ordinals, shape (n_dates,).
    ic_values:
        Spearman ρ for each date, shape (n_dates,).  Dates with fewer than
        3 observations return 0.0.

    When every date carries the same number of observations (the production
    walk-forward case — ``build_panel`` drops NaN rows so each test date has the
    full cross-section), the per-date Spearman loop collapses to a single
    ``rankdata(axis=1)`` + vectorized Pearson on the ``(n_dates, k)`` reshaped
    panel, dropping thousands of ``scipy.stats.spearmanr`` calls.  Ragged groups
    (variable cross-section size) fall back to the per-date loop.
    """
    unique_groups, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    n_dates = len(unique_groups)
    ic_values = np.zeros(n_dates, dtype=np.float64)
    if n_dates == 0:
        return unique_groups, ic_values

    k = int(counts[0])
    if k >= 3 and bool((counts == k).all()):
        # Equal-sized cross-sections: scatter each date's rows into a dense
        # (n_dates, k) panel in encounter order, then batch-rank + Pearson.
        # ``inverse`` maps each sample to its date row; the within-date column is
        # its running position, which np.unique preserves for stably-ordered ties.
        order = np.argsort(inverse, kind="stable")
        Yt = y_true[order].reshape(n_dates, k)
        Yp = y_pred[order].reshape(n_dates, k)
        Rx = stats.rankdata(Yt, axis=1)
        Ry = stats.rankdata(Yp, axis=1)
        ic_values[:] = _pearson_on_ranks(Rx, Ry)
        return unique_groups, ic_values

    # Ragged or tiny cross-sections — per-date loop.
    for i in range(n_dates):
        mask = inverse == i
        ic_values[i] = _spearman(y_true[mask], y_pred[mask])
    return unique_groups, ic_values


def rank_ic_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Mean cross-sectional rank IC across all dates.

    A single scalar summarising the model's cross-sectional predictive power.
    Positive values mean predictions are positively rank-correlated with
    realised returns.
    """
    _, ic_values = rank_ic_series(y_true, y_pred, groups)
    return float(ic_values.mean()) if len(ic_values) > 0 else 0.0


def ic_stats(ic_values: np.ndarray) -> dict[str, float]:
    """Summary statistics for a per-date IC series.

    Returns mean_ic, std_ic, ic_ir (mean/std), and t_stat (mean / (std / sqrt(n))).
    IR and t_stat are set to 0.0 when std is near zero (degenerate series).
    """
    n = len(ic_values)
    mean_ic = float(ic_values.mean()) if n > 0 else 0.0
    std_ic = float(ic_values.std(ddof=1)) if n > 1 else 0.0
    ic_ir = mean_ic / std_ic if std_ic > 1e-12 else 0.0
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if (std_ic > 1e-12 and n > 0) else 0.0
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ic_ir": ic_ir,
        "t_stat": t_stat,
        "n_dates": n,
    }


def held_out_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """OOS R²; returns 0.0 when the target has zero variance (degenerate)."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
