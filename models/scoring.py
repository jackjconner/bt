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
    """
    unique_groups = np.sort(np.unique(groups))
    ic_values = np.empty(len(unique_groups), dtype=np.float64)
    for i, g in enumerate(unique_groups):
        mask = groups == g
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
