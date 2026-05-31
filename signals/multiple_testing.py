"""Multiple-testing correction and rolling IC-IR.

When screening many signals, the best t-stat is biased upward by selection:
you are implicitly choosing the most significant result from a set of tests.
This module provides:

1. **Rolling IC-IR** — trailing-window mean IC / std IC as a time series, so
   you can see regime stability and identify sub-periods where the signal was
   strongest or weakest.

2. **Bonferroni correction** — deflates the significance threshold by the
   number of signals tested (controls family-wise error rate, FWER).  For k
   independent tests at level α, the per-test threshold becomes α / k.

3. **Benjamini-Hochberg (BH) correction** — controls the false discovery rate
   (FDR) at level α.  Ranks p-values, rejects up to the largest index m where
   p_{(m)} ≤ m * α / k.  Less conservative than Bonferroni; appropriate when
   you expect some signals to be genuinely useful and can tolerate a known
   proportion of false discoveries.

4. **p-values from Newey-West t-stats** — two-sided p-values from the standard
   normal.  When the Newey-West HAC variance adjustment is correct, the t-stat
   is approximately N(0,1) under the null of zero IC.

5. **Deflated Sharpe Ratio–style** minimum-t threshold: given k tests, the
   effective significance level needed to claim a signal is 5% significant
   after Bonferroni / BH correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Rolling IC-IR
# ---------------------------------------------------------------------------


def rolling_ic_ir(
    ic_df: pl.DataFrame,
    window: int,
    *,
    ic_col: str = "ic",
) -> pl.DataFrame:
    """Trailing-window IC-IR (mean / std) as a time series.

    Parameters
    ----------
    ic_df:
        DataFrame with at least columns ``(date, ic_col)``.
    window:
        Look-back window in number of periods.
    ic_col:
        Name of the IC column (default "ic").

    Returns
    -------
    DataFrame with columns ``(date, rolling_ic, rolling_ic_std, rolling_ic_ir)``.
    Rows at the start of the series with fewer than ``window`` observations
    will have NaN for the rolling statistics.
    """
    df = ic_df.sort("date")
    return (
        df.with_columns(
            pl.col(ic_col).rolling_mean(window_size=window).alias("rolling_ic"),
            pl.col(ic_col).rolling_std(window_size=window).alias("rolling_ic_std"),
        )
        .with_columns(
            (pl.col("rolling_ic") / pl.col("rolling_ic_std").replace(0.0, None)).alias(
                "rolling_ic_ir"
            )
        )
        .select("date", "rolling_ic", "rolling_ic_std", "rolling_ic_ir")
    )


# ---------------------------------------------------------------------------
# p-values and corrections
# ---------------------------------------------------------------------------


def tstat_to_pvalue(t_stat: float) -> float:
    """Two-sided p-value from a Newey-West t-stat under the N(0,1) null."""
    return float(2.0 * (1.0 - scipy_stats.norm.cdf(abs(t_stat))))


@dataclass(frozen=True)
class MultipleTestingResult:
    """Results from multiple-testing correction."""

    signal_names: tuple[str, ...]
    t_stats: tuple[float, ...]
    p_values: tuple[float, ...]

    bonferroni_rejected: tuple[bool, ...]
    """True where the null is rejected at alpha after Bonferroni correction."""

    bh_rejected: tuple[bool, ...]
    """True where the null is rejected at alpha after Benjamini-Hochberg FDR correction."""

    alpha: float
    n_tests: int

    bonferroni_threshold_t: float
    """Minimum |t| needed to reject at alpha/n_tests level."""

    bh_threshold_p: float
    """Largest p-value threshold allowed by BH procedure (0 if none rejected)."""

    def to_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "signal": list(self.signal_names),
                "t_stat": list(self.t_stats),
                "p_value": list(self.p_values),
                "bonferroni_reject": list(self.bonferroni_rejected),
                "bh_reject": list(self.bh_rejected),
            }
        )


def bonferroni_correct(
    p_values: list[float],
    alpha: float = 0.05,
) -> np.ndarray:
    """Bonferroni-corrected rejection mask.

    Returns a boolean array of the same length as ``p_values``, True where
    the null is rejected at the family-wise error rate ``alpha``.
    """
    k = len(p_values)
    threshold = alpha / k
    return np.array(p_values) <= threshold


def bh_correct(
    p_values: list[float],
    alpha: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Benjamini-Hochberg FDR correction.

    Returns (rejected_mask, threshold_p) where ``rejected_mask`` is a boolean
    array (same order as input) and ``threshold_p`` is the BH cutoff p-value
    (0 if nothing was rejected).

    Implementation follows the original BH 1995 step-up procedure:
    sort p-values, find the largest rank m where p_{(m)} ≤ m * alpha / k,
    reject all hypotheses with p ≤ p_{(m)}.
    """
    p = np.array(p_values, dtype=float)
    k = len(p)
    if k == 0:
        return np.array([], dtype=bool), 0.0

    order = np.argsort(p)
    sorted_p = p[order]
    ranks = np.arange(1, k + 1)
    thresholds = ranks * alpha / k

    # Find largest m where p_{(m)} ≤ threshold
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(k, dtype=bool), 0.0

    m = int(np.where(below)[0].max())
    threshold_p = float(thresholds[m])

    # All p-values ≤ p_{(m)} are rejected
    rejected = p <= sorted_p[m]
    return rejected, threshold_p


def multiple_testing_correction(
    signal_names: list[str],
    t_stats: list[float],
    *,
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Apply both Bonferroni and BH corrections to a set of Newey-West t-stats.

    Parameters
    ----------
    signal_names:
        Names of the signals being tested.
    t_stats:
        Per-signal Newey-West HAC t-statistics.
    alpha:
        Family-wise / FDR significance level.

    Returns
    -------
    MultipleTestingResult with per-signal rejection flags and thresholds.
    """
    k = len(t_stats)
    p_values = [tstat_to_pvalue(t) for t in t_stats]

    bonf_mask = bonferroni_correct(p_values, alpha=alpha)
    bh_mask, bh_thresh = bh_correct(p_values, alpha=alpha)

    # Bonferroni minimum |t| threshold
    bonf_threshold_t = abs(scipy_stats.norm.ppf(alpha / (2 * k)))

    return MultipleTestingResult(
        signal_names=tuple(signal_names),
        t_stats=tuple(t_stats),
        p_values=tuple(p_values),
        bonferroni_rejected=tuple(bonf_mask.tolist()),
        bh_rejected=tuple(bh_mask.tolist()),
        alpha=alpha,
        n_tests=k,
        bonferroni_threshold_t=float(bonf_threshold_t),
        bh_threshold_p=float(bh_thresh),
    )
