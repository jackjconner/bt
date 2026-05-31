"""Cross-signal pair-wise rank-correlation and diversification.

Before committing capital to a book of signals, we want to know how *redundant*
they are.  Two signals that rank the cross-section near-identically every day add
little breadth — they double the exposure of one bet rather than spreading risk
across independent bets.  This module scores a set of signals on that axis
without running a backtest:

1. **Pair-wise rank-correlation matrix.**  Each ``(signal_i, signal_j)`` entry is
   the Spearman correlation of the two signals over the panel.  Spearman is
   Pearson on the within-date ranks (matching the rest of the component: Polars'
   ``rank()`` "average" tie default == ``scipy.stats.rankdata``), so ranks are
   recomputed per date and pooled across the panel.  The matrix is symmetric,
   has unit diagonal, and is bounded ``[-1, 1]``.

2. **Mean absolute correlation.**  The average ``|corr|`` over the off-diagonal
   (unique) pairs — a single scalar summarising how correlated the book is.
   Lower = more independent signals.

3. **Diversification ratio.**  The variance of an equal-weight composite of the
   (rank-standardised) signals relative to the variance of a single signal.  In
   the standardised-rank space each signal has unit variance, so this reduces to
   the **redundancy ratio** ``(1ᵀ C 1) / n²`` where ``C`` is the correlation
   matrix and ``n`` the signal count.  It equals 1 when all signals are perfectly
   correlated (no diversification), ``1/n`` when they are mutually uncorrelated,
   and falls toward 0 as negative correlations actively cancel — so *lower = more
   diversified* and a smaller number is always better, matching the docstring
   contract.  The nominal range is ``[0, 1]``.

NaN / coverage: for each pair, a ``(date, asset)`` cell contributes only when
both signals are finite there (pairwise-complete), and ranks are taken over the
assets that survive that per-date mask — the same masking convention as the IC
engine and ``quantile_spread``.  This naturally handles NaN tails and lets the
function be applied per horizon (the caller passes whatever signal set it wants
scored; horizons differ only in which signals/returns are upstream).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class SignalCorrelationResult:
    """Cross-signal rank-correlation and diversification summary."""

    correlation_matrix: pl.DataFrame
    """Pair-wise Spearman rank-correlation, square ``(n_signals, n_signals)``.

    Column ``signal`` holds the row labels (signal names); the remaining columns
    are one per signal.  Symmetric, unit diagonal, entries in ``[-1, 1]``."""

    signal_names: list[str]
    """Signal names in matrix row/column order."""

    mean_abs_correlation: float
    """Mean ``|corr|`` over the unique off-diagonal pairs.

    Lower = more independent signals.  ``nan`` when there is only one signal (no
    off-diagonal pair exists)."""

    diversification_ratio: float
    """Redundancy ratio ``(1ᵀ C 1) / n²`` in ``[0, 1]``; **lower = more diversified**.

    Equals 1 when all signals are perfectly correlated (no diversification
    benefit), ``1/n`` when mutually uncorrelated, and approaches 0 when negative
    correlations cancel.  ``nan`` when there is only one signal."""

    n_signals: int


def _within_date_ranks(df: pl.DataFrame, signal_col: str) -> pl.DataFrame:
    """Rank ``signal_col`` within each date, dropping non-finite cells first.

    Returns ``(date, id, _rank)`` with ranks taken over the assets finite on that
    date — the "average" tie default, matching the IC engine and scipy.
    """
    c = pl.col(signal_col)
    return (
        df.select("date", "id", signal_col)
        .filter(c.is_not_null() & c.is_finite())
        .with_columns(c.rank().over("date").alias("_rank"))
        .select("date", "id", "_rank")
    )


def _pair_rank_corr(
    ranks_i: pl.DataFrame,
    ranks_j: pl.DataFrame,
    min_obs: int,
) -> float:
    """Pearson correlation of two within-date rank frames, pooled over the panel.

    The two frames are inner-joined on ``(date, id)`` so only pairwise-complete
    cells contribute; ranks were already taken per date over the finite assets.
    Returns ``nan`` when fewer than ``min_obs`` joint cells survive or either
    pooled rank vector is constant.
    """
    joined = ranks_i.join(
        ranks_j.rename({"_rank": "_rank_j"}),
        on=["date", "id"],
        how="inner",
    )
    if joined.height < min_obs:
        return float("nan")

    x = joined["_rank"].to_numpy()
    y = joined["_rank_j"].to_numpy()
    xc = x - x.mean()
    yc = y - y.mean()
    vx = float(xc @ xc)
    vy = float(yc @ yc)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return float((xc @ yc) / np.sqrt(vx * vy))


def signal_pair_correlation(
    signals: dict[str, pl.DataFrame] | list[pl.DataFrame],
    *,
    signal_col: str = "signal",
    names: list[str] | None = None,
    min_obs: int = 10,
) -> SignalCorrelationResult:
    """Pair-wise cross-signal rank-correlation + diversification of a signal book.

    Parameters
    ----------
    signals:
        Either a mapping ``{name: frame}`` or a list of long-format frames, each
        ``(date, id, signal_col)``.  A list pairs with ``names`` (defaulting to
        ``signal_0, signal_1, …``).
    signal_col:
        The numeric signal column present in every frame.
    names:
        Row/column labels when ``signals`` is a list.  Ignored for a mapping
        (its keys are used).  Must match the list length when given.
    min_obs:
        Minimum pairwise-complete cells for a pair's correlation to be computed;
        below it the entry is ``nan``.

    Returns
    -------
    SignalCorrelationResult

    Notes
    -----
    Spearman == Pearson on within-date ranks (Polars ``rank()`` "average" tie
    default).  The matrix is symmetric with unit diagonal; off-diagonal entries
    lie in ``[-1, 1]`` (``nan`` where coverage is insufficient).  A single signal
    yields a ``1.0`` diagonal with ``nan`` summary scalars (no pair exists).
    """
    if isinstance(signals, dict):
        sig_names = list(signals.keys())
        frames = list(signals.values())
    else:
        frames = list(signals)
        if names is not None:
            if len(names) != len(frames):
                raise ValueError("names must match the number of signal frames")
            sig_names = list(names)
        else:
            sig_names = [f"signal_{i}" for i in range(len(frames))]

    if not frames:
        raise ValueError("signals must be non-empty")

    n = len(frames)
    ranked = [_within_date_ranks(f, signal_col) for f in frames]

    mat = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            r = _pair_rank_corr(ranked[i], ranked[j], min_obs)
            mat[i, j] = r
            mat[j, i] = r

    correlation_matrix = pl.DataFrame(
        {"signal": sig_names} | {name: mat[:, k].tolist() for k, name in enumerate(sig_names)}
    )

    if n < 2:
        return SignalCorrelationResult(
            correlation_matrix=correlation_matrix,
            signal_names=sig_names,
            mean_abs_correlation=float("nan"),
            diversification_ratio=float("nan"),
            n_signals=n,
        )

    iu = np.triu_indices(n, k=1)
    off_diag = mat[iu]
    finite = off_diag[np.isfinite(off_diag)]
    mean_abs = float(np.mean(np.abs(finite))) if finite.size > 0 else float("nan")

    # Redundancy ratio = 1 / (classic diversification ratio) = (1ᵀ C 1) / n².
    # Treat unscorable (nan) pairs as 0 correlation so they neither inflate nor
    # deflate the book's redundancy; the diagonal is exactly 1.
    c_filled = np.where(np.isfinite(mat), mat, 0.0)
    np.fill_diagonal(c_filled, 1.0)
    quad_form = float(np.ones(n) @ c_filled @ np.ones(n))
    redundancy = quad_form / (n * n)

    return SignalCorrelationResult(
        correlation_matrix=correlation_matrix,
        signal_names=sig_names,
        mean_abs_correlation=mean_abs,
        diversification_ratio=redundancy,
        n_signals=n,
    )
