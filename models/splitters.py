"""Time-series CV splitters that respect temporal causality.

Standard KFold leaks in two ways that matter in finance:
  1. Test samples are drawn from the same contiguous period as train samples,
     so any lookback window straddles the boundary.
  2. Shuffled KFold trains on future data; even un-shuffled KFold has the last
     fold train on the first N-1 folds' data *after* their test period.

This module provides three scikit-learn-compatible generators (same interface as
KFold.split: yield train_idx, test_idx) that are safe by construction:

* ``PurgedEmbargoCVSplitter`` — fixed k folds, each test fold is contiguous and
  immediately follows (plus embargo) the train fold.  Train samples that fall in
  [test_start - embargo, test_end] are purged from the training set to prevent
  forward-label leakage caused by overlapping return windows.

* ``WalkForwardSplitter`` — expanding-window: train = [0, t), test = [t, t+step).
  Train set always ends strictly before test.  No data from the future is ever
  used for training.

* ``RollingWindowSplitter`` — fixed-size rolling window: same causality as walk-
  forward but drops the oldest training observations to model non-stationarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _sorted_unique_dates(groups: np.ndarray) -> np.ndarray:
    """Return sorted unique date ordinals/indices that appear in ``groups``."""
    return np.sort(np.unique(groups))


def _sample_indices_for_dates(groups: np.ndarray, date_set: np.ndarray) -> np.ndarray:
    """Row indices whose group (date) is in ``date_set``."""
    mask = np.isin(groups, date_set)
    return np.where(mask)[0]


# --------------------------------------------------------------------------- #
# PurgedEmbargoCVSplitter
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PurgedEmbargoCVSplitter:
    """K-fold purged + embargoed time-series CV.

    ``groups`` must be integer date ordinals (one per sample) so we can
    compare dates without calendar libraries.  Use
    ``panel.date_ordinals(groups_series)`` to produce them.

    Purging: train samples whose date ordinal falls within the embargo window
    ``[test_start - embargo_days, test_end]`` are excluded.  This removes
    samples whose *label* was computed with future data that falls in the test
    window (e.g. a 5-day forward return uses up to t+5, so for a test period
    starting at T we must remove train samples at [T-4, T+…]).

    Embargo: the gap between train_end and test_start, expressed in date-ordinal
    units (calendar days if ordinals are calendar-day serial numbers, trading
    days if ordinals are integer position indices).
    """

    n_splits: int = 5
    embargo_periods: int = 0  # units match whatever ordinal scale groups uses

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ):
        """Yield (train_indices, test_indices) for each fold.

        Parameters
        ----------
        X:
            Feature matrix (n_samples, n_features).  Only n_samples is used.
        y:
            Ignored; kept for sklearn compatibility.
        groups:
            1-D array of integer date ordinals, one per sample.  Splitting and
            purging is done in ordinal space.
        """
        if groups is None:
            raise ValueError("PurgedEmbargoCVSplitter requires groups (date ordinals)")
        groups = np.asarray(groups)
        unique_dates = _sorted_unique_dates(groups)
        n_dates = len(unique_dates)
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")

        fold_size = n_dates // self.n_splits
        if fold_size == 0:
            raise ValueError(f"Too few unique dates ({n_dates}) for n_splits={self.n_splits}")

        for k in range(self.n_splits):
            test_start_i = k * fold_size
            test_end_i = test_start_i + fold_size - 1 if k < self.n_splits - 1 else n_dates - 1
            test_dates = unique_dates[test_start_i : test_end_i + 1]
            test_start_ord = int(test_dates[0])
            test_end_ord = int(test_dates[-1])

            # purge boundary: remove any train sample whose label window
            # overlaps the test window by up to embargo_periods
            purge_start_ord = test_start_ord - self.embargo_periods
            train_dates = unique_dates[
                (unique_dates < purge_start_ord) | (unique_dates > test_end_ord)
            ]
            # additionally drop dates *after* the test fold (train only on past
            # relative to each fold start to prevent walk-forward contamination
            # when using this splitter in an expanding/rolling context)
            train_dates = train_dates[train_dates < test_start_ord]

            train_idx = _sample_indices_for_dates(groups, train_dates)
            test_idx = _sample_indices_for_dates(groups, test_dates)
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# --------------------------------------------------------------------------- #
# WalkForwardSplitter
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WalkForwardSplitter:
    """Expanding-window walk-forward splitter.

    Train always uses [first_date, split_date).
    Test uses [split_date + embargo, split_date + embargo + test_size).

    This guarantees zero future leakage: every training set is entirely in the
    past relative to its test set.  The expanding window captures more history
    as the walk advances, which matches real-world model retraining.
    """

    n_splits: int = 5
    embargo_periods: int = 0
    # minimum number of unique date-groups required in the initial train window
    min_train_periods: int = 1

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ):
        if groups is None:
            raise ValueError("WalkForwardSplitter requires groups (date ordinals)")
        groups = np.asarray(groups)
        unique_dates = _sorted_unique_dates(groups)
        n_dates = len(unique_dates)

        # distribute test periods evenly after the minimum train prefix
        available = n_dates - self.min_train_periods
        if available < self.n_splits:
            raise ValueError(
                f"Not enough date periods ({n_dates}) for min_train={self.min_train_periods}"
                f" + n_splits={self.n_splits}"
            )
        step = available // self.n_splits

        for k in range(self.n_splits):
            # train: [0, train_end_i] inclusive
            train_end_i = self.min_train_periods + k * step - 1
            # test: embargo gap after train_end, then test_size dates
            test_start_i = train_end_i + 1 + self.embargo_periods
            test_end_i = min(test_start_i + step - 1, n_dates - 1)

            if test_start_i >= n_dates:
                break

            train_dates = unique_dates[: train_end_i + 1]
            test_dates = unique_dates[test_start_i : test_end_i + 1]

            train_idx = _sample_indices_for_dates(groups, train_dates)
            test_idx = _sample_indices_for_dates(groups, test_dates)
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# --------------------------------------------------------------------------- #
# RollingWindowSplitter
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RollingWindowSplitter:
    """Fixed-size rolling window splitter.

    Like WalkForwardSplitter but drops the oldest observations so the training
    window is always exactly ``train_periods`` date-groups wide.  Useful when
    stationarity is doubtful and a very old regime should not inform today's fit.
    """

    n_splits: int = 5
    train_periods: int = 100  # number of unique date-groups in each train window
    embargo_periods: int = 0

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ):
        if groups is None:
            raise ValueError("RollingWindowSplitter requires groups (date ordinals)")
        groups = np.asarray(groups)
        unique_dates = _sorted_unique_dates(groups)
        n_dates = len(unique_dates)

        available = n_dates - self.train_periods
        if available < self.n_splits:
            raise ValueError(
                f"Not enough date periods ({n_dates}) for train_periods={self.train_periods}"
                f" + n_splits={self.n_splits}"
            )
        step = available // self.n_splits

        for k in range(self.n_splits):
            test_start_i = self.train_periods + k * step + self.embargo_periods
            test_end_i = min(test_start_i + step - 1, n_dates - 1)
            train_start_i = test_start_i - self.embargo_periods - self.train_periods
            train_end_i = test_start_i - self.embargo_periods - 1

            if test_start_i >= n_dates:
                break

            train_dates = unique_dates[train_start_i : train_end_i + 1]
            test_dates = unique_dates[test_start_i : test_end_i + 1]

            train_idx = _sample_indices_for_dates(groups, train_dates)
            test_idx = _sample_indices_for_dates(groups, test_dates)
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# --------------------------------------------------------------------------- #
# calendar-split helper (consumes cv_splits_calendar dataset)
# --------------------------------------------------------------------------- #


def splits_from_calendar(
    cv_df: pl.DataFrame,
    date_ordinals: np.ndarray,
    groups: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convert a ``cv_splits_calendar`` DataFrame into (train_idx, test_idx) pairs.

    ``date_ordinals`` is a 1-D int64 array mapping each unique date to its
    ordinal (``date.toordinal()``).  ``groups`` is the per-sample date ordinal
    array produced by ``panel.date_ordinals``.

    Each fold's train window is [train_start, train_end] and test is
    [test_start, test_end].  Samples between train_end and test_start
    (the embargo gap) are excluded from both sets.
    """
    folds = []
    for row in cv_df.sort("fold").iter_rows(named=True):
        ts_ord = row["train_start"].toordinal()
        te_ord = row["train_end"].toordinal()
        xs_ord = row["test_start"].toordinal()
        xe_ord = row["test_end"].toordinal()

        train_mask = (groups >= ts_ord) & (groups <= te_ord)
        test_mask = (groups >= xs_ord) & (groups <= xe_ord)
        folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return folds
