"""Tests for models.splitters — purged embargo CV, walk-forward, rolling window."""
from __future__ import annotations

import numpy as np
import pytest

from .splitters import (
    PurgedEmbargoCVSplitter,
    RollingWindowSplitter,
    WalkForwardSplitter,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _panel_groups(n_dates: int, n_assets: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (X, groups) where groups are per-date ordinals."""
    # ordinals are just 0..n_dates-1 (integer positions, not calendar days)
    date_ords = np.arange(n_dates, dtype=np.int64)
    groups = np.repeat(date_ords, n_assets)
    X = np.zeros((n_dates * n_assets, 3))
    return X, groups


# --------------------------------------------------------------------------- #
# PurgedEmbargoCVSplitter
# --------------------------------------------------------------------------- #

class TestPurgedEmbargoCVSplitter:
    def test_no_test_in_train(self):
        """Train indices must never overlap test indices."""
        X, groups = _panel_groups(50, 5)
        splitter = PurgedEmbargoCVSplitter(n_splits=5, embargo_periods=0)
        for train_idx, test_idx in splitter.split(X, groups=groups):
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_embargo_removes_boundary_train_samples(self):
        """With embargo_periods=3, no train sample should have a group ordinal
        in [test_start - 3, test_start - 1]."""
        n_dates, n_assets = 60, 4
        embargo = 3
        X, groups = _panel_groups(n_dates, n_assets)
        splitter = PurgedEmbargoCVSplitter(n_splits=4, embargo_periods=embargo)
        for train_idx, test_idx in splitter.split(X, groups=groups):
            test_ords = np.unique(groups[test_idx])
            test_start = int(test_ords.min())
            forbidden = set(range(test_start - embargo, test_start))
            train_ords = set(groups[train_idx].tolist())
            assert forbidden.isdisjoint(train_ords), (
                f"Embargoed ordinals {forbidden} appear in train set"
            )

    def test_train_only_past(self):
        """All training group ordinals must be strictly less than the minimum
        test group ordinal (no future data in training).  The first fold may
        have an empty train set (nothing precedes fold 0); skip those folds."""
        X, groups = _panel_groups(40, 3)
        splitter = PurgedEmbargoCVSplitter(n_splits=4, embargo_periods=2)
        for train_idx, test_idx in splitter.split(X, groups=groups):
            if len(train_idx) == 0:
                continue  # first fold can be empty — no past data
            max_train = groups[train_idx].max()
            min_test = groups[test_idx].min()
            assert max_train < min_test

    def test_n_splits(self):
        X, groups = _panel_groups(50, 5)
        splitter = PurgedEmbargoCVSplitter(n_splits=5)
        folds = list(splitter.split(X, groups=groups))
        assert len(folds) == 5
        assert splitter.get_n_splits() == 5

    def test_requires_groups(self):
        X = np.zeros((10, 2))
        splitter = PurgedEmbargoCVSplitter(n_splits=2)
        with pytest.raises(ValueError, match="groups"):
            list(splitter.split(X))


# --------------------------------------------------------------------------- #
# WalkForwardSplitter
# --------------------------------------------------------------------------- #

class TestWalkForwardSplitter:
    def test_expanding_train_window(self):
        """Each successive fold's training set must be at least as large as the
        previous fold (expanding window property)."""
        X, groups = _panel_groups(80, 4)
        splitter = WalkForwardSplitter(n_splits=5, min_train_periods=10)
        train_sizes = [len(tr) for tr, _ in splitter.split(X, groups=groups)]
        for a, b in zip(train_sizes, train_sizes[1:]):
            assert b >= a, f"Train size shrank: {a} → {b}"

    def test_no_future_data_in_train(self):
        """Train date ordinals must always be strictly less than test date ordinals."""
        X, groups = _panel_groups(60, 3)
        splitter = WalkForwardSplitter(n_splits=4, min_train_periods=10, embargo_periods=2)
        for train_idx, test_idx in splitter.split(X, groups=groups):
            assert groups[train_idx].max() < groups[test_idx].min()

    def test_embargo_gap_respected(self):
        """With embargo_periods=k, the ordinal gap between train_end and test_start
        must be at least k."""
        n_dates, n_assets, embargo = 80, 3, 5
        X, groups = _panel_groups(n_dates, n_assets)
        splitter = WalkForwardSplitter(n_splits=4, min_train_periods=10, embargo_periods=embargo)
        for train_idx, test_idx in splitter.split(X, groups=groups):
            train_end_ord = int(groups[train_idx].max())
            test_start_ord = int(groups[test_idx].min())
            gap = test_start_ord - train_end_ord - 1
            assert gap >= embargo, f"gap={gap} < embargo={embargo}"

    def test_no_overlap(self):
        X, groups = _panel_groups(60, 3)
        splitter = WalkForwardSplitter(n_splits=4, min_train_periods=5)
        for train_idx, test_idx in splitter.split(X, groups=groups):
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_requires_groups(self):
        X = np.zeros((10, 2))
        splitter = WalkForwardSplitter(n_splits=2)
        with pytest.raises(ValueError, match="groups"):
            list(splitter.split(X))


# --------------------------------------------------------------------------- #
# RollingWindowSplitter
# --------------------------------------------------------------------------- #

class TestRollingWindowSplitter:
    def test_fixed_train_window_size(self):
        """Each fold's training window should contain exactly train_periods unique dates."""
        train_periods = 15
        X, groups = _panel_groups(80, 4)
        splitter = RollingWindowSplitter(n_splits=4, train_periods=train_periods)
        for train_idx, _ in splitter.split(X, groups=groups):
            n_unique_train_dates = len(np.unique(groups[train_idx]))
            assert n_unique_train_dates == train_periods, (
                f"expected {train_periods} train dates, got {n_unique_train_dates}"
            )

    def test_no_future_in_train(self):
        X, groups = _panel_groups(80, 4)
        splitter = RollingWindowSplitter(n_splits=4, train_periods=15)
        for train_idx, test_idx in splitter.split(X, groups=groups):
            assert groups[train_idx].max() < groups[test_idx].min()

    def test_no_overlap(self):
        X, groups = _panel_groups(80, 4)
        splitter = RollingWindowSplitter(n_splits=4, train_periods=15)
        for train_idx, test_idx in splitter.split(X, groups=groups):
            assert len(np.intersect1d(train_idx, test_idx)) == 0
