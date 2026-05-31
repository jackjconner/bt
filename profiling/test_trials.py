"""Tests for trials.py — repeated-trial runner + percentile statistics."""

from __future__ import annotations

import numpy as np

from profiling.trials import run_trials


def _noop() -> None:
    return None


def _alloc_10mb() -> bytearray:
    """Allocate ~10 MB of Python-visible memory and hold it until fn returns."""
    return bytearray(10 * 1024 * 1024)


def test_percentiles_ordered() -> None:
    """min ≤ p50 ≤ p90 ≤ p95 must hold for any distribution."""
    count = 0

    def slow_sometimes() -> None:
        nonlocal count
        count += 1
        # Introduce some variability so percentiles aren't all identical.
        if count % 3 == 0:
            _ = list(range(50_000))

    result = run_trials("test", slow_sometimes, lambda _: {}, n_trials=7, warmup=1)

    assert result.elapsed_min <= result.elapsed_p50
    assert result.elapsed_p50 <= result.elapsed_p90
    assert result.elapsed_p90 <= result.elapsed_p95
    assert result.n_trials == 7
    assert len(result.trials) == 7


def test_warmup_discarded() -> None:
    """Warmup executions must not appear in the returned trial measurements."""
    calls: list[int] = []

    def track() -> None:
        calls.append(1)

    result = run_trials("test", track, lambda _: {}, n_trials=3, warmup=2)

    # Total calls = warmup + n_trials; only n_trials appear in result.
    assert sum(calls) == 5  # 2 warmup + 3 kept
    assert len(result.trials) == 3
    assert result.trials[0].trial_idx == 0
    assert result.trials[2].trial_idx == 2


def test_tracemalloc_peak_positive_for_allocating_stage() -> None:
    """peak_traced_mb must be > 0 when the stage allocates Python-visible memory."""
    result = run_trials(
        "alloc",
        _alloc_10mb,
        lambda _: {},
        n_trials=3,
        warmup=0,
    )
    # Median across trials — at least one trial must have traced > 0.
    assert result.peak_traced_mb > 0.0
    # Each individual trial should have captured some traced allocation.
    assert all(m.peak_traced_mb > 0.0 for m in result.trials)


def test_zero_warmup_allowed() -> None:
    """warmup=0 should work without error."""
    result = run_trials("noop", _noop, lambda _: {}, n_trials=2, warmup=0)
    assert result.n_trials == 2


def test_single_trial_stddev_zero() -> None:
    """With n_trials=1 there is no variance — stddev must be 0.0."""
    result = run_trials("noop", _noop, lambda _: {}, n_trials=1, warmup=0)
    assert result.elapsed_stddev == 0.0


def test_frames_after_fn_measured() -> None:
    """result_mb must reflect what frames_after_fn returns."""
    arr = np.zeros(1024 * 1024)  # 8 MB

    result = run_trials(
        "numpy",
        lambda: arr,
        lambda a: {"arr": a},
        n_trials=3,
        warmup=0,
    )
    # 8 MB array — median result_mb should be close to 8.
    assert result.result_mb > 7.0


def test_invalid_n_trials_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        run_trials("bad", _noop, lambda _: {}, n_trials=0)


def test_invalid_warmup_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        run_trials("bad", _noop, lambda _: {}, n_trials=1, warmup=-1)
