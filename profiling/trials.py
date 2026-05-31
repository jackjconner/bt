"""Repeated-trial runner that collects per-trial raw measurements and derives
percentile latency statistics.

Why a separate file: the single-shot `collect_stage` in report.py is the
stable public API that main.py depends on.  This layer wraps it to produce a
richer per-trial record without touching that contract.

Key invariant: warmup trials are discarded *before* any statistics are computed
so JIT / cache-warm effects don't contaminate the reported distribution.  Min is
kept because it represents the machine's best-case throughput; p90/p95 guard
against tail latency that would blow a SLA in a batch pipeline.
"""

from __future__ import annotations

import gc
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from .memory import frames_size_mb, rss_mb

T = TypeVar("T")


@dataclass(frozen=True)
class TrialMeasurement:
    """Single trial observation — one row in ``stage_measurements``."""

    trial_idx: int
    elapsed_s: float
    result_mb: float
    rss_delta_mb: float
    peak_rss_mb: float
    peak_traced_mb: float  # tracemalloc peak for Python-visible allocations


@dataclass(frozen=True)
class TrialStats:
    """Derived percentile statistics across the kept trials for one stage."""

    stage: str
    n_trials: int
    elapsed_min: float
    elapsed_p50: float
    elapsed_p90: float
    elapsed_p95: float
    elapsed_stddev: float
    result_mb: float          # median result size
    rss_delta_mb: float       # median net RSS delta
    peak_rss_mb: float        # median peak RSS
    peak_traced_mb: float     # median peak traced


@dataclass(frozen=True)
class TrialResult(TrialStats):
    """Full output including the raw per-trial measurements."""

    trials: tuple[TrialMeasurement, ...]


def _run_one(
    fn: Callable[[], T],
    frames_after_fn: Callable[[T], dict[str, object]],
    trial_idx: int,
) -> TrialMeasurement:
    """Execute fn() once with full instrumentation.

    tracemalloc is started immediately before the call and stopped immediately
    after to bound what counts as 'peak traced'.  RSS before/after uses the
    same gc.collect() discipline as the original collect_stage so freed
    transients don't inflate the retained figure.
    """
    gc.collect()
    rss_before = rss_mb()

    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result_mb = frames_size_mb(frames_after_fn(result))

    gc.collect()
    rss_after = rss_mb()

    return TrialMeasurement(
        trial_idx=trial_idx,
        elapsed_s=elapsed,
        result_mb=result_mb,
        rss_delta_mb=rss_after - rss_before,
        peak_rss_mb=rss_after,
        peak_traced_mb=traced_peak / 1024 / 1024,
    )


def run_trials(
    stage: str,
    fn: Callable[[], T],
    frames_after_fn: Callable[[T], dict[str, object]],
    n_trials: int = 5,
    warmup: int = 1,
) -> TrialResult:
    """Run ``fn`` repeatedly and return percentile latency statistics.

    ``warmup`` executions run first and are discarded — their purpose is to
    warm Python's JIT, OS page cache, and any lazy initialisation so the kept
    trials reflect steady-state throughput, not cold-start overhead.

    Args:
        stage: Human-readable stage label (used in TrialStats.stage).
        fn: Zero-argument callable that performs the work.
        frames_after_fn: Maps fn's return value to the data structures whose
            size should be measured (Polars DataFrames / NumPy arrays).
        n_trials: Number of timed trials to keep after warmup.
        warmup: Number of warmup trials to discard.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    for _ in range(warmup):
        gc.collect()
        fn()

    measurements: list[TrialMeasurement] = []
    for idx in range(n_trials):
        measurements.append(_run_one(fn, frames_after_fn, idx))

    elapsed_arr = np.array([m.elapsed_s for m in measurements])
    result_mb_arr = np.array([m.result_mb for m in measurements])
    rss_delta_arr = np.array([m.rss_delta_mb for m in measurements])
    peak_rss_arr = np.array([m.peak_rss_mb for m in measurements])
    peak_traced_arr = np.array([m.peak_traced_mb for m in measurements])

    return TrialResult(
        stage=stage,
        n_trials=n_trials,
        elapsed_min=float(np.min(elapsed_arr)),
        elapsed_p50=float(np.percentile(elapsed_arr, 50)),
        elapsed_p90=float(np.percentile(elapsed_arr, 90)),
        elapsed_p95=float(np.percentile(elapsed_arr, 95)),
        elapsed_stddev=float(np.std(elapsed_arr, ddof=1) if n_trials > 1 else 0.0),
        result_mb=float(np.median(result_mb_arr)),
        rss_delta_mb=float(np.median(rss_delta_arr)),
        peak_rss_mb=float(np.median(peak_rss_arr)),
        peak_traced_mb=float(np.median(peak_traced_arr)),
        trials=tuple(measurements),
    )
