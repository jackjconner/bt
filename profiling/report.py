from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .memory import frames_size_mb, rss_mb

T = TypeVar("T")


@dataclass(frozen=True)
class StageProfile:
    stage: str
    elapsed_s: float
    result_mb: float       # size of the data structure(s) the stage produced
    rss_delta_mb: float    # net process RSS change across the stage


@dataclass(frozen=True)
class ScalingResult:
    params: dict[str, int]
    stages: list[StageProfile]


def collect_stage(
    stage: str,
    fn: Callable[[], T],
    frames_after_fn: Callable[[T], dict[str, object]],
) -> tuple[T, StageProfile]:
    """Run fn(), recording wall time, produced-data size, and RSS delta.

    frames_after_fn maps the result to the data structures whose size we want
    measured (Polars DataFrames or NumPy arrays). RSS delta is taken after a
    gc.collect() so freed transients don't inflate the retained figure.
    """
    gc.collect()
    rss_before = rss_mb()

    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0

    result_mb = frames_size_mb(frames_after_fn(result))
    gc.collect()
    rss_after = rss_mb()

    return result, StageProfile(
        stage=stage,
        elapsed_s=elapsed,
        result_mb=result_mb,
        rss_delta_mb=rss_after - rss_before,
    )


def print_report(result: ScalingResult) -> None:
    params_str = "  ".join(f"{k}={v}" for k, v in result.params.items())
    print(f"\n{'─' * 78}")
    print(f"params: {params_str}")
    print(f"{'─' * 78}")
    print(f"{'stage':<26}  {'elapsed_s':>10}  {'result_mb':>10}  {'rss_delta_mb':>12}")
    print(f"{'─' * 26}  {'─' * 10}  {'─' * 10}  {'─' * 12}")
    for s in result.stages:
        print(
            f"{s.stage:<26}  {s.elapsed_s:>10.4f}  {s.result_mb:>10.2f}  "
            f"{s.rss_delta_mb:>12.2f}"
        )
