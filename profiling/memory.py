from __future__ import annotations

import resource
from dataclasses import dataclass

import numpy as np
import polars as pl

_PAGE_SIZE = resource.getpagesize()


def obj_size_mb(obj: object) -> float:
    """Accurate in-memory size of a data structure, in MB.

    tracemalloc only sees Python-level allocations — it misses Polars' Rust
    buffers and NumPy's C buffers entirely. The authoritative size comes from
    each library: estimated_size for Polars, nbytes for NumPy.
    """
    if isinstance(obj, pl.DataFrame):
        return obj.estimated_size("mb")
    if isinstance(obj, np.ndarray):
        return obj.nbytes / 1024 / 1024
    return 0.0


def frames_size_mb(frames: dict[str, object]) -> float:
    return sum(obj_size_mb(o) for o in frames.values())


def rss_mb() -> float:
    """Current resident set size of this process, in MB.

    Reads /proc/self/statm so it reflects Rust- and C-side allocations that
    tracemalloc cannot see. RSS only captures retained memory, not transient
    peaks freed before the read.
    """
    with open("/proc/self/statm") as f:
        resident_pages = int(f.read().split()[1])
    return resident_pages * _PAGE_SIZE / 1024 / 1024


@dataclass(frozen=True)
class MemSnapshot:
    stage: str
    rss_mb: float
    data_mb: float


def snapshot(stage: str, frames: dict[str, object]) -> MemSnapshot:
    return MemSnapshot(stage=stage, rss_mb=rss_mb(), data_mb=frames_size_mb(frames))
