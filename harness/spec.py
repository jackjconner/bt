"""Abstractions for a per-component profiling benchmark.

A ``ComponentBenchmark`` separates the three phases the profiler must treat
differently:

  - ``setup`` builds/loads the component's inputs. It runs ONCE per param point
    and is NOT timed — loading parquet or constructing upstream artifacts is not
    what we are measuring.
  - ``run`` executes the component's production path on those inputs. This is the
    only thing the trial timer wraps, so what it reports is the component's own
    cost, not the cost of preparing its data.
  - ``frames`` maps the run output to the data structures whose in-memory size
    should be attributed to the component (Polars frames / NumPy arrays).

Keeping these explicit means a slow loader can't masquerade as a slow component.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from etl import DatasetLoader
from etl.datasets import GenSpec


@dataclass(frozen=True)
class BenchmarkContext:
    """Everything a benchmark's ``setup`` needs for one param point."""

    spec: GenSpec
    loader: DatasetLoader
    workdir: Path


@dataclass(frozen=True)
class ComponentBenchmark:
    name: str
    setup: Callable[[BenchmarkContext], object]
    run: Callable[[object], object]
    frames: Callable[[object], dict[str, object]]


def no_frames(_: object) -> dict[str, object]:
    """For components whose output is not a sized data structure (timing only)."""
    return {}


__all__ = ["BenchmarkContext", "ComponentBenchmark", "no_frames"]
