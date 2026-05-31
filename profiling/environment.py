"""Run/environment metadata capture.

Captures everything needed to make two profiling runs comparable: the exact
code version (git SHA + dirty flag), the hardware (host, CPU model, cores,
RAM), the Python/Polars/NumPy versions, and the BLAS thread count.

These fields map 1:1 to the ``profiling_runs`` schema in etl.datasets so
results written by storage.py can be queried against that schema.

BLAS thread count is read from environment variables that OpenBLAS / MKL /
BLIS honour (in that priority order).  When none are set we fall back to
``os.cpu_count()`` because most BLAS implementations default to all cores.
"""

from __future__ import annotations

import datetime
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class RunEnvironment:
    """Snapshot of the execution environment at run time.

    Attributes match ``profiling_runs`` schema column names exactly so the
    dict representation can be passed directly to a Polars row constructor.
    """

    run_id: str
    run_ts: datetime.date
    git_sha: str
    git_dirty: bool
    hostname: str
    cpu_model: str
    n_cores: int
    total_ram_mb: float
    python_version: str
    polars_version: str
    numpy_version: str
    blas_threads: int
    trials: int
    warmup_trials: int


def _git_sha() -> str:
    """Return the abbreviated HEAD SHA, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    """True when the working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def _cpu_model() -> str:
    """Best-effort CPU model string from /proc/cpuinfo (Linux) or platform."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _total_ram_mb() -> float:
    """Total physical RAM in MB from /proc/meminfo (Linux) or 0.0."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except OSError:
        pass
    return 0.0


def _blas_threads() -> int:
    """BLAS parallelism limit, highest-priority env var wins.

    Priority: OMP_NUM_THREADS > OPENBLAS_NUM_THREADS > MKL_NUM_THREADS >
    BLIS_NUM_THREADS > os.cpu_count().  This order matches what most BLAS
    builds inspect at startup.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        val = os.environ.get(var)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return os.cpu_count() or 1


def _lib_version(module_name: str) -> str:
    """Return __version__ of an already-imported module, or 'unknown'."""
    import importlib

    try:
        mod = importlib.import_module(module_name)
        return str(getattr(mod, "__version__", "unknown"))
    except ImportError:
        return "unknown"


def capture_environment(
    run_id: str,
    trials: int = 1,
    warmup_trials: int = 0,
) -> RunEnvironment:
    """Capture a full environment snapshot for a profiling run.

    Args:
        run_id: Caller-supplied unique identifier for this run (e.g. a UUID or
            timestamp string).  Written verbatim to ``profiling_runs``.
        trials: Number of timed trials that will be (or were) run.
        warmup_trials: Number of warmup trials that are discarded.
    """
    return RunEnvironment(
        run_id=run_id,
        run_ts=datetime.date.today(),
        git_sha=_git_sha(),
        git_dirty=_git_dirty(),
        hostname=socket.gethostname(),
        cpu_model=_cpu_model(),
        n_cores=os.cpu_count() or 1,
        total_ram_mb=_total_ram_mb(),
        python_version=sys.version.split()[0],
        polars_version=_lib_version("polars"),
        numpy_version=_lib_version("numpy"),
        blas_threads=_blas_threads(),
        trials=trials,
        warmup_trials=warmup_trials,
    )
