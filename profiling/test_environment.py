"""Tests for environment.py — run/environment metadata capture."""

from __future__ import annotations

import sys

from profiling.environment import RunEnvironment, capture_environment


def test_capture_returns_run_environment() -> None:
    env = capture_environment("test-run-001")
    assert isinstance(env, RunEnvironment)


def test_run_id_preserved() -> None:
    env = capture_environment("my-run-42")
    assert env.run_id == "my-run-42"


def test_trials_and_warmup_stored() -> None:
    env = capture_environment("r", trials=7, warmup_trials=2)
    assert env.trials == 7
    assert env.warmup_trials == 2


def test_python_version_matches() -> None:
    env = capture_environment("r")
    assert env.python_version == sys.version.split()[0]


def test_numpy_version_populated() -> None:
    import numpy as np

    env = capture_environment("r")
    assert env.numpy_version == np.__version__


def test_polars_version_populated() -> None:
    import polars as pl

    env = capture_environment("r")
    assert env.polars_version == pl.__version__


def test_n_cores_positive() -> None:
    env = capture_environment("r")
    assert env.n_cores >= 1


def test_total_ram_mb_positive_on_linux() -> None:
    import platform

    env = capture_environment("r")
    if platform.system() == "Linux":
        assert env.total_ram_mb > 0.0


def test_hostname_nonempty() -> None:
    env = capture_environment("r")
    assert len(env.hostname) > 0


def test_git_sha_nonempty() -> None:
    """Git SHA is 'unknown' outside a repo; never empty."""
    env = capture_environment("r")
    assert len(env.git_sha) > 0


def test_blas_threads_positive(monkeypatch) -> None:
    """BLAS thread count must be >= 1 even when no env vars are set."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)
    env = capture_environment("r")
    assert env.blas_threads >= 1


def test_blas_threads_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    env = capture_environment("r")
    assert env.blas_threads == 4
