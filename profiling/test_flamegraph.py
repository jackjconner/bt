"""Tests for flamegraph.py — the artifact index, retention, and capture.

Both capture paths are in-process and unprivileged (pyinstrument for CPU, memray
for memory), so they run directly here alongside the pure index/retention tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from profiling.flamegraph import (
    ProfileArtifact,
    capture_cpu,
    capture_memory,
    prune_profiles,
    read_artifacts,
    write_artifacts,
)


def _fake_artifact(
    profiles_dir: Path,
    run_id: str,
    *,
    stage: str = "build_panel",
    fmt: str = "folded",
    on_regression: bool = False,
) -> ProfileArtifact:
    """Create a blob file on disk and return an index record pointing at it."""
    run_dir = profiles_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    blob = run_dir / f"{stage}.0.{fmt}.bin"
    blob.write_bytes(b"x" * 16)
    return ProfileArtifact(
        run_id=run_id,
        param_point_id=0,
        stage=stage,
        profiler="pyinstrument",
        kind="cpu",
        fmt=fmt,
        path=str(blob.relative_to(profiles_dir)),
        size_bytes=blob.stat().st_size,
        on_regression=on_regression,
        git_sha="deadbee",
    )


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    arts = [_fake_artifact(tmp_path, "run-A")]
    write_artifacts(tmp_path, arts)

    idx = read_artifacts(tmp_path)
    assert len(idx) == 1
    row = idx.row(0, named=True)
    assert row["run_id"] == "run-A"
    assert row["fmt"] == "folded"
    assert row["on_regression"] is False


def test_read_empty_store_returns_empty_schema(tmp_path: Path) -> None:
    idx = read_artifacts(tmp_path)
    assert len(idx) == 0
    assert "run_id" in idx.columns
    assert "on_regression" in idx.columns


def test_write_accumulates_across_runs(tmp_path: Path) -> None:
    write_artifacts(tmp_path, [_fake_artifact(tmp_path, "run-A")])
    write_artifacts(tmp_path, [_fake_artifact(tmp_path, "run-B")])

    idx = read_artifacts(tmp_path)
    assert set(idx["run_id"].to_list()) == {"run-A", "run-B"}


def test_prune_keeps_latest_n(tmp_path: Path) -> None:
    for run in ("r1", "r2", "r3", "r4"):
        write_artifacts(tmp_path, [_fake_artifact(tmp_path, run)])

    pruned = prune_profiles(tmp_path, keep_last_n=2)

    assert set(pruned) == {"r1", "r2"}
    idx = read_artifacts(tmp_path)
    assert set(idx["run_id"].to_list()) == {"r3", "r4"}
    assert not (tmp_path / "r1").exists()
    assert (tmp_path / "r4").exists()


def test_prune_protects_on_regression(tmp_path: Path) -> None:
    write_artifacts(tmp_path, [_fake_artifact(tmp_path, "r1", on_regression=True)])
    for run in ("r2", "r3", "r4"):
        write_artifacts(tmp_path, [_fake_artifact(tmp_path, run)])

    pruned = prune_profiles(tmp_path, keep_last_n=2)

    # r1 is old but flagged on_regression, so it survives; only r2 is dropped.
    assert pruned == ["r2"]
    idx = read_artifacts(tmp_path)
    assert set(idx["run_id"].to_list()) == {"r1", "r3", "r4"}
    assert (tmp_path / "r1").exists()


def test_prune_noop_when_within_window(tmp_path: Path) -> None:
    write_artifacts(tmp_path, [_fake_artifact(tmp_path, "r1")])
    assert prune_profiles(tmp_path, keep_last_n=5) == []


def test_prune_rejects_negative(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        prune_profiles(tmp_path, keep_last_n=-1)


def test_capture_cpu_emits_flamegraph_and_calltree(tmp_path: Path) -> None:
    """In-process pyinstrument capture works without ptrace and emits both blobs."""

    def work() -> int:
        total = 0
        for i in range(2_000_000):
            total += i % 7
        return total

    result, arts = capture_cpu(
        "busy",
        work,
        profiles_dir=tmp_path,
        run_id="cpu-run",
        param_point_id=0,
        git_sha="cafef00d",
    )

    assert result > 0
    fmts = {a.fmt for a in arts}
    assert fmts == {"speedscope", "calltree"}
    for a in arts:
        assert (tmp_path / a.path).exists()
        assert a.kind == "cpu"
        assert a.profiler == "pyinstrument"
        assert a.size_bytes > 0


def test_capture_memory_emits_bin_and_summary(tmp_path: Path) -> None:
    """In-process memray capture works without ptrace and produces both blobs."""

    def work() -> list[bytearray]:
        return [bytearray(1_000_000) for _ in range(3)]

    result, arts = capture_memory(
        "alloc",
        work,
        profiles_dir=tmp_path,
        run_id="mem-run",
        param_point_id=0,
        git_sha="cafef00d",
    )

    assert len(result) == 3
    fmts = {a.fmt for a in arts}
    assert fmts == {"memray-bin", "summary"}
    for a in arts:
        assert (tmp_path / a.path).exists()
        assert a.kind == "memory"
        assert a.size_bytes > 0
