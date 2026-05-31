"""Tests for provenance.py.

Covers:
- hash_data_files: determinism, change detection, empty-dir edge case
- build_manifest: stable hashes over a fixture workdir, env injection
- write_manifest / read_manifest: JSON roundtrip equality
- diff_manifests: identical manifests → identical=True; changed data hash,
  env field, and spec field each → identical=False with correct attribution
"""

from __future__ import annotations

import dataclasses
import datetime
from pathlib import Path

from etl.datasets import GenSpec
from profiling.environment import RunEnvironment
from provenance import (
    ManifestDiff,
    RunManifest,
    build_manifest,
    diff_manifests,
    hash_data_files,
    read_manifest,
    write_manifest,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

_FIXED_TS = "2026-01-01T00:00:00+00:00"

_ENV = RunEnvironment(
    run_id="test-run-001",
    run_ts=datetime.date(2026, 1, 1),
    git_sha="abc1234",
    git_dirty=False,
    hostname="test-host",
    cpu_model="Test CPU @ 3.0GHz",
    n_cores=8,
    total_ram_mb=16384.0,
    python_version="3.13.0",
    polars_version="1.0.0",
    numpy_version="2.0.0",
    blas_threads=4,
    trials=1,
    warmup_trials=0,
)

_SPEC = GenSpec(n_assets=10, n_dates=20, n_features=3, n_factors=2, seed=42)


def _write_parquet_bytes(path, content: bytes) -> None:
    """Write arbitrary bytes to *path* (used to create fake parquet files)."""
    path.write_bytes(content)


def _fixture_workdir(tmp_path) -> object:
    """Create a temporary workdir with two fake parquet files."""
    (tmp_path / "prices.parquet").write_bytes(b"prices-data")
    (tmp_path / "returns.parquet").write_bytes(b"returns-data")
    return tmp_path


# --------------------------------------------------------------------------- #
# hash_data_files
# --------------------------------------------------------------------------- #


def test_hash_data_files_deterministic(tmp_path) -> None:
    """Same bytes always produce the same hash."""
    (tmp_path / "a.parquet").write_bytes(b"hello")
    (tmp_path / "b.parquet").write_bytes(b"world")

    result1 = hash_data_files(tmp_path)
    result2 = hash_data_files(tmp_path)
    assert result1 == result2


def test_hash_data_files_keys_are_relative(tmp_path) -> None:
    """Keys must be relative posix paths, not absolute."""
    (tmp_path / "prices.parquet").write_bytes(b"data")
    hashes = hash_data_files(tmp_path)
    assert list(hashes.keys()) == ["prices.parquet"]


def test_hash_data_files_sorted(tmp_path) -> None:
    """Keys are sorted for determinism regardless of filesystem order."""
    for name in ["z.parquet", "a.parquet", "m.parquet"]:
        (tmp_path / name).write_bytes(name.encode())
    hashes = hash_data_files(tmp_path)
    assert list(hashes.keys()) == sorted(hashes.keys())


def test_hash_data_files_changed_file_changes_hash(tmp_path) -> None:
    """Mutating a file's bytes changes its digest."""
    p = tmp_path / "prices.parquet"
    p.write_bytes(b"original")
    before = hash_data_files(tmp_path)["prices.parquet"]
    p.write_bytes(b"modified")
    after = hash_data_files(tmp_path)["prices.parquet"]
    assert before != after


def test_hash_data_files_empty_dir(tmp_path) -> None:
    """An empty workdir returns an empty dict."""
    assert hash_data_files(tmp_path) == {}


def test_hash_data_files_ignores_non_parquet(tmp_path) -> None:
    """Only *.parquet files are hashed; other files are ignored."""
    (tmp_path / "manifest.json").write_bytes(b"ignored")
    (tmp_path / "prices.parquet").write_bytes(b"counted")
    hashes = hash_data_files(tmp_path)
    assert set(hashes.keys()) == {"prices.parquet"}


def test_hash_data_files_subdirectory(tmp_path) -> None:
    """Files in subdirectories are included with relative paths."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.parquet").write_bytes(b"nested")
    hashes = hash_data_files(tmp_path)
    assert "sub/nested.parquet" in hashes


# --------------------------------------------------------------------------- #
# build_manifest
# --------------------------------------------------------------------------- #


def test_build_manifest_stable_hashes(tmp_path) -> None:
    """Calling build_manifest twice on the same workdir gives equal data_hashes."""
    (tmp_path / "prices.parquet").write_bytes(b"prices-data")
    (tmp_path / "returns.parquet").write_bytes(b"returns-data")

    m1 = build_manifest("run-a", _SPEC, tmp_path, env=_ENV, created_ts=_FIXED_TS)
    m2 = build_manifest("run-a", _SPEC, tmp_path, env=_ENV, created_ts=_FIXED_TS)
    assert m1.data_hashes == m2.data_hashes


def test_build_manifest_injected_env(tmp_path) -> None:
    """When env is injected, build_manifest uses it verbatim."""
    m = build_manifest("run-b", _SPEC, tmp_path, env=_ENV, created_ts=_FIXED_TS)
    assert m.env is _ENV


def test_build_manifest_captures_env_when_none(tmp_path) -> None:
    """When env=None, build_manifest calls capture_environment."""
    m = build_manifest("run-auto", _SPEC, tmp_path, env=None, created_ts=_FIXED_TS)
    assert m.env.run_id == "run-auto"


def test_build_manifest_embeds_spec(tmp_path) -> None:
    """The GenSpec is stored verbatim on the manifest."""
    m = build_manifest("run-c", _SPEC, tmp_path, env=_ENV, created_ts=_FIXED_TS)
    assert m.gen_spec == _SPEC


def test_build_manifest_created_ts(tmp_path) -> None:
    """The supplied created_ts is embedded unchanged."""
    m = build_manifest("run-d", _SPEC, tmp_path, env=_ENV, created_ts=_FIXED_TS)
    assert m.created_ts == _FIXED_TS


def test_build_manifest_real_parquet(tmp_path) -> None:
    """build_manifest over a real polars-written parquet file yields a non-empty hash."""
    import polars as pl

    df = pl.DataFrame({"x": [1, 2, 3]})
    df.write_parquet(tmp_path / "data.parquet")

    m = build_manifest("run-polars", _SPEC, tmp_path, env=_ENV, created_ts=_FIXED_TS)
    assert len(m.data_hashes) == 1
    assert len(m.data_hashes["data.parquet"]) == 64  # sha256 hex = 64 chars


# --------------------------------------------------------------------------- #
# write_manifest / read_manifest roundtrip
# --------------------------------------------------------------------------- #


def _make_manifest(tmp_path: Path) -> tuple[RunManifest, Path]:
    (tmp_path / "prices.parquet").write_bytes(b"prices-data")
    m = build_manifest("round-trip", _SPEC, tmp_path, env=_ENV, created_ts=_FIXED_TS)
    return m, tmp_path


def test_roundtrip_run_id(tmp_path) -> None:
    m, wd = _make_manifest(tmp_path)
    dest = wd / "manifest.json"
    write_manifest(m, dest)
    loaded = read_manifest(dest)
    assert loaded.run_id == m.run_id


def test_roundtrip_created_ts(tmp_path) -> None:
    m, wd = _make_manifest(tmp_path)
    dest = wd / "manifest.json"
    write_manifest(m, dest)
    loaded = read_manifest(dest)
    assert loaded.created_ts == m.created_ts


def test_roundtrip_data_hashes(tmp_path) -> None:
    m, wd = _make_manifest(tmp_path)
    dest = wd / "manifest.json"
    write_manifest(m, dest)
    loaded = read_manifest(dest)
    assert loaded.data_hashes == m.data_hashes


def test_roundtrip_gen_spec(tmp_path) -> None:
    m, wd = _make_manifest(tmp_path)
    dest = wd / "manifest.json"
    write_manifest(m, dest)
    loaded = read_manifest(dest)
    assert loaded.gen_spec == m.gen_spec


def test_roundtrip_env(tmp_path) -> None:
    m, wd = _make_manifest(tmp_path)
    dest = wd / "manifest.json"
    write_manifest(m, dest)
    loaded = read_manifest(dest)
    assert loaded.env == m.env


def test_roundtrip_full_equality(tmp_path) -> None:
    """A written-then-read manifest equals the original in all fields."""
    m, wd = _make_manifest(tmp_path)
    dest = wd / "manifest.json"
    write_manifest(m, dest)
    loaded = read_manifest(dest)
    assert loaded == m


def test_write_manifest_is_valid_json(tmp_path) -> None:
    """The output file is valid JSON."""
    import json as _json

    m, wd = _make_manifest(tmp_path)
    dest = wd / "manifest.json"
    write_manifest(m, dest)
    raw = dest.read_text(encoding="utf-8")
    parsed = _json.loads(raw)
    assert parsed["run_id"] == m.run_id


# --------------------------------------------------------------------------- #
# diff_manifests
# --------------------------------------------------------------------------- #


def _base_manifest(tmp_path) -> RunManifest:
    (tmp_path / "prices.parquet").write_bytes(b"prices-data")
    (tmp_path / "returns.parquet").write_bytes(b"returns-data")
    return build_manifest("diff-base", _SPEC, tmp_path, env=_ENV, created_ts=_FIXED_TS)


def test_diff_identical_manifests(tmp_path) -> None:
    """Two manifests built from the same inputs are identical."""
    (tmp_path / "prices.parquet").write_bytes(b"prices-data")
    m1 = build_manifest("id-1", _SPEC, tmp_path, env=_ENV, created_ts=_FIXED_TS)
    m2 = build_manifest("id-2", _SPEC, tmp_path, env=_ENV, created_ts=_FIXED_TS)
    d = diff_manifests(m1, m2)
    assert d.identical is True
    assert d.added_files == []
    assert d.removed_files == []
    assert d.changed_files == []
    assert d.env_diffs == {}
    assert d.spec_diffs == {}


def test_diff_changed_data_hash(tmp_path) -> None:
    """A modified file is reported in changed_files and identical=False."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    dir_a.joinpath("prices.parquet").write_bytes(b"original")
    dir_b.joinpath("prices.parquet").write_bytes(b"modified")

    m1 = build_manifest("chg-a", _SPEC, dir_a, env=_ENV, created_ts=_FIXED_TS)
    m2 = build_manifest("chg-b", _SPEC, dir_b, env=_ENV, created_ts=_FIXED_TS)
    d = diff_manifests(m1, m2)

    assert d.identical is False
    assert "prices.parquet" in d.changed_files
    assert d.added_files == []
    assert d.removed_files == []


def test_diff_added_file(tmp_path) -> None:
    """A file present only in b is reported in added_files."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    dir_a.joinpath("prices.parquet").write_bytes(b"data")
    dir_b.joinpath("prices.parquet").write_bytes(b"data")
    dir_b.joinpath("extra.parquet").write_bytes(b"extra")

    m1 = build_manifest("add-a", _SPEC, dir_a, env=_ENV, created_ts=_FIXED_TS)
    m2 = build_manifest("add-b", _SPEC, dir_b, env=_ENV, created_ts=_FIXED_TS)
    d = diff_manifests(m1, m2)

    assert d.identical is False
    assert "extra.parquet" in d.added_files
    assert d.removed_files == []


def test_diff_removed_file(tmp_path) -> None:
    """A file present only in a is reported in removed_files."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    dir_a.joinpath("prices.parquet").write_bytes(b"data")
    dir_a.joinpath("extra.parquet").write_bytes(b"extra")
    dir_b.joinpath("prices.parquet").write_bytes(b"data")

    m1 = build_manifest("rm-a", _SPEC, dir_a, env=_ENV, created_ts=_FIXED_TS)
    m2 = build_manifest("rm-b", _SPEC, dir_b, env=_ENV, created_ts=_FIXED_TS)
    d = diff_manifests(m1, m2)

    assert d.identical is False
    assert "extra.parquet" in d.removed_files
    assert d.added_files == []


def test_diff_env_field_difference(tmp_path) -> None:
    """A differing env field is reported in env_diffs and identical=False."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    env_b = dataclasses.replace(_ENV, hostname="other-host")

    m1 = build_manifest("env-a", _SPEC, dir_a, env=_ENV, created_ts=_FIXED_TS)
    m2 = build_manifest("env-b", _SPEC, dir_b, env=env_b, created_ts=_FIXED_TS)
    d = diff_manifests(m1, m2)

    assert d.identical is False
    assert "hostname" in d.env_diffs
    assert d.env_diffs["hostname"] == ("test-host", "other-host")


def test_diff_spec_field_difference(tmp_path) -> None:
    """A differing spec field is reported in spec_diffs and identical=False."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    spec_b = dataclasses.replace(_SPEC, seed=99)

    m1 = build_manifest("spec-a", _SPEC, dir_a, env=_ENV, created_ts=_FIXED_TS)
    m2 = build_manifest("spec-b", spec_b, dir_b, env=_ENV, created_ts=_FIXED_TS)
    d = diff_manifests(m1, m2)

    assert d.identical is False
    assert "seed" in d.spec_diffs
    assert d.spec_diffs["seed"] == (42, 99)


def test_diff_multiple_differences(tmp_path) -> None:
    """Multiple simultaneous differences are all reported."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    dir_a.joinpath("prices.parquet").write_bytes(b"original")
    dir_b.joinpath("prices.parquet").write_bytes(b"changed")

    env_b = dataclasses.replace(_ENV, n_cores=16)
    spec_b = dataclasses.replace(_SPEC, n_assets=20)

    m1 = build_manifest("multi-a", _SPEC, dir_a, env=_ENV, created_ts=_FIXED_TS)
    m2 = build_manifest("multi-b", spec_b, dir_b, env=env_b, created_ts=_FIXED_TS)
    d = diff_manifests(m1, m2)

    assert d.identical is False
    assert "prices.parquet" in d.changed_files
    assert "n_cores" in d.env_diffs
    assert "n_assets" in d.spec_diffs


def test_diff_returns_manifest_diff_type(tmp_path) -> None:
    """diff_manifests always returns a ManifestDiff."""
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    m = build_manifest("type-check", _SPEC, dir_a, env=_ENV, created_ts=_FIXED_TS)
    d = diff_manifests(m, m)
    assert isinstance(d, ManifestDiff)
