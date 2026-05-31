"""Reproducibility/provenance manifest for pipeline runs.

Captures the INPUTS of a pipeline run — content hashes of the generated
parquet files, the full execution environment, and the GenSpec — into a
``RunManifest``.  Two manifests can be compared with ``diff_manifests`` to
prove (or disprove) that two runs share identical inputs.

This module is intentionally standalone: it has no side-effects at import
time, contains only pure typed functions, and depends solely on stdlib
(``hashlib``, ``json``, ``dataclasses``, ``pathlib``) plus modules already
present in the project (``profiling.environment``, ``etl.datasets``).
Pipeline wiring (calling ``build_manifest`` from a real run) is deferred.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from etl.datasets import GenSpec
from profiling.environment import RunEnvironment, capture_environment

# --------------------------------------------------------------------------- #
# Core data structures
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class RunManifest:
    """Immutable record of the inputs that produced a pipeline run.

    Attributes:
        run_id:      Unique identifier for the run (supplied by the caller).
        env:         Execution environment snapshot captured at build time.
        gen_spec:    The GenSpec used to generate the dataset for this run.
        data_hashes: SHA-256 hex digest of each ``*.parquet`` file under the
                     workdir, keyed by the file's path relative to the
                     workdir root.  Sorted for determinism.
        created_ts:  ISO-8601 timestamp supplied by the caller; the module
                     never reads the wall clock itself.
    """

    run_id: str
    env: RunEnvironment
    gen_spec: GenSpec
    data_hashes: dict[str, str]
    created_ts: str


@dataclasses.dataclass(frozen=True)
class ManifestDiff:
    """Structured difference between two ``RunManifest`` instances.

    Attributes:
        identical:         ``True`` iff every field in both manifests agrees.
        added_files:       Files present in ``b`` but not in ``a``.
        removed_files:     Files present in ``a`` but not in ``b``.
        changed_files:     Files whose SHA-256 digest changed between ``a``
                           and ``b``.
        env_diffs:         Mapping of ``RunEnvironment`` field name →
                           ``(a_value, b_value)`` for fields that differ.
        spec_diffs:        Mapping of ``GenSpec`` field name →
                           ``(a_value, b_value)`` for fields that differ.
    """

    identical: bool
    added_files: list[str]
    removed_files: list[str]
    changed_files: list[str]
    env_diffs: dict[str, tuple[object, object]]
    spec_diffs: dict[str, tuple[object, object]]


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def hash_data_files(workdir: Path) -> dict[str, str]:
    """SHA-256 digest of every ``*.parquet`` file under *workdir*.

    Args:
        workdir: Root directory that was written by a pipeline run.

    Returns:
        A sorted ``{relative_path: hex_digest}`` mapping.  Paths use forward
        slashes regardless of OS.  An empty dict is returned when *workdir*
        contains no parquet files.
    """
    result: dict[str, str] = {}
    for path in sorted(workdir.rglob("*.parquet")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(workdir).as_posix()
        result[relative] = digest
    return result


# --------------------------------------------------------------------------- #
# Manifest construction
# --------------------------------------------------------------------------- #


def build_manifest(
    run_id: str,
    gen_spec: GenSpec,
    workdir: Path,
    env: RunEnvironment | None = None,
    *,
    created_ts: str | None = None,
) -> RunManifest:
    """Compose a ``RunManifest`` from the run's inputs.

    Args:
        run_id:     Unique identifier for the run.
        gen_spec:   The ``GenSpec`` used to generate the dataset.
        workdir:    Directory containing the ``*.parquet`` files written by
                    ``etl.datasets.write_all``.
        env:        Pre-captured ``RunEnvironment``; if ``None``,
                    ``capture_environment(run_id)`` is called automatically.
        created_ts: ISO-8601 timestamp string to embed in the manifest.  The
                    caller is responsible for supplying this — the module does
                    not read the wall clock itself.  Defaults to
                    ``datetime.datetime.now(datetime.UTC).isoformat()`` when
                    not provided, but callers should supply a stable value for
                    reproducible tests.

    Returns:
        A frozen ``RunManifest`` ready for serialisation.
    """
    resolved_env = env if env is not None else capture_environment(run_id)
    resolved_ts = (
        created_ts if created_ts is not None else datetime.datetime.now(datetime.UTC).isoformat()
    )
    return RunManifest(
        run_id=run_id,
        env=resolved_env,
        gen_spec=gen_spec,
        data_hashes=hash_data_files(workdir),
        created_ts=resolved_ts,
    )


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #


def _env_to_dict(env: RunEnvironment) -> dict[str, Any]:
    """Convert a ``RunEnvironment`` to a JSON-serialisable dict."""
    d = dataclasses.asdict(env)
    # run_ts is a datetime.date — convert to ISO string
    if isinstance(d.get("run_ts"), datetime.date):
        d["run_ts"] = d["run_ts"].isoformat()
    return d


def _spec_to_dict(spec: GenSpec) -> dict[str, Any]:
    """Convert a ``GenSpec`` to a JSON-serialisable dict."""
    return dataclasses.asdict(spec)


def _manifest_to_dict(manifest: RunManifest) -> dict[str, Any]:
    return {
        "run_id": manifest.run_id,
        "env": _env_to_dict(manifest.env),
        "gen_spec": _spec_to_dict(manifest.gen_spec),
        "data_hashes": manifest.data_hashes,
        "created_ts": manifest.created_ts,
    }


def _dict_to_env(d: dict[str, Any]) -> RunEnvironment:
    """Reconstruct a ``RunEnvironment`` from a plain dict."""
    d = dict(d)
    run_ts_raw = d.get("run_ts")
    if isinstance(run_ts_raw, str):
        d["run_ts"] = datetime.date.fromisoformat(run_ts_raw)
    return RunEnvironment(**d)


def _dict_to_spec(d: dict[str, Any]) -> GenSpec:
    """Reconstruct a ``GenSpec`` from a plain dict."""
    return GenSpec(**d)


def _dict_to_manifest(d: dict[str, Any]) -> RunManifest:
    return RunManifest(
        run_id=d["run_id"],
        env=_dict_to_env(d["env"]),
        gen_spec=_dict_to_spec(d["gen_spec"]),
        data_hashes=dict(d["data_hashes"]),
        created_ts=d["created_ts"],
    )


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def write_manifest(manifest: RunManifest, path: Path) -> None:
    """Serialise *manifest* to a pretty-printed JSON file at *path*.

    Args:
        manifest: The ``RunManifest`` to write.
        path:     Destination file path.  Parent directories must exist.
    """
    path.write_text(json.dumps(_manifest_to_dict(manifest), indent=2) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> RunManifest:
    """Deserialise a ``RunManifest`` from the JSON file at *path*.

    Args:
        path: Source file path previously written by ``write_manifest``.

    Returns:
        A fully reconstructed frozen ``RunManifest``.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _dict_to_manifest(raw)


# --------------------------------------------------------------------------- #
# Diff
# --------------------------------------------------------------------------- #


def diff_manifests(a: RunManifest, b: RunManifest) -> ManifestDiff:
    """Compare two ``RunManifest`` instances and report every difference.

    Args:
        a: The baseline manifest.
        b: The manifest to compare against *a*.

    Returns:
        A frozen ``ManifestDiff``.  ``identical`` is ``True`` iff every field
        in both manifests agrees exactly.
    """
    keys_a = set(a.data_hashes)
    keys_b = set(b.data_hashes)

    added = sorted(keys_b - keys_a)
    removed = sorted(keys_a - keys_b)
    changed = sorted(k for k in keys_a & keys_b if a.data_hashes[k] != b.data_hashes[k])

    env_diffs: dict[str, tuple[object, object]] = {}
    for field in dataclasses.fields(a.env):
        va = getattr(a.env, field.name)
        vb = getattr(b.env, field.name)
        if va != vb:
            env_diffs[field.name] = (va, vb)

    spec_diffs: dict[str, tuple[object, object]] = {}
    for field in dataclasses.fields(a.gen_spec):
        va = getattr(a.gen_spec, field.name)
        vb = getattr(b.gen_spec, field.name)
        if va != vb:
            spec_diffs[field.name] = (va, vb)

    identical = not (added or removed or changed or env_diffs or spec_diffs)

    return ManifestDiff(
        identical=identical,
        added_files=added,
        removed_files=removed,
        changed_files=changed,
        env_diffs=env_diffs,
        spec_diffs=spec_diffs,
    )
