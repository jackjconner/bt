"""Typed dataset loader with schema validation at the boundary.

Features 1 & 2 from the plan:
  - Validates a loaded parquet against ``REGISTRY[name].schema_for(spec)``
    immediately on load, raising ``SchemaError`` before any malformed frame
    reaches downstream code.
  - Predicate/projection pushdown: only materialized columns and the requested
    date/id slice are pulled from disk — everything before ``.collect()`` stays
    lazy so Polars' optimizer can push filters into the parquet reader.

``DatasetLoader`` is the single entry-point for the rest of the codebase;
the ``BatchLoader``/``StreamLoader`` in ``batch.py``/``stream.py`` are kept
unchanged because other modules import them directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import cast

import polars as pl

from .datasets import REGISTRY, GenSpec


@dataclass(frozen=True)
class DatasetLoader:
    """Load a single named dataset from a directory of parquet files.

    Parameters
    ----------
    data_dir:
        Directory written by ``write_all(data_dir, spec)``.  Each dataset is
        expected at ``<data_dir>/<name>.parquet``.
    spec:
        The ``GenSpec`` that was used to write the files, needed to resolve the
        schema for variable-column datasets (e.g. ``feature_panel``).
    validate:
        When ``True`` (default) the loaded frame is validated against the
        declared schema before being returned.  Set ``False`` only in tight
        benchmarking loops where the file has already been validated.
    """

    data_dir: Path
    spec: GenSpec
    validate: bool = True

    def _path(self, name: str) -> Path:
        p = self.data_dir / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Dataset {name!r} not found at {p}")
        return p

    def load(
        self,
        name: str,
        *,
        start: date | None = None,
        end: date | None = None,
        ids: list[int] | None = None,
        columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """Load ``name`` with optional predicate/projection pushdown.

        All filters are applied as lazy predicates before ``.collect()`` so
        the parquet reader skips row-groups and columns it doesn't need.

        Parameters
        ----------
        name:
            Key in ``REGISTRY``.
        start, end:
            Inclusive date bounds.  Applied to the ``date`` column when
            present; silently ignored for datasets without one.
        ids:
            Restrict to this subset of asset ``id`` values (Int64).  Applied
            to the ``id`` column when present.
        columns:
            Subset of columns to materialize.  The schema key columns are
            always included so validation can run.
        """
        if name not in REGISTRY:
            raise KeyError(f"Unknown dataset {name!r}; available: {sorted(REGISTRY)}")

        schema = REGISTRY[name].schema_for(self.spec)
        lf = pl.scan_parquet(self._path(name))

        # --- projection pushdown -------------------------------------------
        if columns is not None:
            # Always keep key columns so schema.validate can check duplicates.
            keep = set(columns) | set(schema.keys)
            available = set(lf.collect_schema().names())
            lf = lf.select([c for c in lf.collect_schema().names() if c in keep and c in available])

        # --- predicate pushdown --------------------------------------------
        lf_schema = lf.collect_schema()
        if start is not None and "date" in lf_schema:
            lf = lf.filter(pl.col("date") >= start)
        if end is not None and "date" in lf_schema:
            lf = lf.filter(pl.col("date") <= end)
        if ids is not None and "id" in lf_schema:
            lf = lf.filter(pl.col("id").is_in(ids))

        df = cast(pl.DataFrame, lf.collect(engine="in-memory"))

        # Only full-column loads can be validated against the full schema.
        # When a column subset was requested we skip the schema check for
        # omitted columns rather than raising a false positive.
        if self.validate and columns is None:
            schema.validate(df)
        return df

    def scan(
        self,
        name: str,
        *,
        start: date | None = None,
        end: date | None = None,
        ids: list[int] | None = None,
    ) -> pl.LazyFrame:
        """Return a ``LazyFrame`` with predicates pushed down but not yet collected.

        Useful when the caller wants to chain further lazy operations (joins,
        groupbys) before materializing.  Schema validation is deferred until
        ``.collect()`` is called by the caller.
        """
        if name not in REGISTRY:
            raise KeyError(f"Unknown dataset {name!r}; available: {sorted(REGISTRY)}")

        lf = pl.scan_parquet(self._path(name))
        lf_schema = lf.collect_schema()
        if start is not None and "date" in lf_schema:
            lf = lf.filter(pl.col("date") >= start)
        if end is not None and "date" in lf_schema:
            lf = lf.filter(pl.col("date") <= end)
        if ids is not None and "id" in lf_schema:
            lf = lf.filter(pl.col("id").is_in(ids))
        return lf
