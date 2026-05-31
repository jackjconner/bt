from __future__ import annotations

from dataclasses import dataclass

import polars as pl


class SchemaError(ValueError):
    """Raised when a DataFrame violates its declared schema."""


@dataclass(frozen=True)
class Column:
    name: str
    dtype: pl.DataType
    nullable: bool = False


@dataclass(frozen=True)
class Schema:
    """Declared shape of a dataset: ordered columns, dtypes, and key columns.

    The schema is the contract the synthetic generator must satisfy and the
    contract a loader validates a source against at the boundary, so a
    malformed parquet fails loudly here rather than corrupting a downstream
    matrix.
    """

    name: str
    columns: tuple[Column, ...]
    keys: tuple[str, ...] = ()

    def column(self, name: str) -> Column:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def validate(self, df: pl.DataFrame) -> None:
        actual = dict(df.schema)
        for col in self.columns:
            if col.name not in actual:
                raise SchemaError(f"{self.name}: missing column {col.name!r}")
            got = actual[col.name]
            if got != col.dtype:
                raise SchemaError(
                    f"{self.name}: column {col.name!r} has dtype {got}, expected {col.dtype}"
                )
            if not col.nullable and df[col.name].null_count() > 0:
                raise SchemaError(f"{self.name}: non-nullable column {col.name!r} has nulls")
        if self.keys:
            n_dupe = df.select(self.keys).is_duplicated().sum()
            if n_dupe > 0:
                raise SchemaError(f"{self.name}: {n_dupe} duplicate rows on key {self.keys}")

    def empty(self) -> pl.DataFrame:
        return pl.DataFrame(schema={c.name: c.dtype for c in self.columns})


def col(name: str, dtype: pl.DataType, nullable: bool = False) -> Column:
    return Column(name=name, dtype=dtype, nullable=nullable)
