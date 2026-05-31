from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from .schema import Schema, SchemaError, col


def _schema() -> Schema:
    return Schema(
        name="toy",
        columns=(
            col("date", pl.Date),
            col("id", pl.Int64),
            col("value", pl.Float64),
            col("note", pl.String, nullable=True),
        ),
        keys=("date", "id"),
    )


def _good() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2020, 1, 1), date(2020, 1, 2)],
            "id": [0, 1],
            "value": [1.0, 2.0],
            "note": [None, "x"],
        },
        schema={"date": pl.Date, "id": pl.Int64, "value": pl.Float64, "note": pl.String},
    )


def test_validate_passes_on_good_frame() -> None:
    _schema().validate(_good())


def test_missing_column_raises() -> None:
    df = _good().drop("value")
    with pytest.raises(SchemaError, match="missing column 'value'"):
        _schema().validate(df)


def test_wrong_dtype_raises() -> None:
    df = _good().with_columns(pl.col("value").cast(pl.Int64))
    with pytest.raises(SchemaError, match="dtype"):
        _schema().validate(df)


def test_null_in_non_nullable_raises() -> None:
    df = _good().with_columns(
        pl.when(pl.col("id") == 0).then(None).otherwise(pl.col("value")).alias("value")
    )
    with pytest.raises(SchemaError, match="nulls"):
        _schema().validate(df)


def test_duplicate_key_raises() -> None:
    df = pl.concat([_good(), _good().head(1)])
    with pytest.raises(SchemaError, match="duplicate"):
        _schema().validate(df)


def test_empty_frame_matches_schema() -> None:
    s = _schema()
    s.validate(s.empty())
