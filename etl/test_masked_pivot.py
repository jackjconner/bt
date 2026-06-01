"""Tests for masked pivot (explicit missing-data policy)."""

from __future__ import annotations

from datetime import date

import polars as pl

from .masked_pivot import to_masked_matrix


def _frame_with_gap() -> pl.DataFrame:
    """3 dates × 3 assets, asset 1 missing on date 2."""
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    ids = [0, 1, 2]
    rows = []
    for d in dates:
        for i in ids:
            if d == date(2020, 1, 3) and i == 1:
                continue  # gap
            rows.append({"date": d, "id": i, "value": float(i + 1)})
    return pl.DataFrame(rows, schema={"date": pl.Date, "id": pl.Int64, "value": pl.Float64})


def test_shape():
    df = _frame_with_gap()
    mat, mask, _dates, _ids = to_masked_matrix(df, "value")
    assert mat.shape == (3, 3)
    assert mask.shape == (3, 3)


def test_axes_sorted():
    df = _frame_with_gap()
    _, _, dates, ids = to_masked_matrix(df, "value")
    assert dates == sorted(dates)
    assert ids == sorted(ids)


def test_missing_cell_is_zero_and_mask_false():
    df = _frame_with_gap()
    mat, mask, _dates, _ids = to_masked_matrix(df, "value")
    # date(2020,1,3) is index 1; id=1 is index 1
    assert mat[1, 1] == 0.0
    assert not mask[1, 1]


def test_present_cell_has_correct_value_and_mask_true():
    df = _frame_with_gap()
    mat, mask, _dates, _ids = to_masked_matrix(df, "value")
    # id=2 has value 3.0 everywhere except the gap
    assert mat[0, 2] == 3.0
    assert mask[0, 2]


def test_fully_populated_frame_all_mask_true():
    dates = [date(2020, 1, 2), date(2020, 1, 3)]
    rows = [{"date": d, "id": i, "value": float(i)} for d in dates for i in range(3)]
    df = pl.DataFrame(rows, schema={"date": pl.Date, "id": pl.Int64, "value": pl.Float64})
    _, mask, _, _ = to_masked_matrix(df, "value")
    assert mask.all()


def test_nan_value_treated_as_missing():
    """Explicit NaN in the source should be flagged as missing, not as a value."""
    rows = [
        {"date": date(2020, 1, 2), "id": 0, "value": 1.0},
        {"date": date(2020, 1, 2), "id": 1, "value": float("nan")},
    ]
    df = pl.DataFrame(rows, schema={"date": pl.Date, "id": pl.Int64, "value": pl.Float64})
    mat, mask, _, _ = to_masked_matrix(df, "value")
    assert not mask[0, 1]
    assert mat[0, 1] == 0.0


def test_non_contiguous_ids_map_to_ascending_columns():
    """Sparse / non-0-based ids still map to ascending column positions."""
    rows = [
        {"date": date(2020, 1, 2), "id": 7, "value": 7.0},
        {"date": date(2020, 1, 2), "id": 42, "value": 42.0},
        {"date": date(2020, 1, 3), "id": 7, "value": 70.0},
        {"date": date(2020, 1, 3), "id": 42, "value": 420.0},
    ]
    df = pl.DataFrame(rows, schema={"date": pl.Date, "id": pl.Int64, "value": pl.Float64})
    mat, mask, _dates, ids = to_masked_matrix(df, "value")
    assert ids == [7, 42]
    assert mat[0, 0] == 7.0 and mat[0, 1] == 42.0
    assert mat[1, 0] == 70.0 and mat[1, 1] == 420.0
    assert mask.all()


def test_duplicate_key_resolves_to_first_row():
    """A duplicate (date, id) key keeps the first such row (matches pivot 'first')."""
    rows = [
        {"date": date(2020, 1, 2), "id": 0, "value": 10.0},
        {"date": date(2020, 1, 2), "id": 0, "value": 20.0},
        {"date": date(2020, 1, 3), "id": 0, "value": 30.0},
    ]
    df = pl.DataFrame(rows, schema={"date": pl.Date, "id": pl.Int64, "value": pl.Float64})
    mat, mask, _dates, _ids = to_masked_matrix(df, "value")
    assert mat.shape == (2, 1)
    assert mat[0, 0] == 10.0
    assert mat[1, 0] == 30.0
    assert mask.all()
