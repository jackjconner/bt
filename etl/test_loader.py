"""Tests for DatasetLoader (features 1 & 2)."""

from __future__ import annotations

from typing import cast

import polars as pl
import pytest

from .datasets import REGISTRY, GenSpec, write_all
from .loader import DatasetLoader

SPEC = GenSpec(n_assets=8, n_dates=20, n_features=4, n_factors=2, seed=42)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("loader_data")
    write_all(d, SPEC)
    return d


def test_load_returns_correct_schema(data_dir):
    loader = DatasetLoader(data_dir, SPEC)
    df = loader.load("prices")
    REGISTRY["prices"].schema_for(SPEC).validate(df)
    assert df.height == SPEC.n_assets * SPEC.n_dates


def test_load_unknown_name_raises(data_dir):
    loader = DatasetLoader(data_dir, SPEC)
    with pytest.raises(KeyError, match="no_such_dataset"):
        loader.load("no_such_dataset")


def test_load_missing_file_raises(tmp_path):
    loader = DatasetLoader(tmp_path, SPEC)
    with pytest.raises(FileNotFoundError):
        loader.load("prices")


def test_date_filter_pushdown(data_dir):
    loader = DatasetLoader(data_dir, SPEC)
    df_all = loader.load("prices")
    all_dates = sorted(df_all["date"].unique().to_list())
    cut = all_dates[len(all_dates) // 2]
    df_filtered = loader.load("prices", end=cut)
    assert df_filtered["date"].max() <= cut
    assert df_filtered.height < df_all.height


def test_start_date_filter(data_dir):
    loader = DatasetLoader(data_dir, SPEC)
    df_all = loader.load("prices")
    all_dates = sorted(df_all["date"].unique().to_list())
    cut = all_dates[len(all_dates) // 2]
    df_filtered = loader.load("prices", start=cut)
    assert df_filtered["date"].min() >= cut


def test_id_filter_pushdown(data_dir):
    loader = DatasetLoader(data_dir, SPEC)
    subset_ids = [0, 1, 2]
    df = loader.load("prices", ids=subset_ids)
    assert set(df["id"].unique().to_list()) == set(subset_ids)
    assert df.height == len(subset_ids) * SPEC.n_dates


def test_column_projection(data_dir):
    loader = DatasetLoader(data_dir, SPEC)
    df = loader.load("prices", columns=["date", "id", "close"])
    assert "close" in df.columns
    assert "open" not in df.columns


def test_validate_false_skips_check(data_dir):
    loader = DatasetLoader(data_dir, SPEC, validate=False)
    df = loader.load("prices")
    assert df.height > 0


def test_scan_returns_lazy_frame(data_dir):
    loader = DatasetLoader(data_dir, SPEC)
    lf = loader.scan("prices")
    assert isinstance(lf, pl.LazyFrame)
    df = cast(pl.DataFrame, lf.collect())
    assert df.height == SPEC.n_assets * SPEC.n_dates


def test_scan_with_date_and_id_filter(data_dir):
    loader = DatasetLoader(data_dir, SPEC)
    df_all = loader.load("prices")
    all_dates = sorted(df_all["date"].unique().to_list())
    cut = all_dates[5]
    lf = loader.scan("prices", end=cut, ids=[0, 1])
    df = cast(pl.DataFrame, lf.collect())
    assert df["date"].max() <= cut
    assert set(df["id"].unique().to_list()).issubset({0, 1})


def test_load_non_panel_dataset(data_dir):
    """Datasets without an 'id' column (e.g. risk_free_rate) load cleanly."""
    loader = DatasetLoader(data_dir, SPEC)
    df = loader.load("risk_free_rate")
    assert "annual_rate" in df.columns
    assert df.height == SPEC.n_dates
