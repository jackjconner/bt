from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from .datasets import REGISTRY, GenSpec, generate, generate_all, write_all
from .source import to_matrix

SPEC = GenSpec(n_assets=12, n_dates=40, n_features=6, n_factors=3, seed=7)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_each_dataset_matches_its_schema(name: str) -> None:
    # generate() validates internally; assert it returns a non-empty frame
    df = generate(name, SPEC)
    assert df.height > 0
    REGISTRY[name].schema_for(SPEC).validate(df)


def test_generate_all_covers_registry() -> None:
    frames = generate_all(SPEC)
    assert set(frames) == set(REGISTRY)


def test_deterministic_under_seed() -> None:
    a = generate("prices", SPEC)
    b = generate("prices", SPEC)
    assert a.equals(b)


def test_prices_reconcile_with_returns() -> None:
    """close[t]/close[t-1]-1 must equal the percent returns / 100 (DECISIONS.md)."""
    from .source import generate_returns

    rets = generate_returns(SPEC.n_assets, SPEC.n_dates, SPEC.start, seed=SPEC.seed)
    R, _ = to_matrix(rets, "return")
    prices = generate("prices", SPEC)
    C, _ = to_matrix(prices, "close")
    implied = (C[1:] / C[:-1] - 1.0) * 100.0
    np.testing.assert_allclose(implied, R[1:], rtol=1e-6, atol=1e-6)


def test_feature_panel_has_injected_predictive_signal() -> None:
    feats = generate("feature_panel", SPEC)
    fwd = generate("forward_returns", SPEC)
    joined = feats.join(fwd.select("date", "id", "fwd_ret_1"), on=["date", "id"]).drop_nulls()
    # at least one feature should correlate with next-day return beyond noise
    corrs = [
        abs(np.corrcoef(joined[f"feat_{i}"], joined["fwd_ret_1"])[0, 1])
        for i in range(SPEC.n_features)
    ]
    assert max(corrs) > 0.05


def test_write_all_roundtrip(tmp_path) -> None:
    paths = write_all(tmp_path, SPEC)
    assert set(paths) == set(REGISTRY)
    for name, path in paths.items():
        loaded = pl.read_parquet(path)
        REGISTRY[name].schema_for(SPEC).validate(loaded)
