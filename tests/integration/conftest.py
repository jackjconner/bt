"""Shared fixtures for the integration suite.

One schema-valid synthetic dataset is generated once per session and loaded
through the production `DatasetLoader`, so every contract test exercises the
same data the components see in `pipeline.py` / `harness/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from etl import DatasetLoader
from etl.datasets import GenSpec, write_all


@dataclass(frozen=True)
class Synth:
    spec: GenSpec
    loader: DatasetLoader
    dir: Path


@pytest.fixture(scope="session")
def synth(tmp_path_factory: pytest.TempPathFactory) -> Synth:
    spec = GenSpec(n_assets=40, n_dates=120, n_features=8, n_factors=4, seed=3)
    data_dir = tmp_path_factory.mktemp("synth_data")
    write_all(data_dir, spec)
    return Synth(spec=spec, loader=DatasetLoader(data_dir, spec), dir=data_dir)
