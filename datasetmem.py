"""Materialize every synthetic dataset for a GenSpec in memory and report size.

Generates the full ``etl.datasets.REGISTRY`` in memory via ``generate_all`` and
reports each polars frame's ``estimated_size`` (and the total), so you can see how
much RAM a given ``(n_assets, n_dates, …)`` configuration costs *before* running
the pipeline or harness on it.

    scripts/dataset-mem --n-assets 3000 --n-dates 5040
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from etl.datasets import GenSpec, generate_all


@dataclass(frozen=True)
class DatasetSize:
    name: str
    rows: int
    cols: int
    n_bytes: int


def measure(spec: GenSpec, *, validate: bool = False) -> list[DatasetSize]:
    """Materialize all datasets for ``spec`` and measure each frame's size.

    Returns the sizes sorted largest-first. ``validate=False`` (the default)
    skips schema validation since we only care about memory footprint here.
    """
    data = generate_all(spec, validate=validate)
    sizes = [
        DatasetSize(name=name, rows=df.height, cols=df.width, n_bytes=int(df.estimated_size()))
        for name, df in data.items()
    ]
    return sorted(sizes, key=lambda s: s.n_bytes, reverse=True)


def _mb(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


def format_report(spec: GenSpec, sizes: list[DatasetSize]) -> str:
    total = sum(s.n_bytes for s in sizes)
    lines = [
        f"GenSpec  n_assets={spec.n_assets}  n_dates={spec.n_dates}  "
        f"n_features={spec.n_features}  n_factors={spec.n_factors}  seed={spec.seed}",
        f"{len(sizes)} datasets materialized in memory (polars estimated_size)",
        "",
        f"{'dataset':<26}{'rows':>14}{'cols':>6}{'est_size':>14}",
        f"{'-' * 26}{'-' * 14}{'-' * 6}{'-' * 14}",
    ]
    lines += [f"{s.name:<26}{s.rows:>14,}{s.cols:>6}{_mb(s.n_bytes):>11.2f} MB" for s in sizes]
    lines.append(f"{'-' * 26}{'-' * 14}{'-' * 6}{'-' * 14}")
    lines.append(f"{'TOTAL':<26}{'':>14}{'':>6}{_mb(total):>11.2f} MB")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Materialize a GenSpec's synthetic datasets in memory and report their size."
    )
    p.add_argument("--n-assets", type=int, required=True)
    p.add_argument("--n-dates", type=int, required=True)
    p.add_argument("--n-features", type=int, default=20)
    p.add_argument("--n-factors", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--validate", action="store_true", help="validate each frame against its schema")
    a = p.parse_args()
    spec = GenSpec(
        n_assets=a.n_assets,
        n_dates=a.n_dates,
        n_features=a.n_features,
        n_factors=a.n_factors,
        seed=a.seed,
    )
    print(format_report(spec, measure(spec, validate=a.validate)))


if __name__ == "__main__":
    main()
