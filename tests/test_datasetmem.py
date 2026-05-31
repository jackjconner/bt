from datasetmem import format_report, measure
from etl.datasets import REGISTRY, GenSpec


def test_measure_covers_registry_largest_first() -> None:
    spec = GenSpec(n_assets=12, n_dates=20, n_features=3, n_factors=2, seed=1)
    sizes = measure(spec)
    assert {s.name for s in sizes} == set(REGISTRY)
    assert sum(s.n_bytes for s in sizes) > 0
    assert [s.n_bytes for s in sizes] == sorted((s.n_bytes for s in sizes), reverse=True)


def test_report_has_total_and_spec() -> None:
    spec = GenSpec(n_assets=12, n_dates=20, n_features=3, n_factors=2, seed=1)
    report = format_report(spec, measure(spec))
    assert "TOTAL" in report
    assert "n_assets=12" in report
