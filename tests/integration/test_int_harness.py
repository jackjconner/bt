"""The profiling harness itself: one timed trial set per component per param
point, persisted and re-readable, with scaling curves fit over the grid."""

from __future__ import annotations

from etl.datasets import GenSpec
from harness import build_components, run_harness
from profiling import read_artifacts, read_measurements, read_runs


def test_harness_profiles_every_component(tmp_path) -> None:
    grid = [
        GenSpec(n_assets=20, n_dates=60, n_features=5, n_factors=3, seed=0),
        GenSpec(n_assets=40, n_dates=60, n_features=5, n_factors=3, seed=0),
        GenSpec(n_assets=80, n_dates=60, n_features=5, n_factors=3, seed=0),
    ]
    n_components = len(build_components())
    report = run_harness(grid, tmp_path, n_trials=2, warmup=1)

    # one TrialResult per (component, param point)
    assert len(report.stats) == n_components * len(grid)
    stages = {s.stage for s in report.stats}
    assert stages == {"etl", "signals", "models", "analysis", "portfolio", "backtest", "profiling"}

    # percentile ordering holds for every measured component
    for s in report.stats:
        assert s.elapsed_min <= s.elapsed_p50 <= s.elapsed_p90 <= s.elapsed_p95

    # measurements persisted and re-readable, schema-compatible
    persisted = read_measurements(tmp_path)
    assert persisted.height == len(report.stats) * 2  # n_trials=2
    assert read_runs(tmp_path).height == 1

    # scaling curves were fit over the grid
    assert len(report.scaling_fits) > 0


def test_harness_flame_graphs_opt_in(tmp_path) -> None:
    """A profiles_dir triggers a separate flame-graph pass: 4 artifacts per
    (component, param point) — CPU speedscope+calltree, memory bin+summary."""
    store = tmp_path / "store"
    profiles = tmp_path / "profiles"
    grid = [GenSpec(n_assets=20, n_dates=60, n_features=5, n_factors=3, seed=0)]
    n_components = len(build_components())

    run_harness(grid, store, n_trials=1, warmup=0, profiles_dir=profiles)

    idx = read_artifacts(profiles)
    assert idx.height == n_components * len(grid) * 4
    assert set(idx["fmt"].to_list()) == {"speedscope", "calltree", "memray-bin", "summary"}
    assert idx["run_id"].n_unique() == 1
    # every indexed blob actually exists on disk
    for path in idx["path"].to_list():
        assert (profiles / path).exists()


def test_harness_no_profiles_dir_skips_capture(tmp_path) -> None:
    """Without profiles_dir the flame-graph pass is skipped entirely."""
    grid = [GenSpec(n_assets=20, n_dates=60, n_features=5, n_factors=3, seed=0)]
    run_harness(grid, tmp_path, n_trials=1, warmup=0)
    assert not (tmp_path / "profile_artifacts.parquet").exists()


def test_harness_regression_check_optional(tmp_path) -> None:
    from etl.datasets import generate

    spec = GenSpec(n_assets=20, n_dates=60, n_features=5, n_factors=3, seed=0)
    report = run_harness(
        [spec],
        tmp_path,
        n_trials=2,
        warmup=0,
        baselines=generate("stage_baselines", spec),
        thresholds=generate("regression_thresholds", spec),
    )
    assert report.regression is not None
    assert report.regression.passed == (len(report.regression.violations) == 0)
