"""Tests for regression.py — threshold-based metric comparison."""

from __future__ import annotations

import polars as pl

from profiling.regression import check_regressions
from profiling.scaling import ScalingFit


def _fit(stage: str, metric: str, r_squared: float, dim: str = "n_assets") -> ScalingFit:
    return ScalingFit(
        run_id="test",
        stage=stage,
        metric=metric,
        scaling_dim=dim,
        log_log_slope=1.0,
        intercept=0.0,
        r_squared=r_squared,
        n_points=4,
    )


def _make_baselines(
    elapsed_p50: float = 1.0, result_mb: float = 100.0, peak_rss_mb: float = 500.0
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "stage": pl.Series(["etl.batch"], dtype=pl.Categorical),
            "param_point_id": [0],
            "baseline_id": ["baseline_main"],
            "elapsed_s_p50": [elapsed_p50],
            "result_mb": [result_mb],
            "peak_rss_mb": [peak_rss_mb],
        }
    )


def _make_thresholds(max_pct: float = 0.20, max_abs: float = 0.05) -> pl.DataFrame:
    metrics = ["elapsed_s", "result_mb", "peak_rss_mb"]
    return pl.DataFrame(
        {
            "stage": pl.Series(["etl.batch"] * len(metrics), dtype=pl.Categorical),
            "metric": pl.Series(metrics, dtype=pl.Categorical),
            "max_pct_increase": [max_pct] * len(metrics),
            "max_abs_increase": [max_abs] * len(metrics),
            "min_samples": [3] * len(metrics),
        }
    )


def _make_current(
    elapsed_s: float = 1.0, result_mb: float = 100.0, peak_rss_mb: float = 500.0
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "stage": pl.Series(["etl.batch"], dtype=pl.Categorical),
            "elapsed_s": [elapsed_s],
            "result_mb": [result_mb],
            "peak_rss_mb": [peak_rss_mb],
        }
    )


def test_no_regression_passes() -> None:
    report = check_regressions(
        _make_current(elapsed_s=1.0),
        _make_baselines(elapsed_p50=1.0),
        _make_thresholds(max_pct=0.20, max_abs=0.05),
    )
    assert report.passed
    assert len(report.violations) == 0


def test_pct_regression_flagged() -> None:
    """30% increase against a 20% threshold must flag a violation."""
    report = check_regressions(
        _make_current(elapsed_s=1.30),
        _make_baselines(elapsed_p50=1.00),
        _make_thresholds(max_pct=0.20, max_abs=999.0),  # only pct matters here
    )
    assert not report.passed
    assert len(report.violations) >= 1
    v = next(v for v in report.violations if v.metric == "elapsed_s")
    assert v.triggered_by == "pct"
    assert abs(v.pct_increase - 0.30) < 1e-9


def test_abs_regression_flagged() -> None:
    """Small absolute increase triggers when pct threshold is loose."""
    # 0.1 s absolute increase; pct is 1% (< 20% threshold) but abs > 0.05 limit
    report = check_regressions(
        _make_current(elapsed_s=10.1),
        _make_baselines(elapsed_p50=10.0),
        _make_thresholds(max_pct=0.20, max_abs=0.05),
    )
    assert not report.passed
    v = next(v for v in report.violations if v.metric == "elapsed_s")
    assert v.triggered_by == "abs"


def test_decrease_never_flagged() -> None:
    """A metric that improved (decreased) must not be reported as a regression."""
    report = check_regressions(
        _make_current(elapsed_s=0.5),
        _make_baselines(elapsed_p50=1.0),
        _make_thresholds(max_pct=0.20, max_abs=0.05),
    )
    assert report.passed


def test_missing_stage_in_baselines_skipped() -> None:
    """A stage with no baseline row should not produce a violation."""
    current = pl.DataFrame(
        {
            "stage": pl.Series(["new_stage"], dtype=pl.Categorical),
            "elapsed_s": [99.0],
            "result_mb": [999.0],
            "peak_rss_mb": [9999.0],
        }
    )
    report = check_regressions(current, _make_baselines(), _make_thresholds())
    assert report.passed


def test_min_r_squared_none_is_byte_identical() -> None:
    """Passing scaling_fits but min_r_squared=None leaves verdicts unchanged."""
    args = (
        _make_current(elapsed_s=1.30),
        _make_baselines(elapsed_p50=1.00),
        _make_thresholds(max_pct=0.20, max_abs=999.0),
    )
    base = check_regressions(*args)
    with_fits = check_regressions(
        *args,
        scaling_fits=[_fit("etl.batch", "elapsed_s", 0.10)],
        min_r_squared=None,
    )
    assert with_fits.passed == base.passed
    assert with_fits.violations == base.violations
    assert with_fits.scaling_fit_confidence_ok is None
    assert with_fits.excluded_low_confidence == ()


def test_low_r_squared_excludes_noisy_regression() -> None:
    """A genuine threshold breach is suppressed when its fit r² is below the floor."""
    report = check_regressions(
        _make_current(elapsed_s=1.30),  # 30% over baseline — would normally flag
        _make_baselines(elapsed_p50=1.00),
        _make_thresholds(max_pct=0.20, max_abs=999.0),
        scaling_fits=[_fit("etl.batch", "elapsed_s", 0.30)],  # noisy fit
        min_r_squared=0.90,
    )
    assert report.passed
    assert len(report.violations) == 0
    assert report.scaling_fit_confidence_ok is False
    assert ("etl.batch", "elapsed_s") in report.excluded_low_confidence


def test_high_r_squared_still_flags_regression() -> None:
    """A real regression on a well-fit (high r²) stage is still reported."""
    report = check_regressions(
        _make_current(elapsed_s=1.30),
        _make_baselines(elapsed_p50=1.00),
        _make_thresholds(max_pct=0.20, max_abs=999.0),
        scaling_fits=[_fit("etl.batch", "elapsed_s", 0.995)],
        min_r_squared=0.90,
    )
    assert not report.passed
    assert any(v.metric == "elapsed_s" for v in report.violations)
    assert report.scaling_fit_confidence_ok is True
    assert report.excluded_low_confidence == ()


def test_missing_fit_with_gating_does_not_exclude() -> None:
    """When gating is on but a (stage, metric) has no fit, it is checked as usual."""
    report = check_regressions(
        _make_current(elapsed_s=1.30),
        _make_baselines(elapsed_p50=1.00),
        _make_thresholds(max_pct=0.20, max_abs=999.0),
        scaling_fits=[_fit("other.stage", "elapsed_s", 0.10)],
        min_r_squared=0.90,
    )
    # No fit for etl.batch/elapsed_s → cannot judge noise → keep the violation.
    assert not report.passed
    assert any(v.metric == "elapsed_s" for v in report.violations)
    assert report.excluded_low_confidence == ()


def test_multiple_metric_violations() -> None:
    """All three metrics exceeding thresholds should each produce a violation."""
    report = check_regressions(
        _make_current(elapsed_s=2.0, result_mb=200.0, peak_rss_mb=1000.0),
        _make_baselines(elapsed_p50=1.0, result_mb=100.0, peak_rss_mb=500.0),
        _make_thresholds(max_pct=0.10, max_abs=999.0),
    )
    assert not report.passed
    metrics_violated = {v.metric for v in report.violations}
    assert "elapsed_s" in metrics_violated
    assert "result_mb" in metrics_violated
    assert "peak_rss_mb" in metrics_violated
