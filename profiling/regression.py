"""Regression detection: compare current-run metrics against stored baselines.

A "regression" is when a current measurement exceeds the baseline value by
more than either a percentage threshold OR an absolute threshold — whichever
is triggered first.  Both thresholds exist because:
  - Relative thresholds (pct) are appropriate for large absolute values where
    a 20% increase matters.
  - Absolute thresholds prevent false alarms on stages that already run in
    milliseconds: a 20% increase of 1 ms is not actionable.

The comparison is against ``stage_baselines`` (p50 elapsed, result_mb,
peak_rss_mb) and ``regression_thresholds`` (per-stage, per-metric limits).

The caller aggregates per-trial ``TrialResult``s into per-stage median metrics
before comparing; that choice (median over trials) is intentional — a single
slow trial can be a fluke, but the median cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from .scaling import ScalingFit, stage_metric_r_squared


@dataclass(frozen=True)
class RegressionViolation:
    """One metric of one stage exceeded its threshold."""

    stage: str
    metric: str
    baseline_value: float
    current_value: float
    pct_increase: float
    abs_increase: float
    triggered_by: str  # "pct" | "abs"


@dataclass(frozen=True)
class RegressionReport:
    """Outcome of a full regression check for one param-point + run combo.

    ``scaling_fit_confidence_ok`` / ``excluded_low_confidence`` are populated
    only when ``check_regressions`` is called with ``min_r_squared`` set; when
    confidence gating is off they stay ``None`` / ``()`` so the report is
    identical to a pre-gating run.
    """

    passed: bool
    violations: tuple[RegressionViolation, ...]
    # None  → confidence gating was not requested (min_r_squared is None).
    # True  → gating on, every checked (stage, metric) cleared the r² floor.
    # False → gating on, at least one (stage, metric) was excluded as too noisy.
    scaling_fit_confidence_ok: bool | None = None
    # (stage, metric) pairs skipped because their scaling-fit r² < min_r_squared.
    excluded_low_confidence: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def _check_metric(
    stage: str,
    metric: str,
    baseline_value: float,
    current_value: float,
    max_pct: float,
    max_abs: float,
) -> RegressionViolation | None:
    """Return a violation if either threshold is exceeded, else None."""
    abs_inc = current_value - baseline_value
    # A decrease is never a regression.
    if abs_inc <= 0:
        return None
    pct_inc = abs_inc / baseline_value if baseline_value > 0 else float("inf")
    triggered: str | None = None
    if pct_inc > max_pct:
        triggered = "pct"
    elif abs_inc > max_abs:
        triggered = "abs"
    if triggered is None:
        return None
    return RegressionViolation(
        stage=stage,
        metric=metric,
        baseline_value=baseline_value,
        current_value=current_value,
        pct_increase=pct_inc,
        abs_increase=abs_inc,
        triggered_by=triggered,
    )


def check_regressions(
    current_metrics: pl.DataFrame,
    baselines: pl.DataFrame,
    thresholds: pl.DataFrame,
    scaling_fits: list[ScalingFit] | None = None,
    min_r_squared: float | None = None,
) -> RegressionReport:
    """Compare ``current_metrics`` against ``baselines`` using ``thresholds``.

    Args:
        current_metrics: Per-stage summary with columns
            ``stage`` (str/cat), ``elapsed_s`` (float), ``result_mb`` (float),
            ``peak_rss_mb`` (float).  Typically the median across trials for
            one param-point.
        baselines: Rows from ``stage_baselines`` for the matching param-point.
            Must have columns ``stage``, ``elapsed_s_p50``, ``result_mb``,
            ``peak_rss_mb``.
        thresholds: Rows from ``regression_thresholds`` with columns
            ``stage``, ``metric``, ``max_pct_increase``, ``max_abs_increase``,
            ``min_samples``.
        scaling_fits: Optional scaling fits (from ``fit_scaling``) used only when
            ``min_r_squared`` is set, to look up each (stage, metric)'s fit r².
        min_r_squared: Optional confidence floor in [0, 1].  When set, any
            (stage, metric) whose best scaling-fit r² is below this threshold is
            treated as too noisy to trust and is *excluded* from the regression
            verdict — guarding against false alarms on noisy grids.  When
            ``None`` (default) no gating happens and the result is byte-identical
            to a call without these arguments.  A (stage, metric) with no fit is
            never excluded (absence of a fit is not evidence of noise).

    Returns:
        ``RegressionReport`` with ``passed=True`` iff no violations found.  When
        ``min_r_squared`` is set, ``scaling_fit_confidence_ok`` /
        ``excluded_low_confidence`` record which pairs were skipped for low
        confidence.

    The three measured metrics are:
      - ``elapsed_s`` compared against ``elapsed_s_p50`` baseline
      - ``result_mb`` compared against ``result_mb`` baseline
      - ``peak_rss_mb`` compared against ``peak_rss_mb`` baseline
    """
    confidence = stage_metric_r_squared(scaling_fits or []) if min_r_squared is not None else {}
    excluded: list[tuple[str, str]] = []

    # Normalise stage column to string for join compatibility (Categorical keys
    # can mismatch across DataFrames built independently).
    def _str_stage(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("stage").cast(pl.String))

    cur = _str_stage(current_metrics)
    bas = _str_stage(baselines)
    thr = _str_stage(thresholds)

    # metric name in current_metrics → baseline column name
    metric_col_map = {
        "elapsed_s": "elapsed_s_p50",
        "result_mb": "result_mb",
        "peak_rss_mb": "peak_rss_mb",
    }

    violations: list[RegressionViolation] = []

    for stage_row in cur.iter_rows(named=True):
        stage = stage_row["stage"]
        bas_rows = bas.filter(pl.col("stage") == stage)
        if bas_rows.is_empty():
            continue  # no baseline for this stage — skip, not a failure
        bas_row = bas_rows.row(0, named=True)

        thr_rows = thr.filter(pl.col("stage") == stage)

        for metric, bas_col in metric_col_map.items():
            current_value = stage_row.get(metric)
            baseline_value = bas_row.get(bas_col)
            if current_value is None or baseline_value is None:
                continue

            # Confidence gate: skip stages whose scaling fit is too noisy to
            # trust. A missing fit is not evidence of noise, so it's not skipped.
            if min_r_squared is not None:
                r2 = confidence.get((stage, metric))
                if r2 is not None and r2 < min_r_squared:
                    excluded.append((stage, metric))
                    continue

            # Look up threshold for this (stage, metric) pair
            thr_match = thr_rows.filter(pl.col("metric") == metric)
            if thr_match.is_empty():
                continue
            thr_row = thr_match.row(0, named=True)

            violation = _check_metric(
                stage=stage,
                metric=metric,
                baseline_value=float(baseline_value),
                current_value=float(current_value),
                max_pct=float(thr_row["max_pct_increase"]),
                max_abs=float(thr_row["max_abs_increase"]),
            )
            if violation is not None:
                violations.append(violation)

    confidence_ok = None if min_r_squared is None else (len(excluded) == 0)
    return RegressionReport(
        passed=len(violations) == 0,
        violations=tuple(violations),
        scaling_fit_confidence_ok=confidence_ok,
        excluded_low_confidence=tuple(excluded),
    )
