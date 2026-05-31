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

from dataclasses import dataclass

import polars as pl


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
    """Outcome of a full regression check for one param-point + run combo."""

    passed: bool
    violations: tuple[RegressionViolation, ...]


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

    Returns:
        ``RegressionReport`` with ``passed=True`` iff no violations found.

    The three measured metrics are:
      - ``elapsed_s`` compared against ``elapsed_s_p50`` baseline
      - ``result_mb`` compared against ``result_mb`` baseline
      - ``peak_rss_mb`` compared against ``peak_rss_mb`` baseline
    """
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

    return RegressionReport(passed=len(violations) == 0, violations=tuple(violations))
