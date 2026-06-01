"""Scaling-curve fitting via OLS in log-log space.

For each (stage, metric, scaling_dim) combination we fit:

    log(metric) = slope * log(dim) + intercept + ε

The slope is the exponent in the power law  metric ∝ dim^slope.  A slope ≥ 2
indicates super-linear (quadratic or worse) scaling that will likely become a
bottleneck at production data sizes.

Why log-log: it linearises power-law relationships so a single OLS fit covers
many orders of magnitude without heteroscedasticity bias.

Why scipy.stats.linregress instead of numpy.polyfit: linregress returns r²
directly and avoids building a full Vandermonde matrix for a two-parameter fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats


@dataclass(frozen=True)
class ScalingFit:
    """One fitted log-log relationship — one row in ``scaling_fits``."""

    run_id: str
    stage: str
    metric: str
    scaling_dim: str
    log_log_slope: float
    intercept: float  # intercept in log-log space (log of scale factor)
    r_squared: float
    n_points: int


def fit_scaling(
    measurements: pl.DataFrame,
    run_id: str,
    metrics: tuple[str, ...] = ("elapsed_s", "result_mb", "peak_traced_mb", "peak_rss_mb"),
    scaling_dims: tuple[str, ...] = ("n_assets", "n_dates", "n_features", "n_factors"),
    min_points: int = 3,
) -> list[ScalingFit]:
    """Fit log-log scaling curves for every (stage, metric, scaling_dim) combo.

    ``measurements`` must contain one median-aggregated row per
    (stage, param_point_id) with columns for each dim and each metric.  The
    caller is responsible for aggregating raw per-trial rows down to medians
    before passing them here — fitting on raw (noisy) measurements would give
    misleading slope estimates.

    Args:
        measurements: DataFrame with columns ``stage``, each dim in
            ``scaling_dims``, and each metric in ``metrics``.
        run_id: Identifier written into every returned ``ScalingFit``.
        metrics: Metric column names to fit.
        scaling_dims: Dimension column names to use as x-axis.
        min_points: Minimum number of distinct dim values required to fit.
            Fewer points makes the slope estimate unreliable.

    Returns:
        List of ``ScalingFit`` records, one per valid (stage, metric, dim).
        Combinations with fewer than ``min_points`` distinct x-values or with
        any non-positive values (which break log) are silently skipped.
    """
    results: list[ScalingFit] = []
    stages = measurements["stage"].cast(pl.String).unique().to_list()
    avail_metrics = [m for m in metrics if m in measurements.columns]
    avail_dims = [d for d in scaling_dims if d in measurements.columns]

    for stage in stages:
        stage_df = measurements.filter(pl.col("stage").cast(pl.String) == stage)

        # Hoist all per-stage Polars work out of the inner loops: pull each dim
        # and metric column to NumPy once (Float64, NaN for nulls — so np.isnan
        # below reproduces the original per-(dim,metric) .drop_nulls()), and take
        # each dim's mode once. The original recomputed these filters/modes for
        # every (metric, dim) pair; the controlled subset and the modes depend
        # only on the stage, so doing them once is exact and ~7× cheaper.
        dim_cols = {d: stage_df[d].to_numpy().astype(float) for d in avail_dims}
        metric_cols = {m: stage_df[m].to_numpy().astype(float) for m in avail_metrics}
        modes: dict[str, object] = {}
        for d in avail_dims:
            mode = stage_df[d].mode()
            modes[d] = None if mode.is_empty() else mode[0]
        n_rows = stage_df.height

        # Control for confounders: fit each dim's slope using only the points
        # where every OTHER varied dimension sits at its baseline (modal) value.
        # On a single-axis grid the others are constant, so this is a no-op; on an
        # anchored multi-axis grid it yields a clean partial slope per dim instead
        # of a confounded pooled one. The controlled subset depends only on the
        # dim (not the metric), so build each per-dim mask + x array once here —
        # the original rebuilt them for every (metric, dim) pair. Masks are NumPy
        # over the Polars-computed modes, preserving Polars' mode tie-break.
        x_by_dim: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for dim in avail_dims:
            controlled = np.ones(n_rows, dtype=bool)
            for other in avail_dims:
                if other == dim or modes[other] is None:
                    continue
                controlled &= dim_cols[other] == modes[other]
            x_by_dim[dim] = (dim_cols[dim][controlled], controlled)

        # Iterate (metric, dim) — the original append order — so the returned
        # list (and the scaling_fits rows it feeds) is byte-identical.
        for metric in avail_metrics:
            for dim in avail_dims:
                x_dim, controlled = x_by_dim[dim]
                y_metric = metric_cols[metric][controlled]

                # Drop rows null in either column (NaN) or non-positive (log
                # undefined) — equals the original drop_nulls + (x>0)&(y>0).
                valid = (x_dim > 0) & (y_metric > 0)
                x, y = x_dim[valid], y_metric[valid]
                if len(x) == 0:
                    continue

                # Aggregate to unique x-values by taking the median y per x to
                # reduce noise from multiple param points with the same dim value.
                unique_x = np.unique(x)
                if len(unique_x) < min_points:
                    continue
                med_y = np.array([np.median(y[x == xi]) for xi in unique_x])

                log_x = np.log(unique_x)
                log_y = np.log(med_y)

                slope, intercept, r, _, _ = stats.linregress(log_x, log_y)

                results.append(
                    ScalingFit(
                        run_id=run_id,
                        stage=stage,
                        metric=metric,
                        scaling_dim=dim,
                        log_log_slope=float(slope),
                        intercept=float(intercept),
                        r_squared=float(r**2),
                        n_points=len(unique_x),
                    )
                )

    return results


def stage_metric_r_squared(fits: list[ScalingFit]) -> dict[tuple[str, str], float]:
    """Best (max) fit r² per (stage, metric), pooling across scaling dims.

    A (stage, metric) is fit independently against each scaling dim; the highest
    r² across those dims is the strongest evidence that the metric scales
    predictably for that stage, so it is the natural confidence score. Regression
    gating uses this to decide whether a stage's measurements are trustworthy
    enough to raise an alarm on (see ``regression.check_regressions``).
    """
    best: dict[tuple[str, str], float] = {}
    for f in fits:
        key = (f.stage, f.metric)
        prev = best.get(key)
        if prev is None or f.r_squared > prev:
            best[key] = f.r_squared
    return best


def fits_to_dataframe(fits: list[ScalingFit]) -> pl.DataFrame:
    """Convert a list of ``ScalingFit`` to a Polars DataFrame.

    The resulting schema matches ``scaling_fits`` in etl.datasets.
    """
    if not fits:
        return pl.DataFrame(
            schema={
                "run_id": pl.String,
                "stage": pl.Categorical,
                "metric": pl.Categorical,
                "scaling_dim": pl.Categorical,
                "log_log_slope": pl.Float64,
                "intercept": pl.Float64,
                "r_squared": pl.Float64,
                "n_points": pl.Int64,
            }
        )
    rows = [
        {
            "run_id": f.run_id,
            "stage": f.stage,
            "metric": f.metric,
            "scaling_dim": f.scaling_dim,
            "log_log_slope": f.log_log_slope,
            "intercept": f.intercept,
            "r_squared": f.r_squared,
            "n_points": f.n_points,
        }
        for f in fits
    ]
    return pl.DataFrame(rows).with_columns(
        pl.col("stage").cast(pl.Categorical),
        pl.col("metric").cast(pl.Categorical),
        pl.col("scaling_dim").cast(pl.Categorical),
        pl.col("n_points").cast(pl.Int64),
    )
