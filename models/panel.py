"""Panel-aware data handling: long (date, id) → aligned numpy arrays.

The synthetic datasets live in long format: one row per (date, id) pair.
Training a model requires stacking these into dense matrices while keeping
date/id provenance so that:
  - CV splitters can split by date (not by sample row), preserving the
    cross-section structure within each date.
  - Predictions can be tagged back to (date, id) for downstream use.

Public API
----------
``build_panel`` — join features + target + optional weights, NaN-mask rows
    with missing data, and return aligned X / y / groups / weights / provenance.
``date_ordinals`` — convert a polars Date series to integer ordinals usable
    as the ``groups`` argument to the splitters in ``models.splitters``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class PanelArrays:
    """Aligned numpy arrays for a (date, id) feature panel.

    Attributes
    ----------
    X:
        (n_samples, n_features) float64 feature matrix; NaN-rows removed.
    y:
        (n_samples,) float64 target vector.
    groups:
        (n_samples,) int64 date ordinals — one per sample, same ordinal for
        all assets on the same date.  Used as ``groups`` in CV splitters.
    weights:
        (n_samples,) float64 sample weights, or ones if none were supplied.
    dates:
        (n_samples,) object array of ``datetime.date`` values; same order as X.
    ids:
        (n_samples,) int64 asset IDs; same order as X.
    feature_names:
        Tuple of feature column names in the order they appear in X.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    weights: np.ndarray
    dates: np.ndarray
    ids: np.ndarray
    feature_names: tuple[str, ...]


def date_ordinals(date_series: pl.Series) -> np.ndarray:
    """Convert a polars Date series to integer ordinals (``date.toordinal()``).

    Using toordinal() rather than integer positions keeps ordinals stable across
    different date ranges, which matters when embargo_periods is expressed in
    calendar days.
    """
    return np.array([d.toordinal() for d in date_series.to_list()], dtype=np.int64)


def _join_features_target(
    features: pl.DataFrame,
    target: pl.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> pl.DataFrame:
    """Inner-join feature panel with forward-return target on (date, id).

    Rows present in one frame but absent from the other are silently dropped,
    which is the correct behaviour when forward-return trailing rows are NaN.
    """
    return features.select(["date", "id", *feature_cols]).join(
        target.select(["date", "id", target_col]),
        on=["date", "id"],
        how="inner",
    )


def _attach_weights(joined: pl.DataFrame, weights: pl.DataFrame | None) -> pl.DataFrame:
    """Left-join per-sample weights onto the joined panel; fill missing with 1.0."""
    if weights is not None:
        joined = joined.join(
            weights.select(["date", "id", "weight"]), on=["date", "id"], how="left"
        )
        return joined.with_columns(pl.col("weight").fill_null(1.0))
    return joined.with_columns(pl.lit(1.0).alias("weight"))


def _drop_null_rows(
    joined: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pl.DataFrame:
    """Drop any row that has a null or NaN in any feature column or the target.

    Polars distinguishes null from floating-point NaN; both are masked here.
    """
    null_exprs = [pl.col(c).is_null() | pl.col(c).is_nan() for c in feature_cols]
    null_exprs += [pl.col(target_col).is_null() | pl.col(target_col).is_nan()]
    return joined.filter(~pl.any_horizontal(*null_exprs))


def _extract_arrays(
    joined: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract aligned numpy arrays from the cleaned joined DataFrame.

    Returns ``(X, y, weights, dates_arr, ids_arr)`` in the row order of
    ``joined`` (caller is responsible for sorting before calling this).
    """
    X = joined.select(feature_cols).to_numpy(allow_copy=True).astype(np.float64)
    y = joined[target_col].to_numpy(allow_copy=True).astype(np.float64)
    w = joined["weight"].to_numpy(allow_copy=True).astype(np.float64)
    dates_arr = np.array(joined["date"].to_list(), dtype=object)
    ids_arr = joined["id"].to_numpy(allow_copy=True).astype(np.int64)
    return X, y, w, dates_arr, ids_arr


def build_panel(
    features: pl.DataFrame,
    target: pl.DataFrame,
    target_col: str,
    *,
    weights: pl.DataFrame | None = None,
    feature_cols: list[str] | None = None,
) -> PanelArrays:
    """Join feature panel + forward-return target into aligned numpy arrays.

    Parameters
    ----------
    features:
        Long-format DataFrame with columns ``date``, ``id``, and one or more
        feature columns.  Typically from ``etl.datasets.gen_feature_panel``.
    target:
        Long-format DataFrame with columns ``date``, ``id``, and at least
        ``target_col``.  Typically from ``etl.datasets.gen_forward_returns``.
    target_col:
        Name of the target column in ``target`` (e.g. ``"fwd_ret_1"``).
    weights:
        Optional long-format DataFrame with columns ``date``, ``id``,
        ``weight``.  If None, uniform weights of 1.0 are used.
    feature_cols:
        Subset of feature columns to include.  Defaults to all columns in
        ``features`` that are not ``date`` or ``id``.

    Returns
    -------
    PanelArrays
        All arrays share the same row order; NaN rows (missing target or any
        feature) are removed before returning.

    Notes
    -----
    The join is an inner join on (date, id), so rows present in features but
    absent in target (or vice versa) are dropped silently.  This is the correct
    behaviour when forward returns have NaN-filled trailing rows.
    """
    if feature_cols is None:
        feature_cols = [c for c in features.columns if c not in ("date", "id")]

    joined = _join_features_target(features, target, target_col, feature_cols)
    joined = _attach_weights(joined, weights)
    # sort by (date, id) for deterministic row order and contiguous date groups
    joined = joined.sort(["date", "id"])
    joined = _drop_null_rows(joined, feature_cols, target_col)

    X, y, w, dates_arr, ids_arr = _extract_arrays(joined, feature_cols, target_col)
    grp = date_ordinals(joined["date"])

    return PanelArrays(
        X=X,
        y=y,
        groups=grp,
        weights=w,
        dates=dates_arr,
        ids=ids_arr,
        feature_names=tuple(feature_cols),
    )
