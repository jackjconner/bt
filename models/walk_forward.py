"""Walk-forward CV loop with per-fold feature standardization and alpha search.

This is the production-grade replacement for the toy ``cv_loop`` in
``cross_val.py``.  ``cv_loop`` is intentionally preserved for backward
compatibility.

Key differences from ``cv_loop``
---------------------------------
1. **Causality**: any splitter from ``models.splitters`` can be plugged in;
   the default is ``WalkForwardSplitter`` which never trains on future data.
2. **Per-fold scaling**: a ``StandardScaler`` is fit on the train fold only and
   applied to the test fold, eliminating feature-mean/std leakage.
3. **Alpha search**: an inner grid search over ``alpha_grid`` finds the best
   regularization strength per fold using a held-out validation slice inside
   the train window.
4. **Sample weighting**: ``groups``-aligned ``weights`` are forwarded to
   ``model.fit`` as ``sample_weight`` when the model supports it.
5. **IC scoring**: ``rank_ic_score`` is reported alongside R² per fold.
6. **Predictions tagged with (date, id)**: ``WFResult`` carries a provenance
   array so predictions can be joined back to the original panel.

Public API
----------
``WalkForwardConfig``  — configuration dataclass.
``FoldScaler``         — thin wrapper around ``StandardScaler`` with explicit
                         fit-on-train / transform-test semantics.
``FoldResult``         — per-fold output.
``WFResult``           — aggregate result from ``walk_forward_cv``.
``walk_forward_cv``    — main entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from .panel import PanelArrays
from .ridge import ModelResult
from .scoring import held_out_r2, ic_stats, rank_ic_score, rank_ic_series

# --------------------------------------------------------------------------- #
# FoldScaler
# --------------------------------------------------------------------------- #


class FoldScaler:
    """Per-fold StandardScaler that enforces fit-on-train-only discipline.

    sklearn's ``StandardScaler`` is stateful; using it naively on the full
    dataset before splitting leaks test-fold statistics into training.  This
    class makes the correct usage the default: call ``fit_transform(X_train)``
    then ``transform(X_test)``; calling ``transform`` before ``fit_transform``
    raises.
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler | None = None

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on training data and return standardized training features."""
        self._scaler = StandardScaler()
        return self._scaler.fit_transform(X_train)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply train-fold statistics to an arbitrary split (test/validation)."""
        if self._scaler is None:
            raise RuntimeError("fit_transform must be called before transform")
        return self._scaler.transform(X)

    @property
    def mean_(self) -> np.ndarray:
        assert self._scaler is not None
        return cast(np.ndarray, self._scaler.mean_)

    @property
    def scale_(self) -> np.ndarray:
        assert self._scaler is not None
        return cast(np.ndarray, self._scaler.scale_)


# --------------------------------------------------------------------------- #
# Alpha search (inner CV)
# --------------------------------------------------------------------------- #


def _best_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha_grid: list[float],
    model_factory,
    weights_train: np.ndarray | None,
    inner_val_frac: float = 0.2,
) -> float:
    """Simple holdout inner validation to pick the best alpha.

    Uses the last ``inner_val_frac`` of the training fold as a validation
    set (preserving time order), fits each alpha on the earlier portion, and
    returns the alpha with the highest validation R².

    Inner validation is intentionally simple (one holdout) rather than nested
    k-fold because the outer fold already provides an unbiased estimate; the
    inner search just needs a stable selector, not an unbiased metric.
    """
    if len(alpha_grid) == 1:
        return alpha_grid[0]

    n = len(X_train)
    split = max(1, int(n * (1.0 - inner_val_frac)))
    X_iv, y_iv = X_train[:split], y_train[:split]
    X_val, y_val = X_train[split:], y_train[split:]
    w_iv = weights_train[:split] if weights_train is not None else None

    if len(X_val) == 0:
        return alpha_grid[0]

    best_alpha = alpha_grid[0]
    best_r2 = -np.inf
    for a in alpha_grid:
        mdl = model_factory(a)
        try:
            mdl.fit(X_iv, y_iv, sample_weight=w_iv)
        except TypeError:
            mdl.fit(X_iv, y_iv)
        preds = mdl.predict(X_val)
        r2 = held_out_r2(y_val, preds)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = a
    return best_alpha


# --------------------------------------------------------------------------- #
# Config / result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for ``walk_forward_cv``.

    Parameters
    ----------
    alpha_grid:
        List of regularization strengths to search over per fold.  A
        length-1 list skips inner CV (uses that alpha directly).
    scale_features:
        Fit a ``StandardScaler`` on each train fold and apply to test.
        Almost always True for regularized linear models.
    use_sample_weights:
        Forward per-sample weights to ``model.fit`` if the model supports it.
    inner_val_frac:
        Fraction of each train fold held out for inner alpha selection.
    """

    alpha_grid: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    scale_features: bool = True
    use_sample_weights: bool = True
    inner_val_frac: float = 0.2


@dataclass(frozen=True)
class FoldResult:
    """Results for a single CV fold.

    Attributes
    ----------
    fit_result:
        ``ModelResult`` from the final fit (best alpha, full train fold).
    test_r2:
        Held-out R² on this fold's test set.
    test_ic:
        Mean cross-sectional rank IC on the test set.
    ic_values:
        Per-date IC array for the test set.
    chosen_alpha:
        Alpha selected by inner validation (or the sole grid alpha).
    n_train:
        Number of training samples.
    n_test:
        Number of test samples.
    """

    fit_result: ModelResult
    test_r2: float
    test_ic: float
    ic_values: np.ndarray
    chosen_alpha: float
    n_train: int
    n_test: int


@dataclass(frozen=True)
class WFResult:
    """Aggregate output from ``walk_forward_cv``.

    Attributes
    ----------
    fold_results:
        One ``FoldResult`` per CV fold.
    mean_r2:
        Mean held-out R² across folds.
    std_r2:
        Std of held-out R² across folds.
    mean_ic:
        Mean cross-sectional rank IC across all test-fold dates.
    ic_ir:
        IC information ratio (mean / std) across all test-fold dates.
    all_preds:
        (n_test_total,) array of OOS predictions in date/id order.
    all_true:
        (n_test_total,) array of true values aligned with ``all_preds``.
    all_groups:
        (n_test_total,) date ordinals aligned with ``all_preds``.
    all_dates:
        (n_test_total,) date objects aligned with ``all_preds``.
    all_ids:
        (n_test_total,) asset IDs aligned with ``all_preds``.
    predictions_panel:
        Long-format ``pl.DataFrame`` with columns ``(date, id, prediction, fold)``
        covering every OOS observation, keyed for downstream joins.  Folds are
        non-overlapping by construction (each date appears in at most one fold).
        ``None`` when no folds completed.  This field is additive — existing call
        sites that do not use it are unaffected.
    """

    fold_results: list[FoldResult]
    mean_r2: float
    std_r2: float
    mean_ic: float
    ic_ir: float
    all_preds: np.ndarray
    all_true: np.ndarray
    all_groups: np.ndarray
    all_dates: np.ndarray
    all_ids: np.ndarray
    # NEW (additive): per-fold OOS predictions keyed by (date, id, prediction, fold).
    # None when no folds completed. Old call-sites are unaffected (default=None).
    predictions_panel: pl.DataFrame | None = None


# --------------------------------------------------------------------------- #
# Per-fold helpers (private)
# --------------------------------------------------------------------------- #


def _scale_fold(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    scale_features: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally apply per-fold StandardScaler fitted on train data only.

    When ``scale_features`` is False the arrays are returned unchanged.
    When True a fresh ``FoldScaler`` is fitted on ``X_tr`` and both ``X_tr``
    and ``X_te`` are standardised using the train-fold statistics, preventing
    test-distribution leakage into the scaling parameters.
    """
    if not scale_features:
        return X_tr, X_te
    scaler = FoldScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    return X_tr_s, X_te_s


def _fit_fold(
    model_factory,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray | None,
    chosen_alpha: float,
) -> tuple[Any, Any]:
    """Fit the model on the full train fold with the chosen alpha.

    Returns ``(fitted_model, fit_result)`` so the caller can call
    ``fitted_model.predict`` without re-fitting.

    Tries to forward ``sample_weight`` if ``w_tr`` is not None; falls back to
    unweighted fit when the model's ``fit`` does not accept that keyword (some
    sklearn estimators do not).
    """
    model = model_factory(chosen_alpha)
    if w_tr is not None:
        try:
            fit_result = model.fit(X_tr, y_tr, sample_weight=w_tr)
            return model, fit_result
        except TypeError:
            pass
    fit_result = model.fit(X_tr, y_tr)
    return model, fit_result


def _score_fold(
    y_te: np.ndarray,
    preds: np.ndarray,
    grp_te: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Compute held-out R², mean IC, and per-date IC values for a test fold.

    Returns ``(test_r2, test_ic, ic_values)`` where ``ic_values`` is the
    per-date Spearman ρ array used to accumulate the aggregate IC stats.
    """
    test_r2 = held_out_r2(y_te, preds)
    test_ic = rank_ic_score(y_te, preds, grp_te)
    _, ic_values = rank_ic_series(y_te, preds, grp_te)
    return test_r2, test_ic, ic_values


def _build_fold_panel_df(
    fold_dates: np.ndarray,
    fold_ids: np.ndarray,
    preds: np.ndarray,
    fold_idx: int,
) -> pl.DataFrame:
    """Build the typed ``pl.DataFrame`` for one fold's OOS predictions.

    Schema: ``(date: Date, id: Int64, prediction: Float64, fold: Int32)``.
    This row block is later concatenated with other folds to form
    ``WFResult.predictions_panel``.
    """
    return pl.DataFrame(
        {
            "date": list(fold_dates),
            "id": fold_ids.tolist(),
            "prediction": preds.tolist(),
            "fold": fold_idx,
        }
    ).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("id").cast(pl.Int64),
        pl.col("prediction").cast(pl.Float64),
        pl.col("fold").cast(pl.Int32),
    )


def _assemble_wf_result(
    fold_results: list[FoldResult],
    all_preds_list: list[np.ndarray],
    all_true_list: list[np.ndarray],
    all_groups_list: list[np.ndarray],
    all_dates_list: list[np.ndarray],
    all_ids_list: list[np.ndarray],
    panel_rows: list[pl.DataFrame],
) -> WFResult:
    """Concatenate per-fold accumulators into the final ``WFResult``.

    Handles the empty-fold edge case for all array fields and for
    ``predictions_panel`` (returns ``None`` when no folds completed).
    """
    all_preds = np.concatenate(all_preds_list) if all_preds_list else np.array([])
    all_true = np.concatenate(all_true_list) if all_true_list else np.array([])
    all_groups = (
        np.concatenate(all_groups_list) if all_groups_list else np.array([], dtype=np.int64)
    )
    all_dates = np.concatenate(all_dates_list) if all_dates_list else np.array([], dtype=object)
    all_ids = np.concatenate(all_ids_list) if all_ids_list else np.array([], dtype=np.int64)

    predictions_panel: pl.DataFrame | None = (
        pl.concat(panel_rows).sort(["date", "id"]) if panel_rows else None
    )

    r2_arr = np.array([f.test_r2 for f in fold_results])
    all_ic_values = (
        np.concatenate([f.ic_values for f in fold_results]) if fold_results else np.array([])
    )
    stats = ic_stats(all_ic_values)

    return WFResult(
        fold_results=fold_results,
        mean_r2=float(r2_arr.mean()) if len(r2_arr) > 0 else 0.0,
        std_r2=float(r2_arr.std()) if len(r2_arr) > 0 else 0.0,
        mean_ic=stats["mean_ic"],
        ic_ir=stats["ic_ir"],
        all_preds=all_preds,
        all_true=all_true,
        all_groups=all_groups,
        all_dates=all_dates,
        all_ids=all_ids,
        predictions_panel=predictions_panel,
    )


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

_DEFAULT_WF_CONFIG = WalkForwardConfig()


def walk_forward_cv(
    panel: PanelArrays,
    splitter,
    model_factory,
    config: WalkForwardConfig = _DEFAULT_WF_CONFIG,
) -> WFResult:
    """Run walk-forward CV on a ``PanelArrays`` dataset.

    Parameters
    ----------
    panel:
        Aligned arrays from ``panel.build_panel``.
    splitter:
        Any splitter from ``models.splitters`` (or sklearn-compatible) that
        accepts ``groups`` as a keyword argument to ``split``.
    model_factory:
        Callable ``(alpha: float) -> FinancialModel``.  Called once per fold
        (and once per alpha candidate in inner CV).  Must accept
        ``sample_weight`` in ``fit`` if ``config.use_sample_weights`` is True.
    config:
        ``WalkForwardConfig`` controlling scaling, alpha search, and weighting.

    Returns
    -------
    WFResult
    """
    X, y = panel.X, panel.y
    groups = panel.groups
    weights = panel.weights
    dates = panel.dates
    ids = panel.ids

    fold_results: list[FoldResult] = []
    all_preds_list: list[np.ndarray] = []
    all_true_list: list[np.ndarray] = []
    all_groups_list: list[np.ndarray] = []
    all_dates_list: list[np.ndarray] = []
    all_ids_list: list[np.ndarray] = []
    panel_rows: list[pl.DataFrame] = []
    fold_idx = 0

    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        w_tr = weights[train_idx] if config.use_sample_weights else None

        X_tr, X_te = _scale_fold(X_tr, X_te, config.scale_features)

        chosen_alpha = _best_alpha(
            X_tr, y_tr, config.alpha_grid, model_factory, w_tr, config.inner_val_frac
        )

        model, fit_result = _fit_fold(model_factory, X_tr, y_tr, w_tr, chosen_alpha)
        preds = model.predict(X_te)

        grp_te = groups[test_idx]
        test_r2, test_ic, ic_values = _score_fold(y_te, preds, grp_te)

        fold_results.append(
            FoldResult(
                fit_result=fit_result,
                test_r2=test_r2,
                test_ic=test_ic,
                ic_values=ic_values,
                chosen_alpha=chosen_alpha,
                n_train=len(train_idx),
                n_test=len(test_idx),
            )
        )
        all_preds_list.append(preds)
        all_true_list.append(y_te)
        all_groups_list.append(grp_te)
        fold_dates = dates[test_idx]
        fold_ids = ids[test_idx]
        all_dates_list.append(fold_dates)
        all_ids_list.append(fold_ids)
        panel_rows.append(_build_fold_panel_df(fold_dates, fold_ids, preds, fold_idx))
        fold_idx += 1

    return _assemble_wf_result(
        fold_results,
        all_preds_list,
        all_true_list,
        all_groups_list,
        all_dates_list,
        all_ids_list,
        panel_rows,
    )
