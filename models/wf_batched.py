"""Batched, numpy-core walk-forward CV engine for closed-form ridge.

This is a structurally different engine from the per-fold Python/sklearn loop in
``walk_forward.py``.  It exploits two facts about the production walk-forward
configuration to collapse the per-window Python overhead into vectorized numpy:

1.  **Closed-form ridge.**  A ridge fit on standardized, weighted-centered
    features is the exact solution of ``(Xᵀ W X + αI) β = Xᵀ W (y − ȳ)``.  This
    matches sklearn's ``Ridge`` to machine epsilon (verified: coef diff ~1e-16),
    so no estimator object, no Python-level ``fit``/``predict`` dispatch, and no
    per-alpha refit from scratch is needed.

2.  **Date-block prefix moments.**  Every train window the splitters produce is a
    *contiguous block of date groups* in the date-sorted panel, and the
    expanding/rolling windows are prefixes (or sliding spans) of those blocks.
    We therefore compute each date block's raw weighted moments
    (``Σw``, ``Σwx``, ``Σwxxᵀ``, ``Σwy``, ``Σwxy``) and unweighted moments
    (``Σx``, ``Σx²``, ``n``) **once** in a single vectorized pass, then assemble
    any window's standardized Gram and cross-product as a *difference of
    cumulative block sums* — O(p²) per fold, with zero rescans of the data and no
    Python loop over samples.

The inner alpha search (a holdout on the last ``inner_val_frac`` of each train
window) reuses the same block-moment machinery: the inner-train sub-window and
its validation slice are also assembled from block sums, and all candidate alphas
are solved against one shared Gram via repeated diagonal-shifted Cholesky solves.

The engine reproduces ``walk_forward_cv``'s numbers (the same scaling, the same
weighted ridge, the same alpha-holdout selector, the same per-date IC) so it can
be dispatched to transparently for ridge-compatible model factories.  When the
factory is not a closed-form ridge, ``walk_forward_cv`` keeps the generic loop.

Public API
----------
``is_ridge_factory``        — probe whether a ``model_factory`` yields a
                              closed-form-ridge-compatible model.
``walk_forward_cv_batched`` — the batched engine; same signature/return as
                              ``walk_forward_cv``.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .panel import PanelArrays
from .ridge import ModelConfig, ModelResult, RidgeModel


# --------------------------------------------------------------------------- #
# Ridge-compatibility probe
# --------------------------------------------------------------------------- #
def is_ridge_factory(model_factory, n_features: int) -> bool:
    """Return True when ``model_factory(alpha)`` is a closed-form ridge.

    The batched engine only reproduces ``RidgeModel`` (the closed-form weighted
    ridge).  Lasso / ElasticNet / gradient boosting have no closed form, so they
    must run the generic per-fold loop.  We probe by constructing a model for a
    sample alpha and checking it is a ``RidgeModel`` whose configured
    ``n_features`` matches the panel.
    """
    try:
        probe = model_factory(1.0)
    except Exception:
        return False
    if not isinstance(probe, RidgeModel):
        return False
    cfg = getattr(probe, "config", None)
    if not isinstance(cfg, ModelConfig):
        return False
    return cfg.n_features == n_features


# --------------------------------------------------------------------------- #
# Block moments
# --------------------------------------------------------------------------- #
class _BlockMoments:
    """Per-date-block raw moments + their cumulative prefix sums.

    All windows the splitters yield are unions of contiguous date blocks, so a
    window's moments are differences of cumulative block sums.  Storing the
    cumulative sums once lets every fold (and every inner-CV sub-window) assemble
    its standardized Gram in O(p²) with no data rescan.

    Cumulative arrays are length ``n_blocks + 1`` (a leading zero row) so the
    moments of block span ``[a, b)`` are ``cum[b] − cum[a]``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray) -> None:
        # the production panel is already date-sorted; the batched engine is only
        # dispatched when groups are non-decreasing (the caller guarantees it), so
        # the sorted view is the identity and fold indices line up directly.
        Xs, ys, ws, gs = X, y, w, groups
        self._X, self._y, self._w = Xs, ys, ws

        # block start indices: where the sorted group id changes
        change = np.empty(len(gs), dtype=bool)
        change[0] = True
        change[1:] = gs[1:] != gs[:-1]
        starts = np.flatnonzero(change)
        self._block_start = starts
        self._block_group = gs[starts]
        n_blocks = len(starts)
        self.n_blocks = n_blocks
        p = X.shape[1]
        self.p = p

        # reduceat sums each block: row i covers [starts[i], starts[i+1])
        wx = Xs * ws[:, None]
        sum_w = np.add.reduceat(ws, starts)
        sum_wx = np.add.reduceat(wx, starts, axis=0)
        sum_wy = np.add.reduceat(ys * ws, starts)
        sum_wyy = np.add.reduceat(ys * ys * ws, starts)
        sum_wxy = np.add.reduceat(Xs * (ys * ws)[:, None], starts, axis=0)
        sum_x = np.add.reduceat(Xs, starts, axis=0)
        sum_xx = np.add.reduceat(Xs * Xs, starts, axis=0)
        count = np.add.reduceat(np.ones(len(gs)), starts)

        # per-block weighted second moment Σ w x xᵀ  (n_blocks, p, p)
        sum_wxx = np.empty((n_blocks, p, p), dtype=np.float64)
        for b in range(n_blocks):
            lo = starts[b]
            hi = starts[b + 1] if b + 1 < n_blocks else len(gs)
            Xb = Xs[lo:hi]
            wb = ws[lo:hi]
            sum_wxx[b] = Xb.T @ (Xb * wb[:, None])

        def _cum(a: np.ndarray) -> np.ndarray:
            z = np.zeros((1, *a.shape[1:]), dtype=np.float64)
            return np.concatenate([z, np.cumsum(a, axis=0)], axis=0)

        self.c_w = _cum(sum_w)
        self.c_wx = _cum(sum_wx)
        self.c_wy = _cum(sum_wy)
        self.c_wyy = _cum(sum_wyy)
        self.c_wxy = _cum(sum_wxy)
        self.c_x = _cum(sum_x)
        self.c_xx = _cum(sum_xx)
        self.c_n = _cum(count)
        self.c_wxx = _cum(sum_wxx)

    def block_index(self, group_ids: np.ndarray) -> np.ndarray:
        """Map sorted-unique date ordinals to block indices (searchsorted)."""
        return np.searchsorted(self._block_group, group_ids)

    def sample_lo(self, block: int) -> int:
        """First sample row of ``block``."""
        return int(self._block_start[block])

    def sample_hi(self, block: int) -> int:
        """One-past-last sample row of ``block``."""
        nxt = block + 1
        return int(self._block_start[nxt]) if nxt < self.n_blocks else int(self.c_n[self.n_blocks])

    def rows(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw ``(X, y, w)`` rows for sample span ``[lo, hi)`` (for partial blocks)."""
        return self._X[lo:hi], self._y[lo:hi], self._w[lo:hi]


def _window_moments(bm: _BlockMoments, lo: int, hi: int) -> tuple:
    """Raw moments for the contiguous block span ``[lo, hi)`` (cumulative diff)."""
    return (
        bm.c_w[hi] - bm.c_w[lo],
        bm.c_wx[hi] - bm.c_wx[lo],
        bm.c_wy[hi] - bm.c_wy[lo],
        bm.c_wxy[hi] - bm.c_wxy[lo],
        bm.c_wxx[hi] - bm.c_wxx[lo],
        bm.c_x[hi] - bm.c_x[lo],
        bm.c_xx[hi] - bm.c_xx[lo],
        bm.c_n[hi] - bm.c_n[lo],
        bm.c_wyy[hi] - bm.c_wyy[lo],
    )


def _scaling_stats(moments: tuple, p: int, scale: bool) -> tuple[np.ndarray, np.ndarray]:
    """StandardScaler mean/std (population, ddof=0) from a window's raw moments."""
    ux, uxx, n = moments[5], moments[6], moments[7]
    if not scale:
        return np.zeros(p), np.ones(p)
    mu = ux / n
    var = uxx / n - mu * mu
    sd = np.sqrt(np.maximum(var, 0.0))
    sd = np.where(sd > 0.0, sd, 1.0)  # mirror StandardScaler: 0-var → scale 1
    return mu, sd


def _solve_moments(
    moments: tuple,
    p: int,
    alphas: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Solve weighted closed-form ridge for every alpha on the window given by
    ``moments``, standardizing with the externally supplied ``mu``/``sd`` (the
    *full train fold* statistics, matching ``walk_forward_cv``: the scaler is fit
    on the whole train fold, then the inner-CV split operates on scaled data).

    Returns ``(betas, intercepts, G, b, ym)`` where ``betas`` has shape
    ``(len(alphas), p)`` in the standardized feature space and ``G``/``b``/``ym``
    are the standardized weighted Gram, cross-product, and weighted target mean
    (reused by ``_weighted_train_r2``).
    """
    W, Sx, Sy, Sxy, Sxx = moments[0], moments[1], moments[2], moments[3], moments[4]

    # weighted mean of standardized X, weighted mean of y
    xm = (Sx / W - mu) / sd
    ym = Sy / W
    # centering offset c so that standardized-centered x = (X − c)/sd
    c = mu + sd * xm
    Dinv = 1.0 / sd

    M = Sxx - np.outer(c, Sx) - np.outer(Sx, c) + W * np.outer(c, c)
    G = (Dinv[:, None] * M) * Dinv[None, :]
    b = Dinv * (Sxy - c * Sy - ym * Sx + W * ym * c)

    betas = np.empty((len(alphas), p), dtype=np.float64)
    eye = np.eye(p)
    for i, a in enumerate(alphas):
        cf = cho_factor(G + a * eye, lower=True, check_finite=False)
        betas[i] = cho_solve(cf, b, check_finite=False)
    intercepts = ym - betas @ xm
    return betas, intercepts, G, b, float(ym)


def _weighted_train_r2(
    moments: tuple, beta: np.ndarray, G: np.ndarray, b: np.ndarray, ym: float
) -> float:
    """Weighted train R² (sklearn ``Ridge.score`` semantics) from moments + fit.

    In the scaled, weighted-centered space the weighted residual sum of squares
    is the quadratic form ``Syy_c − 2 βᵀb + βᵀGβ`` and the weighted total sum of
    squares (about the *weighted* mean) is ``Syy_c``.
    """
    W = moments[0]
    Syy = moments[8]
    syy_c = float(Syy - W * ym * ym)
    if syy_c <= 0.0:
        return 0.0
    ss_res = syy_c - 2.0 * float(beta @ b) + float(beta @ (G @ beta))
    return 1.0 - ss_res / syy_c


def _partial_block_moments(Xb: np.ndarray, yb: np.ndarray, wb: np.ndarray) -> tuple:
    """Raw moments for an explicit set of rows (used for the boundary block of an
    inner-CV split that falls mid-date).  Same tuple layout as ``_window_moments``."""
    return (
        wb.sum(),
        (Xb * wb[:, None]).sum(axis=0),
        (yb * wb).sum(),
        (Xb * (yb * wb)[:, None]).sum(axis=0),
        Xb.T @ (Xb * wb[:, None]),
        Xb.sum(axis=0),
        (Xb * Xb).sum(axis=0),
        float(len(Xb)),
        (yb * yb * wb).sum(),
    )


def _add_moments(a: tuple, b: tuple) -> tuple:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _blockset_moments(bm: _BlockMoments, blocks: np.ndarray) -> tuple:
    """Sum raw moments over an arbitrary sorted set of block indices.

    For a contiguous run this is a single cumulative-sum difference; a
    non-contiguous set (e.g. purged folds) is decomposed into its contiguous
    runs and summed.  Keeps the assembly O(runs · p²) with no data rescan.
    """
    if len(blocks) == 0:
        return _window_moments(bm, 0, 0)
    runs_end = np.flatnonzero(np.diff(blocks) != 1)
    starts = np.concatenate([[0], runs_end + 1])
    stops = np.concatenate([runs_end + 1, [len(blocks)]])
    total = _window_moments(bm, int(blocks[starts[0]]), int(blocks[stops[0] - 1]) + 1)
    for s, e in zip(starts[1:], stops[1:], strict=True):
        lo = int(blocks[s])
        hi = int(blocks[e - 1]) + 1
        total = _add_moments(total, _window_moments(bm, lo, hi))
    return total


def _prefix_moments(bm: _BlockMoments, blocks: np.ndarray, n_rows: int) -> tuple:
    """Raw moments for the first ``n_rows`` samples of a contiguous train block
    set — replicating ``_best_alpha``'s sample-count holdout split exactly.

    ``blocks`` is the contiguous train block range.  ``n_rows`` may fall mid
    date-block; whole blocks before the boundary come from the cumulative sums
    and the partial boundary block is summed directly from its rows.
    """
    base = bm.sample_lo(int(blocks[0]))
    split_row = base + n_rows
    # whole blocks fully inside the prefix
    whole: list[int] = []
    partial_block = None
    for blk in blocks:
        blk = int(blk)
        lo = bm.sample_lo(blk)
        hi = bm.sample_hi(blk)
        if hi <= split_row:
            whole.append(blk)
        elif lo < split_row < hi:
            partial_block = (blk, lo, split_row)
            break
        else:
            break
    total = (
        _blockset_moments(bm, np.array(whole, dtype=np.int64))
        if whole
        else _window_moments(bm, int(blocks[0]), int(blocks[0]))
    )
    if partial_block is not None:
        _, lo, hi = partial_block
        Xb, yb, wb = bm.rows(lo, hi)
        total = _add_moments(total, _partial_block_moments(Xb, yb, wb))
    return total


# --------------------------------------------------------------------------- #
# Per-fold solve
# --------------------------------------------------------------------------- #
def _fold_blocks(bm: _BlockMoments, idx: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Sorted unique block indices touched by sample indices ``idx``."""
    fold_groups = np.unique(groups[idx])
    return bm.block_index(fold_groups)


def _choose_alpha_batched(
    bm: _BlockMoments,
    train_blocks: np.ndarray,
    n_train: int,
    alpha_grid: list[float],
    mu: np.ndarray,
    sd: np.ndarray,
    inner_val_frac: float,
) -> float:
    """Reproduce ``walk_forward._best_alpha`` (holdout inner CV) on scaled data.

    The split is by *sample count* on the full (scaled) train fold — identical to
    the incumbent.  The inner-train sub-window is a date prefix (assembled from
    block moments); the validation slice is the remaining rows, predicted with the
    full-train scaling and scored by held-out R² (the same formula as
    ``scoring.held_out_r2``), and all candidate alphas are scored together.  Ties
    keep the earliest alpha — matching ``_best_alpha``'s strict-``>`` loop.
    """
    if len(alpha_grid) == 1:
        return alpha_grid[0]
    split = max(1, int(n_train * (1.0 - inner_val_frac)))
    if split >= n_train:
        return alpha_grid[0]

    inner_moments = _prefix_moments(bm, train_blocks, split)
    alphas = np.asarray(alpha_grid, dtype=np.float64)
    betas, intercepts, _, _, _ = _solve_moments(inner_moments, bm.p, alphas, mu, sd)

    base = bm.sample_lo(int(train_blocks[0]))
    X_val, y_val, _ = bm.rows(base + split, base + n_train)
    Xs_val = (X_val - mu) / sd
    # preds: (n_val, n_alpha) = Xs_val @ betasᵀ + intercepts
    preds = Xs_val @ betas.T + intercepts[None, :]
    ss_tot = float(((y_val - y_val.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return alpha_grid[0]
    ss_res = ((y_val[:, None] - preds) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / ss_tot
    return alpha_grid[int(np.argmax(r2))]


def walk_forward_cv_batched(panel: PanelArrays, splitter, model_factory, config):
    """Batched numpy-core walk-forward CV — drop-in for ``walk_forward_cv``.

    Same arguments, same ``WFResult``, reproduced via Gram-accumulation closed-form
    ridge instead of per-fold sklearn refits.  Only valid when ``model_factory`` is
    a closed-form ridge (see ``is_ridge_factory``); the caller guards on that.
    """
    # Lazy imports break the import cycle with ``walk_forward`` (which dispatches
    # here): the result/assembly types live there, the math core lives here.
    from .walk_forward import (
        FoldResult,
        _assemble_wf_result,
        _build_fold_panel_df,
        _score_fold,
    )

    X, y, groups = panel.X, panel.y, panel.groups
    weights = panel.weights
    dates, ids = panel.dates, panel.ids
    p = X.shape[1]
    w = weights if config.use_sample_weights else np.ones(len(y))
    bm = _BlockMoments(X, y, w, groups)
    alpha_grid = config.alpha_grid

    fold_results: list[FoldResult] = []
    all_preds_list: list[np.ndarray] = []
    all_true_list: list[np.ndarray] = []
    all_groups_list: list[np.ndarray] = []
    all_dates_list: list[np.ndarray] = []
    all_ids_list: list[np.ndarray] = []
    panel_rows = []
    fold_idx = 0

    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        train_blocks = _fold_blocks(bm, train_idx, groups)
        train_moments = _blockset_moments(bm, train_blocks)
        n_train = round(float(train_moments[7]))
        mu, sd = _scaling_stats(train_moments, p, config.scale_features)

        chosen_alpha = _choose_alpha_batched(
            bm, train_blocks, n_train, alpha_grid, mu, sd, config.inner_val_frac
        )

        betas, intercepts, G, b, ym = _solve_moments(
            train_moments, p, np.array([chosen_alpha]), mu, sd
        )
        beta = betas[0]
        intercept = float(intercepts[0])
        train_r2 = _weighted_train_r2(train_moments, beta, G, b, ym)
        fit_result = ModelResult(coef=beta, intercept=intercept, train_r2=train_r2)

        X_te = X[test_idx]
        Xs_te = (X_te - mu) / sd if config.scale_features else X_te
        preds = Xs_te @ beta + intercept
        y_te = y[test_idx]
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
