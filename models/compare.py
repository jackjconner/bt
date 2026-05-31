"""Model-comparison helper: rank candidate models by out-of-sample IC.

``compare_models`` runs each candidate through ``walk_forward_cv`` and collects
per-model OOS IC, R², and IR.  The result is a ``ModelComparison`` dataclass
that exposes the per-model summary and a stable ranking by mean OOS IC.

Typical usage
-------------
::

    from models import (
        ModelConfig, RidgeModel,
        GradientBoostConfig, GradientBoostModel,
        compare_models,
    )
    from models.splitters import WalkForwardSplitter

    ridge_factory = lambda alpha: RidgeModel(ModelConfig(n_features=5, alpha=alpha))
    boost_factory = lambda alpha: GradientBoostModel(GradientBoostConfig(n_features=5))

    comparison = compare_models(
        models={"ridge": ridge_factory, "boost": boost_factory},
        panel=panel,
        splitter=WalkForwardSplitter(n_splits=5, min_train_periods=20),
    )
    print(comparison.ranking)          # [("boost", 0.12), ("ridge", 0.09)]

Design notes
------------
- ``model_factory`` callables follow the same ``(alpha: float) -> FinancialModel``
  signature required by ``walk_forward_cv``.  Non-regularized models (e.g.
  ``GradientBoostModel``) can simply ignore the alpha argument.
- Ranking is by mean OOS IC (``WFResult.mean_ic``), descending.  The ordering
  is stable: ties preserve insertion order.
- ``ModelComparison`` is a frozen dataclass; ``results`` is a plain dict so
  callers can access the full ``WFResult`` for any model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .panel import PanelArrays
from .walk_forward import WalkForwardConfig, WFResult, walk_forward_cv

_DEFAULT_WF_CONFIG = WalkForwardConfig()


@dataclass(frozen=True)
class ModelComparison:
    """Aggregate output from ``compare_models``.

    Attributes
    ----------
    results:
        Mapping from model name to its ``WFResult``.
    ranking:
        Model names sorted by OOS mean IC, descending.  Each entry is a
        ``(name, mean_ic)`` tuple so the IC value is immediately visible.
    best:
        Name of the model with the highest OOS mean IC.
    """

    results: dict[str, WFResult]
    ranking: list[tuple[str, float]]
    best: str


def compare_models(
    models: dict[str, Callable[[float], object]],
    panel: PanelArrays,
    splitter: object,
    config: WalkForwardConfig = _DEFAULT_WF_CONFIG,
) -> ModelComparison:
    """Run each candidate model through walk-forward CV and rank by OOS IC.

    Parameters
    ----------
    models:
        Mapping from a short name (used as a display key) to a
        ``model_factory`` callable with signature ``(alpha: float) -> FinancialModel``.
        The factory is called once per fold (and once per alpha in the inner
        grid search); non-regularized models may ignore the alpha argument.
    panel:
        Aligned arrays from ``panel.build_panel``.
    splitter:
        Any splitter from ``models.splitters`` that accepts a ``groups``
        keyword argument in ``split``.
    config:
        ``WalkForwardConfig`` forwarded to ``walk_forward_cv`` for every model.
        The same configuration is used for all candidates to ensure a fair
        comparison.

    Returns
    -------
    ModelComparison
        Frozen dataclass with per-model ``WFResult`` and a ranking by OOS IC.
    """
    results: dict[str, WFResult] = {}
    for name, factory in models.items():
        results[name] = walk_forward_cv(panel, splitter, factory, config)

    ranking = sorted(
        ((name, r.mean_ic) for name, r in results.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    best = ranking[0][0] if ranking else next(iter(results))

    return ModelComparison(results=results, ranking=ranking, best=best)
