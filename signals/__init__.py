from ._protocol import SignalEvaluator
from .ic import ICEvaluator, ICMethod, ICResult, ic_series, ic_series_v2, rolling_ic
from .newey_west import default_lags, newey_west_tstat
from .coverage import apply_min_coverage, pairwise_mask
from .horizon import HorizonCurve, HorizonPoint, ic_horizon_curve
from .quantile import QuantileResult, quantile_spread
from .neutralize import (
    NeutralizationResult,
    evaluate_neutralization,
    neutralize_factors,
    neutralize_sector,
)
from .turnover import TurnoverResult, rank_stability, signal_autocorr, turnover_score
from .combine import (
    IncrementalICResult,
    gram_schmidt_orthogonalize,
    ic_weighted_blend,
    incremental_ic,
    zscore_blend,
)
from .multiple_testing import (
    MultipleTestingResult,
    bh_correct,
    bonferroni_correct,
    multiple_testing_correction,
    rolling_ic_ir,
    tstat_to_pvalue,
)

__all__ = [
    # existing
    "SignalEvaluator",
    "ICEvaluator",
    "ICResult",
    "ic_series",
    "rolling_ic",
    "default_lags",
    "newey_west_tstat",
    # new ic
    "ICMethod",
    "ic_series_v2",
    # coverage
    "apply_min_coverage",
    "pairwise_mask",
    # horizon
    "HorizonCurve",
    "HorizonPoint",
    "ic_horizon_curve",
    # quantile
    "QuantileResult",
    "quantile_spread",
    # neutralize
    "NeutralizationResult",
    "evaluate_neutralization",
    "neutralize_factors",
    "neutralize_sector",
    # turnover
    "TurnoverResult",
    "rank_stability",
    "signal_autocorr",
    "turnover_score",
    # combine
    "IncrementalICResult",
    "gram_schmidt_orthogonalize",
    "ic_weighted_blend",
    "incremental_ic",
    "zscore_blend",
    # multiple testing
    "MultipleTestingResult",
    "bh_correct",
    "bonferroni_correct",
    "multiple_testing_correction",
    "rolling_ic_ir",
    "tstat_to_pvalue",
]
