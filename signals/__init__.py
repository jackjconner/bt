from ._protocol import SignalEvaluator
from .combine import (
    IncrementalICResult,
    gram_schmidt_orthogonalize,
    ic_weighted_blend,
    incremental_ic,
    zscore_blend,
)
from .coverage import apply_min_coverage, pairwise_mask
from .horizon import HorizonCurve, HorizonPoint, ic_horizon_curve
from .ic import ICEvaluator, ICMethod, ICResult, ic_series, ic_series_v2, rolling_ic
from .multiple_testing import (
    MultipleTestingResult,
    bh_correct,
    bonferroni_correct,
    multiple_testing_correction,
    rolling_ic_ir,
    tstat_to_pvalue,
)
from .neutralize import (
    NeutralizationResult,
    evaluate_neutralization,
    neutralize_factors,
    neutralize_sector,
)
from .newey_west import default_lags, newey_west_tstat
from .quantile import QuantileResult, quantile_spread
from .turnover import TurnoverResult, rank_stability, signal_autocorr, turnover_score

__all__ = [
    # horizon
    "HorizonCurve",
    "HorizonPoint",
    "ICEvaluator",
    # new ic
    "ICMethod",
    "ICResult",
    # combine
    "IncrementalICResult",
    # multiple testing
    "MultipleTestingResult",
    # neutralize
    "NeutralizationResult",
    # quantile
    "QuantileResult",
    # existing
    "SignalEvaluator",
    # turnover
    "TurnoverResult",
    # coverage
    "apply_min_coverage",
    "bh_correct",
    "bonferroni_correct",
    "default_lags",
    "evaluate_neutralization",
    "gram_schmidt_orthogonalize",
    "ic_horizon_curve",
    "ic_series",
    "ic_series_v2",
    "ic_weighted_blend",
    "incremental_ic",
    "multiple_testing_correction",
    "neutralize_factors",
    "neutralize_sector",
    "newey_west_tstat",
    "pairwise_mask",
    "quantile_spread",
    "rank_stability",
    "rolling_ic",
    "rolling_ic_ir",
    "signal_autocorr",
    "tstat_to_pvalue",
    "turnover_score",
    "zscore_blend",
]
