from ._protocol import BacktestAnalyzer
from .attribution import (
    FactorAttributionResult,
    factor_attribution,
    sector_attribution,
)
from .benchmark import (
    active_returns,
    alpha,
    benchmark_returns_to_fractional,
    beta,
    down_capture,
    information_ratio,
    r_squared,
    relative_drawdown,
    tracking_error,
    up_capture,
)
from .metrics import (
    AnalysisResult,
    BacktestAnalyzerImpl,
    max_drawdown,
    returns_from_nav,
    sharpe,
)
from .periodic import (
    annual_returns,
    monthly_returns,
    monthly_returns_wide,
    quarterly_returns,
)
from .risk import (
    annualized_return_calendar,
    best_day,
    cagr,
    calmar,
    cvar_historical,
    excess_kurtosis,
    hit_rate,
    skewness,
    sortino,
    var_historical,
    worst_day,
)
from .rolling import (
    rolling_beta,
    rolling_max_drawdown,
    rolling_sharpe,
    rolling_vol,
)
from .turnover import (
    effective_n,
    gross_exposure,
    net_exposure,
    net_nav,
    one_way_turnover,
    reconstruct_weights,
    top_n_weight,
    two_way_turnover,
)

__all__ = [
    # metrics (existing public API — never remove)
    "AnalysisResult",
    # protocol
    "BacktestAnalyzer",
    "BacktestAnalyzerImpl",
    # attribution
    "FactorAttributionResult",
    # benchmark
    "active_returns",
    "alpha",
    # periodic
    "annual_returns",
    # risk
    "annualized_return_calendar",
    "benchmark_returns_to_fractional",
    "best_day",
    "beta",
    "cagr",
    "calmar",
    "cvar_historical",
    "down_capture",
    # turnover
    "effective_n",
    "excess_kurtosis",
    "factor_attribution",
    "gross_exposure",
    "hit_rate",
    "information_ratio",
    "max_drawdown",
    "monthly_returns",
    "monthly_returns_wide",
    "net_exposure",
    "net_nav",
    "one_way_turnover",
    "quarterly_returns",
    "r_squared",
    "reconstruct_weights",
    "relative_drawdown",
    "returns_from_nav",
    # rolling
    "rolling_beta",
    "rolling_max_drawdown",
    "rolling_sharpe",
    "rolling_vol",
    "sector_attribution",
    "sharpe",
    "skewness",
    "sortino",
    "top_n_weight",
    "tracking_error",
    "two_way_turnover",
    "up_capture",
    "var_historical",
    "worst_day",
]
