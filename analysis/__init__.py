from ._protocol import BacktestAnalyzer
from .metrics import (
    AnalysisResult,
    BacktestAnalyzerImpl,
    max_drawdown,
    returns_from_nav,
    sharpe,
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
from .rolling import (
    rolling_beta,
    rolling_max_drawdown,
    rolling_sharpe,
    rolling_vol,
)
from .periodic import (
    annual_returns,
    monthly_returns,
    monthly_returns_wide,
    quarterly_returns,
)
from .attribution import (
    FactorAttributionResult,
    factor_attribution,
    sector_attribution,
)

__all__ = [
    # protocol
    "BacktestAnalyzer",
    # metrics (existing public API — never remove)
    "AnalysisResult",
    "BacktestAnalyzerImpl",
    "max_drawdown",
    "returns_from_nav",
    "sharpe",
    # risk
    "annualized_return_calendar",
    "best_day",
    "cagr",
    "calmar",
    "cvar_historical",
    "excess_kurtosis",
    "hit_rate",
    "skewness",
    "sortino",
    "var_historical",
    "worst_day",
    # benchmark
    "active_returns",
    "alpha",
    "benchmark_returns_to_fractional",
    "beta",
    "down_capture",
    "information_ratio",
    "r_squared",
    "relative_drawdown",
    "tracking_error",
    "up_capture",
    # turnover
    "effective_n",
    "gross_exposure",
    "net_exposure",
    "net_nav",
    "one_way_turnover",
    "reconstruct_weights",
    "top_n_weight",
    "two_way_turnover",
    # rolling
    "rolling_beta",
    "rolling_max_drawdown",
    "rolling_sharpe",
    "rolling_vol",
    # periodic
    "annual_returns",
    "monthly_returns",
    "monthly_returns_wide",
    "quarterly_returns",
    # attribution
    "FactorAttributionResult",
    "factor_attribution",
    "sector_attribution",
]
