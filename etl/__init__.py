from ._protocol import Loader
from .adjust import AdjustmentResult, adjust_prices
from .batch import BatchLoader, ETLConfig
from .calendar import align_to_calendar, fill_sessions, sessions_between
from .loader import DatasetLoader
from .masked_pivot import to_masked_matrix
from .pit import as_of_slice, latest_as_of
from .quality import QualityReport, check
from .source import date_axis, generate_returns, to_matrix, write_parquet
from .stream import StreamLoader
from .universe import apply_security_master, resolve_universe

__all__ = [
    # original
    "Loader",
    "BatchLoader",
    "ETLConfig",
    "StreamLoader",
    "date_axis",
    "generate_returns",
    "to_matrix",
    "write_parquet",
    # new
    "AdjustmentResult",
    "adjust_prices",
    "align_to_calendar",
    "fill_sessions",
    "sessions_between",
    "DatasetLoader",
    "to_masked_matrix",
    "as_of_slice",
    "latest_as_of",
    "QualityReport",
    "check",
    "apply_security_master",
    "resolve_universe",
]
