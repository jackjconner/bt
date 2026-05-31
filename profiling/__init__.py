from .environment import RunEnvironment, capture_environment
from .memory import MemSnapshot, frames_size_mb, obj_size_mb, rss_mb, snapshot
from .output import write_json, write_measurements_parquet
from .regression import RegressionReport, RegressionViolation, check_regressions
from .report import ScalingResult, StageProfile, collect_stage, print_report
from .scaling import ScalingFit, fit_scaling, fits_to_dataframe
from .storage import read_measurements, read_runs, write_run
from .timer import StageTimer, TimingResult
from .trials import TrialMeasurement, TrialResult, TrialStats, run_trials

__all__ = [
    # memory
    "MemSnapshot",
    "frames_size_mb",
    "obj_size_mb",
    "rss_mb",
    "snapshot",
    # report (existing stable API)
    "ScalingResult",
    "StageProfile",
    "collect_stage",
    "print_report",
    # timer
    "StageTimer",
    "TimingResult",
    # trials (feature 1 + 2)
    "TrialMeasurement",
    "TrialResult",
    "TrialStats",
    "run_trials",
    # environment (feature 3)
    "RunEnvironment",
    "capture_environment",
    # storage (feature 4)
    "write_run",
    "read_runs",
    "read_measurements",
    # regression (feature 5)
    "RegressionReport",
    "RegressionViolation",
    "check_regressions",
    # scaling (feature 6)
    "ScalingFit",
    "fit_scaling",
    "fits_to_dataframe",
    # output (feature 7)
    "write_json",
    "write_measurements_parquet",
]
