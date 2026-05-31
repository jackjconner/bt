from ._protocol import FinancialModel
from .cross_val import CVConfig, CVResult, cv_loop
from .models_zoo import ElasticNetConfig, ElasticNetModel, LassoConfig, LassoModel
from .panel import PanelArrays, build_panel, date_ordinals
from .persistence import (
    ModelArtifact,
    artifact_from_fold,
    load_artifact,
    predict_from_artifact,
    save_artifact,
)
from .ridge import ModelConfig, ModelResult, RidgeModel
from .scoring import held_out_r2, ic_stats, rank_ic_score, rank_ic_series
from .splitters import (
    PurgedEmbargoCVSplitter,
    RollingWindowSplitter,
    WalkForwardSplitter,
    splits_from_calendar,
)
from .walk_forward import (
    FoldResult,
    FoldScaler,
    WalkForwardConfig,
    WFResult,
    walk_forward_cv,
)

__all__ = [
    "CVConfig",
    "CVResult",
    "ElasticNetConfig",
    "ElasticNetModel",
    # existing
    "FinancialModel",
    "FoldResult",
    # walk-forward CV
    "FoldScaler",
    # model zoo
    "LassoConfig",
    "LassoModel",
    # persistence
    "ModelArtifact",
    "ModelConfig",
    "ModelResult",
    # panel
    "PanelArrays",
    # splitters
    "PurgedEmbargoCVSplitter",
    "RidgeModel",
    "RollingWindowSplitter",
    "WFResult",
    "WalkForwardConfig",
    "WalkForwardSplitter",
    "artifact_from_fold",
    "build_panel",
    "cv_loop",
    "date_ordinals",
    "held_out_r2",
    "ic_stats",
    "load_artifact",
    "predict_from_artifact",
    # scoring
    "rank_ic_score",
    "rank_ic_series",
    "save_artifact",
    "splits_from_calendar",
    "walk_forward_cv",
]
