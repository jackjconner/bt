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
    # existing
    "FinancialModel",
    "CVConfig",
    "CVResult",
    "cv_loop",
    "ModelConfig",
    "ModelResult",
    "RidgeModel",
    # model zoo
    "LassoConfig",
    "LassoModel",
    "ElasticNetConfig",
    "ElasticNetModel",
    # panel
    "PanelArrays",
    "build_panel",
    "date_ordinals",
    # splitters
    "PurgedEmbargoCVSplitter",
    "WalkForwardSplitter",
    "RollingWindowSplitter",
    "splits_from_calendar",
    # scoring
    "rank_ic_score",
    "rank_ic_series",
    "ic_stats",
    "held_out_r2",
    # walk-forward CV
    "FoldScaler",
    "FoldResult",
    "WFResult",
    "WalkForwardConfig",
    "walk_forward_cv",
    # persistence
    "ModelArtifact",
    "artifact_from_fold",
    "save_artifact",
    "load_artifact",
    "predict_from_artifact",
]
