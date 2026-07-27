"""M3 common-backplate array response and screening APIs."""

from .experiment import ArrayExperimentSettings, CommonBackplateExperiment
from .models import (
    M3_MODEL_LEVEL,
    M3_MODULE_VERSION,
    ActivitySets,
    AngleLayout,
    ArrayConfiguration,
    ArrayExperimentResult,
    ArrayPathPoint,
    ArrayPathSummary,
    ArrayPoseResponse,
    ArrayResidualAudit,
    ArrayState,
    LoadSharingMetrics,
)
from .solver import CommonBackplateArray
from .design import (
    build_candidate_pool,
    level_counts,
    screening_gate_status,
    select_balanced_candidates,
)

__all__ = [
    "M3_MODEL_LEVEL",
    "M3_MODULE_VERSION",
    "ActivitySets",
    "AngleLayout",
    "ArrayConfiguration",
    "ArrayExperimentResult",
    "ArrayExperimentSettings",
    "ArrayPathPoint",
    "ArrayPathSummary",
    "ArrayPoseResponse",
    "ArrayResidualAudit",
    "ArrayState",
    "CommonBackplateArray",
    "CommonBackplateExperiment",
    "LoadSharingMetrics",
    "build_candidate_pool",
    "level_counts",
    "screening_gate_status",
    "select_balanced_candidates",
]
