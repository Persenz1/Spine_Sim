"""M2 continuous-preload dynamics and legacy prescribed-pose APIs."""

from .dynamics import (
    DynamicContactSettings,
    DynamicExperimentSettings,
    DynamicIntegratorSettings,
    DynamicPathPoint,
    DynamicPathSummary,
    DynamicSingleSpineExperiment,
    DynamicSingleSpineResult,
    DynamicSingleSpineUnit,
)
from .experiment import (
    ExperimentSettings as LegacyFixedZExperimentSettings,
    SingleSpineExperiment as LegacyFixedZExperiment,
)
from .models import (
    AxialMode,
    ConstitutiveResponse,
    ContactState,
    EventLabel,
    M2_MODEL_LEVEL,
    M2_MODULE_VERSION,
    M2_LEGACY_MODEL_LEVEL,
    PathPoint,
    PathSummary,
    PathTerminalState,
    SingleSpineExperimentResult,
    SingleSpineState,
    SolverSettings,
    SpineParameters,
    SpringState,
)
from .solver import PrescribedPoseConstitutiveCore

# The public production path now means continuous-preload dynamics.  Explicit
# legacy names keep M3 compiling while it migrates to the dynamic unit API.
ExperimentSettings = DynamicExperimentSettings
SingleSpineExperiment = DynamicSingleSpineExperiment

__all__ = [
    "AxialMode",
    "ConstitutiveResponse",
    "ContactState",
    "DynamicContactSettings",
    "DynamicExperimentSettings",
    "DynamicIntegratorSettings",
    "DynamicPathPoint",
    "DynamicPathSummary",
    "DynamicSingleSpineExperiment",
    "DynamicSingleSpineResult",
    "DynamicSingleSpineUnit",
    "EventLabel",
    "ExperimentSettings",
    "LegacyFixedZExperiment",
    "LegacyFixedZExperimentSettings",
    "M2_LEGACY_MODEL_LEVEL",
    "M2_MODEL_LEVEL",
    "M2_MODULE_VERSION",
    "PathPoint",
    "PathSummary",
    "PathTerminalState",
    "PrescribedPoseConstitutiveCore",
    "SingleSpineExperiment",
    "SingleSpineExperimentResult",
    "SingleSpineState",
    "SolverSettings",
    "SpineParameters",
    "SpringState",
]
