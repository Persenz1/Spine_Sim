"""Public M0-M3 interfaces for the spine coupling simulator."""

from .core.config import (
    BackendConfig,
    CampaignSpec,
    M2CaseSpec,
    M3CaseSpec,
    M4CaseSpec,
    ProjectConfig,
    TerrainRecipeRef,
    TerrainRegionSpec,
)
from .core.frames import FrameMetadata, Wrench
from .core.identity import identity, stable_hash
from .core.states import Event, EventType, StateBundle
from .contact import (
    AxialMode,
    ContactState,
    LegacyPrescribedPoseConstitutiveCore,
    SingleSpineExperiment,
    SingleSpineState,
    SpineParameters,
)
from .terrain.models import RegionSpec, TerrainRecipe, TrackGeometry
from .array import (
    AngleLayout,
    ArrayConfiguration,
    ArrayDynamicExperimentSettings,
    ArrayDynamicState,
    DynamicCommonBackplateArray,
    DynamicCommonBackplateExperiment,
    LegacyArrayExperimentSettings,
    LegacyArrayState,
    LegacyCommonBackplateArray,
    LegacyFixedZCommonBackplateExperiment,
)

__all__ = [
    "BackendConfig",
    "CampaignSpec",
    "AxialMode",
    "AngleLayout",
    "ArrayConfiguration",
    "ArrayDynamicExperimentSettings",
    "ArrayDynamicState",
    "ContactState",
    "DynamicCommonBackplateArray",
    "DynamicCommonBackplateExperiment",
    "Event",
    "EventType",
    "FrameMetadata",
    "M2CaseSpec",
    "M3CaseSpec",
    "M4CaseSpec",
    "LegacyArrayExperimentSettings",
    "LegacyArrayState",
    "LegacyCommonBackplateArray",
    "LegacyFixedZCommonBackplateExperiment",
    "ProjectConfig",
    "LegacyPrescribedPoseConstitutiveCore",
    "RegionSpec",
    "StateBundle",
    "SingleSpineExperiment",
    "SingleSpineState",
    "SpineParameters",
    "TerrainRecipe",
    "TerrainRecipeRef",
    "TerrainRegionSpec",
    "TrackGeometry",
    "Wrench",
    "identity",
    "stable_hash",
]

__version__ = "0.5.0"
