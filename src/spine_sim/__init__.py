"""Public M0-M2 interfaces for the spine coupling simulator."""

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
    PrescribedPoseConstitutiveCore,
    SingleSpineExperiment,
    SingleSpineState,
    SpineParameters,
)
from .terrain.models import RegionSpec, TerrainRecipe, TrackGeometry

__all__ = [
    "BackendConfig",
    "CampaignSpec",
    "AxialMode",
    "ContactState",
    "Event",
    "EventType",
    "FrameMetadata",
    "M2CaseSpec",
    "M3CaseSpec",
    "M4CaseSpec",
    "ProjectConfig",
    "PrescribedPoseConstitutiveCore",
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

__version__ = "0.3.0"
