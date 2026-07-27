"""Public M0/M1 interfaces for the spine coupling simulator."""

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
from .terrain.models import RegionSpec, TerrainRecipe, TrackGeometry

__all__ = [
    "BackendConfig",
    "CampaignSpec",
    "Event",
    "EventType",
    "FrameMetadata",
    "M2CaseSpec",
    "M3CaseSpec",
    "M4CaseSpec",
    "ProjectConfig",
    "RegionSpec",
    "StateBundle",
    "TerrainRecipe",
    "TerrainRecipeRef",
    "TerrainRegionSpec",
    "TrackGeometry",
    "Wrench",
    "identity",
    "stable_hash",
]

__version__ = "0.2.0"
