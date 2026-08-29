"""Public canonical terrain-to-single-spine-to-array interfaces."""

from .array import (
    ArrayAcceptedState,
    ArrayResult,
    ArrayTolerances,
    ArrayTrial,
    ControlMode,
    MixedControl,
    SpineInstance,
    commit_array_trial,
    solve_array_equilibrium,
)

from .core.config import (
    BackendConfig,
    BaseCaseSpec,
    CampaignSpec,
    ProjectConfig,
    TerrainRecipeRef,
    TerrainRegionSpec,
)
from .core.frames import FrameMetadata, Wrench
from .core.identity import identity, stable_hash
from .core.states import (
    Event,
    EventType,
    ModelState,
    NumericalState,
    PhysicalState,
    RunState,
    StateBundle,
)
from .geometry import (
    CandidateCursor,
    ContactCandidate,
    SpinePath,
    SpinePose,
    SurfaceState,
    drive_candidate_path,
    query_next_candidate,
)
from .single_spine import (
    BaseMotion,
    FrictionParameters,
    SingleSpineTolerances,
    SpineAcceptedState,
    SpineGeometry,
    SpineMaterial,
    SuspensionParameters,
    commit_single_spine_trial,
    solve_single_spine,
)
from .terrain.models import RegionSpec, TerrainRecipe, TrackGeometry

__all__ = [
    "BackendConfig",
    "BaseMotion",
    "BaseCaseSpec",
    "ArrayAcceptedState",
    "ArrayResult",
    "ArrayTolerances",
    "ArrayTrial",
    "CandidateCursor",
    "CampaignSpec",
    "ContactCandidate",
    "ControlMode",
    "Event",
    "EventType",
    "FrameMetadata",
    "FrictionParameters",
    "MixedControl",
    "ModelState",
    "NumericalState",
    "PhysicalState",
    "ProjectConfig",
    "RegionSpec",
    "RunState",
    "SingleSpineTolerances",
    "SpineAcceptedState",
    "SpineGeometry",
    "SpineInstance",
    "SpineMaterial",
    "SpinePath",
    "SpinePose",
    "StateBundle",
    "SurfaceState",
    "SuspensionParameters",
    "TerrainRecipe",
    "TerrainRecipeRef",
    "TerrainRegionSpec",
    "TrackGeometry",
    "Wrench",
    "commit_array_trial",
    "commit_single_spine_trial",
    "drive_candidate_path",
    "identity",
    "query_next_candidate",
    "solve_array_equilibrium",
    "solve_single_spine",
    "stable_hash",
]

__version__ = "0.3.0"
