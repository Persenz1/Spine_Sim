"""Spine_Sim 的稳定公共接口。

这里集中导出从地形候选查询、单根 spine 求解到阵列平衡求解所需的规范类型；
调用方优先从本模块导入，可减少对包内目录结构的耦合。
"""

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
    BaseCaseSpec,
    CampaignSpec,
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
)
from .geometry import (
    CandidateCursor,
    ContactCandidate,
    SpinePath,
    SpinePose,
    SurfaceState,
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
from .runtime.backend import BackendConfig
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
    "RegionSpec",
    "RunState",
    "SingleSpineTolerances",
    "SpineAcceptedState",
    "SpineGeometry",
    "SpineInstance",
    "SpineMaterial",
    "SpinePath",
    "SpinePose",
    "SurfaceState",
    "SuspensionParameters",
    "TerrainRecipe",
    "TrackGeometry",
    "Wrench",
    "commit_array_trial",
    "commit_single_spine_trial",
    "identity",
    "query_next_candidate",
    "solve_array_equilibrium",
    "solve_single_spine",
    "stable_hash",
]

__version__ = "0.3.0"
