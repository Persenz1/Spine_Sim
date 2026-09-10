"""Spine_Sim 的稳定公共接口。

这里集中导出从地形候选查询、单根 spine 求解到阵列平衡求解所需的规范类型；
调用方优先从本模块导入，可减少对包内目录结构的耦合。

这些单刺/阵列函数保留旧机理语义。IJMS 有限导向杆路径从
``spine_sim.ijms`` 导入；局部新模型从 ``spine_sim.guided_rod`` 和
``spine_sim.guided_array`` 导入，避免将两种物理链混用。
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
    "drive_candidate_path",
    "identity",
    "query_next_candidate",
    "solve_array_equilibrium",
    "solve_single_spine",
    "stable_hash",
]

__version__ = "0.5.0"
