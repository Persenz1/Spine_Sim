"""单刺和阵列结果的标准 metadata 与 schema 校验。

该层只校验结果是否完整表达模型版本、来源、单位、坐标和诊断状态，不判断数值结果
是否“好”或物理性能是否达标。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from spine_sim.core.versions import (
    ARRAY_MODEL_LEVEL,
    MODEL_SCHEMA_VERSION,
    PARAMETER_REGISTRY_VERSION,
    PROJECT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    SINGLE_SPINE_MODEL_LEVEL,
    SOLVER_SEMANTICS_VERSION,
)


@dataclass(frozen=True)
class CanonicalResultMetadata:
    """每个标准物理结果都必须携带的可复现性与适用范围信息。"""

    case_id: str
    normalized_input_hash: str
    model_level: str
    terrain_version: str
    geometry_version: str
    parameter_provenance: Mapping[str, Any]
    units: Mapping[str, str]
    frames: tuple[Mapping[str, Any], ...]
    assumptions: tuple[str, ...] = ()
    omissions: tuple[str, ...] = ()
    applicability: tuple[str, ...] = ()
    cannot_answer: tuple[str, ...] = ()
    project_schema_version: str = PROJECT_SCHEMA_VERSION
    model_schema_version: str = MODEL_SCHEMA_VERSION
    result_schema_version: str = RESULT_SCHEMA_VERSION
    solver_semantics_version: str = SOLVER_SEMANTICS_VERSION
    parameter_registry_version: str = PARAMETER_REGISTRY_VERSION

    def __post_init__(self) -> None:
        """检查模型层级和最小 identity/单位/坐标元数据。"""

        if self.model_level not in {
            SINGLE_SPINE_MODEL_LEVEL,
            ARRAY_MODEL_LEVEL,
        }:
            raise ValueError(f"unsupported model_level: {self.model_level}")
        for name in (
            "case_id",
            "normalized_input_hash",
            "terrain_version",
            "geometry_version",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} cannot be empty")
        if not self.units or not self.frames:
            raise ValueError("canonical results require units and frame metadata")

    def as_dict(self) -> dict[str, Any]:
        """递归转换为可持久化字典。"""

        return asdict(self)


_COMMON_REQUIRED = {
    # 所有物理层级共享的结果字段；阵列层还会在下方追加平衡/稳定性字段。
    "project_schema_version",
    "model_schema_version",
    "result_schema_version",
    "solver_semantics_version",
    "parameter_registry_version",
    "model_level",
    "case_id",
    "normalized_input_hash",
    "terrain_version",
    "geometry_version",
    "parameter_provenance",
    "units",
    "frames",
    "assumptions",
    "omissions",
    "applicability",
    "cannot_answer",
    "physical_state",
    "numerical_state",
    "model_state",
    "residuals",
    "tolerances",
    "per_spine",
}


def validate_canonical_summary(summary: Mapping[str, Any]) -> None:
    """校验 summary 字段完整性及其语义版本是否与当前 writer 一致。"""

    missing = _COMMON_REQUIRED - set(summary)
    if missing:
        raise ValueError(f"canonical result is missing fields: {sorted(missing)}")
    metadata = CanonicalResultMetadata(
        **{
            name: summary[name]
            for name in CanonicalResultMetadata.__dataclass_fields__
            if name in summary
        }
    )
    if metadata.project_schema_version != PROJECT_SCHEMA_VERSION:
        raise ValueError("project schema version does not match this writer")
    if metadata.result_schema_version != RESULT_SCHEMA_VERSION:
        raise ValueError("result schema version does not match this writer")
    if metadata.model_schema_version != MODEL_SCHEMA_VERSION:
        raise ValueError("model schema version does not match this writer")
    if metadata.solver_semantics_version != SOLVER_SEMANTICS_VERSION:
        raise ValueError("solver semantics version does not match this writer")
    if metadata.parameter_registry_version != PARAMETER_REGISTRY_VERSION:
        raise ValueError("parameter registry version does not match this writer")
    if metadata.model_level == ARRAY_MODEL_LEVEL:
        array_required = {
            "rank_status",
            "range_status",
            "equilibrium_status",
            "quasistatic_stability",
            "dynamic_stability",
        }
        array_missing = array_required - set(summary)
        if array_missing:
            raise ValueError(
                f"canonical array result is missing fields: {sorted(array_missing)}"
            )
