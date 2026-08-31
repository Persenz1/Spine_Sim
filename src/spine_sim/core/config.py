"""带语义版本和稳定 identity 的 case/campaign 配置。

配置对象假定物理量已经归一化为 SI；开放的 ``parameters`` 由具体 case adapter 校验，
其规范哈希会直接参与 case ID，因此运行阶段不得再静默改写参数表示。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .errors import ConfigurationError
from .identity import identity, stable_hash
from .versions import (
    GEOMETRY_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    PARAMETER_REGISTRY_VERSION,
    PROJECT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    SOLVER_SEMANTICS_VERSION,
)


def _unknown(data: Mapping[str, Any], allowed: set[str], name: str) -> None:
    """拒绝拼写错误或当前 schema 未声明的配置字段。"""

    extra = set(data) - allowed
    if extra:
        raise ConfigurationError(f"{name} contains unknown fields: {sorted(extra)}")


@dataclass(frozen=True)
class BaseCaseSpec:
    """一个可独立执行、内容寻址的仿真 case。"""

    module: str
    module_version: str
    parameters: Mapping[str, Any]
    upstream_hash: str = ""
    tags: tuple[str, ...] = ()
    project_schema_version: str = PROJECT_SCHEMA_VERSION
    model_schema_version: str = MODEL_SCHEMA_VERSION
    result_schema_version: str = RESULT_SCHEMA_VERSION
    solver_semantics_version: str = SOLVER_SEMANTICS_VERSION
    terrain_version: str = ""
    geometry_version: str = GEOMETRY_SCHEMA_VERSION
    parameter_registry_version: str = PARAMETER_REGISTRY_VERSION

    def __post_init__(self) -> None:
        """检查决定结果语义的模块名和版本字段。"""

        if not self.module:
            raise ConfigurationError("module cannot be empty")
        if not self.module_version:
            raise ConfigurationError("module_version cannot be empty")
        for name in (
            "project_schema_version",
            "model_schema_version",
            "result_schema_version",
            "solver_semantics_version",
            "geometry_version",
            "parameter_registry_version",
        ):
            if not getattr(self, name):
                raise ConfigurationError(f"{name} cannot be empty")

    @property
    def normalized_input_hash(self) -> str:
        """仅对规范参数求哈希，供结果 metadata 和 case identity 复用。"""

        return stable_hash(self.parameters)

    @property
    def case_id(self) -> str:
        """由求解语义、参数和上游来源共同生成 case ID。"""

        return identity(
            "case",
            {
                "module": self.module,
                "project_schema_version": self.project_schema_version,
                "model_schema_version": self.model_schema_version,
                "result_schema_version": self.result_schema_version,
                "solver_semantics_version": self.solver_semantics_version,
                "terrain_version": self.terrain_version,
                "geometry_version": self.geometry_version,
                "parameter_registry_version": self.parameter_registry_version,
                "normalized_input_hash": self.normalized_input_hash,
                "upstream_hash": self.upstream_hash,
            },
            module_version=self.module_version,
        )

    @property
    def config_hash(self) -> str:
        """对包括标签在内的完整 case 配置求哈希。"""

        return stable_hash(asdict(self))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BaseCaseSpec":
        """从 JSON 映射构造严格的 case 配置。"""

        allowed = {
            "module",
            "module_version",
            "parameters",
            "upstream_hash",
            "tags",
            "project_schema_version",
            "model_schema_version",
            "result_schema_version",
            "solver_semantics_version",
            "terrain_version",
            "geometry_version",
            "parameter_registry_version",
        }
        _unknown(data, allowed, "case")
        normalized = dict(data)
        normalized["tags"] = tuple(normalized.get("tags", ()))
        return cls(**normalized)


@dataclass(frozen=True)
class CampaignSpec:
    """共享一个调用入口的一组唯一 case 及其运行策略。"""

    name: str
    module_version: str
    callable: str
    cases: tuple[BaseCaseSpec, ...]
    workers: int = 1
    mode: str = "small"

    def __post_init__(self) -> None:
        """校验调用路径、并行度、存储模式和 case ID 唯一性。"""

        if not self.name or not self.module_version:
            raise ConfigurationError("campaign name/version cannot be empty")
        if ":" not in self.callable:
            raise ConfigurationError("callable must be module.path:function")
        if self.workers < 1:
            raise ConfigurationError("workers must be at least one")
        if self.mode not in {"small", "formal"}:
            raise ConfigurationError("campaign mode must be small or formal")
        if not self.cases:
            raise ConfigurationError("campaign must contain at least one case")
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ConfigurationError("campaign case IDs must be unique")

    @property
    def campaign_id(self) -> str:
        """由 campaign 名称、入口、模式和有序 case 集生成稳定 ID。"""

        return identity(
            "campaign",
            {
                "name": self.name,
                "callable": self.callable,
                "mode": self.mode,
                "case_ids": [case.case_id for case in self.cases],
            },
            module_version=self.module_version,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CampaignSpec":
        """从 JSON 映射递归构造 campaign 与其 case。"""

        allowed = {"name", "module_version", "callable", "cases", "workers", "mode"}
        _unknown(data, allowed, "campaign")
        cases = [
            BaseCaseSpec.from_mapping(raw)
            for raw in data.get("cases", [])
        ]
        return cls(
            name=str(data["name"]),
            module_version=str(data["module_version"]),
            callable=str(data["callable"]),
            cases=tuple(cases),
            workers=int(data.get("workers", 1)),
            mode=str(data.get("mode", "small")),
        )
