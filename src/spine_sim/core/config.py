"""Small versioned configuration schema normalized to SI."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigurationError
from .identity import identity, stable_hash
from .units import require_range, to_si


def _unknown(data: Mapping[str, Any], allowed: set[str], name: str) -> None:
    extra = set(data) - allowed
    if extra:
        raise ConfigurationError(f"{name} contains unknown fields: {sorted(extra)}")


@dataclass(frozen=True)
class BackendConfig:
    preference: str = "auto"
    allow_gpu: bool = True
    device_index: int = 0

    def __post_init__(self) -> None:
        if self.preference not in {"auto", "cpu", "cuda"}:
            raise ConfigurationError("backend preference must be auto, cpu or cuda")
        if self.device_index < 0:
            raise ConfigurationError("device_index must be non-negative")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BackendConfig":
        _unknown(data, {"preference", "allow_gpu", "device_index"}, "backend")
        return cls(**data)


@dataclass(frozen=True)
class TerrainRecipeRef:
    recipe_name: str
    recipe_version: str
    seed: int
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.recipe_name or not self.recipe_version:
            raise ConfigurationError("terrain recipe name/version cannot be empty")
        if self.seed < 0:
            raise ConfigurationError("terrain seed must be non-negative")

    @property
    def terrain_recipe_id(self) -> str:
        return identity("terrain_recipe", asdict(self), module_version=self.recipe_version)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TerrainRecipeRef":
        _unknown(data, {"recipe_name", "recipe_version", "seed", "parameters"}, "terrain")
        return cls(**data)


@dataclass(frozen=True)
class TerrainRegionSpec:
    terrain_recipe_id: str
    origin_x_m: float
    origin_y_m: float
    size_x_m: float
    size_y_m: float
    resolution_m: float

    def __post_init__(self) -> None:
        for name in ("size_x_m", "size_y_m", "resolution_m"):
            require_range(getattr(self, name), name=name, minimum=0.0, inclusive_min=False)
        if not self.terrain_recipe_id:
            raise ConfigurationError("terrain_recipe_id cannot be empty")

    @property
    def region_id(self) -> str:
        return identity("region", asdict(self))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TerrainRegionSpec":
        allowed = {
            "terrain_recipe_id", "origin_x", "origin_y", "size_x", "size_y", "resolution"
        }
        _unknown(data, allowed, "terrain_region")
        return cls(
            terrain_recipe_id=str(data["terrain_recipe_id"]),
            origin_x_m=to_si(data["origin_x"], "length", name="origin_x"),
            origin_y_m=to_si(data["origin_y"], "length", name="origin_y"),
            size_x_m=to_si(data["size_x"], "length", name="size_x"),
            size_y_m=to_si(data["size_y"], "length", name="size_y"),
            resolution_m=to_si(data["resolution"], "length", name="resolution"),
        )


@dataclass(frozen=True)
class BaseCaseSpec:
    module: str
    module_version: str
    parameters: Mapping[str, Any]
    upstream_hash: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.module:
            raise ConfigurationError("module cannot be empty")
        if not self.module_version:
            raise ConfigurationError("module_version cannot be empty")

    @property
    def case_id(self) -> str:
        return identity(
            "case",
            {
                "module": self.module,
                "parameters": self.parameters,
                "upstream_hash": self.upstream_hash,
            },
            module_version=self.module_version,
        )

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BaseCaseSpec":
        allowed = {"module", "module_version", "parameters", "upstream_hash", "tags"}
        _unknown(data, allowed, "case")
        normalized = dict(data)
        normalized["tags"] = tuple(normalized.get("tags", ()))
        return cls(**normalized)


@dataclass(frozen=True)
class CampaignSpec:
    name: str
    module_version: str
    callable: str
    cases: tuple[BaseCaseSpec, ...]
    workers: int = 1
    mode: str = "small"

    def __post_init__(self) -> None:
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
        return identity(
            "campaign",
            {"name": self.name, "case_ids": [case.case_id for case in self.cases]},
            module_version=self.module_version,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CampaignSpec":
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


@dataclass(frozen=True)
class ProjectConfig:
    schema_version: str
    module_version: str
    results_root: Path
    backend: BackendConfig = field(default_factory=BackendConfig)
    absolute_tolerance: float = 1e-9
    relative_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ConfigurationError("only ProjectConfig schema_version '1' is supported")
        if not self.module_version:
            raise ConfigurationError("module_version cannot be empty")
        require_range(self.absolute_tolerance, name="absolute_tolerance", minimum=0, inclusive_min=False)
        require_range(self.relative_tolerance, name="relative_tolerance", minimum=0, inclusive_min=False)

    @property
    def config_hash(self) -> str:
        return stable_hash(self.normalized())

    def normalized(self) -> dict[str, Any]:
        value = asdict(self)
        value["results_root"] = self.results_root.as_posix()
        return value

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, base_dir: Path) -> "ProjectConfig":
        allowed = {
            "schema_version", "module_version", "results_root", "backend",
            "absolute_tolerance", "relative_tolerance",
        }
        _unknown(data, allowed, "project")
        raw_root = Path(str(data.get("results_root", "results")))
        root = raw_root if raw_root.is_absolute() else base_dir / raw_root
        return cls(
            schema_version=str(data["schema_version"]),
            module_version=str(data["module_version"]),
            results_root=root.resolve(),
            backend=BackendConfig.from_mapping(data.get("backend", {})),
            absolute_tolerance=float(data.get("absolute_tolerance", 1e-9)),
            relative_tolerance=float(data.get("relative_tolerance", 1e-7)),
        )


def load_json_config(path: str | Path) -> tuple[dict[str, Any], ProjectConfig]:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read JSON config {config_path}: {exc}") from exc
    return raw, ProjectConfig.from_mapping(raw, base_dir=config_path.parent)
