"""版本化材料 profile 加载，并严格隔离不同材料的 subtype。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from spine_sim.core.identity import stable_hash

from .errors import TerrainConfigurationError


PROFILE_SCHEMA_VERSION = "material-profile-v1"
SUPPORTED_MATERIALS = ("sandpaper", "red_brick", "concrete")
_PROFILE_DIRECTORY = Path(__file__).with_name("material_profiles")


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """深拷贝递归合并 defaults 与 subtype 覆盖，避免修改缓存文档。"""

    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def available_profiles() -> dict[str, tuple[str, ...]]:
    """不导入生成算法，仅列出配置文件中的材料和 subtype。"""

    result: dict[str, tuple[str, ...]] = {}
    for material in SUPPORTED_MATERIALS:
        document = _load_document(material)
        result[material] = tuple(sorted(document["subtypes"]))
    return result


def _load_document(material: str) -> dict[str, Any]:
    """读取并校验一个材料 JSON 的 schema、标签和 subtype 表。"""

    if material not in SUPPORTED_MATERIALS:
        raise TerrainConfigurationError(
            f"unsupported material {material!r}; choose {SUPPORTED_MATERIALS}"
        )
    path = _PROFILE_DIRECTORY / f"{material}.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TerrainConfigurationError(
            f"cannot load material profile {path}: {exc}"
        ) from exc
    if document.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise TerrainConfigurationError(
            f"unsupported profile schema in {path.name}"
        )
    if document.get("material") != material:
        raise TerrainConfigurationError(
            f"profile file {path.name} is labelled as another material"
        )
    if not isinstance(document.get("subtypes"), dict):
        raise TerrainConfigurationError(f"profile {path.name} has no subtypes")
    return document


def load_material_profile(
    material: str, subtype: str | None = None
) -> dict[str, Any]:
    """返回 defaults 已合并的 profile，并拒绝跨材料复用 subtype。"""

    document = _load_document(material)
    selected = subtype or document.get("default_subtype")
    if selected not in document["subtypes"]:
        owner = None
        for other_material in SUPPORTED_MATERIALS:
            if other_material == material:
                continue
            if selected in _load_document(other_material)["subtypes"]:
                owner = other_material
                break
        detail = f"; it belongs to {owner!r}" if owner else ""
        raise TerrainConfigurationError(
            f"unknown subtype {selected!r} for material {material!r}{detail}"
        )
    resolved = _merge(document.get("defaults", {}), document["subtypes"][selected])
    resolved.update(
        {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "material": material,
            "subtype": selected,
        }
    )
    required = {"status", "parameter_basis", "generation"}
    missing = required - resolved.keys()
    if missing:
        raise TerrainConfigurationError(
            f"profile {material}/{selected} is missing {sorted(missing)}"
        )
    if resolved["status"] not in {
        "validated",
        "partially_validated",
        "provisional",
    }:
        raise TerrainConfigurationError(
            f"invalid validation status for {material}/{selected}"
        )
    # 哈希在加入 profile_hash 自身之前计算，绑定全部实际生成参数和验证状态。
    resolved["profile_hash"] = stable_hash(resolved)
    return resolved
