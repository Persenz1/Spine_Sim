"""工程参数注册表及旧版设计输入迁移。

本模块把参数的数值与其证据状态、来源绑定在一起，避免把“候选值”、
“数值协议”或“旧版回归值”误当成已经标定的物理真值。它还负责重建旧版
M3 参数扫描、生成可复现的历史设计编号，并把旧数据显式转换为当前 SI 制
规范下的 :class:`CanonicalDesign`。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from spine_sim.core.identity import identity
from spine_sim.core.versions import PARAMETER_REGISTRY_VERSION


REGISTRY_SCHEMA_VERSION = "spine-parameter-registry-v1"

# 注册表接受的证据类别。这里采用白名单，拼错或新增但未处理的状态会在
# 加载时立即失败，而不会悄悄进入仿真结果。
_EVIDENCE_STATUSES = frozenset(
    {
        "frozen_design_semantics",
        "candidate_design_value",
        "legacy_regression",
        "numerical_protocol",
        "legacy_analysis",
        "unclosed",
    }
)
_ANGLE_PATTERN_ALIASES = {
    "fixed": "fixed",
    "60_to_80": "60_to_80",
    "80_to_60": "80_to_60",
    "80_to_50": "80_to_50",
    "gradient_60_to_80": "60_to_80",
    "gradient_80_to_60": "80_to_60",
    "gradient_80_to_50": "80_to_50",
}


@dataclass(frozen=True, slots=True)
class EvidenceValue:
    """一个带证据等级和来源引用的参数值。"""

    value: Any
    evidence_status: str
    source: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Installation:
    """安装方式：刚性固定，或只能受压的单边弹簧。"""

    mode: str
    stiffness_N_per_m: float | None = None
    travel_m: float | None = None
    preload_N: float = 0.0
    tension_allowed: bool = False

    def __post_init__(self) -> None:
        """检查标签与附加参数是否构成一个自洽的安装配置。"""

        if self.mode == "rigid":
            if self.stiffness_N_per_m is not None or self.travel_m is not None:
                raise ValueError("rigid installation cannot carry spring values")
            return
        if self.mode != "unilateral_spring":
            raise ValueError("installation mode must be rigid or unilateral_spring")
        if (
            self.stiffness_N_per_m is None
            or not math.isfinite(self.stiffness_N_per_m)
            or self.stiffness_N_per_m <= 0.0
        ):
            raise ValueError("unilateral spring stiffness must be positive")
        if (
            self.travel_m is None
            or not math.isfinite(self.travel_m)
            or self.travel_m <= 0.0
        ):
            raise ValueError("unilateral spring travel must be positive")
        if self.preload_N != 0.0 or self.tension_allowed:
            raise ValueError("legacy unilateral springs are unpreloaded and cannot pull")

    def as_dict(self) -> dict[str, Any]:
        """返回适合序列化和身份计算的普通字典。"""

        if self.mode == "rigid":
            return {"mode": "rigid"}
        return {
            "mode": self.mode,
            "stiffness_N_per_m": self.stiffness_N_per_m,
            "travel_m": self.travel_m,
            "preload_N": self.preload_N,
            "tension_allowed": self.tension_allowed,
        }


@dataclass(frozen=True, slots=True)
class SpinePackage:
    """M3a 参数包：单根 spine 的几何、姿态与安装参数。"""

    tip_radius_m: float
    rod_diameter_m: float
    fixed_pitch_rad: float
    installation: Installation
    fixed_length_m: float
    yaw_rad: float
    legacy_package_id: str
    evidence_status: str
    source: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ArrayGeometry:
    """M3b 阵列几何；梯度端点沿 x 列展开。"""

    nx: int
    ny: int
    spacing_m: float
    legacy_angle_pattern: str
    pitch_endpoints_rad: tuple[float, float] | None
    legacy_geometry_id: str
    evidence_status: str
    source: tuple[str, ...]

    @property
    def array_shape(self) -> str:
        """返回旧版使用的 ``nx x ny`` 形状标签。"""

        return f"{self.nx}x{self.ny}"


@dataclass(frozen=True, slots=True)
class CanonicalDesign:
    """求解器使用的规范设计，长度和俯仰角均按 x 列显式给出。"""

    nx: int
    ny: int
    spacing_m: float
    pitch_by_x_column_rad: tuple[float, ...]
    yaw_rad: float
    length_by_x_column_m: tuple[float, ...]
    tip_radius_m: float
    rod_diameter_m: float
    installation: Installation
    material_category: str
    legacy_angle_pattern: str
    legacy_design_id: str | None
    evidence_status: str
    source: tuple[str, ...]

    @property
    def mount_x_by_column_m(self) -> tuple[float, ...]:
        """返回以阵列中心为原点的各列安装点 x 坐标。"""

        return tuple(
            (index - 0.5 * (self.nx - 1)) * self.spacing_m
            for index in range(self.nx)
        )

    @property
    def axis_by_x_column(self) -> tuple[tuple[float, float, float], ...]:
        """把每列的俯仰角与公共偏航角转换为三维单位轴向。"""

        return tuple(
            axis_from_pitch_yaw(pitch, self.yaw_rad)
            for pitch in self.pitch_by_x_column_rad
        )


@dataclass(frozen=True, slots=True)
class LegacyDesign:
    """旧版映射、历史编号与规范设计之间的可追溯绑定。"""

    legacy_design_id: str
    legacy_mapping: Mapping[str, Any]
    canonical: CanonicalDesign
    evidence_status: str
    source: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Protocol:
    """一组命名仿真协议参数及其逐字段证据元数据。"""

    protocol_id: str
    evidence_status: str
    source: tuple[str, ...]
    fields: Mapping[str, EvidenceValue]
    registry_version: str

    @property
    def values(self) -> dict[str, Any]:
        """只提取字段值；证据审计时应直接读取 :attr:`fields`。"""

        return {name: field.value for name, field in self.fields.items()}

    @property
    def protocol_identity(self) -> str:
        """计算包含参数、证据来源和注册表版本的稳定协议身份。"""

        return identity(
            "parameter_protocol",
            {
                "protocol_id": self.protocol_id,
                "evidence_status": self.evidence_status,
                "source": self.source,
                "fields": {
                    name: {
                        "value": field.value,
                        "evidence_status": field.evidence_status,
                        "source": field.source,
                    }
                    for name, field in self.fields.items()
                },
            },
            module_version=self.registry_version,
        )


@dataclass(frozen=True, slots=True)
class TerminalPreset:
    """按机理和用途命名的终端候选设计。"""

    role_id: str
    legacy_design_id: str
    role: str
    mechanism: str
    reason: str
    design: LegacyDesign
    evidence_status: str
    source: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SelectionSet:
    """一组有顺序的终端角色及其对应旧版设计编号。"""

    selection_id: str
    role_ids: tuple[str, ...]
    legacy_design_ids: tuple[str, ...]
    evidence_status: str
    source: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MigrationField:
    """旧字段到规范字段的一条转换记录。"""

    legacy_field: str
    canonical_field: str
    legacy_value: Any
    canonical_value: Any
    conversion: str
    evidence_status: str
    source: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """将转换记录序列化为 JSON 兼容字典。"""

        return {
            "legacy_field": self.legacy_field,
            "canonical_field": self.canonical_field,
            "legacy_value": self.legacy_value,
            "canonical_value": self.canonical_value,
            "conversion": self.conversion,
            "evidence_status": self.evidence_status,
            "source": list(self.source),
        }


@dataclass(frozen=True, slots=True)
class MigrationReport:
    """一次旧版设计导入的逐字段审计报告。"""

    protocol_id: str
    legacy_design_id: str
    fields: tuple[MigrationField, ...]
    unexplained_differences: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """将完整迁移报告序列化为 JSON 兼容字典。"""

        return {
            "protocol_id": self.protocol_id,
            "legacy_design_id": self.legacy_design_id,
            "fields": [field.as_dict() for field in self.fields],
            "unexplained_differences": list(self.unexplained_differences),
        }


@dataclass(frozen=True, slots=True)
class ImportedLegacyDesign:
    """迁移后的规范设计及与之一一对应的报告。"""

    design: CanonicalDesign
    report: MigrationReport


def axis_from_pitch_yaw(
    pitch_rad: float, yaw_rad: float
) -> tuple[float, float, float]:
    """由俯仰角和偏航角构造右手坐标系中的三维单位轴向。

    约定为 ``(cos(pitch) cos(yaw), cos(pitch) sin(yaw), sin(pitch))``；
    因此负俯仰角指向负 z 方向。
    """

    if not math.isfinite(float(pitch_rad)) or not math.isfinite(float(yaw_rad)):
        raise ValueError("pitch and yaw must be finite")
    cos_pitch = math.cos(float(pitch_rad))
    return (
        cos_pitch * math.cos(float(yaw_rad)),
        cos_pitch * math.sin(float(yaw_rad)),
        math.sin(float(pitch_rad)),
    )


def equal_height_length_m(pitch_rad: float) -> float:
    """计算保持旧版“80°、4 mm”竖直高度不变所需的杆长。

    梯度阵列采用 ``L_j |sin(pitch_j)| = 4 mm sin(80°)``，使不同
    俯仰角列的初始端点高度一致。
    """

    pitch = float(pitch_rad)
    if not math.isfinite(pitch):
        raise ValueError("equal-height pitch must be finite")
    denominator = abs(math.sin(pitch))
    if denominator <= 0.0:
        raise ValueError("equal-height pitch must have a nonzero sine")
    reference_height_m = 0.004 * math.sin(math.radians(80.0))
    return reference_height_m / denominator


def _columns(start: float, end: float, count: int) -> tuple[float, ...]:
    """在起止值之间生成包含两个端点的等间距列参数。"""

    if count < 1:
        raise ValueError("column count must be positive")
    if not math.isfinite(float(start)) or not math.isfinite(float(end)):
        raise ValueError("column endpoint pitches must be finite")
    if count == 1:
        return (float(start),)
    return tuple(
        float(start + (end - start) * index / (count - 1))
        for index in range(count)
    )


def _legacy_angle_deg(pitch_rad: float) -> float:
    """把当前有符号俯仰角还原为旧版向下为正的角度值。"""

    return float(round(-math.degrees(float(pitch_rad)), 12))


def _spring_family(stiffness_N_per_m: float | None) -> str:
    """复现旧版按刚度划分的 rigid/compliant/stiff 标签。"""

    if stiffness_N_per_m is None:
        return "rigid"
    return "compliant" if stiffness_N_per_m <= 500.0 else "stiff"


def _legacy_package_mapping(
    tip_radius_m: float,
    rod_diameter_m: float,
    fixed_pitch_rad: float,
    installation: Installation,
) -> dict[str, Any]:
    """由规范物理量生成确定性的旧版 package 映射和标签。"""

    angle_deg = _legacy_angle_deg(fixed_pitch_rad)
    stiffness = (
        None
        if installation.mode == "rigid"
        else float(installation.stiffness_N_per_m)  # type: ignore[arg-type]
    )
    spring_label = "rigid" if stiffness is None else f"k{int(round(stiffness))}"
    package_id = (
        f"rt{int(round(tip_radius_m * 1e6))}um_"
        f"d{int(round(rod_diameter_m * 1e6))}um_"
        f"a{int(round(angle_deg))}deg_{spring_label}"
    )
    return {
        "tip_radius_m": float(tip_radius_m),
        "diameter_m": float(rod_diameter_m),
        "fixed_angle_deg": angle_deg,
        "spring_stiffness_N_per_m": stiffness,
        "package_id": package_id,
        "spring_family": _spring_family(stiffness),
    }


def _legacy_geometry_mapping(
    nx: int, ny: int, spacing_m: float, angle_pattern: str
) -> dict[str, Any]:
    """由规范阵列尺寸生成旧版 geometry 映射和可读编号。"""

    geometry_id = (
        f"{nx}x{ny}_s{int(round(spacing_m * 1e3))}mm_{angle_pattern}"
    )
    return {
        "nx": int(nx),
        "ny": int(ny),
        "spacing_m": float(spacing_m),
        "angle_pattern": angle_pattern,
        "array_shape": f"{nx}x{ny}",
        "geometry_id": geometry_id,
    }


def _legacy_digest(value: Mapping[str, Any]) -> str:
    """按旧版规范 JSON 编码计算截断为 20 位十六进制的摘要。"""

    # 禁止 NaN，并固定键序和分隔符，保证同一物理设计跨运行得到同一摘要。
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def legacy_design_id(value: Mapping[str, Any]) -> str:
    """严格复现历史 M3 全扫描的 20 位十六进制设计编号。

    该编号仅用于来源追踪，不替代当前模型的规范 identity。
    """

    package = _normalize_legacy_package(value["package"])
    geometry = _normalize_legacy_geometry(value["geometry"])
    return f"m3_full_design_{_legacy_digest({'package': package, 'geometry': geometry})}"


def _normalize_legacy_package(value: Any) -> dict[str, Any]:
    """校验旧 package 的原始字段，并重新计算所有派生标签。"""

    if not isinstance(value, Mapping):
        raise TypeError("legacy package must be a mapping")
    try:
        radius = float(value["tip_radius_m"])
        diameter = float(value["diameter_m"])
        angle_deg = float(value["fixed_angle_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("legacy package has invalid physical fields") from exc
    if (
        not math.isfinite(radius)
        or radius <= 0.0
        or not math.isfinite(diameter)
        or diameter <= 0.0
        or not math.isfinite(angle_deg)
        or not 0.0 < angle_deg < 90.0
    ):
        raise ValueError("legacy package physical fields are outside their domain")
    # 旧格式以 null 表示刚性固定；数值刚度表示无预紧、只能受压的弹簧。
    raw_stiffness = value.get("spring_stiffness_N_per_m")
    if raw_stiffness is None:
        installation = Installation("rigid")
    else:
        installation = Installation(
            "unilateral_spring", float(raw_stiffness), 0.004
        )
    normalized = _legacy_package_mapping(
        radius, diameter, -math.radians(angle_deg), installation
    )
    # 不信任输入中的派生字段，防止“物理量已改但标签未更新”的脏数据。
    for derived in ("package_id", "spring_family"):
        if derived in value and value[derived] != normalized[derived]:
            raise ValueError(f"legacy package {derived} does not match its inputs")
    return normalized


def _normalize_legacy_geometry(value: Any) -> dict[str, Any]:
    """校验旧 geometry，并把历史梯度别名归一到固定名称。"""

    if not isinstance(value, Mapping):
        raise TypeError("legacy geometry must be a mapping")
    try:
        raw_nx = value["nx"]
        raw_ny = value["ny"]
        nx = int(raw_nx)
        ny = int(raw_ny)
        spacing_m = float(value["spacing_m"])
        raw_pattern = str(value.get("angle_pattern", "fixed"))
        pattern = _ANGLE_PATTERN_ALIASES[raw_pattern]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("legacy geometry has invalid fields") from exc
    if (
        isinstance(raw_nx, bool)
        or isinstance(raw_ny, bool)
        or nx != raw_nx
        or ny != raw_ny
        or nx < 1
        or ny < 1
        or not math.isfinite(spacing_m)
        or spacing_m <= 0.0
    ):
        raise ValueError("legacy geometry fields are outside their domain")
    normalized = _legacy_geometry_mapping(nx, ny, spacing_m, pattern)
    # 与 package 相同，输入中的形状/编号只能作为一致性断言。
    for derived in ("array_shape", "geometry_id"):
        if derived in value and value[derived] != normalized[derived]:
            raise ValueError(f"legacy geometry {derived} does not match its inputs")
    return normalized


class ParameterRegistry:
    """经过模式、版本和证据元数据校验的只读参数注册表。

    ``candidate_axes`` 只描述各参数可取的候选值，并不自动定义它们的笛卡尔
    积；只有 ``generators`` 中明确列出的组合规则才会实例化设计空间。
    """

    def __init__(self, document: Mapping[str, Any]):
        """从已解析的 JSON 对象构建注册表，并一次性完成结构校验。"""

        if document.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise ValueError("unsupported parameter registry schema")
        self.registry_version = str(document.get("registry_version", ""))
        if self.registry_version != PARAMETER_REGISTRY_VERSION:
            raise ValueError(
                "parameter registry version must be "
                f"{PARAMETER_REGISTRY_VERSION!r}"
            )
        self._evidence_sections: dict[str, dict[str, EvidenceValue]] = {}
        self._generators: dict[
            str, tuple[Mapping[str, Any], str, tuple[str, ...]]
        ] = {}
        self._protocols: dict[str, Protocol] = {}
        self._validate_evidence(document)

    def _validate_evidence(self, document: Mapping[str, Any]) -> None:
        """解析所有证据段、生成器和协议，并保留逐字段来源。"""

        # 普通证据段的每个叶节点都必须是 EvidenceValue 形状。
        for section_name in (
            "coordinate_contract",
            "candidate_axes",
            "legacy_model_baseline",
            "unclosed_parameters",
            "legacy_id_schemes",
            "paired_seed_sets",
            "terminal_presets",
            "selection_sets",
        ):
            section = document.get(section_name)
            if not isinstance(section, Mapping):
                raise TypeError(f"registry section {section_name!r} must be a mapping")
            parsed: dict[str, EvidenceValue] = {}
            for name, raw in section.items():
                if not isinstance(name, str):
                    raise TypeError(f"{section_name} keys must be strings")
                evidence = _evidence(raw, f"{section_name}.{name}")
                if section_name == "terminal_presets" and not isinstance(
                    evidence.value, Mapping
                ):
                    raise TypeError(
                        f"terminal preset {name} must contain a mapping"
                    )
                parsed[name] = evidence
            self._evidence_sections[section_name] = parsed
        # 生成器只需保存用于组合的裸值；协议还需保留逐字段证据，以便把来源
        # 纳入 protocol_identity 和最终结果元数据。
        for section_name in ("generators", "protocols"):
            section = document.get(section_name)
            if not isinstance(section, Mapping):
                raise TypeError(f"registry section {section_name!r} must be a mapping")
            for name, raw in section.items():
                if not isinstance(name, str):
                    raise TypeError(f"{section_name} keys must be strings")
                if not isinstance(raw, Mapping):
                    raise TypeError(f"{section_name}.{name} must be a mapping")
                status, source = _metadata(raw, f"{section_name}.{name}")
                fields = raw.get("fields")
                if not isinstance(fields, Mapping) or not fields:
                    raise ValueError(f"{section_name}.{name}.fields cannot be empty")
                parsed_fields: dict[str, EvidenceValue] = {}
                for field_name, field in fields.items():
                    if not isinstance(field_name, str):
                        raise TypeError(f"{section_name}.{name} field names must be strings")
                    parsed_fields[field_name] = _evidence(
                        field, f"{section_name}.{name}.{field_name}"
                    )
                if section_name == "generators":
                    self._generators[name] = (
                        {
                            field_name: evidence.value
                            for field_name, evidence in parsed_fields.items()
                        },
                        status,
                        source,
                    )
                else:
                    self._protocols[name] = Protocol(
                        name,
                        status,
                        source,
                        parsed_fields,
                        self.registry_version,
                    )

    def evidence(self, section: str, name: str) -> EvidenceValue:
        """按段名和条目名读取带来源的参数值。"""

        return self._evidence_sections[section][name]

    def candidate_axis(self, name: str) -> EvidenceValue:
        """读取一个候选轴，但不生成任何参数组合。"""

        return self.evidence("candidate_axes", name)

    def protocol(self, protocol_id: str) -> Protocol:
        """按编号读取已验证的数值/分析协议。"""

        return self._protocols[protocol_id]

    def paired_seed_set(self, seed_set_id: str) -> tuple[int, ...]:
        """读取用于配对比较的随机种子，保持注册表中的顺序。"""

        raw = self.evidence("paired_seed_sets", seed_set_id).value
        return tuple(int(seed) for seed in raw)

    def paired_seeds_for_protocol(self, protocol_id: str) -> tuple[int, ...]:
        """返回协议引用的配对种子；未配置时返回空元组。"""

        protocol = self.protocol(protocol_id)
        seed_set = protocol.fields.get("paired_seed_set")
        return () if seed_set is None else self.paired_seed_set(str(seed_set.value))

    def generate_source_defined_m3a(self) -> tuple[SpinePackage, ...]:
        """按注册表明示的笛卡尔积生成来源定义的 M3a 单杆参数包。"""

        fields, status, source = self._generators["source_defined_m3a"]
        fixed_length_m = float(self.candidate_axis("fixed_length_m").value)
        yaw_rad = float(self.candidate_axis("yaw_rad").value[0])
        packages: list[SpinePackage] = []
        # 只有此生成器显式列出的四个轴参与组合；其它候选轴不会被隐式展开。
        for radius in fields["tip_radius_m"]:
            for diameter in fields["rod_diameter_m"]:
                for pitch in fields["fixed_pitch_rad"]:
                    for stiffness in fields["spring_stiffness_N_per_m"]:
                        installation = self._installation(stiffness)
                        mapping = _legacy_package_mapping(
                            float(radius),
                            float(diameter),
                            float(pitch),
                            installation,
                        )
                        packages.append(
                            SpinePackage(
                                float(radius),
                                float(diameter),
                                float(pitch),
                                installation,
                                fixed_length_m,
                                yaw_rad,
                                str(mapping["package_id"]),
                                status,
                                source,
                            )
                        )
        self._check_count("source_defined_m3a", fields, len(packages))
        return tuple(packages)

    def generate_source_defined_m3b(self) -> tuple[ArrayGeometry, ...]:
        """生成来源定义的 M3b 固定角与梯度角阵列几何。"""

        fields, status, source = self._generators["source_defined_m3b"]
        geometries: list[ArrayGeometry] = []
        # 固定角布局只有形状和间距；俯仰角由所配套的 SpinePackage 给出。
        for nx, ny in fields["fixed_shapes"]:
            for spacing in fields["fixed_spacing_m"]:
                geometries.append(
                    self._geometry(
                        int(nx), int(ny), float(spacing), "fixed", None, status, source
                    )
                )
        # 梯度布局单独携带首末列俯仰角，后续按 x 列线性插值。
        for layout in fields["gradient_layouts"]:
            endpoints = tuple(float(value) for value in layout["pitch_endpoints_rad"])
            for nx, ny in layout["shapes"]:
                geometries.append(
                    self._geometry(
                        int(nx),
                        int(ny),
                        float(layout["spacing_m"]),
                        str(layout["legacy_angle_pattern"]),
                        (endpoints[0], endpoints[1]),
                        status,
                        source,
                    )
                )
        self._check_count("source_defined_m3b", fields, len(geometries))
        return tuple(geometries)

    def generate_legacy_full_scan(self) -> tuple[LegacyDesign, ...]:
        """完整复建旧版 M3 扫描空间及其历史设计编号。"""

        fields, status, source = self._generators["legacy_full_scan"]
        designs: list[LegacyDesign] = []
        fixed_length_m = float(self.candidate_axis("fixed_length_m").value)
        # 循环顺序属于历史数据契约：既影响输出顺序，也便于与旧结果逐项对齐。
        for nx, ny in fields["shapes"]:
            for spacing in fields["spacing_m"]:
                for radius in fields["tip_radius_m"]:
                    for diameter in fields["rod_diameter_m"]:
                        for stiffness in fields["spring_stiffness_N_per_m"]:
                            installation = self._installation(stiffness)
                            for pitch in fields["fixed_pitch_rad"]:
                                designs.append(
                                    self._full_design(
                                        int(nx),
                                        int(ny),
                                        float(spacing),
                                        float(radius),
                                        float(diameter),
                                        float(pitch),
                                        installation,
                                        "fixed",
                                        None,
                                        fixed_length_m,
                                        status,
                                        source,
                                    )
                                )
                            # 每组 package/geometry 除固定角外再追加一个梯度设计。
                            endpoints = tuple(
                                float(value)
                                for value in fields["gradient_pitch_endpoints_rad"]
                            )
                            designs.append(
                                self._full_design(
                                    int(nx),
                                    int(ny),
                                    float(spacing),
                                    float(radius),
                                    float(diameter),
                                    endpoints[0],
                                    installation,
                                    str(fields["legacy_gradient_name"]),
                                    (endpoints[0], endpoints[1]),
                                    fixed_length_m,
                                    status,
                                    source,
                                )
                            )
        self._check_count("legacy_full_scan", fields, len(designs))
        # 期望数量和 ID 唯一性共同捕获注册表漏项、重复项或组合规则漂移。
        ids = [design.legacy_design_id for design in designs]
        if len(ids) != len(set(ids)):
            raise ValueError("legacy full-scan design IDs are not unique")
        return tuple(designs)

    def import_legacy_design(
        self,
        value: Mapping[str, Any],
        *,
        protocol_id: str = "legacy_full_scan",
    ) -> ImportedLegacyDesign:
        """把一条旧版设计严格转换为规范设计，并生成逐字段迁移报告。

        若输入带有 ``design_id``，会先用规范化后的物理载荷重算并核对，防止
        把错误编号继续传播到新结果中。
        """

        package = _normalize_legacy_package(value["package"])
        geometry = _normalize_legacy_geometry(value["geometry"])
        computed_id = legacy_design_id({"package": package, "geometry": geometry})
        supplied_id = value.get("design_id")
        if supplied_id is not None and supplied_id != computed_id:
            raise ValueError("legacy design_id does not match its physical payload")
        protocol = self.protocol(protocol_id)
        migration_source = protocol.source
        pitch_deg = float(package["fixed_angle_deg"])
        pattern = str(geometry["angle_pattern"])
        nx = int(geometry["nx"])
        # 先按历史角度模式恢复每个 x 列的俯仰角，再按同一模式恢复杆长。
        pitch_columns = self._pitch_columns(pattern, pitch_deg, nx)
        length_columns = self._length_columns(pattern, pitch_columns)
        installation = (
            Installation("rigid")
            if package["spring_stiffness_N_per_m"] is None
            else self._installation(float(package["spring_stiffness_N_per_m"]))
        )
        material = str(self.candidate_axis("material_category").value)
        design = CanonicalDesign(
            nx=nx,
            ny=int(geometry["ny"]),
            spacing_m=float(geometry["spacing_m"]),
            pitch_by_x_column_rad=pitch_columns,
            yaw_rad=0.0,
            length_by_x_column_m=length_columns,
            tip_radius_m=float(package["tip_radius_m"]),
            rod_diameter_m=float(package["diameter_m"]),
            installation=installation,
            material_category=material,
            legacy_angle_pattern=pattern,
            legacy_design_id=computed_id,
            evidence_status=protocol.evidence_status,
            source=migration_source,
        )
        # 报告明确记录每个旧字段的转换公式，使迁移结果可独立审计。
        fields = self._migration_fields(
            package,
            geometry,
            design,
            protocol.evidence_status,
            migration_source,
        )
        return ImportedLegacyDesign(
            design,
            MigrationReport(protocol_id, computed_id, fields),
        )

    def terminal_presets(self) -> tuple[TerminalPreset, ...]:
        """解析 12 个命名终端角色，并绑定到已复建的旧版设计。"""

        by_id = {
            design.legacy_design_id: design
            for design in self.generate_legacy_full_scan()
        }
        presets: list[TerminalPreset] = []
        for role_id, evidence in self._evidence_sections["terminal_presets"].items():
            value = evidence.value
            design_id = str(value["legacy_design_id"])
            try:
                design = by_id[design_id]
            except KeyError as exc:
                raise ValueError(
                    f"terminal preset {role_id} references an unknown legacy design"
                ) from exc
            presets.append(
                TerminalPreset(
                    role_id,
                    design_id,
                    str(value["role"]),
                    str(value["mechanism"]),
                    str(value["reason"]),
                    design,
                    evidence.evidence_status,
                    evidence.source,
                )
            )
        if len(presets) != 12:
            raise ValueError("terminal preset registry must contain exactly 12 roles")
        return tuple(presets)

    def selection_set(self, selection_id: str) -> SelectionSet:
        """把角色选择集解析为保持顺序的历史设计编号集合。"""

        evidence = self.evidence("selection_sets", selection_id)
        role_ids = tuple(str(value) for value in evidence.value)
        preset_by_role = {preset.role_id: preset for preset in self.terminal_presets()}
        try:
            design_ids = tuple(
                preset_by_role[role_id].legacy_design_id for role_id in role_ids
            )
        except KeyError as exc:
            raise ValueError(
                f"selection {selection_id} references an unknown terminal role"
            ) from exc
        return SelectionSet(
            selection_id,
            role_ids,
            design_ids,
            evidence.evidence_status,
            evidence.source,
        )

    def _installation(self, stiffness: Any) -> Installation:
        """将注册表的 ``rigid``/刚度值转换为显式安装标签。"""

        if stiffness == "rigid" or stiffness is None:
            return Installation("rigid")
        travel_m = float(self.candidate_axis("spring_travel_m").value)
        return Installation("unilateral_spring", float(stiffness), travel_m)

    @staticmethod
    def _geometry(
        nx: int,
        ny: int,
        spacing_m: float,
        pattern: str,
        endpoints: tuple[float, float] | None,
        status: str,
        source: tuple[str, ...],
    ) -> ArrayGeometry:
        """创建阵列几何，同时复用旧版规则生成可追踪编号。"""

        mapping = _legacy_geometry_mapping(nx, ny, spacing_m, pattern)
        return ArrayGeometry(
            nx,
            ny,
            spacing_m,
            pattern,
            endpoints,
            str(mapping["geometry_id"]),
            status,
            source,
        )

    def _full_design(
        self,
        nx: int,
        ny: int,
        spacing_m: float,
        radius_m: float,
        diameter_m: float,
        package_pitch_rad: float,
        installation: Installation,
        pattern: str,
        endpoints: tuple[float, float] | None,
        fixed_length_m: float,
        status: str,
        source: tuple[str, ...],
    ) -> LegacyDesign:
        """由一组扫描参数同时构造旧版映射与规范求解器设计。"""

        package = _legacy_package_mapping(
            radius_m, diameter_m, package_pitch_rad, installation
        )
        geometry = _legacy_geometry_mapping(nx, ny, spacing_m, pattern)
        mapping: dict[str, Any] = {"package": package, "geometry": geometry}
        design_id = legacy_design_id(mapping)
        mapping["design_id"] = design_id
        if endpoints is None:
            # 固定角设计沿所有 x 列复用相同俯仰角和 4 mm 杆长。
            pitches = (package_pitch_rad,) * nx
            lengths = (fixed_length_m,) * nx
        else:
            # 梯度设计线性插值俯仰角，并用等高规则逐列修正杆长。
            pitches = _columns(endpoints[0], endpoints[1], nx)
            lengths = tuple(equal_height_length_m(pitch) for pitch in pitches)
        canonical = CanonicalDesign(
            nx,
            ny,
            spacing_m,
            pitches,
            0.0,
            lengths,
            radius_m,
            diameter_m,
            installation,
            str(self.candidate_axis("material_category").value),
            pattern,
            design_id,
            status,
            source,
        )
        return LegacyDesign(design_id, mapping, canonical, status, source)

    @staticmethod
    def _pitch_columns(pattern: str, fixed_angle_deg: float, nx: int) -> tuple[float, ...]:
        """按旧版模式恢复每个 x 列的有符号俯仰角。"""

        if pattern == "fixed":
            return (-math.radians(fixed_angle_deg),) * nx
        legacy_endpoints = {
            "60_to_80": (60.0, 80.0),
            "80_to_60": (80.0, 60.0),
            "80_to_50": (80.0, 50.0),
        }
        start_deg, end_deg = legacy_endpoints[pattern]
        return _columns(-math.radians(start_deg), -math.radians(end_deg), nx)

    @staticmethod
    def _length_columns(
        pattern: str, pitches: tuple[float, ...]
    ) -> tuple[float, ...]:
        """恢复固定 4 mm 长度，或对梯度列应用等高长度规则。"""

        if pattern == "fixed":
            return (0.004,) * len(pitches)
        return tuple(equal_height_length_m(pitch) for pitch in pitches)

    @staticmethod
    def _migration_fields(
        package: Mapping[str, Any],
        geometry: Mapping[str, Any],
        design: CanonicalDesign,
        status: str,
        source: tuple[str, ...],
    ) -> tuple[MigrationField, ...]:
        """生成覆盖几何、角度、安装和来源编号的迁移台账。"""

        unchanged = "identity_after_explicit_SI_parse"
        return (
            MigrationField(
                "geometry.nx",
                "nx",
                geometry["nx"],
                design.nx,
                unchanged,
                status,
                source,
            ),
            MigrationField(
                "geometry.ny",
                "ny",
                geometry["ny"],
                design.ny,
                unchanged,
                status,
                source,
            ),
            MigrationField(
                "geometry.spacing_m",
                "spacing_m",
                geometry["spacing_m"],
                design.spacing_m,
                unchanged,
                status,
                source,
            ),
            MigrationField(
                "package.tip_radius_m",
                "tip_radius_m",
                package["tip_radius_m"],
                design.tip_radius_m,
                unchanged,
                status,
                source,
            ),
            MigrationField(
                "package.diameter_m",
                "rod_diameter_m",
                package["diameter_m"],
                design.rod_diameter_m,
                unchanged,
                status,
                source,
            ),
            MigrationField(
                "package.fixed_angle_deg + geometry.angle_pattern",
                "pitch_by_x_column_rad",
                {
                    "fixed_angle_deg": package["fixed_angle_deg"],
                    "angle_pattern": geometry["angle_pattern"],
                },
                list(design.pitch_by_x_column_rad),
                "preserve_x_column_order_then_pitch_rad=-deg2rad(alpha_old_deg)",
                status,
                source,
            ),
            MigrationField(
                "implicit_legacy_yaw_deg",
                "yaw_rad",
                0.0,
                design.yaw_rad,
                "yaw_rad=deg2rad(beta_yaw_old_deg)",
                status,
                source,
            ),
            MigrationField(
                "fixed_length_or_equal_height_rule",
                "length_by_x_column_m",
                "4 mm fixed; L_j*sin(alpha_j)=4 mm*sin(80 deg) for gradient",
                list(design.length_by_x_column_m),
                "explicit_SI_equal_height_conversion",
                status,
                source,
            ),
            MigrationField(
                "package.spring_stiffness_N_per_m",
                "installation",
                package["spring_stiffness_N_per_m"],
                design.installation.as_dict(),
                "legacy_null_to_rigid_tag_else_unilateral_spring_tag",
                status,
                source,
            ),
            MigrationField(
                "legacy_design_id",
                "legacy_design_id",
                design.legacy_design_id,
                design.legacy_design_id,
                "provenance_only_not_canonical_identity",
                status,
                source,
            ),
        )

    @staticmethod
    def _check_count(
        generator_id: str, fields: Mapping[str, Any], observed: int
    ) -> None:
        """用注册表声明的期望数量检测组合规则或数据的意外变化。"""

        expected = int(fields["expected_count"])
        if observed != expected:
            raise ValueError(
                f"{generator_id} generated {observed} values; expected {expected}"
            )


def _metadata(
    raw: Mapping[str, Any], name: str
) -> tuple[str, tuple[str, ...]]:
    """校验并提取一个证据节点共有的状态和非空来源列表。"""

    status = str(raw.get("evidence_status", ""))
    if status not in _EVIDENCE_STATUSES:
        raise ValueError(f"{name} has invalid evidence_status {status!r}")
    source_raw = raw.get("source")
    if not isinstance(source_raw, list) or not source_raw:
        raise ValueError(f"{name} requires a non-empty source list")
    source = tuple(str(value) for value in source_raw)
    if any(not value for value in source):
        raise ValueError(f"{name} source values cannot be empty")
    return status, source


def _evidence(raw: Any, name: str) -> EvidenceValue:
    """把 JSON 证据节点解析为不可变的 :class:`EvidenceValue`。"""

    if not isinstance(raw, Mapping) or "value" not in raw:
        raise TypeError(f"{name} must be an evidence value")
    status, source = _metadata(raw, name)
    return EvidenceValue(raw["value"], status, source)


def load_registry(path: str | Path | None = None) -> ParameterRegistry:
    """从指定 JSON 文件加载注册表；省略路径时使用随包发布的版本。"""

    registry_path = (
        Path(__file__).with_name("registry.json")
        if path is None
        else Path(path)
    )
    try:
        document = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load parameter registry {registry_path}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("parameter registry root must be an object")
    return ParameterRegistry(document)


__all__ = [
    "ArrayGeometry",
    "CanonicalDesign",
    "EvidenceValue",
    "ImportedLegacyDesign",
    "Installation",
    "LegacyDesign",
    "MigrationField",
    "MigrationReport",
    "ParameterRegistry",
    "Protocol",
    "REGISTRY_SCHEMA_VERSION",
    "SelectionSet",
    "SpinePackage",
    "TerminalPreset",
    "axis_from_pitch_yaw",
    "equal_height_length_m",
    "legacy_design_id",
    "load_registry",
]
