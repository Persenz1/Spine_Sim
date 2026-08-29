"""Canonical engineering parameter registry and legacy-input migration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from spine_sim.core.identity import identity


REGISTRY_SCHEMA_VERSION = "spine-parameter-registry-v1"

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
    value: Any
    evidence_status: str
    source: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Installation:
    """Canonical tagged union for rigid or unilateral-spring installation."""

    mode: str
    stiffness_N_per_m: float | None = None
    travel_m: float | None = None
    preload_N: float = 0.0
    tension_allowed: bool = False

    def __post_init__(self) -> None:
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
        return f"{self.nx}x{self.ny}"


@dataclass(frozen=True, slots=True)
class CanonicalDesign:
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
        return tuple(
            (index - 0.5 * (self.nx - 1)) * self.spacing_m
            for index in range(self.nx)
        )

    @property
    def axis_by_x_column(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            axis_from_pitch_yaw(pitch, self.yaw_rad)
            for pitch in self.pitch_by_x_column_rad
        )


@dataclass(frozen=True, slots=True)
class LegacyDesign:
    legacy_design_id: str
    legacy_mapping: Mapping[str, Any]
    canonical: CanonicalDesign
    evidence_status: str
    source: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Protocol:
    protocol_id: str
    evidence_status: str
    source: tuple[str, ...]
    fields: Mapping[str, EvidenceValue]
    registry_version: str

    @property
    def values(self) -> dict[str, Any]:
        return {name: field.value for name, field in self.fields.items()}

    @property
    def protocol_identity(self) -> str:
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
    selection_id: str
    role_ids: tuple[str, ...]
    legacy_design_ids: tuple[str, ...]
    evidence_status: str
    source: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MigrationField:
    legacy_field: str
    canonical_field: str
    legacy_value: Any
    canonical_value: Any
    conversion: str
    evidence_status: str
    source: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
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
    protocol_id: str
    legacy_design_id: str
    fields: tuple[MigrationField, ...]
    unexplained_differences: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "legacy_design_id": self.legacy_design_id,
            "fields": [field.as_dict() for field in self.fields],
            "unexplained_differences": list(self.unexplained_differences),
        }


@dataclass(frozen=True, slots=True)
class ImportedLegacyDesign:
    design: CanonicalDesign
    report: MigrationReport


def axis_from_pitch_yaw(
    pitch_rad: float, yaw_rad: float
) -> tuple[float, float, float]:
    if not math.isfinite(float(pitch_rad)) or not math.isfinite(float(yaw_rad)):
        raise ValueError("pitch and yaw must be finite")
    cos_pitch = math.cos(float(pitch_rad))
    return (
        cos_pitch * math.cos(float(yaw_rad)),
        cos_pitch * math.sin(float(yaw_rad)),
        math.sin(float(pitch_rad)),
    )


def equal_height_length_m(pitch_rad: float) -> float:
    """Length required to retain the legacy 4 mm at 80 degree height."""

    pitch = float(pitch_rad)
    if not math.isfinite(pitch):
        raise ValueError("equal-height pitch must be finite")
    denominator = abs(math.sin(pitch))
    if denominator <= 0.0:
        raise ValueError("equal-height pitch must have a nonzero sine")
    reference_height_m = 0.004 * math.sin(math.radians(80.0))
    return reference_height_m / denominator


def _columns(start: float, end: float, count: int) -> tuple[float, ...]:
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
    return float(round(-math.degrees(float(pitch_rad)), 12))


def _spring_family(stiffness_N_per_m: float | None) -> str:
    if stiffness_N_per_m is None:
        return "rigid"
    return "compliant" if stiffness_N_per_m <= 500.0 else "stiff"


def _legacy_package_mapping(
    tip_radius_m: float,
    rod_diameter_m: float,
    fixed_pitch_rad: float,
    installation: Installation,
) -> dict[str, Any]:
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
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def legacy_design_id(value: Mapping[str, Any]) -> str:
    """Reproduce the exact historical 20-hex M3 full-scan design ID."""

    package = _normalize_legacy_package(value["package"])
    geometry = _normalize_legacy_geometry(value["geometry"])
    return f"m3_full_design_{_legacy_digest({'package': package, 'geometry': geometry})}"


def _normalize_legacy_package(value: Any) -> dict[str, Any]:
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
    for derived in ("package_id", "spring_family"):
        if derived in value and value[derived] != normalized[derived]:
            raise ValueError(f"legacy package {derived} does not match its inputs")
    return normalized


def _normalize_legacy_geometry(value: Any) -> dict[str, Any]:
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
    for derived in ("array_shape", "geometry_id"):
        if derived in value and value[derived] != normalized[derived]:
            raise ValueError(f"legacy geometry {derived} does not match its inputs")
    return normalized


class ParameterRegistry:
    def __init__(self, document: Mapping[str, Any]):
        self._document = dict(document)
        if self._document.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise ValueError("unsupported parameter registry schema")
        self.registry_version = str(self._document.get("registry_version", ""))
        if not self.registry_version:
            raise ValueError("parameter registry version cannot be empty")
        self._validate_evidence()

    def _validate_evidence(self) -> None:
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
            section = self._mapping(section_name)
            for name, raw in section.items():
                _evidence(raw, f"{section_name}.{name}")
        for section_name in ("generators", "protocols"):
            section = self._mapping(section_name)
            for name, raw in section.items():
                if not isinstance(raw, Mapping):
                    raise TypeError(f"{section_name}.{name} must be a mapping")
                _metadata(raw, f"{section_name}.{name}")
                fields = raw.get("fields")
                if not isinstance(fields, Mapping) or not fields:
                    raise ValueError(f"{section_name}.{name}.fields cannot be empty")
                for field_name, field in fields.items():
                    _evidence(field, f"{section_name}.{name}.{field_name}")

    def _mapping(self, name: str) -> Mapping[str, Any]:
        value = self._document.get(name)
        if not isinstance(value, Mapping):
            raise TypeError(f"registry section {name!r} must be a mapping")
        return value

    def evidence(self, section: str, name: str) -> EvidenceValue:
        return _evidence(self._mapping(section)[name], f"{section}.{name}")

    def candidate_axis(self, name: str) -> EvidenceValue:
        """Return one axis without materializing any parameter combinations."""

        return self.evidence("candidate_axes", name)

    def protocol(self, protocol_id: str) -> Protocol:
        raw = self._mapping("protocols")[protocol_id]
        assert isinstance(raw, Mapping)
        status, source = _metadata(raw, f"protocols.{protocol_id}")
        fields = {
            name: _evidence(value, f"protocols.{protocol_id}.{name}")
            for name, value in raw["fields"].items()
        }
        return Protocol(
            protocol_id,
            status,
            source,
            fields,
            self.registry_version,
        )

    def paired_seed_set(self, seed_set_id: str) -> tuple[int, ...]:
        raw = self.evidence("paired_seed_sets", seed_set_id).value
        return tuple(int(seed) for seed in raw)

    def paired_seeds_for_protocol(self, protocol_id: str) -> tuple[int, ...]:
        protocol = self.protocol(protocol_id)
        seed_set = protocol.fields.get("paired_seed_set")
        return () if seed_set is None else self.paired_seed_set(str(seed_set.value))

    def generate_source_defined_m3a(self) -> tuple[SpinePackage, ...]:
        raw, status, source = self._generator("source_defined_m3a")
        fields = _field_values(raw)
        fixed_length_m = float(self.candidate_axis("fixed_length_m").value)
        yaw_rad = float(self.candidate_axis("yaw_rad").value[0])
        packages: list[SpinePackage] = []
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
        self._check_count("source_defined_m3a", raw, len(packages))
        return tuple(packages)

    def generate_source_defined_m3b(self) -> tuple[ArrayGeometry, ...]:
        raw, status, source = self._generator("source_defined_m3b")
        fields = _field_values(raw)
        geometries: list[ArrayGeometry] = []
        for nx, ny in fields["fixed_shapes"]:
            for spacing in fields["fixed_spacing_m"]:
                geometries.append(
                    self._geometry(
                        int(nx), int(ny), float(spacing), "fixed", None, status, source
                    )
                )
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
        self._check_count("source_defined_m3b", raw, len(geometries))
        return tuple(geometries)

    def generate_legacy_full_scan(self) -> tuple[LegacyDesign, ...]:
        raw, status, source = self._generator("legacy_full_scan")
        fields = _field_values(raw)
        designs: list[LegacyDesign] = []
        fixed_length_m = float(self.candidate_axis("fixed_length_m").value)
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
        self._check_count("legacy_full_scan", raw, len(designs))
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
        package = _normalize_legacy_package(value["package"])
        geometry = _normalize_legacy_geometry(value["geometry"])
        computed_id = legacy_design_id({"package": package, "geometry": geometry})
        supplied_id = value.get("design_id")
        if supplied_id is not None and supplied_id != computed_id:
            raise ValueError("legacy design_id does not match its physical payload")
        protocol = self.protocol(protocol_id)
        migration_source = protocol.source
        if protocol_id == "legacy_terminal_archive":
            migration_source = (
                *migration_source,
                "docs/archive/legacy_simulation_evidence/manifests/"
                "terminal_input_selected_designs.json",
            )
        pitch_deg = float(package["fixed_angle_deg"])
        pattern = str(geometry["angle_pattern"])
        nx = int(geometry["nx"])
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
        by_id = {
            design.legacy_design_id: design
            for design in self.generate_legacy_full_scan()
        }
        presets: list[TerminalPreset] = []
        for role_id, raw in self._mapping("terminal_presets").items():
            evidence = _evidence(raw, f"terminal_presets.{role_id}")
            value = evidence.value
            if not isinstance(value, Mapping):
                raise TypeError(f"terminal preset {role_id} must contain a mapping")
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

    def _generator(
        self, generator_id: str
    ) -> tuple[Mapping[str, Any], str, tuple[str, ...]]:
        raw = self._mapping("generators")[generator_id]
        assert isinstance(raw, Mapping)
        status, source = _metadata(raw, f"generators.{generator_id}")
        return raw, status, source

    def _installation(self, stiffness: Any) -> Installation:
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
        package = _legacy_package_mapping(
            radius_m, diameter_m, package_pitch_rad, installation
        )
        geometry = _legacy_geometry_mapping(nx, ny, spacing_m, pattern)
        mapping: dict[str, Any] = {"package": package, "geometry": geometry}
        design_id = legacy_design_id(mapping)
        mapping["design_id"] = design_id
        if endpoints is None:
            pitches = (package_pitch_rad,) * nx
            lengths = (fixed_length_m,) * nx
        else:
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
        generator_id: str, raw: Mapping[str, Any], observed: int
    ) -> None:
        expected = int(_field_values(raw)["expected_count"])
        if observed != expected:
            raise ValueError(
                f"{generator_id} generated {observed} values; expected {expected}"
            )


def _metadata(
    raw: Mapping[str, Any], name: str
) -> tuple[str, tuple[str, ...]]:
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
    if not isinstance(raw, Mapping) or "value" not in raw:
        raise TypeError(f"{name} must be an evidence value")
    status, source = _metadata(raw, name)
    return EvidenceValue(raw["value"], status, source)


def _field_values(raw: Mapping[str, Any]) -> dict[str, Any]:
    fields = raw["fields"]
    assert isinstance(fields, Mapping)
    return {
        name: _evidence(value, name).value for name, value in fields.items()
    }


def load_registry(path: str | Path | None = None) -> ParameterRegistry:
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


def generate_source_defined_m3a() -> tuple[SpinePackage, ...]:
    return load_registry().generate_source_defined_m3a()


def generate_source_defined_m3b() -> tuple[ArrayGeometry, ...]:
    return load_registry().generate_source_defined_m3b()


def generate_legacy_full_scan() -> tuple[LegacyDesign, ...]:
    return load_registry().generate_legacy_full_scan()


def import_legacy_design(
    value: Mapping[str, Any], *, protocol_id: str = "legacy_full_scan"
) -> ImportedLegacyDesign:
    return load_registry().import_legacy_design(value, protocol_id=protocol_id)


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
    "generate_legacy_full_scan",
    "generate_source_defined_m3a",
    "generate_source_defined_m3b",
    "import_legacy_design",
    "legacy_design_id",
    "load_registry",
]
