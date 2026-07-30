"""M3-fast case generation, grouped execution, Parquet summaries and ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .model import SpineBatch, build_spine_batch
from .solver import (
    MAX_STATION_EVALUATIONS,
    PathSettings,
    PathTrace,
    simulate_path,
)
from .terrain import (
    TerrainCondition,
    load_catalog,
    load_track_bank,
    select_conditions,
)


SUMMARY_FIELDS = (
    "case_id",
    "terrain_id",
    "seed",
    "array_shape",
    "spacing",
    "angle_pattern",
    "spine_package",
    "preload",
    "path_length",
    "completion_ratio",
    "traversal_attempt_ratio",
    "path_end_attempted",
    "initial_preload_established",
    "Fx_q10",
    "Fx_median",
    "Fx_peak_qs",
    "contact_ratio",
    "Neff_q10",
    "Neff_median",
    "max_load_share_q90",
    "slide_ratio",
    "hard_stop_ratio",
    "recontact_count",
    "recontact_ratio",
    "detach_count",
    "landing_change_count",
    "landing_change_ratio",
    "max_abs_landing_offset_m",
    "unsupported_station_count",
    "unsupported_station_ratio",
    "track_invalid_station_count",
    "numerical_failure_station_count",
    "preload_unreachable_station_count",
    "support_loss_position",
    "case_status",
)

DEFAULT_CATALOG = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "m1_material_formal_300"
    / "terrain_catalog.json"
)
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[3] / "results" / "m3_fast"
)
FULL_SCAN_SOLVER_SEMANTICS = "constant-preload-reseat-v3"


@dataclass(frozen=True)
class SpinePackage:
    tip_radius_m: float
    diameter_m: float
    fixed_angle_deg: float
    spring_stiffness_N_per_m: float | None

    @property
    def package_id(self) -> str:
        radius_um = int(round(self.tip_radius_m * 1e6))
        diameter_um = int(round(self.diameter_m * 1e6))
        angle_deg = int(round(self.fixed_angle_deg))
        spring = (
            "rigid"
            if self.spring_stiffness_N_per_m is None
            else f"k{int(round(self.spring_stiffness_N_per_m))}"
        )
        return (
            f"rt{radius_um}um_d{diameter_um}um_"
            f"a{angle_deg}deg_{spring}"
        )

    @property
    def spring_family(self) -> str:
        if self.spring_stiffness_N_per_m is None:
            return "rigid"
        if self.spring_stiffness_N_per_m <= 500.0:
            return "compliant"
        return "stiff"

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["package_id"] = self.package_id
        value["spring_family"] = self.spring_family
        return value

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SpinePackage":
        return cls(
            tip_radius_m=float(value["tip_radius_m"]),
            diameter_m=float(value["diameter_m"]),
            fixed_angle_deg=float(value["fixed_angle_deg"]),
            spring_stiffness_N_per_m=(
                None
                if value.get("spring_stiffness_N_per_m") is None
                else float(value["spring_stiffness_N_per_m"])
            ),
        )


@dataclass(frozen=True)
class ArrayGeometry:
    nx: int
    ny: int
    spacing_m: float
    angle_pattern: str = "fixed"

    @property
    def array_shape(self) -> str:
        return f"{self.nx}x{self.ny}"

    @property
    def geometry_id(self) -> str:
        spacing_mm = int(round(self.spacing_m * 1e3))
        return (
            f"{self.array_shape}_s{spacing_mm}mm_{self.angle_pattern}"
        )

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["array_shape"] = self.array_shape
        value["geometry_id"] = self.geometry_id
        return value

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ArrayGeometry":
        return cls(
            nx=int(value["nx"]),
            ny=int(value["ny"]),
            spacing_m=float(value["spacing_m"]),
            angle_pattern=str(value.get("angle_pattern", "fixed")),
        )


@dataclass(frozen=True)
class ModelSettings:
    young_modulus_Pa: float = 200e9
    poisson_ratio: float = 0.29
    shear_correction: float = 6.0 / 7.0
    static_friction: float = 0.45
    kinetic_friction: float = 0.35
    spring_delta_max_m: float = 0.004

    def __post_init__(self) -> None:
        if not math.isfinite(self.young_modulus_Pa) or self.young_modulus_Pa <= 0:
            raise ValueError("young_modulus_Pa must be positive and finite")
        if not math.isfinite(self.poisson_ratio) or not -0.99 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie in (-0.99, 0.5)")
        if not math.isfinite(self.shear_correction) or self.shear_correction <= 0:
            raise ValueError("shear_correction must be positive and finite")
        if (
            not math.isfinite(self.static_friction)
            or not math.isfinite(self.kinetic_friction)
            or self.static_friction < 0.0
            or self.kinetic_friction < 0.0
            or self.kinetic_friction > self.static_friction
        ):
            raise ValueError(
                "friction coefficients require 0 <= kinetic <= static"
            )
        if (
            not math.isfinite(self.spring_delta_max_m)
            or self.spring_delta_max_m <= 0.0
        ):
            raise ValueError("spring_delta_max_m must be positive and finite")


@dataclass(frozen=True)
class FullScanDesign:
    """One hardware/geometry point shared by all three preload cases."""

    package: SpinePackage
    geometry: ArrayGeometry

    @property
    def design_id(self) -> str:
        return _stable_id(
            "m3_full_design",
            {
                "package": self.package.to_mapping(),
                "geometry": self.geometry.to_mapping(),
            },
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "design_id": self.design_id,
            "package": self.package.to_mapping(),
            "geometry": self.geometry.to_mapping(),
        }


FULL_SCAN_SHAPES = (
    (2, 2),
    (2, 5),
    (5, 2),
    (3, 5),
    (5, 3),
    (4, 4),
    (6, 6),
)
FULL_SCAN_PRELOADS_N = (0.5, 1.0, 2.0)
FULL_SCAN_COARSE_SEEDS = tuple(
    list(range(41001, 41006))
    + list(range(41051, 41056))
    + list(range(41096, 41101))
)
FULL_SCAN_FINE_SEEDS = tuple(
    seed
    for start in (
        41001,
        41011,
        41021,
        41031,
        41041,
        41051,
        41061,
        41071,
        41081,
        41096,
    )
    for seed in range(start, start + 5)
)
FULL_SCAN_FINAL_SEEDS = tuple(range(41001, 41101))


def generate_full_scan_designs() -> list[FullScanDesign]:
    """Generate the frozen seven-shape, 1,344-point hardware design."""

    designs: list[FullScanDesign] = []
    for nx, ny in FULL_SCAN_SHAPES:
        for spacing_m in (0.004, 0.005, 0.006):
            for radius_m in (50e-6, 100e-6):
                for diameter_m in (0.6e-3, 0.8e-3):
                    for spring_stiffness in (300.0, 800.0, 2000.0, None):
                        for fixed_angle_deg in (60.0, 70.0, 80.0):
                            designs.append(
                                FullScanDesign(
                                    SpinePackage(
                                        radius_m,
                                        diameter_m,
                                        fixed_angle_deg,
                                        spring_stiffness,
                                    ),
                                    ArrayGeometry(
                                        nx,
                                        ny,
                                        spacing_m,
                                        "fixed",
                                    ),
                                )
                            )
                        designs.append(
                            FullScanDesign(
                                SpinePackage(
                                    radius_m,
                                    diameter_m,
                                    60.0,
                                    spring_stiffness,
                                ),
                                ArrayGeometry(
                                    nx,
                                    ny,
                                    spacing_m,
                                    "60_to_80",
                                ),
                            )
                        )
    if len(designs) != 1344:
        raise AssertionError(
            "full scan must contain exactly 1,344 mechanical designs"
        )
    if len({item.design_id for item in designs}) != len(designs):
        raise AssertionError("full scan design identities must be unique")
    return designs


def generate_m3a_packages() -> list[SpinePackage]:
    packages: list[SpinePackage] = []
    for radius_m in (50e-6, 100e-6):
        for diameter_m in (0.6e-3, 0.8e-3):
            for angle_deg in (50.0, 60.0, 70.0, 80.0):
                for spring_stiffness in (None, 200.0, 500.0, 1000.0, 2000.0):
                    packages.append(
                        SpinePackage(
                            tip_radius_m=radius_m,
                            diameter_m=diameter_m,
                            fixed_angle_deg=angle_deg,
                            spring_stiffness_N_per_m=spring_stiffness,
                        )
                    )
    if len(packages) != 80:
        raise AssertionError("M3-A design must contain exactly 80 packages")
    return packages


def generate_m3b_geometries() -> list[ArrayGeometry]:
    shapes = ((2, 2), (2, 5), (5, 2), (3, 5), (5, 3), (4, 4), (6, 6))
    geometries = [
        ArrayGeometry(nx, ny, spacing_m, "fixed")
        for nx, ny in shapes
        for spacing_m in (0.004, 0.005, 0.006)
    ]
    for nx, ny in shapes[1:]:
        geometries.append(ArrayGeometry(nx, ny, 0.005, "80_to_60"))
    geometries.append(ArrayGeometry(4, 4, 0.005, "80_to_50"))
    if len(geometries) != 28:
        raise AssertionError("M3-B coverage table must contain 28 geometries")
    return geometries


def _array_y_values(geometry: ArrayGeometry) -> np.ndarray:
    return (
        np.arange(geometry.ny, dtype=np.float64)
        - 0.5 * (geometry.ny - 1)
    ) * geometry.spacing_m


def _stable_id(prefix: str, value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:20]}"


def _build_batch(
    package: SpinePackage,
    geometry: ArrayGeometry,
    model: ModelSettings,
) -> SpineBatch:
    return build_spine_batch(
        geometry.nx,
        geometry.ny,
        geometry.spacing_m,
        angle_pattern=geometry.angle_pattern,
        fixed_angle_deg=package.fixed_angle_deg,
        tip_radius_m=package.tip_radius_m,
        diameter_m=package.diameter_m,
        spring_stiffness_N_per_m=package.spring_stiffness_N_per_m,
        spring_delta_max_m=model.spring_delta_max_m,
        young_modulus_Pa=model.young_modulus_Pa,
        poisson_ratio=model.poisson_ratio,
        shear_correction=model.shear_correction,
        static_friction=model.static_friction,
        kinetic_friction=model.kinetic_friction,
    )


def _case_summary(
    condition: TerrainCondition,
    package: SpinePackage,
    geometry: ArrayGeometry,
    path: PathSettings,
    model: ModelSettings,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    identity = {
        "terrain_id": condition.terrain_id,
        "region_id": condition.region_id,
        "seed": condition.seed,
        "package": package.to_mapping(),
        "geometry": geometry.to_mapping(),
        "path": asdict(path),
        "model": asdict(model),
    }
    summary: dict[str, Any] = {
        "case_id": _stable_id("m3_case", identity),
        "terrain_id": condition.terrain_id,
        "seed": int(condition.seed),
        "array_shape": geometry.array_shape,
        "spacing": geometry.spacing_m,
        "angle_pattern": geometry.angle_pattern,
        "spine_package": package.package_id,
        "preload": path.preload_N,
        "path_length": path.path_length_m,
    }
    for name in SUMMARY_FIELDS[9:]:
        summary[name] = metrics[name]
    return summary


def _load_banks(
    library_root: str,
    condition: TerrainCondition,
    packages: Sequence[SpinePackage],
    y_values_m: Iterable[float],
) -> dict[float, Any]:
    unique_y = np.unique(np.asarray(tuple(y_values_m), dtype=np.float64))
    banks: dict[float, Any] = {}
    for radius_m in sorted({package.tip_radius_m for package in packages}):
        banks[radius_m] = load_track_bank(
            library_root,
            condition,
            radius_m=radius_m,
            y_values_m=unique_y,
            verify_hash=False,
        )
    return banks


def _m3a_condition_worker(
    payload: tuple[
        str,
        TerrainCondition,
        tuple[SpinePackage, ...],
        PathSettings,
        ModelSettings,
    ],
) -> tuple[list[dict[str, Any]], int]:
    library_root, condition, packages, path, model = payload
    geometry = ArrayGeometry(4, 4, 0.005, "fixed")
    y_values = _array_y_values(geometry)
    banks = _load_banks(library_root, condition, packages, y_values)
    rows: list[dict[str, Any]] = []
    maximum_evaluations = 0
    for package in packages:
        batch = _build_batch(package, geometry, model)
        bank = banks[package.tip_radius_m]
        track_rows = bank.rows_for_y(batch.y_m)
        metrics, diagnostics = simulate_path(batch, bank, track_rows, path)
        maximum_evaluations = max(
            maximum_evaluations,
            int(diagnostics["max_station_evaluations"]),
        )
        rows.append(
            _case_summary(
                condition,
                package,
                geometry,
                path,
                model,
                metrics,
            )
        )
    return rows, maximum_evaluations


def _m3b_condition_worker(
    payload: tuple[
        str,
        TerrainCondition,
        tuple[SpinePackage, ...],
        tuple[ArrayGeometry, ...],
        PathSettings,
        ModelSettings,
    ],
) -> tuple[list[dict[str, Any]], int]:
    library_root, condition, packages, geometries, path, model = payload
    all_y_values = np.unique(
        np.concatenate([_array_y_values(item) for item in geometries])
    )
    banks = _load_banks(library_root, condition, packages, all_y_values)
    rows: list[dict[str, Any]] = []
    maximum_evaluations = 0
    for package in packages:
        bank = banks[package.tip_radius_m]
        for geometry in geometries:
            batch = _build_batch(package, geometry, model)
            track_rows = bank.rows_for_y(batch.y_m)
            metrics, diagnostics = simulate_path(batch, bank, track_rows, path)
            maximum_evaluations = max(
                maximum_evaluations,
                int(diagnostics["max_station_evaluations"]),
            )
            rows.append(
                _case_summary(
                    condition,
                    package,
                    geometry,
                    path,
                    model,
                    metrics,
                )
            )
    return rows, maximum_evaluations


def _run_grouped(
    worker: Callable[[Any], tuple[list[dict[str, Any]], int]],
    payloads: Sequence[Any],
    workers: int,
) -> tuple[list[dict[str, Any]], int]:
    if not payloads:
        return [], 0
    maximum_evaluations = 0
    all_rows: list[dict[str, Any]] = []
    resolved_workers = max(1, min(int(workers), len(payloads)))
    if resolved_workers == 1:
        for payload in payloads:
            rows, group_maximum = worker(payload)
            all_rows.extend(rows)
            maximum_evaluations = max(maximum_evaluations, group_maximum)
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            futures = [executor.submit(worker, payload) for payload in payloads]
            for future in as_completed(futures):
                rows, group_maximum = future.result()
                all_rows.extend(rows)
                maximum_evaluations = max(
                    maximum_evaluations, group_maximum
                )
    all_rows.sort(key=lambda row: row["case_id"])
    return all_rows, maximum_evaluations


def _finite_quantile(values: Iterable[Any], q: float, fallback: float) -> float:
    numeric = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if numeric.size == 0:
        return fallback
    return float(np.quantile(numeric, q))


def _aggregate_rankings(
    rows: Sequence[dict[str, Any]],
    packages: Sequence[SpinePackage],
    *,
    include_geometry: bool,
) -> list[dict[str, Any]]:
    package_by_id = {item.package_id: item for item in packages}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if include_geometry:
            key = (
                row["spine_package"],
                row["array_shape"],
                row["spacing"],
                row["angle_pattern"],
            )
        else:
            key = (row["spine_package"],)
        groups.setdefault(key, []).append(row)

    ranking: list[dict[str, Any]] = []
    for key, cases in groups.items():
        package = package_by_id[str(key[0])]
        support_loss_rate = sum(
            case["case_status"] != "complete" for case in cases
        ) / len(cases)
        completion_q10 = _finite_quantile(
            (case["completion_ratio"] for case in cases), 0.10, 0.0
        )
        hard_stop_q90 = _finite_quantile(
            (case["hard_stop_ratio"] for case in cases), 0.90, 1.0
        )
        record: dict[str, Any] = {
            "spine_package": package.package_id,
            "tip_radius_um": package.tip_radius_m * 1e6,
            "diameter_mm": package.diameter_m * 1e3,
            "fixed_angle_deg": package.fixed_angle_deg,
            "spring_stiffness_N_per_m": package.spring_stiffness_N_per_m,
            "spring_family": package.spring_family,
            "paired_seed_count": len(cases),
            "completion_q10": completion_q10,
            "support_lost_rate": support_loss_rate,
            "Fx_cross_seed_q10": _finite_quantile(
                (case["Fx_q10"] for case in cases), 0.10, -math.inf
            ),
            "Fx_cross_seed_median": _finite_quantile(
                (case["Fx_median"] for case in cases), 0.50, -math.inf
            ),
            "Neff_cross_seed_q10": _finite_quantile(
                (case["Neff_q10"] for case in cases), 0.10, -math.inf
            ),
            "max_load_share_cross_seed_q90": _finite_quantile(
                (case["max_load_share_q90"] for case in cases),
                0.90,
                math.inf,
            ),
            "hard_stop_cross_seed_q90": hard_stop_q90,
            "slide_cross_seed_median": _finite_quantile(
                (case["slide_ratio"] for case in cases), 0.50, math.inf
            ),
        }
        if include_geometry:
            record.update(
                {
                    "array_shape": key[1],
                    "spacing": key[2],
                    "angle_pattern": key[3],
                }
            )
            record["configuration_id"] = _stable_id(
                "m3_config",
                {
                    "spine_package": key[0],
                    "array_shape": key[1],
                    "spacing": key[2],
                    "angle_pattern": key[3],
                },
            )
        record["eligible"] = bool(
            completion_q10 >= 0.90
            and support_loss_rate <= (1.0 / len(cases))
            and hard_stop_q90 <= 0.35
        )
        ranking.append(record)

    ranking.sort(
        key=lambda item: (
            not item["eligible"],
            -item["completion_q10"],
            -item["Fx_cross_seed_q10"],
            -item["Neff_cross_seed_q10"],
            item["max_load_share_cross_seed_q90"],
            item["hard_stop_cross_seed_q90"],
            item["spine_package"],
            item.get("array_shape", ""),
            item.get("spacing", 0.0),
            item.get("angle_pattern", ""),
        )
    )
    for rank, record in enumerate(ranking, start=1):
        record["rank"] = rank
    return ranking


def _select_m3a_packages(
    ranking: Sequence[dict[str, Any]],
    packages: Sequence[SpinePackage],
) -> tuple[list[SpinePackage], list[SpinePackage], list[str]]:
    package_by_id = {item.package_id: item for item in packages}
    ordered = [
        package_by_id[item["spine_package"]]
        for item in ranking
        if item["eligible"] and item["spine_package"] in package_by_id
    ]
    if len(ordered) < 4:
        raise RuntimeError(
            "M3-A eligibility gate retained fewer than four packages"
        )
    selected: list[SpinePackage] = []

    def add_first(predicate: Callable[[SpinePackage], bool]) -> None:
        if any(predicate(package) for package in selected):
            return
        for package in ordered:
            if predicate(package) and package not in selected:
                selected.append(package)
                return

    if ordered:
        selected.append(ordered[0])
    add_first(lambda package: package.spring_family == "compliant")
    add_first(lambda package: package.spring_family in {"stiff", "rigid"})
    if selected:
        first_radius = selected[0].tip_radius_m
        add_first(lambda package: package.tip_radius_m != first_radius)
        first_diameter = selected[0].diameter_m
        add_first(lambda package: package.diameter_m != first_diameter)
        first_angle = selected[0].fixed_angle_deg
        add_first(lambda package: package.fixed_angle_deg != first_angle)
    for package in ordered:
        if package not in selected and len(selected) < 6:
            selected.append(package)
    retained = selected[:6]

    m3b: list[SpinePackage] = []
    for spring_family in ("compliant", "rigid", "stiff"):
        for package in retained:
            if (
                package.spring_family == spring_family
                and package not in m3b
                and len(m3b) < 4
            ):
                m3b.append(package)
                break
    for package in retained:
        if package not in m3b and len(m3b) < 4:
            m3b.append(package)
    available_families = {item.spring_family for item in ordered}
    unavailable_families = [
        family
        for family in ("compliant", "rigid", "stiff")
        if family not in available_families
    ]
    return retained, m3b, unavailable_families


def _select_m3b_configurations(
    ranking: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = [item for item in ranking if item["eligible"]]
    selected = list(eligible[:18])
    selected_ids = {item["configuration_id"] for item in selected}
    unavailable: list[dict[str, Any]] = []

    def add_best(field: str, value: Any) -> None:
        if any(item[field] == value for item in selected):
            return
        for item in eligible:
            if item[field] != value:
                continue
            if (
                len(selected) < 24
                and item["configuration_id"] not in selected_ids
            ):
                selected.append(item)
                selected_ids.add(item["configuration_id"])
                return
            return
        unavailable.append({"field": field, "value": value})

    for shape in ("2x2", "2x5", "5x2", "3x5", "5x3", "4x4", "6x6"):
        add_best("array_shape", shape)
    for pattern in ("fixed", "80_to_60", "80_to_50"):
        add_best("angle_pattern", pattern)
    for spacing in (0.004, 0.005, 0.006):
        add_best("spacing", spacing)
    for family in ("compliant", "stiff", "rigid"):
        add_best("spring_family", family)
    for item in eligible:
        if len(selected) >= 24:
            break
        if item["configuration_id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(item["configuration_id"])
    return selected, unavailable


def _summary_schema() -> Any:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow; install spine-sim[parquet]"
        ) from exc
    return pa.schema(
        [
            pa.field("case_id", pa.string(), nullable=False),
            pa.field("terrain_id", pa.string(), nullable=False),
            pa.field("seed", pa.int64(), nullable=False),
            pa.field("array_shape", pa.string(), nullable=False),
            pa.field("spacing", pa.float64(), nullable=False),
            pa.field("angle_pattern", pa.string(), nullable=False),
            pa.field("spine_package", pa.string(), nullable=False),
            pa.field("preload", pa.float64(), nullable=False),
            pa.field("path_length", pa.float64(), nullable=False),
            pa.field("completion_ratio", pa.float64(), nullable=False),
            pa.field(
                "traversal_attempt_ratio", pa.float64(), nullable=False
            ),
            pa.field("path_end_attempted", pa.bool_(), nullable=False),
            pa.field(
                "initial_preload_established", pa.bool_(), nullable=False
            ),
            pa.field("Fx_q10", pa.float64(), nullable=False),
            pa.field("Fx_median", pa.float64(), nullable=False),
            pa.field("Fx_peak_qs", pa.float64(), nullable=False),
            pa.field("contact_ratio", pa.float64(), nullable=False),
            pa.field("Neff_q10", pa.float64(), nullable=False),
            pa.field("Neff_median", pa.float64(), nullable=False),
            pa.field("max_load_share_q90", pa.float64(), nullable=False),
            pa.field("slide_ratio", pa.float64(), nullable=False),
            pa.field("hard_stop_ratio", pa.float64(), nullable=False),
            pa.field("recontact_count", pa.int64(), nullable=False),
            pa.field("recontact_ratio", pa.float64(), nullable=False),
            pa.field("detach_count", pa.int64(), nullable=False),
            pa.field("landing_change_count", pa.int64(), nullable=False),
            pa.field("landing_change_ratio", pa.float64(), nullable=False),
            pa.field(
                "max_abs_landing_offset_m", pa.float64(), nullable=False
            ),
            pa.field(
                "unsupported_station_count", pa.int64(), nullable=False
            ),
            pa.field(
                "unsupported_station_ratio", pa.float64(), nullable=False
            ),
            pa.field(
                "track_invalid_station_count", pa.int64(), nullable=False
            ),
            pa.field(
                "numerical_failure_station_count",
                pa.int64(),
                nullable=False,
            ),
            pa.field(
                "preload_unreachable_station_count",
                pa.int64(),
                nullable=False,
            ),
            pa.field("support_loss_position", pa.float64(), nullable=True),
            pa.field("case_status", pa.string(), nullable=False),
        ]
    )


def _write_parquet(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    summary_schema: bool = False,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow; install spine-sim[parquet]"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = _summary_schema() if summary_schema else None
    table = pa.Table.from_pylist(list(rows), schema=schema)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary_path = Path(temporary_name)
        if temporary_path.exists():
            temporary_path.unlink()


def _json_safe(value: Any) -> Any:
    """Replace non-finite ranking sentinels before strict JSON serialization."""

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _catalog_document(catalog_path: Path) -> dict[str, Any]:
    value = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("terrain catalog must contain a JSON object")
    return value


def _condition_mapping(condition: TerrainCondition) -> dict[str, Any]:
    value = asdict(condition)
    value["catalog_path"] = str(condition.catalog_path)
    value["library_root"] = str(condition.library_root)
    return value


def run_smoke_checks(
    catalog_path: str | Path = DEFAULT_CATALOG,
    output_dir: str | Path = DEFAULT_OUTPUT_ROOT / "smoke",
    *,
    model: ModelSettings = ModelSettings(),
    backplate_travel_m: float = 0.006,
) -> dict[str, Any]:
    started = time.perf_counter()
    catalog_path = Path(catalog_path).resolve()
    output_dir = Path(output_dir).resolve()
    catalog_document = _catalog_document(catalog_path)
    catalog = load_catalog(catalog_path)
    conditions = select_conditions(
        catalog,
        terrain_family="sandpaper",
        subtype="P240",
        seeds=(41005,),
    )
    if len(conditions) != 1:
        raise RuntimeError("P240 seed 41005 smoke terrain is unavailable")
    condition = conditions[0]
    library_root = condition.library_root
    packages = (
        SpinePackage(50e-6, 0.8e-3, 50.0, 2000.0),
        SpinePackage(50e-6, 0.8e-3, 50.0, 1000.0),
    )
    geometries = (
        ArrayGeometry(2, 2, 0.005, "fixed"),
        ArrayGeometry(6, 6, 0.005, "fixed"),
    )
    paths = (
        PathSettings(
            preload_N=1.0,
            path_length_m=0.002,
            dx_m=0.0001,
            backplate_travel_m=backplate_travel_m,
        ),
        PathSettings(
            preload_N=1.0,
            path_length_m=0.010,
            dx_m=0.0001,
            backplate_travel_m=backplate_travel_m,
        ),
    )
    y_values = np.unique(
        np.concatenate([_array_y_values(item) for item in geometries])
    )
    bank = load_track_bank(
        library_root,
        condition,
        radius_m=packages[0].tip_radius_m,
        y_values_m=y_values,
    )
    case_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    for package, geometry, path in zip(
        packages, geometries, paths, strict=True
    ):
        batch = _build_batch(package, geometry, model)
        metrics, diagnostics = simulate_path(
            batch,
            bank,
            bank.rows_for_y(batch.y_m),
            path,
        )
        summary = _case_summary(
            condition, package, geometry, path, model, metrics
        )
        finite = all(
            math.isfinite(float(summary[name]))
            for name in ("Fx_q10", "Fx_median", "Fx_peak_qs", "Neff_q10")
        )
        passed = bool(
            summary["case_status"] == "complete"
            and finite
            and diagnostics["max_station_evaluations"]
            <= MAX_STATION_EVALUATIONS
        )
        check_rows.append(
            {
                "name": (
                    "2x2_material_track_2mm"
                    if geometry.nx == 2
                    else "6x6_material_track_10mm"
                ),
                "passed": passed,
                "case_id": summary["case_id"],
                "case_status": summary["case_status"],
                "finite_summary": finite,
                "max_station_evaluations": diagnostics[
                    "max_station_evaluations"
                ],
                "station_count_completed": diagnostics[
                    "station_count_completed"
                ],
            }
        )
        case_rows.append(summary)
    _write_parquet(output_dir / "cases.parquet", case_rows, summary_schema=True)
    report = {
        "schema_version": "m3-fast-smoke-v1",
        "terrain_semantics": (
            "M1 formal P240 material-derived terrain driven by measured "
            "topography; not an unmodified full-size measured surface"
        ),
        "condition": _condition_mapping(condition),
        "checks": check_rows,
        "all_passed": all(item["passed"] for item in check_rows),
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(output_dir / "checks.json", report)
    if not report["all_passed"]:
        raise RuntimeError(f"M3 smoke checks failed: {check_rows}")
    return report


def run_m3a(
    catalog_path: str | Path = DEFAULT_CATALOG,
    output_dir: str | Path = DEFAULT_OUTPUT_ROOT / "m3a",
    *,
    material: str = "sandpaper",
    subtype: str = "P240",
    seeds: Sequence[int] = (41005, 41010, 41015, 41020, 41025, 41030),
    workers: int = 6,
    model: ModelSettings = ModelSettings(),
    backplate_travel_m: float = 0.006,
) -> dict[str, Any]:
    started = time.perf_counter()
    catalog_path = Path(catalog_path).resolve()
    output_dir = Path(output_dir).resolve()
    catalog_document = _catalog_document(catalog_path)
    catalog = load_catalog(catalog_path)
    conditions = select_conditions(
        catalog,
        terrain_family=material,
        subtype=subtype,
        seeds=tuple(int(seed) for seed in seeds),
    )
    if len(conditions) != 6:
        raise RuntimeError(
            f"M3-A requires exactly six paired terrain conditions, got {len(conditions)}"
        )
    library_root = conditions[0].library_root
    packages = tuple(generate_m3a_packages())
    path = PathSettings(
        preload_N=1.0,
        path_length_m=0.010,
        dx_m=0.0001,
        backplate_travel_m=backplate_travel_m,
    )
    payloads = [
        (str(library_root), condition, packages, path, model)
        for condition in conditions
    ]
    rows, maximum_evaluations = _run_grouped(
        _m3a_condition_worker, payloads, workers
    )
    if len(rows) != 480:
        raise RuntimeError(f"M3-A expected 480 cases, got {len(rows)}")
    ranking = _aggregate_rankings(rows, packages, include_geometry=False)
    retained, m3b_packages, unavailable_families = _select_m3a_packages(
        ranking, packages
    )
    _write_parquet(output_dir / "cases.parquet", rows, summary_schema=True)
    _write_parquet(output_dir / "package_ranking.parquet", ranking)
    selection = {
        "schema_version": "m3-fast-selected-packages-v1",
        "retained_4_to_6": [item.to_mapping() for item in retained],
        "m3b_packages_max_4": [item.to_mapping() for item in m3b_packages],
        "unavailable_after_eligibility_gate": [
            {"field": "spring_family", "value": value}
            for value in unavailable_families
        ],
        "selection_rule": (
            "eligibility gate first; then cross-seed Fx_q10 order, Neff "
            "tie-break, load-share/hard-stop checks, and only working "
            "spring/radius/diameter/angle controls"
        ),
    }
    _write_json(output_dir / "selected_packages.json", _json_safe(selection))
    manifest = {
        "schema_version": "m3-fast-campaign-v1",
        "stage": "M3-A",
        "catalog_path": str(catalog_path),
        "library_root": str(library_root),
        "terrain_material": material,
        "terrain_subtype": subtype,
        "paired_seeds": [condition.seed for condition in conditions],
        "case_count": len(rows),
        "package_count": len(packages),
        "path_settings": asdict(path),
        "model_settings": asdict(model),
        "max_station_evaluations_observed": maximum_evaluations,
        "formal_catalog_ranking_eligible_upstream": bool(
            catalog_document.get("formal_ranking_eligible", False)
        ),
        "upstream_limitations": catalog_document.get("limitations", []),
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {
        "manifest": manifest,
        "selection": selection,
        "top_ranked": ranking[:10],
    }


def run_m3b(
    selected_packages_path: str | Path,
    catalog_path: str | Path = DEFAULT_CATALOG,
    output_dir: str | Path = DEFAULT_OUTPUT_ROOT / "m3b",
    *,
    material: str = "sandpaper",
    subtype: str = "P240",
    seeds: Sequence[int] = (41005, 41010, 41015, 41020, 41025, 41030),
    workers: int = 6,
    model: ModelSettings = ModelSettings(),
    backplate_travel_m: float = 0.006,
) -> dict[str, Any]:
    started = time.perf_counter()
    catalog_path = Path(catalog_path).resolve()
    output_dir = Path(output_dir).resolve()
    selected_value = json.loads(
        Path(selected_packages_path).read_text(encoding="utf-8")
    )
    packages = tuple(
        SpinePackage.from_mapping(value)
        for value in selected_value["m3b_packages_max_4"]
    )
    if not 1 <= len(packages) <= 4:
        raise RuntimeError("M3-B requires one to four selected packages")
    geometries = tuple(generate_m3b_geometries())
    catalog_document = _catalog_document(catalog_path)
    catalog = load_catalog(catalog_path)
    conditions = select_conditions(
        catalog,
        terrain_family=material,
        subtype=subtype,
        seeds=tuple(int(seed) for seed in seeds),
    )
    if len(conditions) != 6:
        raise RuntimeError(
            f"M3-B requires exactly six paired terrain conditions, got {len(conditions)}"
        )
    library_root = conditions[0].library_root
    path = PathSettings(
        preload_N=1.0,
        path_length_m=0.010,
        dx_m=0.0001,
        backplate_travel_m=backplate_travel_m,
    )
    payloads = [
        (str(library_root), condition, packages, geometries, path, model)
        for condition in conditions
    ]
    rows, maximum_evaluations = _run_grouped(
        _m3b_condition_worker, payloads, workers
    )
    expected_cases = len(packages) * len(geometries) * len(conditions)
    if len(rows) != expected_cases or len(rows) > 672:
        raise RuntimeError(
            f"M3-B expected {expected_cases} cases within 672, got {len(rows)}"
        )
    ranking = _aggregate_rankings(rows, packages, include_geometry=True)
    selected, unavailable_categories = _select_m3b_configurations(ranking)
    _write_parquet(output_dir / "cases.parquet", rows, summary_schema=True)
    _write_parquet(output_dir / "configuration_ranking.parquet", ranking)
    selection = {
        "schema_version": "m3-fast-selected-configurations-v1",
        "selected_18_to_24": selected,
        "selection_count": len(selected),
        "unavailable_after_eligibility_gate": unavailable_categories,
        "selection_rule": (
            "eligibility gate first; then cross-seed Fx_q10 order, Neff "
            "tie-break, load-share/hard-stop checks, and only working "
            "shape/spacing/angle/spring mechanism controls"
        ),
    }
    _write_json(
        output_dir / "selected_configurations.json", _json_safe(selection)
    )
    manifest = {
        "schema_version": "m3-fast-campaign-v1",
        "stage": "M3-B",
        "catalog_path": str(catalog_path),
        "library_root": str(library_root),
        "terrain_material": material,
        "terrain_subtype": subtype,
        "paired_seeds": [condition.seed for condition in conditions],
        "case_count": len(rows),
        "package_count": len(packages),
        "geometry_count": len(geometries),
        "path_settings": asdict(path),
        "model_settings": asdict(model),
        "max_station_evaluations_observed": maximum_evaluations,
        "formal_catalog_ranking_eligible_upstream": bool(
            catalog_document.get("formal_ranking_eligible", False)
        ),
        "upstream_limitations": catalog_document.get("limitations", []),
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {
        "manifest": manifest,
        "selection": selection,
        "top_ranked": ranking[:10],
    }


def _terrain_stratum(condition: TerrainCondition) -> str:
    if condition.terrain_family == "sandpaper":
        return condition.subtype
    return condition.terrain_family


def _full_design_record(design: FullScanDesign) -> dict[str, Any]:
    package = design.package
    geometry = design.geometry
    return {
        "design_id": design.design_id,
        "array_shape": geometry.array_shape,
        "nx": geometry.nx,
        "ny": geometry.ny,
        "spine_count": geometry.nx * geometry.ny,
        "spacing_m": geometry.spacing_m,
        "angle_pattern": geometry.angle_pattern,
        "fixed_angle_deg": package.fixed_angle_deg,
        "tip_radius_m": package.tip_radius_m,
        "diameter_m": package.diameter_m,
        "spring_stiffness_N_per_m": package.spring_stiffness_N_per_m,
        "spring_family": package.spring_family,
        "spine_package": package.package_id,
    }


def _full_case_summary(
    condition: TerrainCondition,
    design: FullScanDesign,
    path: PathSettings,
    model: ModelSettings,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    row = _case_summary(
        condition,
        design.package,
        design.geometry,
        path,
        model,
        metrics,
    )
    row.update(
        {
            "design_id": design.design_id,
            "terrain_family": condition.terrain_family,
            "terrain_subtype": condition.subtype,
            "terrain_stratum": _terrain_stratum(condition),
            "tip_radius_m": design.package.tip_radius_m,
            "diameter_m": design.package.diameter_m,
            "fixed_angle_deg": design.package.fixed_angle_deg,
            "spring_stiffness_N_per_m": (
                design.package.spring_stiffness_N_per_m
            ),
            "spine_count": design.geometry.nx * design.geometry.ny,
            "max_station_evaluations": diagnostics[
                "max_station_evaluations"
            ],
            "max_station_total_evaluations": diagnostics[
                "max_station_total_evaluations"
            ],
            "max_station_attempts": diagnostics[
                "max_station_attempts"
            ],
            "station_count_completed": diagnostics[
                "station_count_completed"
            ],
        }
    )
    return row


def _allocate_trace_shard(
    case_count: int,
    station_count: int,
    maximum_spines: int,
    path_x_m: np.ndarray,
) -> dict[str, np.ndarray]:
    global_shape = (case_count, station_count)
    spine_shape = (case_count, station_count, maximum_spines)
    float_globals = (
        "backplate_z_m",
        "force_x_N",
        "force_z_N",
        "force_residual_N",
        "neff",
        "max_load_share",
        "stick_ratio",
        "slide_ratio",
        "hard_stop_ratio",
        "landing_offset_m",
    )
    float_spines = (
        "spine_force_x_N",
        "spine_force_z_N",
        "spine_lambda_n_N",
        "spine_tangent_force_N",
        "spine_u_t_history_m",
        "spine_spring_load_N",
        "spine_spring_displacement_m",
    )
    values: dict[str, np.ndarray] = {
        "schema_version": np.asarray("m3-full-path-v2"),
        "path_x_m": np.asarray(path_x_m, dtype=np.float64),
        "case_id": np.empty(case_count, dtype="<U40"),
        "design_id": np.empty(case_count, dtype="<U48"),
        "preload_N": np.empty(case_count, dtype=np.float64),
        "spine_count": np.empty(case_count, dtype=np.int16),
        "case_status": np.empty(case_count, dtype="<U24"),
        "completion_ratio": np.empty(case_count, dtype=np.float32),
        "accepted": np.zeros(global_shape, dtype=np.bool_),
        "station_status": np.full(global_shape, -1, dtype=np.int8),
        "root_evaluations": np.zeros(global_shape, dtype=np.int16),
        "solve_attempts": np.zeros(global_shape, dtype=np.int8),
        "recontacted": np.zeros(global_shape, dtype=np.bool_),
        "contact_count": np.zeros(global_shape, dtype=np.int16),
        "spine_mode": np.full(spine_shape, -1, dtype=np.int8),
        "spine_spring_branch": np.full(
            spine_shape, -1, dtype=np.int8
        ),
    }
    for name in float_globals:
        values[name] = np.full(global_shape, np.nan, dtype=np.float32)
    for name in float_spines:
        values[name] = np.full(spine_shape, np.nan, dtype=np.float32)
    return values


def _copy_trace_to_shard(
    shard: dict[str, np.ndarray],
    case_index: int,
    trace: PathTrace,
    *,
    case_id: str,
    design: FullScanDesign,
    preload_N: float,
    metrics: dict[str, Any],
) -> None:
    spine_count = design.geometry.nx * design.geometry.ny
    shard["case_id"][case_index] = case_id
    shard["design_id"][case_index] = design.design_id
    shard["preload_N"][case_index] = preload_N
    shard["spine_count"][case_index] = spine_count
    shard["case_status"][case_index] = metrics["case_status"]
    shard["completion_ratio"][case_index] = metrics["completion_ratio"]
    for name in (
        "accepted",
        "station_status",
        "root_evaluations",
        "solve_attempts",
        "recontacted",
        "landing_offset_m",
        "backplate_z_m",
        "force_x_N",
        "force_z_N",
        "force_residual_N",
        "contact_count",
        "neff",
        "max_load_share",
        "stick_ratio",
        "slide_ratio",
        "hard_stop_ratio",
    ):
        shard[name][case_index] = getattr(trace, name)
    for name in (
        "spine_force_x_N",
        "spine_force_z_N",
        "spine_lambda_n_N",
        "spine_tangent_force_N",
        "spine_mode",
        "spine_u_t_history_m",
        "spine_spring_branch",
        "spine_spring_load_N",
        "spine_spring_displacement_m",
    ):
        source = getattr(trace, name)
        if source is None:
            raise RuntimeError("fine/final trace is missing per-spine data")
        shard[name][case_index, :, :spine_count] = source


def _write_npz_atomic(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary_path = Path(temporary_name)
        if temporary_path.exists():
            temporary_path.unlink()


def _full_payload_id(
    stage: str,
    condition: TerrainCondition,
    designs: Sequence[FullScanDesign],
    path: PathSettings,
    model: ModelSettings,
    capture_paths: bool,
) -> str:
    return _stable_id(
        "m3_full_payload",
        {
            "stage": stage,
            "terrain_id": condition.terrain_id,
            "design_ids": [item.design_id for item in designs],
            "preloads_N": list(FULL_SCAN_PRELOADS_N),
            "path": asdict(path),
            "model": asdict(model),
            "capture_paths": capture_paths,
            "solver_semantics": FULL_SCAN_SOLVER_SEMANTICS,
        },
    )


def _full_condition_worker(
    payload: tuple[
        str,
        str,
        TerrainCondition,
        tuple[FullScanDesign, ...],
        PathSettings,
        ModelSettings,
        bool,
    ],
) -> dict[str, Any]:
    (
        stage,
        output_dir_text,
        condition,
        designs,
        base_path,
        model,
        capture_paths,
    ) = payload
    output_dir = Path(output_dir_text)
    summary_path = (
        output_dir / "summaries" / f"{condition.terrain_id}.parquet"
    )
    trace_path = output_dir / "paths" / f"{condition.terrain_id}.npz"
    marker_path = output_dir / "complete" / f"{condition.terrain_id}.json"
    payload_id = _full_payload_id(
        stage,
        condition,
        designs,
        base_path,
        model,
        capture_paths,
    )
    if marker_path.is_file() and summary_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        trace_ready = not capture_paths or trace_path.is_file()
        if marker.get("payload_id") == payload_id and trace_ready:
            return {
                "terrain_id": condition.terrain_id,
                "case_count": int(marker["case_count"]),
                "max_station_evaluations": int(
                    marker["max_station_evaluations"]
                ),
                "max_station_total_evaluations": int(
                    marker["max_station_total_evaluations"]
                ),
                "max_station_attempts": int(
                    marker["max_station_attempts"]
                ),
                "resumed": True,
            }

    all_y_values = np.unique(
        np.concatenate(
            [_array_y_values(item.geometry) for item in designs]
        )
    )
    packages = tuple(item.package for item in designs)
    banks = _load_banks(
        str(condition.library_root),
        condition,
        packages,
        all_y_values,
    )
    rows: list[dict[str, Any]] = []
    maximum_evaluations = 0
    maximum_total_evaluations = 0
    maximum_attempts = 0
    case_count = len(designs) * len(FULL_SCAN_PRELOADS_N)
    trace_shard: dict[str, np.ndarray] | None = None
    if capture_paths:
        example_batch = _build_batch(designs[0].package, designs[0].geometry, model)
        example_settings = replace(
            base_path, preload_N=FULL_SCAN_PRELOADS_N[0]
        )
        example_trace = PathTrace.allocate(
            example_batch,
            example_settings,
            include_spines=True,
        )
        maximum_spines = max(
            item.geometry.nx * item.geometry.ny for item in designs
        )
        trace_shard = _allocate_trace_shard(
            case_count,
            example_settings.station_count + 1,
            maximum_spines,
            example_trace.path_x_m,
        )

    case_index = 0
    for design in designs:
        batch = _build_batch(design.package, design.geometry, model)
        bank = banks[design.package.tip_radius_m]
        track_rows = bank.rows_for_y(batch.y_m)
        for preload_N in FULL_SCAN_PRELOADS_N:
            path = replace(base_path, preload_N=preload_N)
            trace = (
                PathTrace.allocate(batch, path, include_spines=True)
                if capture_paths
                else None
            )
            metrics, diagnostics = simulate_path(
                batch,
                bank,
                track_rows,
                path,
                trace=trace,
            )
            maximum_evaluations = max(
                maximum_evaluations,
                int(diagnostics["max_station_evaluations"]),
            )
            maximum_total_evaluations = max(
                maximum_total_evaluations,
                int(diagnostics["max_station_total_evaluations"]),
            )
            maximum_attempts = max(
                maximum_attempts,
                int(diagnostics["max_station_attempts"]),
            )
            row = _full_case_summary(
                condition,
                design,
                path,
                model,
                metrics,
                diagnostics,
            )
            rows.append(row)
            if capture_paths:
                assert trace is not None
                assert trace_shard is not None
                _copy_trace_to_shard(
                    trace_shard,
                    case_index,
                    trace,
                    case_id=row["case_id"],
                    design=design,
                    preload_N=preload_N,
                    metrics=metrics,
                )
            case_index += 1

    _write_parquet(summary_path, rows)
    if trace_shard is not None:
        trace_shard["terrain_id"] = np.asarray(condition.terrain_id)
        trace_shard["terrain_stratum"] = np.asarray(
            _terrain_stratum(condition)
        )
        _write_npz_atomic(trace_path, trace_shard)
    marker = {
        "schema_version": "m3-full-condition-complete-v1",
        "payload_id": payload_id,
        "terrain_id": condition.terrain_id,
        "case_count": len(rows),
        "max_station_evaluations": maximum_evaluations,
        "max_station_total_evaluations": maximum_total_evaluations,
        "max_station_attempts": maximum_attempts,
        "summary_path": str(summary_path),
        "trace_path": str(trace_path) if capture_paths else None,
    }
    _write_json(marker_path, marker)
    return {
        "terrain_id": condition.terrain_id,
        "case_count": len(rows),
        "max_station_evaluations": maximum_evaluations,
        "max_station_total_evaluations": maximum_total_evaluations,
        "max_station_attempts": maximum_attempts,
        "resumed": False,
    }


def _run_full_workers(
    payloads: Sequence[Any],
    workers: int,
) -> list[dict[str, Any]]:
    resolved_workers = max(1, min(int(workers), len(payloads)))
    results: list[dict[str, Any]] = []
    if resolved_workers == 1:
        for payload in payloads:
            results.append(_full_condition_worker(payload))
    else:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            futures = [
                executor.submit(_full_condition_worker, payload)
                for payload in payloads
            ]
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda item: item["terrain_id"])
    return results


def _read_full_summary_rows(summary_dir: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "full scan aggregation requires pyarrow"
        ) from exc
    rows: list[dict[str, Any]] = []
    for path in sorted(summary_dir.glob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _rank_full_designs(
    rows: Sequence[dict[str, Any]],
    designs: Sequence[FullScanDesign],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    rows_by_design: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_design.setdefault(str(row["design_id"]), []).append(row)
    if stage == "coarse":
        completion_gate = 0.80
        support_loss_gate = 1.0 / 3.0
    elif stage == "fine":
        completion_gate = 0.90
        support_loss_gate = 0.20
    else:
        completion_gate = 0.90
        support_loss_gate = 0.10

    ranking: list[dict[str, Any]] = []
    for design in designs:
        cases = rows_by_design.get(design.design_id, [])
        grouped: dict[tuple[float, str], list[dict[str, Any]]] = {}
        for case in cases:
            key = (float(case["preload"]), str(case["terrain_stratum"]))
            grouped.setdefault(key, []).append(case)
        completion_floor = 1.0
        support_loss_ceiling = 0.0
        fx_normalized_floor = math.inf
        neff_fraction_floor = math.inf
        max_load_share_ceiling = 0.0
        hard_stop_ceiling = 0.0
        recontact_ratio_ceiling = 0.0
        landing_change_ratio_ceiling = 0.0
        group_count = 0
        for (preload_N, _), group_cases in grouped.items():
            group_count += 1
            completion_floor = min(
                completion_floor,
                _finite_quantile(
                    (
                        item["completion_ratio"]
                        for item in group_cases
                    ),
                    0.10,
                    0.0,
                ),
            )
            support_loss_ceiling = max(
                support_loss_ceiling,
                sum(
                    item["case_status"] != "complete"
                    for item in group_cases
                )
                / len(group_cases),
            )
            fx_normalized_floor = min(
                fx_normalized_floor,
                _finite_quantile(
                    (
                        float(item["Fx_q10"]) / preload_N
                        for item in group_cases
                    ),
                    0.10,
                    -math.inf,
                ),
            )
            spine_count = design.geometry.nx * design.geometry.ny
            neff_fraction_floor = min(
                neff_fraction_floor,
                _finite_quantile(
                    (
                        float(item["Neff_q10"]) / spine_count
                        for item in group_cases
                    ),
                    0.10,
                    -math.inf,
                ),
            )
            max_load_share_ceiling = max(
                max_load_share_ceiling,
                _finite_quantile(
                    (
                        item["max_load_share_q90"]
                        for item in group_cases
                    ),
                    0.90,
                    math.inf,
                ),
            )
            hard_stop_ceiling = max(
                hard_stop_ceiling,
                _finite_quantile(
                    (
                        item["hard_stop_ratio"]
                        for item in group_cases
                    ),
                    0.90,
                    math.inf,
                ),
            )
            recontact_ratio_ceiling = max(
                recontact_ratio_ceiling,
                _finite_quantile(
                    (
                        item["recontact_ratio"]
                        for item in group_cases
                    ),
                    0.90,
                    math.inf,
                ),
            )
            landing_change_ratio_ceiling = max(
                landing_change_ratio_ceiling,
                _finite_quantile(
                    (
                        item["landing_change_ratio"]
                        for item in group_cases
                    ),
                    0.90,
                    math.inf,
                ),
            )
        if not grouped:
            completion_floor = 0.0
            support_loss_ceiling = 1.0
            fx_normalized_floor = -math.inf
            neff_fraction_floor = -math.inf
            max_load_share_ceiling = math.inf
            hard_stop_ceiling = math.inf
            recontact_ratio_ceiling = math.inf
            landing_change_ratio_ceiling = math.inf
        record = _full_design_record(design)
        record.update(
            {
                "case_count": len(cases),
                "preload_stratum_group_count": group_count,
                "completion_floor": completion_floor,
                "support_loss_ceiling": support_loss_ceiling,
                "Fx_over_preload_floor": fx_normalized_floor,
                "Neff_fraction_floor": neff_fraction_floor,
                "max_load_share_ceiling": max_load_share_ceiling,
                "hard_stop_ceiling": hard_stop_ceiling,
                "recontact_ratio_ceiling": recontact_ratio_ceiling,
                "landing_change_ratio_ceiling": (
                    landing_change_ratio_ceiling
                ),
                "eligible": bool(
                    completion_floor >= completion_gate
                    and support_loss_ceiling <= support_loss_gate
                    and hard_stop_ceiling <= 0.35
                    and math.isfinite(fx_normalized_floor)
                    and math.isfinite(neff_fraction_floor)
                ),
            }
        )
        ranking.append(record)
    ranking.sort(
        key=lambda item: (
            not item["eligible"],
            -item["completion_floor"],
            item["support_loss_ceiling"],
            -item["Fx_over_preload_floor"],
            -item["Neff_fraction_floor"],
            item["recontact_ratio_ceiling"],
            item["landing_change_ratio_ceiling"],
            item["max_load_share_ceiling"],
            item["hard_stop_ceiling"],
            item["design_id"],
        )
    )
    for rank, record in enumerate(ranking, start=1):
        record["rank"] = rank
    return ranking


def _select_full_designs(
    ranking: Sequence[dict[str, Any]],
    designs: Sequence[FullScanDesign],
    target: int,
) -> list[FullScanDesign]:
    by_id = {item.design_id: item for item in designs}
    ordered = [
        by_id[item["design_id"]]
        for item in ranking
        if item["design_id"] in by_id
    ]
    eligible_ids = {
        item["design_id"] for item in ranking if item["eligible"]
    }
    preferred = [
        item for item in ordered if item.design_id in eligible_ids
    ]
    fallback = [
        item for item in ordered if item.design_id not in eligible_ids
    ]
    selected: list[FullScanDesign] = []

    def add_best(predicate: Callable[[FullScanDesign], bool]) -> None:
        for candidate in preferred:
            if predicate(candidate) and candidate not in selected:
                selected.append(candidate)
                return

    for shape in (
        "2x2",
        "2x5",
        "5x2",
        "3x5",
        "5x3",
        "4x4",
        "6x6",
    ):
        add_best(lambda item, value=shape: item.geometry.array_shape == value)
    for spacing_m in (0.004, 0.005, 0.006):
        add_best(
            lambda item, value=spacing_m: item.geometry.spacing_m == value
        )
    for angle_pattern in ("fixed", "60_to_80"):
        add_best(
            lambda item, value=angle_pattern: (
                item.geometry.angle_pattern == value
            )
        )
    for spring in (300.0, 800.0, 2000.0, None):
        add_best(
            lambda item, value=spring: (
                item.package.spring_stiffness_N_per_m == value
            )
        )
    for candidate in preferred + fallback:
        if len(selected) >= target:
            break
        if candidate not in selected:
            selected.append(candidate)
    return selected[:target]


def _select_final_six(
    ranking: Sequence[dict[str, Any]],
    designs: Sequence[FullScanDesign],
) -> list[FullScanDesign]:
    by_id = {item.design_id: item for item in designs}
    ordered = [
        by_id[item["design_id"]]
        for item in ranking
        if item["design_id"] in by_id and item["eligible"]
    ]
    if len(ordered) < 6:
        ordered = [
            by_id[item["design_id"]]
            for item in ranking
            if item["design_id"] in by_id
        ]
    selected = list(ordered[:3])

    def mechanism_values(item: FullScanDesign) -> tuple[Any, ...]:
        return (
            item.geometry.array_shape,
            item.geometry.spacing_m,
            item.geometry.angle_pattern,
            item.package.tip_radius_m,
            item.package.diameter_m,
            item.package.spring_family,
        )

    while len(selected) < 6:
        represented = [
            set(values)
            for values in zip(
                *(mechanism_values(item) for item in selected),
                strict=True,
            )
        ]
        best: FullScanDesign | None = None
        best_score = -1
        for candidate in ordered:
            if candidate in selected:
                continue
            score = sum(
                value not in represented[index]
                for index, value in enumerate(
                    mechanism_values(candidate)
                )
            )
            if score > best_score:
                best = candidate
                best_score = score
        if best is None:
            break
        selected.append(best)
    return selected[:6]


def _load_design_selection(
    selection_path: Path,
    all_designs: Sequence[FullScanDesign],
) -> list[FullScanDesign]:
    value = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_ids = set(value["selected_design_ids"])
    selected = [
        item for item in all_designs if item.design_id in selected_ids
    ]
    if len(selected) != len(selected_ids):
        raise RuntimeError("selected design file contains unknown identities")
    return selected


def _full_stage_definition(
    stage: str,
) -> tuple[tuple[int, ...], PathSettings, bool, int]:
    if stage == "coarse":
        return (
            FULL_SCAN_COARSE_SEEDS,
            PathSettings(
                preload_N=1.0,
                path_length_m=0.010,
                dx_m=0.0001,
            ),
            False,
            96,
        )
    if stage == "fine":
        return (
            FULL_SCAN_FINE_SEEDS,
            PathSettings(
                preload_N=1.0,
                path_length_m=0.010,
                dx_m=0.0001,
            ),
            True,
            24,
        )
    if stage == "final":
        return (
            FULL_SCAN_FINAL_SEEDS,
            PathSettings(
                preload_N=1.0,
                path_length_m=0.020,
                dx_m=0.0001,
            ),
            True,
            6,
        )
    raise ValueError("full scan stage must be coarse, fine, or final")


def run_full_scan_stage(
    stage: str,
    catalog_path: str | Path = DEFAULT_CATALOG,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT / "full_scan",
    *,
    workers: int = 6,
    model: ModelSettings = ModelSettings(),
    backplate_travel_m: float = 0.006,
) -> dict[str, Any]:
    started = time.perf_counter()
    catalog_path = Path(catalog_path).resolve()
    output_root = Path(output_root).resolve()
    stage_dir = output_root / stage
    seeds, base_path, capture_paths, target = _full_stage_definition(stage)
    base_path = replace(
        base_path, backplate_travel_m=backplate_travel_m
    )
    catalog_document = _catalog_document(catalog_path)
    catalog = load_catalog(catalog_path, require_formal_300=True)
    conditions = [
        item for item in catalog if item.seed in set(seeds)
    ]
    expected_conditions = len(seeds) * 3
    if len(conditions) != expected_conditions:
        raise RuntimeError(
            f"{stage} expected {expected_conditions} terrain conditions, "
            f"got {len(conditions)}"
        )
    all_designs = generate_full_scan_designs()
    if stage == "coarse":
        designs = all_designs
    else:
        previous_stage = "coarse" if stage == "fine" else "fine"
        designs = _load_design_selection(
            output_root / previous_stage / "selected_designs.json",
            all_designs,
        )
    designs_tuple = tuple(designs)
    design_rows = [_full_design_record(item) for item in all_designs]
    _write_parquet(output_root / "design_table.parquet", design_rows)
    terrain_rows = [
        {
            **_condition_mapping(item),
            "terrain_stratum": _terrain_stratum(item),
        }
        for item in conditions
    ]
    _write_parquet(stage_dir / "terrain_table.parquet", terrain_rows)
    running_manifest = {
        "schema_version": "m3-full-scan-stage-v1",
        "status": "running",
        "stage": stage,
        "catalog_path": str(catalog_path),
        "output_root": str(output_root),
        "condition_count": len(conditions),
        "design_count": len(designs),
        "preloads_N": list(FULL_SCAN_PRELOADS_N),
        "expected_case_count": (
            len(conditions) * len(designs) * len(FULL_SCAN_PRELOADS_N)
        ),
        "path_settings": asdict(base_path),
        "model_settings": asdict(model),
        "capture_full_paths": capture_paths,
        "solver_semantics": FULL_SCAN_SOLVER_SEMANTICS,
        "formal_catalog_ranking_eligible_upstream": bool(
            catalog_document.get("formal_ranking_eligible", False)
        ),
        "upstream_limitations": catalog_document.get("limitations", []),
    }
    _write_json(stage_dir / "manifest.json", running_manifest)
    payloads = [
        (
            stage,
            str(stage_dir),
            condition,
            designs_tuple,
            base_path,
            model,
            capture_paths,
        )
        for condition in conditions
    ]
    worker_results = _run_full_workers(payloads, workers)
    observed_case_count = sum(
        int(item["case_count"]) for item in worker_results
    )
    if observed_case_count != running_manifest["expected_case_count"]:
        raise RuntimeError(
            f"{stage} wrote {observed_case_count} cases, expected "
            f"{running_manifest['expected_case_count']}"
        )
    rows = _read_full_summary_rows(stage_dir / "summaries")
    if len(rows) != observed_case_count:
        raise RuntimeError(
            f"{stage} summary shards contain {len(rows)} rows, "
            f"expected {observed_case_count}"
        )
    ranking = _rank_full_designs(rows, designs, stage=stage)
    if stage == "final":
        selected = _select_final_six(ranking, designs)
    else:
        selected = _select_full_designs(ranking, designs, target)
    selection = {
        "schema_version": "m3-full-selected-designs-v1",
        "stage": stage,
        "selected_design_ids": [item.design_id for item in selected],
        "selected_designs": [item.to_mapping() for item in selected],
        "selection_count": len(selected),
        "eligible_selected_count": sum(
            ranking[
                next(
                    index
                    for index, record in enumerate(ranking)
                    if record["design_id"] == item.design_id
                )
            ]["eligible"]
            for item in selected
        ),
        "selection_rule": (
            "completion/support gate, worst preload/terrain-stratum "
            "Fx/preload and Neff/N, recontact/alternate-landing frequency, "
            "load-share/hard-stop checks, then working mechanism coverage"
        ),
    }
    _write_parquet(stage_dir / "ranking.parquet", ranking)
    _write_json(
        stage_dir / "selected_designs.json", _json_safe(selection)
    )
    completed_manifest = {
        **running_manifest,
        "status": "complete",
        "observed_case_count": observed_case_count,
        "selected_design_count": len(selected),
        "max_station_evaluations_observed": max(
            int(item["max_station_evaluations"])
            for item in worker_results
        ),
        "max_station_total_evaluations_observed": max(
            int(item["max_station_total_evaluations"])
            for item in worker_results
        ),
        "max_station_attempts_observed": max(
            int(item["max_station_attempts"])
            for item in worker_results
        ),
        "resumed_condition_count": sum(
            bool(item["resumed"]) for item in worker_results
        ),
        "elapsed_s": time.perf_counter() - started,
    }
    _write_json(stage_dir / "manifest.json", completed_manifest)
    return {
        "manifest": completed_manifest,
        "selection": selection,
        "top_ranked": ranking[:10],
    }


def run_full_scan_auto(
    catalog_path: str | Path = DEFAULT_CATALOG,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT / "full_scan",
    *,
    workers: int = 6,
    model: ModelSettings = ModelSettings(),
    backplate_travel_m: float = 0.006,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for stage in ("coarse", "fine", "final"):
        results[stage] = run_full_scan_stage(
            stage,
            catalog_path,
            output_root,
            workers=workers,
            model=model,
            backplate_travel_m=backplate_travel_m,
        )
    return {
        "schema_version": "m3-full-scan-auto-v1",
        "status": "complete",
        "output_root": str(Path(output_root).resolve()),
        "stages": {
            name: {
                "case_count": value["manifest"]["observed_case_count"],
                "selected_design_count": value["manifest"][
                    "selected_design_count"
                ],
            }
            for name, value in results.items()
        },
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--static-friction", type=float, default=0.45)
    parser.add_argument("--kinetic-friction", type=float, default=0.35)
    parser.add_argument("--backplate-travel-mm", type=float, default=6.0)


def _add_campaign_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--material", default="sandpaper")
    parser.add_argument("--subtype", default="P240")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs=6,
        default=(41005, 41010, 41015, 41020, 41025, 41030),
        metavar=("S1", "S2", "S3", "S4", "S5", "S6"),
    )


def _model_from_args(args: argparse.Namespace) -> ModelSettings:
    return ModelSettings(
        static_friction=args.static_friction,
        kinetic_friction=args.kinetic_friction,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("smoke", "m3a", "m3b", "all"):
        child = subparsers.add_parser(command)
        _add_common_arguments(child)
        if command in {"m3a", "m3b", "all"}:
            _add_campaign_arguments(child)
        if command == "m3b":
            child.add_argument("--selected-packages", type=Path)
    for command in (
        "full-coarse",
        "full-fine",
        "full-final",
        "full-auto",
    ):
        child = subparsers.add_parser(command)
        _add_common_arguments(child)
    args = parser.parse_args(argv)
    model = _model_from_args(args)
    travel_m = args.backplate_travel_mm * 1e-3

    if args.command.startswith("full-"):
        output = args.output or (DEFAULT_OUTPUT_ROOT / "full_scan")
        if args.command == "full-auto":
            result = run_full_scan_auto(
                args.catalog,
                output,
                workers=args.workers,
                model=model,
                backplate_travel_m=travel_m,
            )
        else:
            result = run_full_scan_stage(
                args.command.removeprefix("full-"),
                args.catalog,
                output,
                workers=args.workers,
                model=model,
                backplate_travel_m=travel_m,
            )
    elif args.command == "smoke":
        output = args.output or (DEFAULT_OUTPUT_ROOT / "smoke")
        result = run_smoke_checks(
            args.catalog,
            output,
            model=model,
            backplate_travel_m=travel_m,
        )
    elif args.command == "m3a":
        output = args.output or (DEFAULT_OUTPUT_ROOT / "m3a")
        result = run_m3a(
            args.catalog,
            output,
            material=args.material,
            subtype=args.subtype,
            seeds=args.seeds,
            workers=args.workers,
            model=model,
            backplate_travel_m=travel_m,
        )
    elif args.command == "m3b":
        output = args.output or (DEFAULT_OUTPUT_ROOT / "m3b")
        selected_path = args.selected_packages
        if selected_path is None:
            selected_path = (
                DEFAULT_OUTPUT_ROOT / "m3a" / "selected_packages.json"
            )
        result = run_m3b(
            selected_path,
            args.catalog,
            output,
            material=args.material,
            subtype=args.subtype,
            seeds=args.seeds,
            workers=args.workers,
            model=model,
            backplate_travel_m=travel_m,
        )
    else:
        output = args.output or DEFAULT_OUTPUT_ROOT
        smoke = run_smoke_checks(
            args.catalog,
            output / "smoke",
            model=model,
            backplate_travel_m=travel_m,
        )
        m3a = run_m3a(
            args.catalog,
            output / "m3a",
            material=args.material,
            subtype=args.subtype,
            seeds=args.seeds,
            workers=args.workers,
            model=model,
            backplate_travel_m=travel_m,
        )
        m3b = run_m3b(
            output / "m3a" / "selected_packages.json",
            args.catalog,
            output / "m3b",
            material=args.material,
            subtype=args.subtype,
            seeds=args.seeds,
            workers=args.workers,
            model=model,
            backplate_travel_m=travel_m,
        )
        result = {
            "smoke_all_passed": smoke["all_passed"],
            "m3a_case_count": m3a["manifest"]["case_count"],
            "m3b_case_count": m3b["manifest"]["case_count"],
            "m3b_selection_count": m3b["selection"]["selection_count"],
        }
    print(
        json.dumps(
            _json_safe(result),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
