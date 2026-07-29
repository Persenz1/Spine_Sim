"""Complete M3 hardware design and strictly paired terrain campaign shards."""

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Any, Iterable, Mapping, Sequence

from spine_sim.contact import AxialMode, SpineParameters
from spine_sim.core.config import CampaignSpec
from spine_sim.core.identity import identity, stable_hash

from .models import AngleLayout, ArrayConfiguration, M3_MODULE_VERSION


REPRESENTATIVE_SHAPES = (
    (2, 2),
    (2, 5),
    (5, 2),
    (3, 5),
    (5, 3),
    (4, 4),
    (6, 6),
)
SPACINGS_M = (4e-3, 5e-3, 6e-3)
TIP_RADII_M = (50e-6, 100e-6)
DIAMETERS_M = (0.6e-3, 0.8e-3)
FIXED_ANGLES_DEG = (60.0, 70.0, 80.0)
AXIAL_OPTIONS = (
    (AxialMode.SPRING, 300.0),
    (AxialMode.SPRING, 800.0),
    (AxialMode.SPRING, 2000.0),
    (AxialMode.RIGID, None),
)
GRADIENT_LAYOUTS = (AngleLayout.GRADIENT_80_TO_60,)
TOTAL_PRELOADS_N = (0.5, 1.0, 2.0)
DRAG_LENGTH_M = 0.1
TERRAIN_FAMILIES = ("sandpaper", "red_brick", "concrete")


def _proxy_spine(
    *,
    tip_radius_m: float,
    diameter_m: float,
    installation_angle_deg: float,
    axial_mode: AxialMode,
    spring_stiffness_n_m: float | None,
) -> SpineParameters:
    return SpineParameters(
        tip_radius_m=tip_radius_m,
        diameter_m=diameter_m,
        exposed_length_m=4e-3,
        installation_angle_deg=installation_angle_deg,
        axial_mode=axial_mode,
        spring_stiffness_n_m=spring_stiffness_n_m,
        spring_travel_m=4e-3,
        young_modulus_pa=200e9,
        poisson_ratio=0.29,
        shear_correction=6.0 / 7.0,
        static_friction=0.30,
        kinetic_friction=0.20,
        beam_enabled=True,
        material_assumption=(
            "high_carbon_steel_proxy_E200GPa_rho7850_yield800MPa"
        ),
        rod_clearance_mode="proxy_cylindrical_shank_postcheck",
        density_kg_m3=7850.0,
        axial_modal_mass_factor=1.0 / 3.0,
        transverse_modal_mass_factor=0.236,
        axial_damping_ratio=0.05,
        transverse_damping_ratio=0.05,
        yield_strength_pa=800e6,
    )


def build_base_hardware() -> list[dict[str, Any]]:
    """Return the required 2×2×3×4 = 48 fixed-angle hardware product."""

    rows: list[dict[str, Any]] = []
    for radius_m, diameter_m, angle_deg, axial in product(
        TIP_RADII_M,
        DIAMETERS_M,
        FIXED_ANGLES_DEG,
        AXIAL_OPTIONS,
    ):
        axial_mode, stiffness = axial
        spine = _proxy_spine(
            tip_radius_m=radius_m,
            diameter_m=diameter_m,
            installation_angle_deg=angle_deg,
            axial_mode=axial_mode,
            spring_stiffness_n_m=stiffness,
        )
        identity_fields = {
            "tip_radius_m": radius_m,
            "diameter_m": diameter_m,
            "installation_angle_deg": angle_deg,
            "axial_mode": axial_mode.value,
            "spring_stiffness_n_m": stiffness,
        }
        rows.append(
            {
                **identity_fields,
                "base_hardware_id": identity(
                    "base_hardware",
                    identity_fields,
                    module_version=M3_MODULE_VERSION,
                ),
                "priority_role": (
                    "primary"
                    if radius_m == 100e-6 and diameter_m == 0.8e-3
                    else "auxiliary"
                ),
                "spine": spine.as_dict(),
            }
        )
    rows.sort(key=lambda row: row["base_hardware_id"])
    if len(rows) != 48:
        raise AssertionError(f"expected 48 base hardware rows, got {len(rows)}")
    return rows


def build_full_array_design(
    *,
    include_gradient_80_to_60: bool = True,
) -> list[dict[str, Any]]:
    """Expand all fixed arrays and the non-duplicated 80°→60° layout."""

    rows: list[dict[str, Any]] = []
    base_hardware = build_base_hardware()
    for hardware, (nx, ny), spacing_m in product(
        base_hardware,
        REPRESENTATIVE_SHAPES,
        SPACINGS_M,
    ):
        configuration = ArrayConfiguration(
            nx=nx,
            ny=ny,
            spacing_m=spacing_m,
            base_spine=SpineParameters.from_mapping(hardware["spine"]),
            angle_layout=AngleLayout.FIXED,
        )
        rows.append(
            {
                "array_configuration_id": configuration.configuration_id,
                "base_hardware_id": hardware["base_hardware_id"],
                "priority_role": hardware["priority_role"],
                "nx": nx,
                "ny": ny,
                "spacing_m": spacing_m,
                "angle_layout": AngleLayout.FIXED.value,
                "fixed_angle_deg": hardware["installation_angle_deg"],
                "tip_radius_m": hardware["tip_radius_m"],
                "diameter_m": hardware["diameter_m"],
                "axial_mode": hardware["axial_mode"],
                "spring_stiffness_n_m": hardware[
                    "spring_stiffness_n_m"
                ],
                "configuration": configuration.as_dict(),
            }
        )
    if include_gradient_80_to_60:
        gradient_hardware: dict[tuple[Any, ...], dict[str, Any]] = {}
        for hardware in base_hardware:
            key = (
                hardware["tip_radius_m"],
                hardware["diameter_m"],
                hardware["axial_mode"],
                hardware["spring_stiffness_n_m"],
            )
            gradient_hardware.setdefault(key, hardware)
        for hardware, (nx, ny), spacing_m in product(
            gradient_hardware.values(),
            REPRESENTATIVE_SHAPES,
            SPACINGS_M,
        ):
            spine_data = dict(hardware["spine"])
            spine_data["installation_angle_deg"] = 80.0
            configuration = ArrayConfiguration(
                nx=nx,
                ny=ny,
                spacing_m=spacing_m,
                base_spine=SpineParameters.from_mapping(spine_data),
                angle_layout=AngleLayout.GRADIENT_80_TO_60,
            )
            layout_hardware_fields = {
                "tip_radius_m": hardware["tip_radius_m"],
                "diameter_m": hardware["diameter_m"],
                "axial_mode": hardware["axial_mode"],
                "spring_stiffness_n_m": hardware[
                    "spring_stiffness_n_m"
                ],
                "angle_layout": AngleLayout.GRADIENT_80_TO_60.value,
            }
            rows.append(
                {
                    "array_configuration_id": configuration.configuration_id,
                    "base_hardware_id": identity(
                        "layout_hardware",
                        layout_hardware_fields,
                        module_version=M3_MODULE_VERSION,
                    ),
                    "priority_role": hardware["priority_role"],
                    "nx": nx,
                    "ny": ny,
                    "spacing_m": spacing_m,
                    "angle_layout": (
                        AngleLayout.GRADIENT_80_TO_60.value
                    ),
                    "fixed_angle_deg": None,
                    "tip_radius_m": hardware["tip_radius_m"],
                    "diameter_m": hardware["diameter_m"],
                    "axial_mode": hardware["axial_mode"],
                    "spring_stiffness_n_m": hardware[
                        "spring_stiffness_n_m"
                    ],
                    "configuration": configuration.as_dict(),
                }
            )
    unique = {row["array_configuration_id"]: row for row in rows}
    output = [unique[key] for key in sorted(unique)]
    expected = 1344 if include_gradient_80_to_60 else 1008
    if len(output) != expected:
        raise AssertionError(
            f"expected {expected} array configurations, got {len(output)}"
        )
    return output


def build_candidate_pool(
    parameter_packs: Sequence[Mapping[str, Any]],
    *,
    shapes: Sequence[tuple[int, int]] = REPRESENTATIVE_SHAPES,
    spacings_m: Sequence[float] = SPACINGS_M,
    fixed_angles_deg: Sequence[float] = FIXED_ANGLES_DEG,
) -> list[dict[str, Any]]:
    """Build a legacy coverage fixture; never use it for formal M3 scanning."""

    pool: list[dict[str, Any]] = []
    for pack in parameter_packs:
        pack_id = str(pack["parameter_pack_id"])
        spine = SpineParameters.from_mapping(pack["spine"])
        for nx, ny in shapes:
            for spacing_m in spacings_m:
                for angle_deg in fixed_angles_deg:
                    row = {
                        "parameter_pack_id": pack_id,
                        "nx": int(nx),
                        "ny": int(ny),
                        "spacing_m": float(spacing_m),
                        "angle_layout": AngleLayout.FIXED.value,
                        "fixed_angle_deg": float(angle_deg),
                        "tip_radius_m": spine.tip_radius_m,
                        "diameter_m": spine.diameter_m,
                        "axial_mode": spine.axial_mode.value,
                        "spring_stiffness_n_m": spine.spring_stiffness_n_m,
                    }
                    row["hardware_candidate_id"] = identity(
                        "hardware_candidate",
                        row,
                        module_version=M3_MODULE_VERSION,
                    )
                    pool.append(row)
                for layout in GRADIENT_LAYOUTS:
                    row = {
                        "parameter_pack_id": pack_id,
                        "nx": int(nx),
                        "ny": int(ny),
                        "spacing_m": float(spacing_m),
                        "angle_layout": layout.value,
                        "fixed_angle_deg": None,
                        "tip_radius_m": spine.tip_radius_m,
                        "diameter_m": spine.diameter_m,
                        "axial_mode": spine.axial_mode.value,
                        "spring_stiffness_n_m": spine.spring_stiffness_n_m,
                    }
                    row["hardware_candidate_id"] = identity(
                        "hardware_candidate",
                        row,
                        module_version=M3_MODULE_VERSION,
                    )
                    pool.append(row)
    unique = {row["hardware_candidate_id"]: row for row in pool}
    return [unique[key] for key in sorted(unique)]


def _tokens(row: Mapping[str, Any]) -> frozenset[str]:
    angle = (
        f"fixed_{row['fixed_angle_deg']:g}"
        if row["angle_layout"] == "fixed"
        else str(row["angle_layout"])
    )
    stiffness = (
        "rigid"
        if row["spring_stiffness_n_m"] is None
        else f"{float(row['spring_stiffness_n_m']):g}"
    )
    main = {
        f"pack={row['parameter_pack_id']}",
        f"shape={row['nx']}x{row['ny']}",
        f"nx={row['nx']}",
        f"ny={row['ny']}",
        f"spacing={float(row['spacing_m']):g}",
        f"layout={row['angle_layout']}",
        f"angle={angle}",
        f"tip={float(row['tip_radius_m']):g}",
        f"diameter={float(row['diameter_m']):g}",
        f"axial={row['axial_mode']}",
        f"stiffness={stiffness}",
    }
    interactions = {
        f"installation_mode*scale={row['axial_mode']}*{row['nx']}x{row['ny']}",
        f"stiffness*spacing={stiffness}*{float(row['spacing_m']):g}",
        f"angle*direction={angle}*{row['nx']}x{row['ny']}",
        f"gradient*nx={row['angle_layout']}*{row['nx']}",
        f"diameter*stiffness={float(row['diameter_m']):g}*{stiffness}",
        f"tip*pack={float(row['tip_radius_m']):g}*{row['parameter_pack_id']}",
    }
    return frozenset(main | interactions)


def select_balanced_candidates(
    pool: Sequence[Mapping[str, Any]],
    target_count: int,
) -> list[dict[str, Any]]:
    """Select a legacy deterministic fixture, not a formal M3 design."""

    if target_count < 1:
        raise ValueError("target_count must be positive")
    if target_count > len(pool):
        raise ValueError("target_count cannot exceed the unique candidate pool")
    candidates = [dict(row) for row in pool]
    token_map = {
        row["hardware_candidate_id"]: _tokens(row)
        for row in candidates
    }
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    remaining = {
        row["hardware_candidate_id"]: row
        for row in candidates
    }
    while len(selected) < target_count:
        best_id: str | None = None
        best_score: tuple[float, float, float, str] | None = None
        for candidate_id, row in remaining.items():
            tokens = token_map[candidate_id]
            novelty = float(sum(counts[token] == 0 for token in tokens))
            balance = float(sum(1.0 / (1.0 + counts[token]) for token in tokens))
            if selected:
                max_overlap = max(
                    len(tokens & token_map[item["hardware_candidate_id"]])
                    / len(tokens | token_map[item["hardware_candidate_id"]])
                    for item in selected
                )
                distance = 1.0 - max_overlap
            else:
                distance = 1.0
            score = (novelty, balance, distance, candidate_id)
            if best_score is None or score > best_score:
                best_score = score
                best_id = candidate_id
        assert best_id is not None
        chosen = remaining.pop(best_id)
        selected.append(chosen)
        counts.update(token_map[best_id])
    return selected


def level_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    fields = (
        "parameter_pack_id",
        "nx",
        "ny",
        "spacing_m",
        "angle_layout",
        "fixed_angle_deg",
        "tip_radius_m",
        "diameter_m",
        "axial_mode",
        "spring_stiffness_n_m",
    )
    output: dict[str, dict[str, int]] = {}
    materialized = list(rows)
    for field in fields:
        counts = Counter(str(row[field]) for row in materialized)
        output[field] = dict(sorted(counts.items()))
    return output


def _terrain_family(condition: Mapping[str, Any]) -> str:
    raw = str(
        condition.get(
            "terrain_family",
            condition.get("family", ""),
        )
    ).strip().lower()
    aliases = {
        "sandpaper": "sandpaper",
        "砂纸": "sandpaper",
        "red_brick": "red_brick",
        "red-brick": "red_brick",
        "brick": "red_brick",
        "红砖": "red_brick",
        "concrete": "concrete",
        "混凝土": "concrete",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(
            "every M3 terrain condition must identify sandpaper, red_brick "
            "or concrete"
        ) from exc


def validate_terrain_catalog(
    catalog: Mapping[str, Any],
    *,
    require_formal_300: bool = True,
) -> list[dict[str, Any]]:
    """Validate that M1 supplied complete, hashed, uniquely paired terrain."""

    if catalog.get("status") != "complete":
        raise ValueError("M1 terrain catalog status must be complete")
    if not catalog.get("all_full_hashes_verified"):
        raise ValueError("M1 terrain catalog full hashes are not all verified")
    raw_conditions = catalog.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ValueError("M1 terrain catalog contains no conditions")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for raw_condition in raw_conditions:
        condition = dict(raw_condition)
        family = _terrain_family(condition)
        seed = int(condition["seed"])
        key = (family, seed)
        if key in seen:
            raise ValueError(
                f"duplicate terrain pairing condition {family}/seed={seed}"
            )
        seen.add(key)
        if not condition.get("full_sha256_verified", False):
            raise ValueError(
                f"{family}/seed={seed} does not have a verified full hash"
            )
        for field in (
            "terrain_recipe_id",
            "region_id",
            "data_sha256",
        ):
            if not condition.get(field):
                raise ValueError(
                    f"{family}/seed={seed} is missing {field}"
                )
        condition["terrain_family"] = family
        condition["seed"] = seed
        condition["terrain_condition_id"] = identity(
            "terrain_condition",
            {
                "terrain_family": family,
                "seed": seed,
                "terrain_recipe_id": condition["terrain_recipe_id"],
                "region_id": condition["region_id"],
                "data_sha256": condition["data_sha256"],
            },
            module_version=str(catalog.get("m1_module_version", "m1")),
        )
        normalized.append(condition)
    normalized.sort(
        key=lambda condition: (
            TERRAIN_FAMILIES.index(condition["terrain_family"]),
            condition["seed"],
        )
    )
    if require_formal_300:
        counts = Counter(
            condition["terrain_family"] for condition in normalized
        )
        expected = {family: 100 for family in TERRAIN_FAMILIES}
        if dict(counts) != expected or len(normalized) != 300:
            raise ValueError(
                "formal M3 requires exactly 100 verified seeds for each of "
                "sandpaper, red_brick and concrete"
            )
    return normalized


def _loading_protocol(
    preload_n: float,
    *,
    output_spacing_m: float,
    time_step_s: float,
) -> dict[str, Any]:
    if preload_n not in TOTAL_PRELOADS_N:
        raise ValueError("M3 total preload must be 0.5, 1.0 or 2.0 N")
    protocol = {
        "external_total_preload_n": float(preload_n),
        "drag_length_m": DRAG_LENGTH_M,
        "drag_speed_m_s": 1e-3,
        "preload_ramp_profile": "minimum_jerk_quintic",
        "preload_ramp_time_s": 0.25,
        "settlement_damping_scale": 10.0,
        "output_spacing_m": float(output_spacing_m),
        "time_step_s": float(time_step_s),
    }
    protocol["loading_protocol_id"] = identity(
        "loading_protocol",
        protocol,
        module_version=M3_MODULE_VERSION,
    )
    return protocol


def build_campaign_shard(
    catalog: Mapping[str, Any],
    *,
    terrain_family: str,
    seed_min: int,
    seed_max: int,
    preload_n: float,
    output_level: str = "summary",
    workers: int = 1,
    output_spacing_m: float = 50e-6,
    time_step_s: float = 1e-3,
    require_formal_300: bool = True,
) -> dict[str, Any]:
    """Materialize one bounded family/seed/preload shard, never a full run."""

    if terrain_family not in TERRAIN_FAMILIES:
        raise ValueError(
            f"terrain_family must be one of {TERRAIN_FAMILIES}"
        )
    if seed_max < seed_min:
        raise ValueError("seed_max cannot be less than seed_min")
    if output_level not in {
        "summary",
        "aggregate_trace",
        "full_pin_trace",
    }:
        raise ValueError("invalid M3 output level")
    all_conditions = validate_terrain_catalog(
        catalog,
        require_formal_300=require_formal_300,
    )
    conditions = [
        condition
        for condition in all_conditions
        if condition["terrain_family"] == terrain_family
        and seed_min <= condition["seed"] <= seed_max
    ]
    if not conditions:
        raise ValueError("selected M3 terrain shard contains no conditions")
    designs = build_full_array_design(include_gradient_80_to_60=True)
    protocol = _loading_protocol(
        preload_n,
        output_spacing_m=output_spacing_m,
        time_step_s=time_step_s,
    )
    library_root = str(catalog["library_root"])
    catalog_id = str(
        catalog.get(
            "terrain_catalog_id",
            stable_hash(
                [
                    condition["terrain_condition_id"]
                    for condition in all_conditions
                ]
            ),
        )
    )
    cases: list[dict[str, Any]] = []
    for condition, design in product(conditions, designs):
        configuration = ArrayConfiguration.from_mapping(
            design["configuration"]
        )
        track_requests = [
            {
                "radius_m": parameters.tip_radius_m,
                "y_global_m": offset[1],
            }
            for parameters, offset in zip(
                configuration.pin_parameters,
                configuration.holder_offsets_xyz_m,
            )
        ]
        parameters = {
            "terrain_library_root": library_root,
            "terrain_catalog_id": catalog_id,
            "terrain_condition_id": condition["terrain_condition_id"],
            "terrain_family": terrain_family,
            "terrain_seed": condition["seed"],
            "terrain_recipe_id": condition["terrain_recipe_id"],
            "region_id": condition["region_id"],
            "track_requests": track_requests,
            "configuration": design["configuration"],
            "unit_origin_xy_m": [0.0, 0.0],
            "loading_protocol_id": protocol["loading_protocol_id"],
            "experiment": {
                "drag_length_m": DRAG_LENGTH_M,
                "external_total_preload_n": preload_n,
                "initial_common_ux_m": 0.0,
                "drag_speed_m_s": 1e-3,
                "backplate_mass_kg": 0.10,
                "backplate_vertical_damping_n_s_m": 2.0,
                "backplate_rotational_dofs": "locked",
                "backplate_inertia_kg_m2": None,
                "maximum_preload_approach_m": 8e-3,
                "preload_ramp_time_s": protocol["preload_ramp_time_s"],
                "preload_ramp_profile": protocol[
                    "preload_ramp_profile"
                ],
                "settlement_damping_scale": protocol[
                    "settlement_damping_scale"
                ],
                "settling_reaction_force_tolerance_n": 0.01,
                "settling_reaction_force_relative_tolerance": 0.02,
                "settling_dynamic_residual_tolerance_n": 1e-8,
                "settling_stable_steps": 20,
                "dynamic_residual_tolerance_n": 1e-3,
                "coupled_projection_relaxation": 0.8,
                "output_spacing_m": output_spacing_m,
                "effective_pin_normal_force_min_n": 0.05,
                "unclosed_parameter_names": [
                    "m3_dynamic_parameters_not_experimentally_calibrated",
                    "m3_contact_parameters_not_experimentally_calibrated",
                ],
                "time_step_convergence_checked": False,
                "contact_parameter_convergence_checked": False,
                "settlement_damping_convergence_checked": False,
                "terrain_resolution_convergence_checked": False,
                "physical_calibration_completed": False,
            },
            "contact": {
                "normal_model": "rigid_moreau",
                "restitution_coefficient": 0.0,
                "position_correction": 1.0,
                "activation_tolerance_m": 2e-9,
                "impact_velocity_threshold_m_s": 1e-5,
                "maximum_contact_force_n": 250.0,
                "projection_iterations": 20,
            },
            "integrator": {
                "method": "moreau_implicit_euler",
                "time_step_s": time_step_s,
                "settling_time_s": 0.25,
                "settling_velocity_tolerance_m_s": 2e-5,
                "maximum_settling_time_s": 2.0,
                "maximum_steps": 200_000,
            },
            "output": {"level": output_level},
            "screening_policy": {
                "ranking_scope": "project_model_proxy",
                "formal_ranking_eligible": False,
                "paired_design": True,
                "requires_experimental_calibration": True,
            },
        }
        cases.append(
            {
                "module": "m3",
                "module_version": M3_MODULE_VERSION,
                "parameters": parameters,
                "upstream_hash": stable_hash(
                    {
                        "terrain_data_sha256": condition["data_sha256"],
                        "array_configuration_id": design[
                            "array_configuration_id"
                        ],
                        "loading_protocol_id": protocol[
                            "loading_protocol_id"
                        ],
                    }
                ),
                "tags": [
                    "m3_dynamic",
                    "proxy_model",
                    terrain_family,
                    f"seed_{condition['seed']}",
                    f"preload_{preload_n:g}N",
                    output_level,
                ],
            }
        )
    campaign = {
        "name": (
            f"m3_{terrain_family}_seed_{seed_min}_{seed_max}_"
            f"preload_{preload_n:g}N_{output_level}"
        ),
        "module_version": M3_MODULE_VERSION,
        "callable": "spine_sim.array.case:run_case",
        "cases": cases,
        "workers": workers,
        "mode": "formal",
    }
    parsed = CampaignSpec.from_mapping(campaign)
    expected_count = len(conditions) * len(designs)
    if len(parsed.cases) != expected_count:
        raise AssertionError("M3 shard lost a paired design condition")
    return campaign


def validate_paired_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    configuration_ids: Sequence[str],
    terrain_condition_ids: Sequence[str],
    loading_protocol_ids: Sequence[str],
) -> None:
    """Reject missing or duplicated config×terrain×protocol identities."""

    actual: Counter[tuple[str, str, str]] = Counter()
    for case in cases:
        parameters = case["parameters"]
        configuration = ArrayConfiguration.from_mapping(
            parameters["configuration"]
        )
        actual[
            (
                configuration.configuration_id,
                str(parameters["terrain_condition_id"]),
                str(parameters["loading_protocol_id"]),
            )
        ] += 1
    expected = {
        (configuration_id, terrain_id, protocol_id)
        for configuration_id, terrain_id, protocol_id in product(
            configuration_ids,
            terrain_condition_ids,
            loading_protocol_ids,
        )
    }
    missing = expected - set(actual)
    duplicates = [key for key, count in actual.items() if count != 1]
    unexpected = set(actual) - expected
    if missing or duplicates or unexpected:
        raise ValueError(
            "M3 paired case matrix is incomplete: "
            f"missing={len(missing)}, duplicates={len(duplicates)}, "
            f"unexpected={len(unexpected)}"
        )


def screening_gate_status(
    *,
    terrain_catalog_complete: bool,
    physical_calibration_completed: bool,
    convergence_checks_completed: bool,
    paired_design_verified: bool,
) -> dict[str, Any]:
    blockers = []
    if not terrain_catalog_complete:
        blockers.append("M1 3-family x 100-seed catalog incomplete")
    if not physical_calibration_completed:
        blockers.append("M3 physical/contact parameters uncalibrated")
    if not convergence_checks_completed:
        blockers.append("100 mm time-step/contact/terrain convergence open")
    if not paired_design_verified:
        blockers.append("configuration x terrain x preload pairing unverified")
    return {
        "formal_m3_round1_allowed": not blockers,
        "blockers": blockers,
    }
