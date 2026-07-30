"""Materialize executable M3 convergence cases on one formal M1 terrain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from spine_sim.array.convergence import (
    build_convergence_sentinels,
    build_convergence_variants,
)
from spine_sim.array.design import (
    PLACEMENT_SEARCH_OFFSETS_XY_M,
    TOTAL_PRELOADS_N,
    validate_terrain_catalog,
)
from spine_sim.array.models import M3_MODULE_VERSION, ArrayConfiguration
from spine_sim.array.proxy_parameters import (
    EngineeringProxyScenario,
    estimate_backplate_dynamics,
)
from spine_sim.core.config import CampaignSpec
from spine_sim.core.identity import identity, stable_hash
from spine_sim.io.results import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--terrain-family", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--variant",
        action="append",
        dest="variant_names",
    )
    parser.add_argument(
        "--sentinel-index",
        action="append",
        type=int,
        dest="sentinel_indices",
    )
    parser.add_argument(
        "--preload-n",
        action="append",
        type=float,
        dest="preloads_n",
    )
    parser.add_argument("--drag-length-mm", type=float, default=2.0)
    parser.add_argument("--output-spacing-um", type=float, default=50.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/m3_convergence_campaign.json"),
    )
    args = parser.parse_args()
    if args.drag_length_mm <= 0.0:
        parser.error("--drag-length-mm must be positive")
    if args.output_spacing_um <= 0.0:
        parser.error("--output-spacing-um must be positive")

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    conditions = validate_terrain_catalog(
        catalog, require_formal_300=True
    )
    matches = [
        condition
        for condition in conditions
        if condition["terrain_family"] == args.terrain_family
        and condition["seed"] == args.seed
    ]
    if len(matches) != 1:
        parser.error(
            "convergence campaign requires exactly one matching formal "
            "terrain condition"
        )
    condition = matches[0]

    variants_by_name = {
        variant.name: variant
        for variant in build_convergence_variants()
    }
    requested_variant_names = tuple(
        dict.fromkeys(args.variant_names or variants_by_name)
    )
    unknown_variants = set(requested_variant_names) - set(
        variants_by_name
    )
    if unknown_variants:
        parser.error(
            f"unknown convergence variants: {sorted(unknown_variants)}"
        )
    variants = [
        variants_by_name[name] for name in requested_variant_names
    ]

    all_sentinels = build_convergence_sentinels()
    sentinel_indices = tuple(
        dict.fromkeys(
            args.sentinel_indices or range(len(all_sentinels))
        )
    )
    if any(
        index < 0 or index >= len(all_sentinels)
        for index in sentinel_indices
    ):
        parser.error(
            f"--sentinel-index must be between 0 and "
            f"{len(all_sentinels) - 1}"
        )
    preloads_n = tuple(
        dict.fromkeys(args.preloads_n or TOTAL_PRELOADS_N)
    )
    if any(preload not in TOTAL_PRELOADS_N for preload in preloads_n):
        parser.error(
            f"--preload-n must be one of {TOTAL_PRELOADS_N}"
        )

    scenario = EngineeringProxyScenario("baseline")
    drag_length_m = args.drag_length_mm * 1e-3
    requested_output_spacing_m = args.output_spacing_um * 1e-6
    maximum_variant_step_distance_m = max(
        variant.drag_speed_m_s * variant.time_step_s
        for variant in variants
    )
    output_spacing_m = max(
        maximum_variant_step_distance_m,
        min(requested_output_spacing_m, drag_length_m / 100.0),
    )
    cases = []
    for sentinel_index in sentinel_indices:
        sentinel = all_sentinels[sentinel_index]
        configuration = ArrayConfiguration.from_mapping(
            sentinel["configuration"]
        )
        backplate = estimate_backplate_dynamics(
            configuration, scenario
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
        for preload_n in preloads_n:
            for variant in variants:
                convergence_case = {
                    "variant_name": variant.name,
                    "variant_id": variant.variant_id,
                    "axis": variant.axis,
                    "sentinel_index": sentinel_index,
                    "sentinel_configuration_id": (
                        configuration.configuration_id
                    ),
                    "drag_length_m": drag_length_m,
                    "output_spacing_m": output_spacing_m,
                    "terrain_condition_id": condition[
                        "terrain_condition_id"
                    ],
                    "preload_n": preload_n,
                }
                loading_protocol_id = identity(
                    "m3_convergence_loading_protocol",
                    {
                        **convergence_case,
                        "drag_speed_m_s": variant.drag_speed_m_s,
                        "time_step_s": variant.time_step_s,
                        "projection_iterations": (
                            variant.projection_iterations
                        ),
                        "position_correction": (
                            variant.position_correction
                        ),
                        "settlement_damping_scale": (
                            variant.settlement_damping_scale
                        ),
                    },
                    module_version=M3_MODULE_VERSION,
                )
                maximum_steps = max(
                    200_000,
                    int(
                        math.ceil(
                            drag_length_m
                            / variant.drag_speed_m_s
                            / variant.time_step_s
                        )
                    )
                    + 1,
                )
                parameters = {
                    "terrain_library_root": str(catalog["library_root"]),
                    "terrain_catalog_id": catalog[
                        "terrain_catalog_id"
                    ],
                    "terrain_condition_id": condition[
                        "terrain_condition_id"
                    ],
                    "terrain_family": condition["terrain_family"],
                    "terrain_seed": condition["seed"],
                    "terrain_realization_id": condition.get(
                        "realization_id"
                    ),
                    "terrain_recipe_id": condition[
                        "terrain_recipe_id"
                    ],
                    "region_id": condition["region_id"],
                    "terrain_data_sha256": condition["data_sha256"],
                    "track_cache_mode": "read_only",
                    "track_requests": track_requests,
                    "configuration": sentinel["configuration"],
                    "engineering_proxy": {
                        "scenario": scenario.as_dict(),
                        "backplate": backplate,
                    },
                    "unit_origin_xy_m": [0.0, 0.0],
                    "placement_search": {
                        "enabled": True,
                        "selection_rule": "first_collision_free",
                        "offsets_xy_m": [
                            list(offset)
                            for offset in (
                                PLACEMENT_SEARCH_OFFSETS_XY_M
                            )
                        ],
                    },
                    "loading_protocol_id": loading_protocol_id,
                    "convergence_case": convergence_case,
                    "experiment": {
                        "drag_length_m": drag_length_m,
                        "external_total_preload_n": preload_n,
                        "initial_common_ux_m": 0.0,
                        "drag_speed_m_s": variant.drag_speed_m_s,
                        "backplate_mass_kg": backplate[
                            "backplate_mass_kg"
                        ],
                        "backplate_vertical_damping_n_s_m": backplate[
                            "backplate_vertical_damping_n_s_m"
                        ],
                        "backplate_rotational_dofs": "locked",
                        "backplate_inertia_kg_m2": None,
                        "maximum_preload_approach_m": 8e-3,
                        "preload_ramp_time_s": 0.25,
                        "preload_ramp_profile": (
                            "minimum_jerk_quintic"
                        ),
                        "settlement_damping_scale": (
                            variant.settlement_damping_scale
                        ),
                        "settling_reaction_force_tolerance_n": 0.01,
                        "settling_reaction_force_relative_tolerance": (
                            0.02
                        ),
                        "settling_dynamic_residual_tolerance_n": 1e-8,
                        "settling_stable_steps": 20,
                        "dynamic_residual_tolerance_n": 1e-3,
                        "coupled_projection_relaxation": 0.8,
                        "coupled_projection_position_tolerance_m": (
                            1e-12
                        ),
                        "output_spacing_m": output_spacing_m,
                        "effective_pin_normal_force_min_n": 0.05,
                        "unclosed_parameter_names": [
                            "bounded_convergence_not_calibration",
                            (
                                "dynamic_parameters_not_"
                                "experimentally_calibrated"
                            ),
                        ],
                        "time_step_convergence_checked": False,
                        "contact_parameter_convergence_checked": False,
                        "settlement_damping_convergence_checked": False,
                        "terrain_resolution_convergence_checked": False,
                        "physical_calibration_completed": False,
                    },
                    "contact": {
                        "normal_model": "rigid_moreau",
                        "restitution_coefficient": (
                            scenario.restitution_coefficient
                        ),
                        "position_correction": (
                            variant.position_correction
                        ),
                        "activation_tolerance_m": 2e-9,
                        "impact_velocity_threshold_m_s": 1e-5,
                        "maximum_contact_force_n": 250.0,
                        "projection_iterations": (
                            variant.projection_iterations
                        ),
                    },
                    "integrator": {
                        "method": "moreau_implicit_euler",
                        "time_step_s": variant.time_step_s,
                        "settling_time_s": 0.25,
                        "settling_velocity_tolerance_m_s": 2e-5,
                        "maximum_settling_time_s": 2.0,
                        "maximum_steps": maximum_steps,
                    },
                    "output": {"level": "summary"},
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
                                "terrain_data_sha256": condition[
                                    "data_sha256"
                                ],
                                "configuration_id": (
                                    configuration.configuration_id
                                ),
                                "loading_protocol_id": (
                                    loading_protocol_id
                                ),
                            }
                        ),
                        "tags": [
                            "m3_convergence",
                            variant.name,
                            f"sentinel_{sentinel_index}",
                            f"preload_{preload_n:g}N",
                        ],
                    }
                )

    campaign = {
        "name": (
            f"m3_convergence_{args.terrain_family}_{args.seed}_"
            f"{args.drag_length_mm:g}mm"
        ),
        "module_version": M3_MODULE_VERSION,
        "callable": "spine_sim.array.case:run_case",
        "cases": cases,
        "workers": args.workers,
        "mode": "formal",
    }
    parsed = CampaignSpec.from_mapping(campaign)
    atomic_write_json(args.output, campaign)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "campaign_id": parsed.campaign_id,
                "case_count": len(parsed.cases),
                "compact_summary_storage": True,
                "formal_ranking_eligible": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
