"""Build a small real-M1 M3 campaign for runner/recovery validation.

This command is intentionally separate from the formal 300-condition shard
builder.  It accepts an incomplete test catalog only when one condition is
selected explicitly, uses a short path, and always marks the campaign small
and ineligible for formal ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spine_sim.array.design import (
    PLACEMENT_SEARCH_OFFSETS_XY_M,
    build_full_array_design,
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


def _select_condition(
    catalog: dict,
    *,
    terrain_family: str,
    seed: int,
) -> dict:
    matches = [
        condition
        for condition in validate_terrain_catalog(
            catalog,
            require_formal_300=False,
        )
        if condition["terrain_family"] == terrain_family
        and condition["seed"] == seed
    ]
    if len(matches) != 1:
        raise ValueError(
            "validation campaign requires exactly one matching "
            f"{terrain_family}/seed={seed} condition; got {len(matches)}"
        )
    return matches[0]


def _configuration(size: int) -> dict:
    matches = [
        row
        for row in build_full_array_design()
        if row["nx"] == size
        and row["ny"] == size
        and row["spacing_m"] == 4e-3
        and row["angle_layout"] == "fixed"
        and row["fixed_angle_deg"] == 70.0
        and row["tip_radius_m"] == 100e-6
        and row["diameter_m"] == 0.8e-3
        and row["axial_mode"] == "spring"
        and row["spring_stiffness_n_m"] == 800.0
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one {size}x{size} validation configuration"
        )
    return matches[0]["configuration"]


def build_validation_campaign(
    catalog: dict,
    *,
    terrain_family: str,
    seed: int,
    sizes: tuple[int, ...],
    preload_n: float,
    drag_length_m: float,
    workers: int,
    placement_search: bool,
) -> dict:
    condition = _select_condition(
        catalog,
        terrain_family=terrain_family,
        seed=seed,
    )
    scenario = EngineeringProxyScenario("baseline")
    protocol = {
        "scope": "bounded_M3_runner_validation",
        "external_total_preload_n": preload_n,
        "drag_length_m": drag_length_m,
        "drag_speed_m_s": 1e-3,
        "time_step_s": 1e-3,
        "output_spacing_m": min(20e-6, drag_length_m),
        "placement_search_enabled": placement_search,
        "placement_search_offsets_xy_m": (
            [
                list(offset)
                for offset in PLACEMENT_SEARCH_OFFSETS_XY_M
            ]
            if placement_search
            else [[0.0, 0.0]]
        ),
    }
    protocol_id = identity(
        "m3_validation_loading_protocol",
        protocol,
        module_version=M3_MODULE_VERSION,
    )
    catalog_id = str(
        catalog.get(
            "terrain_catalog_id",
            stable_hash(
                [
                    condition["terrain_condition_id"],
                ]
            ),
        )
    )
    cases = []
    for size in sizes:
        configuration_data = _configuration(size)
        configuration = ArrayConfiguration.from_mapping(
            configuration_data
        )
        backplate = estimate_backplate_dynamics(
            configuration,
            scenario,
        )
        parameters = {
            "terrain_library_root": str(catalog["library_root"]),
            "terrain_catalog_id": catalog_id,
            "terrain_condition_id": condition["terrain_condition_id"],
            "terrain_family": condition["terrain_family"],
            "terrain_seed": condition["seed"],
            "terrain_recipe_id": condition["terrain_recipe_id"],
            "region_id": condition["region_id"],
            "terrain_data_sha256": condition["data_sha256"],
            "track_requests": [
                {
                    "radius_m": pin.tip_radius_m,
                    "y_global_m": offset[1],
                }
                for pin, offset in zip(
                    configuration.pin_parameters,
                    configuration.holder_offsets_xyz_m,
                )
            ],
            "configuration": configuration_data,
            "engineering_proxy": {
                "scenario": scenario.as_dict(),
                "backplate": backplate,
            },
            "unit_origin_xy_m": [0.0, 0.0],
            "placement_search": {
                "enabled": placement_search,
                "selection_rule": "first_collision_free",
                "offsets_xy_m": protocol[
                    "placement_search_offsets_xy_m"
                ],
            },
            "loading_protocol_id": protocol_id,
            "experiment": {
                "drag_length_m": drag_length_m,
                "external_total_preload_n": preload_n,
                "initial_common_ux_m": 0.0,
                "drag_speed_m_s": 1e-3,
                "backplate_mass_kg": backplate["backplate_mass_kg"],
                "backplate_vertical_damping_n_s_m": backplate[
                    "backplate_vertical_damping_n_s_m"
                ],
                "backplate_rotational_dofs": "locked",
                "backplate_inertia_kg_m2": None,
                "maximum_preload_approach_m": 8e-3,
                "preload_ramp_time_s": 0.25,
                "preload_ramp_profile": "minimum_jerk_quintic",
                "settlement_damping_scale": (
                    scenario.settlement_damping_scale
                ),
                "settling_reaction_force_tolerance_n": 0.01,
                "settling_reaction_force_relative_tolerance": 0.02,
                "settling_dynamic_residual_tolerance_n": 1e-8,
                "settling_stable_steps": 20,
                "dynamic_residual_tolerance_n": 1e-3,
                "coupled_projection_relaxation": 0.8,
                "output_spacing_m": min(20e-6, drag_length_m),
                "effective_pin_normal_force_min_n": 0.05,
                "unclosed_parameter_names": [
                    "bounded_validation_not_formal_campaign",
                    "dynamic_parameters_not_experimentally_calibrated",
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
                    scenario.contact_position_correction
                ),
                "activation_tolerance_m": 2e-9,
                "impact_velocity_threshold_m_s": 1e-5,
                "maximum_contact_force_n": 250.0,
                "projection_iterations": 20,
            },
            "integrator": {
                "method": "moreau_implicit_euler",
                "time_step_s": 1e-3,
                "settling_time_s": 0.25,
                "settling_velocity_tolerance_m_s": 2e-5,
                "maximum_settling_time_s": 2.0,
                "maximum_steps": 200_000,
            },
            "output": {"level": "summary"},
            "screening_policy": {
                "ranking_scope": "project_model_proxy",
                "formal_ranking_eligible": False,
                "paired_design": False,
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
                        "configuration_id": (
                            configuration.configuration_id
                        ),
                        "loading_protocol_id": protocol_id,
                    }
                ),
                "tags": [
                    "m3_validation",
                    "bounded",
                    terrain_family,
                    f"seed_{seed}",
                    f"{size}x{size}",
                ],
            }
        )
    campaign = {
        "name": (
            f"m3_validation_{terrain_family}_seed_{seed}_"
            f"{drag_length_m * 1e3:g}mm"
        ),
        "module_version": M3_MODULE_VERSION,
        "callable": "spine_sim.array.case:run_case",
        "cases": cases,
        "workers": workers,
        "mode": "small",
    }
    parsed = CampaignSpec.from_mapping(campaign)
    if len(parsed.cases) != len(sizes):
        raise AssertionError("validation campaign lost a requested size")
    return campaign


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--terrain-family", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--size",
        action="append",
        type=int,
        choices=(2, 4, 6),
        dest="sizes",
    )
    parser.add_argument("--preload-n", type=float, default=1.0)
    parser.add_argument("--drag-length-mm", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-placement-search", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/m3_validation_campaign.json"),
    )
    args = parser.parse_args()
    if args.preload_n not in (0.5, 1.0, 2.0):
        parser.error("--preload-n must be 0.5, 1.0 or 2.0")
    if args.drag_length_mm <= 0.0:
        parser.error("--drag-length-mm must be positive")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    campaign = build_validation_campaign(
        catalog,
        terrain_family=args.terrain_family,
        seed=args.seed,
        sizes=tuple(args.sizes or (2, 4, 6)),
        preload_n=args.preload_n,
        drag_length_m=args.drag_length_mm * 1e-3,
        workers=args.workers,
        placement_search=not args.no_placement_search,
    )
    parsed = CampaignSpec.from_mapping(campaign)
    atomic_write_json(args.output, campaign)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "campaign_id": parsed.campaign_id,
                "case_count": len(parsed.cases),
                "formal_campaign": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
