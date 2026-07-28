"""Materialize the approved M2 round-one proxy-model baseline campaign.

This generator never touches the thirty round-two reserve seeds.  It makes the
absence of experimental calibration explicit in every case and produces
project-model screening evidence, not an absolute hardware certification.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

from spine_sim.core.config import CampaignSpec
from spine_sim.core.identity import stable_hash


MODULE_VERSION = "m2.2.0"
PARAMETER_SET_ID = "m2_proxy_v1"
RADII_M = (50e-6, 100e-6)
DIAMETERS_M = (0.6e-3, 0.8e-3)
ANGLES_DEG = (60.0, 70.0, 80.0)
AXIAL_OPTIONS = (
    ("rigid", None),
    ("spring", 100.0),
    ("spring", 300.0),
    ("spring", 800.0),
    ("spring", 2000.0),
)


def _track_for_radius(condition: dict[str, Any], radius_m: float) -> dict[str, Any]:
    matches = [
        track
        for track in condition["tracks"]
        if abs(float(track["radius_m"]) - radius_m) <= 1e-12
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{condition['name']} does not contain exactly one {radius_m:g} m track"
        )
    return matches[0]


def build_campaign(
    catalog: dict[str, Any],
    *,
    workers: int,
    drag_length_m: float,
    time_step_s: float,
    output_spacing_m: float,
) -> dict[str, Any]:
    if catalog.get("status") != "complete":
        raise ValueError("formal terrain catalog is not complete")
    if not catalog.get("all_full_hashes_verified"):
        raise ValueError("formal terrain catalog hashes have not all been verified")
    round_one = [
        condition
        for condition in catalog["conditions"]
        if condition.get("screening_round") == "round1"
    ]
    round_one.sort(key=lambda condition: int(condition["seed"]))
    expected = list(catalog["round1_seed_values"])
    if [int(condition["seed"]) for condition in round_one] != expected:
        raise ValueError("round-one catalog membership does not match frozen seed list")
    if len(round_one) != 15:
        raise ValueError("M2 round one requires exactly fifteen paired seeds")

    library_root = str(Path(catalog["library_root"]).resolve())
    cases: list[dict[str, Any]] = []
    for condition in round_one:
        seed = int(condition["seed"])
        for radius_m, diameter_m, axial, angle_deg in product(
            RADII_M,
            DIAMETERS_M,
            AXIAL_OPTIONS,
            ANGLES_DEG,
        ):
            axial_mode, stiffness = axial
            track = _track_for_radius(condition, radius_m)
            screening_policy = {
                "ranking_scope": "project_model_proxy",
                "parameter_set_id": PARAMETER_SET_ID,
                "case_role": "baseline",
                "paired_design": True,
                "round": 1,
                "seed": seed,
                "requires_experimental_calibration": True,
                "robustness_audit_status": "pending",
            }
            parameters = {
                "terrain_library_root": library_root,
                "terrain_recipe_id": condition["terrain_recipe_id"],
                "region_id": condition["region_id"],
                "track_id": track["track_id"],
                "radius_m": radius_m,
                "spine": {
                    "tip_radius_m": radius_m,
                    "diameter_m": diameter_m,
                    "exposed_length_m": 4e-3,
                    "installation_angle_deg": angle_deg,
                    "axial_mode": axial_mode,
                    "spring_stiffness_n_m": stiffness,
                    "spring_travel_m": 4e-3,
                    "young_modulus_pa": 200e9,
                    "poisson_ratio": 0.29,
                    "shear_correction": 6.0 / 7.0,
                    "static_friction": 0.30,
                    "kinetic_friction": 0.20,
                    "beam_enabled": True,
                    "material_assumption": (
                        "high_carbon_steel_proxy_E200GPa_rho7850_yield800MPa"
                    ),
                    "rod_clearance_mode": "proxy_cylindrical_shank_postcheck",
                    "density_kg_m3": 7850.0,
                    "axial_modal_mass_factor": 1.0 / 3.0,
                    "transverse_modal_mass_factor": 0.236,
                    "axial_damping_ratio": 0.05,
                    "transverse_damping_ratio": 0.05,
                    "yield_strength_pa": 800e6,
                },
                "experiment": {
                    "initial_center_x_m": 2e-3,
                    "drag_length_m": drag_length_m,
                    "drag_speed_m_s": 1e-3,
                    "constant_preload_n": 0.5,
                    "holder_effective_mass_kg": 0.05,
                    "holder_vertical_damping_n_s_m": 1.0,
                    "maximum_preload_approach_m": 8e-3,
                    "output_spacing_m": output_spacing_m,
                    "effective_normal_force_min_n": 0.05,
                    "initial_preload_force_tolerance_n": 1e-4,
                },
                "dynamic_contact": {
                    "normal_model": "rigid_moreau",
                    "restitution_coefficient": 0.0,
                    "position_correction": 1.0,
                    "activation_tolerance_m": 2e-9,
                    "impact_velocity_threshold_m_s": 1e-5,
                    "maximum_contact_force_n": 250.0,
                    "projection_iterations": 6,
                },
                "dynamic_integrator": {
                    "method": "moreau_implicit_euler",
                    "time_step_s": time_step_s,
                    "settling_time_s": 0.25,
                    "settling_velocity_tolerance_m_s": 2e-5,
                    "maximum_settling_time_s": 2.0,
                    "maximum_steps": 2_000_000,
                },
                "screening_policy": screening_policy,
            }
            cases.append(
                {
                    "module": "m2",
                    "module_version": MODULE_VERSION,
                    "parameters": parameters,
                    "upstream_hash": stable_hash(
                        {
                            "terrain_data_sha256": condition["data_sha256"],
                            "track_id": track["track_id"],
                        }
                    ),
                    "tags": [
                        "m2_dynamic",
                        "round1",
                        "proxy_model",
                        f"seed_{seed}",
                    ],
                }
            )
    campaign = {
        "name": "m2_dynamic_round1_proxy_baseline",
        "module_version": MODULE_VERSION,
        "callable": "spine_sim.contact.case:run_case",
        "cases": cases,
        "workers": workers,
        "mode": "formal",
    }
    parsed = CampaignSpec.from_mapping(campaign)
    if len(parsed.cases) != 900:
        raise AssertionError(f"expected 900 baseline cases, got {len(parsed.cases)}")
    return campaign


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("results/m2_formal_terrains/terrain_catalog.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/m2_dynamic_round1_proxy_baseline.json"),
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--drag-length-mm", type=float, default=100.0)
    parser.add_argument("--time-step-ms", type=float, default=1.0)
    parser.add_argument("--output-spacing-um", type=float, default=50.0)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    campaign = build_campaign(
        catalog,
        workers=args.workers,
        drag_length_m=args.drag_length_mm * 1e-3,
        time_step_s=args.time_step_ms * 1e-3,
        output_spacing_m=args.output_spacing_um * 1e-6,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(campaign, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parsed = CampaignSpec.from_mapping(campaign)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "campaign_id": parsed.campaign_id,
                "case_count": len(parsed.cases),
                "worker_count": parsed.workers,
                "round2_seed_count": 0,
                "ranking_scope": "project_model_proxy",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
