"""Auditable engineering-proxy parameters and bounded sensitivity scenarios."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any

from spine_sim.core.identity import identity

from .models import ArrayConfiguration, M3_MODULE_VERSION


PROXY_POLICY_VERSION = "engineering_proxy_v1"

BASELINE_PROXY: dict[str, float | str] = {
    "young_modulus_pa": 200e9,
    "poisson_ratio": 0.29,
    "density_kg_m3": 7850.0,
    "yield_strength_pa": 800e6,
    "static_friction": 0.30,
    "kinetic_friction": 0.20,
    "axial_damping_ratio": 0.05,
    "transverse_damping_ratio": 0.05,
    "restitution_coefficient": 0.0,
    "contact_position_correction": 0.20,
    "backplate_carriage_equivalent_mass_kg": 0.050,
    "backplate_material_density_kg_m3": 2700.0,
    "backplate_thickness_m": 2e-3,
    "backplate_edge_margin_m": 4e-3,
    "backplate_vertical_damping_ratio": 0.10,
    "settlement_damping_scale": 10.0,
    "provenance": (
        "engineering_proxy_not_experimentally_calibrated"
    ),
}

SENSITIVITY_LEVELS: dict[str, tuple[float, ...]] = {
    "yield_strength_pa": (600e6, 800e6, 1200e6),
    "static_friction": (0.20, 0.30, 0.45),
    "kinetic_friction": (0.15, 0.20, 0.35),
    "pin_modal_damping_ratio": (0.02, 0.05, 0.10),
    "restitution_coefficient": (0.0, 0.10, 0.20),
    "contact_position_correction": (0.20, 0.50, 1.0),
    "backplate_mass_scale": (0.5, 1.0, 2.0),
    "backplate_vertical_damping_ratio": (0.05, 0.10, 0.20),
    "settlement_damping_scale": (5.0, 10.0, 20.0),
}


@dataclass(frozen=True)
class EngineeringProxyScenario:
    name: str
    static_friction: float = 0.30
    kinetic_friction: float = 0.20
    pin_modal_damping_ratio: float = 0.05
    restitution_coefficient: float = 0.0
    contact_position_correction: float = 0.20
    backplate_mass_scale: float = 1.0
    backplate_vertical_damping_ratio: float = 0.10
    settlement_damping_scale: float = 10.0
    yield_strength_pa: float = 800e6

    @property
    def scenario_id(self) -> str:
        return identity(
            "m3_proxy_scenario",
            asdict(self),
            module_version=M3_MODULE_VERSION,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "scenario_id": self.scenario_id,
            "proxy_policy_version": PROXY_POLICY_VERSION,
            "formal_calibration": False,
        }


def build_engineering_proxy_scenarios() -> tuple[EngineeringProxyScenario, ...]:
    """Return a one-factor-at-a-time set without a sensitivity Cartesian blowup."""

    baseline = EngineeringProxyScenario("baseline")
    scenarios = [
        baseline,
        EngineeringProxyScenario(
            "friction_low",
            static_friction=0.20,
            kinetic_friction=0.15,
        ),
        EngineeringProxyScenario(
            "friction_high",
            static_friction=0.45,
            kinetic_friction=0.35,
        ),
        EngineeringProxyScenario(
            "pin_damping_low",
            pin_modal_damping_ratio=0.02,
        ),
        EngineeringProxyScenario(
            "pin_damping_high",
            pin_modal_damping_ratio=0.10,
        ),
        EngineeringProxyScenario(
            "restitution_0p1",
            restitution_coefficient=0.10,
        ),
        EngineeringProxyScenario(
            "restitution_0p2",
            restitution_coefficient=0.20,
        ),
        EngineeringProxyScenario(
            "position_correction_0p5",
            contact_position_correction=0.50,
        ),
        EngineeringProxyScenario(
            "position_correction_1p0",
            contact_position_correction=1.0,
        ),
        EngineeringProxyScenario(
            "moving_mass_half",
            backplate_mass_scale=0.5,
        ),
        EngineeringProxyScenario(
            "moving_mass_double",
            backplate_mass_scale=2.0,
        ),
        EngineeringProxyScenario(
            "plate_damping_low",
            backplate_vertical_damping_ratio=0.05,
        ),
        EngineeringProxyScenario(
            "plate_damping_high",
            backplate_vertical_damping_ratio=0.20,
        ),
        EngineeringProxyScenario(
            "settlement_damping_low",
            settlement_damping_scale=5.0,
        ),
        EngineeringProxyScenario(
            "settlement_damping_high",
            settlement_damping_scale=20.0,
        ),
        EngineeringProxyScenario(
            "yield_low",
            yield_strength_pa=600e6,
        ),
        EngineeringProxyScenario(
            "yield_high",
            yield_strength_pa=1200e6,
        ),
    ]
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise AssertionError("engineering proxy scenario IDs must be unique")
    return tuple(scenarios)


def nominal_vertical_stiffness_n_m(
    configuration: ArrayConfiguration,
) -> float:
    """Estimate flat-contact vertical stiffness from all nominal pin modes."""

    total = 0.0
    for parameters in configuration.pin_parameters:
        axial_compliance = parameters.axial_compliance_m_n
        if parameters.axial_mode.value == "spring":
            axial_compliance += 1.0 / float(
                parameters.spring_stiffness_n_m
            )
        axial_stiffness = 1.0 / axial_compliance
        transverse_stiffness = 1.0 / parameters.transverse_compliance_m_n
        vertical_compliance = (
            parameters.axis_xz[1] ** 2 / axial_stiffness
            + parameters.transverse_xz[1] ** 2 / transverse_stiffness
        )
        total += 1.0 / vertical_compliance
    return float(total)


def estimate_backplate_dynamics(
    configuration: ArrayConfiguration,
    scenario: EngineeringProxyScenario | None = None,
) -> dict[str, float | str]:
    """Derive moving mass and damping from explicit plate/carriage assumptions."""

    selected = scenario or EngineeringProxyScenario("baseline")
    margin = float(BASELINE_PROXY["backplate_edge_margin_m"])
    width_x = (configuration.nx - 1) * configuration.spacing_m + 2.0 * margin
    width_y = (configuration.ny - 1) * configuration.spacing_m + 2.0 * margin
    plate_mass = (
        width_x
        * width_y
        * float(BASELINE_PROXY["backplate_thickness_m"])
        * float(BASELINE_PROXY["backplate_material_density_kg_m3"])
    )
    baseline_mass = (
        float(BASELINE_PROXY["backplate_carriage_equivalent_mass_kg"])
        + plate_mass
    )
    moving_mass = baseline_mass * selected.backplate_mass_scale
    vertical_stiffness = nominal_vertical_stiffness_n_m(configuration)
    damping = (
        2.0
        * selected.backplate_vertical_damping_ratio
        * math.sqrt(moving_mass * vertical_stiffness)
    )
    return {
        "backplate_mass_kg": float(moving_mass),
        "backplate_vertical_damping_n_s_m": float(damping),
        "nominal_vertical_stiffness_n_m": float(vertical_stiffness),
        "backplate_plate_mass_kg": float(plate_mass),
        "backplate_carriage_equivalent_mass_kg": float(
            BASELINE_PROXY["backplate_carriage_equivalent_mass_kg"]
        ),
        "backplate_vertical_damping_ratio": float(
            selected.backplate_vertical_damping_ratio
        ),
        "backplate_mass_scale": float(selected.backplate_mass_scale),
        "mass_model": (
            "constant_equivalent_carriage_plus_geometry_scaled_aluminum_plate"
        ),
    }


def engineering_proxy_manifest() -> dict[str, Any]:
    scenarios = build_engineering_proxy_scenarios()
    return {
        "proxy_policy_version": PROXY_POLICY_VERSION,
        "baseline": dict(BASELINE_PROXY),
        "sensitivity_levels": {
            key: list(values)
            for key, values in SENSITIVITY_LEVELS.items()
        },
        "scenario_count": len(scenarios),
        "scenarios": [scenario.as_dict() for scenario in scenarios],
        "interpretation": (
            "trend robustness proxy only; not absolute-force calibration"
        ),
        "formal_ranking_eligible": False,
    }
