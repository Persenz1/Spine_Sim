"""Prepare a non-executable M3 round-one balanced design draft."""

from __future__ import annotations

import argparse
from pathlib import Path

from spine_sim.array.design import (
    build_candidate_pool,
    level_counts,
    screening_gate_status,
    select_balanced_candidates,
)
from spine_sim.contact import AxialMode, SpineParameters
from spine_sim.io.results import atomic_write_json, utc_now


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/m3_round1_design_draft.json"),
    )
    parser.add_argument("--target-count", type=int, default=100)
    args = parser.parse_args()

    fixture_pack = {
        "parameter_pack_id": "stage_i_fixture_pack_not_for_formal_ranking",
        "spine": SpineParameters(
            tip_radius_m=50e-6,
            diameter_m=0.8e-3,
            exposed_length_m=4e-3,
            installation_angle_deg=70.0,
            axial_mode=AxialMode.SPRING,
            spring_stiffness_n_m=2000.0,
            static_friction=0.30,
            kinetic_friction=0.20,
            rod_clearance_mode="unclosed",
        ).as_dict(),
        "approval_status": "fixture_only_not_user_approved",
    }
    pool = build_candidate_pool([fixture_pack])
    selected = select_balanced_candidates(pool, args.target_count)
    gate = screening_gate_status(
        full_chain_manifest_present=False,
        m2_formal_round1_completed=False,
        m2_parameter_packs_approved=False,
        explicit_m3_round1_approval=False,
    )
    document = {
        "schema_version": "1",
        "created_at_utc": utc_now(),
        "status": "design_algorithm_fixture_draft_not_executable_campaign",
        "gate": gate,
        "selection_algorithm": (
            "deterministic greedy balanced categorical coverage with maximin "
            "Jaccard distance and explicit required second-order interaction tokens"
        ),
        "selection_rationale": [
            "balance main hardware levels before repeating them",
            "cover installation-mode*scale, stiffness*spacing, angle*direction, "
            "gradient*nx, diameter*stiffness and tip*pack interactions",
            "keep 2x5 and 5x2 as distinct configurations",
            "do not combine the hardware matrix with terrain seeds here",
        ],
        "fixture_parameter_pack": fixture_pack,
        "candidate_pool_count": len(pool),
        "selected_count": len(selected),
        "selected_level_counts": level_counts(selected),
        "design_matrix": selected,
        "formal_campaign_started": False,
        "required_rebuild_after_gate_opens": (
            "replace the fixture pack with 4-6 representatives from the "
            "user-approved M2 pack file, then rerun this deterministic selector"
        ),
    }
    atomic_write_json(args.output, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
