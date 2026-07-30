"""Prepare the complete M3 design or one explicitly bounded campaign shard.

This command never starts a campaign.  Without ``--catalog`` it writes only the
hardware/loading design manifest.  With a catalog it first requires the exact
300-condition M1 inventory.  Cases are materialized only when a family, seed
range and one preload are all supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spine_sim.array.design import (
    DRAG_LENGTH_M,
    TERRAIN_FAMILIES,
    TOTAL_PRELOADS_N,
    build_base_hardware,
    build_campaign_shard,
    build_full_array_design,
    validate_terrain_catalog,
)
from spine_sim.array.models import M3_MODULE_VERSION
from spine_sim.array.proxy_parameters import engineering_proxy_manifest
from spine_sim.core.config import CampaignSpec
from spine_sim.core.identity import stable_hash
from spine_sim.io.results import atomic_write_json, utc_now
from prepare_m3_track_cache import (
    validate_complete_track_cache_manifest,
)


def _manifest(catalog: dict | None) -> dict:
    base_hardware = build_base_hardware()
    array_design = build_full_array_design(include_gradient_80_to_60=True)
    fixed = [
        row for row in array_design if row["angle_layout"] == "fixed"
    ]
    gradient = [
        row
        for row in array_design
        if row["angle_layout"] == "gradient_80_to_60"
    ]
    terrain_count = 300
    catalog_evidence = None
    if catalog is not None:
        conditions = validate_terrain_catalog(
            catalog,
            require_formal_300=True,
        )
        terrain_count = len(conditions)
        catalog_evidence = {
            "terrain_catalog_id": catalog.get(
                "terrain_catalog_id",
                stable_hash(
                    [
                        condition["terrain_condition_id"]
                        for condition in conditions
                    ]
                ),
            ),
            "library_root": catalog["library_root"],
            "condition_count": len(conditions),
            "family_counts": {
                family: sum(
                    condition["terrain_family"] == family
                    for condition in conditions
                )
                for family in TERRAIN_FAMILIES
            },
            "all_full_hashes_verified": True,
            "source": "M1 terrain catalog; M3 generated no terrain",
        }
    return {
        "schema_version": "2",
        "created_at_utc": utc_now(),
        "m3_module_version": M3_MODULE_VERSION,
        "status": (
            "complete_design_manifest_not_executed"
            if catalog is not None
            else "complete_hardware_plan_waiting_for_m1_300_catalog"
        ),
        "formal_campaign_started": False,
        "experiment_goal": (
            "compare pull-force trends over a 100 mm +x drag while one "
            "constant total unit preload remains applied"
        ),
        "base_hardware_count": len(base_hardware),
        "fixed_array_configuration_count": len(fixed),
        "gradient_80_to_60_configuration_count": len(gradient),
        "total_array_configuration_count": len(array_design),
        "terrain_condition_count": terrain_count,
        "total_preloads_n": list(TOTAL_PRELOADS_N),
        "drag_length_m": DRAG_LENGTH_M,
        "fixed_case_count": (
            len(fixed) * terrain_count * len(TOTAL_PRELOADS_N)
        ),
        "gradient_case_count": (
            len(gradient) * terrain_count * len(TOTAL_PRELOADS_N)
        ),
        "total_planned_case_count": (
            len(array_design) * terrain_count * len(TOTAL_PRELOADS_N)
        ),
        "terrain_catalog_evidence": catalog_evidence,
        "pairing_rule": (
            "every array_configuration_id appears exactly once for every "
            "terrain_condition_id and loading_protocol_id"
        ),
        "case_identity_rule": (
            "case identity includes full hardware/array configuration, M1 "
            "terrain condition/data hash, and the 0.5/1/2 N 100 mm protocol"
        ),
        "engineering_proxy_policy": engineering_proxy_manifest(),
        "default_output_policy": {
            "summary": "all cases",
            "aggregate_trace": (
                "designated seeds, anomalies, representative configurations"
            ),
            "full_pin_trace": (
                "designated seeds, anomalies and final candidates only"
            ),
        },
        "required_sharding": (
            "one terrain family + bounded seed range + one preload per shard"
        ),
        "formal_ranking_eligible": False,
        "formal_ranking_blockers": [
            "dynamic/contact parameters are not experimentally calibrated",
            "time-step convergence is not closed for all candidates",
            "10/5 um convergence is not closed for final candidates",
            *(
                ["the planned 300-condition M1 catalog must exist and verify"]
                if catalog is None
                else []
            ),
        ],
        "base_hardware": base_hardware,
        "array_design": array_design,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/m3_complete_design_manifest.json"),
    )
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--terrain-family", choices=TERRAIN_FAMILIES)
    parser.add_argument("--seed-min", type=int)
    parser.add_argument("--seed-max", type=int)
    parser.add_argument("--preload-n", type=float, choices=TOTAL_PRELOADS_N)
    parser.add_argument(
        "--output-level",
        choices=("summary", "aggregate_trace", "full_pin_trace"),
        default="summary",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    shard_fields = (
        args.terrain_family,
        args.seed_min,
        args.seed_max,
        args.preload_n,
    )
    materialize_shard = any(value is not None for value in shard_fields)
    if materialize_shard and (
        args.catalog is None or any(value is None for value in shard_fields)
    ):
        parser.error(
            "a shard requires --catalog, --terrain-family, --seed-min, "
            "--seed-max and --preload-n together"
        )

    catalog = (
        json.loads(args.catalog.read_text(encoding="utf-8"))
        if args.catalog is not None
        else None
    )
    if materialize_shard:
        assert catalog is not None
        validate_complete_track_cache_manifest(catalog)
        document = build_campaign_shard(
            catalog,
            terrain_family=args.terrain_family,
            seed_min=args.seed_min,
            seed_max=args.seed_max,
            preload_n=args.preload_n,
            output_level=args.output_level,
            workers=args.workers,
            require_formal_300=True,
        )
        parsed = CampaignSpec.from_mapping(document)
        status = {
            "kind": "campaign_shard",
            "case_count": len(parsed.cases),
            "campaign_id": parsed.campaign_id,
        }
    else:
        document = _manifest(catalog)
        status = {
            "kind": "design_manifest",
            "configuration_count": document[
                "total_array_configuration_count"
            ],
            "planned_case_count": document["total_planned_case_count"],
        }
    atomic_write_json(args.output, document)
    print(
        json.dumps(
            {"output": str(args.output.resolve()), **status},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
