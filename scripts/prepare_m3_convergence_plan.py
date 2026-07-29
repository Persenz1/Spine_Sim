"""Write the bounded M3 convergence plan without running any simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spine_sim.array.convergence import convergence_plan_manifest
from spine_sim.io.results import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/m3_convergence_plan.json"),
    )
    args = parser.parse_args()
    manifest = convergence_plan_manifest()
    atomic_write_json(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sentinel_count": manifest["sentinel_count"],
                "variant_count": manifest["variant_count"],
                "case_count_per_terrain_condition": manifest[
                    "case_count_per_terrain_condition"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
