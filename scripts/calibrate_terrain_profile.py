"""Extract a reproducible material-profile calibration artifact from a scan.

This command does not modify the raw scan or silently update a production
profile.  The JSON output is a reviewable input for a new profile revision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from spine_sim.terrain.descriptors import compute_descriptors
from spine_sim.terrain.measured import load_measured_surface


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--material", required=True, choices=("sandpaper", "red_brick", "concrete")
    )
    parser.add_argument("--subtype", required=True)
    parser.add_argument("--format", default="auto")
    parser.add_argument("--height-unit")
    parser.add_argument("--lateral-unit", default="m")
    parser.add_argument("--spacing-x-um", type=float)
    parser.add_argument("--spacing-y-um", type=float)
    parser.add_argument("--invalid-value", action="append", type=float, default=[])
    parser.add_argument("--dataset-zero-is-invalid", action="store_true")
    parser.add_argument("--invalid-margin-samples", type=int, default=0)
    parser.add_argument(
        "--level", choices=("robust_plane", "mean", "none"), default="robust_plane"
    )
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--doi")
    parser.add_argument("--license")
    return parser


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    provenance = {
        key: value
        for key, value in {
            "source_id": args.source_id,
            "doi": args.doi,
            "license": args.license,
        }.items()
        if value is not None
    }
    surface = load_measured_surface(
        args.path,
        format=args.format,
        height_unit=args.height_unit,
        lateral_unit=args.lateral_unit,
        spacing_x_m=(
            None if args.spacing_x_um is None else args.spacing_x_um * 1e-6
        ),
        spacing_y_m=(
            None if args.spacing_y_um is None else args.spacing_y_um * 1e-6
        ),
        invalid_values=args.invalid_value,
        dataset_zero_is_invalid=args.dataset_zero_is_invalid,
        invalid_margin_samples=args.invalid_margin_samples,
        level=args.level,
        provenance=provenance,
    )
    descriptors = compute_descriptors(
        surface.height_m,
        dx_m=surface.dx_m,
        dy_m=surface.dy_m,
        valid_mask=surface.valid_mask,
        include_curves=True,
    )
    return {
        "schema_version": "material-profile-calibration-v1",
        "material": args.material,
        "subtype": args.subtype,
        "status": "single_sample_fit",
        "population_validation": False,
        "source": surface.metadata,
        "descriptors": descriptors,
        "suggested_profile_fields": {
            "rms_height_m": descriptors["height"]["rms_about_mean_m"],
            "correlation_x_m": descriptors["spatial"]["correlation_length_x_m"],
            "correlation_y_m": descriptors["spatial"]["correlation_length_y_m"],
            "height_quantiles_m": descriptors["height"]["quantiles_m"],
        },
        "review_required": [
            "Confirm surface type, batch, orientation, z sign, and invalid-value semantics.",
            "Compare multiple specimens before changing status to validated.",
            "Keep train/calibration and held-out validation patches separate."
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = calibrate(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"output": str(args.output.resolve())}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
