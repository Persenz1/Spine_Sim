"""Command-line entry points for production dynamic and explicit legacy M3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .dynamic_validation import (
    run_dynamic_analytic_validation,
    run_existing_m1_terrain_smoke,
)
from .validation import (
    run_legacy_analytic_validation,
    run_legacy_m1_suite_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spine-m3")
    subcommands = parser.add_subparsers(dest="command", required=True)
    analytic = subcommands.add_parser("validate-analytic")
    analytic.add_argument(
        "--output",
        type=Path,
        default=Path("results/m3_validation/dynamic_analytic_validation.json"),
    )
    existing = subcommands.add_parser("smoke-existing-m1")
    existing.add_argument(
        "catalog",
        type=Path,
    )
    existing.add_argument(
        "--output",
        type=Path,
        default=Path("results/m3_validation/existing_m1_smoke.json"),
    )
    existing.add_argument("--drag-length-mm", type=float, default=0.1)
    existing.add_argument("--seed", type=int)
    legacy = subcommands.add_parser("validate-legacy-fixed-z")
    legacy.add_argument(
        "--output",
        type=Path,
        default=Path("results/m3_validation/legacy_fixed_z_validation.json"),
    )
    smoke = subcommands.add_parser("smoke-legacy-fixed-z")
    smoke.add_argument(
        "suite_report",
        type=Path,
        nargs="?",
        default=Path("results/m1_gpu_suite/suite_report.json"),
    )
    smoke.add_argument(
        "--output",
        type=Path,
        default=Path("results/m3_validation/legacy_fixed_z_m1_smoke.json"),
    )
    smoke.add_argument("--drag-length-mm", type=float, default=1.0)
    smoke.add_argument("--path-step-um", type=float, default=50.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-analytic":
        report = run_dynamic_analytic_validation(args.output)
    elif args.command == "smoke-existing-m1":
        report = run_existing_m1_terrain_smoke(
            args.catalog,
            output_path=args.output,
            drag_length_m=args.drag_length_mm * 1e-3,
            seed=args.seed,
        )
    elif args.command == "validate-legacy-fixed-z":
        report = run_legacy_analytic_validation(args.output)
    else:
        report = run_legacy_m1_suite_smoke(
            args.suite_report,
            drag_length_m=args.drag_length_mm * 1e-3,
            path_step_m=args.path_step_um * 1e-6,
            output_path=args.output,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
