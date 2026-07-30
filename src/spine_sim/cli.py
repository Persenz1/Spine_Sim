"""Command line entry points for environment, case, campaign and recovery."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from spine_sim.core.config import CampaignSpec
from spine_sim.io.results import open_result_store
from spine_sim.runtime.backend import discover_backend
from spine_sim.runtime.runner import CampaignRunner
from spine_sim.validation.environment import validate_environment


def _load_campaign(path: Path) -> tuple[dict, CampaignSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw, CampaignSpec.from_mapping(raw)


def _runner(args: argparse.Namespace) -> CampaignRunner:
    raw, campaign = _load_campaign(args.config)
    if args.command == "run-case":
        if args.case_id:
            cases = tuple(case for case in campaign.cases if case.case_id == args.case_id)
            if not cases:
                raise SystemExit(f"unknown case ID: {args.case_id}")
        else:
            cases = campaign.cases[:1]
        campaign = CampaignSpec(
            campaign.name,
            campaign.module_version,
            campaign.callable,
            cases,
            workers=1,
            mode="small",
        )
    backend = discover_backend()
    target = args.output / campaign.campaign_id
    runner = CampaignRunner(campaign, target, backend)
    if not (target / "manifest.json").exists():
        runner.initialize(raw)
    return runner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spine-sim")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-env")
    validate.add_argument("--output", type=Path)

    for name in ("run-case", "run-campaign", "resume", "retry-failed"):
        command = sub.add_parser(name)
        command.add_argument("config", type=Path)
        command.add_argument("--output", type=Path, default=Path("results"))
        command.add_argument("--workers", type=int)
        if name == "run-case":
            command.add_argument("--case-id")

    summary = sub.add_parser("summarize")
    summary.add_argument("campaign_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-env":
        report = validate_environment(writable_path=args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 2
    if args.command == "summarize":
        records = open_result_store(args.campaign_dir).list_records()
        counts: dict[str, int] = {}
        for record in records:
            counts[record.run_state] = counts.get(record.run_state, 0) + 1
        print(json.dumps({"case_count": len(records), "status_counts": counts}, indent=2))
        return 0

    runner = _runner(args)
    if args.command == "run-case":
        records = runner.run(workers=1)
    else:
        records = runner.run(
            resume=args.command == "resume",
            failed_only=args.command == "retry-failed",
            workers=args.workers,
        )
    counts: dict[str, int] = {}
    for record in records:
        counts[record.run_state] = counts.get(record.run_state, 0) + 1
    print(json.dumps({"campaign_dir": str(runner.store.root), "status_counts": counts}, indent=2))
    return 0 if counts.get("execution_error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
