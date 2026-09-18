"""``spine-sim`` 命令行入口：环境检查、case/campaign 运行与恢复。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from spine_sim.core.config import CampaignSpec
from spine_sim.io.results import open_result_store
from spine_sim.runtime.backend import (
    BackendConfig,
    discover_backend,
    validate_environment,
)
from spine_sim.runtime.runner import CampaignRunner


def _expected_case_ids(campaign_dir: Path) -> set[str] | None:
    """从受 provenance 保护的 lineage 读取精确预期 case ID 集。"""

    path = campaign_dir / "lineage.json"
    try:
        lineage = json.loads(path.read_text(encoding="utf-8"))
        case_lineage = lineage["case_lineage"]
        if not isinstance(case_lineage, Mapping) or not case_lineage or not all(
            isinstance(case_id, str) and case_id
            for case_id in case_lineage
        ):
            return None
        return set(case_lineage)
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _runner(args: argparse.Namespace) -> CampaignRunner:
    """读取配置、按子命令裁剪 case，并创建或恢复对应 runner。"""

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    campaign = CampaignSpec.from_mapping(raw)
    if args.command == "run-case":
        # run-case 始终退化为单 case、单进程 small 模式，确保命令语义明确。
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
    backend = discover_backend(
        BackendConfig(
            preference=args.backend,
            device_index=args.device_index,
        )
    )
    target = args.output / campaign.campaign_id
    runner = CampaignRunner(campaign, target, backend)
    if args.command == "retry-failed" and not (target / "manifest.json").is_file():
        raise SystemExit("retry-failed requires an existing campaign manifest")
    runner.prepare(raw)
    return runner


def build_parser() -> argparse.ArgumentParser:
    """声明所有子命令和参数，不在解析阶段执行文件或后端操作。"""

    parser = argparse.ArgumentParser(prog="spine-sim")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-env")
    validate.add_argument("--output", type=Path)

    for name in ("run-case", "run-campaign", "resume", "retry-failed"):
        command = sub.add_parser(name)
        command.add_argument("config", type=Path)
        command.add_argument("--output", type=Path, default=Path("results"))
        command.add_argument(
            "--backend", choices=("auto", "cpu", "cuda"), default="auto"
        )
        command.add_argument("--device-index", type=int, default=0)
        if name == "run-case":
            command.add_argument("--case-id")
        else:
            command.add_argument("--workers", type=int)

    summary = sub.add_parser("summarize")
    summary.add_argument("campaign_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """分派子命令，并用退出码区分成功、case 错误和环境错误。"""

    args = build_parser().parse_args(argv)
    if args.command == "validate-env":
        report = validate_environment(writable_path=args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 2
    if args.command == "summarize":
        if not (args.campaign_dir / "manifest.json").is_file():
            raise SystemExit("summarize requires an existing campaign manifest")
        records = open_result_store(args.campaign_dir).list_records()
        expected_ids = _expected_case_ids(args.campaign_dir)
        actual_ids = {record.case_id for record in records}
        missing_ids = (
            None if expected_ids is None else expected_ids - actual_ids
        )
        unexpected_ids = (
            None if expected_ids is None else actual_ids - expected_ids
        )
        counts: dict[str, int] = {}
        for record in records:
            # 未由 lineage 声明的完整目录也不能出现在成功计数中。
            state = (
                "execution_error"
                if unexpected_ids is not None
                and record.case_id in unexpected_ids
                else record.run_state
            )
            counts[state] = counts.get(state, 0) + 1
        print(
            json.dumps(
                {
                    "case_count": len(records),
                    "expected_case_count": (
                        None if expected_ids is None else len(expected_ids)
                    ),
                    "case_ids_match": expected_ids == actual_ids,
                    "missing_case_count": (
                        None if missing_ids is None else len(missing_ids)
                    ),
                    "unexpected_case_count": (
                        None if unexpected_ids is None else len(unexpected_ids)
                    ),
                    "status_counts": counts,
                },
                indent=2,
            )
        )
        return (
            0
            if expected_ids == actual_ids
            and all(record.run_state == "complete" for record in records)
            else 1
        )

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
    return (
        0
        if {record.case_id for record in records}
        == {case.case_id for case in runner.campaign.cases}
        and all(record.run_state == "complete" for record in records)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
