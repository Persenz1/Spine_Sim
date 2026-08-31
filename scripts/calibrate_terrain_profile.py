"""从一次表面扫描中提取可复现的材料剖面标定候选。

本命令不会修改原始扫描，也不会静默覆盖生产剖面。输出 JSON 仅是一份可审查的
新版本输入，状态固定为单样本拟合；必须经过多试件总体验证后才能升级为标定值。
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
    """定义扫描格式、单位、无效值语义和来源信息等命令行参数。"""

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
    """加载并整平扫描，计算描述符，返回不落盘的标定候选文档。"""

    # 只记录用户实际提供的可追踪字段，避免用 null 冒充已知来源信息。
    provenance = {
        key: value
        for key, value in {
            "source_id": args.source_id,
            "doi": args.doi,
            "license": args.license,
        }.items()
        if value is not None
    }
    # 所有微米输入在边界处转换为米；模块内部统一采用 SI 制。
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
    # 曲线描述符也一并输出，供后续比较相关尺度和高度分布，而不只比较 RMS。
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
        # 明确禁止下游把这份单扫描输出当作总体已验证参数。
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
    """执行 CLI、原子化前置目录创建，并把预期错误映射为退出码 2。"""

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
