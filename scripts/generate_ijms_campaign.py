"""从一个生产 case、设计覆盖项和固定样本表生成配对 campaign 分片。

示例（仓库根目录）：
  python scripts/generate_ijms_campaign.py --base examples/ijms_campaign.json \
      --designs designs.json --samples samples.json --output-dir campaigns/run1

designs.json: {"designs": [{"name": "baseline", "parameters": {}}, ...]}
samples.json: [{"surface_seed": 17, "placement_seed": 3, "split": "test",
                "placement_xy_m": [0.0, 0.0]}, ...]

参数字典递归覆盖，列表整体替换；module、版本和 callable 继承 base。
脚本不运行仿真、不随机重采样、不改地形文件。placement_seed 是记录值，实际
落点需通过 placement_xy_m 指定，由生产 adapter 加到 path.start_xy_m。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from spine_sim.core.config import BaseCaseSpec, CampaignSpec
from spine_sim.core.identity import stable_hash


def _merge_parameters(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_parameters(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def write_campaign_shards(
    base_campaign: Mapping[str, Any], designs: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]], *, output_directory: Path,
    shard_size: int = 1000,
) -> tuple[int, int]:
    """返回 (case 数, 分片数)，每次仅物化一个分片，使用现有 case identity。"""

    base = CampaignSpec.from_mapping(base_campaign)
    if len(base.cases) != 1:
        raise ValueError("base campaign must contain exactly one template case")
    if shard_size < 1 or not designs or not samples:
        raise ValueError("shard_size, designs, and samples must be non-empty/positive")
    template = base.cases[0]
    prepared_designs: list[BaseCaseSpec] = []
    design_hashes: set[str] = set()
    for design in designs:
        if set(design) != {"name", "parameters"}:
            raise ValueError("each design must contain only name and parameters; module/version overrides are not supported")
        parameters = _merge_parameters(template.parameters, design["parameters"])
        parameters.pop("sample", None)
        parameter_hash = stable_hash(parameters)
        if parameter_hash in design_hashes:
            raise ValueError(f"duplicate effective design parameters: {design['name']}")
        design_hashes.add(parameter_hash)
        prepared_designs.append(replace(
            template, parameters=parameters, tags=(*template.tags, f"design:{design['name']}"),
        ))
    sample_hashes: set[str] = set()
    for sample in samples:
        for seed_name in ("surface_seed", "placement_seed"):
            if type(sample.get(seed_name)) is not int:
                raise ValueError(f"sample {seed_name} must be an integer")
        if sample.get("split") not in {"train", "validation", "test"}:
            raise ValueError("sample split must be train, validation, or test")
        # A split label cannot turn the same recorded realization/placement into
        # an independent held-out sample.
        sample_hash = stable_hash({key: value for key, value in sample.items() if key != "split"})
        if sample_hash in sample_hashes:
            raise ValueError("duplicate sample realization/placement (possibly across dataset splits)")
        sample_hashes.add(sample_hash)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    case_count = len(prepared_designs) * len(samples)
    shard_count = (case_count + shard_size - 1) // shard_size
    # Fail before writing when reusing an existing shard destination. No manifest
    # or extra process artifacts are produced.
    if any((output_directory / f"campaign_{index:05d}.json").exists() for index in range(1, shard_count + 1)):
        raise FileExistsError("campaign shard already exists; choose a new output directory")

    cases: list[BaseCaseSpec] = []
    shard_index = 0

    def write_shard() -> None:
        nonlocal shard_index
        shard_index += 1
        campaign = replace(base, name=f"{base.name}-{shard_index:05d}", cases=tuple(cases))
        output_path = output_directory / f"campaign_{shard_index:05d}.json"
        with output_path.open("x", encoding="utf-8") as stream:
            json.dump(asdict(campaign), stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")

    for design in prepared_designs:
        for sample in samples:
            parameters = dict(design.parameters)
            parameters["sample"] = dict(sample)
            cases.append(replace(design, parameters=parameters, tags=(*design.tags, sample["split"])))
            if len(cases) == shard_size:
                write_shard()
                cases.clear()
    if cases:
        write_shard()
    return case_count, shard_index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, type=Path, help="single-case production campaign JSON")
    parser.add_argument("--designs", required=True, type=Path, help='JSON object with a "designs" list')
    parser.add_argument("--samples", required=True, type=Path, help="JSON list of fixed paired surface/placement samples")
    parser.add_argument("--output-dir", required=True, type=Path, help="dedicated output directory for campaign shards")
    parser.add_argument("--shard-size", type=int, default=1000, help="maximum cases per campaign (default: 1000)")
    args = parser.parse_args(argv)

    def read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    case_count, shard_count = write_campaign_shards(
        read_json(args.base), read_json(args.designs)["designs"], read_json(args.samples),
        output_directory=args.output_dir, shard_size=args.shard_size,
    )
    print(f"Wrote {case_count} cases in {shard_count} campaign shards to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
