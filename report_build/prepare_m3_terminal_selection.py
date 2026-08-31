from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LEGACY_SELECTION_PATH = (
    Path(__file__).resolve().parents[1]
    / "provenance"
    / "legacy_simulation"
    / "terminal_input_selected_designs.json"
)


def load_archived_designs() -> dict[str, dict[str, object]]:
    payload = json.loads(LEGACY_SELECTION_PATH.read_text(encoding="utf-8"))
    designs = payload.get("selected_designs")
    if not isinstance(designs, list):
        raise RuntimeError(
            f"archived terminal selection has no design list: {LEGACY_SELECTION_PATH}"
        )

    by_id: dict[str, dict[str, object]] = {}
    for design in designs:
        if not isinstance(design, dict) or not isinstance(design.get("design_id"), str):
            raise RuntimeError(
                f"archived terminal selection contains an invalid design: {design!r}"
            )
        by_id[design["design_id"]] = design
    return by_id


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the reviewed 12-design M3 terminal selection."
    )
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    with args.selection_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected_ids = [row["design_id"] for row in rows]
    if len(selected_ids) != 12 or len(set(selected_ids)) != 12:
        raise RuntimeError("terminal selection must contain exactly 12 unique IDs")

    by_id = load_archived_designs()
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise RuntimeError(f"unknown design IDs: {unknown}")
    selected = [by_id[design_id] for design_id in selected_ids]

    selection = {
        "schema_version": "m3-full-selected-designs-v1",
        "stage": "reviewed_terminal_input",
        "selected_design_ids": selected_ids,
        "selected_designs": selected,
        "selection_count": len(selected),
        "selection_rule": (
            "mechanism-reviewed selection: five advantage candidates and "
            "seven matched/mechanism controls"
        ),
        "roles": {
            row["design_id"]: {
                "role": row["role"],
                "mechanism": row["mechanism"],
                "reason": row["reason"],
            }
            for row in rows
        },
    }
    write_json_atomic(
        args.output_root / "fine" / "selected_designs.json",
        selection,
    )
    write_json_atomic(
        args.output_root / "terminal_plan.json",
        {
            "schema_version": "m3-terminal-plan-v1",
            "design_count": 12,
            "terrain_condition_count": 300,
            "preloads_N": [0.5, 1.0, 2.0],
            "expected_case_count": 10800,
            "path_length_mm": 10.0,
            "dx_mm": 0.05,
            "station_count_including_origin": 201,
            "selection_source": str(args.selection_csv.resolve()),
            "selection_file": str(
                (
                    args.output_root / "fine" / "selected_designs.json"
                ).resolve()
            ),
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "selection_count": len(selected),
                "selected_design_ids": selected_ids,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
