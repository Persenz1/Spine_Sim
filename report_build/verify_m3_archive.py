from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_map(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }


def verify_raw_copy(
    source_root: Path,
    archived_root: Path,
    workers: int,
) -> dict[str, object]:
    source_files = file_map(source_root)
    archived_files = file_map(archived_root)
    missing = sorted(set(source_files) - set(archived_files))
    extra = sorted(set(archived_files) - set(source_files))
    common = sorted(set(source_files) & set(archived_files))
    size_mismatches = [
        rel
        for rel in common
        if source_files[rel].stat().st_size != archived_files[rel].stat().st_size
    ]
    hash_candidates = [rel for rel in common if rel not in size_mismatches]

    def compare_one(rel: str) -> tuple[str, str, str]:
        return rel, sha256(source_files[rel]), sha256(archived_files[rel])

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        comparisons = list(pool.map(compare_one, hash_candidates))
    hash_mismatches = [
        rel
        for rel, source_hash, archived_hash in comparisons
        if source_hash != archived_hash
    ]
    all_match = not (missing or extra or size_mismatches or hash_mismatches)
    return {
        "source": str(source_root),
        "archive": str(archived_root),
        "source_file_count": len(source_files),
        "archive_file_count": len(archived_files),
        "source_bytes": sum(p.stat().st_size for p in source_files.values()),
        "archive_bytes": sum(p.stat().st_size for p in archived_files.values()),
        "missing": missing,
        "extra": extra,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "all_match": all_match,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the M3 paper-data archive.")
    parser.add_argument("--source-full-scan", type=Path, required=True)
    parser.add_argument("--source-terminal-scan", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    source_root = args.source_full_scan.resolve()
    terminal_source_root = args.source_terminal_scan.resolve()
    archive_root = args.archive_root.resolve()
    archived_raw_root = archive_root / "01_原始数据" / "full_scan"
    archived_terminal_root = archive_root / "01_原始数据" / "terminal_scan_12"
    verification_dir = archive_root / "99_校验"
    verification_dir.mkdir(parents=True, exist_ok=True)
    verification_path = verification_dir / "archive_verification.json"
    checksum_path = verification_dir / "SHA256SUMS.txt"

    full_raw_copy = verify_raw_copy(source_root, archived_raw_root, args.workers)
    terminal_raw_copy = verify_raw_copy(
        terminal_source_root,
        archived_terminal_root,
        args.workers,
    )

    analysis_manifest_path = archive_root / "02_派生数据" / "analysis_manifest.json"
    analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))
    terminal_analysis_manifest_path = (
        archive_root / "02_派生数据" / "终筛" / "terminal_analysis_manifest.json"
    )
    terminal_analysis_manifest = json.loads(
        terminal_analysis_manifest_path.read_text(encoding="utf-8")
    )
    report_path = (
        archive_root / "03_报告"
        / "M3_独立阵列细筛_承载建立与构型筛选报告_v1.md"
    )
    terminal_report_path = (
        archive_root / "03_报告" / "M3_12构型终筛分析报告_v1.md"
    )
    workbook_path = archive_root / "03_报告" / "M3_细筛论文分析数据_v1.xlsx"
    inspect_sidecars = list(archive_root.rglob("*.xlsx.inspect.ndjson"))

    raw_match = (
        bool(full_raw_copy["all_match"])
        and bool(terminal_raw_copy["all_match"])
    )
    verification = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_full_scan": str(source_root),
        "archived_full_scan": str(archived_raw_root),
        "source_terminal_scan": str(terminal_source_root),
        "archived_terminal_scan": str(archived_terminal_root),
        "raw_copy": {
            "full_scan": full_raw_copy,
            "terminal_scan": terminal_raw_copy,
            "all_match": raw_match,
        },
        "analysis": {
            "fine": {
                "case_count": analysis_manifest.get("case_count"),
                "design_count": analysis_manifest.get("design_count"),
                "expected_case_count": 43200,
                "expected_design_count": 96,
                "counts_match": (
                    analysis_manifest.get("case_count") == 43200
                    and analysis_manifest.get("design_count") == 96
                ),
            },
            "terminal": {
                "case_count": terminal_analysis_manifest.get("case_count"),
                "design_count": terminal_analysis_manifest.get("design_count"),
                "expected_case_count": 10800,
                "expected_design_count": 12,
                "counts_match": (
                    terminal_analysis_manifest.get("case_count") == 10800
                    and terminal_analysis_manifest.get("design_count") == 12
                ),
            },
        },
        "deliverables": {
            "markdown_report_exists": report_path.is_file(),
            "terminal_markdown_report_exists": terminal_report_path.is_file(),
            "workbook_exists": workbook_path.is_file(),
            "xlsx_inspect_sidecars": [
                p.relative_to(archive_root).as_posix() for p in inspect_sidecars
            ],
            "terrain_matrix_library_copied": False,
            "terrain_metadata_only": [
                "01_原始数据/M1_terrain_catalog.json",
                "01_原始数据/M1_generation_report.json",
            ],
        },
    }
    verification["all_checks_pass"] = (
        raw_match
        and verification["analysis"]["fine"]["counts_match"]
        and verification["analysis"]["terminal"]["counts_match"]
        and verification["deliverables"]["markdown_report_exists"]
        and verification["deliverables"]["terminal_markdown_report_exists"]
        and verification["deliverables"]["workbook_exists"]
        and not inspect_sidecars
    )
    verification_path.write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    archive_files = [
        path
        for path in archive_root.rglob("*")
        if path.is_file() and path != checksum_path
    ]
    archive_files.sort(key=lambda p: p.relative_to(archive_root).as_posix())
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        archive_hashes = list(pool.map(sha256, archive_files))
    checksum_lines = [
        f"{digest}  {path.relative_to(archive_root).as_posix()}"
        for path, digest in zip(archive_files, archive_hashes, strict=True)
    ]
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    if not verification["all_checks_pass"]:
        raise SystemExit(json.dumps(verification, ensure_ascii=False, indent=2))
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
