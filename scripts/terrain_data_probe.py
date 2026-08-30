"""探测、下载并解析公开的表面形貌证据。

这个 Phase 0–1 工具有意不生成地形，也不修改 M1 运行接口。当前首个受支持的
实测格式是 Sandpaper Wind Turbine Blade Benchmark Dataset
（DOI 10.17632/hcgcnm269w.2）采用的 Hirox CSV 导出格式。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sys
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray

from spine_sim.io.files import atomic_write_json, sha256_file


LOGGER = logging.getLogger("terrain_data_probe")
DATASET_ID = "hcgcnm269w"
DATASET_VERSION = 2
DATASET_DOI = "10.17632/hcgcnm269w.2"
LANDING_PAGE = "https://data.mendeley.com/datasets/hcgcnm269w/2"
PUBLIC_API_URL = "https://data.mendeley.com/public-api/datasets/hcgcnm269w"
DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/"
    f"{DATASET_ID}/files/{{file_id}}/file_downloaded"
)
PARSER_VERSION = "hirox-csv-v1"
DEFAULT_MAX_BYTES = 100 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class PublicFile:
    """公开文件的冻结 ID、字节数和 SHA-256 期望值。"""

    file_id: str
    size_bytes: int
    sha256: str

    @property
    def download_url(self) -> str:
        """根据不可变文件 ID 构造下载地址。"""

        return DOWNLOAD_URL.format(file_id=self.file_id)


# 2026-07-29 从公共 API 观测到的 v2 不可变标识和哈希。数据集中虽有 P80
# 图片，但公共文件列表没有 P80.csv 真值文件，因此将其明确记为数据缺口。
SANDPAPER_FILES: dict[str, PublicFile] = {
    "P40.csv": PublicFile(
        "c14550e5-b38f-4c1a-9c85-e2287c50654b",
        64_216_747,
        "919ab87e037c41e0c63a65e9dcf0144cf7d077520cce2d746c240544ebf83aeb",
    ),
    "P60.csv": PublicFile(
        "87b880f3-beaa-43f0-8364-21797c29c75d",
        80_661_252,
        "38d886371e459c8d2d3337f53f7167054669735ad3f32f4d2c7783197372bb43",
    ),
    "P100.csv": PublicFile(
        "154152c0-fbc7-4550-b61b-0d28d09871b7",
        68_274_019,
        "a085c14fbbae9999dd29395d69f6ce77c655a5ed8d541c5e62e4a5a59bce1d18",
    ),
    "P120.csv": PublicFile(
        "08fe3024-30be-47f6-a1ed-e8cb09250822",
        62_972_724,
        "aab3065a2efdd483ee7d231c29b9f7c3ecb39246ace6015c25b5ce21f3f176e2",
    ),
    "P180.csv": PublicFile(
        "fbed499f-8e92-419a-b6f9-a6780896d43f",
        70_729_475,
        "3e716caafafcb13df9b1eace19f34b754015bed800860d2fd857ad6e7965e6e8",
    ),
    "P240.csv": PublicFile(
        "610f3b48-6a9e-41bf-b3c1-4beedb2db0c6",
        68_515_861,
        "585fc5e128689f5444b32996703c2d38b103a7c34465d0ed0dc81f269e7fce46",
    ),
    "ReadMe.txt": PublicFile(
        "75846188-5cd6-4bdb-879c-28342a7113c6",
        2_092,
        "ced061a185abf5f232b836f69ddfa1b25562735cc68013fbda8104170d4d9981",
    ),
}


@dataclass(frozen=True)
class HiroxHeader:
    """Hirox CSV 前五行中与空间解释有关的元数据。"""

    measured_date: str
    lateral_calibration_um_per_pixel: float
    height_unit: str
    x_size: int
    y_size: int


class EvidenceError(RuntimeError):
    """证据无法下载、校验或解析时抛出的统一业务错误。"""


def _open_url(url: str, *, byte_range: str | None = None) -> BinaryIO:
    """打开公共 URL，并把网络层异常转换为可报告的证据错误。"""

    headers = {
        "User-Agent": "Spine-Sim-terrain-evidence/phase01",
        "Accept": "*/*",
    }
    if byte_range is not None:
        headers["Range"] = byte_range
    try:
        return urlopen(Request(url, headers=headers), timeout=60)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise EvidenceError(f"request failed for {url}: {exc}") from exc


def _fetch_json(url: str) -> dict[str, Any]:
    """下载 JSON，并要求顶层必须是对象。"""

    with _open_url(url) as response:
        try:
            value = json.load(response)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvidenceError(f"invalid JSON returned by {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected a JSON object from {url}")
    return value


def _metadata_pairs(text: str) -> dict[str, str]:
    """把 Hirox 固定的前五行 ``key,value`` 元数据解析为字典。"""

    lines = text.lstrip("\ufeff").splitlines()
    if len(lines) < 5:
        raise EvidenceError("Hirox CSV header has fewer than five lines")
    pairs: dict[str, str] = {}
    for line_number, line in enumerate(lines[:5], start=1):
        if "," not in line:
            raise EvidenceError(
                f"Hirox CSV header line {line_number} is not key,value"
            )
        key, value = line.split(",", maxsplit=1)
        pairs[key.strip()] = value.strip()
    return pairs


def parse_hirox_header_text(text: str) -> HiroxHeader:
    """严格解析 Hirox 头部文本，并规范化微米单位写法。"""

    pairs = _metadata_pairs(text)
    required = {"Measured Date", "Calibration", "Height Unit", "X size", "Y size"}
    missing = sorted(required - pairs.keys())
    if missing:
        raise EvidenceError(f"Hirox CSV header is missing fields: {missing}")

    # 接受三种常见的微米字符，但不猜测其它单位或换算关系。
    calibration = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:μm|µm|um)/pxl",
        pairs["Calibration"],
        flags=re.IGNORECASE,
    )
    if calibration is None:
        raise EvidenceError(
            f"unsupported Hirox calibration value: {pairs['Calibration']!r}"
        )
    height_unit = pairs["Height Unit"]
    if height_unit not in {"μm", "µm", "um"}:
        raise EvidenceError(f"unsupported Hirox height unit: {height_unit!r}")
    try:
        x_size = int(pairs["X size"])
        y_size = int(pairs["Y size"])
    except ValueError as exc:
        raise EvidenceError("Hirox X/Y size is not an integer") from exc
    if x_size < 2 or y_size < 2:
        raise EvidenceError("Hirox height map must contain at least 2x2 samples")
    lateral = float(calibration.group(1))
    if not math.isfinite(lateral) or lateral <= 0.0:
        raise EvidenceError("Hirox lateral calibration must be positive and finite")
    return HiroxHeader(
        measured_date=pairs["Measured Date"],
        lateral_calibration_um_per_pixel=lateral,
        height_unit="um",
        x_size=x_size,
        y_size=y_size,
    )


def read_hirox_header(path: Path) -> HiroxHeader:
    """只读取 CSV 前五行，适合在加载大型高度矩阵前快速探测。"""

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            text = "".join(stream.readline() for _ in range(5))
    except OSError as exc:
        raise EvidenceError(f"cannot read Hirox CSV header {path}: {exc}") from exc
    return parse_hirox_header_text(text)


def read_hirox_csv(path: Path) -> tuple[HiroxHeader, NDArray[np.float64]]:
    """读取完整 Hirox 高度矩阵（单位 μm）并核对头部声明的形状。"""

    header = read_hirox_header(path)
    try:
        heights = np.loadtxt(
            path,
            delimiter=",",
            skiprows=5,
            dtype=np.float64,
            encoding="utf-8-sig",
        )
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"cannot parse Hirox height matrix {path}: {exc}") from exc
    if heights.ndim != 2:
        raise EvidenceError(f"Hirox height matrix must be 2-D, got {heights.shape}")
    expected = (header.y_size, header.x_size)
    if heights.shape != expected:
        raise EvidenceError(
            f"Hirox matrix shape {heights.shape} does not match header {expected}"
        )
    return header, heights


def summarize_hirox(
    path: Path,
    header: HiroxHeader,
    heights_um: NDArray[np.float64],
) -> dict[str, Any]:
    """把原始 μm 高度转换为 SI 统计摘要，并保留来源与解释边界。"""

    finite = np.isfinite(heights_um)
    finite_count = int(np.count_nonzero(finite))
    total_count = int(heights_um.size)
    if finite_count == 0:
        raise EvidenceError("Hirox height matrix contains no finite values")
    # 统计量只使用有限样本，但缺失数量和比例会显式写入结果。
    values_um = heights_um[finite]
    mean_um = float(np.mean(values_um))
    rms_zero_um = float(np.sqrt(np.mean(np.square(values_um))))
    rms_mean_um = float(np.sqrt(np.mean(np.square(values_um - mean_um))))
    pitch_um = header.lateral_calibration_um_per_pixel
    # N 个采样节点只有 N-1 个间隔，物理跨度不能写成 N*pitch。
    x_span_um = (header.x_size - 1) * pitch_um
    y_span_um = (header.y_size - 1) * pitch_um
    return {
        "parser_version": PARSER_VERSION,
        "source": {
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "doi": DATASET_DOI,
            "landing_page": LANDING_PAGE,
            "local_path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        },
        "header": asdict(header),
        "array_shape_y_x": [header.y_size, header.x_size],
        "physical_node_span_m": {
            "x": x_span_um * 1e-6,
            "y": y_span_um * 1e-6,
        },
        "lateral_sampling_interval_m": {
            "x": pitch_um * 1e-6,
            "y": pitch_um * 1e-6,
        },
        "height_unit": "m",
        "missing_value_count": total_count - finite_count,
        "missing_value_ratio": (total_count - finite_count) / total_count,
        "minimum_height_m": float(np.min(values_um)) * 1e-6,
        "maximum_height_m": float(np.max(values_um)) * 1e-6,
        "mean_height_m": mean_um * 1e-6,
        "rms_height_about_zero_m": rms_zero_um * 1e-6,
        "rms_height_about_mean_m": rms_mean_um * 1e-6,
        "interpretation_notes": [
            "CSV rows are interpreted as y and columns as x.",
            "Physical node span uses (sample_count - 1) * sampling_interval.",
            "The file does not declare row orientation or positive-z convention.",
            "Zero is treated as a measured value, not a missing-value sentinel.",
            "No detrending, filtering, filling, or roughness-model calibration is applied.",
        ],
    }


def write_preview(
    heights_um: NDArray[np.float64],
    header: HiroxHeader,
    output: Path,
    *,
    max_preview_pixels: int = 900,
) -> None:
    """绘制降采样预览；数值摘要仍使用未经降采样的完整矩阵。"""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise EvidenceError(
            "preview output requires the optional 'plot' dependencies"
        ) from exc

    # 分轴选择步长，将图片限制在可交互尺寸而不改变源数据。
    stride_y = max(1, math.ceil(header.y_size / max_preview_pixels))
    stride_x = max(1, math.ceil(header.x_size / max_preview_pixels))
    preview = heights_um[::stride_y, ::stride_x]
    finite = preview[np.isfinite(preview)]
    if finite.size == 0:
        raise EvidenceError("cannot plot a preview without finite heights")
    # 色限裁剪只改善显示对比度，不会裁剪写入 NPY 或摘要的高度值。
    lower, upper = np.percentile(finite, [1.0, 99.0])
    if not lower < upper:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
    extent_mm = (
        0.0,
        (header.x_size - 1) * header.lateral_calibration_um_per_pixel / 1000.0,
        (header.y_size - 1) * header.lateral_calibration_um_per_pixel / 1000.0,
        0.0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10.5, 3.2), constrained_layout=True)
    image = axis.imshow(
        preview,
        cmap="viridis",
        aspect="auto",
        extent=extent_mm,
        vmin=lower,
        vmax=upper,
        interpolation="nearest",
    )
    axis.set_title("P240 Hirox measured height (1st–99th percentile color scale)")
    axis.set_xlabel("x (mm; column direction)")
    axis.set_ylabel("y (mm; row direction)")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("height (um)")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _record_download_metadata(
    target_dir: Path,
    filename: str,
    expected: PublicFile,
) -> None:
    """在原始数据旁原子更新来源、许可和已验证文件信息。"""

    metadata_path = target_dir / "source_metadata.json"
    metadata: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "doi": DATASET_DOI,
        "landing_page": LANDING_PAGE,
        "license": "CC BY 4.0",
        "license_evidence": LANDING_PAGE,
        "downloaded_files": {},
    }
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvidenceError(
                f"cannot update existing source metadata {metadata_path}: {exc}"
            ) from exc
        if not isinstance(existing, dict):
            raise EvidenceError(f"source metadata is not an object: {metadata_path}")
        metadata.update(existing)
    downloaded = metadata.setdefault("downloaded_files", {})
    if not isinstance(downloaded, dict):
        raise EvidenceError(
            f"source metadata downloaded_files is not an object: {metadata_path}"
        )
    downloaded[filename] = {
        **asdict(expected),
        "download_url": expected.download_url,
    }
    atomic_write_json(metadata_path, metadata)


def probe_sandpaper(output: Path | None) -> dict[str, Any]:
    """将公共 API 当前元数据与代码中冻结的 v2 文件清单逐项比较。"""

    metadata = _fetch_json(PUBLIC_API_URL)
    api_files = {
        item.get("filename"): item
        for item in metadata.get("files", [])
        if isinstance(item, dict)
    }
    probes: list[dict[str, Any]] = []
    for filename, expected in SANDPAPER_FILES.items():
        item = api_files.get(filename)
        if item is None:
            probes.append(
                {
                    "filename": filename,
                    "status": "not_listed_by_public_api",
                    "expected": asdict(expected),
                }
            )
            continue
        details = item.get("content_details", {})
        comparison = {
            "size_matches": details.get("size") == expected.size_bytes,
            "sha256_matches": details.get("sha256_hash") == expected.sha256,
            "file_id_matches": item.get("id") == expected.file_id,
        }
        record: dict[str, Any] = {
            "filename": filename,
            "status": (
                "verified_metadata"
                if all(comparison.values())
                else "metadata_mismatch"
            ),
            "comparison": comparison,
            "file_id": item.get("id"),
            "size_bytes": details.get("size"),
            "sha256": details.get("sha256_hash"),
            "download_url": expected.download_url,
        }
        if filename.endswith(".csv"):
            # Range 请求只取头部 4 KiB，探测格式时无需下载几十 MB 的矩阵。
            with _open_url(expected.download_url, byte_range="bytes=0-4095") as stream:
                header_bytes = stream.read(4096)
            record["header"] = asdict(
                parse_hirox_header_text(header_bytes.decode("utf-8-sig"))
            )
        probes.append(record)
    result = {
        "checked_date": "2026-07-29",
        "dataset_id": DATASET_ID,
        "dataset_version": metadata.get("version"),
        "doi": DATASET_DOI,
        "landing_page": LANDING_PAGE,
        "public_api_url": PUBLIC_API_URL,
        "public_api_file_count": len(api_files),
        "p80_csv_listed": "P80.csv" in api_files,
        "files": probes,
    }
    if output is not None:
        atomic_write_json(output, result)
    return result


def download_file(filename: str, destination_root: Path, max_bytes: int) -> Path:
    """下载白名单文件，校验长度和 SHA-256 后再原子发布。"""

    try:
        expected = SANDPAPER_FILES[filename]
    except KeyError as exc:
        raise EvidenceError(
            f"unsupported file {filename!r}; choose one of {sorted(SANDPAPER_FILES)}"
        ) from exc
    if expected.size_bytes > max_bytes:
        raise EvidenceError(
            f"refusing {filename}: {expected.size_bytes} bytes exceeds "
            f"--max-bytes={max_bytes}"
        )
    target_dir = destination_root / f"mendeley_{DATASET_ID}_v{DATASET_VERSION}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if target.exists():
        # 已存在文件只允许验证和复用；失败时不自动覆盖用户的本地数据。
        actual_hash = sha256_file(target)
        if target.stat().st_size == expected.size_bytes and actual_hash == expected.sha256:
            _record_download_metadata(target_dir, filename, expected)
            LOGGER.info("verified existing file %s", target)
            return target
        raise EvidenceError(
            f"existing file failed verification and was not overwritten: {target}"
        )

    # 下载写入同目录 .part 文件，完全校验后用 os.replace 原子改名。
    temporary = target.with_suffix(target.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    LOGGER.info("downloading %s (%d bytes)", filename, expected.size_bytes)
    digest = hashlib.sha256()
    written = 0
    try:
        with _open_url(expected.download_url) as source, temporary.open("xb") as sink:
            while True:
                chunk = source.read(CHUNK_BYTES)
                if not chunk:
                    break
                sink.write(chunk)
                digest.update(chunk)
                written += len(chunk)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    # 长度和哈希必须同时匹配冻结证据，任一失败即删除临时文件。
    actual_hash = digest.hexdigest()
    if written != expected.size_bytes or actual_hash != expected.sha256:
        temporary.unlink()
        raise EvidenceError(
            f"download verification failed for {filename}: size={written}, "
            f"sha256={actual_hash}"
        )
    os.replace(temporary, target)
    _record_download_metadata(target_dir, filename, expected)
    LOGGER.info("verified sha256 %s", actual_hash)
    return target


def _parse_command(args: argparse.Namespace) -> dict[str, Any]:
    """执行本地 Hirox 解析，并按需输出预览、SI 制 NPY 和摘要。"""

    path = Path(args.path).resolve()
    header, heights = read_hirox_csv(path)
    summary = summarize_hirox(path, header, heights)
    if args.preview is not None:
        preview = Path(args.preview).resolve()
        write_preview(heights, header, preview)
        summary["preview_path"] = str(preview)
    if args.npy_output is not None:
        # 原始 CSV 是 μm；规范 NPY 明确转换为 m 且禁用 pickle。
        npy_output = Path(args.npy_output).resolve()
        npy_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_output, heights * 1e-6, allow_pickle=False)
        summary["normalized_height_npy_m"] = str(npy_output)
    if args.summary is not None:
        atomic_write_json(Path(args.summary).resolve(), summary)
    return summary


def _surface_topography_probe() -> dict[str, Any]:
    """只探测可选 SurfaceTopography 包，不安装或修改当前环境。"""

    try:
        import SurfaceTopography  # type: ignore[import-not-found]
    except ImportError as exc:
        return {
            "available": False,
            "status": "not_installed",
            "error": f"{type(exc).__name__}: {exc}",
            "action": (
                "No dependency was added in Phase 0-1. See the ingestion ADR "
                "before installing."
            ),
        }
    return {
        "available": True,
        "status": "import_succeeded",
        "version": getattr(SurfaceTopography, "__version__", "unknown"),
    }


def build_parser() -> argparse.ArgumentParser:
    """定义元数据探测、受控下载、本地解析和依赖探测四个子命令。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    probe = commands.add_parser("probe-sandpaper")
    probe.add_argument("--output", type=Path)

    download = commands.add_parser("download-sandpaper")
    download.add_argument(
        "--file",
        action="append",
        dest="files",
        default=[],
        help="Repeat for multiple files; defaults to P240.csv and ReadMe.txt.",
    )
    download.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    download.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)

    parse = commands.add_parser("parse-hirox")
    parse.add_argument("path", type=Path)
    parse.add_argument("--summary", type=Path)
    parse.add_argument("--preview", type=Path)
    parse.add_argument("--npy-output", type=Path)

    commands.add_parser("probe-surfacetopography")
    return parser


def main(argv: list[str] | None = None) -> int:
    """分派子命令；证据、文件和参数错误统一返回退出码 2。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "probe-sandpaper":
            result = probe_sandpaper(args.output)
        elif args.command == "download-sandpaper":
            filenames = args.files or ["P240.csv", "ReadMe.txt"]
            result = {
                "downloaded": [
                    str(download_file(name, args.raw_root, args.max_bytes))
                    for name in filenames
                ]
            }
        elif args.command == "parse-hirox":
            result = _parse_command(args)
        else:
            result = _surface_topography_probe()
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    except (EvidenceError, OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
