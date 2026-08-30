"""CPU/CUDA 能力探测与最小运行环境检查。

CUDA 在本项目中只用于部分地形生成路径；探测失败时 ``auto`` 可继续使用 CPU，
但用户显式请求 ``cuda`` 时必须明确报错。
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spine_sim.core.errors import ConfigurationError


@dataclass(frozen=True)
class BackendConfig:
    """后端偏好、GPU 开关和设备编号。"""

    preference: str = "auto"
    allow_gpu: bool = True
    device_index: int = 0

    def __post_init__(self) -> None:
        """校验后端名称和非负设备编号。"""

        if self.preference not in {"auto", "cpu", "cuda"}:
            raise ConfigurationError("backend preference must be auto, cpu or cuda")
        if self.device_index < 0:
            raise ConfigurationError("device_index must be non-negative")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BackendConfig":
        """从配置映射构造后端设置并拒绝未知字段。"""

        extra = set(data) - {"preference", "allow_gpu", "device_index"}
        if extra:
            raise ConfigurationError(
                f"backend contains unknown fields: {sorted(extra)}"
            )
        return cls(**data)


@dataclass(frozen=True)
class BackendCapabilities:
    """一次环境探测得到的可用能力、最终选择和诊断说明。"""

    cpu_available: bool
    cuda_available: bool
    cuda_provider: str | None
    selected: str
    device_index: int | None
    platform: str
    detection_notes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """转换为可写入 validation.json 的字典。"""

        return asdict(self)


def _detect_cuda() -> tuple[bool, str | None, list[str]]:
    """通过 CuPy 做轻量 CUDA 探测，并把失败原因作为诊断而非异常返回。"""

    notes: list[str] = []
    if importlib.util.find_spec("cupy") is not None:
        try:
            import cupy  # type: ignore

            count = int(cupy.cuda.runtime.getDeviceCount())
            return count > 0, "cupy", [f"cupy_device_count={count}"]
        except Exception as exc:  # capability probing must not make CPU unusable
            notes.append(f"cupy_probe_failed:{type(exc).__name__}")
    return False, None, notes


def discover_backend(config: BackendConfig | None = None) -> BackendCapabilities:
    """根据配置和实际设备能力选择 CPU 或 CUDA。"""

    config = config or BackendConfig()
    cuda_available, provider, notes = _detect_cuda()
    if not config.allow_gpu:
        cuda_available = False
        notes.append("gpu_disabled_by_config")
    if config.preference == "cuda" and not cuda_available:
        raise ConfigurationError("CUDA was requested but is not available")
    selected = "cuda" if cuda_available and config.preference in {"auto", "cuda"} else "cpu"
    return BackendCapabilities(
        cpu_available=True,
        cuda_available=cuda_available,
        cuda_provider=provider,
        selected=selected,
        device_index=config.device_index if selected == "cuda" else None,
        platform=platform.system().lower(),
        detection_notes=tuple(notes),
    )


def validate_environment(
    backend: BackendConfig | None = None,
    *,
    writable_path: str | Path | None = None,
) -> dict[str, Any]:
    """检查 Python/NumPy 版本、输出父目录和计算后端。"""

    checks = [
        {
            "name": "python_version",
            "passed": sys.version_info >= (3, 11),
            "value": platform.python_version(),
        },
        {
            "name": "numpy",
            "passed": tuple(int(part) for part in np.__version__.split(".")[:2])
            >= (1, 26),
            "value": np.__version__,
        },
    ]
    if writable_path is not None:
        path = Path(writable_path)
        checks.append(
            {
                "name": "results_parent_exists",
                "passed": path.resolve().parent.exists(),
                "value": str(path.resolve().parent),
            }
        )
    capabilities = discover_backend(backend)
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "backend": capabilities.as_dict(),
        "platform": platform.platform(),
    }
