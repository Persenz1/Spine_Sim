"""Explicit CPU/CUDA capability discovery without requiring a GPU package."""

from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import asdict, dataclass

from spine_sim.core.config import BackendConfig
from spine_sim.core.errors import ConfigurationError


@dataclass(frozen=True)
class BackendCapabilities:
    cpu_available: bool
    cuda_available: bool
    cuda_provider: str | None
    selected: str
    device_index: int | None
    platform: str
    detection_notes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _detect_cuda() -> tuple[bool, str | None, list[str]]:
    notes: list[str] = []
    if os.environ.get("SPINE_SIM_FORCE_CUDA") == "1":
        return True, "environment_override", notes
    if importlib.util.find_spec("cupy") is not None:
        try:
            import cupy  # type: ignore

            count = int(cupy.cuda.runtime.getDeviceCount())
            return count > 0, "cupy", [f"cupy_device_count={count}"]
        except Exception as exc:  # capability probing must not make CPU unusable
            notes.append(f"cupy_probe_failed:{type(exc).__name__}")
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  # type: ignore

            return bool(torch.cuda.is_available()), "torch", notes
        except Exception as exc:
            notes.append(f"torch_probe_failed:{type(exc).__name__}")
    return False, None, notes


def discover_backend(config: BackendConfig | None = None) -> BackendCapabilities:
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
