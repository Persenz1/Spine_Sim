"""Read-only environment validation."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from spine_sim.core.config import BackendConfig
from spine_sim.runtime.backend import discover_backend


def validate_environment(
    backend: BackendConfig | None = None, *, writable_path: str | Path | None = None
) -> dict[str, Any]:
    checks = [
        {
            "name": "python_version",
            "passed": sys.version_info >= (3, 11),
            "value": platform.python_version(),
        },
        {
            "name": "numpy",
            "passed": tuple(int(part) for part in np.__version__.split(".")[:2]) >= (1, 26),
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
    checks.append({"name": "cpu_available", "passed": capabilities.cpu_available, "value": True})
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "backend": capabilities.as_dict(),
        "platform": platform.platform(),
    }
