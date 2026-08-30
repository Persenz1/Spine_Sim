"""Atomic file writes, file hashes, and UTC timestamps."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spine_sim.core.identity import canonicalize


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: str | Path, value: Any) -> None:
    data = json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    atomic_write_bytes(path, data + b"\n")


def atomic_write_npz(
    path: str | Path, arrays: Mapping[str, np.ndarray]
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
