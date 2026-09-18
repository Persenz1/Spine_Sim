"""结果文件的原子写入、哈希和 UTC 时间工具。

写入流程统一为“同目录临时文件 → flush/fsync → 原子替换”，因此不会把半写文件
误认作完整结果；临时文件在异常路径中也会清理。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spine_sim.core.identity import canonicalize


def _atomic_replace(source: Path, target: Path) -> None:
    """原子替换文件，并容忍 Windows 读句柄造成的瞬时共享冲突。"""

    deadline = time.monotonic() + 1.0
    delay_s = 0.005
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or time.monotonic() >= deadline:
                raise
            time.sleep(delay_s)
            delay_s = min(delay_s * 2.0, 0.05)


def utc_now() -> str:
    """返回带 UTC 时区的 ISO-8601 时间戳。"""

    return datetime.now(UTC).isoformat()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """在目标目录内通过临时文件原子替换二进制内容。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary_path = Path(temporary)
    try:
        # fsync 保证替换前数据已经交给操作系统持久化，随后 replace 只暴露完整文件。
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_replace(temporary_path, target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: str | Path, value: Any) -> None:
    """规范化对象并以稳定、可读的 UTF-8 JSON 原子写入。"""

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
    """把一组 NumPy 数组压缩为 NPZ，并以原子替换方式发布。"""

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
        _atomic_replace(temporary_path, target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def sha256_file(path: str | Path) -> str:
    """分块计算文件 SHA-256，避免把大型地形文件一次读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
