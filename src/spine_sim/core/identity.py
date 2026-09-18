"""规范化序列化、稳定哈希和确定性对象 ID。

同一物理输入必须在不同进程和机器上得到相同标识，因此所有参与 identity 的对象
先转为无歧义的 JSON 兼容结构，再计算 SHA-256。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def canonicalize(value: Any) -> Any:
    """递归转成键序稳定、可写入 JSON 的基础类型。"""

    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        # 显式排序避免映射构造顺序影响哈希。
        return {str(key): canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        # 集合没有天然顺序，用元素自身的规范 JSON 建立确定顺序。
        normalized = [canonicalize(item) for item in value]
        return sorted(normalized, key=lambda item: canonical_json(item))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return canonicalize(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("non-finite values cannot be hashed")
        if value == 0:
            # 统一 +0.0 与 -0.0，避免物理等价输入出现两个 identity。
            return 0.0
    return value


def canonical_json(value: Any) -> str:
    """生成无多余空白且禁止 NaN/Infinity 的规范 JSON 文本。"""

    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: Any) -> str:
    """返回对象规范 JSON 的完整 SHA-256 十六进制摘要。"""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def identity(kind: str, normalized_input: Any, *, module_version: str = "1") -> str:
    """生成 ``<类型>_<20 位摘要>`` 形式的稳定、可读 ID。"""

    if not kind or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in kind):
        raise ValueError("identity kind must use lowercase ASCII letters, digits or underscore")
    digest = stable_hash(
        {"kind": kind, "module_version": module_version, "input": normalized_input}
    )
    return f"{kind}_{digest[:20]}"


def lineage_hash(*upstream_records: Any) -> str:
    """把有序的上游记录合并为一个来源链摘要。"""

    return stable_hash({"upstream": list(upstream_records)})
