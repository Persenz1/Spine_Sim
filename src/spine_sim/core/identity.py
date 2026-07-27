"""Canonical serialization, hashes and deterministic prefixed IDs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def canonicalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {str(key): canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
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
            return 0.0
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def identity(kind: str, normalized_input: Any, *, module_version: str = "1") -> str:
    if not kind or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in kind):
        raise ValueError("identity kind must use lowercase ASCII letters, digits or underscore")
    digest = stable_hash(
        {"kind": kind, "module_version": module_version, "input": normalized_input}
    )
    return f"{kind}_{digest[:20]}"


def lineage_hash(*upstream_records: Any) -> str:
    return stable_hash({"upstream": list(upstream_records)})


def terrain_recipe_id(value: Any, *, module_version: str) -> str:
    return identity("terrain_recipe", value, module_version=module_version)


def region_id(value: Any, *, module_version: str = "1") -> str:
    return identity("region", value, module_version=module_version)


def track_id(value: Any, *, module_version: str = "1") -> str:
    return identity("track", value, module_version=module_version)


def case_id(value: Any, *, module_version: str) -> str:
    return identity("case", value, module_version=module_version)


def campaign_id(value: Any, *, module_version: str) -> str:
    return identity("campaign", value, module_version=module_version)
