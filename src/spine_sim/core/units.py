"""Explicit conversion of supported input quantities to SI."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Any

from .errors import ConfigurationError


_UNITS: dict[str, tuple[str, float]] = {
    "1": ("dimensionless", 1.0),
    "m": ("length", 1.0),
    "mm": ("length", 1e-3),
    "um": ("length", 1e-6),
    "µm": ("length", 1e-6),
    "nm": ("length", 1e-9),
    "rad": ("angle", 1.0),
    "deg": ("angle", pi / 180.0),
    "N": ("force", 1.0),
    "N/m": ("stiffness", 1.0),
    "Pa": ("pressure", 1.0),
    "MPa": ("pressure", 1e6),
    "GPa": ("pressure", 1e9),
    "s": ("time", 1.0),
    "m/s": ("velocity", 1.0),
    "mm/s": ("velocity", 1e-3),
}


@dataclass(frozen=True)
class Quantity:
    value_si: float
    dimension: str
    input_unit: str


def to_si(value: Any, dimension: str, *, name: str = "value") -> float:
    """Normalize a number already in SI or ``{"value": x, "unit": u}``."""
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be numeric, not bool")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, dict) and set(value) >= {"value", "unit"}:
        unit = str(value["unit"])
        if unit not in _UNITS:
            raise ConfigurationError(f"{name} uses unsupported unit {unit!r}")
        actual_dimension, scale = _UNITS[unit]
        if actual_dimension != dimension:
            raise ConfigurationError(
                f"{name} requires {dimension}, got {actual_dimension} ({unit})"
            )
        try:
            result = float(value["value"]) * scale
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{name} has a non-numeric value") from exc
    else:
        raise ConfigurationError(
            f"{name} must be an SI number or a value/unit mapping"
        )
    if not isfinite(result):
        raise ConfigurationError(f"{name} must be finite")
    return result


def require_range(
    value: float,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
) -> float:
    lower_bad = minimum is not None and (
        value < minimum if inclusive_min else value <= minimum
    )
    upper_bad = maximum is not None and (
        value > maximum if inclusive_max else value >= maximum
    )
    if lower_bad or upper_bad:
        left = "[" if inclusive_min else "("
        right = "]" if inclusive_max else ")"
        raise ConfigurationError(
            f"{name}={value} is outside {left}{minimum}, {maximum}{right}"
        )
    return value
