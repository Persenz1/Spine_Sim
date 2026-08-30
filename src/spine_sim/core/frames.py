"""Coordinate metadata and six-dimensional wrench operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import ConfigurationError


Vector3 = NDArray[np.float64]
Matrix3 = NDArray[np.float64]


def _vector3(value: ArrayLike, name: str) -> Vector3:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ConfigurationError(f"{name} must be a finite 3-vector")
    return array


def _rotation(value: ArrayLike) -> Matrix3:
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ConfigurationError("rotation must be a finite 3x3 matrix")
    if not np.allclose(array.T @ array, np.eye(3), atol=1e-10):
        raise ConfigurationError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(array), 1.0, atol=1e-10):
        raise ConfigurationError("rotation must be proper (determinant +1)")
    return array


@dataclass(frozen=True)
class FrameMetadata:
    name: str
    parent: str | None
    origin_m: tuple[float, float, float]
    rotation_to_parent: tuple[tuple[float, float, float], ...]
    convention: str = "right_handed"

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigurationError("frame name cannot be empty")
        _vector3(self.origin_m, "origin_m")
        _rotation(self.rotation_to_parent)
        if self.convention != "right_handed":
            raise ConfigurationError("only right-handed frames are supported")


@dataclass(frozen=True)
class Wrench:
    """Force and moment with mandatory frame, object and reference-point labels."""

    force_N: tuple[float, float, float]
    moment_Nm: tuple[float, float, float]
    frame: str
    reference_point: str
    acting_on: str
    exerted_by: str
    sign_convention: str = "right_hand_rule"

    def __post_init__(self) -> None:
        _vector3(self.force_N, "force_N")
        _vector3(self.moment_Nm, "moment_Nm")
        for field in ("frame", "reference_point", "acting_on", "exerted_by"):
            if not getattr(self, field):
                raise ConfigurationError(f"{field} cannot be empty")

    @property
    def vector(self) -> NDArray[np.float64]:
        return np.asarray((*self.force_N, *self.moment_Nm), dtype=np.float64)

    @property
    def interaction_label(self) -> str:
        return f"{self.exerted_by}_on_{self.acting_on}"

    def rotate(self, rotation_new_from_old: ArrayLike, *, new_frame: str) -> "Wrench":
        rotation = _rotation(rotation_new_from_old)
        force = rotation @ np.asarray(self.force_N, dtype=np.float64)
        moment = rotation @ np.asarray(self.moment_Nm, dtype=np.float64)
        return Wrench(
            tuple(force),
            tuple(moment),
            new_frame,
            self.reference_point,
            self.acting_on,
            self.exerted_by,
            self.sign_convention,
        )

    def move_reference(
        self,
        old_reference_from_new_m: ArrayLike,
        *,
        new_reference_point: str,
    ) -> "Wrench":
        """Move from O to P using vector P→O, so M_P = M_O + r_PO × F."""
        offset = _vector3(old_reference_from_new_m, "old_reference_from_new_m")
        force = np.asarray(self.force_N, dtype=np.float64)
        moment = np.asarray(self.moment_Nm, dtype=np.float64) + np.cross(offset, force)
        return Wrench(
            self.force_N,
            tuple(moment),
            self.frame,
            new_reference_point,
            self.acting_on,
            self.exerted_by,
            self.sign_convention,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "force_N": list(self.force_N),
            "moment_Nm": list(self.moment_Nm),
            "frame": self.frame,
            "reference_point": self.reference_point,
            "acting_on": self.acting_on,
            "exerted_by": self.exerted_by,
            "interaction_label": self.interaction_label,
            "sign_convention": self.sign_convention,
        }
