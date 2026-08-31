"""坐标系元数据与六维力旋量（wrench）运算。

力和力矩必须同时携带坐标系、参考点、受力对象和施力对象，避免在单刺到阵列的
传递过程中混淆方向或重复计入力矩。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import ConfigurationError


Vector3 = NDArray[np.float64]
Matrix3 = NDArray[np.float64]


def _vector3(value: ArrayLike, name: str) -> Vector3:
    """把输入校验为有限三维向量。"""

    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ConfigurationError(f"{name} must be a finite 3-vector")
    return array


def _rotation(value: ArrayLike) -> Matrix3:
    """校验右手正交旋转矩阵（``R.T @ R = I`` 且 ``det(R)=+1``）。"""

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
    """描述一个相对父坐标系的右手坐标系。"""

    name: str
    parent: str | None
    origin_m: tuple[float, float, float]
    rotation_to_parent: tuple[tuple[float, float, float], ...]
    convention: str = "right_handed"

    def __post_init__(self) -> None:
        """冻结对象前验证原点和旋转定义。"""

        if not self.name:
            raise ConfigurationError("frame name cannot be empty")
        _vector3(self.origin_m, "origin_m")
        _rotation(self.rotation_to_parent)
        if self.convention != "right_handed":
            raise ConfigurationError("only right-handed frames are supported")


@dataclass(frozen=True)
class Wrench:
    """带坐标系、作用对象和参考点标签的力与力矩。"""

    force_N: tuple[float, float, float]
    moment_Nm: tuple[float, float, float]
    frame: str
    reference_point: str
    acting_on: str
    exerted_by: str
    sign_convention: str = "right_hand_rule"

    def __post_init__(self) -> None:
        """保证六个分量有限，且所有语义标签非空。"""

        _vector3(self.force_N, "force_N")
        _vector3(self.moment_Nm, "moment_Nm")
        for field in ("frame", "reference_point", "acting_on", "exerted_by"):
            if not getattr(self, field):
                raise ConfigurationError(f"{field} cannot be empty")

    @property
    def vector(self) -> NDArray[np.float64]:
        """按 ``[Fx, Fy, Fz, Mx, My, Mz]`` 顺序返回六维向量。"""

        return np.asarray((*self.force_N, *self.moment_Nm), dtype=np.float64)

    @property
    def interaction_label(self) -> str:
        """返回 ``施力者_on_受力者`` 形式的关系标签。"""

        return f"{self.exerted_by}_on_{self.acting_on}"

    def rotate(self, rotation_new_from_old: ArrayLike, *, new_frame: str) -> "Wrench":
        """把力和力矩共同旋转到新坐标系，参考点和作用对象保持不变。"""

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
        """将参考点从 O 移到 P；输入 P→O 向量，使用 ``M_P=M_O+r_PO×F``。"""
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
        """转换为可序列化且保留全部力学语义的映射。"""

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
