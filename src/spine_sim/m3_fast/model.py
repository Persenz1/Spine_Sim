"""Vectorized closed-form spine contact kernel for the fast M3 solver.

This module deliberately contains no M2 compatibility layer.  A case owns one
immutable :class:`SpineBatch`, one accepted :class:`ContactState`, and one
reusable :class:`ModelWorkspace`.  Every equilibrium trial reads the same
accepted state and writes only to the workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FREE = 0
STICK = 1
SLIDE = 2

RIGID = 0
INTERIOR = 1
LOWER_STOP = 2
HARD_STOP = 3

_FLOAT: TypeAlias = NDArray[np.float64]
_INT8: TypeAlias = NDArray[np.int8]
_BOOL: TypeAlias = NDArray[np.bool_]
_ANGLE_PATTERNS = {
    "fixed": "fixed",
    "60_to_80": "60_to_80",
    "80_to_60": "80_to_60",
    "80_to_50": "80_to_50",
    # Accept the names used by the removed array configuration only as input
    # spelling; no old class or runtime code is imported.
    "gradient_80_to_60": "80_to_60",
    "gradient_80_to_50": "80_to_50",
    "gradient_60_to_80": "60_to_80",
}


def _positive_scalar(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _spine_float_array(
    name: str,
    value: ArrayLike,
    count: int,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> _FLOAT:
    source = np.asarray(value, dtype=np.float64)
    if source.ndim == 0:
        result = np.full(count, float(source), dtype=np.float64)
    elif source.shape == (count,):
        result = np.ascontiguousarray(source, dtype=np.float64)
    else:
        raise ValueError(f"{name} must be scalar or have shape ({count},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    return result


def _check_float_vector(name: str, value: _FLOAT, count: int) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.shape != (count,) or value.dtype != np.float64:
        raise ValueError(f"{name} must be float64 with shape ({count},)")


@dataclass(frozen=True, slots=True)
class SpineBatch:
    """Case-constant geometry, material parameters, and compliances.

    Arrays use row-major ``(y, x)`` flattening, so adjacent entries traverse
    local x.  ``tip_x_offset_m`` is the complete tip-centre offset from the
    unit origin (mount x plus the axial projection), as required by track
    interpolation.
    """

    nx: int
    ny: int
    spacing_m: float
    angle_pattern: str
    mount_x_m: _FLOAT
    y_m: _FLOAT
    tip_x_offset_m: _FLOAT
    tip_z_offset_m: _FLOAT
    angle_rad: _FLOAT
    length_m: _FLOAT
    tip_radius_m: _FLOAT
    diameter_m: _FLOAT
    axis_x: _FLOAT
    axis_z: _FLOAT
    transverse_x: _FLOAT
    transverse_z: _FLOAT
    axial_compliance_m_per_N: _FLOAT
    transverse_compliance_m_per_N: _FLOAT
    spring_stiffness_N_per_m: _FLOAT
    spring_delta_max_m: _FLOAT
    rigid_mask: _BOOL
    interior_parallel_compliance_m_per_N: _FLOAT
    static_friction: _FLOAT
    kinetic_friction: _FLOAT

    def __post_init__(self) -> None:
        if (
            isinstance(self.nx, (bool, np.bool_))
            or isinstance(self.ny, (bool, np.bool_))
            or not isinstance(self.nx, (int, np.integer))
            or not isinstance(self.ny, (int, np.integer))
            or self.nx < 1
            or self.ny < 1
        ):
            raise ValueError("nx and ny must be positive integers")
        _positive_scalar("spacing_m", self.spacing_m)
        if self.angle_pattern not in {
            "fixed",
            "60_to_80",
            "80_to_60",
            "80_to_50",
        }:
            raise ValueError("unsupported angle_pattern")
        count = int(self.nx * self.ny)
        for name in (
            "mount_x_m",
            "y_m",
            "tip_x_offset_m",
            "tip_z_offset_m",
            "angle_rad",
            "length_m",
            "tip_radius_m",
            "diameter_m",
            "axis_x",
            "axis_z",
            "transverse_x",
            "transverse_z",
            "axial_compliance_m_per_N",
            "transverse_compliance_m_per_N",
            "spring_stiffness_N_per_m",
            "spring_delta_max_m",
            "interior_parallel_compliance_m_per_N",
            "static_friction",
            "kinetic_friction",
        ):
            _check_float_vector(name, getattr(self, name), count)
        if (
            not isinstance(self.rigid_mask, np.ndarray)
            or self.rigid_mask.shape != (count,)
            or self.rigid_mask.dtype != np.bool_
        ):
            raise ValueError(
                f"rigid_mask must be bool with shape ({count},)"
            )

    @property
    def spine_count(self) -> int:
        return self.nx * self.ny

    @property
    def size(self) -> int:
        return self.spine_count

    @property
    def N(self) -> int:
        return self.spine_count

    @property
    def mount_y_m(self) -> _FLOAT:
        return self.y_m

    @property
    def a_x(self) -> _FLOAT:
        return self.axis_x

    @property
    def a_z(self) -> _FLOAT:
        return self.axis_z

    @property
    def b_x(self) -> _FLOAT:
        return self.transverse_x

    @property
    def b_z(self) -> _FLOAT:
        return self.transverse_z

    @property
    def ca(self) -> _FLOAT:
        return self.axial_compliance_m_per_N

    @property
    def cb(self) -> _FLOAT:
        return self.transverse_compliance_m_per_N


@dataclass(slots=True)
class ContactState:
    """Accepted per-spine history; mutated only after a station is accepted."""

    mode: _INT8
    u_t_history_m: _FLOAT
    spring_branch: _INT8

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mode, np.ndarray)
            or self.mode.ndim != 1
            or self.mode.dtype != np.int8
        ):
            raise ValueError("mode must be a one-dimensional int8 array")
        count = self.mode.size
        _check_float_vector("u_t_history_m", self.u_t_history_m, count)
        if (
            not isinstance(self.spring_branch, np.ndarray)
            or self.spring_branch.shape != (count,)
            or self.spring_branch.dtype != np.int8
        ):
            raise ValueError(
                f"spring_branch must be int8 with shape ({count},)"
            )

    @property
    def size(self) -> int:
        return self.mode.size

    @property
    def N(self) -> int:
        return self.mode.size


@dataclass(slots=True)
class ModelWorkspace:
    """Reusable full-array trial outputs and algebra scratch arrays."""

    force_x_N: _FLOAT
    force_z_N: _FLOAT
    lambda_n_N: _FLOAT
    tangent_force_N: _FLOAT
    mode: _INT8
    u_t_history_m: _FLOAT
    spring_branch: _INT8
    vertical_stiffness_N_per_m: _FLOAT
    spring_axial_load_N: _FLOAT
    tangent_x: _FLOAT
    tangent_z: _FLOAT
    normal_x: _FLOAT
    normal_z: _FLOAT
    normal_closure_m: _FLOAT
    tangent_trial_m: _FLOAT
    normal_axis_dot: _FLOAT
    tangent_axis_dot: _FLOAT
    normal_transverse_dot: _FLOAT
    tangent_transverse_dot: _FLOAT
    g_nn: _FLOAT
    g_nt: _FLOAT
    g_tt: _FLOAT
    determinant: _FLOAT
    rhs_normal_m: _FLOAT
    rhs_tangent_m: _FLOAT
    trial_lambda_n_N: _FLOAT
    trial_tangent_force_N: _FLOAT
    slide_denominator_m_per_N: _FLOAT

    @property
    def size(self) -> int:
        return self.mode.size

    @property
    def N(self) -> int:
        return self.mode.size

    @property
    def fx(self) -> _FLOAT:
        return self.force_x_N

    @property
    def fz(self) -> _FLOAT:
        return self.force_z_N

    @property
    def lambda_n(self) -> _FLOAT:
        return self.lambda_n_N

    @property
    def kz(self) -> _FLOAT:
        return self.vertical_stiffness_N_per_m

    @property
    def u(self) -> _FLOAT:
        return self.u_t_history_m

    @property
    def branch(self) -> _INT8:
        return self.spring_branch


def build_spine_batch(
    nx: int,
    ny: int,
    spacing_m: float,
    *,
    angle_pattern: str = "fixed",
    fixed_angle_deg: float = 70.0,
    tip_radius_m: ArrayLike = 50e-6,
    diameter_m: ArrayLike = 0.8e-3,
    spring_stiffness_N_per_m: ArrayLike | None = 800.0,
    spring_delta_max_m: ArrayLike = 4e-3,
    young_modulus_Pa: ArrayLike = 200e9,
    poisson_ratio: ArrayLike = 0.29,
    shear_correction: ArrayLike = 6.0 / 7.0,
    static_friction: ArrayLike = 0.45,
    kinetic_friction: ArrayLike = 0.35,
) -> SpineBatch:
    """Build all geometry and compliance arrays once for one M3 case.

    ``spring_stiffness_N_per_m=None`` selects a rigid axial installation.
    A finite scalar selects the same independent spring for every spine; a
    length-``nx*ny`` array may be used for a mixed fixture.  ``np.inf`` entries
    in such an array are treated as rigid.
    """

    if (
        isinstance(nx, (bool, np.bool_))
        or isinstance(ny, (bool, np.bool_))
        or not isinstance(nx, (int, np.integer))
        or not isinstance(ny, (int, np.integer))
        or nx < 1
        or ny < 1
    ):
        raise ValueError("nx and ny must be positive integers")
    nx = int(nx)
    ny = int(ny)
    count = nx * ny
    spacing_m = _positive_scalar("spacing_m", spacing_m)
    try:
        normalized_pattern = _ANGLE_PATTERNS[str(angle_pattern)]
    except KeyError as exc:
        raise ValueError(
            "angle_pattern must be fixed, 60_to_80, 80_to_60, or 80_to_50"
        ) from exc

    if normalized_pattern == "fixed":
        if isinstance(fixed_angle_deg, (bool, np.bool_)):
            raise ValueError("fixed_angle_deg must lie in (0, 90)")
        angle = float(fixed_angle_deg)
        if not math.isfinite(angle) or not 0.0 < angle < 90.0:
            raise ValueError("fixed_angle_deg must lie in (0, 90)")
        column_angle_deg = np.full(nx, angle, dtype=np.float64)
    elif normalized_pattern == "60_to_80":
        column_angle_deg = np.linspace(
            60.0, 80.0, nx, dtype=np.float64
        )
    else:
        endpoint_deg = 60.0 if normalized_pattern == "80_to_60" else 50.0
        column_angle_deg = np.linspace(
            80.0, endpoint_deg, nx, dtype=np.float64
        )

    x_columns = (
        np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)
    ) * spacing_m
    y_rows = (
        np.arange(ny, dtype=np.float64) - 0.5 * (ny - 1)
    ) * spacing_m
    mount_x = np.tile(x_columns, ny)
    mount_y = np.repeat(y_rows, nx)
    angle_rad = np.tile(np.deg2rad(column_angle_deg), ny)

    if normalized_pattern == "fixed":
        length = np.full(count, 4e-3, dtype=np.float64)
    else:
        reference_vertical_projection_m = 4e-3 * math.sin(
            math.radians(80.0)
        )
        length = reference_vertical_projection_m / np.sin(angle_rad)

    axis_x = np.cos(angle_rad)
    axis_z = -np.sin(angle_rad)
    transverse_x = np.sin(angle_rad)
    transverse_z = np.cos(angle_rad)
    tip_x_offset = mount_x + length * axis_x
    tip_z_offset = length * axis_z

    radii = _spine_float_array(
        "tip_radius_m", tip_radius_m, count, positive=True
    )
    diameters = _spine_float_array(
        "diameter_m", diameter_m, count, positive=True
    )
    young = _spine_float_array(
        "young_modulus_Pa", young_modulus_Pa, count, positive=True
    )
    poisson = _spine_float_array(
        "poisson_ratio", poisson_ratio, count
    )
    if np.any((poisson <= -1.0) | (poisson >= 0.5)):
        raise ValueError("poisson_ratio must lie in (-1, 0.5)")
    correction = _spine_float_array(
        "shear_correction", shear_correction, count, positive=True
    )
    mu_s = _spine_float_array(
        "static_friction", static_friction, count, nonnegative=True
    )
    mu_k = _spine_float_array(
        "kinetic_friction", kinetic_friction, count, nonnegative=True
    )
    if np.any(mu_k > mu_s):
        raise ValueError(
            "friction coefficients require kinetic_friction <= static_friction"
        )
    travel = _spine_float_array(
        "spring_delta_max_m", spring_delta_max_m, count, positive=True
    )

    if spring_stiffness_N_per_m is None:
        stiffness = np.full(count, np.inf, dtype=np.float64)
    else:
        stiffness_source = np.asarray(
            spring_stiffness_N_per_m, dtype=np.float64
        )
        if stiffness_source.ndim == 0:
            stiffness = np.full(
                count, float(stiffness_source), dtype=np.float64
            )
        elif stiffness_source.shape == (count,):
            stiffness = np.ascontiguousarray(
                stiffness_source, dtype=np.float64
            )
        else:
            raise ValueError(
                "spring_stiffness_N_per_m must be scalar, None, or "
                f"have shape ({count},)"
            )
        if np.any(np.isnan(stiffness)) or np.any(stiffness <= 0.0):
            raise ValueError(
                "spring stiffness entries must be positive or np.inf"
            )
    rigid = np.isposinf(stiffness)
    if np.any(np.isneginf(stiffness)):
        raise ValueError(
            "spring stiffness entries must be positive or np.inf"
        )

    area_m2 = np.pi * diameters * diameters / 4.0
    second_moment_m4 = np.pi * diameters**4 / 64.0
    shear_modulus_Pa = young / (2.0 * (1.0 + poisson))
    ca = length / (young * area_m2)
    cb = (
        length**3 / (3.0 * young * second_moment_m4)
        + length / (correction * shear_modulus_Pa * area_m2)
    )
    c_parallel = ca.copy()
    flexible = ~rigid
    c_parallel[flexible] += 1.0 / stiffness[flexible]

    return SpineBatch(
        nx=nx,
        ny=ny,
        spacing_m=spacing_m,
        angle_pattern=normalized_pattern,
        mount_x_m=np.ascontiguousarray(mount_x),
        y_m=np.ascontiguousarray(mount_y),
        tip_x_offset_m=np.ascontiguousarray(tip_x_offset),
        tip_z_offset_m=np.ascontiguousarray(tip_z_offset),
        angle_rad=np.ascontiguousarray(angle_rad),
        length_m=np.ascontiguousarray(length),
        tip_radius_m=radii,
        diameter_m=diameters,
        axis_x=np.ascontiguousarray(axis_x),
        axis_z=np.ascontiguousarray(axis_z),
        transverse_x=np.ascontiguousarray(transverse_x),
        transverse_z=np.ascontiguousarray(transverse_z),
        axial_compliance_m_per_N=np.ascontiguousarray(ca),
        transverse_compliance_m_per_N=np.ascontiguousarray(cb),
        spring_stiffness_N_per_m=np.ascontiguousarray(stiffness),
        spring_delta_max_m=travel,
        rigid_mask=np.ascontiguousarray(rigid),
        interior_parallel_compliance_m_per_N=np.ascontiguousarray(c_parallel),
        static_friction=mu_s,
        kinetic_friction=mu_k,
    )


def make_contact_state(batch: SpineBatch) -> ContactState:
    """Allocate the accepted state once at case start."""

    count = batch.spine_count
    branch = np.full(count, LOWER_STOP, dtype=np.int8)
    branch[batch.rigid_mask] = RIGID
    return ContactState(
        mode=np.full(count, FREE, dtype=np.int8),
        u_t_history_m=np.zeros(count, dtype=np.float64),
        spring_branch=branch,
    )


def reset_contact_state(batch: SpineBatch, state: ContactState) -> None:
    """Reset all spines before a load-controlled re-seating attempt."""

    if state.size != batch.spine_count:
        raise ValueError("batch and state sizes do not match")
    state.mode.fill(FREE)
    state.u_t_history_m.fill(0.0)
    state.spring_branch.fill(LOWER_STOP)
    state.spring_branch[batch.rigid_mask] = RIGID


def _empty_float_vectors(count: int, number: int) -> list[_FLOAT]:
    # This helper is used only during case setup, never in a station trial.
    return [np.empty(count, dtype=np.float64) for _ in range(number)]


def make_model_workspace(batch: SpineBatch) -> ModelWorkspace:
    """Allocate all trial outputs and scratch storage once at case start."""

    count = batch.spine_count
    vectors = _empty_float_vectors(count, 25)
    branch = np.full(count, LOWER_STOP, dtype=np.int8)
    branch[batch.rigid_mask] = RIGID
    workspace = ModelWorkspace(
        force_x_N=vectors[0],
        force_z_N=vectors[1],
        lambda_n_N=vectors[2],
        tangent_force_N=vectors[3],
        mode=np.empty(count, dtype=np.int8),
        u_t_history_m=vectors[4],
        spring_branch=branch,
        vertical_stiffness_N_per_m=vectors[5],
        spring_axial_load_N=vectors[6],
        tangent_x=vectors[7],
        tangent_z=vectors[8],
        normal_x=vectors[9],
        normal_z=vectors[10],
        normal_closure_m=vectors[11],
        tangent_trial_m=vectors[12],
        normal_axis_dot=vectors[13],
        tangent_axis_dot=vectors[14],
        normal_transverse_dot=vectors[15],
        tangent_transverse_dot=vectors[16],
        g_nn=vectors[17],
        g_nt=vectors[18],
        g_tt=vectors[19],
        determinant=vectors[20],
        rhs_normal_m=vectors[21],
        rhs_tangent_m=vectors[22],
        trial_lambda_n_N=vectors[23],
        trial_tangent_force_N=vectors[24],
        slide_denominator_m_per_N=np.empty(count, dtype=np.float64),
    )
    workspace.force_x_N.fill(0.0)
    workspace.force_z_N.fill(0.0)
    workspace.lambda_n_N.fill(0.0)
    workspace.tangent_force_N.fill(0.0)
    workspace.mode.fill(FREE)
    workspace.u_t_history_m.fill(0.0)
    workspace.vertical_stiffness_N_per_m.fill(0.0)
    workspace.spring_axial_load_N.fill(0.0)
    return workspace


def commit_model_workspace(
    state: ContactState, workspace: ModelWorkspace
) -> None:
    """Atomically copy one accepted full-array trial into the case state."""

    if state.size != workspace.size:
        raise ValueError("state and workspace sizes do not match")
    np.copyto(state.mode, workspace.mode)
    np.copyto(state.u_t_history_m, workspace.u_t_history_m)
    np.copyto(state.spring_branch, workspace.spring_branch)


def _validate_evaluation_inputs(
    batch: SpineBatch,
    previous_state: ContactState,
    workspace: ModelWorkspace,
    backplate_z_m: float,
    envelope_height_m: ArrayLike,
    envelope_slope_x: ArrayLike,
    delta_arc_m: ArrayLike,
    valid_mask: ArrayLike,
) -> tuple[_FLOAT, _FLOAT, _FLOAT, _BOOL]:
    count = batch.spine_count
    if previous_state.size != count or workspace.size != count:
        raise ValueError("batch, state, and workspace sizes must match")
    if (
        np.any(
            (previous_state.mode != FREE)
            & (previous_state.mode != STICK)
            & (previous_state.mode != SLIDE)
        )
        or np.any(
            (previous_state.spring_branch != RIGID)
            & (previous_state.spring_branch != INTERIOR)
            & (previous_state.spring_branch != LOWER_STOP)
            & (previous_state.spring_branch != HARD_STOP)
        )
        or not np.all(np.isfinite(previous_state.u_t_history_m))
    ):
        raise ValueError("previous_state contains an invalid mode or history")
    if isinstance(backplate_z_m, (bool, np.bool_)):
        raise ValueError("backplate_z_m must be finite")
    backplate = float(backplate_z_m)
    if not math.isfinite(backplate):
        raise ValueError("backplate_z_m must be finite")

    height = np.asarray(envelope_height_m, dtype=np.float64)
    slope = np.asarray(envelope_slope_x, dtype=np.float64)
    delta_arc = np.asarray(delta_arc_m, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    for name, value in (
        ("envelope_height_m", height),
        ("envelope_slope_x", slope),
        ("delta_arc_m", delta_arc),
        ("valid_mask", valid),
    ):
        if value.shape != (count,):
            raise ValueError(f"{name} must have shape ({count},)")
    if (
        not np.all(np.isfinite(height[valid]))
        or not np.all(np.isfinite(slope[valid]))
        or not np.all(np.isfinite(delta_arc[valid]))
    ):
        raise ValueError("valid terrain entries must be finite")
    return height, slope, delta_arc, valid


def _solve_closed_form(
    batch: SpineBatch,
    workspace: ModelWorkspace,
    solve_mask: _BOOL,
    parallel_compliance_m_per_N: _FLOAT,
    hard_mask: _BOOL | None,
) -> None:
    """Solve all selected contacts once for a prescribed spring branch."""

    if not np.any(solve_mask):
        return

    workspace.lambda_n_N[solve_mask] = 0.0
    workspace.tangent_force_N[solve_mask] = 0.0
    workspace.force_x_N[solve_mask] = 0.0
    workspace.force_z_N[solve_mask] = 0.0
    workspace.vertical_stiffness_N_per_m[solve_mask] = 0.0
    workspace.u_t_history_m[solve_mask] = 0.0
    workspace.mode[solve_mask] = FREE

    np.multiply(
        parallel_compliance_m_per_N,
        workspace.normal_axis_dot * workspace.normal_axis_dot,
        out=workspace.g_nn,
    )
    workspace.g_nn += (
        batch.transverse_compliance_m_per_N
        * workspace.normal_transverse_dot
        * workspace.normal_transverse_dot
    )
    np.multiply(
        parallel_compliance_m_per_N,
        workspace.normal_axis_dot * workspace.tangent_axis_dot,
        out=workspace.g_nt,
    )
    workspace.g_nt += (
        batch.transverse_compliance_m_per_N
        * workspace.normal_transverse_dot
        * workspace.tangent_transverse_dot
    )
    np.multiply(
        parallel_compliance_m_per_N,
        workspace.tangent_axis_dot * workspace.tangent_axis_dot,
        out=workspace.g_tt,
    )
    workspace.g_tt += (
        batch.transverse_compliance_m_per_N
        * workspace.tangent_transverse_dot
        * workspace.tangent_transverse_dot
    )

    np.copyto(workspace.rhs_normal_m, workspace.normal_closure_m)
    np.copyto(workspace.rhs_tangent_m, workspace.tangent_trial_m)
    if hard_mask is not None and np.any(hard_mask):
        workspace.rhs_normal_m[hard_mask] += (
            batch.spring_delta_max_m[hard_mask]
            * workspace.normal_axis_dot[hard_mask]
        )
        workspace.rhs_tangent_m[hard_mask] += (
            batch.spring_delta_max_m[hard_mask]
            * workspace.tangent_axis_dot[hard_mask]
        )

    np.multiply(
        workspace.g_nn, workspace.g_tt, out=workspace.determinant
    )
    workspace.determinant -= workspace.g_nt * workspace.g_nt
    algebraic = (
        solve_mask
        & np.isfinite(workspace.determinant)
        & (workspace.determinant > 0.0)
    )

    workspace.trial_lambda_n_N.fill(0.0)
    workspace.trial_tangent_force_N.fill(0.0)
    numerator = (
        workspace.rhs_normal_m * workspace.g_tt
        - workspace.rhs_tangent_m * workspace.g_nt
    )
    np.divide(
        numerator,
        workspace.determinant,
        out=workspace.trial_lambda_n_N,
        where=algebraic,
    )
    numerator = (
        workspace.g_nn * workspace.rhs_tangent_m
        - workspace.g_nt * workspace.rhs_normal_m
    )
    np.divide(
        numerator,
        workspace.determinant,
        out=workspace.trial_tangent_force_N,
        where=algebraic,
    )

    positive_trial = algebraic & (workspace.trial_lambda_n_N > 0.0)
    stick = positive_trial & (
        np.abs(workspace.trial_tangent_force_N)
        <= batch.static_friction * workspace.trial_lambda_n_N
    )
    workspace.lambda_n_N[stick] = workspace.trial_lambda_n_N[stick]
    workspace.tangent_force_N[stick] = (
        workspace.trial_tangent_force_N[stick]
    )
    workspace.mode[stick] = STICK
    workspace.u_t_history_m[stick] = workspace.tangent_trial_m[stick]
    stick_stiffness = (
        workspace.normal_z
        * (
            workspace.normal_z * workspace.g_tt
            - workspace.tangent_z * workspace.g_nt
        )
    )
    np.divide(
        stick_stiffness,
        workspace.determinant,
        out=workspace.vertical_stiffness_N_per_m,
        where=stick,
    )

    slide_candidate = positive_trial & ~stick
    np.subtract(
        workspace.g_nn,
        batch.kinetic_friction * workspace.g_nt,
        out=workspace.slide_denominator_m_per_N,
    )
    slide_algebraic = (
        slide_candidate
        & np.isfinite(workspace.slide_denominator_m_per_N)
        & (workspace.slide_denominator_m_per_N > 0.0)
    )
    workspace.trial_lambda_n_N.fill(0.0)
    np.divide(
        workspace.rhs_normal_m,
        workspace.slide_denominator_m_per_N,
        out=workspace.trial_lambda_n_N,
        where=slide_algebraic,
    )
    slide = slide_algebraic & (workspace.trial_lambda_n_N > 0.0)
    workspace.lambda_n_N[slide] = workspace.trial_lambda_n_N[slide]
    workspace.tangent_force_N[slide] = (
        -batch.kinetic_friction[slide]
        * workspace.trial_lambda_n_N[slide]
    )
    workspace.mode[slide] = SLIDE
    workspace.u_t_history_m[slide] = (
        workspace.g_nt[slide] * workspace.lambda_n_N[slide]
        + workspace.g_tt[slide] * workspace.tangent_force_N[slide]
    )
    if hard_mask is not None:
        hard_slide = slide & hard_mask
        workspace.u_t_history_m[hard_slide] -= (
            batch.spring_delta_max_m[hard_slide]
            * workspace.tangent_axis_dot[hard_slide]
        )
    slide_stiffness = (
        workspace.normal_z
        * (
            workspace.normal_z
            - batch.kinetic_friction * workspace.tangent_z
        )
    )
    np.divide(
        slide_stiffness,
        workspace.slide_denominator_m_per_N,
        out=workspace.vertical_stiffness_N_per_m,
        where=slide,
    )

    accepted = stick | slide
    workspace.force_x_N[accepted] = (
        workspace.lambda_n_N[accepted] * workspace.normal_x[accepted]
        + workspace.tangent_force_N[accepted]
        * workspace.tangent_x[accepted]
    )
    workspace.force_z_N[accepted] = (
        workspace.lambda_n_N[accepted] * workspace.normal_z[accepted]
        + workspace.tangent_force_N[accepted]
        * workspace.tangent_z[accepted]
    )


def evaluate_spines(
    batch: SpineBatch,
    previous_state: ContactState,
    workspace: ModelWorkspace,
    *,
    backplate_z_m: float,
    envelope_height_m: ArrayLike,
    envelope_slope_x: ArrayLike,
    delta_arc_m: ArrayLike,
    valid_mask: ArrayLike,
) -> tuple[float, float]:
    """Evaluate one common-backplate trial without committing contact history.

    Flexible spines are first evaluated on ``INTERIOR`` compliance.  Their
    axial loads select ``LOWER_STOP``, ``INTERIOR``, or ``HARD_STOP`` in one
    vectorized pass; lower/hard contacts are then recomputed exactly once with
    beam-only axial compliance.  A hard-stop recomputation includes the fixed
    ``delta_max`` displacement as an affine right-hand-side translation.
    """

    height, slope, delta_arc, valid = _validate_evaluation_inputs(
        batch,
        previous_state,
        workspace,
        backplate_z_m,
        envelope_height_m,
        envelope_slope_x,
        delta_arc_m,
        valid_mask,
    )

    workspace.force_x_N.fill(0.0)
    workspace.force_z_N.fill(0.0)
    workspace.lambda_n_N.fill(0.0)
    workspace.tangent_force_N.fill(0.0)
    workspace.mode.fill(FREE)
    workspace.u_t_history_m.fill(0.0)
    workspace.vertical_stiffness_N_per_m.fill(0.0)
    workspace.spring_axial_load_N.fill(0.0)
    workspace.spring_branch.fill(LOWER_STOP)
    workspace.spring_branch[batch.rigid_mask] = RIGID

    np.hypot(1.0, slope, out=workspace.determinant)
    np.reciprocal(workspace.determinant, out=workspace.tangent_x)
    workspace.tangent_z[:] = slope * workspace.tangent_x
    workspace.normal_x[:] = -workspace.tangent_z
    np.copyto(workspace.normal_z, workspace.tangent_x)

    np.subtract(
        height,
        float(backplate_z_m) + batch.tip_z_offset_m,
        out=workspace.normal_closure_m,
    )
    workspace.normal_closure_m *= workspace.normal_z
    np.maximum(
        workspace.normal_closure_m,
        0.0,
        out=workspace.normal_closure_m,
    )
    np.add(
        previous_state.u_t_history_m,
        delta_arc,
        out=workspace.tangent_trial_m,
    )
    # A spine that was FREE has no contact anchor and therefore cannot
    # accumulate wall arc length while detached.  Its first trial at a new
    # landing point starts with zero tangential history.
    workspace.tangent_trial_m[previous_state.mode == FREE] = 0.0
    geometric = valid & (workspace.normal_closure_m > 0.0)

    workspace.normal_axis_dot[:] = (
        workspace.normal_x * batch.axis_x
        + workspace.normal_z * batch.axis_z
    )
    workspace.tangent_axis_dot[:] = (
        workspace.tangent_x * batch.axis_x
        + workspace.tangent_z * batch.axis_z
    )
    workspace.normal_transverse_dot[:] = (
        workspace.normal_x * batch.transverse_x
        + workspace.normal_z * batch.transverse_z
    )
    workspace.tangent_transverse_dot[:] = (
        workspace.tangent_x * batch.transverse_x
        + workspace.tangent_z * batch.transverse_z
    )

    flexible_contact = geometric & ~batch.rigid_mask
    workspace.spring_branch[flexible_contact] = INTERIOR
    _solve_closed_form(
        batch,
        workspace,
        geometric,
        batch.interior_parallel_compliance_m_per_N,
        None,
    )

    workspace.spring_axial_load_N[:] = -(
        workspace.force_x_N * batch.axis_x
        + workspace.force_z_N * batch.axis_z
    )
    # Section 4.4's lambda<=0 decision is terminal for this trial.  In
    # particular, Q_s=0 on an algebraically FREE interior trial must not be
    # mistaken for a loaded LOWER_STOP contact and then "resurrected" by the
    # much stiffer beam-only solve.  Only a positive-reaction contact proceeds
    # to the one allowed spring-branch correction.
    responsive_flexible = flexible_contact & (workspace.mode != FREE)
    workspace.spring_branch[
        flexible_contact & (workspace.mode == FREE)
    ] = LOWER_STOP
    lower = responsive_flexible & (workspace.spring_axial_load_N <= 0.0)
    hard = responsive_flexible & (
        workspace.spring_axial_load_N
        >= batch.spring_stiffness_N_per_m * batch.spring_delta_max_m
    )
    correction = lower | hard
    workspace.spring_branch[lower] = LOWER_STOP
    workspace.spring_branch[hard] = HARD_STOP

    _solve_closed_form(
        batch,
        workspace,
        correction,
        batch.axial_compliance_m_per_N,
        hard,
    )
    workspace.spring_axial_load_N[:] = -(
        workspace.force_x_N * batch.axis_x
        + workspace.force_z_N * batch.axis_z
    )

    # FREE always carries zero history, including invalid/no-closure entries
    # and any branch recomputation that loses positive normal reaction.
    free = workspace.mode == FREE
    workspace.u_t_history_m[free] = 0.0

    return (
        float(np.sum(workspace.force_z_N, dtype=np.float64)),
        float(
            np.sum(
                workspace.vertical_stiffness_N_per_m,
                dtype=np.float64,
            )
        ),
    )


__all__ = [
    "FREE",
    "STICK",
    "SLIDE",
    "RIGID",
    "INTERIOR",
    "LOWER_STOP",
    "HARD_STOP",
    "SpineBatch",
    "ContactState",
    "ModelWorkspace",
    "build_spine_batch",
    "make_contact_state",
    "make_model_workspace",
    "reset_contact_state",
    "commit_model_workspace",
    "evaluate_spines",
]
