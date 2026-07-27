"""Auditable interpolation of M1 finite-tip tracks for M2."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from spine_sim.terrain.models import TrackGeometry

from .errors import ContactConfigurationError, ContactGeometryError
from .models import GeometrySample, SpineParameters


class TrackInterpolator:
    """Linear, validity-aware queries over a frozen ``TrackGeometry``."""

    def __init__(self, track: TrackGeometry, parameters: SpineParameters) -> None:
        if not math.isclose(
            track.radius_m,
            parameters.tip_radius_m,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ContactConfigurationError(
                "track radius does not match SpineParameters.tip_radius_m"
            )
        x = np.asarray(track.x_global_m, dtype=np.float64)
        if (
            x.size < 2
            or not np.all(np.isfinite(x))
            or not np.all(np.diff(x) > 0.0)
        ):
            raise ContactConfigurationError(
                "TrackGeometry x_global_m must be finite and strictly increasing"
            )
        self.track = track
        self.parameters = parameters
        self._x = x

    @property
    def valid_x_range_m(self) -> tuple[float, float]:
        valid = np.asarray(self.track.valid_mask, dtype=np.bool_)
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            raise ContactGeometryError("geometry_out_of_domain: track has no valid samples")
        return float(self._x[indices[0]]), float(self._x[indices[-1]])

    def _bracket(self, center_x_m: float) -> tuple[int, int, float]:
        if not math.isfinite(center_x_m):
            raise ContactGeometryError("invalid_geometry: non-finite center x")
        index = int(np.searchsorted(self._x, center_x_m, side="right") - 1)
        if index < 0:
            raise ContactGeometryError(
                "geometry_out_of_domain: sphere center left the M1 track"
            )
        valid = np.asarray(self.track.valid_mask, dtype=np.bool_)
        if (
            index < self._x.size - 1
            and bool(valid[index] and valid[index + 1])
        ):
            lower = index
            upper = index + 1
        elif (
            index > 0
            and math.isclose(
                center_x_m,
                float(self._x[index]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            and bool(valid[index - 1] and valid[index])
        ):
            lower = index - 1
            upper = index
        else:
            raise ContactGeometryError(
                "geometry_out_of_domain: interpolation bracket is not M1-valid"
            )
        fraction = (center_x_m - self._x[lower]) / (
            self._x[upper] - self._x[lower]
        )
        return lower, upper, float(fraction)

    @staticmethod
    def _lerp(array: NDArray[np.float64], i0: int, i1: int, t: float) -> float:
        return float((1.0 - t) * array[i0] + t * array[i1])

    def query(self, center_x_m: float) -> GeometrySample:
        i0, i1, fraction = self._bracket(center_x_m)
        height = self._lerp(
            np.asarray(self.track.envelope_height_m, dtype=np.float64),
            i0,
            i1,
            fraction,
        )
        slope = self._lerp(
            np.asarray(self.track.envelope_slope_x, dtype=np.float64),
            i0,
            i1,
            fraction,
        )
        support_x = self._lerp(
            np.asarray(self.track.support_x_m, dtype=np.float64),
            i0,
            i1,
            fraction,
        )
        support_y = self._lerp(
            np.asarray(self.track.support_y_m, dtype=np.float64),
            i0,
            i1,
            fraction,
        )
        if not all(
            math.isfinite(value)
            for value in (height, slope, support_x, support_y)
        ):
            raise ContactGeometryError(
                "invalid_geometry: M1 interpolation returned non-finite data"
            )
        radius_squared = self.track.radius_m**2
        lateral_squared = (
            (support_x - center_x_m) ** 2
            + (support_y - self.track.y_global_m) ** 2
        )
        radicand = radius_squared - lateral_squared
        roundoff = max(1e-24, 1e-10 * radius_squared)
        if radicand < -roundoff:
            raise ContactGeometryError(
                "invalid_geometry: interpolated support lies outside the finite tip"
            )
        support_z = height - math.sqrt(max(0.0, radicand))
        normalization = math.hypot(1.0, slope)
        tangent = (1.0 / normalization, slope / normalization)
        normal = (-slope / normalization, 1.0 / normalization)
        axis = self.parameters.axis_xz
        cap_margin = (
            (support_x - center_x_m) * axis[0]
            + (support_z - height) * axis[1]
        )
        near_tie = bool(
            np.asarray(self.track.near_tie_flag, dtype=np.bool_)[i0]
            or np.asarray(self.track.near_tie_flag, dtype=np.bool_)[i1]
        )
        return GeometrySample(
            center_x_m=float(center_x_m),
            envelope_height_m=height,
            envelope_slope_x=slope,
            support_xyz_m=(support_x, float(self.track.y_global_m), support_z),
            tangent_xz=tangent,
            normal_xz=normal,
            valid=True,
            near_tie=near_tie,
            cap_margin_m=float(cap_margin),
        )
