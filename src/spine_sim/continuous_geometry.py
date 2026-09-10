"""Continuous sphere/rod queries against fixed planes and triangulated height fields.

The height samples are vertices of continuous planar triangles, *not* independent
point obstacles. Each cell uses its lower-left to upper-right diagonal. ``gap_m``
is signed Euclidean distance to this surface minus the sphere radius; its gradient
is the unit physical contact normal. It is not the vertical envelope gap.

Queries contain all distinct nearest supports within the requested distance
tolerance. A single-point mechanical solver must stop or reduce its step when
``selected`` is None; choosing the first normal would hide a multi-contact state.
Feature IDs identify mesh features, not friction anchors or contact episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SphereContact:
    center_m: FloatArray
    contact_point_m: FloatArray
    normal: FloatArray
    gap_m: float
    feature_id: str
    # Derivative of normal with respect to the actual sphere center, on this
    # smooth nearest-feature branch. None means that this derivative is not unique.
    normal_jacobian: FloatArray | None


@dataclass(frozen=True)
class SphereQuery:
    status: Literal["ok", "multiple_contacts", "out_of_domain", "invalid_surface"]
    contacts: tuple[SphereContact, ...] = ()

    @property
    def selected(self) -> SphereContact | None:
        return self.contacts[0] if self.status == "ok" else None

    @property
    def gap_m(self) -> float | None:
        return min(c.gap_m for c in self.contacts) if self.contacts else None


@dataclass(frozen=True)
class RodClearance:
    status: Literal["clear", "collision", "out_of_domain", "invalid_surface", "indeterminate"]
    gap_m: float | None
    segment_index: int | None = None
    centerline_point_m: FloatArray | None = None
    surface_point_m: FloatArray | None = None


class ContinuousSurface(Protocol):
    def query_sphere(
        self, center_m: ArrayLike, radius_m: float, *, tie_tolerance_m: float = 1e-10
    ) -> SphereQuery: ...

    def query_centerline(self, points_m: ArrayLike, radius_m: float) -> RodClearance: ...


def _point(value: ArrayLike) -> FloatArray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("expected a finite three-dimensional point")
    return point


def _radius(value: float) -> float:
    if not np.isfinite(value) or value < 0:
        raise ValueError("radius_m must be finite and non-negative")
    return float(value)


def _centerline(value: ArrayLike) -> FloatArray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("centerline requires at least two 3-D points")
    if not np.all(np.isfinite(points)):
        raise ValueError("centerline must be finite")
    return points


class PlaneSurface:
    """Infinite plane, with the wall on the negative-normal side."""

    def __init__(self, point_m: ArrayLike = (0, 0, 0), normal: ArrayLike = (0, 0, 1)):
        self.point_m = _point(point_m).copy()
        self.normal = _point(normal).copy()
        length = np.linalg.norm(self.normal)
        if length == 0:
            raise ValueError("plane normal must be nonzero")
        self.normal /= length

    def query_sphere(
        self, center_m: ArrayLike, radius_m: float, *, tie_tolerance_m: float = 1e-10
    ) -> SphereQuery:
        center = _point(center_m)
        radius = _radius(radius_m)
        distance = float((center - self.point_m) @ self.normal)
        contact = SphereContact(
            center.copy(), center - distance * self.normal, self.normal.copy(),
            distance - radius, "plane", np.zeros((3, 3)),
        )
        return SphereQuery("ok", (contact,))

    def query_centerline(self, points_m: ArrayLike, radius_m: float) -> RodClearance:
        points, radius = _centerline(points_m), _radius(radius_m)
        distances = (points - self.point_m) @ self.normal
        index = int(np.argmin(distances))
        gap = float(distances[index] - radius)
        return RodClearance(
            "collision" if gap < 0 else "clear", gap, min(index, len(points) - 2),
            points[index].copy(), points[index] - distances[index] * self.normal,
        )


def _closest_triangles(point: FloatArray, triangles: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Closest point and barycentric coordinates on each continuous triangle."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    ap = point - a
    aa, bb = np.einsum("ij,ij->i", ab, ab), np.einsum("ij,ij->i", ac, ac)
    mixed = np.einsum("ij,ij->i", ab, ac)
    rhs_a, rhs_b = np.einsum("ij,ij->i", ap, ab), np.einsum("ij,ij->i", ap, ac)
    denominator = aa * bb - mixed**2
    u = (bb * rhs_a - mixed * rhs_b) / denominator
    v = (aa * rhs_b - mixed * rhs_a) / denominator
    bary = np.stack((1 - u - v, u, v), axis=1)
    projection = np.einsum("ni,nij->nj", bary, triangles)
    inside = np.min(bary, axis=1) >= 0
    best = projection.copy()
    best_distance = np.where(inside, np.sum((point - projection)**2, axis=1), np.inf)
    best_bary = bary.copy()
    for first, second in ((0, 1), (1, 2), (2, 0)):
        start = triangles[:, first]
        edge = triangles[:, second] - start
        fraction = np.clip(
            np.einsum("ij,ij->i", point - start, edge)
            / np.einsum("ij,ij->i", edge, edge), 0, 1,
        )
        nearest = start + fraction[:, None] * edge
        distance = np.sum((point - nearest)**2, axis=1)
        replace = distance < best_distance
        best[replace], best_distance[replace] = nearest[replace], distance[replace]
        edge_bary = np.zeros_like(bary)
        edge_bary[:, first], edge_bary[:, second] = 1 - fraction, fraction
        best_bary[replace] = edge_bary[replace]
    return best, best_bary


def _segment_edges(
    start: FloatArray, end: FloatArray, edge_start: FloatArray, edge_end: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Closest points between one segment and an array of edge segments."""
    u, v, w = end - start, edge_end - edge_start, start - edge_start
    a = float(u @ u)
    c = np.einsum("ij,ij->i", v, v)
    if a == 0:
        t = np.clip(np.einsum("ij,ij->i", start - edge_start, v) / c, 0, 1)
        return np.broadcast_to(start, edge_start.shape).copy(), edge_start + t[:, None] * v
    b, d, e = v @ u, w @ u, np.einsum("ij,ij->i", v, w)
    denominator = a * c - b*b
    s = np.zeros(len(v))
    nonparallel = denominator > a*c*1e-14
    s[nonparallel] = np.clip(
        (b[nonparallel]*e[nonparallel] - c[nonparallel]*d[nonparallel])
        / denominator[nonparallel], 0, 1,
    )
    t = (b*s + e) / c
    before, after = t < 0, t > 1
    s[before] = np.clip(-d[before]/a, 0, 1)
    s[after] = np.clip((b[after] - d[after])/a, 0, 1)
    t = np.clip(t, 0, 1)
    return start + s[:, None]*u, edge_start + t[:, None]*v


def _segment_triangles(
    start: FloatArray, end: FloatArray, triangles: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Exact unsigned segment-to-triangle distance witnesses."""
    near_start, _ = _closest_triangles(start, triangles)
    near_end, _ = _closest_triangles(end, triangles)
    d_start = np.sum((near_start - start)**2, axis=1)
    d_end = np.sum((near_end - end)**2, axis=1)
    take_end = d_end < d_start
    surface = np.where(take_end[:, None], near_end, near_start)
    centerline = np.where(take_end[:, None], end, start)
    distance = np.minimum(d_start, d_end)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        line_point, face_point = _segment_edges(start, end, triangles[:, first], triangles[:, second])
        candidate_distance = np.sum((line_point - face_point)**2, axis=1)
        replace = candidate_distance < distance
        centerline[replace], surface[replace] = line_point[replace], face_point[replace]
        distance[replace] = candidate_distance[replace]
    ab, ac = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    normal = np.cross(ab, ac)
    denominator = normal @ (end - start)
    fraction = np.full(len(triangles), np.inf)
    crossing = np.abs(denominator) > np.linalg.norm(normal, axis=1)*np.linalg.norm(end-start)*1e-14
    fraction[crossing] = np.einsum(
        "ij,ij->i", triangles[crossing, 0] - start, normal[crossing],
    ) / denominator[crossing]
    crossing &= (fraction >= 0) & (fraction <= 1)
    indices = np.flatnonzero(crossing)
    if len(indices):
        intersection = start + fraction[indices, None]*(end-start)
        # Barycentric coordinates of the intersection with each supporting plane.
        ap = intersection - triangles[indices, 0]
        aa = np.sum(ab[indices]**2, axis=1)
        bb = np.sum(ac[indices]**2, axis=1)
        mixed = np.sum(ab[indices]*ac[indices], axis=1)
        ra, rb = np.sum(ap*ab[indices], axis=1), np.sum(ap*ac[indices], axis=1)
        det = aa*bb-mixed*mixed
        u, v = (bb*ra-mixed*rb)/det, (aa*rb-mixed*ra)/det
        inside = (u >= -1e-12) & (v >= -1e-12) & (u+v <= 1+1e-12)
        centerline[indices[inside]] = intersection[inside]
        surface[indices[inside]] = intersection[inside]
    return centerline, surface


class HeightFieldSurface:
    """Finite continuous triangulation of ``height_m[y, x]``.

    No extrapolation is performed. The full horizontal sphere/capsule footprint
    must remain in the data rectangle. Invalid vertices within that footprint
    make the query indeterminate, rather than silently filling a measured hole.
    Nearest-point searches use the vertical projection as an upper distance bound.
    """

    def __init__(
        self, height_m: ArrayLike, dx_m: float, dy_m: float,
        origin_xy_m: tuple[float, float] = (0.0, 0.0), *, valid_mask: ArrayLike | None = None,
    ):
        self.height_m = np.asarray(height_m, dtype=float)
        if self.height_m.ndim != 2 or min(self.height_m.shape) < 2:
            raise ValueError("height_m requires at least a 2 by 2 grid")
        if not np.isfinite(dx_m) or not np.isfinite(dy_m) or min(dx_m, dy_m) <= 0:
            raise ValueError("grid spacings must be positive and finite")
        self.dx_m, self.dy_m = float(dx_m), float(dy_m)
        self.origin_xy_m = np.asarray(origin_xy_m, dtype=float)
        if self.origin_xy_m.shape != (2,) or not np.all(np.isfinite(self.origin_xy_m)):
            raise ValueError("origin_xy_m must be a finite 2-vector")
        self.valid_mask = np.isfinite(self.height_m)
        if valid_mask is not None:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.shape != self.height_m.shape:
                raise ValueError("valid_mask must have the height field shape")
            self.valid_mask &= mask
        self.maximum_xy_m = self.origin_xy_m + np.array(
            [(self.height_m.shape[1]-1)*self.dx_m, (self.height_m.shape[0]-1)*self.dy_m],
        )

    @classmethod
    def from_terrain(cls, terrain, *, origin_xy_m=(0.0, 0.0)) -> HeightFieldSurface:
        """Use the existing terrain API's grid and mask without track extraction."""
        return cls(terrain.height, terrain.dx, terrain.dy, origin_xy_m, valid_mask=terrain.valid_mask)

    def _contains(self, xy: FloatArray, radius: float) -> bool:
        return bool(np.all(xy-radius >= self.origin_xy_m) and np.all(xy+radius <= self.maximum_xy_m))

    def height_at(self, x_m: float, y_m: float) -> float:
        """Piecewise-linear height at an arbitrary in-domain horizontal position."""
        xy = np.array((x_m, y_m))
        if not self._contains(xy, 0):
            raise ValueError("height query is outside the finite height field")
        uv = (xy-self.origin_xy_m)/np.array((self.dx_m, self.dy_m))
        col = min(int(np.floor(uv[0])), self.height_m.shape[1]-2)
        row = min(int(np.floor(uv[1])), self.height_m.shape[0]-2)
        u, v = uv-np.array((col, row))
        a, b = self.height_m[row, col], self.height_m[row, col+1]
        c, d = self.height_m[row+1, col+1], self.height_m[row+1, col]
        return float((1-u)*a+(u-v)*b+v*c if u >= v else (1-v)*a+u*c+(v-u)*d)

    def _triangles(self, lower: FloatArray, upper: FloatArray) -> tuple[FloatArray, NDArray[np.int64], bool]:
        spacing = np.array((self.dx_m, self.dy_m))
        first = np.floor((lower-self.origin_xy_m)/spacing).astype(int)
        last = np.floor((upper-self.origin_xy_m)/spacing).astype(int)
        # Include cells incident to an exact grid coordinate on the lower bound.
        first -= 1
        limits = np.array((self.height_m.shape[1]-2, self.height_m.shape[0]-2))
        first, last = np.clip(first, 0, limits), np.clip(last, 0, limits)
        rows, cols = np.meshgrid(np.arange(first[1], last[1]+1), np.arange(first[0], last[0]+1), indexing="ij")
        width = self.height_m.shape[1]
        a = (rows*width+cols).ravel()
        vertex_ids = np.stack((np.stack((a, a+1, a+width+1), axis=1),
                               np.stack((a, a+width+1, a+width), axis=1)), axis=1).reshape(-1, 3)
        valid = np.all(self.valid_mask.ravel()[vertex_ids], axis=1)
        vertex_ids = vertex_ids[valid]
        x = self.origin_xy_m[0]+(vertex_ids % width)*self.dx_m
        y = self.origin_xy_m[1]+(vertex_ids // width)*self.dy_m
        z = self.height_m.ravel()[vertex_ids]
        return np.stack((x, y, z), axis=2), vertex_ids, bool(np.all(valid))

    def query_sphere(
        self, center_m: ArrayLike, radius_m: float, *, tie_tolerance_m: float = 1e-10,
    ) -> SphereQuery:
        center, radius = _point(center_m), _radius(radius_m)
        if tie_tolerance_m < 0:
            raise ValueError("tie_tolerance_m must be non-negative")
        if not self._contains(center[:2], radius):
            return SphereQuery("out_of_domain")
        vertical = center[2]-self.height_at(*center[:2])
        if not np.isfinite(vertical):
            return SphereQuery("invalid_surface")
        reach = max(abs(vertical)+tie_tolerance_m, radius)
        triangles, ids, complete = self._triangles(center[:2]-reach, center[:2]+reach)
        if not complete:
            return SphereQuery("invalid_surface")
        closest, barycentric = _closest_triangles(center, triangles)
        distances = np.linalg.norm(center-closest, axis=1)
        nearest = np.flatnonzero(distances <= np.min(distances)+tie_tolerance_m)
        sign = 1.0 if vertical >= 0 else -1.0
        # A closest point on one triangle edge may still be downhill along its
        # neighbor. It is then a triangulation artifact, not a second physical
        # support (especially near the diagonal seam of a flat cell).
        local_minima = []
        for index in nearest:
            active = ids[index][barycentric[index] > 1e-10]
            if len(active) < 3:
                incident = (active[:, None, None] == ids[None, :, :]).any(axis=2).all(axis=0)
                directions = triangles[incident].reshape(-1, 3)-closest[index]
                tolerance = 1e-12*max(distances[index], self.dx_m, self.dy_m)*max(self.dx_m, self.dy_m)
                if np.any(directions @ (center-closest[index]) > tolerance):
                    continue
            local_minima.append(index)
        nearest = np.asarray(local_minima, dtype=int)
        # Duplicate face representations at a shared edge/vertex are one support.
        spatial_tolerance = max(1e-13*max(self.dx_m, self.dy_m), 1e-15)
        groups: list[list[int]] = []
        for index in nearest:
            for group in groups:
                if np.linalg.norm(closest[index]-closest[group[0]]) <= spatial_tolerance:
                    group.append(int(index))
                    break
            else:
                groups.append([int(index)])
        contacts = []
        for group in groups:
            index = min(group, key=lambda item: distances[item])
            distance = float(distances[index])
            faces = triangles[group]
            face_normals = np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])
            face_normals /= np.linalg.norm(face_normals, axis=1)[:, None]
            active_vertices = ids[index][barycentric[index] > 1e-10]
            label = ":".join(str(int(v)) for v in sorted(active_vertices))
            feature = ("vertex", "edge", "face")[len(active_vertices)-1] + ":" + label
            if distance <= 1e-15:
                # At a sharp feature occupied by the sphere center, the normal
                # cone has no uniquely selected radial direction.
                if np.max(np.linalg.norm(face_normals-face_normals[0], axis=1)) > 1e-10:
                    for normal in face_normals:
                        contacts.append(SphereContact(center.copy(), closest[index].copy(), normal,
                                                      -radius, feature, None))
                    continue
                normal, jacobian = face_normals[0], np.zeros((3, 3))
            else:
                normal = sign*(center-closest[index])/distance
                _, singular, vh = np.linalg.svd(face_normals, full_matrices=True)
                rank = int(np.sum(singular > singular[0]*1e-10))
                tangent = vh[rank:].T
                projector = tangent @ tangent.T
                jacobian = sign*(np.eye(3)-projector-np.outer(normal, normal))/distance
                # On a ridge/corner boundary, a neighboring smooth branch has a
                # different Hessian even though its first derivative agrees.
                if rank > 1 and np.max(face_normals @ normal) > 1-1e-10:
                    jacobian = None
            contacts.append(SphereContact(center.copy(), closest[index].copy(), normal,
                                          sign*distance-radius, feature, jacobian))
        status = "ok" if len(contacts) == 1 else "multiple_contacts"
        if status != "ok":
            contacts = [SphereContact(c.center_m, c.contact_point_m, c.normal, c.gap_m,
                                      c.feature_id, None) for c in contacts]
        return SphereQuery(status, tuple(contacts))

    def query_centerline(self, points_m: ArrayLike, radius_m: float) -> RodClearance:
        """Clearance of a union of equal-radius capsules along a curved rod.

        Positive clearance is exact for this polyline/triangular discretization.
        Negative values certify collision; they are not penetration-depth solutions
        for an already submerged segment. Refine the rod polyline separately from
        the surface mesh when assessing the continuum rod's geometric error.
        """
        points, radius = _centerline(points_m), _radius(radius_m)
        if not all(self._contains(point[:2], radius) for point in points):
            return RodClearance("out_of_domain", None)
        best: RodClearance | None = None
        for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
            endpoint_queries = [self.query_sphere(point, radius) for point in (start, end)]
            if any(query.status == "invalid_surface" for query in endpoint_queries):
                return RodClearance("invalid_surface", None)
            submerged = next((query for query in endpoint_queries if query.gap_m is not None and query.gap_m < 0), None)
            if submerged is not None:
                contact = min(submerged.contacts, key=lambda item: item.gap_m)
                return RodClearance("collision", contact.gap_m, index, contact.center_m, contact.contact_point_m)
            reach = min(abs(point[2]-self.height_at(*point[:2])) for point in (start, end))
            triangles, _, complete = self._triangles(
                np.minimum(start[:2], end[:2])-max(reach, radius),
                np.maximum(start[:2], end[:2])+max(reach, radius),
            )
            if not complete:
                return RodClearance("invalid_surface", None)
            on_line, on_surface = _segment_triangles(start, end, triangles)
            distances = np.linalg.norm(on_line-on_surface, axis=1)
            nearest = int(np.argmin(distances))
            gap = float(distances[nearest]-radius)
            result = RodClearance("collision" if gap < 0 else "clear", gap, index,
                                  on_line[nearest], on_surface[nearest])
            if result.status == "collision":
                return result
            if best is None or gap < best.gap_m:
                best = result
        assert best is not None
        return best
