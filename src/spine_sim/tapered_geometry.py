"""Clearance of the actual bent circular shaft/taper, without capsule end caps.

The exposed body is the union of normal circular sections along the exact rod
arcs. The forward spherical cap is checked by the contact solver separately.
Axial intervals are enclosed by their midpoint disk plus a proved Hausdorff
error. A height-field check minimizes each triangle's plane height difference
over the projected disk/triangle intersection; it never samples only vertices
or the rod endpoints. Intervals that cannot be resolved at the requested error
are reported as indeterminate, rather than being declared collisions.
"""

from __future__ import annotations

import numpy as np

from .continuous_geometry import HeightFieldSurface, PlaneSurface, RodClearance


def _disk_frame(tangent):
    reference = np.eye(3)[int(np.argmin(np.abs(tangent)))]
    first = reference - tangent * (reference @ tangent)
    first /= np.linalg.norm(first)
    return np.column_stack((first, np.cross(tangent, first)))


def _disk_triangle_minima(center, frame, radius, triangles, expansion):
    """Signed normal gaps on disks clipped by triangle XY half-planes.

    Expanding each XY half-plane by ``expansion`` includes the projection of
    every point within that distance of the disk. Subtracting the same distance
    from each unit-plane gap gives a conservative bound for that neighborhood.
    Returns one minimum per triangle (infinity for an empty intersection).
    """
    xy = triangles[:, :, :2]
    edges = np.roll(xy, -1, axis=1) - xy
    outward = np.stack((edges[:, :, 1], -edges[:, :, 0]), axis=2)
    outward /= np.linalg.norm(outward, axis=2)[:, :, None]
    matrix = radius * np.einsum("nei,ij->nej", outward, frame[:2])
    bounds = np.einsum("nei,nei->ne", outward, xy-center[:2]) + expansion
    normals = np.cross(triangles[:, 1]-triangles[:, 0],
                       triangles[:, 2]-triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    normals *= np.where(normals[:, 2] >= 0, 1.0, -1.0)[:, None]
    constant = np.einsum("ni,ni->n", normals, center-triangles[:, 0])
    linear = radius * (normals @ frame)
    best = np.full(len(triangles), np.inf)
    witnesses = np.zeros((len(triangles), 2))
    # These tolerances affect only floating-point feasibility of intersections,
    # not the physical collision tolerance or the axial enclosure error.
    feasible_tolerance = 4e-13 * max(radius, float(np.max(np.linalg.norm(edges, axis=2))))

    def consider(candidate, valid=None):
        finite = np.all(np.isfinite(candidate), axis=1)
        if valid is not None:
            finite &= valid
        feasible = (finite & (np.sum(candidate*candidate, axis=1) <= 1+2e-12)
                    & np.all(np.einsum("nei,ni->ne", matrix, candidate)
                             <= bounds+feasible_tolerance, axis=1))
        values = constant + np.sum(linear*candidate, axis=1)
        improve = feasible & (values < best)
        best[improve] = values[improve]
        witnesses[improve] = candidate[improve]

    norm = np.linalg.norm(linear, axis=1)
    consider(-linear / np.where(norm > 0, norm, 1)[:, None])
    # A linear minimum on disk intersect half-planes occurs on its free circle,
    # a line/circle intersection, or an intersection of two boundary lines.
    for edge in range(3):
        row, offset = matrix[:, edge], bounds[:, edge]
        length2 = np.sum(row*row, axis=1)
        valid = (length2 > 1e-30) & (offset*offset <= length2*(1+2e-12))
        denominator = np.where(length2 > 1e-30, length2, 1)
        middle = row * (offset/denominator)[:, None]
        transverse = np.stack((-row[:, 1], row[:, 0]), axis=1)
        scale = np.sqrt(np.maximum(length2-offset*offset, 0)) / denominator
        consider(middle + transverse*scale[:, None], valid)
        consider(middle - transverse*scale[:, None], valid)
    for first, second in ((0, 1), (1, 2), (2, 0)):
        a, b = matrix[:, first], matrix[:, second]
        determinant = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
        valid = np.abs(determinant) > 1e-14*np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
        denominator = np.where(valid, determinant, 1)
        candidate = np.column_stack((
            (bounds[:, first]*b[:, 1]-a[:, 1]*bounds[:, second])/denominator,
            (a[:, 0]*bounds[:, second]-bounds[:, first]*b[:, 0])/denominator,
        ))
        consider(candidate, valid)
    return best-expansion, center+radius*(witnesses @ frame.T)


def _heightfield_disk(surface, center, tangent, radius, expansion):
    extent = radius*np.sqrt(np.maximum(1-tangent[:2]**2, 0)) + expansion
    lower, upper = center[:2]-extent, center[:2]+extent
    if np.any(lower < surface.origin_xy_m) or np.any(upper > surface.maximum_xy_m):
        return RodClearance("out_of_domain", None)
    spacing = np.array((surface.dx_m, surface.dy_m))
    first = np.floor((lower-surface.origin_xy_m)/spacing).astype(int)-1
    last = np.floor((upper-surface.origin_xy_m)/spacing).astype(int)
    limits = np.array((surface.height_m.shape[1]-2, surface.height_m.shape[0]-2))
    first, last = np.clip(first, 0, limits), np.clip(last, 0, limits)
    patch = surface.height_m[first[1]:last[1]+2, first[0]:last[0]+2]
    mask = surface.valid_mask[first[1]:last[1]+2, first[0]:last[0]+2]
    vertical_lower = center[2]-radius*np.sqrt(max(1-tangent[2]**2, 0))-expansion
    if np.all(mask) and vertical_lower > np.max(patch):
        return RodClearance("clear", float(vertical_lower-np.max(patch)),
                            centerline_point_m=center)
    rows, cols = np.meshgrid(np.arange(first[1], last[1]+1),
                             np.arange(first[0], last[0]+1), indexing="ij")
    width = surface.height_m.shape[1]
    start = (rows*width+cols).ravel()
    ids = np.stack((np.stack((start, start+1, start+width+1), axis=1),
                    np.stack((start, start+width+1, start+width), axis=1)), axis=1).reshape(-1, 3)
    valid = np.all(surface.valid_mask.ravel()[ids], axis=1)
    # Invalid heights are filled only to test horizontal overlap; any overlap
    # still produces invalid_surface and never contributes a physical height.
    height = np.where(np.isfinite(surface.height_m.ravel()[ids]),
                      surface.height_m.ravel()[ids], 0)
    triangles = np.stack((surface.origin_xy_m[0]+(ids % width)*surface.dx_m,
                          surface.origin_xy_m[1]+(ids // width)*surface.dy_m,
                          height), axis=2)
    gaps, witnesses = _disk_triangle_minima(center, _disk_frame(tangent), radius, triangles, expansion)
    if np.any(np.isfinite(gaps) & ~valid):
        return RodClearance("invalid_surface", None)
    index = int(np.argmin(gaps))
    if not np.isfinite(gaps[index]):
        return RodClearance("indeterminate", None)
    point = witnesses[index]
    face = triangles[index]
    normal = np.cross(face[1]-face[0], face[2]-face[0])
    height_at_xy = face[0, 2] - normal[:2] @ (point[:2]-face[0, :2])/normal[2]
    return RodClearance("collision" if gaps[index] < 0 else "clear", float(gaps[index]),
                        centerline_point_m=center.copy(),
                        surface_point_m=np.array((point[0], point[1], height_at_xy)))


def _disk_query(surface, center, tangent, radius, expansion):
    if isinstance(surface, PlaneSurface):
        transverse = surface.normal-tangent*(surface.normal @ tangent)
        length = np.linalg.norm(transverse)
        nearest = center-radius*transverse/length if length > 0 else center.copy()
        gap = float((nearest-surface.point_m) @ surface.normal-expansion)
        return RodClearance("collision" if gap < 0 else "clear", gap,
                            centerline_point_m=center.copy(),
                            surface_point_m=nearest-(gap+expansion)*surface.normal)
    if isinstance(surface, HeightFieldSurface):
        return _heightfield_disk(surface, center, tangent, radius, expansion)
    raise TypeError("Tapered body clearance requires a PlaneSurface or HeightFieldSurface")


def query_tapered_rod_clearance(surface, rod, coordinates, position, tolerance_m):
    """Certified body classification to ``tolerance_m`` for the current rod arcs.

    ``position`` is the shared backplate translation, added to the rod's existing
    guide coordinates. ``clear`` permits penetration no deeper than the supplied
    tolerance. Its gap is a conservative lower bound, not an exact nearest-body
    distance. ``collision`` always has an actual section/surface witness.
    """
    if tolerance_m <= 0:
        raise ValueError("A positive body geometry tolerance is required")
    boundaries = rod.segment_boundaries_m(coordinates)
    length = float(boundaries[-1])
    if length <= 0:
        return RodClearance("indeterminate", None)
    nodes, tangents = rod.centerline_at(coordinates, boundaries)
    cosines = np.clip(np.sum(tangents[:-1]*tangents[1:], axis=1), -1, 1)
    crosses = np.cross(tangents[:-1], tangents[1:])
    sines = np.linalg.norm(crosses, axis=1)
    angles = np.arctan2(sines, cosines)
    curvatures = angles/np.diff(boundaries)
    p = rod.parameters
    kink = length-p.taper_length_m
    breaks = np.unique(np.append(boundaries, kink)) if 0 < kink < length else boundaries
    stack = [(float(a), float(b), 0) for a, b in zip(breaks[:-1], breaks[1:])]
    best = np.inf
    best_witness = None
    translation = np.asarray(position, dtype=float)
    while stack:
        first, last, depth = stack.pop()
        middle, half = (first+last)/2, (last-first)/2
        segment = min(int(np.searchsorted(boundaries, middle, side="right")-1), len(curvatures)-1)
        offset = middle-boundaries[segment]
        left, curvature = tangents[segment], curvatures[segment]
        if angles[segment] < 1e-10:
            center = nodes[segment]+offset*left+translation
            tangent = left
        else:
            bend = np.cross(crosses[segment]/sines[segment], left)
            angle = offset*curvature
            center = (nodes[segment]+(np.sin(angle)*left+2*np.sin(angle/2)**2*bend)/curvature
                      +translation)
            tangent = np.cos(angle)*left+np.sin(angle)*bend
        radius = float(p.section_radius_m(length-middle))
        radii = p.section_radius_m(length-np.array((first, last)))
        radius_slope = abs(float(radii[1]-radii[0]))/(last-first)
        error = half*(1+radius_slope+float(np.max(radii))*curvatures[segment])
        enclosure = _disk_query(surface, center, tangent, radius, error)
        if enclosure.gap_m is not None and enclosure.gap_m >= -tolerance_m:
            if enclosure.gap_m < best:
                best, best_witness = enclosure.gap_m, enclosure
            continue
        actual = _disk_query(surface, center, tangent, radius, 0)
        if actual.status in ("out_of_domain", "invalid_surface"):
            return actual
        if actual.gap_m is not None and actual.gap_m < -tolerance_m:
            return RodClearance("collision", actual.gap_m, segment,
                                actual.centerline_point_m, actual.surface_point_m)
        if error <= tolerance_m/4 or depth >= 24:
            return RodClearance("indeterminate", actual.gap_m, segment,
                                actual.centerline_point_m, actual.surface_point_m)
        stack.extend(((first, middle, depth+1), (middle, last, depth+1)))
    return RodClearance("clear", float(best),
                        centerline_point_m=best_witness.centerline_point_m,
                        surface_point_m=best_witness.surface_point_m)
