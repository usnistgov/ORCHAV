"""Position checks against preview geometry, without modifying simulation data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlacementIssue:
    actor_id: object
    actor_name: str
    code: str
    message: str
    position: tuple


class Surface:
    def __init__(self, mesh):
        self.vertices, self.faces = mesh.vertices, mesh.triangles
        triangles = self.vertices[self.faces]
        self.low = triangles[:, :, :2].min(axis=1)
        self.high = triangles[:, :, :2].max(axis=1)

    def heights(self, xy):
        """Intersect a vertical ray with triangles, retaining sloped surfaces."""
        xy = np.asarray(xy)
        candidates = np.flatnonzero(
            np.all(self.low <= xy + 1e-7, axis=1) & np.all(self.high >= xy - 1e-7, axis=1)
        )
        if not len(candidates):
            return np.empty(0)
        triangles = np.asarray(self.vertices[self.faces[candidates]], dtype=float)
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        u, v, q = b - a, c - a, xy - a[:, :2]
        det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        valid = np.abs(det) > 1e-10
        safe = np.where(valid, det, 1)
        s = (q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0]) / safe
        t = (u[:, 0] * q[:, 1] - u[:, 1] * q[:, 0]) / safe
        valid &= (s >= -1e-7) & (t >= -1e-7) & (s + t <= 1 + 1e-7)
        z = a[:, 2] + s * u[:, 2] + t * v[:, 2]
        return np.unique(np.round(z[valid], 6))


class SpatialScene:
    """Checks named ground/building meshes at the displayed actor positions.

    Bounding rectangles are used only for outside-environment checks. Building
    interiors use actual mesh intersections, never the aggregate building AABB.
    """

    def __init__(self, assets):
        self.ground, self.buildings = [], []
        bounds = []
        for asset in assets:
            if asset.material.shader_variant == "reference_image" or not len(asset.mesh.vertices):
                continue
            bounds.extend((asset.mesh.vertices.min(axis=0), asset.mesh.vertices.max(axis=0)))
            name = asset.name.lower()
            if any(word in name for word in ("ground", "terrain", "street", "floor", "landuse")):
                self.ground.append(Surface(asset.mesh))
            elif any(word in name for word in ("building", "obstacle", "cube")):
                self.buildings.append(Surface(asset.mesh))
        self.bounds = (np.min(bounds, axis=0), np.max(bounds, axis=0)) if bounds else None

    def ground_height(self, point):
        heights = [h for surface in self.ground for h in surface.heights(point[:2])]
        return max(heights) if heights else None

    def inside_building(self, point):
        for surface in self.buildings:
            heights = surface.heights(point[:2])
            if len(heights) >= 2 and heights[0] + 0.01 < point[2] < heights[-1] - 0.01:
                if np.count_nonzero(heights > point[2]) % 2:
                    return True
        return False

    def check(self, actor_id, name, position, *, bottom_z=None):
        position = tuple(float(v) for v in position)
        results = []

        def issue(code, message):
            results.append(PlacementIssue(actor_id, name, code, message, position))

        if self.bounds is not None:
            low, high = self.bounds
            if np.any(np.asarray(position[:2]) < low[:2] - 0.05) or np.any(
                np.asarray(position[:2]) > high[:2] + 0.05
            ):
                issue("outside", "Outside the environment's horizontal bounds")
        height = self.ground_height(position)
        bottom = position[2] if bottom_z is None else bottom_z
        if height is not None and bottom < height - 0.05:
            issue("below_ground", f"Below the ground by {height-bottom:.2f} m")
        if self.inside_building(position):
            issue("inside_building", "Actor origin is inside a building mesh")
        return tuple(results)

    def corrected_position(self, position, *, bottom_z=None, clear=False):
        """Snap in place or find nearby clear ground; never guess a missing floor."""
        position = np.asarray(position, dtype=float)
        offset = max(0.0, position[2] - bottom_z) if bottom_z is not None else 0.0
        candidates = [position[:2]]
        if clear and self.bounds is not None:
            low, high = self.bounds
            centre = np.clip(position[:2], low[:2] + 0.1, high[:2] - 0.1)
            candidates.append(centre)
            radius = max(np.linalg.norm(high[:2] - low[:2]), 1)
            for distance in np.geomspace(0.5, radius, 24):
                for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                    candidates.append(
                        np.clip(
                            centre + distance * np.array((np.cos(angle), np.sin(angle))),
                            low[:2] + 0.1,
                            high[:2] - 0.1,
                        )
                    )
            candidates.sort(key=lambda xy: np.linalg.norm(xy - position[:2]))
        for xy in candidates:
            height = self.ground_height(xy)
            if height is None:
                continue
            point = (*xy, height + offset + 0.02)
            # Test above the floor even for point radios whose anchor is at Z=0.
            probe = (*xy, max(point[2], height + 0.1))
            if not clear or not self.check(None, "", probe):
                return np.asarray(point)
        raise ValueError(
            "No suitable ground was found. Choose a different location or edit the route."
        )
