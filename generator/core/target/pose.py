"""Keep rigid target geometry independent of its pose-update history."""

from __future__ import annotations

import drjit as dr
import mitsuba as mi
from sionna.rt import SceneObject
from sionna.rt.utils import rotation_matrix


class MeshPoseReference:
    """Retain unposed vertices and restore a target's current absolute pose.

    Incremental float32 rotations can make a planar mesh face nonplanar after
    repeated updates. Rebuilding from the loaded vertices bounds rounding to
    one pose, while keeping Sionna's public position/orientation/scale contract.
    """

    def __init__(self, target: SceneObject):
        parameters = mi.traverse(target.mi_mesh)
        vertices = dr.unravel(mi.Point3f, dr.copy(parameters["vertex_positions"]))
        self.vertices = vertices - target.position

    def restore(self, target: SceneObject) -> None:
        """Write rigid vertices without changing the requested AABB center."""
        position = target.position
        vertices = rotation_matrix(target.orientation) @ (target.scaling * self.vertices)
        center = mi.Point3f(
            (dr.min(vertices.x) + dr.max(vertices.x)) * 0.5,
            (dr.min(vertices.y) + dr.max(vertices.y)) * 0.5,
            (dr.min(vertices.z) + dr.max(vertices.z)) * 0.5,
        )
        scene = target.scene
        if scene is None:
            parameters = mi.traverse(target.mi_mesh)
            key = "vertex_positions"
        else:
            parameters = scene.mi_scene_params
            key = f"{target.mi_mesh.id()}.vertex_positions"
        parameters[key] = dr.ravel(vertices - center + position)
        parameters.update()
        if scene is not None:
            scene.scene_geometry_updated()
