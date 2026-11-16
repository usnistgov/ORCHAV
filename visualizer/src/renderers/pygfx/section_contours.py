"""Architectural section lines on the existing world-space cutaway planes.

Native slice materials share static mesh buffers. A broad-phase bounds check
keeps meshes away from the cut out of the extra draw pass; targets and sensing
overlays never participate.
"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _section_material_class():
    """Clip native slice endpoints before perspective division.

    pygfx 0.17 expands slices in screen space with w=1. Without first clipping
    to the camera near plane, segments behind the eye can streak across a city
    view. Keep the native buffer bindings and line shader, correcting only that
    projection step for our contour material.
    """
    import pygfx as gfx
    from pygfx.renderers.wgpu import register_wgpu_render_function
    from pygfx.renderers.wgpu.shaders.meshshader import MeshSliceShader

    class SectionMaterial(gfx.MeshSliceMaterial):
        pass

    @register_wgpu_render_function(gfx.Mesh, SectionMaterial)
    class SectionShader(MeshSliceShader):
        def get_code(self):
            code = super().get_code()
            anchor = '// Don\'t forget to "normalize"!'
            if code.count(anchor) != 1:
                raise RuntimeError("Unsupported pygfx mesh slice projection")
            code = code.replace(
                anchor,
                """
                // WebGPU clip coordinates have z >= 0 at the near plane.
                if (npos_a.z <= 0.0 && npos_b.z <= 0.0) {
                    var varyings: Varyings;
                    varyings.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
                    return varyings;
                }
                if (npos_a.z < 0.0) {
                    npos_a = mix(npos_a, npos_b, -npos_a.z / (npos_b.z - npos_a.z));
                } else if (npos_b.z < 0.0) {
                    npos_b = mix(npos_b, npos_a, -npos_b.z / (npos_a.z - npos_b.z));
                }
                """,
            )
            anchor = "segment_length = length(v0);"
            if code.count(anchor) != 1:
                raise RuntimeError("Unsupported pygfx mesh slice segment expansion")
            return code.replace(
                anchor,
                anchor + """
                if (segment_length < 0.00001) {
                    var varyings: Varyings;
                    varyings.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
                    return varyings;
                }
                """,
            )

    return SectionMaterial


def _visible(obj):
    while obj is not None:
        if not obj.visible:
            return False
        obj = obj.parent
    return True


def _crosses_plane(bounds, plane):
    if bounds is None:
        return False
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        return False
    normal = np.asarray(plane[:3])
    extent = float(np.abs(normal) @ ((bounds[1] - bounds[0]) * 0.5))
    # A coplanar ground sheet has no well-defined section contour.
    return extent > 1e-8 and abs(normal @ bounds.mean(axis=0) - plane[3]) <= extent + 1e-8


class PygfxSectionContoursMixin:
    """Update optional cut lines only when planes or static geometry change."""

    def set_section_contours(self, enabled: bool, *, width: float = 2.0) -> bool:
        width = float(width)
        if not np.isfinite(width) or not 0.5 <= width <= 6.0:
            raise ValueError("Section contour width must be between 0.5 and 6 pixels")
        state = self.get_section_contour_state()
        if state["enabled"] == bool(enabled) and state["width"] == width:
            return True
        self._section_contours_enabled = bool(enabled)
        self._section_contour_width = width
        self._sync_section_contours()
        self.request_redraw()
        return True

    def get_section_contour_state(self) -> dict:
        return {
            "enabled": getattr(self, "_section_contours_enabled", False),
            "width": getattr(self, "_section_contour_width", 2.0),
            "mesh_count": len(getattr(self, "_section_contours", {})),
        }

    def _clear_section_contours(self, *, reset: bool = True) -> None:
        group = getattr(self, "_section_contour_group", None)
        if group is not None and group.parent is not None:
            group.parent.remove(group)
        self._section_contour_group = None
        self._section_contours = {}
        self._section_contour_signature = None
        if reset:
            self._section_contours_enabled = False

    def _sync_section_contours(self) -> None:
        planes = tuple(getattr(self, "_clipping_planes", ()))
        static = getattr(self, "_static_group", None)
        if (
            not getattr(self, "_section_contours_enabled", False)
            or not planes
            or static is None
            or getattr(self, "_scene", None) is None
        ):
            if getattr(self, "_section_contour_group", None) is not None:
                self._clear_section_contours(reset=False)
            return

        sources, revisions = [], []
        for source in static.children:
            name = self._reverse_objects.get(id(source))
            if not (
                name
                and self._is_scene_mesh_name(name)
                and isinstance(source, self._gfx.Mesh)
                and _visible(source)
                and source.material.opacity > 0
            ):
                continue
            geometry = source.geometry
            sources.append(source)
            revisions.append(
                (
                    id(source),
                    id(geometry),
                    geometry.positions.rev,
                    geometry.indices.rev,
                    source.world.last_modified,
                )
            )
        width = getattr(self, "_section_contour_width", 2.0)
        signature = (planes, width, tuple(revisions))
        if signature == getattr(self, "_section_contour_signature", None):
            return

        group = getattr(self, "_section_contour_group", None)
        contours = getattr(self, "_section_contours", {})
        wanted = set()
        for source in sources:
            bounds = source.get_world_bounding_box()
            for index, plane in enumerate(planes):
                if not _crosses_plane(bounds, plane):
                    continue
                if group is None:
                    group = self._gfx.Group()
                    self._scene.add(group)
                    self._section_contour_group = group
                key = (id(source), index)
                wanted.add(key)
                contour = contours.get(key)
                if contour is None:
                    material = _section_material_class()(
                        color="#ffd28a", thickness=width, depth_write=False, pick_write=False
                    )
                    contour = self._gfx.Mesh(source.geometry, material)
                    contour.cast_shadow = contour.receive_shadow = False
                    group.add(contour)
                    contours[key] = contour
                contour.geometry = source.geometry
                contour.local.matrix = np.array(source.world.matrix, copy=True)
                # Slice uses n·p + d = 0, while clipping keeps n·p >= d.
                contour.material.plane = (*plane[:3], -plane[3])
                contour.material.thickness = width
                contour.material.clipping_planes = [p for j, p in enumerate(planes) if j != index]
        for key in set(contours) - wanted:
            group.remove(contours.pop(key))
        self._section_contours = contours
        self._section_contour_signature = signature
