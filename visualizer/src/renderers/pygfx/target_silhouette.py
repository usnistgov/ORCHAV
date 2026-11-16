"""A depth-tested silhouette around one explicitly selected scene actor.

The back-face shell shares the actor's GPU geometry. Only its projected vertex
positions expand; recorded mesh vertices, picking, and shadows stay unchanged.
"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def silhouette_material_class():
    """Register the small vertex adaptation against the pinned pygfx mesh shader."""
    from pygfx import Mesh, MeshBasicMaterial
    from pygfx.renderers.wgpu import register_wgpu_render_function
    from pygfx.renderers.wgpu.shaders.meshshader import MeshShader

    class TargetSilhouetteMaterial(MeshBasicMaterial):
        uniform_type = dict(MeshBasicMaterial.uniform_type, silhouette_width="f4")

        def __init__(self):
            super().__init__(
                color="#56dce8",
                color_mode="uniform",
                side="back",
                depth_test=True,
                depth_write=False,
                pick_write=False,
            )
            self.uniform_buffer.data["silhouette_width"] = 2.5
            self.uniform_buffer.update_full()

    @register_wgpu_render_function(Mesh, TargetSilhouetteMaterial)
    class TargetSilhouetteShader(MeshShader):
        def get_code(self):
            code = super().get_code()
            anchor = "varyings.position = vec4<f32>(ndc_pos.xyz, ndc_pos.w);"
            if code.count(anchor) != 1:
                raise RuntimeError("Unsupported pygfx mesh shader for target silhouettes")
            return code.replace(
                anchor,
                """
                let shell_normal = normalize((transpose(u_wobject.world_transform_inv)
                    * vec4<f32>(raw_normal, 0.0)).xyz);
                let shell_direction = u_stdinfo.projection_transform
                    * u_stdinfo.cam_transform * vec4<f32>(shell_normal, 0.0);
                let projected = (shell_direction.xy
                    - ndc_pos.xy / ndc_pos.w * shell_direction.w) * u_stdinfo.logical_size;
                let outward = projected / max(length(projected), 0.00001);
                let expansion = outward * (2.0 * u_material.silhouette_width)
                    / u_stdinfo.logical_size * ndc_pos.w;
                varyings.position = vec4<f32>(ndc_pos.xy + expansion, ndc_pos.zw);
                """,
            )

    return TargetSilhouetteMaterial


class PygfxTargetSilhouetteMixin:
    """Keep an optional actor shell synchronized without a timer or mesh copies."""

    def set_target_silhouette(self, object_id: str | None) -> bool:
        """Emphasize one target by its stable scene ID, or clear the emphasis."""
        self._target_silhouette_id = object_id
        self._sync_target_silhouette()
        self.request_redraw()
        return True

    def _clear_target_silhouette(self) -> None:
        shell = getattr(self, "_target_silhouette", None)
        if shell is not None and shell.parent is not None:
            shell.parent.remove(shell)
        self._target_silhouette = None
        self._target_silhouette_id = None

    def _sync_target_silhouette(self) -> None:
        object_id = getattr(self, "_target_silhouette_id", None)
        shell = getattr(self, "_target_silhouette", None)
        source = getattr(self, "_objects", {}).get(object_id)
        if source is None:
            if shell is not None and shell.parent is not None:
                shell.parent.remove(shell)
            self._target_silhouette = None
            return
        material = getattr(source, "material", None)
        visible = isinstance(source, self._gfx.Mesh) and source.parent is not None
        parent = source
        while visible and parent is not None:
            visible = bool(parent.visible)
            parent = parent.parent
        # An opaque hull would fill a wireframe or a translucent actor.
        visible = visible and not bool(getattr(material, "wireframe", False))
        visible = visible and bool(getattr(material, "depth_write", True))
        visible = visible and float(getattr(material, "opacity", 1.0)) >= 1.0
        if not visible:
            if shell is not None:
                shell.visible = False
            return
        if shell is None:
            shell = self._gfx.Mesh(source.geometry, silhouette_material_class()())
            shell.cast_shadow = False
            shell.receive_shadow = False
            shell.render_order = 10_000
            self._scene.add(shell)
            self._target_silhouette = shell
        if shell.geometry is not source.geometry:
            shell.geometry = source.geometry
        shell.local.matrix = np.array(source.world.matrix, copy=True)
        shell.material.clipping_planes = tuple(getattr(material, "clipping_planes", ()))
        shell.material.clipping_mode = getattr(material, "clipping_mode", "ANY")
        shell.visible = True
