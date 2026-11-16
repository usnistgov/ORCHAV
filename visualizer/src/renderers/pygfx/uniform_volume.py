"""Unlit ISO surfaces using pygfx's unchanged volume ray-marching algorithm.

The backend adapter replaces only the final lighting expression. Sampling,
threshold crossing, palette lookup and opacity remain upstream-owned.
A guarded correction restores translation to pygfx 0.17 ISO hit depth.
The guard deliberately fails if a dependency update changes this shader contract.
"""

from __future__ import annotations

import pygfx as gfx
from pygfx.renderers.wgpu import register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.volumeshader import VolumeRayShader


def correct_iso_hit_world_position(code):
    """Correct the pinned 0.17 ISO hit transform, never its sampling algorithm.

    The upstream ISO branch uses its normal's mat3 to transform a position,
    dropping translation. This breaks scene occlusion and therefore visible-hit
    picking. Fail loudly on dependency changes instead of applying a fuzzy patch.
    """
    source = "let world_pos = world_matrix * data_pos;"
    if code.count(source) != 1:
        raise RuntimeError("Unsupported pygfx 0.17 ISO hit-position shader contract")
    return code.replace(
        source,
        "let world_pos = (u_wobject.world_transform * vec4<f32>(data_pos, 1.0)).xyz;",
    )


class SceneVolumeIsoMaterial(gfx.VolumeIsoMaterial):
    """Native ISO sampling with scene-correct translated hit depth."""


@register_wgpu_render_function(gfx.Volume, SceneVolumeIsoMaterial)
class SceneVolumeIsoShader(VolumeRayShader):
    def get_code(self):
        return correct_iso_hit_world_position(super().get_code())


class UniformVolumeIsoMaterial(gfx.VolumeIsoMaterial):
    """ISO rendering whose palette colors do not depend on surface normals."""


@register_wgpu_render_function(gfx.Volume, UniformVolumeIsoMaterial)
class UniformVolumeIsoShader(VolumeRayShader):
    """Retain upstream ISO geometry and pass through its physical palette color."""

    def get_code(self):
        code = correct_iso_hit_world_position(super().get_code())
        lighting = "let physical_color = calculate_light(physical_albedo, world_pos.xyz, reoriented_normal, view_dir);"
        if code.count(lighting) != 1:
            raise RuntimeError("Unsupported pygfx ISO lighting shader contract")
        return code.replace(lighting, "let physical_color = physical_albedo;")


class CellGridVolumeIsoMaterial(gfx.VolumeIsoMaterial):
    """Lit ISO surface with a cell-boundary guide evaluated at the visible hit."""


class UniformCellGridVolumeIsoMaterial(UniformVolumeIsoMaterial):
    """Unlit ISO surface with the same visible-surface cell guide."""


@register_wgpu_render_function(gfx.Volume, CellGridVolumeIsoMaterial)
@register_wgpu_render_function(gfx.Volume, UniformCellGridVolumeIsoMaterial)
class CellGridVolumeIsoShader(VolumeRayShader):
    """Draw the tangent cell grid in the volume pass, retaining scene occlusion.

    Separate exact lattice lines can be hidden by the approximate raymarch hit
    depth on a broad surface. The guide colors that hit instead of moving cell
    geometry or disabling depth testing. On the volume's outer faces, use the
    ray-entry face rather than a scalar gradient, which can vanish or reflect
    category changes instead of the face direction. Interior cells remain hidden.
    """

    def __init__(self, wobject):
        super().__init__(wobject)
        self["orchav_unlit"] = isinstance(wobject.material, UniformCellGridVolumeIsoMaterial)

    def get_code(self):
        code = correct_iso_hit_world_position(super().get_code())
        lighting = "let physical_color = calculate_light(physical_albedo, world_pos.xyz, reoriented_normal, view_dir);"
        if code.count(lighting) != 1:
            raise RuntimeError("Unsupported pygfx cell-grid ISO shader contract")
        return code.replace(
            lighting,
            """
        $$ if orchav_unlit
            let surface_color = physical_albedo;
        $$ else
            let surface_color = calculate_light(physical_albedo, world_pos.xyz, reoriented_normal, view_dir);
        $$ endif
        // Texture cell boundaries are integer coordinates, including the box faces.
        var lattice = the_coord * sizef;
        var magnitude = abs(normal);
        // A first-sample hit lies on the box entry face. Refinement can move
        // it back along the ray because the sampler clamps at the box edge.
        // Classify that geometric face independently of the stored values.
        if dot(the_coord - start_coord, step_coord) <= 0.0 {
            lattice = start_coord * sizef;
            let face_distance = min(abs(lattice), abs(lattice - sizef));
            if face_distance.x <= face_distance.y && face_distance.x <= face_distance.z {
                magnitude = vec3<f32>(1.0, 0.0, 0.0);
            } else if face_distance.y <= face_distance.z {
                magnitude = vec3<f32>(0.0, 1.0, 0.0);
            } else {
                magnitude = vec3<f32>(0.0, 0.0, 1.0);
            }
        }
        let distance = abs(lattice - round(lattice));
        var tangent_distance = min(distance.x, distance.y);
        if magnitude.x >= magnitude.y && magnitude.x >= magnitude.z {
            tangent_distance = min(distance.y, distance.z);
        } else if magnitude.y >= magnitude.z {
            tangent_distance = min(distance.x, distance.z);
        }
        let grid_coverage = 1.0 - smoothstep(0.018, 0.035, tangent_distance);
        let grid_color = srgb2physical(vec3<f32>(0.094, 0.141, 0.169));
        let physical_color = mix(surface_color, grid_color, grid_coverage);
        """,
        )
