"""Exact cell traversal for sampled volumes, with native depth and cell picking.

An ISO surface search can skip voxel corners and refine outside the volume.
These discrete results instead intersect cell faces directly. Palette codes
never influence geometry, and picking returns the cell that supplied the color.
"""

import pygfx as gfx
from pygfx.renderers.wgpu import register_wgpu_render_function
from pygfx.renderers.wgpu.shaders.volumeshader import VolumeRayShader


class CellVolumeMaterial(gfx.VolumeIsoMaterial):
    def __init__(self, *, lighting=False, surface_grid=False, **kwargs):
        super().__init__(**kwargs)
        self.cell_lighting = lighting
        self.cell_grid = surface_grid


@register_wgpu_render_function(gfx.Volume, CellVolumeMaterial)
class CellVolumeShader(VolumeRayShader):
    def __init__(self, wobject):
        super().__init__(wobject)
        self["cell_lighting"] = wobject.material.cell_lighting
        self["cell_grid"] = wobject.material.cell_grid

    def get_code(self):
        code = super().get_code()
        start = "$$ elif mode == 'iso'"
        end = "\n$$ else\n    fn raycast"
        if code.count(start) != 1 or code.count(end) != 1:
            raise RuntimeError("Unsupported pygfx cell-volume shader contract")
        short_ray = (
            "let nsteps = i32(-dist / relative_step_size + 0.5);\n    if( nsteps < 1 ) { discard; }"
        )
        if code.count(short_ray) != 1:
            raise RuntimeError("Unsupported pygfx cell-volume entry shader contract")
        code = code.replace(
            short_ray,
            "if dist >= 0.0 { discard; }\n    let nsteps = max(1, i32(-dist / relative_step_size + 0.5));",
        )
        before, tail = code.split(start)
        _, after = tail.split(end)
        return before + start + _CELL_RAYCAST + end + after


_CELL_RAYCAST = r"""
    fn raycast(sizef: vec3<f32>, nsteps: i32, start_coord: vec3<f32>, step_coord: vec3<f32>) -> RenderOutput {
        let direction = normalize(step_coord * sizef);
        let origin = clamp(start_coord * sizef, vec3f(0.0), sizef);
        let moving = abs(direction) > vec3f(1e-8);
        let divisor = select(vec3f(1.0), direction, moving);
        let stride = vec3i(sign(direction));
        var cell = clamp(vec3i(floor(origin + direction * 1e-4)), vec3i(0), vec3i(sizef) - 1);
        let boundary = vec3f(cell) + select(vec3f(0.0), vec3f(1.0), direction > vec3f(0.0));
        var next_face = select(vec3f(1e20), (boundary - origin) / divisor, moving);
        let face_step = select(vec3f(1e20), abs(1.0 / divisor), moving);
        var distance = 0.0;
        var normal = -direction;
        let edge_distance = min(abs(origin), abs(origin - sizef));
        if edge_distance.x <= edge_distance.y && edge_distance.x <= edge_distance.z {
            normal = vec3f(-f32(stride.x), 0.0, 0.0);
        } else if edge_distance.y <= edge_distance.z {
            normal = vec3f(0.0, -f32(stride.y), 0.0);
        } else {
            normal = vec3f(0.0, 0.0, -f32(stride.z));
        }
        // Each iteration crosses at least one cell face; this bound covers
        // even a grazing ray across the complete rectangular lattice.
        for (var i = 0; i <= i32(sizef.x + sizef.y + sizef.z); i++) {
            if any(cell < vec3i(0)) || any(cell >= vec3i(sizef)) { break; }
            let center = (vec3f(cell) + 0.5) / sizef;
            let value = sample_vol(center, sizef);
            let color = sampled_value_to_color(value);
            if value.r > u_material.threshold && color.a > 0.0 {
                let lattice = origin + max(0.0, distance) * direction;
                let world_pos = u_wobject.world_transform * vec4f(lattice - 0.5, 1.0);
                let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
                $$ if colorspace == 'srgb'
                    var physical_color = srgb2physical(color.rgb);
                $$ else
                    var physical_color = color.rgb;
                $$ endif
                $$ if cell_lighting
                    let normal_matrix = transpose(u_wobject.world_transform_inv);
                    let world_normal = normalize((normal_matrix * vec4f(normal, 0.0)).xyz);
                    let view_dir = normalize((u_wobject.world_transform * vec4f(-direction, 0.0)).xyz);
                    physical_color = calculate_light(physical_color, world_pos.xyz, world_normal, view_dir);
                $$ endif
                $$ if cell_grid
                    let edge = abs(lattice - round(lattice));
                    let magnitude = abs(normal);
                    var tangent = min(edge.x, edge.y);
                    if magnitude.x >= magnitude.y && magnitude.x >= magnitude.z {
                        tangent = min(edge.y, edge.z);
                    } else if magnitude.y >= magnitude.z {
                        tangent = min(edge.x, edge.z);
                    }
                    let grid = 1.0 - smoothstep(0.018, 0.035, tangent);
                    physical_color = mix(physical_color, srgb2physical(vec3f(0.094, 0.141, 0.169)), grid);
                $$ endif
                var out: RenderOutput;
                out.color = vec4f(physical_color, color.a * u_material.opacity);
                // Depth uses the exact face; quantized picks use its source
                // cell center so a boundary never selects the adjacent cell.
                out.coord = center;
                out.depth = ndc_pos.z / ndc_pos.w;
                return out;
            }
            distance = min(next_face.x, min(next_face.y, next_face.z));
            let crossed = next_face <= vec3f(distance + 1e-6);
            if crossed.x {
                normal = vec3f(-f32(stride.x), 0.0, 0.0);
            } else if crossed.y {
                normal = vec3f(0.0, -f32(stride.y), 0.0);
            } else {
                normal = vec3f(0.0, 0.0, -f32(stride.z));
            }
            cell += select(vec3i(0), stride, crossed);
            next_face += select(vec3f(0.0), face_step, crossed);
        }
        discard;
    }
"""
