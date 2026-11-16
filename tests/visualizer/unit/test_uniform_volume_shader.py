"""Qualify the backend adaptation against the installed pygfx ISO shader."""

import numpy as np
import pygfx as gfx
import pytest

from visualizer.src.renderers.pygfx.uniform_volume import (
    UniformVolumeIsoMaterial,
    UniformVolumeIsoShader,
    correct_iso_hit_world_position,
)
from visualizer.src.types.render_payloads import VolumePayload


def test_installed_uniform_shader_preserves_raymarch_and_bypasses_lighting():
    from pygfx.renderers.wgpu.shaders.volumeshader import VolumeRayShader

    volume = gfx.Volume(
        gfx.Geometry(grid=gfx.Texture(np.ones((4, 4, 4), np.float32), dim=3)),
        UniformVolumeIsoMaterial(),
    )
    upstream = VolumeRayShader(volume).get_code()
    unlit = UniformVolumeIsoShader(volume).get_code()
    assert unlit != upstream
    lighting = "let physical_color = calculate_light(physical_albedo, world_pos.xyz, reoriented_normal, view_dir);"
    assert upstream.count(lighting) == 1
    assert unlit == correct_iso_hit_world_position(upstream).replace(
        lighting, "let physical_color = physical_albedo;"
    )
    assert "let opacity = color.a * u_material.opacity;" in unlit
    assert "out.coord = the_coord;" in unlit


@pytest.mark.parametrize("lighting", [0, "false", None])
def test_volume_rejects_nonboolean_lighting(lighting):
    with pytest.raises(ValueError, match="lighting must be boolean"):
        VolumePayload(np.ones((2, 2, 2)), np.ones((2, 4)), (0, 1), lighting=lighting)


@pytest.mark.parametrize("unlit", [False, True])
def test_surface_grid_preserves_raymarch_depth_and_opacity(unlit):
    from visualizer.src.renderers.pygfx.uniform_volume import (
        CellGridVolumeIsoMaterial,
        CellGridVolumeIsoShader,
        UniformCellGridVolumeIsoMaterial,
    )

    material = UniformCellGridVolumeIsoMaterial() if unlit else CellGridVolumeIsoMaterial()
    volume = gfx.Volume(
        gfx.Geometry(grid=gfx.Texture(np.ones((4, 4, 4), np.float32), dim=3)), material
    )
    shader = CellGridVolumeIsoShader(volume)
    code = shader.get_code()
    from pygfx.renderers.wgpu.shaders.volumeshader import VolumeRayShader

    upstream = VolumeRayShader(volume).get_code()
    lighting = "let physical_color = calculate_light(physical_albedo, world_pos.xyz, reoriented_normal, view_dir);"
    prefix, suffix = correct_iso_hit_world_position(upstream).split(lighting)
    assert code.startswith(prefix) and code.endswith(suffix)
    assert "let surface_color = physical_albedo;" in code
    assert (
        "let surface_color = calculate_light(physical_albedo, world_pos.xyz, reoriented_normal, view_dir);"
        in code
    )
    assert "let physical_color = mix(surface_color, grid_color, grid_coverage);" in code
    assert shader["orchav_unlit"] == unlit
    assert "var lattice = the_coord * sizef;" in code
    assert "let opacity = color.a * u_material.opacity;" in code
    assert "out.coord = the_coord;" in code
    assert "out.depth = ndc_pos.z / ndc_pos.w;" in code


@pytest.mark.parametrize("value", [0, "true", None])
def test_surface_grid_requires_a_boolean(value):
    with pytest.raises(ValueError, match="surface_grid must be boolean"):
        VolumePayload(np.ones((2, 2, 2)), np.ones((2, 4)), (0, 1), surface_grid=value)


@pytest.mark.parametrize("contract", ["missing", "duplicated"])
@pytest.mark.parametrize("grid", [False, True])
def test_unreviewed_upstream_shader_changes_fail_closed(monkeypatch, contract, grid):
    from pygfx.renderers.wgpu.shaders.volumeshader import VolumeRayShader

    from visualizer.src.renderers.pygfx.uniform_volume import (
        CellGridVolumeIsoMaterial,
        CellGridVolumeIsoShader,
    )

    material = CellGridVolumeIsoMaterial() if grid else UniformVolumeIsoMaterial()
    volume = gfx.Volume(
        gfx.Geometry(grid=gfx.Texture(np.ones((4, 4, 4), np.float32), dim=3)), material
    )
    code = VolumeRayShader(volume).get_code()
    lighting = "let physical_color = calculate_light(physical_albedo, world_pos.xyz, reoriented_normal, view_dir);"
    changed = code.replace(lighting, "") if contract == "missing" else code + lighting
    monkeypatch.setattr(VolumeRayShader, "get_code", lambda self: changed)
    shader_type = CellGridVolumeIsoShader if grid else UniformVolumeIsoShader
    with pytest.raises(RuntimeError, match="Unsupported pygfx.*shader contract"):
        shader_type(volume).get_code()


@pytest.mark.parametrize("code", ["", "let world_pos = world_matrix * data_pos;" * 2])
def test_iso_depth_fix_requires_exact_pinned_contract(code):
    with pytest.raises(RuntimeError, match="ISO hit-position"):
        correct_iso_hit_world_position(code)
