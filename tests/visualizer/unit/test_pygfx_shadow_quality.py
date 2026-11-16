"""Native public-API shadow fitting tests, without a GPU adapter."""

from types import SimpleNamespace

import numpy as np
import pygfx as gfx
import pytest

from visualizer.src.renderers.pygfx.renderer import PygfxRenderer
from visualizer.src.renderers.pygfx.shadows import bounds_corners, fitted_shadow_box
from visualizer.src.types.camera_state import CameraState


def renderer_with_mesh(center=(1200.0, -800.0, 0.0), size=(60.0, 25.0, 12.0)):
    renderer = PygfxRenderer(SimpleNamespace())
    renderer._gfx = gfx
    renderer._scene = gfx.Scene()
    renderer._install_default_lights()
    mesh = gfx.Mesh(gfx.box_geometry(*size), gfx.MeshStandardMaterial())
    mesh.local.position = center
    mesh.cast_shadow = mesh.receive_shadow = True
    renderer._objects["scene:building::mesh"] = mesh
    renderer._scene.add(mesh)
    renderer._update_shadow_extent(max(size))
    renderer.get_camera_state = lambda: CameraState(
        eye=(1200.0, -900.0, 50.0), lookat=center, up=(0.0, 0.0, 1.0), fov_deg=60.0
    )
    return renderer, mesh


def assert_mesh_inside_shadow(renderer, mesh):
    camera = renderer._key_light.shadow.camera
    corners = bounds_corners(mesh.get_world_bounding_box())
    hom = np.column_stack([corners, np.ones(8)])
    clip = hom @ camera.camera_matrix.T
    ndc = clip[:, :3] / clip[:, 3:4]
    assert np.max(np.abs(ndc[:, :2])) <= 1.0
    # pygfx camera projection uses WebGPU's [0, 1] depth range.
    assert ndc[:, 2].min() >= 0.0
    assert ndc[:, 2].max() <= 1.0


def test_defaults_and_standard_restore_are_exact():
    renderer, _ = renderer_with_mesh()
    light = renderer._key_light
    original = (
        light.world.position.copy(),
        light.shadow.camera.width,
        light.shadow.camera.height,
        light.shadow.camera.depth_range,
        light.shadow.bias,
    )
    assert renderer.get_shadow_quality() == "standard"
    assert renderer.set_shadow_quality("scene")
    assert renderer.set_shadow_quality("standard")
    np.testing.assert_allclose(light.world.position, original[0])
    assert (
        light.shadow.camera.width,
        light.shadow.camera.height,
        light.shadow.camera.depth_range,
        light.shadow.bias,
    ) == original[1:]


def test_fitted_scene_covers_translated_geometry_and_preserves_light_direction():
    renderer, mesh = renderer_with_mesh()
    direction = renderer._key_light.target.world.position - renderer._key_light.world.position
    direction /= np.linalg.norm(direction)
    renderer.set_shadow_quality("scene")
    fitted = renderer._key_light.target.world.position - renderer._key_light.world.position
    np.testing.assert_allclose(fitted, direction, atol=1e-12)
    assert_mesh_inside_shadow(renderer, mesh)
    state = renderer.get_shadow_quality_state()
    assert state["fit_available"]
    assert state["map_size"] == 1024
    assert state["world_units_per_texel"] < 90 / 1024


def test_fitted_scene_covers_large_city_depth_and_tiny_objects():
    for center, size in [((10000, -20000, 0), (2000, 1500, 400)), ((0, 0, 0), (0.01, 0.03, 0.02))]:
        renderer, mesh = renderer_with_mesh(center, size)
        renderer.set_shadow_quality("scene")
        assert_mesh_inside_shadow(renderer, mesh)
        assert 0 < renderer._key_light.shadow.bias < 1


def test_cache_updates_on_transforms_and_vertex_revisions():
    renderer, mesh = renderer_with_mesh()
    renderer.set_shadow_quality("scene")
    first = renderer._shadow_fit_signature
    eye = renderer._key_light.world.position.copy()
    renderer._update_shadow_quality()
    assert renderer._shadow_fit_signature is first
    np.testing.assert_array_equal(renderer._key_light.world.position, eye)
    mesh.local.x += 300
    renderer._update_shadow_quality()
    assert renderer._shadow_fit_signature != first
    assert_mesh_inside_shadow(renderer, mesh)
    signature = renderer._shadow_fit_signature
    changed = mesh.geometry.positions.data.copy()
    changed[:, 2] *= 2
    mesh.geometry.positions = gfx.Buffer(changed)
    renderer._update_shadow_quality()
    assert renderer._shadow_fit_signature != signature
    assert_mesh_inside_shadow(renderer, mesh)


def test_focus_pins_camera_target_and_keeps_off_focus_caster_depth():
    renderer, mesh = renderer_with_mesh()
    renderer.set_shadow_focus_radius(5)
    renderer.set_shadow_quality("focus")
    center = renderer._shadow_focus_center.copy()
    original_eye = renderer._key_light.world.position.copy()
    renderer.get_camera_state = lambda: CameraState(
        eye=(100, 100, 100), lookat=(0, 0, 0), up=(0, 0, 1), fov_deg=60
    )
    renderer._update_shadow_quality()
    np.testing.assert_array_equal(renderer._shadow_focus_center, center)
    np.testing.assert_array_equal(renderer._key_light.world.position, original_eye)
    # A tall upstream box expands depth without diluting focus XY resolution.
    original_width = renderer._key_light.shadow.camera.width
    camera = renderer._key_light.shadow.camera
    original_far = camera.far
    caster = gfx.Mesh(gfx.box_geometry(1, 1, 1000), gfx.MeshStandardMaterial())
    caster.local.position = (1200, -800, 500)
    caster.cast_shadow = True
    renderer._objects["scene:tower::mesh"] = caster
    renderer._scene.add(caster)
    renderer._update_shadow_quality()
    assert camera.width == original_width
    assert camera.far > original_far
    assert renderer.refocus_shadows()
    np.testing.assert_array_equal(renderer._shadow_focus_center, (0, 0, 0))


def test_scientific_overlays_do_not_affect_shadow_fitting():
    renderer, _ = renderer_with_mesh()
    renderer.set_shadow_quality("scene")
    before = renderer.get_shadow_quality_state()
    overlay = gfx.Mesh(gfx.box_geometry(1e6, 1e6, 1e6), gfx.MeshBasicMaterial())
    overlay.cast_shadow = True
    renderer._objects["coverage_overlay"] = overlay
    renderer._update_shadow_quality()
    assert renderer.get_shadow_quality_state() == before


def test_light_angles_and_reenable_refit_without_losing_direction():
    renderer, mesh = renderer_with_mesh()
    renderer.set_shadow_quality("scene")
    renderer.set_shadow_enabled(False)
    renderer.set_key_light_angles(65, -30)
    renderer.set_shadow_enabled(True)
    assert_mesh_inside_shadow(renderer, mesh)
    delta = renderer._key_light.target.world.position - renderer._key_light.world.position
    expected = (
        np.cos(np.deg2rad(-30)) * np.cos(np.deg2rad(65)),
        np.cos(np.deg2rad(-30)) * np.sin(np.deg2rad(65)),
        np.sin(np.deg2rad(-30)),
    )
    np.testing.assert_allclose(delta, expected, atol=1e-12)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, "broken"])
def test_invalid_controls_leave_state_unchanged(value):
    renderer, _ = renderer_with_mesh()
    before = renderer.get_shadow_quality_state()
    assert not renderer.set_shadow_focus_radius(value)
    assert not renderer.set_shadow_bias_texels(value)
    assert not renderer.set_shadow_quality("unknown")
    assert renderer.get_shadow_quality_state() == before


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        bounds_corners(np.array([[0, 0, 0], [np.nan, 1, 1]]))


def test_pure_fit_retains_caster_depth_in_focus_region():
    corners = bounds_corners(np.array([[-10, -10, -1000], [10, 10, 1000]]))
    eye, width, height, far = fitted_shadow_box(
        corners, np.eye(3), focus_center=np.zeros(3), focus_radius=1
    )
    assert width == pytest.approx(2.1)
    assert height == pytest.approx(2.1)
    assert far > 2000
    assert eye[2] > 1000


def test_in_place_stream_updates_before_upload_do_not_stale_shadow_bounds():
    renderer, mesh = renderer_with_mesh()
    renderer.set_shadow_quality("scene")
    camera = renderer._key_light.shadow.camera
    original_width = camera.width
    old_signature = renderer._shadow_fit_signature
    for scale in (2, 4):
        changed = mesh.geometry.positions.data.copy()
        changed[:, :2] *= scale
        renderer._push_buffer(
            mesh.geometry.positions, changed, label="mesh_vertex_stream_positions"
        )
        renderer._update_shadow_quality()
        assert renderer._shadow_fit_signature != old_signature
        assert camera.width > original_width
        # Check actual updated data, not pygfx's coalesced native bounds cache.
        points = changed @ mesh.world.matrix[:3, :3].T + mesh.world.position
        hom = np.column_stack([points, np.ones(len(points))])
        clip = hom @ camera.camera_matrix.T
        assert np.max(np.abs(clip[:, :2] / clip[:, 3:4])) <= 1.0
        original_width = camera.width
        old_signature = renderer._shadow_fit_signature


def test_custom_standard_bias_restores_after_fitted_modes():
    renderer, _ = renderer_with_mesh()
    renderer.set_key_light_angles(25, -30)
    original = renderer._key_light.shadow.bias
    renderer.set_shadow_quality("scene")
    renderer.set_shadow_quality("standard")
    assert renderer._key_light.shadow.bias == original == 0.001


def test_writes_in_standard_mode_invalidate_next_fitted_bounds():
    renderer, mesh = renderer_with_mesh()
    renderer.set_shadow_quality("scene")
    original_width = renderer._key_light.shadow.camera.width
    renderer.set_shadow_quality("standard")
    data = mesh.geometry.positions.data.copy() * 4
    renderer._push_buffer(mesh.geometry.positions, data, label="mesh_positions")
    renderer.set_shadow_quality("scene")
    assert renderer._key_light.shadow.camera.width > original_width * 3
