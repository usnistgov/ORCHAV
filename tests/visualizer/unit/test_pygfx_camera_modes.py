from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pygfx = pytest.importorskip("pygfx")
la = pytest.importorskip("pylinalg")

from visualizer.src.materials.appearance import (
    AppearanceIntent,
    MaterialDisplayMode,
    resolve_appearance,
)
from visualizer.src.renderers.camera_ops import object_contributes_to_camera_bounds
from visualizer.src.renderers.pygfx.camera import PygfxCameraMixin, SceneBounds
from visualizer.src.renderers.pygfx.renderer import PygfxRenderer
from visualizer.src.types.camera_state import CameraState


def _make_renderer() -> PygfxRenderer:
    renderer = PygfxRenderer.__new__(PygfxRenderer)
    renderer._initialized = True
    renderer._gfx = pygfx
    renderer._camera = pygfx.PerspectiveCamera(60.0, 1200 / 800)
    renderer._camera.set_view_size(1200, 800)
    renderer._camera.world.reference_up = (0.0, 0.0, 1.0)
    renderer._controller = SimpleNamespace(target=(0.0, 0.0, 0.0), distance=0.0)
    renderer._camera_state = None
    renderer._last_camera_look_distance = 10.0
    renderer._follow_target_lookat = None
    renderer._minimap_camera = pygfx.OrthographicCamera(1.0, 1.0, depth=100.0)
    renderer._minimap_overlay_scene = None
    renderer._minimap_overlay_camera = None
    renderer._minimap_overlay_size = None
    renderer._minimap_overlay_objects = {}
    renderer._minimap_world_rect = None
    renderer._width = 1200
    renderer._height = 800
    renderer._focus_canvas = lambda: None
    renderer.request_redraw = lambda: None
    renderer.visualizer = SimpleNamespace(animation_running=False)
    return renderer


def _project_bounds_to_ndc(renderer: PygfxRenderer, bounds: SceneBounds) -> np.ndarray:
    corners = []
    for x in (bounds.min_bound[0], bounds.max_bound[0]):
        for y in (bounds.min_bound[1], bounds.max_bound[1]):
            for z in (bounds.min_bound[2], bounds.max_bound[2]):
                local = la.vec_transform(
                    np.array([x, y, z], dtype=np.float64),
                    renderer._camera.world.inverse_matrix,
                )
                corners.append(la.vec_transform(local, renderer._camera.projection_matrix))
    return np.asarray(corners, dtype=np.float64)


def test_renderer_composes_pygfx_camera_mixin():
    renderer = _make_renderer()

    assert isinstance(renderer, PygfxCameraMixin)


def test_authoring_decorations_are_excluded_from_camera_fit_bounds():
    assert not object_contributes_to_camera_bounds("authoring:document-id:work_plane")
    assert not object_contributes_to_camera_bounds("authoring:document-id:actor-id:label")
    assert not object_contributes_to_camera_bounds(
        "authoring:document-id:actor-id:mobility_control_label_center"
    )
    assert object_contributes_to_camera_bounds("authoring:document-id:actor-id:path")


def test_reset_camera_bounds_uses_aspect_aware_overview_fit():
    renderer = _make_renderer()
    bounds = SceneBounds(
        min_bound=np.array([-8.0, -4.0, 1.0]),
        max_bound=np.array([8.0, 5.0, 9.0]),
    )
    overview_calls = []
    shadow_extents = []
    renderer.compute_scene_bounds = lambda *, scope: bounds
    renderer.set_overview_camera = (
        lambda view, value, *, fov: overview_calls.append((view, value, fov)) or True
    )
    renderer._update_shadow_extent = shadow_extents.append

    renderer.reset_camera_bounds()

    assert overview_calls == [("isometric", bounds, 60.0)]
    assert shadow_extents == [16.0]


def test_set_camera_state_honors_requested_up_vector_for_top_view():
    renderer = _make_renderer()
    state = CameraState(
        eye=(0.0, 0.0, 50.0),
        lookat=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov_deg=45.0,
    )

    assert renderer.set_camera_state(state) is True
    got = renderer.get_camera_state()

    assert got is not None
    np.testing.assert_allclose(got.lookat, state.lookat, atol=1e-5)
    np.testing.assert_allclose(got.up, state.up, atol=1e-5)


def test_set_camera_state_rejects_untyped_payload():
    renderer = _make_renderer()

    assert (
        renderer.set_camera_state(  # type: ignore[arg-type]
            {
                "eye": [0.0, 0.0, 50.0],
                "lookat": [0.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov": 45.0,
            }
        )
        is False
    )


def test_set_overview_camera_applies_named_view_from_bounds():
    renderer = _make_renderer()
    bounds = SceneBounds(
        min_bound=np.array([-10.0, -5.0, 0.0], dtype=np.float64),
        max_bound=np.array([10.0, 5.0, 4.0], dtype=np.float64),
    )

    assert renderer.set_overview_camera("top", bounds, fov=50.0, distance=25.0) is True
    got = renderer.get_camera_state()

    assert got is not None
    np.testing.assert_allclose(got.lookat, [0.0, 0.0, 2.0], atol=1e-5)
    np.testing.assert_allclose(got.eye, [0.0, 0.0, 27.0], atol=1e-5)
    np.testing.assert_allclose(got.up, [0.0, 1.0, 0.0], atol=1e-5)
    assert got.fov_deg == pytest.approx(50.0)


def test_set_overview_camera_auto_fit_accounts_for_pygfx_extent_fov():
    renderer = _make_renderer()
    bounds = SceneBounds(
        min_bound=np.array([-10.0, -5.0, 0.0], dtype=np.float64),
        max_bound=np.array([10.0, 5.0, 4.0], dtype=np.float64),
    )

    assert renderer.set_overview_camera("top", bounds, fov=50.0) is True
    got = renderer.get_camera_state()

    assert got is not None
    np.testing.assert_allclose(got.lookat, [0.0, 0.0, 2.0], atol=1e-5)
    ndc = _project_bounds_to_ndc(renderer, bounds)
    assert np.max(np.abs(ndc[:, 0])) <= 1.0
    assert np.max(np.abs(ndc[:, 1])) <= 1.0


def test_whole_scene_bounds_use_realized_pygfx_objects_only():
    renderer = _make_renderer()
    scene_points = np.array(
        [
            [-100.0, -80.0, 0.0],
            [120.0, 90.0, 45.0],
        ],
        dtype=np.float32,
    )
    renderer._objects = {
        "scene:city::mesh": SimpleNamespace(
            geometry=SimpleNamespace(positions=SimpleNamespace(data=scene_points)),
            local=SimpleNamespace(matrix=np.eye(4, dtype=np.float64)),
        )
    }
    renderer._hidden = set()
    renderer.visualizer = SimpleNamespace(
        animation_running=False,
        mesh_entries=[{"visible": True, "mesh": object()}],
        target_entries=[],
        current_view_model=SimpleNamespace(
            tx_positions=[[0.0, 0.0, 10.0]],
            rx_positions=[[5.0, 0.0, 1.5]],
            target_positions=[],
        ),
        current_tx_positions=[],
        current_rx_positions=[],
    )

    bounds = renderer.compute_scene_bounds(scope="whole")

    assert bounds is not None
    np.testing.assert_allclose(bounds.get_min_bound(), [-100.0, -80.0, 0.0])
    np.testing.assert_allclose(bounds.get_max_bound(), [120.0, 90.0, 45.0])


@pytest.mark.parametrize(
    "appearance",
    [
        AppearanceIntent(manual_visible=False),
        AppearanceIntent(material_mode=MaterialDisplayMode.HIDDEN),
        AppearanceIntent(pov_visible=False),
    ],
    ids=["manual", "material", "pov"],
)
@pytest.mark.parametrize("object_id", ["scene:hidden::mesh", "target:hidden::mesh"])
def test_visible_bounds_exclude_hidden_scene_and_target_appearances(
    appearance: AppearanceIntent,
    object_id: str,
):
    renderer = _make_renderer()
    anchor = SimpleNamespace(
        geometry=SimpleNamespace(
            positions=SimpleNamespace(
                data=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float32)
            )
        ),
        local=SimpleNamespace(matrix=np.eye(4, dtype=np.float64)),
        visible=True,
    )
    distant_transform = np.eye(4, dtype=np.float64)
    distant_transform[0, 3] = 100.0
    distant = SimpleNamespace(
        geometry=anchor.geometry,
        local=SimpleNamespace(matrix=distant_transform),
        visible=True,
    )
    renderer._objects = {"scene:anchor::mesh": anchor, object_id: distant}
    renderer._name_to_handle = {"scene:anchor::mesh": 1, object_id: 2}
    renderer._hidden = set()

    assert renderer.set_visible(object_id, resolve_appearance(appearance).visible) is True
    visible = renderer.compute_scene_bounds(scope="visible")
    whole = renderer.compute_scene_bounds(scope="whole")

    assert visible is not None
    assert whole is not None
    np.testing.assert_allclose(visible.get_max_bound(), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(whole.get_max_bound(), [101.0, 1.0, 1.0])


def test_failed_visibility_keeps_pygfx_bounds_truthful_until_retry():
    class _FailOnceObject:
        def __init__(self) -> None:
            self.geometry = SimpleNamespace(
                positions=SimpleNamespace(
                    data=np.asarray([[20, 0, 0], [22, 0, 0], [20, 2, 2]], dtype=np.float32)
                )
            )
            self.local = SimpleNamespace(matrix=np.eye(4, dtype=np.float64))
            self._visible = True
            self.failures = 1

        @property
        def visible(self) -> bool:
            return self._visible

        @visible.setter
        def visible(self, value: bool) -> None:
            if self.failures:
                self.failures -= 1
                raise RuntimeError("native visibility failed")
            self._visible = bool(value)

    renderer = _make_renderer()
    object_id = "target:retry::mesh"
    native = _FailOnceObject()
    renderer._objects = {object_id: native}
    renderer._name_to_handle = {object_id: 1}
    renderer._hidden = set()

    assert renderer.set_visible(object_id, False) is False
    failed_bounds = renderer.compute_scene_bounds(scope="visible")
    assert failed_bounds is not None
    np.testing.assert_allclose(failed_bounds.get_max_bound(), [22.0, 2.0, 2.0])

    assert renderer.set_visible(object_id, False) is True
    assert renderer.compute_scene_bounds(scope="visible") is None
    whole_bounds = renderer.compute_scene_bounds(scope="whole")
    assert whole_bounds is not None
    np.testing.assert_allclose(whole_bounds.get_max_bound(), [22.0, 2.0, 2.0])


@pytest.mark.parametrize("view", ["top", "front", "side", "isometric"])
def test_set_overview_camera_auto_fit_handles_deep_scene_bounds(view):
    renderer = _make_renderer()
    bounds = SceneBounds(
        min_bound=np.array([-100.0, -100.0, 0.0], dtype=np.float64),
        max_bound=np.array([100.0, 100.0, 40.0], dtype=np.float64),
    )

    assert renderer.set_overview_camera(view, bounds, fov=60.0) is True

    ndc = _project_bounds_to_ndc(renderer, bounds)
    assert np.max(np.abs(ndc[:, 0])) <= 1.0
    assert np.max(np.abs(ndc[:, 1])) <= 1.0


def test_focus_camera_resets_follow_state_and_preserves_offset():
    renderer = _make_renderer()
    renderer._follow_target_lookat = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    initial = CameraState(
        eye=(10.0, -5.0, 6.0),
        lookat=(1.0, 2.0, 3.0),
        up=(0.0, 0.0, 1.0),
        fov_deg=55.0,
    )
    assert renderer.set_camera_state(initial) is True

    new_target = np.array([4.0, 6.0, 3.0], dtype=np.float64)
    expected_eye = new_target + (np.asarray(initial.eye) - np.asarray(initial.lookat))

    assert renderer.focus_camera(new_target) is True
    got = renderer.get_camera_state()

    assert renderer._follow_target_lookat is not None
    np.testing.assert_allclose(renderer._follow_target_lookat, new_target, atol=1e-5)
    np.testing.assert_allclose(got.eye, expected_eye, atol=1e-5)
    np.testing.assert_allclose(got.lookat, new_target, atol=1e-5)


def test_set_pov_camera_uses_entity_orientation():
    renderer = _make_renderer()

    assert renderer.set_pov_camera([1.0, 2.0, 3.0], [np.pi / 2.0, 0.0, 0.0], "forward") is True
    got = renderer.get_camera_state()

    assert got is not None
    np.testing.assert_allclose(got.eye, [1.0, 2.0, 3.0], atol=1e-5)
    np.testing.assert_allclose(got.lookat, [1.0, 12.0, 3.0], atol=1e-5)


def test_set_lookat_preserves_camera_offset_across_follow_updates():
    renderer = _make_renderer()
    initial = CameraState(
        eye=(10.0, -5.0, 6.0),
        lookat=(1.0, 2.0, 3.0),
        up=(0.0, 0.0, 1.0),
        fov_deg=55.0,
    )
    assert renderer.set_camera_state(initial) is True

    new_target = np.array([4.0, 6.0, 3.0], dtype=np.float64)
    expected_eye = new_target + (np.asarray(initial.eye) - np.asarray(initial.lookat))

    assert renderer.set_lookat(new_target) is True
    got = renderer.get_camera_state()

    assert got is not None
    np.testing.assert_allclose(got.eye, expected_eye, atol=1e-5)
    np.testing.assert_allclose(got.lookat, new_target, atol=1e-5)


def test_update_follow_camera_uses_neutral_follow_protocol():
    renderer = _make_renderer()
    initial = CameraState(
        eye=(10.0, -5.0, 6.0),
        lookat=(1.0, 2.0, 3.0),
        up=(0.0, 0.0, 1.0),
        fov_deg=55.0,
    )
    assert renderer.set_camera_state(initial) is True

    next_target = np.array([7.0, 8.0, 3.0], dtype=np.float64)
    expected_eye = next_target + (np.asarray(initial.eye) - np.asarray(initial.lookat))

    assert renderer.update_follow_camera(next_target) is True
    got = renderer.get_camera_state()

    assert got is not None
    np.testing.assert_allclose(got.eye, expected_eye, atol=1e-5)
    np.testing.assert_allclose(got.lookat, next_target, atol=1e-5)


def test_follow_uses_latest_user_adjusted_offset():
    renderer = _make_renderer()
    current = CameraState(
        eye=(12.0, -2.0, 4.0),
        lookat=(4.0, 6.0, 3.0),
        up=(0.0, 0.0, 1.0),
        fov_deg=55.0,
    )
    assert renderer.set_camera_state(current) is True

    next_target = np.array([7.0, 8.0, 3.0], dtype=np.float64)
    expected_eye = next_target + (np.asarray(current.eye) - np.asarray(current.lookat))

    assert renderer.set_lookat(next_target) is True
    got = renderer.get_camera_state()

    assert got is not None
    np.testing.assert_allclose(got.eye, expected_eye, atol=1e-5)


def test_follow_keeps_world_up_reference_after_orbit_updates():
    renderer = _make_renderer()
    renderer._controller = pygfx.OrbitController(renderer._camera)
    initial = CameraState(
        eye=(10.0, -5.0, 6.0),
        lookat=(1.0, 2.0, 3.0),
        up=(0.0, 0.0, 1.0),
        fov_deg=55.0,
    )

    assert renderer.set_camera_state(initial) is True
    np.testing.assert_allclose(renderer._camera.world.reference_up, (0.0, 0.0, 1.0), atol=1e-6)

    renderer._controller.rotate((0.8, 0.7), (0, 0, 1000, 800), animate=False)
    assert renderer.set_lookat(np.array([4.0, 6.0, 3.0], dtype=np.float64)) is True
    renderer._controller.rotate((0.2, 0.6), (0, 0, 1000, 800), animate=False)
    assert renderer.set_lookat(np.array([7.0, 8.0, 3.0], dtype=np.float64)) is True

    got = renderer.get_camera_state()

    assert got is not None
    np.testing.assert_allclose(got.up, (0.0, 0.0, 1.0), atol=1e-6)
    np.testing.assert_allclose(renderer._camera.world.reference_up, (0.0, 0.0, 1.0), atol=1e-6)
    np.testing.assert_allclose(renderer._controller.target, (7.0, 8.0, 3.0), atol=1e-6)


def test_minimap_camera_uses_whole_scene_bounds():
    renderer = _make_renderer()
    bounds = SceneBounds(
        min_bound=np.array([-100.0, -50.0, 0.0], dtype=np.float64),
        max_bound=np.array([60.0, 90.0, 20.0], dtype=np.float64),
    )

    renderer._update_minimap_camera(bounds)

    left, right, top, bottom = renderer._minimap_world_rect
    center = bounds.get_center()
    assert left <= bounds.min_bound[0] <= right
    assert left <= bounds.max_bound[0] <= right
    assert bottom <= bounds.min_bound[1] <= top
    assert bottom <= bounds.max_bound[1] <= top
    np.testing.assert_allclose(renderer._minimap_camera.local.position[:2], center[:2], atol=1e-5)


def test_world_to_minimap_uv_maps_center_to_middle():
    renderer = _make_renderer()
    renderer._minimap_world_rect = (-50.0, 50.0, 40.0, -40.0)

    uv = renderer._world_to_minimap_uv(np.array([0.0, 0.0, 10.0], dtype=np.float64))

    assert uv is not None
    np.testing.assert_allclose(uv, (0.5, 0.5), atol=1e-6)


def test_minimap_gpu_overlay_updates_marker_positions_and_visibility():
    renderer = _make_renderer()
    renderer._minimap_world_rect = (-10.0, 10.0, 10.0, -10.0)
    renderer.visualizer.camera_scene_query_service = SimpleNamespace(
        get_focus_position=lambda: np.array([0.0, 0.0, 2.0], dtype=np.float64)
    )
    renderer.get_camera_state = lambda: CameraState(
        eye=(10.0, 0.0, 5.0),
        lookat=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
        fov_deg=60.0,
    )

    renderer._ensure_minimap_gpu_overlay(200, 200)
    renderer._update_minimap_gpu_overlay_state(200, 200)

    tracked = renderer._minimap_overlay_objects["tracked"]
    camera = renderer._minimap_overlay_objects["camera"]
    arrow = renderer._minimap_overlay_objects["arrow"]
    assert tracked.visible is True
    assert camera.visible is True
    assert arrow.visible is True
    np.testing.assert_allclose(tracked.geometry.positions.data[0], [100.0, 100.0, 0.0])
    np.testing.assert_allclose(camera.geometry.positions.data[0], [186.0, 100.0, 0.0])
    np.testing.assert_allclose(arrow.geometry.positions.data[0], [186.0, 100.0, 0.0])

    renderer.visualizer.camera_scene_query_service.get_focus_position = lambda: None
    renderer.get_camera_state = lambda: None
    renderer._update_minimap_gpu_overlay_state(200, 200)

    assert tracked.visible is False
    assert camera.visible is False
    assert arrow.visible is False


def test_render_minimap_draws_scene_before_gpu_overlay():
    renderer = _make_renderer()
    bounds = object()
    scene = object()
    minimap_camera = object()
    overlay_scene = object()
    overlay_camera = object()
    render_calls = []
    viewport = SimpleNamespace(
        rect=None,
        render=lambda scene_arg, camera_arg, *, flush: render_calls.append(
            (scene_arg, camera_arg, flush)
        ),
    )
    renderer._minimap_enabled = True
    renderer._renderer = object()
    renderer._scene = scene
    renderer._minimap_camera = minimap_camera
    renderer._minimap_viewport = viewport
    renderer._minimap_overlay_scene = overlay_scene
    renderer._minimap_overlay_camera = overlay_camera
    renderer.compute_scene_bounds = lambda *, scope: bounds
    renderer._update_minimap_camera = lambda value: None
    renderer._compute_minimap_rect = lambda: (10, 20, 200, 200)
    renderer._ensure_minimap_gpu_overlay = lambda width, height: None
    renderer._update_minimap_gpu_overlay_state = lambda width, height: None

    assert renderer._render_minimap() is True

    assert viewport.rect == (10, 20, 200, 200)
    assert render_calls == [
        (scene, minimap_camera, False),
        (overlay_scene, overlay_camera, True),
    ]


def test_resize_invalidates_minimap_gpu_overlay_size():
    renderer = _make_renderer()
    renderer._minimap_overlay_size = (200, 200)
    calls = []
    renderer._reposition_hud_overlays = lambda: calls.append("hud")
    renderer._request_canvas_draw = lambda: calls.append("draw")

    renderer.resize(640, 480)

    assert renderer._width == 640
    assert renderer._height == 480
    assert renderer._minimap_overlay_size is None
    assert calls == ["hud", "draw"]
