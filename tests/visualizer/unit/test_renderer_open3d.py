"""Unit tests for Open3D renderer object ownership and native synchronization."""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from visualizer.src.materials.appearance import (
    AppearanceIntent,
    MaterialDisplayMode,
    resolve_appearance,
)
from visualizer.src.model import RenderObjectState, Transform, make_text_label_state
from visualizer.src.pipeline.core import ViewModel
from visualizer.src.renderers.open3d.lighting_controls import Open3DLightingControlsMixin
from visualizer.src.renderers.open3d.renderer import Open3DRenderer
from visualizer.src.scene.orientation_frame_payloads import make_orientation_frame_handle
from visualizer.src.scene.surface_payloads import BeamformingSurface
from visualizer.src.services.object_identity import make_node_geometry_name
from visualizer.src.state import MpcVisibility
from visualizer.src.types.render_payloads import (
    MaterialPayload,
    MeshPayload,
    SurfaceColorSource,
)


def test_display_pbr_factors_soften_untextured_target_metal() -> None:
    roughness, metallic, reflectance = Open3DRenderer._display_pbr_factors(
        name="target:parkedcar::mesh",
        roughness=0.2,
        metallic=1.0,
        reflectance=0.8,
        has_albedo=False,
    )

    assert roughness == pytest.approx(0.55)
    assert metallic == pytest.approx(0.45)
    assert reflectance == pytest.approx(0.35)


def test_display_pbr_factors_keep_textured_or_scene_metal() -> None:
    textured = Open3DRenderer._display_pbr_factors(
        name="target:parkedcar::mesh",
        roughness=0.2,
        metallic=1.0,
        reflectance=0.8,
        has_albedo=True,
    )
    scene = Open3DRenderer._display_pbr_factors(
        name="scene:roof::mesh",
        roughness=0.2,
        metallic=1.0,
        reflectance=0.8,
        has_albedo=False,
    )

    assert textured == pytest.approx((0.2, 1.0, 0.8))
    assert scene == pytest.approx((0.2, 1.0, 0.8))


class _DummyFilamentScene:
    def __init__(self) -> None:
        self.culling_calls: list[tuple[str, bool]] = []

    def set_geometry_culling(self, name: str, enabled: bool) -> None:
        self.culling_calls.append((name, bool(enabled)))


class _DummyOpen3DScene:
    def __init__(self) -> None:
        self.scene = _DummyFilamentScene()
        self.transforms: list[tuple[str, np.ndarray]] = []
        self.visibility: dict[str, bool] = {}
        self.visibility_calls: list[tuple[str, bool]] = []

    def set_geometry_transform(self, name: str, transform: np.ndarray) -> None:
        self.transforms.append((name, np.asarray(transform, dtype=float).copy()))

    def show_geometry(self, name: str, visible: bool) -> None:
        self.visibility_calls.append((name, bool(visible)))
        self.visibility[name] = bool(visible)

    def geometry_is_visible(self, name: str) -> bool:
        return self.visibility.get(name, False)


class _DummyO3DVisualizer:
    def __init__(self) -> None:
        self.scene = _DummyOpen3DScene()
        self.added: list[str] = []
        self.removed: list[str] = []
        self.shown: list[tuple[str, bool]] = []
        self.visibility_draw_snapshots: list[dict[str, bool]] = []
        self.camera_calls: list[tuple[float, list[float], list[float], list[float]]] = []
        self.geometries: dict[str, object] = {}
        self.materials: dict[str, object] = {}
        self.ibl_calls: list[str] = []
        self.skybox_calls: list[bool] = []
        self._ibl_intensity = 30000.0
        self.content_rect = SimpleNamespace(width=1200, height=800)

    def add_geometry(self, name: str, geometry, material) -> None:
        self.added.append(name)
        self.geometries[name] = geometry
        self.materials[name] = material
        self.scene.visibility[name] = True

    def modify_geometry_material(self, name: str, material) -> None:
        self.materials[name] = material

    def remove_geometry(self, name: str) -> None:
        self.removed.append(name)
        self.scene.visibility.pop(name, None)

    def show_geometry(self, name: str, visible: bool) -> None:
        self.shown.append((name, bool(visible)))
        self.scene.show_geometry(name, visible)
        # Windows' O3DVisualizer path can synchronously present here. Tests
        # retain the complete low-level scene snapshot seen by that draw.
        self.visibility_draw_snapshots.append(dict(self.scene.visibility))

    def setup_camera(self, fov: float, lookat, eye, up) -> None:
        self.camera_calls.append((float(fov), list(lookat), list(eye), list(up)))

    def post_redraw(self) -> None:
        pass

    def set_ibl_intensity(self, intensity: float) -> None:
        self._ibl_intensity = float(intensity)

    def set_ibl(self, ibl_name: str) -> None:
        self.ibl_calls.append(ibl_name)

    def show_skybox(self, show: bool) -> None:
        self.skybox_calls.append(bool(show))


class _DummyGeometry:
    def __init__(self, center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self._center = np.asarray(center, dtype=float)
        self.vertices = np.asarray([self._center], dtype=float)

    def get_center(self) -> np.ndarray:
        return self._center.copy()

    def translate(self, value, relative: bool = True) -> None:
        vec = np.asarray(value, dtype=float).reshape(-1)
        if vec.size < 3:
            return
        vec = vec[:3]
        if relative:
            self._center = self._center + vec
        else:
            self._center = vec.copy()
        self.vertices = np.asarray([self._center], dtype=float)


def _make_marker_handle(
    kind: str = "tx",
    index: int = 0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RenderObjectState:
    vertices = np.asarray(
        [[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.2, 0.0]],
        dtype=float,
    )
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    return RenderObjectState(
        id=make_node_geometry_name(kind, index, "marker"),
        payload=MeshPayload(vertices=vertices, triangles=triangles),
        world_transform=Transform.from_translation(center),
    )


def _make_visualizer_stub() -> SimpleNamespace:
    return SimpleNamespace(
        tx_markers=[],
        rx_markers=[],
        tx_labels=[],
        rx_labels=[],
        _tx_label_base_vertices=[],
        _rx_label_base_vertices=[],
        label_offset_x=0.0,
        label_offset_y=0.0,
        label_offset_z=0.5,
    )


@pytest.fixture
def renderer() -> Open3DRenderer:
    viz = _make_visualizer_stub()
    renderer = Open3DRenderer(viz)
    renderer._o3d_vis = _DummyO3DVisualizer()
    return renderer


def test_open3d_renderer_composes_lighting_controls(renderer: Open3DRenderer) -> None:
    assert isinstance(renderer, Open3DLightingControlsMixin)


@pytest.mark.parametrize("setting", ["line", "point"])
def test_global_size_change_restores_backend_cached_native_material(
    renderer: Open3DRenderer,
    setting: str,
) -> None:
    """Global Open3D knobs must not route material repair through app services."""
    name = "scene:building::mesh"
    cached_material = object()
    renderer._geometry_names.add(name)
    renderer._pbr_materials[name] = cached_material
    renderer._o3d_vis.materials[name] = object()

    def _unexpected_app_repair(_alpha: float) -> None:
        raise AssertionError("renderer material repair crossed into an app service")

    renderer.visualizer.current_building_alpha = 0.4
    renderer.visualizer.current_target_alpha = 0.5
    renderer.visualizer.set_building_transparency = _unexpected_app_repair
    renderer.visualizer.set_target_transparency = _unexpected_app_repair

    if setting == "line":
        assert renderer.set_line_width(4.0) is True
    else:
        assert renderer.set_point_size(7.0) is True

    assert renderer._o3d_vis.materials[name] is cached_material
    assert renderer._pbr_materials[name] is cached_material


def test_apply_frame_does_not_override_service_owned_pov_hidden_node(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = _make_marker_handle(kind="rx")
    marker.visible = False
    renderer.visualizer.rx_markers = [marker]
    assert renderer.ensure_object(marker.to_render_object()) is True
    assert renderer._applied_render_objects[marker.id].visible is False

    renderer._o3d_vis.shown.clear()
    monkeypatch.setattr(renderer, "_apply_mpc_data", lambda _packet: True)
    monkeypatch.setattr(renderer, "_apply_coverage_data", lambda _packet: True)
    monkeypatch.setattr(renderer, "_apply_beamforming", lambda _packet: True)
    packet = SimpleNamespace(
        colorbar=None,
        stats_text="",
        beamforming_meshes=[],
    )

    assert renderer.apply_frame(packet) is True

    assert marker.visible is False
    assert renderer._applied_render_objects[marker.id].visible is False
    assert renderer.is_named_visible(marker.id) is False
    assert (marker.id, True) not in renderer._o3d_vis.shown


def test_open3d_apply_frame_retries_failed_overlay_before_committing_packet(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_packet = SimpleNamespace(colorbar=None, stats_text="old", beamforming_meshes=())
    new_packet = SimpleNamespace(colorbar=None, stats_text="new", beamforming_meshes=())
    renderer.last_frame_packet = old_packet
    coverage_results = iter((False, True))
    coverage_calls: list[object] = []

    monkeypatch.setattr(renderer, "_apply_mpc_data_diff", lambda _old, _new: True)

    def apply_coverage(_old, packet) -> bool:
        coverage_calls.append(packet)
        return next(coverage_results)

    monkeypatch.setattr(renderer, "_apply_coverage_data_diff", apply_coverage)
    monkeypatch.setattr(renderer, "_apply_beamforming", lambda _packet: True)
    monkeypatch.setattr(renderer, "_apply_colorbar_diff", lambda _old, _new: None)
    monkeypatch.setattr(renderer, "_apply_stats_diff", lambda _old, _new: None)

    assert renderer.apply_frame(new_packet) is False
    assert renderer.last_frame_packet is old_packet

    assert renderer.apply_frame(new_packet) is True
    assert coverage_calls == [new_packet, new_packet]
    assert renderer.last_frame_packet is new_packet


def test_open3d_apply_frame_retries_failed_mpc_before_committing_packet(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_packet = SimpleNamespace(colorbar=None, stats_text="old", beamforming_meshes=())
    new_packet = SimpleNamespace(colorbar=None, stats_text="new", beamforming_meshes=())
    renderer.last_frame_packet = old_packet
    mpc_results = iter((False, True))
    mpc_calls: list[object] = []

    def apply_mpc(_old, packet) -> bool:
        mpc_calls.append(packet)
        return next(mpc_results)

    monkeypatch.setattr(renderer, "_apply_mpc_data_diff", apply_mpc)
    monkeypatch.setattr(renderer, "_apply_coverage_data_diff", lambda _old, _new: True)
    monkeypatch.setattr(renderer, "_apply_beamforming", lambda _packet: True)
    monkeypatch.setattr(renderer, "_apply_colorbar_diff", lambda _old, _new: None)
    monkeypatch.setattr(renderer, "_apply_stats_diff", lambda _old, _new: None)

    assert renderer.apply_frame(new_packet) is False
    assert renderer.last_frame_packet is old_packet

    assert renderer.apply_frame(new_packet) is True
    assert mpc_calls == [new_packet, new_packet]
    assert renderer.last_frame_packet is new_packet


def test_open3d_runtime_stats_report_native_content_size(renderer: Open3DRenderer) -> None:
    renderer._o3d_vis.os_frame = SimpleNamespace(width=1280, height=900)

    stats = renderer.get_runtime_stats()

    assert stats["renderer_content_size"] == [1200.0, 800.0]
    assert stats["renderer_window_size"] == [1280.0, 900.0]


def test_lighting_preset_routes_through_open3d_lighting_controls(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[bool] = []
    redraws: list[bool] = []
    monkeypatch.setattr(renderer, "_resolve_default_ibl", lambda: "default")
    monkeypatch.setattr(renderer, "update_renderer", lambda: updates.append(True))
    monkeypatch.setattr(renderer, "_post_redraw", lambda: redraws.append(True))

    assert renderer.set_lighting_preset("soft") is True

    assert renderer._o3d_vis.ibl_calls == ["default"]
    assert renderer._o3d_vis._ibl_intensity == pytest.approx(15000.0)
    assert renderer._ibl_name == "default"
    assert renderer._ibl_intensity == pytest.approx(15000.0)
    assert updates == [True]
    assert redraws == [True, True]


def test_set_overview_camera_applies_camera_inside_open3d_renderer(
    renderer: Open3DRenderer,
) -> None:
    center = np.array([10.0, 20.0, 3.0], dtype=np.float64)
    extent = np.array([40.0, 20.0, 6.0], dtype=np.float64)

    assert renderer.set_overview_camera("front", (center, extent), fov=50.0, distance=30.0)

    fov, lookat, eye, up = renderer._o3d_vis.camera_calls[-1]
    assert fov == pytest.approx(50.0)
    np.testing.assert_allclose(lookat, center, atol=1e-6)
    np.testing.assert_allclose(eye, [-20.0, 20.0, 9.0], atol=1e-6)
    np.testing.assert_allclose(up, [0.0, 0.0, 1.0], atol=1e-6)


def test_compute_scene_bounds_uses_applied_payloads_and_world_transforms(
    renderer: Open3DRenderer,
) -> None:
    scene = RenderObjectState(
        id="scene:building::mesh",
        payload=MeshPayload(
            vertices=np.asarray(
                [[-1.0, -2.0, 0.0], [3.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
                dtype=float,
            ),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )
    target = RenderObjectState(
        id="target:walker::mesh",
        payload=MeshPayload(
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [1.0, 0.0, 1.0]],
                dtype=float,
            ),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
        world_transform=Transform.from_translation((10.0, 20.0, 30.0)),
    )
    assert renderer.ensure_object(scene.to_render_object()) is True
    assert renderer.ensure_object(target.to_render_object()) is True

    bbox = renderer.compute_scene_bounds(scope="whole")

    assert bbox is not None
    np.testing.assert_allclose(bbox.get_center(), [5.5, 10.0, 16.0])
    np.testing.assert_allclose(bbox.get_extent(), [13.0, 24.0, 32.0])


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
    renderer: Open3DRenderer,
    appearance: AppearanceIntent,
    object_id: str,
) -> None:
    anchor = RenderObjectState(
        id="scene:anchor::mesh",
        payload=MeshPayload(
            vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=float),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )
    distant = RenderObjectState(
        id=object_id,
        payload=anchor.payload,
        world_transform=Transform.from_translation((100.0, 0.0, 0.0)),
        visible=resolve_appearance(appearance).visible,
    )
    assert renderer.ensure_object(anchor.to_render_object()) is True
    assert renderer.ensure_object(distant.to_render_object()) is True

    visible = renderer.compute_scene_bounds(scope="visible")
    whole = renderer.compute_scene_bounds(scope="whole")

    assert visible is not None
    assert whole is not None
    np.testing.assert_allclose(visible.get_max_bound(), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(whole.get_max_bound(), [101.0, 1.0, 1.0])


def test_set_pov_camera_converts_pose_inside_open3d_renderer(
    renderer: Open3DRenderer,
) -> None:
    assert renderer.set_pov_camera([1.0, 2.0, 3.0], [np.pi / 2.0, 0.0, 0.0], "forward")

    fov, lookat, eye, up = renderer._o3d_vis.camera_calls[-1]
    assert fov == pytest.approx(60.0)
    np.testing.assert_allclose(eye, [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(lookat, [1.0, 12.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(up, [0.0, 0.0, 1.0], atol=1e-6)


def test_ensure_named_geometry_preserves_label_color_material(
    renderer: Open3DRenderer,
) -> None:
    label = o3d.geometry.TriangleMesh.create_box(width=0.2, height=0.2, depth=0.02)
    label.paint_uniform_color([0.2, 0.4, 0.8])

    assert renderer.ensure_named_geometry("target:car_1::label", label)

    added = renderer._o3d_vis.geometries["target:car_1::label"]
    np.testing.assert_allclose(np.asarray(added.vertex_colors), np.asarray(label.vertex_colors))
    material = renderer._o3d_vis.materials["target:car_1::label"]
    assert material.shader == "defaultUnlit"
    assert list(material.base_color) == pytest.approx([0.2, 0.4, 0.8, 1.0])


def test_mpc_diff_updates_independent_line_and_color_changes(
    renderer: Open3DRenderer,
) -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    old_packet = SimpleNamespace(
        mpc_points=points,
        mpc_lines=np.asarray([[0, 1]], dtype=np.int32),
        mpc_colors=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        mpc_bounce_points=None,
        mpc_bounce_colors=None,
        mpc_visibility=MpcVisibility(bounce_points=False),
    )
    new_packet = SimpleNamespace(
        mpc_points=points,
        mpc_lines=np.asarray([[1, 2]], dtype=np.int32),
        mpc_colors=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64),
        mpc_bounce_points=None,
        mpc_bounce_colors=None,
        mpc_visibility=MpcVisibility(bounce_points=False),
    )

    renderer._apply_mpc_data(old_packet)
    renderer._apply_mpc_data_diff(old_packet, new_packet)

    np.testing.assert_array_equal(np.asarray(renderer.mpc_lineset.lines), new_packet.mpc_lines)
    np.testing.assert_allclose(np.asarray(renderer.mpc_lineset.colors), new_packet.mpc_colors)


def test_open3d_mpc_native_upload_failure_is_reported(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = SimpleNamespace(
        mpc_points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        mpc_lines=np.asarray([[0, 1]], dtype=np.int32),
        mpc_colors=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        mpc_bounce_points=None,
        mpc_bounce_colors=None,
        mpc_visibility=MpcVisibility(bounce_points=False),
    )
    upload_results = iter((False, True))
    monkeypatch.setattr(
        renderer,
        "_add_or_update_geometry",
        lambda *_args: next(upload_results),
    )

    assert renderer._apply_mpc_data(packet) is False
    assert renderer._apply_mpc_data(packet) is True


def test_apply_mpc_data_accepts_readonly_view_model_arrays(renderer: Open3DRenderer) -> None:
    view_model = ViewModel(
        tx_positions=np.empty((0, 3), dtype=np.float64),
        rx_positions=np.empty((0, 3), dtype=np.float64),
        tx_orientations=np.empty((0, 3), dtype=np.float64),
        rx_orientations=np.empty((0, 3), dtype=np.float64),
        mpc_points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
        mpc_lines=np.asarray([[0, 1]], dtype=np.int32),
        mpc_colors=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        colorbar=None,
        stats_text="",
        mpc_visibility=MpcVisibility(),
        mpc_bounce_points=np.asarray([[0.5, 0.0, 0.0]], dtype=np.float64),
        mpc_bounce_colors=np.asarray([[1.0, 1.0, 1.0]], dtype=np.float64),
        target_positions=np.empty((0, 3), dtype=np.float64),
        target_orientations=np.empty((0, 3), dtype=np.float64),
        target_mesh_files=[],
        target_use_ply_positions=[],
        target_metadata=[],
    )

    assert view_model.mpc_points.flags.writeable is False

    renderer._apply_mpc_data(view_model.to_render_packet())

    np.testing.assert_allclose(np.asarray(renderer.mpc_lineset.points), view_model.mpc_points)
    np.testing.assert_array_equal(np.asarray(renderer.mpc_lineset.lines), view_model.mpc_lines)
    np.testing.assert_allclose(np.asarray(renderer.mpc_pcd.points), view_model.mpc_bounce_points)


def test_open3d_mpc_visibility_transitions_keep_only_physical_bounces(
    renderer: Open3DRenderer,
) -> None:
    enabled = ViewModel(
        tx_positions=np.empty((0, 3), dtype=np.float64),
        rx_positions=np.empty((0, 3), dtype=np.float64),
        tx_orientations=np.empty((0, 3), dtype=np.float64),
        rx_orientations=np.empty((0, 3), dtype=np.float64),
        mpc_points=np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        mpc_lines=np.asarray([[0, 1], [1, 2]], dtype=np.int32),
        mpc_colors=np.ones((2, 3), dtype=np.float64),
        colorbar=None,
        stats_text="",
        mpc_visibility=MpcVisibility(),
        mpc_bounce_points=np.asarray([[0.5, 0.0, 0.0]], dtype=np.float64),
        mpc_bounce_colors=np.asarray([[0.2, 0.4, 0.8]], dtype=np.float64),
        target_positions=np.empty((0, 3), dtype=np.float64),
        target_orientations=np.empty((0, 3), dtype=np.float64),
        target_mesh_files=[],
        target_use_ply_positions=[],
        target_metadata=[],
    )
    enabled_packet = enabled.to_render_packet()
    renderer._apply_mpc_data(enabled_packet)

    bounces_only = replace(
        enabled,
        mpc_visibility=MpcVisibility(paths=False, bounce_points=True),
    )
    bounces_only_packet = bounces_only.to_render_packet()
    renderer._apply_mpc_data_diff(enabled_packet, bounces_only_packet)

    assert renderer.MPC_LINES_NAME not in renderer._geometry_names
    assert renderer.MPC_POINTS_NAME in renderer._geometry_names
    np.testing.assert_allclose(np.asarray(renderer.mpc_pcd.points), bounces_only.mpc_bounce_points)

    disabled = replace(
        bounces_only,
        mpc_visibility=MpcVisibility(enabled=False, paths=True, bounce_points=True),
    )
    renderer._apply_mpc_data_diff(bounces_only_packet, disabled.to_render_packet())

    assert renderer.MPC_LINES_NAME not in renderer._geometry_names
    assert renderer.MPC_POINTS_NAME not in renderer._geometry_names


def _make_beamforming_surface(
    surface_id: str = "beamforming:tx_0_pair_0:mesh",
    *,
    offset: float = 0.0,
) -> BeamformingSurface:
    vertices = np.asarray(
        [
            [offset, 0.0, 0.0],
            [offset + 0.5, 0.0, 0.0],
            [offset, 0.5, 0.0],
        ],
        dtype=np.float32,
    )
    return BeamformingSurface(
        id=surface_id,
        payload=MeshPayload(
            vertices=vertices,
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
            normals=np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
            vertex_colors=np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
        ),
    )


def test_open3d_beamforming_create_noop_and_payload_replacement(
    renderer: Open3DRenderer,
) -> None:
    surface = _make_beamforming_surface()
    packet = SimpleNamespace(beamforming_meshes=(surface,))

    assert renderer._apply_beamforming(packet) is True

    geometry = renderer._applied_beamforming_surfaces[surface.id].geometry
    material = renderer._o3d_vis.materials[surface.id]
    assert geometry.has_vertex_colors()
    assert material.shader == "defaultUnlit"
    assert list(material.base_color) == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert renderer._o3d_vis.added == [surface.id]
    assert surface.id in renderer._geometry_names

    assert renderer._apply_beamforming(packet) is True
    assert renderer._o3d_vis.added == [surface.id]
    assert renderer._o3d_vis.removed == []

    replacement = _make_beamforming_surface(offset=2.0)
    assert renderer._apply_beamforming(SimpleNamespace(beamforming_meshes=(replacement,))) is True
    assert renderer._o3d_vis.added == [surface.id, surface.id]
    assert renderer._o3d_vis.removed == [surface.id]
    assert renderer._applied_beamforming_surfaces[surface.id].payload is replacement.payload
    assert renderer._o3d_vis.materials[surface.id].shader == "defaultUnlit"


def test_open3d_beamforming_failed_replacement_is_retried(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = _make_beamforming_surface()
    replacement = _make_beamforming_surface(offset=2.0)
    assert renderer._apply_beamforming(SimpleNamespace(beamforming_meshes=(surface,)))
    original_add = renderer._add_or_update_geometry
    fail_once = True

    def flaky_add(name, geometry, material):
        nonlocal fail_once
        if name == surface.id and fail_once:
            fail_once = False
            return False
        return original_add(name, geometry, material)

    monkeypatch.setattr(renderer, "_add_or_update_geometry", flaky_add)

    packet = SimpleNamespace(beamforming_meshes=(replacement,))
    assert renderer._apply_beamforming(packet) is False
    assert renderer._applied_beamforming_surfaces[surface.id].payload is surface.payload

    assert renderer._apply_beamforming(packet) is True
    assert renderer._applied_beamforming_surfaces[surface.id].payload is replacement.payload


def test_open3d_beamforming_failed_stale_removal_is_retried(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept = _make_beamforming_surface("beamforming:kept:mesh")
    stale = _make_beamforming_surface("beamforming:stale:mesh", offset=2.0)
    assert renderer._apply_beamforming(SimpleNamespace(beamforming_meshes=(kept, stale)))
    original_remove = renderer._remove_geometry
    fail_once = True

    def flaky_remove(name: str) -> bool:
        nonlocal fail_once
        if name == stale.id and fail_once:
            fail_once = False
            return False
        return original_remove(name)

    monkeypatch.setattr(renderer, "_remove_geometry", flaky_remove)
    packet = SimpleNamespace(beamforming_meshes=(kept,))

    assert renderer._apply_beamforming(packet) is False
    assert stale.id in renderer._applied_beamforming_surfaces
    assert stale.id in renderer._geometry_names

    assert renderer._apply_beamforming(packet) is True
    assert stale.id not in renderer._applied_beamforming_surfaces
    assert stale.id not in renderer._geometry_names


def test_open3d_reset_clears_beamforming_snapshot(renderer: Open3DRenderer) -> None:
    surface = _make_beamforming_surface()
    assert renderer._apply_beamforming(SimpleNamespace(beamforming_meshes=(surface,)))

    renderer.reset_state()

    assert renderer._applied_beamforming_surfaces == {}
    assert renderer._beamforming_owned_names == set()


def test_open3d_native_textures_survive_reset_and_release_on_close(
    renderer: Open3DRenderer,
) -> None:
    native_texture = object()
    renderer._texture_image_cache["revision"] = native_texture
    renderer._texture_source_identities["texture.png"] = "revision"

    renderer.reset_state()

    assert renderer._texture_image_cache == {"revision": native_texture}
    assert renderer._texture_source_identities == {"texture.png": "revision"}

    renderer._o3d_vis = None
    renderer.close()

    assert renderer._texture_image_cache == {}
    assert renderer._texture_source_identities == {}


def test_open3d_native_texture_hit_does_not_redecode_evicted_pixels(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from visualizer.src.materials import texture_assets

    texture_path = tmp_path / "cached.png"
    texture_path.write_bytes(b"native-cache-identity")
    identity_result = texture_assets.texture_asset_identity(texture_path)
    assert identity_result is not None
    identity, _resolved = identity_result
    native_texture = object()
    renderer._texture_image_cache[identity] = native_texture
    monkeypatch.setattr(
        texture_assets,
        "load_decoded_texture",
        lambda _path: pytest.fail("a warm native texture must not be decoded again"),
    )

    assert renderer._load_texture_cached(str(texture_path)) is native_texture


def test_open3d_reset_retains_failed_beamforming_ownership_for_retry(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface = _make_beamforming_surface()
    assert renderer._apply_beamforming(SimpleNamespace(beamforming_meshes=(surface,)))
    original_remove = renderer._remove_geometry
    fail_once = True

    def flaky_remove(name: str) -> bool:
        nonlocal fail_once
        if name == surface.id and fail_once:
            fail_once = False
            return False
        return original_remove(name)

    monkeypatch.setattr(renderer, "_remove_geometry", flaky_remove)

    renderer.reset_state()

    assert surface.id in renderer._geometry_names
    assert surface.id in renderer._beamforming_owned_names
    assert surface.id in renderer._applied_beamforming_surfaces

    renderer.reset_state()

    assert surface.id not in renderer._geometry_names
    assert surface.id not in renderer._beamforming_owned_names
    assert surface.id not in renderer._applied_beamforming_surfaces


def test_set_geometry_transform_fast_uses_mapped_geometry_name(renderer: Open3DRenderer) -> None:
    sphere = _DummyGeometry()
    sphere_name = make_node_geometry_name("tx", 0, "marker")
    assert renderer.add_or_update_named_geometry(name=sphere_name, geometry=sphere)

    ok = renderer.set_geometry_transform_fast(sphere, np.asarray([4.0, 5.0, 6.0], dtype=float))

    assert ok is True
    assert renderer._o3d_vis.scene.transforms[-1][0] == sphere_name


def test_remove_geometry_from_visualizer_resolves_mapped_name(renderer: Open3DRenderer) -> None:
    sphere = _DummyGeometry()
    sphere_name = make_node_geometry_name("tx", 0, "marker")
    assert renderer.add_or_update_named_geometry(name=sphere_name, geometry=sphere)

    renderer.remove_geometry_from_visualizer(sphere)

    assert sphere_name in renderer._o3d_vis.removed
    assert sphere_name not in renderer._geometry_names
    assert id(sphere) not in renderer._geometry_id_to_name


def test_set_culling_applies_to_existing_geometry_and_redraws(renderer: Open3DRenderer) -> None:
    renderer._geometry_names.update({"scene_mesh", "target_mesh"})
    redraws: list[bool] = []
    renderer._post_redraw = lambda: redraws.append(True)  # type: ignore[method-assign]

    renderer.set_culling(True)

    assert ("scene_mesh", True) in renderer._o3d_vis.scene.scene.culling_calls
    assert ("target_mesh", True) in renderer._o3d_vis.scene.scene.culling_calls
    assert redraws == [True]


def test_new_open3d_geometry_inherits_current_culling_state(renderer: Open3DRenderer) -> None:
    renderer._culling_enabled = True
    assert renderer.add_or_update_named_geometry(name="scene_mesh", geometry=_DummyGeometry())

    assert ("scene_mesh", True) in renderer._o3d_vis.scene.scene.culling_calls

    renderer._o3d_vis.scene.scene.culling_calls.clear()
    renderer._culling_enabled = False
    assert renderer.add_or_update_named_geometry(name="scene_mesh_2", geometry=_DummyGeometry())

    assert ("scene_mesh_2", False) in renderer._o3d_vis.scene.scene.culling_calls


def test_open3d_screenshot_dimensions_apply_resolution_scale(
    renderer: Open3DRenderer,
) -> None:
    renderer._o3d_vis.content_rect = SimpleNamespace(width=1200, height=800)

    assert renderer._capture_dimensions(scale=0.5) == (600, 400)
    assert renderer._capture_dimensions(scale=2.0) == (2400, 1600)
    assert renderer._capture_dimensions(scale=-1.0) == (1200, 800)


def test_open3d_desktop_capture_fallback_uses_content_rect(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures: list[dict[str, int]] = []
    renderer._o3d_vis.scene = None
    renderer._o3d_vis.content_rect = SimpleNamespace(x=11, y=13, width=5, height=4)

    class _FakeMss:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def grab(self, region):
            captures.append(dict(region))
            image = np.zeros((region["height"], region["width"], 4), dtype=np.uint8)
            image[:, :] = [10, 20, 30, 255]
            return image

    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(mss=lambda: _FakeMss()))

    image = renderer.export_screenshot_to_array(resolution_scale=2.0)

    assert captures == [{"left": 11, "top": 13, "width": 5, "height": 4}]
    assert image.shape == (8, 10, 3)
    np.testing.assert_array_equal(image[0, 0], [30, 20, 10])


def test_open3d_desktop_capture_skips_unscoped_full_monitor(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renderer._o3d_vis.scene = None
    renderer._o3d_vis.content_rect = SimpleNamespace(width=5, height=4)

    class _FakeMss:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def grab(self, region):  # noqa: ARG002
            raise AssertionError("full-monitor screenshot fallback should not run")

    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(mss=lambda: _FakeMss()))

    image = renderer.export_screenshot_to_array()

    assert image.shape == (480, 640, 3)
    assert np.all(image == 0)


def test_update_geometry_in_visualizer_resolves_mapped_name(renderer: Open3DRenderer) -> None:
    sphere = _DummyGeometry()
    sphere_name = make_node_geometry_name("tx", 0, "marker")
    fallback_name = f"geometry_{id(sphere)}"
    assert renderer.add_or_update_named_geometry(name=sphere_name, geometry=sphere)

    renderer.update_geometry_in_visualizer(sphere)

    assert renderer._o3d_vis.added[-1] == sphere_name
    assert renderer._geometry_id_to_name[id(sphere)] == sphere_name
    assert fallback_name not in renderer._geometry_names


def test_raw_named_replacement_preserves_hidden_visibility(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sphere = _DummyGeometry()
    sphere_name = make_node_geometry_name("rx", 0, "marker")
    assert renderer.add_or_update_named_geometry(name=sphere_name, geometry=sphere)
    assert renderer.set_named_visibility(sphere_name, False)
    renderer._o3d_vis.shown.clear()
    redraw_order: list[str] = []
    monkeypatch.setattr(renderer, "_post_redraw", lambda: redraw_order.append("post") or True)
    monkeypatch.setattr(
        renderer,
        "_request_visibility_settle_redraw",
        lambda _reason: redraw_order.append("settle"),
    )

    renderer.update_geometry_in_visualizer(sphere)

    assert renderer.is_named_visible(sphere_name) is False
    assert renderer._o3d_vis.shown[-1] == (sphere_name, False)
    assert redraw_order == ["post", "settle"]


def test_orientation_frame_payload_uses_open3d_coordinate_frame_mesh(
    renderer: Open3DRenderer,
) -> None:
    handle = make_orientation_frame_handle(
        make_node_geometry_name("tx", 0, "orientation_frame"),
        size=4.0,
        visible=True,
    )

    assert renderer.ensure_object(handle.to_render_object()) is True

    geometry = renderer._o3d_vis.geometries[handle.id]
    assert isinstance(geometry, o3d.geometry.TriangleMesh)
    assert renderer._geometry_types[handle.id] == "mesh"
    assert handle.id in renderer._pbr_materials
    colors = np.asarray(geometry.vertex_colors)
    assert colors.ndim == 2
    assert colors.shape[0] == len(geometry.vertices)
    assert np.ptp(colors, axis=0).max() > 0.0


def test_modify_geometry_with_missing_albedo_keeps_material_color(
    renderer: Open3DRenderer, tmp_path
) -> None:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh.create_box()
    assert renderer.ensure_named_geometry(
        "solid_box",
        mesh,
        material=MaterialPayload(base_color=(0.2, 0.3, 0.4, 1.0)),
    )

    assert renderer.modify_geometry_material_pbr(
        "solid_box",
        color=[0.4, 0.5, 0.6],
        texture_path=str(tmp_path / "missing.png"),
    )

    material = renderer._pbr_materials["solid_box"]
    assert list(material.base_color) == pytest.approx([0.4, 0.5, 0.6, 1.0])


def test_set_triangle_uvs_updates_native_mesh(
    renderer: Open3DRenderer,
) -> None:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    )
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray([[0, 1, 2]], dtype=np.int32))
    uvs = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    renderer.set_triangle_uvs(mesh, uvs)

    np.testing.assert_allclose(np.asarray(mesh.triangle_uvs), uvs)


def test_ensure_object_accepts_render_object_state_snapshot(
    renderer: Open3DRenderer,
) -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = [2.0, 3.0, 4.0]
    state = RenderObjectState(
        id="target:pedestrian::mesh",
        payload=MeshPayload(vertices=vertices, triangles=triangles),
        material=MaterialPayload(base_color=(0.2, 0.3, 0.4, 1.0)),
        world_transform=Transform(transform),
        visible=False,
    )

    assert renderer.ensure_object(state.to_render_object()) is True

    added = renderer._o3d_vis.geometries[state.id]
    assert isinstance(added, o3d.geometry.TriangleMesh)
    assert added is not state
    assert id(state) not in renderer._geometry_id_to_name
    assert renderer._o3d_vis.scene.transforms[-1][0] == state.id
    np.testing.assert_allclose(renderer._o3d_vis.scene.transforms[-1][1], transform)
    assert renderer._o3d_vis.shown[-1] == (state.id, False)


def test_ensure_object_is_idempotent_and_updates_components_in_place(
    renderer: Open3DRenderer,
) -> None:
    state = _make_marker_handle()
    first = state.to_render_object()

    assert renderer.ensure_object(first) is True
    add_count = len(renderer._o3d_vis.added)
    remove_count = len(renderer._o3d_vis.removed)
    transform_count = len(renderer._o3d_vis.scene.transforms)
    visibility_count = len(renderer._o3d_vis.shown)

    assert renderer.ensure_object(first) is True
    assert len(renderer._o3d_vis.added) == add_count
    assert len(renderer._o3d_vis.removed) == remove_count
    assert len(renderer._o3d_vis.scene.transforms) == transform_count
    assert len(renderer._o3d_vis.shown) == visibility_count

    state.world_transform = Transform.from_translation([2.0, 3.0, 4.0])
    assert renderer.ensure_object(state.to_render_object()) is True
    assert len(renderer._o3d_vis.added) == add_count
    assert len(renderer._o3d_vis.removed) == remove_count
    assert len(renderer._o3d_vis.scene.transforms) == transform_count + 1

    state.visible = False
    assert renderer.ensure_object(state.to_render_object()) is True
    assert len(renderer._o3d_vis.added) == add_count
    assert renderer._o3d_vis.shown[-1] == (state.id, False)


def test_named_visibility_updates_applied_cache_and_ensure_restores_intent(
    renderer: Open3DRenderer,
) -> None:
    state = _make_marker_handle()
    assert renderer.ensure_object(state.to_render_object()) is True

    assert renderer.set_named_visibility(state.id, False) is True
    assert renderer._applied_render_objects[state.id].visible is False
    assert state.visible is True

    assert renderer.ensure_object(state.to_render_object()) is True
    assert renderer._applied_render_objects[state.id].visible is True
    assert renderer.is_named_visible(state.id) is True
    assert renderer._o3d_vis.shown[-1] == (state.id, True)


def test_named_visibility_same_value_does_not_call_native_or_redraw(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime visibility polling must be a backend-local no-op when unchanged."""
    state = _make_marker_handle()
    assert renderer.ensure_object(state.to_render_object()) is True
    renderer._o3d_vis.shown.clear()
    redraws: list[bool] = []
    settle_redraws: list[str] = []
    monkeypatch.setattr(renderer, "_post_redraw", lambda: redraws.append(True))
    monkeypatch.setattr(
        renderer,
        "_request_visibility_settle_redraw",
        lambda reason: settle_redraws.append(reason),
    )

    assert renderer.set_named_visibility(state.id, True) is True
    assert renderer._o3d_vis.shown == []
    assert redraws == []
    assert settle_redraws == []

    assert renderer.set_named_visibility(state.id, False) is True
    assert renderer.set_named_visibility(state.id, False) is True
    assert renderer._o3d_vis.shown == [(state.id, False)]
    assert redraws == [True]
    assert len(settle_redraws) == 1


def test_batched_visibility_reaches_complete_scene_before_any_visualizer_draw(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open3D's per-object redraws must never expose a partial visibility batch."""
    first = _make_marker_handle(kind="rx", index=0)
    second = _make_marker_handle(kind="rx", index=1)
    assert renderer.ensure_object(first.to_render_object()) is True
    assert renderer.ensure_object(second.to_render_object()) is True
    renderer._o3d_vis.shown.clear()
    renderer._o3d_vis.visibility_draw_snapshots.clear()
    monkeypatch.setattr(renderer, "_submit_redraw_now", lambda: True)

    with renderer.batch_updates():
        assert renderer.set_named_visibility(first.id, False) is True
        assert renderer.set_named_visibility(second.id, False) is True

        # The rendered scene changes immediately, but O3DVisualizer's
        # redraw-producing bookkeeping waits for the outer batch boundary.
        assert renderer._o3d_vis.scene.visibility[first.id] is False
        assert renderer._o3d_vis.scene.visibility[second.id] is False
        assert renderer._o3d_vis.shown == []

    assert renderer._o3d_vis.shown == [(first.id, False), (second.id, False)]
    assert renderer._o3d_vis.visibility_draw_snapshots
    for snapshot in renderer._o3d_vis.visibility_draw_snapshots:
        assert snapshot[first.id] is False
        assert snapshot[second.id] is False
    assert renderer._pending_o3d_visualizer_visibility == {}


def test_frame_visibility_bookkeeping_waits_for_end_frame_update(
    renderer: Open3DRenderer,
) -> None:
    first = _make_marker_handle(kind="tx", index=0)
    second = _make_marker_handle(kind="rx", index=0)
    assert renderer.ensure_object(first.to_render_object()) is True
    assert renderer.ensure_object(second.to_render_object()) is True
    renderer._o3d_vis.shown.clear()
    renderer._o3d_vis.visibility_draw_snapshots.clear()

    renderer.begin_frame_update()
    assert renderer.set_named_visibility(first.id, False) is True
    assert renderer.set_named_visibility(second.id, False) is True
    assert renderer._o3d_vis.shown == []

    assert renderer.end_frame_update() is True
    assert renderer._o3d_vis.shown == [(first.id, False), (second.id, False)]
    for snapshot in renderer._o3d_vis.visibility_draw_snapshots:
        assert snapshot[first.id] is False
        assert snapshot[second.id] is False


def test_failed_named_visibility_does_not_update_applied_cache(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_marker_handle()
    assert renderer.ensure_object(state.to_render_object()) is True
    previous = renderer._applied_render_objects[state.id]
    settle_redraws: list[str] = []
    monkeypatch.setattr(
        renderer,
        "_request_visibility_settle_redraw",
        lambda reason: settle_redraws.append(reason),
    )

    def fail_show(_name: str, _visible: bool) -> None:
        raise RuntimeError("native visibility failed")

    monkeypatch.setattr(renderer._o3d_vis.scene, "show_geometry", fail_show)

    assert renderer.set_named_visibility(state.id, False) is False
    assert renderer._applied_render_objects[state.id] is previous
    assert renderer.is_named_visible(state.id) is True
    assert settle_redraws == []


def test_failed_visibility_keeps_camera_bounds_truthful_until_retry(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RenderObjectState(
        id="target:retry::mesh",
        payload=MeshPayload(
            vertices=np.asarray([[0, 0, 0], [2, 0, 0], [0, 2, 2]], dtype=float),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
        world_transform=Transform.from_translation((20.0, 0.0, 0.0)),
    )
    assert renderer.ensure_object(state.to_render_object()) is True
    original_show = renderer._o3d_vis.scene.show_geometry
    failures = 1

    def fail_once(name: str, visible: bool) -> None:
        nonlocal failures
        if failures:
            failures -= 1
            raise RuntimeError("native visibility failed")
        original_show(name, visible)

    monkeypatch.setattr(renderer._o3d_vis.scene, "show_geometry", fail_once)

    assert renderer.set_visible(state.id, False) is False
    failed_bounds = renderer.compute_scene_bounds(scope="visible")
    assert failed_bounds is not None
    np.testing.assert_allclose(failed_bounds.get_max_bound(), [22.0, 2.0, 2.0])

    assert renderer.set_visible(state.id, False) is True
    assert renderer.compute_scene_bounds(scope="visible") is None
    whole_bounds = renderer.compute_scene_bounds(scope="whole")
    assert whole_bounds is not None
    np.testing.assert_allclose(whole_bounds.get_max_bound(), [22.0, 2.0, 2.0])


def test_visualizer_bookkeeping_failure_keeps_verified_scene_state_for_retry(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_marker_handle()
    assert renderer.ensure_object(state.to_render_object()) is True
    original_show = renderer._o3d_vis.show_geometry

    def fail_show(_name: str, _visible: bool) -> None:
        raise RuntimeError("geometry-tree update failed")

    monkeypatch.setattr(renderer._o3d_vis, "show_geometry", fail_show)

    assert renderer.set_named_visibility(state.id, False) is True
    assert renderer._o3d_vis.scene.geometry_is_visible(state.id) is False
    assert renderer.is_named_visible(state.id) is False
    assert renderer._applied_render_objects[state.id].visible is False
    assert renderer._pending_o3d_visualizer_visibility == {state.id: False}

    monkeypatch.setattr(renderer._o3d_vis, "show_geometry", original_show)

    assert renderer._flush_visualizer_visibility_updates() is True
    assert renderer._pending_o3d_visualizer_visibility == {}
    assert renderer._o3d_vis.shown[-1] == (state.id, False)


def test_show_axes_requests_visibility_settle_redraw_after_success(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    immediate_redraws: list[bool] = []
    settle_redraws: list[str] = []
    monkeypatch.setattr(renderer, "_post_redraw", lambda: immediate_redraws.append(True))
    monkeypatch.setattr(
        renderer,
        "_request_visibility_settle_redraw",
        lambda reason: settle_redraws.append(reason),
    )

    assert renderer.show_axes(False) is True
    assert renderer._o3d_vis.show_axes is False
    assert immediate_redraws == [True]
    assert settle_redraws == ["coordinate axes visibility"]


def test_failed_show_axes_does_not_request_visibility_settle_redraw(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingAxesVisualizer(_DummyO3DVisualizer):
        @property
        def show_axes(self) -> bool:
            return True

        @show_axes.setter
        def show_axes(self, _show: bool) -> None:
            raise RuntimeError("native axes visibility failed")

    renderer._o3d_vis = _FailingAxesVisualizer()
    immediate_redraws: list[bool] = []
    settle_redraws: list[str] = []
    monkeypatch.setattr(renderer, "_post_redraw", lambda: immediate_redraws.append(True))
    monkeypatch.setattr(
        renderer,
        "_request_visibility_settle_redraw",
        lambda reason: settle_redraws.append(reason),
    )

    assert renderer.show_axes(False) is False
    assert immediate_redraws == []
    assert settle_redraws == []


def test_ensure_object_revision_replaces_once_and_restores_state(
    renderer: Open3DRenderer,
) -> None:
    state = _make_marker_handle()
    state.visible = False
    state.world_transform = Transform.from_translation([1.0, 2.0, 3.0])
    assert renderer.ensure_object(state.to_render_object()) is True
    add_count = len(renderer._o3d_vis.added)
    remove_count = len(renderer._o3d_vis.removed)

    replacement = MeshPayload(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=float,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    state.replace_payload(replacement)

    assert renderer.ensure_object(state.to_render_object()) is True
    assert len(renderer._o3d_vis.added) == add_count + 1
    assert len(renderer._o3d_vis.removed) == remove_count + 1
    np.testing.assert_allclose(
        renderer._o3d_vis.scene.transforms[-1][1],
        state.world_transform.matrix,
    )
    assert renderer._o3d_vis.shown[-1] == (state.id, False)

    assert renderer.ensure_object(state.to_render_object()) is True
    assert len(renderer._o3d_vis.added) == add_count + 1
    assert len(renderer._o3d_vis.removed) == remove_count + 1


def test_hidden_replacement_add_failure_rolls_back_and_retries(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_marker_handle(kind="rx")
    state.visible = False
    assert renderer.ensure_object(state.to_render_object()) is True
    previous = renderer._applied_render_objects[state.id]
    replacement = MeshPayload(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
            dtype=float,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    state.replace_payload(replacement)

    original_add = renderer._o3d_vis.add_geometry
    fail_next_add = True

    def add_geometry(name: str, geometry, material) -> None:
        nonlocal fail_next_add
        if fail_next_add:
            fail_next_add = False
            raise RuntimeError("native add failed")
        original_add(name, geometry, material)

    monkeypatch.setattr(renderer._o3d_vis, "add_geometry", add_geometry)

    assert renderer.ensure_object(state.to_render_object()) is False
    assert renderer._applied_render_objects[state.id] is previous
    assert renderer.has_named_geometry(state.id) is True
    assert renderer.is_named_visible(state.id) is False
    assert state.id not in renderer._pending_hidden_geometry_names

    assert renderer.ensure_object(state.to_render_object()) is True
    assert renderer._applied_render_objects[state.id].payload is replacement
    assert renderer.is_named_visible(state.id) is False


def test_hidden_replacement_show_failure_rolls_back_and_retries(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_marker_handle(kind="rx")
    state.visible = False
    assert renderer.ensure_object(state.to_render_object()) is True
    previous = renderer._applied_render_objects[state.id]
    replacement = MeshPayload(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
            dtype=float,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    state.replace_payload(replacement)

    original_show = renderer._o3d_vis.scene.show_geometry
    fail_next_hide = True

    def show_geometry(name: str, visible: bool) -> None:
        nonlocal fail_next_hide
        if not visible and fail_next_hide:
            fail_next_hide = False
            raise RuntimeError("native hide failed")
        original_show(name, visible)

    monkeypatch.setattr(renderer._o3d_vis.scene, "show_geometry", show_geometry)

    assert renderer.ensure_object(state.to_render_object()) is False
    assert renderer._applied_render_objects[state.id] is previous
    assert renderer.has_named_geometry(state.id) is True
    assert renderer.is_named_visible(state.id) is False
    assert state.id not in renderer._pending_hidden_geometry_names

    assert renderer.ensure_object(state.to_render_object()) is True
    assert renderer._applied_render_objects[state.id].payload is replacement
    assert renderer.is_named_visible(state.id) is False


def test_declarative_replacement_can_make_hidden_object_visible(
    renderer: Open3DRenderer,
) -> None:
    state = _make_marker_handle(kind="rx")
    state.visible = False
    assert renderer.ensure_object(state.to_render_object()) is True
    assert renderer.is_named_visible(state.id) is False

    state.replace_payload(
        MeshPayload(
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
                dtype=float,
            ),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        )
    )
    state.visible = True

    assert renderer.ensure_object(state.to_render_object()) is True
    assert renderer.is_named_visible(state.id) is True
    assert renderer._o3d_vis.shown[-1] == (state.id, True)
    assert renderer._applied_render_objects[state.id].visible is True


def test_ensure_object_material_update_does_not_replace_geometry(
    renderer: Open3DRenderer,
) -> None:
    state = _make_marker_handle()
    assert renderer.ensure_object(state.to_render_object()) is True
    add_count = len(renderer._o3d_vis.added)
    remove_count = len(renderer._o3d_vis.removed)
    previous_material = renderer._o3d_vis.materials[state.id]
    state.material = MaterialPayload(
        base_color=(0.2, 0.4, 0.6, 1.0),
        roughness=0.25,
    )

    assert renderer.ensure_object(state.to_render_object()) is True

    assert len(renderer._o3d_vis.added) == add_count
    assert len(renderer._o3d_vis.removed) == remove_count
    assert renderer._o3d_vis.materials[state.id] is not previous_material


def test_color_multiplier_is_material_only_and_preserves_vertex_payload(
    renderer: Open3DRenderer,
) -> None:
    state = _make_marker_handle()
    colors = np.asarray(
        [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.4, 0.2]],
        dtype=float,
    )
    state.replace_payload(
        replace(
            state.payload,
            vertex_colors=colors,
            color_source=SurfaceColorSource.VERTEX,
        )
    )
    assert renderer.ensure_object(state.to_render_object()) is True
    add_count = len(renderer._o3d_vis.added)
    remove_count = len(renderer._o3d_vis.removed)
    native_colors = np.asarray(renderer._o3d_vis.geometries[state.id].vertex_colors).copy()

    state.material = replace(
        state.material,
        color_multiplier=(1.0, 0.3, 0.3),
    )
    assert renderer.ensure_object(state.to_render_object()) is True

    assert len(renderer._o3d_vis.added) == add_count
    assert len(renderer._o3d_vis.removed) == remove_count
    np.testing.assert_array_equal(
        np.asarray(renderer._o3d_vis.geometries[state.id].vertex_colors),
        native_colors,
    )
    assert list(renderer._o3d_vis.materials[state.id].base_color) == pytest.approx(
        [1.0, 0.3, 0.3, 1.0]
    )


def test_material_only_update_requests_scene_settle_redraw(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_marker_handle()
    assert renderer.ensure_object(state.to_render_object()) is True
    redraws: list[bool] = []
    settle_redraws: list[str] = []
    monkeypatch.setattr(renderer, "_post_redraw", lambda: redraws.append(True) or True)
    monkeypatch.setattr(
        renderer,
        "_request_visibility_settle_redraw",
        lambda reason: settle_redraws.append(reason),
    )

    state.material = replace(state.material, color_multiplier=(1.0, 0.3, 0.3))
    assert renderer.ensure_object(state.to_render_object()) is True

    assert redraws == [True]
    assert settle_redraws == [f"PBR material '{state.id}'"]


def test_frame_time_persistent_mutations_do_not_add_scene_settle_redraw(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_marker_handle()
    assert renderer.ensure_object(state.to_render_object()) is True
    settle_redraws: list[str] = []
    monkeypatch.setattr(
        renderer,
        "_request_visibility_settle_redraw",
        lambda reason: settle_redraws.append(reason),
    )

    renderer._frame_update_in_progress = True
    try:
        state.material = replace(state.material, color_multiplier=(1.0, 0.3, 0.3))
        assert renderer.ensure_object(state.to_render_object()) is True
        state.replace_payload(
            replace(
                state.payload,
                vertices=np.asarray(state.payload.vertices, dtype=float) + 1.0,
            )
        )
        assert renderer.ensure_object(state.to_render_object()) is True
        assert renderer.remove_object(state.id) is True
    finally:
        renderer._frame_update_in_progress = False

    assert settle_redraws == []


def test_textured_multiplier_reuses_cached_albedo_without_geometry_upload(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    texture_path = tmp_path / "albedo.png"
    from PIL import Image as PILImage

    PILImage.fromarray(np.full((2, 2, 4), 255, dtype=np.uint8), mode="RGBA").save(texture_path)
    state = _make_marker_handle()
    state.material = MaterialPayload(texture_path=str(texture_path))
    assert renderer.ensure_object(state.to_render_object()) is True
    cached_image = renderer._o3d_vis.materials[state.id].albedo_img
    assert cached_image is not None
    add_count = len(renderer._o3d_vis.added)
    remove_count = len(renderer._o3d_vis.removed)

    monkeypatch.setattr(
        PILImage,
        "open",
        lambda *_args, **_kwargs: pytest.fail("cached highlight must not decode the texture"),
    )

    state.material = replace(state.material, color_multiplier=(1.0, 0.3, 0.3))
    assert renderer.ensure_object(state.to_render_object()) is True

    assert len(renderer._o3d_vis.added) == add_count
    assert len(renderer._o3d_vis.removed) == remove_count
    assert renderer._o3d_vis.materials[state.id].albedo_img is cached_image
    assert list(renderer._o3d_vis.materials[state.id].base_color) == pytest.approx(
        [1.0, 0.3, 0.3, 1.0]
    )


def test_dict_material_update_keeps_applied_cache_coherent(
    renderer: Open3DRenderer,
) -> None:
    state = _make_marker_handle()
    original = state.material
    assert renderer.ensure_object(state.to_render_object()) is True
    updated = {"base_color": [0.8, 0.3, 0.1, 0.7], "roughness": 0.2}

    assert renderer.set_material(state.id, updated) is True
    applied = renderer._applied_render_objects[state.id]
    assert isinstance(applied.material, MaterialPayload)
    assert applied.material.base_color == pytest.approx((0.8, 0.3, 0.1, 0.7))

    assert renderer.ensure_object(state.to_render_object()) is True
    assert renderer._applied_render_objects[state.id].material == original
    assert list(renderer._o3d_vis.materials[state.id].base_color) == pytest.approx(
        original.base_color
    )


def test_culling_failure_is_not_cached_and_retries(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_marker_handle()
    filament_scene = renderer._o3d_vis.scene.scene
    original_set_culling = filament_scene.set_geometry_culling
    failures_remaining = 1

    def flaky_set_culling(name: str, enabled: bool) -> None:
        nonlocal failures_remaining
        if failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("native culling failure")
        original_set_culling(name, enabled)

    monkeypatch.setattr(filament_scene, "set_geometry_culling", flaky_set_culling)

    assert renderer.ensure_object(state.to_render_object()) is False
    assert state.id not in renderer._applied_render_objects
    assert renderer.has_named_geometry(state.id) is False

    assert renderer.ensure_object(state.to_render_object()) is True
    assert renderer._applied_render_objects[state.id].payload is state.payload
    assert filament_scene.culling_calls[-1] == (state.id, renderer._culling_enabled)


def test_failed_component_update_is_not_recorded(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_marker_handle()
    assert renderer.ensure_object(state.to_render_object()) is True
    previous = renderer._applied_render_objects[state.id]
    state.world_transform = Transform.from_translation([9.0, 8.0, 7.0])
    monkeypatch.setattr(renderer, "set_named_transform", lambda *_args, **_kwargs: False)

    assert renderer.ensure_object(state.to_render_object()) is False
    assert renderer._applied_render_objects[state.id] is previous


def test_failed_replacement_restores_previous_applied_snapshot(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_marker_handle()
    assert renderer.ensure_object(state.to_render_object()) is True
    previous = renderer._applied_render_objects[state.id]
    state.replace_payload(
        MeshPayload(
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
                dtype=float,
            ),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        )
    )
    monkeypatch.setattr(renderer, "_apply_full_render_object", lambda _obj: False)
    restored: list[str] = []
    monkeypatch.setattr(
        renderer,
        "_restore_applied_render_object",
        lambda object_id, _state: restored.append(object_id) or True,
    )

    assert renderer.ensure_object(state.to_render_object()) is False
    assert restored == [state.id]
    assert renderer._applied_render_objects[state.id] is previous


def test_remove_object_is_idempotent_and_clears_applied_state(
    renderer: Open3DRenderer,
) -> None:
    state = _make_marker_handle()
    assert renderer.ensure_object(state.to_render_object()) is True

    assert renderer.remove_object(state.id) is True
    assert state.id not in renderer._applied_render_objects
    assert renderer.remove_object(state.id) is True


def test_persistent_create_and_remove_request_scene_settle_redraw(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redraws: list[bool] = []
    settle_redraws: list[str] = []
    monkeypatch.setattr(renderer, "_post_redraw", lambda: redraws.append(True) or True)
    monkeypatch.setattr(
        renderer,
        "_request_visibility_settle_redraw",
        lambda reason: settle_redraws.append(reason),
    )
    state = _make_marker_handle()

    assert renderer.ensure_object(state.to_render_object()) is True
    assert redraws
    assert settle_redraws[-1] == f"create or replace persistent geometry '{state.id}'"

    redraws.clear()
    settle_redraws.clear()
    assert renderer.remove_object(state.id) is True

    assert redraws == [True]
    assert settle_redraws == [f"remove persistent geometry '{state.id}'"]


def test_failed_remove_preserves_applied_state(
    renderer: Open3DRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_marker_handle()
    assert renderer.ensure_object(state.to_render_object()) is True

    def fail_remove(_name: str) -> None:
        raise RuntimeError("native remove failed")

    monkeypatch.setattr(renderer._o3d_vis, "remove_geometry", fail_remove)

    assert renderer.remove_object(state.id) is False
    assert state.id in renderer._applied_render_objects
    assert state.id in renderer._geometry_names


def test_text_label_uses_idempotent_render_object_contract(
    renderer: Open3DRenderer,
) -> None:
    label_id = "target:car::label"
    label = make_text_label_state(
        label_id,
        "Car",
        [0.2, 0.4, 0.8],
        position=[1.5, 2.0, 4.0],
    )

    assert renderer.ensure_object(label.to_render_object())
    add_count = len(renderer._o3d_vis.added)
    remove_count = len(renderer._o3d_vis.removed)
    np.testing.assert_allclose(renderer.get_named_position(label_id), [1.5, 2.0, 4.0])

    assert renderer.ensure_object(label.to_render_object())
    assert len(renderer._o3d_vis.added) == add_count
    assert len(renderer._o3d_vis.removed) == remove_count

    label.visible = False
    assert renderer.ensure_object(label.to_render_object())
    assert renderer.is_named_visible(label_id) is False
    assert renderer.remove_object(label_id)
    assert renderer.has_named_geometry(label_id) is False


def test_text_label_material_matches_requested_color_and_unlit_cache(
    renderer: Open3DRenderer,
) -> None:
    label_id = "target:car::label"
    label = make_text_label_state(
        label_id,
        "Car",
        [0.2, 0.4, 0.8],
    )

    assert renderer.ensure_object(label.to_render_object()) is True
    native_material = renderer._o3d_vis.materials[label_id]
    assert native_material.shader == "defaultUnlit"
    assert list(native_material.base_color) == pytest.approx([0.2, 0.4, 0.8, 1.0])
    applied = renderer._applied_render_objects[label_id]
    assert applied.material == label.material
    assert applied.material.shader == "unlit"

    add_count = len(renderer._o3d_vis.added)
    remove_count = len(renderer._o3d_vis.removed)
    assert renderer.ensure_object(label.to_render_object()) is True
    assert renderer._o3d_vis.materials[label_id] is native_material
    assert len(renderer._o3d_vis.added) == add_count
    assert len(renderer._o3d_vis.removed) == remove_count


def test_text_label_material_update_stays_unlit_without_reupload(
    renderer: Open3DRenderer,
) -> None:
    label_id = "target:car::label"
    label = make_text_label_state(label_id, "Car", [0.2, 0.4, 0.8])
    assert renderer.ensure_object(label.to_render_object()) is True
    add_count = len(renderer._o3d_vis.added)
    remove_count = len(renderer._o3d_vis.removed)
    label.material = MaterialPayload(
        base_color=(0.8, 0.3, 0.1, 0.6),
        shader="lit",
    )

    assert renderer.ensure_object(label.to_render_object()) is True
    native_material = renderer._o3d_vis.materials[label_id]
    assert native_material.shader == "defaultUnlit"
    assert list(native_material.base_color) == pytest.approx([0.8, 0.3, 0.1, 0.6])
    applied = renderer._applied_render_objects[label_id]
    assert applied.material.base_color == label.material.base_color
    assert applied.material.shader == "unlit"
    assert len(renderer._o3d_vis.added) == add_count
    assert len(renderer._o3d_vis.removed) == remove_count

    assert renderer.ensure_object(label.to_render_object()) is True
    assert renderer._o3d_vis.materials[label_id] is native_material


def test_ensure_named_geometry_preserves_requested_vertex_colors(
    renderer: Open3DRenderer,
) -> None:
    vertex_colors = np.asarray(
        [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]],
        dtype=float,
    )
    payload = MeshPayload(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        vertex_colors=vertex_colors,
        color_source=SurfaceColorSource.VERTEX,
    )
    material = MaterialPayload(base_color=(0.3, 0.4, 0.5, 0.7))

    assert renderer.ensure_named_geometry(
        "target:colored::mesh",
        payload,
        material=material,
    )

    added = renderer._o3d_vis.geometries["target:colored::mesh"]
    np.testing.assert_allclose(np.asarray(added.vertex_colors), vertex_colors)
    assert list(renderer._o3d_vis.materials["target:colored::mesh"].base_color) == pytest.approx(
        [0.3, 0.4, 0.5, 0.7]
    )


def test_ensure_named_geometry_neutralizes_vertex_colors_by_default(
    renderer: Open3DRenderer,
) -> None:
    payload = MeshPayload(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        vertex_colors=np.asarray(
            [[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]],
            dtype=float,
        ),
    )

    assert renderer.ensure_named_geometry(
        "target:neutral::mesh",
        payload,
        material=MaterialPayload(base_color=(0.3, 0.4, 0.5, 1.0)),
    )

    added = renderer._o3d_vis.geometries["target:neutral::mesh"]
    np.testing.assert_allclose(np.asarray(added.vertex_colors), np.ones((3, 3)))
