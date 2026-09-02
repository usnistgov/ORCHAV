"""pygfx renderer specific tests.

These tests focus on topology-stable updates and basic protocol operations.
They are skipped unless the pygfx runtime profile is explicitly enabled.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.model import RenderObject, RenderObjectState, Transform, make_text_label_state
from visualizer.src.pipeline.core import FrameRenderPacket, ViewModel
from visualizer.src.services.object_identity import (
    ensure_target_entry_identity,
    make_node_geometry_name,
    make_target_entry_geometry_name,
)
from visualizer.src.state import MpcVisibility
from visualizer.src.types.render_payloads import (
    MaterialPayload,
    MeshPayload,
    PointCloudPayload,
    SurfaceColorSource,
)


def _pygfx_available() -> bool:
    try:
        import pygfx  # noqa: F401
        from PySide6 import QtWidgets as _QtWidgets  # noqa: F401
        from rendercanvas.qt import RenderCanvas as _RenderCanvas  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.pygfx_runtime,
    pytest.mark.skipif(
        (not _pygfx_available()) or (os.getenv("ORCHAV_RUN_PYGFX_TESTS", "0") != "1"),
        reason="pygfx runtime tests disabled (set ORCHAV_RUN_PYGFX_TESTS=1) or runtime unavailable",
    ),
]


def _make_mesh(n_verts: int = 100) -> MeshPayload:
    verts = np.random.rand(n_verts, 3).astype(np.float32)
    n_tris = n_verts - 2
    tris = np.column_stack(
        [np.zeros(n_tris, dtype=np.int32), np.arange(1, n_tris + 1), np.arange(2, n_tris + 2)]
    ).astype(np.int32)
    colors = np.random.rand(n_verts, 3).astype(np.float32)
    return MeshPayload(vertices=verts, triangles=tris, vertex_colors=colors)


def _make_uv_mesh(n_verts: int = 20) -> MeshPayload:
    mesh = _make_mesh(n_verts)
    triangle_uvs = np.zeros((len(mesh.triangles) * 3, 2), dtype=np.float32)
    return MeshPayload(
        vertices=mesh.vertices,
        triangles=mesh.triangles,
        normals=mesh.normals,
        vertex_colors=mesh.vertex_colors,
        triangle_uvs=triangle_uvs,
        cache_key=mesh.cache_key,
    )


def _material_rgba(material) -> np.ndarray:
    return np.asarray(material.color, dtype=np.float32)


def _make_mpc_view_model(
    points: np.ndarray,
    lines: np.ndarray,
    *,
    colors: np.ndarray | None = None,
    point_colors: np.ndarray | None = None,
    show_mpc_bounce_points: bool = True,
    show_paths: bool = True,
) -> ViewModel:
    points_arr = np.array(points, dtype=np.float32, copy=True)
    lines_arr = np.array(lines, dtype=np.int32, copy=True)
    point_count = len(points)
    line_count = len(lines)
    return ViewModel(
        tx_positions=np.empty((0, 3), dtype=np.float32),
        rx_positions=np.empty((0, 3), dtype=np.float32),
        tx_orientations=np.empty((0, 3), dtype=np.float32),
        rx_orientations=np.empty((0, 3), dtype=np.float32),
        mpc_points=points_arr,
        mpc_lines=lines_arr,
        mpc_colors=(
            np.array(colors, dtype=np.float32, copy=True)
            if colors is not None
            else np.ones((line_count, 3), dtype=np.float32)
        ),
        colorbar=None,
        stats_text="",
        mpc_visibility=MpcVisibility(paths=show_paths, bounce_points=show_mpc_bounce_points),
        mpc_bounce_points=points_arr.copy(),
        mpc_bounce_colors=(
            np.array(point_colors, dtype=np.float32, copy=True)
            if point_colors is not None
            else np.ones((point_count, 3), dtype=np.float32)
        ),
        target_positions=np.empty((0, 3), dtype=np.float32),
        target_orientations=np.empty((0, 3), dtype=np.float32),
        target_mesh_files=[],
        target_use_ply_positions=[],
        target_metadata=[],
        show_coverage=False,
        beamforming_meshes=[],
    )


@pytest.fixture
def renderer(qapp):
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    r = PygfxRenderer(SimpleNamespace())
    try:
        r.initialize_visualizer(width=64, height=64)
    except Exception as exc:
        pytest.skip(f"pygfx unavailable in this environment: {exc}")
    yield r
    r.close()


class TestInPlaceVertexUpdate:
    def test_same_vertex_count_preserves_handle(self, renderer):
        mesh_v1 = _make_mesh(100)
        renderer.ensure_named_geometry("obj", mesh_v1)

        handle_before = renderer._get_entity_handle("obj")

        mesh_v2 = _make_mesh(100)
        renderer.ensure_named_geometry("obj", mesh_v2)

        handle_after = renderer._get_entity_handle("obj")
        assert handle_before == handle_after


class TestTopologyChangeRecreate:
    def test_different_vertex_count_changes_handle(self, renderer):
        mesh_small = _make_mesh(100)
        renderer.ensure_named_geometry("obj", mesh_small)
        handle_before = renderer._get_entity_handle("obj")

        mesh_large = _make_mesh(200)
        renderer.ensure_named_geometry("obj", mesh_large)
        handle_after = renderer._get_entity_handle("obj")
        assert handle_before != handle_after

    @pytest.mark.parametrize("attribute", ["normals", "vertex_colors", "triangle_uvs"])
    def test_optional_buffer_add_remove_recreates_entity(self, renderer, attribute):
        vertices = np.zeros((4, 3), dtype=np.float32)
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        plain = MeshPayload(vertices=vertices, triangles=triangles)
        attribute_values = {
            "normals": np.ones((4, 3), dtype=np.float32),
            "vertex_colors": np.ones((4, 3), dtype=np.float32),
            "triangle_uvs": np.zeros((4, 2), dtype=np.float32),
        }
        attributed = MeshPayload(
            vertices=vertices,
            triangles=triangles,
            color_source=(
                SurfaceColorSource.VERTEX
                if attribute == "vertex_colors"
                else SurfaceColorSource.MATERIAL
            ),
            **{attribute: attribute_values[attribute]},
        )

        assert renderer.ensure_named_geometry(
            "layout_obj",
            plain,
        )
        plain_handle = renderer._get_entity_handle("layout_obj")
        assert renderer.ensure_named_geometry(
            "layout_obj",
            attributed,
        )
        attributed_handle = renderer._get_entity_handle("layout_obj")
        assert attributed_handle != plain_handle

        assert renderer.ensure_named_geometry(
            "layout_obj",
            plain,
        )
        assert renderer._get_entity_handle("layout_obj") != attributed_handle


class TestGeometryUpdateFailure:
    def test_failed_same_layout_update_is_not_cached_as_applied(self, renderer, monkeypatch):
        first = RenderObject(id="retryable_obj", payload=_make_mesh(20))
        desired = RenderObject(id="retryable_obj", payload=_make_mesh(20))

        assert renderer.ensure_object(first) is True
        installed_native_geometry = renderer._objects[first.id].geometry
        installed_center = renderer._geometry_upload_center[first.id].copy()

        def fail_push(*_args, **_kwargs):
            raise RuntimeError("simulated native buffer failure")

        def fail_rebuild(*_args, **_kwargs):
            raise RuntimeError("simulated fallback failure")

        monkeypatch.setattr(renderer, "_push_buffer", fail_push)
        monkeypatch.setattr(renderer, "_build_mesh_geometry", fail_rebuild)

        assert renderer.ensure_object(desired) is False
        assert renderer._render_object_snapshots[first.id][0] is first.payload
        assert renderer._objects[first.id].geometry is installed_native_geometry
        np.testing.assert_array_equal(
            renderer._geometry_upload_center[first.id],
            installed_center,
        )

    def test_partial_same_layout_update_is_repaired_when_desired_reverts(
        self, renderer, monkeypatch
    ):
        first_payload = _make_mesh(20)
        first_payload = replace(
            first_payload,
            normals=np.zeros_like(first_payload.vertices),
        )
        desired_payload = _make_mesh(20)
        desired_payload = replace(
            desired_payload,
            normals=np.ones_like(desired_payload.vertices),
        )
        first = RenderObject(id="retryable_partial_obj", payload=first_payload)
        desired = RenderObject(id=first.id, payload=desired_payload)
        assert renderer.ensure_object(first) is True

        original_push = renderer._push_buffer
        original_fallback = renderer._update_in_place_fallback

        def fail_after_positions(buf, data, *, label="buffer"):
            if label == "mesh_normals":
                raise RuntimeError("simulated failure after position upload")
            return original_push(buf, data, label=label)

        monkeypatch.setattr(renderer, "_push_buffer", fail_after_positions)
        monkeypatch.setattr(
            renderer,
            "_update_in_place_fallback",
            lambda *_args, **_kwargs: False,
        )

        assert renderer.ensure_object(desired) is False
        native_positions = renderer._objects[first.id].geometry.positions.data
        np.testing.assert_array_equal(native_positions, desired.payload.vertices)
        assert first.id in renderer._dirty_render_object_geometry

        monkeypatch.setattr(renderer, "_push_buffer", original_push)
        monkeypatch.setattr(renderer, "_update_in_place_fallback", original_fallback)

        assert renderer.ensure_object(first) is True
        np.testing.assert_array_equal(
            renderer._objects[first.id].geometry.positions.data,
            first.payload.vertices,
        )
        assert first.id not in renderer._dirty_render_object_geometry


class TestTransformAndPosition:
    def test_ensure_object_accepts_render_object_state_snapshot(self, renderer):
        mesh = _make_mesh(30)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = [7.0, 8.0, 9.0]
        state = RenderObjectState(
            id="target:pedestrian::mesh",
            payload=mesh,
            material=MaterialPayload(base_color=(0.2, 0.3, 0.4, 1.0)),
            world_transform=Transform(transform),
            visible=False,
        )

        assert renderer.ensure_object(state.to_render_object()) is True

        assert renderer.has_named_geometry(state.id) is True
        assert renderer.is_named_visible(state.id) is False
        np.testing.assert_allclose(
            renderer.get_named_position(state.id),
            [7.0, 8.0, 9.0],
            atol=1e-6,
        )

    def test_ensure_object_skips_unchanged_payload_upload(self, renderer, monkeypatch):
        state = RenderObjectState(id="scene:stable::mesh", payload=_make_mesh(30))
        assert renderer.ensure_object(state.to_render_object()) is True
        updates: list[str] = []
        original = renderer._update_in_place

        def record_update(name, payload, **kwargs):
            updates.append(name)
            return original(name, payload, **kwargs)

        monkeypatch.setattr(renderer, "_update_in_place", record_update)

        assert renderer.ensure_object(state.to_render_object()) is True
        assert updates == []

        state.replace_payload(_make_mesh(30))
        assert renderer.ensure_object(state.to_render_object()) is True
        assert updates == [state.id]

    def test_remove_object_is_idempotent(self, renderer):
        state = RenderObjectState(id="scene:remove::mesh", payload=_make_mesh(30))
        assert renderer.ensure_object(state.to_render_object()) is True

        assert renderer.remove_object(state.id) is True
        assert renderer.remove_object(state.id) is True

    def test_set_transform_updates_named_position(self, renderer):
        mesh = _make_mesh(50)
        renderer.ensure_named_geometry("tr_obj", mesh)

        t = np.eye(4, dtype=np.float64)
        t[:3, 3] = [7.0, 8.0, 9.0]

        assert renderer.set_named_transform("tr_obj", t) is True

        pos = renderer.get_named_position("tr_obj")
        assert pos is not None
        np.testing.assert_allclose(pos, [7.0, 8.0, 9.0], atol=1e-6)

    def test_node_transform_gizmo_emits_selection_change_and_commit_phases(
        self, renderer, monkeypatch
    ):
        """Drive the pygfx gizmo bridge through the same phases as a node drag."""

        class FakeTransformGizmo:
            def __init__(self, *args, **kwargs):
                self.object = None
                self.visible = True
                self.processed_events = []

            def set_object(self, obj):
                self.object = obj

            def toggle_mode(self, mode):
                self.mode = mode

            def add_event_handler(self, callback, *event_types):
                self.event_handler = (callback, event_types)

            def process_event(self, event):
                self.processed_events.append(getattr(event, "type", None))

        class FakeScene:
            def __init__(self):
                self.objects = []

            def add(self, obj):
                self.objects.append(obj)

            def remove(self, obj):
                self.objects.remove(obj)

        renderer._dispose_transform_gizmo()
        monkeypatch.setattr(renderer._gfx, "TransformGizmo", FakeTransformGizmo, raising=False)

        marker_name = make_node_geometry_name("tx", 0, "marker")
        renderer.ensure_named_geometry(marker_name, _make_mesh(12))
        renderer._scene = FakeScene()
        marker_obj = renderer._objects[marker_name]
        events = []
        stops = []

        assert renderer.begin_live_preview_transform_session(events.append)
        renderer.pygfx_interaction_router()._route_event(
            SimpleNamespace(
                type="pointer_down",
                button=1,
                modifiers=(),
                target=marker_obj,
                stop_propagation=lambda: stops.append(True),
            )
        )

        marker_obj.local.position = (1.0, 2.0, 3.0)
        renderer._transform_gizmo.process_event(SimpleNamespace(type="pointer_move"))
        marker_obj.local.position = (4.0, 5.0, 6.0)
        renderer._transform_gizmo.process_event(SimpleNamespace(type="pointer_up"))

        assert stops == [True]
        assert renderer._transform_gizmo.object is marker_obj
        assert [event["phase"] for event in events] == ["selected", "changed", "committed"]
        assert events[0]["object_id"] == marker_name
        assert events[0]["kind"] == "tx"
        assert events[0]["index"] == 0
        assert events[1]["position"] == pytest.approx((1.0, 2.0, 3.0))
        assert events[2]["position"] == pytest.approx((4.0, 5.0, 6.0))

    def test_transform_gizmo_can_select_target_mesh(self, renderer, monkeypatch):
        """Target meshes are edited through their semantic-center proxy."""

        class FakeTransformGizmo:
            def __init__(self, *args, **kwargs):
                self.object = None
                self.visible = True

            def set_object(self, obj):
                self.object = obj

            def toggle_mode(self, mode):
                self.mode = mode

            def add_event_handler(self, callback, *event_types):
                self.event_handler = (callback, event_types)

            def process_event(self, event):
                pass

        class FakeScene:
            def __init__(self):
                self.objects = []

            def add(self, obj):
                self.objects.append(obj)

            def remove(self, obj):
                self.objects.remove(obj)

        renderer._dispose_transform_gizmo()
        monkeypatch.setattr(renderer._gfx, "TransformGizmo", FakeTransformGizmo, raising=False)

        target_entry = {"target_name": "Walker"}
        ensure_target_entry_identity(target_entry, 0)
        renderer.visualizer.target_entries = [target_entry]
        mesh_name = make_target_entry_geometry_name(target_entry, "mesh")
        renderer.ensure_named_geometry(mesh_name, _make_mesh(12))
        renderer._scene = FakeScene()
        target_obj = renderer._objects[mesh_name]
        events = []

        assert renderer.begin_live_preview_transform_session(events.append)
        renderer.pygfx_interaction_router()._route_event(
            SimpleNamespace(
                type="pointer_down",
                button=1,
                modifiers=(),
                target=target_obj,
                stop_propagation=lambda: None,
            )
        )

        target_proxy = renderer._transform_gizmo_target_proxy
        assert target_proxy is not None
        assert renderer._transform_gizmo.object is target_proxy
        assert target_proxy is not target_obj
        assert events[0]["object_id"] == mesh_name
        assert events[0]["kind"] == "target"
        assert events[0]["index"] == 0
        assert "transform" in events[0]

    def test_text_label_render_object_applies_layout_metadata(self, renderer):
        label = make_text_label_state("label_obj", "TX1", [1.0, 0.0, 0.0])
        label.metadata["layout_anchor"] = (1.0, 2.0, 3.0)
        label.metadata["layout_offset"] = (0.5, 0.0, 1.0)

        assert renderer.ensure_object(label.to_render_object()) is True
        pos = renderer.get_named_position(label.id)
        assert pos is not None
        expected = np.asarray([1.5, 2.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(pos, expected, atol=1e-6)

    def test_set_named_transform_is_noop_when_transform_is_unchanged(self, renderer):
        mesh = _make_mesh(18)
        renderer.ensure_named_geometry("stable_obj", mesh)
        t = np.eye(4, dtype=np.float32)
        t[:3, 3] = [3.0, 4.0, 5.0]

        assert renderer.set_named_transform("stable_obj", t) is True
        original_matrix = renderer._objects["stable_obj"].local.matrix
        assert renderer.set_named_transform("stable_obj", t.copy()) is True
        assert renderer._objects["stable_obj"].local.matrix is original_matrix

    def test_text_label_repeated_ensure_preserves_renderer_transform(self, renderer):
        name = make_node_geometry_name("tx", 0, "label")
        label = make_text_label_state(name, "TX1", [1.0, 0.0, 0.0])
        label.metadata["layout_anchor"] = (4.0, 5.0, 6.0)
        label.metadata["layout_offset"] = (1.0, 0.0, 0.5)

        assert renderer.ensure_object(label.to_render_object()) is True
        native = renderer._objects[name]
        assert renderer.ensure_object(label.to_render_object()) is True
        assert renderer._objects[name] is native

        pos = renderer.get_named_position(name)
        assert pos is not None
        np.testing.assert_allclose(pos, [5.0, 5.0, 6.5], atol=1e-6)

    def test_ensure_object_accepts_text_label(self, renderer):
        name = make_node_geometry_name("tx", 0, "label")
        label = make_text_label_state(name, "TX1", [1.0, 0.0, 0.0])

        assert renderer.ensure_object(label.to_render_object()) is True
        assert renderer.has_named_geometry(name) is True
        assert renderer._kinds[name] == "text"
        assert renderer.is_named_visible(name) is True

        label.visible = False
        assert renderer.ensure_object(label.to_render_object()) is True
        assert renderer.is_named_visible(name) is False

    def test_label_uses_common_visibility_and_removal_contract(self, renderer):
        name = make_node_geometry_name("rx", 0, "label")
        label = make_text_label_state(
            name,
            "RX1",
            [0.0, 0.0, 1.0],
            position=[1.5, 2.0, 4.0],
        )

        assert renderer.ensure_object(label.to_render_object())
        assert renderer.is_named_visible(name) is True
        assert renderer.set_visible(name, False)
        assert renderer.is_named_visible(name) is False
        assert renderer.remove_object(name)
        assert renderer.has_named_geometry(name) is False
        assert renderer.is_named_visible(name) is None


class TestVisibilityAndMaterial:
    def test_visibility_roundtrip(self, renderer):
        mesh = _make_mesh(20)
        renderer.ensure_named_geometry("v_obj", mesh)
        calls = []
        orig_request_redraw = renderer.request_redraw

        def _track_redraw():
            calls.append(1)
            orig_request_redraw()

        renderer.request_redraw = _track_redraw

        assert renderer.set_named_visibility("v_obj", False) is True
        assert renderer.is_named_visible("v_obj") is False

        assert renderer.set_named_visibility("v_obj", True) is True
        assert renderer.is_named_visible("v_obj") is True
        assert len(calls) >= 2

    def test_visibility_noop_does_not_request_extra_redraw(self, renderer):
        mesh = _make_mesh(20)
        renderer.ensure_named_geometry("v_obj_noop", mesh)
        calls = []
        orig_request_redraw = renderer.request_redraw

        def _track_redraw():
            calls.append(1)
            orig_request_redraw()

        renderer.request_redraw = _track_redraw

        assert renderer.set_named_visibility("v_obj_noop", True) is True
        assert calls == []

    def test_material_update(self, renderer):
        mesh = _make_mesh(20)
        renderer.ensure_named_geometry("m_obj", mesh)
        calls = []
        orig_request_redraw = renderer.request_redraw

        def _track_redraw():
            calls.append(1)
            orig_request_redraw()

        renderer.request_redraw = _track_redraw

        ok = renderer.set_named_material(
            "m_obj",
            {"base_color": [0.2, 0.4, 0.8, 0.7], "roughness": 0.3, "metallic": 0.1},
        )
        assert ok is True
        assert calls == [1]
        assert renderer._geometry_color_sources["m_obj"] is SurfaceColorSource.MATERIAL
        mat = renderer._objects["m_obj"].material
        assert getattr(mat, "color_mode", None) == "uniform"
        np.testing.assert_allclose(_material_rgba(mat), [0.2, 0.4, 0.8, 0.7], atol=1e-6)

    def test_material_update_noop_does_not_request_extra_redraw(self, renderer):
        mesh = _make_mesh(20)
        renderer.ensure_named_geometry("m_obj_noop", mesh)
        calls = []
        orig_request_redraw = renderer.request_redraw

        def _track_redraw():
            calls.append(1)
            orig_request_redraw()

        renderer.request_redraw = _track_redraw
        material = {"base_color": [0.2, 0.4, 0.8, 0.7], "roughness": 0.3, "metallic": 0.1}

        assert renderer.set_named_material("m_obj_noop", material) is True
        assert calls == [1]

        assert renderer.set_named_material("m_obj_noop", material) is True
        assert calls == [1]

    def test_material_update_replays_when_vertex_color_context_changes(self, renderer):
        name = "target:pedestrian::mesh"
        material = MaterialPayload(base_color=(1.0, 1.0, 1.0, 1.0))

        assert (
            renderer.ensure_named_geometry(
                name,
                replace(_make_mesh(20), color_source=SurfaceColorSource.VERTEX),
                material=material,
            )
            is True
        )
        assert renderer._geometry_color_sources[name] is SurfaceColorSource.VERTEX
        assert getattr(renderer._objects[name].material, "color_mode", None) == "vertex"

        assert (
            renderer.ensure_named_geometry(
                name,
                replace(_make_mesh(20), color_source=SurfaceColorSource.MATERIAL),
                material=material,
            )
            is True
        )
        assert renderer._geometry_color_sources[name] is SurfaceColorSource.MATERIAL
        assert getattr(renderer._objects[name].material, "color_mode", None) == "uniform"

    def test_material_update_can_preserve_true_vertex_colors(self, renderer):
        mesh = replace(_make_mesh(20), color_source=SurfaceColorSource.VERTEX)

        assert (
            renderer.ensure_named_geometry(
                "textured_obj",
                mesh,
                material=MaterialPayload(base_color=(1.0, 1.0, 1.0, 1.0)),
            )
            is True
        )

        assert renderer._geometry_color_sources["textured_obj"] is SurfaceColorSource.VERTEX
        mat = renderer._objects["textured_obj"].material
        assert getattr(mat, "color_mode", None) == "vertex"

    def test_ensure_object_preserves_explicit_vertex_color_source(self, renderer):
        mesh = replace(_make_mesh(20), color_source=SurfaceColorSource.VERTEX)
        obj = RenderObject(
            id="target:pedestrian::mesh",
            payload=mesh,
            material=MaterialPayload(base_color=(1.0, 1.0, 1.0, 1.0)),
        )

        assert renderer.ensure_object(obj) is True

        assert renderer._geometry_color_sources[obj.id] is SurfaceColorSource.VERTEX
        mat = renderer._objects[obj.id].material
        assert getattr(mat, "color_mode", None) == "vertex"

    def test_topology_recreate_preserves_vertex_color_material_mode(self, renderer):
        name = "target:pedestrian::mesh"
        first_mesh = replace(_make_mesh(20), color_source=SurfaceColorSource.VERTEX)
        next_mesh = replace(_make_mesh(23), color_source=SurfaceColorSource.VERTEX)

        assert (
            renderer.ensure_named_geometry(
                name,
                first_mesh,
                material=MaterialPayload(base_color=(1.0, 1.0, 1.0, 1.0)),
            )
            is True
        )
        assert (
            renderer.ensure_named_geometry(
                name,
                next_mesh,
            )
            is True
        )

        obj = renderer._objects[name]
        colors = getattr(obj.geometry, "colors", None)
        assert colors is not None
        assert getattr(colors, "data").shape == next_mesh.vertex_colors.shape
        assert renderer._geometry_color_sources[name] is SurfaceColorSource.VERTEX
        assert getattr(obj.material, "color_mode", None) == "vertex"

    def test_material_update_clears_stale_emissive_color(self, renderer):
        mesh = _make_mesh(20)
        renderer.ensure_named_geometry("glow_obj", mesh)

        assert (
            renderer.set_named_material(
                "glow_obj",
                {
                    "base_color": [0.2, 0.4, 0.8, 1.0],
                    "emissive_color": (1.0, 0.4, 0.0),
                    "emissive_intensity": 2.0,
                },
            )
            is True
        )
        assert (
            renderer.set_named_material(
                "glow_obj",
                {
                    "base_color": [0.2, 0.4, 0.8, 1.0],
                    "emissive_color": (0.0, 0.0, 0.0),
                    "emissive_intensity": 2.0,
                },
            )
            is True
        )

        mat = renderer._objects["glow_obj"].material
        assert mat.emissive.r == pytest.approx(0.0, abs=1e-5)
        assert mat.emissive.g == pytest.approx(0.0, abs=1e-5)
        assert mat.emissive.b == pytest.approx(0.0, abs=1e-5)
        assert mat.emissive_intensity == pytest.approx(2.0, abs=1e-5)

    def test_material_update_rgb_compat(self, renderer):
        mesh = _make_mesh(20)
        renderer.ensure_named_geometry("m_obj_rgb", mesh)
        ok = renderer.modify_geometry_material_pbr(
            "m_obj_rgb",
            color=[0.3, 0.5, 0.7],
            alpha=0.35,
            roughness=0.2,
            metallic=0.0,
        )
        assert ok is True

    def test_material_payload_rgb_triplet_is_normalized(self, renderer):
        mesh = _make_mesh(20)
        renderer.ensure_named_geometry("m_obj_payload", mesh)
        payload = MaterialPayload(base_color=(0.1, 0.2, 0.3))  # session-compatible
        ok = renderer.set_named_material("m_obj_payload", payload)
        assert ok is True

    def test_active_albedo_forces_white_base_color(self, renderer, monkeypatch):
        monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
        monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
        mesh = _make_uv_mesh(20)
        renderer.ensure_named_geometry("flat_texture_obj", mesh)

        ok = renderer.modify_geometry_material_pbr(
            "flat_texture_obj",
            color=[0.2, 0.3, 0.4],
            texture_path="libraries/textures/gold.png",
        )

        assert ok is True
        mat = renderer._materials["flat_texture_obj"]
        assert mat.base_color == pytest.approx((1.0, 1.0, 1.0, 1.0))
        assert Path(mat.texture_path) == Path("libraries/textures/gold.png")

    def test_detail_only_texture_maps_keep_base_color(self, renderer, monkeypatch):
        monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
        monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
        mesh = _make_uv_mesh(20)
        renderer.ensure_named_geometry("detail_only_obj", mesh)

        ok = renderer.modify_geometry_material_pbr(
            "detail_only_obj",
            color=[0.2, 0.3, 0.4],
            normal_map_path="libraries/textures/pbr/brick/normal.png",
        )

        assert ok is True
        mat = renderer._materials["detail_only_obj"]
        assert mat.base_color == pytest.approx((0.2, 0.3, 0.4, 1.0))
        assert mat.texture_path is None
        assert Path(mat.normal_map_path) == Path("libraries/textures/pbr/brick/normal.png")

    def test_colored_point_material_update_preserves_vertex_colors(self, renderer):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        colors = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        payload = PointCloudPayload(points=points, colors=colors)
        renderer.ensure_named_geometry("sensing_detections", payload)

        ok = renderer.set_named_material(
            "sensing_detections",
            MaterialPayload(base_color=(1.0, 1.0, 1.0, 1.0), point_size=12.0),
        )

        assert ok is True
        mat = renderer._objects["sensing_detections"].material
        assert getattr(mat, "color_mode", None) == "vertex"

    def test_transparent_mesh_uses_weighted_blend_without_depth_write(self):
        from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

        renderer = PygfxRenderer(SimpleNamespace())
        mesh_name = "scene:wall_0::mesh"
        renderer._objects[mesh_name] = SimpleNamespace(material=renderer._build_mesh_material())
        renderer._kinds[mesh_name] = "mesh"

        ok = renderer.set_named_material(
            mesh_name,
            {"base_color": [0.2, 0.4, 0.8, 0.2], "roughness": 0.3, "metallic": 0.1},
        )

        assert ok is True
        obj_mat = renderer._objects[mesh_name].material
        assert getattr(obj_mat, "alpha_mode", None) == "weighted_blend"
        assert bool(getattr(obj_mat, "depth_write", True)) is False
        assert bool(getattr(obj_mat, "depth_test", False)) is True

    def test_opaque_mesh_restores_default_alpha_depth_state(self):
        from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

        renderer = PygfxRenderer(SimpleNamespace())
        mesh_name = "scene:wall_1::mesh"
        renderer._objects[mesh_name] = SimpleNamespace(material=renderer._build_mesh_material())
        renderer._kinds[mesh_name] = "mesh"

        assert (
            renderer.set_named_material(
                mesh_name,
                {"base_color": [0.2, 0.4, 0.8, 0.2], "roughness": 0.3, "metallic": 0.1},
            )
            is True
        )
        assert (
            renderer.set_named_material(
                mesh_name,
                {"base_color": [0.2, 0.4, 0.8, 1.0], "roughness": 0.3, "metallic": 0.1},
            )
            is True
        )

        obj_mat = renderer._objects[mesh_name].material
        assert getattr(obj_mat, "alpha_mode", None) == "auto"
        assert bool(getattr(obj_mat, "depth_write", False)) is True

    def test_transparency_persists_across_topology_recreate(self, renderer):
        renderer.ensure_named_geometry("scene_mesh", _make_mesh(20))
        assert (
            renderer.set_geometry_transparency(
                "scene_mesh",
                alpha=0.3,
                color=[0.2, 0.4, 0.6],
                roughness=0.1,
                metallic=0.2,
                reflectance=0.4,
            )
            is True
        )
        # Topology change triggers remove/recreate.
        renderer.ensure_named_geometry("scene_mesh", _make_mesh(35))
        mat = renderer._materials.get("scene_mesh")
        assert mat is not None
        assert abs(float(mat.base_color[3]) - 0.3) < 1e-6


class _CanvasStub:
    def __init__(self) -> None:
        self.calls = 0

    def request_draw(self, *_args) -> None:
        self.calls += 1


class _RendererStub:
    def __init__(self) -> None:
        self.calls = []

    def render(self, scene, camera, **kwargs) -> None:
        self.calls.append(kwargs)
        return None


class _SceneStub:
    def traverse(self, callback) -> None:
        callback(self)


def _make_idle_renderer():
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer(SimpleNamespace(animation_running=False))
    renderer._initialized = True
    renderer._canvas = _CanvasStub()
    renderer._renderer = _RendererStub()
    renderer._scene = _SceneStub()
    renderer._camera = SimpleNamespace(
        local=SimpleNamespace(position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0)),
        fov=60.0,
    )
    renderer._update_headlight_pose = lambda: None
    renderer._render_attempts = 1
    return renderer


def test_request_redraw_requests_exactly_one_canvas_draw():
    renderer = _make_idle_renderer()

    renderer.request_redraw()

    assert renderer._canvas.calls == 1
    assert renderer.get_runtime_stats()["redraw_requests"] >= 1
    assert renderer.get_runtime_stats()["idle_loop_active"] is False


def test_animate_does_not_reschedule_after_one_requested_draw():
    renderer = _make_idle_renderer()

    renderer._animate()

    assert renderer._canvas.calls == 0


def test_animate_does_not_use_deprecated_clear_color_argument():
    renderer = _make_idle_renderer()
    renderer.set_background_color([0.2, 0.2, 0.2])

    renderer._animate()

    assert renderer._clear_color == (0.2, 0.2, 0.2, 1.0)
    assert "clear_color" not in renderer._renderer.calls[-1]


def test_animate_does_not_duplicate_animation_frame_requests():
    renderer = _make_idle_renderer()
    renderer.visualizer.animation_running = True

    renderer._animate()

    assert renderer._canvas.calls == 0


def test_deferred_default_ibl_load_applies_scene_and_requests_redraw():
    renderer = _make_idle_renderer()
    renderer._deferred_default_ibl_name = "neutral_outdoor"
    redraws = []
    applied_scenes = []
    applied_materials = []

    renderer._objects = {
        "mesh": SimpleNamespace(material=SimpleNamespace(env_map=None)),
        "light": SimpleNamespace(material=object()),
    }
    renderer._ibl_manager = SimpleNamespace(
        load_ibl=lambda name: object(),
        apply_to_scene=lambda scene: applied_scenes.append(scene) or True,
        apply_to_material=lambda material: applied_materials.append(material),
        set_skybox_visible=lambda visible, scene: None,
    )
    renderer.request_redraw = lambda: redraws.append(1)

    renderer._load_deferred_default_ibl()

    assert renderer._ibl_loaded is True
    assert renderer._deferred_default_ibl_name is None
    assert applied_scenes == [renderer._scene]
    assert applied_materials == [renderer._objects["mesh"].material]
    assert redraws == [1]


def test_pygfx_ibl_manager_roundtrips_cubemap_cache(tmp_path):
    from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

    manager = PygfxIBLManager(SimpleNamespace(), Path(tmp_path))
    manager._face_size = 4
    cubemap = np.full((6, 4, 4, 4), 0.5, dtype=np.float32)

    manager._write_cubemap_cache("neutral_outdoor", cubemap)
    loaded = manager._load_cached_cubemap("neutral_outdoor")

    assert loaded is not None
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, cubemap, atol=1e-3)


def test_schedule_deferred_default_ibl_load_schedules_once(monkeypatch):
    renderer = _make_idle_renderer()
    renderer._deferred_default_ibl_name = "neutral_outdoor"
    calls = []

    monkeypatch.setattr(
        "PySide6.QtCore.QTimer.singleShot",
        lambda delay_ms, callback: calls.append((delay_ms, callback)),
    )

    renderer._schedule_deferred_default_ibl_load()
    renderer._schedule_deferred_default_ibl_load()

    assert len(calls) == 1
    assert calls[0][0] == 0
    assert renderer._deferred_ibl_load_scheduled is True


class TestWireframeAndCache:
    def test_wireframe_targets_scene_and_target_mesh_components(self, renderer):
        scene_mesh = _make_mesh(16)
        target_mesh = _make_mesh(16)
        node_mesh = _make_mesh(16)
        label_mesh = _make_mesh(16)

        renderer.ensure_named_geometry("scene:wall_0::mesh", scene_mesh)
        renderer.ensure_named_geometry("target:car_0::mesh", target_mesh)
        renderer.ensure_named_geometry("node:tx_0::marker", node_mesh)
        renderer.ensure_named_geometry("target:car_0::label", label_mesh)

        renderer.set_wireframe(True)

        assert bool(getattr(renderer._objects["scene:wall_0::mesh"].material, "wireframe", False))
        assert bool(getattr(renderer._objects["target:car_0::mesh"].material, "wireframe", False))
        assert not bool(
            getattr(renderer._objects["node:tx_0::marker"].material, "wireframe", False)
        )
        assert not bool(
            getattr(renderer._objects["target:car_0::label"].material, "wireframe", False)
        )

    def test_force_refresh_clears_payload_cache(self, renderer):
        geometry = object()
        renderer._payload_cache[id(geometry)] = (("old",), _make_mesh(8))

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(renderer, "_external_name_for_geometry", lambda _g: "force_refresh_obj")
            mp.setattr(renderer, "_coerce_geometry_payload", lambda _g: _make_mesh(8))
            mp.setattr(renderer, "ensure_named_geometry", lambda *args, **kwargs: True)
            renderer.update_geometry_in_visualizer(geometry, force_refresh=True)

        assert id(geometry) not in renderer._payload_cache


class TestCamera:
    def test_camera_orbit_roundtrip(self, renderer):
        from visualizer.src.types.camera_state import CameraState

        state = CameraState(
            eye=(15.0, 8.0, 6.0),
            lookat=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            fov_deg=50.0,
        )

        assert renderer.set_camera_state(state) is True
        got = renderer.get_camera_state()
        assert got is not None
        np.testing.assert_allclose(got.eye, state.eye, atol=1e-2)
        np.testing.assert_allclose(got.lookat, state.lookat, atol=1e-2)
        np.testing.assert_allclose(got.up, state.up, atol=1e-2)
        assert abs(got.fov_deg - state.fov_deg) < 2.0

    def test_set_fly_mode_toggle_preserves_controller(self, renderer):
        assert renderer.set_fly_mode(True) is True
        assert renderer._active_controller_type == "fly"
        assert renderer._controller is not None

        assert renderer.set_fly_mode(False) is True
        assert renderer._active_controller_type == "orbit"
        assert renderer._controller is not None

    def test_set_camera_state_keeps_orbit_target_after_mode_switches(self, renderer):
        from visualizer.src.types.camera_state import CameraState

        state = CameraState(
            eye=(20.0, -10.0, 8.0),
            lookat=(1.0, 2.0, 3.0),
            up=(0.0, 0.0, 1.0),
            fov_deg=45.0,
        )
        assert renderer.set_camera_state(state) is True
        assert renderer.set_fly_mode(True) is True
        assert renderer.set_fly_mode(False) is True

        ctrl = renderer._controller
        assert ctrl is not None
        assert hasattr(ctrl, "target")
        np.testing.assert_allclose(np.asarray(ctrl.target), np.asarray(state.lookat), atol=1e-3)


class TestMpcPayloadConsistency:
    def test_mpc_style_persists_across_capacity_reuse(self, renderer):
        renderer.set_line_width(7.0)
        renderer.set_point_size(11.0)

        vm1 = _make_mpc_view_model(
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]], dtype=np.float32),
            np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
        )
        renderer.apply_frame(vm1.to_render_packet())

        line_obj_1 = renderer._objects.get("mpc_lines")
        point_obj_1 = renderer._objects.get("mpc_points")
        assert line_obj_1 is not None
        assert point_obj_1 is not None
        assert float(line_obj_1.material.thickness) == pytest.approx(7.0)
        assert float(point_obj_1.material.size) == pytest.approx(11.0)

        line_handle_1 = renderer._get_entity_handle("mpc_lines")
        point_handle_1 = renderer._get_entity_handle("mpc_points")

        vm2 = _make_mpc_view_model(
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
            np.array([[0, 1], [1, 2]], dtype=np.int32),
        )
        renderer.apply_frame(vm2.to_render_packet())

        line_obj_2 = renderer._objects.get("mpc_lines")
        point_obj_2 = renderer._objects.get("mpc_points")
        assert line_obj_2 is not None
        assert point_obj_2 is not None
        assert renderer._get_entity_handle("mpc_lines") == line_handle_1
        assert renderer._get_entity_handle("mpc_points") == point_handle_1
        assert float(line_obj_2.material.thickness) == pytest.approx(7.0)
        assert float(point_obj_2.material.size) == pytest.approx(11.0)
        assert tuple(line_obj_2.geometry.positions.draw_range) == (0, 4)
        assert getattr(line_obj_2.geometry, "indices", None) is None
        assert tuple(point_obj_2.geometry.positions.draw_range) == (0, 3)
        assert tuple(point_obj_2.geometry.colors.draw_range) == (0, 3)

    def test_mpc_line_segment_colors_expand_consistently(self, renderer):
        view_model = _make_mpc_view_model(
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]], dtype=np.float32),
            np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
            colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            show_mpc_bounce_points=False,
        )

        renderer.apply_frame(view_model.to_render_packet())
        line_obj = renderer._objects.get("mpc_lines")
        assert line_obj is not None
        geom = line_obj.geometry
        assert geom.positions.data.shape[0] >= 6
        assert geom.colors.data.shape[0] >= 6
        assert geom.colors.data.shape[1] == 3
        assert getattr(geom, "indices", None) is None
        assert tuple(geom.positions.draw_range) == (0, 6)
        np.testing.assert_allclose(
            geom.positions.data[:6],
            np.array(
                [[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [2, 1, 0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            geom.colors.data[:6],
            np.array(
                [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
                dtype=np.float32,
            ),
        )

    def test_mpc_invalid_line_colors_are_ignored(self, renderer):
        view_model = _make_mpc_view_model(
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
            np.array([[0, 1], [1, 2]], dtype=np.int32),
            colors=np.array([[1, 0, 0]], dtype=np.float32),
            show_mpc_bounce_points=False,
        )

        renderer.apply_frame(view_model.to_render_packet())
        line_obj = renderer._objects.get("mpc_lines")
        assert line_obj is not None
        assert getattr(line_obj.geometry, "colors", None) is None

    def test_invalid_line_indices_are_dropped(self, renderer):
        view_model = _make_mpc_view_model(
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
            np.array([[0, 1], [1, 99], [1, 2]], dtype=np.int32),
            colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            show_mpc_bounce_points=False,
        )

        renderer.apply_frame(view_model.to_render_packet())
        line_obj = renderer._objects.get("mpc_lines")
        assert line_obj is not None
        # One invalid segment should be dropped, leaving 2 segments.
        assert line_obj.geometry.positions.data.shape[0] >= 4
        assert getattr(line_obj.geometry, "indices", None) is None
        assert tuple(line_obj.geometry.positions.draw_range) == (0, 4)

    def test_mpc_expanded_line_cache_replays_previous_frame(self, renderer):
        renderer._mpc_expanded_line_cache_max_bytes = 4 * 1024 * 1024
        renderer._clear_mpc_expanded_line_cache()

        vm1 = _make_mpc_view_model(
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]], dtype=np.float32),
            np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
            colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            show_mpc_bounce_points=False,
        )

        vm2 = _make_mpc_view_model(
            np.array([[10, 0, 0], [11, 0, 0], [12, 0, 0]], dtype=np.float32),
            np.array([[0, 1], [1, 2]], dtype=np.int32),
            colors=np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]], dtype=np.float32),
            show_mpc_bounce_points=False,
        )

        renderer.apply_frame(vm1.to_render_packet())
        stores_after_first = renderer._mpc_expanded_line_cache_stores
        renderer.apply_frame(vm2.to_render_packet())
        hits_before_replay = renderer._mpc_expanded_line_cache_hits
        renderer.apply_frame(vm1.to_render_packet())

        assert renderer._mpc_expanded_line_cache_stores >= stores_after_first
        assert renderer._mpc_expanded_line_cache_hits == hits_before_replay + 1
        line_obj = renderer._objects.get("mpc_lines")
        assert line_obj is not None
        np.testing.assert_allclose(
            line_obj.geometry.positions.data[:6],
            np.array(
                [[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [2, 1, 0]],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            line_obj.geometry.colors.data[:6],
            np.array(
                [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
                dtype=np.float32,
            ),
        )

    def test_mpc_expanded_line_cache_detects_in_place_point_mutation(self, renderer):
        renderer._mpc_expanded_line_cache_max_bytes = 4 * 1024 * 1024
        renderer._clear_mpc_expanded_line_cache()

        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32)
        lines = np.array([[0, 1], [1, 2]], dtype=np.int32)
        colors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        view_model = _make_mpc_view_model(
            points, lines, colors=colors, show_mpc_bounce_points=False
        )

        renderer.apply_frame(view_model.to_render_packet())
        points[1] = [10, 0, 0]
        mutated_view_model = _make_mpc_view_model(
            points,
            lines,
            colors=colors,
            show_mpc_bounce_points=False,
        )
        renderer.apply_frame(mutated_view_model.to_render_packet())

        line_obj = renderer._objects.get("mpc_lines")
        assert line_obj is not None
        np.testing.assert_allclose(
            line_obj.geometry.positions.data[:4],
            np.array(
                [[0, 0, 0], [10, 0, 0], [10, 0, 0], [1, 1, 0]],
                dtype=np.float32,
            ),
        )

    def test_mpc_expanded_line_cache_prewarm_populates_replay_cache(self, renderer):
        renderer._mpc_expanded_line_cache_max_bytes = 4 * 1024 * 1024
        renderer._clear_mpc_expanded_line_cache()

        view_model = _make_mpc_view_model(
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]], dtype=np.float32),
            np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
            colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            show_mpc_bounce_points=False,
        )

        assert renderer.prewarm_mpc_line_cache(view_model.to_render_packet()) is True
        assert renderer._mpc_expanded_line_cache_prewarm_stores == 1
        assert renderer._mpc_expanded_line_cache_stores == 1

        hits_before_apply = renderer._mpc_expanded_line_cache_hits
        renderer.apply_frame(view_model.to_render_packet())

        assert renderer._mpc_expanded_line_cache_hits == hits_before_apply + 1
        line_obj = renderer._objects.get("mpc_lines")
        assert line_obj is not None
        np.testing.assert_allclose(
            line_obj.geometry.positions.data[:6],
            np.array(
                [[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [2, 1, 0]],
                dtype=np.float32,
            ),
        )

        stats = renderer.get_runtime_stats()
        assert stats["mpc_line_cache_prewarm_enabled"] is True
        assert stats["mpc_line_cache_prewarm_attempts"] == 1
        assert stats["mpc_line_cache_prewarm_stores"] == 1
        assert stats["mpc_line_cache_hit_rate"] == 1.0

    def test_mpc_expanded_line_cache_prewarm_skips_when_disabled(self, renderer):
        renderer._mpc_expanded_line_cache_max_bytes = 0
        renderer._clear_mpc_expanded_line_cache()

        view_model = _make_mpc_view_model(
            np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int32),
            show_mpc_bounce_points=False,
        )

        assert renderer.prewarm_mpc_line_cache(view_model.to_render_packet()) is False
        assert renderer._mpc_expanded_line_cache_prewarm_attempts == 0
        assert renderer._mpc_expanded_line_cache_stores == 0

    def test_mpc_line_capacity_hint_survives_line_geometry_removal(self, renderer):
        large_points = np.column_stack(
            [
                np.arange(20, dtype=np.float32),
                np.zeros(20, dtype=np.float32),
                np.zeros(20, dtype=np.float32),
            ]
        )
        large_lines = np.column_stack(
            [np.arange(19, dtype=np.int32), np.arange(1, 20, dtype=np.int32)]
        )
        renderer.apply_frame(
            _make_mpc_view_model(
                large_points,
                large_lines,
                show_mpc_bounce_points=False,
            ).to_render_packet()
        )
        capacity_before = renderer._mpc_segment_capacity
        assert capacity_before >= len(large_lines)

        assert renderer.remove_named_geometry("mpc_lines") is True
        assert renderer._mpc_segment_capacity_hint >= capacity_before

        small_vm = _make_mpc_view_model(
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
            np.array([[0, 1], [1, 2]], dtype=np.int32),
            show_mpc_bounce_points=False,
        )
        renderer.apply_frame(small_vm.to_render_packet())
        assert renderer._mpc_segment_capacity >= capacity_before

    def test_mpc_point_capacity_hint_survives_point_geometry_removal(self, renderer):
        large_points = np.column_stack(
            [
                np.arange(20, dtype=np.float32),
                np.zeros(20, dtype=np.float32),
                np.zeros(20, dtype=np.float32),
            ]
        )
        large_lines = np.column_stack(
            [np.arange(19, dtype=np.int32), np.arange(1, 20, dtype=np.int32)]
        )
        renderer.apply_frame(
            _make_mpc_view_model(
                large_points,
                large_lines,
                show_mpc_bounce_points=True,
            ).to_render_packet()
        )
        capacity_before = renderer._mpc_point_capacity
        assert capacity_before >= len(large_points)

        assert renderer.remove_named_geometry("mpc_points") is True
        assert renderer._mpc_point_capacity_hint >= capacity_before

        small_vm = _make_mpc_view_model(
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
            np.array([[0, 1], [1, 2]], dtype=np.int32),
            show_mpc_bounce_points=True,
        )
        renderer.apply_frame(small_vm.to_render_packet())
        assert renderer._mpc_point_capacity >= capacity_before
        point_obj = renderer._objects.get("mpc_points")
        assert point_obj is not None
        assert tuple(point_obj.geometry.positions.draw_range) == (0, 3)

    def test_mpc_line_cache_runtime_stats_report_fit_status(self, renderer):
        renderer.visualizer.total_animation_steps = 3
        renderer._mpc_expanded_line_cache_max_bytes = 4 * 1024 * 1024
        renderer._clear_mpc_expanded_line_cache()

        view_model = _make_mpc_view_model(
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]], dtype=np.float32),
            np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
            colors=np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                dtype=np.float32,
            ),
            show_mpc_bounce_points=False,
        )

        renderer.apply_frame(view_model.to_render_packet())
        stats = renderer.get_runtime_stats()

        assert stats["mpc_line_cache_fit_status"] == "fits_observed_workload"
        assert stats["mpc_line_cache_largest_entry_bytes"] > 0
        assert stats["mpc_line_cache_suggested_full_loop_bytes"] == (
            stats["mpc_line_cache_largest_entry_bytes"] * 3
        )
        assert stats["mpc_line_cache_suggested_full_loop_mb"] >= 1


class TestTrajectories:
    def test_apply_and_remove_trajectories(self, renderer):
        data = {
            "tx_positions": {0: [(0, 0.0, 0.0, 0.0), (1, 1.0, 0.0, 0.0)]},
            "rx_positions": {0: [(0, 0.0, 1.0, 0.0), (1, 1.0, 1.0, 0.0)]},
            "target_positions": {"car": [(0, 0.0, 0.0, 1.0), (1, 0.0, 1.0, 1.0)]},
        }
        renderer.apply_trajectory("tx", data)
        renderer.apply_trajectory("rx", data)
        renderer.apply_trajectory("target", data)

        assert renderer.has_named_geometry(renderer.TRAJECTORY_TX_LINES_NAME)
        assert renderer.has_named_geometry(renderer.TRAJECTORY_TX_POINTS_NAME)
        assert renderer.has_named_geometry(renderer.TRAJECTORY_RX_LINES_NAME)
        assert renderer.has_named_geometry(renderer.TRAJECTORY_RX_POINTS_NAME)
        assert any(
            name.startswith(renderer.TRAJECTORY_TARGET_LINES_PREFIX)
            for name in renderer._name_to_handle
        )
        assert any(
            name.startswith(renderer.TRAJECTORY_TARGET_POINTS_PREFIX)
            for name in renderer._name_to_handle
        )

        assert renderer.set_trajectory_line_width(5.0) is True
        assert renderer.set_trajectory_point_size(9.0) is True

        renderer.remove_trajectory("tx")
        renderer.remove_trajectory("rx")
        renderer.remove_trajectory("target")
        assert not renderer.has_named_geometry(renderer.TRAJECTORY_TX_LINES_NAME)
        assert not renderer.has_named_geometry(renderer.TRAJECTORY_TX_POINTS_NAME)
        assert not renderer.has_named_geometry(renderer.TRAJECTORY_RX_LINES_NAME)
        assert not renderer.has_named_geometry(renderer.TRAJECTORY_RX_POINTS_NAME)


def _make_coverage_packet(
    show_coverage: bool = True,
    coverage_opacity: float = 0.7,
    coverage_signature: str = "sig_abc",
    show_isolines: bool = False,
) -> FrameRenderPacket:
    """Create a minimal frame packet with coverage mesh data."""
    # 4-vertex quad split into 2 triangles.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float64)
    isoline_points = np.array([[0.25, 0.0, 0.05], [0.75, 1.0, 0.05]], dtype=np.float64)
    isoline_lines = np.array([[0, 1]], dtype=np.int32)
    isoline_colors = np.array([[0.05, 0.05, 0.05]], dtype=np.float64)
    base = _make_mpc_view_model(
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 2), dtype=np.int32),
        show_mpc_bounce_points=False,
    ).to_render_packet()
    return replace(
        base,
        show_coverage=show_coverage,
        coverage_vertices=vertices if show_coverage else None,
        coverage_triangles=triangles if show_coverage else None,
        coverage_colors=colors if show_coverage else None,
        coverage_isoline_points=isoline_points if show_coverage and show_isolines else None,
        coverage_isoline_lines=isoline_lines if show_coverage and show_isolines else None,
        coverage_isoline_colors=isoline_colors if show_coverage and show_isolines else None,
        coverage_opacity=coverage_opacity,
        coverage_signature=coverage_signature,
        beamforming_meshes=[],
        stats_text="",
    )


class TestCoverageMesh:
    def test_coverage_mesh_creation(self, renderer):
        """Coverage mesh is created when show_coverage=True with valid data."""
        packet = _make_coverage_packet(show_coverage=True)
        renderer.apply_frame(packet)
        assert renderer.has_named_geometry(renderer.COVERAGE_MESH_NAME)
        assert renderer._last_coverage_signature == "sig_abc"

    def test_coverage_isolines_creation(self, renderer):
        """Coverage isolines are uploaded when isoline data is present."""
        packet = _make_coverage_packet(show_coverage=True, show_isolines=True)
        renderer.apply_frame(packet)
        assert renderer.has_named_geometry(renderer.COVERAGE_MESH_NAME)
        assert renderer.has_named_geometry(renderer.COVERAGE_ISOLINES_NAME)

    def test_coverage_mesh_not_created_when_hidden(self, renderer):
        """Coverage mesh is not created when show_coverage=False."""
        packet = _make_coverage_packet(show_coverage=False)
        renderer.apply_frame(packet)
        assert not renderer.has_named_geometry(renderer.COVERAGE_MESH_NAME)
        assert renderer._last_coverage_signature is None

    def test_coverage_mesh_removed_on_toggle_off(self, renderer):
        """Coverage mesh is removed when show_coverage toggles from True to False."""
        packet_on = _make_coverage_packet(show_coverage=True)
        renderer.apply_frame(packet_on)
        assert renderer.has_named_geometry(renderer.COVERAGE_MESH_NAME)

        packet_off = _make_coverage_packet(show_coverage=False)
        renderer.apply_frame(packet_off)
        assert not renderer.has_named_geometry(renderer.COVERAGE_MESH_NAME)
        assert not renderer.has_named_geometry(renderer.COVERAGE_ISOLINES_NAME)

    def test_coverage_transparency_update(self, renderer):
        """Opacity-only change uses set_coverage_transparency (no re-upload)."""
        packet1 = _make_coverage_packet(coverage_opacity=0.8, coverage_signature="sig_1")
        renderer.apply_frame(packet1)
        assert renderer.has_named_geometry(renderer.COVERAGE_MESH_NAME)

        packet2 = _make_coverage_packet(coverage_opacity=0.3, coverage_signature="sig_1")
        renderer.apply_frame(packet2)
        # Geometry should still exist — opacity-only change.
        assert renderer.has_named_geometry(renderer.COVERAGE_MESH_NAME)
        # Signature should not have changed (no full re-apply).
        assert renderer._last_coverage_signature == "sig_1"

    def test_coverage_signature_change_triggers_re_upload(self, renderer):
        """Signature change triggers full re-apply."""
        packet1 = _make_coverage_packet(coverage_signature="sig_A")
        renderer.apply_frame(packet1)
        assert renderer._last_coverage_signature == "sig_A"

        packet2 = _make_coverage_packet(coverage_signature="sig_B")
        renderer.apply_frame(packet2)
        assert renderer._last_coverage_signature == "sig_B"

    def test_coverage_clear_resets_signature(self, renderer):
        """clear() resets coverage tracking state."""
        packet = _make_coverage_packet()
        renderer.apply_frame(packet)
        assert renderer._last_coverage_signature is not None

        renderer.clear()
        assert renderer._last_coverage_signature is None
        assert not renderer.has_named_geometry(renderer.COVERAGE_MESH_NAME)

    def test_clear_resets_beamforming_snapshot(self, renderer):
        """clear() forgets backend-applied beamforming payloads."""
        renderer._applied_beamforming_surfaces["beamforming:test:mesh"] = object()

        renderer.clear()

        assert renderer._applied_beamforming_surfaces == {}

    def test_set_coverage_transparency_returns_bool(self, renderer):
        """set_coverage_transparency updates the coverage material in place."""
        packet = _make_coverage_packet()
        renderer.apply_frame(packet)
        assert renderer.set_coverage_transparency(0.5) is True

    def test_coverage_transparency_preserves_vertex_colors(self, renderer):
        """Opacity changes must not reset the coverage heatmap to a white material."""
        packet = _make_coverage_packet(coverage_opacity=0.9)
        renderer.apply_frame(packet)

        obj = renderer._objects[renderer.COVERAGE_MESH_NAME]
        mat = obj.material
        if hasattr(mat, "color_mode"):
            assert mat.color_mode == "vertex"
        if hasattr(mat, "opacity"):
            assert mat.opacity == pytest.approx(0.9)
        if hasattr(mat, "depth_write"):
            assert bool(mat.depth_write) is False

        assert renderer.set_coverage_transparency(1.0) is True
        mat = renderer._objects[renderer.COVERAGE_MESH_NAME].material
        if hasattr(mat, "color_mode"):
            assert mat.color_mode == "vertex"
        if hasattr(mat, "opacity"):
            assert mat.opacity == pytest.approx(1.0)
        if hasattr(mat, "depth_write"):
            assert bool(mat.depth_write) is True

    def test_set_coverage_transparency_no_geometry(self, renderer):
        """set_coverage_transparency returns False when no coverage exists."""
        assert renderer.set_coverage_transparency(0.5) is False


class TestShadows:
    def test_shadow_toggle_capability(self):
        from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

        assert PygfxRenderer.capabilities.shadow_toggle is True

    def test_shadow_enabled_toggle(self, renderer):
        assert renderer._shadows_enabled is True
        assert renderer.set_shadow_enabled(False) is True
        assert renderer._shadows_enabled is False
        assert renderer.set_shadow_enabled(True) is True
        assert renderer._shadows_enabled is True

    def test_scene_mesh_gets_cast_shadow(self, renderer):
        mesh = _make_mesh(20)
        renderer.ensure_named_geometry("scene_floor", mesh)
        obj = renderer._objects.get("scene_floor")
        assert obj is not None
        if hasattr(obj, "cast_shadow"):
            assert obj.cast_shadow is True
        if hasattr(obj, "receive_shadow"):
            assert obj.receive_shadow is True

    def test_target_mesh_gets_cast_and_receive_shadow(self):
        from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

        target_entry = {"target_name": "Walker"}
        ensure_target_entry_identity(target_entry, 0)
        mesh_name = make_target_entry_geometry_name(target_entry, "mesh")
        renderer = object.__new__(PygfxRenderer)
        obj = SimpleNamespace()

        renderer._apply_shadow_flags(mesh_name, obj)

        assert obj.cast_shadow is True
        assert obj.receive_shadow is True

    def test_mpc_geometry_no_cast_shadow(self, renderer):
        mesh = _make_mesh(20)
        renderer.ensure_named_geometry("mpc_lines", mesh)
        obj = renderer._objects.get("mpc_lines")
        assert obj is not None
        if hasattr(obj, "cast_shadow"):
            assert obj.cast_shadow is False

    def test_shadow_extent_update(self, renderer):
        renderer._update_shadow_extent(500.0)
        assert renderer._scene_extent == 500.0

    def test_shadow_extent_clamp_minimum(self, renderer):
        renderer._update_shadow_extent(1.0)
        assert renderer._scene_extent == 10.0


# ---------------------------------------------------------------------------
# IBL Manager tests (no GPU/Qt required — tests the helper class directly)
# ---------------------------------------------------------------------------


def _ibl_manager_available() -> bool:
    try:
        import cv2  # noqa: F401
        import pygfx  # noqa: F401

        return True
    except ImportError:
        return False


ibl_pytestmark = pytest.mark.skipif(
    not _ibl_manager_available(),
    reason="pygfx or opencv-python-headless not available",
)


@ibl_pytestmark
class TestIBLManagerDiscovery:
    def test_discover_returns_sorted_names(self):
        import pygfx as gfx

        from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

        ibl_dir = Path(__file__).resolve().parent.parent.parent.parent / "libraries" / "ibl"
        mgr = PygfxIBLManager(gfx, ibl_dir)
        names = mgr.discover_available()
        assert isinstance(names, list)
        assert len(names) > 0
        assert names == sorted(names)
        assert "neutral_outdoor" in names

    def test_discover_empty_dir(self, tmp_path):
        import pygfx as gfx

        from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

        mgr = PygfxIBLManager(gfx, tmp_path)
        assert mgr.discover_available() == []

    def test_discover_missing_dir(self):
        import pygfx as gfx

        from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

        mgr = PygfxIBLManager(gfx, Path("/nonexistent/ibl"))
        assert mgr.discover_available() == []


@ibl_pytestmark
class TestIBLManagerResolveHDR:
    def test_resolve_zup_preferred(self, tmp_path):
        import pygfx as gfx

        from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

        (tmp_path / "my_env_zup.hdr").write_bytes(b"dummy")
        (tmp_path / "my_env.hdr").write_bytes(b"dummy")
        mgr = PygfxIBLManager(gfx, tmp_path)
        path = mgr._resolve_hdr_path("my_env")
        assert path is not None
        assert path.name == "my_env_zup.hdr"

    def test_resolve_fallback_to_plain(self, tmp_path):
        import pygfx as gfx

        from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

        (tmp_path / "my_env.hdr").write_bytes(b"dummy")
        mgr = PygfxIBLManager(gfx, tmp_path)
        path = mgr._resolve_hdr_path("my_env")
        assert path is not None
        assert path.name == "my_env.hdr"

    def test_resolve_returns_none_for_missing(self, tmp_path):
        import pygfx as gfx

        from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

        mgr = PygfxIBLManager(gfx, tmp_path)
        assert mgr._resolve_hdr_path("nonexistent") is None


@ibl_pytestmark
class TestIBLManagerEquirectToCubemap:
    def test_small_equirect_shape(self):
        from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

        equirect = np.random.rand(4, 8, 3).astype(np.float32)
        cubemap = PygfxIBLManager._equirect_to_cubemap(equirect, face_size=2)
        assert cubemap.shape == (6, 2, 2, 3)
        assert cubemap.dtype == np.float32

    def test_values_are_finite_and_nonnegative(self):
        from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

        equirect = np.random.rand(64, 128, 3).astype(np.float32) * 10.0
        cubemap = PygfxIBLManager._equirect_to_cubemap(equirect, face_size=16)
        assert np.all(np.isfinite(cubemap))
        assert cubemap.min() >= 0.0

    def test_four_channel_input(self):
        from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

        equirect = np.random.rand(4, 8, 4).astype(np.float32)
        cubemap = PygfxIBLManager._equirect_to_cubemap(equirect, face_size=2)
        assert cubemap.shape == (6, 2, 2, 4)


@ibl_pytestmark
class TestIBLManagerTextureLoading:
    def test_load_real_ibl(self):
        import pygfx as gfx

        from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

        ibl_dir = Path(__file__).resolve().parent.parent.parent.parent / "libraries" / "ibl"
        mgr = PygfxIBLManager(gfx, ibl_dir)
        tex = mgr.load_ibl("neutral_outdoor")
        assert tex is not None
        # Cached second load returns same object.
        tex2 = mgr.load_ibl("neutral_outdoor")
        assert tex is tex2

    def test_load_missing_returns_none(self, tmp_path):
        import pygfx as gfx

        from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

        mgr = PygfxIBLManager(gfx, tmp_path)
        assert mgr.load_ibl("nonexistent") is None


@ibl_pytestmark
class TestIBLManagerSceneIntegration:
    def test_apply_and_toggle_skybox(self):
        import pygfx as gfx

        from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

        ibl_dir = Path(__file__).resolve().parent.parent.parent.parent / "libraries" / "ibl"
        mgr = PygfxIBLManager(gfx, ibl_dir)
        mgr.load_ibl("neutral_outdoor")

        scene = gfx.Scene()
        assert mgr.apply_to_scene(scene) is True
        assert any(isinstance(c, gfx.Background) for c in scene.children)

        # Hide skybox.
        mgr.set_skybox_visible(False, scene)
        assert not any(isinstance(c, gfx.Background) for c in scene.children)

        # Show skybox.
        mgr.set_skybox_visible(True, scene)
        assert any(isinstance(c, gfx.Background) for c in scene.children)
        if mgr.uses_scene_environment:
            assert scene.environment is not None

    def test_apply_to_scene_uses_scene_environment_when_available(self):
        import pygfx as gfx

        from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

        ibl_dir = Path(__file__).resolve().parent.parent.parent.parent / "libraries" / "ibl"
        mgr = PygfxIBLManager(gfx, ibl_dir)
        mgr.load_ibl("neutral_outdoor")

        scene = gfx.Scene()
        mgr.apply_to_scene(scene)

        if mgr.uses_scene_environment:
            assert scene.environment is not None
        else:
            assert getattr(scene, "environment", None) is None

    def test_set_intensity_propagates(self):
        import pygfx as gfx

        from visualizer.src.renderers.pygfx.lighting import PygfxIBLManager

        ibl_dir = Path(__file__).resolve().parent.parent.parent.parent / "libraries" / "ibl"
        mgr = PygfxIBLManager(gfx, ibl_dir)
        mgr.load_ibl("neutral_outdoor")

        mat = gfx.MeshStandardMaterial()
        mgr.apply_to_material(mat)
        if mgr.uses_scene_environment:
            assert mat.env_map is None
        else:
            assert mat.env_map is not None

        mgr.set_intensity(0.42)
        assert abs(mat.env_map_intensity - 0.42) < 1e-6
