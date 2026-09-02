from __future__ import annotations

import json
import pickle
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PySide6 import QtWidgets
from PySide6.QtWidgets import QLabel as _REAL_Q_LABEL

from shared.frames import StandardMPCFrame
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from visualizer.src.io.frame_sources import LiveGrpcSource, RemoteHdf5Source
from visualizer.src.model import RenderObjectState, Transform
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.scene.target_transforms import (
    TargetGeometryMeta,
    build_sionna_rotation_matrix,
    rotated_aabb_center,
)
from visualizer.src.services.live_preview_payloads import build_live_overrides
from visualizer.src.services.live_preview_service import LivePreviewService
from visualizer.src.services.node_service import NodeService
from visualizer.src.services.raytracing_settings_service import RaytracingSettingsService
from visualizer.src.services.target_asset_cache import (
    ResolvedTargetAssetSource,
    TargetAsset,
    TargetAssetCache,
    TargetAssetKey,
    TargetSourceRevision,
)
from visualizer.src.services.target_service import TargetService
from visualizer.src.types.render_payloads import MeshPayload


class _FakeRenderer:
    capabilities = RendererCapabilities(transform_gizmo=True)

    def __init__(self) -> None:
        self.callback = None
        self.session_begin_calls = 0
        self.session_end_calls = 0
        self.transforms = {}
        self.objects = {}
        self.ensure_calls = []
        self.named_position_reads = []
        self.set_named_transform_calls = []
        self.active_target_pose_syncs = []
        self.active_transform_target = None
        self._initialized = True
        self.positions = {
            "node:tx_0::marker": np.asarray([1.0, 2.0, 3.0]),
            "node:rx_0::marker": np.asarray([4.0, 5.0, 6.0]),
        }

    def begin_live_preview_transform_session(self, callback):
        self.session_begin_calls += 1
        self.callback = callback
        return True

    def end_live_preview_transform_session(self):
        self.session_end_calls += 1
        self.callback = None

    def get_named_position(self, name):
        self.named_position_reads.append(name)
        return self.positions.get(name)

    def set_named_transform(self, name, transform):
        self.set_named_transform_calls.append(name)
        self.transforms[name] = np.asarray(transform, dtype=float)
        return True

    def ensure_object(self, render_object):
        self.ensure_calls.append(render_object)
        self.objects[render_object.id] = render_object
        return True

    def remove_object(self, object_id):
        self.objects.pop(object_id, None)
        return True

    def batch_updates(self):
        return nullcontext()

    def request_redraw(self):
        return None

    def get_active_transform_target(self):
        return self.active_transform_target

    def sync_active_transform_target_pose(self, name, transform):
        self.active_target_pose_syncs.append((name, np.asarray(transform, dtype=float)))
        return True


class _FakeCacheService:
    def __init__(self) -> None:
        self.invalidate_canonical_step = Mock()
        self.invalidate = Mock()


class _FakeStdin:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.closed = False

    def write(self, text: str) -> None:
        self.lines.append(text)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.stdout = None
        self.stderr = None
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


def _messages(proc: _FakeProcess) -> list[dict]:
    return [json.loads(line) for line in proc.stdin.lines if line.strip()]


def _mark_worker_ready(service: LivePreviewService) -> None:
    service._handle_worker_message(
        {
            "type": "ready",
            "source_identity": service._source_identity.to_dict(),
        }
    )


def _restore_real_qt_labels(monkeypatch, *panel_modules) -> None:
    """Bind real QLabel classes for layout tests."""

    monkeypatch.setattr(QtWidgets, "QLabel", _REAL_Q_LABEL)
    loaded_panels = {
        module
        for name, module in tuple(sys.modules.items())
        if name.startswith("visualizer.src.panels") and module is not None
    }
    for module in loaded_panels.union(panel_modules):
        if hasattr(module, "QLabel"):
            monkeypatch.setattr(module, "QLabel", _REAL_Q_LABEL)


def _make_viz() -> SimpleNamespace:
    renderer = _FakeRenderer()
    cache_service = _FakeCacheService()
    target_vertices = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 4.0, 6.0], [2.0, 0.0, 0.0]],
        dtype=float,
    )
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
    marker_payload = MeshPayload(
        vertices=np.asarray(
            [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
            dtype=float,
        ),
        triangles=triangles,
    )
    tx_marker = RenderObjectState(
        id="node:tx_0::marker",
        payload=marker_payload,
        world_transform=Transform.from_translation([1.0, 2.0, 3.0]),
    )
    rx_marker = RenderObjectState(
        id="node:rx_0::marker",
        payload=marker_payload,
        world_transform=Transform.from_translation([4.0, 5.0, 6.0]),
    )
    target_mesh = RenderObjectState(
        id="target:walker::mesh",
        payload=MeshPayload(vertices=target_vertices, triangles=triangles),
    )
    viz = SimpleNamespace(
        renderer=renderer,
        cache_service=cache_service,
        vis=None,
        vis_initialized=True,
        current_tx_positions=np.asarray([[1.0, 2.0, 3.0]], dtype=float),
        current_rx_positions=np.asarray([[4.0, 5.0, 6.0]], dtype=float),
        current_tx_orientations=None,
        current_rx_orientations=None,
        current_target_positions=None,
        current_target_orientations=None,
        tx_markers=[tx_marker],
        rx_markers=[rx_marker],
        tx_labels=[],
        rx_labels=[],
        target_labels=[],
        tx_entries=[{"position": [1.0, 2.0, 3.0], "visible": True, "show_label": True}],
        rx_entries=[{"position": [4.0, 5.0, 6.0], "visible": True, "show_label": True}],
        target_entries=[
            {
                "target_name": "walker",
                "display_name": "Walker",
                "position": [10.0, 20.0, 30.0],
                "orientation": [0.0, 0.0, 0.0],
                "_target_position": [10.0, 20.0, 30.0],
                "_mesh_center": [1.0, 2.0, 3.0],
                "mesh_file": "walker.ply",
                "mesh": target_mesh,
                "visible": True,
                "node_index": 0,
                "_use_ply_position": False,
            }
        ],
        target_outlines_enabled=False,
        label_offset_x=0.0,
        label_offset_y=0.0,
        label_offset_z=0.5,
        show_labels=True,
        selected_tx="all",
        selected_rx="all",
        app_state=SimpleNamespace(
            selected_tx="all",
            selected_rx="all",
            show_labels=True,
            show_target_labels=True,
            pov_hidden_node=None,
        ),
        num_tx=1,
        num_rx=1,
        animation_step=7,
        force_update_next_frame=False,
        update_calls=0,
        scenario=SimpleNamespace(
            root=Path("/tmp/live-preview-scenario"),
            data_mode="files",
            raytracing={"quality": {"preset": "ultra-low"}},
        ),
        current_project_root=Path("/tmp/live-preview-project"),
        raytracing_settings_service=RaytracingSettingsService(),
        _live_preview_enabled=False,
        _live_preview_frame=None,
        _live_preview_step=None,
        _live_preview_sequence=None,
        _live_preview_status="Preview off",
        ui_manager=SimpleNamespace(panels={}),
    )

    def schedule_update():
        viz.update_calls += 1

    viz.schedule_update = schedule_update
    viz._set_status_message = Mock()
    viz.scene_edit_service = SimpleNamespace(edit_node_properties=Mock(return_value=True))
    viz.target_service = TargetService(viz)
    target_asset_cache = viz.target_asset_cache
    assert isinstance(target_asset_cache, TargetAssetCache)
    target_asset_key = ("walker", "walker.ply")
    canonical_path = "memory://walker/walker.ply"
    target_asset = TargetAsset(
        source=ResolvedTargetAssetSource(
            target_name="walker",
            mesh_filename="walker.ply",
            canonical_path=canonical_path,
            key=TargetAssetKey(
                canonical_path=canonical_path,
                revision=TargetSourceRevision(0, 0, 0, 0),
                target_name="walker",
                mesh_filename="walker.ply",
            ),
        ),
        mesh=target_mesh,
        original_vertices=target_vertices.copy(),
        scaled_vertices=target_vertices.copy(),
        geometry_meta=TargetGeometryMeta(
            scaled_aabb_center=(target_vertices.min(axis=0) + target_vertices.max(axis=0)) / 2.0
        ),
    )
    target_asset_cache.put(target_asset)
    assert target_asset_cache.asset_for_logical_key(target_asset_key) is target_asset
    viz.target_entries[0]["_target_asset"] = target_asset
    viz.node_service = NodeService(viz, target_service=viz.target_service)
    return viz


def _preview_frame(
    *,
    frame_index: int = 7,
    target_positions: np.ndarray | None = None,
    targets_metadata: tuple[dict[str, object], ...] = (),
) -> StandardMPCFrame:
    """Build one complete worker result with live-preview provenance."""
    positions = (
        np.empty((0, 3), dtype=np.float64)
        if target_positions is None
        else np.asarray(target_positions, dtype=np.float64).reshape((-1, 3))
    )
    return replace(
        build_standard_mpc_frame("baseline", frame_idx=frame_index),
        target_positions_m=positions,
        targets_metadata=targets_metadata,
        provenance={
            "provider": "live_preview",
            "preview": True,
            "frame_idx": frame_index,
        },
    )


def _service_with_worker(qapp):
    viz = _make_viz()
    service = LivePreviewService(viz)
    proc = _FakeProcess()
    launch = patch.object(service, "_launch_worker_process", return_value=proc)
    readers = patch.object(service, "_start_worker_readers")
    launch_mock = launch.start()
    readers_mock = readers.start()

    def cleanup():
        launch.stop()
        readers.stop()

    return viz, service, proc, launch_mock, readers_mock, cleanup


def test_construction_does_not_start_worker(qapp):
    viz = _make_viz()
    service = LivePreviewService(viz)

    with patch.object(service, "_launch_worker_process") as launch:
        assert service.enabled is False
        launch.assert_not_called()


def test_remote_hdf5_rejects_actor_editing_without_starting_worker(qapp):
    viz = _make_viz()
    viz.frame_source = RemoteHdf5Source("localhost:50052")
    viz.scenario.data_mode = "remote_hdf5"
    service = LivePreviewService(viz)

    with patch.object(service, "_launch_worker_process") as launch:
        assert service.edit_mode() == "remote_hdf5"
        assert service.is_available() is False
        assert service.set_enabled(True) is False

    launch.assert_not_called()
    assert viz.renderer.session_begin_calls == 0
    assert "read-only" in viz._live_preview_status


def test_live_commit_sends_one_rpc_on_release_without_local_worker(qapp):
    viz = _make_viz()
    viz.frame_source = LiveGrpcSource("grpc://localhost:50051")
    viz.scenario.data_mode = "live_grpc"
    service = LivePreviewService(viz)

    with patch.object(service, "_launch_worker_process") as launch:
        assert service.set_enabled(True) is True
        service.handle_node_transform_event(
            {
                "phase": "changed",
                "kind": "tx",
                "index": 0,
                "position": (9.0, 8.0, 7.0),
            }
        )
        viz.scene_edit_service.edit_node_properties.assert_not_called()

        service.handle_node_transform_event(
            {
                "phase": "committed",
                "kind": "tx",
                "index": 0,
                "position": (9.0, 8.0, 7.0),
            }
        )

    launch.assert_not_called()
    assert service._worker_process is None
    assert not service._drag_timer.isActive()
    viz.scene_edit_service.edit_node_properties.assert_called_once_with(
        viz.tx_entries[0],
        {"position": [9.0, 8.0, 7.0]},
    )
    np.testing.assert_allclose(viz.current_tx_positions[0], [9.0, 8.0, 7.0])
    assert service.dirty_edit_count() == 1


def test_live_target_commit_serializes_canonical_rpc_degrees(qapp):
    viz = _make_viz()
    viz.frame_source = LiveGrpcSource("grpc://localhost:50051")
    viz.scenario.data_mode = "live_grpc"
    service = LivePreviewService(viz)
    assert service.set_enabled(True) is True

    service.handle_node_transform_event(
        {
            "phase": "selected",
            "kind": "target",
            "index": 0,
            "position": (9.0, 18.0, 27.0),
            "transform": np.eye(4, dtype=float).tolist(),
        }
    )
    yaw, pitch, roll = np.pi / 2.0, -np.pi / 4.0, np.pi
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = build_sionna_rotation_matrix(yaw, pitch, roll)
    transform[:3, 3] = [9.0, 18.0, 27.0]
    service.handle_node_transform_event(
        {
            "phase": "committed",
            "kind": "target",
            "index": 0,
            "position": (9.0, 18.0, 27.0),
            "transform": transform.tolist(),
        }
    )

    viz.scene_edit_service.edit_node_properties.assert_called_once()
    entry, values = viz.scene_edit_service.edit_node_properties.call_args.args
    assert entry is viz.target_entries[0]
    np.testing.assert_allclose(values["position"], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(values["orientation"], [90.0, -45.0, -180.0])
    assert all(-180.0 <= value < 180.0 for value in values["orientation"])


def test_rejected_live_commit_restores_last_generator_accepted_pose(qapp):
    viz = _make_viz()
    viz.frame_source = LiveGrpcSource("grpc://localhost:50051")
    viz.scenario.data_mode = "live_grpc"
    viz.scene_edit_service.edit_node_properties.side_effect = [True, False]
    service = LivePreviewService(viz)
    assert service.set_enabled(True) is True

    service.handle_node_transform_event(
        {
            "phase": "committed",
            "kind": "rx",
            "index": 0,
            "position": (4.0, 9.0, 6.0),
        }
    )
    service.handle_node_transform_event(
        {
            "phase": "committed",
            "kind": "rx",
            "index": 0,
            "position": (4.0, 12.0, 6.0),
        }
    )

    assert viz.scene_edit_service.edit_node_properties.call_count == 2
    np.testing.assert_allclose(viz.current_rx_positions[0], [4.0, 9.0, 6.0])
    assert service._edit_session.get("rx", 0).current.position == (4.0, 9.0, 6.0)
    assert "restored the last accepted pose" in viz._live_preview_status


def test_live_reset_sends_loaded_baseline_back_to_generator(qapp):
    viz = _make_viz()
    viz.frame_source = LiveGrpcSource("grpc://localhost:50051")
    viz.scenario.data_mode = "live_grpc"
    service = LivePreviewService(viz)
    assert service.set_enabled(True) is True
    service.handle_node_transform_event(
        {
            "phase": "committed",
            "kind": "tx",
            "index": 0,
            "position": (9.0, 8.0, 7.0),
        }
    )

    assert service.reset_edit("tx", 0) is True

    assert viz.scene_edit_service.edit_node_properties.call_count == 2
    assert viz.scene_edit_service.edit_node_properties.call_args.args[1] == {
        "position": [1.0, 2.0, 3.0]
    }
    np.testing.assert_allclose(viz.current_tx_positions[0], [1.0, 2.0, 3.0])
    assert service.dirty_edit_count() == 0


def test_worker_launch_is_bound_to_loaded_visualizer_source(qapp):
    viz = _make_viz()
    service = LivePreviewService(viz)
    proc = _FakeProcess()

    with patch(
        "visualizer.src.services.live_preview_service.subprocess.Popen",
        return_value=proc,
    ) as launch:
        assert service._launch_worker_process() is proc

    command = launch.call_args.args[0]
    assert Path(command[0]).samefile(sys.executable)
    assert command[1:3] == ("-I", "-u")
    assert str(service._source_identity.source_root) in command
    assert "visualizer.src.services.live_preview_worker" in command


def test_enable_acquires_transform_session_starts_worker_and_sends_init(qapp):
    viz, service, proc, launch, readers, cleanup = _service_with_worker(qapp)
    try:
        assert service.set_enabled(True) is True

        assert service.enabled is True
        assert viz._live_preview_enabled is True
        assert viz.renderer.callback.__self__ is service
        assert viz.renderer.callback.__func__ is service.handle_node_transform_event.__func__
        assert viz.renderer.session_begin_calls == 1
        assert viz.renderer.session_end_calls == 0
        launch.assert_called_once()
        readers.assert_called_once_with(proc)
        messages = _messages(proc)
        assert messages[0]["command"] == "init"
        assert messages[0]["request"]["scenario_root"] == str(Path("/tmp/live-preview-scenario"))
    finally:
        cleanup()


def test_worker_ready_rejects_a_different_source_tree(qapp, tmp_path):
    viz, service, _proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        assert service.set_enabled(True) is True
        wrong_identity = service._source_identity.to_dict()
        wrong_identity["source_root"] = str(tmp_path / "other-checkout")

        service._handle_worker_message(
            {
                "type": "ready",
                "source_identity": wrong_identity,
            }
        )

        assert service._worker_ready is False
        assert service._worker_process is None
        assert "does not match" in viz._live_preview_status
    finally:
        cleanup()


def test_worker_ready_rejects_missing_source_identity(qapp):
    viz, service, _proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        assert service.set_enabled(True) is True

        service._handle_worker_message({"type": "ready"})

        assert service._worker_ready is False
        assert service._worker_process is None
        assert "source identity is invalid" in viz._live_preview_status
    finally:
        cleanup()


def test_worker_result_before_source_identity_is_rejected(qapp, tmp_path):
    viz, service, _proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        assert service.set_enabled(True) is True
        output = tmp_path / "unverified.pkl"
        with output.open("wb") as handle:
            pickle.dump(_preview_frame(frame_index=0), handle)

        service._handle_worker_message(
            {
                "type": "result",
                "status": "ok",
                "sequence": 0,
                "output_path": str(output),
            }
        )

        assert viz._live_preview_frame is None
        assert service._worker_process is None
        assert "before source identity was verified" in viz._live_preview_status
    finally:
        cleanup()


def test_worker_result_path_must_match_service_owned_path(qapp, tmp_path):
    viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        assert service.set_enabled(True) is True
        _mark_worker_ready(service)
        proc.stdin.lines.clear()
        service._request_solve("release")
        request = _messages(proc)[0]["request"]
        unexpected = tmp_path / "unexpected.pkl"
        with unexpected.open("wb") as handle:
            pickle.dump(_preview_frame(frame_index=0), handle)

        service._handle_worker_message(
            {
                "type": "result",
                "status": "ok",
                "sequence": request["sequence"],
                "output_path": str(unexpected),
            }
        )

        assert viz._live_preview_frame is None
        assert service._worker_process is None
        assert "unexpected result path" in viz._live_preview_status
    finally:
        cleanup()


def test_first_init_uses_preview_default_after_eager_data_source_build(
    qapp,
    monkeypatch,
):
    """Hidden live-gRPC controls must not replace the local preview preset."""
    from visualizer.src.panels import data_source_panel as data_source_panel_module
    from visualizer.src.panels import nodes_panel as nodes_panel_module

    _restore_real_qt_labels(
        monkeypatch,
        data_source_panel_module,
        nodes_panel_module,
    )
    DataSourcePanel = data_source_panel_module.DataSourcePanel
    NodesSelectionPanel = nodes_panel_module.NodesSelectionPanel

    viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    viz.live_preview_service = service
    nodes_panel = NodesSelectionPanel(viz)
    preview_widget = nodes_panel.create_interactive_preview_panel()
    data_source_panel = DataSourcePanel(viz)
    data_source_widget = data_source_panel.create_panel()
    viz.ui_manager.panels = {
        "nodes": nodes_panel,
        "data_source": data_source_panel,
    }

    try:
        assert nodes_panel.widgets["rt_preset_combo"].currentText() == "ultra-low"
        assert viz.raytracing_settings_service.current_preset == "ultra-low"

        assert service.set_enabled(True) is True

        first_message = _messages(proc)[0]
        assert first_message["command"] == "init"
        assert first_message["request"]["solver_settings"] == {
            "max_depth": 2,
            "samples_per_src": 100000,
            "max_num_paths_per_src": 100000,
            "seed": 42,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": False,
            "refraction": False,
            "diffraction": False,
            "synthetic_array": True,
        }
    finally:
        if service.enabled:
            service.set_enabled(False)
        cleanup()
        preview_widget.deleteLater()
        data_source_widget.deleteLater()


def test_edit_panel_owns_live_raytracing_controls(qapp, monkeypatch):
    """Keep live solver mutations in Edit and runtime telemetry in Data Source."""
    from visualizer.src.panels import data_source_panel as data_source_panel_module
    from visualizer.src.panels import nodes_panel as nodes_panel_module

    _restore_real_qt_labels(monkeypatch, data_source_panel_module, nodes_panel_module)
    DataSourcePanel = data_source_panel_module.DataSourcePanel
    NodesSelectionPanel = nodes_panel_module.NodesSelectionPanel

    viz = _make_viz()
    viz.frame_source = LiveGrpcSource("grpc://localhost:50051")
    viz.scenario.data_mode = "live_grpc"
    viz.scenario.raytracing = {"quality": {"preset": "low"}}
    service = LivePreviewService(viz)
    viz.live_preview_service = service
    nodes_panel = NodesSelectionPanel(viz)
    edit_widget = nodes_panel.create_interactive_preview_panel()
    data_source_panel = DataSourcePanel(viz)
    data_source_widget = data_source_panel.create_panel()

    try:
        assert "rt_preset_combo" not in data_source_panel.widgets
        assert nodes_panel.widgets["rt_preset_combo"].currentText() == "low"
        assert not nodes_panel.widgets["rt_apply_btn"].isHidden()
        assert viz.raytracing_settings_service.current_preset == "low"
    finally:
        edit_widget.deleteLater()
        data_source_widget.deleteLater()


def test_transform_event_updates_position_and_throttles_drag_solve(qapp):
    viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        service.set_enabled(True)
        _mark_worker_ready(service)
        proc.stdin.lines.clear()

        with patch.object(
            viz.node_service,
            "update_tx_rx_positions",
            wraps=viz.node_service.update_tx_rx_positions,
        ) as publish_positions:
            service.handle_node_transform_event(
                {
                    "phase": "changed",
                    "kind": "tx",
                    "index": 0,
                    "position": (9.0, 8.0, 7.0),
                }
            )

        publish_positions.assert_called_once()
        np.testing.assert_allclose(viz.current_tx_positions[0], [9.0, 8.0, 7.0])
        np.testing.assert_allclose(
            viz.tx_markers[0].world_transform.matrix[:3, 3],
            [9.0, 8.0, 7.0],
        )
        viz.node_service.update_tx_rx_visibility()
        np.testing.assert_allclose(
            viz.renderer.objects["node:tx_0::marker"].transform.matrix[:3, 3],
            [9.0, 8.0, 7.0],
        )
        assert viz.renderer.set_named_transform_calls == []
        assert viz.renderer.named_position_reads == []
        assert service._drag_timer.isActive()
        assert _messages(proc) == []

        service._drag_timer.stop()
        service._flush_drag_preview_request()

        messages = _messages(proc)
        assert messages[0]["command"] == "solve"
        assert messages[0]["request"]["quality"] == "drag"
        assert messages[0]["request"]["solver_settings"]["max_depth"] == 2
        assert messages[0]["request"]["solver_settings"]["samples_per_src"] == 4096
        assert messages[0]["request"]["solver_settings"]["max_num_paths_per_src"] == 30000
        assert messages[0]["request"]["solver_settings"]["diffuse_reflection"] is False
    finally:
        cleanup()


def test_commit_event_sends_release_solve_immediately(qapp):
    _viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        service.set_enabled(True)
        _mark_worker_ready(service)
        proc.stdin.lines.clear()

        service.handle_node_transform_event(
            {
                "phase": "committed",
                "kind": "rx",
                "index": 0,
                "position": (4.0, 9.0, 6.0),
            }
        )

        messages = _messages(proc)
        assert messages[0]["command"] == "solve"
        assert messages[0]["request"]["quality"] == "release"
        assert messages[0]["request"]["solver_settings"]["max_depth"] == 3
        assert messages[0]["request"]["solver_settings"]["samples_per_src"] == 1_000_000
        assert messages[0]["request"]["solver_settings"]["max_num_paths_per_src"] == 500_000
        assert messages[0]["request"]["solver_settings"]["diffuse_reflection"] is False
    finally:
        cleanup()


def test_commit_event_cancels_pending_drag_preview(qapp):
    _viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        service.set_enabled(True)
        _mark_worker_ready(service)
        proc.stdin.lines.clear()

        service.handle_node_transform_event(
            {
                "phase": "changed",
                "kind": "tx",
                "index": 0,
                "position": (8.0, 8.0, 7.0),
            }
        )
        assert service._drag_timer.isActive()

        service.handle_node_transform_event(
            {
                "phase": "committed",
                "kind": "tx",
                "index": 0,
                "position": (9.0, 8.0, 7.0),
            }
        )

        assert not service._drag_timer.isActive()
        messages = _messages(proc)
        assert len(messages) == 1
        assert messages[0]["request"]["quality"] == "release"
    finally:
        cleanup()


def test_target_translation_event_updates_position_and_release_request(qapp):
    viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        service.set_enabled(True)
        _mark_worker_ready(service)
        proc.stdin.lines.clear()

        service.handle_node_transform_event(
            {
                "phase": "selected",
                "kind": "target",
                "index": 0,
                "position": (100.0, 200.0, 300.0),
                "transform": np.eye(4, dtype=float).tolist(),
            }
        )

        transform = np.eye(4, dtype=float)
        transform[:3, 3] = [102.0, 199.0, 303.0]
        with patch.object(
            viz.target_service,
            "sync_target_entry_snapshot",
            wraps=viz.target_service.sync_target_entry_snapshot,
        ) as sync_snapshot:
            service.handle_node_transform_event(
                {
                    "phase": "committed",
                    "kind": "target",
                    "index": 0,
                    "position": (102.0, 199.0, 303.0),
                    "transform": transform.tolist(),
                }
            )

        sync_snapshot.assert_called_once_with(viz.target_entries[0])
        np.testing.assert_allclose(viz.current_target_positions[0], [12.0, 19.0, 33.0])
        np.testing.assert_allclose(viz.current_target_orientations[0], [0.0, 0.0, 0.0])
        assert viz.target_entries[0]["position"] == [12.0, 19.0, 33.0]
        expected_transform = viz.target_entries[0]["mesh"].world_transform.matrix.copy()
        assert viz.target_service.sync_target_entry_snapshot(viz.target_entries[0]) is True
        np.testing.assert_allclose(
            viz.renderer.objects["target:walker::mesh"].transform.matrix,
            expected_transform,
        )
        assert viz.renderer.set_named_transform_calls == []
        assert viz.renderer.named_position_reads == []

        messages = _messages(proc)
        assert messages[0]["command"] == "solve"
        assert messages[0]["request"]["quality"] == "release"
        assert messages[0]["request"]["target_positions"] == [[12.0, 19.0, 33.0]]
    finally:
        cleanup()


def test_target_rotation_event_keeps_semantic_position_centered(qapp):
    viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        service.set_enabled(True)
        _mark_worker_ready(service)
        proc.stdin.lines.clear()

        service.handle_node_transform_event(
            {
                "phase": "selected",
                "kind": "target",
                "index": 0,
                "position": (9.0, 18.0, 27.0),
                "transform": np.eye(4, dtype=float).tolist(),
            }
        )

        yaw, pitch, roll = 0.5, 0.1, -0.2
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = build_sionna_rotation_matrix(yaw, pitch, roll)
        transform[:3, 3] = [9.0, 18.0, 27.0]
        service.handle_node_transform_event(
            {
                "phase": "committed",
                "kind": "target",
                "index": 0,
                "position": (9.0, 18.0, 27.0),
                "transform": transform.tolist(),
            }
        )

        np.testing.assert_allclose(viz.current_target_positions[0], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(
            viz.current_target_orientations[0],
            [yaw, pitch, roll],
            atol=1e-6,
        )
        assert viz.target_entries[0]["position"] == [10.0, 20.0, 30.0]
        renderer_transform = viz.target_entries[0]["mesh"].world_transform.matrix
        target_asset = viz.target_asset_cache.asset_for_logical_key(("walker", "walker.ply"))
        assert isinstance(target_asset, TargetAsset)
        center = rotated_aabb_center(
            target_asset.scaled_vertices,
            build_sionna_rotation_matrix(yaw, pitch, roll),
        )
        final_center = renderer_transform[:3, 3] + center
        np.testing.assert_allclose(final_center, [10.0, 20.0, 30.0], atol=1e-6)
        assert viz.renderer.active_target_pose_syncs == []

        messages = _messages(proc)
        assert messages[0]["command"] == "solve"
        assert messages[0]["request"]["quality"] == "release"
        assert messages[0]["request"]["target_positions"] == [[10.0, 20.0, 30.0]]
        np.testing.assert_allclose(
            messages[0]["request"]["target_orientations"],
            [[yaw, pitch, roll]],
            atol=1e-6,
        )
    finally:
        cleanup()


def test_target_move_then_rotate_then_move_keeps_committed_pose(qapp):
    viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        service.set_enabled(True)
        _mark_worker_ready(service)
        proc.stdin.lines.clear()

        service.handle_node_transform_event(
            {
                "phase": "selected",
                "kind": "target",
                "index": 0,
                "position": (9.0, 18.0, 27.0),
                "transform": np.eye(4, dtype=float).tolist(),
            }
        )

        moved_transform = np.eye(4, dtype=float)
        moved_transform[:3, 3] = [11.0, 20.0, 27.0]
        service.handle_node_transform_event(
            {
                "phase": "committed",
                "kind": "target",
                "index": 0,
                "position": (11.0, 20.0, 27.0),
                "transform": moved_transform.tolist(),
            }
        )
        np.testing.assert_allclose(viz.current_target_positions[0], [12.0, 22.0, 30.0])
        service._worker_busy = False

        yaw, pitch, roll = 0.5, 0.1, -0.2
        rotated_transform = np.eye(4, dtype=float)
        rotated_transform[:3, :3] = build_sionna_rotation_matrix(yaw, pitch, roll)
        rotated_transform[:3, 3] = [11.0, 20.0, 27.0]
        service.handle_node_transform_event(
            {
                "phase": "committed",
                "kind": "target",
                "index": 0,
                "position": (11.0, 20.0, 27.0),
                "transform": rotated_transform.tolist(),
            }
        )

        np.testing.assert_allclose(viz.current_target_positions[0], [12.0, 22.0, 30.0])
        np.testing.assert_allclose(
            viz.current_target_orientations[0],
            [yaw, pitch, roll],
            atol=1e-6,
        )
        assert viz.renderer.active_target_pose_syncs == []
        service._worker_busy = False

        moved_again_transform = np.array(rotated_transform, dtype=float, copy=True)
        moved_again_transform[:3, 3] = [13.0, 20.0, 27.0]
        service.handle_node_transform_event(
            {
                "phase": "committed",
                "kind": "target",
                "index": 0,
                "position": (13.0, 20.0, 27.0),
                "transform": moved_again_transform.tolist(),
            }
        )

        np.testing.assert_allclose(viz.current_target_positions[0], [14.0, 22.0, 30.0])
        np.testing.assert_allclose(
            viz.current_target_orientations[0],
            [yaw, pitch, roll],
            atol=1e-6,
        )
        renderer_transform = viz.target_entries[0]["mesh"].world_transform.matrix
        target_asset = viz.target_asset_cache.asset_for_logical_key(("walker", "walker.ply"))
        assert isinstance(target_asset, TargetAsset)
        center = rotated_aabb_center(
            target_asset.scaled_vertices,
            build_sionna_rotation_matrix(yaw, pitch, roll),
        )
        np.testing.assert_allclose(renderer_transform[:3, 3] + center, [14.0, 22.0, 30.0])
    finally:
        cleanup()


def test_target_drag_rotation_updates_mesh_without_syncing_active_proxy(qapp):
    viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        service.set_enabled(True)
        _mark_worker_ready(service)
        proc.stdin.lines.clear()

        service.handle_node_transform_event(
            {
                "phase": "selected",
                "kind": "target",
                "index": 0,
                "position": (9.0, 18.0, 27.0),
                "transform": np.eye(4, dtype=float).tolist(),
            }
        )

        yaw, pitch, roll = 0.35, -0.15, 0.25
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = build_sionna_rotation_matrix(yaw, pitch, roll)
        transform[:3, 3] = [9.0, 18.0, 27.0]
        service.handle_node_transform_event(
            {
                "phase": "changed",
                "kind": "target",
                "index": 0,
                "position": (9.0, 18.0, 27.0),
                "transform": transform.tolist(),
            }
        )

        renderer_transform = viz.target_entries[0]["mesh"].world_transform.matrix
        np.testing.assert_allclose(renderer_transform[:3, :3], transform[:3, :3], atol=1e-6)
        assert viz.renderer.active_target_pose_syncs == []
        assert service._drag_timer.isActive()
    finally:
        cleanup()


def test_preview_frame_maps_dirty_target_to_reordered_metadata_by_identity(qapp):
    viz = _make_viz()
    service = LivePreviewService(viz)

    service.handle_node_transform_event({"kind": "target", "index": 0, "position": (0, 0, 0)})
    assert service.dirty_edit_count() == 0

    service._enabled = True
    service.handle_node_transform_event(
        {
            "phase": "selected",
            "kind": "target",
            "index": 0,
            "position": (9.0, 18.0, 27.0),
            "transform": np.eye(4, dtype=float).tolist(),
        }
    )
    yaw, pitch, roll = 0.35, -0.15, 0.25
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = build_sionna_rotation_matrix(yaw, pitch, roll)
    transform[:3, 3] = [9.0, 18.0, 27.0]
    service.handle_node_transform_event(
        {
            "phase": "changed",
            "kind": "target",
            "index": 0,
            "position": (9.0, 18.0, 27.0),
            "transform": transform.tolist(),
        }
    )
    edit = service._edit_session.get("target", 0)
    assert edit is not None
    assert "walker" in edit.identity_aliases

    frame = _preview_frame(
        target_positions=np.asarray(
            [[40.0, 50.0, 60.0], [1.0, 2.0, 3.0]],
            dtype=np.float64,
        ),
        targets_metadata=(
            {
                "name": "drone",
                "current_position": [40.0, 50.0, 60.0],
                "orientation": [0.1, 0.2, 0.3],
            },
            {
                "name": "walker",
                "current_position": [1.0, 2.0, 3.0],
                "orientation": [0.0, 0.0, 0.0],
            },
        ),
    )

    updated = service._apply_interactive_target_pose_overrides(frame)

    assert updated.targets_metadata[0] == {
        "name": "drone",
        "current_position": [40.0, 50.0, 60.0],
        "orientation": [0.1, 0.2, 0.3],
    }
    np.testing.assert_allclose(
        updated.targets_metadata[1]["orientation"],
        [yaw, pitch, roll],
    )
    np.testing.assert_allclose(
        updated.targets_metadata[1]["current_position"],
        [10.0, 20.0, 30.0],
    )
    np.testing.assert_allclose(updated.target_positions_m[0], [40.0, 50.0, 60.0])
    np.testing.assert_allclose(updated.target_positions_m[1], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(frame.target_positions_m[1], [1.0, 2.0, 3.0])


def test_preview_frame_maps_sparse_canonical_target_to_its_only_frame_row(qapp):
    viz = _make_viz()
    service = LivePreviewService(viz)
    service._edit_session.record(
        kind="target",
        index=3,
        position=[70.0, 80.0, 90.0],
        orientation=[0.4, 0.5, 0.6],
        baseline_position=[7.0, 8.0, 9.0],
        baseline_orientation=[0.0, 0.0, 0.0],
        identity_aliases={"drone", "target-drone-uid"},
    )
    frame = _preview_frame(
        target_positions=np.asarray([[7.0, 8.0, 9.0]], dtype=np.float64),
        targets_metadata=(
            {
                "stable_target_id": "target-drone-uid",
                "name": "drone",
                "current_position": [7.0, 8.0, 9.0],
                "orientation": [0.0, 0.0, 0.0],
            },
        ),
    )

    updated = service._apply_interactive_target_pose_overrides(frame)

    np.testing.assert_allclose(
        updated.targets_metadata[0]["current_position"],
        [70.0, 80.0, 90.0],
    )
    np.testing.assert_allclose(
        updated.targets_metadata[0]["orientation"],
        [0.4, 0.5, 0.6],
    )
    np.testing.assert_allclose(updated.target_positions_m[0], [70.0, 80.0, 90.0])


def test_preview_frame_missing_edited_target_does_not_mutate_another_target(qapp):
    viz = _make_viz()
    service = LivePreviewService(viz)
    service._edit_session.record(
        kind="target",
        index=0,
        position=[70.0, 80.0, 90.0],
        orientation=[0.4, 0.5, 0.6],
        baseline_position=[10.0, 20.0, 30.0],
        baseline_orientation=[0.0, 0.0, 0.0],
        identity_aliases={"walker"},
    )
    frame = _preview_frame(
        target_positions=np.asarray([[7.0, 8.0, 9.0]], dtype=np.float64),
        targets_metadata=(
            {
                "name": "drone",
                "current_position": [7.0, 8.0, 9.0],
                "orientation": [0.1, 0.2, 0.3],
            },
        ),
    )
    expected_metadata = tuple(dict(item) for item in frame.targets_metadata)
    expected_positions = np.array(frame.target_positions_m, copy=True)

    updated = service._apply_interactive_target_pose_overrides(frame)

    assert updated is frame
    assert frame.targets_metadata == expected_metadata
    np.testing.assert_array_equal(frame.target_positions_m, expected_positions)


def test_reset_selected_edit_restores_original_tx_position(qapp):
    viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        service.set_enabled(True)
        _mark_worker_ready(service)
        proc.stdin.lines.clear()
        viz.renderer.active_transform_target = {
            "object_id": "node:tx_0::marker",
            "kind": "tx",
            "index": 0,
        }

        service.handle_node_transform_event(
            {
                "phase": "committed",
                "kind": "tx",
                "index": 0,
                "position": (9.0, 8.0, 7.0),
            }
        )
        assert service.dirty_edit_count() == 1
        proc.stdin.lines.clear()
        service._worker_busy = False

        assert service.reset_selected_edit() is True

        np.testing.assert_allclose(viz.current_tx_positions[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            viz.tx_markers[0].world_transform.matrix[:3, 3],
            [1.0, 2.0, 3.0],
        )
        assert service.dirty_edit_count() == 0
        assert _messages(proc)[0]["request"]["quality"] == "release"
    finally:
        cleanup()


def test_reset_all_edits_restores_target_position_and_orientation(qapp):
    viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        service.set_enabled(True)
        _mark_worker_ready(service)
        proc.stdin.lines.clear()

        service.handle_node_transform_event(
            {
                "phase": "selected",
                "kind": "target",
                "index": 0,
                "position": (9.0, 18.0, 27.0),
                "transform": np.eye(4, dtype=float).tolist(),
            }
        )
        transform = np.eye(4, dtype=float)
        transform[:3, 3] = [11.0, 20.0, 27.0]
        service.handle_node_transform_event(
            {
                "phase": "committed",
                "kind": "target",
                "index": 0,
                "position": (11.0, 20.0, 27.0),
                "transform": transform.tolist(),
            }
        )
        assert service.dirty_edit_count() == 1
        proc.stdin.lines.clear()
        service._worker_busy = False

        assert service.reset_all_edits() is True

        np.testing.assert_allclose(viz.current_target_positions[0], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(viz.current_target_orientations[0], [0.0, 0.0, 0.0])
        assert viz.target_entries[0]["position"] == [10.0, 20.0, 30.0]
        assert service.dirty_edit_count() == 0
        assert _messages(proc)[0]["request"]["quality"] == "release"
    finally:
        cleanup()


def test_live_preview_uses_shared_custom_settings_for_drag_and_release(qapp):
    viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        viz.raytracing_settings_service.set_custom(
            {
                "max_depth": 5,
                "samples_per_src": 500000,
                "max_num_paths_per_src": 600000,
                "seed": 123,
                "los": True,
                "specular_reflection": True,
                "diffuse_reflection": False,
                "refraction": True,
                "diffraction": False,
                "synthetic_array": True,
            }
        )
        service.set_enabled(True)
        _mark_worker_ready(service)
        proc.stdin.lines.clear()

        service._request_solve("drag")
        drag_message = _messages(proc)[0]
        drag_output = Path(drag_message["request"]["output_path"])
        with drag_output.open("wb") as handle:
            pickle.dump(_preview_frame(), handle)
        service._handle_worker_message(
            {
                "type": "result",
                "status": "ok",
                "sequence": service._requested_sequence,
                "output_path": str(drag_output),
            }
        )
        service._request_solve("release")

        drag_settings = _messages(proc)[0]["request"]["solver_settings"]
        release_settings = _messages(proc)[1]["request"]["solver_settings"]

        assert drag_settings["max_depth"] == 2
        assert drag_settings["samples_per_src"] == 4096
        assert drag_settings["max_num_paths_per_src"] == 30000
        assert drag_settings["seed"] == 123
        assert drag_settings["diffuse_reflection"] is False
        assert drag_settings["refraction"] is True

        assert release_settings["max_depth"] == 5
        assert release_settings["samples_per_src"] == 500000
        assert release_settings["max_num_paths_per_src"] == 600000
        assert release_settings["seed"] == 123
        assert release_settings["diffuse_reflection"] is False
        assert release_settings["refraction"] is True
    finally:
        cleanup()


def test_busy_worker_queues_release_and_discards_old_result(qapp):
    viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        service.set_enabled(True)
        _mark_worker_ready(service)
        proc.stdin.lines.clear()

        service._request_solve("drag")
        first_sequence = service._requested_sequence
        service._request_solve("release")
        assert service._pending_profile == "release"

        old_output = Path(proc.stdin.lines[-1] and _messages(proc)[0]["request"]["output_path"])
        with old_output.open("wb") as handle:
            pickle.dump(_preview_frame(), handle)
        service._handle_worker_message(
            {
                "type": "result",
                "status": "ok",
                "sequence": first_sequence,
                "output_path": str(old_output),
            }
        )

        assert viz._live_preview_frame is None
        messages = _messages(proc)
        assert messages[-1]["request"]["quality"] == "release"
    finally:
        cleanup()


def test_worker_result_applies_latest_preview_frame(qapp):
    viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        service.set_enabled(True)
        _mark_worker_ready(service)
        proc.stdin.lines.clear()

        service._request_solve("release")
        message = _messages(proc)[0]
        output = Path(message["request"]["output_path"])
        frame = _preview_frame()
        with output.open("wb") as handle:
            pickle.dump(frame, handle)

        service._handle_worker_message(
            {
                "type": "result",
                "status": "ok",
                "sequence": message["request"]["sequence"],
                "output_path": str(output),
            }
        )

        assert viz._live_preview_frame["preview"] is True
        assert viz._live_preview_frame["_source"]["provider"] == "live_preview"
        assert viz._live_preview_frame["canonical_data"] is not None
        assert viz._live_preview_step == 7
        assert viz.update_calls == 1
    finally:
        cleanup()


def test_apply_preview_frame_invalidates_canonical_step(qapp):
    viz = _make_viz()
    service = LivePreviewService(viz)
    frame = _preview_frame()

    service._apply_preview_frame(frame, sequence=3)

    assert viz._live_preview_frame["preview"] is True
    assert viz._live_preview_frame["canonical_data"] is not None
    assert viz._live_preview_step == 7
    assert viz._live_preview_sequence == 3
    viz.cache_service.invalidate_canonical_step.assert_called_once_with(
        7,
        reason="live_preview",
    )
    assert viz.update_calls == 1


def test_worker_exit_reports_failure_without_applying_frame(qapp):
    viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        service.set_enabled(True)
        _mark_worker_ready(service)
        proc.returncode = -11

        service._poll_worker_messages()

        assert viz._live_preview_frame is None
        assert "Preview failed" in viz._live_preview_status
        assert "code -11" in viz._live_preview_status
    finally:
        cleanup()


def test_disable_clears_preview_frame_releases_session_and_stops_worker(qapp):
    viz, service, proc, _launch, _readers, cleanup = _service_with_worker(qapp)
    try:
        service.set_enabled(True)
        viz._live_preview_frame = {"preview": True}
        viz._live_preview_step = 7

        assert service.set_enabled(False) is True

        assert service.enabled is False
        assert viz.renderer.callback is None
        assert viz.renderer.session_begin_calls == 1
        assert viz.renderer.session_end_calls == 1
        assert viz._live_preview_frame is None
        assert viz._live_preview_step is None
        assert viz.cache_service.invalidate.called
        assert proc.stdin.closed is True
    finally:
        cleanup()


def test_build_live_overrides_uses_config_names():
    simulation = SimpleNamespace(
        tx_configs=[SimpleNamespace(name="tx-main")],
        rx_configs=[SimpleNamespace(name="rx-main")],
        target_configs=[SimpleNamespace(name="walker")],
    )

    overrides = build_live_overrides(
        simulation,
        np.asarray([[1.0, 2.0, 3.0]], dtype=float),
        np.asarray([[4.0, 5.0, 6.0]], dtype=float),
        np.asarray([[7.0, 8.0, 9.0]], dtype=float),
        np.asarray([[np.pi / 2.0, -np.pi / 4.0, np.pi / 6.0]], dtype=float),
    )

    assert overrides[:2] == [
        {"name": "tx-main", "category": "tx", "position": (1.0, 2.0, 3.0)},
        {"name": "rx-main", "category": "rx", "position": (4.0, 5.0, 6.0)},
    ]
    assert overrides[2]["name"] == "walker"
    assert overrides[2]["category"] == "target"
    assert overrides[2]["position"] == (7.0, 8.0, 9.0)
    np.testing.assert_allclose(overrides[2]["orientation"], (90.0, -45.0, 30.0))
