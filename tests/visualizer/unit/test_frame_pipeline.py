"""Unit tests for FramePipeline."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from generator.io.storage.coverage_writer import save_coverage_hdf5
from shared.frames.types import StandardMPCFrame
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from visualizer.src.benchmarking.recorder import BenchmarkRecorder
from visualizer.src.pipeline.core import ViewModel
from visualizer.src.pipeline.frame_pipeline import FramePipeline
from visualizer.src.scene.surface_payloads import BeamformingSurface
from visualizer.src.services.animation_service import AnimationService
from visualizer.src.services.cache_service import CacheService
from visualizer.src.services.coverage_service import CoverageService
from visualizer.src.state import MpcVisibility, create_initial_state
from visualizer.src.types.render_payloads import MeshPayload


def _make_beamforming_surface(surface_id: str) -> BeamformingSurface:
    return BeamformingSurface(
        id=surface_id,
        payload=MeshPayload(
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )


def _make_view_model(**overrides):
    defaults = dict(
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[1.0, 0.0, 0.0]]),
        tx_orientations=np.zeros((1, 3)),
        rx_orientations=np.zeros((1, 3)),
        mpc_points=np.zeros((0, 3), dtype=np.float32),
        mpc_lines=np.zeros((0, 2), dtype=np.int32),
        mpc_colors=np.zeros((0, 3), dtype=np.float32),
        colorbar=None,
        stats_text="stats",
        mpc_visibility=MpcVisibility(),
        target_positions=np.empty((0, 3)),
        target_orientations=np.empty((0, 3)),
        target_mesh_files=[],
        target_use_ply_positions=[],
        target_metadata=[],
    )
    defaults.update(overrides)
    return ViewModel(**defaults)


def _make_coverage_data(
    values: np.ndarray | None = None,
    *,
    metric_name: str = "sinr_db",
) -> dict[str, Any]:
    """Return one small valid coverage dataset for pipeline behavior tests."""
    values_3d = (
        np.asarray([[[0.0, 1.0]]], dtype=np.float32)
        if values is None
        else np.asarray(values, dtype=np.float32)
    )
    nz, ny, nx = values_3d.shape
    return {
        "grid_origin": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.asarray([nx, ny, nz], dtype=np.int32),
        "value_min": float(np.nanmin(values_3d)),
        "value_max": float(np.nanmax(values_3d)),
        "metric_name": metric_name,
        "values_3d": values_3d,
        "heights": [float(index) for index in range(nz)],
    }


class DummyMpcCore:
    """Simple stand-in for MPCCore that records create_view_model calls."""

    def __init__(self):
        self.calls = 0
        self.last_kwargs = {}
        self.return_value = _make_view_model(
            mpc_points=np.array([[0.0, 0.0, 0.0]]),
        )
        self.breakdown = {
            "canonical_lookup_ms": 1.25,
            "filter_ms": 2.5,
            "canonical_cache_hit": 1.0,
            "mpc_visible_segments_count": 7.0,
        }

    def create_view_model(self, **kwargs: dict):
        self.calls += 1
        self.last_kwargs = kwargs
        return replace(
            self.return_value,
            mpc_visibility=kwargs.get("mpc_visibility", self.return_value.mpc_visibility),
        )

    def stats(self, payload):
        return payload

    def get_last_viewmodel_breakdown(self):
        return dict(self.breakdown)


@pytest.fixture
def dummy_visualizer():
    vis = SimpleNamespace()
    vis.ready = True
    vis.app_state = create_initial_state(step=0)
    vis.last_app_state = None
    vis.renderer = SimpleNamespace(
        apply_frame=lambda packet: True,
        begin_frame_update=lambda: None,
        end_frame_update=lambda: True,
    )
    vis.force_update_next_frame = False
    vis.update_calls = 0
    vis.metrics_window = None
    vis.frame_times = []
    vis.last_frame_duration_ms = 0.0
    vis._scene_boot_start = None
    vis._scene_boot_logged = True
    vis.show_tx_orientation = False
    vis.show_rx_orientation = False
    vis.show_target_orientation = False
    vis.mpc_core = DummyMpcCore()
    vis.mpc_view_cache = OrderedDict()
    vis.mpc_allowed_materials = None
    vis.show_tx_segments = True
    vis.use_preload_mode = False
    vis._live_preview_frame = None
    vis._live_preview_step = None
    vis._live_preview_sequence = None
    vis.coverage_data = None
    vis.coverage_opacity = 1.0
    vis.coverage_height_index = 0
    vis.coverage_heights = []
    vis.coverage_interpolation_method = "nearest"
    vis.set_state = lambda **changes: setattr(vis, "last_state_update", changes)
    vis.mpc_allowed_materials = None
    vis.mpc_allowed_materials = None
    vis._ensure_tx_rx_markers_created = lambda tx, rx: setattr(vis, "last_marker_counts", (tx, rx))
    vis.node_service = SimpleNamespace()
    vis.node_service.ensure_tx_rx_markers_created = lambda tx, rx: setattr(
        vis, "last_marker_counts", (tx, rx)
    )
    vis.node_service.update_available_tx_rx_from_frame = lambda tx, rx: None
    vis.node_service.retry_pending_node_syncs = lambda: True
    vis.node_service.create_orientation_frames = lambda step: True
    vis.target_service = SimpleNamespace(
        process_targets_from_view_model=lambda step, view_model: True
    )
    vis.tx_markers = []
    vis.rx_markers = []
    vis.node_service.update_tx_rx_positions = lambda tx, rx: (
        setattr(
            vis,
            "last_tx_rx_position_update",
            (np.asarray(tx), np.asarray(rx)),
        )
        or True
    )
    vis.cache_service = CacheService(vis)

    def schedule_update():
        vis.update_calls += 1

    vis.schedule_update = schedule_update
    vis.beamforming_ui_controller = SimpleNamespace(
        apply_selector_state=lambda: None,
        fail_computation=lambda _reason: None,
        set_frame_beamforming_available=lambda _available: None,
        update_node_options=lambda _info, _pairs=None: None,
        update_standalone_buttons_state=lambda: None,
    )
    vis.ui_controller = SimpleNamespace(
        update_frame_context=lambda *args, **kwargs: None,
        handle_frame_timing_update=lambda _step, _elapsed: None,
    )
    return vis


def test_update_early_exit_when_not_ready(dummy_visualizer):
    """Pipeline should skip processing when the visualizer reports not ready."""
    dummy_visualizer.ready = False
    pipeline = FramePipeline(dummy_visualizer)

    load_called = False

    def fake_load(step):
        nonlocal load_called
        load_called = True
        return {}

    pipeline.load_frame = fake_load

    assert pipeline.update(0) is False

    assert load_called is False
    assert dummy_visualizer.update_calls == 0


def test_update_processes_frame_and_calls_renderer(dummy_visualizer, monkeypatch):
    """Happy-path update should project a frame packet and call the renderer."""
    pipeline = FramePipeline(dummy_visualizer)
    frame_payload = {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
    }
    pipeline.load_frame = lambda step: frame_payload

    view_model = _make_view_model(
        tx_positions=np.array([[2.0, 0.0, 0.0]]),
        rx_positions=np.array([[3.0, 0.0, 0.0]]),
    )
    pipeline._derive_view_model = lambda step, frame: view_model

    applied = {}

    def fake_apply(packet) -> bool:
        applied["packet"] = packet
        return True

    dummy_visualizer.renderer.apply_frame = fake_apply

    assert pipeline.update(0) is True

    assert applied["packet"].mpc_points is view_model.mpc_points
    assert dummy_visualizer.current_view_model is view_model
    np.testing.assert_array_equal(dummy_visualizer.current_tx_positions, view_model.tx_positions)
    np.testing.assert_array_equal(dummy_visualizer.current_rx_positions, view_model.rx_positions)
    np.testing.assert_array_equal(
        dummy_visualizer.current_tx_orientations,
        view_model.tx_orientations,
    )
    np.testing.assert_array_equal(
        dummy_visualizer.current_rx_orientations,
        view_model.rx_orientations,
    )


def test_closed_mpc_explorer_has_only_nullable_accepted_frame_check(dummy_visualizer, monkeypatch):
    """A closed Explorer must not enter its notification helper at all."""
    pipeline = FramePipeline(dummy_visualizer)
    pipeline.load_frame = lambda _step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
    }
    pipeline._derive_view_model = lambda _step, _frame: _make_view_model()
    publish_calls = []
    monkeypatch.setattr(
        pipeline,
        "_publish_mpc_explorer_presented_frame",
        lambda **kwargs: publish_calls.append(kwargs),
    )

    assert pipeline.update(0) is True
    assert publish_calls == []


def test_mpc_explorer_receives_only_renderer_accepted_frames(dummy_visualizer):
    """Explorer delivery occurs after end_frame_update accepts the transaction."""
    pipeline = FramePipeline(dummy_visualizer)
    view_model = _make_view_model()
    pipeline.load_frame = lambda _step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
    }
    pipeline._derive_view_model = lambda _step, _frame: view_model
    accepted = iter((False, True))
    dummy_visualizer.renderer.end_frame_update = lambda: next(accepted)
    deliveries = []

    def callback(*args):
        deliveries.append(args)

    pipeline.set_mpc_explorer_presented_callback(callback)

    assert pipeline.update(0) is False
    assert deliveries == []

    assert pipeline.update(0) is True
    assert len(deliveries) == 1
    source_epoch, step, delivered_view_model, render_packet = deliveries[0]
    assert source_epoch == 0
    assert step == 0
    assert delivered_view_model is view_model
    assert render_packet.canonical_data is view_model.canonical_data

    pipeline.clear_mpc_explorer_presented_callback(callback)
    assert pipeline._mpc_explorer_presented_callback is None


def test_paths_type_legend_receives_only_renderer_accepted_frame_codes(dummy_visualizer):
    """The shared panel must not publish a frame that submission rejected."""
    deliveries: list[tuple[int, ...]] = []
    panel = SimpleNamespace(set_present_mpc_type_codes=deliveries.append)
    dummy_visualizer.ui_manager = SimpleNamespace(panels={"mpc": panel})
    view_model = _make_view_model(
        mpc_line_itypes=np.array([2, 99], dtype=np.uint8),
        mpc_line_itype_codes=(2, 99),
    )
    empty_view_model = _make_view_model()
    pipeline = FramePipeline(dummy_visualizer)
    pipeline.load_frame = lambda _step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
    }
    pipeline._derive_view_model = lambda step, _frame: (
        view_model if step == 0 else empty_view_model
    )
    submissions = iter((False, True, True))
    dummy_visualizer.renderer.end_frame_update = lambda: next(submissions)

    assert pipeline.update(0) is False
    assert deliveries == []

    assert pipeline.update(0) is True
    assert deliveries == [(2, 99)]

    dummy_visualizer.app_state = replace(dummy_visualizer.app_state, step=1)
    assert pipeline.update(1) is True
    assert deliveries == [(2, 99), ()]


def test_empty_node_frame_replaces_cached_anchors_and_triggers_sync(dummy_visualizer) -> None:
    """A missing-anchor frame hides stable inventory nodes instead of reusing stale data."""
    pipeline = FramePipeline(dummy_visualizer)
    nonempty = _make_view_model(
        tx_positions=np.asarray([[2.0, 0.0, 0.0]]),
        rx_positions=np.asarray([[3.0, 0.0, 0.0]]),
    )
    empty = _make_view_model(
        tx_positions=np.empty((0, 3)),
        rx_positions=np.empty((0, 3)),
        tx_orientations=np.empty((0, 3)),
        rx_orientations=np.empty((0, 3)),
    )
    pipeline.load_frame = lambda step: {
        "num_tx": 1 if step == 0 else 0,
        "num_rx": 1 if step == 0 else 0,
    }
    pipeline._derive_view_model = lambda step, _frame: nonempty if step == 0 else empty
    dummy_visualizer.tx_markers = [object()]
    dummy_visualizer.rx_markers = [object()]
    syncs: list[tuple[np.ndarray, np.ndarray]] = []
    dummy_visualizer.node_service.update_tx_rx_positions = lambda tx, rx: (
        syncs.append((np.asarray(tx), np.asarray(rx))) or True
    )

    dummy_visualizer.force_update_next_frame = True
    assert pipeline.update(0)
    dummy_visualizer.force_update_next_frame = True
    assert pipeline.update(1)

    assert dummy_visualizer.current_tx_positions.shape == (0, 3)
    assert dummy_visualizer.current_rx_positions.shape == (0, 3)
    assert len(syncs) == 2
    assert syncs[-1][0].shape == (0, 3)
    assert syncs[-1][1].shape == (0, 3)


def test_failed_frame_submission_is_not_recorded_as_completed(dummy_visualizer):
    pipeline = FramePipeline(dummy_visualizer)
    pipeline.load_frame = lambda _step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
    }
    pipeline._derive_view_model = lambda _step, _frame: _make_view_model()
    events = []
    dummy_visualizer.renderer.begin_frame_update = lambda: events.append("begin")
    dummy_visualizer.renderer.end_frame_update = lambda: events.append("end") or False
    beam_failures: list[str] = []
    dummy_visualizer.beamforming_ui_controller.fail_computation = beam_failures.append

    assert pipeline.update(0) is False

    assert events == ["begin", "end"]
    assert beam_failures == ["Renderer did not present the frame update"]
    assert dummy_visualizer.frame_times == []
    assert dummy_visualizer.last_app_state is None


def test_rejected_frame_apply_retries_without_committing_app_state(dummy_visualizer):
    """A transient renderer rejection must not suppress an identical retry."""
    pipeline = FramePipeline(dummy_visualizer)
    pipeline.load_frame = lambda _step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
    }
    pipeline._derive_view_model = lambda _step, _frame: _make_view_model()
    apply_results = iter((False, True))
    apply_calls: list[Any] = []

    def apply_frame(packet) -> bool:
        apply_calls.append(packet)
        return next(apply_results)

    dummy_visualizer.renderer.apply_frame = apply_frame
    beam_failures: list[str] = []
    dummy_visualizer.beamforming_ui_controller.fail_computation = beam_failures.append

    assert pipeline.update(0) is False
    assert dummy_visualizer.last_app_state is None
    assert beam_failures == ["Renderer rejected the frame update"]

    assert pipeline.update(0) is True
    assert len(apply_calls) == 2
    assert dummy_visualizer.last_app_state is dummy_visualizer.app_state


def test_failed_node_entity_sync_retries_without_committing_app_state(dummy_visualizer):
    """A failed persistent entity sync must be retried on an identical frame."""
    pipeline = FramePipeline(dummy_visualizer)
    view_model = _make_view_model()
    pipeline.load_frame = lambda _step: {
        "tx_positions": view_model.tx_positions,
        "rx_positions": view_model.rx_positions,
    }
    pipeline._derive_view_model = lambda _step, _frame: view_model
    dummy_visualizer.tx_markers = [object()]
    initial_sync_calls: list[tuple[np.ndarray, np.ndarray]] = []
    retry_calls: list[bool] = []

    def reject_initial_sync(tx, rx) -> bool:
        initial_sync_calls.append((np.asarray(tx), np.asarray(rx)))
        return False

    dummy_visualizer.node_service.update_tx_rx_positions = reject_initial_sync
    dummy_visualizer.node_service.retry_pending_node_syncs = lambda: (
        retry_calls.append(True) or True
    )

    assert pipeline.update(0) is False
    assert dummy_visualizer.last_app_state is None

    assert pipeline.update(0) is True
    assert len(initial_sync_calls) == 1
    assert retry_calls == [True]
    assert dummy_visualizer.last_app_state is dummy_visualizer.app_state


@pytest.mark.parametrize("domain", ["target", "orientation"])
def test_failed_persistent_entity_domain_retries_without_committing_app_state(
    dummy_visualizer,
    domain,
):
    pipeline = FramePipeline(dummy_visualizer)
    view_model = _make_view_model()
    pipeline.load_frame = lambda _step: {
        "tx_positions": view_model.tx_positions,
        "rx_positions": view_model.rx_positions,
    }
    pipeline._derive_view_model = lambda _step, _frame: view_model
    sync_results = iter((False, True))
    sync_calls: list[int] = []

    def sync_domain(_step, *_args) -> bool:
        sync_calls.append(_step)
        return next(sync_results)

    if domain == "target":
        dummy_visualizer.target_entries = [{}]
        dummy_visualizer.target_service.process_targets_from_view_model = sync_domain
    else:
        dummy_visualizer.show_tx_orientation = True
        dummy_visualizer.node_service.create_orientation_frames = sync_domain

    assert pipeline.update(0) is False
    assert dummy_visualizer.last_app_state is None

    assert pipeline.update(0) is True
    assert sync_calls == [0, 0]
    assert dummy_visualizer.last_app_state is dummy_visualizer.app_state


def test_non_boolean_frame_apply_is_rejected(dummy_visualizer):
    pipeline = FramePipeline(dummy_visualizer)
    pipeline.load_frame = lambda _step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
    }
    pipeline._derive_view_model = lambda _step, _frame: _make_view_model()
    dummy_visualizer.renderer.apply_frame = lambda _packet: None

    assert pipeline.update(0) is False
    assert dummy_visualizer.last_app_state is None


def test_non_boolean_frame_submission_is_rejected(dummy_visualizer):
    pipeline = FramePipeline(dummy_visualizer)
    pipeline.load_frame = lambda _step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
    }
    pipeline._derive_view_model = lambda _step, _frame: _make_view_model()
    dummy_visualizer.renderer.end_frame_update = lambda: None

    assert pipeline.update(0) is False

    assert dummy_visualizer.frame_times == []
    assert dummy_visualizer.last_app_state is None


def test_post_apply_exception_closes_frame_transaction_once(dummy_visualizer):
    events: list[str] = []
    renderer = dummy_visualizer.renderer

    def begin_frame_update() -> None:
        events.append("begin")
        renderer._frame_update_in_progress = True

    def end_frame_update() -> bool:
        events.append("end")
        renderer._frame_update_in_progress = False
        return True

    renderer.begin_frame_update = begin_frame_update
    renderer.end_frame_update = end_frame_update

    def update_metrics(_view_model) -> None:
        raise RuntimeError("metrics failed")

    metrics_service = SimpleNamespace(update_metrics=update_metrics)
    pipeline = FramePipeline(dummy_visualizer, metrics_service=metrics_service)
    pipeline.load_frame = lambda _step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
    }
    pipeline._derive_view_model = lambda _step, _frame: _make_view_model()
    beam_failures: list[str] = []
    dummy_visualizer.beamforming_ui_controller.fail_computation = beam_failures.append

    with pytest.raises(RuntimeError, match="metrics failed"):
        pipeline.update(0)

    assert events == ["begin", "end"]
    assert beam_failures == ["Frame update failed before presentation"]
    assert renderer._frame_update_in_progress is False
    assert dummy_visualizer.frame_times == []
    assert dummy_visualizer.last_app_state is None


def test_renderer_programming_error_propagates_and_closes_transaction(
    dummy_visualizer,
) -> None:
    """The transaction boundary must not disguise a backend programming defect."""
    events: list[str] = []
    dummy_visualizer.renderer.begin_frame_update = lambda: events.append("begin")
    dummy_visualizer.renderer.end_frame_update = lambda: events.append("end") or True

    def fail_apply(_packet) -> bool:
        raise AssertionError("renderer invariant")

    dummy_visualizer.renderer.apply_frame = fail_apply
    pipeline = FramePipeline(dummy_visualizer)
    pipeline.load_frame = lambda _step: {"num_tx": 0, "num_rx": 0}
    pipeline._derive_view_model = lambda _step, _frame: _make_view_model()

    with pytest.raises(AssertionError, match="renderer invariant"):
        pipeline.update(0)

    assert events == ["begin", "end"]
    assert dummy_visualizer.frame_times == []
    assert dummy_visualizer.last_app_state is None


def test_camera_programming_error_propagates_and_closes_transaction(dummy_visualizer) -> None:
    """A maintained camera callback failure is not an optional-extension failure."""
    events: list[str] = []
    dummy_visualizer.renderer.begin_frame_update = lambda: events.append("begin")
    dummy_visualizer.renderer.end_frame_update = lambda: events.append("end") or True
    dummy_visualizer.app_state = replace(dummy_visualizer.app_state, camera_mode="follow")

    def fail_camera_update() -> None:
        raise AssertionError("camera invariant")

    dummy_visualizer.camera_controller = SimpleNamespace(
        update_follow_camera_focus=fail_camera_update
    )
    pipeline = FramePipeline(dummy_visualizer)
    pipeline.load_frame = lambda _step: {"num_tx": 0, "num_rx": 0}
    pipeline._derive_view_model = lambda _step, _frame: _make_view_model()

    with pytest.raises(AssertionError, match="camera invariant"):
        pipeline.update(0)

    assert events == ["begin", "end"]
    assert dummy_visualizer.last_app_state is None


def test_update_refreshes_node_markers_when_only_orientation_changes(dummy_visualizer):
    """Static TX/RX nodes still need marker transforms when look-at orientation changes."""
    pipeline = FramePipeline(dummy_visualizer)
    tx_positions = np.array([[2.0, 0.0, 0.0]])
    rx_positions = np.array([[3.0, 0.0, 0.0]])
    tx_orientations_by_step = {
        0: np.array([[0.0, 0.0, 0.0]]),
        1: np.array([[np.pi / 2.0, 0.0, 0.0]]),
    }
    rx_orientations = np.array([[0.0, 0.0, 0.0]])
    pipeline.load_frame = lambda step: {
        "tx_positions": tx_positions,
        "rx_positions": rx_positions,
        "num_tx": 1,
        "num_rx": 1,
    }
    pipeline._derive_view_model = lambda step, frame: _make_view_model(
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        tx_orientations=tx_orientations_by_step[step],
        rx_orientations=rx_orientations,
    )
    dummy_visualizer.renderer.apply_frame = lambda packet: True
    dummy_visualizer.tx_markers = [object()]
    dummy_visualizer.rx_markers = []
    marker_syncs: list[tuple[np.ndarray, np.ndarray]] = []

    def record_marker_sync(tx, rx):
        marker_syncs.append((np.asarray(tx), np.asarray(rx)))
        return True

    dummy_visualizer.node_service.update_tx_rx_positions = record_marker_sync

    dummy_visualizer.force_update_next_frame = True
    pipeline.update(0)
    dummy_visualizer.force_update_next_frame = True
    pipeline.update(1)

    assert len(marker_syncs) == 2
    np.testing.assert_array_equal(
        dummy_visualizer.current_tx_orientations,
        tx_orientations_by_step[1],
    )


def test_update_skips_when_state_unchanged(dummy_visualizer):
    """If app_state is unchanged and no dirty flags, renderer should not be called."""
    dummy_visualizer.last_app_state = dummy_visualizer.app_state
    pipeline = FramePipeline(dummy_visualizer)

    dummy_visualizer.renderer.apply_frame = lambda packet: (
        setattr(dummy_visualizer, "applied_packet", packet) or True
    )

    # Stub out load/derive to ensure they are not invoked
    pipeline.load_frame = lambda step: {
        "tx_positions": np.array([[0, 0, 0]]),
        "rx_positions": np.array([[0, 0, 0]]),
    }
    pipeline._derive_view_model = lambda step, frame: _make_view_model()

    pipeline.update(0)

    assert not hasattr(dummy_visualizer, "applied_packet")


def test_force_update_bypasses_state_skip(dummy_visualizer):
    """force_update_next_frame should trigger a render even if state is unchanged."""
    dummy_visualizer.last_app_state = dummy_visualizer.app_state
    dummy_visualizer.force_update_next_frame = True
    pipeline = FramePipeline(dummy_visualizer)

    view_model = _make_view_model()
    pipeline.load_frame = lambda step: {
        "tx_positions": np.array([[0, 0, 0]]),
        "rx_positions": np.array([[0, 0, 0]]),
    }
    pipeline._derive_view_model = lambda step, frame: view_model

    applied = {}

    def fake_apply(packet) -> bool:
        applied["packet"] = packet
        return True

    dummy_visualizer.renderer.apply_frame = fake_apply

    pipeline.update(0)

    assert applied["packet"].mpc_points is view_model.mpc_points
    assert dummy_visualizer.force_update_next_frame is False


def test_record_frame_timing_notifies_scene_boot_completion(dummy_visualizer):
    """First completed frame should publish the scene-ready callback once."""
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer._scene_boot_start = time.perf_counter() - 0.01
    dummy_visualizer._scene_boot_logged = False
    dummy_visualizer.scene_ready_calls = 0
    dummy_visualizer._set_status_message = lambda *args, **kwargs: None

    def _on_scene_boot_completed():
        dummy_visualizer.scene_ready_calls += 1

    dummy_visualizer._on_scene_boot_completed = _on_scene_boot_completed

    pipeline._record_frame_timing(0, time.perf_counter() - 0.005, completed=True)

    assert dummy_visualizer._scene_boot_logged is True
    assert dummy_visualizer._scene_boot_start is None
    assert dummy_visualizer.scene_boot_duration_ms is not None
    assert dummy_visualizer.scene_ready_calls == 1


def test_update_captures_startup_first_frame_breakdown(dummy_visualizer):
    """The first completed frame during scene boot should record a timing breakdown."""
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer._scene_boot_start = time.perf_counter() - 0.01
    dummy_visualizer._scene_boot_logged = False
    dummy_visualizer._set_status_message = lambda *args, **kwargs: None
    captured: dict[str, float] = {}
    detail_captured: dict[str, dict[str, float]] = {}
    dummy_visualizer.set_startup_first_frame_timing = lambda timings: captured.update(timings)
    dummy_visualizer.set_startup_detail_timing = lambda name, timings: detail_captured.setdefault(
        name, dict(timings)
    )

    pipeline.load_frame = lambda step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "num_tx": 1,
        "num_rx": 1,
    }
    pipeline._derive_view_model = lambda step, frame: _make_view_model(
        tx_positions=np.array([[2.0, 0.0, 0.0]]),
        rx_positions=np.array([[3.0, 0.0, 0.0]]),
    )
    dummy_visualizer.renderer.apply_frame = lambda packet: True
    dummy_visualizer.renderer.get_last_end_frame_update_breakdown = lambda: {
        "force_draw_ms": 5.0,
        "total_ms": 5.5,
    }

    pipeline.update(0)

    assert "load_ms" in captured
    assert "viewmodel_ms" in captured
    assert "apply_ms" in captured
    assert "end_frame_update_ms" in captured
    assert "total_before_record_ms" in captured
    assert captured["total_before_record_ms"] >= captured["apply_ms"]
    assert detail_captured["first_frame_end_update_breakdown_ms"]["force_draw_ms"] == 5.0


def test_update_records_benchmark_breakdowns(dummy_visualizer, tmp_path):
    """Benchmark mode should capture per-frame ViewModel and renderer breakdowns."""
    recorder = BenchmarkRecorder(n_frames=1, n_warmup=0, output_path=tmp_path / "bench.json")
    pipeline = FramePipeline(dummy_visualizer, benchmark_recorder=recorder)
    pipeline.load_frame = lambda step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "num_tx": 1,
        "num_rx": 1,
    }
    pipeline._derive_view_model = lambda step, frame: _make_view_model()
    dummy_visualizer.renderer.apply_frame = lambda packet: True
    dummy_visualizer.renderer.get_last_end_frame_update_breakdown = lambda: {
        "force_draw_ms": 4.0,
        "total_ms": 4.5,
    }

    pipeline.update(0)

    record = recorder._records[0]
    assert record.breakdown_ms["viewmodel_cache_hit"] == 0.0
    assert record.breakdown_ms["canonical_lookup_ms"] == 1.25
    assert record.breakdown_ms["filter_ms"] == 2.5
    assert record.breakdown_ms["mpc_visible_segments_count"] == 7.0
    assert record.breakdown_ms["renderer_apply_ms"] >= 0.0
    assert record.breakdown_ms["force_draw_ms"] == 4.0
    assert record.breakdown_ms["end_frame_update_ms"] >= 0.0
    assert record.breakdown_ms["total_before_end_ms"] >= 0.0


def test_update_records_orientation_frame_benchmark_breakdowns(dummy_visualizer, tmp_path):
    """Benchmark mode should carry orientation-frame counters into frame records."""
    recorder = BenchmarkRecorder(n_frames=1, n_warmup=0, output_path=tmp_path / "bench.json")
    pipeline = FramePipeline(dummy_visualizer, benchmark_recorder=recorder)
    pipeline.load_frame = lambda step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "num_tx": 1,
        "num_rx": 1,
    }
    pipeline._derive_view_model = lambda step, frame: _make_view_model()
    dummy_visualizer.renderer.apply_frame = lambda packet: True
    dummy_visualizer.show_tx_orientation = True
    dummy_visualizer.node_service.create_orientation_frames = lambda step: (
        setattr(
            dummy_visualizer,
            "_orientation_frame_breakdown",
            {
                "orientation_frame_sync_count": 2.0,
                "orientation_frame_sync_ms": 1.5,
            },
        )
        or True
    )

    pipeline.update(0)

    record = recorder._records[0]
    assert record.breakdown_ms["orientation_frame_sync_count"] == 2.0
    assert record.breakdown_ms["orientation_frame_sync_ms"] == 1.5


def test_label_visibility_does_not_invalidate_view_model_cache(dummy_visualizer):
    """Service-owned label visibility must not rebuild frame-heavy state."""
    pipeline = FramePipeline(dummy_visualizer)
    raw_frame = {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "num_tx": 1,
        "num_rx": 1,
    }
    dummy_visualizer.app_state = replace(
        dummy_visualizer.app_state,
        mpc_visibility=MpcVisibility(enabled=False),
        show_labels=False,
    )

    first_vm = pipeline._derive_view_model(0, dict(raw_frame))
    calls_after_first = dummy_visualizer.mpc_core.calls
    cached_vm = pipeline._derive_view_model(0, dict(raw_frame))

    dummy_visualizer.app_state = replace(dummy_visualizer.app_state, show_labels=True)
    labels_enabled_vm = pipeline._derive_view_model(0, dict(raw_frame))

    assert cached_vm.mpc_points is first_vm.mpc_points
    assert labels_enabled_vm.mpc_points is first_vm.mpc_points
    assert dummy_visualizer.mpc_core.calls == calls_after_first
    assert "show_labels" not in dummy_visualizer.mpc_core.last_kwargs


def test_update_skips_orientation_frames_when_overlays_disabled(dummy_visualizer):
    """Orientation-frame helpers should not run when all orientation overlays are off."""
    pipeline = FramePipeline(dummy_visualizer)
    pipeline.load_frame = lambda step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "num_tx": 1,
        "num_rx": 1,
    }
    pipeline._derive_view_model = lambda step, frame: _make_view_model()
    dummy_visualizer.renderer.apply_frame = lambda packet: True

    orientation_calls: list[tuple[str, int]] = []
    dummy_visualizer._update_tx_rx_orientations = lambda step, source: orientation_calls.append(
        ("orient", step)
    )
    dummy_visualizer.node_service.create_orientation_frames = lambda step: (
        orientation_calls.append(("frames", step)) or True
    )

    pipeline.update(0)

    assert orientation_calls == []


def test_update_runs_orientation_frame_work_when_overlay_enabled(dummy_visualizer):
    """Orientation overlays should update frame objects without rotating markers."""
    pipeline = FramePipeline(dummy_visualizer)
    pipeline.load_frame = lambda step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "num_tx": 1,
        "num_rx": 1,
    }
    pipeline._derive_view_model = lambda step, frame: _make_view_model()
    dummy_visualizer.renderer.apply_frame = lambda packet: True
    dummy_visualizer.show_tx_orientation = True

    orientation_calls: list[tuple[str, int]] = []
    dummy_visualizer._update_tx_rx_orientations = lambda step, source: orientation_calls.append(
        ("orient", step)
    )
    dummy_visualizer.node_service.create_orientation_frames = lambda step: (
        orientation_calls.append(("frames", step)) or True
    )

    pipeline.update(0)

    assert orientation_calls == [("frames", 0)]


def test_update_refreshes_focus_dropdown_only_when_signature_changes(dummy_visualizer):
    """Camera target-focus dropdown should not rebuild when TX/RX topology is unchanged."""
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.force_update_next_frame = True
    dummy_visualizer.renderer.apply_frame = lambda packet: True
    dummy_visualizer.camera_controller = SimpleNamespace()
    dummy_visualizer.tx_markers = [object()]
    dummy_visualizer.rx_markers = [object()]
    focus_refreshes: list[str] = []
    dummy_visualizer.camera_controller.update_target_focus_dropdown = (
        lambda: focus_refreshes.append("refresh")
    )

    frame_payload = {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "num_tx": 1,
        "num_rx": 1,
    }
    pipeline.load_frame = lambda step: frame_payload
    pipeline._derive_view_model = lambda step, frame: _make_view_model(
        tx_positions=np.array([[2.0, 0.0, 0.0]]),
        rx_positions=np.array([[3.0, 0.0, 0.0]]),
        target_metadata=[{"name": "Car A"}],
    )
    dummy_visualizer.node_service.update_available_tx_rx_from_frame = lambda tx, rx: False

    pipeline.update(0)
    dummy_visualizer.force_update_next_frame = True
    pipeline.update(1)

    assert focus_refreshes == ["refresh"]


def test_update_refreshes_focus_dropdown_when_signature_changes(dummy_visualizer):
    """Camera target-focus dropdown should rebuild when node inventory changes."""
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.force_update_next_frame = True
    dummy_visualizer.renderer.apply_frame = lambda packet: True
    dummy_visualizer.camera_controller = SimpleNamespace()
    dummy_visualizer.tx_markers = [object()]
    dummy_visualizer.rx_markers = [object()]
    focus_refreshes: list[str] = []
    dummy_visualizer.camera_controller.update_target_focus_dropdown = (
        lambda: focus_refreshes.append("refresh")
    )

    frames = iter(
        [
            _make_view_model(
                tx_positions=np.array([[2.0, 0.0, 0.0]]),
                rx_positions=np.array([[3.0, 0.0, 0.0]]),
                target_metadata=[{"name": "Car A"}],
            ),
            _make_view_model(
                tx_positions=np.array([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
                rx_positions=np.array([[3.0, 0.0, 0.0]]),
                target_metadata=[{"name": "Car A"}],
            ),
        ]
    )
    pipeline.load_frame = lambda step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "num_tx": 1,
        "num_rx": 1,
    }
    pipeline._derive_view_model = lambda step, frame: next(frames)
    dummy_visualizer.node_service.update_available_tx_rx_from_frame = lambda tx, rx: False

    pipeline.update(0)
    dummy_visualizer.force_update_next_frame = True
    pipeline.update(1)

    assert focus_refreshes == ["refresh", "refresh"]


def test_update_refreshes_focus_dropdown_when_target_metadata_changes_without_motion(
    dummy_visualizer,
):
    """Dropdown signature changes should refresh even when TX/RX arrays are unchanged."""
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.force_update_next_frame = True
    dummy_visualizer.renderer.apply_frame = lambda packet: True
    dummy_visualizer.camera_controller = SimpleNamespace()
    dummy_visualizer.tx_markers = [object()]
    dummy_visualizer.rx_markers = [object()]
    focus_refreshes: list[str] = []
    dummy_visualizer.camera_controller.update_target_focus_dropdown = (
        lambda: focus_refreshes.append("refresh")
    )

    shared_tx = np.array([[2.0, 0.0, 0.0]])
    shared_rx = np.array([[3.0, 0.0, 0.0]])
    frames = iter(
        [
            _make_view_model(
                tx_positions=shared_tx,
                rx_positions=shared_rx,
                target_metadata=[{"name": "Car A"}],
            ),
            _make_view_model(
                tx_positions=shared_tx,
                rx_positions=shared_rx,
                target_metadata=[{"name": "Car B"}],
            ),
        ]
    )
    pipeline.load_frame = lambda step: {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "num_tx": 1,
        "num_rx": 1,
    }
    pipeline._derive_view_model = lambda step, frame: next(frames)
    dummy_visualizer.node_service.update_available_tx_rx_from_frame = lambda tx, rx: False

    pipeline.update(0)
    dummy_visualizer.force_update_next_frame = True
    pipeline.update(1)

    assert focus_refreshes == ["refresh", "refresh"]


def test_update_refreshes_beam_selector_after_live_node_inventory(dummy_visualizer):
    """Antennas selectors should refresh after live TX/RX counts are known."""
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.force_update_next_frame = True
    dummy_visualizer.renderer.apply_frame = lambda packet: True
    dummy_visualizer.camera_controller = SimpleNamespace(update_target_focus_dropdown=lambda: None)
    dummy_visualizer.available_tx = []
    dummy_visualizer.available_rx = []
    selector_refreshes: list[tuple[list[int], list[int]]] = []
    dummy_visualizer.beamforming_ui_controller.apply_selector_state = (
        lambda: selector_refreshes.append(
            (list(dummy_visualizer.available_tx), list(dummy_visualizer.available_rx))
        )
    )

    def update_available_tx_rx_from_frame(tx_count, rx_count):
        dummy_visualizer.available_tx = list(range(int(tx_count)))
        dummy_visualizer.available_rx = list(range(int(rx_count)))
        return True

    dummy_visualizer.node_service.update_available_tx_rx_from_frame = (
        update_available_tx_rx_from_frame
    )
    pipeline.load_frame = lambda step: {"num_tx": 1, "num_rx": 2}
    pipeline._derive_view_model = lambda step, frame: _make_view_model(
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )

    pipeline.update(0)

    assert selector_refreshes
    assert selector_refreshes[-1] == ([0], [0, 1])


def test_load_frame_uses_animation_service_cache(dummy_visualizer):
    """The pipeline should respect frames cached by AnimationService."""
    pipeline = FramePipeline(dummy_visualizer)
    service = AnimationService(pipeline, dummy_visualizer, max_cached_steps=2)
    dummy_visualizer.animation_service = service

    steps_loaded: list[int] = []

    class TrackedFrameSource:
        def load_frame(self_inner, step: int) -> StandardMPCFrame:
            steps_loaded.append(step)
            return build_standard_mpc_frame(frame_idx=step)

    dummy_visualizer.frame_source = TrackedFrameSource()

    cached_frame = service.load_step(5)
    assert cached_frame is not None
    assert cached_frame["_source"]["frame_idx"] == 5
    assert "canonical_data" in cached_frame
    assert steps_loaded == [5]

    loaded = pipeline.load_frame(5)
    assert loaded == cached_frame
    assert steps_loaded == [5]


def test_load_frame_prefers_live_preview_frame(dummy_visualizer):
    """Live preview frames should bypass the normal frame cache."""
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.cache_service.store_frame(5, {"step": 5, "source": "cached"})
    preview_frame = {"step": 5, "source": "preview"}
    dummy_visualizer._live_preview_frame = preview_frame
    dummy_visualizer._live_preview_step = 5

    loaded = pipeline.load_frame(5)

    assert loaded is preview_frame


def test_pipeline_uses_coverage_service(dummy_visualizer):
    """Ensure coverage helpers go through the injected CoverageService."""

    class StubCoverageService:
        def __init__(self):
            self.events = []
            self.get_kwargs = []

        def compute_cache_key(self, *args, **kwargs):
            self.events.append("compute")
            return "key"

        def get_mesh(self, *args, **kwargs):
            self.events.append("get")
            self.get_kwargs.append(kwargs)
            return None

        def put_mesh(self, *args, **kwargs):
            self.events.append("put")

        def interpolate_values(self, values, interpolation):
            self.events.append("interp")
            return values

        def clear(self):
            self.events.append("clear")

        def stats(self):
            self.events.append("stats")
            return {"hits": 0, "misses": 0, "cache_size": 0}

        def log_stats(self):
            self.events.append("log")

    dummy_visualizer.coverage_service = StubCoverageService()
    dummy_visualizer.coverage_data = {
        "grid_origin": np.zeros(3),
        "grid_spacing": np.ones(3),
        "grid_shape": np.array([1, 1, 1]),
        "values": np.array([0.5]),
        "value_min": 0.0,
        "value_max": 1.0,
        "metric_name": "metric",
        "heights": [0],
    }
    dummy_visualizer.coverage_heights = [0.0]
    dummy_visualizer.coverage_opacity = 0.7
    dummy_visualizer.coverage_interpolation_method = "nearest"

    pipeline = FramePipeline(dummy_visualizer)
    view_model = _make_view_model()

    result = pipeline._add_coverage_to_view_model(view_model)

    assert result.coverage_vertices is not None
    assert "compute" in dummy_visualizer.coverage_service.events
    assert "get" in dummy_visualizer.coverage_service.events
    assert "interp" in dummy_visualizer.coverage_service.events
    assert "put" in dummy_visualizer.coverage_service.events
    assert result.coverage_signature == "key|mask=off"
    assert any(
        kwargs.get("copy") is False for kwargs in dummy_visualizer.coverage_service.get_kwargs
    )


@pytest.mark.parametrize("bounce_points", [False, True])
def test_derive_view_model_when_mpc_layer_disabled(dummy_visualizer, bounce_points):
    state = create_initial_state(
        mpc_layer_enabled=False,
        show_mpc_bounce_points=bounce_points,
    )
    dummy_visualizer.app_state = state
    pipeline = FramePipeline(dummy_visualizer)
    raw_frame = {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "all_padded_vertices": [],
        "all_padded_interactions": [],
        "all_path_lengths": [],
        "tx_rx_pairs": [],
    }

    view_model = pipeline._derive_view_model(0, raw_frame)

    assert view_model.mpc_lines.shape[0] == 0
    assert dummy_visualizer.mpc_core.calls == 1
    assert dummy_visualizer.mpc_core.last_kwargs["mpc_visibility"] == state.mpc_visibility
    assert len(dummy_visualizer.mpc_view_cache) == 1


@pytest.mark.parametrize(
    ("enabled", "paths", "bounce_points"),
    [
        (enabled, paths, bounce_points)
        for enabled in (False, True)
        for paths in (False, True)
        for bounce_points in (False, True)
    ],
)
def test_pipeline_forwards_all_mpc_visibility_combinations(
    dummy_visualizer,
    enabled,
    paths,
    bounce_points,
):
    dummy_visualizer.app_state = create_initial_state(
        mpc_layer_enabled=enabled,
        show_mpc_paths=paths,
        show_mpc_bounce_points=bounce_points,
    )
    pipeline = FramePipeline(dummy_visualizer)
    raw_frame = {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "all_padded_vertices": [],
        "all_padded_interactions": [],
        "all_path_lengths": [],
        "tx_rx_pairs": [],
    }

    pipeline._derive_view_model(0, raw_frame)

    visibility = dummy_visualizer.mpc_core.last_kwargs["mpc_visibility"]
    assert visibility == MpcVisibility(
        enabled=enabled,
        paths=paths,
        bounce_points=bounce_points,
    )
    assert visibility.effective_paths is (enabled and paths)
    assert visibility.effective_bounce_points is (enabled and bounce_points)


def test_derive_view_model_reuses_cache_when_disabled(dummy_visualizer):
    state = create_initial_state(show_mpc_paths=False, show_mpc_bounce_points=True)
    dummy_visualizer.app_state = state
    pipeline = FramePipeline(dummy_visualizer)
    raw_frame = {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "all_padded_vertices": [],
        "all_padded_interactions": [],
        "all_path_lengths": [],
        "tx_rx_pairs": [],
    }

    pipeline._derive_view_model(0, raw_frame)
    dummy_visualizer.mpc_core.calls = 0

    cached = pipeline._derive_view_model(0, raw_frame)

    assert dummy_visualizer.mpc_core.calls == 0
    cached_source = next(iter(dummy_visualizer.mpc_view_cache.values()))
    assert cached.mpc_points is cached_source.mpc_points


def test_derive_view_model_preserves_beamforming_when_mpcs_disabled(dummy_visualizer):
    """Beam-pattern meshes must still render when MPC path lines are hidden."""
    tx_surface = _make_beamforming_surface("beamforming:tx_0:mesh")
    rx_surface = _make_beamforming_surface("beamforming:rx_0:mesh")
    state = create_initial_state(
        mpc_layer_enabled=False,
        show_beamforming=True,
        standalone_antenna_rows=4,
        standalone_antenna_cols=4,
        beamforming_db_scale=True,
        beamforming_dynamic_range_db=27.0,
        beamforming_colormap="viridis",
        beamforming_element_pattern="isotropic",
        beamforming_tx_element_pattern="dipole",
        beamforming_rx_element_pattern="tr38901",
    )
    dummy_visualizer.app_state = state
    dummy_visualizer.mpc_core.return_value = _make_view_model(
        beamforming_meshes=(tx_surface, rx_surface),
        beamforming_pairs=[{"tx_name": "tx_1", "rx_name": "rx_1"}],
        beamforming_info={
            "resolved_tx_node": "tx_1",
            "resolved_rx_node": "rx_1",
            "status": "Beam patterns: tx_1 -> rx_1",
        },
    )
    pipeline = FramePipeline(dummy_visualizer)
    raw_frame = {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "all_padded_vertices": [],
        "all_padded_interactions": [],
        "all_path_lengths": [],
        "tx_rx_pairs": [],
        "_source": {"origin": "cache"},
    }

    view_model = pipeline._derive_view_model(0, raw_frame)

    assert view_model.mpc_lines.shape[0] == 0
    assert view_model.beamforming_meshes == (tx_surface, rx_surface)
    assert view_model.beamforming_info["resolved_rx_node"] == "rx_1"
    kwargs = dummy_visualizer.mpc_core.last_kwargs
    assert kwargs["beamforming_db_scale"] is True
    assert kwargs["beamforming_dynamic_range_db"] == 27.0
    assert kwargs["beamforming_colormap"] == "viridis"
    assert kwargs["beamforming_element_pattern"] == "isotropic"
    assert kwargs["beamforming_tx_element_pattern"] == "dipole"
    assert kwargs["beamforming_rx_element_pattern"] == "tr38901"
    assert "standalone_beamforming_mode" not in raw_frame
    assert "standalone_beamforming_params" not in raw_frame
    assert raw_frame["_source"] == {"origin": "cache"}
    view_frame = kwargs["raw_frame"]
    assert view_frame is not raw_frame
    assert view_frame["_source"] is not raw_frame["_source"]
    assert view_frame["standalone_beamforming_mode"] == "standalone"
    assert view_frame["standalone_beamforming_params"]["antenna_rows"] == 4


def test_disabled_mpc_cache_key_includes_standalone_array_controls(dummy_visualizer):
    """Changing array dimensions must rebuild beam meshes even with MPCs hidden."""
    raw_frame = {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "all_padded_vertices": [],
        "all_padded_interactions": [],
        "all_path_lengths": [],
        "tx_rx_pairs": [],
    }
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.app_state = create_initial_state(
        show_mpc_paths=False,
        show_beamforming=True,
        standalone_antenna_rows=2,
        standalone_antenna_cols=2,
    )
    pipeline._derive_view_model(0, raw_frame)
    dummy_visualizer.app_state = create_initial_state(
        show_mpc_paths=False,
        show_beamforming=True,
        standalone_antenna_rows=4,
        standalone_antenna_cols=4,
    )

    pipeline._derive_view_model(0, raw_frame)

    assert dummy_visualizer.mpc_core.calls == 2


def test_disabled_mpc_cache_key_changes_for_material_filters_when_bounces_visible(
    dummy_visualizer,
):
    """Hidden MPC lines still need material filters for visible bounce payloads."""
    raw_frame = {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "all_padded_vertices": [],
        "all_padded_interactions": [],
        "all_path_lengths": [],
        "tx_rx_pairs": [],
    }
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.app_state = create_initial_state(
        show_mpc_paths=False,
        show_mpc_bounce_points=True,
    )
    dummy_visualizer.mpc_allowed_materials = None
    pipeline._derive_view_model(0, raw_frame)

    dummy_visualizer.app_state = create_initial_state(
        show_mpc_paths=False,
        show_mpc_bounce_points=True,
    )
    dummy_visualizer.mpc_allowed_materials = {"glass"}
    dummy_visualizer.mpc_material_filter_scope = "segment"
    pipeline._derive_view_model(0, raw_frame)

    assert dummy_visualizer.mpc_core.calls == 2
    assert dummy_visualizer.mpc_core.last_kwargs["mpc_allowed_materials"] == {"glass"}


def test_derive_view_model_uses_unique_node_counts_for_spheres(dummy_visualizer):
    """Sphere creation must use unique TX/RX counts, not TX-RX pair array lengths."""
    dummy_visualizer.app_state = create_initial_state(show_mpc_paths=True)
    pipeline = FramePipeline(dummy_visualizer)
    raw_frame = {
        "num_tx": 2,
        "num_rx": 4,
        "tx_positions": np.zeros((8, 3), dtype=np.float32),
        "rx_positions": np.zeros((8, 3), dtype=np.float32),
    }

    pipeline._derive_view_model(0, raw_frame)

    assert dummy_visualizer.last_marker_counts == (2, 4)


def test_derive_view_model_syncs_frame_device_names_for_services(dummy_visualizer):
    """Frame-provided TX/RX names stay in state without entering MPC derivation."""
    dummy_visualizer.app_state = create_initial_state(show_mpc_paths=True, node_label_mode="name")

    def _set_state(**changes):
        dummy_visualizer.app_state = replace(dummy_visualizer.app_state, **changes)

    dummy_visualizer.set_state = _set_state
    pipeline = FramePipeline(dummy_visualizer)
    raw_frame = {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "tx_names": ["MainTransmitter"],
        "rx_names": ["WalkingReceiver"],
    }

    pipeline._derive_view_model(0, raw_frame)

    assert dummy_visualizer.app_state.tx_device_names == ("MainTransmitter",)
    assert dummy_visualizer.app_state.rx_device_names == ("WalkingReceiver",)
    assert "node_label_mode" not in dummy_visualizer.mpc_core.last_kwargs
    assert "tx_device_names" not in dummy_visualizer.mpc_core.last_kwargs
    assert "rx_device_names" not in dummy_visualizer.mpc_core.last_kwargs


def test_derive_view_model_decodes_frame_device_name_bytes(dummy_visualizer):
    """HDF5-style byte strings are decoded before label resolution."""
    dummy_visualizer.app_state = create_initial_state(show_mpc_paths=True, node_label_mode="name")

    def _set_state(**changes):
        dummy_visualizer.app_state = replace(dummy_visualizer.app_state, **changes)

    dummy_visualizer.set_state = _set_state
    pipeline = FramePipeline(dummy_visualizer)
    raw_frame = {
        "tx_positions": np.array([[0.0, 0.0, 0.0]]),
        "rx_positions": np.array([[1.0, 0.0, 0.0]]),
        "tx_names": np.array([b"MainTransmitter"]),
        "rx_names": np.array([b"WalkingReceiver"]),
    }

    pipeline._derive_view_model(0, raw_frame)

    assert dummy_visualizer.app_state.tx_device_names == ("MainTransmitter",)
    assert dummy_visualizer.app_state.rx_device_names == ("WalkingReceiver",)


def test_add_coverage_mesh_populates_view_model(dummy_visualizer):
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.coverage_data = {
        "grid_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.array([1, 1, 1], dtype=int),
        "value_min": 0.0,
        "value_max": 1.0,
        "metric_name": "coverage",
        "values_3d": np.array([[[0.5]]], dtype=np.float32),
        "heights": [0.0],
        "values": np.array([0.5], dtype=np.float32),
    }
    view_model = _make_view_model()
    result = pipeline._add_coverage_to_view_model(view_model)

    assert result.coverage_vertices is not None
    assert result.coverage_triangles is not None
    assert result.coverage_colors is not None
    assert result.coverage_signature is not None
    assert result.coverage_metadata["metric_name"] == "coverage"


def test_malformed_base_coverage_data_is_not_reported_as_a_successful_frame(
    dummy_visualizer,
) -> None:
    """A core coverage contract defect must remain visible to its caller."""
    dummy_visualizer.app_state = replace(dummy_visualizer.app_state, show_coverage=True)
    dummy_visualizer.coverage_data = _make_coverage_data()
    del dummy_visualizer.coverage_data["grid_spacing"]
    pipeline = FramePipeline(dummy_visualizer)
    pipeline.load_frame = lambda _step: {"num_tx": 0, "num_rx": 0}

    with pytest.raises(KeyError, match="grid_spacing"):
        pipeline.update(0)

    assert dummy_visualizer.last_app_state is None
    assert dummy_visualizer.frame_times == []


def test_threshold_visual_data_error_preserves_base_coverage_and_reports_not_applied(
    dummy_visualizer,
    monkeypatch,
) -> None:
    """The optional threshold visual may fail without hiding valid base coverage."""
    dummy_visualizer.coverage_data = _make_coverage_data()
    dummy_visualizer.coverage_heights = [0.0]
    dummy_visualizer.coverage_threshold_enabled = True
    dummy_visualizer.coverage_threshold_mask_enabled = True
    dummy_visualizer.coverage_threshold_value = 0.5

    def fail_threshold(*_args, **_kwargs):
        raise ValueError("bad threshold")

    monkeypatch.setattr(
        "visualizer.src.pipeline.frame_pipeline.compute_coverage_threshold_mask",
        fail_threshold,
    )

    result = FramePipeline(dummy_visualizer)._add_coverage_to_view_model(_make_view_model())

    assert result.show_coverage is True
    assert result.coverage_vertices.shape == (8, 3)
    assert result.coverage_signature.endswith("|mask=off")
    assert result.coverage_metadata["threshold_mask_enabled"] is True
    assert result.coverage_metadata["threshold_mask_applied"] is False


def test_isoline_data_error_preserves_base_coverage_and_reports_not_applied(
    dummy_visualizer,
    monkeypatch,
) -> None:
    """The optional contour visual may fail without hiding valid base coverage."""
    dummy_visualizer.coverage_data = _make_coverage_data()
    dummy_visualizer.coverage_heights = [0.0]
    dummy_visualizer.coverage_isolines_enabled = True
    dummy_visualizer.coverage_isoline_count = 3
    pipeline = FramePipeline(dummy_visualizer)

    def fail_isolines(*_args, **_kwargs):
        raise ValueError("bad contour")

    monkeypatch.setattr(pipeline, "_build_coverage_isolines", fail_isolines)

    result = pipeline._add_coverage_to_view_model(_make_view_model())

    assert result.show_coverage is True
    assert result.coverage_vertices.shape == (8, 3)
    assert result.coverage_metadata["isolines_enabled"] is True
    assert result.coverage_metadata["isolines_applied"] is False
    assert result.coverage_metadata["isoline_signature"] is None


def test_add_serving_tx_coverage_mesh_omits_no_service_cells(dummy_visualizer):
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.coverage_data = {
        "grid_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.array([2, 2, 1], dtype=int),
        "value_min": 0.0,
        "value_max": 1.0,
        "metric_name": "serving_tx",
        "values_3d": np.array([[[-1, 0], [1, -1]]], dtype=np.float32),
        "heights": [0.0],
        "tx_names": ["TX1", "TX2"],
        "serving_tx_count": 2,
    }
    dummy_visualizer.coverage_heights = [0.0]
    view_model = _make_view_model()

    result = pipeline._add_coverage_to_view_model(view_model)

    assert result.coverage_vertices is not None
    assert result.coverage_triangles is not None
    assert result.coverage_colors is not None
    assert result.coverage_vertices.shape[0] == 8
    assert result.coverage_triangles.shape[0] == 4
    assert result.coverage_colors.shape[0] == 8
    assert result.coverage_metadata["tx_names"] == ["TX1", "TX2"]
    assert result.coverage_metadata["tx_count"] == 2
    assert result.coverage_metadata["valid_cell_count"] == 2
    assert result.coverage_metadata["no_data_fraction"] == pytest.approx(0.5)


def test_add_serving_tx_coverage_forces_raw_categories(dummy_visualizer):
    """Categorical transmitter IDs must not be blurred into other categories."""
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.coverage_interpolation_method = "cubic"
    dummy_visualizer.coverage_data = {
        "grid_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.array([2, 1, 1], dtype=int),
        "value_min": 0.0,
        "value_max": 1.0,
        "metric_name": "serving_tx",
        "values_3d": np.array([[[0.0, 1.0]]], dtype=np.float32),
        "heights": [0.0],
        "tx_names": ["TX1", "TX2"],
        "serving_tx_count": 2,
    }
    dummy_visualizer.coverage_heights = [0.0]

    result = pipeline._add_coverage_to_view_model(_make_view_model())

    assert dummy_visualizer.coverage_interpolation_method == "none"
    assert result.coverage_colors is not None
    assert not np.allclose(result.coverage_colors[:4], result.coverage_colors[4:8])


def test_add_scalar_coverage_omits_no_data_cells(dummy_visualizer):
    """No-data scalar cells are holes, not opaque poor-coverage cells."""
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.coverage_data = {
        "grid_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.array([2, 1, 1], dtype=int),
        "value_min": 80.0,
        "value_max": 100.0,
        "metric_name": "best_path_loss_db",
        "values_3d": np.array([[[np.nan, 90.0]]], dtype=np.float32),
        "heights": [0.0],
    }
    dummy_visualizer.coverage_heights = [0.0]

    result = pipeline._add_coverage_to_view_model(_make_view_model())

    assert result.coverage_vertices.shape == (4, 3)
    assert result.coverage_triangles.shape == (2, 3)
    assert result.coverage_metadata["no_data_fraction"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("metric_name", "low_channel", "high_channel"),
    [
        ("best_path_loss_db", 1, 0),
        ("sinr_db", 0, 1),
    ],
)
def test_coverage_colors_make_favorable_values_green(
    dummy_visualizer,
    metric_name: str,
    low_channel: int,
    high_channel: int,
):
    """Semantic coverage palettes keep green at the favorable range endpoint."""
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.coverage_data = {
        "grid_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.array([2, 1, 1], dtype=int),
        "value_min": 0.0,
        "value_max": 100.0,
        "metric_name": metric_name,
        "values_3d": np.array([[[0.0, 100.0]]], dtype=np.float32),
        "heights": [0.0],
    }
    dummy_visualizer.coverage_heights = [0.0]

    result = pipeline._add_coverage_to_view_model(_make_view_model())
    low_color = result.coverage_colors[0]
    high_color = result.coverage_colors[4]

    assert low_color[low_channel] > low_color[high_channel]
    assert high_color[high_channel] > high_color[low_channel]


def test_positive_linear_rf_colors_use_log_scale_and_omit_nonpositive_cells(
    dummy_visualizer,
):
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.coverage_data = {
        "grid_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.array([4, 1, 1], dtype=int),
        "value_min": 1.0e-12,
        "value_max": 1.0e-6,
        "metric_name": "path_gain_linear/TX1",
        "values_3d": np.array([[[0.0, 1.0e-12, 1.0e-9, 1.0e-6]]], dtype=np.float32),
        "heights": [0.0],
        "tx_names": ["TX1"],
    }
    dummy_visualizer.coverage_heights = [0.0]
    dummy_visualizer.coverage_threshold_enabled = True
    dummy_visualizer.coverage_threshold_value = 1.0e-9
    dummy_visualizer.coverage_threshold_mask_enabled = True

    mpc_colorbar = ("Delay (ns)", (1.0, 9.0))
    result = pipeline._add_coverage_to_view_model(_make_view_model(colorbar=mpc_colorbar))

    assert result.coverage_vertices.shape == (12, 3)
    assert result.coverage_metadata["valid_cell_count"] == 3
    assert result.coverage_metadata["no_data_fraction"] == pytest.approx(0.25)
    assert result.coverage_metadata["color_scale"] == "logarithmic"
    assert result.coverage_metadata["threshold_mask_applied"] is True
    assert result.colorbar == mpc_colorbar
    cell_colors = result.coverage_colors[::4]
    assert np.unique(np.round(cell_colors, decimals=6), axis=0).shape[0] == 3


def test_logarithmic_isolines_cover_the_visible_decades():
    values = np.array([[1.0e-12, 1.0e-9, 1.0e-6]], dtype=np.float32)

    levels = FramePipeline._coverage_isoline_levels(
        values,
        6,
        "path_gain_linear/TX1",
    )

    boundaries = np.array([1.0e-12, *levels, 1.0e-6])
    np.testing.assert_allclose(
        np.diff(np.log10(boundaries)),
        np.full(7, 6.0 / 7.0),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_linear_isolines_remain_arithmetic():
    levels = FramePipeline._coverage_isoline_levels(
        np.array([[0.0, 10.0]], dtype=np.float32),
        4,
        "sinr_db",
    )

    np.testing.assert_allclose(levels, [2.0, 4.0, 6.0, 8.0])


def test_all_nonpositive_logarithmic_data_has_no_rendered_range(dummy_visualizer):
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.coverage_data = {
        "grid_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.array([3, 1, 1], dtype=int),
        "value_min": 0.0,
        "value_max": 1.0,
        "metric_name": "path_gain_linear/TX1",
        "values_3d": np.array([[[0.0, -1.0, np.nan]]], dtype=np.float32),
        "heights": [0.0],
        "tx_names": ["TX1"],
    }
    dummy_visualizer.coverage_heights = [0.0]

    result = pipeline._add_coverage_to_view_model(_make_view_model())

    assert result.coverage_vertices.shape == (0, 3)
    assert result.coverage_metadata["valid_cell_count"] == 0
    assert result.coverage_metadata["no_data_fraction"] == pytest.approx(1.0)
    assert result.coverage_metadata["value_min"] is None
    assert result.coverage_metadata["value_max"] is None
    assert result.colorbar is None


def test_all_nan_linear_data_has_no_rendered_range(dummy_visualizer):
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.coverage_data = {
        "grid_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.array([2, 1, 1], dtype=int),
        "value_min": -120.0,
        "value_max": -60.0,
        "metric_name": "rss_dbm/TX1",
        "values_3d": np.array([[[np.nan, np.nan]]], dtype=np.float32),
        "heights": [0.0],
        "tx_names": ["TX1"],
    }
    dummy_visualizer.coverage_heights = [0.0]

    result = pipeline._add_coverage_to_view_model(_make_view_model())

    assert result.coverage_vertices.shape == (0, 3)
    assert result.coverage_metadata["valid_cell_count"] == 0
    assert result.coverage_metadata["value_min"] is None
    assert result.coverage_metadata["value_max"] is None


def test_invalid_logarithmic_range_is_recovered_once_for_mesh_and_metadata(
    dummy_visualizer,
):
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.coverage_data = {
        "grid_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.array([2, 1, 1], dtype=int),
        "value_min": 0.0,
        "value_max": float("inf"),
        "metric_name": "path_gain_linear/TX1",
        "values_3d": np.array([[[1.0e-12, 1.0e-6]]], dtype=np.float32),
        "heights": [0.0],
        "tx_names": ["TX1"],
    }
    dummy_visualizer.coverage_heights = [0.0]

    result = pipeline._add_coverage_to_view_model(_make_view_model())

    assert result.coverage_metadata["value_min"] == pytest.approx(1.0e-12)
    assert result.coverage_metadata["value_max"] == pytest.approx(1.0e-6)
    assert dummy_visualizer.coverage_data["value_min"] == pytest.approx(1.0e-12)
    assert dummy_visualizer.coverage_data["value_max"] == pytest.approx(1.0e-6)
    assert not np.allclose(result.coverage_colors[0], result.coverage_colors[4])


def test_small_db_range_is_not_collapsed_by_absolute_tolerance(dummy_visualizer):
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.coverage_data = {
        "grid_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.array([2, 1, 1], dtype=int),
        "value_min": 1.0e-9,
        "value_max": 2.0e-9,
        "metric_name": "sinr_db",
        "values_3d": np.array([[[1.0e-9, 2.0e-9]]], dtype=np.float32),
        "heights": [0.0],
    }
    dummy_visualizer.coverage_heights = [0.0]

    result = pipeline._add_coverage_to_view_model(_make_view_model())

    assert result.coverage_metadata["color_scale"] == "linear"
    assert not np.allclose(result.coverage_colors[0], result.coverage_colors[4])


def test_scene_only_pipeline_applies_coverage_without_frame_source(dummy_visualizer):
    """A valid coverage dataset renders even when the scenario has no MPC frames."""
    packets: list[Any] = []
    dummy_visualizer.ready = False
    dummy_visualizer._scene_only_mode = True
    dummy_visualizer.app_state = create_initial_state(show_coverage=True)
    dummy_visualizer.renderer.apply_frame = lambda packet: (packets.append(packet) or True)
    dummy_visualizer.coverage_data = {
        "grid_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.array([1, 1, 1], dtype=int),
        "value_min": 0.0,
        "value_max": 1.0,
        "metric_name": "sinr_db",
        "values_3d": np.array([[[0.5]]], dtype=np.float32),
        "heights": [0.0],
    }
    dummy_visualizer.coverage_heights = [0.0]

    assert FramePipeline(dummy_visualizer).update(0) is True

    assert len(packets) == 1
    assert packets[0].show_coverage is True
    assert packets[0].coverage_vertices.shape == (4, 3)


def test_precache_coverage_heights_does_not_change_displayed_slice(dummy_visualizer):
    pipeline = FramePipeline(dummy_visualizer)
    rendered: list[Any] = []
    dummy_visualizer.renderer.apply_frame = lambda packet: (rendered.append(packet) or True)
    dummy_visualizer.coverage_height_index = 0
    dummy_visualizer.coverage_heights = [1.0, 2.0]
    dummy_visualizer.coverage_data = {
        "dataset_fingerprint": "precache-dataset",
        "grid_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.array([2, 1, 2], dtype=int),
        "value_min": 0.0,
        "value_max": 3.0,
        "metric_name": "sinr_db",
        "values_3d": np.arange(4, dtype=np.float32).reshape(2, 1, 2),
        "heights": [1.0, 2.0],
    }

    assert pipeline.precache_coverage_heights() == (2, 0)
    assert pipeline.precache_coverage_heights() == (0, 2)

    assert dummy_visualizer.coverage_height_index == 0
    assert dummy_visualizer.app_state.coverage_height_index == 0
    assert rendered == []


def test_file_backed_coverage_streams_requested_height_and_precaches_all(
    dummy_visualizer,
    tmp_path,
):
    coverage_file = tmp_path / "coverage_maps.h5"
    path_gain = np.asarray(
        [[[[[1e-8, 1e-8]]], [[[1e-9, 1e-9]]]]],
        dtype=np.float32,
    )
    save_coverage_hdf5(
        {
            "grid_origin": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            "grid_spacing": np.asarray([1.0, 1.0], dtype=np.float32),
            "grid_shape": np.asarray([2, 1, 2], dtype=np.int32),
            "heights": np.asarray([1.0, 5.0], dtype=np.float32),
            "path_gain_linear": path_gain,
            "derived": {},
            "metric_name": "best_path_loss_db",
            "tx_positions": np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32),
            "rx_positions": np.empty((0, 3), dtype=np.float32),
            "tx_names": ["TX1"],
            "rx_names": [],
            "tx_power_dbm": np.asarray([0.0], dtype=np.float32),
            "value_min": 80.0,
            "value_max": 90.0,
            "metadata": {
                "metrics_store": ["path_gain_linear"],
                "metrics_derived": ["path_loss_db"],
                "noise_power_w": 1e-12,
            },
        },
        coverage_file,
        compression=None,
    )

    coverage_service = CoverageService(max_cache_size=4)
    coverage_data = coverage_service._load_v2_coverage_hdf5(coverage_file)
    coverage_data["dataset_fingerprint"] = "file-backed-coverage"
    dummy_visualizer.coverage_data = coverage_data
    dummy_visualizer.coverage_heights = [1.0, 5.0]
    dummy_visualizer.coverage_height_index = 1
    dummy_visualizer.app_state = replace(
        dummy_visualizer.app_state,
        coverage_height_index=1,
    )
    pipeline = FramePipeline(dummy_visualizer, coverage_service=coverage_service)

    result = pipeline._add_coverage_to_view_model(_make_view_model())

    assert coverage_data["_active_height_index"] == 1
    assert coverage_data["values_3d"].shape == (1, 1, 2)
    assert np.all(result.coverage_vertices[:, 2] == pytest.approx(5.0))
    assert result.coverage_metadata["selected_height_index"] == 1
    assert result.coverage_metadata["value_min"] == 80.0
    assert result.coverage_metadata["value_max"] == 90.0
    assert pipeline.precache_coverage_heights() == (1, 1)
    assert pipeline.precache_coverage_heights() == (0, 2)
    interpolation = dummy_visualizer.coverage_interpolation_method
    first_key = coverage_service.compute_cache_key(coverage_data, 0, interpolation)
    second_key = coverage_service.compute_cache_key(coverage_data, 1, interpolation)
    first_mesh = coverage_service.get_mesh(first_key, copy=False)
    second_mesh = coverage_service.get_mesh(second_key, copy=False)
    assert first_mesh is not None
    assert second_mesh is not None
    assert np.allclose(first_mesh[0][:, 2], 1.0)
    assert np.allclose(second_mesh[0][:, 2], 5.0)
    assert first_key != second_key
    assert coverage_data["_active_height_index"] == 1
    assert np.allclose(coverage_data["values_3d"], 90.0)
    assert dummy_visualizer.coverage_height_index == 1
    assert dummy_visualizer.app_state.coverage_height_index == 1


def test_add_coverage_mesh_applies_threshold_mask_and_isolines(dummy_visualizer):
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.coverage_data = {
        "grid_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.array([2, 2, 1], dtype=int),
        "value_min": 80.0,
        "value_max": 140.0,
        "metric_name": "best_path_loss_db",
        "values_3d": np.array([[[80.0, 140.0], [140.0, 80.0]]], dtype=np.float32),
        "heights": [0.0],
        "values": np.array([80.0, 140.0, 140.0, 80.0], dtype=np.float32),
    }
    dummy_visualizer.coverage_heights = [0.0]
    dummy_visualizer.coverage_threshold_enabled = True
    dummy_visualizer.coverage_threshold_value = 110.0
    dummy_visualizer.coverage_threshold_mask_enabled = True
    dummy_visualizer.coverage_isolines_enabled = True
    dummy_visualizer.coverage_isoline_count = 3
    view_model = _make_view_model()

    result = pipeline._add_coverage_to_view_model(view_model)

    assert result.coverage_signature.endswith("|mask=110.0")
    assert result.coverage_metadata["isoline_signature"] is not None
    assert result.coverage_isoline_points is not None
    assert result.coverage_isoline_lines is not None
    assert result.coverage_isoline_lines.shape[0] >= 1
    assert result.coverage_metadata["isoline_segments"] >= 1
    assert len(result.coverage_metadata["isoline_levels"]) == 3
    assert result.coverage_metadata["threshold_mask_enabled"] is True
    assert result.coverage_metadata["threshold_mask_applied"] is True
    assert result.coverage_metadata["isolines_applied"] is True


def test_add_coverage_mesh_reuses_cached_isolines(dummy_visualizer):
    pipeline = FramePipeline(dummy_visualizer)
    dummy_visualizer.coverage_data = {
        "dataset_fingerprint": "isoline-cache-dataset",
        "grid_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "grid_spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "grid_shape": np.array([3, 3, 1], dtype=int),
        "value_min": 0.0,
        "value_max": 8.0,
        "metric_name": "sinr_db",
        "values_3d": np.arange(9, dtype=np.float32).reshape(1, 3, 3),
        "heights": [1.5],
    }
    dummy_visualizer.coverage_heights = [1.5]
    dummy_visualizer.coverage_isolines_enabled = True
    dummy_visualizer.coverage_isoline_count = 4

    first = pipeline._add_coverage_to_view_model(_make_view_model())
    second = pipeline._add_coverage_to_view_model(_make_view_model())
    stats = pipeline.get_coverage_cache_stats()

    assert first.coverage_metadata["isoline_signature"]
    assert (
        second.coverage_metadata["isoline_signature"]
        == first.coverage_metadata["isoline_signature"]
    )
    assert stats["isoline_misses"] == 1
    assert stats["isoline_hits"] == 1


def test_pipeline_delegates_metrics_updates(dummy_visualizer):
    recorded: dict[str, Any] = {}

    class DummyMetricsService:
        def update_metrics(self, view_model: ViewModel) -> None:
            recorded["view_model"] = view_model

    metrics_service = DummyMetricsService()
    pipeline = FramePipeline(dummy_visualizer, metrics_service=metrics_service)

    view_model = _make_view_model()
    view_model.canonical_data = SimpleNamespace()  # minimal payload
    raw_frame = {"all_path_lengths": [[1]]}

    pipeline.load_frame = lambda step: raw_frame
    pipeline._derive_view_model = lambda step, frame: view_model
    dummy_visualizer.renderer.apply_frame = lambda packet: True

    pipeline.update(0)

    assert recorded["view_model"] is view_model


def test_record_frame_timing_preserves_scenario_status_and_updates_playback_telemetry(
    dummy_visualizer,
):
    """Completed frames must not replace scenario context with per-frame details."""
    messages: list[tuple[str, int]] = []
    timing_updates: list[tuple[int, float]] = []
    dummy_visualizer._set_status_message = lambda text, timeout=0: messages.append((text, timeout))
    dummy_visualizer.ui_controller.handle_frame_timing_update = (
        lambda step, elapsed: timing_updates.append((step, elapsed))
    )
    pipeline = FramePipeline(dummy_visualizer)

    pipeline._record_frame_timing(
        step=2,
        frame_start=time.perf_counter() - 0.01,
        completed=True,
    )

    assert messages == []
    assert len(timing_updates) == 1
    assert timing_updates[0][0] == 2
    assert timing_updates[0][1] > 0.0


def test_failed_runtime_extension_sync_marks_frame_incomplete(
    dummy_visualizer,
    monkeypatch,
) -> None:
    """The pipeline must not report success for an unapplied extension layer set."""
    sync_calls: list[tuple[Any, dict[str, Any], int]] = []

    def reject_extension_sync(viz, raw_frame, step) -> bool:
        sync_calls.append((viz, raw_frame, step))
        return False

    monkeypatch.setattr(
        "visualizer.src.pipeline.frame_pipeline.sync_runtime_extensions",
        reject_extension_sync,
    )
    pipeline = FramePipeline(dummy_visualizer)
    raw_frame = {
        "tx_positions": np.asarray([[0.0, 0.0, 0.0]]),
        "rx_positions": np.asarray([[1.0, 0.0, 0.0]]),
    }
    pipeline.load_frame = lambda _step: raw_frame
    pipeline._derive_view_model = lambda _step, _frame: _make_view_model()

    assert pipeline.update(0) is False
    assert len(sync_calls) == 1
    assert sync_calls[0] == (dummy_visualizer, raw_frame, 0)
    assert dummy_visualizer.last_app_state is None


def test_expected_runtime_extension_error_is_isolated_and_marks_frame_incomplete(
    dummy_visualizer,
    monkeypatch,
) -> None:
    """Expected optional extension failures do not escape the frame boundary."""
    events: list[str] = []
    dummy_visualizer.renderer.begin_frame_update = lambda: events.append("begin")
    dummy_visualizer.renderer.end_frame_update = lambda: events.append("end") or True

    def fail_extension_sync(*_args, **_kwargs):
        raise RuntimeError("optional extension unavailable")

    monkeypatch.setattr(
        "visualizer.src.pipeline.frame_pipeline.sync_runtime_extensions",
        fail_extension_sync,
    )
    pipeline = FramePipeline(dummy_visualizer)
    pipeline.load_frame = lambda _step: {
        "tx_positions": np.asarray([[0.0, 0.0, 0.0]]),
        "rx_positions": np.asarray([[1.0, 0.0, 0.0]]),
    }
    pipeline._derive_view_model = lambda _step, _frame: _make_view_model()

    assert pipeline.update(0) is False
    assert events == ["begin", "end"]
    assert dummy_visualizer.last_app_state is None


def test_runtime_extension_programming_error_propagates_and_closes_transaction(
    dummy_visualizer,
    monkeypatch,
) -> None:
    """Optional-extension isolation must not conceal its own invariant failures."""
    events: list[str] = []
    dummy_visualizer.renderer.begin_frame_update = lambda: events.append("begin")
    dummy_visualizer.renderer.end_frame_update = lambda: events.append("end") or True

    def fail_extension_sync(*_args, **_kwargs):
        raise AssertionError("extension invariant")

    monkeypatch.setattr(
        "visualizer.src.pipeline.frame_pipeline.sync_runtime_extensions",
        fail_extension_sync,
    )
    pipeline = FramePipeline(dummy_visualizer)
    pipeline.load_frame = lambda _step: {
        "tx_positions": np.asarray([[0.0, 0.0, 0.0]]),
        "rx_positions": np.asarray([[1.0, 0.0, 0.0]]),
    }
    pipeline._derive_view_model = lambda _step, _frame: _make_view_model()

    with pytest.raises(AssertionError, match="extension invariant"):
        pipeline.update(0)

    assert events == ["begin", "end"]
    assert dummy_visualizer.last_app_state is None


def test_failed_aperture_sync_retries_before_committing_app_state(dummy_visualizer) -> None:
    """A failed built-in overlay sync remains pending for an identical retry."""
    dummy_visualizer.app_state = replace(dummy_visualizer.app_state, show_aoa_aperture=True)
    results = iter((False, True))
    calls: list[bool] = []
    dummy_visualizer.aperture_service = SimpleNamespace(
        update_apertures=lambda: calls.append(True) or next(results)
    )
    pipeline = FramePipeline(dummy_visualizer)
    pipeline.load_frame = lambda _step: {
        "tx_positions": np.asarray([[0.0, 0.0, 0.0]]),
        "rx_positions": np.asarray([[1.0, 0.0, 0.0]]),
    }
    pipeline._derive_view_model = lambda _step, _frame: _make_view_model()

    assert pipeline.update(0) is False
    assert dummy_visualizer.last_app_state is None
    assert pipeline.update(0) is True
    assert calls == [True, True]
    assert dummy_visualizer.last_app_state is dummy_visualizer.app_state
