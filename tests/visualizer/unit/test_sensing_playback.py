"""Sensing-only playback skips pose rows without resampling recorded results."""

from concurrent.futures import Future
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QCheckBox, QComboBox, QSlider, QSpinBox

from shared.frames import FrameComponent
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload
from tests.visualizer.unit.test_services_sensing import _CountingMultiFrameProvider
from visualizer.src.sensing.playback import (
    SensingPlayback,
    load_sensing_playback_plan,
    prepared_playback_plan,
)
from visualizer.visualizer import OrchavVisualizer


class _SnapshotProvider(_CountingMultiFrameProvider):
    def __init__(self, frames):
        super().__init__(frames)
        self.snapshot = tuple(sorted(frames))

    @property
    def frame_ids_snapshot(self):
        return self.snapshot


def _provider():
    frames = {
        step: replace(build_standard_mpc_frame(frame_idx=step), timestamp_s=stamp)
        for step, stamp in ((0, 0.0), (1, 0.01), (7, 0.064), (8, 0.065), (23, 0.201))
    }
    for storage, bound in ((1, 0), (8, 7), (23, 23)):
        frames[storage] = replace(
            frames[storage],
            sensing=build_visualizer_sensing_payload(
                {"detections": []}, scene_frame_id=bound, sample_time_s=frames[bound].timestamp_s
            ),
        )
    return _SnapshotProvider(frames)


def test_plan_reads_only_metadata_and_preserves_irregular_scene_timestamps():
    provider = _provider()
    steps, times = load_sensing_playback_plan(provider, Event())
    assert steps == (0, 7, 23)
    assert times == {0: 0.0, 7: 0.064, 23: 0.201}
    assert all(not request.all_sensing_products for _, request in provider.requests)
    assert all(
        request.components in ({FrameComponent.SENSING}, {FrameComponent.PROVENANCE})
        for _, request in provider.requests
    )
    assert all(request.sensing_products <= {"products"} for _, request in provider.requests)


def _playback(qapp):
    provider = _provider()
    updates, counts, preloads, messages = [], [], [], []
    viz = SimpleNamespace(
        frame_loader=SimpleNamespace(provider=provider),
        frame_source=provider,
        animation_running=False,
        animation_step=8,
        total_animation_steps=5,
        step_slider=QSlider(),
        frame_input=QSpinBox(),
        stride_combo=QComboBox(),
        cancel_startup_preload=lambda: None,
        reset_preloading_state=lambda: None,
        start_preloading=lambda: preloads.append(viz.get_available_animation_steps()),
        ui_manager=SimpleNamespace(update_total_steps=counts.append, panels={}),
        _set_status_message=lambda text, *a: messages.append(text),
    )
    viz.stride_combo.setEditable(True)
    viz.stride_combo.addItems(["x1", "x128"])
    viz.stride_combo.setCurrentText("x128")
    viz.get_available_animation_steps = lambda: OrchavVisualizer.get_available_animation_steps(viz)

    def update(step):
        viz.animation_step = step
        updates.append(step)

    viz.update_frame = update
    checkbox = QCheckBox()
    plan = SensingPlayback(viz, checkbox)
    viz.sensing_playback = plan
    return plan, provider, updates, counts, preloads, messages


def test_checkbox_filters_navigation_preload_and_cadence_then_restores(qapp):
    plan, provider, updates, counts, preloads, _ = _playback(qapp)
    try:
        plan.checkbox.setChecked(True)
        assert plan.visualizer.get_available_animation_steps() == []
        plan._future.result(timeout=5)
        plan._poll()
        assert plan.visualizer.get_available_animation_steps() == [0, 7, 23]
        assert updates == [7]
        assert counts == [3]
        assert preloads == [[0, 7, 23]]
        assert plan.visualizer.stride_combo.currentText() == "x1"
        assert plan.interval_ms(0, 7) == 64
        assert plan.interval_ms(7, 23) == plan.interval_ms(23, 7) == 137
        plan.checkbox.setChecked(False)
        assert plan.visualizer.get_available_animation_steps() == provider.list_frames()
        assert counts[-1] == 5
        assert preloads[-1] == provider.list_frames()
    finally:
        plan.close()


def test_cli_plan_selects_only_bound_frames_and_can_be_disabled(qapp):
    plan, provider, updates, _, preloads, _ = _playback(qapp)
    errors = []
    try:
        plan.enable_for_startup(on_error=errors.append)
        assert plan.checkbox.isChecked()
        assert plan.steps == () and not preloads
        plan._future.result(timeout=5)
        plan._poll()
        assert not errors
        assert updates == [7]
        assert preloads == [[0, 7, 23]]
        plan.checkbox.setChecked(False)
        assert preloads[-1] == provider.list_frames()
    finally:
        plan.close()


@pytest.mark.parametrize("outcome", ["empty", "failure", "unsupported"])
def test_cli_plan_errors_never_fall_back_to_full_preload(qapp, monkeypatch, outcome):
    def load(*args):
        if outcome == "failure":
            raise ValueError("invalid binding")
        return (), {}

    monkeypatch.setattr("visualizer.src.sensing.playback.load_sensing_playback_plan", load)
    plan, provider, updates, counts, preloads, _ = _playback(qapp)
    errors = []
    try:
        if outcome == "unsupported":
            provider.snapshot = None
        plan.enable_for_startup(on_error=errors.append)
        if plan._future is not None:
            # Wait for completion without raising the worker's intentional error.
            plan._future.exception(timeout=5)
            plan._poll()
        assert len(errors) == 1
        assert plan.steps == ()
        assert not updates and not counts and not preloads
        assert plan._future is None
    finally:
        plan.close()


@pytest.mark.parametrize("outcome", ["empty", "failure", "replacement", "cancelled"])
def test_pending_plan_cannot_install_empty_failed_or_retired_results(qapp, monkeypatch, outcome):
    future = Future()

    class Executor:
        def __init__(self, **kwargs):
            pass

        def submit(self, *args):
            return future

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr("visualizer.src.sensing.playback.ThreadPoolExecutor", Executor)
    plan, provider, _, counts, _, messages = _playback(qapp)
    try:
        plan.checkbox.setChecked(True)
        if outcome == "cancelled":
            plan.checkbox.setChecked(False)
        elif outcome == "failure":
            future.set_exception(ValueError("invalid binding"))
        else:
            future.set_result(((), {}) if outcome == "empty" else ((0, 7), {0: 0, 7: 0.064}))
            if outcome == "replacement":
                plan.visualizer.frame_loader.provider = _provider()
        plan._poll()
        assert plan.steps is None
        assert not plan.checkbox.isChecked()
        assert 2 not in counts and 0 not in counts
        assert bool(messages) is (outcome != "cancelled")
    finally:
        plan.close()


def test_prepared_plan_uses_bound_scene_times_and_keeps_empty_results():
    scene = SimpleNamespace(
        records=[
            {
                "_sensing_display_provenance": {"bound_scene_frame_id": step},
                "timestamp_s": timestamp,
                "sensing": {"detections": [], "temporal": {"result": {"reference_time_s": 99}}},
            }
            for step, timestamp in ((7, 0.064), (0, 0.0), (7, 0.064), (23, 0.201))
        ]
    )
    assert prepared_playback_plan(scene) == ((0, 7, 23), {0: 0, 7: 0.064, 23: 0.201})


def test_default_preparation_gates_preload_until_verified_plan_and_scene_ready(qapp):
    plan, provider, updates, counts, preloads, messages = _playback(qapp)
    future = Future()
    try:
        plan.visualizer.ready = False
        plan.prepare_for_scenario(provider, future)
        assert plan.checkbox.isChecked() and plan.steps == ()
        assert not preloads and not provider.requests
        future.set_result(((0, 7, 23), {0: 0, 7: 0.064, 23: 0.201}))
        plan._poll()
        assert not updates and not preloads
        plan.visualizer.ready = True
        plan._poll()
        assert plan.steps == (0, 7, 23)
        assert updates == [7] and counts == [3]
        assert preloads == [[0, 7, 23]] and not messages
        assert not provider.requests
    finally:
        plan.close()


def test_opt_out_during_preparation_is_preserved_and_toggle_reuses_finished_plan(qapp):
    plan, provider, updates, _, preloads, _ = _playback(qapp)
    future = Future()
    try:
        plan.prepare_for_scenario(provider, future)
        plan.checkbox.setChecked(False)
        assert not future.cancelled()
        future.set_result(((0, 7, 23), {0: 0, 7: 0.064, 23: 0.201}))
        plan._poll()
        assert plan.steps is None and not plan.checkbox.isChecked()
        assert updates == [8] and preloads == [provider.list_frames()]
        plan.checkbox.setChecked(True)
        assert plan.steps == (0, 7, 23) and updates == [8, 7]
        plan.checkbox.setChecked(False)
        plan.checkbox.setChecked(True)
        assert plan.interval_ms(7, 23) == 137
        assert not provider.requests and plan._executor is None
    finally:
        plan.close()


def test_manually_indexed_plan_is_reused_until_provider_generation_changes(qapp):
    plan, provider, _, _, _, _ = _playback(qapp)
    try:
        plan.checkbox.setChecked(True)
        plan._future.result(timeout=5)
        plan._poll()
        reads = len(provider.requests)
        plan.checkbox.setChecked(False)
        plan.checkbox.setChecked(True)
        assert plan.steps == (0, 7, 23) and len(provider.requests) == reads
        plan.checkbox.setChecked(False)
        provider._generation_epoch = 1
        plan.checkbox.setChecked(True)
        assert plan.steps == ()
        plan._future.result(timeout=5)
        plan._poll()
        assert len(provider.requests) > reads
    finally:
        plan.close()


def test_default_empty_recording_falls_back_quietly(qapp):
    plan, provider, updates, _, preloads, messages = _playback(qapp)
    future = Future()
    try:
        plan.prepare_for_scenario(provider, future)
        future.set_result(((), {}))
        plan._poll()
        assert plan.steps is None and not plan.checkbox.isChecked()
        assert preloads == [provider.list_frames()] and updates == [8]
        assert not messages
    finally:
        plan.close()


def test_default_does_not_change_cli_benchmark_navigation(qapp):
    plan, provider, updates, counts, preloads, _ = _playback(qapp)
    try:
        plan.visualizer._cli_driven_frame_run = True
        future = Future()
        plan.prepare_for_scenario(provider, future)
        future.set_result(((0, 7), {0: 0, 7: 0.064}))
        plan._poll()
        assert plan.steps is None and not plan.checkbox.isChecked()
        assert not updates and not counts and not preloads
    finally:
        plan.close()


@pytest.mark.parametrize("steps", [(), (0, 7)])
def test_automatic_plan_respects_disabled_startup_preload(qapp, steps):
    plan, provider, _, _, preloads, _ = _playback(qapp)
    try:
        plan.visualizer.use_preload_mode = False
        future = Future()
        plan.prepare_for_scenario(provider, future)
        future.set_result((steps, {step: float(step) for step in steps}))
        plan._poll()
        assert not preloads
    finally:
        plan.close()


@pytest.mark.parametrize("change", ["replacement", "inventory", "closed"])
def test_default_preparation_rejects_retired_plan(qapp, change):
    plan, provider, updates, _, _, _ = _playback(qapp)
    future = Future()
    try:
        plan.prepare_for_scenario(provider, future)
        if change == "replacement":
            plan.visualizer.frame_loader.provider = _provider()
        elif change == "inventory":
            provider.snapshot = (*provider.snapshot, 100)
        else:
            plan.close()
        future.set_result(((0, 7), {0: 0, 7: 0.064}))
        plan._poll()
        assert plan.steps is None
        assert 7 not in updates
    finally:
        plan.close()


def test_authored_stride_survives_sensing_filter_and_retirement(qapp):
    plan, _, _, _, _, _ = _playback(qapp)
    try:
        plan.checkbox.setChecked(True)
        plan.set_full_timeline_stride("x32")
        plan._future.result(timeout=5)
        plan._poll()
        assert plan.visualizer.stride_combo.currentText() == "x1"
        plan.checkbox.setChecked(False)
        assert plan.visualizer.stride_combo.currentText() == "x32"
        plan.checkbox.setChecked(True)
        plan.close()
        assert plan._cached_plan is None and plan._prepared_future is None
        assert plan._token is None and not plan.timestamps and plan.steps is None
        assert not plan.checkbox.isChecked()
    finally:
        plan.close()


def test_async_source_inventory_does_not_overwrite_active_ordinal_count(qapp):
    plan, _, _, counts, _, _ = _playback(qapp)
    viz = plan.visualizer
    viz.ui_controller = SimpleNamespace(update_file_source_summary=lambda: None)
    viz.animation_service = SimpleNamespace(
        detect_mpc_frames=lambda **kw: kw["on_frame_count_updated"](1024)
    )
    try:
        plan.checkbox.setChecked(True)
        plan._future.result(timeout=5)
        plan._poll()
        assert viz.total_animation_steps == 3
        OrchavVisualizer.detect_mpc_frames(viz)
        assert viz.source_animation_steps == 1024
        assert viz.total_animation_steps == 3 and counts[-1] == 3
    finally:
        plan.close()


def test_completed_plan_retirement_rebinds_a_reused_panel_to_new_source(qapp):
    from visualizer.src.panels.sensing_panel import SensingPanel

    plan, provider, _, _, preloads, _ = _playback(qapp)
    viz = plan.visualizer
    retired = []
    panel = SimpleNamespace(
        parent=viz,
        playback=plan,
        preparation=SimpleNamespace(close=lambda: retired.append("preparation")),
        _retire_dashboard_session=lambda: retired.append("dashboard"),
        _dashboard_cache=object(),
        _sensing_window=SimpleNamespace(reset_recording=lambda **kw: retired.append("window")),
    )
    try:
        plan.checkbox.setChecked(True)
        plan._future.result(timeout=5)
        plan._poll()
        assert plan.steps == (0, 7, 23)
        SensingPanel.retire_recording_source(panel)
        assert retired == ["preparation", "dashboard", "window"]
        assert panel.preparation is None and panel._dashboard_cache is None
        assert viz.sensing_playback is None and plan._cached_plan is None
        assert not plan.timestamps and plan._prepared_future is None
        # Overlapping frame IDs now have different physical times and ownership.
        changed = _provider()
        changed.frames = {0: replace(provider.frames[0], timestamp_s=12.0)}
        changed.snapshot = (0,)
        viz.frame_source = changed
        viz.frame_loader.provider = changed
        viz.sensing_playback = plan
        future = Future()
        future.set_result(((0,), {0: 12.0}))
        plan.prepare_for_scenario(changed, future)
        assert plan.steps == (0,) and plan.timestamps == {0: 12.0}
        assert viz.total_animation_steps == 1 and preloads[-1] == [0]
        plan.checkbox.setChecked(False)
        assert viz.get_available_animation_steps() == [0]
    finally:
        plan.close()
