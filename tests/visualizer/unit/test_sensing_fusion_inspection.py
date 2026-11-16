"""Actual Qt lifetime, shared reader and native selection worker regressions."""

from concurrent.futures import Future
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent

from shared.sensing.fusion.core import MonostaticFusion, Observation, Sensor, measurement_model
from shared.sensing.fusion.result import FusionResult
from visualizer.src.sensing.fusion_inspection import FusionSelection, track_details, world_layers
from visualizer.src.sensing.fusion_window import FusionWindow


@pytest.fixture
def pretracking_request():
    from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload
    from visualizer.src.sensing.update_request import SensingUpdateRequest

    report = {
        "range_m": 5.0,
        "velocity_m_s": 1.0,
        "x_m": 5.0,
        "y_m": 0.0,
        "aoa_az_deg": 0.0,
        "range_bin": 7,
        "doppler_bin": 3,
    }
    sensing = build_visualizer_sensing_payload(
        {
            "detections": [],  # The first hit is not yet a confirmed track.
            "detections_before_tracking": [report],
            "detections_before_tracking_metadata": {
                "schema_version": 1,
                "stage": "after_angle_refinement_before_tracking",
                "coordinate_frame": "receiver_local",
                "tracking_applied": False,
                "reference_time_s": 0.0,
            },
        }
    )
    request = SensingUpdateRequest(display_frame={"sensing": sensing}, state_frame={}, state_step=0)
    lineage = {
        "acquisition_id": sensing["temporal"]["acquisition"]["acquisition_id"],
        "time_s": 0.0,
        "processing_recipe_id": sensing["temporal"]["result"]["processing_recipe_id"],
        "detection_collection": "detections_before_tracking",
    }
    return request, lineage


def test_native_report_projection_keeps_tentative_rows_and_collection_identity(pretracking_request):
    from visualizer.src.sensing.fusion_inspection import native_observation_request
    from visualizer.src.sensing.selection import matches_detection, select_estimate

    request, lineage = pretracking_request
    original = request.display_frame["sensing"]
    display = native_observation_request(request, lineage)
    projected = display.display_frame["sensing"]
    assert original["detections"] == []
    assert display.state_frame is request.state_frame
    assert projected["products"] is original["products"]
    assert projected["detections"] is original["detections_before_tracking"]
    selection = select_estimate(projected, detection_index=0)
    assert selection.track_id is None
    assert selection.detection_collection == "detections_before_tracking"
    assert projected["detections"][selection.detection_index]["range_bin"] == 7
    assert matches_detection(selection, projected, 0, projected["detections"][0])
    assert not matches_detection(selection, original, 0, projected["detections"][0])
    assert native_observation_request(request, {}) is request


def test_native_detection_plots_receive_exact_selected_reports(qtbot, pretracking_request):
    from visualizer.src.sensing.fusion_inspection import (
        native_observation_request,
        native_result_request,
    )
    from visualizer.src.sensing.sensing_window import SensingDashboard

    request, lineage = pretracking_request
    request = native_result_request(native_observation_request(request, lineage))
    assert request.recording_history.records[0].detections == tuple(
        request.display_frame["sensing"]["detections_before_tracking"]
    )
    window = SensingDashboard()
    qtbot.addWidget(window)
    window._set_view_mode("analysis")
    window._render_tracking_history(request.recording_history, 0)
    assert window._gt_range_det_scatter.getData()[1].tolist() == [5.0]
    assert window._gt_speed_det_scatter.getData()[1].tolist() == [1.0]
    assert "loading" not in window.gt_range_plot.getPlotItem().titleLabel.text.lower()


@pytest.mark.parametrize("mismatch", ["epoch", "collection", "missing", "metadata"])
def test_native_report_projection_rejects_inexact_source_binding(pretracking_request, mismatch):
    from visualizer.src.sensing.fusion_inspection import native_observation_request

    request, lineage = pretracking_request
    if mismatch == "epoch":
        lineage["time_s"] = 0.1
    elif mismatch == "collection":
        lineage["detection_collection"] = "tracker_decisions"
    elif mismatch == "missing":
        del request.display_frame["sensing"]["detections_before_tracking"]
    else:
        request.display_frame["sensing"]["detections_before_tracking_metadata"][
            "reference_time_s"
        ] = 0.1
    with pytest.raises(ValueError):
        native_observation_request(request, lineage)


def test_fusion_linked_selection_uses_raw_collection_in_both_directions(pretracking_request):
    from dataclasses import replace

    from visualizer.src.sensing.fusion_inspection import native_observation_request

    request, lineage = pretracking_request
    native = {**lineage, "detection_index": 0}
    track = {"track_id": 9, "sources": [{"sensor_id": "a", "native_source": native}]}
    highlighted, selected = [], []
    window = SimpleNamespace(
        selected=SimpleNamespace(track_id=9),
        _last_request=native_observation_request(request, lineage),
        reader=SimpleNamespace(
            epoch=lambda index: {"tracks": [track]}, inspection_id="saved-fusion"
        ),
        timeline=SimpleNamespace(value=lambda: 0),
        sensor_choice=SimpleNamespace(currentText=lambda: "a"),
        set_linked_selection=highlighted.append,
        select_fusion=selected.append,
    )
    FusionWindow._highlight_native(window)
    assert len(highlighted) == 1
    FusionWindow._native_selected(window, highlighted[0])
    assert selected == [FusionSelection("saved-fusion", 9, 0)]
    FusionWindow._native_selected(
        window, replace(highlighted[0], detection_collection="detections")
    )
    assert len(selected) == 1
    window._last_request = request
    FusionWindow._highlight_native(window)
    assert len(highlighted) == 1  # Never redirect a raw row to confirmed row zero.


@pytest.fixture
def result():
    import numpy as np

    s = Sensor("a", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    processor = MonostaticFusion([s])
    history = []
    for i in range(5):
        z, _ = measurement_model(np.array([5.0 + i * 0.1, 0.0, 0.0, 1.0, 0.0, 0.0]), s)
        history.append(processor.step(i * 0.1, [Observation("a", str(i), i * 0.1, *z)]))
    return FusionResult(
        {
            "schema_version": 1,
            "histories": {"fused": history, "a": deepcopy(history)},
            "sources": {
                "a": {
                    "sensor": {
                        "sensor_id": "a",
                        "position_m": [0, 0, 0],
                        "orientation_rad": [0, 0, 0],
                    }
                }
            },
            "truth_for_evaluation": [],
            "observations": {"a": [{"detections": []} for _ in history]},
        }
    )


def test_layers_and_details_do_not_change_estimation(result):
    before = deepcopy(dict(result))
    selection = FusionSelection(result.inspection_id, 1, 2)
    assert len(world_layers(result, 2, {"sensors", "fused"}, selection)) == 2
    assert not world_layers(result, 2, set(), selection)
    details = track_details(result, selection, "a")
    assert details["world_speed_m_s"] > 0.9
    assert details["a_predicted_radial_closing_velocity_m_s"] < -0.9
    assert len(details["confirmation_history"]) == 3
    assert dict(result) == before


def test_all_saved_sensor_labels_follow_pose_visibility_without_changing_sources(result):
    import numpy as np

    from visualizer.src.sensing.fusion_inspection import world_sensor_labels

    data = deepcopy(dict(result))
    data["sources"]["South"] = {
        "sensor": {
            "sensor_id": "South",
            "position_m": [10, -10, 1.5],
            "orientation_rad": [1.0, 0, 0],
        }
    }
    reader = FusionResult(data)
    labels = world_sensor_labels(reader, {"sensors"})
    assert [label.payload.text for label in labels] == ["a radar", "South radar"]
    for label, source in zip(labels, data["sources"].values(), strict=True):
        position = np.asarray(source["sensor"]["position_m"])
        np.testing.assert_array_equal(label.metadata["layout_anchor"], position)
        np.testing.assert_array_equal(label.transform_matrix[:3, 3], position + [0, 0, 0.5])
        assert label.metadata["locator_role"] == "radio"
        assert label.payload.screen_space
    assert world_sensor_labels(reader, {"fused"}) == []
    assert dict(reader) == data


def test_fusion_trails_use_only_saved_past_states_and_keep_modes_separate(result):
    import numpy as np

    before = deepcopy(dict(result))
    layers = world_layers(result, 2, {"fused", "tracks:a", "trails"})
    trails = [layer for layer in layers if "trail" in layer.name]
    assert {layer.name for layer in trails} == {"fusion_trail_fused_1", "fusion_trail_a_1"}
    for layer in trails:
        mode = layer.metadata["fusion_selection"].mode
        expected = [epoch["tracks"][0]["position_m"] for epoch in result["histories"][mode][:3]]
        np.testing.assert_allclose(layer.payload.points, expected)
        np.testing.assert_array_equal(layer.payload.lines, [[0, 1], [1, 2]])
        assert layer.metadata["epoch_indices"] == (0, 1, 2)
        assert layer.metadata["fusion_selection"].epoch_index == 2
    assert not world_layers(result, 2, {"trails"})
    assert not any("trail" in layer.name for layer in world_layers(result, 0, {"fused", "trails"}))
    assert dict(result) == before


def test_fusion_trails_preserve_gaps_and_recorded_coasting_states(result):
    import numpy as np

    data = deepcopy(dict(result))
    history = data["histories"]["fused"]
    history[1]["tracks"] = []
    history[2]["tracks"][0]["predicted_only"] = True
    history[4]["tracks"][0]["track_id"] = 2
    changed = FusionResult(data)
    layers = world_layers(changed, 4, {"fused", "trails"})
    trail = next(layer for layer in layers if "trail" in layer.name)
    assert trail.name == "fusion_trail_fused_1"
    assert trail.metadata["epoch_indices"] == (0, 2, 3)
    assert trail.metadata["predicted_only"] == (False, True, False)
    assert trail.metadata["current"] is False
    assert trail.metadata["fusion_selection"].epoch_index == 3
    np.testing.assert_array_equal(trail.payload.lines, [[1, 2]])
    np.testing.assert_allclose(
        trail.payload.points, [history[i]["tracks"][0]["position_m"] for i in (0, 2, 3)]
    )
    assert dict(changed) == data


def test_saved_tracking_adapter_keeps_native_detections_and_radial_semantics(result):
    from visualizer.src.sensing.saved_tracking import saved_tracking_history

    before = deepcopy(dict(result))
    history = saved_tracking_history(result, "a")
    assert len(history.records) == 5
    assert not history.records[0].detections
    track = history.records[-1].detections[0]
    assert track["velocity_m_s"] < -0.9  # Receding motion is negative closing speed.
    assert track["source_kind"] == "saved_world_track"
    assert dict(result) == before


def test_adjacent_tracking_discovery_requires_exact_single_source(tmp_path, result):
    import json

    from visualizer.src.sensing.saved_tracking import adjacent_single_sensor_result

    native = tmp_path / "native"
    native.mkdir()
    tracking = tmp_path / "tracking"
    tracking.mkdir()
    path = tracking / "fusion_results.json"
    data = deepcopy(dict(result))
    data["sources"]["a"]["source"] = "../native"
    path.write_text(json.dumps(data))
    assert adjacent_single_sensor_result(native / "scenario.yaml") == path
    data["sources"]["a"]["source"] = "../unrelated"
    path.write_text(json.dumps(data))
    assert adjacent_single_sensor_result(native) is None


def test_saved_tracking_hashes_are_verified_and_changes_invalidate_cache(
    tmp_path, result, monkeypatch
):
    import json
    from threading import Event

    from shared.sensing.fusion.experiment import fingerprints
    from visualizer.src.sensing.saved_tracking import SavedTrackingLoader

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "display-cache"))

    native = tmp_path / "native"
    frames = native / "frames"
    frames.mkdir(parents=True)
    (frames / "frames_manifest.json").write_text("{}")
    chunk = frames / "frames.h5"
    chunk.write_bytes(b"original bytes; no provider read is needed for hash verification")
    tracking = tmp_path / "tracking"
    tracking.mkdir()
    data = deepcopy(dict(result))
    data["sources"]["a"].update(source="../native", hashes=fingerprints(frames))
    path = tracking / "fusion_results.json"
    path.write_text(json.dumps(data))
    loader = SavedTrackingLoader(native)
    saved, error = loader.load(Event())
    assert error is None and saved is not None
    assert loader.load(Event())[0] is saved
    # A new process/loader can reuse verification only for these exact files.
    with monkeypatch.context() as patch:
        patch.setattr(
            "shared.sensing.fusion.experiment.fingerprints",
            lambda *_: pytest.fail("Unchanged native chunks should not be hashed again"),
        )
        reopened, error = SavedTrackingLoader(native).load(Event())
        assert reopened is not None and error is None
    chunk.write_bytes(b"different authoritative recording")
    saved, error = loader.load(Event())
    assert saved is None and "hash mismatch" in error
    assert loader.load(Event())[0] is None


def test_single_source_result_launch_uses_the_normal_dashboard(tmp_path, result):
    import json

    from visualizer.src.sensing.fusion_window import open_fusion_window

    native = tmp_path / "native"
    native.mkdir()
    (native / "scenario.yaml").write_text("{}")
    saved = tmp_path / "tracking"
    saved.mkdir()
    data = deepcopy(dict(result))
    data["sources"]["a"]["source"] = "../native"
    path = saved / "fusion_results.json"
    path.write_text(json.dumps(data))
    calls = []
    dashboard = object()
    panel = SimpleNamespace(
        _retire_dashboard_session=lambda: calls.append("retire"),
        _on_open_dashboard=lambda: calls.append("dashboard"),
        _sensing_window=dashboard,
    )
    visualizer = SimpleNamespace(
        open_scenario=lambda *args, **kwargs: SimpleNamespace(succeeded=True),
        ui_manager=SimpleNamespace(ensure_panel=lambda key: None, panels={"sensing": panel}),
    )
    assert open_fusion_window(visualizer, path) is dashboard
    assert panel.saved_result_path == str(path)
    assert calls == ["retire", "dashboard"]
    assert not hasattr(visualizer, "fusion_window")


def test_saved_scene_history_has_no_future_states_and_no_implicit_prediction(result):
    from visualizer.src.sensing.saved_tracking import SavedTracking, saved_tracking_history
    from visualizer.src.sensing.saved_tracking_scene import saved_track_layers
    from visualizer.src.services.sensing_service import SensingService

    before = deepcopy(dict(result))
    binding = SavedTracking(result, "a", saved_tracking_history(result, "a"))
    service = SensingService(SimpleNamespace())
    layers, labels = saved_track_layers(
        binding, 2, history_count=None, arrows=True, labels=True, service=service
    )
    trail = next(layer for layer in layers if "trail" in layer.name)
    assert len(trail.payload.points) == 3
    assert any("arrow" in layer.name for layer in layers)
    assert labels and "confirmed" in labels[0].payload.text
    historical, labels = saved_track_layers(
        binding, 2, history_count=2, arrows=True, labels=True, service=service, at_epoch=False
    )
    assert not any("arrow" in layer.name for layer in historical)
    assert all(layer.metadata["current"] is False for layer in historical)
    assert "history" in labels[0].payload.text
    assert len(next(layer for layer in historical if "trail" in layer.name).payload.points) == 2
    assert dict(result) == before


def test_saved_tracking_inspector_clears_cross_source_selection(qtbot, result):
    from visualizer.src.sensing.saved_tracking import SavedTracking, saved_tracking_history
    from visualizer.src.sensing.tracking_inspector import TrackingInspector

    widget = TrackingInspector()
    qtbot.addWidget(widget)
    widget.update_result(
        SavedTracking(result, "a", saved_tracking_history(result, "a")), 0.2, "tracking"
    )
    widget.select_track(1)
    assert widget.selected_track()["confirmed"]
    # Another run with the same numeric track ID must not inherit the selection.
    changed = deepcopy(dict(result))
    changed["histories"]["fused"][0]["tracks"][0]["position_m"][0] += 1
    other = FusionResult(changed)
    widget.update_result(
        SavedTracking(other, "a", saved_tracking_history(other, "a")), 0.2, "tracking"
    )
    assert widget.selected_id is None


class FakeSession:
    instances = []

    def __init__(self):
        self.closed = False
        self.requests = []
        self.instances.append(self)

    def open(self, path):
        self.opening = Future()
        return self.opening

    def request(self, *key):
        future = Future()
        self.requests.append((key, future))
        return future

    def close(self):
        self.closed = True


def test_fusion_native_views_keep_whole_recording_choices_unavailable(qtbot, monkeypatch):
    from tests.visualizer.unit.test_sensing_dashboard_recording import _request
    from visualizer.src.sensing import fusion_window
    from visualizer.src.sensing.fusion_inspection import native_result_request

    monkeypatch.setattr(fusion_window, "FusionInspectionSession", FakeSession)
    window = FusionWindow(SimpleNamespace(renderer=None), "result.json")
    qtbot.addWidget(window)
    window.show()
    window.native_section.expand()
    for step in (0, 3, 1):
        # A sampled native RD result normally enables Motion. Fusion has only
        # the selected observation record, so that capability must stay off.
        request = native_result_request(_request(step))
        window.update_from_request(request)
        assert window._rendered_request is request
        assert len(request.recording_history.records) == 1
        for mode in ("summary", "micro-doppler"):
            for combo in (window.workspace_box, window._view_mode_box):
                item = combo.model().item(combo.findData(mode))
                assert not item.isEnabled()
                assert "source recording" in item.toolTip()
        for stale_mode in ("summary", "evaluation", "micro-doppler"):
            window.apply_display_preset({"view_mode": stale_mode})
            assert window.workspace_box.currentData() == "current"
            assert window.request_view_mode() == "analysis"
        assert (
            window.workspace_box.model().item(window.workspace_box.findData("current")).isEnabled()
        )
    window.close()


def test_native_recording_dashboard_retains_whole_recording_workspaces(qtbot):
    from tests.visualizer.unit.test_sensing_dashboard_recording import _history, _request
    from visualizer.src.sensing.sensing_window import SensingDashboard

    window = SensingDashboard()
    qtbot.addWidget(window)
    window.show()
    window.update_from_request(_request(0, _history()))
    assert [window.workspace_box.itemData(i) for i in range(window.workspace_box.count())] == [
        "summary",
        "current",
        "micro-doppler",
        "tracking",
    ]
    for mode in ("summary", "micro-doppler"):
        index = window.workspace_box.findData(mode)
        assert window.workspace_box.model().item(index).isEnabled()
        window.workspace_box.setCurrentIndex(index)
        assert window.request_view_mode() == mode
    assert window.collapsed_doppler_image.image is not None
    window.close()


def test_single_sensor_report_is_optional_and_opens_saved_html(
    qtbot, monkeypatch, tmp_path, result
):
    from visualizer.src.sensing import fusion_window

    monkeypatch.setattr(fusion_window, "FusionInspectionSession", FakeSession)
    opened = []
    monkeypatch.setattr(
        fusion_window.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile())
    )
    window = FusionWindow(SimpleNamespace(renderer=None), tmp_path)
    window.session.opening.set_result(result)
    window._poll()
    assert window.uav_report_button.isHidden()
    assert window.visibility["fused"].text() == "3D tracks"
    assert "tracks:a" not in window.visibility
    report = tmp_path / "uav_evaluation/index.html"
    report.parent.mkdir()
    report.write_text("<h1>Saved evaluation</h1>")
    window._open_uav_report()
    assert opened == [str(report)]
    window.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_qt_seek_stale_completion_cancel_restore_and_deferred_delete(qtbot, monkeypatch, result):
    import shiboken6

    from visualizer.src.sensing import fusion_window

    monkeypatch.setattr(fusion_window, "FusionInspectionSession", FakeSession)
    viz = SimpleNamespace(renderer=None)
    window = FusionWindow(viz, "result.json")
    window.native_section.expand()
    window.session.opening.set_result(result)
    window._poll()
    first = window.pending
    window.timeline.setValue(2)
    first.set_exception(ValueError("stale completion must not be displayed"))
    window._poll()
    assert "stale completion" not in window.fusion_status.text()
    window.pending.set_exception(FileNotFoundError("missing native data"))
    window._poll()
    assert "missing native data" in window.fusion_status.text()
    window._pause_loading()
    assert window.session.closed
    window._pause_loading()
    assert not window.session.closed
    assert window.visibility["trails"].isChecked()
    window.visibility["trails"].setChecked(False)
    window.visibility["truth"].setChecked(True)
    window.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not shiboken6.isValid(window)
    reopened = FusionWindow(viz, "result.json")
    reopened.session.opening.set_result(result)
    reopened._poll()
    assert reopened.timeline.value() == 2 and reopened.visibility["truth"].isChecked()
    assert not reopened.visibility["trails"].isChecked()
    session = reopened.session
    reopened.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert session.closed and not shiboken6.isValid(reopened)


def test_native_source_change_during_read_and_cache_invalidation(
    tmp_path, monkeypatch, result, pretracking_request
):
    import os

    from shared.sensing.fusion.experiment import fingerprints
    from visualizer.src.sensing import fusion_inspection

    source = tmp_path / "source"
    source.mkdir()
    (source / "frames_manifest.json").write_text("{}")
    chunk = source / "one.h5"
    chunk.write_bytes(b"first")
    payload = deepcopy(dict(result))
    payload["sources"]["a"].update(
        source="source",
        hashes=fingerprints(source),
        lineage=[{"scene_frame_id": i} for i in range(5)],
    )
    (tmp_path / "fusion_results.json").write_text(__import__("json").dumps(payload))
    reader = FusionResult(tmp_path)
    closed = []
    provider = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(fusion_inspection, "_open_provider", lambda path: (source, None, provider))
    monkeypatch.setattr(
        fusion_inspection,
        "SensingService",
        lambda proxy: SimpleNamespace(configure_background_reader=lambda *args: None),
    )
    native = pretracking_request[0]
    monkeypatch.setattr(fusion_inspection, "load_dashboard_request", lambda *args: native)
    session = fusion_inspection.FusionInspectionSession()
    try:
        request = session.request(reader, 0, "a").result(timeout=5)
        assert request.source_identity == fusion_inspection.DashboardSession.provider_identity(
            provider
        )
        assert request.source_identity
        assert request.display_frame is native.display_frame
        assert request.state_frame is native.state_frame
        assert native.source_identity == ()
        assert session.request(reader, 0, "a").result(timeout=5) is request
        stamp = chunk.stat()
        chunk.write_bytes(b"other")
        os.utime(chunk, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        with pytest.raises(ValueError, match="hash mismatch"):
            session.request(reader, 0, "a").result(timeout=5)
        chunk.write_bytes(b"first")

        def changed(*args):
            chunk.write_bytes(b"later-content-change")
            return "must not cache"

        monkeypatch.setattr(fusion_inspection, "load_dashboard_request", changed)
        with pytest.raises(RuntimeError, match="changed while loading"):
            session.request(reader, 1, "a").result(timeout=5)
        assert not session.cache
    finally:
        session.close()
        session.executor.shutdown(wait=True)
    assert closed


def test_scene_playback_follows_fusion_without_seeking_or_pausing(
    qtbot, monkeypatch, result, tmp_path
):
    from visualizer.src.sensing import fusion_window

    monkeypatch.setattr(fusion_window, "FusionInspectionSession", FakeSession)
    data = deepcopy(dict(result))
    data["sources"]["a"].update(
        source=str(tmp_path), lineage=[{"scene_frame_id": i * 10} for i in range(5)]
    )
    calls = []
    viz = SimpleNamespace(
        renderer=None,
        current_scenario_path=tmp_path / "scenario.yaml",
        animation_running=True,
        animation_controller=SimpleNamespace(toggle_animation=lambda: calls.append("pause")),
        ui_controller=SimpleNamespace(handle_frame_input_changed=lambda i: calls.append(i)),
        get_available_animation_steps=lambda: list(range(41)),
    )
    window = FusionWindow(viz, "result.json")
    qtbot.addWidget(window)
    path = tmp_path / "fusion_results.json"
    path.write_text(__import__("json").dumps(data))
    window.session.opening.set_result(FusionResult(path))
    window._poll()
    assert window.reader is not None, window.fusion_status.text()
    calls.clear()
    for step in (10, 11, 20, 30):
        window.follow_scene(step)
    assert window.timeline.value() == 3
    assert calls == [] and viz.animation_running
    assert window.session.requests == []  # Optional sensor plots stay lazy.
    window.timeline.setValue(2)  # An explicit seek still navigates the main scene.
    assert calls == ["pause", 21]
    window.close()


def test_fusion_is_embedded_and_closing_removes_its_tab(qtbot, monkeypatch, result):
    from PySide6.QtWidgets import QTabWidget, QWidget

    from visualizer.src.panels.sensing_panel import SensingPanel
    from visualizer.src.sensing import fusion_window

    monkeypatch.setattr(fusion_window, "FusionInspectionSession", FakeSession)
    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    tabs.addTab(QWidget(), "Results")
    panel = SensingPanel.__new__(SensingPanel)
    panel.recording_tabs = tabs
    viz = SimpleNamespace(renderer=None, ui_manager=SimpleNamespace(panels={"sensing": panel}))
    window = FusionWindow(viz, "result.json")
    panel.attach_fusion(window)
    window.session.opening.set_result(result)
    window._poll()
    assert tabs.currentWidget() is window and not window.isWindow()
    assert "saved tracks" in window.track_summary.text()
    window.close()
    assert tabs.count() == 1 and tabs.tabText(0) == "Results"


def test_deferred_volume_close_does_not_show_results_over_the_fusion_tab(qtbot):
    from PySide6.QtWidgets import QTabWidget, QWidget

    from visualizer.src.panels.sensing_panel import SensingPanel

    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    results, fusion = QWidget(), QWidget()
    tabs.addTab(results, "Results")
    tabs.addTab(fusion, "Fusion")
    tabs.setCurrentWidget(fusion)
    tabs.show()
    panel = SensingPanel.__new__(SensingPanel)
    panel.parent = SimpleNamespace(sensing_volume_window=SimpleNamespace(closed=True))
    panel.recording_tabs, panel.recording_controls = tabs, results
    panel.volume_host = QWidget(tabs)
    panel._volume_closed()
    assert results.isHidden() and not fusion.isHidden()
    tabs.setCurrentWidget(results)
    panel._volume_closed()
    assert not results.isHidden() and fusion.isHidden()


def test_saved_only_fusion_uses_main_transport_without_native_cache(qtbot, monkeypatch, result):
    from PySide6.QtWidgets import QLabel, QSlider, QSpinBox

    from tests.visualizer.unit.test_animation_scrubbing import _PlaybackControls, _Timer
    from visualizer.src.controllers.animation_controller import AnimationController
    from visualizer.src.sensing import fusion_window
    from visualizer.src.sensing.fusion_playback import FusionPlayback
    from visualizer.visualizer import OrchavVisualizer

    monkeypatch.setattr(fusion_window, "FusionInspectionSession", FakeSession)
    viz = OrchavVisualizer.__new__(OrchavVisualizer)
    viz.renderer = None
    viz.total_animation_steps = 60
    viz.animation_step = 0
    viz.animation_running = False
    viz.play_direction = 1
    viz.animation_timer = _Timer()
    viz.update_timer = _Timer()
    viz._active_poll_interval_ms = viz._idle_poll_interval_ms = 100
    viz._pending_slider_scrub_index = None
    viz.slider_scrub_timer = _Timer()
    viz.step_slider, viz.frame_input, viz.step_label = QSlider(), QSpinBox(), QLabel()
    viz.set_state = lambda **kw: None
    viz.schedule_update = lambda: None
    viz.ui_manager = SimpleNamespace(
        update_total_steps=lambda count: (
            viz.step_slider.setRange(0, count - 1),
            viz.frame_input.setRange(1, count),
        )
    )
    viz.ui_controller = SimpleNamespace(
        update_performance_display=lambda: None,
        refresh_status_telemetry=lambda: None,
        update_frame_context=lambda step: pytest.fail("Tried to read a native frame"),
    )
    cache = SimpleNamespace(has_frame=lambda step: pytest.fail("Waited for absent native data"))
    controller = AnimationController(
        viz, SimpleNamespace(cache_service=cache), playback_controls=_PlaybackControls()
    )
    viz.animation_controller = controller
    window = FusionWindow(viz, "result.json")
    qtbot.addWidget(window)
    window.reader = result
    window.timeline.setRange(0, len(result.times) - 1)
    # Install independently of scene loading to exercise the actual transport.
    window.result_playback = FusionPlayback(window)
    assert viz.get_available_animation_steps() == list(range(5))
    assert controller._cached_timestamp_interval_ms(1, 2) == 100
    controller.toggle_animation()
    for _ in range(3):
        controller.handle_animation_tick()
    assert viz.animation_step == window.timeline.value() == 3
    assert viz.animation_running and not controller._buffering
    controller.toggle_animation()
    controller.previous_frame()
    assert viz.animation_step == window.timeline.value() == 2
    window.result_playback.close()
    assert viz.result_playback is None and viz.total_animation_steps == 60
    assert viz.animation_step == 0
    window.close()
