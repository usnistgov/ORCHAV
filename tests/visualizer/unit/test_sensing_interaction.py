"""Saved identities, physical-time navigation and disposable display-cache checks."""

import os
import time
from concurrent.futures import CancelledError
from types import SimpleNamespace

import numpy as np
import pytest

from tests.visualizer.unit.test_sensing_dashboard_recording import _frame, _history, _request
from visualizer.src.sensing.dashboard_data import DashboardSession
from visualizer.src.sensing.display_style import color_rgba, identity_color
from visualizer.src.sensing.recording_cache import RecordingCache
from visualizer.src.sensing.result_navigation import (
    adjacent_result,
    nearest_result,
    result_locations,
)
from visualizer.src.sensing.selection import matches_detection, select_estimate
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.services.sensing_service import SensingService


def test_result_navigation_uses_bound_scenes_not_storage_rows_or_stride():
    frames = [(800, _frame(3)), (900, _frame(19)), (1000, _frame(27))]
    empty = _frame(2)
    empty["sensing"]["temporal"]["result"]["reference_time_s"] = None
    rows = result_locations(tuple([*frames, (2, empty)]))
    assert [r.scene_id for r in rows] == [3, 19, 27]
    assert adjacent_result(rows, 10, list(range(30)), 1).scene_id == 19
    assert adjacent_result(rows, 10, list(range(30)), -1).scene_id == 3
    assert adjacent_result(rows, 27, list(range(30)), 1) is None
    assert nearest_result(rows, rows[1].time_s + 0.01).storage_id == 900
    assert nearest_result(rows, -1) is None
    assert (
        rows[1].acquisition_id
        == frames[1][1]["sensing"]["temporal"]["acquisition"]["acquisition_id"]
    )


def test_result_navigation_keeps_metadata_only_results_and_zero_detections():
    frames = [(800, _frame(3)), (900, _frame(19))]
    for _, frame in frames:
        frame["sensing"]["products"] = {}
        frame["sensing"]["detections"] = []
    rows = result_locations(tuple(frames))
    assert [row.scene_id for row in rows] == [3, 19]
    assert adjacent_result(rows, 19, list(range(20)), -1).storage_id == 800


def test_selection_does_not_cross_clocks_runs_pairs_or_acquisitions():
    sensing = _frame()["sensing"]
    sensing["detections"] = [{"track_id": 7}, {}]
    selection = select_estimate(sensing, detection_index=0)
    assert matches_detection(selection, sensing, 0, sensing["detections"][0])
    other = _frame(1)["sensing"]
    assert not matches_detection(selection, other, 0, {"track_id": 7})
    track_selection = select_estimate(sensing, track_id=7)
    assert matches_detection(track_selection, other, 0, {"track_id": 7})
    other["temporal"]["result"]["processing_run_id"] = "different-run"
    assert not matches_detection(selection, other, 0, {"track_id": 7})
    untracked = select_estimate(sensing, detection_index=1)
    assert not matches_detection(untracked, _frame(1)["sensing"], 1, {})


def test_track_colors_do_not_change_with_confirmation_or_visible_neighbors():
    service = SensingService(SimpleNamespace())
    np.testing.assert_allclose(
        service._track_rgba(9, True)[:3], color_rgba(identity_color("9"))[:3]
    )
    np.testing.assert_array_equal(
        service._track_rgba(9, True)[:3], service._track_rgba(9, False)[:3]
    )
    assert service._track_rgba(9, False)[3] < service._track_rgba(9, True)[3]


def test_display_cache_invalidation_detects_in_place_rewrites_even_with_restored_mtime(tmp_path):
    manifest, chunk = tmp_path / "frames_manifest.json", tmp_path / "chunk.h5"
    manifest.write_text('{"generation":"one"}')
    chunk.write_bytes(b"old data")
    cache = RecordingCache((manifest, chunk), tmp_path / "cache")
    cache.save("overview", {"spectrum": np.arange(8.0)})
    assert cache.load("overview")["spectrum"] == list(range(8))
    stamp = chunk.stat()
    # Some filesystems advance ctime only once per clock tick. The fixture
    # must actually change that stamp before checking restored-mtime detection.
    time.sleep(0.02)
    chunk.write_bytes(b"new data")
    os.utime(chunk, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert cache.load("overview") is None
    changed = RecordingCache((manifest, chunk), cache.root)
    assert changed.load("overview") is None
    changed.save("index", {"rows": [1, 3]})
    manifest.write_text('{"generation":"two"}')
    assert changed.load("index") is None


def test_corrupt_or_unwritable_cache_falls_back_without_changing_source(tmp_path):
    source = tmp_path / "manifest"
    source.write_bytes(b"manifest")
    cache = RecordingCache((source,), tmp_path / "cache")
    cache.save("index", {"rows": []})
    next(cache.root.glob("*.json.gz")).write_bytes(b"broken")
    assert cache.load("index") is None
    blocked = tmp_path / "blocked"
    blocked.write_bytes(b"not a directory")
    RecordingCache((source,), blocked).save("index", {})
    assert source.read_bytes() == b"manifest"


def test_cancelled_index_job_stops_before_reading():
    session = DashboardSession(SimpleNamespace())
    session.close()
    with pytest.raises(CancelledError):
        session.service.report_read_progress("Indexing", 0, 10)


def test_plot_click_selection_marks_saved_bin_and_retains_track_identity(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    request = _request(0, _history())
    sensing = request.display_frame["sensing"]
    # The GUI consumes authored detections; it does not manufacture assignments.
    sensing["detections"] = [
        {
            "track_id": 7,
            "range_m": 1.0,
            "velocity_m_s": 0.0,
            "doppler_hz": 0.0,
            "range_bin": 1,
            "doppler_bin": 2,
        }
    ]
    window._initial_view_selected = True
    window._set_view_mode("analysis")
    window.update_from_request(request)
    assert not window.selection_tools.isVisible()
    assert not window.selection_tools_toggle.isVisible()
    assert not hasattr(window, "documentation_button")
    window.select_detection_index(0)
    assert window.selection_tools_toggle.isVisible()
    assert window.selection_tools_toggle.parentWidget() is window.result_scroll.widget()
    assert not window.selection_tools.isVisible()
    window.selection_tools_toggle.setChecked(True)
    assert window.selection_tools.isVisible()
    assert window._linked_selection.track_id == 7
    assert "Track 7" in window.selection_label.text()
    assert window._selected_rd_cells.data.size == 1
    window.set_linked_selection(None)
    assert not window.selection_tools.isVisible()
    assert not window.selection_tools_toggle.isVisible()
    assert window.request_view_mode() == "analysis"
    assert not window.cell_explanation.isVisible()
    assert window._selected_rd_cells.data.size == 0
    window.select_detection_index(0)
    assert window.selection_tools.isVisible()
    with qtbot.waitSignal(window.selection_changed) as signal:
        window.reset_recording()
    assert signal.args == [None]
    assert window._linked_selection is None
    assert not window.selection_tools.isVisible()
    assert not window.selection_tools_toggle.isVisible()
    assert not window.locate_selection_button.isEnabled()
    assert not window.next_result_button.isEnabled()


def test_pygfx_click_maps_visible_vertex_and_rejects_drags():
    from visualizer.src.renderers.pygfx.picking import PygfxPickingMixin
    from visualizer.src.renderers.pygfx.sensing import PygfxSensingAdapter

    class Renderer(PygfxPickingMixin):
        pass

    renderer = Renderer()
    target = object()
    renderer._reverse_objects = {id(target): "sensing_detections"}
    selections = []
    renderer.visualizer = SimpleNamespace(
        ui_manager=SimpleNamespace(
            panels={"sensing": SimpleNamespace(select_scene_estimate=selections.append)}
        )
    )
    adapter = PygfxSensingAdapter(renderer)
    frame = _frame()
    frame["sensing"]["detections"] = [{}, {"track_id": 8}, {"track_id": 9}]
    adapter._selection_frame = frame
    adapter._selection_layers = {
        "sensing_detections": SimpleNamespace(metadata={"detection_indices": (0, 2)})
    }
    event = SimpleNamespace(
        target=target, x=10, y=20, button=1, modifiers=[], pick_info={"vertex_index": 1}
    )
    event.type = "pointer_down"
    adapter._on_sensing_pointer(event)
    event.type = "pointer_up"
    adapter._on_sensing_pointer(event)
    assert selections[-1].track_id == 9
    event.type = "pointer_down"
    adapter._on_sensing_pointer(event)
    event.type, event.x = "pointer_move", 80
    adapter._on_sensing_pointer(event)
    event.type, event.x = "pointer_up", 10
    adapter._on_sensing_pointer(event)
    assert len(selections) == 1


def test_spectrogram_click_seeks_physical_time_binding(qtbot):
    from PySide6.QtCore import QPointF, Qt

    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    locations = result_locations(tuple((i, _frame(i)) for i in (0, 4, 8)))
    window.set_result_navigation(locations, previous=False, following=True)
    plot = window.micro_plot
    plot.getViewBox().setRange(xRange=(0, 1), yRange=(-1, 1), padding=0)
    position = plot.getViewBox().mapViewToScene(QPointF(locations[1].time_s, 0))
    event = SimpleNamespace(
        button=lambda: Qt.MouseButton.LeftButton,
        modifiers=lambda: Qt.KeyboardModifier.NoModifier,
        scenePos=lambda: position,
        accept=lambda: None,
    )
    with qtbot.waitSignal(window.result_seek_requested) as signal:
        window._seek_plot_time(plot, event)
    assert signal.args == [locations[1]]


def test_plot_selection_pauses_and_seeks_its_authored_scene():
    from visualizer.src.panels.sensing_panel import SensingPanel

    selection = select_estimate(_frame(19)["sensing"], track_id=7)
    location = result_locations(((900, _frame(19)),))[0]
    actions = []
    state_changes = []
    panel = SimpleNamespace(
        parent=SimpleNamespace(
            app_state=SimpleNamespace(extension_state={}),
            set_state=lambda **changes: state_changes.append(changes),
            schedule_update=lambda: None,
            animation_running=True,
            animation_controller=SimpleNamespace(toggle_animation=lambda: actions.append("pause")),
            renderer=SimpleNamespace(renderer_id="none"),
        ),
        _dashboard_session=SimpleNamespace(navigation=(location,)),
        _current_step=lambda: 22,
        _set_widget_checked=lambda *args: None,
        _sync_scene_selection_controls=lambda: None,
        _seek_result=actions.append,
    )
    SensingPanel._on_linked_selection(panel, selection)
    assert actions == ["pause", location]
    assert state_changes == []  # Selection preserves the user's 3D visibility choice.
    assert panel.parent.force_update_next_frame
    assert panel.parent.sensing_selection == selection


def test_unselected_scene_does_not_rewrite_marker_buffers():
    from visualizer.src.renderers.pygfx.sensing import PygfxSensingAdapter

    class Renderer:
        pass

    renderer = Renderer()
    adapter = PygfxSensingAdapter(renderer)
    # No renderer geometry API exists here: the normal unselected frame path
    # must return before requesting buffers or materials.
    adapter.highlight_selection(_frame(), [object()], None)
    adapter.clear_selection_bindings()
    assert adapter.selected_position(None) is None


def test_picking_is_restored_after_material_update_without_duplicate_handlers():
    from visualizer.src.renderers.pygfx.sensing import PygfxSensingAdapter

    class Renderer:
        pass

    renderer = Renderer()
    handlers = []
    obj = SimpleNamespace(
        material=SimpleNamespace(pick_write=False),
        add_event_handler=lambda *args: handlers.append(args),
    )
    renderer._objects = {"sensing_track_point_7": obj}
    layer = SimpleNamespace(name="sensing_track_point_7", metadata={"track_id": 7})
    adapter = PygfxSensingAdapter(renderer)
    adapter.bind_selection(_frame(), [layer])
    assert obj.material.pick_write
    obj.material.pick_write = False  # A normal renderer material transaction.
    adapter.bind_selection(_frame(1), [layer])
    assert obj.material.pick_write
    assert len(handlers) == 1


def test_pygfx_retirement_removes_labels_picking_and_hud():
    from visualizer.src.renderers.pygfx.sensing import PygfxSensingAdapter

    class Renderer:
        pass

    renderer = Renderer()
    calls = []
    renderer.remove_object = lambda name: calls.append(("object", name))
    renderer._clear_hud_overlay = lambda name: calls.append(("hud", name))
    adapter = PygfxSensingAdapter(renderer)
    adapter._sensing_annotation_names.add("old_track_label")
    adapter._sensing_marker_specs["old"] = (1,)
    adapter._pick_objects["old"] = SimpleNamespace(
        remove_event_handler=lambda *args: calls.append(("pick", args))
    )
    adapter.retire()
    assert ("object", "old_track_label") in calls
    assert ("hud", "sensing_scene_legend") in calls
    assert sum(kind == "pick" for kind, _ in calls) == 1
    assert not adapter._pick_objects and not adapter._selection_layers
    assert not adapter._sensing_annotation_names and not adapter._sensing_marker_specs
