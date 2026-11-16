"""Saved-volume opening, exact cell feedback and source-bound report navigation."""

import json
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from shared.sensing.volume.evaluation import load_evaluation
from shared.sensing.volume.inspection import PairedVolume, load_bookmark, save_bookmark
from shared.sensing.volume.reader import SensingVolume
from tests.shared.test_sensing_volume_inspection import write_fixture
from tests.visualizer.unit.test_volume_evidence_workflow import opened as _opened
from visualizer.src.sensing.volume_sources import (
    condition_sources,
    resolve_volume_source,
    volume_candidates,
)

opened = _opened


def test_discovery_is_shallow_and_preserves_ordinary_scenarios(tmp_path):
    a = write_fixture(tmp_path / "site_a", "site_a")
    b = write_fixture(tmp_path / "site_b", "site_b")
    write_fixture(tmp_path / "trials" / "unrelated")
    assert volume_candidates(tmp_path) == [a.path, b.path]
    assert volume_candidates(a.path) == [a.path]
    assert volume_candidates(a.scene_path) == [a.path]
    assert volume_candidates(a.path.parent) == [a.path]
    assert condition_sources(a) == [a.path, b.path]
    (tmp_path / "scenario.yaml").write_text("{}")
    assert volume_candidates(tmp_path) == []


def test_volume_camera_actions_use_shared_presentation(opened):
    w, _, _ = opened
    calls = []
    presentation = SimpleNamespace(
        prepare=lambda: calls.append("prepare"), free_camera=lambda: calls.append("free")
    )
    w.visualizer.camera_controller = SimpleNamespace(presentation=lambda: presentation)
    w.visualizer.renderer.set_overview_camera = lambda *args: calls.append("frame")
    w._frame_grid()
    assert calls == ["prepare", "free", "frame"]
    w.workflow.select_volume_cell("sensing:outcome_volume", (0, 0, 0), "iso")
    w.inspection_controls.table.selectRow(0)
    for action in ("frame_reports", "side_reports", "frame_focus", "locate_report"):
        calls.clear()
        w.workflow.action(action)
        assert calls == ["prepare", "free", "frame"], action


def test_label_style_shortcut_opens_the_shared_scene_controls(opened):
    from PySide6.QtWidgets import QComboBox, QWidget

    w, _, _ = opened
    calls = []
    controls = QWidget(w)
    controls.mode = QComboBox(controls)
    w.visualizer.ui_manager = SimpleNamespace(
        set_panel_visible=lambda *args: calls.append(args),
        reveal_control=lambda *args: calls.append(args),
        panels={"nodes": SimpleNamespace(widgets={"label_style_controls": controls})},
    )
    w.scene_label_settings.click()
    assert calls == [("nodes", True), ("nodes", controls.mode)]
    assert w.inspection_controls.vectors.isHidden()
    assert w.inspection_controls.through.isHidden()
    assert w.inspection_controls.report_note.isHidden()


def test_fusion_controls_omit_single_radar_measurements(opened, tmp_path, qtbot):
    from tests.shared.test_wall_fusion_volume import fusion_fixture
    from visualizer.src.sensing.volume_window import SensingVolumeWindow

    existing, _, _ = opened
    reader = fusion_fixture(tmp_path / "fused")
    existing.visualizer.current_scenario_path = reader.scene_path
    window = SensingVolumeWindow(existing.visualizer, reader.path)
    window.show()
    try:
        qtbot.waitUntil(lambda: window.evaluation_controls.ready)
        assert window.geometry_section.isHidden()
        assert window.inspection_controls.table.isColumnHidden(4)
        assert window.inspection_controls.table.isColumnHidden(5)
        basis = window.evaluation_controls.basis
        assert not basis.model().item(basis.findData("range_velocity")).isEnabled()
        window.workflow.select_volume_cell("sensing:outcome_volume", (0, 0, 0), "iso")
        assert window.open_native.text() == "Inspect fusion & sensors"
        assert window.contributor_note.text() == "Contributors: West"
        assert not window.contributor_note.isHidden()
        window.quantity.setCurrentIndex(window.quantity.findData("contributions"))
        assert window.workflow.contributions["labels"][:3] == (
            "West only",
            "East only",
            "West + East",
        )
        assert not window.contribution_plot.isHidden()
        assert window.contribution_plot.data is window.workflow.contributions
        assert "Contributors to matched estimate" in window.legend.text()
        context = window.workflow.context()
        from shared.sensing.volume.inspection import validate_context

        assert validate_context(reader, context) == window.selected_key
    finally:
        window.close()


def test_multiple_sources_require_selection(tmp_path, monkeypatch):
    a = write_fixture(tmp_path / "a", "a")
    b = write_fixture(tmp_path / "b", "b")
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getItem", lambda *args: ("", False))
    assert resolve_volume_source(None, tmp_path) is False
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args: ("b", True))
    assert resolve_volume_source(None, tmp_path) == b.path
    assert resolve_volume_source(None, a.path) == a.path


def test_generic_pair_keeps_alignment_and_strict_wall_contract(tmp_path):
    a = write_fixture(tmp_path / "a", "deployment A")
    b = write_fixture(tmp_path / "b", "deployment B")
    with pytest.raises(ValueError, match="mutually declared"):
        PairedVolume(a, b)
    pair = PairedVolume(a, b, strict_experiment=False)
    before = (a.path.read_bytes(), b.path.read_bytes())
    result = pair.evaluate(load_evaluation(a), load_evaluation(b), 0, position_gate_m=1)
    assert result["common"] == 4 and result["counts"]["Matched in both"] == 2
    assert before == (a.path.read_bytes(), b.path.read_bytes())
    data = b.metadata
    data["world_frame"]["id"] = "other"
    b.path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identical world frame"):
        PairedVolume(a, SensingVolume(b.path), strict_experiment=False)


def test_selection_has_exact_bounds_and_clears_without_changing_results(opened):
    w, objects, _ = opened
    snapshot = w.evaluation_snapshot
    assert w.selected_key is None and not any("selected_cell" in name for name in objects)
    w.workflow.select_volume_cell("sensing:outcome_volume", (1, 0, 0), "iso")
    assert w.selected_key == (0, 0, 0, 1)
    box = objects["sensing:volume:selected_cell:outline"]
    np.testing.assert_allclose(box.payload.points.min(0), [11, 18.5, 28])
    np.testing.assert_allclose(box.payload.points.max(0), [13, 21.5, 32])
    assert box.metadata["depth_test"] is False
    w.mode.setCurrentIndex(w.mode.findData("slice"))
    np.testing.assert_allclose(objects[box.id].payload.points[:, 2], 30)
    w.inspect.setChecked(False)
    assert w.selected_key is None and not any("selected_cell" in name for name in objects)
    assert "No cell selected" in w.cell_note.text()
    assert w.evaluation_snapshot is snapshot


def test_inspector_explains_sample_size_and_acquired_empty_results(opened):
    w, _, _ = opened
    w.workflow.select_volume_cell("sensing:outcome_volume", (1, 0, 0), "iso")
    assert "Sampling cell: 2 m × 3 m × 4 m" in w.cell_note.text()
    assert "X 11 m to 13 m" in w.cell_note.text()
    assert "not the matching gate" in w.cell_note.text()
    table = w.inspection_controls.table
    assert table.item(0, 6).text() == "Inside"
    assert table.item(1, 6).text() == "Outside"
    w.quantity.setCurrentIndex(w.quantity.findData("reports"))
    w.workflow.select_volume_cell("sensing:outcome_volume", (3, 0, 0), "iso")
    assert "not defined — no saved detector reports" in w.cell_note.text()
    assert "Raw source value: unavailable" not in w.cell_note.text()


def test_report_markers_preserve_out_of_grid_positions_and_association(opened):
    w, objects, _ = opened
    w.workflow.select_volume_cell("sensing:outcome_volume", (1, 0, 0), "iso")
    c = w.inspection_controls
    c.table.selectRow(1)
    assert w.workflow.selected_report == "1"
    np.testing.assert_allclose(objects["sensing:volume:report:1"].payload.points, [[0, 0, -5]])
    assert c.table.item(1, 1).text() == "Extra · outside gate"
    label = objects["sensing:volume:report_label:1"]
    assert "extra" in label.payload.text
    assert label.metadata["label_callout"]["on_top"] is False
    c.through.setChecked(True)
    assert objects[label.id].metadata["label_callout"]["on_top"] is True
    assert objects[label.id].metadata["alpha_mode"] == "blend"


def test_idle_poll_stops_and_edges_reuse_on_report_selection(opened, qtbot, monkeypatch):
    w, _, _ = opened
    qtbot.waitUntil(lambda: not w.timer.isActive())
    w.cell_edges.setChecked(True)
    edges = w.workflow.edge_objects
    from visualizer.src.sensing import volume_workflow

    monkeypatch.setattr(
        volume_workflow, "cell_edge_objects", lambda *a, **k: pytest.fail("rebuilt unchanged grid")
    )
    w.opacity.setValue(0.7)
    w.workflow.select_volume_cell("sensing:outcome_volume", (1, 0, 0), "iso")
    w.inspection_controls.table.selectRow(1)
    assert w.workflow.edge_objects is edges


def test_pair_bookmark_paths_are_relocatable(opened, tmp_path):
    w, _, _ = opened
    state = w.workflow.context()
    peer = tmp_path / "peer" / "sensing_volume.json"
    state["pair"] = dict(volume=str(peer), volume_sha256="a" * 64, changed_variable="conditions")
    path = tmp_path / "bookmark.json"
    save_bookmark(path, state)
    assert json.loads(path.read_text())["pair"]["volume"] == "peer/sensing_volume.json"
    assert load_bookmark(path)["pair"]["volume"] == str(peer)


@pytest.mark.parametrize("plane,axis", [("xy", "z"), ("xz", "y"), ("yz", "x")])
def test_slice_coordinate_works_without_selecting_a_cell(opened, plane, axis):
    w, objects, _ = opened
    w.mode.setCurrentIndex(w.mode.findData("slice"))
    w.inspection_controls.orientation.setCurrentIndex(
        w.inspection_controls.orientation.findData(plane)
    )
    assert not w.slice_coordinate.isHidden()
    assert w.slice_coordinate.property("axis") == axis
    assert w.inspect_section.isHidden()
    assert w.inspector.isHidden()
    assert w.selected_key is None
    assert not any("selected_cell" in name for name in objects)


def test_clearing_selection_cancels_pending_native_open(opened):
    w, _, _ = opened
    w.workflow.select_volume_cell("sensing:outcome_volume", (1, 0, 0), "iso")
    w.native_pending = Future()
    w.native_pending_key = w._cell_key()
    w.native_pending_generation = w.selection_generation
    w.native_pending.set_result(w.reader.scene_path.parent)
    w.visualizer.open_scenario = lambda *a, **kw: pytest.fail("opened a deselected trial")
    w.inspect.setChecked(False)
    w._poll()
    assert w.native_pending is None and not w.closed


@pytest.mark.parametrize("change", ["grid", "motion"])
def test_switching_condition_does_not_silently_select_a_different_trial(opened, qtbot, change):
    from visualizer.src.sensing.volume_window import SensingVolumeWindow

    w, _, _ = opened
    w.inspect.setChecked(True)
    state = w._inspection_view_state()
    if change == "grid":
        state["x"] = -12345.0
    else:
        state["heading_identity"] = dict(state["heading_identity"], velocity_m_s=[0, 0, 99])
    restored = SensingVolumeWindow(w.visualizer, w.reader.path, view_state=state)
    try:
        qtbot.waitUntil(lambda: restored.evaluation_controls.ready, timeout=20000)
        assert not restored.inspect.isChecked()
        assert restored.selected_key is None
    finally:
        restored.close()


def test_normal_open_recognizes_volume_and_forces_saved_file_source(tmp_path, monkeypatch):
    from tests.visualizer.unit.test_scenario_workflow import (
        _FakeProgressReporter,
        _make_open_scenario_viz,
    )
    from visualizer.src.app import scenario_workflow
    from visualizer.src.renderers import protocol
    from visualizer.src.sensing import volume_window
    from visualizer.src.services.scenario_loader_service import ScenarioLoadResult

    reader = write_fixture(tmp_path / "experiment")
    calls = []
    scenario = SimpleNamespace(
        root=reader.path.parent,
        view_defaults={},
        visualizer_cfg={},
        sensing={},
        raytracing={},
        coverage_cfg={},
    )
    viz = _make_open_scenario_viz(ScenarioLoadResult(scenario, None, object(), False), calls)
    viz.renderer = object()
    monkeypatch.setattr(protocol, "volume_rendering_unavailable_reason", lambda renderer: None)
    attached = []
    monkeypatch.setattr(
        volume_window,
        "open_sensing_volume",
        lambda *args, **kwargs: attached.append((args, kwargs)),
    )
    result = scenario_workflow.open_scenario(
        viz, str(reader.path), _progress_factory=lambda _: _FakeProgressReporter()
    )
    assert result.succeeded
    assert ("preflight_scenario", str(reader.scene_path), {"data_mode_override": "files"}) in calls
    assert attached == [((viz, reader.path), {"scene_is_open": True})]


def test_invalid_volume_keeps_current_scene(tmp_path, monkeypatch):
    from tests.visualizer.unit.test_scenario_workflow import _make_open_scenario_viz
    from visualizer.src.app import scenario_workflow
    from visualizer.src.renderers import protocol

    path = tmp_path / "sensing_volume.json"
    path.write_text("{}")
    calls = []
    viz = _make_open_scenario_viz(None, calls)
    viz.renderer = object()
    viz.current_scenario_path = "active"
    viz.vis_initialized = True
    monkeypatch.setattr(protocol, "volume_rendering_unavailable_reason", lambda renderer: None)
    result = scenario_workflow.open_scenario(viz, str(path))
    assert not result.succeeded
    assert "scene_cleanup" not in calls
    assert viz.current_scenario_path == "active"


def test_controls_have_one_filter_and_common_selection_display(opened):
    w, objects, _ = opened
    assert [w.tabs.tabText(i) for i in range(w.tabs.count())] == ["View", "Analysis"]
    assert w.mode.itemText(w.mode.findData("both")) == "3D volume"
    assert w.inspect_section.isHidden()
    assert not w.inspection_controls.buttons["next"].isEnabled()
    assert w.readable_sensor.isHidden()
    assert not any("volume_sensor" in key for key in objects)
    snapshot = w.evaluation_snapshot
    w.tabs.setCurrentIndex(1)
    assert w.appearance_section.isVisibleTo(w)
    assert w.summary_plots.snapshot is snapshot
    w.workflow.select_volume_cell("sensing:outcome_volume", (1, 0, 0), "iso")
    assert w.tabs.currentIndex() == 1
    assert not w.inspect_section.isHidden()
    assert w.inspection_controls.buttons["next"].isEnabled()
    w.clear_selection.click()
    assert w.selected_key is None and w.inspect_section.isHidden()
    assert w.evaluation_snapshot is snapshot


def test_filter_parameters_and_overlay_scale_only_appear_when_relevant(opened):
    w, _, legends = opened
    c = w.inspection_controls
    assert c.threshold.isHidden() and c.minimum.isHidden()
    c.focus.setCurrentIndex(c.focus.findData("error"))
    assert not c.threshold.isHidden() and c.minimum.isHidden()
    c.focus.setCurrentIndex(c.focus.findData("count"))
    assert c.threshold.isHidden() and not c.minimum.isHidden()
    c.reset_filter()
    assert c.threshold.isHidden() and c.minimum.isHidden()
    for key, scalar in [("extras", True), ("reports", True), ("details", False)]:
        w.quantity.setCurrentIndex(w.quantity.findData(key))
        assert (legends[-1].colorbar_range is not None) is scalar
        assert len(legends[-1].lines) <= 1
    assert not hasattr(w, "scalar_scale")
    assert w.count_scale_row.isHidden()
    assert w.evaluation_controls.error_settings.isHidden()


def test_cutaway_sweep_preserves_selection_and_evaluation(opened):
    w, _, _ = opened
    snapshot = w.evaluation_snapshot
    w.mode.setCurrentIndex(w.mode.findData("both"))
    w.cutaway.axis.setCurrentIndex(w.cutaway.axis.findData("x"))
    w.cutaway.boundary.setCurrentIndex(0)
    w.workflow.step_plane(1)
    assert not w.cutaway.active.isChecked()
    assert w.cutaway.boundary.currentIndex() == 0
    assert w.inspection_controls.sweep_controls.isHidden()
    w.workflow.action("play")
    assert not w.workflow.sweep.isActive()
    w.cutaway.active.setChecked(True)
    w.workflow.step_plane(1)
    assert w.cutaway.boundary.currentIndex() == 1
    assert w.selected_key is None
    assert w.evaluation_snapshot is snapshot
    w.workflow.action("play")
    assert w.workflow.sweep.isActive()
    w.cutaway.active.setChecked(False)
    assert not w.workflow.sweep.isActive()
    w.cutaway.mode.setCurrentIndex(w.cutaway.mode.findData("box"))
    assert w.inspection_controls.sweep_controls.isHidden()


def test_cell_navigation_stays_inside_the_displayed_cut(opened):
    w, _, _ = opened
    w.cutaway.axis.setCurrentIndex(w.cutaway.axis.findData("x"))
    w.cutaway.boundary.setCurrentIndex(1)
    w.cutaway.active.setChecked(True)
    w.workflow.select_volume_cell("sensing:outcome_volume", (1, 0, 0), "iso")
    w.workflow.action("next")
    assert w._cell_key() == (0, 0, 0, 0)
    w.workflow.action("previous")
    assert w._cell_key() == (0, 0, 0, 1)


def test_report_focus_hides_only_volume_and_clearing_cell_restores_it(opened):
    w, objects, legends = opened
    assert w.cell_edges.isChecked()
    snapshot = w.evaluation_snapshot
    w.workflow.select_volume_cell("sensing:outcome_volume", (1, 0, 0), "iso")
    w.inspection_controls.table.selectRow(1)
    selected = w.workflow.selected_report
    w.hide_volume.setChecked(True)
    assert not objects["sensing:outcome_volume"].visible
    assert not any("cell_edges" in key or "selected_cell" in key for key in objects)
    assert objects["sensing:volume:report:1"].visible
    assert w.workflow.selected_report == selected
    assert w.evaluation_snapshot is snapshot and legends[-1] is None
    assert w.workflow.context()["view"]["hide_volume"] is True
    assert w._inspection_view_state()["hide_volume"] is True
    w.clear_selection.click()
    assert objects["sensing:outcome_volume"].visible
    assert any("cell_edges" in key for key in objects)
    assert legends[-1] is not None


def test_source_provenance_is_reachable_and_dialogs_accept_the_real_parent(opened, qtbot):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QFileDialog

    w, _, _ = opened
    w.workflow.select_volume_cell("sensing:outcome_volume", (0, 0, 0), "iso")
    assert not w.inspect_section.isHidden()
    assert w.cell_note.parentWidget().layout().indexOf(w.cell_note) >= 0
    assert "Source index XYZ:" in w.cell_note.text()
    w.visualizer.show()
    w.show()
    disclosure = w.cell_note.parentWidget().parentWidget()
    assert "Source cell details" in disclosure._toggle.text()
    disclosure._toggle.click()
    qtbot.waitUntil(lambda: w.cell_note.isVisible())
    dialogs = []

    def cancel_dialog():
        for dialog in QApplication.topLevelWidgets():
            if isinstance(dialog, QFileDialog):
                dialogs.append(dialog.parentWidget() is w)
                dialog.reject()

    QApplication.setAttribute(
        __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.AA_DontUseNativeDialogs, True
    )
    for action in ("bookmark_save", "bookmark_load", "capture"):
        QTimer.singleShot(0, cancel_dialog)
        w.workflow.action(action)
        assert len(dialogs) == ("bookmark_save", "bookmark_load", "capture").index(action) + 1
        assert dialogs[-1]
