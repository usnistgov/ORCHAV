"""Coverage runs preserve sources, bind assets and remain cancellable."""

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import shiboken6
import yaml
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QWidget

from tests.shared.test_coverage_volume import write_volume
from tests.visualizer.unit.test_unified_coverage_panel import loaded
from tests.visualizer.unit.test_unified_coverage_panel import panel_view as panel_view
from visualizer.src.app.scenario_workflow import ScenarioOpenOutcome
from visualizer.src.coverage.generation import (
    coverage_mapping,
    estimate_work,
    parse_heights,
    prepare_run,
)
from visualizer.src.coverage.generation_window import CoverageGenerationWindow

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "scenarios/generator/coverage/single_tx/scenario.yaml"
SETTINGS = dict(
    heights=[10, 15],
    resolution=[5, 5],
    bbox=[[-140, -110], [25, 55]],
    samples=1000,
    depth=1,
    mechanisms={"los": True, "specular_reflection": True},
)


def test_height_sweeps_and_work_preflight():
    assert parse_heights("10:70:5") == list(range(10, 71, 5))
    assert parse_heights("1.5, 10, 30") == [1.5, 10, 30]
    for invalid in ("0:1:0", "0:1:.3", "0:10000:.01", "2,1", "nan", ""):
        with pytest.raises(ValueError):
            parse_heights(invalid)
    original = yaml.safe_load(SOURCE.read_text())
    changed = coverage_mapping(original, **SETTINGS)
    assert changed["timeline"] == {"steps": 1, "duration_s": 0}
    assert original["coverage"]["grid"]["heights_m"] != changed["coverage"]["grid"]["heights_m"]
    assert "72 cells" in estimate_work(changed)
    changed["coverage"]["grid"]["resolution_m"] = [0.001, 0.001]
    with pytest.raises(ValueError, match="Reduce"):
        estimate_work(changed)


def test_prepare_new_independent_run(tmp_path):
    before = SOURCE.read_bytes()
    output = tmp_path / "new-run"
    saved = prepare_run(SOURCE, output, SETTINGS)
    mapping = yaml.safe_load(saved.read_text())
    assert mapping["scene"]["id"] == "etoile"
    assert mapping["coverage"]["solver"]["samples_per_tx"] == 1000
    assert not mapping["raytracing"]["enabled"]
    assert not (output / "frames").exists()
    assert SOURCE.read_bytes() == before
    with pytest.raises(FileExistsError):
        prepare_run(SOURCE, output, SETTINGS)
    assert yaml.safe_load(saved.read_text()) == mapping


def window(qtbot, tmp_path):
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.current_scenario_path = str(SOURCE.parent)
    opened = []

    def open_scenario(path):
        opened.append(path)
        return ScenarioOpenOutcome.success(Path(path), frame_source_ready=True)

    parent.open_scenario = open_scenario
    parent.ui_controller = SimpleNamespace(handle_coverage_toggled=Mock())
    parent.ui_manager = SimpleNamespace(show_panel=Mock())
    dialog = CoverageGenerationWindow(parent)
    qtbot.addWidget(dialog)
    dialog.destination.setText(str(tmp_path / "new-run"))
    return dialog, opened


def test_close_during_preparation_cannot_start_solver(qtbot, tmp_path, monkeypatch):
    dialog, _ = window(qtbot, tmp_path)
    future = Future()
    monkeypatch.setattr(dialog.executor, "submit", lambda *args: future)
    dialog._start()
    dialog.close()
    future.set_result(tmp_path / "new-run/scenario.yaml")
    assert not dialog.poll.isActive()
    assert dialog.job is None


def test_cancel_during_preparation_never_launches(qtbot, tmp_path, monkeypatch):
    dialog, _ = window(qtbot, tmp_path)
    future = Future()
    monkeypatch.setattr(dialog.executor, "submit", lambda *args: future)
    dialog._start()
    dialog._cancel()
    future.set_result(tmp_path / "new-run/scenario.yaml")
    dialog._poll()
    assert dialog.job is None
    assert not dialog.open_result.isEnabled()
    assert "Cancelled" in dialog.status.text()


def test_success_failure_and_explicit_open(qtbot, tmp_path):
    dialog, opened = window(qtbot, tmp_path)
    path = tmp_path / "new-run"
    (path / "coverage").mkdir(parents=True)
    (path / "coverage/coverage_maps.h5").write_bytes(b"verified by generator")
    result = SimpleNamespace(
        succeeded=True, paths=SimpleNamespace(scenario_yaml=path / "scenario.yaml")
    )
    dialog.job = SimpleNamespace(stdout_log=(), stderr_log=(), result=result)
    dialog._poll()
    assert dialog.open_result.isEnabled()
    assert not opened
    dialog._open()
    assert opened == [str(path)]
    dialog.visualizer.ui_controller.handle_coverage_toggled.assert_called_once_with(True)
    dialog.visualizer.ui_manager.show_panel.assert_called_once_with("coverage")


@pytest.mark.parametrize(
    "outcome",
    [ScenarioOpenOutcome.failure("Cannot load result"), ScenarioOpenOutcome.cancelled("Cancelled")],
)
def test_open_failure_keeps_result_dialog_available(qtbot, tmp_path, outcome):
    dialog, _ = window(qtbot, tmp_path)
    dialog.result_path = tmp_path / "new-run"
    dialog.visualizer.open_scenario = Mock(return_value=outcome)
    dialog.show()
    dialog._open()
    assert dialog.isVisible()
    assert outcome.message in dialog.status.text()
    dialog.visualizer.ui_controller.handle_coverage_toggled.assert_not_called()


@pytest.mark.parametrize("mode,axis", [("slice", 1), ("iso", 2), ("iso", 0)])
def test_open_result_replaces_visible_controls_and_preserves_view(
    panel_view, qtbot, tmp_path, mode, axis
):
    viz, panel, controller = panel_view
    panel.volume_view.widget.window().show()
    controller.handle_coverage_toggled(True)
    previous = loaded(qtbot, panel)
    previous.mode.setCurrentIndex(previous.mode.findData(mode))
    previous.orientation.setCurrentIndex(previous.orientation.findData(axis))
    previous.cut_axis.setCurrentIndex(previous.cut_axis.findData(axis))
    previous.cut_enabled.setChecked(True)
    previous.cut_position.setCurrentIndex(0)
    source = dict(viz.coverage_data)
    source.pop("_volume_layer", None)
    path = write_volume(tmp_path / "new_coverage.h5")
    heights = list(range(10, 71, 10))
    with h5py.File(path, "r+") as h:
        gains = np.repeat(h["values/path_gain_linear"][:, :1], len(heights), axis=1)
        del h["values/path_gain_linear"], h["grid/heights_m"]
        h.create_dataset("values/path_gain_linear", data=gains)
        h.create_dataset("grid/heights_m", data=heights)
        h["grid/shape_xyz"][2] = len(heights)
        h["grid/origin"][2] = heights[0]
    source.update(coverage_file=str(path), heights=heights, grid_shape=[5, 4, 7])

    def open_scenario(directory):
        # Use the normal teardown, including the visible old embedded dialog.
        viz.ui_controller.coverage_service.reset_runtime_state(viz)
        viz.coverage_data = source
        viz.coverage_heights = heights
        panel.update_coverage_status(True, source)
        panel.set_heights(heights)
        return ScenarioOpenOutcome.success(Path(directory), frame_source_ready=True)

    viz.current_scenario_path = str(SOURCE.parent)
    viz.open_scenario = open_scenario
    viz.ui_manager.show_panel = Mock()
    dialog = CoverageGenerationWindow(viz)
    qtbot.addWidget(dialog)
    dialog.result_path = tmp_path
    dialog.show()
    dialog._open()
    current = loaded(qtbot, panel)
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not shiboken6.isValid(previous)
    assert panel.volume_view.widget.findChildren(type(current)) == [current]
    assert viz.app_state.show_coverage and current.mode.currentData() == mode
    assert current.orientation.currentData() == axis and current.cut_axis.currentData() == axis
    assert current.model.values.shape == (7, 4, 5)
    assert "5 × 4 × 7" in current.status.text()
    assert not current.cut_enabled.isChecked()
    viz.ui_manager.show_panel.assert_called_once_with("coverage")
    if mode == "iso":
        current.cut_axis.setCurrentIndex(current.cut_axis.findData(2))
        assert [current.cut_position.itemData(i) for i in range(7)] == heights
        current._play()
        assert current.sweep.isActive() and current.cut_position.currentIndex() == 0
        current._step()
        assert current.cut_position.currentData() == 20


def test_close_cancels_running_job(qtbot, tmp_path):
    dialog, _ = window(qtbot, tmp_path)
    calls = []
    dialog.job = SimpleNamespace(cancel=lambda: calls.append("cancel"))
    dialog.close()
    assert calls == ["cancel"]


def test_moving_source_timeline_remains_valid_for_initial_pose_preparation():
    source = yaml.safe_load(SOURCE.read_text())
    source["timeline"] = {"steps": 8, "duration_s": 1.0}
    source["actors"]["tx"][0]["mobility"] = {
        "type": "linear",
        "start_m": [0, 0, 30],
        "end_m": [1, 0, 30],
    }
    new = coverage_mapping(source, **SETTINGS)
    assert new["timeline"] == source["timeline"]
    assert new["actors"] == source["actors"]
    assert not new["raytracing"]["enabled"]


def test_invalid_destination_is_not_reused(qtbot, tmp_path):
    dialog, _ = window(qtbot, tmp_path)
    dialog.destination.setText(str(tmp_path))
    assert not dialog.generate.isEnabled()
    assert "non-existing" in dialog.status.text()
