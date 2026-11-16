"""Native controls share the same saved-report comparison/export routines."""

import json

from shared.sensing.volume.comparison import compare_volumes, load_definition, save_definition
from tests.shared.test_sensing_volume_inspection import write_fixture
from tests.visualizer.unit.test_volume_evidence_workflow import opened as _opened
from visualizer.src.sensing.volume_comparison import VolumeComparisonDialog

opened = _opened


def test_controls_recipe_gate_export_and_inspection(qtbot, tmp_path):
    a, b = write_fixture(tmp_path / "first"), write_fixture(tmp_path / "second")
    dialog = VolumeComparisonDialog(initial_volume=a.path)
    qtbot.addWidget(dialog)
    dialog.add_volume(b.path, label="A different option")
    dialog.sources_table.item(0, 0).setText("Baseline")
    dialog.gate.setValue(1)
    dialog.analyze()
    qtbot.waitUntil(lambda: dialog.pending is None, timeout=20000)
    assert dialog.result == compare_volumes(dialog.definition())
    assert dialog.summary.item(0, 2).text() == "2"
    dialog.export_to(tmp_path / "export")
    assert json.loads((tmp_path / "export/comparison.json").read_text()) == dialog.result
    before = dialog.result["conditions"][0]["report_snapshot_sha256"]
    dialog.gate.setValue(100)
    assert not dialog.export_button.isEnabled() and dialog.result is None
    dialog.analyze()
    qtbot.waitUntil(lambda: dialog.pending is None, timeout=20000)
    assert dialog.summary.item(0, 2).text() == "3"
    assert dialog.result["conditions"][0]["report_snapshot_sha256"] == before
    path = tmp_path / "saved.json"
    save_definition(path, dialog.definition())
    dialog.set_definition(load_definition(path))
    assert len(dialog.sources) == 2 and dialog.gate.value() == 100
    received = []
    dialog.inspect_requested.connect(received.append)
    dialog.sources_table.selectRow(1)
    dialog.inspect_button.click()
    assert received == [str(b.path)]
    dialog.remove_button.click()
    assert len(dialog.sources) == 1
    dialog.close()


def test_invalid_metadata_keeps_export_disabled(qtbot, tmp_path):
    dialog = VolumeComparisonDialog()
    qtbot.addWidget(dialog)
    dialog.analyze()
    qtbot.waitUntil(lambda: dialog.pending is None, timeout=20000)
    assert "unavailable" in dialog.status.text() and not dialog.export_button.isEnabled()
    dialog.close()


def test_private_extension_absent_keeps_single_volume_operation(opened, qtbot, monkeypatch):
    from visualizer.src.sensing.volume_window import SensingVolumeWindow

    original, _, _ = opened
    monkeypatch.setattr("visualizer.src.sensing.volume_controls.find_spec", lambda _: None)
    second = SensingVolumeWindow(original.visualizer, original.reader.path)
    qtbot.addWidget(second)
    second.show()
    qtbot.waitUntil(lambda: second.evaluation_controls.ready)
    assert second.compare_saved.isHidden()
    assert second.evaluation_snapshot["tp"] == original.evaluation_snapshot["tp"]
    second.close()
