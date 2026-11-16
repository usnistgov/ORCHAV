"""Native controls preserve source definitions and cannot export stale settings."""

import numpy as np

from tests.shared.test_sensing_motion_analysis import fixture_comparison
from visualizer.src.sensing.motion_comparison import MotionComparisonDialog


def test_controls_roundtrip_and_export_invalidation(qtbot, tmp_path, monkeypatch):
    dialog = MotionComparisonDialog()
    qtbot.addWidget(dialog)
    comparison = fixture_comparison()
    dialog.set_definition(comparison["definition"])
    assert dialog.definition()["recipe"] == comparison["definition"]["recipe"]
    assert dialog.definition()["sources"][0]["label"] == "Independent tone"
    dialog.show_comparison(comparison)
    assert dialog.export_button.isEnabled()
    destination = tmp_path / "export"
    monkeypatch.setattr(
        "visualizer.src.sensing.motion_comparison.QFileDialog.getSaveFileName",
        lambda *a: (str(destination), ""),
    )
    dialog.export_button.click()
    assert (destination / "spectra.npz").is_file()
    assert (destination / "motion_comparison.pdf").is_file()
    dialog.range_max.setValue(10)
    assert dialog.comparison is None and not dialog.export_button.isEnabled()
    dialog.close()


def test_worker_and_portable_inspection_use_same_arrays(qtbot, monkeypatch):
    comparison = fixture_comparison()
    monkeypatch.setattr(
        "visualizer.src.sensing.motion_comparison.analyze_comparison", lambda definition: comparison
    )
    dialog = MotionComparisonDialog()
    qtbot.addWidget(dialog)
    dialog.set_definition(comparison["definition"])
    dialog.analyze_button.click()
    qtbot.waitUntil(lambda: dialog.comparison is not None, timeout=10000)
    np.testing.assert_array_equal(
        dialog.comparison["results"][0]["retained_psd"], comparison["results"][0]["retained_psd"]
    )
    assert "complete" in dialog.status.text()
    dialog.show_comparison(comparison, portable=True)
    assert "raw data are needed" in dialog.status.text()
    dialog.close()
