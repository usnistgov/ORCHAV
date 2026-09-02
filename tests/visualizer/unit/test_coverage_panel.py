"""Headless Qt regression tests for the compact Coverage panel."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication, QGroupBox, QLabel

from visualizer.src.panels.coverage_panel import CoverageMapPanel

_APP = QApplication.instance() or QApplication([])
_REAL_QLABEL = QLabel


@pytest.fixture(autouse=True)
def restore_real_qt_labels(monkeypatch):
    """Bind the real QLabel for Coverage panel tests."""
    monkeypatch.setattr(QtWidgets, "QLabel", _REAL_QLABEL)
    module = sys.modules[CoverageMapPanel.__module__]
    monkeypatch.setattr(module, "QLabel", _REAL_QLABEL)


class _RecordingController:
    def __init__(self):
        self.metrics = []
        self.stop_calls = 0

    def handle_coverage_metric_changed(self, metric):
        self.metrics.append(metric)

    def handle_coverage_height_animation_stop(self):
        self.stop_calls += 1


def _build_panel(controller=None):
    parent = SimpleNamespace(ui_controller=controller) if controller else SimpleNamespace()
    panel = CoverageMapPanel(parent)
    group = panel.create_panel()
    return panel, group


def _scalar_metadata(metric="best_path_loss_db"):
    return {
        "metric_name": metric,
        "available_metrics": [metric],
        "grid_shape": np.array([2, 3, 2], dtype=np.int32),
        "heights": np.array([1.0, 2.0], dtype=np.float32),
        "value_min": 80.0,
        "value_max": 130.0,
        "values_3d": np.array(
            [
                [[80.0, 90.0, 100.0], [110.0, 120.0, 130.0]],
                [[82.0, 92.0, 102.0], [112.0, 122.0, 128.0]],
            ],
            dtype=np.float32,
        ),
    }


def test_panel_defaults_are_grouped_accessible_and_fully_opaque():
    panel, group = _build_panel()

    subgroup_titles = {child.title() for child in group.findChildren(QGroupBox)}
    assert {"Layer", "Analysis", "Height playback"} <= subgroup_titles
    assert panel.widgets["coverage_opacity"].value() == 100
    assert panel.widgets["opacity_label"].text() == "100%"
    assert panel.widgets["coverage_height_speed_label"].text() == "0.67 s"
    for key in (
        "coverage_metric_combo",
        "coverage_serving_tx_toggle",
        "coverage_height_combo",
        "coverage_interpolation",
        "coverage_opacity",
        "coverage_threshold_value",
        "coverage_isoline_count",
    ):
        assert panel.widgets[key].accessibleName()

    group.deleteLater()


def test_status_accepts_numpy_metadata_and_uses_human_labels_and_units():
    panel, group = _build_panel()
    metadata = _scalar_metadata()

    panel.update_coverage_status(True, metadata)

    status = panel.widgets["coverage_status"].text()
    assert "Best path loss" in status
    assert "2×3×2 cells" in status
    assert "80.0–130.0 dB" in status
    assert "Heights: 1.00, 2.00 m" in status
    assert panel.widgets["coverage_height_combo"].count() == 2
    assert "RdYlGn_r" in panel.widgets["coverage_legend_gradient"].toolTip()
    assert panel.widgets["coverage_missing_label"].text() == ("Missing cells: hidden (transparent)")
    assert panel.widgets["coverage_slice_summary"].text() == (
        "Valid: 6/6 cells (100.0%); no data: 0.0% | " "P10/P50/P90: 85.0/105.0/125.0 dB"
    )

    group.deleteLater()


def test_metric_families_switch_between_serving_and_selected_transmitter():
    recorder = _RecordingController()
    panel, group = _build_panel(recorder)
    metrics = [
        "path_gain_linear/West",
        "path_gain_linear/East",
        "serving_path_gain_linear",
        "path_gain_db/West",
        "path_loss_db/West",
        "path_loss_db/East",
        "best_path_loss_db",
        "rss_dbm/West",
        "rss_dbm/East",
        "best_rss_dbm",
        "sinr_db",
        "sinr_db/West",
        "sinr_db/East",
        "serving_tx",
    ]
    panel.set_metrics(metrics, "best_path_loss_db")
    metric_combo = panel.widgets["coverage_metric_combo"]
    tx_combo = panel.widgets["coverage_tx_combo"]
    serving_toggle = panel.widgets["coverage_serving_tx_toggle"]

    assert [metric_combo.itemData(index) for index in range(metric_combo.count())] == [
        "path_gain_linear",
        "path_gain_db",
        "path_loss_db",
        "rss_dbm",
        "sinr_db",
        "serving_tx",
    ]
    assert not serving_toggle.isHidden()
    assert serving_toggle.isChecked()
    assert tx_combo.isHidden()

    serving_toggle.setChecked(False)
    assert not tx_combo.isHidden()
    assert recorder.metrics[-1] == "path_loss_db/West"

    tx_combo.setCurrentIndex(1)
    assert recorder.metrics[-1] == "path_loss_db/East"

    metric_combo.setCurrentIndex(metric_combo.findData("rss_dbm"))
    assert not serving_toggle.isChecked()
    assert not tx_combo.isHidden()
    assert tx_combo.currentText() == "East"
    assert recorder.metrics[-1] == "rss_dbm/East"

    metric_combo.setCurrentIndex(metric_combo.findData("sinr_db"))
    assert not serving_toggle.isChecked()
    assert tx_combo.currentText() == "East"
    assert recorder.metrics[-1] == "sinr_db/East"

    metric_combo.setCurrentIndex(metric_combo.findData("path_gain_linear"))
    assert not serving_toggle.isHidden()
    assert not serving_toggle.isChecked()
    assert not tx_combo.isHidden()
    assert [tx_combo.itemText(index) for index in range(tx_combo.count())] == [
        "West",
        "East",
    ]
    assert tx_combo.currentText() == "East"
    assert recorder.metrics[-1] == "path_gain_linear/East"

    metric_combo.setCurrentIndex(metric_combo.findData("path_gain_db"))
    assert serving_toggle.isHidden()
    assert not tx_combo.isHidden()
    assert tx_combo.count() == 1
    assert recorder.metrics[-1] == "path_gain_db/West"

    metric_combo.setCurrentIndex(metric_combo.findData("path_loss_db"))
    assert not serving_toggle.isChecked()
    assert tx_combo.currentText() == "East"
    assert recorder.metrics[-1] == "path_loss_db/East"

    serving_toggle.setChecked(True)
    assert tx_combo.isHidden()
    assert recorder.metrics[-1] == "best_path_loss_db"

    metric_combo.setCurrentIndex(metric_combo.findData("path_gain_linear"))
    assert recorder.metrics[-1] == "serving_path_gain_linear"

    metric_combo.setCurrentIndex(metric_combo.findData("rss_dbm"))
    assert serving_toggle.isChecked()
    assert tx_combo.isHidden()
    assert recorder.metrics[-1] == "best_rss_dbm"

    metric_combo.setCurrentIndex(metric_combo.findData("serving_tx"))
    assert serving_toggle.isHidden()
    assert tx_combo.isHidden()
    assert recorder.metrics[-1] == "serving_tx"

    metric_combo.setCurrentIndex(metric_combo.findData("sinr_db"))
    assert serving_toggle.isChecked()
    assert recorder.metrics[-1] == "sinr_db"

    panel.reset_view_state()
    panel.set_metrics(metrics, "path_loss_db/East")
    assert metric_combo.currentData() == "path_loss_db"
    assert not serving_toggle.isChecked()
    assert not tx_combo.isHidden()
    assert tx_combo.currentData() == "path_loss_db/East"

    group.deleteLater()


def test_single_tx_metric_hides_selector_and_labels_logarithmic_color_scale():
    recorder = _RecordingController()
    panel, group = _build_panel(recorder)
    metadata = {
        "metric_name": "path_gain_linear/BaseStation",
        "available_metrics": [
            "path_gain_linear/BaseStation",
            "sinr_db/BaseStation",
        ],
        "grid_shape": np.array([2, 1, 1], dtype=np.int32),
        "heights": np.array([1.5], dtype=np.float32),
        "value_min": 1.0e-12,
        "value_max": 1.0e-6,
        "values_3d": np.array([[[1.0e-12, 1.0e-6]]], dtype=np.float32),
    }

    panel.update_coverage_status(True, metadata)

    metric_combo = panel.widgets["coverage_metric_combo"]
    tx_combo = panel.widgets["coverage_tx_combo"]
    serving_toggle = panel.widgets["coverage_serving_tx_toggle"]
    assert tx_combo.count() == 1
    assert tx_combo.isHidden()
    assert serving_toggle.isHidden()
    assert panel.widgets["coverage_legend_title"].text() == "Scale (logarithmic):"
    assert "(logarithmic)" in panel.widgets["coverage_legend_gradient"].toolTip()

    metric_combo.setCurrentIndex(metric_combo.findData("sinr_db"))
    metric_combo.setCurrentIndex(metric_combo.findData("path_gain_linear"))
    assert recorder.metrics[-1] == "path_gain_linear/BaseStation"

    group.deleteLater()


def test_multi_tx_count_keeps_serving_mode_when_one_selected_variant_is_advertised():
    recorder = _RecordingController()
    panel, group = _build_panel(recorder)

    panel.set_metrics(
        ["sinr_db", "sinr_db/West"],
        "sinr_db",
        tx_count=2,
    )

    serving_toggle = panel.widgets["coverage_serving_tx_toggle"]
    tx_combo = panel.widgets["coverage_tx_combo"]
    assert not serving_toggle.isHidden()
    assert serving_toggle.isChecked()
    serving_toggle.setChecked(False)
    assert not tx_combo.isHidden()
    assert tx_combo.currentData() == "sinr_db/West"
    assert recorder.metrics[-1] == "sinr_db/West"

    group.deleteLater()


def test_categorical_metric_forces_raw_and_shows_transmitter_color_legend():
    panel, group = _build_panel()
    metadata = {
        "metric_name": "serving_tx",
        "available_metrics": ["serving_tx"],
        "grid_shape": np.array([2, 2, 1], dtype=np.int32),
        "heights": np.array([1.5], dtype=np.float32),
        "value_min": 0.0,
        "value_max": 1.0,
        "tx_names": np.array(["West", "East"]),
        "serving_tx_count": 2,
        "values_3d": np.array([[[0, 1], [1, -1]]], dtype=np.float32),
    }

    panel.update_coverage_status(True, metadata)

    smoothing = panel.widgets["coverage_interpolation"]
    assert smoothing.currentText() == "Raw"
    assert not smoothing.isEnabled()
    assert not panel.widgets["coverage_threshold_toggle"].isEnabled()
    assert panel.widgets["coverage_tx_combo"].isHidden()
    legend = panel.widgets["coverage_legend_gradient"].text()
    assert "West" in legend
    assert "East" in legend
    assert "#1f77b4" in legend
    assert "#d62728" in legend
    assert panel.widgets["coverage_missing_label"].text().endswith("(transparent)")
    assert panel.widgets["coverage_slice_summary"].text() == (
        "No service: 1/4 cells (25.0%, 1.0 m^2) | "
        "West: 1/4 cells (25.0%, 1.0 m^2) | "
        "East: 2/4 cells (50.0%, 2.0 m^2)"
    )

    group.deleteLater()


def test_categorical_legend_keeps_overflow_marker_visible_at_narrow_width():
    panel, group = _build_panel()
    tx_names = [f"TX {index + 1}" for index in range(12)]
    metadata = {
        "metric_name": "serving_tx",
        "available_metrics": ["serving_tx"],
        "grid_shape": np.array([2, 2, 1], dtype=np.int32),
        "heights": np.array([1.5], dtype=np.float32),
        "value_min": 0.0,
        "value_max": 11.0,
        "tx_names": tx_names,
        "serving_tx_count": len(tx_names),
        "values_3d": np.zeros((1, 2, 2), dtype=np.float32),
    }

    panel.update_coverage_status(True, metadata)

    legend = panel.widgets["coverage_legend_gradient"]
    assert "+4 more" in legend.text()
    assert legend.maximumHeight() >= 80

    group.deleteLater()


@pytest.mark.parametrize("metric", ["path_gain_linear/TX1", "rss_w/TX1"])
def test_tiny_linear_thresholds_remain_editable_in_scientific_notation(metric):
    panel, group = _build_panel()
    metadata = {
        "metric_name": metric,
        "available_metrics": [metric],
        "grid_shape": np.array([1, 3, 1]),
        "heights": np.array([1.0]),
        "value_min": 1.0e-21,
        "value_max": 1.0e-20,
        "values_3d": np.array([[[1.0e-21, 5.0e-21, 1.0e-20]]]),
    }

    panel.update_coverage_status(True, metadata)
    spin = panel.widgets["coverage_threshold_value"]
    spin.setValue(5.0e-21)

    assert np.isclose(spin.value(), 5.0e-21, rtol=1.0e-12, atol=1.0e-30)
    assert 0.0 < spin.singleStep() < 1.0e-20
    assert "e" in spin.cleanText().lower()
    assert float(spin.cleanText()) == pytest.approx(5.0e-21)

    group.deleteLater()


def test_reset_view_state_restores_deterministic_defaults():
    panel, group = _build_panel()
    panel.update_coverage_status(True, _scalar_metadata())
    panel.widgets["coverage_toggle"].setChecked(True)
    panel.widgets["coverage_opacity"].setValue(45)
    panel.widgets["coverage_interpolation"].setCurrentText("Smooth+")
    panel.widgets["coverage_threshold_toggle"].setChecked(True)
    panel.widgets["coverage_isolines_toggle"].setChecked(True)
    panel.widgets["coverage_isoline_count"].setValue(10)
    panel.widgets["coverage_height_speed"].setValue(8)

    panel.reset_view_state()

    assert panel.widgets["coverage_toggle"].isChecked() is False
    assert panel.widgets["coverage_opacity"].value() == 100
    assert panel.widgets["coverage_interpolation"].currentText() == "Raw"
    assert panel.widgets["coverage_metric_combo"].count() == 0
    assert panel.widgets["coverage_height_combo"].count() == 0
    assert panel.widgets["coverage_threshold_toggle"].isChecked() is False
    assert panel.widgets["coverage_threshold_mask_toggle"].isChecked() is False
    assert panel.widgets["coverage_isolines_toggle"].isChecked() is False
    assert panel.widgets["coverage_isoline_count"].value() == 6
    assert panel.widgets["coverage_height_speed"].value() == 3
    assert panel.widgets["coverage_height_stop_btn"].isEnabled() is False
    assert panel.widgets["coverage_slice_summary"].text() == "Unavailable"

    group.deleteLater()


def test_selected_height_updates_only_the_active_slice_summary():
    panel, group = _build_panel()
    metadata = {
        "metric_name": "sinr_db",
        "available_metrics": ["sinr_db"],
        "grid_shape": np.array([2, 1, 2], dtype=np.int32),
        "heights": np.array([1.0, 2.0], dtype=np.float32),
        "value_min": 0.0,
        "value_max": 40.0,
        "values_3d": np.array(
            [
                [[0.0, 10.0]],
                [[30.0, 40.0]],
            ],
            dtype=np.float32,
        ),
    }

    panel.update_coverage_status(True, metadata)
    assert "P10/P50/P90: 1.0/5.0/9.0 dB" in panel.widgets["coverage_slice_summary"].text()

    panel.set_height_index(1)

    assert "P10/P50/P90: 31.0/35.0/39.0 dB" in panel.widgets["coverage_slice_summary"].text()
    group.deleteLater()
