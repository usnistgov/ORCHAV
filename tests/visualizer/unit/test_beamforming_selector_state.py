"""Tests for Antennas Pattern Source selector synchronization."""

from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtCore import QObject

from visualizer.src.controllers.beamforming_ui_controller import BeamformingUIController
from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.state import create_initial_state, update_state


class _FakeNodeDisplay:
    def __init__(self) -> None:
        self.text_value = ""
        self.enabled = False

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def setText(self, text: str) -> None:
        self.text_value = str(text)


class _FakeLabel:
    def __init__(self) -> None:
        self.text_value = ""

    def setText(self, text: str) -> None:
        self.text_value = str(text)


class _FakeRadio(QObject):
    def __init__(self, checked: bool = False) -> None:
        super().__init__()
        self.checked = checked
        self.enabled = True
        self.tooltip = ""

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def setToolTip(self, tooltip: str) -> None:
        self.tooltip = str(tooltip)

    def setChecked(self, checked: bool) -> None:
        self.checked = bool(checked)

    def isChecked(self) -> bool:
        return self.checked


class _FakeSpin(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.value = None
        self.enabled = False
        self.read_only = True

    def setValue(self, value) -> None:
        self.value = value

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def setReadOnly(self, read_only: bool) -> None:
        self.read_only = bool(read_only)


class _ValueSpin(QObject):
    def __init__(self, value: int) -> None:
        super().__init__()
        self._value = int(value)

    def value(self) -> int:
        return self._value

    def setValue(self, value: int) -> None:
        self._value = int(value)


def _make_viz(**state_updates):
    viz = SimpleNamespace()
    viz.app_state = create_initial_state(**state_updates)
    viz.beam_tx_selector = _FakeNodeDisplay()
    viz.beam_rx_selector = _FakeNodeDisplay()
    viz.beam_status_label = _FakeLabel()
    viz.beam_gain_label = _FakeLabel()
    viz.standalone_mode_frame = _FakeRadio()
    viz.standalone_mode_standalone = _FakeRadio(checked=True)
    viz._latest_beamforming_info = None
    viz._latest_beamforming_pairs = None
    viz._beamforming_tx_nodes = []
    viz._beamforming_rx_nodes = []
    viz._frame_beamforming_available = False
    viz.available_tx = [0]
    viz.available_rx = [0, 1]

    def set_state(**updates):
        viz.app_state = update_state(viz.app_state, **updates)

    viz.set_state = set_state
    return viz


def _make_controller(viz):
    return BeamformingUIController(viz)


def test_pattern_source_global_all_does_not_fallback_to_previous_pair():
    viz = _make_viz(beamforming_tx_node="tx_1", beamforming_rx_node="rx_2")

    _make_controller(viz).apply_selector_state()

    assert viz.app_state.beamforming_tx_node == "auto"
    assert viz.app_state.beamforming_rx_node == "auto"
    assert viz.beam_tx_selector.text_value == "All TX"
    assert viz.beam_rx_selector.text_value == "All RX"
    assert viz.beam_tx_selector.enabled is True
    assert viz.beam_rx_selector.enabled is True
    assert viz.beam_status_label.text_value == "Hidden"


def test_pattern_source_requires_both_sides_to_be_concrete():
    viz = _make_viz(selected_tx=0, beamforming_tx_node="auto", beamforming_rx_node="auto")

    _make_controller(viz).apply_selector_state()

    assert viz.app_state.beamforming_tx_node == "tx_1"
    assert viz.app_state.beamforming_rx_node == "auto"
    assert viz.beam_tx_selector.text_value == "TX1"
    assert viz.beam_rx_selector.text_value == "All RX"
    assert viz.beam_status_label.text_value == "Hidden"


def test_pattern_source_prefers_nodes_selection_over_stale_beam_state():
    viz = _make_viz(
        selected_tx=0,
        selected_rx=1,
        show_beamforming=True,
        beamforming_tx_node="tx_1",
        beamforming_rx_node="rx_1",
    )
    viz._latest_beamforming_info = {
        "available_tx_nodes": ["tx_1"],
        "available_rx_nodes": ["rx_1"],
        "resolved_tx_node": "tx_1",
        "resolved_rx_node": "rx_1",
        "pairs": [{"tx_index": 0, "rx_index": 0, "tx_name": "tx_1", "rx_name": "rx_1"}],
        "status": "Beam patterns: tx_1 -> rx_1",
    }

    _make_controller(viz).apply_selector_state()

    assert viz.app_state.beamforming_rx_node == "rx_2"
    assert viz.beam_rx_selector.text_value == "RX2"
    assert "RX2" in viz.beam_status_label.text_value
    assert "tx_1 -> rx_1" not in viz.beam_status_label.text_value


def test_pattern_source_reports_matching_missing_pair_result():
    """A completed exact-pair miss should replace the transient waiting status."""
    viz = _make_viz(
        selected_tx=0,
        selected_rx=1,
        show_beamforming=True,
        beamforming_tx_node="tx_1",
        beamforming_rx_node="rx_2",
    )
    viz._latest_beamforming_info = {
        "available_tx_nodes": ["tx_1"],
        "available_rx_nodes": ["rx_1"],
        "resolved_tx_node": "tx_1",
        "resolved_rx_node": "rx_2",
        "requested_tx_index": 0,
        "requested_rx_index": 1,
        "pairs": [{"tx_index": 0, "rx_index": 0}],
        "status": "No beamforming data for TX1 -> RX2",
    }

    _make_controller(viz).apply_selector_state()

    assert viz.beam_status_label.text_value == "Unavailable: TX1 -> RX2"
    assert viz.beam_gain_label.text_value == "Metrics: \u2014"


def test_pattern_source_formats_linear_gain_and_pattern_metrics():
    viz = _make_viz(
        selected_tx=0,
        selected_rx=0,
        show_beamforming=True,
        beamforming_tx_node="tx_1",
        beamforming_rx_node="rx_1",
    )
    viz._latest_beamforming_info = {
        "available_tx_nodes": ["tx_1"],
        "available_rx_nodes": ["rx_1"],
        "resolved_tx_node": "tx_1",
        "resolved_rx_node": "rx_1",
        "pairs": [{"tx_index": 0, "rx_index": 0, "tx_name": "tx_1", "rx_name": "rx_1"}],
        "status": "Beam patterns: tx_1 -> rx_1",
        "gain_by_role": {"tx": 32.0, "rx": 1.0},
        "metrics_by_role": {
            "tx": {"hpbw_az_deg": 12.4, "hpbw_el_deg": 24.6, "sll_db": -13.2},
            "rx": {
                "peak_gain_dbi": 5.5,
                "hpbw_az_deg": 90.0,
                "hpbw_el_deg": 45.0,
            },
        },
    }

    _make_controller(viz).apply_selector_state()

    text = viz.beam_gain_label.text_value
    assert viz.beam_status_label.text_value == "Ready: TX1 -> RX1"
    assert "TX Gain 15.1 dB" in text
    assert "HPBW 12/25 deg" in text
    assert "SLL -13.2 dB" in text
    assert "RX Gain 0.0 dB" in text


def test_visible_pair_clears_stale_metrics_while_computing():
    viz = _make_viz(selected_tx=0, selected_rx=0, show_beamforming=True)
    viz._latest_beamforming_info = {
        "resolved_tx_node": "tx_1",
        "resolved_rx_node": "rx_1",
        "requested_tx_index": 0,
        "requested_rx_index": 0,
        "gain_by_role": {"tx": 10.0},
    }
    controller = _make_controller(viz)
    controller.apply_selector_state()
    assert "TX Gain" in viz.beam_gain_label.text_value

    controller.begin_computation()

    assert viz.beam_status_label.text_value == "Computing: TX1 -> RX1..."
    assert viz.beam_gain_label.text_value == "Metrics: \u2014"


def test_completed_update_without_beam_result_clears_computing_state():
    viz = _make_viz(selected_tx=0, selected_rx=0, show_beamforming=True)
    viz._latest_beamforming_pairs = [{"tx_index": 0, "rx_index": 0}]
    controller = _make_controller(viz)
    controller.begin_computation()

    controller.update_node_options(None, None)

    assert viz._beamforming_computing is False
    assert viz._latest_beamforming_pairs == []
    assert viz.beam_status_label.text_value == "Unavailable: no beam pattern result for TX1 -> RX1"
    assert viz.beam_gain_label.text_value == "Metrics: \u2014"


def test_terminal_pipeline_failure_clears_computing_state():
    viz = _make_viz(selected_tx=0, selected_rx=0, show_beamforming=True)
    controller = _make_controller(viz)
    controller.begin_computation()

    controller.fail_computation("Renderer rejected the frame update")

    assert viz._beamforming_computing is False
    assert (
        viz.beam_status_label.text_value == "Error: TX1 -> RX1. Renderer rejected the frame update"
    )
    assert viz.beam_gain_label.text_value == "Metrics: \u2014"


def test_clear_result_metadata_resets_transient_error_state():
    viz = _make_viz(selected_tx=0, selected_rx=0, show_beamforming=True)
    viz._latest_beamforming_info = {"status": "stale"}
    viz._latest_beamforming_pairs = [{"tx_index": 0, "rx_index": 0}]
    viz._beamforming_tx_nodes = ["tx_1"]
    viz._beamforming_rx_nodes = ["rx_1"]
    viz._beamforming_computing = True
    viz._beamforming_completed_without_result = True
    viz._beamforming_error_message = "stale failure"

    _make_controller(viz).clear_result_metadata()

    assert viz._latest_beamforming_info is None
    assert viz._latest_beamforming_pairs == []
    assert viz._beamforming_tx_nodes == []
    assert viz._beamforming_rx_nodes == []
    assert viz._beamforming_computing is False
    assert viz._beamforming_completed_without_result is False
    assert viz._beamforming_error_message is None


def test_partial_pair_result_is_not_reported_as_ready():
    viz = _make_viz(selected_tx=0, selected_rx=0, show_beamforming=True)
    viz._latest_beamforming_info = {
        "requested_tx_index": 0,
        "requested_rx_index": 0,
        "pairs": [{"tx_index": 0, "rx_index": 0}],
        "status": "Beam pattern partial: missing RX surface",
        "gain_by_role": {"tx": 4.0},
    }

    _make_controller(viz).apply_selector_state()

    assert viz.beam_status_label.text_value == "Partial: TX1 -> RX1. missing RX surface"
    assert "TX Gain" in viz.beam_gain_label.text_value


def test_frame_array_sampling_limit_is_visible_in_status():
    viz = _make_viz(
        selected_tx=0,
        selected_rx=0,
        show_beamforming=True,
        beamforming_azimuth_samples=180,
        beamforming_elevation_samples=91,
    )
    viz._latest_beamforming_info = {
        "requested_tx_index": 0,
        "requested_rx_index": 0,
        "pairs": [{"tx_index": 0, "rx_index": 0}],
        "status": "Beam patterns: tx_1 -> rx_1",
        "sampling_by_role": {"tx": {"azimuth": 180, "elevation": 10}},
    }

    _make_controller(viz).apply_selector_state()

    assert viz.beam_status_label.text_value == (
        "Ready: TX1 -> RX1. Sampling limited for memory safety"
    )


def test_preview_work_budget_clamps_combined_array_and_sampling_cost():
    viz = _make_viz(
        standalone_antenna_rows=32,
        standalone_antenna_cols=32,
        beamforming_azimuth_samples=180,
        beamforming_elevation_samples=91,
    )
    viz.standalone_rows = _ValueSpin(32)
    viz.standalone_cols = _ValueSpin(32)
    viz.beam_azimuth_spin = _ValueSpin(180)
    viz.beam_elevation_spin = _ValueSpin(91)
    viz.beam_complexity_note = _FakeLabel()
    controller = UIController.__new__(UIController)
    controller.visualizer = viz

    assert controller._apply_beam_preview_work_budget("elevation") is True

    work_items = (
        viz.app_state.standalone_antenna_rows
        * viz.app_state.standalone_antenna_cols
        * viz.app_state.beamforming_azimuth_samples
        * viz.app_state.beamforming_elevation_samples
    )
    assert work_items <= 8_000_000
    assert viz.beam_elevation_spin.value() < 91
    assert "memory budget" in viz.beam_complexity_note.text_value


def test_enabling_beams_applies_work_budget_before_first_visible_build():
    viz = _make_viz(
        standalone_antenna_rows=32,
        standalone_antenna_cols=32,
        beamforming_azimuth_samples=180,
        beamforming_elevation_samples=91,
    )
    viz.standalone_rows = _ValueSpin(32)
    viz.standalone_cols = _ValueSpin(32)
    viz.beam_azimuth_spin = _ValueSpin(180)
    viz.beam_elevation_spin = _ValueSpin(91)
    viz.beam_complexity_note = _FakeLabel()
    controller = UIController.__new__(UIController)
    controller.visualizer = viz
    calls = []
    controller._refresh_beam_colorbar = lambda: calls.append("colorbar")
    controller._invalidate_beam_patterns = lambda **kwargs: calls.append(("invalidate", kwargs))

    controller.handle_beamforming_toggled(True)

    work_items = (
        viz.app_state.standalone_antenna_rows
        * viz.app_state.standalone_antenna_cols
        * viz.app_state.beamforming_azimuth_samples
        * viz.app_state.beamforming_elevation_samples
    )
    assert viz.app_state.show_beamforming is True
    assert work_items <= 8_000_000
    assert calls == ["colorbar", ("invalidate", {"force_refresh": True})]


def test_hidden_beam_setting_invalidates_cache_without_scheduling_redraw():
    viz = _make_viz(show_beamforming=False)
    selector_calls = []
    viz.beamforming_ui_controller = SimpleNamespace(
        apply_selector_state=lambda: selector_calls.append("hidden"),
        begin_computation=lambda: selector_calls.append("computing"),
    )
    redraws = []
    viz.schedule_update = lambda: redraws.append(True)
    controller = UIController.__new__(UIController)
    controller.visualizer = viz
    invalidations = []
    controller._invalidate_cache = lambda scope, *, reason: invalidations.append((scope, reason))

    controller._invalidate_beam_patterns()

    assert invalidations
    assert selector_calls == ["hidden"]
    assert redraws == []


def test_resolution_controls_sync_from_app_state_and_refresh_selectors():
    viz = _make_viz(
        selected_tx=0,
        selected_rx=0,
        beamforming_azimuth_samples=36,
        beamforming_elevation_samples=19,
        beamforming_tx_scale=2.5,
        beamforming_rx_scale=1.75,
    )
    viz.beam_azimuth_spin = _FakeSpin()
    viz.beam_elevation_spin = _FakeSpin()
    viz.beam_tx_scale_spin = _FakeSpin()
    viz.beam_rx_scale_spin = _FakeSpin()

    _make_controller(viz).update_resolution_controls()

    assert viz.beam_azimuth_spin.value == 36
    assert viz.beam_elevation_spin.value == 19
    assert viz.beam_tx_scale_spin.value == 2.5
    assert viz.beam_rx_scale_spin.value == 1.75
    assert viz.beam_azimuth_spin.enabled is True
    assert viz.beam_azimuth_spin.read_only is False
    assert viz.beam_tx_selector.text_value == "TX1"


def test_frame_data_mode_disabled_without_frame_beamforming_metadata():
    viz = _make_viz(standalone_beamforming_mode="frame")
    viz.standalone_mode_frame.checked = True
    viz.standalone_mode_standalone.checked = False

    _make_controller(viz).set_frame_beamforming_available(False)

    assert viz._frame_beamforming_available is False
    assert viz.standalone_mode_frame.enabled is False
    assert "Unavailable" in viz.standalone_mode_frame.tooltip
    assert viz.standalone_mode_frame.checked is False
    assert viz.standalone_mode_standalone.checked is True
    assert viz.app_state.standalone_beamforming_mode == "standalone"


def test_frame_data_mode_enabled_when_frame_beamforming_metadata_exists():
    viz = _make_viz()

    _make_controller(viz).set_frame_beamforming_available(True)

    assert viz._frame_beamforming_available is True
    assert viz.standalone_mode_frame.enabled is True
    assert "Unavailable" not in viz.standalone_mode_frame.tooltip
