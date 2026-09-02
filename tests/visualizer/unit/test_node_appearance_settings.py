"""Focused contracts for node-appearance controls and workspace persistence."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from PySide6.QtWidgets import QDoubleSpinBox

from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.panels.nodes_panel import NodesSelectionPanel
from visualizer.src.scene.defaults import (
    DEFAULT_LABEL_OFFSET_M,
    LABEL_FONT_SIZE_BOUNDS,
    LABEL_OFFSET_BOUNDS_M,
    NODE_MARKER_SIZE_BOUNDS_M,
    ORIENTATION_SCALE_BOUNDS_M,
    TRAJECTORY_LINE_WIDTH_BOUNDS_PX,
    TRAJECTORY_POINT_SIZE_BOUNDS_PX,
)
from visualizer.src.services.session_service import SessionService
from visualizer.src.state import create_initial_state


def _nodes_panel_parent(**overrides):
    values = {
        "target_entries": [],
        "app_state": create_initial_state(),
        "label_offset_x": DEFAULT_LABEL_OFFSET_M[0],
        "label_offset_y": DEFAULT_LABEL_OFFSET_M[1],
        "label_offset_z": DEFAULT_LABEL_OFFSET_M[2],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _session_service(viz) -> SessionService:
    service = SessionService.__new__(SessionService)
    service.viz = viz
    return service


def test_node_appearance_controls_use_direct_units_and_broad_safety_bounds(qapp) -> None:
    """Appearance spinboxes accept values beyond the former convenience ranges."""
    panel = NodesSelectionPanel(_nodes_panel_parent())
    group = panel.create_panel()

    expected_bounds = {
        "tx_marker_size_spin": NODE_MARKER_SIZE_BOUNDS_M,
        "rx_marker_size_spin": NODE_MARKER_SIZE_BOUNDS_M,
        "label_font_size_spin": LABEL_FONT_SIZE_BOUNDS,
        "x_offset_spinbox": LABEL_OFFSET_BOUNDS_M,
        "y_offset_spinbox": LABEL_OFFSET_BOUNDS_M,
        "z_offset_spinbox": LABEL_OFFSET_BOUNDS_M,
        "orientation_scale_spin": ORIENTATION_SCALE_BOUNDS_M,
        "trajectory_line_width_spin": TRAJECTORY_LINE_WIDTH_BOUNDS_PX,
        "trajectory_point_size_spin": TRAJECTORY_POINT_SIZE_BOUNDS_PX,
    }
    for key, (minimum, maximum) in expected_bounds.items():
        widget = panel.widgets[key]
        assert isinstance(widget, QDoubleSpinBox)
        assert widget.minimum() == minimum
        assert widget.maximum() == maximum

    assert panel.widgets["x_offset_spinbox"].value() == DEFAULT_LABEL_OFFSET_M[0]
    assert panel.widgets["x_offset_spinbox"].suffix() == " m"
    panel.widgets["tx_marker_size_spin"].setValue(250.0)
    panel.widgets["x_offset_spinbox"].setValue(-125.5)
    panel.widgets["trajectory_line_width_spin"].setValue(40.5)
    assert panel.widgets["tx_marker_size_spin"].value() == 250.0
    assert panel.widgets["x_offset_spinbox"].value() == -125.5
    assert panel.widgets["trajectory_line_width_spin"].value() == 40.5

    group.deleteLater()


def test_label_offset_handler_reads_decimal_meters_without_legacy_scaling() -> None:
    """The controller forwards direct meter values instead of dividing by ten."""
    node_service = SimpleNamespace(apply_label_offsets=Mock())
    viz = SimpleNamespace(
        x_offset_spinbox=SimpleNamespace(value=lambda: 1.25),
        y_offset_spinbox=SimpleNamespace(value=lambda: -2.5),
        z_offset_spinbox=SimpleNamespace(value=lambda: 0.375),
        node_service=node_service,
    )
    controller = UIController.__new__(UIController)
    controller.visualizer = viz

    controller.handle_label_offset_changed()

    assert (viz.label_offset_x, viz.label_offset_y, viz.label_offset_z) == (
        1.25,
        -2.5,
        0.375,
    )
    node_service.apply_label_offsets.assert_called_once_with()


def test_node_appearance_workspace_state_round_trips_values_beyond_old_limits(qapp) -> None:
    """Workspace extraction and restore preserve the exact displayed values."""

    class Renderer:
        def __init__(self) -> None:
            self.trajectory_line_width = 3.0
            self.trajectory_point_size = 6.0

        def set_trajectory_line_width(self, value: float) -> bool:
            self.trajectory_line_width = float(value)
            return True

        def set_trajectory_point_size(self, value: float) -> bool:
            self.trajectory_point_size = float(value)
            return True

    renderer = Renderer()
    viz = _nodes_panel_parent(
        renderer=renderer,
        tx_marker_size=0.3,
        rx_marker_size=0.3,
        label_font_size=0.3,
        orientation_scale=3.0,
        current_building_alpha=1.0,
        current_target_alpha=1.0,
        vis_initialized=True,
    )
    panel = NodesSelectionPanel(viz)
    group = panel.create_panel()
    viz.ui_manager = SimpleNamespace(panels={"nodes": panel})

    saved_values = {
        "tx_marker_size_m": 250.0,
        "rx_marker_size_m": 125.0,
        "label_font_size": 8.75,
        "label_offset_x_m": -125.5,
        "label_offset_y_m": 42.25,
        "label_offset_z_m": 500.0,
        "orientation_scale_m": 350.0,
        "trajectory_line_width_px": 40.5,
        "trajectory_point_size_px": 80.0,
    }
    for key, value in saved_values.items():
        panel.widgets[
            {
                "tx_marker_size_m": "tx_marker_size_spin",
                "rx_marker_size_m": "rx_marker_size_spin",
                "label_font_size": "label_font_size_spin",
                "label_offset_x_m": "x_offset_spinbox",
                "label_offset_y_m": "y_offset_spinbox",
                "label_offset_z_m": "z_offset_spinbox",
                "orientation_scale_m": "orientation_scale_spin",
                "trajectory_line_width_px": "trajectory_line_width_spin",
                "trajectory_point_size_px": "trajectory_point_size_spin",
            }[key]
        ].setValue(value)

    service = _session_service(viz)
    snapshot = service._get_node_appearance_state()
    assert snapshot == saved_values

    node_service = SimpleNamespace(
        update_tx_marker_sizes=Mock(),
        update_rx_marker_sizes=Mock(),
        recreate_tx_rx_labels=Mock(),
        recreate_target_labels=Mock(),
        apply_label_offsets=Mock(),
    )
    viz.node_service = node_service
    viz.ui_controller = SimpleNamespace(handle_orientation_scale_changed=Mock())

    service._restore_node_appearance_state(snapshot)
    service._sync_nodes_panel(viz.ui_manager.panels, viz.app_state)

    assert viz.tx_marker_size == 250.0
    assert viz.rx_marker_size == 125.0
    assert viz.label_font_size == 8.75
    assert (viz.label_offset_x, viz.label_offset_y, viz.label_offset_z) == (
        -125.5,
        42.25,
        500.0,
    )
    assert renderer.trajectory_line_width == 40.5
    assert renderer.trajectory_point_size == 80.0
    assert panel.widgets["x_offset_spinbox"].value() == -125.5
    assert panel.widgets["trajectory_line_width_spin"].value() == 40.5
    node_service.update_tx_marker_sizes.assert_called_once_with()
    node_service.update_rx_marker_sizes.assert_called_once_with()
    node_service.recreate_tx_rx_labels.assert_called_once_with(8.75)
    node_service.recreate_target_labels.assert_called_once_with(8.75)
    node_service.apply_label_offsets.assert_called_once_with()
    viz.ui_controller.handle_orientation_scale_changed.assert_called_once_with(350.0)

    group.deleteLater()


def test_workspace_validation_accepts_missing_appearance_and_rejects_unsafe_values() -> None:
    """The optional block is backward compatible and invalid values fail preflight."""
    payload = {
        "app_state": {},
        "animation": {},
        "rendering": {},
        "entry_state": {},
    }
    assert SessionService._restore_payload_validation_error(payload) is None

    payload["rendering"]["node_appearance"] = {"label_offset_x_m": float("inf")}
    error = SessionService._restore_payload_validation_error(payload)
    assert error is not None
    assert "label_offset_x_m" in error
