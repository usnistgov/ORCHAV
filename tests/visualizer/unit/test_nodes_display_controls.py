"""Scenario-aware node controls and independent world/pixel workspace sizes."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.panels.label_style_controls import LabelStyleControls
from visualizer.src.panels.nodes_panel import NodesSelectionPanel
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.session_service import SessionService
from visualizer.src.state import AppState, create_initial_state
from visualizer.src.types.label_style import LabelCalloutStyle


@pytest.fixture
def nodes(qtbot):
    viz = SimpleNamespace(
        app_state=create_initial_state(),
        target_entries=[],
        scenario_config=None,
        renderer=SimpleNamespace(capabilities=RendererCapabilities(label_callouts=True)),
        tx_marker_size=0.3,
        rx_marker_size=0.4,
        node_service=Mock(),
        schedule_update=Mock(),
        vis_initialized=True,
    )
    viz.set_state = lambda **values: setattr(viz, "app_state", replace(viz.app_state, **values))
    controller = UIController.__new__(UIController)
    controller.visualizer = viz
    controller._tx_marker_size_timer = Mock()
    controller._rx_marker_size_timer = Mock()
    viz.ui_controller = controller
    panel = NodesSelectionPanel(viz)
    group = panel.create_panel()
    qtbot.addWidget(group)
    viz.ui_manager = SimpleNamespace(panels={"nodes": panel})
    for kind in ("tx", "rx"):
        panel.widgets[f"{kind}_marker_size_spin"].valueChanged.connect(
            getattr(controller, f"handle_{kind}_marker_size_changed")
        )
    return viz, panel, group


@pytest.mark.parametrize("sensing", [None, {}, {"enabled": False}, {"enabled": True}])
def test_sensing_text_choice_follows_scenario_configuration(nodes, sensing):
    viz, panel, _ = nodes
    viz.scenario_config = SimpleNamespace(sensing=sensing)
    viz.set_state(node_label_mode="name")
    panel.refresh_live_preview_state()
    combo = panel.widgets["node_label_mode_combo"]
    assert (combo.findData("sensing") >= 0) == bool(sensing and sensing.get("enabled"))
    assert combo.currentData() == "name"


def test_sensing_choice_is_removed_on_scenario_change_and_unload(nodes):
    viz, panel, _ = nodes
    combo = panel.widgets["node_label_mode_combo"]
    for replacement in (SimpleNamespace(sensing={"enabled": False}), None):
        viz.scenario_config = SimpleNamespace(sensing={"enabled": True})
        viz.set_state(node_label_mode="sensing")
        panel.refresh_live_preview_state()
        assert combo.currentData() == "sensing"
        viz.scenario_config = replacement
        panel.refresh_live_preview_state()
        assert combo.findData("sensing") == -1
        assert combo.currentData() == viz.app_state.node_label_mode == "role"


def test_sensing_volume_keeps_roles_when_live_processing_is_disabled(nodes):
    viz, panel, _ = nodes
    viz.scenario_config = SimpleNamespace(sensing={"enabled": False})
    viz.set_state(node_label_mode="sensing")
    viz._opening_sensing_volume_scene = True
    panel.refresh_display_controls()
    combo = panel.widgets["node_label_mode_combo"]
    assert combo.currentData() == "sensing"
    viz._opening_sensing_volume_scene = False
    viz.sensing_volume_window = SimpleNamespace(reader=object(), closed=False)
    panel.refresh_display_controls()
    assert combo.currentData() == "sensing"
    viz.sensing_volume_window.closed = True
    viz.fusion_window = SimpleNamespace(
        reader=None, volume_context={"selected_report": "track:1"}, closed=False
    )
    panel.refresh_display_controls()
    assert combo.currentData() == "sensing"
    viz.fusion_window.closed = True
    panel.refresh_display_controls()
    assert combo.findData("sensing") == -1
    assert viz.app_state.node_label_mode == "role"


def test_labels_lead_and_axis_size_tracks_available_actor_axes(nodes):
    viz, panel, group = nodes
    assert group.layout().itemAt(0).widget().title() == "Labels"
    size_row = panel.widgets["orientation_scale_spin"].parentWidget()
    assert size_row.isHidden()
    for kind in ("tx", "rx"):
        panel.widgets[f"{kind}_orient_cb"].setChecked(True)
        assert not size_row.isHidden()
        panel.widgets[f"{kind}_orient_cb"].setChecked(False)
        assert size_row.isHidden()
    panel.widgets["target_orient_cb"].setChecked(True)
    assert size_row.isHidden()  # A stale target toggle does not expose irrelevant controls.
    viz.target_entries = [{"name": "Car"}]
    panel.update_target_controls()
    assert not size_row.isHidden()
    viz.target_entries = []
    panel.update_target_controls()
    assert size_row.isHidden()


@pytest.mark.parametrize("mode", ["card", "callout"])
def test_one_marker_control_switches_units_and_preserves_both_sizes(nodes, mode):
    viz, panel, _ = nodes
    controls = panel.widgets["label_style_controls"]
    assert "marker_size" not in controls.spins
    tx, rx = (panel.widgets[f"{kind}_marker_size_spin"] for kind in ("tx", "rx"))
    tx.setValue(2.5)
    rx.setValue(1.25)
    # Switch before the geometry debounce fires: edited metre values must survive.
    controls.mode.setCurrentIndex(controls.mode.findData(mode))
    assert tx.suffix() == rx.suffix() == " px"
    tx.setValue(34)
    rx.setValue(22)
    assert viz.app_state.label_callout.node_marker_size("tx") == 34
    assert viz.app_state.label_callout.node_marker_size("rx") == 22
    viz.ui_controller._apply_tx_marker_size()
    viz.ui_controller._apply_rx_marker_size()
    assert (viz.tx_marker_size, viz.rx_marker_size) == (2.5, 1.25)
    controls.mode.setCurrentIndex(controls.mode.findData("plain"))
    assert tx.suffix() == rx.suffix() == " m"
    assert (tx.value(), rx.value()) == (2.5, 1.25)
    controls.mode.setCurrentIndex(controls.mode.findData(mode))
    assert (tx.value(), rx.value()) == (34, 22)


def test_unsupported_callouts_keep_world_marker_controls(nodes):
    viz, panel, _ = nodes
    viz.renderer.capabilities = RendererCapabilities()
    viz.set_state(label_callout=LabelCalloutStyle(mode="card", tx_marker_size=42))
    panel.refresh_display_controls()
    spin = panel.widgets["tx_marker_size_spin"]
    assert spin.suffix() == " m"
    spin.setValue(2)
    assert viz.tx_marker_size == 2
    assert viz.app_state.label_callout.tx_marker_size == 42


def test_label_style_shares_font_and_shows_only_relevant_offsets(nodes, qtbot):
    viz, panel, group = nodes
    controls = panel.widgets["label_style_controls"]
    world_section = panel.widgets["label_world_offset_section"]
    world_section.expand()
    controls.details_container.expand()
    font = panel.widgets["label_font_size_spin"]
    font.valueChanged.connect(viz.ui_controller.handle_label_font_size_changed)
    offsets = [panel.widgets[f"{axis}_offset_spinbox"] for axis in "xyz"]
    font.setValue(0.52)
    expected_font = 0.52
    for spin, value in zip(offsets, (0.25, -0.75, 1.5)):
        spin.setValue(value)
    group.show()
    qtbot.waitUntil(group.isVisible)

    for mode in ("plain", "callout", "card", "plain", "card", "callout", "plain"):
        controls.mode.setCurrentIndex(controls.mode.findData(mode))
        assert world_section.isVisible() == (mode == "plain")
        assert font.isVisible()
        assert all(spin.isVisible() == (mode == "plain") for spin in offsets)
        assert controls.details_container.isVisible() == (mode != "plain")
        assert font.value() == viz.label_font_size == expected_font
        assert [spin.value() for spin in offsets] == [0.25, -0.75, 1.5]
        if mode == "card":
            font.setValue(0.64)
            expected_font = 0.64
            viz.node_service.recreate_tx_rx_labels.assert_called_with(0.64)
            viz.node_service.recreate_target_labels.assert_called_with(0.64)
        if mode != "plain":
            controls.spins["offset_x"].setValue(95)
            controls.spins["offset_y"].setValue(-30)
    assert viz.app_state.label_callout.offset_x == 95
    assert viz.app_state.label_callout.offset_y == -30
    assert world_section.is_expanded()
    assert controls.details_container.is_expanded()


@pytest.mark.parametrize("mode", ["plain", "callout", "card"])
@pytest.mark.parametrize("supported", [False, True])
def test_restored_label_style_updates_relevant_controls_silently(nodes, mode, supported):
    viz, panel, _ = nodes
    controls = panel.widgets["label_style_controls"]
    world_section = panel.widgets["label_world_offset_section"]
    changed = Mock()
    controls.styleChanged.connect(changed)
    viz.renderer.capabilities = RendererCapabilities(label_callouts=supported)
    style = LabelCalloutStyle(mode=mode, offset_x=95, offset_y=-30)
    viz.app_state = AppState.from_dict(create_initial_state(label_callout=style).to_dict())
    viz.label_font_size = 0.52
    viz.label_offset_x, viz.label_offset_y, viz.label_offset_z = (0.25, -0.75, 1.5)
    service = SessionService.__new__(SessionService)
    service.viz = viz
    service._sync_nodes_panel(viz.ui_manager.panels, viz.app_state)

    active = supported and mode != "plain"
    assert world_section.isHidden() == active
    assert controls.details_container.isHidden() != active
    assert controls.mode.currentData() == mode
    assert controls.mode.isEnabled() == supported
    assert controls.spins["offset_x"].value() == 95
    assert controls.spins["offset_y"].value() == -30
    assert panel.widgets["label_font_size_spin"].value() == 0.52
    assert [panel.widgets[f"{axis}_offset_spinbox"].value() for axis in "xyz"] == [
        0.25,
        -0.75,
        1.5,
    ]
    changed.assert_not_called()


def test_workspace_roundtrip_keeps_pixels_out_of_metre_fields(nodes):
    viz, panel, _ = nodes
    viz.ui_controller.handle_label_callout_changed(
        LabelCalloutStyle(mode="card", marker_size=26, tx_marker_size=34, rx_marker_size=22)
    )
    service = SessionService.__new__(SessionService)
    service.viz = viz
    appearance = service._get_node_appearance_state()
    assert appearance["tx_marker_size_m"] == 0.3
    assert appearance["rx_marker_size_m"] == 0.4
    saved_state = viz.app_state.to_dict()
    viz.app_state = AppState.from_dict(saved_state)
    viz.vis_initialized = False
    service._restore_node_appearance_state(appearance)
    service._sync_nodes_panel(viz.ui_manager.panels, viz.app_state)
    assert panel.widgets["tx_marker_size_spin"].value() == 34
    assert panel.widgets["rx_marker_size_spin"].value() == 22
    assert (viz.tx_marker_size, viz.rx_marker_size) == (0.3, 0.4)


def test_legacy_shared_size_and_other_windows_preserve_node_overrides(nodes, qtbot):
    viz, panel, _ = nodes
    legacy_style = LabelCalloutStyle.from_dict({"mode": "card", "marker_size": 26})
    viz.ui_controller.handle_label_callout_changed(legacy_style)
    assert panel.widgets["tx_marker_size_spin"].value() == 26
    assert panel.widgets["rx_marker_size_spin"].value() == 26
    panel.widgets["tx_marker_size_spin"].setValue(34)
    other = LabelStyleControls(viz)
    qtbot.addWidget(other)
    other.spins["opacity"].setValue(0.5)
    assert viz.app_state.label_callout.tx_marker_size == 34
    assert viz.app_state.label_callout.marker_size == 26
    assert panel.widgets["tx_marker_size_spin"].value() == 34
