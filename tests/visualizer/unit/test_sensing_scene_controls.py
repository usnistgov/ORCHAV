"""Compact scene controls expose linked selection without opening deep diagnostics."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QRect, QSettings

from tests.visualizer.unit.test_sensing_selection_geometry import spatial_request
from visualizer.src.panels import sensing_panel as module
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.sensing.runtime import state_value
from visualizer.src.sensing.sensing_window import SensingWindow


@pytest.fixture
def scene_panel(qtbot, monkeypatch):
    settings = QSettings()
    keys = (
        "layout/sections/sensing_results",
        "layout/sections/sensing_options",
        "visualizer/sensing_dashboard_geometry",
    )
    previous = {key: settings.value(key) for key in keys}
    for key in keys:
        settings.remove(key)
    panels = []

    def make(*, enabled=False, renderer_id="pygfx", mode="detections"):
        positions = []
        state = SimpleNamespace(
            extension_state={"sensing": {"show_detections": enabled, "scene_mode": mode}}
        )
        parent = SimpleNamespace(
            app_state=state,
            renderer=SimpleNamespace(
                renderer_id=renderer_id,
                focus_camera=positions.append,
                capabilities=RendererCapabilities(),
            ),
            schedule_update=lambda: None,
            extension_services={},
        )

        def update_state(viz, **values):
            viz.app_state.extension_state.setdefault("sensing", {}).update(values)

        monkeypatch.setattr(module, "update_state", update_state)
        panel = module.SensingPanel(parent)
        monkeypatch.setattr(panel, "_start_auto_refresh", lambda: None)
        monkeypatch.setattr(panel, "_refresh_live_plots", lambda *args, **kwargs: None)
        monkeypatch.setattr(panel, "_on_linked_selection", lambda selection: None)
        container = panel.create_panel()
        panel.options_section.expand()
        qtbot.addWidget(container)
        container.resize(1280, 1000)
        container.show()
        qtbot.waitUntil(lambda: panel._sensing_window is not None)
        panels.append((panel, container))
        return panel, container, positions

    yield make
    for panel, container in panels:
        panel.cleanup()
        container.close()
    for key, value in previous.items():
        settings.remove(key) if value is None else settings.setValue(key, value)


@pytest.mark.parametrize("initially_enabled", [False, True])
def test_scene_options_follow_overlay_visibility_without_a_second_toggle(
    scene_panel, initially_enabled
):
    panel, _, _ = scene_panel(enabled=initially_enabled)
    enabled = panel.widgets["show_sensing_detections"]
    options = panel.widgets["scene_options"]
    assert "scene_settings" not in panel.widgets
    if not initially_enabled:
        enabled.click()
    assert options.isVisible()
    enabled.click()
    assert options.isHidden()
    panel._sync_runtime_controls()
    assert options.isHidden()
    enabled.setChecked(False)
    enabled.setChecked(True)
    assert options.isVisible()


def test_disabling_overlays_hides_options_but_preserves_preferences(scene_panel):
    panel, _, _ = scene_panel(enabled=True)
    panel.widgets["sensing_scene_use_marker_glyphs"].setChecked(False)
    panel.widgets["show_sensing_detections"].setChecked(False)
    assert panel.widgets["scene_options"].isHidden()
    assert not panel.widgets["sensing_scene_mode"].isEnabled()
    assert not state_value(panel.parent.app_state, "show_detections")
    panel.widgets["show_sensing_detections"].setChecked(True)
    assert panel.widgets["scene_options"].isVisible()
    assert not panel.widgets["sensing_scene_use_marker_glyphs"].isChecked()


def test_highlight_preference_is_enabled_by_default_and_writes_only_its_state(
    scene_panel, monkeypatch
):
    from visualizer.src.sensing import runtime

    refreshed = []
    monkeypatch.setattr(
        runtime,
        "refresh_scene_selection",
        lambda viz, selection: refreshed.append((viz, selection)),
    )
    panel, _, _ = scene_panel(enabled=True)
    highlight = panel.widgets["sensing_scene_highlight_selection"]
    assert highlight.isChecked()
    highlight.click()
    assert not state_value(panel.parent.app_state, "highlight_selection")
    assert state_value(panel.parent.app_state, "show_detections")
    assert refreshed == [(panel.parent, None)]
    panel.parent.app_state.extension_state["sensing"]["highlight_selection"] = True
    panel._sync_runtime_controls()
    assert highlight.isChecked()
    assert len(refreshed) == 1


def test_main_selection_controls_follow_dashboard_while_floating_and_after_redock(
    scene_panel, qtbot
):
    panel, _, positions = scene_panel()
    panel.widgets["show_sensing_detections"].setChecked(True)
    dashboard = panel._sensing_window
    assert dashboard.scene_selection_controls.isHidden()
    request, _ = spatial_request()
    dashboard._set_view_mode("current")
    dashboard.update_from_request(request)
    dashboard.select_detection_index(0)
    focus = panel.widgets["scene_focus_selection"]
    assert focus.isEnabled()
    focus.click()
    np.testing.assert_allclose(positions, [[10, 25, 30]])
    panel.widgets["scene_range_shell"].setChecked(True)
    assert dashboard.range_shell_enabled.isChecked()
    dashboard.range_shell_cutaway.setChecked(True)
    assert panel.widgets["scene_range_cutaway"].isChecked()
    panel.dashboard_host.detach()
    qtbot.waitUntil(panel.dashboard_host.is_detached)
    assert panel._sensing_window is dashboard
    dashboard.range_shell_enabled.setChecked(False)
    assert not panel.widgets["scene_range_shell"].isChecked()
    dashboard.set_linked_selection(None, notify=True)
    assert not focus.isEnabled()
    panel.dashboard_host.dock()
    assert panel._sensing_window is dashboard
    assert not dashboard.range_shell_enabled.isChecked()
    assert dashboard.range_shell_cutaway.isChecked()
    assert dashboard.scene_selection_controls.isHidden()


def test_range_only_and_unsupported_renderer_keep_capabilities_explicit(scene_panel):
    panel, _, _ = scene_panel(renderer_id="modern")
    dashboard = panel._sensing_window
    request, _ = spatial_request()
    request.display_frame["sensing"]["detections"][0].pop("aoa_az_deg")
    dashboard._set_view_mode("current")
    dashboard.update_from_request(request)
    dashboard.select_detection_index(0)
    assert not panel.widgets["scene_focus_selection"].isEnabled()
    assert not panel.widgets["scene_range_shell"].isEnabled()
    assert not panel.widgets["scene_range_cutaway"].isEnabled()
    assert not panel.widgets["scene_frame_range_shell"].isEnabled()


def test_shell_frame_action_requires_a_selected_bin_and_clear_retires_it(scene_panel):
    panel, _, _ = scene_panel()
    framed = []
    panel.parent.extension_services["experiments"] = SimpleNamespace(
        sync_sensing_selection=lambda _window: None,
        range_shell=SimpleNamespace(frame=lambda: framed.append(True)),
    )
    dashboard = panel._sensing_window
    dashboard._set_view_mode("current")
    dashboard.update_from_request(spatial_request()[0])
    button = panel.widgets["scene_frame_range_shell"]
    assert not button.isEnabled()
    panel.widgets["scene_range_shell"].setChecked(True)
    assert not button.isEnabled()
    dashboard._select_inspection_coordinate(1, 1)
    assert button.isEnabled()
    button.click()
    assert framed == [True]
    dashboard.set_linked_selection(None, notify=True)
    assert not button.isEnabled()


def test_destroying_dashboard_retires_main_selection_capabilities(scene_panel):
    panel, _, _ = scene_panel()
    dashboard = panel._sensing_window
    dashboard._set_view_mode("current")
    dashboard.update_from_request(spatial_request()[0])
    dashboard.select_detection_index(0)
    assert panel.widgets["scene_focus_selection"].isEnabled()
    dashboard.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert panel._sensing_window is None
    assert not panel.widgets["scene_focus_selection"].isEnabled()
    assert not panel.widgets["scene_frame_range_shell"].isEnabled()


def test_standalone_dashboard_retains_scene_actions(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    assert not window.scene_selection_controls.isHidden()
    assert not window.scene_selection_controls.isVisible()
    window._set_view_mode("current")
    window.update_from_request(spatial_request()[0])
    assert window.selection_tools_toggle.isHidden()
    window.select_detection_index(0)
    window.selection_tools_toggle.setChecked(True)
    assert window.scene_selection_controls.isVisible()
    assert window.range_shell_enabled.isVisible()
    assert window.range_shell_cutaway.isVisible()
    assert window.locate_selection_button.isEnabled()
    with qtbot.waitSignal(window.range_shell_changed):
        window.range_shell_enabled.click()
    assert window.range_shell_enabled.isChecked()
    window._set_view_mode("summary")
    assert not window.scene_selection_controls.isVisible()
    window._set_view_mode("current")
    assert window.scene_selection_controls.isVisible()
    window.set_linked_selection(None, notify=True)
    assert not window.scene_selection_controls.isVisible()


def test_saved_world_track_selection_notifies_once_and_refresh_does_not_reselect(qtbot):
    from tests.visualizer.unit.test_sensing_dashboard_recording import _history, _request
    from tests.visualizer.unit.test_sensing_tracking_presentation import SavedReader
    from visualizer.src.sensing.fusion_inspection import FusionSelection

    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._set_view_mode("tracking")
    request = replace(
        _request(4, _history()),
        saved_tracking=SimpleNamespace(reader=SavedReader(), sensor_id="sensor"),
    )
    window.update_from_request(request)
    selections = []
    window.selection_changed.connect(selections.append)
    window.select_track(1)
    assert selections == [FusionSelection("saved-run", 1, 0)]
    window.update_from_request(request)
    assert selections == [FusionSelection("saved-run", 1, 0)]
    assert window.tracking_inspector.selected_id == 1


def test_scene_options_pack_wide_rows_and_wrap_without_clipping(scene_panel, qtbot):
    panel, container, _ = scene_panel(enabled=True, mode="full")
    options = panel.widgets["scene_options"]
    keys = (
        "sensing_scene_mode",
        "sensing_scene_full_history",
        "sensing_scene_show_velocity_arrows",
        "sensing_scene_show_track_labels",
        "sensing_scene_show_truth",
        "sensing_scene_use_marker_glyphs",
        "show_recorded_paths",
        "pin_fresh_sensing",
        "sensing_scene_highlight_selection",
        "scene_range_shell",
        "scene_range_cutaway",
        "scene_focus_selection",
        "scene_frame_range_shell",
    )

    def controls_fit():
        return all(
            options.rect().contains(QRect(widget.mapTo(options, QPoint()), widget.size()))
            for widget in (panel.widgets[key] for key in keys)
            if widget.isVisible()
        )

    qtbot.waitUntil(controls_fit)
    wide_height = options.height()
    assert wide_height < 160  # The previous sparse grid used 226 px at this width.
    container.resize(440, 1000)
    qtbot.waitUntil(lambda: options.width() < 450 and controls_fit())
    assert options.height() > wide_height
    container.resize(1280, 1000)
    qtbot.waitUntil(lambda: options.height() == wide_height and controls_fit())
