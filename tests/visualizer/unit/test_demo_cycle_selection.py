"""User-selected demo cycles retain order, restoration and offline sky boundaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import Qt

from tests.visualizer.unit.test_demo_service import Appearance
from tests.visualizer.unit.test_demo_service import rig as rig
from tests.visualizer.unit.test_demo_showcase import showcase as showcase
from visualizer.src.panels.demo_panel import DemoPanel
from visualizer.src.services import demo_service


def installed_skies(monkeypatch):
    catalog = {
        key: SimpleNamespace(id=key, label=key.title()) for key in ("sunny", "missing", "cloudy")
    }
    monkeypatch.setattr(demo_service, "environment_catalog", lambda: catalog)
    monkeypatch.setattr(
        demo_service,
        "cached_environment",
        lambda asset: None if asset.id == "missing" else "cached.hdr",
    )
    return catalog


def select(widget, values):
    for i in range(widget.count()):
        item = widget.item(i)
        item.setCheckState(
            Qt.CheckState.Checked
            if item.data(Qt.ItemDataRole.UserRole) in values
            else Qt.CheckState.Unchecked
        )


def test_selected_appearances_keep_order_and_restore_custom_look(rig):
    viz, service, _tools = rig
    appearance = viz.appearance_preset_service = Appearance(viz.renderer)
    original = appearance.capture()
    assert service.start(
        orbit=False,
        playback=False,
        cycle_appearance=True,
        appearance_presets=("Glow", "Paper", "Glow"),
    )
    for expected in ("Glow", "Paper", "Custom", "Glow"):
        service.tick(service._next_look)
        assert appearance.active_name == expected
    service.stop()
    assert appearance.capture() == original
    assert not any(name in ("Detailed", "Plan", "Dark") for _, name in appearance.calls)


@pytest.mark.parametrize("selection", [(), ("Unknown",)])
def test_invalid_appearance_selection_changes_nothing(rig, selection):
    viz, service, tools = rig
    appearance = viz.appearance_preset_service = Appearance(viz.renderer)
    assert not service.start(cycle_appearance=True, appearance_presets=selection)
    assert not appearance.calls and not viz.animation_running and tools.autorotate is None
    assert not service._timer.isActive()


def test_texture_locked_explicit_selection_cannot_become_empty(rig, monkeypatch):
    viz, service, _tools = rig
    viz.appearance_preset_service = Appearance(viz.renderer)
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    assert not service.start(cycle_appearance=True, appearance_presets=("Detailed",))
    assert "Select at least one" in service.status


def test_selected_sky_skips_unselected_installed_skies_and_restores(rig, monkeypatch):
    viz, service, _tools = rig
    installed_skies(monkeypatch)
    original_lights = viz.renderer.get_light_rig_state()
    assert service.start(
        orbit=False, playback=False, cycle_ibl=True, ibl_environments=("cloudy", "cloudy")
    )
    service.tick(service._next_look)
    assert viz.renderer.sky == "cloudy" and viz.renderer.skybox
    assert viz.renderer.get_shadow_quality() == "scene"
    service.tick(service._next_look)
    assert viz.renderer.sky == "neutral_outdoor" and not viz.renderer.skybox
    assert viz.renderer.get_light_rig_state() == original_lights
    assert viz.renderer.get_shadow_quality() == "focus"
    service.tick(service._next_look)
    assert viz.renderer.get_shadow_quality() == "scene"
    service.stop()
    assert viz.renderer.get_shadow_quality() == "focus"
    assert "sunny" not in viz.renderer.environment_calls


@pytest.mark.parametrize("selection", [(), ("missing",), ("unknown",), ("neutral_outdoor",)])
def test_empty_or_unavailable_sky_selection_changes_nothing(rig, monkeypatch, selection):
    viz, service, tools = rig
    installed_skies(monkeypatch)
    assert not service.start(cycle_ibl=True, ibl_environments=selection)
    assert not viz.renderer.environment_calls and not viz.animation_running
    assert tools.autorotate is None and not service._timer.isActive()


def test_showcase_visits_only_selected_presets_then_plays_recording(showcase):
    viz, service, _tools, _experiments, _calls = showcase
    assert service.start(showcase=True, orbit=False, appearance_presets=("Paper", "Glow"))
    for expected in ("Paper", "Glow", "Custom"):
        service.tick(service._next_look)
        assert viz.appearance_preset_service.active_name == expected
        assert not viz.animation_running
    service.tick(service._next_look)
    assert viz.animation_running and service._showcase_phase == "playback"
    assert [name for action, name in viz.appearance_preset_service.calls if action == "apply"] == [
        "Paper",
        "Glow",
    ]


def test_panel_forwards_selections_and_keeps_them_when_skies_refresh(showcase, qtbot, monkeypatch):
    viz, _service, _tools, _experiments, _calls = showcase
    catalog = installed_skies(monkeypatch)
    panel = DemoPanel(viz)
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    try:
        appearances = panel.widgets["demo_appearance_choices"]
        skies = panel.widgets["demo_ibl_choices"]
        select(appearances, ("Paper", "Glow"))
        select(skies, ("cloudy",))
        panel.widgets["demo_ibl"].setChecked(True)
        assert skies.count() == 2 and skies.isEnabled()
        panel.service.start = Mock()
        panel.start()
        options = panel.service.start.call_args.kwargs
        assert options["appearance_presets"] == ("Paper", "Glow")
        assert options["ibl_environments"] == ("cloudy",)
        panel.widgets["demo_showcase"].setChecked(True)
        assert appearances.isEnabled() and not appearances.isHidden()
        assert not skies.isEnabled()
        panel.widgets["demo_showcase"].setChecked(False)
        catalog["night"] = SimpleNamespace(id="night", label="Night")
        panel.refresh()
        assert panel._selected_choices("demo_ibl_choices") == ("cloudy", "night")
        viz.renderer.sky = "cloudy"
        panel.refresh()
        assert panel._selected_choices("demo_ibl_choices") == ("night",)
        viz.renderer.sky = "neutral_outdoor"
        panel.refresh()
        assert panel._selected_choices("demo_ibl_choices") == ("cloudy", "night")
        assert panel._selected_choices("demo_appearance_choices") == ("Paper", "Glow")
    finally:
        panel.cleanup()
