"""Presentation recipes expose only usable controls and keep user choices stable."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QObject, Signal

from tests.visualizer.unit.test_demo_service import Appearance
from tests.visualizer.unit.test_demo_service import rig as rig
from visualizer.src.panels.demo_panel import DemoPanel


class DemoStub(QObject):
    changed = Signal()

    def __init__(self, _):
        super().__init__()
        self.active = self.camera_paused = False
        self.status = "Ready"
        self.capabilities = dict.fromkeys(
            (
                "ready",
                "orbit",
                "tour",
                "targets",
                "playback",
                "showcase",
                "sensing_story",
                "sensing_presentation",
                "appearance",
                "ibl",
                "lighting",
                "dusk",
                "exhibition",
                "clean_view",
                "captions",
            ),
            True,
        )
        self.start = Mock()
        self.stop = Mock()
        self.resume_camera = Mock()

    def available_options(self):
        return dict(self.capabilities)

    def available_environments(self):
        return (("cloudy", "Cloudy"),) if self.capabilities["ibl"] else ()

    def close(self):
        pass


@pytest.fixture
def panel(monkeypatch, qtbot):
    monkeypatch.setattr("visualizer.src.panels.demo_panel.DemoService", DemoStub)
    panel = DemoPanel(SimpleNamespace(ready=True))
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    yield panel
    panel.cleanup()


def select(panel, name):
    combo = panel.widgets["demo_preset"]
    combo.setCurrentIndex(combo.findData(name))


def test_presentation_default_uses_authored_tour_and_clean_captioned_view(panel):
    assert panel.widgets["demo_preset"].currentData() == "presentation"
    assert not panel.widgets["demo_advanced"].is_expanded()
    panel.start()
    options = panel.service.start.call_args.kwargs
    assert options["tour"] and not options["orbit"]
    assert options["cycle_appearance"] and options["playback"]
    assert options["clean_view"] and options["captions"]
    assert not options["showcase"] and not options["sensing_story"]


@pytest.mark.parametrize("explicit_explore", [False, True])
def test_panel_created_before_scene_load_chooses_presentation_unless_user_selected_explore(
    monkeypatch, qtbot, explicit_explore
):
    class LoadingDemo(DemoStub):
        def __init__(self, parent):
            super().__init__(parent)
            self.capabilities = dict.fromkeys(self.capabilities, False)

    monkeypatch.setattr("visualizer.src.panels.demo_panel.DemoService", LoadingDemo)
    panel = DemoPanel(SimpleNamespace(ready=False))
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    try:
        assert panel.widgets["demo_preset"].currentData() == "explore"
        assert not panel.widgets["demo_start"].isEnabled()
        if explicit_explore:
            combo = panel.widgets["demo_preset"]
            combo.activated.emit(combo.currentIndex())
        panel.service.capabilities.update(dict.fromkeys(panel.service.capabilities, True))
        panel.refresh()
        expected = "explore" if explicit_explore else "presentation"
        assert panel.widgets["demo_preset"].currentData() == expected
        assert panel.widgets["demo_start"].isEnabled()
        assert panel.widgets["demo_appearance"].isChecked() is (not explicit_explore)
    finally:
        panel.cleanup()


def test_real_service_capabilities_select_presentation_after_initial_scene_load(rig, qtbot):
    viz, _service, _tools = rig
    viz.ready = False
    panel = DemoPanel(viz)
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    widget.show()
    try:
        assert panel.widgets["demo_preset"].currentData() == "explore"
        assert panel.widgets["demo_orbit"].isHidden()
        viz.appearance_preset_service = Appearance(viz.renderer)
        viz.get_available_animation_steps = lambda: [0, 1, 2]
        viz.ready = True
        qtbot.waitUntil(lambda: panel.widgets["demo_preset"].currentData() == "presentation")
        assert panel.widgets["demo_appearance"].isChecked()
        assert panel.widgets["demo_orbit"].isChecked()
        assert panel.widgets["demo_playback"].isChecked()
        widget.hide()
        assert not widget.refresh_timer.isActive()
    finally:
        panel.cleanup()


def test_presentation_falls_back_to_orbit_and_explore_preserves_look(panel):
    panel.service.capabilities["tour"] = False
    panel.refresh()
    panel.start()
    assert panel.service.start.call_args.kwargs["orbit"]
    select(panel, "explore")
    panel.start()
    options = panel.service.start.call_args.kwargs
    assert options["playback"] and options["orbit"]
    assert not options["cycle_appearance"] and options["presentation"] is None


def test_unavailable_controls_rows_and_recipes_disappear_without_latent_options(panel):
    select(panel, "custom")
    for name in ("tour", "targets", "showcase", "ibl", "playback", "exhibition"):
        panel.widgets[f"demo_{name}"].setChecked(True)
    panel.widgets["demo_lighting_mode"].setCurrentIndex(1)
    panel.service.capabilities.update(
        dict.fromkeys(
            (
                "tour",
                "targets",
                "showcase",
                "sensing_story",
                "ibl",
                "playback",
                "exhibition",
                "dusk",
            ),
            False,
        )
    )
    panel.refresh()
    for name in ("tour", "targets", "showcase", "sensing_story", "ibl", "playback", "exhibition"):
        assert panel.widgets[f"demo_{name}"].isHidden()
        assert not panel.widgets[f"demo_{name}"].isChecked()
    assert panel.widgets["demo_lighting_mode"].isHidden()
    assert panel._form_labels["demo_lighting_mode"].isHidden()
    for name in ("mpc_story", "sensing_story", "follow_targets"):
        assert panel.widgets["demo_preset"].findData(name) < 0
    panel.start()
    options = panel.service.start.call_args.kwargs
    assert not any(
        options[key]
        for key in (
            "tour",
            "cycle_targets",
            "showcase",
            "sensing_story",
            "cycle_ibl",
            "playback",
            "scenario_dusk",
            "scenario_sunset",
            "exhibition",
        )
    )


def test_selected_recipe_becomes_explore_when_required_content_disappears(panel):
    select(panel, "follow_targets")
    assert panel.widgets["demo_targets"].isChecked()
    panel.service.capabilities["targets"] = False
    panel.refresh()
    assert panel.widgets["demo_preset"].currentData() == "explore"
    assert panel.widgets["demo_orbit"].isChecked()


def test_manual_changes_use_custom_and_survive_refresh(panel):
    panel.widgets["demo_speed"].setValue(7)
    assert panel.widgets["demo_preset"].currentData() == "custom"
    panel.refresh()
    assert panel.widgets["demo_speed"].value() == 7
    assert panel.widgets["demo_preset"].currentData() == "custom"
    panel.widgets["demo_exhibition"].setChecked(True)
    assert not panel.widgets["demo_clean_view"].isChecked()
    panel.widgets["demo_clean_view"].setChecked(True)
    assert not panel.widgets["demo_exhibition"].isChecked()


def test_active_camera_changes_do_not_reconfigure_recipe_and_resume_remains_available(panel):
    before = panel.widgets["demo_preset"].currentData()
    panel.service.active = panel.service.camera_paused = True
    panel.service.capabilities.update(dict.fromkeys(panel.service.capabilities, False))
    panel.refresh()
    assert panel.widgets["demo_preset"].currentData() == before
    assert not panel.widgets["demo_preset"].isEnabled()
    assert not panel.widgets["demo_resume_camera"].isHidden()
    assert panel.widgets["demo_resume_camera"].property("demoAction")
    panel.widgets["demo_resume_camera"].click()
    panel.service.resume_camera.assert_called_once()


def test_story_modes_forward_only_their_owned_behaviors(panel):
    select(panel, "mpc_story")
    panel.start()
    options = panel.service.start.call_args.kwargs
    assert options["showcase"] and not options["sensing_story"]
    assert not options["cycle_appearance"] and options["presentation"] is None
    select(panel, "sensing_story")
    panel.start()
    options = panel.service.start.call_args.kwargs
    assert options["sensing_story"] and not options["showcase"]
    assert not any(
        options[key]
        for key in (
            "orbit",
            "tour",
            "cycle_targets",
            "cycle_appearance",
            "cycle_ibl",
            "cycle_lighting",
            "playback",
        )
    )
    assert options["presentation"] is None
    assert options["look_interval"] == 12
    assert panel._form_labels["demo_interval"].text() == "Reveal every"
    select(panel, "presentation")
    assert panel.widgets["demo_interval"].value() == 5


def test_sensing_presentation_has_fixed_detailed_look_and_two_figure_exhibition(panel):
    select(panel, "sensing_presentation")
    panel.start()
    options = panel.service.start.call_args.kwargs
    assert options["presentation"] == "Detailed"
    assert options["exhibition"] and options["exhibition_content"] == "sensing"
    assert options["playback"] and options["captions"]
    assert not any(
        options[key]
        for key in (
            "orbit",
            "clean_view",
            "sensing_story",
            "cycle_appearance",
            "cycle_lighting",
            "cycle_ibl",
        )
    )
    panel.widgets["demo_orbit"].setChecked(True)
    assert panel.widgets["demo_preset"].currentData() == "custom"
    panel.start()
    assert panel.service.start.call_args.kwargs["exhibition_content"] == "sensing"
    panel.service.capabilities["sensing_presentation"] = False
    panel.refresh()
    assert panel.widgets["demo_preset"].findData("sensing_presentation") < 0
    assert panel.widgets["demo_exhibition_content"].currentData() == "actor"
