"""HUD appearance is saved independently of content detail and visibility."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from visualizer.src.panels.render_panel import RenderPanel
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.state import (
    AppState,
    create_initial_state,
    normalize_viewport_hud_style,
    update_state,
)


@pytest.mark.parametrize("style", ["classic", "card", "minimal"])
def test_hud_style_round_trips_without_changing_content_policy(style):
    state = create_initial_state(
        viewport_hud_style=style,
        viewport_hud_mode="detailed",
        viewport_hud_enabled=False,
        viewport_hud_show_status=False,
    )

    restored = AppState.from_dict(state.to_dict())

    assert restored.viewport_hud_style == style
    assert restored.viewport_hud_mode == "detailed"
    assert not restored.viewport_hud_enabled
    assert not restored.viewport_hud_show_status


def test_missing_or_unsupported_hud_style_uses_classic():
    saved = create_initial_state().to_dict()
    saved.pop("viewport_hud_style")
    assert AppState.from_dict(saved).viewport_hud_style == "classic"
    saved["viewport_hud_style"] = "unknown"
    assert AppState.from_dict(saved).viewport_hud_style == "classic"
    assert normalize_viewport_hud_style(" CARD ") == "card"
    assert normalize_viewport_hud_style(None) == "classic"
    state = create_initial_state(viewport_hud_style="card")
    assert update_state(state, viewport_hud_style="unknown").viewport_hud_style == "classic"


@pytest.fixture
def hud_panel(qtbot, monkeypatch):
    monkeypatch.setattr(RenderPanel, "_load_ibl_library_from_disk", lambda _self: None)
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(viewport_hud=True), refresh_viewport_hud=Mock()
    )
    viz = SimpleNamespace(
        app_state=create_initial_state(
            viewport_hud_enabled=False,
            viewport_hud_mode="detailed",
            viewport_hud_show_status=False,
        ),
        renderer=renderer,
    )
    viz.set_state = Mock(
        side_effect=lambda **changes: setattr(
            viz, "app_state", update_state(viz.app_state, **changes)
        )
    )
    panel = RenderPanel(viz)
    group = panel.create_viewport_hud_panel()
    qtbot.addWidget(group)
    yield panel
    panel.cleanup()


def test_style_control_updates_once_while_hud_is_hidden(hud_panel):
    combo = hud_panel.widgets["viewport_hud_style_combo"]
    assert [combo.itemText(i) for i in range(combo.count())] == ["Classic", "Card", "Minimal"]
    assert combo.isEnabled()

    combo.setCurrentIndex(combo.findData("card"))

    viz = hud_panel.parent
    viz.set_state.assert_called_once_with(viewport_hud_style="card")
    viz.renderer.refresh_viewport_hud.assert_called_once_with()
    assert viz.app_state.viewport_hud_style == "card"
    assert viz.app_state.viewport_hud_mode == "detailed"
    assert not viz.app_state.viewport_hud_enabled
    assert not viz.app_state.viewport_hud_show_status


def test_style_control_mirrors_workspace_state_without_emitting_change(hud_panel):
    viz = hud_panel.parent
    viz.app_state = AppState.from_dict({**viz.app_state.to_dict(), "viewport_hud_style": "minimal"})

    hud_panel._sync_viewport_hud_controls()

    assert hud_panel.widgets["viewport_hud_style_combo"].currentData() == "minimal"
    viz.set_state.assert_not_called()
    viz.renderer.refresh_viewport_hud.assert_not_called()
