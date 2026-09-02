"""Tests for visualizer keyboard shortcut registry and help table."""

from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtWidgets import QDialogButtonBox, QTableWidget

from visualizer.src.app.shortcuts import SHORTCUTS, shortcut, shortcut_rows
from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.panels.help_dialog import HelpDialog
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.state import create_initial_state, update_state
from visualizer.visualizer import OrchavVisualizer


class _FakeSignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self) -> None:
        for callback in self._callbacks:
            callback()


def test_shortcut_registry_sequences_are_qt_parseable() -> None:
    """Every documented shortcut should map to a non-empty Qt key sequence."""
    assert {spec.identifier for spec in SHORTCUTS} == {
        "play_pause",
        "previous_frame",
        "next_frame",
        "toggle_hud",
        "save_session",
        "load_session",
        "toggle_metrics",
    }

    for spec in SHORTCUTS:
        sequence = spec.key_sequence()
        assert sequence.count() == 1
        assert sequence.toString(), spec.identifier
        assert shortcut(spec.identifier) is spec


def test_workspace_save_does_not_conflict_with_scenario_authoring_save() -> None:
    """Ctrl+S remains available to the authoring workflow."""
    assert shortcut("save_session").sequence == "Ctrl+Alt+S"


def test_context_hud_switch_only_toggles_when_requested_state_differs() -> None:
    """The persistent checkbox owns master state without changing HUD detail."""

    class Renderer:
        capabilities = RendererCapabilities(viewport_hud=True)

        def __init__(self):
            self.refresh_calls = 0

        def refresh_viewport_hud(self):
            self.refresh_calls += 1

    class Visualizer:
        app_state = create_initial_state(viewport_hud_mode="detailed")
        renderer = Renderer()
        ui_manager = None

        def set_state(self, **changes):
            self.app_state = update_state(self.app_state, **changes)

    controller = UIController.__new__(UIController)
    controller.visualizer = Visualizer()

    controller.handle_viewport_hud_enabled_toggled(True)
    controller.handle_viewport_hud_enabled_toggled(False)
    controller.handle_viewport_hud_enabled_toggled(False)
    controller.handle_viewport_hud_enabled_toggled(True)

    assert controller.visualizer.app_state.viewport_hud_enabled is True
    assert controller.visualizer.app_state.viewport_hud_mode == "detailed"
    assert controller.visualizer.renderer.refresh_calls == 2


def test_help_dialog_rows_come_from_shortcut_registry(qapp) -> None:
    """The help dialog should render the same rows as the runtime registry."""
    dialog = HelpDialog()
    table = dialog.findChild(QTableWidget)

    assert table is not None
    assert table.rowCount() == len(SHORTCUTS)

    rendered_rows = [
        (table.item(row, 0).text(), table.item(row, 1).text()) for row in range(table.rowCount())
    ]
    assert rendered_rows == shortcut_rows()

    buttons = dialog.findChild(QDialogButtonBox)
    assert buttons is not None
    assert buttons.button(QDialogButtonBox.Close) is not None


def test_runtime_shortcuts_call_animation_controller(monkeypatch) -> None:
    """Runtime playback shortcuts should target AnimationController directly."""
    import visualizer.visualizer as visualizer_module

    shortcuts = []

    class _FakeShortcut:
        def __init__(self, _sequence, _parent) -> None:
            self.activated = _FakeSignal()
            shortcuts.append(self)

    calls = []

    class _AnimationController:
        def toggle_animation(self, direction=None) -> None:
            calls.append(("toggle", direction))

        def previous_frame(self) -> None:
            calls.append(("previous",))

        def next_frame(self) -> None:
            calls.append(("next",))

    class _PlayButton:
        def setChecked(self, checked) -> None:
            calls.append(("checked", checked))

    class _UIController:
        def toggle_viewport_hud(self) -> None:
            calls.append(("hud",))

    monkeypatch.setattr(visualizer_module, "QShortcut", _FakeShortcut)
    viz = OrchavVisualizer.__new__(OrchavVisualizer)
    viz.animation_controller = _AnimationController()
    viz.animation_running = False
    viz.play_btn = _PlayButton()
    viz.ui_controller = _UIController()

    OrchavVisualizer._setup_keyboard_shortcuts(viz)

    assert len(shortcuts) == 4
    shortcuts[0].activated.emit()
    shortcuts[1].activated.emit()
    shortcuts[2].activated.emit()
    shortcuts[3].activated.emit()

    assert calls == [
        ("toggle", 1),
        ("checked", False),
        ("previous",),
        ("next",),
        ("hud",),
    ]


def test_playback_shortcut_is_workspace_scoped_not_viewport_focus() -> None:
    calls = []
    viz = OrchavVisualizer.__new__(OrchavVisualizer)
    viz.workspace_mode_controller = SimpleNamespace(mode=SimpleNamespace(value="visualization"))

    OrchavVisualizer._run_viewport_shortcut(viz, lambda: calls.append("visualization"))
    viz.workspace_mode_controller.mode = SimpleNamespace(value="authoring")
    OrchavVisualizer._run_viewport_shortcut(viz, lambda: calls.append("authoring"))

    assert calls == ["visualization"]
