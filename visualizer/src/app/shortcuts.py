"""Shared keyboard shortcut registry for the visualizer shell and help dialog."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QKeySequence


@dataclass(frozen=True)
class ShortcutSpec:
    """One runtime shortcut and its user-facing help-table description."""

    identifier: str
    sequence: str
    display: str
    description: str

    def key_sequence(self) -> QKeySequence:
        """Return the Qt key sequence used by runtime shortcut registration."""
        return QKeySequence(self.sequence)


SHORTCUTS: tuple[ShortcutSpec, ...] = (
    ShortcutSpec("open_scenario", "Ctrl+O", "Ctrl+O", "Open scenario"),
    ShortcutSpec("find_control", "Ctrl+K", "Ctrl+K", "Find a control"),
    ShortcutSpec("help", "F1", "F1", "Help and controls"),
    ShortcutSpec("play_pause", "Space", "Space", "Play / Pause animation"),
    ShortcutSpec("previous_frame", "Left", "Left Arrow", "Previous frame"),
    ShortcutSpec("next_frame", "Right", "Right Arrow", "Next frame"),
    ShortcutSpec("toggle_hud", "Ctrl+H", "Ctrl+H", "Hide / restore viewport HUD"),
    ShortcutSpec(
        "save_session",
        "Ctrl+Alt+S",
        "Ctrl+Alt+S",
        "Save workspace snapshot",
    ),
    ShortcutSpec("load_session", "Ctrl+L", "Ctrl+L", "Open workspace snapshot"),
    ShortcutSpec("toggle_metrics", "M", "M", "Show / hide live frame metrics"),
)

_SHORTCUTS_BY_ID = {shortcut.identifier: shortcut for shortcut in SHORTCUTS}


def shortcut(identifier: str) -> ShortcutSpec:
    """Return a shortcut specification by stable identifier."""
    return _SHORTCUTS_BY_ID[identifier]


def shortcut_rows() -> list[tuple[str, str]]:
    """Return help-dialog rows from the shared runtime shortcut registry."""
    return [(item.display, item.description) for item in SHORTCUTS]
