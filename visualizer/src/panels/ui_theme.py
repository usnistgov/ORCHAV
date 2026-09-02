"""Small helpers for theme-aware panel-local Qt styling."""

from __future__ import annotations

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QLabel, QWidget

EMPTY_DISPLAY_VALUE = "--"


def _fixed_width_font() -> QFont:
    """Return Qt's platform-selected fixed-width UI font."""
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


def set_widget_role(widget: QWidget, role: str | None) -> None:
    """Set a stylesheet role property and refresh Qt polish immediately."""
    widget.setProperty("role", role or "")
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def configure_label(
    label: QLabel,
    *,
    role: str | None = None,
    min_width: int | None = None,
    font_size: int | None = None,
    bold: bool = False,
    italic: bool = False,
    monospace: bool = False,
    word_wrap: bool | None = None,
) -> QLabel:
    """Apply non-color label styling while color remains theme-driven."""
    if role is not None:
        set_widget_role(label, role)
    if min_width is not None:
        label.setMinimumWidth(int(min_width))
    if word_wrap is not None:
        label.setWordWrap(bool(word_wrap))

    font = label.font()
    changed = False
    if font_size is not None:
        font.setPointSize(int(font_size))
        changed = True
    if bold:
        font.setWeight(QFont.Weight.Bold)
        changed = True
    if italic:
        font.setItalic(True)
        changed = True
    if monospace:
        fixed_font = _fixed_width_font()
        font.setFamily(fixed_font.family())
        font.setStyleHint(QFont.StyleHint.Monospace)
        changed = True
    if changed:
        label.setFont(font)
    return label


def compact_button_style() -> str:
    """Return compact action-button sizing without overriding theme colors."""
    return """
        QPushButton {
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 11px;
            min-height: 20px;
        }
    """


def compact_text_edit_style(*, monospace: bool = False, font_size: int = 9) -> str:
    """Return sizing-only QTextEdit QSS, leaving colors to the app theme."""
    family = ""
    if monospace:
        fixed_family = _fixed_width_font().family()
        fixed_family = fixed_family.replace("\\", "\\\\").replace('"', '\\"')
        family = f'font-family: "{fixed_family}";'
    return f"""
        QTextEdit {{
            border-radius: 4px;
            padding: 4px;
            {family}
            font-size: {int(font_size)}px;
        }}
    """


def compact_progress_bar_style(*, font_size: int = 9) -> str:
    """Return sizing-only QProgressBar QSS, leaving colors to the app theme."""
    return f"""
        QProgressBar {{
            border-radius: 4px;
            text-align: center;
            font-size: {int(font_size)}px;
        }}
        QProgressBar::chunk {{
            border-radius: 3px;
        }}
    """
