"""Search existing controls and lazy section titles without building hidden plots."""

import re
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass
class ControlMatch:
    panel: str
    label: str
    terms: str
    widget: QWidget | None = None


def control_matches(manager):
    matches, seen = [], set()
    for key, section in manager.sections.items():
        tab = manager.get_panel_tab_label(key) or ""
        index = manager.find_tab_index_by_label(tab)
        if index >= 0 and not manager._tab_available_at(index):
            continue
        title = section._title
        matches.append(ControlMatch(key, f"{tab} › {title}", f"{tab} {title} {key}".lower()))
        for widget in section.findChildren(QWidget):
            if not isinstance(widget, (QAbstractButton, QComboBox, QLabel, QLineEdit)):
                continue
            text = widget.text() if hasattr(widget, "text") else widget.currentText()
            tip = widget.toolTip()
            options = (
                " ".join(widget.itemText(i) for i in range(widget.count()))
                if isinstance(widget, QComboBox)
                else ""
            )
            label = re.sub("<[^>]+>", "", str(text or tip)).strip().replace("&", "")
            if not label or len(label) > 150:
                continue
            identity = (key, label)
            if identity in seen:
                continue
            seen.add(identity)
            matches.append(
                ControlMatch(
                    key,
                    f"{tab} › {title} › {label[:85]}",
                    f"{tab} {title} {label} {tip} {options}".lower(),
                    widget,
                )
            )
    for key in ("animation", "camera", "context"):
        panel = manager.panels.get(key)
        if panel is None:
            continue
        for name, widget in panel.widgets.items():
            if not isinstance(widget, QWidget):
                continue
            label = re.sub(r"_(cb|combo|spin|spinbox|btn|rb|label|container)$", "", name).replace(
                "_", " "
            )
            tip = widget.toolTip()
            matches.append(
                ControlMatch(
                    key, f"{key.title()} › {label.title()}", f"{key} {label} {tip}".lower(), widget
                )
            )
    return matches


class ControlSearchDialog(QDialog):
    """Jump to matching settings, expanding their containing section."""

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.matches = control_matches(manager)
        self.setWindowTitle("Find a control")
        self.resize(650, 420)
        layout = QVBoxLayout(self)
        self.query = QLineEdit()
        self.query.setPlaceholderText("Search controls, section names, or tooltips…")
        layout.addWidget(self.query)
        self.results = QListWidget()
        layout.addWidget(self.results)
        self.query.textChanged.connect(self._filter)
        self.query.returnPressed.connect(self._activate)
        self.results.itemActivated.connect(self._activate)
        self._filter("")
        self.query.setFocus()

    def _filter(self, query):
        self.results.clear()
        words = query.lower().split()
        for match in self.matches:
            if all(word in match.terms for word in words):
                item = QListWidgetItem(match.label)
                item.setData(Qt.UserRole, match)
                self.results.addItem(item)
        if self.results.count():
            self.results.setCurrentRow(0)

    def _activate(self, *_):
        item = self.results.currentItem()
        if item is None:
            return
        match = item.data(Qt.UserRole)
        self.accept()
        self.manager.reveal_control(match.panel, match.widget)
