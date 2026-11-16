"""Read-only inspection of imported sensing and radio settings."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget


class SensingInspector(QWidget):
    """Expose preserved configuration without adding sensing authoring controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.PlainText)
        layout.addWidget(self.summary)
        self.show_guides = QCheckBox("Show sensing guides in the scene")
        self.show_guides.setToolTip(
            "Read-only sensor boresights, configured range gates and angular path-selection bounds. These are not predicted detection coverage."
        )
        layout.addWidget(self.show_guides)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(("Setting", "Imported value"))
        self.tree.setAlternatingRowColors(True)
        layout.addWidget(self.tree)
        self.set_mapping({})

    def set_mapping(self, mapping):
        self.tree.clear()
        sensing = mapping.get("sensing") or {}
        self.show_guides.setEnabled(isinstance(sensing, dict) and bool(sensing.get("enabled")))
        if not isinstance(sensing, dict):
            self.summary.setText(
                "Sensing settings could not be read. See Problems for configuration errors."
            )
            return
        self.summary.setText(
            ("Sensing enabled." if sensing.get("enabled") else "Sensing disabled.")
            + " Imported parameters are read-only and preserved when saving. "
            "Use the viewport and timeline to inspect the environment, sensors and targets."
            if sensing
            else "No sensing configuration is present. The Builder creates environments and actors; it does not create sensing parameters."
        )

        def append(parent, key, value, path):
            nested = isinstance(value, (dict, list))
            label = str(key).replace("_", " ")
            display = "" if nested else str(value)
            if str(key).endswith("_hz") and isinstance(value, (int, float)) and value > 0:
                for factor, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"), (1, "Hz")):
                    if value >= factor:
                        display = f"{value / factor:g} {unit}"
                        break
                label = str(key)[:-3].replace("_", " ")
            item = QTreeWidgetItem((label, display))
            item.setToolTip(0, path)
            if not nested:
                item.setToolTip(1, str(value))
            (
                parent.addChild(item)
                if isinstance(parent, QTreeWidgetItem)
                else parent.addTopLevelItem(item)
            )
            if isinstance(value, dict):
                for name, child in value.items():
                    append(item, name, child, f"{path}.{name}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    append(item, index, child, f"{path}.{index}")

        if sensing:
            append(self.tree, "Sensing", sensing, "sensing")
            radio = mapping.get("raytracing") or {}
            if not isinstance(radio, dict):
                radio = {}
            for name in ("carrier_frequency_hz", "antenna"):
                if name in radio:
                    append(self.tree, name, radio[name], f"raytracing.{name}")
            self.tree.expandToDepth(1)
            self.tree.resizeColumnToContents(0)
