"""Shared callout controls for actors, sensing tracks and volume locators."""

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QWidget

from ..renderers.protocol import renderer_capabilities
from ..types.label_style import CALLOUT_MARKER_SIZE_BOUNDS_PX, LabelCalloutStyle
from ..widgets.form_layout import AlignedFormLayout
from .collapsible_section import CollapsibleSection


class LabelStyleControls(QWidget):
    """One global label appearance, also available without coverage data."""

    styleChanged = Signal(object)
    displayModeChanged = Signal(str)

    def __init__(
        self, visualizer, parent=None, *, show_marker_size=True, collapsible_details=False
    ):
        super().__init__(parent)
        self.visualizer = visualizer
        layout = AlignedFormLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        self.mode = QComboBox()
        self.mode.addItem("Plain text", "plain")
        self.mode.addItem("Callout", "callout")
        self.mode.addItem("Card", "card")
        self.mode.setToolTip(
            "Shared by actor and sensing-track labels. "
            "Callout uses a square marker and dashed guide. Card adds rounded corners, "
            "text padding, a black border and a solid guide. Requires pygfx. "
            "Callout and Card use screen offsets in pixels and retain the saved "
            "font size and world offset."
        )
        layout.addRow("Label style", self.mode)
        self.details = QWidget()
        form = AlignedFormLayout(self.details)
        form.setContentsMargins(0, 0, 0, 0)
        self.background = QComboBox()
        for text, value in (
            ("White Tx / tinted Rx, targets and tracks", "paper"),
            ("Tinted identity colour", "node"),
            ("White", "white"),
            ("Dark navy", "ink"),
        ):
            self.background.addItem(text, value)
        form.addRow("Background", self.background)
        self.spins = {}
        for key, text, bounds, step, suffix in (
            ("opacity", "Background opacity", (0.1, 1), 0.05, ""),
            ("offset_x", "Horizontal offset", (-500, 500), 5, " px"),
            ("offset_y", "Vertical offset", (-500, 500), 5, " px"),
            ("marker_size", "Position marker", CALLOUT_MARKER_SIZE_BOUNDS_PX, 2, " px"),
        ):
            if key == "marker_size" and not show_marker_size:
                continue
            spin = QDoubleSpinBox()
            spin.setRange(*bounds)
            spin.setSingleStep(step)
            spin.setDecimals(2 if key == "opacity" else 0)
            spin.setSuffix(suffix)
            if key in ("offset_x", "offset_y"):
                spin.setToolTip(
                    "Screen offset in pixels, relative to the saved world label position. "
                    "Changes the annotation only; actor positions are unchanged."
                )
            form.addRow(text, spin)
            self.spins[key] = spin
        self.leader = QCheckBox("Position guide")
        self.on_top = QCheckBox("Keep annotations visible through geometry")
        self.on_top.setToolTip(
            "Display-only overlay: visibility of a marker does not establish a clear radio path."
        )
        form.addRow(self.leader)
        form.addRow(self.on_top)
        self.details_container = self.details
        if collapsible_details:
            self.details_container = CollapsibleSection(
                "Callout appearance", start_open=False, flat_header=True
            )
            self.details_container.content_layout().addWidget(self.details)
        layout.addRow(self.details_container)
        self.sync_from_state()
        self.mode.currentIndexChanged.connect(self._changed)
        self.background.currentIndexChanged.connect(self._changed)
        self.leader.toggled.connect(self._changed)
        self.on_top.toggled.connect(self._changed)
        for spin in self.spins.values():
            spin.valueChanged.connect(self._changed)
        self.styleChanged.connect(
            lambda style: visualizer.ui_controller.handle_label_callout_changed(style)
        )

    def sync_from_state(self):
        """Restore settings silently, including after loading a workspace."""
        style = LabelCalloutStyle.from_dict(
            getattr(getattr(self.visualizer, "app_state", None), "label_callout", None)
        )
        for widget, value in ((self.mode, style.mode), (self.background, style.background)):
            with QSignalBlocker(widget):
                widget.setCurrentIndex(widget.findData(value))
        for key, widget in self.spins.items():
            with QSignalBlocker(widget):
                widget.setValue(getattr(style, key))
        for key in ("leader", "on_top"):
            widget = getattr(self, key)
            with QSignalBlocker(widget):
                widget.setChecked(getattr(style, key))
        supported = renderer_capabilities(getattr(self.visualizer, "renderer", None)).label_callouts
        self.mode.setEnabled(supported)
        self.details_container.setEnabled(supported)
        self._sync_mode_visibility(style)

    def _sync_mode_visibility(self, style):
        """Refresh relevant controls, including after a silent workspace restore."""
        supported = renderer_capabilities(getattr(self.visualizer, "renderer", None)).label_callouts
        mode = style.mode if supported else "plain"
        self.details_container.setVisible(mode != "plain")
        self.displayModeChanged.emit(mode)

    def _changed(self, *_args):
        # Each surface edits only its own fields; retain TX/RX overrides elsewhere.
        values = LabelCalloutStyle.from_dict(
            getattr(getattr(self.visualizer, "app_state", None), "label_callout", None)
        ).to_dict()
        values.update({key: spin.value() for key, spin in self.spins.items()})
        values.update(
            mode=self.mode.currentData(),
            background=self.background.currentData(),
            leader=self.leader.isChecked(),
            on_top=self.on_top.isChecked(),
        )
        style = LabelCalloutStyle.from_dict(values)
        self._sync_mode_visibility(style)
        self.styleChanged.emit(style)
