"""Whole-retained-run scientific overview, independent of the playback cursor."""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QSignalBlocker, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..app.theme import current_theme
from ..widgets.dashboard_plot import DashboardPlot
from .dashboard_evaluation import DEFAULT_RANGE_GATE_M, DEFAULT_VELOCITY_GATE_M_S
from .display_style import identity_color
from .evaluation_settings import EvaluationSettings
from .time_interval import TimeBoundSpinBox

FIGURE_HELP = {
    "counts": (
        "Detection outcomes",
        "TP: matched outputs. FP: unmatched outputs. FN: unmatched eligible targets.",
    ),
    "population": (
        "Truth and saved outputs",
        "Eligible targets and scored outputs in each acquisition.",
    ),
    "rates": (
        "Cumulative recall and precision",
        "Recall = TP / (TP + FN); precision = TP / (TP + FP), using cumulative counts.",
    ),
    "coverage": (
        "Per-target coverage",
        "Matched appearances divided by eligible appearances, per target.",
    ),
    "range": (
        "Range: truth and saved outputs",
        "Recorded range at each acquisition time.",
    ),
    "radial": (
        "Radial velocity: truth and saved outputs",
        "Sensor-relative radial velocity from the saved reference metadata.",
    ),
    "range_error": (
        "Matched absolute range error",
        "Share of matched pairs below each absolute range error.",
    ),
    "radial_error": (
        "Matched absolute radial error",
        "Share of matched pairs below each absolute radial-velocity error.",
    ),
    "track_population": (
        "Saved track confirmation",
        "Confirmed and tentative output rows; internal/coasting states are excluded.",
    ),
    "identity": (
        "Target-to-track assignments",
        "Saved track ID matched to each truth target using the selected tolerances.",
    ),
}


# Summary-local semantic palette: shape/line patterns also distinguish every series.
COLORS = {"tp": "#00866b", "fp": "#b5475a", "fn": "#a66b00", "gt": "#2865b0", "output": "#bf5b13"}
METRICS = (
    ("tp", "TRUE POSITIVES", "matched outputs", COLORS["tp"]),
    ("fp", "FALSE POSITIVES", "unmatched outputs", COLORS["fp"]),
    ("fn", "FALSE NEGATIVES", "missed targets", COLORS["fn"]),
    ("recall", "RECALL", "TP / (TP + FN)", "#2865b0"),
    ("precision", "PRECISION", "TP / (TP + FP)", "#6251a3"),
    ("range", "RANGE RMSE", "matched pairs · m", "#64748b"),
    ("radial", "RADIAL RMSE", "matched pairs · m/s", "#64748b"),
)


def _summary_style(theme):
    return f"""
QWidget#SensingRunSummary {{ background: {theme.bg_primary}; color: {theme.text_primary}; font-family: 'DejaVu Sans'; font-size: 12px; }}
QWidget#SensingRunSummary QLabel {{ background: transparent; color: {theme.text_primary}; }}
QWidget#SensingRunSummary QLabel#summaryTitle {{ font-size: 24px; font-weight: 700; }}
QWidget#SensingRunSummary QLabel#scope {{ color: {theme.text_secondary}; font-size: 12px; }}
QWidget#SensingRunSummary QFrame#metricCard, QWidget#SensingRunSummary QFrame#plotCard {{ background: {theme.bg_secondary}; border: 1px solid {theme.border_subtle}; border-radius: 9px; }}
QWidget#SensingRunSummary QLabel#metricName {{ color: {theme.text_secondary}; font-size: 10px; font-weight: 600; }}
QWidget#SensingRunSummary QLabel#metricValue {{ color: {theme.text_primary}; font-size: 24px; font-weight: 700; }}
QWidget#SensingRunSummary QLabel#metricCaption {{ color: {theme.text_secondary}; font-size: 10px; }}
QWidget#SensingRunSummary QLabel#coverageNote {{ color: {theme.text_secondary}; font-size: 12px; }}
QWidget#SensingRunSummary QLabel#methodNote, QWidget#SensingRunSummary QLabel#plotNote {{ color: {theme.text_secondary}; font-size: 11px; }}
QWidget#SensingRunSummary QLabel#plotNote {{ padding: 3px 8px 0 8px; }}
QWidget#SensingRunSummary QFrame#settingsDrawer {{ background: {theme.bg_tertiary}; border: 1px solid {theme.border_subtle}; border-radius: 7px; padding: 6px; }}
QWidget#SensingRunSummary QPushButton {{ background: {theme.bg_secondary}; color: {theme.text_secondary}; border: 1px solid {theme.border_primary}; border-radius: 6px; padding: 7px 11px; font-size: 11px; }}
QWidget#SensingRunSummary QPushButton:hover, QWidget#SensingRunSummary QPushButton:checked {{ background: {theme.bg_tertiary}; border-color: {theme.accent_hover}; }}
QWidget#SensingRunSummary QPushButton#applySettings {{ background: {theme.accent}; color: {"#172b4d" if theme.is_dark else "white"}; border: none; }}
QWidget#SensingRunSummary QDoubleSpinBox {{ background: {theme.bg_secondary}; color: {theme.text_primary}; border: 1px solid {theme.border_primary}; border-radius: 4px; padding: 4px; min-width: 70px; }}
QWidget#SensingRunSummary QDoubleSpinBox:disabled {{ color: {theme.text_muted}; background: {theme.bg_primary}; }}
QWidget#SensingRunSummary QTabWidget::pane {{ border: none; background: transparent; }}
QWidget#SensingRunSummary QTabBar::tab {{ background: transparent; color: {theme.text_secondary}; border: none; border-bottom: 2px solid transparent; padding: 8px 20px; font-size: 12px; }}
QWidget#SensingRunSummary QTabBar::tab:selected {{ color: {theme.accent}; font-weight: 600; border-bottom: 2px solid {theme.accent}; }}
QWidget#SensingRunSummary QTabBar::tab:disabled {{ color: {theme.text_muted}; }}
QWidget#SensingRunSummary QScrollArea, QWidget#SensingRunSummary QScrollArea > QWidget > QWidget {{ background: transparent; }}
"""


class _SummaryLayout(QVBoxLayout):
    """Embedded rows already resolve wrapping before publishing a fixed height."""

    def hasHeightForWidth(self):
        if self.parentWidget()._embedded_graph_height is not None:
            return False
        return super().hasHeightForWidth()


class RunSummaryPanel(QWidget):
    """Readable cards plus linked 2D figures; no detector or renderer is invoked."""

    settings_changed = Signal()
    time_selected = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SensingRunSummary")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_summary_style(current_theme()))
        self.setMinimumHeight(700)
        self.data = self._rendered = None
        self.plots, self.plot_cards, self.plot_notes = {}, {}, {}
        self._page_grids = []
        self._columns = 2
        self._empty_keys = set()
        self._embedded_graph_height = None
        self._embedded_size_timer = QTimer(self)
        self._embedded_size_timer.setSingleShot(True)
        self._embedded_size_timer.timeout.connect(self._size_embedded_content)
        outer = _SummaryLayout(self)
        outer.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        outer.setContentsMargins(12, 10, 12, 8)
        outer.setSpacing(8)
        header = self.header = QBoxLayout(QBoxLayout.Direction.TopToBottom)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("Run summary")
        title.setObjectName("summaryTitle")
        titles.addWidget(title)
        header.addLayout(titles)
        self.settings_toggle = QPushButton("Evaluation settings")
        self.settings_toggle.setCheckable(True)
        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(self.settings_toggle)
        header.addLayout(actions)
        outer.addLayout(header)
        self.scope = QLabel("All retained acquisitions · select a saved result to begin")
        self.scope.setObjectName("scope")
        self.scope.setWordWrap(True)
        outer.addWidget(self.scope)

        self.controls = QFrame()
        self.controls.setObjectName("settingsDrawer")
        controls = self.controls_grid = QGridLayout(self.controls)
        controls.setContentsMargins(12, 8, 12, 8)
        self.range_gate = self._spin(DEFAULT_RANGE_GATE_M, 0.001, 10000, " m")
        self.velocity_gate = self._spin(DEFAULT_VELOCITY_GATE_M_S, 0.001, 10000, " m/s")
        self.outputs = QComboBox()
        self.outputs.addItem("All saved outputs", False)
        self.outputs.addItem("Confirmed tracks only", True)
        self.interval = QCheckBox("Interval")
        self.start = self._spin(0, -1e12, 1e12, " s")
        self.end = self._spin(1, -1e12, 1e12, " s")
        for spin in (self.start, self.end):
            spin.setDecimals(6)
            spin.setEnabled(False)
        self.apply = QPushButton("Apply")
        self.apply.setObjectName("applySettings")
        self.apply.clicked.connect(self._apply_settings)
        self.control_items = [
            QLabel("Matching gates ±"),
            self.range_gate,
            self.velocity_gate,
            self.outputs,
            self.interval,
            self.start,
            self.end,
            self.apply,
        ]
        self.controls.hide()
        outer.addWidget(self.controls)
        self.settings_toggle.toggled.connect(self.controls.setVisible)
        self.interval.toggled.connect(self.start.setEnabled)
        self.interval.toggled.connect(self.end.setEnabled)

        # Keep a machine-readable/accessibility summary alongside the visual cards.
        self.cards = QLabel("Summary not loaded", self)
        self.cards.hide()
        metric_row = self.metric_grid = QGridLayout()
        self.metric_cards = []
        metric_row.setSpacing(9)
        self.metric_values, self.metric_captions = {}, {}
        for key, label, caption, color in METRICS:
            card = QFrame()
            card.setObjectName("metricCard")
            card.setStyleSheet(f"QFrame#metricCard {{ border-top: 3px solid {color}; }}")
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            box = QVBoxLayout(card)
            box.setContentsMargins(11, 8, 10, 8)
            box.setSpacing(2)
            name = QLabel(label)
            name.setObjectName("metricName")
            value = QLabel("—")
            value.setObjectName("metricValue")
            note = QLabel(caption)
            note.setObjectName("metricCaption")
            box.addWidget(name)
            box.addWidget(value)
            box.addWidget(note)
            self.metric_cards.append(card)
            self.metric_values[key], self.metric_captions[key] = value, note
        outer.addLayout(metric_row)
        self.coverage_note = QLabel("Choose a recording to see scoring coverage")
        self.coverage_note.setObjectName("coverageNote")
        self.coverage_note.setWordWrap(True)
        outer.addWidget(self.coverage_note)
        self.tabs = QTabWidget()
        # Reserve both plot rows independently of the header and metric cards.
        self.tabs.setMinimumHeight(540)
        self.tabs.currentChanged.connect(self._schedule_embedded_size)
        outer.addWidget(self.tabs, 1)
        self._applied_settings = (DEFAULT_RANGE_GATE_M, DEFAULT_VELOCITY_GATE_M_S, None, False)
        self.evaluation_settings = None
        self.bind_evaluation_settings(EvaluationSettings(self))
        self._reflow()

    def set_embedded_graph_height(self, height: int) -> None:
        """Use the host's scroll area and reserve this data height in each plot.

        Standalone panels keep their own scrollable pages. An embedded summary
        sizes its pages to their figures, so the surrounding workspace owns
        scrolling without stretching every plot to a second scroll viewport.
        """
        self._embedded_graph_height = max(1, int(height))
        self.setMinimumHeight(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tabs.setMinimumHeight(0)
        for plot in self.plots.values():
            plot.set_data_minimum_height(self._embedded_graph_height)
        self._schedule_embedded_size()

    def _schedule_embedded_size(self, *_args):
        if self._embedded_graph_height is not None and not self._embedded_size_timer.isActive():
            self._embedded_size_timer.start(0)

    def event(self, event):
        if event.type() == QEvent.Type.LayoutRequest and hasattr(self, "_embedded_size_timer"):
            self._schedule_embedded_size()
        return super().event(event)

    def eventFilter(self, watched, event):
        # A legend or conditional note can grow after its card was laid out.
        if event.type() == QEvent.Type.LayoutRequest:
            self._schedule_embedded_size()
        return super().eventFilter(watched, event)

    def _size_embedded_content(self):
        if self._embedded_graph_height is None or not self._page_grids:
            return
        for index, (grid, keys) in enumerate(self._page_grids):
            heights = []
            for key in keys:
                card, plot, note = self.plot_cards[key], self.plots[key], self.plot_notes[key]
                box = card.layout()
                margins = box.contentsMargins()
                height = plot.minimumHeight() + margins.top() + margins.bottom()
                height += 2 * card.frameWidth()
                if not note.isHidden():
                    width = max(1, card.width() - margins.left() - margins.right())
                    height += max(note.minimumSizeHint().height(), note.heightForWidth(width))
                    height += box.spacing()
                heights.append(height)
            rows = [
                max(heights[start : start + self._columns])
                for start in range(0, len(heights), self._columns)
            ]
            for row in range(grid.rowCount()):
                if grid.rowStretch(row):
                    grid.setRowStretch(row, 0)
                height = rows[row] if row < len(rows) else 0
                if grid.rowMinimumHeight(row) != height:
                    grid.setRowMinimumHeight(row, height)
            for item, key in enumerate(keys):
                self._set_content_height(self.plot_cards[key], rows[item // self._columns])
            margins = grid.contentsMargins()
            height = sum(rows) + grid.verticalSpacing() * max(0, len(rows) - 1)
            height += margins.top() + margins.bottom()
            scroll = self.tabs.widget(index)
            if scroll.verticalScrollBarPolicy() != Qt.ScrollBarPolicy.ScrollBarAlwaysOff:
                scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._set_content_height(scroll.widget(), height)
            self._set_content_height(scroll, height)
        current = self.tabs.currentWidget()
        frame = self.tabs.style().pixelMetric(
            QStyle.PixelMetric.PM_DefaultFrameWidth, None, self.tabs
        )
        self._set_content_height(
            self.tabs, current.height() + self.tabs.tabBar().sizeHint().height() + 2 * frame
        )
        layout = self.layout()
        height = max(layout.minimumSize().height(), layout.heightForWidth(self.width()))
        self._set_content_height(self, height)

    @staticmethod
    def _set_content_height(widget, height):
        if widget.minimumHeight() != height or widget.maximumHeight() != height:
            widget.setFixedHeight(height)

    def _ensure_plots(self):
        if self.plots:
            return
        for title, keys in (
            ("Overview", ("counts", "population", "rates", "coverage")),
            ("Accuracy", ("range", "radial", "range_error", "radial_error")),
            ("Tracking", ("track_population", "identity")),
        ):
            page = QWidget()
            grid = QGridLayout(page)
            grid.setContentsMargins(0, 10, 0, 0)
            grid.setSpacing(12)
            for index, key in enumerate(keys):
                card = QFrame()
                card.setObjectName("plotCard")
                card.installEventFilter(self)
                box = QVBoxLayout(card)
                box.setContentsMargins(8, 9, 10, 6)
                box.setSpacing(0)
                plot = DashboardPlot()
                plot.setToolTip(FIGURE_HELP[key][1])
                plot.setMinimumSize(270, 155)
                if self._embedded_graph_height is not None:
                    plot.set_data_minimum_height(self._embedded_graph_height)
                plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
                plot.setMenuEnabled(False)
                plot.setTitle(FIGURE_HELP[key][0])
                plot.add_external_legend(
                    labelTextSize="8pt",
                    labelTextColor="#475569",
                    colCount=3,
                    verSpacing=1,
                    horSpacing=8,
                )
                box.addWidget(plot, 1)
                note = QLabel()
                note.setObjectName("plotNote")
                note.setWordWrap(True)
                note.hide()
                box.addWidget(note)
                self.plots[key], self.plot_cards[key], self.plot_notes[key] = plot, card, note
                grid.addWidget(card, index // 2, index % 2)
                grid.setRowStretch(index // 2, 1)
                grid.setColumnStretch(index % 2, 1)
                if key not in {"coverage", "range_error", "radial_error"}:
                    plot.setLabel("bottom", "Result time", units="s")
                    plot.scene().sigMouseClicked.connect(
                        lambda event, widget=plot: self._clicked(widget, event)
                    )
            scroll = QScrollArea()
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self._page_grids.append((grid, keys))
            self.tabs.addTab(scroll, title)
        self.tabs.setTabEnabled(2, False)
        self.apply_theme()
        self._reflow()

    def apply_theme(self):
        """Style only this workspace; leave the application's other plots untouched."""
        theme = current_theme()
        self.setStyleSheet(_summary_style(theme))
        dark_colors = {
            "#00866b": "#4ed4ad",
            "#b5475a": "#ff8b9e",
            "#a66b00": "#eeb958",
            "#2865b0": "#83b9ff",
            "#bf5b13": "#ffb477",
            "#6251a3": "#b9a1f3",
            "#7552a3": "#c4a0ef",
            "#197a78": "#6ad6d0",
            "#64748b": "#aab9cc",
        }

        def color(value):
            return dark_colors.get(value, value) if theme.is_dark else value

        for card, (_, _, _, accent) in zip(self.metric_cards, METRICS):
            card.setStyleSheet(f"QFrame#metricCard {{ border-top: 3px solid {color(accent)}; }}")
        for key, plot in self.plots.items():
            plot.setBackground(theme.bg_secondary)
            plot.setTitle(
                plot.getPlotItem().titleLabel.text, color=theme.text_primary, size="10pt", bold=True
            )
            plot.showGrid(x=False, y=key not in self._empty_keys, alpha=0.10)
            item = plot.getPlotItem()
            item.getViewBox().setBorder(None)
            item.layout.setContentsMargins(6, 0, 6, 0)
            for name in ("bottom", "left"):
                axis = plot.getAxis(name)
                axis.setPen(pg.mkPen(theme.border_primary, width=1))
                axis.setTextPen(pg.mkPen(theme.text_secondary))
                axis.setStyle(tickFont=QFont("DejaVu Sans", 8), maxTickLevel=0)
                axis.setLabel(
                    axis.labelText,
                    units=axis.labelUnits or None,
                    color=theme.text_secondary,
                    **{"font-size": "9pt"},
                )
            if item.legend is not None:
                item.legend.setLabelTextColor(theme.text_secondary)
                # pyqtgraph's setter updates defaults but leaves existing label
                # HTML unchanged until setText() is called.
                for _, label in item.legend.items:
                    label.setText(label.text, color=theme.text_secondary)
                item.legend.setBrush(pg.mkBrush(theme.bg_secondary))
                item.legend.setPen(None)
            # Recolor existing graphics, keeping samples, zoom and object ownership.
            for series in plot.listDataItems():
                base = getattr(series, "_summary_color", None)
                if base is None:
                    continue
                for option, setter in (("pen", series.setPen), ("symbolPen", series.setSymbolPen)):
                    pen = series.opts.get(option)
                    if pen is not None:
                        pen = pg.mkPen(pen)
                        pen.setColor(pg.mkColor(color(base)))
                        setter(pen)
                brush = series.opts.get("symbolBrush")
                if brush is not None and pg.mkBrush(brush).color().alpha():
                    series.setSymbolBrush(color(base))
            for graphic in item.items:
                if isinstance(graphic, pg.TextItem):
                    graphic.setColor(theme.text_secondary)
                elif isinstance(graphic, pg.BarGraphItem):
                    graphic.setOpts(brush=color("#2865b0"))

    @staticmethod
    def _reflow_grid(grid, widgets, columns):
        # Remove old cell constraints before shrinking; otherwise an empty column
        # can retain its width and defeat the compact layout.
        for widget in widgets:
            grid.removeWidget(widget)
        for column in range(grid.columnCount()):
            grid.setColumnStretch(column, 0)
            grid.setColumnMinimumWidth(column, 0)
        for index, widget in enumerate(widgets):
            grid.addWidget(widget, index // columns, index % columns)
        for column in range(columns):
            grid.setColumnStretch(column, 1)

    def _reflow(self):
        compact = self.width() < 850
        self.header.setDirection(
            QBoxLayout.Direction.TopToBottom if compact else QBoxLayout.Direction.LeftToRight
        )
        available = max(1, self.width() - 24)
        card_width = max(card.minimumSizeHint().width() for card in self.metric_cards)
        card_columns = max(1, min(len(self.metric_cards), (available + 9) // (card_width + 9)))
        self._reflow_grid(self.metric_grid, self.metric_cards, card_columns)
        self._reflow_grid(self.controls_grid, self.control_items, 3 if compact else 8)
        columns = 2 if self.width() >= 640 else 1
        for grid, keys in self._page_grids:
            self._reflow_grid(grid, [self.plot_cards[key] for key in keys], columns)
            for key in keys:
                if self._embedded_graph_height is None:
                    self.plot_cards[key].setMinimumHeight(265 if columns == 1 else 220)
            for row in range(grid.rowCount()):
                grid.setRowStretch(row, int(row < (len(keys) + columns - 1) // columns))
        self._columns = columns
        self._size_plot_markers()
        # Let an embedded scroll host allocate the requested figure area rather
        # than squeeze it below the two-row minimum as the card header wraps.
        if self._embedded_graph_height is not None:
            self._schedule_embedded_size()
        elif self.isVisible():
            self.setMinimumHeight(self.layout().minimumSize().height())

    def _size_plot_markers(self):
        for plot in self.plots.values():
            for series in plot.listDataItems():
                size = getattr(series, "_summary_symbol_size", None)
                if size is not None:
                    series.setSymbolSize(max(4, size * (0.7 if self._columns == 2 else 0.85)))

    def minimumSizeHint(self):
        # The wide arrangement is not the minimum: controls reflow on resize.
        return QSize(360, super().minimumSizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow()

    def showEvent(self, event):
        self._ensure_plots()
        super().showEvent(event)

    @staticmethod
    def _spin(value, low, high, suffix):
        spin = TimeBoundSpinBox() if suffix == " s" else QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(3)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    def settings(self):
        return self._applied_settings

    def bind_evaluation_settings(self, settings):
        """Share committed gates/cohort; this page's interval remains local."""
        if self.evaluation_settings is settings:
            return
        if self.evaluation_settings is not None:
            self.evaluation_settings.changed.disconnect(self._sync_evaluation_settings)
        self.evaluation_settings = settings
        settings.changed.connect(self._sync_evaluation_settings)
        self._sync_evaluation_settings()

    def _sync_evaluation_settings(self):
        settings = self.evaluation_settings
        with (
            QSignalBlocker(self.range_gate),
            QSignalBlocker(self.velocity_gate),
            QSignalBlocker(self.outputs),
        ):
            self.range_gate.setValue(settings.range_gate_m)
            self.velocity_gate.setValue(settings.velocity_gate_m_s)
            self.outputs.setCurrentIndex(self.outputs.findData(settings.confirmed_only))
        values = (
            settings.range_gate_m,
            settings.velocity_gate_m_s,
            self._applied_settings[2],
            settings.confirmed_only,
        )
        if values != self._applied_settings:
            self._applied_settings = values
            self.settings_changed.emit()

    def _apply_settings(self):
        interval = (self.start.value(), self.end.value()) if self.interval.isChecked() else None
        self._applied_settings = (
            self.range_gate.value(),
            self.velocity_gate.value(),
            interval,
            bool(self.outputs.currentData()),
        )
        self.evaluation_settings.set_values(
            range_gate_m=self.range_gate.value(),
            velocity_gate_m_s=self.velocity_gate.value(),
            confirmed_only=bool(self.outputs.currentData()),
        )
        self.settings_changed.emit()

    def _clicked(self, plot, event):
        if (
            event.button() != Qt.MouseButton.LeftButton
            or event.modifiers() != Qt.KeyboardModifier.NoModifier
            or event.double()
            or not plot.getPlotItem().vb.sceneBoundingRect().contains(event.scenePos())
        ):
            return
        self.time_selected.emit(float(plot.getPlotItem().vb.mapSceneToView(event.scenePos()).x()))

    def set_loading(self, progress=None):
        if self._rendered is not None:
            self.clear()
        stage, completed, total = progress or ("Preparing summary", 0, 0)
        self.cards.setText(f"{stage} · {completed}/{total}" if total else stage + "…")
        self.coverage_note.setText(self.cards.text())

    def clear(self, reason="Summary not loaded"):
        self.data = self._rendered = None
        self.cards.setText(reason)
        self.coverage_note.setText(reason)
        for value in self.metric_values.values():
            value.setText("—")
        self.scope.setText("All retained sensing acquisitions")
        self._empty_keys.clear()
        for key, plot in self.plots.items():
            plot.clear()
            plot.showAxis("left")
            plot.showAxis("bottom")
            plot.enableAutoRange()
            plot.getAxis("left").setTickSpacing()
            plot.setTitle(
                FIGURE_HELP[key][0], color=current_theme().text_primary, size="10pt", bold=True
            )
            self._set_plot_note(key, "")
        self.tabs.setTabEnabled(2, False)

    def _set_plot_note(self, key, text):
        note = self.plot_notes[key]
        note.setText(text)
        note.setVisible(bool(text))

    @staticmethod
    def _number(value, suffix="", percent=False):
        if value is None or not np.isfinite(value):
            return "unavailable"
        return f"{value * 100:.1f}%" if percent else f"{value:.3g}{suffix}"

    def _empty_plot(self, key, message):
        plot = self.plots[key]
        self._empty_keys.add(key)
        plot.hideAxis("left")
        plot.hideAxis("bottom")
        plot.setXRange(0, 1, padding=0)
        plot.setYRange(0, 1, padding=0)
        label = pg.TextItem(message, color=current_theme().text_secondary, anchor=(0.5, 0.5))
        label.setFont(QFont("DejaVu Sans", 12))
        label.setPos(0.5, 0.5)
        plot.addItem(label, ignoreBounds=True)

    @staticmethod
    def _target_color(identity):
        # Keep the established identity assignment, with stronger light-theme contrast.
        palette = {
            "#56a9df": "#2865b0",
            "#ef8a47": "#bf5b13",
            "#70b85a": "#00866b",
            "#b883d5": "#7552a3",
            "#df6384": "#b5475a",
            "#42bcb1": "#197a78",
        }
        return palette[identity_color(identity, truth=True)]

    @staticmethod
    def _line(plot, x, y, name, color, *, scatter=False):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if not np.any(np.isfinite(x) & np.isfinite(y)):
            # An undefined statistic has no points. Keep its legend entry,
            # without sending an all-NaN series to the scatter bounds reader.
            x, y = np.empty(0), np.empty(0)
        # Hollow references and smaller filled estimates remain distinguishable
        # at identical coordinates; no jitter or fabricated samples are added.
        styles = {
            "TP": (COLORS["tp"], "o", 6, True, Qt.PenStyle.SolidLine),
            "FP": (COLORS["fp"], "x", 7, False, Qt.PenStyle.DashLine),
            "FN": (COLORS["fn"], "t", 10, False, Qt.PenStyle.DotLine),
            "Eligible GT": (COLORS["gt"], "o", 10, False, Qt.PenStyle.DashLine),
            "Saved outputs": (COLORS["output"], "d", 6, True, Qt.PenStyle.SolidLine),
            "Recall": ("#2865b0", "o", 10, False, Qt.PenStyle.DashLine),
            "Precision": ("#7552a3", "d", 5, True, Qt.PenStyle.SolidLine),
            "Confirmed": (COLORS["tp"], "o", 7, True, Qt.PenStyle.SolidLine),
            "Tentative": (COLORS["fn"], "t", 9, False, Qt.PenStyle.DashLine),
            "Matched pairs": ("#2865b0", "o", 5, True, Qt.PenStyle.SolidLine),
        }
        color, symbol, size, fill, pattern = styles.get(
            name, (color, "o", 10, False, Qt.PenStyle.SolidLine)
        )
        pen = pg.mkPen(color, width=1.9, style=pattern)
        item = plot.plot(
            x,
            y,
            name=name,
            pen=None if scatter else pen,
            symbol=symbol,
            symbolSize=size,
            symbolBrush=pg.mkBrush(color) if fill else pg.mkBrush(0, 0, 0, 0),
            symbolPen=pg.mkPen(color, width=1.7),
            connect="finite",
            antialias=True,
        )
        item._summary_color = color
        item._summary_symbol_size = size

    def set_summary(self, data):
        self._ensure_plots()
        if data is self._rendered:
            return
        self.clear()
        self.data = self._rendered = data
        if data.error or data.cancelled:
            self.cards.setText("Summary unavailable: " + (data.error or "Cancelled"))
            self.coverage_note.setText(self.cards.text())
            return
        summary = data.summary
        fmt = self._number
        context_text = (
            "Clock: "
            + str(data.context[0])
            + "\nProcessing run: "
            + str(data.context[1])
            + "\nSensor pair: "
            + str(data.context[2])
        )
        self.scope.setToolTip(context_text)
        self.cards.setText(
            f"TP {summary.get('total_tp', 0):,}    FP {summary.get('total_fp', 0):,}    FN {summary.get('total_fn', 0):,}    "
            f"Recall {fmt(summary.get('recall'), percent=True)}    Precision {fmt(summary.get('precision'), percent=True)}\n"
            f"Matched range RMSE {fmt(summary.get('cumulative_range_rmse'), ' m')}    "
            f"Matched radial RMSE {fmt(summary.get('cumulative_velocity_rmse'), ' m/s')}"
        )
        if summary.get("velocity_excluded_matched_pair_count", 0):
            self.cards.setText(
                self.cards.text()
                + f"\nVelocity pairs {summary['velocity_matched_pair_count']:,} / "
                f"{summary['matched_pair_count']:,}; range-only matches excluded from radial RMSE"
            )
        if not data.evaluated_count:
            self.cards.setText(
                "Scoring unavailable · no acquisitions have usable time-aligned truth and outputs"
            )
        gates = f"Applied gates ±{summary.get('range_gate_m', DEFAULT_RANGE_GATE_M):g} m / ±{summary.get('velocity_gate_m_s', DEFAULT_VELOCITY_GATE_M_S):g} m/s"
        interval = summary.get("interval")
        interval_text = (
            f" · interval [{interval[0]:g}, {interval[1]:g}] s"
            if interval is not None
            else " · full retained interval"
        )
        self.scope.setText(
            ("Confirmed tracks" if summary.get("confirmed_only") else "Saved outputs")
            + interval_text
            + "  ·  "
            + gates
        )
        values = {
            "tp": summary.get("total_tp", 0),
            "fp": summary.get("total_fp", 0),
            "fn": summary.get("total_fn", 0),
            "recall": summary.get("recall"),
            "precision": summary.get("precision"),
            "range": summary.get("cumulative_range_rmse"),
            "radial": summary.get("cumulative_velocity_rmse"),
        }
        for key, value in values.items():
            label = self.metric_values[key]
            unavailable = value is None or not data.evaluated_count
            label.setText(
                "—"
                if unavailable
                else (
                    f"{value:,}"
                    if key in {"tp", "fp", "fn"}
                    else fmt(value, percent=key in {"recall", "precision"})
                )
            )
            label.setToolTip(
                "Unavailable: no scored evidence or zero denominator"
                if unavailable
                else self.cards.text()
            )
            label.setAccessibleName(key + ": " + label.text())
        notes = [
            f"Scored {data.evaluated_count:,} / {data.acquisition_count:,} retained acquisitions",
            f"{summary.get('total_tp', 0) + summary.get('total_fn', 0):,} target opportunities",
            f"{summary.get('total_tp', 0):,} matched pairs",
        ]
        if summary.get("velocity_excluded_matched_pair_count", 0):
            notes.append(
                f"{summary['velocity_matched_pair_count']:,} matched velocity pairs; "
                f"{summary['velocity_excluded_matched_pair_count']:,} range-only matches and "
                f"{summary['velocity_unavailable_opportunity_count']:,} target opportunities "
                "have no eligible instantaneous velocity reference"
            )
        skipped = summary.get("skipped", {})
        if skipped:
            notes.append(
                "Excluded: " + "; ".join(f"{key} ({value})" for key, value in skipped.items())
            )
        notes.extend(data.uncertainty)
        if data.plot_sampled:
            notes.append(
                "Plot points are sampled; scalar counts and RMSE use every scored acquisition"
            )
        coverage = f"{data.evaluated_count:,} / {data.acquisition_count:,} acquisitions scored  ·  {summary.get('total_tp', 0) + summary.get('total_fn', 0):,} target opportunities  ·  {summary.get('total_tp', 0):,} matched pairs"
        if skipped:
            coverage += "  ·  Excluded: " + "; ".join(
                f"{reason} ({count})" for reason, count in skipped.items()
            )
        if data.plot_sampled:
            coverage += "  ·  Sampled plots; exact full-run scores"
        if any("share timestamps" in note for note in data.uncertainty):
            coverage += "  ·  Equal timestamps: saved inventory order"
        self.coverage_note.setText(coverage)
        self.coverage_note.setToolTip("\n".join(notes))
        time = data.time_s
        for name, values, color in (
            ("TP", data.event_tp, "#229979"),
            ("FP", data.event_fp, "#cf5267"),
            ("FN", data.event_fn, "#c78924"),
        ):
            self._line(self.plots["counts"], time, values, name, color)
        self.plots["counts"].setLabel("left", "Outputs / targets")
        self._line(self.plots["population"], time, data.event_gt_count, "Eligible GT", "#4388c9")
        self._line(
            self.plots["population"], time, data.event_output_count, "Saved outputs", "#ef8a47"
        )
        self.plots["population"].setLabel("left", "Count")
        self._line(
            self.plots["rates"], time, np.asarray(data.cumulative_recall) * 100, "Recall", "#229979"
        )
        self._line(
            self.plots["rates"],
            time,
            np.asarray(data.cumulative_precision) * 100,
            "Precision",
            "#4388c9",
        )
        self.plots["rates"].setLabel("left", "Cumulative", units="%")
        self.plots["rates"].setYRange(0, 100, padding=0.05)
        targets = summary.get("per_target", {})
        keys = list(targets)
        if not keys:
            self._set_plot_note("coverage", "Unavailable · no scored target opportunities")
        coverage = [targets[key].get("recall", np.nan) * 100 for key in keys]
        self.plots["coverage"].addItem(
            pg.BarGraphItem(
                x=np.arange(len(keys)), height=coverage, width=0.48, brush="#497fb9", pen=None
            )
        )
        self.plots["coverage"].getAxis("bottom").setTicks([list(enumerate(map(str, keys)))])
        self.plots["coverage"].setLabel("left", "Matched / eligible", units="%")
        self.plots["coverage"].setYRange(0, 115, padding=0)
        for index, key in enumerate(keys):
            row = targets[key]
            annotation = pg.TextItem(
                f"{coverage[index]:.1f}% · {row.get('matched_count', 0)}/{row.get('eligible_count', 0)}",
                color="#334e68",
                anchor=(0.5, 1),
            )
            annotation.setFont(QFont("DejaVu Sans", 9))
            annotation.setPos(index, coverage[index] + 3)
            self.plots["coverage"].addItem(annotation)
        self.plots["coverage"].setToolTip(
            FIGURE_HELP["coverage"][1]
            + "\n"
            + "\n".join(
                f"{key}: {targets[key].get('matched_count', 0)} / {targets[key].get('eligible_count', 0)}"
                for key in keys
            )
        )
        for key, gt_values, output_values, units in (
            ("range", data.gt_range_m, data.output_range_m, "m"),
            ("radial", data.gt_radial_m_s, data.output_radial_m_s, "m/s"),
        ):
            plot = self.plots[key]
            ids = np.asarray(data.gt_target_ids)
            for tid in dict.fromkeys(data.gt_target_ids):
                mask = ids == tid
                self._line(
                    plot,
                    np.asarray(data.gt_time_s)[mask],
                    np.asarray(gt_values)[mask],
                    f"GT {tid}",
                    self._target_color(tid),
                    scatter=True,
                )
            self._line(
                plot, data.output_time_s, output_values, "Saved outputs", "#ef8a47", scatter=True
            )
            plot.setLabel("left", "Range" if key == "range" else "Radial velocity", units=units)
        for key, errors, units in (
            ("range_error", data.range_errors_m, "m"),
            ("radial_error", data.radial_errors_m_s, "m/s"),
        ):
            values = np.sort(np.abs(np.asarray(errors, dtype=float)))
            if not len(values):
                self._set_plot_note(key, "Unavailable · no matched pairs for this cohort")
            self._line(
                self.plots[key],
                values,
                np.arange(1, len(values) + 1) / max(1, len(values)) * 100,
                "Matched pairs",
                "#229979",
            )
            self.plots[key].setLabel("bottom", "Absolute error", units=units)
            self.plots[key].setLabel("left", "Matched pairs ≤ error", units="%")
            self.plots[key].setYRange(0, 100, padding=0.05)
            if len(values):
                # Absolute-error axes include zero and never magnify floating-point
                # spread around an almost constant error into a misleading trend.
                upper = float(values[-1]) * 1.08 if values[-1] > 0 else 1.0
                self.plots[key].setXRange(0, upper, padding=0.02)
                if values[-1] == 0:
                    self._set_plot_note(key, "All matched errors are zero")
        has_tracking = bool(data.tracking_population_available or len(data.tracking_time_s))
        self.tabs.setTabEnabled(2, has_tracking)
        if has_tracking:
            if not data.tracking_population_available:
                self._set_plot_note(
                    "track_population", "Unavailable · confirmation states were not recorded"
                )
            self._line(
                self.plots["track_population"],
                data.time_s,
                data.confirmed_count,
                "Confirmed",
                "#229979",
            )
            self._line(
                self.plots["track_population"],
                data.time_s,
                data.tentative_count,
                "Tentative",
                "#c78924",
            )
            self.plots["track_population"].setLabel("left", "Saved outputs")
            ids = list(dict.fromkeys(data.tracking_track_ids))
            rows = {tid: index for index, tid in enumerate(ids)}
            tids = np.asarray(data.tracking_target_ids)
            for tid in dict.fromkeys(data.tracking_target_ids):
                mask = tids == tid
                y = [rows[x] for x in np.asarray(data.tracking_track_ids)[mask]]
                self._line(
                    self.plots["identity"],
                    np.asarray(data.tracking_time_s)[mask],
                    y,
                    f"GT {tid}",
                    self._target_color(tid),
                    scatter=True,
                )
            identity_axis = self.plots["identity"].getAxis("left")
            # These row positions label identities, rather than physical values.
            identity_axis.enableAutoSIPrefix(False)
            identity_axis.setTicks([[(index, str(tid)) for tid, index in rows.items()]])
            self.plots["identity"].setLabel("left", "Saved track ID")
            switches = summary.get("id_switches")
            self._set_plot_note(
                "identity", f"ID switches: {switches if switches is not None else 'unavailable'}"
            )
        for key in ("counts", "population"):
            series = (
                (data.event_tp, data.event_fp, data.event_fn)
                if key == "counts"
                else (data.event_gt_count, data.event_output_count)
            )
            finite = np.concatenate([np.asarray(values)[np.isfinite(values)] for values in series])
            if len(finite) and max(finite) <= 5:
                self.plots[key].getAxis("left").setTickSpacing(1, 1)
            self.plots[key].enableAutoRange()
        if not data.evaluated_count:
            for key in ("counts", "population", "rates"):
                self._set_plot_note(
                    key, "Scoring unavailable · no usable time-aligned truth and outputs"
                )
                self._empty_plot(key, "No scored acquisitions")
        if not keys:
            self._empty_plot("coverage", "No scored target opportunities")
        if not len(data.range_errors_m):
            self._empty_plot("range_error", "No matched pairs")
        if not len(data.radial_errors_m_s):
            self._empty_plot("radial_error", "No matched pairs")
        if has_tracking and not len(data.tracking_time_s):
            self._empty_plot("identity", "Identity evidence unavailable")
        self.apply_theme()
        self._size_plot_markers()
