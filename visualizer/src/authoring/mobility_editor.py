"""Qt editor for editable mobility models and preserved read-only imports."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TypeAlias
from uuid import UUID

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from shared.scenarios.actors import (
    MAX_RANDOM_SEED,
    ActorRole,
    CircularMobilitySpec,
    ConstantSpeedTraversalSpec,
    Figure8MobilitySpec,
    FitDurationTraversalSpec,
    GaussMarkovMobilitySpec,
    GridScanMobilitySpec,
    GroupMemberMobilitySpec,
    GroupOffsetSpec,
    LinearMobilitySpec,
    ManhattanGridMobilitySpec,
    MeshSequenceMobilitySpec,
    NetworkRouteMobilitySpec,
    OscillatingMobilitySpec,
    PendulumMobilitySpec,
    RandomSamplingMobilitySpec,
    RandomWaypointMobilitySpec,
    SpiralMobilitySpec,
    StationaryMobilitySpec,
    SurveyMobilitySpec,
    TraversalSpec,
    WaypointMobilitySpec,
)

from .mobility_models import (
    MOBILITY_MODELS,
    MobilityKind,
    MobilityModel,
    mobility_kind,
)
from .model_capabilities import mobility_capability

_NUMBER_LIMIT = 1_000_000_000.0
_ANGLE_LIMIT_DEGREES = 360_000.0
_MINIMUM_POSITIVE = 0.000_000_001
_MAXIMUM_PENDULUM_ANGLE = 179.999_999_999
_AXES = ("x", "y", "z")

Vector3: TypeAlias = tuple[float, float, float]
Vector2: TypeAlias = tuple[float, float]
Range2: TypeAlias = tuple[float, float]
VectorSpins: TypeAlias = tuple[QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox]
PairSpins: TypeAlias = tuple[QDoubleSpinBox, QDoubleSpinBox]
GroupChoice: TypeAlias = tuple[UUID, str]


def _configured_spin(
    parent: QWidget,
    object_name: str,
    *,
    minimum: float = -_NUMBER_LIMIT,
    maximum: float = _NUMBER_LIMIT,
    decimals: int = 9,
    suffix: str = "",
) -> QDoubleSpinBox:
    """Create a keyboard-editable numeric control with adaptive arrow steps."""

    spin = QDoubleSpinBox(parent)
    spin.setObjectName(object_name)
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    spin.setKeyboardTracking(False)
    spin.setAccelerated(True)
    spin.setSuffix(suffix)
    return spin


def _configured_int_spin(
    parent: QWidget,
    object_name: str,
    *,
    minimum: int = -2_147_483_648,
    maximum: int = 2_147_483_647,
) -> QSpinBox:
    spin = QSpinBox(parent)
    spin.setObjectName(object_name)
    spin.setRange(minimum, maximum)
    spin.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    spin.setKeyboardTracking(False)
    spin.setAccelerated(True)
    return spin


def _configured_seed_spin(parent: QWidget, object_name: str) -> QSpinBox:
    """Create a seed control whose complete range is accepted by the schema."""

    return _configured_int_spin(
        parent,
        object_name,
        minimum=0,
        maximum=MAX_RANDOM_SEED,
    )


def _vector_editor(
    parent: QWidget,
    prefix: str,
    *,
    dimensions: tuple[str, ...] = _AXES,
) -> tuple[QWidget, tuple[QDoubleSpinBox, ...]]:
    editor = QWidget(parent)
    editor.setObjectName(f"{prefix}Editor")
    layout = QHBoxLayout(editor)
    layout.setContentsMargins(0, 0, 0, 0)
    spins: list[QDoubleSpinBox] = []
    for dimension in dimensions:
        layout.addWidget(QLabel(dimension.upper(), editor))
        spin = _configured_spin(
            editor,
            f"{prefix}{dimension.upper()}Spin",
            suffix=" m",
        )
        layout.addWidget(spin, 1)
        spins.append(spin)
    return editor, tuple(spins)


def _range_editor(
    parent: QWidget,
    prefix: str,
    *,
    minimum: float = -_NUMBER_LIMIT,
    maximum: float = _NUMBER_LIMIT,
    suffix: str = "",
) -> tuple[QWidget, PairSpins]:
    editor = QWidget(parent)
    editor.setObjectName(f"{prefix}RangeEditor")
    layout = QHBoxLayout(editor)
    layout.setContentsMargins(0, 0, 0, 0)
    low = _configured_spin(
        editor,
        f"{prefix}MinSpin",
        minimum=minimum,
        maximum=maximum,
        suffix=suffix,
    )
    high = _configured_spin(
        editor,
        f"{prefix}MaxSpin",
        minimum=minimum,
        maximum=maximum,
        suffix=suffix,
    )
    layout.addWidget(QLabel("Min", editor))
    layout.addWidget(low, 1)
    layout.addWidget(QLabel("Max", editor))
    layout.addWidget(high, 1)
    return editor, (low, high)


def _spin_values(spins: Iterable[QDoubleSpinBox]) -> tuple[float, ...]:
    return tuple(float(spin.value()) for spin in spins)


def _set_spin_values(spins: Iterable[QDoubleSpinBox], values: Iterable[float]) -> None:
    for spin, value in zip(spins, values, strict=True):
        spin.setValue(float(value))


def _range_values(spins: PairSpins) -> Range2:
    return float(spins[0].value()), float(spins[1].value())


def _set_range_values(spins: PairSpins, values: Iterable[float]) -> None:
    _set_spin_values(spins, values)


def _set_combo_data(combo: QComboBox, value: object) -> None:
    index = combo.findData(value)
    if index < 0:
        raise ValueError(f"{combo.objectName()} has no value {value!r}")
    combo.setCurrentIndex(index)


def _optional_int_editor(
    parent: QWidget,
    prefix: str,
) -> tuple[QWidget, QCheckBox, QSpinBox]:
    editor = QWidget(parent)
    editor.setObjectName(f"{prefix}OptionalEditor")
    layout = QHBoxLayout(editor)
    layout.setContentsMargins(0, 0, 0, 0)
    enabled = QCheckBox("Set", editor)
    enabled.setObjectName(f"{prefix}EnabledCheck")
    spin = _configured_seed_spin(editor, f"{prefix}Spin")
    spin.setEnabled(False)
    enabled.toggled.connect(spin.setEnabled)
    layout.addWidget(enabled)
    layout.addWidget(spin, 1)
    return editor, enabled, spin


def _set_optional_int(enabled: QCheckBox, spin: QSpinBox, value: int | None) -> None:
    enabled.setChecked(value is not None)
    if value is not None:
        spin.setValue(int(value))


def _optional_int(enabled: QCheckBox, spin: QSpinBox) -> int | None:
    return int(spin.value()) if enabled.isChecked() else None


class _CollapsibleSection(QWidget):
    """A compact section whose content is hidden until requested."""

    def __init__(self, title: str, object_name: str, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.toggle = QToolButton(self)
        self.toggle.setObjectName(f"{object_name}Toggle")
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(False)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        layout.addWidget(self.toggle)
        self.content = QWidget(self)
        self.content.setObjectName(f"{object_name}Content")
        self.content.setHidden(True)
        layout.addWidget(self.content)
        self.toggle.toggled.connect(self.set_expanded)

    @property
    def expanded(self) -> bool:
        return bool(self.toggle.isChecked())

    def set_expanded(self, expanded: bool) -> None:
        normalized = bool(expanded)
        if self.toggle.isChecked() != normalized:
            self.toggle.setChecked(normalized)
            return
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if normalized else Qt.ArrowType.RightArrow)
        self.content.setVisible(normalized)


class _TraversalEditor(QWidget):
    """Edit a discriminated traversal value without constructing it eagerly."""

    def __init__(self, prefix: str, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName(f"{prefix}TraversalEditor")
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.type_combo = QComboBox(self)
        self.type_combo.setObjectName(f"{prefix}TraversalTypeCombo")
        self.type_combo.addItem("Fit full path to duration", "fit_duration")
        self.type_combo.addItem("Constant speed", "constant_speed")
        self.speed_spin = _configured_spin(
            self,
            f"{prefix}TraversalSpeedSpin",
            minimum=_MINIMUM_POSITIVE,
            maximum=_NUMBER_LIMIT,
            suffix=" m/s",
        )
        self.speed_spin.setValue(1.0)
        self.after_end_combo = QComboBox(self)
        self.after_end_combo.setObjectName(f"{prefix}TraversalAfterEndCombo")
        for label, value in (
            ("Hold", "hold"),
            ("Loop", "loop"),
            ("Ping pong", "ping_pong"),
        ):
            self.after_end_combo.addItem(label, value)
        form.addRow("Traversal", self.type_combo)
        form.addRow("Speed", self.speed_spin)
        form.addRow("After end", self.after_end_combo)
        self.type_combo.currentIndexChanged.connect(self._update_mode)
        self._update_mode()

    def _update_mode(self, _index: int = -1) -> None:
        constant = self.type_combo.currentData() == "constant_speed"
        self.speed_spin.setEnabled(constant)
        self.after_end_combo.setEnabled(constant)

    def set_traversal(self, traversal: TraversalSpec) -> None:
        _set_combo_data(self.type_combo, traversal.type)
        if isinstance(traversal, ConstantSpeedTraversalSpec):
            self.speed_spin.setValue(traversal.speed_mps)
            _set_combo_data(self.after_end_combo, traversal.after_end)
        self._update_mode()

    def traversal(self) -> TraversalSpec:
        if self.type_combo.currentData() == "fit_duration":
            return FitDurationTraversalSpec()
        return ConstantSpeedTraversalSpec(
            speed_mps=self.speed_spin.value(),
            after_end=str(self.after_end_combo.currentData()),
        )


def _form_page(
    parent: QWidget,
    object_name: str,
) -> tuple[QWidget, QVBoxLayout, QFormLayout]:
    page = QWidget(parent)
    page.setObjectName(object_name)
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    primary = QWidget(page)
    form = QFormLayout(primary)
    form.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(primary)
    return page, layout, form


def _add_combo(
    parent: QWidget,
    object_name: str,
    options: Iterable[tuple[str, str]],
) -> QComboBox:
    combo = QComboBox(parent)
    combo.setObjectName(object_name)
    for label, value in options:
        combo.addItem(label, value)
    return combo


class MobilityEditor(QWidget):
    """Edit one immutable mobility without owning document conversion policy."""

    mobility_type_changed = Signal(str)
    apply_requested = Signal()
    draw_waypoints_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mobilityEditor")
        self._editing_enabled = True
        self._model_editable = True
        self._preserved_mobility: MobilityModel | None = None
        self._read_only_type_token: str | None = None
        self._waypoint_drawing_active = False
        self._network_route_random_walk = False
        self._network_route_shortest_seed_state: tuple[bool, int] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        type_row = QWidget(self)
        type_layout = QFormLayout(type_row)
        type_layout.setContentsMargins(0, 0, 0, 0)
        self.type_combo = QComboBox(type_row)
        self.type_combo.setObjectName("mobilityTypeCombo")
        for kind, descriptor in MOBILITY_MODELS.items():
            self.type_combo.addItem(descriptor.label, kind.value)
        type_layout.addRow("Type", self.type_combo)
        layout.addWidget(type_row)

        self.page_stack = QStackedWidget(self)
        self.page_stack.setObjectName("mobilityPageStack")
        builders = {
            MobilityKind.STATIONARY: self._build_stationary_page,
            MobilityKind.LINEAR: self._build_linear_page,
            MobilityKind.WAYPOINT: self._build_waypoint_page,
            MobilityKind.CIRCULAR: self._build_circular_page,
            MobilityKind.SURVEY: self._build_survey_page,
            MobilityKind.GRID_SCAN: self._build_grid_scan_page,
            MobilityKind.OSCILLATING: self._build_oscillating_page,
            MobilityKind.PENDULUM: self._build_pendulum_page,
            MobilityKind.FIGURE8: self._build_figure8_page,
            MobilityKind.SPIRAL: self._build_spiral_page,
            MobilityKind.RANDOM_SAMPLING: self._build_random_sampling_page,
            MobilityKind.GAUSS_MARKOV: self._build_gauss_markov_page,
            MobilityKind.RANDOM_WAYPOINT: self._build_random_waypoint_page,
            MobilityKind.MANHATTAN_GRID: self._build_manhattan_grid_page,
            MobilityKind.NETWORK_ROUTE: self._build_network_route_page,
            MobilityKind.MESH_SEQUENCE: self._build_mesh_sequence_page,
            MobilityKind.GROUP_MEMBER: self._build_group_member_page,
        }
        self._pages = {kind.value: builders[kind]() for kind in MOBILITY_MODELS}
        for page in self._pages.values():
            self.page_stack.addWidget(page)
        self.read_only_page = QWidget(self.page_stack)
        self.read_only_page.setObjectName("mobilityReadOnlyPage")
        read_only_layout = QVBoxLayout(self.read_only_page)
        read_only_layout.setContentsMargins(0, 0, 0, 0)
        self.read_only_mobility_label = QLabel(self.read_only_page)
        self.read_only_mobility_label.setObjectName("mobilityReadOnlySummary")
        self.read_only_mobility_label.setWordWrap(True)
        read_only_layout.addWidget(self.read_only_mobility_label)
        read_only_layout.addStretch()
        self.page_stack.addWidget(self.read_only_page)
        layout.addWidget(self.page_stack)

        self.average_speed_label = QLabel("Average speed: —", self)
        self.average_speed_label.setObjectName("mobilityAverageSpeedLabel")
        layout.addWidget(self.average_speed_label)
        action_row = QHBoxLayout()
        action_row.addStretch()
        self.apply_button = QPushButton("Apply Mobility", self)
        self.apply_button.setObjectName("mobilityApplyButton")
        self.apply_button.clicked.connect(self.apply_requested.emit)
        action_row.addWidget(self.apply_button)
        layout.addLayout(action_row)

        self.type_combo.currentIndexChanged.connect(self._type_index_changed)
        self._show_type(MobilityKind.STATIONARY.value)

    def _advanced(
        self,
        page: QWidget,
        layout: QVBoxLayout,
        prefix: str,
    ) -> tuple[_CollapsibleSection, QFormLayout]:
        section = _CollapsibleSection(
            "Advanced",
            f"mobility{prefix}AdvancedSection",
            page,
        )
        form = QFormLayout(section.content)
        layout.addWidget(section)
        return section, form

    def _add_traversal(
        self,
        form: QFormLayout,
        prefix: str,
    ) -> _TraversalEditor:
        editor = _TraversalEditor(f"mobility{prefix}", form.parentWidget())
        form.addRow(editor)
        return editor

    def _build_stationary_page(self) -> QWidget:
        page, _layout, form = _form_page(self, "mobilityStationaryPage")
        widget, spins = _vector_editor(page, "stationaryPosition")
        self.stationary_position_spins = spins
        form.addRow("Position", widget)
        return page

    def _build_linear_page(self) -> QWidget:
        page, layout, form = _form_page(self, "mobilityLinearPage")
        start, self.linear_start_spins = _vector_editor(page, "linearStart")
        end, self.linear_end_spins = _vector_editor(page, "linearEnd")
        form.addRow("Start", start)
        form.addRow("End", end)
        self.linear_advanced_section, advanced = self._advanced(page, layout, "Linear")
        self.linear_traversal_editor = self._add_traversal(advanced, "Linear")
        self._expose_traversal("linear", self.linear_traversal_editor)
        return page

    def _build_waypoint_page(self) -> QWidget:
        page = QWidget(self)
        page.setObjectName("mobilityWaypointPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.waypoint_table = QTableWidget(0, 3, page)
        self.waypoint_table.setObjectName("mobilityWaypointTable")
        self.waypoint_table.setHorizontalHeaderLabels(("X", "Y", "Z"))
        self.waypoint_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.waypoint_table.verticalHeader().setVisible(False)
        self.waypoint_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.waypoint_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.waypoint_table.itemSelectionChanged.connect(self._update_waypoint_buttons)
        layout.addWidget(self.waypoint_table)
        controls = QHBoxLayout()
        self.waypoint_add_button = QPushButton("Add", page)
        self.waypoint_add_button.setObjectName("mobilityWaypointAddButton")
        self.waypoint_remove_button = QPushButton("Remove", page)
        self.waypoint_remove_button.setObjectName("mobilityWaypointRemoveButton")
        self.waypoint_up_button = QPushButton("Up", page)
        self.waypoint_up_button.setObjectName("mobilityWaypointUpButton")
        self.waypoint_down_button = QPushButton("Down", page)
        self.waypoint_down_button.setObjectName("mobilityWaypointDownButton")
        self.draw_waypoints_button = QPushButton("Draw Waypoints", page)
        self.draw_waypoints_button.setObjectName("mobilityDrawWaypointsButton")
        for button in (
            self.waypoint_add_button,
            self.waypoint_remove_button,
            self.waypoint_up_button,
            self.waypoint_down_button,
        ):
            controls.addWidget(button)
        controls.addStretch()
        controls.addWidget(self.draw_waypoints_button)
        layout.addLayout(controls)
        self.waypoint_advanced_section, advanced = self._advanced(page, layout, "Waypoint")
        self.waypoint_interpolation_combo = _add_combo(
            self.waypoint_advanced_section.content,
            "mobilityWaypointInterpolationCombo",
            (("Linear", "linear"), ("Catmull-Rom", "catmull_rom")),
        )
        advanced.addRow("Interpolation", self.waypoint_interpolation_combo)
        self.waypoint_traversal_editor = self._add_traversal(advanced, "Waypoint")
        self._expose_traversal("waypoint", self.waypoint_traversal_editor)
        self.waypoint_add_button.clicked.connect(self._add_waypoint_row)
        self.waypoint_remove_button.clicked.connect(self._remove_waypoint_row)
        self.waypoint_up_button.clicked.connect(lambda: self._move_waypoint_row(-1))
        self.waypoint_down_button.clicked.connect(lambda: self._move_waypoint_row(1))
        self.draw_waypoints_button.clicked.connect(self.draw_waypoints_requested.emit)
        self._set_waypoint_rows(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        return page

    def _build_circular_page(self) -> QWidget:
        page, layout, form = _form_page(self, "mobilityCircularPage")
        center, self.circular_center_spins = _vector_editor(page, "circularCenter")
        self.circular_radius_spin = _configured_spin(
            page,
            "circularRadiusSpin",
            minimum=_MINIMUM_POSITIVE,
            maximum=_NUMBER_LIMIT,
            suffix=" m",
        )
        self.circular_radius_spin.setValue(1.0)
        form.addRow("Center", center)
        form.addRow("Radius", self.circular_radius_spin)
        self.circular_advanced_section, advanced = self._advanced(page, layout, "Circular")
        self.circular_start_angle_spin = _configured_spin(
            self.circular_advanced_section.content,
            "circularStartAngleDegreesSpin",
            minimum=-_ANGLE_LIMIT_DEGREES,
            maximum=_ANGLE_LIMIT_DEGREES,
            suffix="°",
        )
        self.circular_clockwise_check = QCheckBox(
            "Clockwise",
            self.circular_advanced_section.content,
        )
        self.circular_clockwise_check.setObjectName("circularClockwiseCheck")
        self.circular_turns_spin = _configured_spin(
            self.circular_advanced_section.content,
            "circularTurnsSpin",
            minimum=_MINIMUM_POSITIVE,
            maximum=_NUMBER_LIMIT,
        )
        self.circular_turns_spin.setValue(1.0)
        advanced.addRow("Start angle", self.circular_start_angle_spin)
        advanced.addRow("Direction", self.circular_clockwise_check)
        advanced.addRow("Turns", self.circular_turns_spin)
        self.circular_traversal_editor = self._add_traversal(advanced, "Circular")
        self._expose_traversal("circular", self.circular_traversal_editor)
        return page

    def _build_survey_page(self) -> QWidget:
        page, layout, form = _form_page(self, "mobilitySurveyPage")
        origin, self.survey_origin_spins = _vector_editor(page, "surveyOrigin")
        self.survey_width_spin = self._positive_spin(page, "surveyWidthSpin")
        self.survey_height_spin = self._positive_spin(page, "surveyHeightSpin")
        self.survey_row_spacing_spin = self._positive_spin(page, "surveyRowSpacingSpin")
        form.addRow("Origin", origin)
        form.addRow("Width", self.survey_width_spin)
        form.addRow("Height", self.survey_height_spin)
        form.addRow("Row spacing", self.survey_row_spacing_spin)
        self.survey_advanced_section, advanced = self._advanced(page, layout, "Survey")
        self.survey_heading_spin = self._angle_spin(
            self.survey_advanced_section.content,
            "surveyHeadingSpin",
        )
        advanced.addRow("Heading", self.survey_heading_spin)
        self.survey_traversal_editor = self._add_traversal(advanced, "Survey")
        self._expose_traversal("survey", self.survey_traversal_editor)
        return page

    def _build_grid_scan_page(self) -> QWidget:
        page, layout, form = _form_page(self, "mobilityGridScanPage")
        x_widget, self.grid_scan_x_bounds_spins = _range_editor(
            page, "gridScanXBounds", suffix=" m"
        )
        y_widget, self.grid_scan_y_bounds_spins = _range_editor(
            page, "gridScanYBounds", suffix=" m"
        )
        z_widget, self.grid_scan_z_bounds_spins = _range_editor(
            page, "gridScanZBounds", suffix=" m"
        )
        self.grid_scan_x_steps_spin = _configured_int_spin(
            page, "gridScanXStepsSpin", minimum=1, maximum=1_000_000
        )
        self.grid_scan_y_steps_spin = _configured_int_spin(
            page, "gridScanYStepsSpin", minimum=1, maximum=1_000_000
        )
        self.grid_scan_z_steps_spin = _configured_int_spin(
            page, "gridScanZStepsSpin", minimum=1, maximum=1_000_000
        )
        for spin in (
            self.grid_scan_x_steps_spin,
            self.grid_scan_y_steps_spin,
            self.grid_scan_z_steps_spin,
        ):
            spin.setValue(1)
        form.addRow("X bounds", x_widget)
        form.addRow("Y bounds", y_widget)
        form.addRow("Z bounds", z_widget)
        form.addRow("X steps", self.grid_scan_x_steps_spin)
        form.addRow("Y steps", self.grid_scan_y_steps_spin)
        form.addRow("Z steps", self.grid_scan_z_steps_spin)
        self.grid_scan_advanced_section, advanced = self._advanced(page, layout, "GridScan")
        self.grid_scan_traversal_pattern_combo = _add_combo(
            self.grid_scan_advanced_section.content,
            "gridScanTraversalPatternCombo",
            (("Snake", "snake"), ("Raster", "raster")),
        )
        self.grid_scan_start_corner_combo = _add_combo(
            self.grid_scan_advanced_section.content,
            "gridScanStartCornerCombo",
            (
                ("Bottom left", "bottom_left"),
                ("Bottom right", "bottom_right"),
                ("Top left", "top_left"),
                ("Top right", "top_right"),
            ),
        )
        self.grid_scan_interpolation_combo = _add_combo(
            self.grid_scan_advanced_section.content,
            "gridScanInterpolationCombo",
            (("Linear", "linear"), ("Catmull-Rom", "catmull_rom")),
        )
        advanced.addRow("Pattern", self.grid_scan_traversal_pattern_combo)
        advanced.addRow("Start corner", self.grid_scan_start_corner_combo)
        advanced.addRow("Interpolation", self.grid_scan_interpolation_combo)
        self.grid_scan_traversal_editor = self._add_traversal(advanced, "GridScan")
        self._expose_traversal("grid_scan", self.grid_scan_traversal_editor)
        return page

    def _build_oscillating_page(self) -> QWidget:
        page, layout, form = _form_page(self, "mobilityOscillatingPage")
        center, self.oscillating_center_spins = _vector_editor(page, "oscillatingCenter")
        axis, self.oscillating_axis_spins = _vector_editor(page, "oscillatingAxis")
        _set_spin_values(self.oscillating_axis_spins, (1.0, 0.0, 0.0))
        self.oscillating_amplitude_spin = _configured_spin(
            page,
            "oscillatingAmplitudeSpin",
            minimum=0.0,
            maximum=_NUMBER_LIMIT,
            suffix=" m",
        )
        self.oscillating_frequency_spin = self._positive_spin(
            page, "oscillatingFrequencySpin", suffix=" Hz"
        )
        form.addRow("Center", center)
        form.addRow("Axis", axis)
        form.addRow("Amplitude", self.oscillating_amplitude_spin)
        form.addRow("Frequency", self.oscillating_frequency_spin)
        self.oscillating_advanced_section, advanced = self._advanced(page, layout, "Oscillating")
        self.oscillating_phase_spin = self._angle_spin(
            self.oscillating_advanced_section.content,
            "oscillatingPhaseSpin",
        )
        advanced.addRow("Phase", self.oscillating_phase_spin)
        return page

    def _build_pendulum_page(self) -> QWidget:
        page, layout, form = _form_page(self, "mobilityPendulumPage")
        pivot, self.pendulum_pivot_spins = _vector_editor(page, "pendulumPivot")
        self.pendulum_length_spin = self._positive_spin(page, "pendulumLengthSpin")
        self.pendulum_max_angle_spin = _configured_spin(
            page,
            "pendulumMaxAngleSpin",
            minimum=_MINIMUM_POSITIVE,
            maximum=_MAXIMUM_PENDULUM_ANGLE,
            suffix="°",
        )
        self.pendulum_max_angle_spin.setValue(30.0)
        self.pendulum_frequency_spin = self._positive_spin(
            page, "pendulumFrequencySpin", suffix=" Hz"
        )
        form.addRow("Pivot", pivot)
        form.addRow("Length", self.pendulum_length_spin)
        form.addRow("Maximum angle", self.pendulum_max_angle_spin)
        form.addRow("Frequency", self.pendulum_frequency_spin)
        self.pendulum_advanced_section, advanced = self._advanced(page, layout, "Pendulum")
        self.pendulum_plane_combo = _add_combo(
            self.pendulum_advanced_section.content,
            "pendulumPlaneCombo",
            (("XY", "xy"), ("XZ", "xz"), ("YZ", "yz")),
        )
        _set_combo_data(self.pendulum_plane_combo, "xz")
        self.pendulum_phase_spin = self._angle_spin(
            self.pendulum_advanced_section.content,
            "pendulumPhaseSpin",
        )
        advanced.addRow("Plane", self.pendulum_plane_combo)
        advanced.addRow("Phase", self.pendulum_phase_spin)
        return page

    def _build_figure8_page(self) -> QWidget:
        page, layout, form = _form_page(self, "mobilityFigure8Page")
        center, self.figure8_center_spins = _vector_editor(page, "figure8Center")
        self.figure8_size_spin = self._positive_spin(page, "figure8SizeSpin")
        form.addRow("Center", center)
        form.addRow("Size", self.figure8_size_spin)
        self.figure8_advanced_section, advanced = self._advanced(page, layout, "Figure8")
        self.figure8_plane_combo = _add_combo(
            self.figure8_advanced_section.content,
            "figure8PlaneCombo",
            (("XY", "xy"), ("XZ", "xz"), ("YZ", "yz")),
        )
        self.figure8_turns_spin = self._positive_spin(
            self.figure8_advanced_section.content,
            "figure8TurnsSpin",
            suffix="",
        )
        advanced.addRow("Plane", self.figure8_plane_combo)
        advanced.addRow("Turns", self.figure8_turns_spin)
        self.figure8_traversal_editor = self._add_traversal(advanced, "Figure8")
        self._expose_traversal("figure8", self.figure8_traversal_editor)
        return page

    def _build_spiral_page(self) -> QWidget:
        page, layout, form = _form_page(self, "mobilitySpiralPage")
        center, self.spiral_center_spins = _vector_editor(page, "spiralCenter")
        self.spiral_radius_spin = self._positive_spin(page, "spiralRadiusSpin")
        self.spiral_start_altitude_spin = _configured_spin(
            page, "spiralStartAltitudeSpin", suffix=" m"
        )
        self.spiral_end_altitude_spin = _configured_spin(page, "spiralEndAltitudeSpin", suffix=" m")
        self.spiral_turns_spin = self._positive_spin(page, "spiralTurnsSpin", suffix="")
        form.addRow("Center", center)
        form.addRow("Radius", self.spiral_radius_spin)
        form.addRow("Start altitude", self.spiral_start_altitude_spin)
        form.addRow("End altitude", self.spiral_end_altitude_spin)
        form.addRow("Turns", self.spiral_turns_spin)
        self.spiral_advanced_section, advanced = self._advanced(page, layout, "Spiral")
        self.spiral_start_angle_spin = self._angle_spin(
            self.spiral_advanced_section.content,
            "spiralStartAngleSpin",
        )
        self.spiral_clockwise_check = QCheckBox(
            "Clockwise",
            self.spiral_advanced_section.content,
        )
        self.spiral_clockwise_check.setObjectName("spiralClockwiseCheck")
        advanced.addRow("Start angle", self.spiral_start_angle_spin)
        advanced.addRow("Direction", self.spiral_clockwise_check)
        self.spiral_traversal_editor = self._add_traversal(advanced, "Spiral")
        self._expose_traversal("spiral", self.spiral_traversal_editor)
        return page

    def _build_random_sampling_page(self) -> QWidget:
        page, layout, form = _form_page(self, "mobilityRandomSamplingPage")
        initial_widget = QWidget(page)
        initial_widget.setObjectName("randomSamplingInitialPositionOptionalEditor")
        initial_layout = QHBoxLayout(initial_widget)
        initial_layout.setContentsMargins(0, 0, 0, 0)
        self.random_sampling_initial_position_enabled_check = QCheckBox(
            "Set",
            initial_widget,
        )
        self.random_sampling_initial_position_enabled_check.setObjectName(
            "randomSamplingInitialPositionEnabledCheck"
        )
        (
            self.random_sampling_initial_position_editor,
            self.random_sampling_initial_position_spins,
        ) = _vector_editor(initial_widget, "randomSamplingInitialPosition")
        initial_layout.addWidget(self.random_sampling_initial_position_enabled_check)
        initial_layout.addWidget(self.random_sampling_initial_position_editor, 1)
        self.random_sampling_initial_position_editor.setEnabled(False)
        self.random_sampling_initial_position_enabled_check.toggled.connect(
            self.random_sampling_initial_position_editor.setEnabled
        )
        x_widget, self.random_sampling_x_bounds_spins = _range_editor(
            page, "randomSamplingXBounds", suffix=" m"
        )
        y_widget, self.random_sampling_y_bounds_spins = _range_editor(
            page, "randomSamplingYBounds", suffix=" m"
        )
        z_widget, self.random_sampling_z_bounds_spins = _range_editor(
            page, "randomSamplingZBounds", suffix=" m"
        )
        self.random_sampling_seed_spin = _configured_seed_spin(page, "randomSamplingSeedSpin")
        self.random_sampling_sampling_combo = _add_combo(
            page,
            "randomSamplingSamplingCombo",
            (("Uniform", "uniform"), ("Poisson disk", "poisson_disk")),
        )
        self.random_sampling_explanation_label = QLabel(page)
        self.random_sampling_explanation_label.setObjectName("randomSamplingExplanationLabel")
        self.random_sampling_explanation_label.setWordWrap(True)
        form.addRow("Initial sample", initial_widget)
        form.addRow("X bounds", x_widget)
        form.addRow("Y bounds", y_widget)
        form.addRow("Z bounds", z_widget)
        form.addRow("Seed", self.random_sampling_seed_spin)
        form.addRow("Sampling", self.random_sampling_sampling_combo)
        layout.addWidget(self.random_sampling_explanation_label)
        self.random_sampling_advanced_section, advanced = self._advanced(
            page, layout, "RandomSampling"
        )
        self.random_sampling_min_distance_spin = self._positive_spin(
            self.random_sampling_advanced_section.content,
            "randomSamplingMinDistanceSpin",
        )
        self.random_sampling_min_distance_spin.setValue(1.0)
        self.random_sampling_min_distance_spin.setToolTip(
            "Required center-to-center spacing between every pair of sampled positions."
        )
        advanced.addRow("Minimum spacing", self.random_sampling_min_distance_spin)
        self.random_sampling_sampling_combo.currentIndexChanged.connect(
            self._update_random_sampling_mode
        )
        self._update_random_sampling_mode()
        return page

    def _build_gauss_markov_page(self) -> QWidget:
        page, layout, form = _form_page(self, "mobilityGaussMarkovPage")
        initial, self.gauss_markov_initial_position_spins = _vector_editor(
            page, "gaussMarkovInitialPosition"
        )
        x_widget, self.gauss_markov_x_bounds_spins = _range_editor(
            page, "gaussMarkovXBounds", suffix=" m"
        )
        y_widget, self.gauss_markov_y_bounds_spins = _range_editor(
            page, "gaussMarkovYBounds", suffix=" m"
        )
        z_widget, self.gauss_markov_z_bounds_spins = _range_editor(
            page, "gaussMarkovZBounds", suffix=" m"
        )
        form.addRow("Initial position", initial)
        form.addRow("X bounds", x_widget)
        form.addRow("Y bounds", y_widget)
        form.addRow("Z bounds", z_widget)
        self.gauss_markov_explanation_label = QLabel(
            "Correlated continuous motion. Memory α=0 follows the preferred motion "
            "with fresh noise; α near 1 keeps the previous speed and heading. "
            "Heading 0° is +X and 90° is +Y. Bounds clamp the resulting position.",
            page,
        )
        self.gauss_markov_explanation_label.setObjectName("gaussMarkovExplanationLabel")
        self.gauss_markov_explanation_label.setWordWrap(True)
        layout.addWidget(self.gauss_markov_explanation_label)
        self.gauss_markov_advanced_section, advanced = self._advanced(page, layout, "GaussMarkov")
        advanced_parent = self.gauss_markov_advanced_section.content
        self.gauss_markov_alpha_spin = _configured_spin(
            advanced_parent, "gaussMarkovAlphaSpin", minimum=0.0, maximum=1.0
        )
        self.gauss_markov_mean_speed_spin = _configured_spin(
            advanced_parent,
            "gaussMarkovMeanSpeedSpin",
            minimum=0.0,
            maximum=_NUMBER_LIMIT,
            suffix=" m/s",
        )
        self.gauss_markov_mean_direction_spin = self._angle_spin(
            advanced_parent, "gaussMarkovMeanDirectionSpin"
        )
        self.gauss_markov_speed_std_spin = _configured_spin(
            advanced_parent,
            "gaussMarkovSpeedStdSpin",
            minimum=0.0,
            maximum=_NUMBER_LIMIT,
            suffix=" m/s",
        )
        self.gauss_markov_direction_std_spin = _configured_spin(
            advanced_parent,
            "gaussMarkovDirectionStdSpin",
            minimum=0.0,
            maximum=_ANGLE_LIMIT_DEGREES,
            suffix="°",
        )
        self.gauss_markov_seed_spin = _configured_seed_spin(
            advanced_parent,
            "gaussMarkovSeedSpin",
        )
        self.gauss_markov_alpha_spin.setToolTip(
            "Motion memory: 0 adds fresh variation at each update; 1 is steady motion."
        )
        self.gauss_markov_mean_speed_spin.setToolTip(
            "The preferred speed that the correlated process returns toward."
        )
        self.gauss_markov_mean_direction_spin.setToolTip(
            "Preferred XY heading: 0° points along +X; 90° points along +Y."
        )
        self.gauss_markov_speed_std_spin.setToolTip(
            "Standard deviation of the random speed variation; 0 keeps speed deterministic."
        )
        self.gauss_markov_direction_std_spin.setToolTip(
            "Standard deviation of random heading variation; 0 keeps heading deterministic."
        )
        self.gauss_markov_seed_spin.setToolTip(
            "The same seed and parameters reproduce the same trajectory."
        )
        advanced.addRow("Memory α", self.gauss_markov_alpha_spin)
        advanced.addRow("Preferred speed", self.gauss_markov_mean_speed_spin)
        advanced.addRow("Preferred heading", self.gauss_markov_mean_direction_spin)
        advanced.addRow("Speed noise σ", self.gauss_markov_speed_std_spin)
        advanced.addRow("Heading noise σ", self.gauss_markov_direction_std_spin)
        advanced.addRow("Seed", self.gauss_markov_seed_spin)
        return page

    def _build_random_waypoint_page(self) -> QWidget:
        page, layout, form = _form_page(self, "mobilityRandomWaypointPage")
        initial, self.random_waypoint_initial_position_spins = _vector_editor(
            page, "randomWaypointInitialPosition"
        )
        x_widget, self.random_waypoint_x_bounds_spins = _range_editor(
            page, "randomWaypointXBounds", suffix=" m"
        )
        y_widget, self.random_waypoint_y_bounds_spins = _range_editor(
            page, "randomWaypointYBounds", suffix=" m"
        )
        z_widget, self.random_waypoint_z_bounds_spins = _range_editor(
            page, "randomWaypointZBounds", suffix=" m"
        )
        speed_widget, self.random_waypoint_speed_range_spins = _range_editor(
            page,
            "randomWaypointSpeed",
            minimum=_MINIMUM_POSITIVE,
            suffix=" m/s",
        )
        form.addRow("Initial position", initial)
        form.addRow("X bounds", x_widget)
        form.addRow("Y bounds", y_widget)
        form.addRow("Z bounds", z_widget)
        form.addRow("Speed range", speed_widget)
        self.random_waypoint_advanced_section, advanced = self._advanced(
            page, layout, "RandomWaypoint"
        )
        pause_widget, self.random_waypoint_pause_range_spins = _range_editor(
            self.random_waypoint_advanced_section.content,
            "randomWaypointPause",
            minimum=0.0,
            suffix=" s",
        )
        self.random_waypoint_seed_spin = _configured_seed_spin(
            self.random_waypoint_advanced_section.content,
            "randomWaypointSeedSpin",
        )
        advanced.addRow("Pause range", pause_widget)
        advanced.addRow("Seed", self.random_waypoint_seed_spin)
        return page

    def _build_manhattan_grid_page(self) -> QWidget:
        page, layout, form = _form_page(self, "mobilityManhattanGridPage")
        origin, self.manhattan_grid_origin_spins = _vector_editor(
            page, "manhattanGridOrigin", dimensions=("x", "y")
        )
        self.manhattan_grid_block_size_spin = self._positive_spin(
            page, "manhattanGridBlockSizeSpin"
        )
        self.manhattan_grid_width_spin = _configured_int_spin(
            page, "manhattanGridWidthSpin", minimum=1, maximum=1_000_000
        )
        self.manhattan_grid_height_spin = _configured_int_spin(
            page, "manhattanGridHeightSpin", minimum=1, maximum=1_000_000
        )
        self.manhattan_grid_altitude_spin = _configured_spin(
            page, "manhattanGridAltitudeSpin", suffix=" m"
        )
        speed_widget, self.manhattan_grid_speed_range_spins = _range_editor(
            page,
            "manhattanGridSpeed",
            minimum=_MINIMUM_POSITIVE,
            suffix=" m/s",
        )
        form.addRow("Origin", origin)
        form.addRow("Block size", self.manhattan_grid_block_size_spin)
        form.addRow("Grid width", self.manhattan_grid_width_spin)
        form.addRow("Grid height", self.manhattan_grid_height_spin)
        form.addRow("Altitude", self.manhattan_grid_altitude_spin)
        form.addRow("Speed range", speed_widget)
        self.manhattan_grid_advanced_section, advanced = self._advanced(
            page, layout, "ManhattanGrid"
        )
        self.manhattan_grid_turn_probability_spin = _configured_spin(
            self.manhattan_grid_advanced_section.content,
            "manhattanGridTurnProbabilitySpin",
            minimum=0.0,
            maximum=1.0,
        )
        pause_widget, self.manhattan_grid_pause_range_spins = _range_editor(
            self.manhattan_grid_advanced_section.content,
            "manhattanGridPause",
            minimum=0.0,
            suffix=" s",
        )
        self.manhattan_grid_seed_spin = _configured_seed_spin(
            self.manhattan_grid_advanced_section.content,
            "manhattanGridSeedSpin",
        )
        advanced.addRow("Turn probability", self.manhattan_grid_turn_probability_spin)
        advanced.addRow("Pause range", pause_widget)
        advanced.addRow("Seed", self.manhattan_grid_seed_spin)
        return page

    def _build_network_route_page(self) -> QWidget:
        page, layout, form = _form_page(self, "mobilityNetworkRoutePage")
        self.network_route_travel_mode_combo = _add_combo(
            page,
            "networkRouteTravelModeCombo",
            (
                ("Pedestrian", "pedestrian"),
                ("Bike", "bike"),
                ("Car", "car"),
                ("Drone", "drone"),
            ),
        )
        self.network_route_route_combo = _add_combo(
            page,
            "networkRouteRouteCombo",
            (("Shortest path", "shortest_path"), ("Random walk", "random_walk")),
        )
        self.network_route_altitude_spin = _configured_spin(
            page, "networkRouteAltitudeSpin", suffix=" m"
        )
        graph_widget, self.network_route_graph_path_edit, self.network_route_browse_button = (
            self._path_editor(
                page,
                "networkRouteGraphPath",
                "Browse…",
                self._browse_network_graph,
            )
        )
        form.addRow("Travel mode", self.network_route_travel_mode_combo)
        form.addRow("Route", self.network_route_route_combo)
        form.addRow("Altitude", self.network_route_altitude_spin)
        form.addRow("Network graph", graph_widget)
        self.network_route_advanced_section, advanced = self._advanced(page, layout, "NetworkRoute")
        seed_widget, self.network_route_seed_enabled_check, self.network_route_seed_spin = (
            _optional_int_editor(
                self.network_route_advanced_section.content,
                "networkRouteSeed",
            )
        )
        self.network_route_start_node_edit = QLineEdit(self.network_route_advanced_section.content)
        self.network_route_start_node_edit.setObjectName("networkRouteStartNodeEdit")
        self.network_route_end_node_edit = QLineEdit(self.network_route_advanced_section.content)
        self.network_route_end_node_edit.setObjectName("networkRouteEndNodeEdit")
        advanced.addRow("Seed", seed_widget)
        advanced.addRow("Start node", self.network_route_start_node_edit)
        advanced.addRow("End node", self.network_route_end_node_edit)
        self.network_route_traversal_editor = self._add_traversal(advanced, "NetworkRoute")
        self._expose_traversal("network_route", self.network_route_traversal_editor)
        self.network_route_route_combo.currentIndexChanged.connect(self._update_network_route_mode)
        self._update_network_route_mode()
        return page

    def _build_mesh_sequence_page(self) -> QWidget:
        page, layout, form = _form_page(self, "mobilityMeshSequencePage")
        (
            positions_widget,
            self.mesh_sequence_positions_path_edit,
            self.mesh_sequence_browse_button,
        ) = self._path_editor(
            page,
            "meshSequencePositionsPath",
            "Browse…",
            self._browse_position_sequence,
        )
        form.addRow("Positions", positions_widget)
        self.mesh_sequence_advanced_section, advanced = self._advanced(page, layout, "MeshSequence")
        self.mesh_sequence_position_key_edit = QLineEdit(
            self.mesh_sequence_advanced_section.content
        )
        self.mesh_sequence_position_key_edit.setObjectName("meshSequencePositionKeyEdit")
        self.mesh_sequence_position_key_edit.setText("positions")
        self.mesh_sequence_interpolation_combo = _add_combo(
            self.mesh_sequence_advanced_section.content,
            "meshSequenceInterpolationCombo",
            (("Linear", "linear"), ("Step", "step")),
        )
        advanced.addRow("Position key", self.mesh_sequence_position_key_edit)
        advanced.addRow("Interpolation", self.mesh_sequence_interpolation_combo)
        self.mesh_sequence_traversal_editor = self._add_traversal(advanced, "MeshSequence")
        self._expose_traversal("mesh_sequence", self.mesh_sequence_traversal_editor)
        return page

    def _build_group_member_page(self) -> QWidget:
        page, _layout, form = _form_page(self, "mobilityGroupMemberPage")
        explanation = QLabel(
            "Fixed formation offset in the group's local right, forward, and up frame.",
            page,
        )
        explanation.setWordWrap(True)
        form.addRow(explanation)
        self.group_member_group_combo = QComboBox(page)
        self.group_member_group_combo.setObjectName("groupMemberGroupCombo")
        self.group_member_group_combo.setPlaceholderText("Select a group")
        self.group_member_group_combo.setCurrentIndex(-1)
        right = _configured_spin(page, "groupMemberRightSpin", suffix=" m")
        forward = _configured_spin(page, "groupMemberForwardSpin", suffix=" m")
        up = _configured_spin(page, "groupMemberUpSpin", suffix=" m")
        self.group_member_offset_spins = (right, forward, up)
        self.group_member_right_spin = right
        self.group_member_forward_spin = forward
        self.group_member_up_spin = up
        form.addRow("Group", self.group_member_group_combo)
        form.addRow("Fixed right offset", right)
        form.addRow("Fixed forward offset", forward)
        form.addRow("Fixed up offset", up)
        return page

    def _positive_spin(
        self,
        parent: QWidget,
        object_name: str,
        *,
        suffix: str = " m",
    ) -> QDoubleSpinBox:
        spin = _configured_spin(
            parent,
            object_name,
            minimum=_MINIMUM_POSITIVE,
            maximum=_NUMBER_LIMIT,
            suffix=suffix,
        )
        spin.setValue(1.0)
        return spin

    def _angle_spin(self, parent: QWidget, object_name: str) -> QDoubleSpinBox:
        return _configured_spin(
            parent,
            object_name,
            minimum=-_ANGLE_LIMIT_DEGREES,
            maximum=_ANGLE_LIMIT_DEGREES,
            suffix="°",
        )

    def _path_editor(
        self,
        parent: QWidget,
        prefix: str,
        button_text: str,
        callback,
    ) -> tuple[QWidget, QLineEdit, QPushButton]:
        editor = QWidget(parent)
        editor.setObjectName(f"{prefix}Editor")
        layout = QHBoxLayout(editor)
        layout.setContentsMargins(0, 0, 0, 0)
        line_edit = QLineEdit(editor)
        line_edit.setObjectName(f"{prefix}Edit")
        button = QPushButton(button_text, editor)
        button.setObjectName(f"{prefix}BrowseButton")
        button.clicked.connect(callback)
        layout.addWidget(line_edit, 1)
        layout.addWidget(button)
        return editor, line_edit, button

    def _expose_traversal(self, prefix: str, editor: _TraversalEditor) -> None:
        setattr(self, f"{prefix}_traversal_type_combo", editor.type_combo)
        setattr(self, f"{prefix}_traversal_combo", editor.type_combo)
        setattr(self, f"{prefix}_traversal_speed_spin", editor.speed_spin)
        setattr(self, f"{prefix}_traversal_after_end_combo", editor.after_end_combo)

    def _type_index_changed(self, _index: int) -> None:
        token = str(self.type_combo.currentData())
        self._show_type(token)
        self.mobility_type_changed.emit(token)

    def _show_type(self, token: str) -> None:
        page = self._pages.get(str(token))
        if page is None:
            raise ValueError(f"unsupported mobility type: {token}")
        self.page_stack.setCurrentWidget(page)

    def _remove_read_only_type_item(self) -> None:
        token = self._read_only_type_token
        if token is None:
            return
        blocker = QSignalBlocker(self.type_combo)
        try:
            index = self.type_combo.findData(token)
            if index >= 0:
                self.type_combo.removeItem(index)
        finally:
            del blocker
        self._read_only_type_token = None

    def _show_read_only_mobility(self, mobility: MobilityModel) -> None:
        token = str(mobility.type)
        capability = mobility_capability(token)
        if capability.editable:
            raise TypeError(f"editable mobility type {token!r} has no editor page")

        self._remove_read_only_type_item()
        blocker = QSignalBlocker(self.type_combo)
        try:
            label = token.replace("_", " ").title()
            self.type_combo.addItem(f"{label} (read-only)", token)
            index = self.type_combo.count() - 1
            item = self.type_combo.model().item(index)
            if item is not None:
                item.setEnabled(False)
            self.type_combo.setCurrentIndex(index)
            self.page_stack.setCurrentWidget(self.read_only_page)
        finally:
            del blocker

        positions = getattr(mobility, "positions_m", ())
        detail = f"{len(positions):,} exact timeline positions" if positions else token
        self.read_only_mobility_label.setText(
            f"{detail}. This mobility is preserved unchanged and is not editable here."
        )
        self._read_only_type_token = token
        self._preserved_mobility = mobility
        self._model_editable = False
        self._update_editing_state()

    def _update_random_sampling_mode(self, _index: int = -1) -> None:
        poisson = self.random_sampling_sampling_combo.currentData() == "poisson_disk"
        self.random_sampling_min_distance_spin.setEnabled(poisson)
        if poisson:
            self.random_sampling_advanced_section.set_expanded(True)
            hint = (
                "Each frame is still an independent spatial observation. Poisson disk "
                "rejects candidates that are closer than Minimum spacing, so samples "
                "spread out instead of clustering."
            )
        else:
            hint = (
                "Each frame is an independent spatial observation, not continuous motion. "
                "Uniform gives every point in the bounds equal probability and allows clusters."
            )
        self.random_sampling_sampling_combo.setToolTip(hint)
        self.random_sampling_explanation_label.setText(hint)

    def _update_network_route_mode(self, _index: int = -1) -> None:
        random_walk = self.network_route_route_combo.currentData() == "random_walk"
        if random_walk and not self._network_route_random_walk:
            self._network_route_shortest_seed_state = (
                self.network_route_seed_enabled_check.isChecked(),
                self.network_route_seed_spin.value(),
            )
        elif (
            not random_walk
            and self._network_route_random_walk
            and self._network_route_shortest_seed_state is not None
        ):
            enabled, value = self._network_route_shortest_seed_state
            check_blocker = QSignalBlocker(self.network_route_seed_enabled_check)
            spin_blocker = QSignalBlocker(self.network_route_seed_spin)
            try:
                self.network_route_seed_enabled_check.setChecked(enabled)
                self.network_route_seed_spin.setValue(value)
            finally:
                del spin_blocker
                del check_blocker
            self._network_route_shortest_seed_state = None
        self._network_route_random_walk = random_walk
        self.network_route_start_node_edit.setEnabled(not random_walk)
        self.network_route_end_node_edit.setEnabled(not random_walk)
        if random_walk:
            self.network_route_seed_enabled_check.setChecked(True)
        self.network_route_seed_enabled_check.setEnabled(not random_walk)
        self.network_route_seed_spin.setEnabled(
            random_walk or self.network_route_seed_enabled_check.isChecked()
        )

    def _browse_network_graph(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select Network Graph",
            self.network_route_graph_path_edit.text(),
            "Network graphs (*.graphml *.xml *.json);;All files (*)",
        )
        if path:
            self.network_route_graph_path_edit.setText(path)

    def _browse_position_sequence(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select Position Sequence",
            self.mesh_sequence_positions_path_edit.text(),
            "Position sequences (*.npy *.npz *.csv *.h5 *.hdf5);;All files (*)",
        )
        if path:
            self.mesh_sequence_positions_path_edit.setText(path)

    def set_mobility(
        self,
        mobility: MobilityModel,
        *,
        preserve_type_signal: bool = False,
    ) -> None:
        """Populate the matching page, normally without emitting a type change."""

        if str(mobility.type) not in self._pages:
            self._show_read_only_mobility(mobility)
            return

        self._remove_read_only_type_item()
        self._preserved_mobility = None
        self._model_editable = True
        kind = mobility_kind(mobility)
        if isinstance(mobility, StationaryMobilitySpec):
            _set_spin_values(self.stationary_position_spins, mobility.position_m)
        elif isinstance(mobility, LinearMobilitySpec):
            _set_spin_values(self.linear_start_spins, mobility.start_m)
            _set_spin_values(self.linear_end_spins, mobility.end_m)
            self.linear_traversal_editor.set_traversal(mobility.traversal)
        elif isinstance(mobility, WaypointMobilitySpec):
            self._set_waypoint_rows(mobility.points_m)
            _set_combo_data(self.waypoint_interpolation_combo, mobility.interpolation)
            self.waypoint_traversal_editor.set_traversal(mobility.traversal)
        elif isinstance(mobility, CircularMobilitySpec):
            _set_spin_values(self.circular_center_spins, mobility.center_m)
            self.circular_radius_spin.setValue(mobility.radius_m)
            self.circular_start_angle_spin.setValue(mobility.start_angle_deg)
            self.circular_clockwise_check.setChecked(mobility.clockwise)
            self.circular_turns_spin.setValue(mobility.turns)
            self.circular_traversal_editor.set_traversal(mobility.traversal)
        elif isinstance(mobility, SurveyMobilitySpec):
            _set_spin_values(self.survey_origin_spins, mobility.origin_m)
            self.survey_width_spin.setValue(mobility.width_m)
            self.survey_height_spin.setValue(mobility.height_m)
            self.survey_row_spacing_spin.setValue(mobility.row_spacing_m)
            self.survey_heading_spin.setValue(mobility.heading_deg)
            self.survey_traversal_editor.set_traversal(mobility.traversal)
        elif isinstance(mobility, GridScanMobilitySpec):
            _set_range_values(self.grid_scan_x_bounds_spins, mobility.x_bounds_m)
            _set_range_values(self.grid_scan_y_bounds_spins, mobility.y_bounds_m)
            _set_range_values(self.grid_scan_z_bounds_spins, mobility.z_bounds_m)
            self.grid_scan_x_steps_spin.setValue(mobility.x_steps)
            self.grid_scan_y_steps_spin.setValue(mobility.y_steps)
            self.grid_scan_z_steps_spin.setValue(mobility.z_steps)
            _set_combo_data(
                self.grid_scan_traversal_pattern_combo,
                mobility.traversal_pattern,
            )
            _set_combo_data(self.grid_scan_start_corner_combo, mobility.start_corner)
            _set_combo_data(self.grid_scan_interpolation_combo, mobility.interpolation)
            self.grid_scan_traversal_editor.set_traversal(mobility.traversal)
        elif isinstance(mobility, OscillatingMobilitySpec):
            _set_spin_values(self.oscillating_center_spins, mobility.center_m)
            _set_spin_values(self.oscillating_axis_spins, mobility.axis)
            self.oscillating_amplitude_spin.setValue(mobility.amplitude_m)
            self.oscillating_frequency_spin.setValue(mobility.frequency_hz)
            self.oscillating_phase_spin.setValue(mobility.phase_deg)
        elif isinstance(mobility, PendulumMobilitySpec):
            _set_spin_values(self.pendulum_pivot_spins, mobility.pivot_m)
            self.pendulum_length_spin.setValue(mobility.length_m)
            self.pendulum_max_angle_spin.setValue(mobility.max_angle_deg)
            self.pendulum_frequency_spin.setValue(mobility.frequency_hz)
            _set_combo_data(self.pendulum_plane_combo, mobility.plane)
            self.pendulum_phase_spin.setValue(mobility.phase_deg)
        elif isinstance(mobility, Figure8MobilitySpec):
            _set_spin_values(self.figure8_center_spins, mobility.center_m)
            self.figure8_size_spin.setValue(mobility.size_m)
            _set_combo_data(self.figure8_plane_combo, mobility.plane)
            self.figure8_turns_spin.setValue(mobility.turns)
            self.figure8_traversal_editor.set_traversal(mobility.traversal)
        elif isinstance(mobility, SpiralMobilitySpec):
            _set_spin_values(self.spiral_center_spins, mobility.center_m)
            self.spiral_radius_spin.setValue(mobility.radius_m)
            self.spiral_start_altitude_spin.setValue(mobility.start_altitude_m)
            self.spiral_end_altitude_spin.setValue(mobility.end_altitude_m)
            self.spiral_turns_spin.setValue(mobility.turns)
            self.spiral_start_angle_spin.setValue(mobility.start_angle_deg)
            self.spiral_clockwise_check.setChecked(mobility.clockwise)
            self.spiral_traversal_editor.set_traversal(mobility.traversal)
        elif isinstance(mobility, RandomSamplingMobilitySpec):
            has_initial = mobility.initial_position_m is not None
            self.random_sampling_initial_position_enabled_check.setChecked(has_initial)
            if mobility.initial_position_m is not None:
                _set_spin_values(
                    self.random_sampling_initial_position_spins,
                    mobility.initial_position_m,
                )
            _set_range_values(self.random_sampling_x_bounds_spins, mobility.x_bounds_m)
            _set_range_values(self.random_sampling_y_bounds_spins, mobility.y_bounds_m)
            _set_range_values(self.random_sampling_z_bounds_spins, mobility.z_bounds_m)
            self.random_sampling_seed_spin.setValue(mobility.seed)
            _set_combo_data(self.random_sampling_sampling_combo, mobility.sampling)
            if mobility.min_distance_m is not None:
                self.random_sampling_min_distance_spin.setValue(mobility.min_distance_m)
            self._update_random_sampling_mode()
        elif isinstance(mobility, GaussMarkovMobilitySpec):
            _set_spin_values(
                self.gauss_markov_initial_position_spins,
                mobility.initial_position_m,
            )
            _set_range_values(self.gauss_markov_x_bounds_spins, mobility.x_bounds_m)
            _set_range_values(self.gauss_markov_y_bounds_spins, mobility.y_bounds_m)
            _set_range_values(self.gauss_markov_z_bounds_spins, mobility.z_bounds_m)
            self.gauss_markov_alpha_spin.setValue(mobility.alpha)
            self.gauss_markov_mean_speed_spin.setValue(mobility.mean_speed_mps)
            self.gauss_markov_mean_direction_spin.setValue(mobility.mean_direction_deg)
            self.gauss_markov_speed_std_spin.setValue(mobility.speed_std_mps)
            self.gauss_markov_direction_std_spin.setValue(mobility.direction_std_deg)
            self.gauss_markov_seed_spin.setValue(mobility.seed)
        elif isinstance(mobility, RandomWaypointMobilitySpec):
            _set_spin_values(
                self.random_waypoint_initial_position_spins,
                mobility.initial_position_m,
            )
            _set_range_values(self.random_waypoint_x_bounds_spins, mobility.x_bounds_m)
            _set_range_values(self.random_waypoint_y_bounds_spins, mobility.y_bounds_m)
            _set_range_values(self.random_waypoint_z_bounds_spins, mobility.z_bounds_m)
            _set_range_values(
                self.random_waypoint_speed_range_spins,
                mobility.speed_range_mps,
            )
            _set_range_values(
                self.random_waypoint_pause_range_spins,
                mobility.pause_range_s,
            )
            self.random_waypoint_seed_spin.setValue(mobility.seed)
        elif isinstance(mobility, ManhattanGridMobilitySpec):
            _set_spin_values(self.manhattan_grid_origin_spins, mobility.origin_xy_m)
            self.manhattan_grid_block_size_spin.setValue(mobility.block_size_m)
            self.manhattan_grid_width_spin.setValue(mobility.grid_width)
            self.manhattan_grid_height_spin.setValue(mobility.grid_height)
            self.manhattan_grid_altitude_spin.setValue(mobility.altitude_m)
            self.manhattan_grid_turn_probability_spin.setValue(mobility.turn_probability)
            _set_range_values(
                self.manhattan_grid_speed_range_spins,
                mobility.speed_range_mps,
            )
            _set_range_values(
                self.manhattan_grid_pause_range_spins,
                mobility.pause_range_s,
            )
            self.manhattan_grid_seed_spin.setValue(mobility.seed)
        elif isinstance(mobility, NetworkRouteMobilitySpec):
            _set_combo_data(self.network_route_travel_mode_combo, mobility.travel_mode)
            _set_combo_data(self.network_route_route_combo, mobility.route)
            self.network_route_altitude_spin.setValue(mobility.altitude_m)
            _set_optional_int(
                self.network_route_seed_enabled_check,
                self.network_route_seed_spin,
                mobility.seed,
            )
            self.network_route_graph_path_edit.setText(mobility.graph_path or "")
            self.network_route_start_node_edit.setText(
                "" if mobility.start_node is None else str(mobility.start_node)
            )
            self.network_route_end_node_edit.setText(
                "" if mobility.end_node is None else str(mobility.end_node)
            )
            self.network_route_traversal_editor.set_traversal(mobility.traversal)
            self._update_network_route_mode()
            self._network_route_shortest_seed_state = None
        elif isinstance(mobility, MeshSequenceMobilitySpec):
            self.mesh_sequence_positions_path_edit.setText(mobility.positions_path)
            self.mesh_sequence_position_key_edit.setText(mobility.position_key)
            _set_combo_data(
                self.mesh_sequence_interpolation_combo,
                mobility.interpolation,
            )
            self.mesh_sequence_traversal_editor.set_traversal(mobility.traversal)
        elif isinstance(mobility, GroupMemberMobilitySpec):
            self._select_group(UUID(mobility.group))
            _set_spin_values(
                self.group_member_offset_spins,
                (
                    mobility.offset_m.right,
                    mobility.offset_m.forward,
                    mobility.offset_m.up,
                ),
            )
        else:
            raise TypeError(f"unsupported mobility value: {type(mobility).__name__}")

        blocker = None if preserve_type_signal else QSignalBlocker(self.type_combo)
        try:
            _set_combo_data(self.type_combo, kind.value)
            self._show_type(kind.value)
        finally:
            del blocker
        self._update_editing_state()

    def mobility(self) -> MobilityModel:
        """Return the validated mobility represented by the active page."""

        if self._preserved_mobility is not None:
            return self._preserved_mobility
        kind = MobilityKind(str(self.type_combo.currentData()))
        if kind is MobilityKind.STATIONARY:
            return StationaryMobilitySpec(position_m=_spin_values(self.stationary_position_spins))
        if kind is MobilityKind.LINEAR:
            return LinearMobilitySpec(
                start_m=_spin_values(self.linear_start_spins),
                end_m=_spin_values(self.linear_end_spins),
                traversal=self.linear_traversal_editor.traversal(),
            )
        if kind is MobilityKind.WAYPOINT:
            return WaypointMobilitySpec(
                points_m=self._waypoint_rows(),
                interpolation=str(self.waypoint_interpolation_combo.currentData()),
                traversal=self.waypoint_traversal_editor.traversal(),
            )
        if kind is MobilityKind.CIRCULAR:
            return CircularMobilitySpec(
                center_m=_spin_values(self.circular_center_spins),
                radius_m=self.circular_radius_spin.value(),
                start_angle_deg=self.circular_start_angle_spin.value(),
                clockwise=self.circular_clockwise_check.isChecked(),
                turns=self.circular_turns_spin.value(),
                traversal=self.circular_traversal_editor.traversal(),
            )
        if kind is MobilityKind.SURVEY:
            return SurveyMobilitySpec(
                origin_m=_spin_values(self.survey_origin_spins),
                width_m=self.survey_width_spin.value(),
                height_m=self.survey_height_spin.value(),
                row_spacing_m=self.survey_row_spacing_spin.value(),
                heading_deg=self.survey_heading_spin.value(),
                traversal=self.survey_traversal_editor.traversal(),
            )
        if kind is MobilityKind.GRID_SCAN:
            return GridScanMobilitySpec(
                x_bounds_m=_range_values(self.grid_scan_x_bounds_spins),
                y_bounds_m=_range_values(self.grid_scan_y_bounds_spins),
                z_bounds_m=_range_values(self.grid_scan_z_bounds_spins),
                x_steps=self.grid_scan_x_steps_spin.value(),
                y_steps=self.grid_scan_y_steps_spin.value(),
                z_steps=self.grid_scan_z_steps_spin.value(),
                traversal_pattern=str(self.grid_scan_traversal_pattern_combo.currentData()),
                start_corner=str(self.grid_scan_start_corner_combo.currentData()),
                interpolation=str(self.grid_scan_interpolation_combo.currentData()),
                traversal=self.grid_scan_traversal_editor.traversal(),
            )
        if kind is MobilityKind.OSCILLATING:
            return OscillatingMobilitySpec(
                center_m=_spin_values(self.oscillating_center_spins),
                axis=_spin_values(self.oscillating_axis_spins),
                amplitude_m=self.oscillating_amplitude_spin.value(),
                frequency_hz=self.oscillating_frequency_spin.value(),
                phase_deg=self.oscillating_phase_spin.value(),
            )
        if kind is MobilityKind.PENDULUM:
            return PendulumMobilitySpec(
                pivot_m=_spin_values(self.pendulum_pivot_spins),
                length_m=self.pendulum_length_spin.value(),
                max_angle_deg=self.pendulum_max_angle_spin.value(),
                frequency_hz=self.pendulum_frequency_spin.value(),
                plane=str(self.pendulum_plane_combo.currentData()),
                phase_deg=self.pendulum_phase_spin.value(),
            )
        if kind is MobilityKind.FIGURE8:
            return Figure8MobilitySpec(
                center_m=_spin_values(self.figure8_center_spins),
                size_m=self.figure8_size_spin.value(),
                plane=str(self.figure8_plane_combo.currentData()),
                turns=self.figure8_turns_spin.value(),
                traversal=self.figure8_traversal_editor.traversal(),
            )
        if kind is MobilityKind.SPIRAL:
            return SpiralMobilitySpec(
                center_m=_spin_values(self.spiral_center_spins),
                radius_m=self.spiral_radius_spin.value(),
                start_altitude_m=self.spiral_start_altitude_spin.value(),
                end_altitude_m=self.spiral_end_altitude_spin.value(),
                turns=self.spiral_turns_spin.value(),
                start_angle_deg=self.spiral_start_angle_spin.value(),
                clockwise=self.spiral_clockwise_check.isChecked(),
                traversal=self.spiral_traversal_editor.traversal(),
            )
        if kind is MobilityKind.RANDOM_SAMPLING:
            poisson = self.random_sampling_sampling_combo.currentData() == "poisson_disk"
            return RandomSamplingMobilitySpec(
                x_bounds_m=_range_values(self.random_sampling_x_bounds_spins),
                y_bounds_m=_range_values(self.random_sampling_y_bounds_spins),
                z_bounds_m=_range_values(self.random_sampling_z_bounds_spins),
                initial_position_m=(
                    _spin_values(self.random_sampling_initial_position_spins)
                    if self.random_sampling_initial_position_enabled_check.isChecked()
                    else None
                ),
                seed=self.random_sampling_seed_spin.value(),
                sampling=str(self.random_sampling_sampling_combo.currentData()),
                min_distance_m=(
                    self.random_sampling_min_distance_spin.value() if poisson else None
                ),
            )
        if kind is MobilityKind.GAUSS_MARKOV:
            return GaussMarkovMobilitySpec(
                initial_position_m=_spin_values(self.gauss_markov_initial_position_spins),
                x_bounds_m=_range_values(self.gauss_markov_x_bounds_spins),
                y_bounds_m=_range_values(self.gauss_markov_y_bounds_spins),
                z_bounds_m=_range_values(self.gauss_markov_z_bounds_spins),
                alpha=self.gauss_markov_alpha_spin.value(),
                mean_speed_mps=self.gauss_markov_mean_speed_spin.value(),
                mean_direction_deg=self.gauss_markov_mean_direction_spin.value(),
                speed_std_mps=self.gauss_markov_speed_std_spin.value(),
                direction_std_deg=self.gauss_markov_direction_std_spin.value(),
                seed=self.gauss_markov_seed_spin.value(),
            )
        if kind is MobilityKind.RANDOM_WAYPOINT:
            return RandomWaypointMobilitySpec(
                initial_position_m=_spin_values(self.random_waypoint_initial_position_spins),
                x_bounds_m=_range_values(self.random_waypoint_x_bounds_spins),
                y_bounds_m=_range_values(self.random_waypoint_y_bounds_spins),
                z_bounds_m=_range_values(self.random_waypoint_z_bounds_spins),
                speed_range_mps=_range_values(self.random_waypoint_speed_range_spins),
                pause_range_s=_range_values(self.random_waypoint_pause_range_spins),
                seed=self.random_waypoint_seed_spin.value(),
            )
        if kind is MobilityKind.MANHATTAN_GRID:
            return ManhattanGridMobilitySpec(
                origin_xy_m=_spin_values(self.manhattan_grid_origin_spins),
                block_size_m=self.manhattan_grid_block_size_spin.value(),
                grid_width=self.manhattan_grid_width_spin.value(),
                grid_height=self.manhattan_grid_height_spin.value(),
                altitude_m=self.manhattan_grid_altitude_spin.value(),
                turn_probability=self.manhattan_grid_turn_probability_spin.value(),
                speed_range_mps=_range_values(self.manhattan_grid_speed_range_spins),
                pause_range_s=_range_values(self.manhattan_grid_pause_range_spins),
                seed=self.manhattan_grid_seed_spin.value(),
            )
        if kind is MobilityKind.NETWORK_ROUTE:
            random_walk = self.network_route_route_combo.currentData() == "random_walk"
            return NetworkRouteMobilitySpec(
                travel_mode=str(self.network_route_travel_mode_combo.currentData()),
                route="random_walk" if random_walk else "shortest_path",
                altitude_m=self.network_route_altitude_spin.value(),
                seed=(
                    int(self.network_route_seed_spin.value())
                    if random_walk
                    else _optional_int(
                        self.network_route_seed_enabled_check,
                        self.network_route_seed_spin,
                    )
                ),
                graph_path=self.network_route_graph_path_edit.text().strip() or None,
                start_node=(
                    None
                    if random_walk
                    else self._node_value(self.network_route_start_node_edit.text())
                ),
                end_node=(
                    None
                    if random_walk
                    else self._node_value(self.network_route_end_node_edit.text())
                ),
                traversal=self.network_route_traversal_editor.traversal(),
            )
        if kind is MobilityKind.MESH_SEQUENCE:
            return MeshSequenceMobilitySpec(
                positions_path=self.mesh_sequence_positions_path_edit.text().strip(),
                position_key=self.mesh_sequence_position_key_edit.text().strip(),
                interpolation=str(self.mesh_sequence_interpolation_combo.currentData()),
                traversal=self.mesh_sequence_traversal_editor.traversal(),
            )
        selected_group = self.group_member_group_combo.currentData()
        if selected_group is None:
            raise ValueError("group-member mobility requires a group")
        right, forward, up = _spin_values(self.group_member_offset_spins)
        return GroupMemberMobilitySpec(
            group=str(UUID(str(selected_group))),
            offset_m=GroupOffsetSpec(right=right, forward=forward, up=up),
        )

    def get_mobility(self) -> MobilityModel:
        """Return the mobility represented by the active editor page."""

        return self.mobility()

    @staticmethod
    def _node_value(text: str) -> str | int | None:
        normalized = text.strip()
        if not normalized:
            return None
        try:
            integer = int(normalized)
        except ValueError:
            return normalized
        return integer if str(integer) == normalized else normalized

    def set_group_choices(
        self,
        choices: Iterable[GroupChoice],
        *,
        preserve_selection: bool = True,
    ) -> None:
        """Replace UUID-backed group labels and optionally retain selection."""

        selected = self.group_member_group_combo.currentData()
        selected_id = UUID(str(selected)) if preserve_selection and selected is not None else None
        normalized = tuple((UUID(str(group_id)), str(label)) for group_id, label in choices)
        blocker = QSignalBlocker(self.group_member_group_combo)
        try:
            self.group_member_group_combo.clear()
            for group_id, label in normalized:
                self.group_member_group_combo.addItem(label, group_id)
            if selected_id is not None:
                self._select_group(selected_id)
            else:
                self.group_member_group_combo.setCurrentIndex(-1)
        finally:
            del blocker

    def _select_group(self, group_id: UUID) -> None:
        normalized = UUID(str(group_id))
        index = next(
            (
                candidate
                for candidate in range(self.group_member_group_combo.count())
                if UUID(str(self.group_member_group_combo.itemData(candidate))) == normalized
            ),
            -1,
        )
        if index < 0:
            self.group_member_group_combo.addItem(
                f"Unavailable group ({normalized})",
                normalized,
            )
            index = self.group_member_group_combo.count() - 1
        self.group_member_group_combo.setCurrentIndex(index)

    def set_actor_role(self, role: ActorRole | str | None) -> None:
        """Surface target-only restrictions without hiding canonical models."""

        normalized = None if role is None else ActorRole(role)
        descriptor = MOBILITY_MODELS[MobilityKind.MESH_SEQUENCE]
        index = self.type_combo.findData(MobilityKind.MESH_SEQUENCE.value)
        item = self.type_combo.model().item(index)
        if item is not None:
            item.setEnabled(normalized is None or descriptor.allows_role(normalized))
            item.setToolTip(
                "" if normalized in (None, ActorRole.TARGET) else "Mesh sequence is target-only."
            )

    def set_editing_enabled(self, enabled: bool) -> None:
        """Enable or disable every value-changing control."""

        self._editing_enabled = bool(enabled)
        self._update_editing_state()

    def set_read_only(self, read_only: bool) -> None:
        """Switch between editable and read-only presentation."""

        self.set_editing_enabled(not read_only)

    def set_waypoint_drawing_active(self, active: bool) -> None:
        """Reserve mobility edits for the viewport drawing transaction."""

        self._waypoint_drawing_active = bool(active)
        self.draw_waypoints_button.setText(
            "Drawing…" if self._waypoint_drawing_active else "Draw Waypoints"
        )
        hint = (
            "Finish or cancel the waypoint drawing session first."
            if self._waypoint_drawing_active
            else ""
        )
        self.apply_button.setToolTip(hint)
        self.draw_waypoints_button.setToolTip(
            hint or "Replace the waypoint list by clicking an ordered path in the viewport."
        )
        self._update_editing_state()

    def _update_editing_state(self) -> None:
        enabled = (
            self._editing_enabled and self._model_editable and not self._waypoint_drawing_active
        )
        self.type_combo.setEnabled(enabled)
        self.page_stack.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)
        self._update_waypoint_buttons()

    def set_average_speed(self, speed_mps: float | None) -> None:
        """Display the externally prepared average speed as read-only text."""

        speed = None if speed_mps is None else float(speed_mps)
        if speed is None or not math.isfinite(speed):
            self.average_speed_label.setText("Average speed: —")
        else:
            self.average_speed_label.setText(f"Average speed: {speed:.3f} m/s (computed)")

    def _waypoint_spin(self, row: int, column: int) -> QDoubleSpinBox:
        widget = self.waypoint_table.cellWidget(row, column)
        if not isinstance(widget, QDoubleSpinBox):
            raise RuntimeError(f"waypoint cell {row},{column} has no coordinate editor")
        return widget

    def _waypoint_rows(self) -> tuple[Vector3, ...]:
        return tuple(
            (
                float(self._waypoint_spin(row, 0).value()),
                float(self._waypoint_spin(row, 1).value()),
                float(self._waypoint_spin(row, 2).value()),
            )
            for row in range(self.waypoint_table.rowCount())
        )

    def _set_waypoint_rows(
        self,
        points: Iterable[Vector3],
        *,
        selected_row: int | None = None,
    ) -> None:
        frozen = tuple(tuple(float(value) for value in point) for point in points)
        self.waypoint_table.clearContents()
        self.waypoint_table.setRowCount(len(frozen))
        for row, point in enumerate(frozen):
            for column, (axis, value) in enumerate(zip(_AXES, point, strict=True)):
                spin = _configured_spin(
                    self.waypoint_table,
                    f"waypointPoint{row}{axis.upper()}Spin",
                    suffix=" m",
                )
                spin.setValue(value)
                self.waypoint_table.setCellWidget(row, column, spin)
        if selected_row is not None and 0 <= selected_row < len(frozen):
            self.waypoint_table.selectRow(selected_row)
        else:
            self.waypoint_table.clearSelection()
        self._update_waypoint_buttons()

    def _add_waypoint_row(self) -> None:
        points = list(self._waypoint_rows())
        current = self.waypoint_table.currentRow()
        seed = points[current] if 0 <= current < len(points) else (0.0, 0.0, 0.0)
        insert_at = current + 1 if 0 <= current < len(points) else len(points)
        points.insert(insert_at, seed)
        self._set_waypoint_rows(points, selected_row=insert_at)

    def _remove_waypoint_row(self) -> None:
        current = self.waypoint_table.currentRow()
        points = list(self._waypoint_rows())
        if current < 0 or current >= len(points) or len(points) <= 2:
            return
        points.pop(current)
        self._set_waypoint_rows(
            points,
            selected_row=min(current, len(points) - 1),
        )

    def _move_waypoint_row(self, offset: int) -> None:
        current = self.waypoint_table.currentRow()
        destination = current + int(offset)
        points = list(self._waypoint_rows())
        if current < 0 or current >= len(points) or destination < 0 or destination >= len(points):
            return
        point = points.pop(current)
        points.insert(destination, point)
        self._set_waypoint_rows(points, selected_row=destination)

    def _update_waypoint_buttons(self) -> None:
        current = self.waypoint_table.currentRow()
        count = self.waypoint_table.rowCount()
        enabled = self._editing_enabled and not self._waypoint_drawing_active
        selected = enabled and 0 <= current < count
        self.waypoint_add_button.setEnabled(enabled)
        self.waypoint_remove_button.setEnabled(selected and count > 2)
        self.waypoint_up_button.setEnabled(selected and current > 0)
        self.waypoint_down_button.setEnabled(selected and current < count - 1)
        self.draw_waypoints_button.setEnabled(enabled)


__all__ = ["GroupChoice", "MobilityEditor"]
