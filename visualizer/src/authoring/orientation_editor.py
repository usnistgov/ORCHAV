"""Typed Qt editor for every shared orientation model and field."""

from __future__ import annotations

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
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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
    AlignMotionOrientationSpec,
    FixedOrientationSpec,
    KeyframesOrientationSpec,
    LookAtOrientationSpec,
    OrientationKeyframeSpec,
    RandomOrientationSpec,
    SpinOrientationSpec,
)

from .orientation_models import (
    ORIENTATION_MODELS,
    OrientationKind,
    OrientationModel,
    actor_look_at_orientation,
    look_at_actor_id,
    orientation_kind,
    point_look_at_orientation,
)

_ANGLE_LIMIT_DEGREES = 360_000.0
_NUMBER_LIMIT = 1_000_000_000.0
_TIME_LIMIT_SECONDS = 1_000_000_000.0
_ANGLE_NAMES = ("yaw", "pitch", "roll")
_VECTOR_NAMES = ("x", "y", "z")
_LOOK_AT_ACTOR = "actor"
_LOOK_AT_POINT = "point"

Vector3: TypeAlias = tuple[float, float, float]
Range2: TypeAlias = tuple[float, float]
TripleSpins: TypeAlias = tuple[QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox]
RangeSpins: TypeAlias = tuple[QDoubleSpinBox, QDoubleSpinBox]
LookAtChoice: TypeAlias = tuple[UUID, str]


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


def _configured_int_spin(parent: QWidget, object_name: str) -> QSpinBox:
    spin = QSpinBox(parent)
    spin.setObjectName(object_name)
    spin.setRange(0, MAX_RANDOM_SEED)
    spin.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
    spin.setKeyboardTracking(False)
    spin.setAccelerated(True)
    return spin


def _angle_spins(parent: QWidget, prefix: str) -> TripleSpins:
    spins = tuple(
        _configured_spin(
            parent,
            f"{prefix}{name.title()}Spin",
            minimum=-_ANGLE_LIMIT_DEGREES,
            maximum=_ANGLE_LIMIT_DEGREES,
            suffix="°",
        )
        for name in _ANGLE_NAMES
    )
    return spins[0], spins[1], spins[2]


def _vector_spins(parent: QWidget, prefix: str) -> TripleSpins:
    spins = tuple(
        _configured_spin(parent, f"{prefix}{name.upper()}Spin", suffix=" m")
        for name in _VECTOR_NAMES
    )
    return spins[0], spins[1], spins[2]


def _triple_values(spins: TripleSpins) -> Vector3:
    return tuple(float(spin.value()) for spin in spins)  # type: ignore[return-value]


def _set_triple_values(spins: TripleSpins, values: Iterable[float]) -> None:
    for spin, value in zip(spins, values, strict=True):
        spin.setValue(float(value))


def _range_widget(
    parent: QWidget,
    prefix: str,
    *,
    suffix: str = "°",
) -> tuple[QWidget, RangeSpins]:
    widget = QWidget(parent)
    widget.setObjectName(f"{prefix}Range")
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    minimum = _configured_spin(
        widget,
        f"{prefix}MinSpin",
        minimum=-_ANGLE_LIMIT_DEGREES,
        maximum=_ANGLE_LIMIT_DEGREES,
        suffix=suffix,
    )
    maximum = _configured_spin(
        widget,
        f"{prefix}MaxSpin",
        minimum=-_ANGLE_LIMIT_DEGREES,
        maximum=_ANGLE_LIMIT_DEGREES,
        suffix=suffix,
    )
    layout.addWidget(QLabel("Min", widget))
    layout.addWidget(minimum)
    layout.addWidget(QLabel("Max", widget))
    layout.addWidget(maximum)
    return widget, (minimum, maximum)


def _range_values(spins: RangeSpins) -> Range2:
    return float(spins[0].value()), float(spins[1].value())


def _set_range_values(spins: RangeSpins, values: Iterable[float]) -> None:
    minimum, maximum = values
    spins[0].setValue(float(minimum))
    spins[1].setValue(float(maximum))


def _optional_spin_widget(
    parent: QWidget,
    prefix: str,
    label: str,
    *,
    suffix: str = "",
) -> tuple[QWidget, QCheckBox, QDoubleSpinBox]:
    widget = QWidget(parent)
    widget.setObjectName(f"{prefix}Optional")
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    enabled = QCheckBox(label, widget)
    enabled.setObjectName(f"{prefix}EnabledCheck")
    value = _configured_spin(
        widget,
        f"{prefix}Spin",
        minimum=0.000_000_001,
        maximum=_NUMBER_LIMIT,
        suffix=suffix,
    )
    value.setEnabled(False)
    enabled.toggled.connect(value.setEnabled)
    layout.addWidget(enabled)
    layout.addWidget(value)
    return widget, enabled, value


def _set_optional_value(
    enabled: QCheckBox,
    spin: QDoubleSpinBox,
    value: float | None,
) -> None:
    enabled.setChecked(value is not None)
    if value is not None:
        spin.setValue(float(value))


def _optional_value(enabled: QCheckBox, spin: QDoubleSpinBox) -> float | None:
    return float(spin.value()) if enabled.isChecked() else None


def _optional_range_widget(
    parent: QWidget,
    prefix: str,
    label: str,
) -> tuple[QWidget, QCheckBox, RangeSpins]:
    widget = QWidget(parent)
    widget.setObjectName(f"{prefix}Optional")
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    enabled = QCheckBox(label, widget)
    enabled.setObjectName(f"{prefix}EnabledCheck")
    range_widget, spins = _range_widget(widget, prefix)
    range_widget.setEnabled(False)
    enabled.toggled.connect(range_widget.setEnabled)
    layout.addWidget(enabled)
    layout.addWidget(range_widget)
    return widget, enabled, spins


def _set_optional_range(
    enabled: QCheckBox,
    spins: RangeSpins,
    value: Range2 | None,
) -> None:
    enabled.setChecked(value is not None)
    if value is not None:
        _set_range_values(spins, value)


def _optional_range(enabled: QCheckBox, spins: RangeSpins) -> Range2 | None:
    return _range_values(spins) if enabled.isChecked() else None


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


class OrientationEditor(QWidget):
    """Edit one immutable orientation without owning document conversion."""

    orientation_type_changed = Signal(str)
    apply_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("orientationEditor")
        self._editing_enabled = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        type_row = QWidget(self)
        type_layout = QFormLayout(type_row)
        type_layout.setContentsMargins(0, 0, 0, 0)
        self.type_combo = QComboBox(type_row)
        self.type_combo.setObjectName("orientationTypeCombo")
        for kind, descriptor in ORIENTATION_MODELS.items():
            self.type_combo.addItem(descriptor.label, kind.value)
        type_layout.addRow("Type", self.type_combo)
        layout.addWidget(type_row)

        self.page_stack = QStackedWidget(self)
        self.page_stack.setObjectName("orientationPageStack")
        page_builders = {
            OrientationKind.FIXED: self._build_fixed_page,
            OrientationKind.KEYFRAMES: self._build_keyframes_page,
            OrientationKind.ALIGN_MOTION: self._build_align_motion_page,
            OrientationKind.LOOK_AT: self._build_look_at_page,
            OrientationKind.SPIN: self._build_spin_page,
            OrientationKind.RANDOM: self._build_random_page,
        }
        self._pages = {kind.value: page_builders[kind]() for kind in ORIENTATION_MODELS}
        for page in self._pages.values():
            self.page_stack.addWidget(page)
        layout.addWidget(self.page_stack)

        action_row = QHBoxLayout()
        action_row.addStretch()
        self.apply_button = QPushButton("Apply Orientation", self)
        self.apply_button.setObjectName("orientationApplyButton")
        self.apply_button.clicked.connect(self.apply_requested.emit)
        action_row.addWidget(self.apply_button)
        layout.addLayout(action_row)

        self.type_combo.currentIndexChanged.connect(self._type_index_changed)
        self._show_type(OrientationKind.FIXED.value)

    def _build_fixed_page(self) -> QWidget:
        page = QWidget(self)
        page.setObjectName("orientationFixedPage")
        form = QFormLayout(page)
        self.fixed_angle_spins = _angle_spins(page, "orientationFixed")
        for label, spin in zip(("Yaw", "Pitch", "Roll"), self.fixed_angle_spins, strict=True):
            form.addRow(label, spin)
        return page

    def _build_keyframes_page(self) -> QWidget:
        page = QWidget(self)
        page.setObjectName("orientationKeyframesPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        self.keyframes_table = QTableWidget(0, 4, page)
        self.keyframes_table.setObjectName("orientationKeyframesTable")
        self.keyframes_table.setHorizontalHeaderLabels(("Time (s)", "Yaw", "Pitch", "Roll"))
        self.keyframes_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.keyframes_table.verticalHeader().setVisible(False)
        self.keyframes_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.keyframes_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.keyframes_table.itemSelectionChanged.connect(self._update_keyframe_buttons)
        layout.addWidget(self.keyframes_table)

        row_controls = QHBoxLayout()
        self.keyframe_add_button = QPushButton("Add", page)
        self.keyframe_add_button.setObjectName("orientationKeyframeAddButton")
        self.keyframe_remove_button = QPushButton("Remove", page)
        self.keyframe_remove_button.setObjectName("orientationKeyframeRemoveButton")
        self.keyframe_up_button = QPushButton("Up", page)
        self.keyframe_up_button.setObjectName("orientationKeyframeUpButton")
        self.keyframe_down_button = QPushButton("Down", page)
        self.keyframe_down_button.setObjectName("orientationKeyframeDownButton")
        for button in (
            self.keyframe_add_button,
            self.keyframe_remove_button,
            self.keyframe_up_button,
            self.keyframe_down_button,
        ):
            row_controls.addWidget(button)
        row_controls.addStretch()
        layout.addLayout(row_controls)

        self.keyframe_add_button.clicked.connect(self._add_keyframe_row)
        self.keyframe_remove_button.clicked.connect(self._remove_keyframe_row)
        self.keyframe_up_button.clicked.connect(lambda: self._move_keyframe_row(-1))
        self.keyframe_down_button.clicked.connect(lambda: self._move_keyframe_row(1))
        self._set_keyframe_rows(
            (
                OrientationKeyframeSpec(time_s=0.0),
                OrientationKeyframeSpec(time_s=1.0),
            )
        )
        return page

    def _build_align_motion_page(self) -> QWidget:
        page = QWidget(self)
        page.setObjectName("orientationAlignMotionPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.align_motion_explanation = QLabel(
            "The actor follows its prepared direction of travel.",
            page,
        )
        self.align_motion_explanation.setObjectName("orientationAlignMotionExplanation")
        self.align_motion_explanation.setWordWrap(True)
        layout.addWidget(self.align_motion_explanation)

        primary = QWidget(page)
        primary_form = QFormLayout(primary)
        primary_form.setContentsMargins(0, 0, 0, 0)
        self.align_allow_pitch_check = QCheckBox("Follow vertical motion", primary)
        self.align_allow_pitch_check.setObjectName("orientationAlignMotionAllowPitchCheck")
        self.align_allow_pitch_check.setChecked(True)
        primary_form.addRow("Pitch", self.align_allow_pitch_check)
        layout.addWidget(primary)

        self.align_advanced_section = _CollapsibleSection(
            "Advanced",
            "orientationAlignMotionAdvancedSection",
            page,
        )
        advanced = QFormLayout(self.align_advanced_section.content)
        self.align_smoothing_time_spin = _configured_spin(
            self.align_advanced_section.content,
            "orientationAlignMotionSmoothingTimeSpin",
            minimum=0.0,
            maximum=_TIME_LIMIT_SECONDS,
            suffix=" s",
        )
        (
            max_yaw_widget,
            self.align_max_yaw_rate_enabled_check,
            self.align_max_yaw_rate_spin,
        ) = _optional_spin_widget(
            self.align_advanced_section.content,
            "orientationAlignMotionMaxYawRate",
            "Limit",
            suffix="°/s",
        )
        (
            max_pitch_widget,
            self.align_max_pitch_rate_enabled_check,
            self.align_max_pitch_rate_spin,
        ) = _optional_spin_widget(
            self.align_advanced_section.content,
            "orientationAlignMotionMaxPitchRate",
            "Limit",
            suffix="°/s",
        )
        self.align_offset_spins = _angle_spins(
            self.align_advanced_section.content,
            "orientationAlignMotion",
        )
        advanced.addRow("Smoothing time", self.align_smoothing_time_spin)
        advanced.addRow("Maximum yaw rate", max_yaw_widget)
        advanced.addRow("Maximum pitch rate", max_pitch_widget)
        for label, spin in zip(
            ("Yaw offset", "Pitch offset", "Roll offset"),
            self.align_offset_spins,
            strict=True,
        ):
            advanced.addRow(label, spin)
        layout.addWidget(self.align_advanced_section)
        layout.addStretch()
        return page

    def _build_look_at_page(self) -> QWidget:
        page = QWidget(self)
        page.setObjectName("orientationLookAtPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        primary = QWidget(page)
        form = QFormLayout(primary)
        form.setContentsMargins(0, 0, 0, 0)
        self.look_at_target_mode_combo = QComboBox(primary)
        self.look_at_target_mode_combo.setObjectName("orientationLookAtTargetModeCombo")
        self.look_at_target_mode_combo.addItem("Actor", _LOOK_AT_ACTOR)
        self.look_at_target_mode_combo.addItem("Point", _LOOK_AT_POINT)
        form.addRow("Target", self.look_at_target_mode_combo)

        self.look_at_target_stack = QStackedWidget(primary)
        self.look_at_target_stack.setObjectName("orientationLookAtTargetStack")
        actor_page = QWidget(self.look_at_target_stack)
        actor_form = QFormLayout(actor_page)
        actor_form.setContentsMargins(0, 0, 0, 0)
        self.look_at_combo = QComboBox(actor_page)
        self.look_at_combo.setObjectName("orientationLookAtCombo")
        self.look_at_combo.setPlaceholderText("Select an actor")
        self.look_at_combo.setCurrentIndex(-1)
        actor_form.addRow("Actor", self.look_at_combo)
        self.look_at_target_stack.addWidget(actor_page)

        point_page = QWidget(self.look_at_target_stack)
        point_form = QFormLayout(point_page)
        point_form.setContentsMargins(0, 0, 0, 0)
        self.look_at_point_spins = _vector_spins(
            point_page,
            "orientationLookAtPoint",
        )
        for label, spin in zip(("X", "Y", "Z"), self.look_at_point_spins, strict=True):
            point_form.addRow(label, spin)
        self.look_at_point_spins[0].setValue(1.0)
        self.look_at_target_stack.addWidget(point_page)
        form.addRow(self.look_at_target_stack)

        self.look_at_allow_pitch_check = QCheckBox("Track elevation", primary)
        self.look_at_allow_pitch_check.setObjectName("orientationLookAtAllowPitchCheck")
        self.look_at_allow_pitch_check.setChecked(True)
        form.addRow("Pitch", self.look_at_allow_pitch_check)
        layout.addWidget(primary)

        self.look_at_advanced_section = _CollapsibleSection(
            "Advanced",
            "orientationLookAtAdvancedSection",
            page,
        )
        advanced = QFormLayout(self.look_at_advanced_section.content)
        self.look_at_smoothing_time_spin = _configured_spin(
            self.look_at_advanced_section.content,
            "orientationLookAtSmoothingTimeSpin",
            minimum=0.0,
            maximum=_TIME_LIMIT_SECONDS,
            suffix=" s",
        )
        (
            max_yaw_widget,
            self.look_at_max_yaw_rate_enabled_check,
            self.look_at_max_yaw_rate_spin,
        ) = _optional_spin_widget(
            self.look_at_advanced_section.content,
            "orientationLookAtMaxYawRate",
            "Limit",
            suffix="°/s",
        )
        (
            max_pitch_widget,
            self.look_at_max_pitch_rate_enabled_check,
            self.look_at_max_pitch_rate_spin,
        ) = _optional_spin_widget(
            self.look_at_advanced_section.content,
            "orientationLookAtMaxPitchRate",
            "Limit",
            suffix="°/s",
        )
        self.look_at_offset_spins = _angle_spins(
            self.look_at_advanced_section.content,
            "orientationLookAt",
        )
        (
            yaw_limits_widget,
            self.look_at_yaw_limits_enabled_check,
            self.look_at_yaw_limit_spins,
        ) = _optional_range_widget(
            self.look_at_advanced_section.content,
            "orientationLookAtYawLimits",
            "Enable",
        )
        (
            pitch_limits_widget,
            self.look_at_pitch_limits_enabled_check,
            self.look_at_pitch_limit_spins,
        ) = _optional_range_widget(
            self.look_at_advanced_section.content,
            "orientationLookAtPitchLimits",
            "Enable",
        )
        advanced.addRow("Smoothing time", self.look_at_smoothing_time_spin)
        advanced.addRow("Maximum yaw rate", max_yaw_widget)
        advanced.addRow("Maximum pitch rate", max_pitch_widget)
        for label, spin in zip(
            ("Yaw offset", "Pitch offset", "Roll offset"),
            self.look_at_offset_spins,
            strict=True,
        ):
            advanced.addRow(label, spin)
        advanced.addRow("Yaw limits", yaw_limits_widget)
        advanced.addRow("Pitch limits", pitch_limits_widget)
        layout.addWidget(self.look_at_advanced_section)
        layout.addStretch()

        self.look_at_target_mode_combo.currentIndexChanged.connect(
            self._look_at_target_mode_changed
        )
        self._look_at_target_mode_changed()
        return page

    def _build_spin_page(self) -> QWidget:
        page = QWidget(self)
        page.setObjectName("orientationSpinPage")
        form = QFormLayout(page)
        self.spin_axis_combo = QComboBox(page)
        self.spin_axis_combo.setObjectName("orientationSpinAxisCombo")
        for axis in ("yaw", "pitch", "roll"):
            self.spin_axis_combo.addItem(axis.title(), axis)
        self.spin_rate_spin = _configured_spin(
            page,
            "orientationSpinRateSpin",
            suffix="°/s",
        )
        self.spin_rate_spin.setValue(30.0)
        self.spin_angle_spins = _angle_spins(page, "orientationSpin")
        (
            self.spin_yaw_spin,
            self.spin_pitch_spin,
            self.spin_roll_spin,
        ) = self.spin_angle_spins
        self.spin_start_yaw_spin = self.spin_yaw_spin
        self.spin_rotations_spin = self.spin_rate_spin
        form.addRow("Axis", self.spin_axis_combo)
        form.addRow("Rate", self.spin_rate_spin)
        for label, spin in zip(
            ("Base yaw", "Base pitch", "Base roll"),
            self.spin_angle_spins,
            strict=True,
        ):
            form.addRow(label, spin)
        return page

    def _build_random_page(self) -> QWidget:
        page = QWidget(self)
        page.setObjectName("orientationRandomPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        primary = QWidget(page)
        form = QFormLayout(primary)
        form.setContentsMargins(0, 0, 0, 0)
        self.random_seed_spin = _configured_int_spin(
            primary,
            "orientationRandomSeedSpin",
        )
        yaw_widget, self.random_yaw_range_spins = _range_widget(
            primary,
            "orientationRandomYaw",
        )
        pitch_widget, self.random_pitch_range_spins = _range_widget(
            primary,
            "orientationRandomPitch",
        )
        roll_widget, self.random_roll_range_spins = _range_widget(
            primary,
            "orientationRandomRoll",
        )
        _set_range_values(self.random_yaw_range_spins, (-180.0, 180.0))
        _set_range_values(self.random_pitch_range_spins, (-90.0, 90.0))
        _set_range_values(self.random_roll_range_spins, (-180.0, 180.0))
        form.addRow("Seed", self.random_seed_spin)
        form.addRow("Yaw range", yaw_widget)
        form.addRow("Pitch range", pitch_widget)
        form.addRow("Roll range", roll_widget)
        layout.addWidget(primary)

        self.random_advanced_section = _CollapsibleSection(
            "Advanced",
            "orientationRandomAdvancedSection",
            page,
        )
        advanced = QFormLayout(self.random_advanced_section.content)
        (
            update_widget,
            self.random_update_interval_enabled_check,
            self.random_update_interval_spin,
        ) = _optional_spin_widget(
            self.random_advanced_section.content,
            "orientationRandomUpdateInterval",
            "Enable",
            suffix=" s",
        )
        advanced.addRow("Update interval", update_widget)
        layout.addWidget(self.random_advanced_section)
        layout.addStretch()
        return page

    def _type_index_changed(self, _index: int) -> None:
        token = str(self.type_combo.currentData())
        self._show_type(token)
        self.orientation_type_changed.emit(token)

    def _show_type(self, token: str) -> None:
        page = self._pages.get(str(token))
        if page is None:
            raise ValueError(f"unsupported orientation type: {token}")
        self.page_stack.setCurrentWidget(page)

    def _look_at_target_mode_changed(self, _index: int = -1) -> None:
        mode = str(self.look_at_target_mode_combo.currentData())
        self.look_at_target_stack.setCurrentIndex(0 if mode == _LOOK_AT_ACTOR else 1)

    def set_orientation(
        self,
        orientation: OrientationModel,
        *,
        preserve_type_signal: bool = False,
    ) -> None:
        """Populate the matching page, normally without emitting a type change."""

        kind = orientation_kind(orientation)
        token = kind.value
        if isinstance(orientation, FixedOrientationSpec):
            _set_triple_values(
                self.fixed_angle_spins,
                (
                    orientation.yaw_deg,
                    orientation.pitch_deg,
                    orientation.roll_deg,
                ),
            )
        elif isinstance(orientation, KeyframesOrientationSpec):
            self._set_keyframe_rows(orientation.keyframes)
        elif isinstance(orientation, AlignMotionOrientationSpec):
            self.align_allow_pitch_check.setChecked(orientation.allow_pitch)
            self.align_smoothing_time_spin.setValue(orientation.smoothing_time_s)
            _set_optional_value(
                self.align_max_yaw_rate_enabled_check,
                self.align_max_yaw_rate_spin,
                orientation.max_yaw_rate_deg_s,
            )
            _set_optional_value(
                self.align_max_pitch_rate_enabled_check,
                self.align_max_pitch_rate_spin,
                orientation.max_pitch_rate_deg_s,
            )
            _set_triple_values(
                self.align_offset_spins,
                (
                    orientation.yaw_offset_deg,
                    orientation.pitch_offset_deg,
                    orientation.roll_offset_deg,
                ),
            )
        elif isinstance(orientation, LookAtOrientationSpec):
            if orientation.actor is not None:
                self.look_at_target_mode_combo.setCurrentIndex(
                    self.look_at_target_mode_combo.findData(_LOOK_AT_ACTOR)
                )
                target_id = look_at_actor_id(orientation)
                assert target_id is not None
                self._select_look_at_target(target_id)
            else:
                self.look_at_target_mode_combo.setCurrentIndex(
                    self.look_at_target_mode_combo.findData(_LOOK_AT_POINT)
                )
                assert orientation.point_m is not None
                _set_triple_values(self.look_at_point_spins, orientation.point_m)
            self.look_at_allow_pitch_check.setChecked(orientation.allow_pitch)
            self.look_at_smoothing_time_spin.setValue(orientation.smoothing_time_s)
            _set_optional_value(
                self.look_at_max_yaw_rate_enabled_check,
                self.look_at_max_yaw_rate_spin,
                orientation.max_yaw_rate_deg_s,
            )
            _set_optional_value(
                self.look_at_max_pitch_rate_enabled_check,
                self.look_at_max_pitch_rate_spin,
                orientation.max_pitch_rate_deg_s,
            )
            _set_triple_values(
                self.look_at_offset_spins,
                (
                    orientation.yaw_offset_deg,
                    orientation.pitch_offset_deg,
                    orientation.roll_offset_deg,
                ),
            )
            _set_optional_range(
                self.look_at_yaw_limits_enabled_check,
                self.look_at_yaw_limit_spins,
                orientation.yaw_limits_deg,
            )
            _set_optional_range(
                self.look_at_pitch_limits_enabled_check,
                self.look_at_pitch_limit_spins,
                orientation.pitch_limits_deg,
            )
        elif isinstance(orientation, SpinOrientationSpec):
            self.spin_axis_combo.setCurrentIndex(self.spin_axis_combo.findData(orientation.axis))
            self.spin_rate_spin.setValue(orientation.rate_deg_s)
            _set_triple_values(
                self.spin_angle_spins,
                (
                    orientation.yaw_deg,
                    orientation.pitch_deg,
                    orientation.roll_deg,
                ),
            )
        elif isinstance(orientation, RandomOrientationSpec):
            self.random_seed_spin.setValue(orientation.seed)
            _set_range_values(
                self.random_yaw_range_spins,
                orientation.yaw_range_deg,
            )
            _set_range_values(
                self.random_pitch_range_spins,
                orientation.pitch_range_deg,
            )
            _set_range_values(
                self.random_roll_range_spins,
                orientation.roll_range_deg,
            )
            _set_optional_value(
                self.random_update_interval_enabled_check,
                self.random_update_interval_spin,
                orientation.update_interval_s,
            )
        else:
            raise TypeError(f"unsupported orientation value: {type(orientation).__name__}")

        blocker = None if preserve_type_signal else QSignalBlocker(self.type_combo)
        try:
            index = self.type_combo.findData(token)
            if index < 0:
                raise RuntimeError(f"orientation type is not registered: {token}")
            self.type_combo.setCurrentIndex(index)
            self._show_type(token)
        finally:
            del blocker

    def orientation(self) -> OrientationModel:
        """Return the validated shared orientation represented by the active page."""

        kind = OrientationKind(str(self.type_combo.currentData()))
        if kind is OrientationKind.FIXED:
            yaw, pitch, roll = _triple_values(self.fixed_angle_spins)
            return FixedOrientationSpec(
                yaw_deg=yaw,
                pitch_deg=pitch,
                roll_deg=roll,
            )
        if kind is OrientationKind.KEYFRAMES:
            keyframes = self._keyframe_rows()
            if len(keyframes) < 2:
                raise ValueError("keyframe orientation requires at least two rows")
            return KeyframesOrientationSpec(keyframes=keyframes)
        if kind is OrientationKind.ALIGN_MOTION:
            yaw, pitch, roll = _triple_values(self.align_offset_spins)
            return AlignMotionOrientationSpec(
                allow_pitch=self.align_allow_pitch_check.isChecked(),
                smoothing_time_s=self.align_smoothing_time_spin.value(),
                max_yaw_rate_deg_s=_optional_value(
                    self.align_max_yaw_rate_enabled_check,
                    self.align_max_yaw_rate_spin,
                ),
                max_pitch_rate_deg_s=_optional_value(
                    self.align_max_pitch_rate_enabled_check,
                    self.align_max_pitch_rate_spin,
                ),
                yaw_offset_deg=yaw,
                pitch_offset_deg=pitch,
                roll_offset_deg=roll,
            )
        if kind is OrientationKind.LOOK_AT:
            yaw, pitch, roll = _triple_values(self.look_at_offset_spins)
            fields = {
                "allow_pitch": self.look_at_allow_pitch_check.isChecked(),
                "smoothing_time_s": self.look_at_smoothing_time_spin.value(),
                "max_yaw_rate_deg_s": _optional_value(
                    self.look_at_max_yaw_rate_enabled_check,
                    self.look_at_max_yaw_rate_spin,
                ),
                "max_pitch_rate_deg_s": _optional_value(
                    self.look_at_max_pitch_rate_enabled_check,
                    self.look_at_max_pitch_rate_spin,
                ),
                "yaw_offset_deg": yaw,
                "pitch_offset_deg": pitch,
                "roll_offset_deg": roll,
                "yaw_limits_deg": _optional_range(
                    self.look_at_yaw_limits_enabled_check,
                    self.look_at_yaw_limit_spins,
                ),
                "pitch_limits_deg": _optional_range(
                    self.look_at_pitch_limits_enabled_check,
                    self.look_at_pitch_limit_spins,
                ),
            }
            if self.look_at_target_mode_combo.currentData() == _LOOK_AT_ACTOR:
                target_id = self.look_at_combo.currentData()
                if target_id is None:
                    raise ValueError("look-at orientation requires a target actor")
                return actor_look_at_orientation(UUID(str(target_id)), **fields)
            return point_look_at_orientation(
                _triple_values(self.look_at_point_spins),
                **fields,
            )
        if kind is OrientationKind.SPIN:
            yaw, pitch, roll = _triple_values(self.spin_angle_spins)
            return SpinOrientationSpec(
                axis=str(self.spin_axis_combo.currentData()),
                rate_deg_s=self.spin_rate_spin.value(),
                yaw_deg=yaw,
                pitch_deg=pitch,
                roll_deg=roll,
            )
        return RandomOrientationSpec(
            seed=self.random_seed_spin.value(),
            yaw_range_deg=_range_values(self.random_yaw_range_spins),
            pitch_range_deg=_range_values(self.random_pitch_range_spins),
            roll_range_deg=_range_values(self.random_roll_range_spins),
            update_interval_s=_optional_value(
                self.random_update_interval_enabled_check,
                self.random_update_interval_spin,
            ),
        )

    def get_orientation(self) -> OrientationModel:
        """Return the orientation represented by the active editor page."""

        return self.orientation()

    def set_look_at_choices(
        self,
        choices: Iterable[LookAtChoice],
        *,
        preserve_selection: bool = True,
    ) -> None:
        """Replace UUID-backed labels, optionally retaining the prior selection."""

        selected_data = self.look_at_combo.currentData()
        selected_id = (
            UUID(str(selected_data)) if preserve_selection and selected_data is not None else None
        )
        normalized = tuple((UUID(str(actor_id)), str(label)) for actor_id, label in choices)

        blocker = QSignalBlocker(self.look_at_combo)
        try:
            self.look_at_combo.clear()
            for actor_id, label in normalized:
                self.look_at_combo.addItem(label, actor_id)
            if selected_id is not None:
                self._select_look_at_target(selected_id)
            else:
                self.look_at_combo.setCurrentIndex(-1)
        finally:
            del blocker

    def set_editing_enabled(self, enabled: bool) -> None:
        """Enable or disable every value-changing control."""

        self._editing_enabled = bool(enabled)
        self.type_combo.setEnabled(self._editing_enabled)
        self.page_stack.setEnabled(self._editing_enabled)
        self.apply_button.setEnabled(self._editing_enabled)
        self._update_keyframe_buttons()

    def set_read_only(self, read_only: bool) -> None:
        """Switch between editable and read-only presentation."""

        self.set_editing_enabled(not read_only)

    def _select_look_at_target(self, target_id: UUID) -> None:
        normalized = UUID(str(target_id))
        index = next(
            (
                candidate
                for candidate in range(self.look_at_combo.count())
                if UUID(str(self.look_at_combo.itemData(candidate))) == normalized
            ),
            -1,
        )
        if index < 0:
            self.look_at_combo.addItem(f"Unavailable actor ({normalized})", normalized)
            index = self.look_at_combo.count() - 1
        self.look_at_combo.setCurrentIndex(index)

    def _keyframe_spin(self, row: int, column: int) -> QDoubleSpinBox:
        widget = self.keyframes_table.cellWidget(row, column)
        if not isinstance(widget, QDoubleSpinBox):
            raise RuntimeError(f"keyframe cell {row},{column} has no numeric editor")
        return widget

    def _keyframe_rows(self) -> tuple[OrientationKeyframeSpec, ...]:
        return tuple(
            OrientationKeyframeSpec(
                time_s=self._keyframe_spin(row, 0).value(),
                yaw_deg=self._keyframe_spin(row, 1).value(),
                pitch_deg=self._keyframe_spin(row, 2).value(),
                roll_deg=self._keyframe_spin(row, 3).value(),
            )
            for row in range(self.keyframes_table.rowCount())
        )

    def _set_keyframe_rows(
        self,
        keyframes: Iterable[OrientationKeyframeSpec],
        *,
        selected_row: int | None = None,
    ) -> None:
        frozen = tuple(keyframes)
        self.keyframes_table.clearContents()
        self.keyframes_table.setRowCount(len(frozen))
        for row, keyframe in enumerate(frozen):
            values = (
                keyframe.time_s,
                keyframe.yaw_deg,
                keyframe.pitch_deg,
                keyframe.roll_deg,
            )
            names = ("Time", "Yaw", "Pitch", "Roll")
            for column, (name, value) in enumerate(zip(names, values, strict=True)):
                if column == 0:
                    spin = _configured_spin(
                        self.keyframes_table,
                        f"orientationKeyframe{row}{name}Spin",
                        minimum=0.0,
                        maximum=_TIME_LIMIT_SECONDS,
                        suffix=" s",
                    )
                else:
                    spin = _configured_spin(
                        self.keyframes_table,
                        f"orientationKeyframe{row}{name}Spin",
                        minimum=-_ANGLE_LIMIT_DEGREES,
                        maximum=_ANGLE_LIMIT_DEGREES,
                        suffix="°",
                    )
                spin.setValue(float(value))
                self.keyframes_table.setCellWidget(row, column, spin)
        if selected_row is not None and 0 <= selected_row < len(frozen):
            self.keyframes_table.selectRow(selected_row)
        else:
            self.keyframes_table.clearSelection()
        self._update_keyframe_buttons()

    def _add_keyframe_row(self) -> None:
        keyframes = list(self._keyframe_rows())
        current = self.keyframes_table.currentRow()
        insert_at = current + 1 if 0 <= current < len(keyframes) else len(keyframes)
        seed = (
            keyframes[current]
            if 0 <= current < len(keyframes)
            else keyframes[-1] if keyframes else OrientationKeyframeSpec(time_s=0.0)
        )
        if insert_at == 0:
            time_s = 0.0
        elif insert_at < len(keyframes):
            time_s = (keyframes[insert_at - 1].time_s + keyframes[insert_at].time_s) / 2.0
        else:
            previous_time = keyframes[-1].time_s if keyframes else -1.0
            previous_gap = (
                keyframes[-1].time_s - keyframes[-2].time_s if len(keyframes) >= 2 else 1.0
            )
            time_s = previous_time + max(previous_gap, 0.000_001)
        keyframes.insert(
            insert_at,
            OrientationKeyframeSpec(
                time_s=time_s,
                yaw_deg=seed.yaw_deg,
                pitch_deg=seed.pitch_deg,
                roll_deg=seed.roll_deg,
            ),
        )
        self._set_keyframe_rows(keyframes, selected_row=insert_at)

    def _remove_keyframe_row(self) -> None:
        current = self.keyframes_table.currentRow()
        keyframes = list(self._keyframe_rows())
        if current < 0 or current >= len(keyframes) or len(keyframes) <= 2:
            return
        keyframes.pop(current)
        self._set_keyframe_rows(
            keyframes,
            selected_row=min(current, len(keyframes) - 1),
        )

    def _move_keyframe_row(self, offset: int) -> None:
        current = self.keyframes_table.currentRow()
        destination = current + int(offset)
        keyframes = list(self._keyframe_rows())
        if (
            current < 0
            or current >= len(keyframes)
            or destination < 0
            or destination >= len(keyframes)
        ):
            return
        source_angles = (
            keyframes[current].yaw_deg,
            keyframes[current].pitch_deg,
            keyframes[current].roll_deg,
        )
        destination_angles = (
            keyframes[destination].yaw_deg,
            keyframes[destination].pitch_deg,
            keyframes[destination].roll_deg,
        )
        keyframes[current] = keyframes[current].model_copy(
            update={
                "yaw_deg": destination_angles[0],
                "pitch_deg": destination_angles[1],
                "roll_deg": destination_angles[2],
            }
        )
        keyframes[destination] = keyframes[destination].model_copy(
            update={
                "yaw_deg": source_angles[0],
                "pitch_deg": source_angles[1],
                "roll_deg": source_angles[2],
            }
        )
        self._set_keyframe_rows(keyframes, selected_row=destination)

    def _update_keyframe_buttons(self) -> None:
        current = self.keyframes_table.currentRow()
        count = self.keyframes_table.rowCount()
        selected = self._editing_enabled and 0 <= current < count
        self.keyframe_add_button.setEnabled(self._editing_enabled)
        self.keyframe_remove_button.setEnabled(selected and count > 2)
        self.keyframe_up_button.setEnabled(selected and current > 0)
        self.keyframe_down_button.setEnabled(selected and current < count - 1)


__all__ = ["LookAtChoice", "OrientationEditor"]
