"""Small exhibition controls built on existing playback and camera owners."""

from contextlib import ExitStack

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..materials.texture_policy import textures_locked_off
from ..services.demo_presets import PRESETS as DEMO_PRESETS
from ..services.demo_presets import available_presets, preset_options
from ..services.demo_service import DEFAULT_APPEARANCE_CYCLE, DemoService
from ..widgets.form_layout import AlignedFormLayout
from .base import BasePanel
from .collapsible_section import CollapsibleSection


class _DemoControls(QGroupBox):
    """Refresh late-loading scenario capabilities only while the controls are visible."""

    refresh_requested = Signal()

    def __init__(self, title):
        super().__init__(title)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(500)
        self.refresh_timer.timeout.connect(self.refresh_requested.emit)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_requested.emit()
        self.refresh_timer.start()

    def hideEvent(self, event):
        self.refresh_timer.stop()
        super().hideEvent(event)


class DemoPanel(BasePanel):
    """Expose reversible presentation automation in the ordinary visualizer."""

    def __init__(self, parent):
        super().__init__(parent)
        self.service = DemoService(parent)
        parent.demo_service = self.service
        self._environment_choices = {}
        self._available = {}
        self._updating = False
        self._preset_chosen = False
        self._form_labels = {}
        self._controls_group = None

    def create_panel(self):
        group = self._controls_group = _DemoControls("Demo mode")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        explanation = QLabel(
            "Choose a presentation for this scene. You can rotate and zoom while it runs."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        presets = self.widgets["demo_preset"] = QComboBox()
        presets.setAccessibleName("Demo preset")
        for key, (label, description) in DEMO_PRESETS.items():
            presets.addItem(label, key)
            presets.setItemData(presets.count() - 1, description, Qt.ToolTipRole)
        presets.setCurrentIndex(presets.findData("presentation"))
        preset_form = AlignedFormLayout()
        preset_form.addRow("Preset", presets)
        layout.addLayout(preset_form)
        description = self.widgets["demo_preset_description"] = QLabel()
        description.setWordWrap(True)
        layout.addWidget(description)
        self.widgets["demo_status"] = QLabel(self.service.status)
        self.widgets["demo_status"].setWordWrap(True)
        layout.addWidget(self.widgets["demo_status"])
        buttons = QHBoxLayout()
        for key, label, callback in (
            ("demo_start", "Start demo", self.start),
            ("demo_stop", "Stop demo", self.service.stop),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            if key == "demo_stop":
                button.setProperty("demoAction", True)
            self.widgets[key] = button
            buttons.addWidget(button)
        layout.addLayout(buttons)
        resume = self.widgets["demo_resume_camera"] = QPushButton("Resume camera")
        resume.setProperty("demoAction", True)
        resume.clicked.connect(self.service.resume_camera)
        layout.addWidget(resume)
        self.widgets["demo_clean_view"] = QCheckBox("Hide interface during demo")
        self.widgets["demo_clean_view"].setToolTip(
            "Show the scene with a small footer. Stop demo or press Esc to restore your layout."
        )
        self.widgets["demo_captions"] = QCheckBox("Show explanatory captions")
        self.widgets["demo_captions"].setToolTip(
            "Describe the current look, path colors and recorded playback beneath the scene."
        )
        layout.addWidget(self.widgets["demo_clean_view"])
        layout.addWidget(self.widgets["demo_captions"])
        advanced = self.widgets["demo_advanced"] = CollapsibleSection(
            "Advanced options", start_open=False, flat_header=True
        )
        advanced_layout = advanced.content_layout()
        layout.addWidget(advanced)
        self.widgets["demo_showcase"] = QCheckBox("Preset tour + MPC pulses, then play once")
        self.widgets["demo_showcase"].setToolTip(
            "Hold the current frame while showing the selected appearance presets and slowed MPC travel. "
            "Return to your look, stop camera motion, then play the recording from its first "
            "available frame to the end. Requires visible MPCs and at least two recorded frames."
        )
        self.widgets["demo_showcase"].toggled.connect(self._sync_cycle_options)
        self.widgets["demo_sensing_story"] = QCheckBox("Replay with periodic detection reveals")
        self.widgets["demo_sensing_story"].setToolTip(
            "Replay saved detections and tracks, pausing to explain a recorded detection."
        )
        self.widgets["demo_sensing_story"].toggled.connect(self._sync_cycle_options)
        self.widgets["demo_orbit"] = QCheckBox("Slow camera orbit")
        self.widgets["demo_orbit"].setChecked(True)
        self.widgets["demo_tour"] = QCheckBox("Scenario camera tour")
        self.widgets["demo_targets"] = QCheckBox("Cycle targets from current camera position")
        self.widgets["demo_targets"].setToolTip(
            "Keep the camera in place and look at each recorded target in turn, "
            "following its motion until the next switch. Uses the Advanced camera "
            "smooth-transition setting and duration."
        )
        self.widgets["demo_exhibition"] = QCheckBox("Exhibition layout")
        self.widgets["demo_exhibition"].setToolTip(
            "Give the scene most of the window, with the chosen views on the right. "
            "Your controls and dock arrangement return when you stop the demo."
        )
        exhibition_content = self.widgets["demo_exhibition_content"] = QComboBox()
        exhibition_content.addItem("Actor view and live figure", "actor")
        exhibition_content.addItem("Sensing map and range profile", "sensing")
        exhibition_content.setToolTip(
            "Sensing uses two linked live figures with saved detection markers. "
            "Existing figures and their placement are restored when the demo stops."
        )
        self.widgets["demo_exhibition"].toggled.connect(self._sync_cycle_options)
        camera_options = ("demo_orbit", "demo_tour", "demo_targets")
        for key in camera_options:
            for other in camera_options:
                if other == key:
                    continue
                self.widgets[key].toggled.connect(
                    lambda checked, other=other: (
                        self.widgets[other].setChecked(False) if checked else None
                    )
                )
            self.widgets[key].toggled.connect(self._sync_camera_options)
        self.widgets["demo_playback"] = QCheckBox("Loop recorded playback")
        self.widgets["demo_playback"].setChecked(True)
        from ..materials.view_styles import DESCRIPTIONS, PRESETS

        look = self.widgets["demo_look"] = QComboBox()
        look.addItem("Current", None)
        for name in PRESETS:
            look.addItem(name, name)
        look.setToolTip(
            "Hold a scene preset with its matching MPC appearance. Dark and Glow show up to "
            "20 strongest paths from your current filters. Stopping restores your settings. "
            "Current allows the look cycles below."
        )
        look.currentIndexChanged.connect(self._sync_cycle_options)
        self.widgets["demo_lighting"] = QCheckBox("Cycle current, soft and side lighting")
        self.widgets["demo_ibl"] = QCheckBox("Cycle sky / IBL environments")
        self.widgets["demo_ibl"].setToolTip(
            "Choose installed skies below to cycle with their matching lighting. Nothing is downloaded. "
            "The original environment and sky visibility return when you stop the demo."
        )
        self.widgets["demo_appearance"] = QCheckBox("Cycle appearance presets")
        self.widgets["demo_appearance"].setChecked(True)
        self.widgets["demo_appearance"].setToolTip(
            "Start with your current look, including unsaved Custom settings, then show "
            "the checked presets in list order with matching MPC styles before returning to your exact look. "
            "Your current look is restored when the demo stops."
        )
        for key in ("demo_appearance_choices", "demo_ibl_choices"):
            choices = self.widgets[key] = QListWidget()
            choices.setAccessibleName(
                "Appearance presets" if key == "demo_appearance_choices" else "IBL environments"
            )
            choices.setSelectionMode(QListWidget.SelectionMode.NoSelection)
            choices.setMaximumHeight(132)
            choices.setToolTip(
                "Check the entries to include, in list order. The current look starts each loop."
            )
        for name, preset in PRESETS.items():
            item = self._add_choice(
                self.widgets["demo_appearance_choices"],
                name,
                name,
                name in DEFAULT_APPEARANCE_CYCLE,
            )
            item.setToolTip(DESCRIPTIONS[name])
            if preset["textures"] and textures_locked_off():
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setHidden(True)
                look.removeItem(look.findData(name))
        self.widgets["demo_ibl_choices"].setToolTip(
            "Check the installed skies to visit in list order, after the current environment. "
            "Install additional skies in Rendering."
        )
        self.widgets["demo_ibl_empty"] = QLabel(
            "Install another sky in Rendering to add it to this loop."
        )
        self.widgets["demo_ibl_empty"].setWordWrap(True)
        lighting = self.widgets["demo_lighting_mode"] = QComboBox()
        for label, mode in (
            ("Off", "off"),
            ("Blue hour", "blue_hour"),
            ("Sunset cycle", "sunset"),
        ):
            lighting.addItem(label, mode)
        lighting.setToolTip(
            "Off keeps your current lighting. Blue hour applies the scenario's authored lights. "
            "Sunset cycles from daylight to blue hour and back over the selected duration, "
            "with brief holds at both endpoints. The current look stays until the cycle finishes. Without a look cycle, "
            "the lighting cycle repeats. "
            "Technical looks remain unchanged; stopping restores your original lighting."
        )
        self.widgets["demo_wireframe"] = QCheckBox("Add wireframe to studio cycle")
        self.widgets["demo_wireframe"].setToolTip(
            "Add an occasional mesh view to the lighting cycle."
        )
        for key in ("demo_lighting", "demo_ibl", "demo_appearance"):
            self.widgets[key].toggled.connect(self._sync_cycle_options)
        for key in (
            "demo_showcase",
            "demo_sensing_story",
            "demo_orbit",
            "demo_tour",
            "demo_targets",
            "demo_exhibition",
            "demo_playback",
            "demo_lighting",
            "demo_wireframe",
            "demo_ibl",
            "demo_appearance",
        ):
            advanced_layout.addWidget(self.widgets[key])
            if key == "demo_appearance":
                advanced_layout.addWidget(self.widgets["demo_appearance_choices"])
            elif key == "demo_ibl":
                advanced_layout.addWidget(self.widgets["demo_ibl_choices"])
                advanced_layout.addWidget(self.widgets["demo_ibl_empty"])
        form = AlignedFormLayout()
        form.addRow("Exhibition views", exhibition_content)
        self._form_labels["demo_exhibition_content"] = form.labelForField(exhibition_content)
        form.addRow("Look", look)
        form.addRow("Scenario lighting", lighting)
        self._form_labels["demo_look"] = form.labelForField(look)
        self._form_labels["demo_lighting_mode"] = form.labelForField(lighting)
        self.widgets["demo_speed"] = QDoubleSpinBox()
        self.widgets["demo_speed"].setRange(0.2, 12)
        self.widgets["demo_speed"].setValue(3)
        self.widgets["demo_speed"].setSuffix(" °/s")
        self.widgets["demo_interval"] = QSpinBox()
        self.widgets["demo_interval"].setRange(1, 120)
        self.widgets["demo_interval"].setValue(5)
        self.widgets["demo_interval"].setSuffix(" s")
        form.addRow("Orbit speed", self.widgets["demo_speed"])
        self._orbit_speed_label = form.labelForField(self.widgets["demo_speed"])
        form.addRow("Change look every", self.widgets["demo_interval"])
        self._form_labels["demo_interval"] = form.labelForField(self.widgets["demo_interval"])
        for key, label in (
            ("demo_target_interval", "Follow each target for"),
            ("demo_sunset_duration", "Sunset cycle duration"),
        ):
            spin = self.widgets[key] = QSpinBox()
            spin.setRange(1, 600)
            spin.setValue(5)
            spin.setSuffix(" s")
            form.addRow(label, spin)
        self._target_interval_label = form.labelForField(self.widgets["demo_target_interval"])
        self._sunset_duration_label = form.labelForField(self.widgets["demo_sunset_duration"])
        lighting.currentIndexChanged.connect(self._sync_camera_options)
        advanced_layout.addLayout(form)
        note = QLabel(
            "Drag or scroll to pause automatic camera motion; the rest of the demo continues. "
            "Resume camera restarts it. Stop demo or Esc restores your appearance, playback and layout, "
            "keeping the camera where you left it."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()
        self.service.changed.connect(self.refresh)
        group.refresh_requested.connect(self._refresh_idle)
        presets.currentIndexChanged.connect(self._apply_preset)
        presets.activated.connect(self._remember_preset_choice)
        for key in ("demo_clean_view", "demo_exhibition"):
            other = "demo_exhibition" if key == "demo_clean_view" else "demo_clean_view"
            self.widgets[key].toggled.connect(
                lambda checked, other=other: (
                    self.widgets[other].setChecked(False) if checked else None
                )
            )
        for key, other in (
            ("demo_showcase", "demo_sensing_story"),
            ("demo_sensing_story", "demo_showcase"),
        ):
            self.widgets[key].toggled.connect(
                lambda checked, other=other: (
                    self.widgets[other].setChecked(False) if checked else None
                )
            )
        for key, control in self.widgets.items():
            if isinstance(control, QCheckBox):
                control.toggled.connect(self._mark_custom)
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.valueChanged.connect(self._mark_custom)
            elif isinstance(control, QComboBox) and key != "demo_preset":
                control.currentIndexChanged.connect(self._mark_custom)
            elif isinstance(control, QListWidget):
                control.itemChanged.connect(self._mark_custom)
        self._sync_cycle_options()
        self._sync_camera_options()
        self.refresh()
        return group

    def _sync_camera_options(self):
        if "demo_speed" in self.widgets:
            visible = self._checked("demo_orbit")
            self.widgets["demo_speed"].setVisible(visible)
            self._orbit_speed_label.setVisible(visible)
            for key, label, visible in (
                (
                    "demo_target_interval",
                    self._target_interval_label,
                    self._checked("demo_targets"),
                ),
                (
                    "demo_sunset_duration",
                    self._sunset_duration_label,
                    self.widgets["demo_lighting_mode"].isEnabled()
                    and self.widgets["demo_lighting_mode"].currentData() == "sunset",
                ),
            ):
                self.widgets[key].setVisible(visible)
                label.setVisible(visible)

    def _sync_cycle_options(self):
        """Hide unavailable content and show only controls used by the selected recipe."""
        sensing = (
            self._available.get("sensing_story", False)
            and self.widgets["demo_sensing_story"].isChecked()
        )
        for suffix in (
            "orbit",
            "tour",
            "targets",
            "showcase",
            "sensing_story",
            "exhibition",
            "clean_view",
            "captions",
        ):
            self._set_option_visible(
                f"demo_{suffix}",
                self._available.get(suffix, False),
                not sensing or suffix not in {"orbit", "tour", "targets", "showcase"},
            )
        fixed = not sensing and self.widgets["demo_look"].currentData() is not None
        self._set_option_visible("demo_exhibition_content", self._checked("demo_exhibition"))
        sensing_index = self.widgets["demo_exhibition_content"].findData("sensing")
        self.widgets["demo_exhibition_content"].model().item(sensing_index).setEnabled(
            self._available.get("sensing_presentation", False)
        )
        showcase = self._checked("demo_showcase")
        self._set_option_visible(
            "demo_look", self._available.get("appearance", False), not showcase and not sensing
        )
        self._set_option_visible(
            "demo_playback", self._available.get("playback", False), not showcase and not sensing
        )
        for suffix in ("ibl", "appearance"):
            self._set_option_visible(
                f"demo_{suffix}",
                self._available.get(suffix, False),
                not fixed and not showcase and not sensing,
            )
        self._set_option_visible(
            "demo_lighting_mode",
            self._available.get("dusk", False),
            not showcase and not fixed and not sensing,
        )
        rich_cycle = (
            sensing
            or showcase
            or fixed
            or self._checked("demo_ibl")
            or self._checked("demo_appearance")
        )
        self._set_option_visible(
            "demo_lighting", self._available.get("lighting", False) and not rich_cycle
        )
        self._set_option_visible("demo_wireframe", self._checked("demo_lighting"))
        self._set_option_visible(
            "demo_interval",
            sensing
            or showcase
            or self._checked("demo_appearance")
            or self._checked("demo_ibl")
            or self._checked("demo_lighting"),
        )
        if "demo_interval" in self._form_labels:
            self._form_labels["demo_interval"].setText(
                "Reveal every" if sensing else "Change look every"
            )
        appearance_choices = self.widgets["demo_appearance_choices"]
        appearance_choices.setVisible(showcase or self._checked("demo_appearance"))
        appearance_choices.setEnabled(showcase or self._checked("demo_appearance"))
        ibl_visible = self._checked("demo_ibl")
        self.widgets["demo_ibl_choices"].setVisible(ibl_visible)
        self.widgets["demo_ibl_choices"].setEnabled(ibl_visible)
        self.widgets["demo_ibl_empty"].hide()
        self._sync_camera_options()

    def _set_option_visible(self, key, available, enabled=True):
        widget = self.widgets[key]
        widget.setVisible(available)
        widget.setEnabled(available and enabled)
        if key in self._form_labels:
            self._form_labels[key].setVisible(available)

    def _checked(self, key):
        widget = self.widgets[key]
        return widget.isEnabled() and widget.isChecked()

    def _mark_custom(self, *_):
        """A manual adjustment remains visible as an editable Custom recipe."""
        if self._updating or self.service.active:
            return
        self._preset_chosen = True
        combo = self.widgets["demo_preset"]
        with QSignalBlocker(combo):
            combo.setCurrentIndex(combo.findData("custom"))
        self.widgets["demo_preset_description"].setText(DEMO_PRESETS["custom"][1])

    def _remember_preset_choice(self, *_):
        """Explicitly choosing even the current fallback prevents automatic replacement."""
        self._preset_chosen = True

    def _apply_preset(self, *selection):
        if self._updating or self.service.active:
            return
        if selection:
            self._remember_preset_choice()
        name = self.widgets["demo_preset"].currentData()
        if name is None:
            return
        self.widgets["demo_preset_description"].setText(DEMO_PRESETS[name][1])
        if name == "custom":
            self.widgets["demo_advanced"].expand()
            return
        options = preset_options(name, self._available)
        self._updating = True
        try:
            with ExitStack() as stack:
                for widget in self.widgets.values():
                    if isinstance(widget, (QCheckBox, QComboBox, QListWidget)):
                        stack.enter_context(QSignalBlocker(widget))
                for key, widget in self.widgets.items():
                    if isinstance(widget, QCheckBox):
                        widget.setChecked(options.get(key.removeprefix("demo_"), False))
                self.widgets["demo_look"].setCurrentIndex(
                    self.widgets["demo_look"].findData("Detailed")
                    if name == "sensing_presentation"
                    else 0
                )
                self.widgets["demo_exhibition_content"].setCurrentIndex(
                    1 if name == "sensing_presentation" else 0
                )
                self.widgets["demo_lighting_mode"].setCurrentIndex(0)
                self.widgets["demo_interval"].setValue(12 if name == "sensing_story" else 5)
                choices = self.widgets["demo_appearance_choices"]
                for index in range(choices.count()):
                    item = choices.item(index)
                    selected = item.data(Qt.UserRole) in DEFAULT_APPEARANCE_CYCLE
                    item.setCheckState(
                        Qt.Checked if selected and item.flags() & Qt.ItemIsEnabled else Qt.Unchecked
                    )
        finally:
            self._updating = False
        self._sync_cycle_options()
        self._sync_camera_options()

    def _refresh_presets(self):
        combo = self.widgets["demo_preset"]
        selected = combo.currentData()
        names = available_presets(self._available)
        if not self._preset_chosen:
            selected = "presentation" if "presentation" in names else "explore"
        if tuple(combo.itemData(i) for i in range(combo.count())) != names:
            with QSignalBlocker(combo):
                combo.clear()
                for name in names:
                    label, description = DEMO_PRESETS[name]
                    combo.addItem(label, name)
                    combo.setItemData(combo.count() - 1, description, Qt.ToolTipRole)
                combo.setCurrentIndex(max(0, combo.findData(selected)))
        elif combo.currentData() != selected:
            with QSignalBlocker(combo):
                combo.setCurrentIndex(combo.findData(selected))

    @staticmethod
    def _add_choice(widget, label, value, checked):
        item = QListWidgetItem(label, widget)
        item.setData(Qt.ItemDataRole.UserRole, value)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        return item

    def _selected_choices(self, key):
        widget = self.widgets[key]
        return tuple(
            item.data(Qt.ItemDataRole.UserRole)
            for item in (widget.item(i) for i in range(widget.count()))
            if item.checkState() == Qt.CheckState.Checked
            and item.flags() & Qt.ItemFlag.ItemIsEnabled
        )

    def _refresh_environments(self):
        """Keep user selections when installed skies or the current environment change."""
        widget = self.widgets["demo_ibl_choices"]
        for i in range(widget.count()):
            item = widget.item(i)
            self._environment_choices[item.data(Qt.ItemDataRole.UserRole)] = (
                item.checkState() == Qt.CheckState.Checked
            )
        available = self.service.available_environments()
        existing = tuple(
            (widget.item(i).data(Qt.ItemDataRole.UserRole), widget.item(i).text())
            for i in range(widget.count())
        )
        if existing == available:
            return
        widget.clear()
        for name, label in available:
            self._add_choice(widget, label, name, self._environment_choices.get(name, True))

    def start(self):
        self.refresh()
        lighting = self.widgets["demo_lighting_mode"]
        self.service.start(
            showcase=self._checked("demo_showcase"),
            sensing_story=self._checked("demo_sensing_story"),
            orbit=self._checked("demo_orbit"),
            tour=self._checked("demo_tour"),
            cycle_targets=self._checked("demo_targets"),
            exhibition=self._checked("demo_exhibition"),
            exhibition_content=self.widgets["demo_exhibition_content"].currentData(),
            clean_view=self._checked("demo_clean_view"),
            captions=self._checked("demo_captions"),
            playback=self._checked("demo_playback"),
            cycle_lighting=self._checked("demo_lighting"),
            include_wireframe=self._checked("demo_wireframe"),
            cycle_ibl=self._checked("demo_ibl"),
            cycle_appearance=self._checked("demo_appearance"),
            appearance_presets=self._selected_choices("demo_appearance_choices"),
            ibl_environments=self._selected_choices("demo_ibl_choices"),
            presentation=(
                self.widgets["demo_look"].currentData()
                if self.widgets["demo_look"].isEnabled()
                else None
            ),
            scenario_dusk=lighting.isEnabled() and lighting.currentData() == "blue_hour",
            scenario_sunset=lighting.isEnabled() and lighting.currentData() == "sunset",
            orbit_speed=self.widgets["demo_speed"].value(),
            look_interval=self.widgets["demo_interval"].value(),
            target_interval=self.widgets["demo_target_interval"].value(),
            sunset_duration=self.widgets["demo_sunset_duration"].value(),
        )

    def refresh(self):
        self.widgets["demo_advanced"].setEnabled(not self.service.active)
        if not self.service.active:
            available = self.service.available_options()
            changed = available != self._available
            self._available = available
            self._updating = True
            try:
                self._refresh_environments()
                self._refresh_presets()
                for name in (
                    "orbit",
                    "tour",
                    "targets",
                    "showcase",
                    "sensing_story",
                    "playback",
                    "appearance",
                    "ibl",
                    "lighting",
                    "exhibition",
                    "clean_view",
                    "captions",
                ):
                    if not available.get(name, False):
                        self.widgets[f"demo_{name}"].setChecked(False)
                if not available.get("dusk", False):
                    self.widgets["demo_lighting_mode"].setCurrentIndex(0)
                if not available.get("appearance", False):
                    self.widgets["demo_look"].setCurrentIndex(0)
                if not available.get("sensing_presentation", False):
                    self.widgets["demo_exhibition_content"].setCurrentIndex(0)
            finally:
                self._updating = False
            if changed:
                self._apply_preset()
            self._sync_cycle_options()
        self.widgets["demo_status"].setText(self.service.status)
        self.widgets["demo_start"].setEnabled(
            not self.service.active
            and self._available.get("ready", getattr(self.parent, "ready", False))
        )
        self.widgets["demo_stop"].setEnabled(self.service.active)
        self.widgets["demo_resume_camera"].setVisible(
            self.service.active and self.service.camera_paused
        )
        self.widgets["demo_preset"].setEnabled(not self.service.active)
        for key in ("demo_clean_view", "demo_captions"):
            self.widgets[key].setEnabled(
                not self.service.active and self._available.get(key.removeprefix("demo_"), False)
            )

    def _refresh_idle(self):
        if not self.service.active:
            self.refresh()

    def cleanup(self):
        if self._controls_group is not None:
            self._controls_group.refresh_timer.stop()
        self.service.close()
        self.service.changed.disconnect(self.refresh)
