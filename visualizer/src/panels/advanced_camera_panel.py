"""Optional pygfx camera tools, kept out of the persistent camera header."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..widgets.form_layout import AlignedFormLayout
from .base import BasePanel


class _CameraTabs(QTabWidget):
    """Follow the active page's styled height without retaining a fixed pixel cap."""

    def _page_hint(self, hint):
        page = self.currentWidget()
        if page is not None:
            height = max(page.sizeHint().height(), page.minimumSizeHint().height())
            if page.hasHeightForWidth():
                height = max(height, page.heightForWidth(page.width()))
            frame = self.style().pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth)
            hint.setHeight(height + self.tabBar().sizeHint().height() + 2 * frame)
        return hint

    def sizeHint(self):
        return self._page_hint(super().sizeHint())

    def minimumSizeHint(self):
        return self._page_hint(super().minimumSizeHint())


class _FramingPage(QWidget):
    """Mirror mouse framing only while the optional controls are visible."""

    def __init__(self, refresh):
        super().__init__()
        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(refresh)

    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)


class AdvancedCameraPanel(BasePanel):
    """Motion, framing and saved shots for everyday navigation and recording."""

    def create_panel(self):
        group = self.create_group_box("Advanced camera")
        layout = QVBoxLayout(group)
        tabs = _CameraTabs()
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(tabs)
        self.status = QLabel("Mouse input or Escape stops automatic camera motion.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        motion = QWidget()
        form = AlignedFormLayout(motion)
        tabs.addTab(motion, "Navigate")
        self.smooth = QCheckBox("Smooth transitions")
        self.smooth.setChecked(True)
        self.duration = self.spin(0, 30, 0.8, " s")
        form.addRow(self.smooth, self.duration)
        self.smooth.toggled.connect(self.settings)
        self.duration.valueChanged.connect(self.settings)
        self.focus_kind = QComboBox()
        for title, data in [
            ("Selected actor", "actor"),
            ("Selected Tx–Rx pair", "pair"),
            ("Inspected coverage cell", "coverage"),
            ("Inspected sensing cell", "sensing"),
            ("Selected sensing report", "report"),
        ]:
            if data != "actor" and not getattr(self.parent, "supports_recorded_results", True):
                continue
            self.focus_kind.addItem(title, data)
        focus = QPushButton("Focus")
        focus.clicked.connect(lambda: self.run(self.focus_selection))
        form.addRow(self.focus_kind, focus)
        back = QPushButton("Previous view")
        back.clicked.connect(lambda: self.run(lambda t: t.previous()))
        form.addRow(back)
        self.orbit_kind = QComboBox()
        self.orbit_kind.addItems(
            ["Scene center", "Selected actor", "Current look-at", "World point"]
        )
        form.addRow("Orbit around", self.orbit_kind)
        point = QWidget()
        row = QHBoxLayout(point)
        row.setContentsMargins(0, 0, 0, 0)
        self.point = [self.spin(-1e6, 1e6, 0, " m") for _ in range(3)]
        for name, spin in zip("XYZ", self.point):
            spin.setPrefix(f"{name}: ")
            row.addWidget(spin)
        form.addRow(point)
        point.setVisible(False)
        self.orbit_kind.currentIndexChanged.connect(lambda i: point.setVisible(i == 3))
        self.speed = self.spin(-90, 90, 8, " °/s")
        self.speed.setToolTip("Positive: counterclockwise from above. Negative: clockwise.")
        form.addRow("Orbit speed", self.speed)
        orbit_button = QPushButton("Start orbit")
        orbit_button.clicked.connect(lambda: self.run(self.start_orbit))
        stop = QPushButton("Stop camera motion")
        stop.clicked.connect(lambda: self.run(lambda t: t.stop()))
        form.addRow(orbit_button, stop)

        framing = _FramingPage(self.refresh_follow_framing)
        form = AlignedFormLayout(framing)
        tabs.addTab(framing, "Framing")
        self.framing_status = QLabel()
        self.framing_status.setWordWrap(True)
        form.addRow(self.framing_status)
        self.heading = QComboBox()
        self.heading.addItem("Keep world heading", "world")
        self.heading.addItem("Behind actor (chase)", "chase")
        self.heading.addItem("Behind motion (direction of travel)", "motion")
        self.heading.addItem("Fixed position, track actor", "fixed")
        self.heading.setToolTip(
            "Behind actor uses recorded orientation. Behind motion follows the horizontal path, "
            "independently of mesh orientation, and holds its heading when stopped."
        )
        form.addRow("Follow behavior", self.heading)
        self.heading.currentIndexChanged.connect(self.framing_settings)
        self.follow_offset = [self.spin(-1e6, 1e6, value, " m") for value in (10, 0, 3.5)]
        for name, spin in zip(
            ("Distance behind", "Side offset (left)", "Height above actor"), self.follow_offset
        ):
            form.addRow(name, spin)
            spin.valueChanged.connect(self.follow_offset_changed)
            spin.setToolTip(
                "Behind actor uses recorded heading; Behind motion uses travel direction; "
                "world-heading Follow uses a fixed heading. "
                "Negative values place the camera ahead, to the right, or below."
            )
        behind = QPushButton("Frame behind selected actor")
        behind.setToolTip("Use the selected Behind actor or Behind motion behavior.")
        behind.clicked.connect(lambda: self.run(self.behind))
        fixed = QPushButton("Track actor from here")
        fixed.setToolTip(
            "Keep this exact camera position and a level horizon; turn to watch the actor during playback."
        )
        fixed.clicked.connect(lambda: self.run(self.track_from_here))
        form.addRow(behind, fixed)
        reset_behind = QPushButton("Reset directly behind actor")
        reset_behind.setToolTip(
            "Keep the following distance and zero side and height offsets. "
            "Uses travel direction when Behind motion is selected."
        )
        reset_behind.clicked.connect(lambda: self.run(self.reset_behind))
        form.addRow(reset_behind)
        self.true_pov = QCheckBox("True POV (zero offsets; hide selected actor)")
        self.true_pov.setChecked(True)
        self.true_pov.setToolTip(
            "Uncheck to use an offset camera and restore the actor's normal visibility. "
            "Check to return to the actor position. Previous offsets are kept; lens settings are unchanged."
        )
        self.true_pov.toggled.connect(self.toggle_true_pov)
        form.addRow(self.true_pov)
        self.offset = [self.spin(-1000, 1000, 0, " m") for _ in range(3)]
        for name, spin in zip(
            ("POV forward offset", "POV left offset", "POV up offset"), self.offset
        ):
            form.addRow(name, spin)
            spin.valueChanged.connect(self.framing_settings)
        self.horizon = QCheckBox("Keep POV horizon level (world Z-up)")
        self.horizon.setChecked(True)
        self.horizon.toggled.connect(self.framing_settings)
        form.addRow(self.horizon)
        hint = QLabel(
            "Changes apply immediately, including while paused. Fixed tracking keeps the camera position; mouse navigation remains available."
        )
        hint.setWordWrap(True)
        form.addRow(hint)

        views = QWidget()
        form = AlignedFormLayout(views)
        tabs.addTab(views, "Views & sequence")
        self.name = QLineEdit()
        self.name.setPlaceholderText("View name")
        self.relative = QCheckBox("Translate with selected actor")
        form.addRow(self.name)
        form.addRow(self.relative)
        save = QPushButton("Save / replace named view")
        save.clicked.connect(lambda: self.run(self.save_view))
        form.addRow(save)
        self.names = QComboBox()
        recall = QPushButton("Go to view")
        recall.clicked.connect(lambda: self.run(lambda t: t.recall_view(self.names.currentText())))
        form.addRow(self.names, recall)
        remove = QPushButton("Delete view")
        remove.clicked.connect(lambda: self.run(self.delete_view))
        form.addRow(remove)
        self.hold = self.spin(0, 3600, 2, " s")
        self.move = self.spin(0, 3600, 1, " s")
        form.addRow("Hold view", self.hold)
        form.addRow("Transition into view", self.move)
        add = QPushButton("Append view to sequence")
        add.clicked.connect(lambda: self.run(self.add_shot))
        form.addRow(add)
        self.shots = QListWidget()
        self.shots.setMaximumHeight(115)
        form.addRow(self.shots)
        actions = QWidget()
        row = QHBoxLayout(actions)
        row.setContentsMargins(0, 0, 0, 0)
        for label, delta in [("Up", -1), ("Down", 1), ("Remove", 0)]:
            button = QPushButton(label)
            button.clicked.connect(lambda _, d=delta: self.run(lambda t: self.edit_shot(t, d)))
            row.addWidget(button)
        form.addRow(actions)
        self.loop = QCheckBox("Loop camera sequence")
        play = QPushButton("Play sequence")
        play.clicked.connect(lambda: self.run(lambda t: t.start_sequence(self.loop.isChecked())))
        form.addRow(self.loop, play)
        stop_sequence = QPushButton("Stop sequence")
        stop_sequence.clicked.connect(lambda: self.run(lambda t: t.stop()))
        form.addRow(stop_sequence)
        export = QPushButton("Export camera sequence as MP4 / GIF…")
        export.clicked.connect(self.export_video)
        export.setVisible(getattr(self.parent, "supports_recorded_results", True))
        form.addRow(export)
        hint = QLabel(
            "Data playback stays independent. Video export can hold one data frame or step through saved frames; it never interpolates scientific results."
        )
        hint.setWordWrap(True)
        if not getattr(self.parent, "supports_recorded_results", True):
            hint.setText("Camera sequences and actor playback can run independently.")
        form.addRow(hint)
        load = QPushButton("Import camera JSON…")
        dump = QPushButton("Export camera JSON…")
        load.clicked.connect(lambda: self.run(lambda t: self.json_file(t, False)))
        dump.clicked.connect(lambda: self.run(lambda t: self.json_file(t, True)))
        form.addRow(load, dump)

        # A long sequence editor should not impose its height on Navigate.
        def fit_current_page(index):
            for i in range(tabs.count()):
                tabs.widget(i).setSizePolicy(
                    QSizePolicy.Expanding,
                    QSizePolicy.Preferred if i == index else QSizePolicy.Ignored,
                )
            tabs.updateGeometry()
            group.updateGeometry()

        tabs.currentChanged.connect(fit_current_page)
        self.refresh()
        fit_current_page(0)
        return group

    @staticmethod
    def spin(low, high, value, suffix):
        widget = QDoubleSpinBox()
        widget.setRange(low, high)
        widget.setDecimals(2)
        widget.setValue(value)
        widget.setSuffix(suffix)
        return widget

    def tools(self):
        tools = self.parent.camera_controller.presentation()
        if tools is not None and tools is not getattr(self, "_bound_tools", None):
            previous = getattr(self, "_bound_tools", None)
            if previous is not None:
                previous.status_changed.disconnect(self.status.setText)
                previous.views_changed.disconnect(self.refresh)
            tools.status_changed.connect(self.status.setText)
            tools.views_changed.connect(self.refresh)
            self._bound_tools = tools
            self.status.setText(tools.message)
        return tools

    def run(self, action):
        """Report unavailable selections inline without interrupting navigation."""
        try:
            tools = self.tools()
            if tools is None:
                raise ValueError("Camera presentation requires pygfx")
            tools.bind_input()
            action(tools)
            self.status.setText(tools.message)
            self.refresh()
        except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
            self.status.setText(str(exc))

    def settings(self, *_):
        if getattr(self, "_syncing", False):
            return
        tools = self.tools()
        tools.smooth = self.smooth.isChecked()
        tools.duration = self.duration.value()

    def framing_settings(self, *_):
        """Only editable controls for the active mode change camera framing."""
        if getattr(self, "_syncing", False):
            return

        def apply(tools):
            mode = getattr(self.parent.app_state, "camera_mode", None)
            if mode == "follow":
                heading = self.heading.currentData()
                if heading != tools.follow_heading:
                    tools.remember_follow_framing()
                    tools.last_yaw = None
                    tools.follow_heading = heading
            elif mode == "pov":
                if not tools.true_pov:
                    tools.pov_offset = [spin.value() for spin in self.offset]
                tools.horizon_lock = self.horizon.isChecked()
            tools.apply_framing()

        self.run(apply)

    def toggle_true_pov(self, checked):
        """Switch camera origin and actor visibility without losing offset values."""
        if getattr(self, "_syncing", False):
            return

        def apply(tools):
            tools.true_pov = bool(checked)
            tools.apply_framing()

        self.run(apply)

    def follow_offset_changed(self, *_):
        if getattr(self, "_syncing", False):
            return

        def apply(tools):
            offset = [spin.value() for spin in self.follow_offset]
            if np.linalg.norm(offset) < 0.01:
                raise ValueError(
                    "Follow offsets must place the camera at least 0.01 m from the actor"
                )
            tools.follow_offset = offset
            tools.apply_framing(offset_changed=True)

        self.run(apply)

    def refresh_follow_framing(self):
        """Reflect mouse movement without overwriting an in-progress numeric edit."""
        if getattr(self.parent.app_state, "camera_mode", None) != "follow":
            return
        if any(spin.hasFocus() for spin in self.follow_offset):
            return
        tools = self.tools()
        if tools is None or tools.exporting:
            return
        tools.remember_follow_framing()
        if tools.follow_offset is not None:
            self._syncing = True
            try:
                for spin, value in zip(self.follow_offset, tools.follow_offset):
                    spin.setValue(value)
            finally:
                self._syncing = False

    def sync_from_state(self, state):
        """Show which framing controls affect the current camera mode."""
        mode = getattr(state, "camera_mode", "overview")
        self.heading.setEnabled(mode == "follow")
        for widget in self.follow_offset:
            widget.setEnabled(mode == "follow" and self.heading.currentData() != "fixed")
        for widget in [self.true_pov, self.horizon]:
            widget.setEnabled(mode == "pov")
        for widget in self.offset:
            widget.setEnabled(mode == "pov" and not self.true_pov.isChecked())
        messages = {
            "overview": "Overview: select Follow or POV in the Camera panel, or use Frame behind below.",
            "follow": "Follow active. Field of view is in the Camera panel; POV offsets require POV.",
            "pov": "POV active. Offsets apply now; field of view is in the Camera panel. Mouse navigation exits POV.",
        }
        self.framing_status.setText(messages.get(mode, messages["overview"]))

    def refresh(self):
        tools = self.tools()
        if tools is None:
            return
        self._syncing = True
        try:
            self.smooth.setChecked(tools.smooth)
            self.duration.setValue(tools.duration)
            self.heading.setCurrentIndex(self.heading.findData(tools.follow_heading))
            if tools.follow_offset is not None:
                for spin, value in zip(self.follow_offset, tools.follow_offset):
                    spin.setValue(value)
            self.true_pov.setChecked(tools.true_pov)
            self.horizon.setChecked(tools.horizon_lock)
            effective_offset = [0.0, 0.0, 0.0] if tools.true_pov else tools.pov_offset
            for spin, value in zip(self.offset, effective_offset):
                spin.setValue(value)
            selected = self.names.currentText()
            self.names.clear()
            self.names.addItems(list(tools.views))
            if selected in tools.views:
                self.names.setCurrentText(selected)
            index = self.shots.currentRow()
            self.shots.clear()
            for i, shot in enumerate(tools.shots):
                self.shots.addItem(
                    f"{i+1}. {shot['view']} — hold {shot['hold']:g} s, move {shot['move'] if i else 0:g} s"
                )
            self.shots.setCurrentRow(min(index, len(tools.shots) - 1))
        finally:
            self._syncing = False
        self.sync_from_state(getattr(self.parent, "app_state", None))

    def focus_selection(self, tools):
        kind = self.focus_kind.currentData()
        query = self.parent.camera_controller.scene_query
        radius = None
        if kind == "actor":
            point = query.get_focus_position()
        elif kind == "pair":
            state = self.parent.app_state
            if not isinstance(state.selected_tx, int) or not isinstance(state.selected_rx, int):
                raise ValueError("Select one transmitter and one receiver in the link controls")
            tx = query.resolve_tx_position(state.selected_tx)
            rx = query.resolve_rx_position(state.selected_rx)
            if tx is None or rx is None:
                raise ValueError("Selected link positions are unavailable")
            point = (np.asarray(tx) + rx) / 2
            radius = max(5, np.linalg.norm(np.asarray(tx) - rx) * 1.2)
        elif kind == "coverage":
            window = getattr(self.parent, "coverage_volume_window", None)
            if window is None or window.model is None or window.selected_indices is None:
                raise ValueError("Inspect a coverage cell first")
            point = [axis[i] for axis, i in zip(window.model.axes, window.selected_indices)]
            radius = 15
        else:
            window = getattr(self.parent, "sensing_volume_window", None)
            if window is None or window.reader is None or not window.inspect.isChecked():
                panels = getattr(getattr(self.parent, "ui_manager", None), "panels", {})
                resolve = getattr(panels.get("sensing"), "selected_camera_position", None)
                if kind == "report" and callable(resolve):
                    tools.focus(resolve(), radius=15)
                    return
                raise ValueError("Inspect a sensing cell first")
            if kind == "report":
                workflow = window.workflow
                point = next(
                    (
                        row["position_m"]
                        for row in workflow.rows
                        if row["detection_id"] == workflow.selected_report
                    ),
                    None,
                )
            else:
                trial = window.reader.cells.get(window._cell_key())
                point = trial["position_m"] if trial else None
            radius = 15
        tools.focus(point, radius=radius)

    def start_orbit(self, tools):
        mode = self.orbit_kind.currentIndex()
        center = None
        if mode == 0:
            bounds = tools.owner.scene_query.compute_scene_bounds(scope="whole")
            center = bounds[0] if bounds else None
        elif mode == 2:
            state = tools.renderer.get_camera_state()
            center = state.lookat if state is not None else None
        elif mode == 3:
            center = [spin.value() for spin in self.point]
        tools.start_orbit(center, self.speed.value(), actor=mode == 1)

    def behind(self, tools):
        position, orientation, _ = (
            tools.owner.scene_query.get_entity_position_orientation_and_info()
        )
        motion = self.heading.currentData() == "motion"
        if position is None or (orientation is None and not motion):
            raise ValueError("The selected actor needs a position and orientation")
        tools.follow_heading = "motion" if motion else "chase"
        self.parent.ui_controller.handle_camera_mode_changed("follow", True)

    def track_from_here(self, tools):
        if tools.owner.scene_query.get_focus_position() is None:
            raise ValueError("Select an available actor in the Camera panel first")
        tools.follow_heading = "fixed"
        self.parent.ui_controller.handle_camera_mode_changed("follow", True)

    def reset_behind(self, tools):
        """Restore centered, level chase framing without discarding follow distance."""
        if tools.exporting:
            raise ValueError("Wait for camera export to finish before changing framing")
        position = tools.owner.scene_query.get_focus_position()
        state = tools.renderer.get_camera_state()
        if position is None or state is None:
            raise ValueError("Select an available actor in the Camera panel first")
        tools.remember_follow_framing()
        if (
            getattr(self.parent.app_state, "camera_mode", None) == "follow"
            and tools.follow_heading != "fixed"
            and tools.follow_offset is not None
        ):
            distance = abs(tools.follow_offset[0])
        else:
            distance = float(np.linalg.norm(np.asarray(state.eye) - position))
        self.behind(tools)
        tools.follow_offset = [max(distance, 0.01), 0.0, 0.0]
        tools.apply_framing(offset_changed=True)

    def save_view(self, tools):
        tools.save_view(self.name.text(), relative=self.relative.isChecked())
        self.refresh()
        self.names.setCurrentText(self.name.text().strip())

    def delete_view(self, tools):
        name = self.names.currentText()
        if any(shot["view"] == name for shot in tools.shots):
            raise ValueError("Remove this view's sequence shots before deleting it")
        tools.views.pop(name, None)

    def add_shot(self, tools):
        if self.names.currentText() not in tools.views:
            raise ValueError("Save a named view first")
        tools.shots.append(
            {"view": self.names.currentText(), "hold": self.hold.value(), "move": self.move.value()}
        )

    def edit_shot(self, tools, delta):
        i = self.shots.currentRow()
        if not 0 <= i < len(tools.shots):
            return
        tools.stop()
        if delta == 0:
            tools.shots.pop(i)
        elif 0 <= i + delta < len(tools.shots):
            tools.shots[i], tools.shots[i + delta] = tools.shots[i + delta], tools.shots[i]
            self.shots.setCurrentRow(i + delta)

    def json_file(self, tools, save):
        dialog = QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName
        name, _ = dialog(
            self.parent, "Camera views and sequence", "camera_views.json", "JSON (*.json)"
        )
        if not name:
            return
        if save:
            Path(name).write_text(json.dumps(tools.payload(), indent=2) + "\n", encoding="utf8")
        else:
            tools.restore(json.loads(Path(name).read_text(encoding="utf8")))

    def export_video(self):
        from .video_export_dialog import VideoExportDialog

        dialog = VideoExportDialog(self.parent, max(1, self.parent.total_animation_steps))
        dialog.camera_sequence_cb.setChecked(True)
        dialog.exec()

    def cleanup(self):
        tools = getattr(getattr(self.parent, "camera_controller", None), "_presentation", None)
        if tools is not None:
            tools.close()
