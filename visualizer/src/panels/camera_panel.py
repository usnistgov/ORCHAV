"""Camera-mode and preset widgets for the visualizer shell.

The panel exposes renderer-neutral controls for preset views, Overview/Follow/
POV mode selection, tracked entity selection, POV look axis, and optional
backend-provided fly/minimap features. Controllers decide what each selection
does for the active renderer.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack

from PySide6.QtCore import QSignalBlocker, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from .base import BasePanel

logger = logging.getLogger(__name__)


class _FieldOfViewSpinBox(QDoubleSpinBox):
    """Read the active lens while visible, without interrupting numeric edits."""

    def __init__(self, refresh):
        super().__init__()
        self._refresh = refresh
        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(refresh)

    def showEvent(self, event):
        super().showEvent(event)
        self.timer.start()
        self._refresh()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)


class CameraControlPanel(BasePanel):
    """Create camera widgets and expose them through stable widget keys."""

    def create_panel(self) -> QGroupBox:
        """Build the camera panel without binding controller behavior."""
        group = self.create_group_box("Camera")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # Quick-view buttons are wired by widget key to camera-controller actions.
        row0 = QHBoxLayout()
        row0.setSpacing(4)
        row0.setContentsMargins(0, 0, 0, 0)

        for name, key in [
            ("Top", "view_top_btn"),
            ("Side", "view_side_btn"),
            ("Iso", "view_iso_btn"),
            ("Front", "view_front_btn"),
        ]:
            btn = QPushButton(name)
            btn.setMinimumWidth(50)
            btn.setToolTip(f"Set camera to {name.lower()} view")
            self.widgets[key] = btn
            row0.addWidget(btn)

        self.widgets["reset_camera_btn"] = QPushButton("Reset")
        self.widgets["reset_camera_btn"].setMinimumWidth(84)
        self.widgets["reset_camera_btn"].setToolTip(
            "Reset to the scenario camera default, or fit the whole scene"
        )
        row0.addWidget(self.widgets["reset_camera_btn"])

        row0.addStretch()
        layout.addLayout(row0)

        # Mode radios remain renderer-neutral; controller code handles support checks.
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.setContentsMargins(0, 0, 0, 0)

        self.widgets["camera_mode_group"] = QButtonGroup(group)

        for label, key, tooltip in [
            ("Overview", "overview_mode_rb", "Free camera control with mouse"),
            ("Follow", "follow_mode_rb", "Auto-follow selected entity (camera looks at entity)"),
            (
                "POV",
                "pov_mode_rb",
                "First-person view from selected entity (camera at entity position)",
            ),
        ]:
            rb = QRadioButton(label)
            rb.setToolTip(tooltip)
            self.widgets["camera_mode_group"].addButton(rb)
            self.widgets[key] = rb
            row2.addWidget(rb)

        self.widgets["overview_mode_rb"].setChecked(True)

        # Fly mode starts hidden because only some renderer backends expose it.
        fly_cb = QCheckBox("Fly")
        fly_cb.setToolTip(
            "Enable native fly camera mode in Overview (WASD / arrow keys to move, mouse to look).\n"
            "Follow and POV disable Fly automatically."
        )
        fly_cb.setVisible(False)
        self.widgets["fly_mode_cb"] = fly_cb
        row2.addWidget(fly_cb)

        row2.addStretch()
        layout.addLayout(row2)

        # Follow and POV share target selection, but POV owns the look-axis selector.
        row3 = QHBoxLayout()
        row3.setSpacing(8)
        row3.setContentsMargins(0, 0, 0, 0)

        track_widget = QWidget()
        track_layout = QHBoxLayout(track_widget)
        track_layout.setSpacing(4)
        track_layout.setContentsMargins(0, 0, 0, 0)

        track_layout.addWidget(QLabel("Track:"))
        self.widgets["target_focus_dropdown"] = QComboBox()
        self.widgets["target_focus_dropdown"].addItem("Auto (First Target)", {"type": "auto"})
        self.widgets["target_focus_dropdown"].setStyleSheet(
            "QComboBox { padding: 2px; min-width: 100px; }"
        )
        self.widgets["target_focus_dropdown"].setToolTip(
            "Select entity to track in Follow/POV mode"
        )
        track_layout.addWidget(self.widgets["target_focus_dropdown"])

        self.widgets["track_group"] = track_widget
        track_widget.setVisible(False)
        row3.addWidget(track_widget)

        pov_container = QWidget()
        pov_layout = QHBoxLayout(pov_container)
        pov_layout.setSpacing(4)
        pov_layout.setContentsMargins(0, 0, 0, 0)

        pov_layout.addWidget(QLabel("Look:"))
        pov_combo = QComboBox()
        pov_combo.setStyleSheet("QComboBox { padding: 2px; min-width: 120px; }")
        for text, data in [
            ("Motion", "motion"),
            ("Forward", "forward"),
            ("+X (East)", "x"),
            ("+Y (North)", "y"),
            ("+Z (Up)", "z"),
            ("-X (West)", "-x"),
            ("-Y (South)", "-y"),
            ("-Z (Down)", "-z"),
        ]:
            pov_combo.addItem(text, data)
        pov_combo.setToolTip(
            "Motion looks along the recorded travel direction, including climbs and descents.\n"
            "It holds the last direction when stopped; Forward uses the actor's recorded orientation."
        )
        self.widgets["pov_axis_combo"] = pov_combo
        pov_layout.addWidget(pov_combo)

        self.widgets["pov_axis_container"] = pov_container
        pov_container.setVisible(False)
        row3.addWidget(pov_container)

        row3.addStretch()
        layout.addLayout(row3)

        # Preset slots are plain numbered buttons; save mode is tracked by the controller.
        self._saved_views_row = QWidget()
        row4 = QHBoxLayout(self._saved_views_row)
        row4.setSpacing(4)
        row4.setContentsMargins(0, 0, 0, 0)

        self._saved_views_label = QLabel("Views:")
        row4.addWidget(self._saved_views_label)

        save_btn = QPushButton("Save")
        save_btn.setMinimumSize(46, 24)
        save_btn.setCheckable(True)
        save_btn.setToolTip("Click, then choose a view number to save the current view")
        self.widgets["camera_preset_save_btn"] = save_btn
        row4.addWidget(save_btn)

        self.widgets["camera_preset_buttons"] = []
        for i in range(1, 5):
            btn = QPushButton(str(i))
            btn.setMinimumSize(24, 24)
            btn.setStyleSheet("QPushButton { font-size: 10px; padding: 2px; }")
            btn.setCheckable(False)
            btn.setToolTip(f"View {i}: empty. Click Save, then {i} to store current view")
            btn.setProperty("preset_num", i)
            row4.addWidget(btn)
            self.widgets["camera_preset_buttons"].append(btn)

        minimap_cb = QCheckBox("Minimap")
        minimap_cb.setMinimumWidth(110)
        minimap_cb.setToolTip("Show the renderer-provided top-view minimap inset.")
        minimap_cb.setVisible(False)
        self.widgets["camera_minimap_cb"] = minimap_cb
        row4.addWidget(minimap_cb)

        row4.addStretch()
        layout.addWidget(self._saved_views_row)

        lens_row = QHBoxLayout()
        lens_row.setSpacing(6)
        lens_row.setContentsMargins(0, 0, 0, 0)
        fov_label = QLabel("FOV:")
        fov = _FieldOfViewSpinBox(self.refresh_field_of_view)
        fov.setRange(5, 150)
        fov.setDecimals(1)
        fov.setSingleStep(5)
        fov.setSuffix(" °")
        fov.setValue(60)
        fov.setKeyboardTracking(False)
        fov.setToolTip(
            "Camera field of view in Overview, Follow and POV.\n"
            "Smaller angles narrow the view; larger angles widen it without moving the camera."
        )
        fov_label.setBuddy(fov)
        self.widgets["camera_fov_spin"] = fov
        lens_row.addWidget(fov_label)
        lens_row.addWidget(fov)
        advanced = QCheckBox("Advanced camera")
        advanced.setToolTip("Show camera motion, framing and saved sequences in the Rendering tab")
        advanced.setVisible(
            getattr(getattr(self.parent, "renderer", None), "renderer_id", None) == "pygfx"
        )
        self.widgets["advanced_camera_cb"] = advanced
        lens_row.addWidget(advanced)
        lens_row.addStretch()
        layout.addLayout(lens_row)
        views_row = QHBoxLayout()
        target_view = QPushButton("Actor view +")
        target_view.setToolTip(
            "Open another camera following a target, TX or RX; drag to orbit and wheel to zoom"
        )
        target_view.setVisible(
            callable(getattr(getattr(self.parent, "renderer", None), "open_target_view", None))
        )
        target_view.clicked.connect(self._open_target_view)
        self.widgets["camera_target_view_btn"] = target_view
        views_row.addWidget(target_view)
        live_figure = QPushButton("Live figure +")
        live_figure.setToolTip("Show up to four live channel or sensing figures beside the scene")
        live_figure.clicked.connect(self._open_live_figure)
        self.widgets["live_figure_btn"] = live_figure
        views_row.addWidget(live_figure)
        views_row.addStretch()
        layout.addLayout(views_row)

        self.sync_from_state(getattr(self.parent, "app_state", None))
        return group

    def _open_live_figure(self) -> None:
        """Create optional figure placement only when explicitly requested."""
        from ..widgets.live_figure_dock import LiveFigureManager

        manager = getattr(self.parent, "live_figure_manager", None)
        if manager is None:
            manager = LiveFigureManager(self.parent)
            self.parent.live_figure_manager = manager
        try:
            manager.add()
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            logger.exception("Could not open a live figure")
            self.parent.statusBar().showMessage(f"Live figure unavailable: {exc}", 6000)

    def _open_target_view(self) -> None:
        """Request a renderer-owned camera without importing a backend into the panel."""
        callback = getattr(getattr(self.parent, "renderer", None), "open_target_view", None)
        if not callable(callback):
            return
        try:
            if callback() is not None:
                return
            message = "Open a scenario with a target, TX or RX first."
        except Exception as exc:
            logger.exception("Could not open an actor view")
            message = f"Actor view unavailable: {exc}"
        status_bar = getattr(self.parent, "statusBar", None)
        if callable(status_bar):
            status_bar().showMessage(message, 6000)

    def refresh_field_of_view(self) -> None:
        """Mirror saved views, resets and animated lenses without driving the camera."""
        fov = self.widgets["camera_fov_spin"]
        renderer = getattr(self.parent, "renderer", None)
        if fov.hasFocus() or renderer is None:
            return
        state = renderer.get_camera_state()
        if state is not None:
            with QSignalBlocker(fov):
                fov.setValue(state.fov_deg)

    def sync_from_state(self, state: object | None) -> None:
        """Reflect restored camera intent without moving the camera or emitting actions."""
        if state is None or "camera_mode_group" not in self.widgets:
            return
        mode = getattr(state, "camera_mode", "overview")
        buttons = self.widgets["camera_mode_group"].buttons()
        # Checking one exclusive radio also unchecks another. Block both ends.
        with ExitStack() as stack:
            for button in buttons:
                stack.enter_context(QSignalBlocker(button))
            button = self.widgets.get(f"{mode}_mode_rb")
            if button is not None:
                button.setChecked(True)
        advanced = self.widgets.get("advanced_camera_cb")
        self.widgets["track_group"].setVisible(
            mode in ("follow", "pov") or bool(advanced and advanced.isChecked())
        )
        self.widgets["pov_axis_container"].setVisible(mode == "pov")
        combo = self.widgets["pov_axis_combo"]
        with QSignalBlocker(combo):
            combo.setCurrentIndex(combo.findData(getattr(state, "pov_axis", "motion")))

    def set_compact_mode(self, compact: bool) -> None:
        """Tighten the saved-view row when the window is narrow."""
        if hasattr(self, "_saved_views_label"):
            self._saved_views_label.setVisible(not compact)
