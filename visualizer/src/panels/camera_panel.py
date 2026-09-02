"""Camera-mode and preset widgets for the visualizer shell.

The panel exposes renderer-neutral controls for preset views, Overview/Follow/
POV mode selection, tracked entity selection, POV look axis, and optional
backend-provided fly/minimap features. Controllers decide what each selection
does for the active renderer.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from .base import BasePanel


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
            btn.setFixedSize(50, 22)
            btn.setToolTip(f"Set camera to {name.lower()} view")
            self.widgets[key] = btn
            row0.addWidget(btn)

        self.widgets["reset_camera_btn"] = QPushButton("Reset")
        self.widgets["reset_camera_btn"].setMinimumWidth(84)
        self.widgets["reset_camera_btn"].setFixedHeight(22)
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
            ("Forward", "forward"),
            ("+X (East)", "x"),
            ("+Y (North)", "y"),
            ("+Z (Up)", "z"),
            ("-X (West)", "-x"),
            ("-Y (South)", "-y"),
            ("-Z (Down)", "-z"),
        ]:
            pov_combo.addItem(text, data)
        pov_combo.setToolTip("Select look direction in POV mode")
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
        save_btn.setFixedSize(46, 24)
        save_btn.setCheckable(True)
        save_btn.setToolTip("Click, then choose a view number to save the current view")
        self.widgets["camera_preset_save_btn"] = save_btn
        row4.addWidget(save_btn)

        self.widgets["camera_preset_buttons"] = []
        for i in range(1, 5):
            btn = QPushButton(str(i))
            btn.setFixedSize(24, 24)
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

        return group

    def set_compact_mode(self, compact: bool) -> None:
        """Tighten the saved-view row when the window is narrow."""
        if hasattr(self, "_saved_views_label"):
            self._saved_views_label.setVisible(not compact)
