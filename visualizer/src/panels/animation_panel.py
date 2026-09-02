"""Playback and frame-navigation controls for frame-backed scenarios.

The panel builds transport buttons, the timeline slider, frame-number input,
and cadence controls. It owns widget registration and live gRPC auto-request
flags only; ``AnimationController`` and app workflow code perform playback,
stepping, and scene-only mode transitions.
"""

from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from shared.logging import get_logger

from ..playback import (
    DEFAULT_FIXED_PLAYBACK_FPS,
    MAX_PLAYBACK_FPS,
    MIN_PLAYBACK_FPS,
    PlaybackMode,
    normalize_playback_mode,
)
from .base import BasePanel

logger = get_logger("orchav.animation_panel")

SETTINGS_PLAYBACK_MODE_KEY = "playback/mode"
SETTINGS_FIXED_PLAYBACK_FPS_KEY = "playback/fixed_fps"


class AnimationControlsPanel(BasePanel):
    """Create transport/timeline widgets consumed by playback controllers."""

    def __init__(
        self,
        parent_widget,
        total_steps: int = 60,
        *,
        settings: Any = None,
    ):
        """Store timeline length and live-stream request defaults."""
        super().__init__(parent_widget)
        self.total_steps = total_steps
        self.is_online_mode = False
        self.auto_request_frames = True
        self._settings = settings
        self._scene_only_mode = False

    def create_panel(self) -> QGroupBox:
        """Create and return the animation controls panel."""
        group = self.create_group_box("Animation")

        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(self._create_step_slider_row())
        layout.addLayout(self._create_animation_buttons_row())
        layout.addLayout(self._create_speed_controls_row())
        return group

    def set_online_mode(self, is_online: bool) -> None:
        """Record whether frame navigation should request live gRPC frames."""
        self.is_online_mode = is_online
        logger.info("Animation panel: live gRPC mode = %s", is_online)

    def is_in_online_mode(self) -> bool:
        """Return whether live gRPC frame-request semantics are active."""
        return self.is_online_mode

    def set_auto_request_frames(self, auto_request: bool) -> None:
        """Enable or disable opportunistic frame requests during live playback."""
        self.auto_request_frames = auto_request
        logger.info("Animation panel: Auto-request frames = %s", auto_request)

    def request_frame_if_needed(self, frame_idx: int) -> bool:
        """Request a missing live frame when online auto-request mode is enabled."""
        if not self.is_online_mode or not self.auto_request_frames:
            return False
        if not hasattr(self.parent, "frame_source") or not self.parent.frame_source:
            return False
        try:
            available_frames = self.parent.frame_source.list_frames()
            if frame_idx in available_frames:
                return True
            logger.info("Animation panel: Auto-requesting frame %s", frame_idx)
            success = self.parent.frame_source.request_frame(frame_idx)
            if success:
                logger.info("Animation panel: Successfully requested frame %s", frame_idx)
                return True
            logger.warning("Animation panel: Failed to request frame %s", frame_idx)
            return False
        except (
            OSError,
            RuntimeError,
            AttributeError,
        ) as exc:  # pragma: no cover - defensive logging
            logger.error("Animation panel: Error requesting frame %s: %s", frame_idx, exc)
            return False

    def _create_step_slider_row(self) -> QHBoxLayout:
        """Create zero-based slider plus one-based frame-number controls."""
        row = QHBoxLayout()
        row.setSpacing(8)
        row.setContentsMargins(4, 0, 4, 0)

        step_label = QLabel("Step:")
        step_label.setStyleSheet("font-weight: bold; min-width: 40px;")
        row.addWidget(step_label)

        self.widgets["step_slider"] = QSlider(Qt.Horizontal)
        self.widgets["step_slider"].setMinimum(0)
        self.widgets["step_slider"].setMaximum(self.total_steps - 1)
        self.widgets["step_slider"].setToolTip("Frame slider - drag to change frame")
        row.addWidget(self.widgets["step_slider"], 1)

        self.widgets["step_label"] = QLabel("1")
        self.widgets["step_label"].setStyleSheet("font-weight: bold; min-width: 40px;")
        row.addWidget(self.widgets["step_label"])

        self.widgets["frame_input"] = QSpinBox()
        self.widgets["frame_input"].setMinimum(1)
        self.widgets["frame_input"].setMaximum(self.total_steps)
        self.widgets["frame_input"].setValue(1)
        self.widgets["frame_input"].setToolTip("Enter frame number to jump")
        self.widgets["frame_input"].setMinimumWidth(70)
        row.addWidget(self.widgets["frame_input"])

        self.widgets["total_steps_label"] = QLabel(f"/ {self.total_steps}")
        row.addWidget(self.widgets["total_steps_label"])
        return row

    def _create_animation_buttons_row(self) -> QHBoxLayout:
        """Create transport buttons and loop toggle using stable widget keys."""
        row = QHBoxLayout()
        row.setSpacing(12)
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch()

        from ..widgets.transport_button import TransportButton

        self.widgets["prev_btn"] = TransportButton("prev")
        self.widgets["prev_btn"].setToolTip("Previous Frame")
        row.addWidget(self.widgets["prev_btn"])

        self.widgets["reverse_play_btn"] = TransportButton("rev_play")
        self.widgets["reverse_play_btn"].setToolTip("Play Backward")
        row.addWidget(self.widgets["reverse_play_btn"])

        self.widgets["play_btn"] = TransportButton("play")
        self.widgets["play_btn"].setToolTip("Play/Pause Animation")
        row.addWidget(self.widgets["play_btn"])

        self.widgets["reset_btn"] = TransportButton("stop")
        self.widgets["reset_btn"].setToolTip("Stop Animation")
        row.addWidget(self.widgets["reset_btn"])

        self.widgets["next_btn"] = TransportButton("next")
        self.widgets["next_btn"].setToolTip("Next Frame")
        row.addWidget(self.widgets["next_btn"])

        self.widgets["loop_cb"] = QCheckBox("Loop")
        self.widgets["loop_cb"].setChecked(True)
        self.widgets["loop_cb"].setToolTip(
            "Wrap playback and frame stepping at sequence boundaries"
        )
        row.addWidget(self.widgets["loop_cb"])

        row.addStretch()
        return row

    def _create_speed_controls_row(self) -> QHBoxLayout:
        """Create explicit playback policy, fixed-rate, and frame-stride controls."""
        row = QHBoxLayout()
        row.setSpacing(15)
        row.setContentsMargins(0, 0, 0, 0)

        playback_label = QLabel("Playback:")
        playback_label.setStyleSheet("font-weight: bold; min-width: 55px;")
        row.addWidget(playback_label)

        mode_combo = QComboBox()
        mode_combo.addItem("Real time", PlaybackMode.REAL_TIME.value)
        mode_combo.addItem("Fixed FPS", PlaybackMode.FIXED_FPS.value)
        mode_combo.addItem("Maximum", PlaybackMode.MAXIMUM.value)
        mode_combo.setToolTip(
            "Real time follows scenario timing; Fixed FPS uses the selected display "
            "rate; Maximum advances immediately after each completed frame. These "
            "settings do not control camera-interaction rendering."
        )
        self.widgets["playback_mode_combo"] = mode_combo
        row.addWidget(mode_combo)

        fps_spinbox = QSpinBox()
        fps_spinbox.setRange(MIN_PLAYBACK_FPS, MAX_PLAYBACK_FPS)
        fps_spinbox.setSuffix(" FPS")
        fps_spinbox.setToolTip(
            "Scenario frames attempted per second in Fixed FPS mode; this does not "
            "limit renderer or camera-interaction FPS"
        )
        self.widgets["playback_fps_spinbox"] = fps_spinbox
        row.addWidget(fps_spinbox)

        self._restore_playback_preferences()
        mode_combo.currentIndexChanged.connect(self._on_playback_mode_changed)
        fps_spinbox.valueChanged.connect(self._on_fixed_fps_changed)
        self._sync_playback_rate_controls()

        stride_label = QLabel("Stride:")
        stride_label.setStyleSheet("font-weight: bold; min-width: 40px;")
        row.addWidget(stride_label)

        self.widgets["stride_combo"] = QComboBox()
        self.widgets["stride_combo"].setEditable(True)
        self.widgets["stride_combo"].addItems(["x1", "x2", "x5", "x10", "x50", "x128", "Mesh"])
        self.widgets["stride_combo"].setCurrentIndex(0)
        self.widgets["stride_combo"].setToolTip(
            "Steps per animation tick. Type a custom value (e.g. x128 for coherent CPI stride).\n"
            "'Mesh' syncs to target mesh update rate."
        )
        row.addWidget(self.widgets["stride_combo"])

        row.addStretch()
        return row

    def _qsettings(self) -> Any:
        """Return the injected preference store or the application QSettings."""
        if self._settings is None:
            self._settings = QSettings()
        return self._settings

    def _restore_playback_preferences(self) -> None:
        """Restore the last playback policy and fixed FPS selection."""
        settings = self._qsettings()
        mode = normalize_playback_mode(
            settings.value(SETTINGS_PLAYBACK_MODE_KEY, PlaybackMode.MAXIMUM.value)
        )
        try:
            fixed_fps = int(
                settings.value(SETTINGS_FIXED_PLAYBACK_FPS_KEY, DEFAULT_FIXED_PLAYBACK_FPS)
            )
        except (TypeError, ValueError):
            fixed_fps = DEFAULT_FIXED_PLAYBACK_FPS

        mode_combo = self.widgets["playback_mode_combo"]
        mode_index = mode_combo.findData(mode.value)
        mode_combo.setCurrentIndex(max(0, mode_index))
        self.widgets["playback_fps_spinbox"].setValue(fixed_fps)

    def _on_playback_mode_changed(self, _index: int) -> None:
        """Persist playback policy and expose only its relevant rate control."""
        self._qsettings().setValue(SETTINGS_PLAYBACK_MODE_KEY, self.playback_mode().value)
        self._sync_playback_rate_controls()

    def _on_fixed_fps_changed(self, value: int) -> None:
        """Persist the fixed playback FPS preference."""
        self._qsettings().setValue(SETTINGS_FIXED_PLAYBACK_FPS_KEY, int(value))

    def _sync_playback_rate_controls(self) -> None:
        """Show the FPS selector only when Fixed FPS mode owns cadence."""
        fixed_mode = self.playback_mode() is PlaybackMode.FIXED_FPS
        fps_spinbox = self.widgets.get("playback_fps_spinbox")
        if fps_spinbox is not None:
            fps_spinbox.setVisible(fixed_mode)
            fps_spinbox.setEnabled(fixed_mode and not self._scene_only_mode)

    def playback_mode(self) -> PlaybackMode:
        """Return the selected scenario-frame playback policy."""
        combo = self.widgets.get("playback_mode_combo")
        return normalize_playback_mode(combo.currentData() if combo is not None else None)

    def fixed_playback_fps(self) -> int:
        """Return the selected fixed scenario-frame rate."""
        spinbox = self.widgets.get("playback_fps_spinbox")
        return int(spinbox.value()) if spinbox is not None else DEFAULT_FIXED_PLAYBACK_FPS

    def compute_mesh_stride(self) -> int:
        """Compute stride that aligns with the target mesh update interval.

        Returns:
            Number of simulation steps per mesh update, or 1 if unavailable.
        """
        viz = self.parent
        if viz is None:
            return 1
        duration_s = getattr(viz, "_frame_duration", None)
        total_steps = getattr(viz, "total_animation_steps", None)
        mesh_interval = getattr(viz, "_mesh_update_interval_s", None)
        if not all((duration_s, total_steps, mesh_interval)):
            return 1
        if total_steps <= 0 or duration_s <= 0:
            return 1
        step_dt = duration_s / total_steps
        if step_dt <= 0:
            return 1
        return max(1, math.ceil(mesh_interval / step_dt))

    def set_scene_only_mode(self, enabled: bool) -> None:
        """Disable animation controls when no frame data is available."""
        self._scene_only_mode = bool(enabled)
        interactive_widgets = [
            "step_slider",
            "frame_input",
            "play_btn",
            "prev_btn",
            "next_btn",
            "reverse_play_btn",
            "reset_btn",
            "playback_mode_combo",
            "playback_fps_spinbox",
            "loop_cb",
            "stride_combo",
        ]
        for name in interactive_widgets:
            widget = self.widgets.get(name)
            if widget is not None:
                widget.setEnabled(not enabled)

        self._sync_playback_rate_controls()

        label = self.widgets.get("step_label")
        if label is not None:
            label.setText("Scene only" if enabled else "1")

    def update_total_steps(self, new_total_steps: int) -> None:
        """Mirror a new frame count into slider/input bounds and total label."""
        self.total_steps = new_total_steps
        if "step_slider" in self.widgets:
            self.widgets["step_slider"].setMaximum(self.total_steps - 1)
        if "frame_input" in self.widgets:
            self.widgets["frame_input"].setMaximum(self.total_steps)
        if "total_steps_label" in self.widgets:
            self.widgets["total_steps_label"].setText(f"/ {self.total_steps}")
