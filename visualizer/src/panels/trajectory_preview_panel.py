"""2D trajectory overview panel for loaded frame sources.

The panel consumes immutable TX/RX/target snapshots from the shared background
trajectory coordinator and can color segments by per-frame KPI metrics supplied
by the statistics workflow.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional

from PySide6.QtCore import QEvent, QObject, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from shared.logging import get_logger

from ..app.plot_theme import apply_matplotlib_legend_theme, apply_matplotlib_theme
from ..app.theme import current_theme, get_theme_manager
from ..services.trajectory_load_service import (
    TrajectoryLoadCoordinator,
    TrajectorySnapshot,
)
from .base import BasePanel
from .ui_theme import compact_progress_bar_style, configure_label

logger = get_logger("orchav.trajectory_preview_panel")


class _PanelVisibilityFilter(QObject):
    """Notify the panel when Qt shows or hides its trajectory group."""

    def __init__(self, callback, parent: QWidget) -> None:
        super().__init__(parent)
        self._callback = callback

    def eventFilter(self, watched, event) -> bool:
        """Forward visibility transitions without consuming the Qt event."""
        if event.type() == QEvent.Type.Show:
            self._callback(True)
        elif event.type() == QEvent.Type.Hide:
            self._callback(False)
        return super().eventFilter(watched, event)


class TrajectoryPreviewPanel(BasePanel):
    """Panel for plotting TX/RX/target trajectories across all frames.

    Only visible for precomputed data modes (files, remote_hdf5).
    """

    # Colors for TX, RX trajectories
    TX_COLOR = "#2196F3"  # Blue
    RX_COLOR = "#4CAF50"  # Green
    CURRENT_COLOR = "#FF5722"  # Orange-red for current frame marker

    # Color palette for targets (each target gets a different color)
    TARGET_COLORS = [
        "#9C27B0",  # Purple
        "#E91E63",  # Pink
        "#FF9800",  # Orange
        "#009688",  # Teal
        "#795548",  # Brown
        "#607D8B",  # Blue Grey
        "#CDDC39",  # Lime
        "#00BCD4",  # Cyan
    ]

    # Metric coloring options for the "Color by" combo box
    COLOR_BY_OPTIONS = [
        "None",
        "Direct-Path Pair Share",
        "Pooled Path Delay Spread",
        "Path Count",
        "Strongest Single-Path Loss",
    ]

    # Matplotlib is expensive to repaint at animation speed. Keep the marker
    # responsive while bounding canvas work to roughly 15 Hz.
    FRAME_UPDATE_INTERVAL_MS = 67

    def __init__(self, parent_widget):
        """Initialize lazy trajectory loading and matplotlib state."""
        super().__init__(parent_widget)
        self._trajectories: TrajectorySnapshot | None = None
        self._trajectory_coordinator: TrajectoryLoadCoordinator = (
            parent_widget.trajectory_load_coordinator
        )
        # The coordinator can already be loading while the application shell is
        # assembling its tabs.  Connect only after this panel has created its
        # controls so queued progress updates cannot address an empty registry.
        self._coordinator_connected = False
        self._has_matplotlib = False
        self._current_frame = 0
        self._figure = None
        self._ax = None
        self._per_frame_stats: Optional[Dict] = None
        self._colorbar = None
        self._panel_widget: QGroupBox | None = None
        self._visibility_filter: _PanelVisibilityFilter | None = None
        self._static_plot_dirty = False
        self._current_marker_artist = None
        self._indexed_trajectories: TrajectorySnapshot | None = None
        self._frame_positions: dict[str, dict[int, tuple[tuple[float, float], ...]]] = {
            "tx": {},
            "rx": {},
            "target": {},
        }
        self._last_frame_update_at = 0.0
        self._frame_update_timer = QTimer()
        self._frame_update_timer.setSingleShot(True)
        self._frame_update_timer.timeout.connect(self._flush_current_frame_update)
        self._theme_manager = get_theme_manager()
        self._theme_changed_callback = self._apply_theme
        self._theme_manager.theme_changed.connect(self._theme_changed_callback)

    def create_panel(self) -> QGroupBox:
        """Build trajectory loading controls, progress display, canvas, and legend."""
        group = self.create_group_box("Trajectory Preview")
        self._panel_widget = group
        self._visibility_filter = _PanelVisibilityFilter(self._on_panel_visibility_changed, group)
        group.installEventFilter(self._visibility_filter)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addLayout(self._create_controls_row())

        self.widgets["progress_bar"] = QProgressBar()
        self.widgets["progress_bar"].setRange(0, 100)
        self.widgets["progress_bar"].setValue(0)
        self.widgets["progress_bar"].setVisible(False)
        self.widgets["progress_bar"].setStyleSheet(compact_progress_bar_style())
        layout.addWidget(self.widgets["progress_bar"])

        layout.addWidget(self._create_trajectory_canvas())
        layout.addLayout(self._create_legend_row())

        self._connect_trajectory_coordinator()
        self._sync_from_trajectory_coordinator()

        return group

    def _connect_trajectory_coordinator(self) -> None:
        """Subscribe after controls exist, avoiding construction-time races."""
        if self._coordinator_connected:
            return
        self._trajectory_coordinator.progress_updated.connect(self._on_progress)
        self._trajectory_coordinator.snapshot_updated.connect(self._on_partial_update)
        self._trajectory_coordinator.loading_complete.connect(self._on_loading_complete)
        self._trajectory_coordinator.error_occurred.connect(self._on_error)
        self._trajectory_coordinator.cleared.connect(self._on_trajectory_cleared)
        self._coordinator_connected = True

    def _sync_from_trajectory_coordinator(self) -> None:
        """Render coordinator state that may have changed before panel creation."""
        snapshot = self._trajectory_coordinator.snapshot
        if snapshot is not None:
            if self._trajectory_coordinator.is_complete:
                self._on_loading_complete(snapshot)
                return
            self._on_partial_update(snapshot)

        if self._trajectory_coordinator.is_loading:
            loaded, total = self._trajectory_coordinator.progress
            self._on_progress(loaded, total)

    def _create_controls_row(self):
        """Create load/filter controls and metric-coloring selection."""
        row = QHBoxLayout()
        row.setSpacing(8)

        self.widgets["load_btn"] = QPushButton("Load Trajectories")
        self.widgets["load_btn"].setToolTip("Load shared TX/RX/target positions from all frames")
        self.widgets["load_btn"].clicked.connect(self._on_load_clicked)
        row.addWidget(self.widgets["load_btn"])

        self.widgets["show_tx"] = QCheckBox("TX")
        self.widgets["show_tx"].setChecked(True)
        self.widgets["show_tx"].setStyleSheet("color: #2196F3; font-weight: bold;")
        self.widgets["show_tx"].toggled.connect(self._update_plot)
        row.addWidget(self.widgets["show_tx"])

        self.widgets["show_rx"] = QCheckBox("RX")
        self.widgets["show_rx"].setChecked(True)
        self.widgets["show_rx"].setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.widgets["show_rx"].toggled.connect(self._update_plot)
        row.addWidget(self.widgets["show_rx"])

        self.widgets["show_targets"] = QCheckBox("Targets")
        self.widgets["show_targets"].setChecked(True)
        self.widgets["show_targets"].setStyleSheet("color: #9C27B0; font-weight: bold;")
        self.widgets["show_targets"].toggled.connect(self._update_plot)
        row.addWidget(self.widgets["show_targets"])

        self.widgets["show_labels"] = QCheckBox("Labels")
        self.widgets["show_labels"].setChecked(True)
        self.widgets["show_labels"].toggled.connect(self._update_plot)
        row.addWidget(self.widgets["show_labels"])

        row.addWidget(QLabel("Color by:"))
        self.widgets["color_by_combo"] = QComboBox()
        for option in self.COLOR_BY_OPTIONS:
            self.widgets["color_by_combo"].addItem(option)
        self.widgets["color_by_combo"].setToolTip(
            "Color trajectory segments by a per-frame channel metric.\n"
            "Requires statistics to be computed first."
        )
        self.widgets["color_by_combo"].setFixedWidth(180)
        self.widgets["color_by_combo"].currentTextChanged.connect(self._update_plot)
        row.addWidget(self.widgets["color_by_combo"])

        self.widgets["status_label"] = QLabel("Not loaded")
        configure_label(self.widgets["status_label"], role="secondary", italic=True)
        row.addWidget(self.widgets["status_label"])

        row.addStretch()
        return row

    def _create_trajectory_canvas(self):
        """Create matplotlib canvas for trajectory visualization."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            import matplotlib

            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._has_matplotlib = True

            fig = Figure(figsize=(6, 4), tight_layout=True)
            self._ax = fig.add_subplot(111)
            self._ax.set_xlabel("X (m)", fontsize=9)
            self._ax.set_ylabel("Y (m)", fontsize=9)
            self._ax.set_title("TX/RX Trajectories (Top-Down View)", fontsize=10)
            apply_matplotlib_theme(fig, self._ax)
            self._ax.set_aspect("equal", adjustable="datalim")
            self._ax.tick_params(axis="both", which="major", labelsize=8)

            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumHeight(250)
            canvas.setMaximumHeight(450)

            self.widgets["canvas"] = canvas
            self._figure = fig
            layout.addWidget(canvas)

        except ImportError:
            self._has_matplotlib = False
            error_label = QLabel("Matplotlib not available.\nInstall with: pip install matplotlib")
            configure_label(error_label, role="error", word_wrap=True)
            layout.addWidget(error_label)

        return container

    def _create_legend_row(self):
        """Create the static legend for trajectory and current-frame colors."""
        row = QHBoxLayout()
        row.setSpacing(8)

        tx_box = QLabel("TX")
        tx_box.setStyleSheet(
            f"background-color: {self.TX_COLOR}; color: white; "
            "padding: 2px 8px; border-radius: 3px; font-size: 10px;"
        )
        row.addWidget(tx_box)

        rx_box = QLabel("RX")
        rx_box.setStyleSheet(
            f"background-color: {self.RX_COLOR}; color: white; "
            "padding: 2px 8px; border-radius: 3px; font-size: 10px;"
        )
        row.addWidget(rx_box)

        current_box = QLabel("Current")
        current_box.setStyleSheet(
            f"background-color: {self.CURRENT_COLOR}; color: white; "
            "padding: 2px 8px; border-radius: 3px; font-size: 10px;"
        )
        row.addWidget(current_box)

        row.addStretch()
        return row

    def _apply_theme(self, *_args) -> None:
        """Apply the current application theme to the trajectory plot."""
        if self._figure is not None and self._ax is not None:
            apply_matplotlib_theme(self._figure, self._ax)
            apply_matplotlib_legend_theme(self._ax)
            canvas = self.widgets.get("canvas")
            if canvas is not None:
                canvas.draw_idle()

    def _on_load_clicked(self):
        """Request the shared background load or consume its current snapshot."""
        frame_source = getattr(self.parent, "frame_source", None)
        started = self._trajectory_coordinator.load(frame_source)
        if started or self._trajectory_coordinator.is_loading:
            self.widgets["load_btn"].setEnabled(False)
            self.widgets["progress_bar"].setVisible(True)
            self.widgets["progress_bar"].setValue(0)
            self.widgets["status_label"].setText("Loading...")

        snapshot = self._trajectory_coordinator.snapshot
        if snapshot is not None:
            if self._trajectory_coordinator.is_complete:
                self._on_loading_complete(snapshot)
            else:
                self._on_partial_update(snapshot)
                loaded, total = self._trajectory_coordinator.progress
                self._on_progress(loaded, total)

    def _on_progress(self, loaded: int, total: int):
        """Handle progress updates."""
        load_button = self.widgets.get("load_btn")
        progress = self.widgets.get("progress_bar")
        status = self.widgets.get("status_label")
        if load_button is None or progress is None or status is None:
            return
        percent = int(loaded / total * 100) if total > 0 else 0
        load_button.setEnabled(False)
        progress.setVisible(True)
        progress.setValue(percent)
        status.setText(f"Loading: {loaded}/{total} frames")

    def _on_partial_update(self, trajectories: TrajectorySnapshot):
        """Plot a shared partial trajectory snapshot."""
        self._set_trajectory_snapshot(trajectories)
        if "show_tx" in self.widgets:
            self._update_plot()

    def _on_loading_complete(self, trajectories: TrajectorySnapshot):
        """Plot the shared final trajectory snapshot."""
        self._set_trajectory_snapshot(trajectories)
        load_button = self.widgets.get("load_btn")
        progress = self.widgets.get("progress_bar")
        status = self.widgets.get("status_label")
        if load_button is None or progress is None or status is None:
            return
        load_button.setEnabled(True)
        progress.setVisible(False)

        total_frames = len(trajectories.frames_loaded)
        status.setText(f"Loaded {total_frames} frames")

        self._update_plot()

    def _on_error(self, error_msg: str):
        """Handle error during loading."""
        load_button = self.widgets.get("load_btn")
        progress = self.widgets.get("progress_bar")
        status = self.widgets.get("status_label")
        if load_button is not None:
            load_button.setEnabled(True)
        if progress is not None:
            progress.setVisible(False)
        if status is not None:
            status.setText(f"Error: {error_msg}")
        logger.error("Trajectory loading error: %s", error_msg)

    def _on_trajectory_cleared(self) -> None:
        """Clear plot state when scenario/source ownership changes."""
        self._trajectories = None
        self._indexed_trajectories = None
        self._frame_positions = {"tx": {}, "rx": {}, "target": {}}
        self._static_plot_dirty = False
        self._current_marker_artist = None
        self._frame_update_timer.stop()
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except (ValueError, AttributeError):
                pass
            self._colorbar = None
        if self._has_matplotlib and self._ax is not None:
            self._ax.clear()
            self._ax.set_xlabel("X (m)", fontsize=9)
            self._ax.set_ylabel("Y (m)", fontsize=9)
            self._ax.set_title("TX/RX/Target Trajectories (Top-Down View)", fontsize=10)
            apply_matplotlib_theme(self._figure, self._ax)
            self._ax.text(
                0.5,
                0.5,
                "No trajectory data",
                ha="center",
                va="center",
                transform=self._ax.transAxes,
                fontsize=12,
                color=current_theme().text_muted,
            )
            canvas = self.widgets.get("canvas")
            if canvas is not None:
                canvas.draw_idle()
        load_button = self.widgets.get("load_btn")
        if load_button is not None:
            load_button.setEnabled(True)
        progress = self.widgets.get("progress_bar")
        if progress is not None:
            progress.setVisible(False)
            progress.setValue(0)
        status = self.widgets.get("status_label")
        if status is not None:
            status.setText("Not loaded")

    def _get_metric_color_mode(self) -> str:
        """Return the currently selected 'Color by' metric name."""
        combo = self.widgets.get("color_by_combo")
        if combo is None:
            return "None"
        return combo.currentText()

    def _build_frame_index_to_metric(self, metric_key: str) -> Optional[Dict[int, float]]:
        """Map frame indices to per-frame metric values.

        Args:
            metric_key: One of ``direct_path_pair_share_evolution``,
                ``delay_spread_evolution``, ``mpc_evolution``, or
                ``strongest_single_path_loss_evolution``.

        Returns:
            Dict mapping frame index to metric value, or ``None`` if the
            required statistics data is not available.
        """
        if self._per_frame_stats is None:
            return None
        frame_indices = self._per_frame_stats.get("frame_indices", [])
        values = self._per_frame_stats.get(metric_key, [])
        if len(frame_indices) == 0 or len(values) == 0 or len(frame_indices) != len(values):
            return None
        return dict(zip(frame_indices, values))

    def _plot_colored_trajectory(
        self,
        sorted_pos: list,
        frame_metric_map: Dict[int, float],
        metric_mode: str,
        label: str,
        linestyle: str = "-",
        marker: str = "o",
    ) -> bool:
        """Plot a single trajectory with per-segment metric coloring.

        Args:
            sorted_pos: Sorted list of ``(frame_idx, x, y, z)`` tuples.
            frame_metric_map: Mapping from frame index to metric value.
            metric_mode: The selected color-by mode string.
            label: Legend label for the trajectory.
            linestyle: Line style (solid or dashed).
            marker: Scatter marker style.

        Returns:
            True if any data was plotted.
        """
        import numpy as np
        from matplotlib.collections import LineCollection

        if len(sorted_pos) < 2:
            return False

        xs = [p[1] for p in sorted_pos]
        ys = [p[2] for p in sorted_pos]
        frame_ids = [p[0] for p in sorted_pos]

        # Build segment values: each segment gets the metric of its starting frame
        seg_values = []
        for fid in frame_ids[:-1]:
            val = frame_metric_map.get(fid)
            if val is None:
                seg_values.append(float("nan"))
            else:
                seg_values.append(float(val))

        points = np.column_stack([xs, ys])
        segments = np.array([[points[i], points[i + 1]] for i in range(len(points) - 1)])
        seg_values_arr = np.array(seg_values, dtype=np.float64)

        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("plasma")
        finite_mask = np.isfinite(seg_values_arr)
        if not finite_mask.any():
            return False

        if metric_mode == "Direct-Path Pair Share":
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.nanmin(seg_values_arr))
            vmax = float(np.nanmax(seg_values_arr))
            if vmin == vmax:
                vmax = vmin + 1.0

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=2.0,
            alpha=0.8,
            label=label,
        )
        if linestyle == "--":
            lc.set_linestyle("--")
        lc.set_array(seg_values_arr)
        self._ax.add_collection(lc)

        if self._colorbar is None:
            theme = current_theme()
            metric_labels = {
                "Direct-Path Pair Share": "Direct-Path Pair Share",
                "Pooled Path Delay Spread": "Pooled Path Delay Spread (ns)",
                "Path Count": "Path Count",
                "Strongest Single-Path Loss": "Single-Path Loss (dB)",
            }
            self._colorbar = self._figure.colorbar(lc, ax=self._ax, pad=0.02, fraction=0.04)
            self._colorbar.set_label(metric_labels.get(metric_mode, metric_mode), fontsize=8)
            self._colorbar.ax.yaxis.label.set_color(theme.text_primary)
            self._colorbar.ax.tick_params(colors=theme.text_primary)

        # Scatter dots at positions
        self._ax.scatter(xs, ys, c=current_theme().text_muted, s=3, alpha=0.3, zorder=1)
        return True

    def _set_trajectory_snapshot(self, trajectories: TrajectorySnapshot) -> None:
        """Store the latest snapshot without eagerly indexing hidden updates."""
        self._trajectories = trajectories

    def _ensure_frame_position_index(self) -> None:
        """Index the latest snapshot once, immediately before presentation."""
        trajectories = self._trajectories
        if trajectories is None or self._indexed_trajectories is trajectories:
            return

        indexed: dict[str, dict[int, list[tuple[float, float]]]] = {
            "tx": {},
            "rx": {},
            "target": {},
        }
        trajectory_keys = (
            ("tx", "tx_positions"),
            ("rx", "rx_positions"),
            ("target", "target_positions"),
        )
        for kind, snapshot_key in trajectory_keys:
            tracks = trajectories.get(snapshot_key, {})
            for positions in tracks.values():
                for frame_idx, x, y, _z in positions:
                    indexed[kind].setdefault(int(frame_idx), []).append((float(x), float(y)))
        self._frame_positions = {
            kind: {frame: tuple(positions) for frame, positions in by_frame.items()}
            for kind, by_frame in indexed.items()
        }
        self._indexed_trajectories = trajectories

    def _is_plot_visible(self) -> bool:
        """Return whether the panel is currently visible through its ancestors."""
        return self._panel_widget is not None and self._panel_widget.isVisible()

    def _on_panel_visibility_changed(self, visible: bool) -> None:
        """Catch the plot up when shown and suspend repaint work when hidden."""
        if not visible:
            self._frame_update_timer.stop()
            return
        if self._static_plot_dirty:
            self._rebuild_static_plot()
        else:
            self._flush_current_frame_update(force=True)

    def _update_plot(self, *_args):
        """Request a static trajectory rebuild after data or settings change."""
        self._static_plot_dirty = True
        if self._is_plot_visible():
            self._rebuild_static_plot()

    def _rebuild_static_plot(self):
        """Rebuild trajectory lines, labels, legend, and plot bounds."""
        if not self._has_matplotlib or not self._trajectories:
            return

        self._ensure_frame_position_index()
        self._static_plot_dirty = False
        self._frame_update_timer.stop()

        # Remove existing colorbar before clearing axes
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except (ValueError, AttributeError):
                pass
            self._colorbar = None

        self._ax.clear()
        self._current_marker_artist = None
        self._ax.set_xlabel("X (m)", fontsize=9)
        self._ax.set_ylabel("Y (m)", fontsize=9)
        self._ax.set_title("TX/RX/Target Trajectories (Top-Down View)", fontsize=10)
        apply_matplotlib_theme(self._figure, self._ax)
        theme = current_theme()

        show_tx = self.widgets["show_tx"].isChecked()
        show_rx = self.widgets["show_rx"].isChecked()
        show_targets = self.widgets["show_targets"].isChecked()
        show_labels = self.widgets["show_labels"].isChecked()
        color_mode = self._get_metric_color_mode()

        # Determine metric map for coloring
        metric_map: Optional[Dict[int, float]] = None
        metric_key_map = {
            "Direct-Path Pair Share": "direct_path_pair_share_evolution",
            "Pooled Path Delay Spread": "delay_spread_evolution",
            "Path Count": "mpc_evolution",
            "Strongest Single-Path Loss": "strongest_single_path_loss_evolution",
        }
        if color_mode != "None":
            metric_key = metric_key_map.get(color_mode)
            if metric_key:
                metric_map = self._build_frame_index_to_metric(metric_key)

        use_metric_coloring = metric_map is not None and color_mode != "None"

        has_data = False
        label_positions = []  # Collect positions for labels

        # Plot TX trajectories
        if show_tx:
            for tx_idx, positions in self._trajectories.get("tx_positions", {}).items():
                if positions:
                    sorted_pos = sorted(positions, key=lambda x: x[0])
                    label = f"TX{tx_idx}"
                    if use_metric_coloring:
                        plotted = self._plot_colored_trajectory(
                            sorted_pos, metric_map, color_mode, label
                        )
                        has_data = has_data or plotted
                    else:
                        xs = [p[1] for p in sorted_pos]
                        ys = [p[2] for p in sorted_pos]
                        self._ax.plot(
                            xs, ys, color=self.TX_COLOR, alpha=0.7, linewidth=1.5, label=label
                        )
                        self._ax.scatter(xs, ys, c=self.TX_COLOR, s=4, alpha=0.5)
                        has_data = True
                    xs = [p[1] for p in sorted_pos]
                    ys = [p[2] for p in sorted_pos]
                    if xs and ys:
                        label_positions.append((xs[0], ys[0], label, self.TX_COLOR, "tx"))

        # Plot RX trajectories
        if show_rx:
            for rx_idx, positions in self._trajectories.get("rx_positions", {}).items():
                if positions:
                    sorted_pos = sorted(positions, key=lambda x: x[0])
                    label = f"RX{rx_idx}"
                    if use_metric_coloring:
                        plotted = self._plot_colored_trajectory(
                            sorted_pos, metric_map, color_mode, label
                        )
                        has_data = has_data or plotted
                    else:
                        xs = [p[1] for p in sorted_pos]
                        ys = [p[2] for p in sorted_pos]
                        self._ax.plot(
                            xs, ys, color=self.RX_COLOR, alpha=0.7, linewidth=1.5, label=label
                        )
                        self._ax.scatter(xs, ys, c=self.RX_COLOR, s=4, alpha=0.5)
                        has_data = True
                    xs = [p[1] for p in sorted_pos]
                    ys = [p[2] for p in sorted_pos]
                    if xs and ys:
                        label_positions.append((xs[0], ys[0], label, self.RX_COLOR, "rx"))

        # Plot Target trajectories (each target gets a different color)
        if show_targets:
            target_names = list(self._trajectories.get("target_positions", {}).keys())
            for i, target_name in enumerate(target_names):
                positions = self._trajectories["target_positions"][target_name]
                if positions:
                    color = self.TARGET_COLORS[i % len(self.TARGET_COLORS)]
                    sorted_pos = sorted(positions, key=lambda x: x[0])
                    if use_metric_coloring:
                        plotted = self._plot_colored_trajectory(
                            sorted_pos,
                            metric_map,
                            color_mode,
                            target_name,
                            linestyle="--",
                            marker="D",
                        )
                        has_data = has_data or plotted
                    else:
                        xs = [p[1] for p in sorted_pos]
                        ys = [p[2] for p in sorted_pos]
                        self._ax.plot(
                            xs,
                            ys,
                            color=color,
                            alpha=0.7,
                            linewidth=1.5,
                            linestyle="--",
                            label=target_name,
                        )
                        self._ax.scatter(xs, ys, c=color, s=6, alpha=0.5, marker="D")
                        has_data = True
                    xs = [p[1] for p in sorted_pos]
                    ys = [p[2] for p in sorted_pos]
                    if xs and ys:
                        label_positions.append((xs[0], ys[0], target_name, color, "target"))

        if show_labels and label_positions:
            for x, y, text, color, node_type in label_positions:
                offset_x = 2 if node_type == "tx" else (-2 if node_type == "rx" else 0)
                offset_y = 2 if node_type == "target" else -2
                self._ax.annotate(
                    text,
                    (x, y),
                    textcoords="offset points",
                    xytext=(offset_x, offset_y),
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                    ha="center",
                    va="bottom" if offset_y > 0 else "top",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=theme.bg_secondary,
                        edgecolor=color,
                        alpha=0.8,
                    ),
                )

        # Create one reusable collection for every current-frame marker. Frame
        # changes only mutate this artist instead of clearing the full axes.
        self._highlight_current_frame()

        if has_data:
            self._ax.set_aspect("equal", adjustable="datalim")
            # Auto-scale for LineCollection (not auto-fitted by matplotlib)
            self._ax.autoscale_view()
            # Only show legend if not too many entries (avoid clutter)
            handles, labels = self._ax.get_legend_handles_labels()
            if len(handles) <= 8:
                self._ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)
                apply_matplotlib_legend_theme(self._ax)
        else:
            self._ax.text(
                0.5,
                0.5,
                "No trajectory data",
                ha="center",
                va="center",
                transform=self._ax.transAxes,
                fontsize=12,
                color=theme.text_muted,
            )

        self.widgets["canvas"].draw_idle()

    def _highlight_current_frame(self) -> None:
        """Create and position the reusable current-frame marker collection."""
        if not self._trajectories or self._ax is None:
            return

        self._current_marker_artist = self._ax.scatter(
            [],
            [],
            c=self.CURRENT_COLOR,
            s=120,
            marker="*",
            zorder=10,
            edgecolors=current_theme().bg_secondary,
            linewidths=0.5,
        )
        self._update_current_frame_marker(draw=False)

    def _visible_current_frame_positions(self) -> list[tuple[float, float]]:
        """Return indexed marker positions enabled by the current filters."""
        current_frame = self._current_frame
        positions: list[tuple[float, float]] = []
        show_tx = self.widgets["show_tx"].isChecked()
        show_rx = self.widgets["show_rx"].isChecked()
        show_targets = self.widgets["show_targets"].isChecked()

        if show_tx:
            positions.extend(self._frame_positions["tx"].get(current_frame, ()))
        if show_rx:
            positions.extend(self._frame_positions["rx"].get(current_frame, ()))
        if show_targets:
            positions.extend(self._frame_positions["target"].get(current_frame, ()))
        return positions

    def _update_current_frame_marker(self, *, draw: bool = True) -> None:
        """Move the reusable marker collection to the current indexed positions."""
        if self._current_marker_artist is None:
            return

        import numpy as np

        positions = self._visible_current_frame_positions()
        offsets = (
            np.asarray(positions, dtype=float).reshape((-1, 2))
            if positions
            else np.empty((0, 2), dtype=float)
        )
        self._current_marker_artist.set_offsets(offsets)
        if draw:
            canvas = self.widgets.get("canvas")
            if canvas is not None:
                canvas.draw_idle()

    def _schedule_current_frame_update(self) -> None:
        """Update now or coalesce rapid playback frames into one repaint."""
        if (
            not self._trajectories
            or self._current_marker_artist is None
            or not self._is_plot_visible()
        ):
            return
        if self._frame_update_timer.isActive():
            return

        elapsed_ms = (time.monotonic() - self._last_frame_update_at) * 1000.0
        remaining_ms = self.FRAME_UPDATE_INTERVAL_MS - elapsed_ms
        if remaining_ms <= 0:
            self._flush_current_frame_update()
            return
        self._frame_update_timer.start(max(1, math.ceil(remaining_ms)))

    def _flush_current_frame_update(self, *, force: bool = False) -> None:
        """Apply the latest frame to the marker artist when presentation is active."""
        if (
            not self._trajectories
            or self._current_marker_artist is None
            or not self._is_plot_visible()
        ):
            return
        if not force:
            elapsed_ms = (time.monotonic() - self._last_frame_update_at) * 1000.0
            if elapsed_ms < self.FRAME_UPDATE_INTERVAL_MS:
                self._frame_update_timer.start(
                    max(1, math.ceil(self.FRAME_UPDATE_INTERVAL_MS - elapsed_ms))
                )
                return
        self._update_current_frame_marker()
        self._last_frame_update_at = time.monotonic()

    def set_per_frame_stats(self, stats: Dict) -> None:
        """Supply per-frame KPI data for metric-based trajectory coloring.

        Args:
            stats: Statistics dict from the collector, containing keys
                ``frame_indices``, ``direct_path_pair_share_evolution``,
                ``delay_spread_evolution``, ``mpc_evolution``, and
                ``strongest_single_path_loss_evolution``.
        """
        self._per_frame_stats = stats
        # Refresh the plot if trajectories are already loaded
        if self._trajectories:
            self._update_plot()

    def set_current_frame(self, frame: int):
        """Update the current-frame marker without rebuilding static trajectories."""
        frame = int(frame)
        if frame == self._current_frame:
            return
        self._current_frame = frame
        self._schedule_current_frame_update()

    def should_be_visible(self) -> bool:
        """Check if this panel should be visible based on data mode."""
        return self._trajectory_coordinator.supports_source(
            getattr(self.parent, "frame_source", None)
        )

    def cleanup(self):
        """Disconnect this consumer; the coordinator owns worker shutdown."""
        self._frame_update_timer.stop()
        if self._theme_manager is not None:
            try:
                self._theme_manager.theme_changed.disconnect(self._theme_changed_callback)
            except (RuntimeError, TypeError):
                pass
            self._theme_manager = None
        if not self._coordinator_connected:
            return
        connections = (
            (self._trajectory_coordinator.progress_updated, self._on_progress),
            (self._trajectory_coordinator.snapshot_updated, self._on_partial_update),
            (self._trajectory_coordinator.loading_complete, self._on_loading_complete),
            (self._trajectory_coordinator.error_occurred, self._on_error),
            (self._trajectory_coordinator.cleared, self._on_trajectory_cleared),
        )
        for signal, callback in connections:
            try:
                signal.disconnect(callback)
            except (RuntimeError, TypeError):
                pass
        self._coordinator_connected = False
