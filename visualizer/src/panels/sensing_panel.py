"""Integrated sensing analysis and compact scene-overlay controls.

The panel bridges frame-backed sensing payloads to the optional pyqtgraph
dashboard and to renderer-owned 3D sensing overlays. Products are selected by
their explicit schema-v4 scene binding; storage proximity is never a display
policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QSettings, QSignalBlocker, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from shared.extensions.sensing import (
    UnsupportedSensingSchemaError,
    assert_sensing_contract,
    assert_supported_sensing_schema,
)
from shared.frames.providers import Hdf5Provider
from shared.logging import get_logger

from ..io.packed_frame_payload import (
    PACKED_PROJECTION_KEY,
    standard_frame_to_visual_frame,
    try_load_packed_visual_frame,
    try_upgrade_packed_visual_frame,
    visual_frame_read_request,
    visualizer_frame_provider,
)
from ..renderers.protocol import volume_rendering_unavailable_reason
from ..sensing.runtime import FEATURE_KEY, state_value, update_state
from ..sensing.startup_cache_ui import (
    SENSING_FIGURES_ENABLED_PROPERTY,
    focused_sensing_figure_policy,
    initial_sensing_motion_mode,
    preferred_sensing_figure_policy,
    sensing_3d_override,
    sensing_cache_policy,
    sensing_figure_policy,
    set_sensing_figure_strategy,
)
from ..services.sensing_service import (
    range_doppler_product_variants,
    range_profile_product_variants,
)
from ..widgets.dashboard_host import dashboard_is_visible
from ..widgets.wrapping_layout import WrappingLayout
from .collapsible_section import CollapsibleSection

logger = get_logger("orchav.sensing_panel")

# Periodic plot refresh is independent of the acquisition PRF.
# Poll finished work promptly; one in-flight selection still coalesces playback
# changes, and a slow read never queues every frame.
_DASHBOARD_REFRESH_MS = 33
_DASHBOARD_PLAYBACK_REFRESH_MS = 67
_DASHBOARD_POLL_MS = 8
_IDLE_REFRESH_MS = 500


def _assert_sensing_block(sensing: Any) -> None:
    """Validate a sensing block before panel-specific interpretation."""
    assert_supported_sensing_schema(sensing)
    assert_sensing_contract(
        {"sensing": sensing},
        require_schema_version=True,
    )


def _validated_sensing_block(frame: dict[str, Any]) -> dict[str, Any]:
    """Return a validated sensing mapping, or an empty mapping when absent."""
    sensing = frame.get("sensing")
    _assert_sensing_block(sensing)
    return sensing if isinstance(sensing, dict) else {}


try:
    from ..sensing.sensing_window import SensingDashboard
    from ..sensing.update_request import SensingUpdateRequest
    from ..widgets.dashboard_host import DashboardHost

    HAS_SENSING_WINDOW = True
except ImportError:  # pragma: no cover
    HAS_SENSING_WINDOW = False
    SensingDashboard = None  # type: ignore
    SensingUpdateRequest = None  # type: ignore


class SensingPanel:
    """Sensing analysis with shared embedded and floating presentation."""

    expand_in_tab = True

    SENSING_PRESETS: dict[str, dict[str, Any]] = {
        "analysis": {
            "label": "Analysis",
            "description": "Practical target analysis view",
            "variant": "auto",
            "dashboard": {
                "view_mode": "summary",
                "micro_doppler_mode": "auto",
                "micro_gate_half_width": 2,
                "rd_dynamic_range_db": 50,
                "micro_dynamic_range_db": 60,
                "show_detections": True,
                "show_gt": True,
                "show_cfar": False,
                "zoom_to_data": False,
            },
        },
        "micro-doppler": {
            "label": "Micro-Doppler",
            "description": "Slow-time micro-motion focus",
            "variant": "auto",
            "dashboard": {
                "view_mode": "micro-doppler",
                "micro_doppler_mode": "auto",
                "micro_gate_half_width": 3,
                "rd_dynamic_range_db": 55,
                "micro_dynamic_range_db": 72,
                "show_detections": True,
                "show_gt": False,
                "show_cfar": False,
                "zoom_to_data": False,
            },
        },
        "diagnostics": {
            "label": "Diagnostics",
            "description": "Engineering diagnostics and debug overlays",
            "variant": "clean",
            "dashboard": {
                "view_mode": "diagnostics",
                "micro_doppler_mode": "auto",
                "micro_gate_half_width": 1,
                "rd_dynamic_range_db": 38,
                "micro_dynamic_range_db": 60,
                "show_detections": True,
                "show_gt": False,
                "show_cfar": True,
                "zoom_to_data": False,
            },
        },
    }

    def __init__(self, parent_widget: Any) -> None:
        """Initialize widget, dashboard, refresh, and sensing-frame cache state."""
        self.parent = parent_widget
        self._container: QWidget | None = None
        self.widgets: dict[str, Any] = {}
        self._sensing_window: Optional[SensingDashboard] = None
        self.dashboard_host = None
        self.results_section = None
        self.options_section = None
        self._refresh_timer: Optional[QTimer] = None
        self._last_cir_identity: Optional[tuple[Any, ...]] = None
        self._last_display_identity: Optional[tuple[Any, ...]] = None
        self._dashboard_generation = 0
        self._closed = False
        self._dashboard_session = None
        self._dashboard_cache = None
        self._dashboard_poll_timer: Optional[QTimer] = None
        self._completed_dashboard_key = None
        self._loading_paused_identity = None
        self._bound_recording_identity = None
        self.recording_figure_policy = None
        self.recording_3d_override = None
        self.recording_cache_policy = None
        self.recording_motion_mode = "auto"
        self._initial_figure_view_pending = False
        self._dashboard_view_identity = None
        self.preparation = None
        from ..sensing.saved_tracking_scene import SavedTrackingScene

        self.saved_track_scene = SavedTrackingScene()

    def title(self) -> str:
        """Return the section title used by panel composition."""
        return "Sensing"

    def create_panel(self) -> QWidget:
        """Build playback, scene controls, and a lazily activated dashboard."""
        container = QWidget()
        self._container = container
        container.destroyed.connect(self.cleanup)
        main_layout = QVBoxLayout(container)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)

        self.volume_host = QWidget()
        self.volume_layout = QVBoxLayout(self.volume_host)
        self.volume_layout.setContentsMargins(0, 0, 0, 0)
        self.volume_host.hide()
        main_layout.addWidget(self.volume_host)
        self.return_host = QWidget()
        self.return_layout = QVBoxLayout(self.return_host)
        self.return_layout.setContentsMargins(0, 0, 0, 0)
        self.return_host.hide()
        main_layout.addWidget(self.return_host)

        self.recording_controls = QWidget()
        self.recording_tabs = QTabWidget()
        self.recording_tabs.setDocumentMode(True)
        self.recording_tabs.setTabBarAutoHide(True)
        self.recording_tabs.addTab(self.recording_controls, "Results")
        main_layout.addWidget(self.recording_tabs, 1)
        main_layout = QVBoxLayout(self.recording_controls)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.options_section = CollapsibleSection(
            "Sensing options", start_open=False, flat_header=True
        )
        self.options_section.setToolTip(
            "Playback, 3D overlays and figure-loading settings. "
            "Result graphs keep updating while these options are collapsed."
        )
        if QSettings().value("layout/sections/sensing_options", False, type=bool):
            self.options_section.expand()
        self.options_section.toggled.connect(
            lambda expanded: QSettings().setValue("layout/sections/sensing_options", expanded)
        )
        main_layout.addWidget(self.options_section)
        options_layout = self.options_section.content_layout()

        self._build_buttons()
        from ..sensing.playback import SensingPlayback

        sensing_playback = QCheckBox("Play sensing frames only")
        sensing_playback.setToolTip(
            "Load and play only scene frames with a saved sensing result. "
            "Uses each result's recorded scene binding, including zero-detection results. "
            "Real-time playback keeps the saved timestamp spacing."
        )
        self.widgets["sensing_playback"] = sensing_playback
        self.playback = SensingPlayback(self.parent, sensing_playback)
        self.parent.sensing_playback = self.playback
        options_layout.addWidget(sensing_playback)
        options_layout.addWidget(self._build_display_row())

        if HAS_SENSING_WINDOW:
            preparation_group = QGroupBox("Sensing figure preparation")
            preparation_layout = QVBoxLayout(preparation_group)
            strategy_row = WrappingLayout()
            strategy_row.addWidget(QLabel("Mode:"))
            preparation_layout.addLayout(strategy_row)
            strategy = QComboBox()
            strategy.addItem("Open quickly", "on-demand")
            strategy.addItem("Motion only", "motion")
            strategy.addItem("Summary only", "summary")
            strategy.addItem("Prepare before playback", "prepare")
            strategy.setCurrentIndex(strategy.findData(preferred_sensing_figure_policy()))
            strategy.setToolTip(
                "Choose the first figures to load, or prepare saved playback figures. "
                "Motion and Summary defer 3D sensing until you enable it. "
                "Other graphs remain available on demand. Changes apply to the next opening."
            )
            strategy.currentIndexChanged.connect(self._on_figure_strategy_changed)
            self.widgets["sensing_figure_strategy"] = strategy
            strategy_row.addWidget(strategy)
            load_figures = QCheckBox("Load sensing figures")
            load_figures.setChecked(self._sensing_figures_enabled())
            load_figures.setToolTip(
                "Temporarily disable for 3D only. The loading strategy is preserved; "
                "scene overlays and sensing-frame playback remain available."
            )
            load_figures.toggled.connect(self._on_sensing_figures_toggled)
            self.widgets["load_sensing_figures"] = load_figures
            preparation_layout.addWidget(load_figures)
            options_layout.addWidget(preparation_group)
            self.dashboard_host = DashboardHost(
                title="ORCHAV Sensing Dashboard",
                settings_key="visualizer/sensing_dashboard_geometry",
            )
            self.dashboard_host.activated.connect(self._activate_dashboard)
            self.dashboard_host.returned.connect(self._on_open_dashboard)
            self.results_section = CollapsibleSection(
                "Result graphs", start_open=True, expand_content=True
            )
            self.results_section.content_layout().addWidget(self.dashboard_host)
            self.results_section.setToolTip(
                "Collapse to hide graphs and pause plot updates. Completed results and zoom are preserved."
            )
            if not QSettings().value("layout/sections/sensing_results", True, type=bool):
                self.results_section.collapse()
            self.results_section.toggled.connect(self._on_results_toggled)
            if not self._sensing_figures_enabled():
                self.results_section.hide()
            main_layout.addWidget(self.results_section, 1)
        else:
            main_layout.addWidget(QLabel("Install pyqtgraph to enable sensing analysis."))
        main_layout.addStretch()

        self._start_auto_refresh()
        self.bind_recording_source()
        return container

    def _on_results_toggled(self, expanded: bool) -> None:
        QSettings().setValue("layout/sections/sensing_results", bool(expanded))
        if not expanded and not self._dashboard_is_visible():
            self._pause_dashboard_updates()

    @staticmethod
    def _sensing_figures_enabled() -> bool:
        return sensing_figure_policy() != "off"

    def _on_sensing_figures_toggled(self, enabled: bool) -> None:
        """Pause figure readers independently of the scene and its playback plan."""
        app = QApplication.instance()
        if app is not None:
            app.setProperty(SENSING_FIGURES_ENABLED_PROPERTY, enabled)
        if not enabled:
            self._retire_dashboard_session()
            if self.dashboard_host is not None:
                self.dashboard_host.suspend()
            if self._sensing_window is not None:
                self._sensing_window._update_live_figure_captions("Sensing figures disabled")
            self._set_status("Sensing figures disabled; 3D sensing remains available.")
        if self.results_section is not None:
            self.results_section.setVisible(enabled)
        if enabled:
            self._loading_paused_identity = None
            self._completed_dashboard_key = None
            self._activate_dashboard()

    def _on_figure_strategy_changed(self, _index: int) -> None:
        strategy = self.widgets["sensing_figure_strategy"].currentData()
        set_sensing_figure_strategy(strategy)
        if strategy == "prepare":
            self._set_status("Figures will be prepared at the next scenario opening.")

    def attach_volume(self, widget) -> None:
        """Present one volume workspace in place of inapplicable recording controls."""
        self._retire_dashboard_session()
        if self.dashboard_host is not None:
            self.dashboard_host.suspend()
        self.volume_layout.addWidget(widget)
        self.volume_host.show()
        self.recording_controls.hide()
        self.recording_tabs.hide()
        widget.destroyed.connect(self._volume_closed)

    def _volume_closed(self, *_args) -> None:
        current = getattr(self.parent, "sensing_volume_window", None)
        if not getattr(self, "_closed", False) and (current is None or current.closed):
            self.volume_host.hide()
            # A volume's deferred destruction can follow attachment of a fusion
            # tab. Let the selected tab own page visibility; showing Results
            # unconditionally would paint both pages over each other.
            self.recording_controls.setVisible(
                self.recording_tabs.currentWidget() is self.recording_controls
            )
            self.recording_tabs.show()

    def attach_fusion(self, widget) -> None:
        """Keep common fusion results in the same sensing workspace."""
        self.recording_tabs.addTab(widget, "Fusion")
        self.recording_tabs.setCurrentWidget(widget)

    def detach_fusion(self, widget) -> None:
        index = self.recording_tabs.indexOf(widget)
        if index >= 0:
            self.recording_tabs.removeTab(index)

    # UI helpers
    def _build_sampling_row(self) -> QWidget:
        """Build manual sampling-frequency controls used as config fallback."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(QLabel("Fs (Hz):"))
        fs_box = QDoubleSpinBox()
        fs_box.setRange(1e3, 50e9)
        fs_box.setDecimals(2)
        fs_box.setSingleStep(1e6)
        fs_box.setValue(1e9)
        fs_box.setToolTip("Sampling frequency (Manual Override)")
        self.widgets["fs_hz"] = fs_box
        h.addWidget(fs_box, stretch=1)
        h.addStretch()
        return row

    def _build_fft_row(self) -> QWidget:
        """Build manual range and Doppler FFT-size fallback controls."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(QLabel("FFT Range:"))
        fft_range = QSpinBox()
        fft_range.setRange(64, 4096)
        fft_range.setValue(256)
        fft_range.setSingleStep(64)
        self.widgets["fft_range"] = fft_range
        h.addWidget(fft_range)

        h.addWidget(QLabel("Doppler:"))
        fft_dopp = QSpinBox()
        fft_dopp.setRange(8, 1024)
        fft_dopp.setValue(64)
        fft_dopp.setSingleStep(8)
        self.widgets["fft_doppler"] = fft_dopp
        h.addWidget(fft_dopp)
        h.addStretch()
        return row

    def _build_prf_row(self) -> QWidget:
        """Build manual pulse-repetition-frequency fallback controls."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(QLabel("PRF (Hz):"))
        prf_box = QDoubleSpinBox()
        prf_box.setRange(0.1, 5e3)
        prf_box.setDecimals(2)
        prf_box.setSingleStep(1.0)
        prf_box.setValue(30.0)
        self.widgets["prf_hz"] = prf_box
        h.addWidget(prf_box, stretch=1)
        h.addStretch()
        return row

    def _build_display_row(self) -> QWidget:
        """Build inline controls for 3D sensing overlays and CPI pinning."""
        # Build directly in the final owners. A temporary row followed by
        # reparenting can silently delete any control omitted from the move.
        compact = QWidget()
        layout = QVBoxLayout(compact)
        layout.setContentsMargins(0, 0, 0, 0)
        common = WrappingLayout()
        layout.addLayout(common)
        options = QGroupBox("3D sensing options", compact)
        self.widgets["scene_options"] = options
        layout.addWidget(options)
        options_layout = QVBoxLayout(options)
        options_layout.setSpacing(6)
        scene_controls = WrappingLayout()
        options_layout.addLayout(scene_controls)
        flags = WrappingLayout()
        flags.addWidget(QLabel("Markers:"))
        options_layout.addLayout(flags)
        selection_controls = WrappingLayout()
        selection_controls.addWidget(QLabel("Selection:"))
        options_layout.addLayout(selection_controls)
        context_controls = WrappingLayout()
        context_controls.addWidget(QLabel("Context:"))
        options_layout.addLayout(context_controls)
        app_state = getattr(self.parent, "app_state", None)

        pin_fresh = QCheckBox("Pin result")
        pin_fresh.setChecked(bool(state_value(app_state, "pin_freshest")))
        pin_fresh.setToolTip(
            "Explicit view override: keep the result with the latest physical reference "
            "time even when it is bound to another scene. Bound/stale provenance remains visible."
        )
        pin_fresh.stateChanged.connect(self._on_pin_freshest_changed)
        self.widgets["pin_fresh_sensing"] = pin_fresh

        show_det_3d = QCheckBox("Show 3D sensing")
        show_det_3d.setChecked(bool(state_value(app_state, "show_detections")))
        show_det_3d.setToolTip(
            "Overlay sensing products in the 3D scene.\n"
            "Modes include raw detections, tracked entities, trails,\n"
            "and a temporal history cloud. Missing elevation is drawn in the\n"
            "sensor plane; that marker height is assumed, not measured."
        )
        show_det_3d.stateChanged.connect(self._on_show_detections_changed)
        self.widgets["show_sensing_detections"] = show_det_3d
        common.addWidget(show_det_3d)

        scene_controls.addWidget(QLabel("Show:"))
        scene_mode = QComboBox()
        scene_mode.addItem("Current detections", "detections")
        scene_mode.addItem("Tracks and trails", "tracks")
        scene_mode.addItem("Detection history", "evidence")
        scene_mode.addItem("All sensing evidence", "full")
        scene_mode.setToolTip(
            "Detections: current 3D detections only.\n"
            "Tracks: confirmed/tentative track anchors and trails.\n"
            "History cloud: temporal detection accumulation.\n"
            "Full: detections + tracks + trails + history cloud.\n"
            "History length still applies; enable Full history for all prior results."
        )
        current_mode = str(state_value(app_state, "scene_mode"))
        mode_index = scene_mode.findData(current_mode)
        if mode_index >= 0:
            scene_mode.setCurrentIndex(mode_index)
        scene_mode.currentIndexChanged.connect(self._on_sensing_scene_mode_changed)
        self.widgets["sensing_scene_mode"] = scene_mode
        scene_controls.addWidget(scene_mode)
        history_label = QLabel("Recent results:")
        self.widgets["sensing_scene_history_label"] = history_label
        history_box = QSpinBox()
        history_box.setRange(1, 64)
        history_box.setMaximumWidth(65)
        history_box.setValue(int(state_value(app_state, "history_frames")))
        history_box.setToolTip(
            "Number of sensing results used for trails and the temporal history cloud."
        )
        history_box.valueChanged.connect(self._on_sensing_scene_history_changed)
        self.widgets["sensing_scene_history_frames"] = history_box

        full_history = QCheckBox("Full history")
        full_history.setChecked(bool(state_value(app_state, "full_history")))
        full_history.setToolTip(
            "Use all saved results up to this scene for trails and the history cloud. "
            "History stays visible between results at its saved positions."
        )
        full_history.stateChanged.connect(self._on_sensing_scene_full_history_changed)
        self.widgets["sensing_scene_full_history"] = full_history
        scene_controls.addWidget(full_history)
        scene_controls.addWidget(history_label)
        scene_controls.addWidget(history_box)

        velocity_arrows = QCheckBox("Velocity arrows")
        velocity_arrows.setChecked(bool(state_value(app_state, "show_velocity_arrows")))
        velocity_arrows.setToolTip(
            "Show per-track velocity vectors when tracking metadata contains vx/vy/vz."
        )
        velocity_arrows.stateChanged.connect(self._on_sensing_velocity_arrows_changed)
        self.widgets["sensing_scene_show_velocity_arrows"] = velocity_arrows

        track_labels = QCheckBox("Track labels")
        track_labels.setChecked(bool(state_value(app_state, "show_track_labels")))
        track_labels.setToolTip(
            "Show track identity labels in the 3D scene when supported by the renderer."
        )
        track_labels.stateChanged.connect(self._on_sensing_track_labels_changed)
        self.widgets["sensing_scene_show_track_labels"] = track_labels

        truth = QCheckBox("Ground truth")
        truth.setChecked(bool(state_value(app_state, "show_truth")))
        truth.setToolTip(
            "Show the stored target reference position at the result time; this is not a detection."
        )
        truth.toggled.connect(self._on_scene_truth_changed)
        self.widgets["sensing_scene_show_truth"] = truth
        flags.addWidget(truth)
        flags.addWidget(track_labels)
        flags.addWidget(velocity_arrows)

        focus = QCheckBox("Focus selected evidence")
        focus.setToolTip(
            "Paused-frame presentation: dim background overlays without changing physical geometry or data. Selection/frame changes restore the view."
        )
        focus.toggled.connect(self._on_evidence_focus_changed)
        flags.addWidget(focus)
        self.widgets["evidence_focus"] = focus

        from ..sensing.uncertainty_controls import TrackingUncertaintyControls

        self.uncertainty_controls = TrackingUncertaintyControls()
        self.uncertainty_controls.embed_toggle(flags)
        options_layout.addWidget(self.uncertainty_controls)
        self.uncertainty_controls.changed.connect(self._on_uncertainty_changed)

        marker_glyphs = QCheckBox("Marker shapes")
        marker_glyphs.setChecked(bool(state_value(app_state, "use_marker_glyphs")))
        marker_glyphs.setToolTip(
            "Distinguish detections, tracks and history by marker shape when supported by the renderer."
        )
        marker_glyphs.stateChanged.connect(self._on_sensing_marker_glyphs_changed)
        self.widgets["sensing_scene_use_marker_glyphs"] = marker_glyphs
        flags.addWidget(marker_glyphs)
        show_paths = QCheckBox("RT paths")
        show_paths.setToolTip(
            "Show or hide the main scene's saved propagation paths. Between analytical-Doppler "
            "solves, show labelled geometry from the source solve. No ray tracing is rerun."
        )
        visibility = getattr(app_state, "mpc_visibility", None)
        show_paths.setChecked(bool(visibility and visibility.enabled and visibility.paths))
        show_paths.toggled.connect(self._show_recorded_paths)
        self.widgets["show_recorded_paths"] = show_paths
        context_controls.addWidget(show_paths)
        context_controls.addWidget(pin_fresh)

        highlight = QCheckBox("Highlight selection")
        highlight.setChecked(bool(state_value(app_state, "highlight_selection")))
        highlight.setToolTip("Emphasize the selected saved estimate in the 3D scene.")
        highlight.toggled.connect(self._on_scene_highlight_changed)
        self.widgets["sensing_scene_highlight_selection"] = highlight
        selection_controls.addWidget(highlight)
        for key, label, attribute in (
            ("scene_range_shell", "Range shell", "range_shell_enabled"),
            ("scene_range_cutaway", "Cutaway", "range_shell_cutaway"),
        ):
            option = QCheckBox(label)
            option.toggled.connect(partial(self._set_scene_range_option, attribute))
            self.widgets[key] = option
            selection_controls.addWidget(option)
        focus = QPushButton("Focus in 3D")
        focus.clicked.connect(self._locate_selection)
        self.widgets["scene_focus_selection"] = focus
        selection_controls.addWidget(focus)
        frame_shell = QPushButton("Frame shell")
        frame_shell.clicked.connect(self._frame_range_shell)
        self.widgets["scene_frame_range_shell"] = frame_shell
        selection_controls.addWidget(frame_shell)
        self._sync_scene_selection_controls()
        self._update_scene_overlay_controls_enabled(show_det_3d.isChecked())
        return compact

    def _build_buttons(self) -> None:
        """Build saved-result actions and the existing preset controls."""
        fusion_btn = QPushButton("Open saved result…", self._container)
        menu = QMenu(fusion_btn)
        menu.addAction("Tracking / fusion result…", self._on_open_fusion)
        volume_action = menu.addAction("Detection outcome volume…", self._on_open_volume)
        self.widgets["open_volume_action"] = volume_action
        menu.setToolTipsVisible(True)
        menu.aboutToShow.connect(self._update_volume_availability)
        self._update_volume_availability()
        fusion_btn.setMenu(menu)
        fusion_btn.setEnabled(HAS_SENSING_WINDOW)
        self.widgets["open_saved_result"] = fusion_btn
        fusion_btn.hide()  # Standalone result tools are outside the recording workflow.

        presentation_btn = QPushButton("Show all 3D evidence", self._container)
        presentation_btn.setToolTip(
            "Apply a sensing presentation overlay: full history, track labels, "
            "velocity arrows, and glyph markers."
        )
        presentation_btn.clicked.connect(self._on_presentation_preset_clicked)
        self.widgets["btn_sensing_presentation"] = presentation_btn
        presentation_btn.hide()  # The explicit scene controls supersede this bundled shortcut.

        preset_box = QComboBox(self._container)
        for preset_id, spec in self.SENSING_PRESETS.items():
            preset_box.addItem(spec["label"], preset_id)
        preset_box.setToolTip("Choose the sensing analysis mode")
        preset_box.currentIndexChanged.connect(self._on_preset_changed)
        self.widgets["sensing_preset"] = preset_box
        preset_box.hide()  # The dashboard owns the single reader-facing View selector.

        variant_box = QComboBox(self._container)
        variant_box.addItem("Auto (primary\u2192raw)", "auto")
        variant_box.addItem("Clean (primary product)", "clean")
        variant_box.addItem("Raw (explicit product)", "raw")
        variant_box.setToolTip(
            "Select an explicit schema-v4 product variant. Clean maps to primary; "
            "an unavailable raw request never falls back to primary."
        )
        self.widgets["rd_variant"] = variant_box
        variant_box.currentIndexChanged.connect(self._on_variant_changed)
        variant_box.hide()  # Product selection lives in the dashboard's Display settings.

    def _start_auto_refresh(self) -> None:
        """Start periodic refresh for dashboard/status while the panel exists."""
        if self._refresh_timer is None:
            self._refresh_timer = QTimer(self._container)
            self._refresh_timer.setInterval(_IDLE_REFRESH_MS)
            self._refresh_timer.timeout.connect(self._refresh_live_plots)
        self._refresh_timer.start()

    # Actions
    def _dashboard_is_visible(self) -> bool:
        """Return whether the sensing analysis window currently needs frame data."""
        if not self._sensing_figures_enabled():
            return False
        window = getattr(self, "_sensing_window", None)
        is_visible = getattr(window, "isVisible", None)
        try:
            return bool(
                callable(is_visible)
                and (
                    dashboard_is_visible(window)
                    or getattr(window, "has_visible_live_figures", lambda: False)()
                    or getattr(window, "_live_catalog_requested", False)
                )
            )
        except RuntimeError:
            self._sensing_window = None
            return False

    def _on_dashboard_destroyed(self, generation: int, *_args: Any) -> None:
        """Release only the window whose destruction signal is being delivered."""
        if generation == self._dashboard_generation:
            self._sensing_window = None
            self._last_cir_identity = None
            self._last_display_identity = None
            if not self._closed:
                self._sync_scene_selection_controls()

    def retire_recording_source(self) -> None:
        """Retire source-owned plans even while this panel remains reusable."""
        preparation = getattr(self, "preparation", None)
        if preparation is not None:
            preparation.close()
        self.preparation = None
        self._bound_recording_identity = None
        self.recording_figure_policy = None
        self.recording_3d_override = None
        self.recording_cache_policy = None
        self.recording_motion_mode = "auto"
        self._initial_figure_view_pending = False
        self._dashboard_view_identity = None
        playback = getattr(self, "playback", None)
        if playback is not None:
            playback.close()
        self._retire_dashboard_session()
        self._dashboard_cache = None
        window = getattr(self, "_sensing_window", None)
        if window is not None:
            window.reset_recording(preserve_options=True)

    def bind_recording_source(self) -> None:
        """Capture opening intent once; build navigation independently of 3D."""
        from ..sensing.dashboard_data import DashboardSession

        provider = visualizer_frame_provider(self.parent)
        identity = (
            *DashboardSession.provider_identity(provider),
            id(getattr(provider, "frame_ids_snapshot", None)),
        )
        if self._bound_recording_identity == identity or self._closed:
            return
        if self._bound_recording_identity is not None:
            self.retire_recording_source()
        self._bound_recording_identity = identity
        focused = focused_sensing_figure_policy()
        self.recording_figure_policy = focused or sensing_figure_policy()
        self.recording_3d_override = sensing_3d_override()
        self.recording_cache_policy = sensing_cache_policy()
        self.recording_motion_mode = initial_sensing_motion_mode()
        if self.recording_3d_override is None and focused:
            self.recording_3d_override = False
        self._initial_figure_view_pending = focused is not None
        self.parent.sensing_playback = self.playback
        automated = bool(getattr(self.parent, "_cli_driven_frame_run", False))
        if (
            self.recording_3d_override is not None
            and getattr(self.parent, "app_state", None) is not None
        ):
            update_state(
                self.parent,
                show_detections=self.recording_3d_override,
                **({"show_tracking_uncertainty": False} if not self.recording_3d_override else {}),
            )
            self._sync_runtime_controls()
        if (
            (automated or self.recording_figure_policy == "prepare")
            and self.recording_3d_override is not False
            or bool(state_value(getattr(self.parent, "app_state", None), "show_detections"))
        ):
            self.start_scene_preparation(prepare_playback=True)
        elif (
            not automated
            and not (self.recording_figure_policy == "off" and self.recording_3d_override is False)
            and isinstance(provider, Hdf5Provider)
            and self._get_service() is not None
        ):
            with QSignalBlocker(self.playback.checkbox):
                self.playback.checkbox.setChecked(True)
            self.playback.set_enabled(True, automatic=True)

    def start_scene_preparation(self, *, prepare_playback: bool = False) -> None:
        """Prepare requested 3D data once without restarting an active timeline."""
        if (
            self._closed
            or getattr(self.parent, "_shutdown_started", False)
            or self.preparation is not None
            or self._bound_recording_identity is None
        ):
            return
        from ..sensing.dashboard_data import DashboardSession
        from ..sensing.preparation_controller import SensingPreparation

        provider = visualizer_frame_provider(self.parent)
        identity = (
            *DashboardSession.provider_identity(provider),
            id(getattr(provider, "frame_ids_snapshot", None)),
        )
        if identity != self._bound_recording_identity:
            return
        self.preparation = SensingPreparation(self, prepare_playback=prepare_playback)

    def _apply_initial_figure_view(self) -> None:
        """Apply focused opening intent after the dashboard's source reset."""
        if not self._initial_figure_view_pending or self._sensing_window is None:
            return
        self._initial_figure_view_pending = False
        view = "micro-doppler" if self.recording_figure_policy == "motion" else "summary"
        window = self._sensing_window
        window.apply_display_preset(
            {"view_mode": view, "micro_doppler_mode": self.recording_motion_mode}
        )
        window._initial_view_selected = True

    def cleanup(self) -> None:
        """Stop refresh callbacks and close the separately owned dashboard."""
        if self._closed:
            return
        self._closed = True
        from ..services.evidence_focus import stop_evidence_focus

        stop_evidence_focus(self.parent)
        preparation = getattr(self, "preparation", None)
        if preparation is not None:
            preparation.close()
        playback = getattr(self, "playback", None)
        if playback is not None:
            playback.close()
        self.saved_track_scene.close()
        from ..sensing.runtime import refresh_tracking_uncertainty

        refresh_tracking_uncertainty(self.parent, clear=True)
        renderer = getattr(self.parent, "renderer", None)
        if renderer is not None:
            self.saved_track_scene.overlay.clear(renderer)
        self._retire_dashboard_session()
        timer, self._refresh_timer = self._refresh_timer, None
        if timer is not None:
            timer.stop()
            timer.deleteLater()
        poll_timer, self._dashboard_poll_timer = self._dashboard_poll_timer, None
        if poll_timer is not None:
            poll_timer.stop()
            poll_timer.deleteLater()
        if self.dashboard_host is not None:
            self.dashboard_host.dispose()
        window, self._sensing_window = self._sensing_window, None
        self._dashboard_generation += 1
        if window is not None:
            try:
                window.close()
            except RuntimeError:
                pass  # A deferred Qt deletion may already have destroyed it.
        self.widgets.clear()
        self._container = None

    def _ensure_sensing_frame(
        self,
        step: int,
        cached: Optional[dict],
    ) -> Optional[dict]:
        """Load all dashboard sensing products without rereading packed geometry."""
        provider = visualizer_frame_provider(self.parent)
        if provider is None:
            if cached is not None:
                _validated_sensing_block(cached)
            return cached
        request = visual_frame_read_request(include_sensing=True)
        try:
            if cached is not None and PACKED_PROJECTION_KEY in cached:
                loaded = try_upgrade_packed_visual_frame(
                    provider,
                    step,
                    cached,
                    request=request,
                    points_dtype=getattr(
                        getattr(self.parent, "mpc_core", None),
                        "canon_points_dtype",
                        np.float32,
                    ),
                )
            elif cached is None:
                loaded = try_load_packed_visual_frame(
                    provider,
                    step,
                    request=request,
                    points_dtype=getattr(
                        getattr(self.parent, "mpc_core", None),
                        "canon_points_dtype",
                        np.float32,
                    ),
                )
            else:
                _validated_sensing_block(cached)
                return cached
        except UnsupportedSensingSchemaError:
            raise
        except (OSError, KeyError, ValueError) as exc:
            logger.debug("Unable to load sensing projection for frame %s: %s", step, exc)
            if cached is not None:
                _validated_sensing_block(cached)
            return cached
        if loaded is None:
            if cached is not None:
                _validated_sensing_block(cached)
            return cached
        _validated_sensing_block(loaded)
        cache = getattr(self.parent, "cache_service", None)
        if cache is not None:
            cache.store_frame(step, loaded)
        return loaded

    def _project_source_frame(
        self,
        frame: Any,
        *,
        include_sensing: bool,
    ) -> Optional[dict[str, Any]]:
        """Convert a complete source frame into the panel's visual payload."""
        if frame is None:
            return None
        return standard_frame_to_visual_frame(
            frame,
            request=visual_frame_read_request(include_sensing=include_sensing),
            points_dtype=getattr(
                getattr(self.parent, "mpc_core", None),
                "canon_points_dtype",
                np.float32,
            ),
        )

    def _resolve_frame(self) -> Optional[dict]:
        """Load only the exact current app-state frame."""
        viz = self.parent
        step = getattr(viz, "app_state", None).step if hasattr(viz, "app_state") else None
        if step is None:
            return None
        cached = viz.cache_service.get_frame(step)
        if cached is not None:
            return (
                self._ensure_sensing_frame(step, cached) if self._dashboard_is_visible() else cached
            )
        if self._dashboard_is_visible():
            projected = self._ensure_sensing_frame(step, None)
            if projected is not None:
                return projected

        provider = visualizer_frame_provider(viz)
        if provider is not None:
            try:
                projected = try_load_packed_visual_frame(
                    provider,
                    step,
                    request=visual_frame_read_request(),
                    points_dtype=getattr(
                        getattr(viz, "mpc_core", None),
                        "canon_points_dtype",
                        np.float32,
                    ),
                )
            except (OSError, KeyError, ValueError):
                projected = None
            if projected is not None:
                viz.cache_service.store_frame(step, projected)
                return projected

        if hasattr(viz, "frame_source") and viz.frame_source is not None:
            try:
                payload = viz.frame_source.load_frame(step)
                return self._project_source_frame(
                    payload,
                    include_sensing=self._dashboard_is_visible(),
                )
            except UnsupportedSensingSchemaError:
                raise
            except (OSError, KeyError, ValueError) as exc:
                logger.warning("Failed to load frame %s for sensing: %s", step, exc)
        return None

    def _get_cached_frame(self, step: int) -> Optional[dict]:
        """Return frame data by step index, preferring cache and falling back to source."""
        viz = self.parent
        cache = getattr(viz, "cache_service", None)
        if cache is not None:
            cached = cache.get_frame(step)
            if cached is not None:
                loaded = self._ensure_sensing_frame(step, cached)
                if loaded is not None:
                    _validated_sensing_block(loaded)
                return loaded
        projected = self._ensure_sensing_frame(step, None)
        if projected is not None:
            _validated_sensing_block(projected)
            return projected
        frame_source = getattr(viz, "frame_source", None)
        if frame_source is not None and hasattr(frame_source, "load_frame"):
            try:
                payload = frame_source.load_frame(step)
            except UnsupportedSensingSchemaError:
                raise
            except (OSError, KeyError, ValueError, RuntimeError):
                return None
            projected = self._project_source_frame(payload, include_sensing=True)
            if projected is not None:
                _validated_sensing_block(projected)
            return projected
        return None

    def _get_service(self) -> Optional[Any]:
        """Return the sensing service attached during scenario setup."""
        services = getattr(self.parent, "extension_services", {})
        if not isinstance(services, dict):
            return None
        return services.get(FEATURE_KEY)

    @staticmethod
    def _has_displayable_sensing_products(sensing_block: Optional[dict[str, Any]]) -> bool:
        """Return whether exact typed RP or RD product descriptors are usable."""
        _assert_sensing_block(sensing_block)
        if not isinstance(sensing_block, dict) or not sensing_block:
            return False
        return bool(
            range_profile_product_variants(sensing_block)
            or range_doppler_product_variants(sensing_block)
        )

    def _resolve_display_selection(
        self,
        service: Any,
        scene_frame: dict[str, Any],
        scene_frame_id: int | str,
    ) -> Any:
        """Resolve dashboard products through schema-v4 descriptors and display binding."""
        return service.resolve_display_sensing_frame(
            scene_frame,
            scene_frame_id=scene_frame_id,
            storage_frame_id=scene_frame_id,
            product_names=(
                "range_profile",
                "range_doppler",
            ),
            load_all_sensing_products=True,
            pin_freshest=self._prefer_fresh_sensing_display(),
            load_scene_frame=self._get_cached_frame,
        )

    def _on_scene_truth_changed(self, checked: bool) -> None:
        """Toggle evaluation markers without reading or changing sensing products."""
        update_state(self.parent, show_truth=checked)
        self.parent.force_update_next_frame = True
        self.parent.schedule_update()

    def _on_open_fusion(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self._container,
            "Open saved tracking / fusion result",
            "",
            "Fusion result (fusion_results.json);;JSON (*.json)",
        )
        if path:
            from ..sensing.fusion_window import open_fusion_window

            open_fusion_window(self.parent, path)

    def _update_volume_availability(self) -> None:
        """Refresh after deferred renderer initialization as well as on first build."""
        action = self.widgets["open_volume_action"]
        reason = volume_rendering_unavailable_reason(getattr(self.parent, "renderer", None))
        action.setEnabled(not reason)
        action.setToolTip(reason or "Inspect saved detection outcomes in the 3D scene")

    def _on_open_volume(self) -> None:
        reason = volume_rendering_unavailable_reason(getattr(self.parent, "renderer", None))
        if reason:
            self._set_status(reason)
            return
        path, _ = QFileDialog.getOpenFileName(
            self._container,
            "Open detection outcome volume",
            "",
            "Sensing volume (sensing_volume.json);;JSON (*.json)",
        )
        if path:
            from ..sensing.volume_window import open_sensing_volume

            try:
                open_sensing_volume(self.parent, path)
            except ValueError as exc:
                self._set_status(str(exc))

    def ensure_live_figure_dashboard(self):
        """Prepare the normal dashboard's figure catalog without changing tabs."""
        if not self._sensing_figures_enabled():
            return None
        self._activate_dashboard()
        window = self._sensing_window
        if window is not None:
            window._live_catalog_owner = True
        if window is not None and window._last_request is None:
            window._live_catalog_requested = True
            self._refresh_live_plots(force=True)
        return window

    def _activate_dashboard(self) -> None:
        """Create plots on first visit, then reuse them across tab/placement changes."""
        if not HAS_SENSING_WINDOW or self._closed or not self._sensing_figures_enabled():
            return
        if self._sensing_window is None or getattr(self._sensing_window, "_plots_released", False):
            self._loading_paused_identity = None
            self._dashboard_generation += 1
            self._sensing_window = SensingDashboard(parent=self.dashboard_host)
            self._sensing_window.view_changed.connect(self._on_dashboard_options_changed)
            self._sensing_window.result_step_requested.connect(self._navigate_result)
            self._sensing_window.result_seek_requested.connect(self._seek_result)
            self._sensing_window.selection_changed.connect(self._on_linked_selection)
            self._sensing_window.range_shell_changed.connect(self._sync_range_shell)
            self._sensing_window.range_shell_frame_requested.connect(self._frame_range_shell)
            self._sensing_window.path_inspection_requested.connect(self._inspect_cell_paths)
            shell_available = (
                getattr(getattr(self.parent, "renderer", None), "renderer_id", None) == "pygfx"
            )
            self._sensing_window.range_shell_enabled.setEnabled(shell_available)
            self._sensing_window.range_shell_cutaway.setEnabled(shell_available)
            explanation = getattr(self._sensing_window, "scene_explanation", None)
            if explanation is not None:
                explanation.setEnabled(shell_available)
            scene_controls = getattr(self._sensing_window, "scene_selection_controls", None)
            if scene_controls is not None:
                scene_controls.hide()
            if not shell_available:
                self._sensing_window.range_shell_status.setText(
                    "Range shells require the pygfx renderer"
                )
            self._sensing_window.loading_control_requested.connect(self._toggle_dashboard_loading)
            self._sensing_window.locate_selection_requested.connect(self._locate_selection)
            self._sensing_window.destroyed.connect(
                partial(self._on_dashboard_destroyed, self._dashboard_generation)
            )
            self._last_cir_identity = None
            self._last_display_identity = None
            self._apply_dashboard_preset()
            if self.dashboard_host is not None:
                self.dashboard_host.set_dashboard(self._sensing_window)
            else:
                self._sensing_window.show()
        self._sync_scene_selection_controls()
        self._refresh_live_plots()

    def _on_open_dashboard(self) -> None:
        """Reveal analysis for commands linked from saved results or 3D evidence."""
        if self._closed:
            return
        manager = getattr(self.parent, "ui_manager", None)
        reveal = getattr(manager, "show_panel", None)
        if callable(reveal):
            reveal("sensing")
        if not self._sensing_figures_enabled():
            self._set_status("Enable Load sensing figures to inspect saved results.")
            return
        if self.results_section is not None:
            self.results_section.expand()
        self._activate_dashboard()
        if self.dashboard_host is not None and self.dashboard_host.is_detached():
            window = self.dashboard_host.floating_window
            window.raise_()
            window.activateWindow()

    def _on_dashboard_options_changed(self) -> None:
        """Load newly requested products without rebuilding hidden views."""
        self._completed_dashboard_key = None
        self._refresh_live_plots(force=True)

    def _toggle_dashboard_loading(self) -> None:
        from ..sensing.dashboard_data import DashboardSession

        if self._loading_paused_identity is not None:
            self._loading_paused_identity = None
            self._refresh_live_plots(force=True)
        else:
            self._loading_paused_identity = DashboardSession.provider_identity(
                visualizer_frame_provider(self.parent)
            )
            self._retire_dashboard_session()
            self._sensing_window.set_loading_progress(
                "Loading cancelled", 0, 0, active=False, cancelled=True
            )
            self._set_status("Loading cancelled; use Resume loading to continue.")

    def _navigate_result(self, direction: int) -> None:
        """Seek by saved scene bindings, without modifying the user's stride."""
        from ..sensing.result_navigation import adjacent_result

        session = self._dashboard_session
        if session is None:
            return
        result = adjacent_result(
            session.navigation,
            self._current_step(),
            self.parent.get_available_animation_steps(),
            direction,
        )
        if result is not None:
            self._seek_result(result)

    def select_scene_estimate(self, selection: Any) -> None:
        """Open the dashboard for a clicked scene estimate and link its saved identity."""
        from ..sensing.fusion_inspection import FusionSelection

        if not self._sensing_figures_enabled():
            self._on_open_dashboard()
            return
        if isinstance(selection, FusionSelection):
            window = getattr(self.parent, "fusion_window", None)
            if window is not None and not window.closed:
                window.select_fusion(selection)
            elif self.saved_track_scene.binding is not None:
                reader = self.saved_track_scene.binding.reader
                if selection.run_id != reader.inspection_id:
                    return
                lineage = reader["sources"][self.saved_track_scene.binding.sensor_id].get(
                    "lineage", []
                )
                if not 0 <= selection.epoch_index < len(lineage):
                    return
                steps = self.parent.get_available_animation_steps()
                scene = lineage[selection.epoch_index]["scene_frame_id"]
                if scene not in steps:
                    return
                if self.parent.animation_running:
                    self.parent.animation_controller.toggle_animation()
                self.parent.ui_controller.handle_frame_input_changed(steps.index(scene) + 1)
                self._on_open_dashboard()
                window = self._sensing_window
                window._set_view_mode("tracking")
                window.tracking_inspector.selected_id = selection.track_id
                window.tracking_inspector.toggle.setChecked(True)
            return
        if self.parent.animation_running:
            self.parent.animation_controller.toggle_animation()
        self._on_open_dashboard()
        self._sensing_window.set_linked_selection(selection, notify=True)

    def _frame_range_shell(self) -> None:
        service = getattr(self.parent, "extension_services", {}).get("experiments")
        if service is not None:
            service.range_shell.frame()

    def _sync_range_shell(self, *_args) -> None:
        service = getattr(self.parent, "extension_services", {}).get("experiments")
        if service is not None and self._sensing_window is not None:
            service.sync_sensing_selection(self._sensing_window)
        self._sync_scene_selection_controls()

    def _set_scene_range_option(self, attribute: str, checked: bool) -> None:
        """Use the dashboard's exact selected-bin state from the scene controls."""
        if self._sensing_window is None:
            self._on_open_dashboard()
        if self._sensing_window is not None:
            getattr(self._sensing_window, attribute).setChecked(checked)

    def _sync_scene_selection_controls(self) -> None:
        """Mirror selection capabilities without opening details or reloading products."""
        window = self._sensing_window
        supported = getattr(getattr(self.parent, "renderer", None), "renderer_id", None) == "pygfx"
        for key, attribute in (
            ("scene_range_shell", "range_shell_enabled"),
            ("scene_range_cutaway", "range_shell_cutaway"),
        ):
            control = self.widgets.get(key)
            if control is None:
                continue
            source = getattr(window, attribute, None)
            with QSignalBlocker(control):
                control.setChecked(bool(source and source.isChecked()))
            control.setEnabled(supported and (source is None or source.isEnabled()))
            control.setToolTip(
                source.toolTip()
                if source is not None
                else "Requires pygfx and a selected saved range bin."
            )
            if key == "scene_range_shell" and source is not None and source.isChecked():
                control.setToolTip(window.range_shell_status.text())
        focus = self.widgets.get("scene_focus_selection")
        if focus is not None:
            source = getattr(window, "locate_selection_button", None)
            focus.setEnabled(bool(source and source.isEnabled()))
            focus.setToolTip(source.toolTip() if source else "Select a saved spatial estimate.")
        frame = self.widgets.get("scene_frame_range_shell")
        if frame is not None:
            selection = getattr(window, "_linked_selection", None)
            shell = getattr(window, "range_shell_enabled", None)
            frame.setEnabled(
                bool(supported and shell and shell.isChecked() and selection and selection.cell)
            )
            frame.setToolTip("Fit the selected range-bin geometry in the 3D scene.")

    def _inspect_cell_paths(self) -> None:
        window = self._sensing_window
        service = getattr(self.parent, "extension_services", {}).get("experiments")
        if window is None:
            return
        try:
            if service is None:
                raise ValueError("Path Analysis requires the pygfx renderer")
            service.open_cell_paths(window)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            window.cell_explanation.paths_note.setText(f"Path Analysis unavailable: {exc}")

    def _on_linked_selection(self, selection: Any) -> None:
        """Pause on the selected result's saved scene and update marker emphasis."""
        from ..sensing.fusion_inspection import FusionSelection
        from ..sensing.runtime import refresh_scene_selection

        refresh_scene_selection(self.parent, selection)
        self._sync_scene_selection_controls()
        if selection is not None:
            self.parent.force_update_next_frame = True
            self.parent.schedule_update()
            if self.parent.animation_running:
                self.parent.animation_controller.toggle_animation()
            if isinstance(selection, FusionSelection):
                return
            session = self._dashboard_session
            if session is not None:
                result = next(
                    (
                        r
                        for r in session.navigation
                        if r.context == selection.scope
                        and r.acquisition_id == selection.acquisition_id
                    ),
                    None,
                )
                if result is not None and result.scene_id != self._current_step():
                    self._seek_result(result)

    def _locate_selection(self) -> None:
        position = self.selected_camera_position()
        renderer = getattr(self.parent, "renderer", None)
        if position is None or renderer is None:
            self._set_status("No saved 3D position is available for this selection.")
            return
        camera = getattr(self.parent, "camera_controller", None)
        tools = camera.presentation() if camera and hasattr(camera, "presentation") else None
        if tools is not None:
            tools.focus(position)
        else:
            renderer.focus_camera(position)

    def selected_camera_position(self):
        """Return a saved estimate for camera framing, never inferred ground truth."""
        from ..sensing.runtime import selected_scene_position

        window = self._sensing_window
        if window is not None:
            return window.selected_world_position()
        return selected_scene_position(self.parent)

    def _seek_result(self, result: Any) -> None:
        """Pause playback and seek only to a binding from the current recording."""
        session = self._dashboard_session
        provider = visualizer_frame_provider(self.parent)
        if (
            session is None
            or session.source_identity != session.provider_identity(provider)
            or result not in session.navigation
        ):
            return
        steps = self.parent.get_available_animation_steps()
        if result.scene_id not in steps:
            self._set_status("The bound scene is not available in this recording.")
            return
        if self.parent.animation_running:
            self.parent.animation_controller.toggle_animation()
        self.parent.cancel_pending_slider_scrub()
        self.parent.ui_controller.handle_frame_input_changed(steps.index(result.scene_id) + 1)
        self._refresh_live_plots(force=True)

    def _update_result_navigation(self) -> None:
        from ..sensing.result_navigation import adjacent_result

        session, window = self._dashboard_session, self._sensing_window
        if session is None or window is None:
            return
        steps = self.parent.get_available_animation_steps()
        current = self._current_step()
        window.set_result_navigation(
            session.navigation,
            previous=adjacent_result(session.navigation, current, steps, -1) is not None,
            following=adjacent_result(session.navigation, current, steps, 1) is not None,
        )

    def _pause_dashboard_updates(self) -> None:
        """Keep the analysis and completed reads while another panel is visible."""
        session = self._dashboard_session
        provider = visualizer_frame_provider(self.parent)
        if session is not None and session.source_identity != session.provider_identity(provider):
            self._retire_dashboard_session()
            return
        if self._dashboard_poll_timer is not None:
            self._dashboard_poll_timer.stop()
        if self._refresh_timer is not None:
            self._refresh_timer.setInterval(_IDLE_REFRESH_MS)

    def _retire_dashboard_session(self) -> None:
        """Detach background readers before a scenario/window can be replaced."""
        if self._dashboard_session is not None:
            self._dashboard_session.close()
            self._dashboard_cache = None if self._closed else self._dashboard_session
            self._dashboard_session = None
        if self._closed:
            self._dashboard_cache = None
        if self._dashboard_poll_timer is not None:
            self._dashboard_poll_timer.stop()
        if self._refresh_timer is not None:
            self._refresh_timer.setInterval(_IDLE_REFRESH_MS)
        self._completed_dashboard_key = None

    def _refresh_dashboard_async(self, provider: Any, step: int, *, force: bool) -> None:
        if not self._sensing_figures_enabled():
            self._pause_dashboard_updates()
            return
        from ..sensing.dashboard_data import DashboardSession
        from ..sensing.qt_lifetime import ensure_gui_garbage_collection

        if self._loading_paused_identity is not None:
            if self._loading_paused_identity == DashboardSession.provider_identity(provider):
                return
            self._loading_paused_identity = None
        ensure_gui_garbage_collection()
        session = self._dashboard_session
        if session is None or session.source_identity != DashboardSession.provider_identity(
            provider
        ):
            self._retire_dashboard_session()
            previous = getattr(provider, "_prepared_sensing_dashboard", None)
            if previous is not None:
                del provider._prepared_sensing_dashboard
            else:
                previous = getattr(self, "_dashboard_cache", None)
            session = self._dashboard_session = DashboardSession(
                provider,
                previous=previous,
                source_path=getattr(self.parent, "current_scenario_path", None),
                saved_result_path=getattr(self, "saved_result_path", None),
                prepared=getattr(self._get_service(), "prepared_sensing", None),
                cache_policy=self.recording_cache_policy,
            )
            self._dashboard_cache = None
            self._sensing_window.reset_recording(
                preserve_options=self._dashboard_view_identity == session.source_identity
            )
            self._dashboard_view_identity = session.source_identity
            self._apply_initial_figure_view()
        if self._dashboard_poll_timer is None:
            self._dashboard_poll_timer = QTimer(self._container)
            self._dashboard_poll_timer.setInterval(_DASHBOARD_POLL_MS)
            self._dashboard_poll_timer.timeout.connect(self._poll_dashboard)
        # Playback shares Qt's thread with the scene. Explicit user actions
        # still request immediately; paused analysis retains its fast cadence.
        refresh_ms = (
            _DASHBOARD_PLAYBACK_REFRESH_MS
            if getattr(self.parent, "animation_running", False)
            else _DASHBOARD_REFRESH_MS
        )
        if self._refresh_timer is not None and self._refresh_timer.interval() != refresh_ms:
            self._refresh_timer.setInterval(refresh_ms)
        self._sync_dashboard_poll_timer()
        window = self._sensing_window
        window.environment.set_scene(
            (
                session.source_identity,
                getattr(getattr(self.parent, "scene_service", None), "scene_path", None),
            ),
            getattr(self.parent, "mesh_entries", None) or [],
        )
        if (
            window.request_view_mode() == "summary"
            and not getattr(window, "has_visible_live_figures", lambda: False)()
            and session.can_reuse_summary(window._last_request)
        ):
            # Playback and reopening cannot change whole-run scores. Poll only
            # the initial aggregate/settings work; do not reread each scene/CPI.
            self._update_result_navigation()
            self._poll_dashboard()
            return
        key = (
            step,
            window.request_view_mode(),
            window.product_variant(),
            self._dashboard_follows_results(),
            window._selected_micro_doppler_mode(),
            *getattr(window, "live_figure_request_suffix", lambda: ())(),
        )
        if (
            not force
            and window.can_move_compact_motion_cursor(session.overview)
            and window._last_request.source_identity == session.source_identity
            and not state_value(
                getattr(self.parent, "app_state", None), "show_tracking_uncertainty"
            )
        ):
            playback = getattr(self.parent, "sensing_playback", None)
            scene_time_s = getattr(playback, "timestamps", {}).get(step)
            if scene_time_s is None:
                # Ordinary playback has no sensing-only timestamp map. Its
                # already rendered raw frame supplies the same physical clock;
                # never invoke the source-loading panel helper for a cursor.
                get_frame = getattr(getattr(self.parent, "cache_service", None), "get_frame", None)
                if callable(get_frame):
                    try:
                        cached_frame = get_frame(step)
                    except (KeyError, RuntimeError, TypeError, ValueError):
                        cached_frame = None
                    if isinstance(cached_frame, dict):
                        scene_time_s = session.service._scene_time_s(cached_frame)
            cursor = session.compact_motion_cursor(
                step,
                key[2],
                window.overview_families(),
                pin_freshest=key[3],
                scene_time_s=scene_time_s,
            )
            previous_cursor = getattr(window, "_compact_motion_cursor", None)
            entering_result = (
                cursor is not None
                and cursor.result is not None
                and (
                    window._last_request.recording_overview_only
                    or previous_cursor is not None
                    and previous_cursor.result is None
                )
            )
            if cursor is not None and not entering_result:
                window.move_compact_motion_cursor(cursor, session.overview.variant)
                self._update_result_navigation()
                self._completed_dashboard_key = key
                self._set_status(window.provenance_label.text())
                label = self.widgets.get("detections_label")
                if isinstance(label, QLabel):
                    label.setText(f"Sensing: scene {step} | Saved Motion overview")
                return
        if (
            not force
            and window.can_move_recording_cursor(session.history)
            and session.history_key is not None
            and (
                key[2] == "auto"
                or ("primary" if key[2] == "clean" else key[2]) == session.history_key[-1]
            )
        ):
            playback = getattr(self.parent, "sensing_playback", None)
            record = session.motion_cursor(
                step,
                pin_freshest=key[3],
                scene_time_s=(getattr(playback, "timestamps", {}).get(step)),
            )
            if record is not None:
                window.move_recording_cursor(record, session.history_key[-1])
                self._update_result_navigation()
                self._completed_dashboard_key = key
                self._set_status(f"Result at {record.time_s:.3f} s · scene {step}")
                label = self.widgets.get("detections_label")
                if isinstance(label, QLabel):
                    label.setText(f"Sensing: scene {step} | Detections {len(record.detections)}")
                return
        if force or key != self._completed_dashboard_key:
            session.request(key)
        self._poll_dashboard()

    def _poll_dashboard(self) -> None:
        """Poll only while foreground results, history, or summaries need delivery."""
        try:
            self._apply_dashboard_completions()
        finally:
            self._sync_dashboard_poll_timer()

    def _sync_dashboard_poll_timer(self) -> None:
        """Retain wakeups for undelivered work, including cursor-only refreshes."""
        timer = self._dashboard_poll_timer
        if timer is None:
            return
        session = self._dashboard_session
        visible = not self._closed and session is not None and self._dashboard_is_visible()
        pending = visible and (
            session.pending is not None
            or session.history_pending is not None
            or session.overview_pending is not None
            or (
                session.run_summary.pending
                and self._sensing_window.request_view_mode() == "summary"
            )
        )
        if pending:
            if not timer.isActive():
                timer.start()
        else:
            timer.stop()
            if visible:
                # Summary completion can clear its future after the progress
                # update at the start of polling. Finish the row in this turn.
                self._sensing_window.set_loading_progress(*session.progress, active=False)

    def _apply_dashboard_completions(self) -> None:
        """Apply completed immutable results on Qt's thread, never from a worker."""
        session = self._dashboard_session
        if session is None:
            return
        provider = visualizer_frame_provider(self.parent)
        if session.source_identity != session.provider_identity(provider):
            self._retire_dashboard_session()
            return
        if not self._dashboard_is_visible():
            self._pause_dashboard_updates()
            return
        window = self._sensing_window
        progress = session.progress
        overview_active = session.overview_pending is not None and bool(window.overview_families())
        if overview_active and session.pending is None:
            progress = session.overview_progress
        if progress[2] and progress[1] >= progress[2]:
            # A completed index/overview stage is not the progress of a later
            # result request. Keep its stale 100% label out of the loading row.
            progress = (
                (
                    "Loading result"
                    if session.pending is not None and not session.pending.done()
                    else "Loading overview"
                ),
                0,
                0,
            )
        window.set_loading_progress(
            *progress,
            active=(session.pending is not None and not session.pending.done())
            or (session.history_pending is not None and not session.history_pending.done())
            or overview_active
            or (session.run_summary.pending and window.request_view_mode() == "summary"),
        )
        if session.pending is not None and session.pending.done():
            future, key = session.pending, session.pending_key
            session.pending = None
            try:
                request = future.result()
            except Exception as exc:  # Worker boundary: show errors without killing Qt's timer.
                self._set_unavailable(f"Sensing load failed: {exc}", step=self._current_step())
                self._completed_dashboard_key = key
            else:
                current_options = (
                    window.request_view_mode(),
                    window.product_variant(),
                    self._dashboard_follows_results(),
                    window._selected_micro_doppler_mode(),
                    *getattr(window, "live_figure_request_suffix", lambda: ())(),
                )
                current_step = self._current_step()
                advanced = False
                if request is not None and getattr(self.parent, "animation_running", False):
                    # A read may finish after playback advances. Present that
                    # completed snapshot with its own provenance, then request
                    # the newest one; otherwise every read can be discarded.
                    # Rewinds/loop wraps and paused seeks still reject old work.
                    steps = self.parent.get_available_animation_steps()
                    if key[0] in steps and current_step in steps:
                        advanced = steps.index(key[0]) < steps.index(current_step)
                if key[1:] != current_options or (key[0] != current_step and not advanced):
                    self._completed_dashboard_key = None
                    return
                self._completed_dashboard_key = key
                self._update_result_navigation()
                if request is None:
                    self._set_unavailable(
                        f"No sensing result is bound to scene {key[0]}.", step=key[0]
                    )
                else:
                    request = session.with_recording_data(request)
                    if not window.advance_playback_context(request):
                        window.update_from_request(request)
                    if state_value(
                        getattr(self.parent, "app_state", None), "show_tracking_uncertainty"
                    ):
                        from ..sensing.runtime import _sync_tracking_uncertainty

                        _sync_tracking_uncertainty(self.parent, self._get_service(), current_step)
                    if request.recording_overview_only:
                        self._set_status(
                            f"Recording overview · no sensing result at scene {current_step}"
                        )
                    else:
                        self._set_status(
                            f"Result at {window.result_time_s():.3f} s · scene {request.effective_display_step}"
                            + (f" · playback scene {current_step}" if advanced else "")
                        )
                        self._update_detection_label(
                            request.display_frame["sensing"],
                            state_step=request.state_step,
                            display_step=request.effective_display_step,
                            variant=request.variant,
                            display_from_cache=request.display_from_cache,
                        )
                    self._request_recording_data(session, request)
        if session.history_pending is not None and session.history_pending.done():
            future, session.history_pending = session.history_pending, None
            try:
                session.history = future.result()
            except Exception as exc:  # Background recording scan may encounter a retired source.
                from ..sensing.dashboard_data import RecordingHistory

                session.history = RecordingHistory((), error=str(exc))
            request = window._last_request
            if request is not None:
                from ..sensing.dashboard_data import result_context

                if (
                    *result_context(request.display_frame["sensing"]),
                    request.variant,
                ) == session.history_key and getattr(
                    window, "_compact_motion_cursor", None
                ) is None:
                    window.update_from_request(session.with_recording_data(request))
        if session.poll_overview() and window._last_request is not None:
            window.update_from_request(session.with_recording_data(window._last_request))
        request = window._last_request
        if (
            request is not None
            and window.request_view_mode() == "summary"
            and not (
                getattr(window, "_live_catalog_owner", False) and not dashboard_is_visible(window)
            )
        ):
            from ..sensing.dashboard_data import result_context

            summary = session.run_summary
            summary.request(
                result_context(request.display_frame["sensing"]),
                session.service.dashboard_inventory_token(),
                window.run_summary_panel.settings(),
            )
            summary.poll()
            if summary.error:
                window.run_summary_panel.clear("Summary unavailable: " + summary.error)
            elif summary.data is not None:
                window.run_summary_panel.set_summary(summary.data)
            else:
                window.run_summary_panel.set_loading(summary.progress)
        if request is not None:
            self._request_recording_data(session, request)
        playback = getattr(self.parent, "sensing_playback", None)
        if playback is not None and playback.steps:
            session.prefetch_next(
                (
                    self._current_step(),
                    window.request_view_mode(),
                    window.product_variant(),
                    True,
                    window._selected_micro_doppler_mode(),
                    *getattr(window, "live_figure_request_suffix", lambda: ())(),
                ),
            )

    def _request_recording_data(self, session, request) -> None:
        """Keep detailed history and visible spectral families independent."""
        window = self._sensing_window
        if window.wants_recording_history():
            session.request_history(
                request.display_frame["sensing"],
                request.variant,
                include_signal_products=False,
            )
        session.request_overview(
            request.display_frame["sensing"],
            request.variant,
            window.overview_families(),
        )

    def _on_variant_changed(self) -> None:
        """Force an immediate full refresh when the variant dropdown changes."""
        self._last_cir_identity = None
        self._last_display_identity = None
        self._refresh_live_plots(force=True, refresh_rd=True)

    def _selected_preset_spec(self) -> dict[str, Any]:
        """Return the active dashboard preset specification."""
        preset_box = self.widgets.get("sensing_preset")
        preset_id = preset_box.currentData() if isinstance(preset_box, QComboBox) else None
        preset_key = str(preset_id or "analysis")
        return self.SENSING_PRESETS.get(preset_key, self.SENSING_PRESETS["analysis"])

    def _apply_dashboard_preset(self) -> None:
        """Apply the active display preset to the open dashboard window."""
        if self._sensing_window is None:
            return
        self._sensing_window.apply_display_preset(self._selected_preset_spec()["dashboard"])

    def _on_preset_changed(self) -> None:
        """Apply preset display settings and refresh unless variant already did."""
        preset = self._selected_preset_spec()
        variant_box = self.widgets.get("rd_variant")
        variant_changed = False
        if isinstance(variant_box, QComboBox):
            variant_index = variant_box.findData(preset["variant"])
            if variant_index >= 0 and variant_index != variant_box.currentIndex():
                variant_changed = True
                variant_box.setCurrentIndex(variant_index)
        self._apply_dashboard_preset()
        if not variant_changed:
            self._refresh_live_plots(force=True, refresh_rd=True)

    def _set_widget_checked(self, key: str, checked: bool) -> None:
        """Set a checkbox by widget key without assuming the widget exists."""
        widget = self.widgets.get(key)
        if isinstance(widget, QCheckBox) and widget.isChecked() != bool(checked):
            widget.setChecked(bool(checked))

    def _on_presentation_preset_clicked(self) -> None:
        """Apply a scene-overlay preset for presentation-style sensing views."""
        viz = self.parent
        if viz is None:
            return
        self.start_scene_preparation()
        update_state(
            viz,
            show_detections=True,
            scene_mode="full",
            history_frames=64,
            full_history=True,
            show_velocity_arrows=True,
            show_track_labels=True,
            use_marker_glyphs=True,
        )
        self._set_widget_checked("show_sensing_detections", True)
        self._set_widget_checked("sensing_scene_full_history", True)
        self._set_widget_checked("sensing_scene_show_velocity_arrows", True)
        self._set_widget_checked("sensing_scene_show_track_labels", True)
        self._set_widget_checked("sensing_scene_use_marker_glyphs", True)

        mode_widget = self.widgets.get("sensing_scene_mode")
        if isinstance(mode_widget, QComboBox):
            mode_index = mode_widget.findData("full")
            if mode_index >= 0 and mode_index != mode_widget.currentIndex():
                mode_widget.setCurrentIndex(mode_index)

        history_widget = self.widgets.get("sensing_scene_history_frames")
        if isinstance(history_widget, QSpinBox):
            history_widget.setValue(64)

        self._update_scene_overlay_controls_enabled(True)
        viz.force_update_next_frame = True
        viz.schedule_update()

    def _set_status(self, text: str) -> None:
        """Update the compact status label."""
        status = self.widgets.get("status")
        if status:
            status.setText(f"Status: {text}")

    def _set_unavailable(self, text: str, *, step: int | str | None = None) -> None:
        """Clear current dashboard ownership as well as the compact status."""
        self._last_display_identity = None
        self._set_status(text)
        window = getattr(self, "_sensing_window", None)
        if self._dashboard_is_visible():
            window.set_unavailable(text, scene_frame_id=step)

    def _resolve_runtime_config(self, sensing_block: dict[str, Any]) -> dict[str, Any]:
        """Return exact persisted runtime parameters for a native-v4 result."""
        _assert_sensing_block(sensing_block)
        resolved = sensing_block.get("config_resolved")
        if not isinstance(resolved, Mapping):
            raise ValueError("Native sensing-v4 visualization requires config_resolved")
        required = (
            "sampling_frequency",
            "prf_hz",
            "fft_size_range",
            "fft_size_doppler",
            "carrier_freq_hz",
            "acquisition",
        )
        missing = [key for key in required if key not in resolved]
        if missing:
            raise ValueError(
                "Native sensing-v4 config_resolved is missing exact fields: " + ", ".join(missing)
            )
        acquisition = resolved["acquisition"]
        if not isinstance(acquisition, Mapping) or "cpi_length_samples" not in acquisition:
            raise ValueError(
                "Native sensing-v4 config_resolved.acquisition requires cpi_length_samples"
            )
        return {
            "sampling_frequency": float(resolved["sampling_frequency"]),
            "prf_hz": float(resolved["prf_hz"]),
            "fft_size_range": int(resolved["fft_size_range"]),
            "fft_size_doppler": int(resolved["fft_size_doppler"]),
            "carrier_freq_hz": float(resolved["carrier_freq_hz"]),
            "cir_time_steps": int(acquisition["cpi_length_samples"]),
        }

    def _prefer_fresh_sensing_display(self) -> bool:
        """Return whether the user explicitly pins the freshest physical result."""
        widget = self.widgets.get("pin_fresh_sensing")
        if isinstance(widget, QCheckBox):
            return bool(widget.isChecked())
        return bool(state_value(getattr(self.parent, "app_state", None), "pin_freshest"))

    def _dashboard_follows_results(self) -> bool:
        """Follow saved results during playback and in the Tracking history view.

        The service selects only results at or before the scene's physical
        time and retains their original binding. Tracking retains prior results
        during paused scrubbing too; other paused views use exact bindings
        unless the explicit hold option is set.
        """
        window = self._sensing_window
        return (
            self._prefer_fresh_sensing_display()
            or bool(getattr(self.parent, "animation_running", False))
            or (window is not None and window.request_view_mode() in {"tracking", "summary"})
        )

    def _update_scene_overlay_controls_enabled(self, enabled: bool) -> None:
        """Expose dependencies on the chosen overlay, retaining user preferences."""
        options = self.widgets.get("scene_options")
        if options is not None:
            options.setVisible(enabled)
        mode_widget = self.widgets.get("sensing_scene_mode")
        mode = mode_widget.currentData() if mode_widget is not None else "detections"
        history_visible = mode in {"tracks", "evidence", "full"}
        tracks_visible = mode in {"tracks", "full"}
        for key in (
            "sensing_scene_mode",
            "sensing_scene_full_history",
            "sensing_scene_show_velocity_arrows",
            "sensing_scene_show_track_labels",
            "sensing_scene_show_truth",
            "sensing_scene_use_marker_glyphs",
            "sensing_scene_highlight_selection",
        ):
            widget = self.widgets.get(key)
            if widget is not None:
                relevant = True
                if key == "sensing_scene_full_history":
                    relevant = history_visible
                elif key in {
                    "sensing_scene_show_velocity_arrows",
                    "sensing_scene_show_track_labels",
                }:
                    relevant = tracks_visible
                widget.setEnabled(bool(enabled) and relevant)
                if key in {
                    "sensing_scene_full_history",
                    "sensing_scene_show_velocity_arrows",
                    "sensing_scene_show_track_labels",
                }:
                    widget.setVisible(relevant)
                    widget.setToolTip(
                        {
                            "sensing_scene_full_history": "Use all saved results for trails or the history cloud. Choose a history or tracks overlay to activate.",
                            "sensing_scene_show_velocity_arrows": "Show saved Cartesian track velocity vectors. Choose a tracks overlay to activate; radial velocity alone cannot supply a vector.",
                            "sensing_scene_show_track_labels": "Show saved track IDs. Choose a tracks overlay to activate; labels require renderer support.",
                        }[key]
                    )
        history_widget = self.widgets.get("sensing_scene_history_frames")
        full_history_widget = self.widgets.get("sensing_scene_full_history")
        if history_widget is not None:
            use_full_history = (
                bool(full_history_widget.isChecked()) if full_history_widget else False
            )
            history_widget.setEnabled(bool(enabled) and history_visible and not use_full_history)
            history_widget.setVisible(history_visible and not use_full_history)
            label = self.widgets.get("sensing_scene_history_label")
            if label is not None:
                label.setVisible(history_visible and not use_full_history)
            history_widget.setToolTip(
                "Number of saved results in trails or history. Active for a history or tracks overlay when All saved history is off."
            )

    @staticmethod
    def _format_display_provenance(selection: Any) -> str:
        """Format explicit result/display identity without deriving time from IDs."""
        parts = [
            f"bound scene {selection.bound_scene_frame_id}",
            f"storage row {selection.storage_frame_id}",
            f"reference {selection.reference_time_s:.6g} s",
        ]
        if selection.current_scene_time_s is not None:
            parts.append(f"scene time {selection.current_scene_time_s:.6g} s")
        if selection.scene_result_time_offset_s is not None:
            parts.append(f"scene-result offset {selection.scene_result_time_offset_s:+.6g} s")
        if selection.stale_age_s is not None:
            parts.append(f"stale age {selection.stale_age_s:.6g} s")
        return "; ".join(parts)

    def _sync_runtime_controls(self) -> None:
        """Reflect scenario-scoped state without replaying stale widget values."""
        state = getattr(self.parent, "app_state", None)
        for widget_key, state_key in (
            ("pin_fresh_sensing", "pin_freshest"),
            ("show_sensing_detections", "show_detections"),
            ("sensing_scene_full_history", "full_history"),
            ("sensing_scene_show_velocity_arrows", "show_velocity_arrows"),
            ("sensing_scene_show_track_labels", "show_track_labels"),
            ("sensing_scene_show_truth", "show_truth"),
            ("sensing_scene_use_marker_glyphs", "use_marker_glyphs"),
            ("sensing_scene_highlight_selection", "highlight_selection"),
            ("show_rt_context", "show_rt_context"),
        ):
            widget = self.widgets.get(widget_key)
            value = bool(state_value(state, state_key))
            if isinstance(widget, QCheckBox) and widget.isChecked() != value:
                with QSignalBlocker(widget):
                    widget.setChecked(value)
        paths = self.widgets.get("show_recorded_paths")
        visibility = getattr(state, "mpc_visibility", None)
        if isinstance(paths, QCheckBox):
            with QSignalBlocker(paths):
                paths.setChecked(bool(visibility and visibility.enabled and visibility.paths))
        mode = self.widgets.get("sensing_scene_mode")
        if isinstance(mode, QComboBox):
            index = mode.findData(str(state_value(state, "scene_mode")))
            if index >= 0 and index != mode.currentIndex():
                with QSignalBlocker(mode):
                    mode.setCurrentIndex(index)
        history = self.widgets.get("sensing_scene_history_frames")
        if isinstance(history, QSpinBox):
            with QSignalBlocker(history):
                history.setValue(int(state_value(state, "history_frames")))
        self._update_scene_overlay_controls_enabled(bool(state_value(state, "show_detections")))

    # Refresh logic
    def _refresh_live_plots(self, force: bool = False, refresh_rd: bool = True) -> None:
        """Refresh compact status, optional dashboard, and sensing display data."""
        if getattr(self, "_closed", False) or getattr(self.parent, "_shutdown_started", False):
            return
        volume = getattr(self.parent, "sensing_volume_window", None)
        if volume is not None and not volume.closed:
            return
        self._sync_runtime_controls()
        pending = self.saved_track_scene.pending
        if pending is not None and pending.done():
            self.parent.force_update_next_frame = True
            self.parent.schedule_update()
        service = self._get_service()
        if service is None:
            self._set_unavailable(
                "Sensing service not initialized. Enable sensing in scenario YAML."
            )
            return
        if not self._dashboard_is_visible():
            # The 3D overlay has its own selective detection loader. A closed
            # dashboard must not load angle/CIR cubes or replay the processor
            # on the GUI thread every time this panel's timer fires.
            self._last_display_identity = None
            self._set_status("Select Sensing to inspect sensing products.")
            label = self.widgets.get("detections_label")
            if label is not None:
                label.setText(f"Scene {self._current_step()} · 3D visibility controls below")
            self._pause_dashboard_updates()
            return
        provider = visualizer_frame_provider(self.parent)
        current_step = self._current_step()
        if provider is not None and current_step is not None:
            self._refresh_dashboard_async(provider, current_step, force=force)
            return
        frame = self._resolve_frame()
        current_frame = frame
        step = self._current_step()

        if frame is None:
            self._set_unavailable("No frame loaded at step %s" % (step,), step=step)
            return

        if step is None:
            self._set_unavailable("No current scene ID is available for sensing display binding.")
            return

        try:
            selection = self._resolve_display_selection(service, frame, step)
        except (KeyError, ValueError, RuntimeError) as exc:
            self._set_unavailable(f"Typed sensing product selection failed: {exc}", step=step)
            return
        if selection is None:
            _validated_sensing_block(frame)
            self._set_unavailable(
                f"No range profile or RD result is explicitly bound to scene {step}.", step=step
            )
            return

        frame = selection.display_frame
        sensing_block = _validated_sensing_block(frame)
        try:
            runtime_cfg = self._resolve_runtime_config(sensing_block)
        except (KeyError, TypeError, ValueError) as exc:
            self._last_display_identity = None
            self._set_unavailable(
                f"Resolved sensing configuration is unavailable: {exc}", step=step
            )
            return
        fs = runtime_cfg["sampling_frequency"]
        prf = runtime_cfg["prf_hz"]
        fft_r = runtime_cfg["fft_size_range"]
        fft_d = runtime_cfg["fft_size_doppler"]

        display_identity = (
            step,
            selection.storage_frame_id,
            selection.bound_scene_frame_id,
            selection.reference_time_s,
            selection.pinned,
            (
                selection.replay_event.manifest_id
                if selection.replay_event is not None
                else selection.replay_reason
            ),
        )
        if not force and display_identity == self._last_display_identity:
            return

        try:
            rp_variants = range_profile_product_variants(sensing_block)
            rd_variants = range_doppler_product_variants(sensing_block)
        except (KeyError, ValueError, RuntimeError) as exc:
            self._set_unavailable(f"Typed sensing product validation failed: {exc}", step=step)
            return

        variant_selector = self.widgets.get("rd_variant")
        variant_preference = variant_selector.currentData() if variant_selector else "auto"
        variant_preference = (variant_preference or "auto").lower()

        rp_variant = _select_variant(variant_preference, rp_variants)
        rd_variant = _select_variant(variant_preference, rd_variants)

        if rp_variant is None and rd_variant is None:
            requested_variant = "primary" if variant_preference == "clean" else variant_preference
            if requested_variant in {"primary", "raw"} and (rp_variants or rd_variants):
                self._set_unavailable(
                    f"The explicitly bound result has no {requested_variant!r} typed "
                    "range-profile or range-Doppler product.",
                    step=step,
                )
            else:
                self._set_unavailable(
                    "The explicitly bound result has no typed range-profile or "
                    "range-Doppler product for display.",
                    step=step,
                )
            return

        using_cached_sensing = selection.storage_frame_id != step
        display_is_stale = selection.stale_for_current_scene

        # Keep CIR history in sync (used for micro-Doppler spectrogram).
        # When CPI-stack mode is active the generator stores pre-computed
        # RP/RD maps but no raw ``sensing.cir`` — so StoredCIRProvider raises
        # RuntimeError. The diagnostic is logged and RP/RD display continues
        # because it does not require per-frame CIR history.
        cir_identity = (selection.storage_frame_id, selection.bound_scene_frame_id)
        if cir_identity != self._last_cir_identity:
            try:
                service.compute_cir(frame, sampling_frequency=fs, store_in_history=True)
            except (KeyError, ValueError, RuntimeError):
                logger.debug(
                    "CIR history sync skipped (no stored CIR on storage frame %s)",
                    selection.storage_frame_id,
                )
            self._last_cir_identity = cir_identity

        rp_result = None
        if rp_variant is not None:
            try:
                rp_result = service.compute_range_profile(
                    frame,
                    sampling_frequency=fs,
                    fft_size=fft_r,
                    window="hann",
                    store_in_history=False,
                    variant=rp_variant,
                )
            except (KeyError, ValueError, RuntimeError) as exc:
                self._set_unavailable(str(exc), step=step)
                return

        rd_result = None
        if refresh_rd and rd_variant is not None:
            try:
                rd_result = service.compute_range_doppler(
                    frame=frame,
                    sampling_frequency=fs,
                    prf_hz=prf,
                    fft_size_range=fft_r,
                    fft_size_doppler=fft_d,
                    variant=rd_variant,
                )
            except (KeyError, ValueError, RuntimeError) as exc:
                self._set_unavailable(str(exc), step=step)
                return

        actual_variant = variant_preference
        if rd_result is not None:
            rd_meta = getattr(rd_result, "metadata", {}) or {}
            actual_variant = rd_meta.get("variant") or rd_variant or variant_preference
        elif rp_result is not None:
            rp_meta = getattr(rp_result, "metadata", {}) or {}
            actual_variant = rp_meta.get("variant") or rp_variant or variant_preference

        pretty = {
            "primary": "generator data (primary)",
            "raw": "generator data (raw)",
        }.get(actual_variant or "", "generator data")
        pretty += " | " + self._format_display_provenance(selection)
        if selection.replay_event is not None:
            pretty += (
                " | replay "
                f"{selection.replay_event.manifest_id[:8]} "
                f"({selection.replay_event.preflight.disposition})"
            )
        elif selection.replay_reason:
            pretty += f" | replay unavailable: {selection.replay_reason}"
        if selection.pinned and display_is_stale:
            state = "STALE" if selection.stale_age_s is not None else "MISMATCH"
            pretty += f" | PINNED {state} for current scene {step}"
        rd_source = sensing_block.get("rd_source")
        frame_role = sensing_block.get("frame_role")
        rd_fresh = sensing_block.get("rd_fresh")
        rp_fresh = sensing_block.get("rp_fresh")
        meta_parts: list[str] = []
        if rd_source:
            meta_parts.append(str(rd_source))
        if frame_role:
            meta_parts.append(str(frame_role))
        if rd_fresh is not None or rp_fresh is not None:
            meta_parts.append(f"fresh(rd={bool(rd_fresh)}, rp={bool(rp_fresh)})")
        if meta_parts:
            pretty += " | " + ", ".join(meta_parts)
        self._set_status(f"Sensing: {pretty}")

        display_step = (
            selection.bound_scene_frame_id
            if isinstance(selection.bound_scene_frame_id, int)
            and not isinstance(selection.bound_scene_frame_id, bool)
            else step
        )
        self._update_detection_label(
            sensing_block,
            state_step=step,
            display_step=display_step,
            variant=actual_variant,
            display_from_cache=using_cached_sensing or display_is_stale,
        )

        # Push to dashboard window
        if self._sensing_window is not None and self._dashboard_is_visible():
            detection_stage_bundle = service.load_detection_stage_bundle(selection)
            scientific_result = rd_result if rd_result is not None else rp_result
            scientific_metadata = (
                getattr(scientific_result, "metadata", {}) or {}
                if scientific_result is not None
                else {}
            )
            range_mode = scientific_metadata.get("range_mode")
            if range_mode not in {"monostatic", "bistatic"}:
                self._set_unavailable(
                    "Typed sensing product is missing exact range-axis coordinate semantics.",
                    step=step,
                )
                return
            carrier = runtime_cfg["carrier_freq_hz"]
            # Extract dense fast-time CIR arrays from service history for micro-Doppler
            cir_history_arrays: Optional[list[np.ndarray]] = None
            cir_history_times_s: Optional[list[float]] = None
            if service is not None and hasattr(service, "cir_history") and service.cir_history:
                timestamped_entries = [
                    entry for entry in service.cir_history if entry.sample_time_s is not None
                ]
                cir_history_arrays = [entry.cir for entry in timestamped_entries]
                cir_history_times_s = [float(entry.sample_time_s) for entry in timestamped_entries]
            request = self._build_update_request(
                display_frame=frame,
                state_frame=current_frame,
                state_step=step,
                display_step=display_step,
                display_from_cache=using_cached_sensing or display_is_stale,
                coherent_cir_time_steps=runtime_cfg["cir_time_steps"],
                rp_result=rp_result,
                rd_result=rd_result,
                variant=actual_variant,
                range_mode=range_mode,
                carrier_freq_hz=carrier,
                prf_hz=prf,
                sampling_frequency=fs,
                cir_history=cir_history_arrays,
                cir_history_times_s=cir_history_times_s,
                replay_event=selection.replay_event,
                detection_stage_bundle=detection_stage_bundle,
            )
            self._sensing_window.update_from_request(request)

        self._last_display_identity = display_identity

    @staticmethod
    def _build_update_request(
        *,
        display_frame: dict[str, Any],
        state_frame: dict[str, Any],
        state_step: Optional[int],
        display_step: Optional[int],
        display_from_cache: bool,
        coherent_cir_time_steps: Optional[int],
        rp_result: Any,
        rd_result: Any,
        variant: str,
        range_mode: str,
        carrier_freq_hz: float,
        prf_hz: float,
        sampling_frequency: float,
        cir_history: Optional[list[np.ndarray]],
        cir_history_times_s: Optional[list[float]],
        replay_event: Any = None,
        detection_stage_bundle: Any = None,
    ) -> "SensingUpdateRequest":
        """Build the typed dashboard update payload from display and state frames."""
        _validated_sensing_block(display_frame)
        return SensingUpdateRequest(
            display_frame=display_frame,
            state_frame=state_frame,
            state_step=state_step,
            display_step=display_step,
            display_from_cache=display_from_cache,
            coherent_cir_time_steps=coherent_cir_time_steps,
            rp_result=rp_result,
            rd_result=rd_result,
            variant=variant,
            range_mode=range_mode,
            carrier_freq_hz=carrier_freq_hz,
            prf_hz=prf_hz,
            sampling_frequency=sampling_frequency,
            cir_history=cir_history,
            cir_history_times_s=cir_history_times_s,
            replay_event=replay_event,
            detection_stage_bundle=detection_stage_bundle,
        )

    def _update_detection_label(
        self,
        sensing_block: dict[str, Any],
        *,
        state_step: Optional[int],
        display_step: Optional[int],
        variant: str,
        display_from_cache: bool,
    ) -> None:
        """Format the summary from the sensing block validated by the caller."""
        detections = sensing_block.get("detections") or []
        widget = self.widgets.get("detections_label")
        label = widget if isinstance(widget, QLabel) else None
        if label is None:
            return

        parts: list[str] = []
        if state_step is not None:
            parts.append(f"State {state_step}")
        if display_step is not None:
            display_label = f"Display {display_step}"
            if display_from_cache and state_step != display_step:
                display_label += " (cached)"
            parts.append(display_label)
        parts.append(f"Variant {variant}")
        parts.append(f"Detections {len(detections)}")

        rd_source = sensing_block.get("rd_source")
        if rd_source:
            parts.append(f"Source {rd_source}")

        rd_fresh = sensing_block.get("rd_fresh")
        rp_fresh = sensing_block.get("rp_fresh")
        if rd_fresh is not None or rp_fresh is not None:
            parts.append(f"Fresh rd={bool(rd_fresh)} rp={bool(rp_fresh)}")

        label.setText("Sensing: " + " | ".join(parts))

    def _current_step(self) -> Optional[int]:
        """Return current step from AppState, falling back to animation state."""
        app_state = getattr(self.parent, "app_state", None)
        if app_state and hasattr(app_state, "step"):
            return app_state.step
        return getattr(self.parent, "animation_step", None)

    def _on_show_detections_changed(self, state: int) -> None:
        """Toggle 3D detection overlay in the scene.

        ``force_update_next_frame + schedule_update()`` is needed because
        ``set_state()`` records the state change while the render pipeline
        handles redraw scheduling separately.
        """
        viz = self.parent
        if viz is None:
            return
        enabled = bool(state)
        if enabled:
            self.start_scene_preparation()
        update_state(viz, show_detections=enabled)
        self._update_scene_overlay_controls_enabled(enabled)
        self._sync_range_shell()
        # Keep the renderer refresh explicit until this control is fully state-driven.
        viz.force_update_next_frame = True
        viz.schedule_update()

    def _on_scene_highlight_changed(self, enabled: bool) -> None:
        from ..sensing.runtime import refresh_scene_selection

        update_state(self.parent, highlight_selection=enabled)
        refresh_scene_selection(self.parent, getattr(self.parent, "sensing_selection", None))

    def _show_recorded_paths(self, enabled: bool) -> None:
        """Toggle the actual MPC layer, including its historical solve context."""
        update_state(self.parent, show_rt_context=enabled)
        self.parent.ui_controller._set_mpc_visibility(
            reason="show_recorded_paths",
            **({"enabled": True, "paths": True} if enabled else {"paths": False}),
        )
        self.parent.force_update_next_frame = True
        self.parent.schedule_update()

    def _on_pin_freshest_changed(self, state: int) -> None:
        """Persist the explicit cross-scene result pin and refresh provenance."""
        viz = self.parent
        if viz is None:
            return
        update_state(viz, pin_freshest=bool(state))
        self._last_display_identity = None
        self._refresh_live_plots(force=True, refresh_rd=True)
        viz.force_update_next_frame = True
        viz.schedule_update()

    def _on_sensing_scene_mode_changed(self, _index: int) -> None:
        """Persist the selected 3D sensing overlay mode and schedule redraw."""
        viz = self.parent
        if viz is None:
            return
        widget = self.widgets.get("sensing_scene_mode")
        mode = widget.currentData() if isinstance(widget, QComboBox) else "detections"
        update_state(viz, scene_mode=str(mode or "detections"))
        self._update_scene_overlay_controls_enabled(
            bool(state_value(viz.app_state, "show_detections"))
        )
        viz.force_update_next_frame = True
        viz.schedule_update()

    def _on_sensing_scene_history_changed(self, value: int) -> None:
        """Persist finite sensing history length and schedule redraw."""
        viz = self.parent
        if viz is None:
            return
        update_state(viz, history_frames=max(1, int(value)))
        viz.force_update_next_frame = True
        viz.schedule_update()

    def _on_sensing_scene_full_history_changed(self, state: int) -> None:
        """Toggle all-frame sensing history and refresh dependent controls."""
        viz = self.parent
        if viz is None:
            return
        enabled = bool(state)
        update_state(viz, full_history=enabled)
        self._update_scene_overlay_controls_enabled(
            bool(state_value(viz.app_state, "show_detections"))
        )
        viz.force_update_next_frame = True
        viz.schedule_update()

    def _on_sensing_velocity_arrows_changed(self, state: int) -> None:
        """Toggle per-track velocity arrows in 3D sensing overlays."""
        viz = self.parent
        if viz is None:
            return
        update_state(viz, show_velocity_arrows=bool(state))
        viz.force_update_next_frame = True
        viz.schedule_update()

    def _on_sensing_track_labels_changed(self, state: int) -> None:
        """Toggle renderer-owned track labels for sensing overlays."""
        viz = self.parent
        if viz is None:
            return
        update_state(viz, show_track_labels=bool(state))
        viz.force_update_next_frame = True
        viz.schedule_update()

    def _on_evidence_focus_changed(self, enabled):
        from ..services.evidence_focus import EvidenceFocus, focus_identity, selected_focus_track
        from ..services.paused_presentation import frames_advancing

        viz = self.parent
        focus = getattr(viz, "_evidence_focus", None)
        if not enabled:
            if focus is not None:
                focus.stop()
            return
        if focus is None:
            focus = viz._evidence_focus = EvidenceFocus(viz, self.widgets["evidence_focus"])
        selection = getattr(viz, "sensing_selection", None)
        request = getattr(getattr(self, "_sensing_window", None), "_last_request", None)
        identity = focus_identity(viz)
        timestamp = None
        from ..io.packed_frame_payload import visualizer_frame_provider
        from ..sensing.dashboard_data import DashboardSession

        source = DashboardSession.provider_identity(visualizer_frame_provider(viz))
        if (
            request is not None
            and request.source_identity == source
            and request.effective_display_step == identity[1]
        ):
            timestamp = (
                request.display_frame.get("sensing", {})
                .get("temporal", {})
                .get("result", {})
                .get("reference_time_s")
            )
        focus.start(
            identity=identity,
            selected_track_id=selected_focus_track(
                selection, request, source, identity[1], getattr(self, "saved_track_scene", None)
            ),
            result_time_s=timestamp,
        )
        if frames_advancing(viz):
            focus.set_playing(True)

    def _on_uncertainty_changed(self):
        from ..sensing.runtime import refresh_tracking_uncertainty

        controls = self.uncertainty_controls
        active = controls.enabled.isChecked()
        update_state(
            self.parent,
            show_tracking_uncertainty=active,
            uncertainty_sigma=controls.scale.currentData(),
            uncertainty_selected_only=controls.scope.currentData(),
        )
        refresh_tracking_uncertainty(self.parent)
        if active:
            self.parent.force_update_next_frame = True
            self.parent.schedule_update()

    def _on_sensing_marker_glyphs_changed(self, state: int) -> None:
        """Toggle pygfx marker glyphs for sensing detections and tracks."""
        viz = self.parent
        if viz is None:
            return
        update_state(viz, use_marker_glyphs=bool(state))
        viz.force_update_next_frame = True
        viz.schedule_update()


def _select_variant(preference: str, available: frozenset[str]) -> Optional[str]:
    """Resolve one explicit descriptor variant without cross-variant fallback."""
    choice = str(preference or "auto").strip().lower()
    if choice == "clean":
        choice = "primary"
    if choice == "auto":
        if "primary" in available:
            return "primary"
        if "raw" in available:
            return "raw"
        return None
    if choice not in {"primary", "raw"}:
        raise ValueError(f"Unsupported typed sensing product variant {choice!r}")
    return choice if choice in available else None
