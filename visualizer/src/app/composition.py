"""Deferred UI and cache composition for the visualizer window.

``OrchavVisualizer`` shows a loading placeholder first, then calls
``finalize_deferred_setup`` after core state and services exist. This module
finishes UI wiring that depends on those services and creates timers/caches
that should not run until the Qt widgets are present.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import QTimer

from ..scene.orientation_frame_payloads import DEFAULT_ORIENTATION_FRAME_THICKNESS
from ..services.coverage_service import (
    DEFAULT_COVERAGE_INTERPOLATION,
    DEFAULT_COVERAGE_ISOLINE_COUNT,
    DEFAULT_COVERAGE_OPACITY,
)
from ..services.viewmodel_warmer import ViewModelWarmer
from .cache import ObservableCache


def finalize_deferred_setup(viz: Any) -> None:
    """Build panels, startup timers, view caches, and remaining UI state.

    This runs after ``construct_services`` and before any scenario is opened.
    It may connect widgets to services, but it must not create renderer scene
    content; renderer boot waits for scenario or manual scene loading.
    """
    viz._init_ui()
    viz._setup_keyboard_shortcuts()
    if getattr(viz, "ui_manager", None):
        viz.ui_manager.set_panel_visible("coverage", False)
    viz.beamforming_ui_controller.update_resolution_controls()

    viz.node_service.update_node_coloring_legend()

    viz.auto_update = True  # Auto-update visualizer when controls change

    viz.update_pending = False
    viz._session_restore_in_progress = False
    viz.update_timer_coalesce = QTimer()
    viz.update_timer_coalesce.timeout.connect(viz._flush_update)
    viz._startup_preload_timer = QTimer(viz)
    viz._startup_preload_timer.setSingleShot(True)
    viz._startup_preload_timer.timeout.connect(viz._run_startup_preload)

    # Coverage map data
    viz.coverage_data = None
    viz.coverage_opacity = DEFAULT_COVERAGE_OPACITY
    viz.coverage_heights = []  # Available coverage heights (float meters)
    viz.coverage_height_index = 0  # Currently selected height index
    viz.coverage_interpolation_method = DEFAULT_COVERAGE_INTERPOLATION
    viz.coverage_metric_name = None  # Active coverage metric layer
    viz.coverage_threshold_enabled = False
    viz.coverage_threshold_value = None
    viz.coverage_threshold_mask_enabled = False
    viz.coverage_isolines_enabled = False
    viz.coverage_isoline_count = DEFAULT_COVERAGE_ISOLINE_COUNT

    viz.orientation_axis_thickness = DEFAULT_ORIENTATION_FRAME_THICKNESS

    # Optional metrics window
    viz.metrics_window = None
    viz.update_timer_coalesce.setSingleShot(True)

    viz.mpc_view_cache: ObservableCache = ObservableCache()  # LRU cache for ViewModels

    # ViewModel pre-warming service (warms cache as frames finish preloading)
    viz._vm_warmer = ViewModelWarmer(viz)
    viz.mpc_view_cache.set_warmer(viz._vm_warmer)

    # Beamforming UI widgets are connected during _init_ui(); preserve them
    # here instead of resetting the Antennas panel back to an unbound state.
    viz.beam_azimuth_spin = getattr(viz, "beam_azimuth_spin", None)
    viz.beam_elevation_spin = getattr(viz, "beam_elevation_spin", None)
    viz.beam_tx_selector = getattr(viz, "beam_tx_selector", None)
    viz.beam_rx_selector = getattr(viz, "beam_rx_selector", None)
    viz._beamforming_tx_nodes: List[str] = []
    viz._beamforming_rx_nodes: List[str] = []
    viz._latest_beamforming_info: Optional[Dict[str, Any]] = None
    viz._latest_beamforming_pairs: List[Dict[str, Any]] = []
    viz._beamforming_computing = False
    viz._beamforming_completed_without_result = False
    viz._beamforming_error_message: Optional[str] = None
    viz._frame_beamforming_available = False

    # Do not boot visualizer immediately: scenario/manual scene loading owns
    # the first renderer scene and camera setup.
    # _init_ui() already replaced the loading placeholder via setCentralWidget().
    viz._loading_widget = None
    viz._loading_label = None
    viz._loading_progress = None
