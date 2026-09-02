"""Initial visualizer state bootstrap helpers.

``set_minimal_startup_flags`` installs attributes that may be read before the
loading placeholder is replaced. ``initialize_runtime_state`` then creates the
full set of runtime attributes, timers, and frame-source placeholders before
services and panels are constructed.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer

from shared.frames.loader import FrameLoaderService
from shared.logging import get_logger

from ..io.config_handlers import RecentFilesHandler
from ..pipeline.core import MPCCore
from ..playback import PlaybackCadenceTracker
from ..scene.defaults import (
    DEFAULT_LABEL_FONT_SIZE,
    DEFAULT_LABEL_OFFSET_M,
    DEFAULT_NODE_MARKER_SIZE_M,
    DEFAULT_ORIENTATION_SCALE_M,
    DEFAULT_SCENE_BACKGROUND_COLOR,
    DEFAULT_SCENE_BACKGROUND_PRESET,
)
from ..services.beamforming_service import BeamformingService
from ..services.target_asset_cache import TargetAssetCache
from ..state import create_initial_state

logger = get_logger(__name__)


def _create_animation_timer(callback: Callable[[], None]) -> QTimer:
    """Create the serial frame-playback timer with accurate deadline behavior.

    Each timeout owns exactly one synchronous frame transaction. The
    controller rearms this single-shot timer only after that transaction
    completes, which gives native renderer/window events a scheduling boundary
    even in Maximum mode. Precise timing keeps fixed-rate deadline rounding
    from accumulating drift on Windows.
    """
    timer = QTimer()
    timer.setTimerType(Qt.PreciseTimer)
    timer.setSingleShot(True)
    timer.timeout.connect(callback)
    return timer


def _create_slider_scrub_timer(callback: Callable[[], None]) -> QTimer:
    """Create the precise single-shot timer used to throttle timeline scrubbing."""
    timer = QTimer()
    timer.setTimerType(Qt.PreciseTimer)
    timer.setSingleShot(True)
    timer.timeout.connect(callback)
    return timer


def set_minimal_startup_flags(viz: Any) -> None:
    """Install attributes that may be read before deferred initialization finishes."""
    viz.ready = False
    viz._scene_only_mode = False
    viz.vis_initialized = False
    viz.vis = None
    viz._pending_camera: Optional[dict] = None
    viz._compact_mode = False
    viz._scenario_load_in_progress = False
    viz._scenario_load_generation = 0
    # The MPC Explorer is imported and constructed only on explicit open.
    # This committed epoch changes when an active source is retired, unlike
    # the load-attempt generation above which may advance on failed preflight.
    viz._mpc_explorer_session = None
    viz._mpc_presented_source_epoch = 0
    viz._frame_retry_token = 0
    viz._frame_retry_count = 0
    viz._frame_retry_pending = False


def initialize_runtime_state(viz: Any, *, default_animation_cache_size: int) -> None:
    """Initialize runtime state, timers, caches, and frame-source placeholders."""
    viz.xml_root = None
    viz.xml_path = None
    viz.mesh_entries = []
    viz.target_entries = []
    viz.tx_entries: List[Dict[str, Any]] = []
    viz.rx_entries: List[Dict[str, Any]] = []
    viz.target_meshes = {}  # Current target meshes by name.
    viz.last_cam = None

    viz.recent_files = []
    viz.max_recent_files = 3

    # Config file for persistent storage
    viz.config_file = os.path.join(os.path.expanduser("~"), ".orchav_config.json")
    logger.debug("Recent files will be stored in: %s", viz.config_file)

    viz.recent_files = RecentFilesHandler.load_recent_files(viz.config_file, viz.max_recent_files)

    # Animation state
    viz.animation_timer = _create_animation_timer(viz.update_animation)
    viz.animation_step = 0
    viz.total_animation_steps = 60
    viz.total_steps_label = None
    viz.animation_running = False
    viz._frame_duration = 1.0
    viz._mesh_update_interval_s = None
    viz.stride_combo = None
    viz.play_direction = 1
    viz.play_btn = None
    viz.reverse_play_btn = None
    viz.original_target_vertices = None
    viz.vis = None
    viz.vis_initialized = False
    viz.update_timer = QTimer()
    viz.update_timer.timeout.connect(viz.update_visualizer)
    viz._idle_poll_interval_ms = 16
    viz._active_poll_interval_ms = 16  # High frequency polling while animating
    viz.update_timer.start(viz._idle_poll_interval_ms)

    # Single-shot throttle for timeline scrubbing. While active, slider changes
    # replace the pending display index without restarting the timer.
    viz.slider_scrub_timer = _create_slider_scrub_timer(viz.flush_pending_slider_scrub)
    viz._pending_slider_scrub_index = None

    viz.last_forward_position = 0.0
    viz.frame_times = []  # For benchmarking
    viz.playback_cadence = PlaybackCadenceTracker()
    viz.last_frame_duration_ms = None
    viz.scene_boot_duration_ms = None
    viz._scene_boot_start = None
    viz._scene_boot_logged = False
    viz._cli_driven_frame_run = False
    viz.startup_stage_timings_ms: OrderedDict[str, float] = OrderedDict()
    viz.startup_first_frame_timings_ms: dict[str, float] = {}
    viz.startup_detail_timings_ms: OrderedDict[str, dict[str, float]] = OrderedDict()
    viz._startup_preload_requested = False
    viz._startup_preload_delay_ms = 1000
    viz._last_avg_frame_ms = None
    viz._last_status_message = ""

    # One typed owner keeps source identity, geometry baselines, transform
    # metadata, and bounded animation-frame residency together.
    viz.target_asset_cache = TargetAssetCache()
    viz.target_scale_overrides: Dict[str, float] = {}
    viz.targets_metadata = []  # Current frame's target metadata from frame data
    viz.num_targets = 0  # Number of targets in current frame

    viz.last_update_time = 0
    viz.update_threshold = 0  # Only update every 16ms minimum
    viz.force_update_next_frame = False  # Force update on next visualizer cycle

    # Scene rendering controls
    viz.outlines_enabled = False
    viz.target_outlines_enabled = False
    viz.outline_color = [0.05, 0.05, 0.05]
    viz.current_background_color = list(DEFAULT_SCENE_BACKGROUND_COLOR)
    viz.current_background_preset = DEFAULT_SCENE_BACKGROUND_PRESET

    # Transparency state (persists across frame changes)
    viz.current_building_alpha = 1.0  # Default fully opaque
    viz.current_target_alpha = 1.0  # Default fully opaque

    # Renderer-owned MPC handles retained for compatibility with older callers.
    viz.mpc_pcd = None  # Will be accessed via viz.renderer.mpc_pcd
    viz.mpc_lineset = None  # Will be accessed via viz.renderer.mpc_lineset
    viz.mpc_loaded_step = None
    # Material filtering behavior
    viz.show_tx_segments = True  # Show segments starting from TX (no-material)
    viz.mpc_allowed_materials = None  # None means no active material allow-list
    viz.mpc_material_filter_scope = "segment"  # "segment" or "path"
    viz._mpc_material_filter_choices: set[str] = set()
    viz._last_material_keys = None  # Track last populated material keys for UI

    # Raw frame cache lives in CacheService; these fields track current app mode.
    viz.current_scenario_path = None
    viz.scenario_config = None  # Active ScenarioConfiguration (set after loading)
    viz._live_preview_enabled = False
    viz._live_preview_frame = None
    viz._live_preview_step = None
    viz._live_preview_sequence = None
    viz._live_preview_status = "Preview off"
    viz.current_target_positions = None
    viz.current_target_orientations = None

    # MPC Core helper for frame data management
    viz.mpc_core = MPCCore(logger, viz)  # Pass visualizer reference for color methods

    # Beamforming service (antenna pattern visualization)
    viz.beamforming_service = BeamformingService(viz)
    viz.mpc_core.beamforming_service = viz.beamforming_service

    # Frame source is intentionally absent until scenario loading selects one.
    viz.frame_source = None  # No default frame source, scenario will set it
    viz.frame_loader: Optional[FrameLoaderService] = None
    viz._frame_loader_cache_size = default_animation_cache_size
    viz.ready = False  # Guard to prevent updates before FrameSource is ready
    logger.debug("No default frame source - will be set by scenario loading")

    # Performance optimization: Track last app state to enable early returns
    viz.last_app_state = None  # Will be set after first update

    # Scenario loading owns file, live gRPC, and remote-HDF5 provider selection.
    logger.debug("Data provider switching handled by scenario loading")

    # Selection and camera code consume these established list attributes; the
    # contents are renderer-neutral marker handles synced as VisualEntity objects.
    viz.tx_markers = []  # List of TX marker handles
    viz.rx_markers = []  # List of RX marker handles
    viz.tx_labels = []  # List of TX text labels
    viz.rx_labels = []  # List of RX text labels
    # Let frame data determine actual node counts.
    viz.num_tx = None  # Will be set from frame data
    viz.num_rx = None  # Will be set from frame data

    # TX/RX Selection: Dropdown controls for selecting specific TX/RX nodes
    viz.tx_dropdown = None  # TX selection dropdown
    viz.rx_dropdown = None  # RX selection dropdown
    viz.available_tx = []  # List of available TX indices
    viz.available_rx = []  # List of available RX indices
    viz.tx_rx_data_loaded = False  # Flag to track if TX/RX data has been discovered
    viz.show_building_labels = False  # Toggle for showing building labels (disabled by default)

    # Position tracking for markers to prevent loss during visibility changes
    viz.current_tx_positions = []  # Current TX positions from latest data
    viz.current_rx_positions = []  # Current RX positions from latest data

    # TX/RX/Target Orientation visualization
    viz.tx_orientation_frames = []  # List of TX orientation coordinate frames
    viz.rx_orientation_frames = []  # List of RX orientation coordinate frames
    viz.target_orientation_frames = []  # List of Target orientation coordinate frames
    viz.show_tx_orientation = False  # Toggle for TX orientation display
    viz.show_rx_orientation = False
    viz.show_target_orientation = False
    viz.positions_loaded_step = None

    # Building labels
    viz.building_labels = []
    viz.target_labels = []

    # Object selection and highlighting
    viz.selected_objects = set()

    # Preloading trades startup work for fast TX/RX switching.
    viz.use_preload_mode = True

    viz.node_coloring_mode = "per_type"
    viz.individual_node_colors = []

    # Color-mode widgets are populated by the MPC panel; the selected mode is
    # owned exclusively by AppState.
    viz.color_mode_group = None
    viz.reflection_order_rb = None
    viz.mpc_type_rb = None
    viz.delay_rb = None
    viz.path_loss_rb = None
    viz.material_rb = None
    viz.reconstruction_type_rb = None
    viz.color_legend_label = None
    viz.color_legend_widget = None
    viz.color_legend_layout = None
    viz.colorbar_widget = None

    # Shared node-appearance defaults. Workspace restore may replace these
    # values after a scenario is hydrated.
    (
        viz.label_offset_x,
        viz.label_offset_y,
        viz.label_offset_z,
    ) = DEFAULT_LABEL_OFFSET_M
    viz.label_font_size = DEFAULT_LABEL_FONT_SIZE
    viz.orientation_scale = DEFAULT_ORIENTATION_SCALE_M

    # Marker size controls for TX/RX visualization. The marker payload may be
    # a sphere, box, or custom mesh, while coloring remains node-color driven.
    viz.tx_marker_size = DEFAULT_NODE_MARKER_SIZE_M
    viz.rx_marker_size = DEFAULT_NODE_MARKER_SIZE_M
    viz.node_marker_config = {
        "default": {
            "shape": "sphere",
            # For shape="mesh", set mesh_path to an OBJ/PLY file. The mesh is
            # centered at the node anchor by default and recolored by the
            # normal node-coloring strategy.
            "center": True,
        }
    }

    # NodeService creates markers once frame data arrives.

    # App state must exist before controls perform initial synchronization.
    viz.app_state = create_initial_state()
