"""Embeddable sensing dashboard for frame-derived radar/ISAC products.

``SensingDashboard`` consumes :class:`SensingUpdateRequest` objects from the
visualizer panel layer and renders cached/shared DSP products without owning
the signal-processing algorithms. Ground-truth and detection histories use
persisted result/truth time; scene IDs are never treated as clocks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import (
    QEvent,
    QRectF,
    QSettings,
    QSignalBlocker,
    Qt,
    Signal,
)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from shared.logging import get_logger
from shared.sensing.evaluation_truth import extract_evaluation_truth
from shared.sensing.products import TEMPORAL_SUPPORT_LAST_VALID_SAMPLE
from shared.sensing.signal_processing import (
    DetectionVelocityDisplay,
    MicroDopplerResult,
    auto_select_range_bin,
    compute_micro_doppler,
    compute_range_collapsed_micro_doppler,
    compute_range_tracked_micro_doppler,
    detection_velocity_display,
    doppler_axis_to_velocity_m_s,
)

from ..app.plot_theme import apply_pyqtgraph_plot_theme
from ..app.theme import get_theme_manager
from ..panels.collapsible_section import CollapsibleSection
from ..widgets.dashboard_host import dashboard_is_visible
from ..widgets.dashboard_plot import DashboardPlot
from ..widgets.plot_lifecycle import ManagedPlotWidget
from ..widgets.wrapping_layout import WrappingLayout
from .cir_provider import extract_timestamped_fast_time_samples
from .dashboard_diagnostics import (
    STAGE_HELP,
    cell_explanation,
    stage_counts,
    stage_image,
    stage_unavailable_text,
)
from .detector_map import DetectorMap
from .display_style import HEATMAP_MARKER_OUTLINE, detection_color, identity_color
from .evaluation_panel import EvaluationPanel
from .evaluation_settings import EvaluationSettings
from .inspection_display import SIGNAL_COMPANIONS, PowerScale, range_label, regular_axis
from .inspection_figure import InspectionFigurePanel, tracking_decision_text
from .linked_selection import LinkedSelectionMixin
from .live_figures import LiveFiguresMixin
from .prepared_angle import PreparedAngle, prepare_angular
from .processing_panel import ProcessingPanel
from .replay_export import export_replay_session
from .result_browser import DetectionTableModel as _DetectionTableModel  # noqa: F401
from .result_browser import SensingResultBrowser
from .result_navigation import nearest_result
from .run_summary_panel import RunSummaryPanel
from .spectrum_surface import SpectrumSurface
from .stage_inspector import (
    DETECTION_STAGE_KINDS,
    CellInspection,
    CellSelection,
    DetectionStageBundle,
    InspectionDisposition,
)
from .tracking_environment import TrackingEnvironment
from .tracking_inspector import TrackingInspector
from .tracking_views import TrackingViewsMixin
from .update_request import SensingUpdateRequest
from .workspace_shell import WorkspaceShellMixin

logger = get_logger("orchav.sensing.window")

# ── Colour palette ──────────────────────────────────────────────────────────
_COLORS = {
    "rp_line": "#1f77b4",
    "cfar_thresh": "#ff5722",
    "det_marker": "#ff5722",
    "gt_marker": "#4caf50",
}

_DET_PALETTE = ["#ff5722", "#e91e63", "#9c27b0", "#673ab7", "#3f51b5", "#00bcd4"]

# Minimum CIR history frames required for micro-Doppler STFT
_MIN_MICRO_DOPPLER_FRAMES = 32
_ORACLE_MICRO_LABEL = "Oracle-assisted (GT tracked)"
_MOTION_VIEW_TOOLTIPS = {
    "auto": (
        "Show motion over the whole recording by summing saved range–Doppler power "
        "across all ranges. This does not compute a new STFT or rerun detection or tracking."
    ),
    "saved-stft": (
        "Show the recorded micro-Doppler STFT over the recording, using its saved "
        "window and range/time selection. This does not rerun detection or tracking."
    ),
    "tracked": (
        "Inspect the current result's saved range–Doppler power around the ground-truth "
        "range. This oracle view uses known truth, not a tracked estimate, "
        "and does not rerun detection or tracking."
    ),
    "gated": (
        "Inspect the current result's saved range–Doppler power near a detected range. "
        "Gate half-width sets the neighboring bins included. "
        "This does not rerun detection or tracking."
    ),
    "fixed-bin": (
        "Inspect one detected range bin in the current result's saved range–Doppler map. "
        "This does not rerun detection or tracking."
    ),
}

# Maximum range-time waterfall columns to retain
_MAX_RANGE_TIME_HISTORY = 96

# View limits never clip the stored/plotted samples. Exact zeros have no finite
# dB value, so their display baseline remains below every positive sample.
_RANGE_PROFILE_DISPLAY_SPAN_DB = 80.0
_RANGE_PROFILE_HEADROOM_DB = 3.0
_RANGE_PROFILE_FLOOR_MARGIN_DB = 6.0
_RANGE_PROFILE_MARKER_SPACING_PX = 10.0
_TRACK_HISTORY_MARKER_PX = 8
_TRACK_CURRENT_MARKER_PX = 18

_DETECTION_STAGE_LABELS = {
    "range_doppler_detection_input_power": "Detection input power",
    "range_doppler_cfar_noise_power": "CFAR noise power",
    "range_doppler_cfar_threshold_power": "CFAR threshold power",
    "range_doppler_cfar_training_cell_count": "CFAR training count",
    "range_doppler_cfar_eligible_mask": "CFAR eligibility",
    "range_doppler_cfar_mask": "CFAR decision",
    "range_doppler_nms_mask": "NMS decision",
    "range_doppler_snr_db": "Local signal/background ratio (SNR)",
    "range_doppler_snr_valid_mask": "SNR validity",
    "range_doppler_cluster_labels": "Cluster labels",
}

_DETECTION_STAGE_CELL_FIELDS = {
    "range_doppler_detection_input_power": ("detection_input_power", "1"),
    "range_doppler_cfar_noise_power": ("cfar_noise_power", "1"),
    "range_doppler_cfar_threshold_power": ("cfar_threshold_power", "1"),
    "range_doppler_cfar_training_cell_count": ("cfar_training_cell_count", "1"),
    "range_doppler_cfar_eligible_mask": ("cfar_eligible", "1"),
    "range_doppler_cfar_mask": ("cfar_passed", "1"),
    "range_doppler_nms_mask": ("nms_retained", "1"),
    "range_doppler_snr_db": ("snr_db", "dB"),
    "range_doppler_snr_valid_mask": ("snr_valid", "1"),
    "range_doppler_cluster_labels": ("cluster_id", "1"),
}


def _mpl_cmap_to_lut(name: str, n: int = 256) -> np.ndarray:
    """Convert a matplotlib colormap name to a pyqtgraph-compatible LUT (Nx4 uint8)."""
    try:
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap(name)
    except (ImportError, AttributeError, ValueError):
        # Fallback: simple grayscale
        g = np.linspace(0, 255, n, dtype=np.uint8)
        return np.column_stack([g, g, g, np.full(n, 255, dtype=np.uint8)])
    vals = cmap(np.linspace(0, 1, n))
    return (vals * 255).astype(np.uint8)


def _build_cfar_mask_lut() -> np.ndarray:
    """Build a 2-entry LUT for the binary CFAR mask overlay.

    Index 0 (no detection) = dark semi-transparent grey
    Index 1 (detection)    = bright green
    """
    lut = np.zeros((2, 4), dtype=np.uint8)
    lut[0] = [30, 30, 30, 140]  # dark overlay where NOT detected
    lut[1] = [0, 230, 64, 180]  # bright green where detected
    return lut


class _DashboardContent(
    LiveFiguresMixin, WorkspaceShellMixin, TrackingViewsMixin, LinkedSelectionMixin
):
    """Shared plots and behavior for the panel widget and standalone analysis windows."""

    # ── Mode layout specs ────────────────────────────────────────────
    # Each entry: (widget_attr, row, col, rowspan, colspan, row_stretch)
    _MODE_LAYOUTS: dict[str, list[tuple[str, int, int, int, int, int]]] = {
        "analysis": [
            ("rd_plot", 0, 0, 1, 1, 3),
            ("rp_plot", 0, 1, 1, 1, 3),
            ("gt_range_plot", 1, 0, 1, 1, 2),
            ("gt_speed_plot", 1, 1, 1, 1, 2),
        ],
        "micro-doppler": [
            ("collapsed_doppler_plot", 0, 0, 1, 2, 4),
            ("range_time_plot", 1, 0, 1, 2, 2),
        ],
        "tracking": [
            ("tracking_plot", 0, 0, 1, 2, 3),
            ("gt_range_plot", 1, 0, 1, 1, 2),
            ("gt_speed_plot", 1, 1, 1, 1, 2),
        ],
        "diagnostics": [
            ("detection_stage_inspector", 0, 0, 1, 2, 0),
            ("cfar_rp_plot", 1, 0, 1, 2, 3),
            ("stage_cfar_plot", 2, 0, 1, 2, 3),
            ("cfar_rd_plot", 3, 0, 1, 1, 3),
            ("stage_threshold_plot", 3, 1, 1, 1, 3),
            ("stage_noise_plot", 4, 0, 1, 1, 3),
            ("stage_snr_plot", 4, 1, 1, 1, 3),
            ("stage_nms_plot", 5, 0, 1, 1, 3),
            ("stage_cluster_plot", 5, 1, 1, 1, 3),
        ],
        "evaluation": [("evaluation_panel", 0, 0, 1, 2, 1)],
        "summary": [("run_summary_panel", 0, 0, 1, 2, 1)],
        "angular": [
            ("rd_plot", 0, 0, 1, 1, 3),
            ("range_azimuth_plot", 0, 1, 1, 1, 3),
            ("doppler_azimuth_plot", 1, 0, 1, 1, 3),
            ("angular_spectrum_plot", 1, 1, 1, 1, 3),
        ],
    }

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Build the sensing dashboard and timestamped product-history state."""
        super().__init__(parent)
        self.setWindowTitle("ORCHAV Sensing Dashboard")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self._hosted = False
        self._live_figures = {}
        self._live_catalog_requested = False
        self._live_catalog_owner = False
        self._live_catalog = ()
        self._live_product_kinds = frozenset()
        self._compact = False
        self._initial_view_selected = False
        self._support_choices: dict[str, str] = {}
        self._support_mode: str | None = None
        self._reconstruct_spectrum = False
        self._recording_render_key = None
        self._locked_levels: dict[str, tuple[float, float]] = {}
        self._tracking_curves: dict[str, Any] = {}
        self._tracking_truth_curves: dict[str, Any] = {}
        self._tracking_coast_curves: dict[str, Any] = {}
        self._track_estimate_curves: dict[tuple, Any] = {}
        self._track_estimate_coasts: dict[tuple, Any] = {}

        # ── LUTs ────────────────────────────────────────────────────────
        self._rd_cmap = pg.colormap.get("viridis")
        self._rd_lut = self._rd_cmap.getLookupTable(nPts=256, alpha=True)
        self._micro_lut = _mpl_cmap_to_lut("inferno")
        self._cfar_mask_lut = _build_cfar_mask_lut()

        # ── State ───────────────────────────────────────────────────────
        self._target_history: Dict[str, Dict[str, List[Any]]] = {}
        self._history_from_recording = False
        self._target_names: Dict[str, str] = {}
        self._detection_history: Dict[str, List[float]] = {
            "times_s": [],
            "range": [],
            "speed": [],
        }
        self._detection_history_colors: list[str] = []
        self._history_marker_styles: dict[tuple[str, bool | None], tuple[Any, Any]] = {}
        self._recorded_marker_styles = None
        self._recorded_scatter_state = None
        self._detection_history_styles = None
        self._history_bounds_cache = None
        self._gt_range_curves: Dict[str, pg.PlotDataItem] = {}
        self._gt_speed_curves: Dict[str, pg.PlotDataItem] = {}
        self._last_history_reference_time_s: Optional[float] = None
        self._last_history_result_identity: Optional[tuple[str, str]] = None
        self._micro_doppler_range_bin: Optional[int] = None
        self._last_request: Optional[SensingUpdateRequest] = None
        self._rendered_request: Optional[SensingUpdateRequest] = None
        self._view_mode: str = "summary"
        self._micro_status: str = "idle"
        self._micro_mode_used: str = "auto"
        self._angular_status: str = "unavailable"
        self._angular_shape: tuple[int, int, int, int] | None = None
        self._micro_range_gate_half_width: int = 2
        self._micro_result_cache: Dict[tuple[Any, ...], MicroDopplerResult] = {}
        self._range_time_records: Dict[tuple[str, str, str, str, float], np.ndarray] = {}
        self._range_time_event_keys: List[tuple[str, str, str, str, float]] = []
        self._range_time_reference_times_s: List[float] = []
        self._range_time_axis_m: Optional[np.ndarray] = None
        self._range_time_axis_label: str = "Path Length (m)"
        self._range_time_status: str = "waiting for schema-v4 temporal data"
        self._acquisition_timeline_status: str = "waiting for schema-v4 temporal data"
        self._detection_stage_status: str = "unavailable: no canonical replay request"
        self._detection_stage_event_identity: tuple[str, str, str, str, str, float] | None = None
        self._cached_range_mode: Optional[str] = None

        # Complete multi-sample acquisitions are accumulated only with exact
        # physical-time alignment, including overlap between adjacent windows.
        self._expanded_cir_history: List[np.ndarray] = []
        self._expanded_cir_times_s: List[float] = []
        self._expanded_range_axis_m: Optional[np.ndarray] = None
        self._sequential_range_axis_m: Optional[np.ndarray] = None
        self._fast_time_range_axis_m: Optional[np.ndarray] = None
        self._last_expanded_acquisition_id: Optional[str] = None
        self._expanded_history_error: Optional[str] = None

        # ── Build UI ────────────────────────────────────────────────────
        if isinstance(self, QMainWindow):
            content = QWidget(self)
            self.setCentralWidget(content)
        else:
            content = self
        root = QVBoxLayout(content)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        title = QLabel("Sensing Analysis")
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        header = QHBoxLayout()
        header.addWidget(title, 1)
        self.placement_button = QPushButton("Pop out")
        self.placement_button.setToolTip(
            "Move this analysis between the panel and a separate window."
        )
        self.placement_button.hide()
        header.addWidget(self.placement_button)
        root.addLayout(header)

        self._grid = QGridLayout()
        self._grid.setSpacing(8)

        # ── Create owned plots; layouts can reparent them as views change. ──

        # Range-Velocity Map
        self.rd_plot = self._make_plot("Range\u2013Velocity")
        self.rd_plot.add_external_legend(labelTextSize="8pt", verSpacing=0, horSpacing=5)
        self.rd_image = pg.ImageItem()
        # One display buffer per persistent heatmap; never a second result cache.
        self._heatmap_buffers = {}
        self._rd_image_geometry = None
        self.rd_image.setLookupTable(self._rd_lut)
        self.rd_plot.addItem(self.rd_image)
        self.rd_cfar_overlay = pg.ImageItem()
        self.rd_cfar_overlay.setZValue(9)
        self.rd_plot.addItem(self.rd_cfar_overlay)
        self.rd_det_scatter = pg.ScatterPlotItem(
            size=12,
            symbol="x",
            brush=pg.mkBrush(_COLORS["det_marker"]),
            pen=pg.mkPen(HEATMAP_MARKER_OUTLINE, width=1.2),
            name="Detection",
        )
        self.rd_plot.addItem(self.rd_det_scatter)
        self.rd_det_scatter.setZValue(20)
        self.rd_gt_scatter = pg.ScatterPlotItem(
            size=14,
            symbol="+",
            brush=pg.mkBrush(_COLORS["gt_marker"]),
            pen=pg.mkPen(HEATMAP_MARKER_OUTLINE, width=1.2),
            name="Ground truth",
        )
        self.rd_plot.addItem(self.rd_gt_scatter)
        self.rd_gt_scatter.setZValue(19)
        # Ground truth is a reference overlay; clicks must reach the saved bin beneath it.
        self.rd_gt_scatter.setEnabled(False)
        self.rd_plot.setLabel("bottom", "Path Length (m)")
        self.rd_plot.setLabel("left", "Monostatic-equivalent Velocity (m/s)")

        self.rd_colorbar = pg.ColorBarItem(
            colorMap=self._rd_cmap,
            label="Power (dB)",
            interactive=False,
            width=15,
        )
        self.rd_colorbar.setImageItem(self.rd_image, insert_in=self.rd_plot.getPlotItem())

        # RD CFAR Mask
        self.cfar_rd_plot = self._make_plot("RD CFAR Mask")
        self.cfar_rd_image = pg.ImageItem()
        self.cfar_rd_image.setLookupTable(self._rd_lut)
        self.cfar_rd_plot.addItem(self.cfar_rd_image)
        self.cfar_rd_mask = pg.ImageItem()
        self.cfar_rd_mask.setZValue(10)
        self.cfar_rd_plot.addItem(self.cfar_rd_mask)
        self.cfar_pass_borders = pg.PlotCurveItem(pen=pg.mkPen("#f44336", width=1.5))
        self.cfar_pass_borders.setZValue(12)
        self.cfar_rd_plot.addItem(self.cfar_pass_borders, ignoreBounds=True)
        self.cfar_rd_plot.setLabel("bottom", "Path Length (m)")
        self.cfar_rd_plot.setLabel("left", "Monostatic-equivalent Velocity (m/s)")
        self.stage_colorbar = pg.ColorBarItem(
            colorMap=self._rd_cmap, label="Power (dB)", interactive=False, width=12
        )
        self.stage_colorbar.setImageItem(
            self.cfar_rd_image, insert_in=self.cfar_rd_plot.getPlotItem()
        )
        self.stage_slice_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#eab83c", width=2))
        self.stage_cell = pg.ScatterPlotItem(
            symbol="s", size=14, pen=pg.mkPen("#eab83c", width=2), brush=pg.mkBrush(None)
        )
        self.cfar_rd_plot.addItem(self.stage_slice_line)
        self.cfar_rd_plot.addItem(self.stage_cell)
        self.stage_slice_line.setZValue(18)
        self.stage_cell.setZValue(20)
        self.stage_bin_grid = pg.PlotCurveItem(pen=pg.mkPen((170, 180, 194, 140), width=1))
        self.stage_bin_grid.setZValue(11)
        self.cfar_rd_plot.addItem(self.stage_bin_grid, ignoreBounds=True)
        self.stage_bin_grid.hide()
        self.cfar_rd_plot.scene().sigMouseClicked.connect(self._select_stage_map_cell)

        self._stage_maps = {}
        for name, kind in (
            ("threshold", "range_doppler_cfar_threshold_power"),
            ("noise", "range_doppler_cfar_noise_power"),
            ("snr", "range_doppler_snr_db"),
            ("cfar", "range_doppler_cfar_mask"),
            ("nms", "range_doppler_nms_mask"),
            ("cluster", "range_doppler_cluster_labels"),
        ):
            plot = DetectorMap(
                _DETECTION_STAGE_LABELS[kind],
                self._rd_cmap,
                categorical=kind == "range_doppler_cluster_labels",
            )
            plot.setParent(self)
            plot.hide()
            plot.setToolTip(STAGE_HELP[kind])
            plot.scene().sigMouseClicked.connect(
                lambda event, p=plot, k=kind: self._select_stage_map_cell(event, p, k)
            )
            setattr(self, f"stage_{name}_plot", plot)
            self._stage_maps[kind] = plot

        # Angular products
        self.range_azimuth_plot = self._make_plot("Range-Azimuth")
        self.range_azimuth_plot.add_external_legend(labelTextSize="8pt")
        self.range_azimuth_image = pg.PColorMeshItem(
            colorMap=self._rd_cmap,
            enableAutoLevels=False,
        )
        self.range_azimuth_image.setVisible(False)
        self.range_azimuth_plot.addItem(self.range_azimuth_image)
        self.range_azimuth_det_scatter = pg.ScatterPlotItem(
            size=10,
            symbol="x",
            brush=pg.mkBrush(_COLORS["det_marker"]),
            pen=pg.mkPen(_COLORS["det_marker"], width=2),
            name="Detection",
        )
        self.range_azimuth_plot.addItem(self.range_azimuth_det_scatter)
        self.range_azimuth_det_scatter.setZValue(20)
        self.range_azimuth_plot.setLabel("bottom", "Path Length (m)")
        self.range_azimuth_plot.setLabel("left", "Azimuth (deg)")

        self.doppler_azimuth_plot = self._make_plot("Doppler-Azimuth")
        self.doppler_azimuth_plot.add_external_legend(labelTextSize="8pt")
        self.doppler_azimuth_image = pg.PColorMeshItem(
            colorMap=self._rd_cmap,
            enableAutoLevels=False,
        )
        self.doppler_azimuth_image.setVisible(False)
        self.doppler_azimuth_plot.addItem(self.doppler_azimuth_image)
        self.doppler_azimuth_det_scatter = pg.ScatterPlotItem(
            size=10,
            symbol="x",
            brush=pg.mkBrush(_COLORS["det_marker"]),
            pen=pg.mkPen(_COLORS["det_marker"], width=2),
            name="Detection",
        )
        self.doppler_azimuth_plot.addItem(self.doppler_azimuth_det_scatter)
        self.doppler_azimuth_det_scatter.setZValue(20)
        self.doppler_azimuth_plot.setLabel("bottom", "Monostatic-equivalent Velocity (m/s)")
        self.doppler_azimuth_plot.setLabel("left", "Azimuth (deg)")
        self.angular_spectrum_plot = self._make_plot("Angular spectrum")
        self.angular_spectrum_plot.add_external_legend(labelTextSize="8pt")
        self.angular_spectrum_image = pg.PColorMeshItem(
            colorMap=self._rd_cmap, enableAutoLevels=False
        )
        self.angular_spectrum_image.hide()
        self.angular_spectrum_plot.addItem(self.angular_spectrum_image)
        self.angular_spectrum_det_scatter = pg.ScatterPlotItem(
            size=12,
            symbol="x",
            name="Detection",
            brush=pg.mkBrush(_COLORS["det_marker"]),
            pen=pg.mkPen(HEATMAP_MARKER_OUTLINE, width=1.2),
        )
        self.angular_spectrum_det_scatter.setZValue(20)
        self.angular_spectrum_plot.addItem(self.angular_spectrum_det_scatter)
        self.angular_spectrum_plot.setLabel("bottom", "Azimuth (deg)")
        self.angular_spectrum_plot.setLabel("left", "Zenith (deg)")
        self.angular_spectrum_plot.getViewBox().invertY(True)
        self.angular_spectrum_plot.setToolTip(
            "RX-array-local azimuth and zenith (90° minus elevation). Color is the maximum "
            "saved angular power over range and Doppler. Cell shapes follow the recorded "
            "direction-cosine grid; bins outside the forward hemisphere are masked."
        )
        self._angular_colorbars = {}
        for angular_plot in (
            self.range_azimuth_plot,
            self.doppler_azimuth_plot,
            self.angular_spectrum_plot,
        ):
            bar = pg.ColorBarItem(
                colorMap=self._rd_cmap, label="Power (dB)", interactive=False, width=12
            )
            bar.setImageItem([], insert_in=angular_plot.getPlotItem())
            self._angular_colorbars[angular_plot] = bar

        # Range Profile
        self._profile_full_dynamic_range_cb = QCheckBox("Full dynamic range")
        self._profile_full_dynamic_range_cb.setToolTip(
            "Range profile only: reveal the full power span within the visible range. "
            "By default, show 80 dB below the profile peak. Only view limits change; "
            "saved samples and the range–Doppler color scale are unchanged."
        )
        self._profile_full_dynamic_range_cb.toggled.connect(self._fit_range_profile_axis)
        self._profile_detections_cb = QCheckBox("Detections on range profile")
        self._profile_detections_cb.setToolTip(
            "Mark saved detection ranges with dashed lines. Detections at the same range "
            "can have different velocities and share a line in this projection."
        )
        self._profile_detections_cb.toggled.connect(self._update_profile_detections)
        self._profile_detection_lines: list[pg.InfiniteLine] = []
        self.rp_plot = self._make_plot("Range Profile")
        self.rp_curve = self.rp_plot.plot(pen=pg.mkPen(_COLORS["rp_line"], width=1.0))
        self.rp_samples = self.rp_plot.plot(
            pen=None, symbol="o", symbolSize=4, symbolBrush=_COLORS["rp_line"], symbolPen=None
        )
        self.rp_plot.setLabel("bottom", "Path Length (m)")
        self.rp_plot.setLabel("left", "Power (dB)")
        self.rp_plot.setXLink(self.rd_plot)
        self.rp_plot.getViewBox().sigXRangeChanged.connect(self._fit_range_profile_axis)
        self.rp_plot.getViewBox().sigResized.connect(self._fit_range_profile_axis)

        # Exact detector input and threshold at the selected Doppler row.
        self.cfar_rp_plot = self._make_plot("Recorded detector threshold slice")
        self.cfar_rp_plot.setMinimumHeight(150)
        self.cfar_rp_plot.add_external_legend(labelTextSize="8pt")
        self.cfar_rp_power = self.cfar_rp_plot.plot(
            pen=pg.mkPen(_COLORS["rp_line"], width=1.5),
            name="Detector input",
        )
        self.cfar_rp_thresh = self.cfar_rp_plot.plot(
            pen=pg.mkPen(_COLORS["cfar_thresh"], width=1.5, style=Qt.DashLine),
            name="CFAR Threshold",
        )
        self.cfar_rp_plot.setLabel("bottom", "Path Length (m)")
        self.cfar_rp_plot.setLabel("left", "Power (dB)")
        # These plots occupy different column spans. Share numerical limits;
        # ViewBox's screen-aligned linking would stretch the wider plot's range.
        self._syncing_detector_range = False
        self.cfar_rp_plot.getViewBox().disableAutoRange(axis=pg.ViewBox.XAxis)
        for plot in (self.cfar_rd_plot, self.cfar_rp_plot):
            plot.getViewBox().sigXRangeChanged.connect(self._sync_detector_range)
        self.stage_range_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#eab83c", width=2))
        self.cfar_rp_plot.addItem(self.stage_range_line)
        self.stage_slice_detections = self.cfar_rp_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=9,
            symbolBrush="#229979",
            name="Published peaks in this bin",
        )
        self.stage_report_input = pg.ScatterPlotItem(
            pen=None, brush="#3d9fec", size=9, hoverable=True
        )
        self.stage_report_threshold = pg.ScatterPlotItem(
            pen="#ec7955", brush=None, symbol="x", size=10, hoverable=True
        )
        self.cfar_rp_plot.addItem(self.stage_report_input)
        self.cfar_rp_plot.addItem(self.stage_report_threshold)

        # GT range at persisted truth/result times
        self.gt_range_plot = self._make_plot("GT Range vs Result Time")
        self.gt_range_plot.setLabel("bottom", "Frame")
        self.gt_range_plot.setLabel("left", "Range (m)")
        self.gt_range_plot.add_external_legend(labelTextSize="8pt", colCount=2)
        self._gt_range_det_scatter = pg.ScatterPlotItem(
            [],
            [],
            size=12,
            symbol="x",
            pen=pg.mkPen(_COLORS["det_marker"], width=2),
            brush=pg.mkBrush(_COLORS["det_marker"]),
            name="Detections",
        )
        self.gt_range_plot.addItem(self._gt_range_det_scatter)

        # GT speed at persisted truth/result times
        self.gt_speed_plot = self._make_plot("GT Speed vs Result Time")
        self.gt_speed_plot.setLabel("bottom", "Frame")
        self.gt_speed_plot.setLabel("left", "Monostatic-equivalent Velocity (m/s)")
        self.gt_speed_plot.add_external_legend(labelTextSize="8pt", colCount=2)
        self._gt_speed_det_scatter = pg.ScatterPlotItem(
            [],
            [],
            size=12,
            symbol="x",
            pen=pg.mkPen(_COLORS["det_marker"], width=2),
            brush=pg.mkBrush(_COLORS["det_marker"]),
            name="Detections",
        )
        self.gt_speed_plot.addItem(self._gt_speed_det_scatter)

        # Range-Collapsed Micro-Doppler Spectrogram (2D: time × Doppler)
        self.collapsed_doppler_plot = self._make_plot("Collapsed Micro-Doppler")
        self.collapsed_doppler_plot.setLabel("bottom", "Time (s)")
        self.collapsed_doppler_plot.setLabel("left", "Doppler (Hz)")
        self.collapsed_doppler_image = pg.ImageItem()
        self.collapsed_doppler_image.setLookupTable(self._micro_lut)
        self.collapsed_doppler_plot.addItem(self.collapsed_doppler_image)
        self.collapsed_doppler_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("#00a8bb", width=2))
        self.collapsed_doppler_plot.addItem(self.collapsed_doppler_cursor)
        self.collapsed_doppler_mesh = pg.PColorMeshItem(
            colorMap=pg.colormap.get("inferno", source="matplotlib"), enableAutoLevels=False
        )
        self.collapsed_doppler_plot.addItem(self.collapsed_doppler_mesh)
        self.collapsed_doppler_mesh.hide()
        self.micro_colorbar = pg.ColorBarItem(
            colorMap=pg.colormap.get("inferno", source="matplotlib"),
            label="Channel power (dB)",
            interactive=False,
            width=12,
        )
        self.micro_colorbar.setImageItem(
            self.collapsed_doppler_image, insert_in=self.collapsed_doppler_plot.getPlotItem()
        )

        self.tracking_plot = self._make_plot("Tracks and ground truth")
        self.tracking_plot.setLabel("bottom", "World X (m)")
        self.tracking_plot.setLabel("left", "World Y (m)")
        self.tracking_plot.setAspectLocked(True)
        self.tracking_plot.add_external_legend(labelTextSize="9pt")
        self.environment = TrackingEnvironment(self.tracking_plot, self)
        self._inspection_scale = PowerScale()
        self._stage_count_key = None
        self._stage_images = {}
        self._stage_image_bundle = None
        self._stage_image_render_key = None
        self.processing_panel = ProcessingPanel(self)
        self.processing_panel.cell_selected.connect(self._select_processing_cell)
        self.processing_panel.hide()
        self.processing_panel.reconstruction_requested.connect(
            self._request_spectrum_reconstruction
        )
        self.stage_surface = SpectrumSurface(self)
        self.stage_surface.scope.setCurrentText("All detections region")
        self.stage_surface.hide()

        # One metadata browser is shared by every workspace.
        self.result_browser = SensingResultBrowser(self)
        self.detection_table_model = self.result_browser.model
        self.detection_table_proxy = self.result_browser.proxy
        self.detection_table = self.result_browser.table
        self.result_browser.detection_selected.connect(self._select_browser_detection)
        self.result_browser.track_selected.connect(self._select_browser_track)
        self.result_browser.selection_cleared.connect(
            lambda: self.set_linked_selection(None, notify=True)
        )

        # Micro-Doppler Spectrogram (shown only in Micro-Doppler view mode)
        self.micro_plot = self._make_plot("Micro-Doppler Spectrogram")
        self.micro_image = pg.ImageItem()
        self.micro_image.setLookupTable(self._micro_lut)
        self.micro_plot.addItem(self.micro_image)
        self.micro_plot.setLabel("bottom", "Time (s)")
        self.micro_plot.setLabel("left", "Doppler (Hz)")

        # Range-Time Waterfall
        self.range_time_plot = self._make_timeline_plot("Range-Time Waterfall")
        self.range_time_raster = pg.ImageItem(axisOrder="col-major")
        self.range_time_raster.setLookupTable(self._rd_cmap.getLookupTable(nPts=256))
        self.range_time_raster.hide()
        self.range_time_plot.addItem(self.range_time_raster)
        self.range_time_image = pg.PColorMeshItem(
            colorMap=self._rd_cmap,
            enableAutoLevels=False,
        )
        self.range_time_image.setVisible(False)
        self.range_time_plot.addItem(self.range_time_image)
        self.range_time_plot.setMinimumHeight(140)
        self.range_time_plot.setMaximumHeight(16777215)
        self.range_time_plot.getPlotItem().getAxis("bottom").setLabel(
            "Result reference time", units="s"
        )
        self.range_time_plot.getPlotItem().showAxis("left", True)
        self.range_time_plot.getPlotItem().getAxis("left").setLabel("Path Length (m)")
        self.range_time_plot.getPlotItem().getAxis("left").setStyle(tickFont=QFont("Segoe UI", 7))
        self.range_time_plot.getPlotItem().getAxis("left").setPen(
            pg.mkPen(color="#888888", width=1)
        )

        # Acquisition/result timeline. Every marker uses schema-v4 physical
        # seconds; opaque scene/storage IDs are never treated as time.
        self.acquisition_timeline_plot = self._make_timeline_plot("Acquisition timeline")
        self.acquisition_timeline_plot.setMinimumHeight(80)
        self.acquisition_timeline_plot.setMaximumHeight(120)
        self.acquisition_timeline_plot.setToolTip(
            "Blue: acquisition span; green circles: valid samples; red crosses: invalid "
            "samples; orange: result reference; purple dashed: current scene time."
        )
        self.acquisition_timeline_plot.getPlotItem().getAxis("bottom").setLabel(
            "Scenario-clock time", units="s"
        )
        self.acquisition_span_curve = self.acquisition_timeline_plot.plot(
            [],
            [],
            pen=pg.mkPen(color="#1976d2", width=5),
        )
        self.acquisition_valid_samples = pg.ScatterPlotItem(
            size=8,
            symbol="o",
            brush=pg.mkBrush("#2e7d32"),
            pen=pg.mkPen("#1b5e20", width=1),
        )
        self.acquisition_invalid_samples = pg.ScatterPlotItem(
            size=10,
            symbol="x",
            brush=pg.mkBrush("#c62828"),
            pen=pg.mkPen("#c62828", width=2),
        )
        self.acquisition_timeline_plot.addItem(self.acquisition_valid_samples)
        self.acquisition_timeline_plot.addItem(self.acquisition_invalid_samples)
        self.acquisition_result_marker = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color="#ef6c00", width=2),
        )
        self.acquisition_scene_time_marker = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(color="#6a1b9a", width=2, style=Qt.DashLine),
        )
        self.acquisition_result_marker.setVisible(False)
        self.acquisition_scene_time_marker.setVisible(False)
        self.acquisition_timeline_plot.addItem(self.acquisition_result_marker)
        self.acquisition_timeline_plot.addItem(self.acquisition_scene_time_marker)
        self.acquisition_timeline_plot.setYRange(-1.0, 1.0, padding=0.0)
        self.timing_panel = QWidget()
        timing_layout = QVBoxLayout(self.timing_panel)
        timing_layout.setContentsMargins(0, 0, 0, 0)
        self.timing_status = QLabel("Acquisition timing unavailable")
        self.timing_status.setWordWrap(True)
        timing_layout.addWidget(self.timing_status)
        self.timing_toggle = QPushButton("Timing details")
        self.timing_toggle.setCheckable(True)
        timing_layout.addWidget(self.timing_toggle)
        timing_layout.addWidget(self.acquisition_timeline_plot)
        self.acquisition_timeline_plot.hide()
        self.timing_toggle.toggled.connect(self.acquisition_timeline_plot.setVisible)

        # Persisted detector-stage inspector. These controls never invoke the
        # signal-processing pipeline; they only index the immutable bundle on
        # the current SensingUpdateRequest.
        self.detection_stage_inspector = QWidget(self)
        self.detection_stage_inspector.hide()
        stage_layout = QVBoxLayout(self.detection_stage_inspector)
        stage_layout.setContentsMargins(6, 6, 6, 6)
        stage_layout.setSpacing(4)
        stage_controls = WrappingLayout()
        stage_controls.setContentsMargins(0, 0, 0, 0)
        stage_controls.setSpacing(5)

        self._detection_stage_kind = "range_doppler_detection_input_power"

        self._detection_stage_pair_box = QComboBox(self.detection_stage_inspector)
        self._detection_stage_pair_box.hide()
        self._detection_stage_pair_box.setMinimumContentsLength(10)
        self._detection_stage_pair_box.setToolTip("Exact canonical sensing pair identity")
        # A bundle contains exactly one pair; its full identity belongs in provenance.

        stage_controls.addWidget(QLabel("Doppler bin:"))
        self._detection_stage_doppler_box = QSpinBox()
        self._detection_stage_doppler_box.setRange(0, 0)
        stage_controls.addWidget(self._detection_stage_doppler_box)

        stage_controls.addWidget(QLabel("Range bin:"))
        self._detection_stage_range_box = QSpinBox()
        self._detection_stage_range_box.setRange(0, 0)
        stage_controls.addWidget(self._detection_stage_range_box)
        self.stage_clear_selection_button = QPushButton("Clear selection")
        self.stage_clear_selection_button.setToolTip(
            "Remove the selected bin or detection while keeping all detector maps visible."
        )
        self.stage_clear_selection_button.clicked.connect(
            lambda: self.set_linked_selection(None, notify=True)
        )
        stage_controls.addWidget(self.stage_clear_selection_button)
        self.stage_comparison_scope = QComboBox()
        self.stage_comparison_scope.addItem("Doppler slice", "slice")
        self.stage_comparison_scope.addItem("All detections", "reports")
        self.stage_comparison_scope.setToolTip(
            "The slice shows one Doppler row. All detections compares input and threshold at each published peak's own range–Doppler cell; it is not a projected range profile."
        )
        self.stage_comparison_scope.currentIndexChanged.connect(self._rerender_last_request)
        stage_controls.addWidget(self.stage_comparison_scope)
        self.stage_surface.scope.currentIndexChanged.connect(
            self._update_detection_stage_inspector_from_controls
        )
        self.stage_surface_toggle = QPushButton("3D surface")
        self.stage_surface_toggle.setCheckable(True)
        self.stage_surface_toggle.setEnabled(False)
        self.stage_surface_toggle.setToolTip(
            "Compare saved input and CFAR threshold across the whole grid, or zoom to the selected neighborhood. Green markers show published peak cells."
        )
        self.stage_surface_toggle.toggled.connect(self._rerender_last_request)
        stage_actions = WrappingLayout()
        stage_actions.addWidget(self.stage_surface_toggle)

        self.detection_stage_export_button = QPushButton("Export replay…")
        self.detection_stage_export_button.setEnabled(False)
        self.detection_stage_export_button.setToolTip(
            "Choose a new package destination and export the canonical replay plus "
            "this persisted cell inspection. Existing destinations are never overwritten."
        )
        self.detection_stage_export_button.clicked.connect(self._export_detection_stage_replay)
        stage_layout.addLayout(stage_controls)
        stage_layout.addLayout(stage_actions)

        self.detection_stage_summary = QLabel("Detector-stage details are not loaded")
        self.detection_stage_summary.setWordWrap(True)
        self.detection_stage_summary.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        self.detection_stage_summary.setMinimumHeight(0)
        self.detection_stage_details_toggle = QPushButton("Source, timing and export")
        self.detection_stage_details_toggle.setCheckable(True)
        self.detection_stage_text = QPlainTextEdit()
        self.detection_stage_text.setReadOnly(True)
        self.detection_stage_text.setMaximumBlockCount(300)
        self.detection_stage_text.setMinimumHeight(150)
        self.detection_stage_text.setMaximumHeight(260)
        self.detection_stage_text.setFont(QFont("Monospace", 8))
        self.detection_stage_text.setPlainText(
            "Detector-stage inspection unavailable: no canonical replay request.\n"
            "Persisted products only; no sensing reprocessing is performed."
        )
        self.stage_technical = QScrollArea()
        self.stage_technical.setWidgetResizable(True)
        self.stage_technical.setMinimumHeight(160)
        self.stage_technical.setMaximumHeight(210)
        technical = QWidget()
        technical_layout = QVBoxLayout(technical)
        technical_layout.addWidget(self.detection_stage_text)
        technical_layout.addWidget(self.timing_panel)
        technical_layout.addWidget(self.detection_stage_export_button)
        self.stage_technical.setWidget(technical)
        self.stage_technical.hide()
        self.detection_stage_details_toggle.toggled.connect(self.stage_technical.setVisible)
        self.detection_stage_details_toggle.toggled.connect(
            self._update_detection_stage_inspector_from_controls
        )

        self._detection_stage_pair_box.currentIndexChanged.connect(
            self._update_detection_stage_inspector_from_controls
        )
        self._detection_stage_doppler_box.valueChanged.connect(self._select_stage_control_cell)
        self._detection_stage_range_box.valueChanged.connect(self._select_stage_control_cell)

        self._plots_widget = QWidget()
        self._parked_plots = QWidget(self)
        self._parked_plots.hide()
        self._plots_widget.setLayout(self._grid)
        root.addWidget(self._plots_widget, stretch=1)
        self._unavailable_label = QLabel()
        self._unavailable_label.setWordWrap(True)
        self._unavailable_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._unavailable_label.hide()
        root.addWidget(self._unavailable_label, stretch=1)
        root.addWidget(self.detection_stage_summary)

        # ── Options bar (two rows to keep minimum width small) ────────
        options_row1 = QGridLayout()
        options_row1.setContentsMargins(0, 0, 0, 0)
        options_row1.setSpacing(6)
        self._view_mode_box = QComboBox()
        self._view_mode_box.addItem("Run summary", "summary")
        self._view_mode_box.addItem("Current result", "current")
        self._view_mode_box.addItem("Detection", "analysis")
        self._view_mode_box.addItem("Motion", "micro-doppler")
        self._view_mode_box.addItem("Tracking", "tracking")
        self._view_mode_box.addItem("Signal & detector", "diagnostics")
        self._view_mode_box.addItem("Angle", "angular")
        self._view_mode_box.addItem("Evaluation", "evaluation")
        angle_item = self._view_mode_box.model().item(self._view_mode_box.findData("angular"))
        angle_item.setEnabled(False)
        angle_item.setToolTip("Load a result with a saved angular product to use this view.")
        tracking_item = self._view_mode_box.model().item(self._view_mode_box.findData("tracking"))
        tracking_item.setEnabled(False)
        tracking_item.setToolTip(
            "Load a recording with saved radial or Cartesian tracks to use this view."
        )
        self._view_mode_box.currentIndexChanged.connect(self._on_view_mode_changed)

        self._micro_label = QLabel("Spectrum view:")
        options_row1.addWidget(self._micro_label, 0, 0)
        self._micro_mode_box = QComboBox()
        self._micro_mode_box.addItem("All ranges (recording)", "auto")
        self._micro_mode_box.addItem("Saved micro-Doppler", "saved-stft")
        saved_stft_item = self._micro_mode_box.model().item(1)
        saved_stft_item.setEnabled(False)
        saved_stft_item.setToolTip("A fixed STFT display product was not recorded.")
        self._micro_mode_box.addItem("Around true range (oracle)", "tracked")
        self._micro_mode_box.addItem("Around detected range", "gated")
        self._micro_mode_box.addItem("One detected range bin", "fixed-bin")
        for token, tooltip in _MOTION_VIEW_TOOLTIPS.items():
            item = self._micro_mode_box.model().item(self._micro_mode_box.findData(token))
            if item.isEnabled():
                item.setToolTip(tooltip)
        self._micro_mode_box.setToolTip(
            "All ranges and saved micro-Doppler show the recording. Range selections "
            "inspect the current result's saved range–Doppler power.\n"
            "The oracle selection uses ground truth. These displays do not rerun "
            "detection or tracking."
        )
        self._micro_mode_box.currentIndexChanged.connect(self._on_view_mode_changed)
        options_row1.addWidget(self._micro_mode_box, 0, 1, 1, 3)

        self._micro_gate_label = QLabel("Gate half-width (bins):")
        options_row1.addWidget(self._micro_gate_label, 0, 4)
        self._micro_gate_box = QSpinBox()
        self._micro_gate_box.setRange(0, 16)
        self._micro_gate_box.setValue(self._micro_range_gate_half_width)
        self._micro_gate_box.setEnabled(False)
        self._micro_gate_box.setToolTip(
            "Include this many bins on either side of the detected or GT range bin."
        )
        self._micro_gate_box.valueChanged.connect(self._rerender_last_request)
        options_row1.addWidget(self._micro_gate_box, 0, 5)

        self._rd_range_label = QLabel("RD display range (dB):")
        options_row1.addWidget(self._rd_range_label, 1, 0)
        self._rd_dynamic_range_box = QSpinBox()
        self._rd_dynamic_range_box.setRange(20, 100)
        self._rd_dynamic_range_box.setValue(50)
        self._rd_dynamic_range_box.setToolTip(
            "Color-scale span below its reference. This does not change detector sensitivity."
        )
        self._rd_dynamic_range_box.valueChanged.connect(self._rerender_last_request)
        options_row1.addWidget(self._rd_dynamic_range_box, 1, 1)

        self._motion_range_label = QLabel("Motion display range (dB):")
        options_row1.addWidget(self._motion_range_label, 2, 0)
        self._micro_dynamic_range_box = QSpinBox()
        self._micro_dynamic_range_box.setRange(20, 100)
        self._micro_dynamic_range_box.setValue(60)
        self._micro_dynamic_range_box.setToolTip(
            "Color-scale span for the motion spectrum; saved signal values remain unchanged."
        )
        self._micro_dynamic_range_box.valueChanged.connect(self._rerender_last_request)
        options_row1.addWidget(self._micro_dynamic_range_box, 2, 1)
        self._variant_box = QComboBox()
        self._variant_box.addItem("Saved spectrum", "primary")
        self._variant_box.addItem("Alternate saved spectrum", "raw")
        self._variant_box.setToolTip(
            "Both spectra include the saved windows and Fourier transforms. "
            "These are not raw receiver samples. Changing the display does not rerun processing."
        )
        self._variant_box.currentIndexChanged.connect(self._on_product_variant_changed)
        options_row1.addWidget(self._variant_box, 3, 0, 1, 6)
        self._product_note = QLabel("Display product selection does not rerun the detector.")
        self._product_note.setWordWrap(True)
        options_row1.addWidget(self._product_note, 4, 0, 1, 6)
        self._environment_opacity_label = QLabel("Scene opacity:")
        options_row1.addWidget(self._environment_opacity_label, 6, 0)
        options_row1.addWidget(self.environment.opacity, 6, 1)
        options_row1.addWidget(self.environment.note, 7, 0, 1, 6)
        options_row1.addWidget(self.detection_stage_details_toggle, 13, 0, 1, 6)
        options_row1.addWidget(self.stage_technical, 14, 0, 1, 6)
        self._advanced_widget = QWidget()
        self._advanced_widget.setLayout(options_row1)
        root.insertWidget(1, self._advanced_widget)

        common = WrappingLayout()
        common.addWidget(QLabel("View:"))
        common.addWidget(self._view_mode_box)
        self._support_box = QComboBox()
        self._support_box.currentIndexChanged.connect(self._on_view_mode_changed)
        common.addWidget(self._support_box)
        self._advanced_motion_cb = QCheckBox("Motion display options")
        self._advanced_motion_cb.setToolTip(
            "Choose a recording overview, saved micro-Doppler, or a range selection "
            "for the current result. These displays do not rerun detection or tracking. "
            "Uncheck to return to the all-range overview."
        )
        self._advanced_motion_cb.toggled.connect(self._on_advanced_motion_toggled)
        common.addWidget(self._advanced_motion_cb)
        options_row1.addWidget(self.environment.toggle, 6, 2, 1, 4)
        self.table_button = QPushButton("Detection table")
        self.table_button.setCheckable(True)
        self.table_button.toggled.connect(self._toggle_detection_table)
        common.addWidget(self.table_button)
        if find_spec("visualizer.src.sensing.motion_comparison") is not None:
            self.motion_comparison_button = QPushButton("Compare motion…")
            self.motion_comparison_button.setToolTip(
                "Open a separate analysis tool for continuous sequential CIR recordings. "
                "Independent coherent CPIs cannot be used as a continuous recording. "
                "Choose source recordings and spectral settings explicitly."
            )
            self.motion_comparison_button.clicked.connect(self._open_motion_comparison)
            options_row1.addWidget(self.motion_comparison_button, 10, 0, 1, 3)
        self._table_requested = False
        self._lock_scale_cb = QCheckBox("Lock color scale")
        self._lock_scale_cb.setChecked(True)
        self._lock_scale_cb.setToolTip(
            "Keep a fixed reference while playing; unlock to scale each result."
        )
        self._lock_scale_cb.toggled.connect(self._reset_color_scale)
        options_row1.addWidget(self._lock_scale_cb, 1, 2, 1, 4)
        self._graph_height_box = QSpinBox()
        self._graph_height_box.setRange(120, 1000)
        self._graph_height_box.setSingleStep(40)
        self._graph_height_box.setSuffix(" px")
        self._graph_height_box.setValue(180)
        self._graph_height_box.setToolTip(
            "Minimum height of the data area, excluding the title, axes and legend. "
            "Increase it for crowded histories; the panel scrolls to fit."
        )
        self._graph_height_box.valueChanged.connect(self._apply_graph_height)
        root.insertLayout(1, common)

        navigation = WrappingLayout()
        self._result_locations = ()
        self.previous_result_button = QPushButton("Previous")
        self.next_result_button = QPushButton("Next")
        for button, direction in ((self.previous_result_button, -1), (self.next_result_button, 1)):
            button.setEnabled(False)
            button.clicked.connect(
                lambda _checked=False, d=direction: self.result_step_requested.emit(d)
            )
            navigation.addWidget(button)
        self.navigation_label = QLabel("Indexing saved results…")
        navigation.addWidget(self.navigation_label)
        self.loading_progress = QProgressBar()
        self.loading_progress.setMaximumWidth(210)
        self.loading_progress.hide()
        navigation.addWidget(self.loading_progress)
        self.cancel_loading_button = QPushButton("Cancel loading")
        self.cancel_loading_button.clicked.connect(self.loading_control_requested.emit)
        self.cancel_loading_button.hide()
        navigation.addWidget(self.cancel_loading_button)
        root.insertLayout(2, navigation)
        self.retention_notice = QLabel()
        self.retention_notice.setWordWrap(True)
        self.retention_notice.hide()
        self.view_description = QLabel()
        self.view_description.setWordWrap(True)
        root.insertWidget(3, self.view_description)
        self.motion_controls = QWidget()
        motion_controls_layout = WrappingLayout(self.motion_controls)
        motion_controls_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (
            self._micro_label,
            self._micro_mode_box,
            self._micro_gate_label,
            self._micro_gate_box,
        ):
            motion_controls_layout.addWidget(widget)
        root.insertWidget(4, self.motion_controls)
        self.setup_linked_selection(root)
        self.association_figure = InspectionFigurePanel("Association decisions", self)
        self.association_figure.hide()
        self.association_figure.toggle.toggled.connect(self._rerender_last_request)
        root.insertWidget(4, self.association_figure)
        self.tracking_inspector = TrackingInspector(self)
        self.tracking_inspector.track_selected.connect(self.select_track)
        self.tracking_inspector.evidence_requested.connect(self._select_saved_evidence)
        root.insertWidget(4, self.tracking_inspector)
        self.run_summary_panel = RunSummaryPanel(self)
        self.run_summary_panel.set_embedded_graph_height(self._graph_height_box.value())
        self.run_summary_panel.settings_changed.connect(self.view_changed.emit)
        self.run_summary_panel.time_selected.connect(self._seek_summary_time)
        self.run_summary_panel.hide()
        self.evaluation_panel = EvaluationPanel(self)
        self.evaluation_panel.history_requested.connect(self.view_changed.emit)
        self.evaluation_panel.hide()
        self.evaluation_settings = EvaluationSettings(self)
        self.run_summary_panel.bind_evaluation_settings(self.evaluation_settings)
        self.evaluation_panel.bind_evaluation_settings(self.evaluation_settings)
        self.evaluation_settings.changed.connect(self._refresh_current_summary)
        for plot in (self.micro_plot, self.collapsed_doppler_plot, self.range_time_plot):
            plot.scene().sigMouseClicked.connect(
                lambda event, p=plot: self._seek_plot_time(p, event)
            )

        options_row2 = WrappingLayout()
        options_row2.setContentsMargins(0, 0, 0, 0)
        options_row2.setSpacing(6)

        self._show_heatmaps_cb = QCheckBox("Heatmaps")
        self._show_heatmaps_cb.setChecked(
            QSettings().value("visualizer/sensing_heatmaps", True, type=bool)
        )
        self._show_heatmaps_cb.setToolTip(
            "Draw range–Doppler and range/Doppler–azimuth power maps. Turn off to reduce "
            "plotting work; markers, profiles and angular spectrum remain. Pinned maps "
            "stay enabled. This display option does not change loaded results."
        )
        self._show_heatmaps_cb.toggled.connect(self._on_heatmaps_toggled)
        options_row2.addWidget(self._show_heatmaps_cb)

        self._show_det_cb = QCheckBox("Detections")
        self._show_det_cb.setChecked(True)
        self._show_det_cb.stateChanged.connect(self._apply_overlay_visibility)
        options_row2.addWidget(self._show_det_cb)

        self._show_gt_cb = QCheckBox("Ground truth (evaluation)")
        self._show_gt_cb.setChecked(True)
        self._show_gt_cb.setToolTip(
            "Show stored ground truth as an evaluation overlay. This display toggle does not "
            "drive blind sensing estimates; oracle-assisted processing must be selected explicitly."
        )
        self._show_gt_cb.stateChanged.connect(self._apply_overlay_visibility)
        options_row2.addWidget(self._show_gt_cb)

        self._show_cfar_cb = QCheckBox("Mark CFAR passes")
        self._show_cfar_cb.setToolTip(
            "Highlight recorded CFAR passes with a green fill and red cell borders. "
            "Later filtering and clustering can remove them; these are not all published detections."
        )
        self._show_cfar_cb.setChecked(True)
        self._show_cfar_cb.stateChanged.connect(self._apply_overlay_visibility)
        options_row2.addWidget(self._show_cfar_cb)

        self._zoom_data_cb = QCheckBox("Zoom to data")
        self._zoom_data_cb.setToolTip(
            "Use the scenario's display bounds when recorded, otherwise fit significant "
            "power and visible markers. Uncheck to show the full saved response. "
            "Histories fit the displayed results when checked; uncheck for the full "
            "recording time span and recorded range, velocity and XY extents."
        )
        self._zoom_data_cb.setChecked(False)
        self._zoom_data_cb.toggled.connect(self._rerender_last_request)
        options_row2.addWidget(self._zoom_data_cb)
        options_row2.addWidget(self._profile_full_dynamic_range_cb)
        options_row2.addWidget(self._profile_detections_cb)

        self._doppler_hz_cb = QCheckBox("Doppler (Hz)")
        self._doppler_hz_cb.setChecked(False)
        self._doppler_hz_cb.setToolTip("Show Doppler axis in Hz instead of velocity (m/s)")
        self._doppler_hz_cb.stateChanged.connect(self._on_doppler_axis_toggle)
        options_row2.addWidget(self._doppler_hz_cb)

        self._show_bin_grid_cb = QCheckBox("Show bin grid")
        self._show_bin_grid_cb.setToolTip(
            "Outline saved range and velocity cells. Clicks select the nearest cell centre. "
            "Bin spacing describes the saved grid, not physical target-separation resolution."
        )
        self._show_bin_grid_cb.toggled.connect(self._update_detector_bin_grid)
        options_row2.addWidget(self._show_bin_grid_cb)
        root.addLayout(options_row2)
        self.bin_spacing_label = QLabel()
        self.bin_spacing_label.setWordWrap(True)
        self.bin_spacing_label.setToolTip(self._show_bin_grid_cb.toolTip())
        root.addWidget(self.bin_spacing_label)

        self.provenance_label = QLabel("Waiting for sensing provenance...")
        self.provenance_label.setAlignment(Qt.AlignCenter)
        self.provenance_label.setFont(QFont("Segoe UI", 8))
        self.provenance_label.setWordWrap(True)
        root.addWidget(self.provenance_label)

        # ── Summary bar ─────────────────────────────────────────────────
        self.summary_label = QLabel("Waiting for sensing data...")
        self.summary_label.setAlignment(Qt.AlignCenter)
        self.summary_label.setFont(QFont("Segoe UI", 9))
        self.summary_label.setWordWrap(True)
        root.addWidget(self.summary_label)

        self._build_workspace_shell(root, navigation, options_row2)
        self._build_figure_options(root)
        self._options_grid = options_row1
        self._options_positions = [
            (options_row1.itemAt(i).widget(), options_row1.getItemPosition(i))
            for i in range(options_row1.count())
        ]
        self._apply_view_mode()
        self._apply_overlay_visibility()
        self._theme_manager = get_theme_manager()
        self._theme_manager.theme_changed.connect(self._apply_theme)
        self._apply_graph_height()
        self._apply_theme()
        logger.info("Sensing window initialized (dynamic layout + timeline)")

    def _build_figure_options(self, root: QVBoxLayout) -> None:
        """Keep optional display controls out of the visible figures' vertical budget."""
        settings = QSettings()
        key = "visualizer/sensing_figure_options_open"
        self.figure_options = CollapsibleSection(
            "Figure options", start_open=settings.value(key, False, type=bool)
        )
        self.figure_options._toggle.setToolTip(
            "Show display settings, overlays and current-result details. "
            "Collapsing these controls keeps the figures updating."
        )
        self.figure_options.toggled.connect(lambda opened: settings.setValue(key, opened))
        evidence = self.evidence_scroll.widget().layout()
        for widget in (
            self.view_description,
            self._advanced_widget,
            self.motion_controls,
            self.current_result_summary,
        ):
            layout = self.figure_options.content_layout()
            index = evidence.indexOf(widget)
            if index >= 0:
                # Transfer the uncached QWidgetItem intact. Deleting it through
                # removeWidget can invalidate PySide's layout-item ownership.
                item = evidence.takeAt(index)
                widget.setParent(layout.parentWidget())
                layout.addItem(item)
            else:
                layout.addWidget(widget)
        evidence.insertWidget(0, self.figure_options)
        # The workspace rebuilds the original root layout. Give coverage
        # guidance a real row outside the optional, collapsible controls.
        evidence.insertWidget(1, self.retention_notice)

        # Table visibility is a common navigation action; keep it beside the
        # workspace selector rather than inside optional figure styling.
        self.table_button.parentWidget().layout().removeWidget(self.table_button)
        root.itemAt(1).layout().addWidget(self.table_button)
        optional = QWidget()
        options = WrappingLayout(optional)
        options.setContentsMargins(0, 0, 0, 0)
        for widget in (
            self._advanced_motion_cb,
            self.tracking_measurements,
            self.truth_zoom.combo,
        ):
            root.itemAt(4).layout().removeWidget(widget)
            options.addWidget(widget)
        self._advanced_widget.layout().addWidget(optional, 17, 0, 1, 6)

        # The panel section or floating shell already names this analysis.
        # Keep result selection and placement together so a laptop does not
        # spend three full rows on the dashboard heading and navigation.
        header = root.takeAt(0).layout()
        tools = root.itemAt(0).layout()
        navigation = root.takeAt(1).layout()
        while header.count():
            item = header.takeAt(0)
            widget = item.widget()
            if widget is self.placement_button:
                tools.addItem(item)
            elif widget is not None:
                widget.hide()
                widget.setParent(self._parked_plots)
        while navigation.count():
            tools.addItem(navigation.takeAt(0))
        header.deleteLater()
        navigation.deleteLater()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        compact = event.size().width() < 760
        if compact == self._compact or not hasattr(self, "_options_positions"):
            return
        self._compact = compact
        grid = self._options_grid
        for widget, _ in self._options_positions:
            grid.removeWidget(widget)
        if compact:
            row = 0
            previous = None
            for widget, (r, c, rs, cs) in sorted(
                self._options_positions, key=lambda item: item[1][:2]
            ):
                if cs == 1 and previous == (r, c - 1) and c % 2 == 1:
                    grid.addWidget(widget, row - 1, 1)
                else:
                    grid.addWidget(widget, row, 0, 1, 1 if cs == 1 and c % 2 == 0 else 2)
                    row += 1
                previous = (r, c) if cs == 1 else None
        else:
            for widget, position in self._options_positions:
                grid.addWidget(widget, *position)
        self._repopulate_grid(self._selected_view_mode())
        self.updateGeometry()

    # ── Plot factories ─────────────────────────────────────────────────
    def _make_timeline_plot(self, title_text: str) -> pg.PlotWidget:
        """Factory for timeline-style plots (no Y axis)."""
        pw = ManagedPlotWidget(parent=self)
        pw.hide()
        apply_pyqtgraph_plot_theme(pw, title=title_text)
        pw.showGrid(x=True, y=False, alpha=0.15)
        pw.setMinimumHeight(90)
        pw.setMaximumHeight(180)
        pw.setMouseEnabled(x=False, y=False)
        pw.hideButtons()
        pi = pw.getPlotItem()
        pi.getViewBox().setBorder(pg.mkPen(color="#cccccc", width=1))
        pi.showAxis("bottom", True)
        pi.getAxis("bottom").setLabel("Frame")
        pi.getAxis("bottom").setStyle(tickFont=QFont("Segoe UI", 7))
        pi.getAxis("bottom").setPen(pg.mkPen(color="#888888", width=1))
        for axis_name in ("left", "right", "top"):
            pi.showAxis(axis_name, False)
        return pw

    def _make_plot(self, title_text: str) -> pg.PlotWidget:
        """Factory for standard 2D plots."""
        pw = DashboardPlot()
        # A figure may never enter a visible layout. Own it immediately so Qt
        # also retires unused plots if the dashboard's parent is destroyed.
        pw.setParent(self)
        pw.hide()
        apply_pyqtgraph_plot_theme(pw, title=title_text)
        pw.setMinimumWidth(180)
        return pw

    def _apply_graph_height(self, *_args: Any) -> None:
        """Resize plotting areas without replaying products or changing zoom."""
        self.run_summary_panel.set_embedded_graph_height(self._graph_height_box.value())
        for widget in (
            *vars(self).values(),
            self.processing_panel.slice,
            self.processing_panel.phase,
            self.evaluation_panel.plots.matches,
            self.evaluation_panel.plots.counts,
        ):
            if isinstance(widget, DashboardPlot):
                widget.set_data_minimum_height(self._graph_height_box.value())

    def _apply_theme(self, *_args: Any) -> None:
        """Style plots, legends, labels and colorbars using the application theme."""
        theme = get_theme_manager().current_theme
        for widget in (
            *vars(self).values(),
            self.evaluation_panel.plots.matches,
            self.evaluation_panel.plots.counts,
        ):
            if not isinstance(widget, pg.PlotWidget):
                continue
            plot = widget.getPlotItem()
            apply_pyqtgraph_plot_theme(widget, title=plot.titleLabel.text)
            legend = plot.legend
            if legend is not None:
                legend.setBrush(pg.mkBrush(theme.bg_secondary))
                legend.setPen(pg.mkPen(theme.border_primary))
                legend.setLabelTextColor(theme.text_primary)
                for _, label in legend.items:
                    label.setText(label.text)
        for bar in (
            self.rd_colorbar,
            self.micro_colorbar,
            self.stage_colorbar,
            *(plot.colorbar for plot in self._stage_maps.values()),
            *self._angular_colorbars.values(),
        ):
            for name in ("left", "right", "bottom", "top"):
                axis = bar.getAxis(name)
                axis.setPen(pg.mkPen(theme.text_secondary))
                axis.setTextPen(pg.mkPen(theme.text_secondary))
        self.provenance_label.setStyleSheet(f"color: {theme.text_secondary};")
        self.summary_label.setStyleSheet(f"color: {theme.text_secondary};")
        self.environment.set_color(theme.text_secondary)
        self.processing_panel.apply_theme()
        self.run_summary_panel.apply_theme()
        if self._last_request is not None:
            self.processing_panel.render()
            self._update_detection_stage_inspector_from_controls()

    def closeEvent(self, event: Any) -> None:
        """Disconnect theme hooks and release plots detached by a view change."""
        from ..widgets.plot_lifecycle import unregister_plot_views

        if getattr(self, "_plots_released", False):
            super().closeEvent(event)
            return
        self._plots_released = True
        plots = set(self.findChildren(pg.PlotWidget))
        plots.update(widget for widget in vars(self).values() if isinstance(widget, pg.PlotWidget))
        # Nested controllers can close/detach their plots during release. Retire
        # all view registrations while the graphics ownership tree is intact.
        for widget in plots:
            unregister_plot_views(widget)
        self._last_request = self._rendered_request = None
        self._linked_selection = None
        self.range_shell_changed.emit()
        self._stage_image_bundle = None
        self._stage_images.clear()
        self._inspection_scale.reset()
        self.tracking_inspector.binding = None
        self._theme_manager.theme_changed.disconnect(self._apply_theme)
        self.truth_zoom.close()
        self.environment.close()
        self.processing_panel.release()
        self.stage_surface.release()
        self.association_figure.reset()
        for widget in plots:
            # Include color bars as well as evaluation/processing plots.
            # PlotWidget.close detaches the Qt parent, so restore ownership.
            if widget.plotItem is not None:
                widget.close()
            widget.setParent(self._parked_plots)
        super().closeEvent(event)

    def product_variant(self) -> str:
        return str(self._variant_box.currentData() or "primary")

    def _update_product_options(self, sensing, request) -> None:
        """Expose saved capabilities and the processing behind vague variant names."""
        products = list((sensing.get("products") or {}).values())
        saved_stft = (
            self._selected_view_mode() == "micro-doppler"
            and self._selected_micro_doppler_mode() == "saved-stft"
        )
        kinds = {"range_profile", "range_doppler"}
        if self._selected_view_mode() == "angular":
            kinds = {"range_doppler_angle"}
        elif saved_stft:
            kinds = {"micro_doppler"}
            products.extend(
                {"kind": "micro_doppler", "variant": variant}
                for variant in request.recording_stft_variants or ()
            )
            overview = request.motion_overview
            compact = overview.families.get("saved-stft") if overview is not None else None
            if compact is not None:
                products.extend(
                    {"kind": "micro_doppler", "variant": variant}
                    for variant in compact.available_variants
                )
            if not any(d.get("kind") == "micro_doppler" for d in products):
                history = request.recording_history
                recorded = (
                    next((r for r in history.records if r.micro_power is not None), None)
                    if history is not None
                    else None
                )
                if recorded is not None:
                    products.extend((recorded.sensing_metadata or {}).get("products", {}).values())
        variants = {d.get("variant") for d in products if d.get("kind") in kinds}
        primary = next(
            (
                d
                for d in products
                if d.get("kind") == "range_doppler" and d.get("variant") == "primary"
            ),
            {},
        )
        applied = (primary.get("processing") or {}).get("slow_time_mean_removal_applied")
        # Older and non-coherent producers may not declare this operation.
        # Do not infer suppression (or equality of products) from their names.
        primary_label = (
            "After clutter suppression"
            if applied is True
            else "Spectrum (no clutter suppression)" if applied is False else "Saved spectrum"
        )
        raw_label = "Before clutter suppression" if applied is True else "Alternate saved spectrum"
        if saved_stft:
            primary_label = "Saved STFT · mean removed"
            raw_label = "Saved STFT · mean retained"
        primary_index = self._variant_box.findData("primary")
        raw_index = self._variant_box.findData("raw")
        self._variant_box.setItemText(primary_index, primary_label)
        self._variant_box.setItemText(raw_index, raw_label)
        duplicate = not saved_stft and applied is False and "primary" in variants
        self._has_alternate_spectrum = {"primary", "raw"}.issubset(variants) and not duplicate
        self._variant_box.setVisible(
            self._has_alternate_spectrum
            and self._selected_view_mode() in {"current", "analysis", "angular", "micro-doppler"}
        )
        self._variant_box.view().setRowHidden(raw_index, duplicate)
        if duplicate and self.product_variant() == "raw":
            with QSignalBlocker(self._variant_box):
                self._variant_box.setCurrentIndex(primary_index)
        if variants and self.product_variant() not in variants:
            with QSignalBlocker(self._variant_box):
                self._variant_box.setCurrentIndex(
                    primary_index if "primary" in variants else raw_index
                )
        for i in range(self._variant_box.count()):
            available = self._variant_box.itemData(i) in variants
            item = self._variant_box.model().item(i)
            item.setEnabled(available and not (i == raw_index and duplicate))
            item.setToolTip(
                (
                    "Fixed saved STFT mean policy; playback does not rerun its transform."
                    if saved_stft
                    else "Windowed Fourier spectrum; not raw receiver samples."
                )
                if available
                else (
                    "This STFT mean policy was not saved in this recording."
                    if saved_stft
                    else "This spectrum was not saved in this recording."
                )
            )
        processing = (
            "Saved detections use the spectrum after clutter suppression."
            if applied is True
            else (
                "No clutter suppression was applied."
                if applied is False
                else "Clutter-suppression status is not recorded for this spectrum."
            )
        )
        if saved_stft:
            from .recorded_products import saved_micro_doppler_unavailable_reason

            mean = "retained" if self.product_variant() == "raw" else "removed"
            processing = (
                f"Saved STFT with mean {mean}; fixed recorded window and range/time gates."
                if variants
                else saved_micro_doppler_unavailable_reason(sensing)
            )
        elif applied is True and "raw" not in variants:
            processing += " The spectrum before suppression was not saved."
        self._product_note.setText(
            processing + " Changing the display does not rerun detection or evaluation."
        )
        # Keep the old token readable by existing presets; avoid two equivalent
        # choices in the normal saved-recording workflow.
        self._micro_mode_box.setItemText(
            0,
            (
                "All ranges (recording)"
                if request.recording_managed
                else "Automatic saved / sampled motion"
            ),
        )
        if request.recording_managed:
            gt = self._extract_bound_gt_entry(sensing)
            for mode, enabled, reason in (
                (
                    "tracked",
                    gt is not None and len(gt[1]["target_ranges"]) == 1,
                    "Requires exactly one time-aligned ground-truth target.",
                ),
                (
                    "gated",
                    bool(sensing.get("detections")),
                    "Requires a saved detection with a range.",
                ),
                (
                    "fixed-bin",
                    bool(sensing.get("detections")),
                    "Requires a saved detection with a range.",
                ),
            ):
                item = self._micro_mode_box.model().item(self._micro_mode_box.findData(mode))
                item.setEnabled(enabled)
                item.setToolTip(_MOTION_VIEW_TOOLTIPS[mode] if enabled else reason)
        history = (
            request.tracking_history
            if request.tracking_history is not None
            else request.recording_history
        )
        if history is not None:
            tracks = not history.error and bool(
                history.tracking_plot_data.tracks
                or history.tracking_plot_data.estimates
                or history.tracking_plot_data.statuses
                or sensing["config_resolved"].get("track_confirmation_enabled")
            )
        else:
            config = sensing["config_resolved"]
            tracks = bool(
                request.saved_tracking is not None
                or config.get("track_confirmation_enabled")
                or any(d.get("track_id") is not None for d in sensing.get("detections", ()))
            )
        spatial = (
            bool(history.tracking_plot_data.tracks)
            if history is not None
            else bool(
                request.saved_tracking is not None
                or sensing["config_resolved"].get("tracker_type") in {"cartesian", "cartesian3d"}
            )
        )
        if spatial != getattr(self, "_tracking_spatial_available", None):
            self._tracking_spatial_available = spatial
            if self._selected_view_mode() == "tracking":
                self._repopulate_grid("tracking")
        item = self._view_mode_box.model().item(self._view_mode_box.findData("tracking"))
        item.setEnabled(tracks)
        item.setToolTip("" if tracks else "No saved tracking information.")

    def _update_recorded_capabilities(self, sensing, request) -> None:
        """Keep unavailable actions visible with the missing evidence explained."""
        from .recorded_products import (
            inspection_capabilities,
            product_kinds,
            saved_micro_doppler_unavailable_reason,
        )

        kinds = product_kinds(sensing, request.available_product_kinds)
        self._recorded_capabilities = inspection_capabilities(kinds)
        for combo in (self._support_box, self.input_kind_box):
            for index in range(combo.count()):
                capability = self._recorded_capabilities.get(combo.itemData(index))
                if capability is not None:
                    enabled, reason = capability
                    item = combo.model().item(index)
                    item.setEnabled(enabled)
                    item.setToolTip("" if enabled else reason)
        stft = self._micro_mode_box.model().item(self._micro_mode_box.findData("saved-stft"))
        history = request.recording_history
        has_stft = (
            "micro_doppler" in kinds
            or bool(request.recording_stft_variants)
            or (history is not None and any(r.micro_power is not None for r in history.records))
            or bool(
                request.motion_overview is not None
                and request.motion_overview.families.get("saved-stft") is not None
                and request.motion_overview.families["saved-stft"].coverage.available_results
            )
        )
        stft.setEnabled(has_stft)
        stft.setToolTip(
            _MOTION_VIEW_TOOLTIPS["saved-stft"]
            if stft.isEnabled()
            else saved_micro_doppler_unavailable_reason(sensing)
        )
        for mode in (
            ("auto", "tracked", "gated", "fixed-bin") if request.recording_managed else ()
        ):
            item = self._micro_mode_box.model().item(self._micro_mode_box.findData(mode))
            if "range_doppler" not in kinds:
                item.setEnabled(False)
                item.setToolTip("Saved range–Doppler maps are required for this view.")
            elif mode == "auto":
                item.setEnabled(True)
                item.setToolTip(_MOTION_VIEW_TOOLTIPS["auto"])
        selected = self._micro_mode_box.model().item(self._micro_mode_box.currentIndex())
        if not selected.isEnabled() and stft.isEnabled():
            with QSignalBlocker(self._micro_mode_box):
                self._micro_mode_box.setCurrentIndex(self._micro_mode_box.findData("saved-stft"))
            self._apply_view_mode()
        previous_map_availability = (self.rd_plot.isEnabled(), self.rp_plot.isEnabled())
        self.rd_plot.setEnabled("range_doppler" in kinds)
        self.rp_plot.setEnabled("range_profile" in kinds or "range_doppler" in kinds)
        if previous_map_availability != (
            self.rd_plot.isEnabled(),
            self.rp_plot.isEnabled(),
        ) and self._selected_view_mode() in {"current", "analysis"}:
            self._repopulate_grid(self._selected_view_mode())
        motion = self._view_mode_box.model().item(self._view_mode_box.findData("micro-doppler"))
        sampled = (sensing.get("config_resolved", {}).get("acquisition") or {}).get(
            "sample_period_s"
        ) is not None
        motion.setEnabled(
            sampled
            and (
                has_stft
                or bool(
                    kinds
                    & (
                        {"range_doppler", "micro_doppler"}
                        if request.recording_managed
                        else {
                            "range_doppler",
                            "micro_doppler",
                            "delay_domain_slow_time",
                            "delay_domain_cir",
                        }
                    )
                )
            )
        )
        motion.setToolTip(
            "" if motion.isEnabled() else "No sampled RD or STFT display products were recorded."
        )
        self._sync_workspace_navigation()

    def result_time_s(self) -> float:
        compact = getattr(self, "_compact_motion_cursor", None)
        if compact is not None:
            return compact.result.time_s if compact.result is not None else 0.0
        cursor = getattr(self, "_recording_cursor", None)
        if cursor is not None:
            return cursor.time_s
        if self._last_request is None:
            return 0.0
        return float(
            self._last_request.display_frame["sensing"]["temporal"]["result"]["reference_time_s"]
        )

    def can_move_compact_motion_cursor(self, overview) -> bool:
        """Only settled overview figures may follow time without result details."""
        request = self._last_request
        families = self.overview_families()
        return (
            overview is not None
            and request is not None
            and request is self._rendered_request
            and request.motion_overview is overview
            and self._selected_view_mode() == "micro-doppler"
            and self._selected_micro_doppler_mode() in {"auto", "collapsed", "saved-stft"}
            and not self.has_visible_live_figures()
            and not self._table_requested
            and self._linked_selection is None
            and not self.selection_tools_toggle.isChecked()
            and not getattr(self, "_pending_browser_selection", None)
            and bool(families)
            and families.issubset(overview.families)
            and all(
                overview.families[f].coverage.state in {"complete", "unavailable"} for f in families
            )
            and self._recording_render_key
            == (
                id(overview),
                self._selected_micro_doppler_mode(),
                self._micro_dynamic_range_db(),
                self._rd_dynamic_range_db(),
                self._zoom_data_cb.isChecked(),
            )
        )

    def move_compact_motion_cursor(self, cursor, variant: str) -> None:
        """Move a saved-overview marker without presenting stale result details."""
        if getattr(self, "_compact_motion_cursor", None) is None:
            self.evaluation_panel.update_event(None)
            self.detection_stage_export_button.setEnabled(False)
        self._recording_cursor = None
        self._compact_motion_cursor = cursor
        result = cursor.result
        self.collapsed_doppler_cursor.setVisible(result is not None)
        self.result_browser.hide()
        self.result_scroll.hide()
        self.table_button.setVisible(result is not None)
        if result is None:
            self.provenance_label.setText(
                f"Recording overview · no sensing result at scene {cursor.scene_id}"
            )
            self.provenance_label.setToolTip(
                "No saved result satisfies the current display policy."
            )
            for mode in ("tracked", "gated", "fixed-bin"):
                self._micro_mode_box.model().item(self._micro_mode_box.findData(mode)).setEnabled(
                    False
                )
        else:
            self.collapsed_doppler_cursor.setPos(result.time_s)
            self.provenance_label.setText(
                f"Result {result.time_s:.3f} s · bound scene {result.scene_id} · "
                f"playback scene {cursor.scene_id} · {variant}"
            )
            self.provenance_label.setToolTip(
                f"Saved overview · storage row {result.storage_id} · "
                f"clock/run/pair/acquisition: {' / '.join((*result.context, result.acquisition_id))}"
            )
        self.summary_label.setText("Saved Motion overview · select Results to load current details")
        self.summary_label.setToolTip(
            "Spectral data are unchanged; no current detections or scores were loaded."
        )

    def can_move_recording_cursor(self, history: Any) -> bool:
        """Only the complete all-range Motion overview can skip a product read."""
        return (
            history is not None
            and not self.has_visible_live_figures()
            and self._last_request is not None
            and self._last_request.recording_history is history
            and not self._last_request.recording_overview_only
            and not getattr(self, "_pending_browser_selection", None)
            and self._recording_render_key is not None
            and self._selected_view_mode() == "micro-doppler"
            and self._selected_micro_doppler_mode() in {"auto", "collapsed"}
        )

    def move_recording_cursor(self, record: Any, variant: str) -> None:
        """Move within validated saved history without fabricating a full request.

        Retain the last complete request for normal readers. Opening a gated
        spectrum, detection table or another view requests a fresh full result.
        """
        self._recording_cursor = record
        self._update_browser_cursor(record)
        self.evaluation_panel.update_event(record.evaluation, self.evaluation_panel.history)
        self.collapsed_doppler_cursor.setPos(record.time_s)
        self.provenance_label.setText(
            f"Result {record.time_s:.3f} s · scene {record.scene_frame_id} · {variant}"
        )
        self.provenance_label.setToolTip(
            f"Saved overview · storage row {record.storage_id} · "
            f"clock/run/pair/acquisition: {' / '.join(record.identity)}"
        )
        self.summary_label.setText(
            f"Saved Motion overview · {len(record.detections)} detections · no reprocessing"
        )
        self.summary_label.setToolTip("Cursor follows exact recorded scene bindings.")

    def _on_product_variant_changed(self, *_args: Any) -> None:
        self._recording_render_key = None
        self._locked_levels.clear()
        self._recorded_marker_styles = None
        self._history_bounds_cache = None
        self.view_changed.emit()

    def set_result_navigation(self, locations: tuple, *, previous: bool, following: bool) -> None:
        """Update compact navigation without touching images or loading products."""
        self._result_locations = locations
        self.previous_result_button.setEnabled(previous)
        self.next_result_button.setEnabled(following)
        self.navigation_label.setText(f"{len(locations)} results")

    def set_loading_progress(
        self, stage: str, completed: int, total: int, *, active: bool, cancelled: bool = False
    ) -> None:
        """Present background progress without blocking playback or clearing loaded plots."""
        started = getattr(self, "_loading_started_at", None)
        if active:
            if started is None:
                started = monotonic()
            self._loading_started_at = started
            # Keep fast scene-context refreshes from flashing the progress row
            # and resizing every plot. Initial loads remain immediately visible.
            active = self._last_request is None or monotonic() - started >= 0.15
        else:
            self._loading_started_at = None
        # Compare after debounce: a slow request with unchanged progress
        # must still reveal its progress row once the delay expires. Compare
        # the live widgets so a source reset or explicit hide is repaired too.
        display = (
            active,
            0,
            total,
            completed,
            f"{stage} · %p%" if total else stage,
            active or cancelled,
            "Resume loading" if cancelled else "Cancel loading",
        )
        current = (
            not self.loading_progress.isHidden(),
            self.loading_progress.minimum(),
            self.loading_progress.maximum(),
            self.loading_progress.value(),
            self.loading_progress.format(),
            not self.cancel_loading_button.isHidden(),
            self.cancel_loading_button.text(),
        )
        if current == display:
            return
        self.loading_progress.setVisible(active)
        self.loading_progress.setRange(0, total)
        self.loading_progress.setValue(completed)
        self.loading_progress.setFormat(f"{stage} · %p%" if total else stage)
        self.cancel_loading_button.setVisible(active or cancelled)
        self.cancel_loading_button.setText("Resume loading" if cancelled else "Cancel loading")

    def _seek_summary_time(self, time_s: float) -> None:
        result = nearest_result(self._result_locations, time_s)
        if result is not None:
            self.result_seek_requested.emit(result)

    def _seek_plot_time(self, plot: Any, event: Any) -> None:
        if (
            event.button() != Qt.MouseButton.LeftButton
            or event.modifiers() != Qt.KeyboardModifier.NoModifier
        ):
            return
        view = plot.getViewBox()
        if not view.sceneBoundingRect().contains(event.scenePos()):
            return
        result = nearest_result(
            self._result_locations, float(view.mapSceneToView(event.scenePos()).x())
        )
        if result is not None:
            self.result_seek_requested.emit(result)
            event.accept()

    def _toggle_detection_table(self, checked: bool) -> None:
        self._set_browser_visible(checked)
        if checked and getattr(self, "_compact_motion_cursor", None) is not None:
            # The retained full request can belong to an older cursor position.
            self.result_browser.hide()
            self.result_scroll.hide()
            self.view_changed.emit()

    def _reset_color_scale(self, *_args: Any) -> None:
        self._locked_levels.clear()
        self._inspection_scale.reset()
        self.processing_panel.scale.reset()
        self._recording_render_key = None
        self._rerender_last_request()

    def _levels(self, key: str, power_db: np.ndarray, span: float) -> tuple[float, float]:
        """Lock display levels without modifying scientific power values."""
        locked = self._lock_scale_cb.isChecked()
        if locked and key in self._locked_levels:
            maximum = self._locked_levels[key][1]
        else:
            maximum = float(np.nanmax(power_db))
        if locked:
            maximum = self._locked_levels.setdefault(key, (maximum - span, maximum))[1]
        return maximum - span, maximum

    def _heatmap_db(self, image, power: np.ndarray) -> np.ndarray:
        """Reuse the displayed dB buffer only for the same immutable numeric array."""
        previous = self._heatmap_buffers.get(image)
        if previous is not None and previous[0] is power and not power.flags.writeable:
            return previous[1]
        db = 10.0 * np.log10(np.maximum(power, 1e-30))
        self._heatmap_buffers[image] = (power if not power.flags.writeable else None, db)
        return db

    def reset_recording(self, *, preserve_options: bool = False) -> None:
        """Retire plot/cache ownership when the source provider changes."""
        self.retention_notice.hide()
        self._last_request = self._rendered_request = None
        self.detection_stage_export_button.setEnabled(False)
        self.truth_zoom.reset()
        self._heatmap_buffers.clear()
        self._rd_image_geometry = None
        self._live_product_kinds = frozenset()
        self._has_alternate_spectrum = False
        self._variant_box.hide()
        if self._live_catalog:
            self._live_catalog = ()
            self.live_figures_changed.emit()
        self._recorded_capabilities = {}
        self._browser_snapshot_key = None
        self._recorded_marker_styles = None
        self._recorded_scatter_state = None
        self._history_bounds_cache = None
        self._pending_browser_selection = None
        self.result_browser.update_result((), context_key=())
        self.tracking_inspector.selected_id = None
        self.tracking_inspector.update_result(None, 0, "analysis")
        self._inspection_scale.reset()
        self._stage_count_key = None
        self._stage_images.clear()
        self._stage_image_bundle = None
        self._stage_image_render_key = None
        if not preserve_options:
            with (
                QSignalBlocker(self._variant_box),
                QSignalBlocker(self.stage_surface_toggle),
                QSignalBlocker(self._micro_mode_box),
                QSignalBlocker(self._advanced_motion_cb),
                QSignalBlocker(self._show_bin_grid_cb),
                QSignalBlocker(self._profile_full_dynamic_range_cb),
                QSignalBlocker(self._profile_detections_cb),
            ):
                self._variant_box.setCurrentIndex(self._variant_box.findData("primary"))
                self.stage_surface_toggle.setChecked(False)
                self._micro_mode_box.setCurrentIndex(self._micro_mode_box.findData("auto"))
                self._advanced_motion_cb.setChecked(False)
                self._show_bin_grid_cb.setChecked(False)
                self._profile_full_dynamic_range_cb.setChecked(False)
                self._profile_detections_cb.setChecked(False)
            self._reconstruct_spectrum = False
        self.processing_panel.reset(preserve_options=preserve_options)
        self.stage_surface.clear()
        self.association_figure.reset(preserve_options=preserve_options)
        self.evaluation_panel.update_event(None)
        self.run_summary_panel.clear()
        self.set_linked_selection(None, notify=True)
        self.set_result_navigation((), previous=False, following=False)
        self._recording_cursor = None
        self._compact_motion_cursor = None
        if not preserve_options:
            self._initial_view_selected = False
        self._recording_render_key = None
        self._locked_levels.clear()
        self._update_profile_detections()
        self._reset_result_histories()
        self._clear_range_time_history()
        self._expanded_cir_history.clear()
        self._expanded_cir_times_s.clear()
        self._last_expanded_acquisition_id = None
        self._clear_tracking()
        for (tid, column), curve in self._track_estimate_curves.items():
            (self.gt_range_plot, self.gt_speed_plot)[column].removeItem(curve)
        self._track_estimate_curves.clear()
        for (_, column), curve in self._track_estimate_coasts.items():
            (self.gt_range_plot, self.gt_speed_plot)[column].removeItem(curve)
        self._track_estimate_coasts.clear()
        self.environment.clear()
        self._target_names.clear()
        self._clear_recording_images()
        self.set_unavailable("Loading sensing results…", scene_frame_id=None)

    def _clear_recording_images(self) -> None:
        """Never leave a previous recording visible behind an error message."""
        self.collapsed_doppler_image.clear()
        self.collapsed_doppler_mesh.hide()
        self.range_time_raster.clear()
        self.range_time_raster.hide()
        self.range_time_image.hide()
        self.micro_image.clear()

    def _clear_tracking(self) -> None:
        self._track_estimate_present = set()
        for curve in (*self._track_estimate_curves.values(), *self._track_estimate_coasts.values()):
            curve.hide()
        for curves in (
            self._tracking_curves,
            self._tracking_truth_curves,
            self._tracking_coast_curves,
        ):
            for curve in curves.values():
                self.tracking_plot.removeItem(curve)
            curves.clear()

    def _rerender_last_request(self, *_args: Any) -> None:
        """Reapply the last typed update after a display-option change."""
        if getattr(self, "_plots_released", False):
            return
        if getattr(self, "_compact_motion_cursor", None) is not None:
            # A display/detail change needs the current selected result, not
            # the older full request retained as the immutable overview owner.
            self._plots_widget.hide()
            self.view_changed.emit()
            return
        self._micro_gate_box.setEnabled(self._selected_micro_doppler_mode() in {"gated", "tracked"})
        self._recording_render_key = None
        self._apply_view_mode()
        if self._last_request is not None:
            self.update_from_request(self._last_request)
        if self._selected_view_mode() == "micro-doppler":
            self.view_changed.emit()

    def _on_heatmaps_toggled(self, enabled: bool) -> None:
        """Redraw the current result without changing reader demand or selection."""
        QSettings().setValue("visualizer/sensing_heatmaps", enabled)
        self._rerender_last_request()

    def _heatmap_enabled(self, plot: pg.PlotWidget) -> bool:
        """Explicitly pinned maps remain visible independently of dashboard options."""
        return self._show_heatmaps_cb.isChecked() or any(
            getattr(self, key) is plot for key in self._visible_live_figure_keys()
        )

    def _selected_view_mode(self) -> str:
        """Return the dashboard mode selected in the view combo."""
        mode = self._view_mode_box.currentData()
        return str(mode or self._view_mode or "analysis")

    def request_view_mode(self) -> str:
        """Separate deeper product requests from the stable dashboard view token."""
        if not dashboard_is_visible(self) and (self._live_figures or self._live_catalog_requested):
            if self.has_visible_live_figures():
                return "tracking" if self._live_history_visible() else "live"
            return "summary"
        if (
            self._selected_view_mode() in {"analysis", "current"}
            and self._linked_selection is not None
            and self.selection_tools_toggle.isChecked()
        ):
            return "analysis-inspection"
        if self._selected_view_mode() == "diagnostics":
            detail = self._support_box.currentData()
            if detail in {"clutter", "inputs", "frequency"}:
                return f"processing-{detail}" + (
                    "-rebuild" if detail == "clutter" and self._reconstruct_spectrum else ""
                )
        return "analysis" if self._selected_view_mode() == "current" else self._selected_view_mode()

    def _request_spectrum_reconstruction(self) -> None:
        """Explicitly request a spectrum derivation on the existing reader worker."""
        self._reconstruct_spectrum = True
        self.processing_panel.notice.setText(
            "Reconstructing the missing spectrum from retained samples… Detector not rerun."
        )
        self.processing_panel.reconstruct.hide()
        self.view_changed.emit()

    def _set_view_mode(self, mode: Any) -> None:
        """Synchronize the stored view mode, combo box, and active grid layout."""
        if mode == "evaluation":
            mode = "summary"  # Restore older sessions to the overview and its gate controls.
        self._view_mode = str(mode or "analysis")
        index = self._view_mode_box.findData(self._view_mode)
        if index >= 0 and index != self._view_mode_box.currentIndex():
            self._view_mode_box.setCurrentIndex(index)
        self._apply_view_mode()

    def _repopulate_grid(self, mode: str) -> None:
        """Remove all widgets from the grid and re-add for *mode*."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(self._parked_plots)
        for i in range(self._grid.rowCount()):
            self._grid.setRowStretch(i, 0)

        # Current result and existing analysis presets share one detection view.
        layout_spec = self._MODE_LAYOUTS.get(mode, self._MODE_LAYOUTS["analysis"])
        if mode in {"current", "analysis"}:
            layout_spec = [
                spec
                for spec in layout_spec
                if spec[0] not in {"rd_plot", "rp_plot"} or getattr(self, spec[0]).isEnabled()
            ]
            rows = {r: index for index, r in enumerate(sorted({spec[1] for spec in layout_spec}))}
            one_map = sum(spec[0] in {"rd_plot", "rp_plot"} for spec in layout_spec) == 1
            layout_spec = [
                (
                    attr,
                    rows[r],
                    0 if one_map and attr in {"rd_plot", "rp_plot"} else c,
                    rs,
                    2 if one_map and attr in {"rd_plot", "rp_plot"} else cs,
                    stretch,
                )
                for attr, r, c, rs, cs, stretch in layout_spec
            ]
        bundle = self._last_request.detection_stage_bundle if self._last_request else None
        if mode == "diagnostics" and (bundle is None or not bundle.available):
            # Omit duplicate plots for stages that were never recorded. The
            # inspector explains their absence without inventing detector data.
            layout_spec = [
                ("detection_stage_inspector", 0, 0, 1, 2, 0),
            ]
        if mode == "diagnostics" and self._support_box.currentData() in {
            "clutter",
            "inputs",
            "frequency",
        }:
            layout_spec = [("processing_panel", 0, 0, 1, 2, 1)]
        elif (
            mode == "diagnostics"
            and self.stage_surface_toggle.isChecked()
            and bundle is not None
            and bundle.available
        ):
            layout_spec = [
                ("detection_stage_inspector", 0, 0, 1, 2, 0),
                ("stage_surface", 1, 0, 1, 2, 4),
                ("cfar_rp_plot", 2, 0, 1, 2, 1),
            ]
        if mode == "tracking" and getattr(self, "_tracking_spatial_available", True) is False:
            layout_spec = [("gt_range_plot", 0, 0, 1, 1, 3), ("gt_speed_plot", 0, 1, 1, 1, 3)]
        for row, (attr, r, c, rs, cs, stretch) in enumerate(layout_spec):
            if attr == "cfar_rd_plot" and self.stage_surface_toggle.isChecked():
                attr = "stage_surface"
            if (
                mode == "micro-doppler"
                and attr == "collapsed_doppler_plot"
                and self._selected_micro_doppler_mode()
                not in {"auto", "stored", "collapsed", "saved-stft"}
            ):
                attr = "micro_plot"
            widget = getattr(self, attr)
            if attr in self._live_figures:
                widget = self._live_figures[attr].placeholder
            if self._compact:
                r, c, rs, cs = row, 0, 1, 1
            self._grid.addWidget(widget, r, c, rs, cs)
            widget.show()
            self._grid.setRowStretch(r, stretch)
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnStretch(1, 0 if self._compact else 1)
        # A newly visible history may not have received the last marker update.
        self._recorded_scatter_state = None

    def _apply_view_mode(self) -> None:
        """Apply the selected mode's grid layout and mode-gated controls."""
        mode = self._selected_view_mode()
        self._view_mode = mode
        # Presets and programmatic selections must expose an active range gate.
        if self._selected_micro_doppler_mode() not in {"auto", "stored", "collapsed"}:
            with QSignalBlocker(self._advanced_motion_cb):
                self._advanced_motion_cb.setChecked(True)
        self._advanced_motion_cb.setVisible(mode == "micro-doppler")
        self.motion_controls.setVisible(
            mode == "micro-doppler" and self._advanced_motion_cb.isChecked()
        )
        if hasattr(self, "motion_comparison_button"):
            self.motion_comparison_button.setVisible(mode == "micro-doppler")
        for widget in (
            self._micro_label,
            self._micro_mode_box,
            self._micro_gate_label,
            self._micro_gate_box,
            self._motion_range_label,
            self._micro_dynamic_range_box,
        ):
            widget.setVisible(mode == "micro-doppler")
        for widget in (self._rd_range_label, self._rd_dynamic_range_box):
            widget.setVisible(
                mode in {"current", "analysis", "angular", "micro-doppler", "diagnostics"}
            )
        choices = {
            "diagnostics": [(value[0], key) for key, value in SIGNAL_COMPANIONS.items()],
        }
        if self._support_mode != mode:
            if self._support_mode is not None and self._support_box.currentData() is not None:
                self._support_choices[self._support_mode] = self._support_box.currentData()
            with QSignalBlocker(self._support_box):
                self._support_box.clear()
                for label, value in choices.get(mode, []):
                    self._support_box.addItem(label, value)
                selected = self._support_box.findData(self._support_choices.get(mode))
                if selected >= 0:
                    self._support_box.setCurrentIndex(selected)
            self._support_mode = mode
        self.evaluation_panel.set_workspace_active(mode == "evaluation")
        detailed_signal = mode == "diagnostics" and self._support_box.currentData() in {
            "clutter",
            "inputs",
            "frequency",
        }
        detector_view = mode == "diagnostics" and not detailed_signal
        self.detection_stage_summary.setVisible(detector_view)
        self.detection_stage_details_toggle.setVisible(detector_view)
        self.stage_technical.setVisible(
            detector_view and self.detection_stage_details_toggle.isChecked()
        )
        surface = (
            mode == "diagnostics" and self.stage_surface_toggle.isChecked() and not detailed_signal
        )
        self._repopulate_grid(mode)
        self._show_bin_grid_cb.setVisible(
            mode == "diagnostics" and not self.cfar_rd_plot.isHidden()
        )
        self._update_detector_bin_grid()
        self.cfar_rp_plot.setMinimumHeight(140 if surface else 150)
        self._show_cfar_cb.setVisible(mode == "diagnostics" and not detailed_signal and not surface)
        self._show_det_cb.setVisible(mode in {"current", "analysis", "tracking", "angular"})
        self._show_heatmaps_cb.setVisible(mode in {"current", "analysis", "angular"})
        self._show_gt_cb.setVisible(mode in {"current", "analysis", "tracking", "angular"})
        self._doppler_hz_cb.setVisible(mode in {"current", "analysis", "angular", "diagnostics"})
        self._zoom_data_cb.setVisible(
            mode in {"current", "analysis", "tracking", "angular", "micro-doppler", "diagnostics"}
        )
        self._profile_full_dynamic_range_cb.setVisible(
            mode in {"current", "analysis"} and not self.rp_plot.isHidden()
        )
        self._profile_detections_cb.setVisible(mode in {"current", "analysis"})
        self._lock_scale_cb.setVisible(
            mode in {"current", "analysis", "angular", "micro-doppler", "diagnostics"}
        )
        self._support_box.setVisible(mode in choices)
        self.table_button.setVisible(
            mode in {"current", "analysis", "tracking", "angular"} or surface
        )
        self._advanced_widget.show()
        for widget in (
            self.environment.toggle,
            self._environment_opacity_label,
            self.environment.opacity,
            self.environment.note,
        ):
            widget.setVisible(mode == "tracking")
        for widget in (self._variant_box, self._product_note):
            widget.setVisible(
                mode in {"current", "analysis", "angular", "micro-doppler"}
                and getattr(self, "_has_alternate_spectrum", False)
            )
        if detailed_signal or surface:
            self._doppler_hz_cb.hide()
            self._zoom_data_cb.hide()
        if detailed_signal:
            self._lock_scale_cb.setVisible(self._support_box.currentData() == "clutter")
        gated = mode == "micro-doppler" and self._selected_micro_doppler_mode() in {
            "gated",
            "tracked",
        }
        self._micro_gate_label.setVisible(gated)
        self._micro_gate_box.setVisible(gated)
        self.stage_surface_toggle.setText("Return to 2D" if surface else "3D surface")
        self._sync_workspace_navigation()
        self._apply_overlay_visibility()
        self.truth_zoom.sync()

    def wants_recording_history(self) -> bool:
        """Request a bounded overview only for history plots or expanded evaluation."""
        if self._live_history_visible():
            return True
        if (self._live_figures or self._live_catalog_owner) and not dashboard_is_visible(self):
            return False
        return self._selected_view_mode() in {
            "current",
            "analysis",
            "tracking",
            "evaluation",
        }

    def wants_signal_history(self) -> bool:
        """Rich detail rows never own the recording's spectral overview."""
        return False

    def overview_families(self) -> frozenset[str]:
        """Declare compact history only for the currently visible Motion plots."""
        if not dashboard_is_visible(self) or self._selected_view_mode() != "micro-doppler":
            return frozenset()
        from .figure_requirements import figure_requirements

        return figure_requirements(
            "micro-doppler",
            variant=self.product_variant(),
            micro_mode=self._selected_micro_doppler_mode(),
            range_time_visible=dashboard_is_visible(self.range_time_plot),
        ).spectral_families

    def _update_retention_notice(self, request) -> None:
        """Make limited coverage and the relevant configurable allowance visible."""
        notices = []
        history = request.recording_history
        if self.wants_recording_history() and history is not None and history.truncated:
            notices.append(
                f"{history.coverage_text()}. Increase --sensing-history-cache-mb "
                f"(current {history.byte_limit / 1024**2:g} MiB) or --sensing-history-results "
                f"(current {history.result_limit:,})."
            )
        overview = request.motion_overview
        if self._selected_view_mode() == "micro-doppler" and overview is not None:
            for item in overview.families.values():
                if item.coverage.state in {"limited", "error"}:
                    notices.append(
                        f"{item.family}: {item.coverage_text()}. {item.coverage.reason}."
                    )
            if any(item.coverage.state == "limited" for item in overview.families.values()):
                notices.append(
                    f"Combined spectral allowance: {overview.byte_limit / 1024**2:g} MiB; this is separate from detailed-result memory."
                )
        self.retention_notice.setText("\n".join(notices))
        self.retention_notice.setVisible(bool(notices))

    def _on_view_mode_changed(self, *_args: Any) -> None:
        """Refresh layout and plots after the user changes dashboard mode."""
        self._initial_view_selected = True
        self._apply_view_mode()
        self._rerender_last_request()
        self.view_changed.emit()

    def _on_advanced_motion_toggled(self, enabled: bool) -> None:
        """Return to the overview when hiding gate controls, leaving no hidden gate."""
        if enabled:
            self._apply_view_mode()
            return
        with QSignalBlocker(self._micro_mode_box):
            self._micro_mode_box.setCurrentIndex(self._micro_mode_box.findData("auto"))
        self._on_view_mode_changed()

    def _selected_micro_doppler_mode(self) -> str:
        """Return the requested micro-Doppler strategy."""
        mode = self._micro_mode_box.currentData()
        return str(mode or "auto")

    @staticmethod
    def _micro_mode_display_label(mode: str) -> str:
        """Return an honest user-facing label for a micro-Doppler route."""
        return {
            "tracked": _ORACLE_MICRO_LABEL,
            "collapsed": "all ranges",
            "gated": "detection gate",
        }.get(mode, mode)

    def _rd_dynamic_range_db(self) -> float:
        """Return the selected range-Doppler color scale span in dB."""
        return float(self._rd_dynamic_range_box.value())

    def _micro_dynamic_range_db(self) -> float:
        """Return the selected micro-Doppler color scale span in dB."""
        return float(self._micro_dynamic_range_box.value())

    def _micro_doppler_defer_reason(self) -> str | None:
        """Return why expensive micro-Doppler rendering should be skipped now."""
        if self.window().isMinimized():
            return "deferred (window minimized)"
        if not self.isVisible():
            return "deferred (window hidden)"
        if not (self.micro_plot.isVisible() or self.collapsed_doppler_plot.isVisible()):
            return "deferred (plot hidden)"
        return None

    @staticmethod
    def _micro_history_signature(cir_history: Optional[List[np.ndarray]]) -> tuple[Any, ...]:
        """Build a lightweight cache signature for the current CIR history list."""
        if not cir_history:
            return ()
        first = np.asarray(cir_history[0])
        last = np.asarray(cir_history[-1])
        return (
            id(cir_history),
            len(cir_history),
            id(cir_history[0]),
            id(cir_history[-1]),
            tuple(first.shape),
            tuple(last.shape),
        )

    @staticmethod
    def _micro_gt_signature(gt_entries: list[dict[str, Any]]) -> tuple[Any, ...]:
        """Build a cache signature for exact identity/time truth joins."""
        if not gt_entries:
            return ()

        def _entry_signature(entry: dict[str, Any]) -> tuple[Any, ...]:
            """Return exact identity, time, range, and speed from one truth row."""
            target_ids = entry.get("target_ids") or []
            ranges = entry.get("target_ranges") or []
            speeds = entry.get("target_speeds") or []
            reference_time_s = entry.get("reference_time_s")
            range_value = float(ranges[0]) if ranges else None
            speed_value = float(speeds[0]) if speeds else None
            return (
                float(reference_time_s) if reference_time_s is not None else None,
                tuple(str(value) for value in target_ids),
                range_value,
                speed_value,
            )

        return (len(gt_entries), _entry_signature(gt_entries[0]), _entry_signature(gt_entries[-1]))

    def _cache_micro_result(
        self,
        cache_key: tuple[Any, ...],
        result: MicroDopplerResult,
    ) -> None:
        """Store a bounded micro-Doppler result cache for UI rerenders."""
        self._micro_result_cache[cache_key] = result
        while len(self._micro_result_cache) > 8:
            oldest_key = next(iter(self._micro_result_cache))
            self._micro_result_cache.pop(oldest_key, None)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: Any) -> None:
        """Select a combo-box item by user data without raising on missing values."""
        index = combo.findData(value)
        if index >= 0 and index != combo.currentIndex():
            combo.setCurrentIndex(index)

    @staticmethod
    def _replay_stage_event_identity(replay_event: Any) -> tuple[str, str, str, str, str, float]:
        """Return the exact replay identity used to join a stage bundle."""
        key = getattr(replay_event, "key", None)
        recipe = getattr(replay_event, "recipe", None)
        if key is None or recipe is None:
            raise ValueError("Detector-stage replay_event must expose key and recipe")
        values = (
            getattr(key, "clock_id", None),
            getattr(key, "acquisition_id", None),
            getattr(key, "processing_run_id", None),
            getattr(recipe, "processing_recipe_id", None),
            getattr(key, "pair_id", None),
        )
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError("Detector-stage replay_event has an incomplete event identity")
        recipe_run_id = getattr(recipe, "processing_run_id", None)
        if recipe_run_id != values[2]:
            raise ValueError("Detector-stage replay recipe/run identity is contradictory")
        reference_time_s = getattr(key, "reference_time_s", None)
        if (
            not isinstance(reference_time_s, (int, float))
            or isinstance(reference_time_s, bool)
            or not math.isfinite(float(reference_time_s))
        ):
            raise ValueError("Detector-stage replay reference_time_s must be finite")
        clock_id, acquisition_id, run_id, recipe_id, pair_id = values
        assert isinstance(clock_id, str)
        assert isinstance(acquisition_id, str)
        assert isinstance(run_id, str)
        assert isinstance(recipe_id, str)
        assert isinstance(pair_id, str)
        return (
            clock_id,
            acquisition_id,
            run_id,
            recipe_id,
            pair_id,
            float(reference_time_s),
        )

    @staticmethod
    def _bundle_stage_event_identity(
        bundle: DetectionStageBundle,
    ) -> tuple[str, str, str, str, str, float] | None:
        """Return one available bundle's exact occurrence identity."""
        event = bundle.event
        if event is None:
            return None
        return (
            event.clock_id,
            event.acquisition_id,
            event.processing_run_id,
            event.processing_recipe_id,
            event.pair_id,
            event.reference_time_s,
        )

    @classmethod
    def _assert_detection_stage_request_identity(cls, request: SensingUpdateRequest) -> None:
        """Fail before UI mutation when replay and stage occurrence identities differ."""
        replay_event = request.replay_event
        bundle = request.detection_stage_bundle
        if replay_event is None or bundle is None:
            return
        replay_identity = cls._replay_stage_event_identity(replay_event)
        bundle_identity = cls._bundle_stage_event_identity(bundle)
        if bundle.disposition is InspectionDisposition.AVAILABLE and bundle_identity is None:
            raise ValueError("Available detector-stage bundle has no event identity")
        if bundle_identity is not None and bundle_identity != replay_identity:
            fields = ("clock", "acquisition", "run", "recipe", "pair", "reference_time_s")
            mismatched = [
                field
                for field, replay_value, bundle_value in zip(
                    fields,
                    replay_identity,
                    bundle_identity,
                    strict=True,
                )
                if replay_value != bundle_value
            ]
            raise ValueError(
                "Detector-stage replay/bundle event identity mismatch: " + ", ".join(mismatched)
            )

    @staticmethod
    def _stage_scalar_text(value: Any) -> str:
        """Format one exact persisted scalar without display-time rounding."""
        if value is None:
            return "unavailable"
        if isinstance(value, (bool, np.bool_)):
            return "true" if bool(value) else "false"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            parsed = float(value)
            return format(parsed, ".17g") if math.isfinite(parsed) else "nonfinite"
        return str(value)

    @staticmethod
    def _stage_inventory_lines(bundle: DetectionStageBundle | None) -> list[str]:
        """List all ten contracts and their explicit availability state."""
        by_kind = (
            {product.kind: product for product in bundle.products} if bundle is not None else {}
        )
        lines = ["Stage inventory (all Phase-4 contracts):"]
        for ordinal, kind in enumerate(DETECTION_STAGE_KINDS, start=1):
            product = by_kind.get(kind)
            label = _DETECTION_STAGE_LABELS[kind]
            unit = _DETECTION_STAGE_CELL_FIELDS[kind][1]
            if product is None:
                lines.append(
                    f"  {ordinal:02d}. {label} [{kind}] | unavailable | unit={unit} | "
                    "product=unavailable"
                )
                continue
            reason = f" | reason={product.reason}" if product.reason else ""
            lines.append(
                f"  {ordinal:02d}. {label} [{kind}] | {product.disposition.value} | "
                f"representation={product.representation} | unit={unit} | "
                f"product={product.product_id or 'unavailable'}{reason}"
            )
        return lines

    def _set_detection_stage_cell_controls_enabled(self, enabled: bool) -> None:
        """Enable exact cell controls only for a usable joined bundle."""
        surface = (
            self.stage_surface_toggle.isChecked() and self._selected_view_mode() == "diagnostics"
        )
        overview = surface and self.stage_surface.scope.currentText() != "Selected neighborhood"
        previous = getattr(self, "_stage_2d_comparison_scope", None)
        with QSignalBlocker(self.stage_comparison_scope):
            if surface:
                if previous is None:
                    self._stage_2d_comparison_scope = self.stage_comparison_scope.currentIndex()
                self.stage_comparison_scope.setCurrentIndex(1 if overview else 0)
            elif previous is not None:
                self.stage_comparison_scope.setCurrentIndex(previous)
                self._stage_2d_comparison_scope = None
        self.stage_comparison_scope.setEnabled(enabled and not surface)
        self._detection_stage_pair_box.setEnabled(enabled)
        self._detection_stage_doppler_box.setEnabled(enabled and not overview)
        self._detection_stage_range_box.setEnabled(enabled and not overview)
        for box in (self._detection_stage_doppler_box, self._detection_stage_range_box):
            box.setToolTip(
                "Choose Selected neighborhood in the 3D view to inspect a specific cell."
                if overview
                else "Select the exact saved cell for detailed inspection."
            )

    def _initialize_detection_stage_controls(
        self,
        bundle: DetectionStageBundle,
        identity: tuple[str, str, str, str, str, float],
    ) -> None:
        """Initialize a useful saved slice independently of linked object selection."""
        assert bundle.event is not None
        assert bundle.doppler_axis_hz is not None
        assert bundle.range_axis_m is not None
        event_changed = identity != self._detection_stage_event_identity
        previous_doppler = self._detection_stage_doppler_box.value()
        previous_range = self._detection_stage_range_box.value()
        default_doppler, default_range = 0, 0
        if event_changed:
            product = bundle.product("range_doppler_detection_input_power")
            if product.available:
                power = product.values[0]
                usable = np.isfinite(power)
                eligible = bundle.product("range_doppler_cfar_eligible_mask")
                if eligible.available and np.any(usable & eligible.values[0]):
                    usable = usable & eligible.values[0]
                if np.any(usable):
                    default_doppler, default_range = np.unravel_index(
                        np.argmax(np.where(usable, power, -np.inf)), power.shape
                    )

        blockers = [
            QSignalBlocker(self._detection_stage_pair_box),
            QSignalBlocker(self._detection_stage_doppler_box),
            QSignalBlocker(self._detection_stage_range_box),
        ]
        try:
            self._detection_stage_pair_box.clear()
            self._detection_stage_pair_box.addItem(bundle.event.pair_id, bundle.event.pair_id)
            self._detection_stage_doppler_box.setRange(
                0,
                max(0, int(bundle.doppler_axis_hz.size) - 1),
            )
            self._detection_stage_range_box.setRange(
                0,
                max(0, int(bundle.range_axis_m.size) - 1),
            )
            self._detection_stage_doppler_box.setValue(
                default_doppler if event_changed else previous_doppler
            )
            self._detection_stage_range_box.setValue(
                default_range if event_changed else previous_range
            )
        finally:
            blockers.clear()
        self._detection_stage_event_identity = identity
        self._set_detection_stage_cell_controls_enabled(True)

    def _current_detection_stage_selection(
        self,
        bundle: DetectionStageBundle,
    ) -> CellSelection:
        """Return the exact control selection with an unambiguous cluster link."""
        pair_id = self._detection_stage_pair_box.currentData()
        if not isinstance(pair_id, str) or not pair_id:
            assert bundle.event is not None
            pair_id = bundle.event.pair_id
        doppler_bin = self._detection_stage_doppler_box.value()
        range_bin = self._detection_stage_range_box.value()
        cell_links = [
            link
            for link in bundle.detection_links
            if link.doppler_bin == doppler_bin and link.range_bin == range_bin
        ]
        cluster_id = cell_links[0].cluster_id if len(cell_links) == 1 else None
        return CellSelection(
            pair_id=pair_id,
            doppler_bin=doppler_bin,
            range_bin=range_bin,
            cluster_id=cluster_id,
        )

    @staticmethod
    def _stage_value_text(
        kind: str,
        inspection: CellInspection,
        product_available: bool,
    ) -> str:
        """Return one selected persisted value with its exact declared unit."""
        field_name, unit = _DETECTION_STAGE_CELL_FIELDS[kind]
        value = getattr(inspection, field_name)
        if not product_available:
            return f"unavailable {unit}"
        if kind == "range_doppler_snr_db" and value is None:
            return "suppressed by persisted SNR-validity mask"
        if kind == "range_doppler_cluster_labels" and value is None:
            return "-1 1 (background)"
        return f"{_DashboardContent._stage_scalar_text(value)} {unit}"

    def _detection_stage_text(
        self,
        replay_event: Any,
        bundle: DetectionStageBundle,
        selection: CellSelection,
    ) -> str:
        """Build stable, complete text for one request-driven cell inspection."""
        replay_identity = self._replay_stage_event_identity(replay_event)
        clock_id, acquisition_id, run_id, recipe_id, pair_id, reference_time_s = replay_identity
        kind = self._detection_stage_kind
        if not isinstance(kind, str) or kind not in DETECTION_STAGE_KINDS:
            kind = DETECTION_STAGE_KINDS[0]
        product = bundle.product(kind)
        inspection = bundle.inspect(selection)
        lines = [
            "Persisted detector-stage inspection",
            "Source: immutable typed Phase-4 products; no sensing reprocessing performed.",
            f"Bundle disposition: {bundle.disposition.value}"
            + (f" ({bundle.reason})" if bundle.reason else ""),
            "Event identity:",
            f"  clock_id={clock_id}",
            f"  acquisition_id={acquisition_id}",
            f"  processing_run_id={run_id}",
            f"  processing_recipe_id={recipe_id}",
            f"  pair_id={pair_id}",
            f"  reference_time_s={self._stage_scalar_text(reference_time_s)} s",
        ]
        if bundle.time_span is not None:
            lines.extend(
                [
                    "Product time span:",
                    f"  start={self._stage_scalar_text(bundle.time_span.start_time_s)} s",
                    f"  end={self._stage_scalar_text(bundle.time_span.end_time_s)} s",
                    f"  policy={bundle.time_span.reference_time_policy}",
                ]
            )
        lines.extend(
            [
                "Selected cell:",
                f"  pair_id={selection.pair_id}",
                f"  doppler_bin={selection.doppler_bin} -> "
                f"{self._stage_scalar_text(inspection.doppler_hz)} Hz",
                f"  range_bin={selection.range_bin} -> "
                f"{self._stage_scalar_text(inspection.range_m)} m",
                f"  eligibility={inspection.eligibility_disposition}",
                f"  CFAR={inspection.cfar_disposition}",
                f"  NMS={inspection.nms_disposition}",
                f"  cluster={inspection.cluster_disposition}",
                "Selected stage:",
                f"  kind={kind}",
                f"  disposition={product.disposition.value}"
                + (f" ({product.reason})" if product.reason else ""),
                f"  representation={product.representation}",
                f"  product_id={product.product_id or 'unavailable'}",
                f"  value={self._stage_value_text(kind, inspection, product.available)}",
            ]
        )
        link = inspection.detection_link
        if link is None:
            lines.append(f"Detection link: {inspection.detection_disposition}")
        else:
            lines.extend(
                [
                    "Detection link:",
                    f"  acquisition_id={link.acquisition_id}",
                    f"  processing_run_id={link.processing_run_id}",
                    f"  pair_id={link.pair_id}",
                    f"  cluster_id={link.cluster_id}",
                    f"  doppler_bin={link.doppler_bin}",
                    f"  range_bin={link.range_bin}",
                ]
            )
        trace = product.lineage_trace
        lines.append("Ordered lineage (root first):")
        if trace is None:
            lines.append("  unavailable")
        else:
            for ordinal, node in enumerate(trace.nodes):
                sources = ",".join(node.source_product_ids) or "none"
                lines.append(
                    f"  {ordinal:02d}. kind={node.kind or 'external'} "
                    f"product_id={node.product_id} sources={sources} "
                    f"descriptor={str(node.descriptor_available).lower()} "
                    f"loaded={str(node.data_loaded).lower()}"
                )
        lines.extend(self._stage_inventory_lines(bundle))
        return "\n".join(lines)

    def _set_detection_stage_unavailable(
        self,
        *,
        replay_event: Any,
        bundle: DetectionStageBundle | None,
        reason: str,
    ) -> None:
        """Expose missing/invalid request state without consulting frame payloads."""
        self.cfar_pass_borders.clear()
        self.stage_bin_grid.hide()
        self.bin_spacing_label.hide()
        self._set_detection_stage_cell_controls_enabled(False)
        self.stage_surface_toggle.setEnabled(False)
        self.stage_surface.clear(stage_unavailable_text(bundle, reason))
        self.stage_report_input.clear()
        self.stage_report_threshold.clear()
        self.detection_stage_summary.setText(stage_unavailable_text(bundle, reason))
        pair_blocker = QSignalBlocker(self._detection_stage_pair_box)
        try:
            self._detection_stage_pair_box.clear()
        finally:
            del pair_blocker
        self._detection_stage_event_identity = None
        disposition = bundle.disposition.value if bundle is not None else "unavailable"
        lines = [
            f"Persisted detector-stage inspection {disposition}: {reason}",
            "Source: request replay_event + detection_stage_bundle only.",
            "No sensing reprocessing is performed.",
        ]
        if replay_event is not None:
            try:
                identity = self._replay_stage_event_identity(replay_event)
            except ValueError as exc:
                lines.append(f"Replay event invalid: {exc}")
            else:
                lines.extend(
                    [
                        "Replay event:",
                        f"  clock_id={identity[0]}",
                        f"  acquisition_id={identity[1]}",
                        f"  processing_run_id={identity[2]}",
                        f"  processing_recipe_id={identity[3]}",
                        f"  pair_id={identity[4]}",
                        f"  reference_time_s={self._stage_scalar_text(identity[5])} s",
                    ]
                )
        lines.extend(self._stage_inventory_lines(bundle))
        self._detection_stage_status = f"{disposition}: {reason}"
        self.detection_stage_text.setPlainText("\n".join(lines))

    def _update_detection_stage_inspector(self, request: SensingUpdateRequest) -> None:
        """Render only the canonical replay event and stage bundle on ``request``."""
        replay_event = request.replay_event
        bundle = request.detection_stage_bundle
        if replay_event is None:
            self._set_detection_stage_unavailable(
                replay_event=None,
                bundle=bundle,
                reason="canonical replay_event unavailable",
            )
            return
        if bundle is None:
            self._set_detection_stage_unavailable(
                replay_event=replay_event,
                bundle=None,
                reason="detection_stage_bundle unavailable",
            )
            return
        if bundle.disposition is not InspectionDisposition.AVAILABLE:
            self._set_detection_stage_unavailable(
                replay_event=replay_event,
                bundle=bundle,
                reason=bundle.reason or bundle.disposition.value,
            )
            return
        identity = self._replay_stage_event_identity(replay_event)
        if (
            bundle.event is None
            or bundle.doppler_axis_hz is None
            or bundle.range_axis_m is None
            or bundle.doppler_axis_hz.size == 0
            or bundle.range_axis_m.size == 0
        ):
            self._set_detection_stage_unavailable(
                replay_event=replay_event,
                bundle=bundle,
                reason="available bundle lacks exact event or cell coordinates",
            )
            return
        self._initialize_detection_stage_controls(bundle, identity)
        selection = self._current_detection_stage_selection(bundle)
        cell = bundle.inspect(selection)
        kind = self._detection_stage_kind
        values, units = self._stage_image(bundle, kind)
        selected = (
            values[selection.doppler_bin, selection.range_bin] if values is not None else np.nan
        )
        value = f"{selected:.4g} {units}" if np.isfinite(selected) else "not available"
        count_key = tuple((p.product_id, id(p.values)) for p in bundle.products)
        if count_key != self._stage_count_key:
            self._stage_count_key = count_key
            self._stage_counts_text = stage_counts(
                bundle, len(request.display_frame["sensing"].get("detections", []))
            )
        self.detection_stage_summary.setText(
            f"Selected cell · {bundle.range_axis_m[selection.range_bin]:.3g} m · value: {value}\n"
            + cell_explanation(cell).split("\n", 1)[0]
        )
        self.detection_stage_summary.setToolTip(STAGE_HELP[kind] + "\n" + self._stage_counts_text)
        if (
            self.stage_surface_toggle.isChecked()
            and self.stage_surface.scope.currentText() != "Selected neighborhood"
        ):
            self.detection_stage_summary.setText(
                "Overview of published peak cells and their saved input / thresholds. "
                "Choose Selected neighborhood for an individual cell."
            )
        self.detection_stage_text.setPlainText(
            self._detection_stage_text(replay_event, bundle, selection)
        )
        self._update_cfar_rp(request.display_frame["sensing"], request.rp_result, "")
        self._update_cfar_rd(
            request.display_frame["sensing"],
            request.rd_result,
            request.display_frame["sensing"]["config_resolved"]["carrier_freq_hz"],
            "",
        )
        self._update_stage_surface(bundle, selection)
        self._detection_stage_status = (
            f"available: {self._detection_stage_kind} "
            f"cell=({selection.doppler_bin},{selection.range_bin})"
        )
        if self._cell_selection_cleared():
            self._clear_stage_cell_presentation(bundle)
            self._detection_stage_status = "available: no cell selected"

    def _update_detection_stage_inspector_from_controls(self, *_args: Any) -> None:
        """Reinspect persisted values without rerendering or recomputing sensing."""
        if (
            self._last_request is not None
            and self._selected_view_mode() == "diagnostics"
            and self._support_box.currentData() == "cfar"
        ):
            self._update_detection_stage_inspector(self._last_request)

    def _update_stage_surface(self, bundle, selection) -> None:
        input_product = bundle.product("range_doppler_detection_input_power")
        threshold = bundle.product("range_doppler_cfar_threshold_power")
        available = input_product.available and threshold.available
        self.stage_surface_toggle.setEnabled(available)
        if not self.stage_surface_toggle.isChecked() or self._support_box.currentData() != "cfar":
            return
        if not available:
            self.stage_surface.clear(
                "Saved detector input and threshold are required for this surface."
            )
            return
        eligible = bundle.product("range_doppler_cfar_eligible_mask")
        self.stage_surface.set_spectra(
            bundle.range_axis_m,
            bundle.doppler_axis_hz,
            input_product.values[0],
            threshold.values[0],
            selected=(selection.doppler_bin, selection.range_bin),
            labels=("Detector input", "CFAR threshold"),
            threshold=True,
            eligible=eligible.values[0].astype(bool) if eligible.available else None,
            span_db=self._rd_dynamic_range_db(),
            cache_key=tuple((p.product_id, id(p.values)) for p in bundle.products),
            levels=self._stage_power_levels(bundle),
            range_label=range_label(self._last_request.range_mode),
            detections=tuple((link.doppler_bin, link.range_bin) for link in bundle.detection_links),
        )

    def _select_processing_cell(self, r, d, toggle=False) -> None:
        self._select_inspection_coordinate(r, d, model=self.processing_panel.model, toggle=toggle)

    def _select_stage_control_cell(self, *_args) -> None:
        if not self._select_inspection_coordinate(
            self._detection_stage_range_box.value(), self._detection_stage_doppler_box.value()
        ):
            self._update_detection_stage_inspector_from_controls()

    def _select_inspection_coordinate(self, r, d, *, model=None, toggle=False) -> bool:
        from .selection import SensingSelection, inspection_cell, selection_scope

        request = self._last_request
        if request is None:
            return False
        sensing = request.display_frame["sensing"]
        cell = inspection_cell(request, r, d, model=model)
        if cell is not None:
            self.set_linked_selection(
                SensingSelection(
                    selection_scope(sensing),
                    sensing["temporal"]["acquisition"]["acquisition_id"],
                    cell=cell,
                    source=request.source_identity,
                ),
                notify=True,
                toggle=toggle,
            )
            return True
        return False

    def _sync_selected_stage_cell(self, selection) -> None:
        """Link exact compatible coordinates, clearing stale event-owned evidence."""
        from dataclasses import replace

        from .selection import catalog_identity, inspection_cell, matches_detection

        request = self._last_request
        if request is None:
            return
        cell = selection.cell if selection is not None else None
        sensing = request.display_frame["sensing"]
        event = sensing["temporal"]
        valid = (
            selection is not None
            and selection.source == request.source_identity
            and selection.acquisition_id == event["acquisition"]["acquisition_id"]
        )
        if selection is not None and (cell is None or not valid) and selection.track_id is not None:
            # Following a track uses a recorded associated report, never a coast
            # or a truth-nearest cell.
            for index, detection in enumerate(sensing.get("detections", [])):
                if matches_detection(selection, sensing, index, detection):
                    r, d = detection.get("range_bin"), detection.get("doppler_bin")
                    if type(r) is int and type(d) is int:
                        cell = inspection_cell(request, r, d, origin="detection")
                        selection = replace(
                            selection,
                            cell=cell,
                            acquisition_id=event["acquisition"]["acquisition_id"],
                            detection_index=selection.detection_index,
                        )
                        self._linked_selection = selection
                        valid = True
                    break
        valid = (
            valid
            and cell is not None
            and cell.recipe == event["result"]["processing_recipe_id"]
            and cell.variant == request.variant
            and cell.range_mode == request.range_mode
            and cell.catalog_identity == catalog_identity(sensing)
        )
        if not valid:
            cell = None
        self.processing_panel.apply_cell(
            cell, explicit=selection is not None, cleared=self._cell_selection_cleared()
        )
        bundle = request.detection_stage_bundle
        if (
            self._selected_view_mode() != "diagnostics"
            or self._support_box.currentData() != "cfar"
            or bundle is None
            or not bundle.available
        ):
            return
        compatible = (
            cell is not None
            and np.array_equal(bundle.range_axis_m, cell.ranges)
            and (not cell.doppler or np.array_equal(bundle.doppler_axis_hz, cell.doppler))
        )
        if compatible:
            with (
                QSignalBlocker(self._detection_stage_range_box),
                QSignalBlocker(self._detection_stage_doppler_box),
            ):
                self._detection_stage_range_box.setValue(cell.range_bin)
                if cell.doppler_bin is not None:
                    self._detection_stage_doppler_box.setValue(cell.doppler_bin)
            self._update_detection_stage_inspector_from_controls()
        elif selection is not None:
            self.stage_cell.setData([], [])
            for plot in self._stage_maps.values():
                plot.select_cell()
            self.stage_surface.clear(
                "Selected return is not mapped to this event/grid; select a new bin explicitly."
            )
            self.cfar_rp_power.setData([], [])
            self.cfar_rp_thresh.setData([], [])
            self.stage_report_input.clear()
            self.stage_report_threshold.clear()
            self.stage_slice_detections.setData([], [])
            self.detection_stage_summary.setText("No saved cell for this selection")
        elif self._cell_selection_cleared():
            self._clear_stage_cell_presentation(bundle)

    def _clear_stage_cell_presentation(self, bundle) -> None:
        """Keep the saved slice and maps; remove only the linked cell emphasis."""
        self.stage_cell.setData([], [])
        for plot in self._stage_maps.values():
            plot.select_cell()
        if self.stage_comparison_scope.currentData() != "reports":
            self._update_cfar_rp({}, None, "")
        self.stage_slice_line.hide()
        self.stage_range_line.hide()
        self.stage_surface.clear("Select a bin")
        self.detection_stage_summary.setText("No cell selected")
        self.detection_stage_text.setPlainText(
            "No cell selected.\n" + "\n".join(self._stage_inventory_lines(bundle))
        )

    def _stage_image(self, bundle, kind):
        if self._stage_image_bundle is not bundle:
            self._stage_image_bundle = bundle
            self._stage_images.clear()
        if kind not in self._stage_images:
            self._stage_images[kind] = stage_image(bundle, kind)
        return self._stage_images[kind]

    def _stage_power_levels(self, bundle):
        powers = [
            bundle.product(kind).values
            for kind in (
                "range_doppler_detection_input_power",
                "range_doppler_cfar_threshold_power",
            )
        ]
        self._inspection_scale.locked = self._lock_scale_cb.isChecked()
        event = bundle.event
        return self._inspection_scale.levels(
            powers,
            identity=tuple((p.product_id, id(p.values)) for p in bundle.products),
            scope=(
                self._last_request.source_identity,
                event.clock_id,
                event.processing_run_id,
                event.processing_recipe_id,
                event.pair_id,
                self._last_request.range_mode,
            ),
            span=self._rd_dynamic_range_db(),
        )

    def _open_motion_comparison(self) -> None:
        """Open optional raw-observation analysis without altering recorded detection."""
        from .motion_comparison import MotionComparisonDialog

        dialog = MotionComparisonDialog(self)
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.show()

    def _set_detection_stage_export_status(self, message: str) -> None:
        """Append one current export outcome to the request-driven inspection."""
        marker = "\n\nReplay export status:\n"
        current = self.detection_stage_text.toPlainText().split(marker, maxsplit=1)[0]
        self.detection_stage_text.setPlainText(f"{current}{marker}{message}")
        self._detection_stage_status = message

    def _export_detection_stage_replay(self) -> None:
        """Publish replay and selected persisted stage data after an explicit dialog."""
        request = self._last_request
        if request is None or request.replay_event is None:
            self._set_detection_stage_export_status(
                "blocked: canonical replay_event is unavailable"
            )
            return
        bundle = request.detection_stage_bundle
        if bundle is not None and bundle.disposition is InspectionDisposition.INVALID:
            self._set_detection_stage_export_status(
                f"blocked: detector-stage bundle is invalid ({bundle.reason or 'unknown reason'})"
            )
            return
        try:
            self._assert_detection_stage_request_identity(request)
            identity = self._replay_stage_event_identity(request.replay_event)
        except ValueError as exc:
            self._set_detection_stage_export_status(f"blocked: {exc}")
            return

        suggested_name = f"sensing_replay_{identity[1][:8]}_{identity[2][:8]}"
        chosen_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export canonical sensing replay",
            str(Path.cwd() / suggested_name),
            "Replay package directory (*)",
        )
        if not chosen_path:
            return
        destination = Path(chosen_path).expanduser().resolve()
        if destination.exists():
            self._set_detection_stage_export_status(
                f"blocked: destination already exists ({destination})"
            )
            return

        export_bundle: DetectionStageBundle | None = None
        selection: CellSelection | None = None
        selected_kind = self._detection_stage_kind
        if (
            bundle is not None
            and bundle.disposition is InspectionDisposition.AVAILABLE
            and bundle.event is not None
            and bundle.doppler_axis_hz is not None
            and bundle.range_axis_m is not None
            and bundle.doppler_axis_hz.size > 0
            and bundle.range_axis_m.size > 0
            and not self._cell_selection_cleared()
        ):
            export_bundle = bundle
            selection = self._current_detection_stage_selection(bundle)
        try:
            replay_result = export_replay_session(
                request.replay_event,
                destination,
                stage_bundle=export_bundle,
                stage_selection=selection,
                stage_kinds=(
                    (selected_kind,)
                    if export_bundle is not None
                    and isinstance(selected_kind, str)
                    and selected_kind in DETECTION_STAGE_KINDS
                    else None
                ),
            )
        except Exception as exc:  # GUI boundary: make every exporter refusal visible.
            logger.exception("Canonical sensing replay export failed")
            publication = (
                f"; inspect unexpected destination {destination}"
                if destination.exists()
                else "; destination not published"
            )
            self._set_detection_stage_export_status(
                f"blocked: {type(exc).__name__}: {exc}{publication}"
            )
            return

        stage_note = (
            "selected persisted detector cell included"
            if export_bundle is not None
            else (
                "no cell selected; replay-only package"
                if self._cell_selection_cleared()
                else "detector-stage bundle unavailable; replay-only package"
            )
        )
        self._set_detection_stage_export_status(
            f"complete: {replay_result.output_directory} ({stage_note})"
        )

    def _apply_overlay_visibility(self, *_args: Any) -> None:
        """Apply detection, GT, and CFAR visibility toggles to all plot overlays."""
        show_det = self._show_det_cb.isChecked()
        show_gt = self._show_gt_cb.isChecked()
        show_cfar = self._show_cfar_cb.isChecked() and self._selected_view_mode() == "diagnostics"

        self.rd_det_scatter.setVisible(show_det)
        self.rd_gt_scatter.setVisible(show_gt)
        self._update_profile_detections()
        self.rd_cfar_overlay.setVisible(show_cfar and self._heatmap_enabled(self.rd_plot))
        self.cfar_rd_mask.setVisible(show_cfar)
        self.cfar_pass_borders.setVisible(show_cfar)
        self._show_cfar_cb.setEnabled(True)
        for curve in self._tracking_curves.values():
            curve.setVisible(show_det)
        self.range_azimuth_det_scatter.setVisible(show_det)
        self.doppler_azimuth_det_scatter.setVisible(show_det)
        self.angular_spectrum_det_scatter.setVisible(show_det)
        self._gt_range_det_scatter.setVisible(show_det)
        self._gt_speed_det_scatter.setVisible(show_det)
        for curve in self._tracking_truth_curves.values():
            curve.setVisible(show_gt)
        for curve in self._gt_range_curves.values():
            curve.setVisible(show_gt)
        for curve in self._gt_speed_curves.values():
            curve.setVisible(show_gt)
        # Tracking owns its final state/measurement titles and legends below.
        # Replacing them with generic detection labels first forces a layout
        # rebuild even when the final text has not changed between results.
        if not self._tracking_figures_active():
            self._show_det_cb.setText("Detections")
            for plot, scatter in (
                (self.gt_range_plot, self._gt_range_det_scatter),
                (self.gt_speed_plot, self._gt_speed_det_scatter),
            ):
                legend_label = plot.getPlotItem().legend.getLabel(scatter)
                if legend_label is not None and legend_label.text != "Detections":
                    legend_label.setText("Detections")
            if self._target_history:
                truth_suffix = "" if show_gt else " (ground truth hidden)"
                self.gt_range_plot.setTitle(f"Range: detections and ground truth{truth_suffix}")
                self.gt_speed_plot.setTitle(
                    f"Radial velocity: detections and ground truth{truth_suffix}"
                )
        if hasattr(self, "tracking_measurements"):
            self._tracking_plot_labels()
        request = self._last_request
        if request is not None and self._selected_view_mode() in {
            "current",
            "analysis",
            "tracking",
        }:
            history = (
                request.tracking_history
                if self._selected_view_mode() == "tracking" and request.tracking_history is not None
                else request.recording_history
            )
            if history is not None and not history.error:
                self._fit_history_axes(history, self.result_time_s())
        zoom = getattr(self, "truth_zoom", None)
        if zoom is not None:
            zoom.apply()

    def _on_doppler_axis_toggle(self, *_args: Any) -> None:
        """Re-render RD plots when the Hz/m-s toggle changes."""
        self._rerender_last_request()

    @property
    def _use_doppler_hz(self) -> bool:
        """True when the Y-axis should show Doppler in Hz."""
        return self._doppler_hz_cb.isChecked()

    def set_unavailable(self, reason: str, *, scene_frame_id: int | str | None) -> None:
        """Hide the previous result when the current scene has no valid selection.

        Keep timestamped history for the next valid acquisition, but invalidate
        rerender/export ownership so display controls cannot revive a stale result.
        """
        self._last_request = self._rendered_request = None
        self.truth_zoom.sync()
        self._update_live_figure_captions(reason)
        for key in self._live_figures:
            getattr(self, key).hide()
        self.result_browser.update_result((), context_key=())
        self.current_result_summary.update_result({})
        self.tracking_inspector.update_result(None, 0, "analysis")
        self.tracking_inspector.select_track(None)
        self.set_linked_selection(None, notify=True)
        self.processing_panel.set_model(None, str(self._support_box.currentData() or ""))
        self.cell_explanation.update_cell(None, None)
        self.stage_surface.clear(reason)
        self.evaluation_panel.update_event(None)
        self.run_summary_panel.clear()
        self._recording_cursor = None
        self._compact_motion_cursor = None
        self.locate_selection_button.setEnabled(False)
        self._selected_rd_cells.setData([], [])
        self.selection_label.setText("No sensing result selected")
        self._plots_widget.hide()
        self.current_product_notice.hide()
        self._unavailable_label.setText(reason)
        self._unavailable_label.show()
        self.summary_label.setText(reason)
        self._unavailable_label.setText(reason)
        self.provenance_label.setText(f"scene={scene_frame_id} | no selected sensing result")
        self._set_detection_stage_unavailable(replay_event=None, bundle=None, reason=reason)
        self.detection_stage_export_button.setEnabled(False)

    def apply_display_preset(self, preset: dict[str, Any]) -> None:
        """Apply display settings without enabling oracle-assisted processing."""
        rerender_needed = self._last_request is not None

        micro_mode = str(preset.get("micro_doppler_mode", "auto")).strip().lower()
        if micro_mode in {"tracked", "collapsed"}:
            micro_mode = "auto"

        blockers = [
            QSignalBlocker(self._view_mode_box),
            QSignalBlocker(self._micro_mode_box),
            QSignalBlocker(self._advanced_motion_cb),
            QSignalBlocker(self._micro_gate_box),
            QSignalBlocker(self._rd_dynamic_range_box),
            QSignalBlocker(self._micro_dynamic_range_box),
        ]
        try:
            self._set_combo_data(self._view_mode_box, preset.get("view_mode", "analysis"))
            self._set_combo_data(self._micro_mode_box, micro_mode)
            self._advanced_motion_cb.setChecked(micro_mode not in {"auto", "stored", "collapsed"})
            self._micro_gate_box.setValue(int(preset.get("micro_gate_half_width", 2)))
            self._rd_dynamic_range_box.setValue(int(preset.get("rd_dynamic_range_db", 50)))
            self._micro_dynamic_range_box.setValue(int(preset.get("micro_dynamic_range_db", 60)))
        finally:
            blockers.clear()

        self._show_det_cb.setChecked(bool(preset.get("show_detections", True)))
        self._show_gt_cb.setChecked(bool(preset.get("show_gt", False)))
        self._show_cfar_cb.setChecked(bool(preset.get("show_cfar", True)))
        self._zoom_data_cb.setChecked(bool(preset.get("zoom_to_data", False)))
        self._set_view_mode(preset.get("view_mode", "analysis"))
        if rerender_needed:
            self._rerender_last_request()

    def showEvent(self, event: Any) -> None:
        """Refresh deferred plots when the dashboard becomes visible."""
        super().showEvent(event)
        if (
            self._last_request is not None
            and self._last_request is not self._rendered_request
            and not self.isMinimized()
        ):
            self._rerender_last_request()

    def changeEvent(self, event: Any) -> None:
        """Refresh deferred plots when the window leaves a minimized state."""
        super().changeEvent(event)
        if (
            event.type() == QEvent.WindowStateChange
            and self._last_request is not None
            and self._last_request is not self._rendered_request
            and self.isVisible()
            and not self.isMinimized()
        ):
            self._rerender_last_request()

    # ── Public API ──────────────────────────────────────────────────────
    def advance_playback_context(self, request: SensingUpdateRequest) -> bool:
        """Keep unchanged result plots while updating the actual playback binding.

        Only worker-owned cache entries have a token. A new result, view, history,
        tracking file or source uses the full update, as do explicit UI changes.
        """
        previous = self._last_request
        if (
            previous is None
            or getattr(self, "_compact_motion_cursor", None) is not None
            or request.recording_overview_only
            or previous.recording_overview_only
            or previous is not self._rendered_request
            or request.display_data_token is None
            or request.display_data_token is not previous.display_data_token
            or request.prepared_views != previous.prepared_views
            or request.prepared_angular is not previous.prepared_angular
            or request.detection_stage_bundle is not previous.detection_stage_bundle
            or request.processing_inspection is not previous.processing_inspection
            or request.recording_history is not previous.recording_history
            or request.motion_overview is not previous.motion_overview
            or request.tracking_history is not previous.tracking_history
            or request.saved_tracking is not previous.saved_tracking
            or request.saved_tracking_error != previous.saved_tracking_error
            or request.source_identity != previous.source_identity
        ):
            return False
        self._last_request = request
        self._rendered_request = request
        self._update_live_figure_captions()
        self._update_provenance(
            request.display_frame["sensing"],
            request,
            request.variant,
        )
        self._update_summary(
            request.display_frame["sensing"],
            request.variant,
            request.state_step,
            display_step=request.effective_display_step,
            display_from_cache=request.display_from_cache,
        )
        bundle = request.detection_stage_bundle
        if (
            self._selected_view_mode() == "diagnostics"
            and self.detection_stage_details_toggle.isChecked()
            and bundle is not None
            and bundle.available
            and request.replay_event is not None
            and not self._cell_selection_cleared()
        ):
            self.detection_stage_text.setPlainText(
                self._detection_stage_text(
                    request.replay_event, bundle, self._current_detection_stage_selection(bundle)
                )
            )
        return True

    def update_from_request(self, request: SensingUpdateRequest) -> None:
        """Update all plots from a typed sensing update request."""
        request.validate_display_frame()
        if request.recording_overview_only:
            self._update_recording_overview(request)
            return
        self._assert_detection_stage_request_identity(request)
        self._recording_cursor = None
        self._compact_motion_cursor = None
        previous = self._last_request
        if previous is not None and previous.source_identity != request.source_identity:
            self.reset_recording(preserve_options=True)
        self._last_request = request
        self._rendered_request = None
        self._update_retention_notice(request)
        bundle = request.detection_stage_bundle
        self.detection_stage_export_button.setEnabled(
            request.replay_event is not None
            and (bundle is None or bundle.disposition is not InspectionDisposition.INVALID)
        )
        if not dashboard_is_visible(self) and (self._live_figures or self._live_catalog_requested):
            self._update_product_options(request.display_frame["sensing"], request)
            self._publish_live_figure_catalog()
            self._live_catalog_requested = False
            for key in self._visible_live_figure_keys():
                getattr(self, key).show()
            self._update_live_figures(request)
            self.truth_zoom.apply()
            return
        self._unavailable_label.hide()
        self._plots_widget.show()
        frame = request.display_frame
        state_frame = request.state_frame
        step = request.state_step

        sensing_block = frame.get("sensing") or {}
        sensing_cfg = sensing_block.get("config_resolved")
        if not isinstance(sensing_cfg, Mapping):
            raise ValueError("Native sensing-v4 dashboard input requires config_resolved")
        required_runtime_fields = ("carrier_freq_hz", "prf_hz", "sampling_frequency")
        missing_runtime_fields = [key for key in required_runtime_fields if key not in sensing_cfg]
        if missing_runtime_fields:
            raise ValueError(
                "Native sensing-v4 config_resolved is missing exact dashboard fields: "
                + ", ".join(missing_runtime_fields)
            )

        frequency_index = self._support_box.findData("frequency")
        if frequency_index >= 0:
            frequency_item = self._support_box.model().item(frequency_index)
            from .recorded_products import product_kinds

            kinds = product_kinds(sensing_block, request.available_product_kinds)
            retained = bool(kinds & {"baseband_frequency_observation", "baseband_frequency_power"})
            frequency_item.setEnabled(retained)
            frequency_item.setToolTip(
                "" if retained else "Frequency observations or display power were not recorded."
            )
        from .recorded_products import product_kinds

        kinds = product_kinds(sensing_block, request.available_product_kinds)
        frequency_choice = self.input_kind_box.model().item(
            self.input_kind_box.findData("frequency")
        )
        frequency_choice.setEnabled(
            bool(kinds & {"baseband_frequency_observation", "baseband_frequency_power"})
        )
        frequency_choice.setToolTip(
            "" if frequency_choice.isEnabled() else "No saved frequency response."
        )

        carrier_freq_hz = float(sensing_cfg["carrier_freq_hz"])
        prf_hz = float(sensing_cfg["prf_hz"])
        sampling_frequency = float(sensing_cfg["sampling_frequency"])
        if request.recording_managed and not self._initial_view_selected:
            self._initial_view_selected = True
            with QSignalBlocker(self._view_mode_box):
                self._set_view_mode("summary")
        kinds = request.available_product_kinds
        if kinds is None:
            kinds = {d.get("kind") for d in sensing_block.get("products", {}).values()}
        item = self._view_mode_box.model().item(self._view_mode_box.findData("angular"))
        has_angle = "range_doppler_angle" in kinds
        item.setEnabled(has_angle)
        item.setToolTip("" if has_angle else "This acquisition has no saved angular product.")
        self._update_product_options(sensing_block, request)
        if (
            self._selected_view_mode() == "tracking"
            and not self._view_mode_box.model()
            .item(self._view_mode_box.findData("tracking"))
            .isEnabled()
        ):
            with QSignalBlocker(self._view_mode_box):
                self._set_view_mode("analysis")
            self.view_changed.emit()
        mode = self._selected_view_mode()
        if mode == "diagnostics":
            available = bool(
                request.detection_stage_bundle and request.detection_stage_bundle.available
            )
            if available != getattr(self, "_diagnostic_stage_available", None):
                self._diagnostic_stage_available = available
                self._apply_view_mode()
            detail = self._support_box.currentData()
            if detail == "cfar":
                self._update_detection_stage_inspector(request)
            if detail in {"clutter", "inputs", "frequency"}:
                self.processing_panel._span = self._rd_dynamic_range_db()
                self.processing_panel.scale.locked = self._lock_scale_cb.isChecked()
                inspection = request.processing_inspection
                self.processing_panel.set_model(
                    inspection if inspection and inspection.mode == detail else None, detail
                )
                self.processing_panel.render()
        self.evaluation_panel.update_event(request.evaluation, request.recording_history)
        if request.recording_managed:
            sampled = (sensing_cfg.get("acquisition") or {}).get("sample_period_s") is not None
            motion_item = self._view_mode_box.model().item(
                self._view_mode_box.findData("micro-doppler")
            )
            motion_item.setEnabled(
                sampled
                and (
                    request.rd_result is not None
                    or bool(kinds & {"range_doppler", "micro_doppler"})
                )
            )
            motion_item.setToolTip(
                "" if sampled else "No sampled acquisition clock in this result."
            )
        self._update_recorded_capabilities(sensing_block, request)
        self._publish_live_figure_catalog()

        # Determine actual variant label
        actual_variant = request.variant
        if request.rp_result is not None:
            meta = getattr(request.rp_result, "metadata", None) or {}
            actual_variant = meta.get("variant", request.variant)
        elif request.rd_result is not None:
            meta = getattr(request.rd_result, "metadata", None) or {}
            actual_variant = meta.get("variant", request.variant)

        variant_suffix = " (alternate saved)" if actual_variant == "raw" else ""

        self._update_target_names(state_frame)
        self._update_target_names(frame)

        # Truth-assisted products consume the current result's canonical truth,
        # never the previous UI update's history tail.
        if mode in {"current", "analysis", "angular", "diagnostics", "tracking"} and not (
            mode == "tracking" and request.recording_managed
        ):
            render_history = not (
                mode in {"current", "analysis"} and request.recording_managed
            ) and (
                mode not in {"angular", "diagnostics"}
                or dashboard_is_visible(self.gt_range_plot)
                or dashboard_is_visible(self.gt_speed_plot)
            )
            self._update_ground_truth(frame, render=render_history)
        elif "rd_plot" in self._visible_live_figure_keys():
            # A pinned map keeps this result's truth even while the main
            # dashboard shows a metadata-only or history workspace.
            self._update_ground_truth(frame, render=False)

        # The compact event timeline consumes only the canonical temporal
        # contract and explicit display provenance physical time.
        if mode == "diagnostics":
            self._update_acquisition_timeline(sensing_block, frame)

        # 1. Range Profile
        if self.rp_plot.isVisible() or not self.rp_plot.isEnabled():
            self._update_range_profile(
                request.rp_result, variant_suffix, rd_result=request.rd_result
            )

        # 1b. Range-time waterfall
        if not request.recording_managed and self.range_time_plot.isVisible():
            self._update_range_time_waterfall(
                request.rp_result,
                sensing_block=sensing_block,
                variant_suffix=variant_suffix,
            )

        # 2. Range-Velocity Map
        if self.rd_plot.isVisible() or not self.rd_plot.isEnabled():
            self._update_range_velocity(request.rd_result, frame, variant_suffix, carrier_freq_hz)

        # 5. Angular products
        if mode == "angular" or self._LIVE_ANGULAR.intersection(self._visible_live_figure_keys()):
            descriptors = sensing_block.get("products") or {}
            pending = (
                request.selective_products
                and request.prepared_angular is None
                and "angular" not in request.prepared_views
                and any(
                    d.get("kind") == "range_doppler_angle" and key not in sensing_block
                    for key, d in descriptors.items()
                )
            )
            if pending:
                self._clear_angular_products("Loading saved angular product…")
            else:
                self._update_angular_products(
                    sensing_block,
                    "primary" if actual_variant in {"clean", "primary"} else actual_variant,
                    variant_suffix,
                    carrier_freq_hz,
                )

        # Compact metadata is available in every workspace, without loading maps.
        self._update_result_browser(request)

        # 7. Collapsed micro-Doppler spectrogram
        saved_stft_selected = (
            mode == "micro-doppler" and self._selected_micro_doppler_mode() == "saved-stft"
        )
        if (request.recording_managed or saved_stft_selected) and mode in {
            "current",
            "analysis",
            "micro-doppler",
            "tracking",
        }:
            self._render_recording(request, apply_overlays=False)
            if mode == "micro-doppler" and self.micro_plot.isVisible():
                self._render_saved_gate(request)
        needs_samples = not saved_stft_selected and (
            (self.micro_plot.isVisible() and not request.recording_managed)
            or (not request.recording_managed and self.collapsed_doppler_plot.isVisible())
        )
        if needs_samples:
            cir_history, cir_history_times_s, cir_history_error = (
                self._resolve_timestamped_fast_time_history(sensing_block, request)
            )
        if (
            not request.recording_managed
            and not saved_stft_selected
            and self.collapsed_doppler_plot.isVisible()
        ):
            self._update_collapsed_doppler(
                sensing_block,
                prf_hz,
                sampling_frequency,
                request.range_mode,
                cir_history=cir_history,
                cir_history_times_s=cir_history_times_s,
                history_error=cir_history_error,
                range_axis_m=self._fast_time_range_axis_m,
            )

        # 8. Micro-Doppler spectrogram (only rendered in Micro-Doppler view)
        if (
            self.micro_plot.isVisible()
            and not request.recording_managed
            and not saved_stft_selected
        ):
            self._update_micro_doppler(
                sensing_block,
                prf_hz,
                sampling_frequency,
                request.range_mode,
                step=step,
                cir_history=cir_history,
                cir_history_times_s=cir_history_times_s,
                history_error=cir_history_error,
                range_axis_m=self._fast_time_range_axis_m,
            )

        # Summary + provenance
        self._update_provenance(
            sensing_block,
            request,
            actual_variant,
        )
        self._update_summary(
            sensing_block,
            actual_variant,
            step,
            display_step=request.effective_display_step,
            display_from_cache=request.display_from_cache,
        )
        self._apply_overlay_visibility()
        self.apply_linked_selection()
        self._update_live_figure_captions()
        if self._live_history_visible() and mode not in {"current", "analysis", "tracking"}:
            history = request.tracking_history or request.recording_history
            self._render_tracking_history(history, self.result_time_s())
        self._sync_selected_stage_cell(self._linked_selection)
        self.association_figure.setVisible(mode == "tracking" and request.saved_tracking is None)
        if mode == "tracking" and request.saved_tracking is None:
            from shared.sensing.plotting.inspection import tracking_decisions_figure
            from shared.sensing.tracking_inspection import tracking_decisions

            sensing = request.display_frame["sensing"]
            self.association_figure.set_source(
                lambda: tracking_decisions_figure(*tracking_decisions(sensing)),
                (
                    tracking_decision_text(sensing)
                    if self.association_figure.toggle.isChecked()
                    else ""
                ),
            )
        self.tracking_inspector.update_result(
            request.saved_tracking,
            self.result_time_s(),
            mode,
            request.saved_tracking_error,
            sensing=sensing_block,
            snapshot=self._browser_snapshot,
        )
        if mode == "tracking" and self.tracking_inspector.selected_id is not None:
            # Following an existing track during playback is not a new user selection.
            with QSignalBlocker(self):
                self.select_track(self.tracking_inspector.selected_id)
        self._sync_workspace_navigation()
        self._refresh_current_summary()
        self._finish_browser_selection(sensing_block)
        self.truth_zoom.apply()
        if dashboard_is_visible(self):
            self._rendered_request = request
        self.range_shell_changed.emit()

    def _update_recording_overview(self, request: SensingUpdateRequest) -> None:
        """Display whole-recording figures without claiming a current acquisition."""
        reason = f"No sensing result is bound to scene {request.state_step}."
        self.set_unavailable(reason, scene_frame_id=request.state_step)
        mode = self._selected_view_mode()
        if mode not in {"micro-doppler", "summary"}:
            return
        self._last_request = request
        self._unavailable_label.hide()
        self._update_retention_notice(request)
        self._plots_widget.show()
        if mode == "micro-doppler":
            sensing = request.display_frame["sensing"]
            self._update_product_options(sensing, request)
            self._update_recorded_capabilities(sensing, request)
            # Range gates and selection tools describe one selected event. During
            # warm-up only the whole-recording products have a meaningful scope.
            for gate_mode in ("tracked", "gated", "fixed-bin"):
                item = self._micro_mode_box.model().item(self._micro_mode_box.findData(gate_mode))
                item.setEnabled(False)
                item.setToolTip("Select a saved sensing result to inspect its range gate.")
        self._apply_view_mode()
        if mode == "micro-doppler":
            self._render_recording(request, apply_overlays=False)
        self.collapsed_doppler_cursor.hide()
        self.provenance_label.setText(
            f"Recording overview · no sensing result at scene {request.state_step}"
        )
        self.provenance_label.setToolTip(
            reason + " Select a saved result to inspect current-result details."
        )
        self._sync_browser_visibility()
        self._rendered_request = request

    def select_track(self, track_id):
        """Keep saved world-track identities separate from native detector IDs."""
        track_id = int(track_id)
        if self._last_request is not None and self._last_request.saved_tracking is not None:
            self._linked_selection = None
            self.tracking_inspector.select_track(track_id)
            self.result_browser.select_track(track_id)
            self.selection_label.setText(f"Saved 3D track {track_id}")
            self._sync_selection_location_button()
            self._highlight_track_selection()
            from .fusion_inspection import FusionSelection

            reader = self._last_request.saved_tracking.reader
            index = reader.epoch_index(self.result_time_s())
            if index is not None:
                self.selection_changed.emit(FusionSelection(reader.inspection_id, track_id, index))
            return
        super().select_track(track_id)
        self.result_browser.select_track(track_id)
        self.tracking_inspector.select_track(track_id)
        self._sync_selection_location_button()

    def _select_saved_evidence(self, native):
        request = self._last_request
        if request is None:
            return
        sensing = request.display_frame["sensing"]
        index = native.get("detection_index")
        if (
            native.get("acquisition_id") == sensing["temporal"]["acquisition"]["acquisition_id"]
            and native.get("detection_collection", "detections")
            == sensing.get("_display_detection_collection", "detections")
            and isinstance(index, int)
            and 0 <= index < len(sensing.get("detections", []))
        ):
            self._set_view_mode("analysis")
            self.select_detection_index(index)

    # ── Target names ────────────────────────────────────────────────────
    def _render_recording(
        self, request: SensingUpdateRequest, *, apply_overlays: bool = True
    ) -> None:
        """Render saved history once; seeking only moves the physical-time cursor."""
        history = request.recording_history
        now = self.result_time_s()
        mode = self._selected_view_mode()
        if mode in {"current", "analysis", "tracking"}:
            if mode == "tracking" and request.tracking_history is not None:
                history = request.tracking_history
            self._render_tracking_history(history, now, apply_overlays=apply_overlays)
            return
        if request.motion_overview is not None:
            self._render_compact_motion(request)
            return
        self.collapsed_doppler_cursor.setPos(now)
        self.collapsed_doppler_cursor.setVisible(not request.recording_overview_only)
        key = (
            id(history),
            request.variant,
            self._selected_micro_doppler_mode(),
            self._micro_dynamic_range_db(),
            self._rd_dynamic_range_db(),
            self._zoom_data_cb.isChecked(),
        )
        if history is not None and key == self._recording_render_key:
            return
        self._clear_recording_images()
        if (
            request.recording_managed
            and history is None
            and (
                request.recording_overview_only
                or self._selected_micro_doppler_mode() != "saved-stft"
            )
        ):
            self.collapsed_doppler_image.clear()
            self.collapsed_doppler_mesh.hide()
            self.collapsed_doppler_plot.setTitle("Loading saved spectral history…")
            return
        if (
            request.display_frame["sensing"]["config_resolved"]["acquisition"]["sample_period_s"]
            is None
        ):
            self.collapsed_doppler_plot.setTitle("Motion requires a sampled acquisition clock")
            return
        if history is not None and history.error:
            self.collapsed_doppler_image.clear()
            self.collapsed_doppler_plot.setTitle("Recording unavailable")
            self.collapsed_doppler_plot.setToolTip(history.error)
            return
        records = history.records if history is not None else ()
        spectra = [r for r in records if r.spectrum is not None]
        saved_stft = self._selected_micro_doppler_mode() == "saved-stft"
        widths = None
        if saved_stft:
            from .recorded_products import (
                micro_doppler_history,
                saved_micro_doppler_unavailable_reason,
            )

            try:
                micro = micro_doppler_history(
                    records, request.display_frame["sensing"], request.variant
                )
            except (KeyError, ValueError) as exc:
                self.collapsed_doppler_plot.setTitle("Saved STFT unavailable")
                self.collapsed_doppler_plot.setToolTip(str(exc))
                return
            if micro is None:
                reason = saved_micro_doppler_unavailable_reason(request.display_frame["sensing"])
                self.collapsed_doppler_plot.setTitle(reason)
                self.collapsed_doppler_plot.setToolTip(reason)
                self._micro_status = reason
                self._micro_mode_used = "unavailable"
                return
            times, axis, power = micro["times_s"], micro["doppler_hz"], micro["power"]
            widths = micro["widths_s"]
            width = widths[0]
            label = "Micro-Doppler · saved STFT · fixed processing"
            self.collapsed_doppler_plot.setToolTip(
                f"Saved STFT recipe: {micro['recipe']}. These spectra were computed from complex samples during generation; playback does not rerun the transform."
            )
        elif not spectra:
            rd = request.rd_result
            if rd is None:
                self.collapsed_doppler_image.clear()
                self.collapsed_doppler_plot.setTitle("No saved RD spectrum")
                return
            times = np.array([now])
            axis = np.asarray(rd.doppler_axis_hz)
            power = np.sum(rd.power_linear, axis=-1)[None, :]
            cfg = request.display_frame["sensing"]["config_resolved"]["acquisition"]
            width = float(cfg["sample_period_s"]) * int(cfg["cpi_hop_samples"])
            label = "Saved spectrum · loading recording…" if history is None else "Saved spectrum"
        else:
            axis = spectra[0].doppler_hz
            if any(not np.array_equal(axis, r.doppler_hz) for r in spectra):
                self.collapsed_doppler_image.clear()
                self.collapsed_doppler_plot.setTitle("Incompatible Doppler grids in this recording")
                return
            times = np.array([r.time_s for r in spectra])
            power = np.stack([r.spectrum for r in spectra])
            width = spectra[0].cadence_s
            if any(r.cadence_s is None for r in spectra):
                self.collapsed_doppler_plot.setTitle("Recording contains unsampled acquisitions")
                return
            label = "Doppler over time · saved RD power, all ranges"
            if history.truncated:
                label += f" · last {len(spectra)} results"
        if not saved_stft:
            self.collapsed_doppler_plot.setToolTip(
                "Saved RD power summed across range. Motion content depends on the acquisition: "
                "coherent one-solve CPIs freeze pose and evolve bulk velocity. Changing poses "
                "between those CPIs does not simulate independent limb velocities."
            )
        db = 10 * np.log10(np.maximum(power, 1e-30))
        levels = self._levels(
            "recording-micro" if records else "current-micro", db, self._micro_dynamic_range_db()
        )
        # Uniform grids use the fast image path. Missing results get explicit
        # empty cells rather than a spectral column stretched across the gap.
        uniform = (
            times.size < 2 or np.allclose(np.diff(times), width, rtol=1e-7, atol=1e-10)
        ) and (widths is None or np.allclose(widths, width))
        if uniform:
            self.collapsed_doppler_mesh.hide()
            self.collapsed_doppler_image.show()
            self.collapsed_doppler_image.setImage(db, autoLevels=False)
            self.collapsed_doppler_image.setLevels(levels)
            self.collapsed_doppler_image.setRect(self._axis_rect(times, axis, x_single_width=width))
        else:
            from .dashboard_data import history_cells

            try:
                edges, columns = history_cells(
                    times,
                    widths if widths is not None else np.array([r.cadence_s for r in spectra]),
                    db,
                )
            except ValueError as exc:
                self.collapsed_doppler_plot.setTitle(str(exc))
                return
            x, y = np.meshgrid(edges, self._axis_edges(axis, axis_name="Doppler"), indexing="ij")
            self.collapsed_doppler_image.hide()
            self.collapsed_doppler_mesh.setData(x, y, columns, autoLevels=False)
            self.collapsed_doppler_mesh.setLevels(levels)
            self.collapsed_doppler_mesh.show()
        self.micro_colorbar.setLevels(values=levels)
        self.collapsed_doppler_plot.setTitle(label)
        self.collapsed_doppler_plot.setLabel(
            "bottom", "STFT window center" if saved_stft else "Result time", units="s"
        )
        self._micro_status = (
            "saved STFT power; fixed recorded recipe"
            if saved_stft
            else "saved RD power summed across range; original processing retained"
        )
        self._micro_mode_used = "saved-stft" if saved_stft else "collapsed"
        profiles = [r for r in records if r.range_power is not None]
        if profiles and self.range_time_plot.isVisible():
            axis_r = profiles[0].range_m
            if all(np.array_equal(axis_r, r.range_m) for r in profiles):
                t = np.array([r.time_s for r in profiles])
                values = 10 * np.log10(
                    np.maximum(np.stack([r.range_power for r in profiles]), 1e-30)
                )
                if t.size == 1 or np.all(np.diff(t) > 0):
                    from .dashboard_data import history_cells

                    try:
                        edges, values = history_cells(
                            t, np.array([r.cadence_s for r in profiles]), values
                        )
                    except ValueError as exc:
                        self.range_time_plot.setTitle(str(exc))
                        return
                    self._set_range_time_image(
                        edges,
                        self._cell_edges(axis_r, axis_name="range"),
                        values,
                        self._levels("recording-range", values, self._rd_dynamic_range_db()),
                    )
                    self._fit_range_time_axis(values, axis_r)
                    self.range_time_plot.setTitle("Range–time · saved profiles")
                    self.range_time_plot.setLabel(
                        "left",
                        "Range" if request.range_mode == "monostatic" else "Path length",
                        units="m",
                    )
        if history is not None:
            self._recording_render_key = key

    def _render_compact_motion(self, request) -> None:
        """Draw exact independent families, without borrowing detailed rows."""
        from .dashboard_data import history_cells

        overview = request.motion_overview
        mode = self._selected_micro_doppler_mode()
        self.collapsed_doppler_cursor.setPos(self.result_time_s())
        self.collapsed_doppler_cursor.setVisible(not request.recording_overview_only)
        key = (
            id(overview),
            mode,
            self._micro_dynamic_range_db(),
            self._rd_dynamic_range_db(),
            self._zoom_data_cb.isChecked(),
        )
        if key == self._recording_render_key:
            return
        self._clear_recording_images()
        family = "saved-stft" if mode == "saved-stft" else "summed-rd"
        item = overview.families.get(family)
        if mode in {"auto", "stored", "collapsed", "saved-stft"}:
            if item is None:
                self.collapsed_doppler_plot.setTitle("Loading saved spectral history…")
            elif not item.times_s.size:
                self.collapsed_doppler_plot.setTitle(
                    item.coverage.reason or "Saved spectral history unavailable"
                )
            else:
                db = 10 * np.log10(np.maximum(item.power, 1e-30))
                levels = self._levels("recording-micro", db, self._micro_dynamic_range_db())
                try:
                    edges, columns = history_cells(item.times_s, item.widths_s, db)
                    uniform = np.allclose(item.widths_s, item.widths_s[0]) and (
                        item.times_s.size < 2
                        or np.allclose(
                            np.diff(item.times_s), item.widths_s[0], rtol=1e-7, atol=1e-10
                        )
                    )
                    if uniform:
                        self.collapsed_doppler_mesh.hide()
                        self.collapsed_doppler_image.show()
                        self.collapsed_doppler_image.setImage(db, autoLevels=False)
                        self.collapsed_doppler_image.setLevels(levels)
                        self.collapsed_doppler_image.setRect(
                            self._axis_rect(
                                item.times_s, item.axis, x_single_width=item.widths_s[0]
                            )
                        )
                    else:
                        x, y = np.meshgrid(
                            edges, self._axis_edges(item.axis, axis_name="Doppler"), indexing="ij"
                        )
                        self.collapsed_doppler_image.hide()
                        self.collapsed_doppler_mesh.setData(x, y, columns, autoLevels=False)
                        self.collapsed_doppler_mesh.setLevels(levels)
                        self.collapsed_doppler_mesh.show()
                    self.micro_colorbar.setLevels(values=levels)
                    label = (
                        "Micro-Doppler · saved STFT"
                        if family == "saved-stft"
                        else "Doppler over time · saved RD power, all ranges"
                    )
                    self.collapsed_doppler_plot.setTitle(f"{label} · {item.coverage_text()}")
                    self.collapsed_doppler_plot.setLabel(
                        "bottom",
                        "STFT window center" if family == "saved-stft" else "Result time",
                        units="s",
                    )
                    self.collapsed_doppler_plot.setToolTip(
                        f"Exact saved {item.variant} product; recipe: {item.recipe}. No reprocessing. {item.coverage.reason}"
                    )
                    self._micro_status = (
                        "saved STFT power; fixed recorded recipe"
                        if family == "saved-stft"
                        else "saved RD power summed across range; original processing retained"
                    )
                    self._micro_mode_used = "saved-stft" if family == "saved-stft" else "collapsed"
                except ValueError as exc:
                    self.collapsed_doppler_plot.setTitle(str(exc))
        profile = overview.families.get("range-profile")
        if profile is not None and dashboard_is_visible(self.range_time_plot):
            if not profile.times_s.size:
                self.range_time_plot.setTitle(
                    profile.coverage.reason or "Saved range profiles unavailable"
                )
            else:
                values = 10 * np.log10(np.maximum(profile.power, 1e-30))
                try:
                    edges, values = history_cells(profile.times_s, profile.widths_s, values)
                    self._set_range_time_image(
                        edges,
                        self._cell_edges(profile.axis, axis_name="range"),
                        values,
                        self._levels("recording-range", values, self._rd_dynamic_range_db()),
                    )
                    self._fit_range_time_axis(values, profile.axis)
                    self.range_time_plot.setTitle(
                        f"Range–time · saved profiles · {profile.coverage_text()}"
                    )
                    self.range_time_plot.setLabel(
                        "left",
                        "Range" if request.range_mode == "monostatic" else "Path length",
                        units="m",
                    )
                    self.range_time_plot.setToolTip(profile.coverage.reason)
                except ValueError as exc:
                    self.range_time_plot.setTitle(str(exc))
        self._recording_render_key = key

    def _render_saved_gate(self, request: SensingUpdateRequest) -> None:
        """Inspect a range selection in the stored spectrum, with no new DSP."""
        rd = request.rd_result
        mode = self._selected_micro_doppler_mode()
        sensing = request.display_frame["sensing"]
        if rd is None:
            self._clear_micro_doppler("No saved RD product for a range gate", "unavailable")
            return
        ranges = np.asarray(rd.range_axis_m)
        if mode == "tracked":
            gt = self._extract_bound_gt_entry(sensing)
            if gt is None or len(gt[1]["target_ranges"]) != 1:
                self._clear_micro_doppler(
                    "Oracle gate requires one bound ground-truth target", "oracle gate unavailable"
                )
                return
            center = int(np.argmin(np.abs(ranges - gt[1]["target_ranges"][0])))
            label = "Oracle-assisted GT range gate"
        else:
            try:
                center = self._resolve_micro_range_bin(sensing, ranges, ranges.size)
            except ValueError as exc:
                self._clear_micro_doppler("No detected range for this gate", str(exc))
                return
            if center is None:
                self._clear_micro_doppler("No detected range for this gate", "unavailable")
                return
            label = "Detected range bin" if mode == "fixed-bin" else "Detection range gate"
        half_width = 0 if mode == "fixed-bin" else self._micro_gate_box.value()
        lo, hi = max(0, center - half_width), min(ranges.size, center + half_width + 1)
        power = np.sum(rd.power_linear[:, lo:hi], axis=1)[None, :]
        db = 10 * np.log10(np.maximum(power, 1e-30))
        cfg = sensing["config_resolved"]["acquisition"]
        width = float(cfg["sample_period_s"]) * int(cfg["cpi_hop_samples"])
        self.micro_image.setImage(db, autoLevels=False)
        self.micro_image.setLevels(
            self._levels("current-gated", db, self._micro_dynamic_range_db())
        )
        self.micro_image.setRect(
            self._axis_rect(
                np.array([self.result_time_s()]), rd.doppler_axis_hz, x_single_width=width
            )
        )
        self.micro_plot.setTitle(
            f"{label}: {ranges[lo]:.2f}–{ranges[hi - 1]:.2f} m · saved RD window"
        )
        self._micro_status = f"{label}; saved RD power, no reprocessing"
        self._micro_mode_used = mode

    def _render_tracking_xy(self, history: Any, now: float) -> None:
        """Project saved world trajectories; retain observed/coasting status."""
        self._draw_track_paths(history, now)

    def _render_tracking_history(
        self, history: Any, now: float, *, apply_overlays: bool = True
    ) -> None:
        """Show saved histories through the selected result time, without future points."""
        if history is None or history.error:
            self._clear_tracking()
            self._clear_gt_plots(
                "loading saved history…" if history is None else "history unavailable"
            )
            self.tracking_plot.setTitle(
                "Loading saved tracks…" if history is None else "Track history unavailable"
            )
            return
        if self._tracking_figures_active() and self._live_plot_needs_update(self.tracking_plot):
            self._render_tracking_xy(history, now)
        if self._live_only() and not any(
            self._live_plot_needs_update(plot) for plot in (self.gt_range_plot, self.gt_speed_plot)
        ):
            return
        prepared = history.tracking_plot_data
        target_history = {}
        for tid, values in prepared.ranges.items():
            times, points = values.through(now)
            if len(times):
                target_history[tid] = {
                    "times_s": times,
                    "range": points[:, 0],
                    "speed": points[:, 1],
                }
        times, points = prepared.detections.through(now)
        self._target_history = target_history
        self._detection_history = {"times_s": times, "range": points[:, 0], "speed": points[:, 1]}
        self._detection_history_colors = list(prepared.detection_colors[: len(times)])
        if self._recorded_marker_styles is None or self._recorded_marker_styles[0] is not prepared:
            styles = [self._history_marker_style(color) for color in prepared.detection_colors]
            pens = np.empty(len(styles), dtype=object)
            brushes = np.empty(len(styles), dtype=object)
            pens[:] = [style[0] for style in styles]
            brushes[:] = [style[1] for style in styles]
            self._recorded_marker_styles = (prepared, pens, brushes)
        _, pens, brushes = self._recorded_marker_styles
        self._detection_history_styles = (pens[: len(times)], brushes[: len(times)])
        self._history_from_recording = True
        self._render_gt_curves(gt_available=bool(target_history), apply_overlays=False)
        self._draw_track_estimates(prepared, now)
        if apply_overlays:
            # Resolve titles after this result's estimates are known.
            self._apply_overlay_visibility()

    def _fit_history_axes(self, history, now):
        """Use either the displayed prefix or the complete retained history bounds."""
        zoom = self._zoom_data_cb.isChecked()
        key = (
            now if zoom else None,
            self._show_det_cb.isChecked(),
            self._show_gt_cb.isChecked(),
            self._tracking_figures_active(),
            self.tracking_projection.currentText(),
        )
        cached = self._history_bounds_cache
        if cached is None or cached[0] is not history or cached[1] != key:
            cached = (history, key, self._history_axis_limits(history, now))
            self._history_bounds_cache = cached
        limits = cached[2]
        if limits is None:
            return
        time_limits, radial_limits, spatial_limits = limits
        for plot, bounds in zip(
            (self.gt_range_plot, self.gt_speed_plot), radial_limits, strict=True
        ):
            if not self._live_plot_needs_update(plot) or self._live_plot_is_inspected(plot):
                continue
            plot.setXRange(*time_limits, padding=0)
            if bounds is not None:
                plot.setYRange(*bounds, padding=0)
        if (
            spatial_limits is not None
            and self._live_plot_needs_update(self.tracking_plot)
            and not self._live_plot_is_inspected(self.tracking_plot)
        ):
            self.tracking_plot.setRange(
                xRange=spatial_limits[0], yRange=spatial_limits[1], padding=0
            )

    def _history_axis_limits(self, history, now):
        """Compute bounds once per immutable history and display choice."""
        zoom = self._zoom_data_cb.isChecked()
        times = [r.time_s for r in history.records if not zoom or r.time_s <= now + 1e-12]
        if not times:
            return

        def bounds(arrays):
            limits = []
            for values in arrays:
                finite = np.asarray(values)[np.isfinite(values)]
                if finite.size:
                    limits.append((float(finite.min()), float(finite.max())))
            if not limits:
                return None
            low, high = min(v[0] for v in limits), max(v[1] for v in limits)
            padding = (high - low) * 0.03 if high > low else max(abs(low), 1) * 0.05
            return low - padding, high + padding

        time_limits = bounds([times])
        prepared = history.tracking_plot_data

        def values(series):
            return series.through(now)[1] if zoom else series.values

        radial = []
        xy = []
        if self._show_det_cb.isChecked():
            radial.append(values(prepared.detections))
            if self._tracking_figures_active():
                radial.extend(values(series) for series in prepared.estimates.values())
            xy.extend(values(series) for series in prepared.tracks.values())
        if self._show_gt_cb.isChecked():
            radial.extend(values(series) for series in prepared.ranges.values())
            xy.extend(values(series) for series in prepared.truths.values())
        radial_limits = tuple(bounds([array[:, column] for array in radial]) for column in (0, 1))
        spatial_limits = None
        if self._tracking_figures_active():
            plane = self.tracking_projection.currentText().lower()
            cols = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
            x_limits = bounds([array[:, cols[0]] for array in xy])
            y_limits = bounds([array[:, cols[1]] for array in xy])
            if x_limits is not None and y_limits is not None:
                spatial_limits = (x_limits, y_limits)
        return time_limits, radial_limits, spatial_limits

    def _update_target_names(self, frame: dict) -> None:
        """Index target display names by exact canonical target ID."""
        metas = frame.get("targets_metadata") or []
        for meta in metas:
            if not isinstance(meta, Mapping):
                continue
            target_id = meta.get("target_id")
            name = meta.get("display_name") or meta.get("name")
            if isinstance(target_id, str) and target_id and isinstance(name, str) and name:
                self._target_names[target_id] = name

        sensing_block = frame.get("sensing")
        catalog = (
            sensing_block.get("target_catalog") if isinstance(sensing_block, Mapping) else None
        )
        catalog_targets = catalog.get("targets") if isinstance(catalog, Mapping) else None
        for target in catalog_targets if isinstance(catalog_targets, list) else []:
            if not isinstance(target, Mapping):
                continue
            target_id = target.get("target_id")
            name = target.get("display_name")
            if isinstance(target_id, str) and target_id and isinstance(name, str) and name:
                self._target_names[target_id] = name

    def _get_target_label(self, target_id: str) -> str:
        """Return a display name while retaining the semantic ID as fallback."""
        name = self._target_names.get(target_id)
        if name and isinstance(name, str):
            return name
        return target_id

    @staticmethod
    def _short_temporal_identity(value: Any) -> str:
        """Return an eight-character UI label without changing canonical identity."""
        text = str(value)
        return text[:8] if len(text) > 8 else text

    @staticmethod
    def _finite_time_s(value: Any) -> float | None:
        """Return a finite authored time in seconds, rejecting IDs and booleans."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        time_s = float(value)
        return time_s if math.isfinite(time_s) else None

    def _clear_acquisition_timeline(self, reason: str) -> None:
        """Clear all timeline glyphs and expose why physical time is unavailable."""
        self.acquisition_span_curve.setData([], [])
        self.acquisition_valid_samples.setData([], [])
        self.acquisition_invalid_samples.setData([], [])
        self.acquisition_result_marker.setVisible(False)
        self.acquisition_scene_time_marker.setVisible(False)
        self._acquisition_timeline_status = f"unavailable: {reason}"
        self.timing_status.setText(f"Acquisition timing unavailable: {reason}")
        self.acquisition_timeline_plot.setTitle(
            f"Acquisition timeline ({self._acquisition_timeline_status})"
        )

    def _update_acquisition_timeline(
        self,
        sensing_block: Mapping[str, Any],
        display_frame: Mapping[str, Any],
    ) -> None:
        """Render exact schema-v4 acquisition, result, and current-scene time."""
        temporal = sensing_block.get("temporal")
        if not isinstance(temporal, Mapping):
            self._clear_acquisition_timeline("schema-v4 temporal contract missing")
            return
        clock = temporal.get("clock")
        acquisition = temporal.get("acquisition")
        result = temporal.get("result")
        if not all(isinstance(value, Mapping) for value in (clock, acquisition, result)):
            self._clear_acquisition_timeline("clock, acquisition, or result metadata missing")
            return
        assert isinstance(clock, Mapping)
        assert isinstance(acquisition, Mapping)
        assert isinstance(result, Mapping)

        identity_values = {
            "clock": clock.get("clock_id"),
            "acquisition": acquisition.get("acquisition_id"),
            "pair": acquisition.get("pair_id"),
            "run": result.get("processing_run_id"),
            "recipe": result.get("processing_recipe_id"),
        }
        missing_identities = [
            name
            for name, value in identity_values.items()
            if not isinstance(value, str) or not value
        ]
        if missing_identities:
            self._clear_acquisition_timeline(
                "missing temporal identity: " + ", ".join(missing_identities)
            )
            return

        start_time_s = self._finite_time_s(acquisition.get("start_time_s"))
        end_time_s = self._finite_time_s(acquisition.get("end_time_s"))
        reference_time_s = self._finite_time_s(result.get("reference_time_s"))
        if start_time_s is None or end_time_s is None or reference_time_s is None:
            self._clear_acquisition_timeline("physical acquisition/result time missing")
            return
        if end_time_s < start_time_s or not start_time_s <= reference_time_s <= end_time_s:
            self._clear_acquisition_timeline("inconsistent acquisition/result span")
            return

        samples = acquisition.get("samples")
        if not isinstance(samples, list) or not samples:
            self._clear_acquisition_timeline("acquisition samples missing")
            return
        valid_times_s: list[float] = []
        invalid_times_s: list[float] = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                self._clear_acquisition_timeline("invalid acquisition sample metadata")
                return
            sample_time_s = self._finite_time_s(sample.get("sample_time_s"))
            valid = sample.get("valid")
            if sample_time_s is None or not isinstance(valid, bool):
                self._clear_acquisition_timeline("sample physical time or validity missing")
                return
            (valid_times_s if valid else invalid_times_s).append(sample_time_s)

        self.acquisition_span_curve.setData(
            [start_time_s, end_time_s],
            [0.0, 0.0],
        )
        self.acquisition_valid_samples.setData(
            valid_times_s,
            np.zeros(len(valid_times_s), dtype=np.float64),
        )
        self.acquisition_invalid_samples.setData(
            invalid_times_s,
            np.zeros(len(invalid_times_s), dtype=np.float64),
        )
        self.acquisition_result_marker.setPos(reference_time_s)
        self.acquisition_result_marker.setVisible(True)

        provenance = display_frame.get("_sensing_display_provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        held_stale = bool(provenance.get("stale_for_current_scene"))
        current_scene_time_s = self._finite_time_s(provenance.get("current_scene_time_s"))
        if current_scene_time_s is None and not held_stale:
            current_scene_time_s = self._finite_time_s(display_frame.get("timestamp_s"))
            if current_scene_time_s is None:
                timestamp_ns = self._finite_time_s(display_frame.get("timestamp_ns"))
                current_scene_time_s = timestamp_ns * 1e-9 if timestamp_ns is not None else None
        if current_scene_time_s is None:
            self.acquisition_scene_time_marker.setVisible(False)
        else:
            self.acquisition_scene_time_marker.setPos(current_scene_time_s)
            self.acquisition_scene_time_marker.setVisible(True)

        disposition = "held-stale" if held_stale else "exact"
        if held_stale:
            stale_age_s = self._finite_time_s(provenance.get("stale_age_s"))
            disposition += (
                f" stale={stale_age_s:.6g}s"
                if stale_age_s is not None and stale_age_s >= 0.0
                else " stale-age=unavailable"
            )

        identity_status = (
            f"acq={self._short_temporal_identity(identity_values['acquisition'])} "
            f"pair={self._short_temporal_identity(identity_values['pair'])} "
            f"run={self._short_temporal_identity(identity_values['run'])} "
            f"recipe={self._short_temporal_identity(identity_values['recipe'])}"
        )
        self._acquisition_timeline_status = f"{identity_status} | {disposition}"
        alignment = (
            f"Scene − result {current_scene_time_s - reference_time_s:+.6g} s"
            if current_scene_time_s is not None
            else "Scene time unavailable"
        )
        self.timing_status.setText(
            f"Acquisition {end_time_s - start_time_s:.4g} s · "
            f"{len(valid_times_s)}/{len(samples)} valid samples · "
            + (f"Held result ({disposition}) · " if held_stale else "")
            + alignment
        )
        self.timing_status.setStyleSheet("color: #e7a24c" if held_stale or invalid_times_s else "")
        self.acquisition_timeline_plot.setTitle(
            f"Acquisition timeline — {self._acquisition_timeline_status}"
        )

        visible_times = [start_time_s, end_time_s, reference_time_s]
        if current_scene_time_s is not None:
            visible_times.append(current_scene_time_s)
        time_min = min(visible_times)
        time_max = max(visible_times)
        span_s = time_max - time_min
        padding_s = span_s * 0.05 if span_s > 0.0 else max(abs(time_min) * 1e-6, 0.5)
        self.acquisition_timeline_plot.setXRange(
            time_min - padding_s,
            time_max + padding_s,
            padding=0.0,
        )

    # ── 1. Range Profile ────────────────────────────────────────────────
    @staticmethod
    def _cfr_window_annotation(result: Any, *, doppler: bool) -> str:
        """Name the retained frequency taper separately from slow-time weighting."""
        metadata = getattr(result, "metadata", None) or {}
        if metadata.get("range_response_model") != "cfr_ifft":
            return ""
        names = {"boxcar": "rectangular", "hann": "Hann", "hanning": "Hann"}
        frequency = metadata.get("frequency_window", "not recorded")
        time = metadata.get("doppler_taper", "not recorded")
        detail = (
            f"Doppler window: {names.get(time, time)}" if doppler else "before Doppler processing"
        )
        return f"Range window: {names.get(frequency, frequency)} · {detail}"

    def _update_range_profile(
        self, rp_result: Any, variant_suffix: str, *, rd_result: Any = None
    ) -> None:
        """Prefer the saved profile; label a display-only RD power reduction explicitly."""
        derived = rp_result is None and rd_result is not None
        result = rd_result if derived else rp_result
        if result is None:
            self.rp_curve.setData([], [])
            self.rp_samples.setData([], [])
            self.rp_plot.setToolTip("")
            self.rp_plot.setTitle("Range profile unavailable")
            return
        power = np.asarray(result.power_linear)
        if derived:
            power = np.sum(power, axis=0)
        with np.errstate(divide="ignore"):
            power_db = 10.0 * np.log10(power)
        positive = power > 0
        zeros = power == 0
        peak_db = float(power_db[positive].max()) if positive.any() else 0.0
        minimum_db = float(power_db[positive].min()) if positive.any() else peak_db
        floor_db = min(
            peak_db - _RANGE_PROFILE_DISPLAY_SPAN_DB,
            minimum_db - _RANGE_PROFILE_FLOOR_MARGIN_DB,
        )
        # Join the saved bin centers directly. These are samples of a response,
        # not piecewise-constant histogram heights. No smoothing or resampling.
        axis = np.asarray(result.range_axis_m)
        self.rp_curve.setData(axis, np.where(zeros, floor_db, power_db), stepMode=None)
        self.rp_samples.setData(axis[positive], power_db[positive])
        self.rp_samples.setVisible(False)
        metadata = getattr(result, "metadata", None) or {}
        support = metadata.get("temporal_support") or {}
        last_sample = support.get("selection_policy") == TEMPORAL_SUPPORT_LAST_VALID_SAMPLE
        profile_suffix = " · last sample" if last_sample and not derived else ""
        label = getattr(result, "range_axis_label", None) or "Path Length (m)"
        self.rp_plot.setLabel("bottom", label)
        self.rp_plot.setTitle(
            (
                f"Range power · Doppler sum{variant_suffix}"
                if derived
                else f"Range profile{variant_suffix}{profile_suffix}"
            )
            + (" · all bins zero" if not positive.any() else "")
        )
        notes = ["Saved RD power summed over Doppler." if derived else "Saved range-profile power."]
        window_note = self._cfr_window_annotation(result, doppler=derived)
        if window_note:
            notes.append(window_note)
        if last_sample and not derived and metadata.get("time_span"):
            notes.append(f"Last sample: t = {metadata['time_span']['reference_time_s']:.6g} s")
        if zeros.any():
            notes.append(f"Zero display floor: {floor_db:.1f} dB")
        self.rp_plot.setToolTip("\n".join(notes))
        self._fit_range_profile_axis()

    def _update_profile_detections(self, *_args: Any) -> None:
        """Project saved estimates onto range without contributing to plot bounds."""
        request = self._last_request
        detections = (
            request.display_frame["sensing"].get("detections", [])
            if request is not None
            and self._profile_detections_cb.isChecked()
            and self._show_det_cb.isChecked()
            and self.rp_curve.getData()[0] is not None
            else []
        )
        markers = {}
        for detection in detections:
            value = detection.get("range_m")
            if value is not None and np.isfinite(value):
                color = detection_color(detection)
                markers[(float(value), color)] = None
        while len(self._profile_detection_lines) < len(markers):
            line = pg.InfiniteLine(angle=90, movable=False)
            line.setZValue(15)
            self.rp_plot.addItem(line, ignoreBounds=True)
            self._profile_detection_lines.append(line)
        for line, (value, color) in zip(self._profile_detection_lines, markers):
            line.setValue(value)
            line.setPen(pg.mkPen(color, width=1.5, style=Qt.PenStyle.DashLine))
            line.show()
        for line in self._profile_detection_lines[len(markers) :]:
            line.hide()

    def _fit_range_profile_axis(self, *_args: Any) -> None:
        """Change view limits only; expose deep nulls on explicit request."""
        if self._live_plot_is_inspected(self.rp_plot):
            return
        axis, power_db = self.rp_curve.getData()
        samples, sample_db = self.rp_samples.getData()
        if axis is None or power_db is None:
            self.rp_samples.setVisible(False)
            return
        full = self._profile_full_dynamic_range_cb.isChecked()
        if not full:
            peak_db = float(np.max(sample_db)) if sample_db is not None else 0.0
            self.rp_plot.setYRange(
                peak_db - _RANGE_PROFILE_DISPLAY_SPAN_DB,
                peak_db + _RANGE_PROFILE_HEADROOM_DB,
                padding=0,
            )
        if not self.rp_plot.isVisible():
            if full:
                self.rp_plot.setYRange(
                    float(np.min(power_db)) - 1,
                    float(np.max(power_db)) + _RANGE_PROFILE_HEADROOM_DB,
                    padding=0.03,
                )
            return
        lo, hi = self.rp_plot.viewRange()[0]
        visible = (axis >= lo) & (axis <= hi)
        centers = axis[visible]
        pixels = self.rp_plot.getViewBox().width()
        spacing = float(np.min(np.diff(centers))) if len(centers) > 1 else hi - lo
        self.rp_samples.setVisible(
            bool(
                samples is not None
                and len(centers)
                and hi > lo
                and spacing * pixels / (hi - lo) >= _RANGE_PROFILE_MARKER_SPACING_PX
            )
        )
        if not full:
            return
        # Boundary values are used only to fit the viewport to the existing
        # straight segments; no interpolated point is added to the plotted data.
        boundaries = np.interp([lo, hi], axis, power_db, left=np.nan, right=np.nan)
        values = np.r_[power_db[visible], boundaries]
        values = values[np.isfinite(values)]
        if values.size:
            self.rp_plot.setYRange(
                float(values.min()) - 1,
                float(values.max()) + _RANGE_PROFILE_HEADROOM_DB,
                padding=0.03,
            )

    def _clear_range_time_history(self, reason: str | None = None) -> None:
        """Clear the canonical range-time cache and expose any contract failure."""
        self._range_time_records.clear()
        self._range_time_event_keys.clear()
        self._range_time_reference_times_s.clear()
        self._range_time_axis_m = None
        self._range_time_axis_label = "Path Length (m)"
        self.range_time_image.setData(np.empty((0, 0)), autoLevels=False)
        self.range_time_image.setVisible(False)
        self.range_time_raster.clear()
        self.range_time_raster.hide()
        self._range_time_status = reason or "waiting for schema-v4 temporal data"
        title = "Range-Time Waterfall"
        if reason:
            title += f" (unavailable: {reason})"
        self.range_time_plot.setTitle(title)
        self.range_time_plot.getPlotItem().getAxis("bottom").setLabel(
            "Result reference time", units="s"
        )
        self.range_time_plot.getPlotItem().getAxis("left").setLabel(self._range_time_axis_label)

    def _set_range_time_image(
        self,
        time_edges: np.ndarray,
        range_edges: np.ndarray,
        values: np.ndarray,
        levels: tuple[float, float],
    ) -> None:
        """Use a raster only when its affine transform preserves every cell edge."""
        regular = all(
            np.allclose(np.diff(edges), np.diff(edges)[0], rtol=1e-9, atol=1e-12)
            for edges in (time_edges, range_edges)
        )
        if regular:
            self.range_time_image.hide()
            self.range_time_image.setData(np.empty((0, 0)), autoLevels=False)
            # Col-major coordinates map values[time, range] directly to x/y.
            self.range_time_raster.setImage(values, levels=levels, autoLevels=False)
            self.range_time_raster.setRect(
                QRectF(
                    float(time_edges[0]),
                    float(range_edges[0]),
                    float(time_edges[-1] - time_edges[0]),
                    float(range_edges[-1] - range_edges[0]),
                )
            )
            self.range_time_raster.show()
        else:
            # Missing hops and nonuniform physical ranges must retain their
            # exact cell boundaries; an affine image would silently stretch them.
            self.range_time_raster.clear()
            self.range_time_raster.hide()
            x, y = np.meshgrid(time_edges, range_edges, indexing="ij")
            self.range_time_image.setData(x, y, values, autoLevels=False)
            self.range_time_image.setLevels(levels)
            self.range_time_image.show()

    def _fit_range_time_axis(self, power_db: np.ndarray, ranges: np.ndarray) -> None:
        """Fit the range axis to visible power across the whole displayed history."""
        view = self.range_time_plot.getViewBox()
        if not self._zoom_data_cb.isChecked() or not np.any(np.isfinite(power_db)):
            view.enableAutoRange(axis=pg.ViewBox.YAxis)
            return
        threshold = float(np.nanmax(power_db)) - self._rd_dynamic_range_db()
        occupied = np.flatnonzero(np.any(np.isfinite(power_db) & (power_db >= threshold), axis=0))
        if not occupied.size:
            view.enableAutoRange(axis=pg.ViewBox.YAxis)
            return
        edges = self._cell_edges(ranges, axis_name="range")
        # Show a neighboring cell on each side of the complete range history.
        lo, hi = max(0, occupied[0] - 1), min(len(ranges), occupied[-1] + 2)
        self.range_time_plot.setYRange(float(edges[lo]), float(edges[hi]), padding=0.03)

    @staticmethod
    def _range_time_event_key(
        sensing_block: Mapping[str, Any],
    ) -> tuple[str, str, str, str, float]:
        """Return canonical event identity and reject frame-number substitutes."""
        temporal = sensing_block.get("temporal")
        if not isinstance(temporal, Mapping):
            raise ValueError("schema-v4 temporal contract missing")
        clock = temporal.get("clock")
        acquisition = temporal.get("acquisition")
        result = temporal.get("result")
        if not all(isinstance(value, Mapping) for value in (clock, acquisition, result)):
            raise ValueError("clock, acquisition, or result metadata missing")
        assert isinstance(clock, Mapping)
        assert isinstance(acquisition, Mapping)
        assert isinstance(result, Mapping)
        identities = (
            clock.get("clock_id"),
            acquisition.get("acquisition_id"),
            result.get("processing_run_id"),
            acquisition.get("pair_id"),
        )
        if any(not isinstance(value, str) or not value for value in identities):
            raise ValueError("canonical clock/acquisition/run/pair identity missing")
        reference_time_s = _DashboardContent._finite_time_s(result.get("reference_time_s"))
        if reference_time_s is None:
            raise ValueError("result reference_time_s missing")
        return (
            str(identities[0]),
            str(identities[1]),
            str(identities[2]),
            str(identities[3]),
            reference_time_s,
        )

    @staticmethod
    def _cell_edges(centers: np.ndarray, *, axis_name: str) -> np.ndarray:
        """Return exact midpoint cell edges for an authored monotonic axis."""
        values = np.asarray(centers, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"{axis_name} axis must be a finite non-empty 1-D array")
        if values.size == 1:
            return np.array([values[0] - 0.5, values[0] + 0.5], dtype=np.float64)
        spacings = np.diff(values)
        if not np.all(spacings > 0.0):
            raise ValueError(f"{axis_name} axis must be strictly increasing")
        edges = np.empty(values.size + 1, dtype=np.float64)
        edges[1:-1] = (values[:-1] + values[1:]) / 2.0
        edges[0] = values[0] - spacings[0] / 2.0
        edges[-1] = values[-1] + spacings[-1] / 2.0
        return edges

    def _update_range_time_waterfall(
        self,
        rp_result: Any,
        *,
        sensing_block: Mapping[str, Any] | None = None,
        display_step: Optional[int] = None,
        variant_suffix: str,
    ) -> None:
        """Upsert and physically sort one canonical range-profile event."""
        del display_step  # Opaque frame IDs are intentionally never interpreted as time.
        if rp_result is None:
            self._clear_range_time_history("range profile unavailable")
            return

        try:
            event_key = self._range_time_event_key(sensing_block or {})
        except ValueError as exc:
            self._clear_range_time_history(str(exc))
            return

        range_axis_m = np.asarray(rp_result.range_axis_m, dtype=float)
        power_linear = np.asarray(rp_result.power_linear, dtype=float)
        axis_label = getattr(rp_result, "range_axis_label", None) or "Path Length (m)"
        if (
            range_axis_m.ndim != 1
            or power_linear.ndim != 1
            or range_axis_m.size == 0
            or power_linear.shape != range_axis_m.shape
            or not np.all(np.isfinite(power_linear))
        ):
            self._clear_range_time_history("invalid range-profile shape or values")
            return
        try:
            range_edges_m = self._cell_edges(range_axis_m, axis_name="range")
        except ValueError as exc:
            self._clear_range_time_history(str(exc))
            return
        power_db = 10.0 * np.log10(np.maximum(power_linear, np.finfo(np.float64).tiny))

        if self._range_time_axis_m is not None and (
            not np.array_equal(self._range_time_axis_m, range_axis_m)
            or self._range_time_axis_label != axis_label
        ):
            self._clear_range_time_history("range-axis disagreement across results")
            return

        existing = self._range_time_records.get(event_key)
        if existing is None or not np.array_equal(existing, power_db):
            self._range_time_records[event_key] = power_db.copy()
        self._range_time_axis_m = range_axis_m.copy()
        self._range_time_axis_label = axis_label

        ordered_keys = sorted(
            self._range_time_records,
            key=lambda key: (key[4], key[0], key[1], key[2], key[3]),
        )
        for expired_key in ordered_keys[:-_MAX_RANGE_TIME_HISTORY]:
            self._range_time_records.pop(expired_key, None)
        ordered_keys = ordered_keys[-_MAX_RANGE_TIME_HISTORY:]
        reference_times_s = np.asarray([key[4] for key in ordered_keys], dtype=np.float64)
        waterfall = np.stack([self._range_time_records[key] for key in ordered_keys], axis=0)

        # Distinct result times use their exact physical coordinates. Multiple
        # runs at one reference epoch fall back to deterministic event columns
        # with exact time labels rather than inventing small time offsets.
        if np.unique(reference_times_s).size == reference_times_s.size:
            x_centers = reference_times_s
            self.range_time_plot.getPlotItem().getAxis("bottom").setTicks(None)
            bottom_label = "Result reference time"
        else:
            x_centers = np.arange(reference_times_s.size, dtype=np.float64)
            ticks = [
                (float(index), f"{reference_time_s:.6g}")
                for index, reference_time_s in enumerate(reference_times_s)
            ]
            self.range_time_plot.getPlotItem().getAxis("bottom").setTicks([ticks])
            bottom_label = "Result reference time (ordered event columns)"
        time_edges_s = self._cell_edges(x_centers, axis_name="result-time")

        self._range_time_event_keys = list(ordered_keys)
        self._range_time_reference_times_s = reference_times_s.tolist()
        vmax = float(np.max(waterfall))
        vmin = vmax - self._rd_dynamic_range_db()
        self._set_range_time_image(time_edges_s, range_edges_m, waterfall, (vmin, vmax))
        self._fit_range_time_axis(waterfall, self._range_time_axis_m)
        self.range_time_plot.getPlotItem().getAxis("bottom").setLabel(bottom_label, units="s")
        self.range_time_plot.getPlotItem().getAxis("left").setLabel(axis_label)
        self._range_time_status = (
            f"{len(ordered_keys)} canonical result(s), physical reference-time order"
        )
        self.range_time_plot.setTitle(
            f"Range-Time Waterfall{variant_suffix} — {self._range_time_status}"
        )

    # ── 2. Range-Velocity Map ───────────────────────────────────────────
    def _update_range_velocity(
        self, rd_result: Any, frame: dict, variant_suffix: str, carrier_freq_hz: float
    ) -> None:
        """Render the range-Doppler map with detection and GT overlays."""
        if rd_result is None:
            self.rd_image.clear()
            self.rd_cfar_overlay.clear()
            self.rd_det_scatter.setData([], [])
            self.rd_gt_scatter.setData([], [])
            self.rd_plot.setTitle("Range–Doppler unavailable · map was not recorded")
            self.rd_plot.setToolTip("Range–Doppler products were not recorded.")
            self.rd_colorbar.hide()
            return

        self.rd_plot.setToolTip("")
        show_heatmap = self._heatmap_enabled(self.rd_plot)
        self.rd_image.setVisible(show_heatmap)
        self.rd_colorbar.setVisible(show_heatmap)
        sensing_block = frame.get("sensing") or {}
        velocity_display = self._resolve_detection_velocity_display(sensing_block, rd_result)

        use_hz = self._use_doppler_hz
        if use_hz:
            y_axis = np.asarray(rd_result.doppler_axis_hz, dtype=np.float64)
            y_label = "Doppler (Hz)"
        else:
            y_axis = self._doppler_to_velocity(rd_result.doppler_axis_hz, carrier_freq_hz)
            y_label = velocity_display.axis_label
        rect = self._rd_rect(rd_result.range_axis_m, y_axis)

        # Disable auto-range before setImage so the retained zoom is not replaced.
        zoom_active = self._zoom_data_cb.isChecked()
        if zoom_active:
            self.rd_plot.getPlotItem().getViewBox().disableAutoRange()

        if show_heatmap:
            power_db = self._heatmap_db(self.rd_image, rd_result.power_linear)
            vmin, vmax = self._levels("rd", power_db, self._rd_dynamic_range_db())
            if self.rd_image.image is None or not np.shares_memory(self.rd_image.image, power_db):
                self.rd_image.setImage(power_db.T, autoLevels=False)
            if self.rd_colorbar.values != (vmin, vmax):
                self.rd_colorbar.setLevels(values=(vmin, vmax))
            geometry = (power_db.shape, rect)
            if geometry != self._rd_image_geometry:
                self.rd_image.setRect(rect)
                self._rd_image_geometry = geometry
            self._update_rd_cfar_overlay(sensing_block, rd_result, rect)
        else:
            self.rd_cfar_overlay.hide()

        # Detection overlay
        detections = sensing_block.get("detections") or []
        if detections:
            xs = [float(d.get("range_m", 0.0)) for d in detections]
            if use_hz:
                ys = [float(d.get("doppler_hz", 0.0)) for d in detections]
            else:
                ys = [float(d.get("velocity_m_s", 0.0)) for d in detections]
            self.rd_det_scatter.setData(
                xs,
                ys,
                data=list(range(len(detections))),
                brush=[pg.mkBrush(detection_color(d)) for d in detections],
            )
        else:
            self.rd_det_scatter.setData([], [])

        # GT overlay
        gt_xs, gt_ys = self._current_gt_range_speed()
        if gt_xs:
            gt_display_y = (
                self._velocity_to_doppler_hz(np.asarray(gt_ys, dtype=np.float64), carrier_freq_hz)
                if use_hz
                else np.asarray(gt_ys, dtype=np.float64)
            )
            self.rd_gt_scatter.setData(gt_xs, gt_display_y.tolist())
        else:
            self.rd_gt_scatter.setData([], [])

        if show_heatmap:
            self._apply_zoom_to_data(self.rd_plot, power_db, rd_result.range_axis_m, y_axis)
        else:
            self._fit_marker_map_axes(self.rd_plot, rd_result.range_axis_m, y_axis)

        label = getattr(rd_result, "range_axis_label", None) or "Path Length (m)"
        self.rd_plot.setLabel("bottom", label)
        self.rd_plot.setLabel("left", y_label)
        title_y = "Doppler" if use_hz else "Velocity"
        cfg = sensing_block.get("config_resolved") or {}
        beam_detections = (cfg.get("processing_recipe") or {}).get(
            "array_detection_mode"
        ) == "beam_scan"
        broadside = cfg.get("rx_combine_policy") == "rx_broadside_unitary_voltage_sum"
        combination = " · broadside" if beam_detections and broadside else ""
        map_note = "" if show_heatmap else " · heatmap hidden"
        self.rd_plot.setTitle(f"Range–{title_y}{variant_suffix}{combination}{map_note}")
        notes = ["Saved power in dB re 1 channel unit."]
        if beam_detections and broadside:
            notes.append("Broadside map; detector decisions use scanned beams.")
        window_note = self._cfr_window_annotation(rd_result, doppler=True)
        if window_note:
            notes.append(window_note)
        self.rd_plot.setToolTip("\n".join(notes))

    def _update_rd_cfar_overlay(self, sensing_block: dict, rd_result: Any, rect: QRectF) -> None:
        """Overlay the exact persisted CFAR decision bound to this RD product."""
        del sensing_block
        mask = self._persisted_cfar_mask_for_rd(rd_result)
        if mask is None:
            self.rd_cfar_overlay.clear()
            return
        self.rd_cfar_overlay.setImage(mask.T.astype(np.uint8, copy=False), autoLevels=False)
        self.rd_cfar_overlay.setLookupTable(self._cfar_mask_lut)
        self.rd_cfar_overlay.setLevels([0, 1])
        self.rd_cfar_overlay.setRect(rect)

    def _persisted_cfar_mask_for_rd(self, rd_result: Any) -> np.ndarray | None:
        """Return a lineage- and coordinate-matched persisted CFAR mask."""

        request = self._last_request
        bundle = request.detection_stage_bundle if request is not None else None
        if bundle is None or bundle.disposition is not InspectionDisposition.AVAILABLE:
            return None
        product = bundle.product("range_doppler_cfar_mask")
        if not product.available or product.values is None:
            return None
        values = np.asarray(product.values)
        rd_power = np.asarray(getattr(rd_result, "power_linear", None))
        rd_doppler = np.asarray(getattr(rd_result, "doppler_axis_hz", None))
        rd_range = np.asarray(getattr(rd_result, "range_axis_m", None))
        if (
            values.shape != (1, *rd_power.shape)
            or bundle.doppler_axis_hz is None
            or bundle.range_axis_m is None
            or not np.array_equal(bundle.doppler_axis_hz, rd_doppler)
            or not np.array_equal(bundle.range_axis_m, rd_range)
        ):
            return None

        rd_metadata = getattr(rd_result, "metadata", None) or {}
        rd_product_id = rd_metadata.get("product_id") if isinstance(rd_metadata, Mapping) else None
        trace = product.lineage_trace
        if not isinstance(rd_product_id, str) or trace is None:
            return None
        if rd_product_id not in {node.product_id for node in trace.nodes}:
            return None
        return np.asarray(values[0], dtype=np.bool_)

    # ── 3. CFAR on Range Profile ────────────────────────────────────────
    def _sync_detector_range(self, source, bounds) -> None:
        """Keep the full-width slice and detector maps on the same range limits."""
        target = (
            self.cfar_rd_plot if source is self.cfar_rp_plot.getViewBox() else self.cfar_rp_plot
        )
        if self._syncing_detector_range or target.viewRange()[0] == list(bounds):
            return
        self._syncing_detector_range = True
        try:
            target.setXRange(*bounds, padding=0)
        finally:
            self._syncing_detector_range = False

    def _update_cfar_rp(self, sensing_block: dict, rp_result: Any, variant_suffix: str) -> None:
        """Show the recorded 2D detector's range slice, never invent 1D CFAR."""
        del sensing_block, rp_result, variant_suffix
        self.cfar_rp_power.setData([], [])
        self.cfar_rp_thresh.setData([], [])
        self.stage_slice_detections.setData([], [])
        self.stage_report_input.clear()
        self.stage_report_threshold.clear()
        bundle = self._last_request.detection_stage_bundle if self._last_request else None
        if bundle is None or not bundle.available:
            self.cfar_rp_plot.setTitle("Detector threshold slice not recorded")
            return
        reports = self.stage_comparison_scope.currentData() == "reports"
        self.stage_range_line.setVisible(not reports)
        self.cfar_rp_plot.getPlotItem().legend.setVisible(not reports)
        if reports:
            self._update_all_report_thresholds(bundle)
            return
        row = self._detection_stage_doppler_box.value()
        for kind, curve in (
            ("range_doppler_detection_input_power", self.cfar_rp_power),
            ("range_doppler_cfar_threshold_power", self.cfar_rp_thresh),
        ):
            product = bundle.product(kind)
            if product.available:
                values = 10 * np.log10(np.maximum(product.values[0, row], 1e-30))
                if kind.endswith("threshold_power"):
                    eligible = bundle.product("range_doppler_cfar_eligible_mask")
                    values = np.where(
                        eligible.values[0, row] if eligible.available else False, values, np.nan
                    )
                curve.setData(bundle.range_axis_m, values, connect="finite")
        self.cfar_rp_plot.setLabel("bottom", range_label(self._last_request.range_mode), units="m")
        self.cfar_rp_plot.setLabel("left", "Power (dB)")
        self.cfar_rp_plot.setToolTip(
            "This is one Doppler row; detections at other velocities are outside this slice. Choose All detections to compare every reported cell. Power is dB re 1 recorded linear unit."
        )
        self.cfar_rp_plot.getPlotItem().legend.setColumnCount(3)
        carrier = self._last_request.display_frame["sensing"]["config_resolved"]["carrier_freq_hz"]
        velocity = self._doppler_to_velocity(bundle.doppler_axis_hz, carrier)[row]
        self.cfar_rp_plot.setTitle(
            f"Input / threshold · Doppler bin {row} · "
            f"{bundle.doppler_axis_hz[row]:.3g} Hz · {velocity:.3g} m/s"
            + self._resolve_detection_velocity_display(
                self._last_request.display_frame["sensing"]
            ).inline_suffix
        )
        self.stage_range_line.setPos(bundle.range_axis_m[self._detection_stage_range_box.value()])
        input_product = bundle.product("range_doppler_detection_input_power")
        if input_product.available:
            bins = sorted(
                {link.range_bin for link in bundle.detection_links if link.doppler_bin == row}
            )
            self.stage_slice_detections.setData(
                bundle.range_axis_m[bins],
                10 * np.log10(np.maximum(input_product.values[0, row, bins], 1e-30)),
            )
        # A heatmap's colour floor must not crop the measured line or its peaks.
        # Fit both curves over the visible range, including when zoom is off.
        view = self.cfar_rp_plot.getViewBox()
        view.setAutoVisible(y=True)
        view.enableAutoRange(axis=pg.ViewBox.YAxis)

    def _update_all_report_thresholds(self, bundle) -> None:
        """Compare each reported peak to its own saved threshold, never a max projection."""
        input_product = bundle.product("range_doppler_detection_input_power")
        threshold = bundle.product("range_doppler_cfar_threshold_power")
        eligible = bundle.product("range_doppler_cfar_eligible_mask")
        cells = sorted({(link.doppler_bin, link.range_bin) for link in bundle.detection_links})
        heights = []
        for product, item, is_threshold in (
            (input_product, self.stage_report_input, False),
            (threshold, self.stage_report_threshold, True),
        ):
            spots = []
            if product.available:
                for d, r in cells:
                    if is_threshold and (not eligible.available or not eligible.values[0, d, r]):
                        continue
                    value = float(product.values[0, d, r])
                    if not np.isfinite(value) or value < 0:
                        continue
                    height = 10 * np.log10(max(value, 1e-30))
                    heights.append(height)
                    spots.append(
                        dict(
                            pos=(float(bundle.range_axis_m[r]), height),
                            data=f"Range bin {r}; Doppler bin {d} ({bundle.doppler_axis_hz[d]:.6g} Hz)",
                        )
                    )
            item.setData(spots, tip=lambda x, y, data: f"{data}\n{y:.3f} dB")
        input_label = "blue: input" if input_product.available else "input not recorded"
        threshold_label = (
            "red ×: own CFAR threshold"
            if threshold.available and eligible.available
            else "threshold / eligibility not recorded"
        )
        self.cfar_rp_plot.setTitle(
            f"All {len(cells)} published peak cells · {input_label} · {threshold_label}"
        )
        self.cfar_rp_plot.setLabel("bottom", range_label(self._last_request.range_mode), units="m")
        self.cfar_rp_plot.setLabel("left", "Power (dB)")
        self.cfar_rp_plot.setToolTip(
            "Each pair uses its own range AND Doppler cell. Hover for coordinates. Equal-range reports can overlap; use the RD map or 3D view to distinguish velocity. No lines connect different Doppler rows; no detector is rerun."
        )
        if heights:
            # Every report remains visible, including weak reports below the
            # map's display floor. Values stay in the same absolute dB units.
            self.cfar_rp_plot.setYRange(min(heights) - 1, max(heights) + 1, padding=0.06)

    # ── 4. CFAR RD Mask ─────────────────────────────────────────────────
    def _update_cfar_rd(
        self, sensing_block: dict, rd_result: Any, carrier_freq_hz: float, variant_suffix: str
    ) -> None:
        """Render the detector input and all saved stages on their recorded grid."""
        del rd_result, variant_suffix
        self.stage_cell.setData([], [])
        self.stage_slice_line.hide()
        self.stage_bin_grid.hide()
        self.bin_spacing_label.hide()
        bundle = self._last_request.detection_stage_bundle if self._last_request else None
        if bundle is None or not bundle.available:
            self.cfar_rd_image.clear()
            self.cfar_rd_mask.clear()
            self.cfar_pass_borders.clear()
            self._stage_image_render_key = None
            self.cfar_rd_plot.setTitle("Detector stages not available")
            for plot in self._stage_maps.values():
                plot.set_map(None, "Not recorded", (0, 1), None, None)
                plot.select_cell()
            return
        kind = "range_doppler_detection_input_power"
        values, units = self._stage_image(bundle, kind)
        self.cfar_rd_plot.setTitle(
            _DETECTION_STAGE_LABELS[kind] + (" · " + units.lower() if values is None else "")
        )
        self.cfar_rd_plot.setToolTip(STAGE_HELP[kind])
        self.stage_colorbar.getAxis("left").setLabel(units)
        y_axis = (
            bundle.doppler_axis_hz
            if self._use_doppler_hz
            else self._doppler_to_velocity(bundle.doppler_axis_hz, carrier_freq_hz)
        )
        if not regular_axis(bundle.range_axis_m) or not (
            regular_axis(y_axis) or regular_axis(y_axis[::-1])
        ):
            self.cfar_rd_image.clear()
            self.cfar_rd_mask.clear()
            self.cfar_pass_borders.clear()
            self._stage_image_render_key = None
            self.cfar_rd_plot.setTitle(
                "Nonuniform coordinates: use exact 3D surface or linked slice"
            )
            for plot in self._stage_maps.values():
                plot.set_map(None, "Nonuniform coordinates", (0, 1), None, None)
                plot.select_cell()
            return
        rect = self._rd_rect(bundle.range_axis_m, y_axis)
        levels = self._stage_power_levels(bundle)
        render_key = (
            id(bundle),
            kind,
            levels,
            self._use_doppler_hz,
            self._zoom_data_cb.isChecked(),
        )
        if render_key != self._stage_image_render_key:
            self.cfar_rd_mask.clear()
            self.cfar_pass_borders.clear()
            if values is None:
                self.cfar_rd_image.clear()
            else:
                self.cfar_rd_image.setImage(values.T, autoLevels=False)
                self.stage_colorbar.setLevels(values=levels)
                self.cfar_rd_image.setRect(rect)
            mask = bundle.product("range_doppler_cfar_mask")
            if mask.available:
                self.cfar_rd_mask.setImage(
                    np.asarray(mask.values[0], dtype=np.uint8).T, autoLevels=False
                )
                self.cfar_rd_mask.setLookupTable(self._cfar_mask_lut)
                self.cfar_rd_mask.setLevels([0, 1])
                self.cfar_rd_mask.setRect(rect)
                self._update_cfar_pass_borders(mask.values[0], bundle.range_axis_m, y_axis)
            reference, _units = self._stage_image(bundle, "range_doppler_detection_input_power")
            if reference is not None:
                self._apply_zoom_to_data(self.cfar_rd_plot, reference, bundle.range_axis_m, y_axis)
            self._stage_image_render_key = render_key
        row, column = (
            self._detection_stage_doppler_box.value(),
            self._detection_stage_range_box.value(),
        )
        self.stage_slice_line.setPos(y_axis[row])
        self.stage_slice_line.show()
        self.stage_cell.setData([bundle.range_axis_m[column]], [y_axis[row]])
        label = (
            "Range (m)"
            if sensing_block["config_resolved"]["range_mode"] == "monostatic"
            else "Path length (m)"
        )
        self.cfar_rd_plot.setLabel("bottom", label)
        self.cfar_rd_plot.setLabel(
            "left",
            (
                "Doppler (Hz)"
                if self._use_doppler_hz
                else self._resolve_detection_velocity_display(sensing_block).axis_label
            ),
        )
        self._update_detector_maps(bundle, y_axis, rect, label)
        self._apply_overlay_visibility()
        self._update_detector_bin_grid()

    def _update_detector_maps(
        self, bundle: DetectionStageBundle, y_axis: np.ndarray, rect: QRectF, x_label: str
    ) -> None:
        """Share input/threshold scales and coordinates across detector maps."""
        reference, _ = self._stage_image(bundle, "range_doppler_detection_input_power")
        row = self._detection_stage_doppler_box.value()
        column = self._detection_stage_range_box.value()
        for kind, plot in self._stage_maps.items():
            values, units = self._stage_image(bundle, kind)
            finite = values[np.isfinite(values)] if values is not None else np.empty(0)
            if kind.endswith("_mask"):
                levels = (0, 1)
            elif kind == "range_doppler_cfar_threshold_power":
                levels = self._stage_power_levels(bundle)
            elif kind.endswith("_power") or kind == "range_doppler_snr_db":
                levels = (
                    self._levels("stage-" + kind, finite, self._rd_dynamic_range_db())
                    if finite.size
                    else (0, 1)
                )
            else:
                levels = (0, max(1, float(np.max(finite)))) if finite.size else (0, 1)
            key = (id(bundle), levels, self._use_doppler_hz, self._zoom_data_cb.isChecked())
            if plot.set_map(values, units, levels, rect, key):
                title = _DETECTION_STAGE_LABELS[kind]
                if values is None:
                    title += " · " + units.lower()
                elif kind == "range_doppler_cluster_labels":
                    count = len(plot.cluster_ids)
                    title += f" · {count} cluster{'s' if count != 1 else ''}"
                elif kind.endswith("_power") and finite.size and np.min(finite) == np.max(finite):
                    title += f" · constant {finite[0]:.3g}"
                plot.setTitle(title)
                plot.setLabel("bottom", x_label)
                plot.setLabel("left", self.cfar_rd_plot.getPlotItem().getAxis("left").labelText)
                if reference is not None:
                    self._apply_zoom_to_data(plot, reference, bundle.range_axis_m, y_axis)
            if values is None or self._cell_selection_cleared():
                plot.select_cell()
            else:
                plot.select_cell(bundle.range_axis_m[column], y_axis[row])

    def _update_cfar_pass_borders(
        self, mask: np.ndarray, range_axis: np.ndarray, y_axis: np.ndarray
    ) -> None:
        """Outline each passing saved cell, drawing shared borders only once."""
        passed = np.asarray(mask, dtype=np.bool_)
        x_edges = self._axis_edges(range_axis, axis_name="range")
        y_edges = self._axis_edges(y_axis, axis_name="Doppler/velocity")
        vertical = np.pad(passed, ((0, 0), (0, 1))) | np.pad(passed, ((0, 0), (1, 0)))
        horizontal = np.pad(passed, ((0, 1), (0, 0))) | np.pad(passed, ((1, 0), (0, 0)))
        v_row, v_col = np.nonzero(vertical)
        h_row, h_col = np.nonzero(horizontal)
        self.cfar_pass_borders.setData(
            x=np.concatenate(
                (
                    np.repeat(x_edges[v_col], 2),
                    np.column_stack((x_edges[h_col], x_edges[h_col + 1])).ravel(),
                )
            ),
            y=np.concatenate(
                (
                    np.column_stack((y_edges[v_row], y_edges[v_row + 1])).ravel(),
                    np.repeat(y_edges[h_row], 2),
                )
            ),
            connect="pairs",
        )

    def _update_detector_bin_grid(self, *_args: Any) -> None:
        """Outline actual saved image cells without changing data or plot bounds."""
        self.stage_bin_grid.hide()
        self.bin_spacing_label.hide()
        self.cfar_rd_plot.showGrid(x=True, y=True)
        for plot in self._stage_maps.values():
            plot.bin_grid.hide()
            plot.showGrid(x=True, y=True)
        request = self._last_request
        bundle = request.detection_stage_bundle if request else None
        if (
            not self._show_bin_grid_cb.isChecked()
            or self._show_bin_grid_cb.isHidden()
            or bundle is None
            or not bundle.available
            or self._stage_image_render_key is None
            or self._stage_image_render_key[0] != id(bundle)
        ):
            return
        sensing = request.display_frame["sensing"]
        carrier = sensing["config_resolved"]["carrier_freq_hz"]
        ranges = bundle.range_axis_m
        y_axis = (
            bundle.doppler_axis_hz
            if self._use_doppler_hz
            else self._doppler_to_velocity(bundle.doppler_axis_hz, carrier)
        )
        x_edges = self._axis_edges(ranges, axis_name="range")
        y_edges = self._axis_edges(y_axis, axis_name="Doppler/velocity")
        # Tick-based grid lines would suggest a second, unrelated cell spacing.
        self.cfar_rd_plot.showGrid(x=False, y=False)
        # One curve holds independent two-point segments, not a widget per line.
        self.stage_bin_grid.setData(
            x=np.concatenate((np.repeat(x_edges, 2), np.tile(x_edges[[0, -1]], y_edges.size))),
            y=np.concatenate((np.tile(y_edges[[0, -1]], x_edges.size), np.repeat(y_edges, 2))),
            connect="pairs",
        )
        self.stage_bin_grid.show()
        grid_x, grid_y = self.stage_bin_grid.getData()
        for plot in self._stage_maps.values():
            if plot.image.image is not None:
                plot.showGrid(x=False, y=False)
                plot.bin_grid.setData(x=grid_x, y=grid_y, connect="pairs")
                plot.bin_grid.show()

        def spacing(axis: np.ndarray, unit: str) -> str:
            if axis.size < 2:
                return "single saved bin"
            return f"{abs(float(axis[1] - axis[0])):.4g} {unit}"

        y_label, y_unit = ("Doppler", "Hz") if self._use_doppler_hz else ("velocity", "m/s")
        self.bin_spacing_label.setText(
            f"Bin spacing: {range_label(request.range_mode).lower()} {spacing(ranges, 'm')} · "
            f"{y_label} {spacing(y_axis, y_unit)}. Clicks snap to cell centres."
        )
        self.bin_spacing_label.show()

    def _select_stage_map_cell(self, event: Any, plot=None, kind=None) -> None:
        """Select exact saved coordinates without requesting data or processing."""
        request = self._last_request
        if event.button() != Qt.MouseButton.LeftButton or request is None:
            return
        bundle = request.detection_stage_bundle
        view = (plot or self.cfar_rd_plot).getViewBox()
        if (
            bundle is None
            or not bundle.available
            or (kind is not None and not bundle.product(kind).available)
            or not view.sceneBoundingRect().contains(event.scenePos())
        ):
            return
        self._detection_stage_kind = kind or "range_doppler_detection_input_power"
        point = view.mapSceneToView(event.scenePos())
        carrier = request.display_frame["sensing"]["config_resolved"]["carrier_freq_hz"]
        y_axis = (
            bundle.doppler_axis_hz
            if self._use_doppler_hz
            else self._doppler_to_velocity(bundle.doppler_axis_hz, carrier)
        )
        with (
            QSignalBlocker(self._detection_stage_range_box),
            QSignalBlocker(self._detection_stage_doppler_box),
        ):
            self._detection_stage_range_box.setValue(
                int(np.argmin(np.abs(bundle.range_axis_m - point.x())))
            )
            self._detection_stage_doppler_box.setValue(int(np.argmin(np.abs(y_axis - point.y()))))
        self._select_inspection_coordinate(
            self._detection_stage_range_box.value(),
            self._detection_stage_doppler_box.value(),
            toggle=True,
        )
        event.accept()

    # ── 4b. Angular products ───────────────────────────────────────────
    def _update_angular_products(
        self,
        sensing_block: dict,
        variant: str,
        variant_suffix: str,
        carrier_freq_hz: float,
    ) -> None:
        """Render projections from one exact typed P,E,A,D,R angular product."""
        request = self._last_request
        prepared = (
            request.prepared_angular
            if request is not None and request.display_frame.get("sensing") is sensing_block
            else None
        )
        if prepared is not None and prepared.variant != variant:
            prepared = None
        if prepared is None:
            try:
                prepared = prepare_angular(sensing_block, variant=variant)
            except (KeyError, ValueError) as exc:
                logger.warning("Typed angular product is not renderable: %s", exc)
                self._clear_angular_products(f"invalid typed product: {exc}")
                return
        if prepared is None:
            self._clear_angular_products(f"no typed {variant} product")
            return
        az_axis = prepared.azimuth_axis_deg
        range_axis = prepared.range_axis_m
        if self._use_doppler_hz:
            doppler_axis = prepared.doppler_axis_hz
            doppler_label = "Doppler (Hz)"
        else:
            doppler_axis = self._doppler_to_velocity(
                prepared.doppler_axis_hz,
                carrier_freq_hz,
            )
            doppler_label = self._resolve_detection_velocity_display(sensing_block).axis_label

        range_label = (
            "Monostatic Range (m)"
            if prepared.range_coordinate_semantics == "monostatic_range"
            else "Bistatic Path Length (m)"
        )
        try:
            if self._live_plot_needs_update(self.range_azimuth_plot):
                self._update_angular_image(
                    self.range_azimuth_image,
                    self.range_azimuth_plot,
                    prepared.range_azimuth_power,
                    range_axis,
                    az_axis,
                    f"Range–broadside azimuth{variant_suffix}<br>Max over Doppler/elevation",
                    range_label,
                )
            if self._live_plot_needs_update(self.doppler_azimuth_plot):
                self._update_angular_image(
                    self.doppler_azimuth_image,
                    self.doppler_azimuth_plot,
                    prepared.doppler_azimuth_power,
                    doppler_axis,
                    az_axis,
                    f"Doppler–broadside azimuth{variant_suffix}<br>Max over range/elevation",
                    doppler_label,
                )
            if self._live_plot_needs_update(self.angular_spectrum_plot):
                self._update_angular_spectrum(prepared, sensing_block, variant_suffix)
        except ValueError as exc:
            logger.warning("Typed angular coordinates are not renderable: %s", exc)
            self._clear_angular_products(f"unsupported coordinates: {exc}")
            return
        self._angular_shape = prepared.shape
        self._angular_status = f"typed {variant}"
        self._update_angular_detection_overlays(
            sensing_block, doppler_axis, has_elevation=prepared.shape[0] > 1
        )

    def _clear_angular_products(self, reason: str) -> None:
        """Clear angular views and surface the exact unavailable reason."""

        self.range_azimuth_image.setVisible(False)
        self.doppler_azimuth_image.setVisible(False)
        self.angular_spectrum_image.setVisible(False)
        self.range_azimuth_det_scatter.setData([], [])
        self.doppler_azimuth_det_scatter.setData([], [])
        self.angular_spectrum_det_scatter.setData([], [])
        self.range_azimuth_plot.setTitle(f"Range-Azimuth — unavailable: {reason}")
        self.doppler_azimuth_plot.setTitle(f"Doppler-Azimuth — unavailable: {reason}")
        self.angular_spectrum_plot.setTitle(f"Angular spectrum — unavailable: {reason}")
        self._angular_status = reason
        self._angular_shape = None

    def _update_angular_spectrum(self, angular: PreparedAngle, sensing_block, variant_suffix):
        """Show a joint azimuth–zenith map only when both coordinates were saved."""
        self.angular_spectrum_det_scatter.setData([], [])
        if angular.spectrum_power is None:
            self.angular_spectrum_image.hide()
            self.angular_spectrum_plot.setTitle("Angular spectrum — elevation not resolved")
            return
        db = self._heatmap_db(self.angular_spectrum_image, angular.spectrum_power)
        levels = self._levels("angular", db, self._rd_dynamic_range_db())
        image = self.angular_spectrum_image
        same_grid = np.array_equal(image.x, angular.spectrum_azimuth_deg) and np.array_equal(
            image.y, angular.spectrum_zenith_deg
        )
        if image.z is not db or not same_grid:
            image.setData(
                None if same_grid else angular.spectrum_azimuth_deg,
                None if same_grid else angular.spectrum_zenith_deg,
                db,
                autoLevels=False,
            )
        if image.getLevels() != levels:
            image.setLevels(levels)
            self._angular_colorbars[self.angular_spectrum_plot].setLevels(values=levels)
        self.angular_spectrum_image.show()
        if not self._live_plot_is_inspected(self.angular_spectrum_plot):
            self.angular_spectrum_plot.setRange(xRange=(-90, 90), yRange=(0, 180), padding=0)
        self.angular_spectrum_plot.setTitle(
            f"Angular spectrum{variant_suffix}<br>Max over range/Doppler"
        )
        detections = sensing_block.get("detections") or []
        points = []
        for index, detection in enumerate(detections):
            az = detection.get("aoa_az_deg", detection.get("azimuth_deg"))
            el = detection.get("aoa_el_deg")
            if az is None or el is None or not np.isfinite([az, el]).all():
                continue
            if abs(az) < 90 and abs(el) < 90:
                points.append(
                    dict(
                        pos=(float(az), 90 - float(el)),
                        data=index,
                        brush=pg.mkBrush(detection_color(detection)),
                    )
                )
        self.angular_spectrum_det_scatter.setData(points)

    def _update_angular_image(
        self,
        image: pg.PColorMeshItem,
        plot: pg.PlotWidget,
        power_linear: np.ndarray,
        x_axis: np.ndarray,
        azimuth_axis: np.ndarray,
        title: str,
        x_label: str,
    ) -> None:
        """Render one angular projection using the shared dashboard color scale."""
        if (
            self._last_request is not None
            and self._last_request.recording_managed
            and not plot.isVisible()
        ):
            return
        if power_linear.size == 0 or x_axis.size == 0 or azimuth_axis.size == 0:
            image.setVisible(False)
            return

        show_heatmap = self._heatmap_enabled(plot)
        image.setVisible(show_heatmap)
        self._angular_colorbars[plot].setVisible(show_heatmap)
        if show_heatmap:
            power_db = self._heatmap_db(image, power_linear)
            vmin, vmax = self._levels("angular", power_db, self._rd_dynamic_range_db())
            x_edges = self._axis_edges(x_axis, axis_name=x_label)
            azimuth_edges = self._axis_edges(azimuth_axis, axis_name="azimuth")
            same_grid = (
                image.x is not None
                and np.array_equal(image.x[:, 0], x_edges)
                and np.array_equal(image.y[0, :], azimuth_edges)
            )
            if not same_grid:
                x_grid, azimuth_grid = np.meshgrid(x_edges, azimuth_edges, indexing="ij")
                image.setData(x_grid, azimuth_grid, power_db.T, autoLevels=False)
            elif not np.shares_memory(image.z, power_db):
                image.setData(None, None, power_db.T, autoLevels=False)
            if image.getLevels() != (vmin, vmax):
                image.setLevels((vmin, vmax))
                self._angular_colorbars[plot].setLevels(values=(vmin, vmax))
            if self._zoom_data_cb.isChecked():
                self._apply_zoom_to_data(plot, power_db, x_axis, azimuth_axis)
            elif not self._live_plot_is_inspected(plot):
                plot.getViewBox().enableAutoRange()
        else:
            self._fit_marker_map_axes(plot, x_axis, azimuth_axis)
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", "Broadside azimuth (deg)")
        plot.setTitle(title if show_heatmap else title + " · heatmap hidden")

    def _fit_marker_map_axes(
        self, plot: pg.PlotWidget, x_axis: np.ndarray, y_axis: np.ndarray
    ) -> None:
        """Fit markers or saved extents without examining a hidden power map."""
        if self._live_plot_is_inspected(plot):
            return
        if self._zoom_data_cb.isChecked():
            plot.getViewBox().enableAutoRange()
        else:
            x_edges = self._axis_edges(x_axis, axis_name="horizontal")
            y_edges = self._axis_edges(y_axis, axis_name="vertical")
            left = max(0.0, x_edges[0]) if plot is self.rd_plot and x_axis[0] >= 0 else x_edges[0]
            plot.setRange(
                xRange=(float(left), float(x_edges[-1])),
                yRange=(float(y_edges[0]), float(y_edges[-1])),
                padding=0,
            )

    def _update_angular_detection_overlays(
        self,
        sensing_block: dict,
        doppler_axis: np.ndarray,
        *,
        has_elevation: bool = False,
    ) -> None:
        """Project spherical detection angles onto the broadside display axis."""
        from shared.sensing.arrays import broadside_azimuth_projection

        detections = sensing_block.get("detections") or []
        range_xs: list[float] = []
        range_ys: list[float] = []
        doppler_xs: list[float] = []
        doppler_ys: list[float] = []
        indices: list[int] = []
        for index, det in enumerate(detections):
            az = det.get("aoa_az_deg")
            if az is None:
                az = det.get("azimuth_deg")
            if az is None:
                continue
            elevation = det.get("aoa_el_deg", np.nan) if has_elevation else 0.0
            az_f = float(broadside_azimuth_projection(az, elevation))
            if not np.isfinite(az_f):
                continue
            indices.append(index)
            range_xs.append(float(det.get("range_m", 0.0)))
            range_ys.append(az_f)
            if self._use_doppler_hz:
                doppler_x = float(det.get("doppler_hz", 0.0))
            else:
                doppler_x = float(det.get("velocity_m_s", 0.0))
            if doppler_axis.size:
                doppler_x = float(np.clip(doppler_x, np.min(doppler_axis), np.max(doppler_axis)))
            doppler_xs.append(doppler_x)
            doppler_ys.append(az_f)

        pens = [pg.mkPen(detection_color(detections[i]), width=2) for i in indices]
        brushes = [pg.mkBrush(detection_color(detections[i])) for i in indices]
        if self._live_plot_needs_update(self.range_azimuth_plot):
            self.range_azimuth_det_scatter.setData(
                range_xs, range_ys, data=indices, pen=pens, brush=brushes
            )
        if self._live_plot_needs_update(self.doppler_azimuth_plot):
            self.doppler_azimuth_det_scatter.setData(
                doppler_xs, doppler_ys, data=indices, pen=pens, brush=brushes
            )

    # ── 5. Detection SNR bars ───────────────────────────────────────────
    def _update_detection_table(self, sensing_block: dict) -> None:
        """Update the detection summary table."""
        detections = sensing_block.get("detections") or []
        config = sensing_block.get("config_resolved") or {}
        # A horizontal ULA may serialize a zero elevation as a projection
        # convention. It does not measure that coordinate. Unknown historical
        # metadata retains its saved-angle display behavior.
        self.detection_table_model.set_detections(
            detections, elevation_available=config.get("antenna_rows") != 1
        )

    @staticmethod
    def _uniform_history_period_s(
        cir_history: list[np.ndarray],
        sample_times_s: list[float],
    ) -> tuple[float | None, str | None]:
        """Validate one-to-one, strictly increasing, uniform physical timestamps."""
        if len(cir_history) != len(sample_times_s):
            return None, "CIR samples and persisted timestamps are not aligned"
        if len(sample_times_s) < 2:
            return None, "at least two persisted sample timestamps are required"
        times = np.asarray(sample_times_s, dtype=float)
        if not np.all(np.isfinite(times)):
            return None, "persisted sample timestamps must be finite"
        periods = np.diff(times)
        if not np.all(periods > 0.0):
            return None, "persisted sample timestamps must be strictly increasing"
        if not np.allclose(periods, periods[0], rtol=1e-9, atol=1e-12):
            return None, "persisted sample timestamps are nonuniform"
        return float(periods[0]), None

    @staticmethod
    def _validated_fast_time_range_axis(
        range_axis_m: np.ndarray | None,
        samples: list[np.ndarray] | tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray | None, str | None]:
        """Validate one exact typed range coordinate for every fast-time row."""
        if range_axis_m is None:
            return None, "exact typed delay-product range_axis_m is unavailable"
        axis = np.asarray(range_axis_m, dtype=np.float64)
        if axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis)):
            return None, "typed delay-product range_axis_m must be a non-empty finite 1-D axis"
        if axis.size > 1 and not np.all(np.diff(axis) > 0.0):
            return None, "typed delay-product range_axis_m must be strictly increasing"
        for sample_index, sample in enumerate(samples):
            if np.asarray(sample).shape != (axis.size,):
                return (
                    None,
                    "typed delay-product range_axis_m does not align with fast-time "
                    f"sample {sample_index}",
                )
        return axis.copy(), None

    @staticmethod
    def _same_range_axis(left: np.ndarray | None, right: np.ndarray | None) -> bool:
        """Return whether two producer-authored range axes are exactly identical."""
        if left is None or right is None:
            return left is right
        return left.shape == right.shape and bool(np.array_equal(left, right))

    def _resolve_timestamped_fast_time_history(
        self,
        sensing_block: dict[str, Any],
        request: SensingUpdateRequest,
    ) -> tuple[list[np.ndarray], list[float], str | None]:
        """Resolve order-safe live CIR history from persisted physical membership."""
        batch = extract_timestamped_fast_time_samples(sensing_block)
        range_axis_m, range_axis_error = self._validated_fast_time_range_axis(
            batch.range_axis_m,
            batch.samples,
        )
        if range_axis_error is not None:
            self._fast_time_range_axis_m = None
            if batch.expanded_acquisition:
                self._expanded_cir_history.clear()
                self._expanded_cir_times_s.clear()
                self._expanded_range_axis_m = None
            self._sequential_range_axis_m = None
            return [], [], batch.alignment_error or range_axis_error
        if batch.alignment_error is not None:
            self._expanded_cir_history.clear()
            self._expanded_cir_times_s.clear()
            self._expanded_range_axis_m = None
            self._sequential_range_axis_m = None
            self._fast_time_range_axis_m = None
            self._last_expanded_acquisition_id = batch.acquisition_id
            self._expanded_history_error = batch.alignment_error
            return [], [], batch.alignment_error

        if batch.expanded_acquisition:
            if not batch.samples:
                self._expanded_cir_history.clear()
                self._expanded_cir_times_s.clear()
                self._expanded_range_axis_m = None
                self._fast_time_range_axis_m = None
                self._last_expanded_acquisition_id = batch.acquisition_id
                self._expanded_history_error = batch.alignment_error
                return [], [], batch.alignment_error
            if (
                batch.acquisition_id is not None
                and batch.acquisition_id == self._last_expanded_acquisition_id
            ):
                if not self._same_range_axis(range_axis_m, self._expanded_range_axis_m):
                    error = "replayed acquisition changed its typed range_axis_m"
                    self._fast_time_range_axis_m = None
                    self._expanded_history_error = error
                    return [], [], error
                self._fast_time_range_axis_m = range_axis_m
                return (
                    list(self._expanded_cir_history),
                    list(self._expanded_cir_times_s),
                    self._expanded_history_error,
                )

            acquisition_samples = [
                np.asarray(sample, dtype=np.complex128) for sample in batch.samples
            ]
            acquisition_times_s = [float(value) for value in batch.sample_times_s]
            samples = list(acquisition_samples)
            times_s = list(acquisition_times_s)
            if self._expanded_cir_history and not self._same_range_axis(
                range_axis_m,
                self._expanded_range_axis_m,
            ):
                self._expanded_cir_history.clear()
                self._expanded_cir_times_s.clear()
            self._expanded_range_axis_m = range_axis_m
            if self._expanded_cir_times_s:
                last_time_s = self._expanded_cir_times_s[-1]
                if times_s[-1] <= last_time_s:
                    self._expanded_cir_history.clear()
                    self._expanded_cir_times_s.clear()
                elif times_s[0] <= last_time_s:
                    append_index = next(
                        index
                        for index, sample_time_s in enumerate(times_s)
                        if sample_time_s > last_time_s
                    )
                    existing_times = np.asarray(self._expanded_cir_times_s, dtype=float)
                    overlap_is_aligned = all(
                        np.any(
                            np.isclose(
                                existing_times,
                                sample_time_s,
                                rtol=0.0,
                                atol=1e-12,
                            )
                        )
                        for sample_time_s in times_s[:append_index]
                    )
                    if overlap_is_aligned:
                        samples = samples[append_index:]
                        times_s = times_s[append_index:]
                    else:
                        self._expanded_cir_history.clear()
                        self._expanded_cir_times_s.clear()

            self._expanded_cir_history.extend(samples)
            self._expanded_cir_times_s.extend(times_s)
            self._last_expanded_acquisition_id = batch.acquisition_id
            error: str | None = None
            if len(self._expanded_cir_times_s) >= 2:
                _period_s, error = self._uniform_history_period_s(
                    self._expanded_cir_history,
                    self._expanded_cir_times_s,
                )
            if error is not None and len(self._expanded_cir_times_s) > len(times_s):
                # A gap between coherent acquisitions does not invalidate the
                # current CPI. Restart from that internally uniform batch so the
                # dashboard can render it without implying cross-gap coherence.
                self._expanded_cir_history = acquisition_samples
                self._expanded_cir_times_s = acquisition_times_s
                _period_s, error = self._uniform_history_period_s(
                    self._expanded_cir_history,
                    self._expanded_cir_times_s,
                )
            # A seek or range-grid change starts a new history; it does not
            # invalidate the selected CPI. Retain only actual timing errors.
            self._expanded_history_error = error
            if error is not None:
                self._fast_time_range_axis_m = None
                return [], [], error
            self._fast_time_range_axis_m = self._expanded_range_axis_m
            return (
                list(self._expanded_cir_history),
                list(self._expanded_cir_times_s),
                None,
            )

        samples = list(request.cir_history or batch.samples)
        times_s = list(request.cir_history_times_s or batch.sample_times_s)
        range_axis_m, range_axis_error = self._validated_fast_time_range_axis(
            range_axis_m,
            samples,
        )
        if range_axis_error is not None:
            self._fast_time_range_axis_m = None
            return [], [], range_axis_error
        if len(samples) > len(batch.samples):
            if self._sequential_range_axis_m is None:
                self._sequential_range_axis_m = range_axis_m
            elif not self._same_range_axis(range_axis_m, self._sequential_range_axis_m):
                self._fast_time_range_axis_m = None
                return [], [], "typed range_axis_m changed within sequential CIR history"
        else:
            self._sequential_range_axis_m = range_axis_m
        self._fast_time_range_axis_m = range_axis_m
        _period_s, error = self._uniform_history_period_s(samples, times_s)
        if error is not None and len(samples) >= 2:
            return [], [], error
        return samples, times_s, error

    @staticmethod
    def _micro_spectral_settings(sensing_block: Mapping[str, Any]) -> dict[str, int]:
        """Use one complete CPI per spectrum for analytical-Doppler input.

        Independent RT solves do not establish phase continuity between CPIs.
        A non-overlapping window preserves each acquisition's spectral content.
        Simulated-Doppler uses the configured FFT length with overlapping windows.
        """
        cfg = sensing_block.get("config_resolved") or {}
        strategy = (cfg.get("processing") or {}).get("strategy")
        if strategy == "coherent_cpi_fft":
            acquisition = (sensing_block.get("temporal") or {}).get("acquisition") or {}
            window = len(acquisition.get("samples") or [])
            if window < 2:
                raise ValueError("analytical-Doppler needs a complete multi-pulse CPI")
            return {"stft_window_frames": window, "stft_hop_frames": window}
        window = max(2, int(cfg.get("fft_size_doppler", _MIN_MICRO_DOPPLER_FRAMES)))
        return {"stft_window_frames": window, "stft_hop_frames": max(1, window // 4)}

    def _update_collapsed_doppler(
        self,
        sensing_block: dict,
        prf_hz: float,
        sampling_frequency: float,
        range_mode: str,
        *,
        cir_history: Optional[List[np.ndarray]] = None,
        cir_history_times_s: Optional[List[float]] = None,
        history_error: str | None = None,
        range_axis_m: Optional[np.ndarray] = None,
    ) -> None:
        """Update blind collapsed micro-Doppler from exact typed coordinates."""
        del prf_hz, sampling_frequency, range_mode
        history = list(cir_history or [])
        sample_times_s = list(cir_history_times_s or [])
        if history_error is not None:
            self.collapsed_doppler_image.clear()
            self.collapsed_doppler_plot.setTitle(f"Collapsed Micro-Doppler ({history_error})")
            return
        frame_dt, cadence_error = self._uniform_history_period_s(history, sample_times_s)
        if cadence_error is not None or frame_dt is None:
            self.collapsed_doppler_image.clear()
            self.collapsed_doppler_plot.setTitle(
                f"Collapsed Micro-Doppler ({cadence_error or 'physical cadence unavailable'})"
            )
            return
        spectral = self._micro_spectral_settings(sensing_block)
        window_frames = spectral["stft_window_frames"]
        if not history or len(history) < window_frames:
            self.collapsed_doppler_image.clear()
            n = len(history) if history else 0
            self.collapsed_doppler_plot.setTitle(
                f"Collapsed Micro-Doppler (need {window_frames} pulses, have {n})"
            )
            return

        n_range_bins = int(np.asarray(history[0]).shape[0])
        exact_range_axis_m, axis_error = self._validated_fast_time_range_axis(
            range_axis_m if range_axis_m is not None else self._fast_time_range_axis_m,
            history,
        )
        if axis_error is not None or exact_range_axis_m is None:
            self.collapsed_doppler_image.clear()
            self.collapsed_doppler_plot.setTitle(
                f"Collapsed Micro-Doppler ({axis_error or 'typed range axis unavailable'})"
            )
            return

        physical_prf_hz = 1.0 / frame_dt
        try:
            result = compute_range_collapsed_micro_doppler(
                fast_time_sequence=history,
                prf_hz=physical_prf_hz,
                frame_duration_s=frame_dt,
                range_axis_m=exact_range_axis_m,
                **spectral,
            )
        except ValueError as exc:
            self.collapsed_doppler_image.clear()
            self.collapsed_doppler_plot.setTitle(f"Collapsed Micro-Doppler ({exc})")
            return

        if result is None or result.spectrogram.size == 0:
            self.collapsed_doppler_image.clear()
            self.collapsed_doppler_plot.setTitle("Collapsed Micro-Doppler (STFT failed)")
            return

        power_db = 10.0 * np.log10(np.maximum(result.spectrogram, 1e-30))
        vmax = float(np.max(power_db))
        vmin = vmax - self._micro_dynamic_range_db()
        t = result.time_axis_s + sample_times_s[0]
        d = result.doppler_axis_hz
        rect = (
            self._axis_rect(
                t,
                d,
                x_axis_name="micro-Doppler time",
                y_axis_name="micro-Doppler frequency",
                x_single_width=spectral["stft_hop_frames"] * frame_dt,
            )
            if t.size and d.size
            else None
        )

        # pyqtgraph ImageItem: array rows → X axis, cols → Y axis.
        # spectrogram is [time, doppler] — pass as-is for time=X, doppler=Y.
        self.collapsed_doppler_image.setImage(power_db, autoLevels=False)
        self.collapsed_doppler_image.setLevels([vmin, vmax])
        if rect is not None:
            self.collapsed_doppler_image.setRect(rect)

        title = (
            f"Collapsed Micro-Doppler (all {n_range_bins} range bins, "
            f"{window_frames} pulses/window)"
        )
        self.collapsed_doppler_plot.setTitle(title)

    # ── 6. Micro-Doppler spectrogram ────────────────────────────────────
    def _update_micro_doppler(
        self,
        sensing_block: dict,
        prf_hz: float,
        sampling_frequency: float,
        range_mode: str,
        *,
        step: Optional[int] = None,
        cir_history: Optional[List[np.ndarray]] = None,
        cir_history_times_s: Optional[List[float]] = None,
        history_error: str | None = None,
        range_axis_m: Optional[np.ndarray] = None,
    ) -> None:
        """Resolve a blind route unless the user explicitly selects the oracle route."""
        del prf_hz, sampling_frequency, range_mode

        preferred_mode = self._selected_micro_doppler_mode()
        defer_reason = self._micro_doppler_defer_reason()
        if defer_reason is not None:
            self._micro_mode_used = "deferred"
            self._micro_status = (
                f"{_ORACLE_MICRO_LABEL}: {defer_reason}"
                if preferred_mode == "tracked"
                else defer_reason
            )
            return

        if preferred_mode == "stored":
            self._clear_micro_doppler(
                "Micro-Doppler (typed stored product is not defined)",
                "stored mode unavailable: no typed product contract",
            )
            return

        if history_error is not None:
            self._clear_micro_doppler(
                f"Micro-Doppler ({history_error})",
                history_error,
            )
            return
        if cir_history is None or len(cir_history) == 0:
            self._clear_micro_doppler("Micro-Doppler (no CIR history)", "no history")
            return

        spectral = self._micro_spectral_settings(sensing_block)
        window_frames = spectral["stft_window_frames"]
        n_history = len(cir_history)
        if n_history < window_frames:
            self._clear_micro_doppler(
                f"Micro-Doppler (need {window_frames} pulses, have {n_history})",
                "insufficient history",
            )
            return

        effective_frame_dt, cadence_error = self._uniform_history_period_s(
            list(cir_history),
            list(cir_history_times_s or []),
        )
        if cadence_error is not None or effective_frame_dt is None:
            self._clear_micro_doppler(
                f"Micro-Doppler ({cadence_error or 'physical cadence unavailable'})",
                cadence_error or "cadence unavailable",
            )
            return

        exact_range_axis_m, axis_error = self._validated_fast_time_range_axis(
            range_axis_m if range_axis_m is not None else self._fast_time_range_axis_m,
            list(cir_history),
        )
        if axis_error is not None or exact_range_axis_m is None:
            reason = axis_error or "typed range axis unavailable"
            self._clear_micro_doppler(f"Micro-Doppler ({reason})", reason)
            return

        sample_times_s = list(cir_history_times_s or [])
        target_id: str | None = None
        gt_entries: list[dict[str, Any]] = []
        tracking_error: str | None = None
        can_track = False
        if preferred_mode == "tracked":
            target_id = self._select_micro_tracking_target()
            if target_id is None:
                target_count = len(self._target_history)
                tracking_error = (
                    f"{_ORACLE_MICRO_LABEL} requires exactly one canonical target_id "
                    f"(found {target_count})"
                )
            else:
                gt_entries = self._build_micro_gt_entries(sample_times_s, target_id)
                tracking_error = (
                    None
                    if len(gt_entries) == n_history
                    else f"{_ORACLE_MICRO_LABEL} requires canonical truth exactly aligned to "
                    "every cir_history_times_s sample"
                )
            can_track = tracking_error is None

        n_range_bins = int(np.asarray(cir_history[0]).shape[0])
        blind_error: str | None = None
        try:
            fixed_bin = self._resolve_micro_range_bin(
                sensing_block,
                exact_range_axis_m,
                n_range_bins,
            )
        except ValueError as exc:
            fixed_bin = None
            blind_error = str(exc)
        gate_bins = self._micro_collapse_bins(fixed_bin, n_range_bins)
        collapse_bins = (
            gate_bins if preferred_mode == "gated" else np.arange(n_range_bins, dtype=np.int64)
        )
        mode = self._resolve_micro_doppler_mode(
            preferred_mode,
            has_stored=False,
            can_track=can_track,
            can_collapse=collapse_bins.size > 0,
            can_fixed=fixed_bin is not None,
            can_gate=gate_bins.size > 0,
        )

        if mode == "unavailable":
            if preferred_mode == "tracked":
                reason = tracking_error or f"{_ORACLE_MICRO_LABEL} unavailable"
            elif preferred_mode in {"gated", "fixed-bin"}:
                reason = blind_error or f"{preferred_mode} mode unavailable"
            else:
                reasons = [reason for reason in (blind_error,) if reason]
                reason = "; ".join(reasons) or "no micro-Doppler route is available"
            self._clear_micro_doppler(f"Micro-Doppler ({reason})", reason)
            return

        physical_prf_hz = 1.0 / effective_frame_dt
        cache_key = (
            mode,
            tuple(spectral.items()),
            step,
            float(physical_prf_hz),
            float(effective_frame_dt),
            int(fixed_bin) if fixed_bin is not None else None,
            tuple(int(bin_idx) for bin_idx in collapse_bins.tolist()),
            tuple(float(value) for value in exact_range_axis_m.tolist()),
            self._micro_history_signature(cir_history),
            (
                len(cir_history_times_s or []),
                (cir_history_times_s or [None])[0],
                (cir_history_times_s or [None])[-1],
            ),
            target_id,
            self._micro_gt_signature(gt_entries),
        )
        cached_result = self._micro_result_cache.get(cache_key)
        if cached_result is not None:
            self._render_micro_doppler(
                cached_result,
                mode=mode,
                step=step,
                cached=True,
                time_bin_width_s=spectral["stft_hop_frames"] * effective_frame_dt,
            )
            return

        result: Optional[MicroDopplerResult] = None
        try:
            if mode == "tracked" and target_id is not None:
                result = compute_range_tracked_micro_doppler(
                    gt_entries,
                    target_id=target_id,
                    fast_time_sequence=cir_history,
                    sample_times_s=sample_times_s,
                    range_axis_m=exact_range_axis_m,
                    **spectral,
                )
            elif mode in {"collapsed", "gated"}:
                result = compute_range_collapsed_micro_doppler(
                    fast_time_sequence=cir_history,
                    prf_hz=physical_prf_hz,
                    frame_duration_s=effective_frame_dt,
                    range_bins=collapse_bins,
                    range_axis_m=exact_range_axis_m,
                    **spectral,
                )
            elif mode == "fixed-bin" and fixed_bin is not None:
                result = compute_micro_doppler(
                    range_bin=fixed_bin,
                    fast_time_sequence=cir_history,
                    prf_hz=physical_prf_hz,
                    frame_duration_s=effective_frame_dt,
                    range_axis_m=exact_range_axis_m,
                    **spectral,
                )
        except ValueError as exc:
            reason = str(exc)
            self._clear_micro_doppler(f"Micro-Doppler ({reason})", reason)
            return

        if result is None:
            label = (
                "Micro-Doppler (STFT failed)"
                if mode != "stored"
                else "Micro-Doppler (stored unavailable)"
            )
            self._clear_micro_doppler(label, f"{mode} failed")
            return

        if mode != "tracked":
            # The oracle helper already uses absolute physical sample times.
            result = replace(result, time_axis_s=result.time_axis_s + sample_times_s[0])
        self._cache_micro_result(cache_key, result)
        self._render_micro_doppler(
            result,
            mode=mode,
            step=step,
            time_bin_width_s=spectral["stft_hop_frames"] * effective_frame_dt,
        )

    def _render_micro_doppler(
        self,
        result: MicroDopplerResult,
        *,
        mode: str,
        step: Optional[int],
        cached: bool = False,
        time_bin_width_s: float = 1.0,
    ) -> None:
        """Render a computed micro-Doppler result with status/provenance text."""
        power_db = 10.0 * np.log10(np.maximum(result.spectrogram, 1e-30))
        vmax = float(np.max(power_db))
        vmin = vmax - self._micro_dynamic_range_db()
        t = result.time_axis_s
        d = result.doppler_axis_hz
        rect = (
            self._axis_rect(
                t,
                d,
                x_axis_name="micro-Doppler time",
                y_axis_name="micro-Doppler frequency",
                x_single_width=time_bin_width_s,
            )
            if t.size and d.size
            else None
        )

        # spectrogram is [time, doppler] — pass as-is for time=X, doppler=Y.
        self.micro_image.setImage(power_db, autoLevels=False)
        self.micro_image.setLevels([vmin, vmax])
        if rect is not None:
            self.micro_image.setRect(rect)

        display_mode = self._micro_mode_display_label(mode)
        if result.collapsed:
            title = (
                f"Micro-Doppler ({display_mode}, bins "
                f"{result.range_bin_start}-{result.range_bin_end}, "
                f"{result.range_m:.1f} m)"
            )
        elif result.range_tracking:
            title = f"Micro-Doppler ({display_mode}, {result.range_m:.1f} m)"
        else:
            title = (
                f"Micro-Doppler ({display_mode}, bin {result.range_bin}, {result.range_m:.1f} m)"
            )
        self.micro_plot.setTitle(title)
        self._micro_mode_used = mode
        step_label = f"step {step}" if step is not None else "current frame"
        status_prefix = f"{display_mode} cached" if cached else display_mode
        self._micro_status = f"{status_prefix} from {step_label}"

    def _clear_micro_doppler(self, title: str, status: str) -> None:
        """Clear the micro-Doppler image and record the reason for the summary bar."""
        if self._selected_micro_doppler_mode() == "tracked":
            if _ORACLE_MICRO_LABEL not in title:
                title = f"{title} — {_ORACLE_MICRO_LABEL}"
            if _ORACLE_MICRO_LABEL not in status:
                status = f"{_ORACLE_MICRO_LABEL}: {status}"
        self.micro_image.clear()
        self.micro_plot.setTitle(title)
        self._micro_mode_used = "unavailable"
        self._micro_status = status

    @staticmethod
    def _resolve_micro_doppler_mode(
        preferred_mode: str,
        *,
        has_stored: bool,
        can_track: bool,
        can_collapse: bool,
        can_fixed: bool,
        can_gate: bool = False,
    ) -> str:
        """Choose a route; ``has_stored`` means a validated blind typed product exists."""
        mode = (preferred_mode or "auto").strip().lower()
        availability = {
            "stored": has_stored,
            "tracked": can_track,
            "collapsed": can_collapse,
            "fixed-bin": can_fixed,
            "gated": can_gate,
        }
        if mode in availability:
            return mode if availability[mode] else "unavailable"
        if has_stored:
            return "stored"
        if can_collapse:
            return "collapsed"
        if can_fixed:
            return "fixed-bin"
        return "unavailable"

    def _resolve_micro_range_bin(
        self,
        sensing_block: dict,
        range_axis_m: np.ndarray,
        n_range_bins: int,
    ) -> Optional[int]:
        """Select a blind range bin from one explicit persisted detection."""
        if n_range_bins <= 0:
            return None
        self._micro_doppler_range_bin = auto_select_range_bin(
            [],
            sensing_block,
            n_range_bins,
            range_axis_m=range_axis_m,
        )
        return int(self._micro_doppler_range_bin)

    def _micro_collapse_bins(self, center_bin: Optional[int], n_range_bins: int) -> np.ndarray:
        """Return the configured gate of range bins around a selected center bin."""
        if center_bin is None or n_range_bins <= 0:
            return np.array([], dtype=np.int64)
        gate_half_width = int(self._micro_gate_box.value())
        self._micro_range_gate_half_width = gate_half_width
        lo = max(0, int(center_bin) - gate_half_width)
        hi = min(n_range_bins, int(center_bin) + gate_half_width + 1)
        return np.arange(lo, hi, dtype=np.int64)

    def _build_micro_gt_entries(
        self,
        sample_times_s: list[float],
        target_id: str,
    ) -> list[dict[str, Any]]:
        """Join one canonical target's truth exactly to all CIR sample epochs."""
        history = self._target_history.get(target_id)
        if not history or not sample_times_s:
            return []
        truth_times = history.get("times_s", [])
        truth_ranges = history.get("range", [])
        truth_speeds = history.get("speed", [])
        if not (
            len(truth_times) == len(truth_ranges) == len(truth_speeds) and len(truth_times) > 0
        ):
            return []

        truth_by_time: dict[float, tuple[float, float]] = {}
        for raw_time, raw_range, raw_speed in zip(
            truth_times,
            truth_ranges,
            truth_speeds,
            strict=True,
        ):
            try:
                truth_time_s = float(raw_time)
                target_range_m = float(raw_range)
                target_speed_m_s = float(raw_speed)
            except (TypeError, ValueError):
                return []
            if not all(
                math.isfinite(value) for value in (truth_time_s, target_range_m, target_speed_m_s)
            ):
                return []
            if truth_time_s in truth_by_time:
                return []
            truth_by_time[truth_time_s] = (target_range_m, target_speed_m_s)

        entries: list[dict[str, Any]] = []
        for raw_sample_time_s in sample_times_s:
            try:
                sample_time_s = float(raw_sample_time_s)
            except (TypeError, ValueError):
                return []
            target_state = truth_by_time.get(sample_time_s)
            if target_state is None:
                return []
            target_range_m, target_speed_m_s = target_state
            entries.append(
                {
                    "reference_time_s": sample_time_s,
                    "target_ids": [target_id],
                    "target_ranges": [target_range_m],
                    "target_speeds": [target_speed_m_s],
                }
            )
        return entries

    def _select_micro_tracking_target(self) -> Optional[str]:
        """Return a target ID only when the canonical truth identity is unique."""
        target_ids = sorted(self._target_history)
        return target_ids[0] if len(target_ids) == 1 else None

    # ── 7 & 8. GT Range / Speed ─────────────────────────────────────────
    @staticmethod
    def _result_history_context(
        sensing_block: Mapping[str, Any],
    ) -> tuple[tuple[str, str], float] | None:
        """Return immutable result identity and physical reference time."""
        temporal = sensing_block.get("temporal")
        if not isinstance(temporal, Mapping):
            return None
        result = temporal.get("result")
        acquisition = temporal.get("acquisition")
        if not isinstance(result, Mapping) or not isinstance(acquisition, Mapping):
            return None
        reference_time = result.get("reference_time_s")
        if not isinstance(reference_time, (int, float)) or isinstance(reference_time, bool):
            return None
        reference_time_s = float(reference_time)
        if not math.isfinite(reference_time_s):
            return None
        return (
            (
                str(result.get("processing_run_id", "")),
                str(acquisition.get("acquisition_id", "")),
            ),
            reference_time_s,
        )

    def _extract_bound_gt_entry(
        self,
        sensing_block: Mapping[str, Any],
        frame: dict | None = None,
    ) -> tuple[float, dict[str, list[Any]]] | None:
        """Return the prepared signal reference only at its exact recorded epoch."""
        temporal = sensing_block.get("temporal")
        if not isinstance(temporal, Mapping):
            return None
        truth_binding = temporal.get("truth")
        if not isinstance(truth_binding, Mapping):
            return None
        reference_time = truth_binding.get("reference_time_s")
        if not isinstance(reference_time, (int, float)) or isinstance(reference_time, bool):
            return None
        reference_time_s = float(reference_time)
        if not math.isfinite(reference_time_s):
            return None
        context = _DashboardContent._result_history_context(sensing_block)
        if context is None or context[1] != reference_time_s:
            return None
        if not isinstance(sensing_block.get("truth"), Mapping):
            return None
        request = self._last_request
        if request is not None and request.display_frame.get("sensing") is sensing_block:
            entry = request.evaluation["truth"]
        else:
            entry = extract_evaluation_truth(sensing_block, frame)
        return (reference_time_s, entry) if entry is not None else None

    def _reset_result_histories(self) -> None:
        """Clear timestamped GT/detection state after reverse, seek, or invalid input."""
        self._history_from_recording = False
        self._target_history.clear()
        self._detection_history = {"times_s": [], "range": [], "speed": []}
        self._detection_history_colors = []
        self._detection_history_styles = None
        self._micro_result_cache.clear()

    def _update_ground_truth(self, frame: dict[str, Any], *, render: bool = True) -> None:
        """Append only persisted truth/detections at the result reference epoch."""
        if self._history_from_recording:
            # Recording views hold immutable array slices. Incremental views own
            # appendable lists and must start from the newly selected result.
            self._reset_result_histories()
            self._last_history_reference_time_s = None
            self._last_history_result_identity = None
        sensing_block = frame.get("sensing") or {}
        self.gt_speed_plot.setLabel(
            "left", self._resolve_detection_velocity_display(sensing_block).axis_label
        )
        context = self._result_history_context(sensing_block)
        if context is None:
            self._reset_result_histories()
            self._last_history_reference_time_s = None
            self._last_history_result_identity = None
            if render:
                self._clear_gt_plots("no physical result reference")
            return

        result_identity, result_reference_time_s = context
        bound_gt = self._extract_bound_gt_entry(sensing_block, frame)
        if result_identity == self._last_history_result_identity:
            if render:
                self._render_gt_curves(gt_available=bound_gt is not None)
            return

        if (
            self._last_history_reference_time_s is not None
            and result_reference_time_s <= self._last_history_reference_time_s
        ):
            self._reset_result_histories()

        if bound_gt is None:
            self._target_history.clear()
        else:
            truth_reference_time_s, stored_gt = bound_gt
            target_ids = stored_gt["target_ids"]
            ranges = stored_gt["target_ranges"]
            speeds = stored_gt["target_speeds"]
            velocity_valid = stored_gt["target_velocity_valid"]
            for target_id, target_range, target_speed, speed_valid in zip(
                target_ids,
                ranges,
                speeds,
                velocity_valid,
                strict=True,
            ):
                history = self._target_history.setdefault(
                    target_id,
                    {"times_s": [], "range": [], "speed": []},
                )
                history["times_s"].append(truth_reference_time_s)
                history["range"].append(float(target_range))
                # Keep unavailable authored truth unavailable.  NumPy renders
                # the resulting None as a gap; truth-assisted processing also
                # rejects it instead of inventing a zero or finite difference.
                history["speed"].append(float(target_speed) if speed_valid else None)

        for detection in sensing_block.get("detections") or []:
            self._detection_history["times_s"].append(result_reference_time_s)
            self._detection_history["range"].append(float(detection.get("range_m", 0.0)))
            self._detection_history["speed"].append(float(detection.get("velocity_m_s", 0.0)))
            self._detection_history_colors.append(detection_color(detection))

        if render:
            self._render_gt_curves(gt_available=bound_gt is not None)
        self._last_history_reference_time_s = result_reference_time_s
        self._last_history_result_identity = result_identity

    def _render_gt_curves(self, *, gt_available: bool, apply_overlays: bool = True) -> None:
        """Render accumulated GT target curves and detection-history scatters."""
        histories = self._target_history
        active_ids = set(histories)
        range_visible = self._live_plot_needs_update(self.gt_range_plot)
        speed_visible = self._live_plot_needs_update(self.gt_speed_plot)
        if (range_visible and active_ids != set(self._gt_range_curves)) or (
            speed_visible and active_ids != set(self._gt_speed_curves)
        ):
            self._reset_gt_curve_items()

        for target_id, history in sorted(histories.items()):
            color = identity_color(target_id, truth=True)
            times_s = np.asarray(history["times_s"], dtype=float)
            ranges = np.asarray(history["range"], dtype=float)
            speeds = np.asarray(history["speed"], dtype=float)
            label = f"GT {self._get_target_label(target_id)}"

            if range_visible and target_id not in self._gt_range_curves:
                self._gt_range_curves[target_id] = self.gt_range_plot.plot(
                    [],
                    [],
                    pen=pg.mkPen(color, width=1.5),
                    symbol="+",
                    symbolSize=12,
                    symbolPen=pg.mkPen(color, width=2),
                    symbolBrush=pg.mkBrush(color),
                    name=label,
                )
            if speed_visible and target_id not in self._gt_speed_curves:
                self._gt_speed_curves[target_id] = self.gt_speed_plot.plot(
                    [],
                    [],
                    pen=pg.mkPen(color, width=1.5),
                    symbol="+",
                    symbolSize=12,
                    symbolPen=pg.mkPen(color, width=2),
                    symbolBrush=pg.mkBrush(color),
                    name=label,
                )

            if range_visible:
                self._gt_range_curves[target_id].setData(times_s, ranges)
            if speed_visible:
                self._gt_speed_curves[target_id].setData(times_s, speeds)

        self._render_detection_history()

        if not self._tracking_figures_active():
            if gt_available:
                suffix = (
                    " (ground truth hidden)"
                    if histories and not self._show_gt_cb.isChecked()
                    else ""
                )
                self.gt_range_plot.setTitle("Range: detections and ground truth" + suffix)
                self.gt_speed_plot.setTitle("Radial velocity: detections and ground truth" + suffix)
            else:
                self.gt_range_plot.setTitle("Detection range (no persisted truth)")
                self.gt_speed_plot.setTitle("Detection radial velocity (no persisted truth)")
        self.gt_range_plot.setLabel("bottom", "Result time", units="s")
        self.gt_speed_plot.setLabel("bottom", "Result time", units="s")
        if apply_overlays:
            self._apply_overlay_visibility()

    def _render_detection_history(self) -> None:
        """Keep saved identity colors aligned with the displayed history prefix."""
        history = self._detection_history
        owner = (
            self._recorded_marker_styles[0]
            if self._detection_history_styles is not None
            and self._recorded_marker_styles is not None
            else None
        )
        cached = self._recorded_scatter_state
        if (
            owner is not None
            and cached is not None
            and cached[0] is owner
            and cached[1] == len(history["times_s"])
            and cached[2] is self._gt_range_det_scatter.data
            and cached[3] is self._gt_speed_det_scatter.data
        ):
            return
        if self._detection_history_styles is None:
            styles = [self._history_marker_style(color) for color in self._detection_history_colors]
            pens = [style[0] for style in styles]
            brushes = [style[1] for style in styles]
        else:
            pens, brushes = self._detection_history_styles
        for plot, scatter, field in (
            (self.gt_range_plot, self._gt_range_det_scatter, "range"),
            (self.gt_speed_plot, self._gt_speed_det_scatter, "speed"),
        ):
            if not self._live_plot_needs_update(plot):
                continue
            scatter.setData(history["times_s"], history[field], pen=pens, brush=brushes)
        self._recorded_scatter_state = (
            owner,
            len(history["times_s"]),
            self._gt_range_det_scatter.data,
            self._gt_speed_det_scatter.data,
        )

    def _history_marker_style(self, color: str, *, selected: bool | None = None):
        """Reuse the finite identity palette so pyqtgraph can reuse marker sprites."""
        key = (color, selected)
        if key not in self._history_marker_styles:
            tint = pg.mkColor(color)
            tint.setAlphaF(0.18 if selected is False else 1.0)
            self._history_marker_styles[key] = (
                pg.mkPen(tint, width=1 if selected is False else 2),
                pg.mkBrush(tint),
            )
        return self._history_marker_styles[key]

    def _reset_gt_curve_items(self) -> None:
        """Remove existing GT curves while preserving persistent legend entries."""
        for curve in self._gt_range_curves.values():
            self.gt_range_plot.removeItem(curve)
        for curve in self._gt_speed_curves.values():
            self.gt_speed_plot.removeItem(curve)
        self._gt_range_curves.clear()
        self._gt_speed_curves.clear()

    def _clear_gt_plots(self, reason: str) -> None:
        """Clear GT plots and display why no GT curve is available."""
        self._reset_gt_curve_items()
        self._gt_range_det_scatter.setData([], [])
        self._gt_speed_det_scatter.setData([], [])
        self.gt_range_plot.setTitle(f"GT Range ({reason})")
        self.gt_speed_plot.setTitle(f"GT Speed ({reason})")

    def _update_provenance(
        self,
        sensing_block: dict,
        request: SensingUpdateRequest,
        variant: str,
    ) -> None:
        """Update the compact label that distinguishes display and state frames."""
        rd_source = sensing_block.get("rd_source") or "unknown"
        frame_role = sensing_block.get("frame_role") or "unspecified"
        display_step = request.effective_display_step
        state_step = request.state_step
        provenance_parts = [
            f"display={display_step if display_step is not None else '-'}",
            f"state={state_step if state_step is not None else '-'}",
            f"variant={variant}",
            f"rd_source={rd_source}",
            f"role={frame_role}",
        ]
        if request.display_from_cache:
            provenance_parts.append("source=cached sensing")

        display_provenance = request.display_frame.get("_sensing_display_provenance") or {}
        if isinstance(display_provenance, Mapping):
            bound_scene = display_provenance.get("bound_scene_frame_id")
            storage_frame = display_provenance.get("storage_frame_id")
            if bound_scene is not None:
                provenance_parts.append(f"bound_scene={bound_scene}")
            if storage_frame is not None:
                provenance_parts.append(f"storage_row={storage_frame}")
            time_offset_s = display_provenance.get("scene_result_time_offset_s")
            if isinstance(time_offset_s, (int, float)) and not isinstance(time_offset_s, bool):
                provenance_parts.append(f"scene_result_offset={float(time_offset_s):+.6g}s")
            stale_age_s = display_provenance.get("stale_age_s")
            if isinstance(stale_age_s, (int, float)) and not isinstance(stale_age_s, bool):
                provenance_parts.append(f"stale_age={float(stale_age_s):.6g}s")

        rd_fresh = sensing_block.get("rd_fresh")
        rp_fresh = sensing_block.get("rp_fresh")
        if rd_fresh is not None or rp_fresh is not None:
            provenance_parts.append(f"fresh(rd={bool(rd_fresh)}, rp={bool(rp_fresh)})")

        self.provenance_label.setToolTip("  |  ".join(provenance_parts))
        result = (sensing_block.get("temporal") or {}).get("result") or {}
        time_s = result.get("reference_time_s")
        time_text = f"{time_s:.3f} s" if isinstance(time_s, (int, float)) else "time unavailable"
        held = " · held result" if request.display_from_cache else ""
        self.provenance_label.setText(
            f"Result {time_text} · scene {display_step} · {variant}{held}"
        )

    # ── Summary ─────────────────────────────────────────────────────────
    def _update_summary(
        self,
        sensing_block: dict,
        variant: str,
        step: Optional[int],
        *,
        display_step: Optional[int] = None,
        display_from_cache: bool = False,
    ) -> None:
        """Update the bottom summary bar with dashboard state and sensing metadata."""
        parts = []
        if step is not None:
            parts.append(f"State frame: {step}")
        if display_step is not None:
            display_label = f"Display frame: {display_step}"
            if display_from_cache and step is not None and display_step != step:
                display_label += " (cached)"
            parts.append(display_label)
        parts.append(f"View: {self._selected_view_mode()}")
        parts.append(f"Variant: {variant}")
        parts.append(f"Micro-Doppler: {self._micro_mode_display_label(self._micro_mode_used)}")
        parts.append(f"Timeline: {self._acquisition_timeline_status}")

        processing_strategy = sensing_block.get("processing_strategy")
        if processing_strategy:
            parts.append(f"Strategy: {processing_strategy}")

        if self._extract_bound_gt_entry(sensing_block) is not None or self._target_history:
            gt_state = "evaluation shown" if self._show_gt_cb.isChecked() else "available (hidden)"
            parts.append(f"GT: {gt_state}")

        detections = sensing_block.get("detections") or []
        parts.append(f"Detections: {len(detections)}")
        if self._angular_shape is not None:
            parts.append("Angular E/A/D/R: " + "x".join(map(str, self._angular_shape)))
        parts.append(f"Angular status: {self._angular_status}")
        parts.append(f"Micro status: {self._micro_status}")
        velocity_display = self._resolve_detection_velocity_display(sensing_block)
        if velocity_display.semantics == "monostatic_equivalent_velocity":
            parts.append("Velocity: monostatic-equivalent")

        n_targets = len(self._target_history)
        if n_targets:
            parts.append(f"Targets: {n_targets}")

        if self._selected_view_mode() == "diagnostics":
            config = sensing_block.get("config") or {}
            bw = config.get("bandwidth_hz") or config.get("sampling_frequency")
            if bw:
                parts.append(f"Fs/BW: {float(bw) / 1e6:.1f} MHz")
            prf = config.get("prf_hz")
            if prf:
                parts.append(f"PRF: {float(prf):.1f} Hz")
            timing_summary = self._format_stage_profile_summary(sensing_block.get("timing"))
            if timing_summary:
                parts.append(timing_summary)

        self.summary_label.setToolTip("  |  ".join(parts))
        strategy = (
            (sensing_block.get("config_resolved") or {}).get("processing", {}).get("strategy")
        )
        acquisition_label = (
            "analytical-Doppler"
            if strategy == "coherent_cpi_fft"
            else (
                "simulated-Doppler"
                if strategy == "sequential_stft"
                else str(strategy or "Stored sensing")
            )
        )
        report_label = (
            "pre-tracking reports"
            if sensing_block.get("_display_detection_collection") == "detections_before_tracking"
            else "detections"
        )
        self.summary_label.setText(f"{acquisition_label} · {len(detections)} {report_label}")

    # ── Helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _format_stage_profile_summary(timing: Any) -> str | None:
        """Format the hottest sensing timing stages for diagnostics mode."""
        if not isinstance(timing, dict):
            return None
        stage_profile = timing.get("stage_profile_ms")
        if not isinstance(stage_profile, dict):
            return None

        numeric_items: list[tuple[str, float]] = []
        for key, value in stage_profile.items():
            if isinstance(value, (int, float)):
                numeric_items.append((str(key), float(value)))
        if not numeric_items:
            return None

        total_ms = next((value for key, value in numeric_items if key == "total_strategy"), None)
        if total_ms is None:
            total_ms = float(sum(value for key, value in numeric_items if key != "total_strategy"))

        hot_stages = sorted(
            ((key, value) for key, value in numeric_items if key != "total_strategy"),
            key=lambda item: item[1],
            reverse=True,
        )[:2]
        if hot_stages:
            stage_text = ", ".join(f"{key} {value:.1f} ms" for key, value in hot_stages)
            return f"Timing: {total_ms:.1f} ms ({stage_text})"
        return f"Timing: {total_ms:.1f} ms"

    def _resolve_detection_velocity_display(
        self,
        sensing_block: dict[str, Any],
        rd_result: Any | None = None,
    ) -> DetectionVelocityDisplay:
        """Resolve velocity-axis wording from result metadata or range-mode fallback."""
        metadata_sources: list[dict[str, Any]] = []
        rd_meta = getattr(rd_result, "metadata", None)
        if isinstance(rd_meta, dict):
            metadata_sources.append(rd_meta)
        metadata_sources.append(sensing_block)
        raw_config_resolved = sensing_block.get("config_resolved")
        config_resolved = raw_config_resolved if isinstance(raw_config_resolved, dict) else {}
        if config_resolved:
            metadata_sources.append(config_resolved)
        raw_config = sensing_block.get("config")
        config = raw_config if isinstance(raw_config, dict) else {}
        if config:
            metadata_sources.append(config)

        axis_label: str | None = None
        is_approximate: bool | None = None
        semantics: str | None = None
        for source in metadata_sources:
            if axis_label is None:
                raw_label = source.get("detection_velocity_axis_label")
                if isinstance(raw_label, str):
                    axis_label = raw_label
            if is_approximate is None:
                raw_approx = source.get("detection_velocity_is_approximate")
                if isinstance(raw_approx, bool):
                    is_approximate = raw_approx
            if semantics is None:
                raw_semantics = source.get("detection_velocity_semantics")
                if isinstance(raw_semantics, str):
                    semantics = raw_semantics

        if axis_label is None or is_approximate is None:
            range_mode = (
                sensing_block.get("range_mode")
                or config_resolved.get("range_mode")
                or config.get("range_mode")
                or self._cached_range_mode
            )
            return detection_velocity_display(range_mode=str(range_mode))

        resolved_semantics = semantics or (
            "approx_bistatic_radial_velocity"
            if is_approximate
            else detection_velocity_display(
                range_mode=str(
                    sensing_block.get("range_mode")
                    or config_resolved.get("range_mode")
                    or config.get("range_mode")
                    or self._cached_range_mode
                )
            ).semantics
        )
        return DetectionVelocityDisplay(
            semantics=resolved_semantics,
            axis_label=axis_label,
            inline_suffix=(
                " approx."
                if is_approximate
                else (
                    " mono. equiv."
                    if resolved_semantics == "monostatic_equivalent_velocity"
                    else ""
                )
            ),
            is_approximate=is_approximate,
        )

    @staticmethod
    def _doppler_to_velocity(doppler_hz: np.ndarray, carrier_freq_hz: float) -> np.ndarray:
        """Convert Doppler bins to the monostatic-equivalent velocity coordinate."""
        return doppler_axis_to_velocity_m_s(doppler_hz, carrier_freq_hz)

    @staticmethod
    def _velocity_to_doppler_hz(velocity_m_s: np.ndarray, carrier_freq_hz: float) -> np.ndarray:
        """Convert the monostatic-equivalent coordinate to Doppler frequency."""
        carrier_freq_hz = float(carrier_freq_hz)
        if carrier_freq_hz <= 0.0:
            return np.zeros_like(velocity_m_s, dtype=np.float64)
        wavelength_m = 299_792_458.0 / carrier_freq_hz
        return np.asarray(velocity_m_s, dtype=np.float64) * 2.0 / wavelength_m

    @staticmethod
    def _rd_rect(range_axis: np.ndarray, velocity_axis: np.ndarray) -> QRectF:
        """Map exact uniform range and Doppler/velocity bin centers to an image."""
        return _DashboardContent._axis_rect(
            range_axis,
            velocity_axis,
            x_axis_name="range",
            y_axis_name="Doppler/velocity",
        )

    @staticmethod
    def _axis_rect(
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        *,
        x_axis_name: str = "x",
        y_axis_name: str = "y",
        x_single_width: float = 1.0,
    ) -> QRectF:
        """Map uniform bin centers exactly and reject unsupported irregular grids.

        ``ImageItem.setRect`` is an affine transform, so interpolating between
        the endpoints of a nonuniform producer axis would silently move its
        interior bins. Nonuniform coordinates require a coordinate-aware mesh
        renderer and are rejected until that renderer is available.
        """

        def _bounds(
            axis: np.ndarray, axis_name: str, single_width: float = 1.0
        ) -> tuple[float, float]:
            coordinates = np.asarray(axis, dtype=np.float64)
            if coordinates.ndim != 1 or coordinates.size == 0:
                raise ValueError(
                    f"Cannot render {axis_name} axis: expected a non-empty 1-D coordinate array"
                )
            if not np.all(np.isfinite(coordinates)):
                raise ValueError(f"Cannot render {axis_name} axis with non-finite coordinates")
            if coordinates.size == 1:
                center = float(coordinates[0])
                return center - single_width / 2.0, center + single_width / 2.0

            spacings = np.diff(coordinates)
            if not np.all(spacings > 0.0):
                raise ValueError(
                    f"Cannot render {axis_name} axis unless coordinates are strictly increasing"
                )
            spacing = float(spacings[0])
            if not np.allclose(spacings, spacing, rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f"Cannot render nonuniform producer-authored {axis_name} axis with "
                    "pyqtgraph ImageItem; coordinate-aware rendering is required"
                )
            return float(coordinates[0] - spacing / 2.0), float(coordinates[-1] + spacing / 2.0)

        x_start, x_end = _bounds(x_axis, x_axis_name, x_single_width)
        y_start, y_end = _bounds(y_axis, y_axis_name)
        return QRectF(x_start, y_start, x_end - x_start, y_end - y_start)

    @staticmethod
    def _axis_edges(axis: np.ndarray, *, axis_name: str) -> np.ndarray:
        """Convert exact strictly increasing bin centers to cell edges."""
        coordinates = np.asarray(axis, dtype=np.float64)
        if coordinates.ndim != 1 or coordinates.size == 0:
            raise ValueError(
                f"Cannot render {axis_name} axis: expected a non-empty 1-D coordinate array"
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(f"Cannot render {axis_name} axis with non-finite coordinates")
        if coordinates.size == 1:
            center = float(coordinates[0])
            return np.array([center - 0.5, center + 0.5], dtype=np.float64)
        spacings = np.diff(coordinates)
        if not np.all(spacings > 0.0):
            raise ValueError(
                f"Cannot render {axis_name} axis unless coordinates are strictly increasing"
            )
        edges = np.empty(coordinates.size + 1, dtype=np.float64)
        edges[1:-1] = 0.5 * (coordinates[:-1] + coordinates[1:])
        edges[0] = coordinates[0] - 0.5 * spacings[0]
        edges[-1] = coordinates[-1] + 0.5 * spacings[-1]
        return edges

    def _restore_full_spectrum_axes(
        self, plot: pg.PlotWidget, range_axis: np.ndarray, velocity_axis: np.ndarray
    ) -> None:
        """Keep the linked profile within saved range cells, independent of overlays."""
        if plot is self.cfar_rd_plot or plot in self._stage_maps.values():
            # A map below the scroll viewport may not paint yet. Deferred
            # auto-range leaves it at [0, 1], including the linked threshold
            # slice. Fit the saved grid immediately, without needing a marker.
            plot.getViewBox().setRange(rect=self._rd_rect(range_axis, velocity_axis), padding=0)
        elif plot is self.rd_plot and range_axis.size:
            # X uses the saved cells. Enabling auto-X before fixing it again
            # briefly expands both this map and its linked range profile.
            plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
            edges = self._axis_edges(range_axis, axis_name="range")
            left = max(0.0, edges[0]) if range_axis[0] >= 0 else edges[0]
            plot.setXRange(float(left), float(edges[-1]), padding=0)
        else:
            plot.getViewBox().enableAutoRange()

    def _apply_zoom_to_data(
        self,
        plot: pg.PlotWidget,
        power_db: np.ndarray,
        range_axis: np.ndarray,
        velocity_axis: np.ndarray,
    ) -> None:
        """Zoom plot to the region containing significant power.

        ``power_db`` has shape ``(n_doppler, n_range)`` (slow-time × fast-time),
        matching ``RangeDopplerResult.power_linear``.  ``range_axis`` indexes
        axis-1 (columns) and ``velocity_axis`` indexes axis-0 (rows).

        When unchecked or data are degenerate, restore the full axes. The RD
        map's shared range axis fits its saved cells without overlay padding.
        """
        vb = plot.getPlotItem().getViewBox()
        if self._live_plot_is_inspected(plot):
            return
        if not self._zoom_data_cb.isChecked():
            self._restore_full_spectrum_axes(plot, range_axis, velocity_axis)
            return

        from shared.sensing.plotting.limits import range_doppler_zoom_limits

        points = []
        if plot is self.rd_plot:
            for scatter in (self.rd_det_scatter, self.rd_gt_scatter):
                if scatter.isVisible():
                    points.extend(zip(*scatter.getData()))
        limits = range_doppler_zoom_limits(
            power_db,
            range_axis,
            velocity_axis,
            dynamic_range_db=self._rd_dynamic_range_db(),
            extra_points=points,
            range_padding_bins=1.5 if plot is self.rd_plot else 0.0,
        )
        if limits is None:
            self._restore_full_spectrum_axes(plot, range_axis, velocity_axis)
            return
        vb.disableAutoRange()
        plot.setXRange(*limits[0], padding=0)
        plot.setYRange(*limits[1], padding=0)
        # Display hints are view limits only. Unchecking Zoom to data above
        # restores the full axes; the saved arrays and linked profile stay intact.
        request = self._last_request
        if plot is self.rd_plot and request is not None:
            config = request.display_frame.get("sensing", {}).get("config_resolved", {})
            for key, setter in (
                ("display_range_xlim", plot.setXRange),
                ("display_velocity_ylim", plot.setYRange),
            ):
                if key == "display_velocity_ylim" and self._use_doppler_hz:
                    continue
                bounds = config.get(key)
                if (
                    isinstance(bounds, (list, tuple))
                    and len(bounds) == 2
                    and np.isfinite(bounds).all()
                    and bounds[0] < bounds[1]
                ):
                    setter(float(bounds[0]), float(bounds[1]), padding=0)

    def _current_gt_range_speed(self) -> tuple[list[float], list[float]]:
        """Return latest GT range/speed pairs with explicitly valid velocity."""
        xs: list[float] = []
        ys: list[float] = []
        for _idx, history in sorted(self._target_history.items()):
            if not len(history["range"]) or not len(history["speed"]):
                continue
            speed = history["speed"][-1]
            if speed is None:
                continue
            try:
                range_m = float(history["range"][-1])
                speed_m_s = float(speed)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(range_m) or not math.isfinite(speed_m_s):
                continue
            xs.append(range_m)
            ys.append(speed_m_s)
        return xs, ys


class SensingDashboard(_DashboardContent, QWidget):
    """Analysis widget hosted directly by the Sensing panel or its pop-out shell."""

    view_changed = Signal()
    live_figures_changed = Signal()
    result_step_requested = Signal(int)
    result_seek_requested = Signal(object)
    selection_changed = Signal(object)
    loading_control_requested = Signal()
    locate_selection_requested = Signal()
    range_shell_changed = Signal()
    range_shell_frame_requested = Signal()
    path_inspection_requested = Signal()


class SensingWindow(_DashboardContent, QMainWindow):
    """Standalone analysis window for scripts and independent inspection."""

    view_changed = Signal()
    live_figures_changed = Signal()
    result_step_requested = Signal(int)
    result_seek_requested = Signal(object)
    selection_changed = Signal(object)
    loading_control_requested = Signal()
    locate_selection_requested = Signal()
    range_shell_changed = Signal()
    range_shell_frame_requested = Signal()
    path_inspection_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)
        self.resize(1100, 900)
