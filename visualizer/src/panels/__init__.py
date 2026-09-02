"""Reusable Qt panels for the ORCHAV visualizer.

Panel classes build widget groups and expose their child widgets through a
``widgets`` mapping. ``UIPanelManager`` owns tab/section placement and signal
wiring, while controllers/services translate widget state into app behavior.
This package exports the stable panel classes used by app composition and tests.
"""

from .animation_panel import AnimationControlsPanel
from .beam_pattern_panel import BeamPatternPanel
from .camera_panel import CameraControlPanel
from .coverage_panel import CoverageMapPanel
from .data_source import (
    FrameComparisonDialog,
    FrameTimelineWidget,
)
from .data_source_panel import DataSourcePanel
from .export_panel import ExportPanel
from .global_context_panel import GlobalContextPanel
from .live_grpc_mode_panel import LiveGrpcModePanel
from .materials_pbr_panel import MaterialsPanel
from .mpc_panel import MPCVisualizationPanel
from .nodes_panel import NodesSelectionPanel
from .object_management import NodePropertiesDialog
from .object_panel import ObjectManagementPanel
from .performance_panel import PerformancePanel
from .render_panel import RenderPanel
from .statistics_panel import StatisticsPanel
from .trajectory_preview_panel import TrajectoryPreviewPanel

__all__ = [
    "AnimationControlsPanel",
    "BeamPatternPanel",
    "CameraControlPanel",
    "CoverageMapPanel",
    "DataSourcePanel",
    "ExportPanel",
    "FrameComparisonDialog",
    "FrameTimelineWidget",
    "GlobalContextPanel",
    "MaterialsPanel",
    "MPCVisualizationPanel",
    "NodePropertiesDialog",
    "NodesSelectionPanel",
    "ObjectManagementPanel",
    "LiveGrpcModePanel",
    "PerformancePanel",
    "RenderPanel",
    "StatisticsPanel",
    "TrajectoryPreviewPanel",
]
