"""Service and controller construction for the visualizer window.

``build_service_bundle`` creates the app service composition from local
variables so dependencies are auditable and testable before they are exposed on
the mutable visualizer object. ``construct_services`` installs that typed bundle
on the Qt composition root and registers the scenario loader.

Keep renderer-specific setup behind ``create_renderer`` and the renderer
protocol so the rest of the composition can stay backend-neutral.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.logging import get_logger

from ..controllers.animation_controller import AnimationController
from ..controllers.beamforming_ui_controller import BeamformingUIController
from ..controllers.camera_controller import CameraController
from ..controllers.main_controller import MainController
from ..controllers.ui_controller import UIController
from ..pipeline.frame_pipeline import FramePipeline
from ..renderers.factory import create_renderer
from ..services.animation_service import AnimationService
from ..services.aperture_service import ApertureService
from ..services.cache_service import CacheService
from ..services.camera_scene_query_service import CameraSceneQueryService
from ..services.coverage_service import CoverageService
from ..services.frame_refresh_service import FrameRefreshService
from ..services.live_preview_service import LivePreviewService
from ..services.material_entry_editor import MaterialEntryEditService
from ..services.material_mode_commands import MaterialModeCommandService
from ..services.material_modes import MaterialModeService
from ..services.material_pbr_service import MaterialPBRService
from ..services.metrics_service import MetricsService
from ..services.node_service import NodeService
from ..services.object_appearance_service import ObjectAppearanceService
from ..services.override_service import OverrideService
from ..services.preset_service import PresetService
from ..services.raytracing_settings_service import RaytracingSettingsService
from ..services.rf_xray_analysis_service import RFXRayAnalysisService
from ..services.scenario_loader_service import ScenarioLoaderService
from ..services.scenario_statistics_service import ScenarioStatisticsService
from ..services.scene_appearance_service import SceneAppearanceService
from ..services.scene_edit_service import SceneEditService
from ..services.scene_service import SceneService
from ..services.session_service import SessionService
from ..services.statistics_cache_service import StatisticsCacheService
from ..services.target_asset_cache import TargetAssetCache
from ..services.target_service import TargetService
from ..services.trajectory_load_service import TrajectoryLoadCoordinator
from ..services.visual_profile_service import VisualProfileService
from .dialog_manager import DialogManager
from .selection_manager import SelectionManager

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AppServiceBundle:
    """Typed bundle of visualizer collaborators before root installation."""

    cache_service: CacheService
    coverage_service: CoverageService
    metrics_service: MetricsService
    statistics_cache_service: StatisticsCacheService
    scenario_statistics_service: ScenarioStatisticsService
    frame_refresh_service: FrameRefreshService
    raytracing_settings_service: RaytracingSettingsService
    material_mode_service: MaterialModeService
    material_mode_command_service: MaterialModeCommandService
    material_entry_edit_service: MaterialEntryEditService
    rf_xray_analysis_service: RFXRayAnalysisService
    renderer: Any
    pipeline: FramePipeline
    live_preview_service: LivePreviewService
    animation_service: AnimationService
    trajectory_load_coordinator: TrajectoryLoadCoordinator
    scene_service: SceneService
    aperture_service: ApertureService
    scene_appearance_service: SceneAppearanceService
    scene_edit_service: SceneEditService
    target_service: TargetService
    node_service: NodeService
    override_service: OverrideService
    animation_controller: AnimationController
    beamforming_ui_controller: BeamformingUIController
    camera_scene_query_service: CameraSceneQueryService
    camera_controller: CameraController
    ui_controller: UIController
    selection_manager: SelectionManager
    main_controller: MainController
    dialog_manager: DialogManager
    scenario_loader_service: ScenarioLoaderService
    session_service: SessionService
    preset_service: PresetService
    material_pbr_service: MaterialPBRService
    object_appearance_service: ObjectAppearanceService
    visual_profile_service: VisualProfileService


def build_service_bundle(viz: Any, *, default_animation_cache_size: int) -> AppServiceBundle:
    """Build services, controllers, and helpers without installing ``viz`` attrs.

    The ordering is intentional: cache/metrics/renderer services are created
    before ``FramePipeline`` and ``AnimationService`` so frame loading, live
    preview, and UI controllers share the same cache and renderer state.
    """
    # Core/cache/metrics/settings/material/renderer state.
    cache_service = CacheService(viz, max_frame_cache_size=viz._frame_loader_cache_size)
    coverage_service = CoverageService(max_cache_size=50)
    metrics_service = MetricsService(viz)
    statistics_cache_service = StatisticsCacheService(viz)
    scenario_statistics_service = ScenarioStatisticsService(statistics_cache_service)
    frame_refresh_service = FrameRefreshService(viz)
    raytracing_settings_service = RaytracingSettingsService()
    material_mode_service = MaterialModeService()
    material_mode_command_service = MaterialModeCommandService(material_mode_service)
    material_entry_edit_service = MaterialEntryEditService()
    rf_xray_analysis_service = RFXRayAnalysisService(viz)
    renderer = create_renderer(viz, viz._renderer_type)

    # Pipeline/live-preview/animation.
    pipeline = FramePipeline(
        viz,
        coverage_service=coverage_service,
        metrics_service=metrics_service,
    )
    live_preview_service = LivePreviewService(viz)
    # Keep the AnimationService around so the pipeline can reuse cached
    # frames before the visualizer makes the unified update call.
    animation_service = AnimationService(
        pipeline=pipeline,
        visualizer=viz,
        max_cached_steps=default_animation_cache_size,
        cache_service=cache_service,
    )
    trajectory_load_coordinator = TrajectoryLoadCoordinator()

    # Scene/node/object services.
    scene_service = SceneService(viz)
    aperture_service = ApertureService(viz)
    scene_appearance_service = SceneAppearanceService(viz)
    scene_edit_service = SceneEditService(viz)
    target_asset_cache = getattr(viz, "target_asset_cache", None)
    if not isinstance(target_asset_cache, TargetAssetCache):
        raise ValueError("visualizer bootstrap must provide one TargetAssetCache owner")
    target_service = TargetService(viz, target_asset_cache=target_asset_cache)
    node_service = NodeService(viz, target_service=target_service)
    override_service = OverrideService(viz)

    # Controllers and selection.
    animation_controller = AnimationController(viz, animation_service)
    beamforming_ui_controller = BeamformingUIController(viz)
    camera_scene_query_service = CameraSceneQueryService(viz)
    camera_controller = CameraController(viz, camera_scene_query_service)
    ui_controller = UIController(
        viz,
        scene_service,
        metrics_service,
        coverage_service,
        material_mode_service=material_mode_service,
        material_mode_command_service=material_mode_command_service,
        material_entry_edit_service=material_entry_edit_service,
        trajectory_load_coordinator=trajectory_load_coordinator,
    )
    selection_manager = SelectionManager(viz, logger)
    main_controller = MainController(
        scene_service=scene_service,
        ui_controller=ui_controller,
    )

    # Dialog/session/profile helpers.
    dialog_manager = DialogManager(viz)
    scenario_loader_service = ScenarioLoaderService(
        viz,
        dialog_manager,
    )
    session_service = SessionService(viz)
    preset_service = PresetService()
    material_pbr_service = MaterialPBRService(viz)
    object_appearance_service = ObjectAppearanceService(viz)
    visual_profile_service = VisualProfileService(viz)

    return AppServiceBundle(
        cache_service=cache_service,
        coverage_service=coverage_service,
        metrics_service=metrics_service,
        statistics_cache_service=statistics_cache_service,
        scenario_statistics_service=scenario_statistics_service,
        frame_refresh_service=frame_refresh_service,
        raytracing_settings_service=raytracing_settings_service,
        material_mode_service=material_mode_service,
        material_mode_command_service=material_mode_command_service,
        material_entry_edit_service=material_entry_edit_service,
        rf_xray_analysis_service=rf_xray_analysis_service,
        renderer=renderer,
        pipeline=pipeline,
        live_preview_service=live_preview_service,
        animation_service=animation_service,
        trajectory_load_coordinator=trajectory_load_coordinator,
        scene_service=scene_service,
        aperture_service=aperture_service,
        scene_appearance_service=scene_appearance_service,
        scene_edit_service=scene_edit_service,
        target_service=target_service,
        node_service=node_service,
        override_service=override_service,
        animation_controller=animation_controller,
        beamforming_ui_controller=beamforming_ui_controller,
        camera_scene_query_service=camera_scene_query_service,
        camera_controller=camera_controller,
        ui_controller=ui_controller,
        selection_manager=selection_manager,
        main_controller=main_controller,
        dialog_manager=dialog_manager,
        scenario_loader_service=scenario_loader_service,
        session_service=session_service,
        preset_service=preset_service,
        material_pbr_service=material_pbr_service,
        object_appearance_service=object_appearance_service,
        visual_profile_service=visual_profile_service,
    )


def _install_service_bundle(viz: Any, bundle: AppServiceBundle) -> None:
    """Install bundle members on ``viz`` in startup dependency order."""
    viz.cache_service = bundle.cache_service
    viz.coverage_service = bundle.coverage_service
    viz.metrics_service = bundle.metrics_service
    viz.statistics_cache_service = bundle.statistics_cache_service
    viz.scenario_statistics_service = bundle.scenario_statistics_service
    viz.frame_refresh_service = bundle.frame_refresh_service
    viz.raytracing_settings_service = bundle.raytracing_settings_service
    viz.extension_services = {}
    viz.material_mode_service = bundle.material_mode_service
    viz.material_mode_command_service = bundle.material_mode_command_service
    viz.material_entry_edit_service = bundle.material_entry_edit_service
    viz.rf_xray_analysis_service = bundle.rf_xray_analysis_service
    viz.renderer = bundle.renderer
    viz.pipeline = bundle.pipeline
    viz.live_preview_service = bundle.live_preview_service
    viz.animation_service = bundle.animation_service
    viz.trajectory_load_coordinator = bundle.trajectory_load_coordinator
    viz.scene_service = bundle.scene_service
    viz.aperture_service = bundle.aperture_service
    viz.scene_appearance_service = bundle.scene_appearance_service
    viz.scene_edit_service = bundle.scene_edit_service
    viz.target_service = bundle.target_service
    viz.node_service = bundle.node_service
    viz.override_service = bundle.override_service
    viz.animation_controller = bundle.animation_controller
    viz.beamforming_ui_controller = bundle.beamforming_ui_controller
    viz.camera_scene_query_service = bundle.camera_scene_query_service
    viz.camera_controller = bundle.camera_controller
    viz._camera_preset_save_mode = False
    viz.default_camera_view = None
    viz.default_camera_dist = None
    viz.default_camera_fov = None
    viz.ui_controller = bundle.ui_controller
    viz.selection_manager = bundle.selection_manager
    viz.main_controller = bundle.main_controller

    # Dialog/session/preset helpers are installed after controllers because they
    # register UI callbacks or wrap services that already exist.
    viz.dialog_manager = bundle.dialog_manager
    viz.scenario_loader_service = bundle.scenario_loader_service
    viz.ui_controller.register_scenario_loader(viz.scenario_loader_service)

    viz.session_service = bundle.session_service
    viz.preset_service = bundle.preset_service
    viz.material_pbr_service = bundle.material_pbr_service
    viz.object_appearance_service = bundle.object_appearance_service
    viz.visual_profile_service = bundle.visual_profile_service


def construct_services(viz: Any, *, default_animation_cache_size: int) -> None:
    """Construct and install the visualizer service bundle."""
    bundle = build_service_bundle(
        viz,
        default_animation_cache_size=default_animation_cache_size,
    )
    _install_service_bundle(viz, bundle)
