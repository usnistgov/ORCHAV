"""Top-level controller facade for cross-service visualizer workflows."""

from __future__ import annotations

from typing import Any

from ..controllers.ui_controller import UIController
from ..services.scene_service import SceneService


class MainController:
    """Coordinate high-level workflows without owning implementation policy.

    Scene loading, animation preloading, metrics display, and UI refresh all
    remain delegated to their focused services/controllers. This class gives
    app wiring one stable place to invoke those workflows.
    """

    def __init__(
        self,
        scene_service: SceneService,
        ui_controller: UIController,
    ) -> None:
        """Store the scene and UI owners used by the load workflow."""
        self.scene_service = scene_service
        self.ui_controller = ui_controller

    def load_scene(
        self,
        xml_path: str,
        *,
        render_immediately: bool = True,
        cleanup_first: bool = True,
    ) -> None:
        """Load a scene and refresh dependent UI panels."""
        self.scene_service.load_scene(
            xml_path,
            render_immediately=render_immediately,
            cleanup_first=cleanup_first,
        )
        self.ui_controller.refresh_scene_panels()

    def load_prepared_scene(
        self,
        xml_path: str,
        xml_root: Any,
        mesh_entries: list[dict[str, Any]],
        *,
        render_immediately: bool = True,
        cleanup_first: bool = True,
    ) -> None:
        """Install a scene payload validated before active-scene teardown."""

        self.scene_service.load_prepared_scene(
            xml_path,
            xml_root,
            mesh_entries,
            render_immediately=render_immediately,
            cleanup_first=cleanup_first,
        )
        self.ui_controller.refresh_scene_panels()
