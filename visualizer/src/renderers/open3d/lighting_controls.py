"""Open3D IBL, skybox, and shadow controls.

The renderer exposes lighting through the shared renderer protocol, but the
actual calls are Open3D/Filament-specific.  This mixin keeps environment-light
state and native API fallbacks separate from material upload and transparency
logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shared.logging import get_logger
from shared.scenarios.paths import find_project_root

logger = get_logger("orchav.renderer_open3d.lighting_controls")


class Open3DLightingControlsMixin:
    """Mixin providing Open3D environment-light and shadow controls."""

    def _resolve_default_ibl(self) -> str:
        """Resolve the default IBL to a neutral Z-up outdoor environment."""
        try:
            root = find_project_root(Path(__file__).parent)
            ibl_path = root / "libraries" / "ibl" / "neutral_outdoor_ibl.ktx"
            skybox_path = root / "libraries" / "ibl" / "neutral_outdoor_skybox.ktx"
            if ibl_path.exists() and skybox_path.exists():
                logger.debug("Using neutral outdoor IBL: %s", ibl_path)
                return str(ibl_path)
        except (OSError, ValueError) as exc:
            logger.debug("Could not resolve default IBL path: %s", exc)
        return "default"

    def set_ibl(self, ibl_name: str = "default") -> bool:
        """Set the Open3D image-based lighting environment.

        Custom IBLs must be paired ``name_ibl.ktx`` and ``name_skybox.ktx``
        files. Open3D receives the shared base path without either suffix.
        """
        if self._o3d_vis is None:
            return False

        self._render_debug("set_ibl_request", ibl=ibl_name)
        if ibl_name == "default":
            # The bundled Z-up IBL replaces Open3D's Y-up built-in environment,
            # which misaligns the sky with Sionna RT scenes.
            ibl_name = self._resolve_default_ibl()

        if ibl_name != "default":
            path = Path(str(ibl_name))
            if path.suffix.lower() != ".ktx":
                logger.warning(
                    "IBL '%s' is not a .ktx IBL pair. Use name_ibl.ktx + name_skybox.ktx.",
                    ibl_name,
                )
                return False
            if path.name.endswith("_skybox.ktx"):
                base = path.name[:-11]
                path = path.with_name(f"{base}_ibl.ktx")
            elif not path.name.endswith("_ibl.ktx"):
                logger.warning(
                    "IBL '%s' must end with _ibl.ktx (paired with _skybox.ktx).",
                    ibl_name,
                )
                return False
            if not path.exists():
                logger.warning("IBL '%s' does not exist.", path)
                return False
            skybox = path.with_name(f"{path.name[:-8]}_skybox.ktx")
            if not skybox.exists():
                logger.warning(
                    "IBL '%s' missing matching skybox '%s'.",
                    path,
                    skybox,
                )
                return False
            # O3DVisualizer.set_ibl() expects the base path (without
            # "_ibl.ktx") so it can find both name_ibl.ktx and name_skybox.ktx.
            ibl_name = str(path)[:-8]

        try:
            self._o3d_vis.set_ibl(ibl_name)
            if not getattr(self, "_skybox_visible", True):
                self._o3d_vis.show_skybox(False)
            self._ibl_name = ibl_name
            self._post_redraw()
            self._render_debug("set_ibl_ok", ibl=self._ibl_name)
            logger.info("Set IBL to '%s'", ibl_name)
            return True
        except (RuntimeError, ValueError) as exc:
            logger.warning("Failed to set IBL '%s': %s", ibl_name, exc)
            return False

    def set_ibl_intensity(self, intensity: float) -> bool:
        """Set IBL intensity and force Open3D to repaint the Filament scene."""
        if self._o3d_vis is None:
            return False

        try:
            self._ibl_intensity = float(intensity)
            # Native call marks the scene dirty for repaint and persists
            # through mouse events (updates O3DVisualizer's internal state).
            self._o3d_vis.set_ibl_intensity(self._ibl_intensity)
            self.update_renderer()
            self._render_debug("set_ibl_intensity", intensity=intensity)
            logger.info("Set IBL intensity to %s", intensity)
            return True
        except (RuntimeError, ValueError) as exc:
            logger.warning("Failed to set IBL intensity: %s", exc)
            return False

    def set_ibl_rotation(self, yaw_degrees: float) -> bool:
        """Rotate IBL/skybox around the Z axis.

        Open3D/Filament uses a Y-up coordinate system while the scene uses
        Z-up. A 90-degree rotation around X converts between the two before
        applying the user's yaw rotation around Z.
        """
        if self._o3d_vis is None:
            return False

        self._ibl_rotation_deg = float(yaw_degrees)
        self._render_debug(
            "set_ibl_rotation_request",
            yaw_degrees=self._ibl_rotation_deg,
            has_set_ibl_rotation=hasattr(self._o3d_vis, "set_ibl_rotation"),
        )
        radians = np.deg2rad(self._ibl_rotation_deg)
        cos_val = float(np.cos(radians))
        sin_val = float(np.sin(radians))

        yaw_rotation = np.array(
            [[cos_val, -sin_val, 0.0], [sin_val, cos_val, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )

        x_rot_90 = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            dtype=float,
        )

        rotation = yaw_rotation @ x_rot_90

        try:
            if hasattr(self._o3d_vis, "set_ibl_rotation"):
                self._o3d_vis.set_ibl_rotation(rotation)
            else:
                scene_widget = self._o3d_vis.scene
                scene = scene_widget.scene if scene_widget is not None else None
                if scene is None:
                    return False
                if hasattr(scene, "set_indirect_light_rotation"):
                    try:
                        scene.set_indirect_light_rotation(rotation)
                    except TypeError:
                        scene.set_indirect_light_rotation(rotation.tolist())
                if hasattr(scene, "set_skybox_rotation"):
                    try:
                        scene.set_skybox_rotation(rotation)
                    except TypeError:
                        scene.set_skybox_rotation(rotation.tolist())
            self._post_redraw()
            logger.info("Set IBL rotation to %.1f deg", self._ibl_rotation_deg)
            return True
        except (RuntimeError, ValueError) as exc:
            logger.debug("Failed to set IBL rotation: %s", exc)
            return False

    def show_skybox(self, show: bool) -> bool:
        """Show or hide the skybox while preserving the selected IBL."""
        if self._o3d_vis is None:
            return False

        try:
            self._o3d_vis.show_skybox(show)
            self._skybox_visible = bool(show)
            self._post_redraw()
            self._render_debug("show_skybox", show=show)
            logger.info("Skybox visibility set to %s", show)
            return True
        except (RuntimeError, ValueError) as exc:
            logger.warning("Failed to set skybox visibility: %s", exc)
            return False

    def set_lighting_preset(self, preset: str) -> bool:
        """Apply an ORCHAV lighting preset as IBL name plus intensity.

        Skybox visibility is not changed here; the user controls it
        independently through the UI.
        """
        presets = {
            "default": {"ibl": "default", "intensity": 30000},
            "bright": {"ibl": "default", "intensity": 50000},
            "soft": {"ibl": "default", "intensity": 15000},
            "studio": {"ibl": "default", "intensity": 40000},
        }

        if preset not in presets:
            logger.warning("Unknown lighting preset '%s'", preset)
            return False

        cfg = presets[preset]
        self.set_ibl(cfg["ibl"])
        self.set_ibl_intensity(cfg["intensity"])
        if self._o3d_vis is not None:
            self._post_redraw()
        return True

    def set_shadowing(self, enabled: bool, shadow_type: str = "PCF") -> bool:
        """Enable or disable Filament shadow mapping.

        Uses the Filament View-level API which persists through mouse
        events, independent of O3DVisualizer's lighting profile.
        """
        self._shadows_enabled = bool(enabled)
        self._shadow_type = shadow_type
        if self._o3d_vis is None:
            return False
        scene_widget = self._o3d_vis.scene
        if scene_widget is None:
            return False
        view = getattr(scene_widget, "view", None)
        if view is None or not hasattr(view, "set_shadowing"):
            return False
        st_enum = getattr(type(view), "ShadowType", None)
        if st_enum is not None:
            st = getattr(st_enum, shadow_type, None)
            if st is not None:
                view.set_shadowing(enabled, st)
                self.update_renderer()
                return True
        view.set_shadowing(enabled)
        self.update_renderer()
        return True
