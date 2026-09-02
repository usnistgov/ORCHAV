"""Scene-level controls for the Open3D renderer backend.

This mixin owns visualizer-wide appearance controls that do not belong to a
single mesh payload: background color/images, axes, the custom ground grid,
scene shader mode, and the Open3D settings panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

from shared.logging import get_logger

logger = get_logger("orchav.renderer_open3d")


class Open3DSceneControlsMixin:
    """Own Open3D background, axes, ground, and scene shader controls."""

    def _build_ground_grid(self) -> o3d.geometry.LineSet:
        """Create the custom X-Y ground grid geometry in scene meters."""
        grid_size = 100.0
        grid_spacing = 5.0
        lines = []
        points = []

        for y in np.arange(-grid_size, grid_size + grid_spacing, grid_spacing):
            idx = len(points)
            points.append([-grid_size, y, 0.0])
            points.append([grid_size, y, 0.0])
            lines.append([idx, idx + 1])

        for x in np.arange(-grid_size, grid_size + grid_spacing, grid_spacing):
            idx = len(points)
            points.append([x, -grid_size, 0.0])
            points.append([x, grid_size, 0.0])
            lines.append([idx, idx + 1])

        ground_grid = o3d.geometry.LineSet()
        ground_grid.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        ground_grid.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        ground_grid.paint_uniform_color([0.5, 0.5, 0.5])
        return ground_grid

    def get_ibl_intensity(self) -> Optional[float]:
        """Return current IBL intensity."""
        return float(self._ibl_intensity)

    def get_ibl_name(self) -> Optional[str]:
        """Return current IBL environment name."""
        return str(self._ibl_name)

    def set_background_color(self, color: list[float]) -> None:
        """Set a solid Open3D background color."""
        if self._o3d_vis is not None:
            if len(color) == 3:
                color = [color[0], color[1], color[2], 1.0]
            self._o3d_vis.set_background(color, None)
            self._post_redraw()
            logger.debug("Open3DRenderer: Set background color to %s", color)

    def show_axes(self, show: bool, size: float = 1.0) -> bool:
        """Show or hide coordinate axes."""
        if self._o3d_vis is None:
            return False

        try:
            self._o3d_vis.show_axes = show
            self._post_redraw()
            self._request_visibility_settle_redraw("coordinate axes visibility")
            logger.info("Open3DRenderer: Axes visibility set to %s", show)
            return True
        except (RuntimeError, AttributeError) as exc:
            logger.warning("Open3DRenderer: Failed to set axes visibility: %s", exc)
            return False

    def show_ground(self, show: bool) -> bool:
        """Show or hide the backend-owned custom X-Y ground grid."""
        if self._o3d_vis is None:
            return False

        try:
            ground_name = "_custom_ground_plane"

            if show and self._ground_grid_geometry is None:
                self._ground_grid_geometry = self._build_ground_grid()

            if show:
                if ground_name not in self._geometry_names:
                    if self._ground_grid_geometry is None:
                        self._ground_grid_geometry = self._build_ground_grid()

                    line_material = rendering.MaterialRecord()
                    line_material.shader = "unlitLine"
                    line_material.line_width = 1.0
                    if not self.add_or_update_named_geometry(
                        name=ground_name,
                        geometry=self._ground_grid_geometry,
                        material=line_material,
                        is_edge=True,
                    ):
                        return False
                    logger.info("Open3DRenderer: Created custom X-Y ground plane grid")
            else:
                if ground_name in self._geometry_names:
                    self.remove_named_geometry(ground_name)
                    logger.info("Open3DRenderer: Removed custom ground plane grid")

            self._post_redraw()
            return True
        except (RuntimeError, AttributeError) as exc:
            logger.warning("Open3DRenderer: Failed to set ground plane visibility: %s", exc)
            return False

    def show_settings(self, show: bool) -> bool:
        """Show or hide O3DVisualizer's built-in settings panel."""
        if self._o3d_vis is None:
            return False
        try:
            self._o3d_vis.show_settings = bool(show)
            logger.info("O3D settings panel %s", "shown" if show else "hidden")
            return True
        except (RuntimeError, AttributeError) as exc:
            logger.warning("Failed to toggle O3D settings panel: %s", exc)
            return False

    def set_scene_shader(self, shader: str) -> bool:
        """Set the Open3D scene shader mode from the UI string vocabulary."""
        if self._o3d_vis is None:
            return False

        shader_map = {
            "standard": o3d.visualization.O3DVisualizer.Shader.STANDARD,
            "unlit": o3d.visualization.O3DVisualizer.Shader.UNLIT,
            "normals": o3d.visualization.O3DVisualizer.Shader.NORMALS,
            "depth": o3d.visualization.O3DVisualizer.Shader.DEPTH,
        }

        shader_enum = shader_map.get(shader.lower())
        if shader_enum is None:
            logger.warning("Open3DRenderer: Unknown shader '%s'", shader)
            return False

        try:
            self._o3d_vis.scene_shader = shader_enum
            self._post_redraw()
            logger.info("Open3DRenderer: Scene shader set to '%s'", shader)
            return True
        except (RuntimeError, AttributeError) as exc:
            logger.warning("Open3DRenderer: Failed to set scene shader: %s", exc)
            return False

    def set_background_image(self, image_path: str) -> bool:
        """Load a background image and fit it to the realized viewport."""
        if self._o3d_vis is None:
            return False

        try:
            path = Path(str(image_path))
            suffix = path.suffix.lower()
            data = None

            def _fit_to_viewport(img_data: np.ndarray) -> np.ndarray:
                """Crop/resize image data to the realized Open3D content rect."""
                try:
                    rect = self._o3d_vis.content_rect
                    target_w = int(rect.width)
                    target_h = int(rect.height)
                except AttributeError:
                    return img_data

                if target_w <= 0 or target_h <= 0:
                    return img_data

                src_h, src_w = img_data.shape[:2]
                if src_h == 0 or src_w == 0:
                    return img_data

                target_aspect = target_w / target_h
                src_aspect = src_w / src_h
                if abs(src_aspect - target_aspect) > 1e-3:
                    if src_aspect > target_aspect:
                        new_w = max(1, int(round(target_aspect * src_h)))
                        x0 = max(0, (src_w - new_w) // 2)
                        img_data = img_data[:, x0 : x0 + new_w]
                    else:
                        new_h = max(1, int(round(src_w / target_aspect)))
                        y0 = max(0, (src_h - new_h) // 2)
                        img_data = img_data[y0 : y0 + new_h, :]

                if img_data.shape[0] == target_h and img_data.shape[1] == target_w:
                    return img_data

                try:
                    from PIL import Image

                    return np.asarray(
                        Image.fromarray(img_data).resize(
                            (target_w, target_h), resample=Image.BILINEAR
                        )
                    )
                except (RuntimeError, ValueError):
                    y_idx = (np.linspace(0, img_data.shape[0] - 1, target_h)).astype(int)
                    x_idx = (np.linspace(0, img_data.shape[1] - 1, target_w)).astype(int)
                    return img_data[np.ix_(y_idx, x_idx)]

            if suffix in {".png", ".jpg", ".jpeg"}:
                img = o3d.io.read_image(str(path))
                if img is None or np.asarray(img).size == 0:
                    logger.warning("Open3DRenderer: Failed to load image: %s", image_path)
                    return False
                data = np.asarray(img)
            elif suffix in {".hdr", ".exr"}:
                try:
                    import imageio.v2 as imageio
                except ImportError:
                    logger.warning(
                        "Open3DRenderer: imageio not available; cannot load HDR/EXR "
                        "background: %s",
                        image_path,
                    )
                    return False
                try:
                    data = imageio.imread(str(path))
                except (OSError, ValueError) as exc:
                    logger.warning(
                        "Open3DRenderer: Failed to read HDR/EXR background '%s': %s",
                        image_path,
                        exc,
                    )
                    return False
                if data is None or np.size(data) == 0:
                    logger.warning("Open3DRenderer: Failed to load HDR/EXR image: %s", image_path)
                    return False
                data = np.asarray(data)
                if data.ndim == 2:
                    data = np.stack([data] * 3, axis=-1)
                if data.shape[-1] > 3:
                    data = data[..., :3]
                data = data.astype(np.float32)
                scale = np.percentile(data, 99)
                if not np.isfinite(scale) or scale <= 0:
                    scale = float(np.max(data)) if np.max(data) > 0 else 1.0
                data = np.clip(data / scale, 0.0, 1.0)
                data = np.power(data, 1.0 / 2.2)
                data = (data * 255.0 + 0.5).astype(np.uint8)
            else:
                logger.warning(
                    "Open3DRenderer: Unsupported background image format: %s",
                    image_path,
                )
                return False

            if data is None:
                logger.warning("Open3DRenderer: Failed to load image: %s", image_path)
                return False
            if data.ndim == 2:
                data = np.stack([data] * 3, axis=-1)
            if data.shape[-1] > 3:
                data = data[..., :3]
            if data.dtype != np.uint8:
                data = np.clip(data, 0, 255).astype(np.uint8)
            data = _fit_to_viewport(data)
            img = o3d.geometry.Image(data)

            bg_color = [0.8, 0.8, 0.8, 1.0]
            self._o3d_vis.set_background(bg_color, img)
            self._post_redraw()
            logger.info("Open3DRenderer: Set background image: %s", image_path)
            return True
        except (RuntimeError, ValueError, AttributeError, KeyError) as exc:
            logger.warning("Open3DRenderer: Failed to set background image: %s", exc)
            return False

    def clear_background_image(self) -> bool:
        """Clear the background image and revert to a solid color."""
        if self._o3d_vis is None:
            return False

        try:
            bg_color = [0.8, 0.8, 0.8, 1.0]
            self._o3d_vis.set_background(bg_color, None)
            self._post_redraw()
            logger.info("Open3DRenderer: Cleared background image")
            return True
        except (RuntimeError, AttributeError) as exc:
            logger.warning("Open3DRenderer: Failed to clear background image: %s", exc)
            return False
