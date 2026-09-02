"""Screenshot capture helpers for the Open3D renderer backend."""

from __future__ import annotations

import numpy as np

from shared.logging import get_logger

logger = get_logger("orchav.renderer_open3d")


class Open3DCaptureMixin:
    """Own Open3D screenshot export and framebuffer readback.

    O3DVisualizer framebuffers are not always current when queried directly, so
    capture uses a prioritized fallback ladder and rejects all-black frames.
    """

    def export_screenshot(
        self,
        path: str,
        *,
        resolution_scale: float = 1.0,
        include_hud: bool = False,
    ) -> bool:
        """Export the current Open3D view as a PNG or JPEG image.

        Uses ``export_screenshot_to_array`` (which forces a render pass via
        ``Application.render_to_image``) and writes the result with Pillow.
        The naive ``export_current_image`` often produces empty/black images
        because the O3DVisualizer framebuffer may not be up to date.
        """
        if self._o3d_vis is None:
            logger.warning("Open3DRenderer: Cannot export screenshot - visualizer not initialized")
            return False

        try:
            from PIL import Image

            image_array = self.export_screenshot_to_array(
                resolution_scale=resolution_scale,
                include_hud=include_hud,
            )
            if image_array is None or image_array.size == 0 or np.all(image_array == 0):
                logger.warning("Open3DRenderer: Captured image is empty, screenshot not saved")
                return False

            pil_image = Image.fromarray(image_array)
            pil_image.save(path)
            logger.info(
                "Open3DRenderer: Exported screenshot to %s (%dx%d)",
                path,
                pil_image.width,
                pil_image.height,
            )
            return True
        except (OSError, ValueError, ImportError) as exc:
            logger.error("Open3DRenderer: Failed to export screenshot: %s", exc)
            return False

    def export_screenshot_to_array(
        self,
        resolution_scale: float = 1.0,
        *,
        include_hud: bool = False,
    ) -> np.ndarray:
        """Capture the current view as an RGB ``uint8`` array.

        The preferred Open3D GUI path renders at the requested scale.
        Framebuffer and desktop compatibility fallbacks use native size and are
        resized so export/video callers still receive the requested dimensions.
        ``include_hud`` is accepted for backend parity; Open3D currently has
        no renderer-owned Qt HUD to composite.
        """
        if self._o3d_vis is None:
            logger.warning("Open3DRenderer: Cannot capture screen - visualizer not initialized")
            return np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            scale = self._normalize_resolution_scale(resolution_scale)
            self._post_redraw()

            # Preferred path: render through Open3D's GUI application context.
            scene = self._o3d_vis.scene
            if scene is not None:
                try:
                    import open3d as o3d

                    app = o3d.visualization.gui.Application.instance
                    width, height = self._capture_dimensions(scale=scale)

                    o3d_image = app.render_to_image(scene, width, height)
                    if o3d_image is not None:
                        image = self._coerce_rgb8_image(o3d_image)
                        if image is not None and not self._is_effectively_black(image):
                            logger.debug(
                                f"Open3DRenderer: Screenshot captured via Application.render_to_image ({width}x{height})"
                            )
                            return image
                except (RuntimeError, ValueError) as e:
                    logger.debug(f"Open3DRenderer: Application.render_to_image failed: {e}")

            # Older Open3D capture APIs are useful fallbacks but may return
            # stale or black frames depending on platform/window state.
            if hasattr(self._o3d_vis, "export_current_image"):
                try:
                    import os
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_path = tmp.name
                    self._o3d_vis.export_current_image(tmp_path)
                    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                        from PIL import Image

                        pil_image = Image.open(tmp_path)
                        image = self._coerce_rgb8_image(pil_image)
                        os.remove(tmp_path)
                        if image is not None and not self._is_effectively_black(image):
                            logger.debug(
                                "Open3DRenderer: Screenshot captured via export_current_image"
                            )
                            return self._resize_capture_if_needed(image, scale)
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except (OSError, ValueError) as e:
                    logger.debug(f"Open3DRenderer: export_current_image failed: {e}")

            if hasattr(self._o3d_vis, "capture_screen_float_buffer"):
                try:
                    image = self._coerce_rgb8_image(
                        self._o3d_vis.capture_screen_float_buffer(do_render=True)
                    )
                    if image is not None and not self._is_effectively_black(image):
                        logger.debug(
                            "Open3DRenderer: Screenshot captured via capture_screen_float_buffer"
                        )
                        return self._resize_capture_if_needed(image, scale)
                except (RuntimeError, ValueError) as e:
                    logger.debug(f"Open3DRenderer: capture_screen_float_buffer failed: {e}")

            if hasattr(self._o3d_vis, "capture_screen_image"):
                try:
                    import os
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_path = tmp.name
                    self._o3d_vis.capture_screen_image(tmp_path, do_render=True)
                    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                        from PIL import Image

                        pil_image = Image.open(tmp_path)
                        image = self._coerce_rgb8_image(pil_image)
                        os.remove(tmp_path)
                        if image is not None and not self._is_effectively_black(image):
                            logger.debug(
                                "Open3DRenderer: Screenshot captured via capture_screen_image"
                            )
                            return self._resize_capture_if_needed(image, scale)
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except (OSError, ValueError) as e:
                    logger.debug(f"Open3DRenderer: capture_screen_image failed: {e}")

            # Last resort: capture only the visible Open3D window content. If
            # Open3D cannot provide a screen-space content rect, skip desktop
            # capture rather than exporting the user's whole primary monitor.
            try:
                import mss

                region = self._content_rect_capture_region()
                if region is None:
                    raise RuntimeError("Open3D content rect unavailable for mss capture")
                with mss.mss() as sct:
                    screenshot = sct.grab(region)
                    image = self._coerce_rgb8_image(screenshot, bgr=True)
                    if image is not None and not self._is_effectively_black(image):
                        logger.debug("Open3DRenderer: Screenshot captured via mss content rect")
                        return self._resize_capture_if_needed(image, scale)
            except ImportError:
                logger.debug("Open3DRenderer: mss not available for screen capture")
            except (RuntimeError, ValueError) as e:
                logger.debug(f"Open3DRenderer: mss screen capture failed: {e}")

            try:
                from PIL import ImageGrab

                bbox = self._content_rect_imagegrab_bbox()
                if bbox is None:
                    raise RuntimeError("Open3D content rect unavailable for ImageGrab capture")
                screenshot = ImageGrab.grab(bbox=bbox)
                image = self._coerce_rgb8_image(screenshot)
                if image is not None and not self._is_effectively_black(image):
                    logger.debug(
                        "Open3DRenderer: Screenshot captured via PIL ImageGrab content rect"
                    )
                    return self._resize_capture_if_needed(image, scale)
            except ImportError:
                logger.debug("Open3DRenderer: PIL ImageGrab not available")
            except (RuntimeError, ValueError) as e:
                logger.debug(f"Open3DRenderer: PIL ImageGrab failed: {e}")

            logger.warning(
                "Open3DRenderer: All screenshot capture methods returned black/empty image"
            )
            return np.zeros((480, 640, 3), dtype=np.uint8)

        except (RuntimeError, ValueError, AttributeError, KeyError):
            logger.exception("Open3DRenderer: Failed to capture screen to array")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    @staticmethod
    def _normalize_resolution_scale(resolution_scale: float) -> float:
        """Return a positive finite export scale with ``1.0`` as fallback."""
        try:
            scale = float(resolution_scale)
        except (TypeError, ValueError):
            return 1.0
        if not np.isfinite(scale) or scale <= 0.0:
            return 1.0
        return scale

    def _capture_dimensions(self, *, scale: float = 1.0) -> tuple[int, int]:
        """Return requested Open3D render-to-image dimensions."""
        width = 800
        height = 600
        try:
            content_rect = self._o3d_vis.content_rect
            if content_rect is not None:
                rect_width = self._rect_attr(content_rect, "width")
                rect_height = self._rect_attr(content_rect, "height")
                if rect_width is not None and float(rect_width) > 0:
                    width = int(float(rect_width))
                if rect_height is not None and float(rect_height) > 0:
                    height = int(float(rect_height))
        except (AttributeError, TypeError, ValueError):
            logger.debug("Content rect not available for screenshot dimensions")

        scale = self._normalize_resolution_scale(scale)
        return (
            max(int(round(width * scale)), 1),
            max(int(round(height * scale)), 1),
        )

    @staticmethod
    def _coerce_rgb8_image(raw: object, *, bgr: bool = False) -> np.ndarray | None:
        """Normalize an image-like object to contiguous RGB uint8 data."""
        try:
            image = np.asarray(raw)
        except (TypeError, ValueError):
            return None
        if image.size == 0:
            return None
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.ndim == 3 and image.shape[2] >= 4:
            if bgr:
                image = image[:, :, [2, 1, 0]]
            else:
                image = image[:, :, :3]
        elif image.ndim == 3 and image.shape[2] >= 3:
            if bgr:
                image = image[:, :, [2, 1, 0]]
            else:
                image = image[:, :, :3]
        else:
            return None

        if image.dtype != np.uint8:
            max_val = float(np.nanmax(image))
            if max_val <= 1.0:
                image = np.clip(image, 0.0, 1.0) * 255.0
            image = np.asarray(np.clip(image, 0.0, 255.0), dtype=np.uint8)
        return np.ascontiguousarray(image, dtype=np.uint8)

    @staticmethod
    def _is_effectively_black(image: np.ndarray) -> bool:
        """Return True when capture data has no visible rendered content."""
        if image.size == 0:
            return True
        return int(np.max(image)) <= 1

    @staticmethod
    def _resize_capture_if_needed(image: np.ndarray, scale: float) -> np.ndarray:
        """Resize fallback captures to match the requested export scale."""
        scale = Open3DCaptureMixin._normalize_resolution_scale(scale)
        if abs(scale - 1.0) <= 1e-6:
            return image
        try:
            from PIL import Image

            pil = Image.fromarray(image)
            width = max(int(round(pil.width * scale)), 1)
            height = max(int(round(pil.height * scale)), 1)
            resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
            return np.asarray(pil.resize((width, height), resample=resample), dtype=np.uint8)
        except (ImportError, OSError, ValueError):
            return image

    def _content_rect_capture_region(self) -> dict[str, int] | None:
        """Return an ``mss`` region for the Open3D content rect, when known."""
        rect = self._screen_content_rect()
        if rect is None:
            return None
        left, top, width, height = rect
        return {"left": left, "top": top, "width": width, "height": height}

    def _content_rect_imagegrab_bbox(self) -> tuple[int, int, int, int] | None:
        """Return a PIL ImageGrab bbox for the Open3D content rect, when known."""
        rect = self._screen_content_rect()
        if rect is None:
            return None
        left, top, width, height = rect
        return (left, top, left + width, top + height)

    def _screen_content_rect(self) -> tuple[int, int, int, int] | None:
        """Resolve the Open3D content rect as screen coordinates."""
        try:
            content_rect = self._o3d_vis.content_rect
        except AttributeError:
            return None
        if content_rect is None:
            return None

        left = self._rect_attr(content_rect, "x", "left")
        top = self._rect_attr(content_rect, "y", "top")
        width = self._rect_attr(content_rect, "width")
        height = self._rect_attr(content_rect, "height")
        if left is None or top is None or width is None or height is None:
            return None
        try:
            left_i = int(round(float(left)))
            top_i = int(round(float(top)))
            width_i = int(round(float(width)))
            height_i = int(round(float(height)))
        except (TypeError, ValueError):
            return None
        if width_i <= 0 or height_i <= 0:
            return None
        return (left_i, top_i, width_i, height_i)

    @staticmethod
    def _rect_attr(rect: object, *names: str) -> object | None:
        """Read the first present rect attribute from Open3D/Qt-like rects."""
        for name in names:
            if hasattr(rect, name):
                value = getattr(rect, name)
                return value() if callable(value) else value
        return None
