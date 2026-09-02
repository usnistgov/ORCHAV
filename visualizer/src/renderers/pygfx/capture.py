"""pygfx screenshot export and canvas readback helpers.

Readback support differs across rendercanvas/wgpu versions. The capture path
tries renderer snapshots first, then canvas/context methods, then a Qt widget
grab fallback, returning an RGB ``uint8`` array for video/export callers.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

__all__ = ["PygfxCaptureMixin"]

logger = logging.getLogger(__name__)


class PygfxCaptureMixin:
    """Screenshot capture behavior for ``PygfxRenderer``."""

    def export_screenshot_to_array(
        self,
        resolution_scale: float = 1.0,
        *,
        include_hud: bool = False,
    ) -> np.ndarray:
        """Return the current rendered frame as an RGB uint8 image.

        Direct GPU/canvas readback intentionally excludes Qt HUD widgets.
        ``include_hud=True`` composites the visible renderer-owned Qt widgets
        over that readback.  The GPU image remains authoritative because
        native direct-screen canvases are not reliably included by
        ``QWidget.grab()``. ``resolution_scale`` is a post-capture bilinear
        resize; it changes output dimensions without increasing rendered scene
        detail.
        """
        if not self._initialized:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        raw = self._capture_canvas_frame(include_hud=include_hud)
        image = self._coerce_rgb8_image(raw)
        if image is None:
            h = max(int(self._height), 1)
            w = max(int(self._width), 1)
            return np.zeros((h, w, 3), dtype=np.uint8)
        scale = float(resolution_scale)
        if abs(scale - 1.0) <= 1e-6:
            return image
        if scale <= 0.0:
            return image
        try:
            from PIL import Image

            pil = Image.fromarray(image)
            width = max(int(round(pil.width * scale)), 1)
            height = max(int(round(pil.height * scale)), 1)
            resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
            return np.asarray(pil.resize((width, height), resample=resample), dtype=np.uint8)
        except Exception:
            return image

    def export_screenshot(
        self,
        path: str,
        *,
        resolution_scale: float = 1.0,
        include_hud: bool = False,
    ) -> bool:
        """Write the current rendered frame to an image file."""
        try:
            from PIL import Image
        except ImportError:
            logger.warning("PygfxRenderer: Pillow is required for screenshot export")
            return False
        if not self._initialized:
            logger.warning("PygfxRenderer: cannot export a screenshot before initialization")
            return False
        image = self.export_screenshot_to_array(
            resolution_scale=resolution_scale,
            include_hud=include_hud,
        )
        if image is None or image.size == 0:
            logger.warning("PygfxRenderer: screenshot readback returned no image")
            return False
        if self._is_effectively_black(image):
            logger.warning("PygfxRenderer: screenshot readback returned an all-black image")
            return False
        try:
            Image.fromarray(image).save(path)
            return True
        except Exception as exc:
            logger.warning("PygfxRenderer: screenshot export failed: %s", exc)
            return False

    def _capture_canvas_frame(self, *, include_hud: bool = False) -> Any:
        """Capture one frame using the best available pygfx/canvas readback API."""
        if self._canvas is None:
            return None
        first_valid: Any = None

        def _usable_candidate(candidate: Any) -> Any:
            """Accept visible pixels while remembering the first black fallback."""
            nonlocal first_valid
            img = self._extract_image_candidate(candidate)
            if img is None:
                return None
            coerced = self._coerce_rgb8_image(img)
            if coerced is None:
                return None
            if first_valid is None:
                first_valid = img
            if not self._is_effectively_black(coerced):
                return img
            return None

        # The explicit renderer snapshot is authoritative. Evaluate later APIs
        # only when it cannot produce visible pixels because canvas draw/present
        # fallbacks may be both expensive and stateful.
        selected = _usable_candidate(self._capture_renderer_snapshot(include_hud=include_hud))

        if selected is None:
            minimap_was_enabled = bool(getattr(self, "_minimap_enabled", False))
            suppress_minimap = minimap_was_enabled and not include_hud
            if suppress_minimap:
                # Canvas draw fallbacks execute normal presentation, so suppress
                # the GPU minimap while preparing a clean fallback buffer.
                self._minimap_enabled = False
            try:
                self.update_renderer()
                for meth in ("snapshot", "draw"):
                    fn = getattr(self._canvas, meth, None)
                    if not callable(fn):
                        continue
                    try:
                        selected = _usable_candidate(fn())
                    except Exception:
                        continue
                    if selected is not None:
                        break

                if selected is None:
                    force_draw = getattr(self._canvas, "force_draw", None)
                    if callable(force_draw):
                        try:
                            selected = _usable_candidate(force_draw())
                        except Exception:
                            selected = None

                if selected is None:
                    get_context = getattr(self._canvas, "get_context", None)
                    if callable(get_context):
                        # rendercanvas has exposed bitmap and wgpu contexts.
                        for ctx_type in ("bitmap", "wgpu"):
                            try:
                                ctx = get_context(ctx_type)
                            except Exception:
                                continue
                            for meth in ("get_bitmap", "present"):
                                fn = getattr(ctx, meth, None)
                                if not callable(fn):
                                    continue
                                try:
                                    selected = _usable_candidate(fn())
                                except Exception:
                                    continue
                                if selected is not None:
                                    break
                            if selected is not None:
                                break

                if selected is None:
                    qt_grab = self._capture_qt_widget_frame()
                    if qt_grab is not None:
                        selected = qt_grab
            finally:
                if suppress_minimap:
                    self._minimap_enabled = True
                    request_redraw = getattr(self, "request_redraw", None)
                    if callable(request_redraw):
                        request_redraw()

        if selected is None and first_valid is not None:
            selected = first_valid

        if not include_hud:
            return selected
        base = self._coerce_rgb8_image(selected)
        if base is None:
            return selected
        return self._composite_visible_qt_overlays(base)

    def _capture_renderer_snapshot(self, *, include_hud: bool = False) -> Any:
        """Synchronously render and read the pygfx renderer color buffer."""
        renderer = getattr(self, "_renderer", None)
        scene = getattr(self, "_scene", None)
        camera = getattr(self, "_camera", None)
        if renderer is None or scene is None or camera is None:
            return None

        try:
            update_headlight = getattr(self, "_update_headlight_pose", None)
            if callable(update_headlight):
                update_headlight()
        except Exception:
            logger.debug("PygfxRenderer: headlight update before screenshot failed", exc_info=True)

        try:
            minimap_drawn = False
            if include_hud and bool(getattr(self, "_minimap_enabled", False)):
                renderer.render(scene, camera, flush=False)
                render_minimap = getattr(self, "_render_minimap", None)
                if callable(render_minimap):
                    try:
                        minimap_drawn = bool(render_minimap())
                    except Exception as exc:
                        logger.debug(
                            "PygfxRenderer: minimap render before screenshot failed: %s",
                            exc,
                        )
            if not minimap_drawn:
                renderer.render(scene, camera, flush=True)
        except Exception as exc:
            logger.debug("PygfxRenderer: synchronous screenshot render failed: %s", exc)
            return None

        snapshot = getattr(renderer, "snapshot", None)
        if not callable(snapshot):
            return None
        try:
            return snapshot()
        except Exception as exc:
            logger.debug("PygfxRenderer: renderer snapshot failed: %s", exc)
            return None

    @staticmethod
    def _extract_image_candidate(candidate: Any) -> Any:
        """Unwrap image-like values returned by renderer/canvas snapshots."""
        if candidate is None:
            return None
        if isinstance(candidate, dict):
            for key in ("data", "bitmap", "image", "array"):
                if key in candidate:
                    return candidate[key]
            return None
        if isinstance(candidate, tuple) and len(candidate) > 0:
            return candidate[0]
        return candidate

    @staticmethod
    def _coerce_rgb8_image(raw: Any) -> Optional[np.ndarray]:
        """Normalize renderer/canvas output to contiguous RGB uint8 arrays."""
        if raw is None:
            return None
        try:
            image = np.asarray(raw)
        except Exception:
            return None
        if image.size == 0:
            return None
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.ndim == 3 and image.shape[2] >= 4:
            image = image[:, :, :3]
        elif image.ndim != 3 or image.shape[2] < 3:
            return None
        if image.dtype != np.uint8:
            max_val = float(np.nanmax(image))
            if max_val <= 1.0:
                image = np.clip(image, 0.0, 1.0) * 255.0
            image = np.asarray(np.clip(image, 0.0, 255.0), dtype=np.uint8)
        return np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)

    @staticmethod
    def _is_effectively_black(image: np.ndarray) -> bool:
        """Return True when an RGB image has no visible rendered content."""
        if image.size == 0:
            return True
        return int(np.max(image)) <= 1

    @staticmethod
    def _qimage_to_rgba8(image: Any) -> Optional[np.ndarray]:
        """Copy a QImage-compatible value into a contiguous RGBA uint8 array."""
        try:
            from PySide6.QtGui import QImage
        except Exception:
            return None
        if image is None or image.isNull():
            return None
        try:
            converted = image.convertToFormat(QImage.Format_RGBA8888)
            width = int(converted.width())
            height = int(converted.height())
            if width <= 0 or height <= 0:
                return None

            ptr = converted.bits()
            if ptr is None:
                return None
            size_fn = getattr(converted, "sizeInBytes", None)
            if not callable(size_fn):
                size_fn = getattr(converted, "byteCount", None)
            if not callable(size_fn):
                return None
            byte_count = int(size_fn())
            set_size = getattr(ptr, "setsize", None)
            if callable(set_size):
                # PyQt exposes a sip.voidptr that must be sized explicitly.
                # Current PySide6 instead returns an already-sized memoryview.
                set_size(byte_count)
            bytes_per_line = int(converted.bytesPerLine())
            required_bytes = height * bytes_per_line
            if byte_count < required_bytes or bytes_per_line % 4:
                return None
            flat = np.frombuffer(ptr, dtype=np.uint8, count=byte_count)
            if flat.size < required_bytes:
                return None
            rgba = flat[:required_bytes].reshape((height, bytes_per_line // 4, 4))
            # QImage owns ``ptr``; force a copy before ``converted`` is released.
            return np.array(rgba[:, :width, :4], dtype=np.uint8, copy=True, order="C")
        except Exception:
            return None

    @classmethod
    def _render_qt_widget_rgba(
        cls,
        widget: Any,
        *,
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        """Render one widget onto a transparent QImage at capture resolution."""
        try:
            from PySide6.QtCore import QPoint, Qt
            from PySide6.QtGui import QImage, QPainter, QRegion
            from PySide6.QtWidgets import QWidget
        except Exception:
            logger.debug("PygfxRenderer: Qt HUD capture imports failed", exc_info=True)
            return None

        target_width = int(width)
        target_height = int(height)
        try:
            source_width = int(widget.width())
            source_height = int(widget.height())
        except Exception:
            logger.debug("PygfxRenderer: Qt HUD widget render failed", exc_info=True)
            return None
        if min(target_width, target_height, source_width, source_height) <= 0:
            return None

        image = QImage(target_width, target_height, QImage.Format_RGBA8888)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        try:
            painter.scale(
                float(target_width) / float(source_width),
                float(target_height) / float(source_height),
            )
            # QLabel paints its stylesheet background in paintEvent. Asking
            # QWidget.render() to draw the window background as well applies a
            # translucent stylesheet twice and corrupts its intended alpha.
            widget.render(
                painter,
                QPoint(0, 0),
                QRegion(),
                QWidget.RenderFlag.DrawChildren,
            )
        except Exception:
            logger.debug("PygfxRenderer: Qt HUD widget render failed", exc_info=True)
            return None
        finally:
            painter.end()
        return cls._qimage_to_rgba8(image)

    @staticmethod
    def _alpha_composite_rgba(
        base: np.ndarray,
        overlay: np.ndarray,
        *,
        left: int,
        top: int,
    ) -> None:
        """Alpha-composite ``overlay`` into ``base`` in place with clipping."""
        base_height, base_width = base.shape[:2]
        overlay_height, overlay_width = overlay.shape[:2]
        dst_left = max(int(left), 0)
        dst_top = max(int(top), 0)
        dst_right = min(int(left) + overlay_width, base_width)
        dst_bottom = min(int(top) + overlay_height, base_height)
        if dst_left >= dst_right or dst_top >= dst_bottom:
            return

        src_left = dst_left - int(left)
        src_top = dst_top - int(top)
        src_right = src_left + (dst_right - dst_left)
        src_bottom = src_top + (dst_bottom - dst_top)
        source = overlay[src_top:src_bottom, src_left:src_right]
        destination = base[dst_top:dst_bottom, dst_left:dst_right]

        alpha = source[:, :, 3:4].astype(np.uint32)
        source_rgb = source[:, :, :3].astype(np.uint32)
        destination_rgb = destination.astype(np.uint32)
        blended = source_rgb * alpha + destination_rgb * (255 - alpha)
        destination[:] = ((blended + 127) // 255).astype(np.uint8)

    def _composite_visible_qt_overlays(self, base: np.ndarray) -> np.ndarray:
        """Composite visible HUD labels and the tooltip over a GPU RGB frame."""
        result = np.array(base, dtype=np.uint8, copy=True, order="C")
        container = getattr(self, "_container", None)
        if container is None:
            return result

        try:
            from PySide6.QtCore import QPoint
        except Exception:
            return result

        try:
            logical_width = int(container.width())
            logical_height = int(container.height())
        except Exception:
            return result
        if logical_width <= 0 or logical_height <= 0:
            return result

        capture_height, capture_width = result.shape[:2]
        scale_x = float(capture_width) / float(logical_width)
        scale_y = float(capture_height) / float(logical_height)

        labels = getattr(self, "_hud_overlay_labels", {})
        widgets = list(labels.values()) if isinstance(labels, dict) else []
        tooltip = getattr(self, "_tooltip_label", None)
        if tooltip is not None:
            # Tooltip widgets are raised above the persistent HUD labels.
            widgets.append(tooltip)

        seen: set[int] = set()
        for widget in widgets:
            if widget is None or id(widget) in seen:
                continue
            seen.add(id(widget))
            try:
                if not widget.isVisible():
                    continue
                origin = widget.mapTo(container, QPoint(0, 0))
                widget_width = int(widget.width())
                widget_height = int(widget.height())
                left = int(round(float(origin.x()) * scale_x))
                top = int(round(float(origin.y()) * scale_y))
                right = int(round(float(origin.x() + widget_width) * scale_x))
                bottom = int(round(float(origin.y() + widget_height) * scale_y))
            except Exception:
                continue
            overlay = self._render_qt_widget_rgba(
                widget,
                width=max(right - left, 1),
                height=max(bottom - top, 1),
            )
            if overlay is None:
                continue
            self._alpha_composite_rgba(
                result,
                overlay,
                left=left,
                top=top,
            )
        return result

    def _capture_qt_widget_frame(self) -> Optional[np.ndarray]:
        """Use a canvas-widget grab only as a last-resort readback fallback."""
        widget = self._canvas_widget
        if widget is None:
            return None
        try:
            from PySide6.QtWidgets import QApplication
        except Exception:
            return None

        app = QApplication.instance()
        if app is not None:
            try:
                app.processEvents()
            except Exception:
                pass

        try:
            pixmap = widget.grab()
            if pixmap is None or pixmap.isNull():
                return None
            image = pixmap.toImage()
            if image is None or image.isNull():
                return None
            rgba = self._qimage_to_rgba8(image)
            if rgba is None:
                return None
            rgb = np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)
            if self._is_effectively_black(rgb):
                return None
            return rgb
        except Exception:
            return None
