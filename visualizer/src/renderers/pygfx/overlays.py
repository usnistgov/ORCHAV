"""pygfx minimap and HUD presentation helpers.

The minimap scene, chrome, and markers stay on the GPU so direct-screen
presentation does not depend on Qt alpha-compositing over a native surface.
Compact status chips, legends, colorbars, and hover tooltips remain Qt widgets.
Their state is derived from renderer state, frame metadata, and app-state
toggles.
"""

from __future__ import annotations

import logging
from html import escape
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from PySide6.QtCore import Qt

from ...coverage.analysis import (
    coverage_metric_color_scale,
    coverage_metric_colormap,
    coverage_metric_label,
    format_coverage_value,
    is_serving_tx_metric,
    serving_tx_color_hex,
    serving_tx_labels,
)
from ...extensions import runtime_status_chips
from ...services.mpc_interaction_style_service import mpc_interaction_legend_entries
from ...services.viewport_hud_service import (
    build_path_filter_summary,
    build_trajectory_hud_legend,
    viewport_hud_policy,
)
from ...utils.colors import ensure_viridis_lut
from .camera import SceneBounds
from .canvas import _env_flag
from .mpc import INTERACTION_MARKER_SPECS, UNKNOWN_INTERACTION_MARKER_SPEC

if TYPE_CHECKING:
    from ...pipeline.core import FrameRenderPacket

__all__ = ["PygfxOverlayMixin"]

logger = logging.getLogger(__name__)


class PygfxOverlayMixin:
    """HUD, minimap, and tooltip overlay behavior for ``PygfxRenderer``.

    Overlay presentation is renderer-owned because it needs canvas-relative
    placement and pygfx-only state. The minimap stays entirely on the GPU;
    compact HUD and tooltip labels remain Qt widgets.
    """

    def _setup_tooltip(self) -> None:
        """Create the tooltip QLabel overlay on the canvas container."""
        try:
            from PySide6.QtWidgets import QLabel
        except ImportError:
            return
        if self._container is None:
            return
        self._tooltip_label = QLabel(self._container)
        self._tooltip_label.setStyleSheet(
            "background: rgba(30,30,30,220); color: white; padding: 6px 10px; "
            "border-radius: 4px; font-size: 12px;"
        )
        self._tooltip_label.setVisible(False)
        self._tooltip_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._tooltip_label.raise_()

    def _setup_hud_overlays(self) -> None:
        """Initialize pygfx HUD overlays after the canvas container exists."""
        self._update_status_chip_overlay()

    def _viewport_hud_policy(self):
        """Return the effective renderer-neutral HUD policy."""
        visualizer = getattr(self, "visualizer", None)
        return viewport_hud_policy(getattr(visualizer, "app_state", None))

    def _hud_role_enabled(self, role: str) -> bool:
        """Return whether one overlay role is enabled by the current policy."""
        policy = self._viewport_hud_policy()
        if not policy.enabled:
            return False
        if role == "chips":
            return policy.show_status
        if role in {"filters", "filter_chips"}:
            return policy.show_filters
        if role == "annotation":
            return policy.show_annotations
        return policy.show_legends

    def _clear_all_hud_overlays(self) -> None:
        """Hide every renderer-owned HUD widget in one bounded pass."""
        changed = False
        for overlay_id, spec in self._hud_overlay_specs.items():
            if not bool(spec.get("visible", False)):
                continue
            spec["visible"] = False
            label = self._hud_overlay_labels.get(overlay_id)
            if label is not None:
                label.setVisible(False)
            changed = True
        if changed:
            self._reposition_hud_overlays()
        self._hide_tooltip()

    def _hide_policy_disabled_hud_overlays(self) -> None:
        """Hide registered overlays whose content role is currently disabled."""
        changed = False
        for overlay_id, spec in self._hud_overlay_specs.items():
            if not bool(spec.get("visible", False)):
                continue
            role = str(spec.get("role", "legend"))
            if self._hud_role_enabled(role):
                continue
            spec["visible"] = False
            label = self._hud_overlay_labels.get(overlay_id)
            if label is not None:
                label.setVisible(False)
            changed = True
        if changed:
            self._reposition_hud_overlays()

    def refresh_viewport_hud(self) -> None:
        """Reconcile all HUD widgets after policy changes without rebuilding scene data."""
        policy = self._viewport_hud_policy()
        if not policy.enabled:
            if not bool(getattr(self, "_hud_suppressed", False)):
                self._clear_all_hud_overlays()
            self._hud_suppressed = True
            self.request_redraw()
            return
        self._hud_suppressed = False
        self._hide_policy_disabled_hud_overlays()
        self._update_mpc_hud_overlays(self.last_frame_packet)
        self._update_trajectory_hud_overlay()
        self._refresh_feature_hud_overlays()
        if not policy.show_annotations:
            self._hide_tooltip()
        self.request_redraw()

    def _refresh_feature_hud_overlays(self) -> None:
        """Reconcile feature-owned legends after a global HUD policy change."""
        self._sync_marker_legend_from_state(self.last_frame_packet)
        refresh_rf_xray = getattr(self, "_refresh_rf_xray_hud_overlay", None)
        if callable(refresh_rf_xray):
            refresh_rf_xray()

    def _ensure_minimap_gpu_overlay(self, width: int, height: int) -> None:
        """Create screen-space minimap chrome and markers on the pygfx GPU path."""
        size = (max(1, int(width)), max(1, int(height)))
        if self._minimap_overlay_scene is not None and self._minimap_overlay_size == size:
            return

        gfx = self._gfx
        width_px, height_px = (float(size[0]), float(size[1]))
        scene = gfx.Scene()
        camera = gfx.ScreenCoordsCamera(invert_y=True)

        background = gfx.Mesh(
            gfx.Geometry(
                positions=np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [width_px, 0.0, 0.0],
                        [width_px, height_px, 0.0],
                        [0.0, height_px, 0.0],
                    ],
                    dtype=np.float32,
                ),
                indices=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
            ),
            gfx.MeshBasicMaterial(
                color=(0.04, 0.055, 0.08, 0.16),
                side="both",
                depth_test=False,
                depth_write=False,
                alpha_mode="blend",
            ),
            render_order=-10,
        )
        border = gfx.Line(
            gfx.Geometry(
                positions=np.asarray(
                    [
                        [1.0, 1.0, 0.0],
                        [width_px - 1.0, 1.0, 0.0],
                        [width_px - 1.0, 1.0, 0.0],
                        [width_px - 1.0, height_px - 1.0, 0.0],
                        [width_px - 1.0, height_px - 1.0, 0.0],
                        [1.0, height_px - 1.0, 0.0],
                        [1.0, height_px - 1.0, 0.0],
                        [1.0, 1.0, 0.0],
                    ],
                    dtype=np.float32,
                )
            ),
            gfx.LineSegmentMaterial(
                color=(0.9, 0.92, 0.96, 0.9),
                thickness=1.5,
                aa=True,
                depth_test=False,
                depth_write=False,
            ),
            render_order=10,
        )
        title = gfx.Text(
            text="Minimap",
            font_size=12,
            screen_space=True,
            anchor="top-left",
            material=gfx.TextMaterial(
                color=(0.94, 0.95, 0.97, 0.95),
                outline_color=(0.04, 0.05, 0.07, 0.9),
                outline_thickness=0.12,
                aa=True,
                depth_test=False,
                depth_write=False,
            ),
            render_order=20,
        )
        title.local.position = (10.0, 6.0, 0.0)

        tracked = gfx.Points(
            gfx.Geometry(positions=np.zeros((1, 3), dtype=np.float32)),
            gfx.PointsMaterial(
                color=(1.0, 0.7, 0.28, 0.95),
                size=10.0,
                aa=True,
                depth_test=False,
                depth_write=False,
            ),
            visible=False,
            render_order=20,
        )
        camera_point = gfx.Points(
            gfx.Geometry(positions=np.zeros((1, 3), dtype=np.float32)),
            gfx.PointsMaterial(
                color=(0.49, 0.89, 1.0, 0.95),
                size=8.0,
                aa=True,
                depth_test=False,
                depth_write=False,
            ),
            visible=False,
            render_order=20,
        )
        camera_arrow = gfx.Line(
            gfx.Geometry(positions=np.zeros((6, 3), dtype=np.float32)),
            gfx.LineSegmentMaterial(
                color=(0.49, 0.89, 1.0, 0.95),
                thickness=2.0,
                aa=True,
                depth_test=False,
                depth_write=False,
            ),
            visible=False,
            render_order=20,
        )
        scene.add(background, border, title, tracked, camera_point, camera_arrow)
        self._minimap_overlay_scene = scene
        self._minimap_overlay_camera = camera
        self._minimap_overlay_size = size
        self._minimap_overlay_objects = {
            "tracked": tracked,
            "camera": camera_point,
            "arrow": camera_arrow,
        }

    def _compute_minimap_rect(self) -> tuple[int, int, int, int]:
        """Return the canvas-relative rectangle reserved for the minimap."""
        side = int(max(160, min(min(self._width, self._height) * 0.28, 260)))
        margin = 16
        x = max(margin, self._width - side - margin)
        y = max(margin, self._height - side - margin)
        return x, y, side, side

    def _update_minimap_camera(self, bounds: SceneBounds) -> None:
        """Frame the top-down minimap camera around the current scene bounds."""
        if self._minimap_camera is None:
            return
        x, y, width_px, height_px = self._compute_minimap_rect()
        del x, y
        center = np.asarray(bounds.get_center(), dtype=np.float64)
        extent = np.asarray(bounds.get_extent(), dtype=np.float64)
        xy_extent = np.maximum(extent[:2], 1.0)
        pad = max(float(np.max(xy_extent)) * 0.08, 2.0)
        span_x = max(float(xy_extent[0]) + 2.0 * pad, 10.0)
        span_y = max(float(xy_extent[1]) + 2.0 * pad, 10.0)
        aspect = max(float(width_px) / max(float(height_px), 1.0), 1e-3)
        if aspect >= 1.0:
            span_x = max(span_x, span_y * aspect)
        else:
            span_y = max(span_y, span_x / aspect)
        self._minimap_world_rect = (
            center[0] - span_x * 0.5,
            center[0] + span_x * 0.5,
            center[1] + span_y * 0.5,
            center[1] - span_y * 0.5,
        )
        depth = max(float(extent[2]) + 2.0 * pad, max(span_x, span_y), 10.0)
        try:
            self._minimap_camera.width = span_x
            self._minimap_camera.height = span_y
            if hasattr(self._minimap_camera, "depth"):
                self._minimap_camera.depth = depth
            self._minimap_camera.world.reference_up = (0.0, 1.0, 0.0)
            self._minimap_camera.local.position = (
                float(center[0]),
                float(center[1]),
                float(bounds.max_bound[2] + depth),
            )
            self._minimap_camera.look_at((float(center[0]), float(center[1]), float(center[2])))
        except Exception as exc:
            logger.debug("Failed to update minimap camera: %s", exc)

    def _world_to_minimap_uv(self, position: np.ndarray) -> Optional[tuple[float, float]]:
        """Project a world XY position into the minimap overlay's normalized space."""
        if self._minimap_world_rect is None:
            return None
        pos = np.asarray(position, dtype=np.float64).reshape(-1)
        if pos.size < 2 or not np.all(np.isfinite(pos[:2])):
            return None
        left, right, top, bottom = self._minimap_world_rect
        span_x = max(right - left, 1e-6)
        span_y = max(top - bottom, 1e-6)
        u = float(np.clip((pos[0] - left) / span_x, 0.0, 1.0))
        v = float(np.clip((top - pos[1]) / span_y, 0.0, 1.0))
        return u, v

    def _get_minimap_tracked_position(self) -> Optional[np.ndarray]:
        """Return the current focus target position for the minimap marker."""
        scene_query = getattr(self.visualizer, "camera_scene_query_service", None)
        if scene_query is None or not hasattr(scene_query, "get_focus_position"):
            return None
        try:
            position = scene_query.get_focus_position()
        except Exception:
            return None
        if position is None:
            return None
        arr = np.asarray(position, dtype=np.float64).reshape(-1)
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return None
        return arr[:3]

    @staticmethod
    def _minimap_overlay_point(
        uv: tuple[float, float],
        width: int,
        height: int,
        inset: float = 14.0,
    ) -> np.ndarray:
        """Convert normalized minimap coordinates to overlay-local pixels."""
        return np.asarray(
            [
                inset + float(uv[0]) * max(float(width) - 2.0 * inset, 1.0),
                inset + float(uv[1]) * max(float(height) - 2.0 * inset, 1.0),
                0.0,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _write_minimap_positions(obj: Any, positions: np.ndarray) -> None:
        """Update one fixed-size minimap geometry buffer."""
        buffer = obj.geometry.positions
        buffer.data[:] = np.asarray(positions, dtype=np.float32)
        buffer.update_full()

    def _update_minimap_gpu_overlay_state(self, width: int, height: int) -> None:
        """Project tracked-target and camera state into GPU overlay markers."""
        objects = self._minimap_overlay_objects
        if not objects:
            return
        tracked_uv = None
        tracked_pos = self._get_minimap_tracked_position()
        if tracked_pos is not None:
            tracked_uv = self._world_to_minimap_uv(tracked_pos)

        tracked = objects["tracked"]
        tracked.visible = tracked_uv is not None
        if tracked_uv is not None:
            self._write_minimap_positions(
                tracked,
                self._minimap_overlay_point(tracked_uv, width, height).reshape(1, 3),
            )

        camera_uv = None
        camera_heading_xy = None
        cam_state = self.get_camera_state()
        if cam_state is not None:
            eye = np.asarray(cam_state.eye, dtype=np.float64)
            lookat = np.asarray(cam_state.lookat, dtype=np.float64)
            camera_uv = self._world_to_minimap_uv(eye)
            heading_xy = lookat[:2] - eye[:2]
            heading_norm = float(np.linalg.norm(heading_xy))
            if heading_norm > 1e-6:
                camera_heading_xy = (
                    float(heading_xy[0] / heading_norm),
                    float(-heading_xy[1] / heading_norm),
                )

        camera_obj = objects["camera"]
        arrow = objects["arrow"]
        camera_obj.visible = camera_uv is not None
        arrow.visible = camera_uv is not None and camera_heading_xy is not None
        if camera_uv is None:
            return

        camera_point = self._minimap_overlay_point(camera_uv, width, height)
        self._write_minimap_positions(camera_obj, camera_point.reshape(1, 3))
        if camera_heading_xy is None:
            return

        heading = np.asarray(camera_heading_xy, dtype=np.float32)
        heading /= max(float(np.linalg.norm(heading)), 1e-6)
        perpendicular = np.asarray([-heading[1], heading[0]], dtype=np.float32)
        tip = camera_point.copy()
        tip[:2] += heading * 18.0
        left = camera_point.copy()
        left[:2] += -heading * 3.0 + perpendicular * 5.0
        right = camera_point.copy()
        right[:2] += -heading * 3.0 - perpendicular * 5.0
        self._write_minimap_positions(
            arrow,
            np.asarray(
                [camera_point, tip, tip, left, tip, right],
                dtype=np.float32,
            ),
        )

    def _render_minimap(self) -> bool:
        """Render the top-down scene and screen-space chrome in one GPU viewport."""
        if (
            not self._viewport_hud_policy().enabled
            or not self._minimap_enabled
            or self._renderer is None
            or self._scene is None
            or self._minimap_camera is None
            or self._minimap_viewport is None
        ):
            return False
        bounds = self.compute_scene_bounds(scope="whole")
        if bounds is None:
            return False
        self._update_minimap_camera(bounds)
        rect = self._compute_minimap_rect()
        self._minimap_viewport.rect = rect
        self._ensure_minimap_gpu_overlay(rect[2], rect[3])
        self._update_minimap_gpu_overlay_state(rect[2], rect[3])
        self._minimap_viewport.render(self._scene, self._minimap_camera, flush=False)
        self._minimap_viewport.render(
            self._minimap_overlay_scene,
            self._minimap_overlay_camera,
            flush=True,
        )
        return True

    def set_camera_minimap_visible(self, visible: bool) -> None:
        """Toggle the pygfx top-down minimap viewport."""
        self._minimap_enabled = bool(visible)
        self.request_redraw()

    @staticmethod
    def _hud_overlay_stylesheet(role: str) -> str:
        """Return stylesheet policy for HUD chips vs legend panels."""
        if role in {"chips", "filter_chips"}:
            return "background: transparent; color: white; font-size: 12px;"
        return (
            "background: rgba(20,20,20,215); color: white; padding: 8px 10px; "
            "border-radius: 6px; font-size: 12px;"
        )

    def _ensure_hud_overlay_label(self, overlay_id: str, role: str) -> Any:
        """Create a HUD overlay label on demand."""
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QLabel
        except ImportError:
            return None
        if self._container is None:
            return None
        label = self._hud_overlay_labels.get(overlay_id)
        if label is None:
            label = QLabel(self._container)
            label.setTextFormat(Qt.RichText)
            label.setAttribute(Qt.WA_TransparentForMouseEvents)
            label.setWordWrap(True)
            label.setStyleSheet(self._hud_overlay_stylesheet(role))
            label.setVisible(False)
            self._hud_overlay_labels[overlay_id] = label
        return label

    def _request_hud_background_redraw(self) -> None:
        """Refresh canvas pixels exposed by a moved, resized, or hidden HUD."""
        request_redraw = getattr(self, "request_redraw", None)
        if callable(request_redraw):
            request_redraw()

    def _set_hud_overlay(
        self,
        overlay_id: str,
        *,
        html: str,
        visible: bool,
        role: str,
        corner: str,
        priority: int,
    ) -> None:
        """Create/update a renderer-owned HUD overlay."""
        if not visible or not self._hud_role_enabled(role):
            self._clear_hud_overlay(overlay_id)
            return
        label = self._ensure_hud_overlay_label(overlay_id, role)
        if label is None:
            return
        normalized = {
            "role": role,
            "corner": corner,
            "priority": int(priority),
            "visible": bool(visible),
            "html": str(html),
        }
        previous = self._hud_overlay_specs.get(overlay_id)
        if previous == normalized:
            # A bounded layout can temporarily hide a lower-priority widget
            # without changing its semantic spec. Reconcile that presentation
            # state instead of treating the next frame as a complete no-op.
            if bool(getattr(label, "_orchav_hud_layout_hidden", False)):
                self._reposition_hud_overlays()
                self._request_hud_background_redraw()
            return

        role_changed = previous is None or previous.get("role") != role
        html_changed = previous is None or previous.get("html") != normalized["html"]
        visible_changed = previous is None or previous.get("visible") != normalized["visible"]
        layout_changed = previous is None or any(
            previous.get(key) != normalized[key] for key in ("corner", "priority")
        )
        content_changed_while_visible = bool(
            previous is not None
            and previous.get("visible", False)
            and (role_changed or html_changed)
        )

        if content_changed_while_visible:
            label.setVisible(False)
        if role_changed:
            label.setStyleSheet(self._hud_overlay_stylesheet(role))
        if role_changed or html_changed:
            label.setText(normalized["html"])
            label.adjustSize()
        if visible_changed or content_changed_while_visible:
            layout_hidden = bool(getattr(label, "_orchav_hud_layout_hidden", False))
            label.setVisible(normalized["visible"] and not layout_hidden)
        self._hud_overlay_specs[overlay_id] = normalized
        if role_changed or html_changed or visible_changed or layout_changed:
            self._reposition_hud_overlays()
            self._request_hud_background_redraw()

    def _clear_hud_overlay(self, overlay_id: str) -> None:
        """Hide a HUD overlay without deleting its widget."""
        spec = self._hud_overlay_specs.get(overlay_id)
        if spec is None or not bool(spec.get("visible", False)):
            return
        spec["visible"] = False
        label = self._hud_overlay_labels.get(overlay_id)
        if label is not None:
            label.setVisible(False)
        self._reposition_hud_overlays()
        self._request_hud_background_redraw()

    def _reposition_hud_overlays(self) -> None:
        """Stack visible HUD overlays by corner and priority."""
        if self._container is None:
            return
        grouped: dict[str, list[tuple[int, str, Any]]] = {}
        for overlay_id, spec in self._hud_overlay_specs.items():
            if not bool(spec.get("visible", False)):
                continue
            label = self._hud_overlay_labels.get(overlay_id)
            if label is None:
                continue
            corner = str(spec.get("corner", "top_right"))
            grouped.setdefault(corner, []).append(
                (int(spec.get("priority", 0)), str(overlay_id), label)
            )

        margin = 12
        gap = 8
        container_height_fn = getattr(self._container, "height", None)
        container_height = (
            int(container_height_fn())
            if callable(container_height_fn)
            else max(1, int(getattr(self, "_height", 480)))
        )
        for corner, entries in grouped.items():
            y = margin
            max_y = max(margin, container_height - margin)
            if corner == "top_right" and bool(getattr(self, "_minimap_enabled", False)):
                minimap_y = self._compute_minimap_rect()[1]
                max_y = min(max_y, max(margin, minimap_y - gap))
            for _, _, label in sorted(entries, key=lambda item: (item[0], item[1])):
                max_width = max(80, min(360, self._container.width() - (2 * margin)))
                set_maximum_width = getattr(label, "setMaximumWidth", None)
                if callable(set_maximum_width):
                    set_maximum_width(max_width)
                set_maximum_height = getattr(label, "setMaximumHeight", None)
                if callable(set_maximum_height):
                    set_maximum_height(16_777_215)
                if callable(set_maximum_width):
                    label.adjustSize()
                remaining_height = max_y - y
                if remaining_height < 28:
                    if not bool(getattr(label, "_orchav_hud_layout_hidden", False)):
                        label.setVisible(False)
                        setattr(label, "_orchav_hud_layout_hidden", True)
                    continue
                if label.height() > remaining_height:
                    if callable(set_maximum_height):
                        set_maximum_height(remaining_height)
                if bool(getattr(label, "_orchav_hud_layout_hidden", False)):
                    label.setVisible(True)
                    setattr(label, "_orchav_hud_layout_hidden", False)
                x = margin
                if corner == "top_right":
                    x = max(0, self._container.width() - label.width() - margin)
                label.move(x, y)
                # Keep the actual QWidget stack synchronized with the semantic
                # priority order after frame-to-frame hide/show transitions.
                label.raise_()
                y += label.height() + gap

    @staticmethod
    def _marker_legend_html() -> str:
        """HTML payload for the interaction-marker legend overlay."""
        specs = (*INTERACTION_MARKER_SPECS, UNKNOWN_INTERACTION_MARKER_SPEC)
        table_rows: list[str] = []
        for start in range(0, len(specs), 2):
            cells: list[str] = []
            for spec in specs[start : start + 2]:
                cells.extend(
                    (
                        "<td style='padding:0 6px 2px 0;'>" f"{spec.html_symbol}</td>",
                        "<td style='padding:0 12px 2px 0;'>" f"{escape(spec.label)}</td>",
                    )
                )
            table_rows.append(f"<tr>{''.join(cells)}</tr>")
        return (
            "<div style='font-weight:600; margin-bottom:4px;'>Interaction Markers</div>"
            f"<table cellspacing='0' cellpadding='0'>{''.join(table_rows)}</table>"
        )

    @staticmethod
    def _rgb_to_hex(color: Any) -> str:
        """Convert normalized RGB-like input to a CSS hex color."""
        arr = np.asarray(color, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            return "#b3b3b3"
        rgb = np.clip(np.round(arr[:3] * 255.0), 0.0, 255.0).astype(np.uint8)
        return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"

    def _semantic_mpc_legend_html(
        self,
        color_mode: str,
        packet: Optional[FrameRenderPacket] = None,
    ) -> str:
        """Return categorical MPC legend HTML for the current color mode."""
        mpc_core = getattr(self.visualizer, "mpc_core", None)
        if mpc_core is None:
            return ""

        rows: list[tuple[str, str]] = []
        title = ""
        if color_mode == "reflection_order":
            palette = getattr(mpc_core, "_order_palette", None)
            if palette is None or len(palette) < 7:
                return ""
            title = "MPC Color Legend"
            rows = [
                ("LoS", self._rgb_to_hex(palette[0])),
                ("1st Order", self._rgb_to_hex(palette[1])),
                ("2nd Order", self._rgb_to_hex(palette[2])),
                ("3rd Order", self._rgb_to_hex(palette[3])),
                ("4th Order", self._rgb_to_hex(palette[4])),
                ("5th Order", self._rgb_to_hex(palette[5])),
                ("6+ Order", self._rgb_to_hex(palette[6])),
            ]
        elif color_mode == "mpc_type":
            palette = getattr(mpc_core, "_type_palette", None)
            if palette is None or len(palette) < 6:
                return ""
            frame_packet = (
                packet if packet is not None else getattr(self, "last_frame_packet", None)
            )
            present_types = tuple(getattr(frame_packet, "mpc_line_itype_codes", ()) or ())
            title = "MPC Type Legend"
            rows = [
                (entry.label, self._rgb_to_hex(entry.color))
                for entry in mpc_interaction_legend_entries(
                    palette,
                    present_types=present_types,
                )
            ]
            if not rows:
                return ""
        elif color_mode == "material":
            material_items = getattr(mpc_core, "material_legend_items", None)
            if not callable(material_items):
                return ""
            state = getattr(self.visualizer, "app_state", None)
            use_distinct = bool(getattr(state, "use_distinct_material_colors", False))
            active_only = getattr(self.visualizer, "mpc_allowed_materials", None) is not None
            try:
                material_rows = list(
                    material_items(
                        use_distinct,
                        active_only=active_only,
                    )
                )
            except (AttributeError, TypeError, ValueError):
                return ""
            if not material_rows:
                return ""
            limit = 12 if self._viewport_hud_policy().detailed else 4
            title = "MPC Material Legend"
            rows = [(str(label), self._rgb_to_hex(color)) for label, color in material_rows[:limit]]
            hidden_count = len(material_rows) - len(rows)
            if hidden_count:
                rows.append((f"+{hidden_count} more", "transparent"))
        else:
            return ""

        lines = "".join(
            (
                "<tr>"
                "<td style='padding:0 8px 3px 0;'>"
                f"<span style='color:{color}; font-size:14px;'>&#9632;</span>"
                "</td>"
                f"<td>{escape(str(label))}</td>"
                "</tr>"
            )
            for label, color in rows
        )
        return (
            f"<div style='font-weight:600; margin-bottom:4px;'>{title}</div>"
            f"<table cellspacing='0' cellpadding='0'>{lines}</table>"
        )

    def _semantic_mpc_legend_cache_signature(
        self,
        color_mode: str,
        packet: Optional[FrameRenderPacket] = None,
    ) -> tuple[Any, ...]:
        """Return a stable key for semantic legend mode and palette content."""
        mpc_core = getattr(self.visualizer, "mpc_core", None)
        if mpc_core is None:
            return (color_mode, None)
        palette_name = {
            "reflection_order": "_order_palette",
            "mpc_type": "_type_palette",
        }.get(color_mode)
        if color_mode == "material":
            material_items = getattr(mpc_core, "material_legend_items", None)
            if not callable(material_items):
                return (color_mode, None)
            state = getattr(self.visualizer, "app_state", None)
            use_distinct = bool(getattr(state, "use_distinct_material_colors", False))
            active_only = getattr(self.visualizer, "mpc_allowed_materials", None) is not None
            try:
                material_rows = list(
                    material_items(
                        use_distinct,
                        active_only=active_only,
                    )
                )
            except (AttributeError, TypeError, ValueError):
                return (color_mode, use_distinct, active_only, None)
            row_signature = tuple(
                (
                    str(label),
                    *tuple(
                        float(value)
                        for value in np.asarray(color, dtype=np.float32).reshape(-1)[:3]
                    ),
                )
                for label, color in material_rows
            )
            return (
                color_mode,
                use_distinct,
                active_only,
                self._viewport_hud_policy().mode,
                row_signature,
            )
        if palette_name is None:
            return (color_mode, None)
        palette = getattr(mpc_core, palette_name, None)
        if palette is None:
            return (color_mode, None)
        values = np.asarray(palette, dtype=np.float32).reshape(-1)
        if color_mode == "mpc_type":
            frame_packet = (
                packet if packet is not None else getattr(self, "last_frame_packet", None)
            )
            packet_signature = tuple(getattr(frame_packet, "mpc_line_itype_codes", ()) or ())
            return (color_mode, packet_signature, *tuple(float(value) for value in values))
        return (color_mode, *tuple(float(value) for value in values))

    def _colorbar_gradient_html(self, lut: Any = None) -> str:
        """Return a compact gradient bar from an explicit or active scalar LUT."""
        if lut is None:
            try:
                from ...utils.colors import ensure_continuous_lut

                lut = ensure_continuous_lut()
            except (ImportError, AttributeError):
                return ""
        try:
            samples = np.asarray(lut, dtype=np.float32)
            if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] < 3:
                return ""
            idx = np.linspace(0, samples.shape[0] - 1, 18).astype(np.int32)
            return "".join(
                f"<span style='color:{self._rgb_to_hex(samples[i])};'>&#9608;</span>" for i in idx
            )
        except Exception:
            return ""

    def _colorbar_overlay_html(
        self,
        label: str,
        value_range: tuple[float, float],
        *,
        lut: Any = None,
    ) -> str:
        """Return HTML for a compact scalar colorbar overlay."""
        min_val, max_val = float(value_range[0]), float(value_range[1])
        gradient = self._colorbar_gradient_html(lut)
        if not gradient:
            return ""
        return (
            f"<div style='font-weight:600; margin-bottom:4px;'>{escape(str(label))}</div>"
            f"<div style='line-height:1;'>{gradient}</div>"
            "<table cellspacing='0' cellpadding='0' width='100%'>"
            f"<tr><td>{min_val:.2f}</td><td align='right'>{max_val:.2f}</td></tr>"
            "</table>"
        )

    def _coverage_colorbar_gradient_html(self, metric_name: Any) -> str:
        """Return the same metric-aware scalar gradient used by coverage meshes."""
        cmap_name = coverage_metric_colormap(metric_name)
        if cmap_name is None:
            return ""
        try:
            from matplotlib import colormaps

            samples = np.asarray(
                colormaps[cmap_name](np.linspace(0.0, 1.0, 18)),
                dtype=np.float32,
            )
        except (ImportError, KeyError, TypeError, ValueError):
            return ""
        if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] < 3:
            return ""
        return "".join(
            f"<span style='color:{self._rgb_to_hex(sample)};'>&#9608;</span>" for sample in samples
        )

    @staticmethod
    def _coverage_metric_label(metric_name: Any) -> tuple[str, str]:
        """Return a human label and unit for a coverage metric key."""
        return coverage_metric_label(metric_name)

    @staticmethod
    def _format_coverage_value(value: Any, unit: str) -> str:
        """Format a coverage legend endpoint for the current unit family."""
        return format_coverage_value(value, unit)

    @staticmethod
    def _coverage_serving_tx_overlay_html(metadata: dict[str, Any]) -> str:
        """Return a categorical legend for the active serving-TX layer."""
        tx_count = int(metadata.get("tx_count", 0) or 0)
        labels = serving_tx_labels(metadata.get("tx_names", []) or [], tx_count)
        if not labels:
            return "<div style='font-weight:600; margin-bottom:4px;'>Coverage: Serving TX</div>"

        rows = "".join(
            (
                "<tr>"
                "<td style='padding:1px 8px 1px 0;'>"
                f"<span style='color:{escape(serving_tx_color_hex(index))};'>&#9632;</span>"
                "</td>"
                f"<td style='padding:1px 8px 1px 0;'>{index}</td>"
                f"<td>{escape(label)}</td>"
                "</tr>"
            )
            for index, label in enumerate(labels)
        )

        height_value = metadata.get("selected_height_value")
        height_row = ""
        try:
            if height_value is not None and np.isfinite(float(height_value)):
                height_row = (
                    f"<div style='margin-top:4px;'>Height: {float(height_value):.2f} m</div>"
                )
        except (TypeError, ValueError):
            height_row = ""

        return (
            "<div style='font-weight:600; margin-bottom:4px;'>Coverage: Serving TX</div>"
            "<table cellspacing='0' cellpadding='0'>"
            f"{rows}"
            "</table>"
            f"{height_row}"
        )

    def _coverage_colorbar_overlay_html(self, metadata: dict[str, Any]) -> str:
        """Return HTML for the active coverage metric legend."""
        metric_name = str(metadata.get("metric_name", "coverage"))
        if is_serving_tx_metric(metric_name):
            return self._coverage_serving_tx_overlay_html(metadata)

        label, unit = self._coverage_metric_label(metric_name)
        gradient = self._coverage_colorbar_gradient_html(metric_name)
        if not gradient:
            return ""

        value_min = metadata.get("value_min", 0.0)
        value_max = metadata.get("value_max", 1.0)
        min_label = self._format_coverage_value(value_min, unit)
        max_label = self._format_coverage_value(value_max, unit)
        unit_label = f" ({escape(unit)})" if unit and unit not in {"linear", "index"} else ""
        title = f"Coverage: {escape(label)}{unit_label}"

        color_scale = str(metadata.get("color_scale") or coverage_metric_color_scale(metric_name))
        scale_row = (
            "<div style='margin-top:4px;'>Color scale: logarithmic</div>"
            if color_scale == "logarithmic"
            else ""
        )

        height_value = metadata.get("selected_height_value")
        height_row = ""
        try:
            if height_value is not None and np.isfinite(float(height_value)):
                height_row = (
                    f"<div style='margin-top:4px;'>Height: {float(height_value):.2f} m</div>"
                )
        except (TypeError, ValueError):
            height_row = ""

        no_data_row = ""
        try:
            no_data_fraction = float(metadata.get("no_data_fraction", 0.0))
        except (TypeError, ValueError):
            no_data_fraction = 0.0
        if no_data_fraction > 0.0:
            no_data_row = (
                "<div style='margin-top:4px; color:#d9d9d9; font-style:italic;'>"
                f"No data (transparent / hidden): {no_data_fraction * 100.0:.1f}%"
                "</div>"
            )

        return (
            f"<div style='font-weight:600; margin-bottom:4px;'>{title}</div>"
            f"<div style='line-height:1;'>{gradient}</div>"
            "<table cellspacing='0' cellpadding='0' width='100%'>"
            f"<tr><td>{min_label}</td><td align='right'>{max_label}</td></tr>"
            "</table>"
            f"{scale_row}"
            f"{height_row}"
            f"{no_data_row}"
        )

    def _status_chip_html(self, packet: Optional[FrameRenderPacket] = None) -> str:
        """HTML payload for top-left status chips."""
        chips: list[str] = []
        app_state = getattr(self.visualizer, "app_state", None)
        if self._aa_mode != "off":
            chips.append(f"AA: {self._aa_mode.upper()}")
        if self._clipping_planes:
            chips.append("Cutaway")
        if self._ground_grid_visible:
            chips.append("Grid")
        if self._depth_pass_enabled:
            chips.append("Depth EXP")
        if _env_flag("ORCHAV_PYGFX_ENABLE_EDL", False):
            chips.append("EDL EXP")
        chips.extend(runtime_status_chips(self, packet))
        if bool(getattr(app_state, "show_mpc_type_markers", False)):
            chips.append("Interaction Markers")
        if bool(app_state is not None and app_state.mpc_visibility.effective_paths):
            frame_packet = packet if packet is not None else self.last_frame_packet
            if frame_packet is not None:
                if len(frame_packet.mpc_lines) == 0 and len(frame_packet.mpc_points) == 0:
                    chips.append("MPC: none")
        if not chips:
            return ""
        chip_html = "".join(
            (
                "<span style='background: rgba(20,20,20,215); color: white; "
                "border-radius: 10px; padding: 3px 8px; margin-right: 6px;'>"
                f"{label}</span>"
            )
            for label in chips
        )
        return f"<div>{chip_html}</div>"

    def _update_status_chip_overlay(
        self,
        packet: Optional[FrameRenderPacket] = None,
    ) -> None:
        """Refresh the top-left status-chip overlay."""
        if not self._viewport_hud_policy().show_status:
            self._clear_hud_overlay("render_status")
            return
        html = self._status_chip_html(packet)
        if not html:
            self._clear_hud_overlay("render_status")
        else:
            self._set_hud_overlay(
                "render_status",
                html=html,
                visible=True,
                role="chips",
                corner="top_left",
                priority=10,
            )

    def _update_coverage_hud_overlay(
        self,
        packet: Optional[FrameRenderPacket],
    ) -> bool:
        """Refresh the pygfx-only coverage metric/colorbar overlay."""
        if not self._viewport_hud_policy().show_legends:
            self._clear_hud_overlay("coverage_colorbar")
            return False
        if packet is None or not packet.show_coverage:
            self._clear_hud_overlay("coverage_colorbar")
            return False
        if packet.coverage_vertices is None or np.asarray(packet.coverage_vertices).size == 0:
            self._clear_hud_overlay("coverage_colorbar")
            return False

        metadata = packet.coverage_metadata
        if not isinstance(metadata, dict):
            self._clear_hud_overlay("coverage_colorbar")
            return False
        try:
            if int(metadata.get("valid_cell_count", 1)) <= 0:
                self._clear_hud_overlay("coverage_colorbar")
                return False
        except (TypeError, ValueError):
            self._clear_hud_overlay("coverage_colorbar")
            return False

        html = self._coverage_colorbar_overlay_html(metadata)
        if not html:
            self._clear_hud_overlay("coverage_colorbar")
            return False

        self._set_hud_overlay(
            "coverage_colorbar",
            html=html,
            visible=True,
            role="legend",
            corner="top_right",
            priority=15,
        )
        return True

    def _update_mpc_hud_overlays(self, packet: Optional[FrameRenderPacket]) -> None:
        """Refresh MPC legends/colorbars from the current view model and app state."""
        policy = self._viewport_hud_policy()
        if not policy.enabled:
            if not bool(getattr(self, "_hud_suppressed", False)):
                self._clear_all_hud_overlays()
            self._hud_suppressed = True
            return
        self._hud_suppressed = False
        self._update_status_chip_overlay(packet)
        self._update_filter_hud_overlay()
        self._update_coverage_hud_overlay(packet)
        self._sync_marker_legend_from_state(packet)
        if packet is None:
            self._clear_hud_overlay("mpc_semantic_legend")
            self._clear_hud_overlay("mpc_colorbar")
            self._update_trajectory_hud_overlay()
            return
        if len(packet.mpc_lines) == 0 and len(packet.mpc_points) == 0:
            self._clear_hud_overlay("mpc_semantic_legend")
            self._clear_hud_overlay("mpc_colorbar")
            self._update_trajectory_hud_overlay()
            return

        app_state = getattr(self.visualizer, "app_state", None)
        color_mode = str(getattr(app_state, "color_mode", "")).strip().lower()

        semantic_html = ""
        if policy.show_legends:
            semantic_key = self._semantic_mpc_legend_cache_signature(color_mode, packet)
            if semantic_key != self._semantic_legend_cache_key:
                self._semantic_legend_cache_key = semantic_key
                self._semantic_legend_cache_html = self._semantic_mpc_legend_html(
                    color_mode,
                    packet,
                )
            semantic_html = self._semantic_legend_cache_html
        if semantic_html:
            self._set_hud_overlay(
                "mpc_semantic_legend",
                html=semantic_html,
                visible=True,
                role="legend",
                corner="top_right",
                priority=30,
            )
        else:
            self._clear_hud_overlay("mpc_semantic_legend")

        colorbar = packet.colorbar
        if policy.show_legends and colorbar is not None and color_mode in ("delay", "path_loss"):
            label, value_range = colorbar
            html = self._colorbar_overlay_html(str(label), value_range)
            if html:
                self._set_hud_overlay(
                    "mpc_colorbar",
                    html=html,
                    visible=True,
                    role="legend",
                    corner="top_right",
                    priority=40,
                )
            else:
                self._clear_hud_overlay("mpc_colorbar")
        else:
            self._clear_hud_overlay("mpc_colorbar")
        self._update_trajectory_hud_overlay()

    def _update_filter_hud_overlay(self) -> None:
        """Show a bounded compact or detailed summary of active path filters."""
        policy = self._viewport_hud_policy()
        if not policy.show_filters:
            self._clear_hud_overlay("path_filters")
            self._clear_hud_overlay("material_filter_swatches")
            return
        state = getattr(self.visualizer, "app_state", None)
        if state is None:
            self._clear_hud_overlay("path_filters")
            self._clear_hud_overlay("material_filter_swatches")
            return
        self._update_material_filter_swatch_overlay()
        summary = build_path_filter_summary(
            state,
            allowed_materials=getattr(self.visualizer, "mpc_allowed_materials", None),
        )
        if not summary.active:
            self._clear_hud_overlay("path_filters")
            return
        if policy.detailed:
            detail_html = "".join(
                f"<div style='margin-top:2px;'>{escape(detail)}</div>" for detail in summary.details
            )
            html = (
                "<div style='font-weight:600; margin-bottom:4px;'>Active Path Filters</div>"
                f"{detail_html}"
            )
            role = "filters"
        else:
            text = escape(summary.compact_text)
            html = (
                "<div><span style='background: rgba(20,20,20,215); color: white; "
                "border-radius: 10px; padding: 3px 8px;'>"
                f"{text}</span></div>"
            )
            role = "filter_chips"
        self._set_hud_overlay(
            "path_filters",
            html=html,
            visible=True,
            role=role,
            corner="top_left",
            priority=20,
        )

    def _update_material_filter_swatch_overlay(self) -> None:
        """Show bounded swatches for an explicit MPC material allow-list."""
        policy = self._viewport_hud_policy()
        visualizer = getattr(self, "visualizer", None)
        allowed = getattr(visualizer, "mpc_allowed_materials", None)
        if not policy.show_filters or allowed is None or not allowed:
            self._clear_hud_overlay("material_filter_swatches")
            return

        mpc_core = getattr(visualizer, "mpc_core", None)
        material_items = getattr(mpc_core, "material_legend_items", None)
        if not callable(material_items):
            self._clear_hud_overlay("material_filter_swatches")
            return

        state = getattr(visualizer, "app_state", None)
        use_distinct = bool(getattr(state, "use_distinct_material_colors", False))
        try:
            rows = list(
                material_items(
                    use_distinct,
                    active_only=True,
                )
            )
        except (AttributeError, TypeError, ValueError):
            self._clear_hud_overlay("material_filter_swatches")
            return
        if not rows:
            self._clear_hud_overlay("material_filter_swatches")
            return

        limit = 12 if policy.detailed else 4
        visible_rows = rows[:limit]
        row_html = "".join(
            (
                "<tr>"
                "<td style='padding:0 7px 2px 0;'>"
                f"<span style='color:{self._rgb_to_hex(color)};'>&#9632;</span>"
                "</td>"
                f"<td>{escape(str(label))}</td>"
                "</tr>"
            )
            for label, color in visible_rows
        )
        hidden_count = len(rows) - len(visible_rows)
        if hidden_count:
            row_html += (
                "<tr><td></td>" f"<td style='font-style:italic;'>+{hidden_count} more</td></tr>"
            )
        content = (
            "<div style='font-weight:600; margin-bottom:3px;'>Selected Materials</div>"
            f"<table cellspacing='0' cellpadding='0'>{row_html}</table>"
        )
        role = "filters"
        if not policy.detailed:
            content = (
                "<div style='background: rgba(20,20,20,215); color: white; "
                "border-radius: 8px; padding: 5px 8px;'>"
                f"{content}</div>"
            )
            role = "filter_chips"
        self._set_hud_overlay(
            "material_filter_swatches",
            html=content,
            visible=True,
            role=role,
            corner="top_left",
            priority=25,
        )

    def _update_trajectory_hud_overlay(self) -> None:
        """Show the active scalar trajectory color legend when trajectories exist."""
        policy = self._viewport_hud_policy()
        if not policy.show_legends:
            self._clear_hud_overlay("trajectory_colorbar")
            return
        visible_kinds = getattr(self, "_visible_trajectory_kinds", set())
        if not visible_kinds:
            self._clear_hud_overlay("trajectory_colorbar")
            return
        legend = build_trajectory_hud_legend(
            str(getattr(self, "_trajectory_hud_color_mode", "node_color")),
            getattr(self, "_trajectory_hud_scalar_range", None),
        )
        if legend is None:
            self._clear_hud_overlay("trajectory_colorbar")
            return
        label = legend.title
        if legend.unit:
            label = f"{label} ({legend.unit})"
        html = self._colorbar_overlay_html(
            label,
            legend.value_range,
            lut=ensure_viridis_lut(),
        )
        if not html:
            self._clear_hud_overlay("trajectory_colorbar")
            return
        self._set_hud_overlay(
            "trajectory_colorbar",
            html=html,
            visible=True,
            role="legend",
            corner="top_right",
            priority=50,
        )

    def _set_marker_legend_visible(self, visible: bool) -> None:
        """Show or hide the MPC marker legend without changing user intent."""
        if visible and self._hud_role_enabled("legend"):
            self._set_hud_overlay(
                "mpc_marker_legend",
                html=self._marker_legend_html(),
                visible=True,
                role="legend",
                corner="top_right",
                priority=20,
            )
            return
        self._clear_hud_overlay("mpc_marker_legend")

    def _sync_marker_legend_from_state(
        self,
        packet: Optional[FrameRenderPacket],
    ) -> None:
        """Keep marker-legend intent stable across sparse animation frames."""
        state = getattr(getattr(self, "visualizer", None), "app_state", None)
        visibility = getattr(state, "mpc_visibility", None)
        bounce_points_visible = bool(getattr(visibility, "effective_bounce_points", True))
        requested = bool(
            packet is not None
            and getattr(state, "show_mpc_type_markers", False)
            and bounce_points_visible
        )
        self._mpc_marker_legend_requested = requested
        self._set_marker_legend_visible(requested)

    def _show_tooltip(self, text: str, event: Any) -> None:
        """Show tooltip near cursor position."""
        if self._tooltip_label is None or not self._viewport_hud_policy().show_annotations:
            return
        self._tooltip_label.setText(text)
        self._tooltip_label.adjustSize()
        self._reposition_tooltip(event)
        self._tooltip_label.setVisible(True)
        self._tooltip_label.raise_()

    def _reposition_tooltip(self, event: Any) -> None:
        """Reposition tooltip near cursor, clamped to container bounds."""
        if self._tooltip_label is None or self._container is None:
            return
        x = int(getattr(event, "x", 0)) + 15
        y = int(getattr(event, "y", 0)) + 15

        cw = self._container.width()
        ch = self._container.height()
        tw = self._tooltip_label.width()
        th = self._tooltip_label.height()

        if x + tw > cw:
            x = max(0, cw - tw - 5)
        if y + th > ch:
            y = max(0, ch - th - 5)
        self._tooltip_label.move(x, y)

    def _hide_tooltip(self) -> None:
        """Hide the tooltip overlay."""
        self._last_hover_identity = None
        if self._tooltip_label is not None:
            self._tooltip_label.setVisible(False)
