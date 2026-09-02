"""MPC hot-path helper types for the pygfx renderer.

This mixin owns the pygfx-specific path from frame-packet MPC arrays to reusable
line and point GPU buffers. It keeps the renderer-neutral MPC semantics from
the pipeline intact while adding backend-local expansion, capacity management,
marker glyphs, picking metadata, and CPU prewarm cache policy.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ...types.render_payloads import LineSetPayload, PointCloudPayload
from .canvas import _env_flag

if TYPE_CHECKING:
    from ...pipeline.core import FrameRenderPacket

__all__ = [
    "INTERACTION_MARKER_SPECS",
    "UNKNOWN_INTERACTION_MARKER_SPEC",
    "InteractionMarkerSpec",
    "MpcExpandedLineCacheEntry",
    "PygfxMpcMixin",
    "interaction_marker_spec",
]


@dataclass(frozen=True, slots=True)
class InteractionMarkerSpec:
    """One canonical interaction-point glyph used by pygfx and its legend."""

    interaction_type: int | None
    label: str
    marker_name: str
    html_symbol: str


INTERACTION_MARKER_SPECS: tuple[InteractionMarkerSpec, ...] = (
    InteractionMarkerSpec(1, "Specular", "circle", "&#9679;"),
    InteractionMarkerSpec(2, "Diffuse", "triangle_up", "&#9650;"),
    InteractionMarkerSpec(4, "Refraction", "diamond", "&#9670;"),
    InteractionMarkerSpec(8, "Diffraction", "plus", "&#10010;"),
    InteractionMarkerSpec(99, "Virtual", "square", "&#9632;"),
)
UNKNOWN_INTERACTION_MARKER_SPEC = InteractionMarkerSpec(
    None,
    "Unknown",
    "cross",
    "&#10005;",
)
_INTERACTION_MARKER_SPECS_BY_TYPE = {
    spec.interaction_type: spec for spec in INTERACTION_MARKER_SPECS
}


def interaction_marker_spec(interaction_type: int) -> InteractionMarkerSpec | None:
    """Return the marker semantics for one canonical interaction type.

    LoS (type 0) has no physical interaction point, so it intentionally has no
    visible marker or legend row. Unrecognized nonzero values use the explicit
    Unknown cross glyph.
    """
    value = int(interaction_type)
    if value == 0:
        return None
    return _INTERACTION_MARKER_SPECS_BY_TYPE.get(value, UNKNOWN_INTERACTION_MARKER_SPEC)


@dataclass(slots=True)
class MpcExpandedLineCacheEntry:
    """CPU cache entry for endpoint-expanded colored MPC line segments."""

    points: np.ndarray
    colors: np.ndarray
    n_segments: int
    color_cols: int
    byte_size: int
    source_refs: tuple[np.ndarray, np.ndarray, np.ndarray]


logger = logging.getLogger(__name__)


class PygfxMpcMixin:
    """Apply frame-packet MPC lines, bounce points, and marker glyphs to pygfx."""

    def _remove_mpc_geometry_if_present(self, name: str) -> bool:
        """Ensure one MPC geometry is absent without treating absence as failure."""
        if not self.has_named_geometry(name):
            return True
        return bool(self.remove_named_geometry(name))

    def _apply_mpc_lines(self, packet: FrameRenderPacket) -> bool:
        """Upload MPC segments and report complete native acceptance."""
        t_func_start = time.perf_counter()
        points = packet.mpc_points
        lines = packet.mpc_lines
        colors = packet.mpc_colors

        if (
            not packet.mpc_visibility.effective_paths
            or points is None
            or len(points) == 0
            or lines is None
            or len(lines) == 0
        ):
            t_remove = time.perf_counter()
            removed = self._remove_mpc_geometry_if_present("mpc_lines")
            self._record_profile_metric(
                "pygfx_mpc_lines_empty_remove_ms", (time.perf_counter() - t_remove) * 1000.0
            )
            if removed:
                self._mpc_lines_source_sig = None
            return removed

        self._record_profile_array_bytes("pygfx_mpc_lines_input_bytes", points, lines, colors)
        t_sig_start = time.perf_counter()
        source_sig = self._mpc_line_source_signature(packet)
        self._record_profile_metric(
            "pygfx_mpc_lines_signature_ms",
            (time.perf_counter() - t_sig_start) * 1000.0,
        )
        if source_sig == self._mpc_lines_source_sig:
            # Check that the geometries still exist (they might have been removed)
            if self.has_named_geometry("mpc_lines"):
                self._record_profile_metric(
                    "pygfx_mpc_lines_cache_skip_ms",
                    (time.perf_counter() - t_func_start) * 1000.0,
                )
                return True

        t_norm_start = time.perf_counter()
        normalized = self._normalize_mpc_lines_input(points, lines, colors)
        t_norm_end = time.perf_counter()
        self._record_profile_metric(
            "pygfx_mpc_lines_normalize_ms", (t_norm_end - t_norm_start) * 1000.0
        )
        if normalized is None:
            t_remove = time.perf_counter()
            removed = self._remove_mpc_geometry_if_present("mpc_lines")
            self._record_profile_metric(
                "pygfx_mpc_lines_invalid_remove_ms", (time.perf_counter() - t_remove) * 1000.0
            )
            if removed:
                self._mpc_lines_source_sig = None
            return removed
        points_arr, lines_arr, colors_arr = normalized

        t_expand_start = time.perf_counter()

        def upload_unified_lines() -> bool:
            """Upload all MPC types through the same geometry."""
            all_synced = True
            # Expand to disjoint segments so each segment gets its own vertex pair.
            fast_path_updated = False
            payload: Optional[LineSetPayload] = None
            if colors_arr is not None and len(colors_arr) == len(lines_arr):
                n_segments = len(lines_arr)
                color_cols = int(colors_arr.shape[1])
                active_rows = n_segments * 2
                cache_key = self._mpc_expanded_line_cache_key(
                    source_sig,
                    points,
                    lines,
                    colors,
                )
                cache_source_refs: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = (
                    (points, lines, colors) if cache_key is not None else None
                )
                t_cache_lookup = time.perf_counter()
                cache_entry = (
                    self._get_mpc_expanded_line_cache(cache_key, n_segments, color_cols)
                    if cache_key is not None
                    else None
                )
                self._record_profile_metric(
                    "pygfx_mpc_lines_expand_cache_lookup_ms",
                    (time.perf_counter() - t_cache_lookup) * 1000.0,
                )
                self._record_profile_metric(
                    "pygfx_mpc_lines_expand_cache_hit",
                    1.0 if cache_entry is not None else 0.0,
                )

                if cache_entry is not None:
                    fast_path_ready = self._ensure_mpc_segment_geometry_capacity(
                        n_segments,
                        color_cols,
                    )
                    self._record_profile_metric("pygfx_mpc_lines_expand_points_ms", 0.0)
                    self._record_profile_metric("pygfx_mpc_lines_expand_colors_ms", 0.0)
                    self._record_profile_array_bytes(
                        "pygfx_mpc_lines_payload_bytes",
                        cache_entry.points,
                        cache_entry.colors,
                    )
                    if fast_path_ready:
                        t_update = time.perf_counter()
                        fast_path_updated = self._update_mpc_segment_geometry_buffers(
                            cache_entry.points,
                            cache_entry.colors,
                            n_segments,
                        )
                        self._record_profile_metric(
                            "pygfx_mpc_lines_fast_update_ms",
                            (time.perf_counter() - t_update) * 1000.0,
                        )
                        self._record_profile_metric(
                            "pygfx_mpc_lines_active_segments",
                            float(n_segments),
                        )
                    if fast_path_updated:
                        self._record_profile_metric("pygfx_mpc_lines_payload_ms", 0.0)
                        self._register_pick_metadata("mpc_lines", {"type": "mpc_lines"})

                if not fast_path_updated:
                    fast_path_ready = self._ensure_mpc_segment_geometry_capacity(
                        n_segments,
                        color_cols,
                    )
                    t_expand_points = time.perf_counter()
                    point_rows = self._mpc_segment_capacity * 2 if fast_path_ready else active_rows
                    segment_points = self._ensure_buffer(
                        "_mpc_segment_points_buf",
                        (point_rows, 3),
                        np.float32,
                    )
                    active_segment_points = segment_points[:active_rows]
                    # Gather start/end positions as one contiguous endpoint stream.
                    # This avoids two temporary fancy-index arrays and strided writes.
                    np.take(
                        points_arr,
                        lines_arr.reshape(active_rows),
                        axis=0,
                        out=active_segment_points,
                    )
                    self._record_profile_metric(
                        "pygfx_mpc_lines_expand_points_ms",
                        (time.perf_counter() - t_expand_points) * 1000.0,
                    )

                    t_expand_colors = time.perf_counter()
                    color_rows = self._mpc_segment_capacity * 2 if fast_path_ready else active_rows
                    segment_colors = self._ensure_buffer(
                        "_mpc_segment_colors_buf",
                        (color_rows, color_cols),
                        np.float32,
                    )
                    active_segment_colors = segment_colors[:active_rows]
                    active_segment_colors[0::2] = colors_arr
                    active_segment_colors[1::2] = colors_arr
                    self._record_profile_metric(
                        "pygfx_mpc_lines_expand_colors_ms",
                        (time.perf_counter() - t_expand_colors) * 1000.0,
                    )
                    self._record_profile_array_bytes(
                        "pygfx_mpc_lines_payload_bytes",
                        active_segment_points,
                        active_segment_colors,
                    )
                    if cache_source_refs is not None:
                        self._store_mpc_expanded_line_cache(
                            cache_key,
                            active_segment_points,
                            active_segment_colors,
                            n_segments,
                            color_cols,
                            cache_source_refs,
                        )

                    if fast_path_ready:
                        t_update = time.perf_counter()
                        fast_path_updated = self._update_mpc_segment_geometry_buffers(
                            segment_points,
                            segment_colors,
                            n_segments,
                        )
                        self._record_profile_metric(
                            "pygfx_mpc_lines_fast_update_ms",
                            (time.perf_counter() - t_update) * 1000.0,
                        )
                        self._record_profile_metric(
                            "pygfx_mpc_lines_active_segments",
                            float(n_segments),
                        )

                    if not fast_path_updated:
                        t_payload = time.perf_counter()
                        payload = LineSetPayload(
                            points=active_segment_points,
                            lines=np.empty((0, 2), dtype=np.int32),
                            colors=active_segment_colors,
                        )
                        self._record_profile_metric(
                            "pygfx_mpc_lines_payload_ms",
                            (time.perf_counter() - t_payload) * 1000.0,
                        )
                    else:
                        self._record_profile_metric("pygfx_mpc_lines_payload_ms", 0.0)
                        self._register_pick_metadata("mpc_lines", {"type": "mpc_lines"})
            else:
                n_segments = len(lines_arr)
                t_payload = time.perf_counter()
                points_buf = self._copy_to_buffer("_mpc_lines_points_buf", points_arr, np.float32)
                lines_buf = self._copy_to_buffer("_mpc_lines_indices_buf", lines_arr, np.int32)
                colors_buf = (
                    None
                    if colors_arr is None
                    else self._copy_to_buffer("_mpc_lines_colors_buf", colors_arr, np.float32)
                )
                self._record_profile_array_bytes(
                    "pygfx_mpc_lines_payload_bytes", points_buf, lines_buf, colors_buf
                )
                payload = LineSetPayload(points=points_buf, lines=lines_buf, colors=colors_buf)
                self._record_profile_metric(
                    "pygfx_mpc_lines_payload_ms",
                    (time.perf_counter() - t_payload) * 1000.0,
                )

            if not fast_path_updated and payload is not None:
                t_ensure = time.perf_counter()
                all_synced = self.ensure_named_geometry("mpc_lines", payload) and all_synced
                self._record_profile_metric(
                    "pygfx_mpc_lines_ensure_named_geometry_ms",
                    (time.perf_counter() - t_ensure) * 1000.0,
                )
                if all_synced:
                    self._register_pick_metadata("mpc_lines", {"type": "mpc_lines"})
            elif fast_path_updated:
                all_synced = all_synced and True
            else:
                all_synced = False
            return all_synced

        all_synced = upload_unified_lines()

        t_expand_end = time.perf_counter()
        n_segments = len(lines_arr)

        if all_synced:
            self._mpc_lines_source_sig = source_sig
        self._record_profile_metric(
            "pygfx_mpc_lines_expand_and_upload_ms",
            (t_expand_end - t_expand_start) * 1000.0,
        )
        self._record_profile_metric(
            "pygfx_mpc_lines_total_ms",
            (t_expand_end - t_func_start) * 1000.0,
        )

        logger.debug(
            "[pygfx-telemetry] _apply_mpc_lines: %d segments, %d points | "
            "normalize=%.1fms expand+gpu=%.1fms total=%.1fms",
            n_segments,
            len(points_arr),
            (t_norm_end - t_norm_start) * 1000,
            (t_expand_end - t_expand_start) * 1000,
            (t_expand_end - t_norm_start) * 1000,
        )
        return all_synced

    def _apply_mpc_points(self, packet: FrameRenderPacket) -> bool:
        """Upload MPC bounce points and report complete native acceptance."""
        t_func_start = time.perf_counter()
        if not packet.mpc_visibility.effective_bounce_points:
            hidden = True
            if self.has_named_geometry("mpc_points"):
                t_hide = time.perf_counter()
                hidden = self.set_named_visibility("mpc_points", False)
                self._record_profile_metric(
                    "pygfx_mpc_points_hide_ms", (time.perf_counter() - t_hide) * 1000.0
                )
            self._set_marker_legend_visible(False)
            return hidden

        points = packet.mpc_bounce_points
        if points is None or len(points) == 0:
            t_remove = time.perf_counter()
            removed = self._remove_mpc_geometry_if_present("mpc_points")
            if removed:
                self._mpc_points_source_sig = None
            self._set_marker_legend_visible(False)
            self._record_profile_metric(
                "pygfx_mpc_points_empty_remove_ms", (time.perf_counter() - t_remove) * 1000.0
            )
            return removed

        point_colors = packet.mpc_bounce_colors
        self._record_profile_array_bytes("pygfx_mpc_points_input_bytes", points, point_colors)

        # Check signature BEFORE normalization to skip work when unchanged.
        t_sig_start = time.perf_counter()
        source_sig = self._mpc_point_source_signature(packet)
        self._record_profile_metric(
            "pygfx_mpc_points_signature_ms",
            (time.perf_counter() - t_sig_start) * 1000.0,
        )
        if source_sig == self._mpc_points_source_sig and self.has_named_geometry("mpc_points"):
            visible = True
            if self.is_named_visible("mpc_points") is False:
                visible = self.set_named_visibility("mpc_points", True)
            self._apply_mpc_point_markers(packet)
            self._record_profile_metric(
                "pygfx_mpc_points_cache_skip_ms",
                (time.perf_counter() - t_func_start) * 1000.0,
            )
            return visible

        t_norm_start = time.perf_counter()
        normalized = self._normalize_mpc_points_input(points, point_colors)
        self._record_profile_metric(
            "pygfx_mpc_points_normalize_ms",
            (time.perf_counter() - t_norm_start) * 1000.0,
        )
        if normalized is None:
            t_remove = time.perf_counter()
            removed = self._remove_mpc_geometry_if_present("mpc_points")
            if removed:
                self._mpc_points_source_sig = None
            self._set_marker_legend_visible(False)
            self._record_profile_metric(
                "pygfx_mpc_points_invalid_remove_ms", (time.perf_counter() - t_remove) * 1000.0
            )
            return removed
        points_arr, colors = normalized

        fast_path_updated = False
        upload_succeeded = False
        payload: Optional[PointCloudPayload] = None
        point_count = int(len(points_arr))
        if colors is not None and len(colors) == point_count:
            color_cols = int(colors.shape[1])
            t_ensure = time.perf_counter()
            fast_path_ready = self._ensure_mpc_point_geometry_capacity(point_count, color_cols)
            self._record_profile_metric(
                "pygfx_mpc_points_ensure_named_geometry_ms",
                (time.perf_counter() - t_ensure) * 1000.0,
            )
            if fast_path_ready:
                t_update = time.perf_counter()
                fast_path_updated = self._update_mpc_point_geometry_buffers(
                    points_arr,
                    colors,
                    point_count,
                )
                self._record_profile_metric(
                    "pygfx_mpc_points_fast_update_ms",
                    (time.perf_counter() - t_update) * 1000.0,
                )
                self._record_profile_metric(
                    "pygfx_mpc_points_active_points",
                    float(point_count),
                )
                self._record_profile_array_bytes(
                    "pygfx_mpc_points_payload_bytes",
                    points_arr,
                    colors,
                )
            if fast_path_updated:
                upload_succeeded = True
                if self.is_named_visible("mpc_points") is False:
                    upload_succeeded = self.set_named_visibility("mpc_points", True)
                if upload_succeeded:
                    self._mpc_points_source_sig = source_sig
                self._record_profile_metric("pygfx_mpc_points_payload_ms", 0.0)

        if not fast_path_updated:
            t_payload = time.perf_counter()
            payload = PointCloudPayload(
                points=self._copy_to_buffer("_mpc_points_points_buf", points_arr, np.float32),
                colors=(
                    None
                    if colors is None
                    else self._copy_to_buffer("_mpc_points_colors_buf", colors, np.float32)
                ),
            )
            self._record_profile_array_bytes(
                "pygfx_mpc_points_payload_bytes", payload.points, payload.colors
            )
            self._record_profile_metric(
                "pygfx_mpc_points_payload_ms",
                (time.perf_counter() - t_payload) * 1000.0,
            )
            # Validation is already performed inside _normalize_mpc_points_input();
            # skip the redundant _validate_mpc_point_payload() call.
            t_ensure = time.perf_counter()
            upload_succeeded = self.ensure_named_geometry("mpc_points", payload, visible=True)
            if upload_succeeded:
                self._mpc_points_source_sig = source_sig
            self._record_profile_metric(
                "pygfx_mpc_points_ensure_named_geometry_ms",
                (time.perf_counter() - t_ensure) * 1000.0,
            )
        # Always refresh marker state — the toggle can change frame-to-frame
        # independently of the point-cloud signature.
        t_markers = time.perf_counter()
        self._apply_mpc_point_markers(packet)
        self._record_profile_metric(
            "pygfx_mpc_points_marker_update_ms",
            (time.perf_counter() - t_markers) * 1000.0,
        )
        self._record_profile_metric(
            "pygfx_mpc_points_total_ms",
            (time.perf_counter() - t_func_start) * 1000.0,
        )
        return upload_succeeded

    def _apply_mpc_point_markers(self, packet: FrameRenderPacket) -> None:
        """Swap MPC points between plain and interaction-marker modes.

        When ``app_state.show_mpc_type_markers`` is False, the mpc_points mesh
        keeps the plain ``PointsMaterial`` that ``_build_points_material``
        installs. When True, the mesh uses a ``PointsMarkerMaterial`` in vertex
        marker mode and populate ``geometry.markers`` with per-bounce glyph
        IDs derived from ``packet.mpc_bounce_itypes``.

        LoS has no physical interaction marker. Specular, diffuse, refraction,
        diffraction, and virtual points use circle, triangle, diamond, plus,
        and square glyphs respectively; unknown values use a cross.
        """
        obj = self._objects.get("mpc_points")
        if obj is None:
            self._mpc_marker_cache_key = None
            self._set_marker_legend_visible(False)
            return
        gfx = self._gfx
        marker_cls = getattr(gfx, "PointsMarkerMaterial", None)
        plain_cls = getattr(gfx, "PointsMaterial", None)
        if marker_cls is None or plain_cls is None:
            self._mpc_marker_cache_key = None
            self._set_marker_legend_visible(False)
            return

        app_state = getattr(self.visualizer, "app_state", None)
        enabled = bool(getattr(app_state, "show_mpc_type_markers", False))
        if app_state is None:
            enabled = _env_flag("ORCHAV_PYGFX_MPC_MARKERS", False)
        current = getattr(obj, "material", None)

        if not enabled:
            self._mpc_marker_cache_key = None
            if isinstance(current, marker_cls):
                has_vc = packet.mpc_bounce_colors is not None
                try:
                    obj.material = self._build_points_material(has_vertex_colors=has_vc)
                    self._apply_named_visual_overrides("mpc_points")
                except Exception as exc:
                    logger.debug("Revert to PointsMaterial failed: %s", exc)
            self._set_marker_legend_visible(False)
            return

        itypes = packet.mpc_bounce_itypes
        points = packet.mpc_bounce_points
        point_count = points.shape[0] if points is not None else 0
        if itypes is None or itypes.size != point_count or point_count == 0:
            self._mpc_marker_cache_key = None
            self._set_marker_legend_visible(False)
            return

        has_vc = packet.mpc_bounce_colors is not None
        if not isinstance(current, marker_cls):
            kwargs: dict[str, Any] = {
                "color": (0.9, 0.9, 0.9, 1.0),
                "size": float(self._point_size),
                "pick_write": True,
                "marker": "circle",
                "marker_mode": "vertex",
                "edge_width": 1.0,
                "edge_color": (0.0, 0.0, 0.0, 1.0),
                # Type 0 is LoS and should not have a bounce glyph. Its
                # per-vertex marker is ``custom`` with this always-outside SDF.
                "custom_sdf": "return size;",
            }
            if has_vc:
                kwargs["color_mode"] = "vertex"
            try:
                new_mat = marker_cls(**kwargs)
            except TypeError:
                kwargs.pop("color_mode", None)
                new_mat = marker_cls(**kwargs)
            self._apply_clipping_to_material(new_mat)
            try:
                obj.material = new_mat
                current = new_mat
            except Exception as exc:
                logger.debug("Swap to PointsMarkerMaterial failed: %s", exc)
                self._mpc_marker_cache_key = None
                self._set_marker_legend_visible(False)
                return
        else:
            if hasattr(current, "size"):
                try:
                    current.size = float(self._point_size)
                except (AttributeError, TypeError):
                    pass
            if hasattr(current, "marker_mode"):
                try:
                    current.marker_mode = "vertex"
                except (AttributeError, ValueError):
                    pass

        geom = getattr(obj, "geometry", None)
        if geom is None:
            self._mpc_marker_cache_key = None
            self._set_marker_legend_visible(False)
            return

        point_revision = getattr(packet, "mpc_point_revision", None)
        itype_revision = (
            "ndarray",
            id(itypes),
            tuple(itypes.shape),
            itypes.dtype.str,
            tuple(itypes.strides),
            int(itypes.nbytes),
        )
        marker_cache_key = (
            "mpc-marker-v1",
            enabled,
            point_revision,
            itype_revision,
        )
        marker_buffer = getattr(geom, "markers", None)
        if (
            marker_cache_key == getattr(self, "_mpc_marker_cache_key", None)
            and marker_buffer is not None
        ):
            self._set_marker_legend_visible(True)
            return

        try:
            from pygfx import MarkerInt
        except ImportError:
            self._mpc_marker_cache_key = None
            self._set_marker_legend_visible(False)
            return

        marker_codes = getattr(self, "_mpc_marker_codes_buf", None)
        markers_arr = self._interaction_marker_codes(itypes, MarkerInt, out=marker_codes)
        self._mpc_marker_codes_buf = markers_arr
        try:
            if marker_buffer is None:
                geom.markers = gfx.Buffer(markers_arr)
            else:
                self._push_buffer(marker_buffer, markers_arr, label="mpc_markers")
        except Exception as exc:
            logger.debug("Failed to assign marker buffer: %s", exc)
            self._mpc_marker_cache_key = None
            self._set_marker_legend_visible(False)
            return
        self._mpc_marker_cache_key = marker_cache_key
        self._set_marker_legend_visible(True)

    @staticmethod
    def _interaction_marker_codes(
        itypes: np.ndarray,
        marker_int: Any,
        *,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Translate canonical interaction values to pygfx marker integers."""
        unknown = int(marker_int[UNKNOWN_INTERACTION_MARKER_SPEC.marker_name])
        hidden = int(marker_int["custom"])
        size = int(itypes.size)
        marker_codes = out
        if (
            marker_codes is None
            or marker_codes.shape != (size,)
            or marker_codes.dtype != np.int32
            or not marker_codes.flags.writeable
        ):
            marker_codes = np.empty(size, dtype=np.int32)
        marker_codes.fill(unknown)
        marker_codes[itypes == 0] = hidden
        for spec in INTERACTION_MARKER_SPECS:
            marker_codes[itypes == spec.interaction_type] = int(marker_int[spec.marker_name])
        return marker_codes

    def refresh_mpc_point_markers(self) -> None:
        """Re-apply the current marker/plain material choice to live MPC points."""
        packet = self.last_frame_packet
        if packet is None:
            self._set_marker_legend_visible(False)
            return
        self._apply_mpc_point_markers(packet)
        self.request_redraw()

    def _clear_mpc_expanded_line_cache(self) -> None:
        """Drop all CPU endpoint-expansion cache entries and byte accounting."""
        self._mpc_expanded_line_cache.clear()
        self._mpc_expanded_line_cache_bytes = 0

    def _mpc_expanded_line_cache_key(
        self,
        source_sig: tuple[Any, ...],
        points: Any,
        lines: Any,
        colors: Any,
    ) -> Optional[tuple[Any, ...]]:
        """Return a cache key only for ndarray-backed, cacheable MPC inputs."""
        if self._mpc_expanded_line_cache_max_bytes <= 0:
            return None
        if not (
            isinstance(points, np.ndarray)
            and isinstance(lines, np.ndarray)
            and isinstance(colors, np.ndarray)
        ):
            return None
        return source_sig

    @staticmethod
    def _mpc_line_source_signature(packet: FrameRenderPacket) -> tuple[Any, ...]:
        """Return the line cache key from the frame-packet MPC revision."""
        return packet.mpc_line_revision

    @staticmethod
    def _mpc_point_source_signature(packet: FrameRenderPacket) -> tuple[Any, ...]:
        """Return the point-cloud cache key from the frame-packet MPC revision."""
        return packet.mpc_point_revision

    def _get_mpc_expanded_line_cache(
        self,
        key: tuple[Any, ...],
        n_segments: int,
        color_cols: int,
    ) -> Optional[MpcExpandedLineCacheEntry]:
        """Fetch a valid endpoint-expansion cache entry and update hit stats."""
        entry = self._mpc_expanded_line_cache.get(key)
        if entry is None:
            self._mpc_expanded_line_cache_misses += 1
            return None
        if entry.n_segments != n_segments or entry.color_cols != color_cols:
            self._mpc_expanded_line_cache.pop(key, None)
            self._mpc_expanded_line_cache_bytes = max(
                0,
                self._mpc_expanded_line_cache_bytes - entry.byte_size,
            )
            self._mpc_expanded_line_cache_misses += 1
            return None
        self._mpc_expanded_line_cache.move_to_end(key)
        self._mpc_expanded_line_cache_hits += 1
        self._mpc_expanded_line_cache_last_entry_bytes = int(entry.byte_size)
        self._mpc_expanded_line_cache_largest_entry_bytes = max(
            self._mpc_expanded_line_cache_largest_entry_bytes,
            int(entry.byte_size),
        )
        return entry

    def _peek_mpc_expanded_line_cache(
        self,
        key: tuple[Any, ...],
        n_segments: int,
        color_cols: int,
    ) -> Optional[MpcExpandedLineCacheEntry]:
        """Return a valid expanded-line cache entry without changing hit stats."""
        entry = self._mpc_expanded_line_cache.get(key)
        if entry is None:
            return None
        if entry.n_segments != n_segments or entry.color_cols != color_cols:
            self._mpc_expanded_line_cache.pop(key, None)
            self._mpc_expanded_line_cache_bytes = max(
                0,
                self._mpc_expanded_line_cache_bytes - entry.byte_size,
            )
            return None
        self._mpc_expanded_line_cache.move_to_end(key)
        self._mpc_expanded_line_cache_last_entry_bytes = int(entry.byte_size)
        self._mpc_expanded_line_cache_largest_entry_bytes = max(
            self._mpc_expanded_line_cache_largest_entry_bytes,
            int(entry.byte_size),
        )
        return entry

    def prewarm_mpc_line_cache(self, packet: FrameRenderPacket) -> bool:
        """Populate the CPU expanded-line cache for a warmed frame packet.

        This intentionally does not create or update pygfx objects.  The normal
        render path still owns GPU buffer creation and upload, but it can skip
        endpoint/color expansion when playback reaches a prewarmed frame.
        """
        if (
            not self._mpc_expanded_line_cache_prewarm_enabled
            or self._mpc_expanded_line_cache_max_bytes <= 0
        ):
            return False

        self._mpc_expanded_line_cache_prewarm_attempts += 1
        t_start = time.perf_counter()
        try:
            if not packet.mpc_visibility.effective_paths:
                self._mpc_expanded_line_cache_prewarm_skips += 1
                return False

            points = packet.mpc_points
            lines = packet.mpc_lines
            colors = packet.mpc_colors
            if points is None or lines is None or colors is None:
                self._mpc_expanded_line_cache_prewarm_skips += 1
                return False
            if len(points) == 0 or len(lines) == 0:
                self._mpc_expanded_line_cache_prewarm_skips += 1
                return False

            source_sig = self._mpc_line_source_signature(packet)
            normalized = self._normalize_mpc_lines_input(points, lines, colors)
            if normalized is None:
                self._mpc_expanded_line_cache_prewarm_skips += 1
                return False

            points_arr, lines_arr, colors_arr = normalized
            if colors_arr is None or len(colors_arr) != len(lines_arr):
                self._mpc_expanded_line_cache_prewarm_skips += 1
                return False

            n_segments = len(lines_arr)
            if n_segments <= 0:
                self._mpc_expanded_line_cache_prewarm_skips += 1
                return False
            color_cols = int(colors_arr.shape[1])
            cache_key = self._mpc_expanded_line_cache_key(source_sig, points, lines, colors)
            if cache_key is None:
                self._mpc_expanded_line_cache_prewarm_skips += 1
                return False

            if self._peek_mpc_expanded_line_cache(cache_key, n_segments, color_cols) is not None:
                self._mpc_expanded_line_cache_prewarm_existing += 1
                return True

            active_rows = n_segments * 2
            segment_points = np.empty((active_rows, 3), dtype=np.float32)
            np.take(points_arr, lines_arr.reshape(active_rows), axis=0, out=segment_points)
            segment_colors = np.empty((active_rows, color_cols), dtype=np.float32)
            segment_colors[0::2] = colors_arr
            segment_colors[1::2] = colors_arr

            stores_before = self._mpc_expanded_line_cache_stores
            self._store_mpc_expanded_line_cache(
                cache_key,
                segment_points,
                segment_colors,
                n_segments,
                color_cols,
                (points, lines, colors),
            )
            if self._mpc_expanded_line_cache_stores > stores_before:
                self._mpc_expanded_line_cache_prewarm_stores += 1
                return True

            self._mpc_expanded_line_cache_prewarm_skips += 1
            return False
        finally:
            self._mpc_expanded_line_cache_prewarm_total_ms += (
                time.perf_counter() - t_start
            ) * 1000.0

    def _trim_mpc_expanded_line_cache(self) -> None:
        """Evict least-recently-used cache entries until under the byte limit."""
        while (
            self._mpc_expanded_line_cache
            and self._mpc_expanded_line_cache_bytes > self._mpc_expanded_line_cache_max_bytes
        ):
            _, entry = self._mpc_expanded_line_cache.popitem(last=False)
            self._mpc_expanded_line_cache_bytes = max(
                0,
                self._mpc_expanded_line_cache_bytes - entry.byte_size,
            )
            self._mpc_expanded_line_cache_evictions += 1

    def _store_mpc_expanded_line_cache(
        self,
        key: Optional[tuple[Any, ...]],
        segment_points: np.ndarray,
        segment_colors: np.ndarray,
        n_segments: int,
        color_cols: int,
        source_refs: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Store endpoint-expanded MPC arrays in the bounded CPU cache."""
        if key is None or self._mpc_expanded_line_cache_max_bytes <= 0:
            return
        points_copy = np.ascontiguousarray(segment_points[: n_segments * 2]).copy()
        colors_copy = np.ascontiguousarray(segment_colors[: n_segments * 2]).copy()
        # Source refs are retained to keep id()-based cache keys safe, but they
        # are not copied here.  Bound the cache by the incremental expanded
        # arrays it owns.
        byte_size = int(points_copy.nbytes + colors_copy.nbytes)
        self._mpc_expanded_line_cache_last_entry_bytes = byte_size
        self._mpc_expanded_line_cache_largest_entry_bytes = max(
            self._mpc_expanded_line_cache_largest_entry_bytes,
            byte_size,
        )
        if byte_size > self._mpc_expanded_line_cache_max_bytes:
            self._mpc_expanded_line_cache_rejected_oversize += 1
            return

        previous = self._mpc_expanded_line_cache.pop(key, None)
        if previous is not None:
            self._mpc_expanded_line_cache_bytes = max(
                0,
                self._mpc_expanded_line_cache_bytes - previous.byte_size,
            )
        self._mpc_expanded_line_cache[key] = MpcExpandedLineCacheEntry(
            points=points_copy,
            colors=colors_copy,
            n_segments=int(n_segments),
            color_cols=int(color_cols),
            byte_size=byte_size,
            source_refs=source_refs,
        )
        self._mpc_expanded_line_cache_bytes += byte_size
        self._mpc_expanded_line_cache_stores += 1
        self._trim_mpc_expanded_line_cache()

    def _clear_mpc_buffers(self) -> None:
        """Drop reusable numpy buffers and cached pygfx capacity hints."""
        self._mpc_lines_points_buf = None
        self._mpc_lines_indices_buf = None
        self._mpc_lines_colors_buf = None
        self._mpc_segment_points_buf = None
        self._mpc_segment_colors_buf = None
        self._mpc_segment_indices_buf = None
        self._clear_mpc_segment_capacity(preserve_hint=False)
        self._mpc_segment_capacity_hint = self._mpc_segment_capacity_floor
        self._clear_mpc_expanded_line_cache()
        self._mpc_points_points_buf = None
        self._mpc_points_colors_buf = None
        self._clear_mpc_point_capacity(preserve_hint=False)
        self._mpc_point_capacity_hint = 0

    def _clear_mpc_segment_capacity(self, *, preserve_hint: bool = True) -> None:
        """Forget realized line-buffer capacity, optionally keeping the growth hint."""
        if preserve_hint and self._mpc_segment_capacity > 0:
            self._mpc_segment_capacity_hint = max(
                self._mpc_segment_capacity_hint,
                self._mpc_segment_capacity,
            )
        self._mpc_segment_capacity = 0
        self._mpc_segment_color_cols = 0

    def _clear_mpc_point_capacity(self, *, preserve_hint: bool = True) -> None:
        """Forget realized point-buffer capacity, optionally keeping the growth hint."""
        if preserve_hint and self._mpc_point_capacity > 0:
            self._mpc_point_capacity_hint = max(
                self._mpc_point_capacity_hint,
                self._mpc_point_capacity,
            )
        self._mpc_point_capacity = 0
        self._mpc_point_color_cols = 0

    @staticmethod
    def _mpc_segment_capacity_for(n_segments: int) -> int:
        """Round requested segment count up to a power-of-two buffer capacity."""
        n_segments = max(1, int(n_segments))
        return 1 << (n_segments - 1).bit_length()

    @staticmethod
    def _mpc_point_capacity_for(n_points: int) -> int:
        """Round requested point count up to a power-of-two buffer capacity."""
        n_points = max(1, int(n_points))
        return 1 << (n_points - 1).bit_length()

    def _ensure_mpc_segment_indices(self, capacity: int) -> np.ndarray:
        """Build endpoint-pair indices for the reusable segment buffer."""
        indices = self._ensure_buffer("_mpc_segment_indices_buf", (capacity, 2), np.int32)
        base = np.arange(capacity, dtype=np.int32)
        indices[:, 0] = base * 2
        indices[:, 1] = base * 2 + 1
        return indices

    def _mpc_segment_geometry_has_capacity(self, n_segments: int, color_cols: int) -> bool:
        """Return whether the live line geometry can hold the next segment upload."""
        obj = self._objects.get("mpc_lines")
        geom = getattr(obj, "geometry", None)
        if obj is None or geom is None or self._mpc_segment_capacity < n_segments:
            return False

        positions = getattr(getattr(geom, "positions", None), "data", None)
        indices = getattr(getattr(geom, "indices", None), "data", None)
        colors = getattr(getattr(geom, "colors", None), "data", None)
        if positions is None or colors is None:
            return False
        if positions.shape[0] < n_segments * 2 or positions.shape[1:] != (3,):
            return False
        if indices is not None and (indices.shape[0] < n_segments or indices.shape[1:] != (2,)):
            return False
        if colors.shape[0] < n_segments * 2 or colors.shape[1:] != (color_cols,):
            return False
        return True

    def _ensure_mpc_segment_geometry_capacity(
        self,
        n_segments: int,
        color_cols: int,
    ) -> bool:
        """Ensure ``mpc_lines`` owns reusable endpoint-pair buffers."""
        if n_segments <= 0:
            return False
        if (
            self.has_named_geometry("mpc_lines")
            and self._mpc_segment_color_cols == color_cols
            and self._mpc_segment_geometry_has_capacity(n_segments, color_cols)
        ):
            self._mpc_segment_capacity_hint = max(
                self._mpc_segment_capacity_hint,
                self._mpc_segment_capacity,
            )
            self._record_profile_metric("pygfx_mpc_lines_ensure_named_geometry_ms", 0.0)
            return True

        t_ensure = time.perf_counter()
        prev_material = self._materials.get("mpc_lines")
        prev_transform = self._transforms.get("mpc_lines")
        prev_visible = self.is_named_visible("mpc_lines")
        if not self._remove_mpc_geometry_if_present("mpc_lines"):
            return False
        target_segments = max(int(n_segments), int(self._mpc_segment_capacity_hint))
        capacity = self._mpc_segment_capacity_for(target_segments)
        if self._mpc_segment_capacity > 0 and self._mpc_segment_color_cols == color_cols:
            capacity = max(capacity, self._mpc_segment_capacity)

        points = self._ensure_buffer("_mpc_segment_points_buf", (capacity * 2, 3), np.float32)
        colors = self._ensure_buffer(
            "_mpc_segment_colors_buf",
            (capacity * 2, color_cols),
            np.float32,
        )
        indices = self._ensure_mpc_segment_indices(capacity)
        payload = LineSetPayload(points=points, lines=indices, colors=colors)
        ok = self.ensure_named_geometry(
            "mpc_lines",
            payload,
            material=prev_material,
            transform=prev_transform,
            visible=prev_visible,
        )
        if ok:
            self._mpc_segment_capacity = capacity
            self._mpc_segment_color_cols = color_cols
            self._mpc_segment_capacity_hint = max(
                self._mpc_segment_capacity_hint,
                capacity,
            )
            self._set_mpc_segment_draw_ranges(0)
        self._record_profile_metric(
            "pygfx_mpc_lines_ensure_named_geometry_ms",
            (time.perf_counter() - t_ensure) * 1000.0,
        )
        self._record_profile_metric("pygfx_mpc_lines_capacity_segments", float(capacity))
        return ok

    def _set_mpc_segment_draw_ranges(self, n_segments: int) -> None:
        """Restrict reusable line buffers to the active segment prefix."""
        obj = self._objects.get("mpc_lines")
        geom = getattr(obj, "geometry", None)
        if geom is None:
            return
        self._set_buffer_draw_range(getattr(geom, "positions", None), n_segments * 2)
        self._set_buffer_draw_range(getattr(geom, "indices", None), n_segments)
        self._set_buffer_draw_range(getattr(geom, "colors", None), n_segments * 2)

    def _update_mpc_segment_geometry_buffers(
        self,
        segment_points: np.ndarray,
        segment_colors: np.ndarray,
        n_segments: int,
    ) -> bool:
        """Push active endpoint/color prefixes into the live line geometry."""
        obj = self._objects.get("mpc_lines")
        geom = getattr(obj, "geometry", None)
        if geom is None:
            return False
        try:
            self._push_buffer_prefix(
                geom.positions,
                segment_points[: n_segments * 2],
                label="mpc_line_positions",
            )
            self._push_buffer_prefix(
                geom.colors,
                segment_colors[: n_segments * 2],
                label="mpc_line_colors",
            )
            self._set_buffer_draw_range(getattr(geom, "indices", None), n_segments)
            return True
        except Exception as exc:
            logger.debug("PygfxRenderer: mpc_lines prefix update failed: %s", exc)
            return False

    def _mpc_point_geometry_has_capacity(self, n_points: int, color_cols: int) -> bool:
        """Return whether the live point geometry can hold the next point upload."""
        obj = self._objects.get("mpc_points")
        geom = getattr(obj, "geometry", None)
        if obj is None or geom is None or self._mpc_point_capacity < n_points:
            return False

        positions = getattr(getattr(geom, "positions", None), "data", None)
        colors = getattr(getattr(geom, "colors", None), "data", None)
        if positions is None or colors is None:
            return False
        if positions.shape[0] < n_points or positions.shape[1:] != (3,):
            return False
        if colors.shape[0] < n_points or colors.shape[1:] != (color_cols,):
            return False
        return True

    def _ensure_mpc_point_geometry_capacity(self, n_points: int, color_cols: int) -> bool:
        """Ensure ``mpc_points`` owns reusable point/color buffers."""
        if n_points <= 0 or color_cols <= 0:
            return False
        if (
            self.has_named_geometry("mpc_points")
            and self._mpc_point_color_cols == color_cols
            and self._mpc_point_geometry_has_capacity(n_points, color_cols)
        ):
            self._mpc_point_capacity_hint = max(
                self._mpc_point_capacity_hint,
                self._mpc_point_capacity,
            )
            return True

        prev_material = self._materials.get("mpc_points")
        prev_transform = self._transforms.get("mpc_points")
        prev_visible = self.is_named_visible("mpc_points")
        if not self._remove_mpc_geometry_if_present("mpc_points"):
            return False
        target_points = max(int(n_points), int(self._mpc_point_capacity_hint))
        capacity = self._mpc_point_capacity_for(target_points)
        if self._mpc_point_capacity > 0 and self._mpc_point_color_cols == color_cols:
            capacity = max(capacity, self._mpc_point_capacity)

        points = self._ensure_buffer("_mpc_points_points_buf", (capacity, 3), np.float32)
        colors = self._ensure_buffer("_mpc_points_colors_buf", (capacity, color_cols), np.float32)
        payload = PointCloudPayload(points=points, colors=colors)
        ok = self.ensure_named_geometry(
            "mpc_points",
            payload,
            material=prev_material,
            transform=prev_transform,
            visible=True if prev_visible is None else prev_visible,
        )
        if ok:
            self._mpc_point_capacity = capacity
            self._mpc_point_color_cols = color_cols
            self._mpc_point_capacity_hint = max(self._mpc_point_capacity_hint, capacity)
            self._set_mpc_point_draw_ranges(0)
        self._record_profile_metric("pygfx_mpc_points_capacity_points", float(capacity))
        return ok

    def _set_mpc_point_draw_ranges(self, n_points: int) -> None:
        """Restrict reusable point buffers to the active point prefix."""
        obj = self._objects.get("mpc_points")
        geom = getattr(obj, "geometry", None)
        if geom is None:
            return
        self._set_buffer_draw_range(getattr(geom, "positions", None), n_points)
        self._set_buffer_draw_range(getattr(geom, "colors", None), n_points)

    def _update_mpc_point_geometry_buffers(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        n_points: int,
    ) -> bool:
        """Push active position/color prefixes into the live point geometry."""
        obj = self._objects.get("mpc_points")
        geom = getattr(obj, "geometry", None)
        if geom is None:
            return False
        try:
            self._push_buffer_prefix(
                geom.positions,
                points[:n_points],
                label="mpc_point_positions",
            )
            self._push_buffer_prefix(
                geom.colors,
                colors[:n_points],
                label="mpc_bounce_colors",
            )
            return True
        except Exception as exc:
            logger.debug("PygfxRenderer: mpc_points prefix update failed: %s", exc)
            return False

    def _copy_to_buffer(self, attr_name: str, src: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Copy *src* into a reusable numpy buffer attribute."""
        t_start = time.perf_counter()
        src_arr = (
            src
            if (isinstance(src, np.ndarray) and src.dtype == dtype)
            else np.asarray(src, dtype=dtype)
        )
        dst = self._ensure_buffer(attr_name, src_arr.shape, dtype)
        np.copyto(dst, src_arr, casting="unsafe")
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        safe_name = attr_name.lstrip("_").replace("::", "_").replace(" ", "_")
        self._record_profile_metric("pygfx_copy_to_numpy_buffer_ms", elapsed_ms)
        self._record_profile_metric(f"pygfx_copy_to_numpy_buffer_{safe_name}_ms", elapsed_ms)
        nbytes = float(getattr(dst, "nbytes", 0))
        self._record_profile_bytes("pygfx_copy_to_numpy_buffer_bytes", nbytes)
        self._record_profile_bytes(f"pygfx_copy_to_numpy_buffer_{safe_name}_bytes", nbytes)
        return dst

    def _ensure_buffer(self, attr_name: str, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Return a contiguous reusable numpy buffer with the requested shape/dtype."""
        dst = getattr(self, attr_name, None)
        if (
            not isinstance(dst, np.ndarray)
            or dst.shape != shape
            or dst.dtype != np.dtype(dtype)
            or not dst.flags.c_contiguous
            or not dst.flags.writeable
            or not dst.flags.owndata
        ):
            dst = np.empty(shape=shape, dtype=dtype)
            setattr(self, attr_name, dst)
        return dst

    def _normalize_mpc_lines_input(
        self, points: Any, lines: Any, colors: Any
    ) -> Optional[tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
        """Normalize frame-packet MPC line arrays to GPU-ready shapes and dtypes."""
        points_arr = points if isinstance(points, np.ndarray) else np.asarray(points)
        lines_arr = lines if isinstance(lines, np.ndarray) else np.asarray(lines)
        if points_arr.ndim != 2 or points_arr.shape[1] < 3:
            logger.warning(
                "PygfxRenderer: invalid MPC points shape %s; expected (N, >=3)",
                points_arr.shape,
            )
            return None
        if lines_arr.ndim == 1 and lines_arr.size % 2 == 0:
            lines_arr = lines_arr.reshape(-1, 2)
        if lines_arr.ndim != 2 or lines_arr.shape[1] != 2:
            logger.warning(
                "PygfxRenderer: invalid MPC lines shape %s; expected (M, 2)",
                lines_arr.shape,
            )
            return None

        # Convert to GPU-ready dtypes, avoiding a copy when already correct.
        pts_slice = points_arr[:, :3]
        if pts_slice.dtype == np.float32 and pts_slice.flags["C_CONTIGUOUS"]:
            points_norm = pts_slice
        else:
            points_norm = np.ascontiguousarray(pts_slice, dtype=np.float32)

        if lines_arr.dtype == np.int32 and lines_arr.flags["C_CONTIGUOUS"]:
            lines_norm = lines_arr
        else:
            lines_norm = np.ascontiguousarray(lines_arr, dtype=np.int32)

        if len(lines_norm) == 0:
            return points_norm, lines_norm, None

        n_pts = len(points_norm)
        all_valid = bool(
            lines_norm[:, 0].min() >= 0
            and lines_norm[:, 1].min() >= 0
            and lines_norm[:, 0].max() < n_pts
            and lines_norm[:, 1].max() < n_pts
        )
        if not all_valid:
            valid = (
                (lines_norm[:, 0] >= 0)
                & (lines_norm[:, 1] >= 0)
                & (lines_norm[:, 0] < n_pts)
                & (lines_norm[:, 1] < n_pts)
            )
            invalid_count = int(np.count_nonzero(~valid))
            logger.warning(
                "PygfxRenderer: dropping %d invalid MPC line indices",
                invalid_count,
            )
            lines_norm = lines_norm[valid]
            if len(lines_norm) == 0:
                return points_norm, lines_norm, None
        else:
            valid = None  # sentinel: all lines valid

        colors_norm: Optional[np.ndarray] = None
        if colors is not None:
            raw_colors = colors if isinstance(colors, np.ndarray) else np.asarray(colors)
            if raw_colors.ndim == 2 and raw_colors.shape[1] in (3, 4):
                n_lines_orig = len(lines_arr)
                if raw_colors.shape[0] == n_lines_orig:
                    if valid is None:
                        # All valid — convert dtype only if needed.
                        if raw_colors.dtype == np.float32 and raw_colors.flags["C_CONTIGUOUS"]:
                            colors_norm = raw_colors
                        else:
                            colors_norm = np.ascontiguousarray(raw_colors, dtype=np.float32)
                    else:
                        colors_norm = np.ascontiguousarray(raw_colors[valid], dtype=np.float32)
                elif raw_colors.shape[0] == n_pts:
                    if raw_colors.dtype == np.float32 and raw_colors.flags["C_CONTIGUOUS"]:
                        colors_norm = raw_colors
                    else:
                        colors_norm = np.ascontiguousarray(raw_colors, dtype=np.float32)
                else:
                    logger.warning(
                        "PygfxRenderer: ignoring MPC colors with incompatible length %d "
                        "(points=%d, lines=%d)",
                        raw_colors.shape[0],
                        n_pts,
                        len(lines_norm),
                    )
            else:
                logger.warning(
                    "PygfxRenderer: ignoring MPC colors with invalid shape %s; expected (K,3|4)",
                    raw_colors.shape,
                )
        return points_norm, lines_norm, colors_norm

    def _normalize_mpc_points_input(
        self, points: Any, point_colors: Any
    ) -> Optional[tuple[np.ndarray, Optional[np.ndarray]]]:
        """Normalize frame-packet bounce-point arrays to GPU-ready shapes and dtypes."""
        points_arr = points if isinstance(points, np.ndarray) else np.asarray(points)
        if points_arr.ndim != 2 or points_arr.shape[1] < 3:
            logger.warning(
                "PygfxRenderer: invalid MPC point-cloud shape %s; expected (N, >=3)",
                points_arr.shape,
            )
            return None
        pts_slice = points_arr[:, :3]
        if pts_slice.dtype == np.float32 and pts_slice.flags["C_CONTIGUOUS"]:
            points_norm = pts_slice
        else:
            points_norm = np.ascontiguousarray(pts_slice, dtype=np.float32)
        colors_norm: Optional[np.ndarray] = None
        if point_colors is not None:
            raw_colors = (
                point_colors if isinstance(point_colors, np.ndarray) else np.asarray(point_colors)
            )
            if (
                raw_colors.ndim == 2
                and raw_colors.shape[0] == len(points_norm)
                and raw_colors.shape[1] in (3, 4)
            ):
                if raw_colors.dtype == np.float32 and raw_colors.flags["C_CONTIGUOUS"]:
                    colors_norm = raw_colors
                else:
                    colors_norm = np.ascontiguousarray(raw_colors, dtype=np.float32)
            else:
                logger.warning(
                    "PygfxRenderer: ignoring MPC point colors with invalid shape %s",
                    raw_colors.shape,
                )
        return points_norm, colors_norm

    @staticmethod
    def _validate_mpc_line_payload(payload: LineSetPayload) -> bool:
        """Validate direct line payloads before using the MPC fast path."""
        points = np.asarray(payload.points)
        lines = np.asarray(payload.lines)
        if points.ndim != 2 or points.shape[1] != 3:
            logger.warning("PygfxRenderer: invalid MPC line points shape %s", points.shape)
            return False
        if lines.ndim != 2 or lines.shape[1] != 2:
            logger.warning("PygfxRenderer: invalid MPC line index shape %s", lines.shape)
            return False
        if payload.colors is not None:
            colors = np.asarray(payload.colors)
            if colors.ndim != 2 or colors.shape[1] not in (3, 4):
                logger.warning("PygfxRenderer: invalid MPC line color shape %s", colors.shape)
                return False
            if colors.shape[0] not in (len(lines), len(points)):
                logger.warning(
                    "PygfxRenderer: invalid MPC line color count %d (points=%d, lines=%d)",
                    colors.shape[0],
                    len(points),
                    len(lines),
                )
                return False
        return True

    @staticmethod
    def _validate_mpc_point_payload(payload: PointCloudPayload) -> bool:
        """Validate direct point payloads before using the MPC fast path."""
        points = np.asarray(payload.points)
        if points.ndim != 2 or points.shape[1] != 3:
            logger.warning("PygfxRenderer: invalid MPC points shape %s", points.shape)
            return False
        if payload.colors is None:
            return True
        colors = np.asarray(payload.colors)
        if colors.ndim != 2 or colors.shape[1] not in (3, 4):
            logger.warning("PygfxRenderer: invalid MPC point color shape %s", colors.shape)
            return False
        if colors.shape[0] != len(points):
            logger.warning(
                "PygfxRenderer: invalid MPC point color count %d (points=%d)",
                colors.shape[0],
                len(points),
            )
            return False
        return True
