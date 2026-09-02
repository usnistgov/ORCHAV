"""pygfx presentation for renderer-neutral RF X-Ray snapshots."""

from __future__ import annotations

import html
import logging
from typing import TYPE_CHECKING, Any

from ...services.rf_xray_analysis_service import (
    RFXRAY_MODE_MATERIAL_MAP,
    RFXRAY_MODE_MATERIAL_PROPERTIES,
    RFXRAY_MODE_MPC_USAGE,
    RFXRayAnalysisSnapshot,
    RFXRayLegendEntry,
)
from ...types.render_payloads import LineSetPayload, MaterialPayload, SurfaceColorSource
from .explanatory_overlay import configure_explanatory_overlay

if TYPE_CHECKING:
    from ...pipeline.core import FrameRenderPacket

logger = logging.getLogger(__name__)

RF_XRAY_BOUNCES_NAME = "rf_xray::bounces"
RF_XRAY_COLORBAR_OVERLAY_ID = "rf_xray_colorbar"
RF_XRAY_LEGEND_OVERLAY_ID = "rf_xray_material_legend"
RF_XRAY_TOP_PATHS_NAME = "rf_xray::top_paths"


class PygfxRFXRayMixin:
    """Apply RF X-Ray material colors and overlay geometry to pygfx."""

    def _initialize_rf_xray_state(self) -> None:
        """Initialize mutable RF X-Ray renderer state."""
        self._rf_xray_base_materials: dict[str, MaterialPayload] = {}
        self._rf_xray_overlaid_names: set[str] = set()
        self._rf_xray_vertex_color_names: set[str] = set()
        self._rf_xray_last_signature: tuple[Any, ...] | None = None
        self._rf_xray_last_snapshot: RFXRayAnalysisSnapshot | None = None
        self._rf_xray_applying_overlay: bool = False

    def _apply_rf_xray_overlay(self, packet: "FrameRenderPacket") -> bool:
        """Apply or clear RF X-Ray overlay state for the frame packet.

        Returns True when renderer-visible state changed.
        """
        service = getattr(self.visualizer, "rf_xray_analysis_service", None)
        if service is None:
            self._rf_xray_last_snapshot = None
            return self._clear_rf_xray_overlay()

        snapshot = service.build_snapshot(packet)
        self._rf_xray_last_snapshot = snapshot
        if snapshot.signature == self._rf_xray_last_signature:
            return False

        self._rf_xray_last_signature = snapshot.signature
        self._publish_rf_xray_summary(snapshot)
        self._update_rf_xray_hud_overlay(snapshot)

        if not snapshot.enabled:
            return self._clear_rf_xray_overlay()

        changed = self._apply_rf_xray_materials(snapshot)
        if self.has_named_geometry(RF_XRAY_BOUNCES_NAME):
            changed = self.remove_named_geometry(RF_XRAY_BOUNCES_NAME) or changed
        changed = self._apply_rf_xray_top_paths(snapshot) or changed
        return changed

    def _refresh_rf_xray_hud_overlay(self) -> None:
        """Reapply the last RF X-Ray legend under the current HUD policy."""
        snapshot = self._rf_xray_last_snapshot
        if snapshot is not None:
            self._update_rf_xray_hud_overlay(snapshot)
            return
        clear_hud = getattr(self, "_clear_hud_overlay", None)
        if callable(clear_hud):
            clear_hud(RF_XRAY_COLORBAR_OVERLAY_ID)
            clear_hud(RF_XRAY_LEGEND_OVERLAY_ID)

    def _clear_rf_xray_overlay(self) -> bool:
        """Restore base materials and remove RF X-Ray-only geometries."""
        changed = False
        clear_hud = getattr(self, "_clear_hud_overlay", None)
        if callable(clear_hud):
            clear_hud(RF_XRAY_COLORBAR_OVERLAY_ID)
            clear_hud(RF_XRAY_LEGEND_OVERLAY_ID)
        self._rf_xray_applying_overlay = True
        try:
            for name, material in list(self._rf_xray_base_materials.items()):
                if name in self._rf_xray_vertex_color_names:
                    self._geometry_color_sources[name] = SurfaceColorSource.VERTEX
                if self.has_named_geometry(name):
                    changed = self.set_named_material(name, material) or changed
            for name in (RF_XRAY_BOUNCES_NAME, RF_XRAY_TOP_PATHS_NAME):
                if self.has_named_geometry(name):
                    changed = self.remove_named_geometry(name) or changed
        finally:
            self._rf_xray_applying_overlay = False
            self._rf_xray_base_materials.clear()
            self._rf_xray_overlaid_names.clear()
            self._rf_xray_vertex_color_names.clear()
        return changed

    def _update_rf_xray_hud_overlay(self, snapshot: RFXRayAnalysisSnapshot) -> None:
        """Show the pygfx RF X-Ray legend or scalar colorbar."""
        clear_hud = getattr(self, "_clear_hud_overlay", None)
        if not snapshot.enabled:
            if callable(clear_hud):
                clear_hud(RF_XRAY_COLORBAR_OVERLAY_ID)
                clear_hud(RF_XRAY_LEGEND_OVERLAY_ID)
            return

        if snapshot.mode == RFXRAY_MODE_MATERIAL_MAP:
            if callable(clear_hud):
                clear_hud(RF_XRAY_COLORBAR_OVERLAY_ID)
            self._update_rf_xray_material_legend(snapshot)
            return

        if callable(clear_hud):
            clear_hud(RF_XRAY_LEGEND_OVERLAY_ID)
        if snapshot.mode == RFXRAY_MODE_MATERIAL_PROPERTIES:
            self._update_rf_xray_property_colorbar(snapshot)
            return
        if snapshot.mode == RFXRAY_MODE_MPC_USAGE and snapshot.usage:
            self._update_rf_xray_usage_colorbar()
            return

        if callable(clear_hud):
            clear_hud(RF_XRAY_COLORBAR_OVERLAY_ID)

    def _update_rf_xray_usage_colorbar(self) -> None:
        """Show the relative MPC-usage scalar colorbar."""
        html_builder = getattr(self, "_colorbar_overlay_html", None)
        set_hud = getattr(self, "_set_hud_overlay", None)
        clear_hud = getattr(self, "_clear_hud_overlay", None)
        if not callable(html_builder) or not callable(set_hud):
            if callable(clear_hud):
                clear_hud(RF_XRAY_COLORBAR_OVERLAY_ID)
            return

        html = html_builder("RF X-Ray MPC Usage (relative)", (0.0, 1.0))
        if not html:
            if callable(clear_hud):
                clear_hud(RF_XRAY_COLORBAR_OVERLAY_ID)
            return
        set_hud(
            RF_XRAY_COLORBAR_OVERLAY_ID,
            html=html,
            visible=True,
            role="legend",
            corner="top_right",
            priority=25,
        )

    def _update_rf_xray_property_colorbar(self, snapshot: RFXRayAnalysisSnapshot) -> None:
        """Show the selected material-property scalar colorbar."""
        html_builder = getattr(self, "_colorbar_overlay_html", None)
        set_hud = getattr(self, "_set_hud_overlay", None)
        clear_hud = getattr(self, "_clear_hud_overlay", None)
        if not callable(html_builder) or not callable(set_hud) or snapshot.scalar_range is None:
            if callable(clear_hud):
                clear_hud(RF_XRAY_COLORBAR_OVERLAY_ID)
            return

        label = snapshot.scalar_property_label or snapshot.scalar_property or "Material property"
        html_text = html_builder(f"RF X-Ray: {label}", snapshot.scalar_range)
        if not html_text:
            if callable(clear_hud):
                clear_hud(RF_XRAY_COLORBAR_OVERLAY_ID)
            return
        set_hud(
            RF_XRAY_COLORBAR_OVERLAY_ID,
            html=html_text,
            visible=True,
            role="legend",
            corner="top_right",
            priority=25,
        )

    def _update_rf_xray_material_legend(self, snapshot: RFXRayAnalysisSnapshot) -> None:
        """Show the material-map swatch legend."""
        set_hud = getattr(self, "_set_hud_overlay", None)
        clear_hud = getattr(self, "_clear_hud_overlay", None)
        if not callable(set_hud) or not snapshot.legend_entries:
            if callable(clear_hud):
                clear_hud(RF_XRAY_LEGEND_OVERLAY_ID)
            return
        html_text = self._rf_xray_material_legend_html(snapshot.legend_entries)
        if not html_text:
            if callable(clear_hud):
                clear_hud(RF_XRAY_LEGEND_OVERLAY_ID)
            return
        set_hud(
            RF_XRAY_LEGEND_OVERLAY_ID,
            html=html_text,
            visible=True,
            role="legend",
            corner="top_right",
            priority=25,
        )

    @classmethod
    def _rf_xray_material_legend_html(
        cls,
        entries: tuple[RFXRayLegendEntry, ...],
    ) -> str:
        """Return compact HTML for the RF X-Ray Material Map legend."""
        visible_entries = tuple(entries[:12])
        rows = []
        for entry in visible_entries:
            color = cls._rf_xray_color_hex(entry.color)
            name = html.escape(entry.display_name)
            suffix = " (unknown)" if entry.missing_data else ""
            rows.append(
                "<tr>"
                "<td style='padding:0 8px 3px 0;'>"
                f"<span style='color:{color}; font-size:14px;'>&#9632;</span>"
                "</td>"
                f"<td>{name}{suffix}</td>"
                "</tr>"
            )
        if len(entries) > len(visible_entries):
            rows.append(
                "<tr><td></td>" f"<td>+{len(entries) - len(visible_entries)} more</td></tr>"
            )
        return (
            "<div style='font-weight:600; margin-bottom:4px;'>RF X-Ray Material Map</div>"
            "<table cellspacing='0' cellpadding='0'>" + "".join(rows) + "</table>"
        )

    @staticmethod
    def _rf_xray_color_hex(color: tuple[float, float, float, float]) -> str:
        """Convert normalized RGB-like RF X-Ray colors to CSS hex."""
        rgb = []
        for channel in color[:3]:
            try:
                value = int(round(max(0.0, min(1.0, float(channel))) * 255.0))
            except (TypeError, ValueError):
                value = 179
            rgb.append(value)
        while len(rgb) < 3:
            rgb.append(179)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    def _apply_rf_xray_materials(self, snapshot: RFXRayAnalysisSnapshot) -> bool:
        """Apply snapshot material colors to existing scene/target meshes."""
        next_names = {
            name
            for name in snapshot.geometry_colors
            if self.has_named_geometry(name) and self._kinds.get(name) == "mesh"
        }
        changed = False

        stale_names = self._rf_xray_overlaid_names - next_names
        self._rf_xray_applying_overlay = True
        try:
            for name in stale_names:
                base_material = self._rf_xray_base_materials.pop(name, None)
                if base_material is None:
                    continue
                if name in self._rf_xray_vertex_color_names:
                    self._geometry_color_sources[name] = SurfaceColorSource.VERTEX
                changed = self.set_named_material(name, base_material) or changed
                self._rf_xray_vertex_color_names.discard(name)

            for name in sorted(next_names):
                if name not in self._rf_xray_base_materials:
                    base_material = self._materials.get(name)
                    if base_material is not None:
                        self._rf_xray_base_materials[name] = base_material
                    if self._geometry_color_sources.get(name) is SurfaceColorSource.VERTEX:
                        self._rf_xray_vertex_color_names.add(name)
                self._geometry_color_sources[name] = SurfaceColorSource.MATERIAL
                color = snapshot.geometry_colors[name]
                overlay_material = self._rf_xray_material(color)
                changed = self.set_named_material(name, overlay_material) or changed
        finally:
            self._rf_xray_applying_overlay = False

        self._rf_xray_overlaid_names = next_names
        return changed

    def _apply_rf_xray_top_paths(self, snapshot: RFXRayAnalysisSnapshot) -> bool:
        """Create/update optional strongest-path line overlay."""
        if (
            snapshot.top_path_points is None
            or snapshot.top_path_lines is None
            or snapshot.top_path_colors is None
        ):
            if self.has_named_geometry(RF_XRAY_TOP_PATHS_NAME):
                return self.remove_named_geometry(RF_XRAY_TOP_PATHS_NAME)
            return False

        payload = LineSetPayload(
            points=snapshot.top_path_points,
            lines=snapshot.top_path_lines,
            colors=snapshot.top_path_colors,
        )
        material = MaterialPayload(
            base_color=(1.0, 1.0, 1.0, 1.0),
            shader="unlit",
            line_width=max(4.0, float(getattr(self, "_line_width", 2.0)) * 1.8),
        )
        applied = bool(
            self.ensure_named_geometry(
                RF_XRAY_TOP_PATHS_NAME,
                payload,
                material=material,
                visible=True,
            )
        )
        if applied:
            # This explanatory overlay sits directly over bulk ``mpc_lines``.
            # Keep the canonical bulk line as the viewport pick target.
            configure_explanatory_overlay(self, RF_XRAY_TOP_PATHS_NAME)
        return applied

    @staticmethod
    def _rf_xray_material(color: tuple[float, float, float, float]) -> MaterialPayload:
        """Return an untextured inspection material for RF X-Ray scene coloring."""
        return MaterialPayload(
            base_color=(
                float(color[0]),
                float(color[1]),
                float(color[2]),
                float(color[3]) if len(color) > 3 else 0.85,
            ),
            roughness=0.65,
            metallic=0.0,
            reflectance=0.35,
            shader="unlit",
            texture_path=None,
            normal_map_path=None,
            roughness_map_path=None,
            ao_map_path=None,
            metallic_map_path=None,
        )

    def _publish_rf_xray_summary(self, snapshot: RFXRayAnalysisSnapshot) -> None:
        """Expose the latest RF X-Ray summary to the Materials panel when present."""
        try:
            self.visualizer.rf_xray_summary_text = snapshot.summary
            panel = getattr(getattr(self.visualizer, "ui_manager", None), "panels", {}).get(
                "materials"
            )
            if panel is not None and hasattr(panel, "set_rf_xray_status"):
                panel.set_rf_xray_status(snapshot.summary, active=snapshot.enabled)
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Unable to publish RF X-Ray summary: %s", exc)
