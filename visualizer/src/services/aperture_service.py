"""Service for managing AOA/AOD aperture visualizations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from shared.logging import get_logger

from ..model import RenderObjectState
from ..renderers.protocol import renderer_capabilities
from ..scene.geometry_helpers import create_text_label
from ..types.render_payloads import MaterialPayload
from ..utils.aperture_geometry import (
    AOA_APERTURE_COLOR,
    AOD_APERTURE_COLOR,
    angular_reference_label_positions,
    create_angular_reference_line_payload,
    create_aperture_line_payload,
    create_aperture_mesh_payload,
)
from .base import BaseService

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = get_logger("orchav.services.aperture_service")


def _same_snapshot_value(left: Any, right: Any) -> bool:
    """Compare immutable renderer state without NumPy's ambiguous truth values."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))
        except (TypeError, ValueError):
            return False
    if is_dataclass(left) or is_dataclass(right):
        if type(left) is not type(right) or not is_dataclass(left):
            return False
        return all(
            _same_snapshot_value(getattr(left, field.name), getattr(right, field.name))
            for field in fields(left)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        return left.keys() == right.keys() and all(
            _same_snapshot_value(left[key], right[key]) for key in left
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
            return False
        return len(left) == len(right) and all(
            _same_snapshot_value(a, b) for a, b in zip(left, right)
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _same_render_state(left: RenderObjectState, right: RenderObjectState) -> bool:
    """Return whether two aperture handles describe the same complete snapshot."""
    return bool(
        left.id == right.id
        and _same_snapshot_value(left.payload, right.payload)
        and _same_snapshot_value(left.material, right.material)
        and np.array_equal(left.world_transform.matrix, right.world_transform.matrix)
        and left.visible is right.visible
        and left.is_edge is right.is_edge
        and _same_snapshot_value(left.metadata, right.metadata)
    )


class ApertureService(BaseService):
    """Service to manage AOA/AOD angular-sector visualizations.

    This service creates and updates filled spherical sectors that visualize the
    angular filter apertures at TX (for AOD) and RX (for AOA) positions.
    """

    def __init__(self, visualizer: OrchavVisualizer) -> None:
        """Bind aperture preview state to the visualizer and active geometry sets."""
        super().__init__()
        self.visualizer = visualizer
        # Track aperture geometries by name -> geometry object
        self._active_aoa_geometries: dict[str, Any] = {}
        self._active_aod_geometries: dict[str, Any] = {}
        self._active_reference_geometries: dict[str, Any] = {}
        self._collecting_desired_geometry = False

    @staticmethod
    def _resolve_preview_bounds(state: Any, prefix: str) -> tuple[float, float, float, float]:
        """Return drawable aperture bounds for a filter prefix.

        Range filters store natural limits as ``None`` to mean "no filtering".
        The visual preview still needs explicit bounds so the user can draw a
        full all-angle aperture when AoA/AoD preview is enabled.
        """
        az_min = getattr(state, f"{prefix}_az_filter_min_deg", None)
        az_max = getattr(state, f"{prefix}_az_filter_max_deg", None)
        el_min = getattr(state, f"{prefix}_el_filter_min_deg", None)
        el_max = getattr(state, f"{prefix}_el_filter_max_deg", None)
        return (
            -180.0 if az_min is None else float(az_min),
            180.0 if az_max is None else float(az_max),
            -90.0 if el_min is None else float(el_min),
            90.0 if el_max is None else float(el_max),
        )

    def update_apertures(self) -> bool:
        """Update aperture visualizations based on current state.

        This method should be called when:
        - Visibility toggles change (show_aoa_aperture, show_aod_aperture)
        - Any angle filter value changes
        - TX/RX positions change (frame navigation)
        - Aperture radius changes
        - Node selection changes (selected_tx, selected_rx)
        """
        viz = self.visualizer
        state = viz.app_state

        logger.info(
            f"ApertureService.update_apertures: show_aoa={state.show_aoa_aperture}, "
            f"show_aod={state.show_aod_aperture}, selected_tx={state.selected_tx}, "
            f"selected_rx={state.selected_rx}, "
            f"show_global_ref={getattr(state, 'show_global_angular_reference', False)}, "
            f"show_local_ref={getattr(state, 'show_local_angular_reference', False)}"
        )

        active_aoa = self._active_aoa_geometries
        active_aod = self._active_aod_geometries
        active_references = self._active_reference_geometries
        self._active_aoa_geometries = {}
        self._active_aod_geometries = {}
        self._active_reference_geometries = {}
        self._collecting_desired_geometry = True
        try:
            if not viz.vis_initialized:
                logger.debug("ApertureService: Visualizer not initialized, clearing previews")
            elif not self._renderer_supports_aperture_preview():
                logger.info(
                    "ApertureService: renderer does not support aperture previews; clearing"
                )
            else:
                if state.show_aoa_aperture:
                    logger.info("ApertureService: Updating AOA aperture")
                    self._update_aoa_aperture()

                if state.show_aod_aperture:
                    logger.info("ApertureService: Updating AOD aperture")
                    self._update_aod_aperture()

                if getattr(state, "show_global_angular_reference", False):
                    logger.info("ApertureService: Updating global angular references")
                    self._update_angular_references(local=False)

                if getattr(state, "show_local_angular_reference", False):
                    logger.info("ApertureService: Updating local angular references")
                    self._update_angular_references(local=True)
        finally:
            desired_aoa = self._active_aoa_geometries
            desired_aod = self._active_aod_geometries
            desired_references = self._active_reference_geometries
            self._active_aoa_geometries = active_aoa
            self._active_aod_geometries = active_aod
            self._active_reference_geometries = active_references
            self._collecting_desired_geometry = False

        renderer = getattr(viz, "renderer", None)
        batch_updates = getattr(renderer, "batch_updates", None)
        batch = batch_updates() if callable(batch_updates) else nullcontext()
        with batch:
            # One aperture change may replace a patch, boundary lines, and
            # several reference labels. Keep the semantic overlay update in
            # one renderer presentation instead of exposing partial guides.
            aoa_ok, aoa_changed = self._sync_geometry_collection(active_aoa, desired_aoa)
            aod_ok, aod_changed = self._sync_geometry_collection(active_aod, desired_aod)
            reference_ok, reference_changed = self._sync_geometry_collection(
                active_references,
                desired_references,
            )
            if aoa_changed or aod_changed or reference_changed:
                request_redraw = getattr(renderer, "request_redraw", None)
                if callable(request_redraw):
                    request_redraw()
        return aoa_ok and aod_ok and reference_ok

    def _get_node_orientation(self, kind: str, index: int) -> tuple[float, float, float] | None:
        """Return cached TX/RX orientation for ``kind`` and ``index`` when available."""
        viz = self.visualizer
        state = viz.app_state
        orientation_key = f"{kind}_orientations"
        current_key = f"current_{kind}_orientations"

        orientations = []
        frame_data = viz.cache_service.get_frame(state.step)
        if frame_data is not None:
            orientations = frame_data.get(orientation_key, [])
            if hasattr(orientations, "tolist"):
                orientations = orientations.tolist()

        if not orientations:
            orientations = getattr(viz, current_key, [])

        if index >= len(orientations):
            return None
        ori = orientations[index]
        if not (hasattr(ori, "__len__") and len(ori) == 3):
            return None
        return (float(ori[0]), float(ori[1]), float(ori[2]))

    def _renderer_supports_aperture_preview(self) -> bool:
        """Return True when the renderer supports filled aperture patches."""
        renderer = getattr(self.visualizer, "renderer", None)
        return renderer is not None and renderer_capabilities(renderer).aperture_preview

    def _iter_selected_node_contexts(
        self,
    ) -> list[tuple[str, int, np.ndarray, tuple[float, float, float] | None]]:
        """Return selected TX/RX node contexts that can host angular overlays."""
        viz = self.visualizer
        state = viz.app_state
        contexts: list[tuple[str, int, np.ndarray, tuple[float, float, float] | None]] = []

        for kind, selection, positions_attr in (
            ("tx", getattr(state, "selected_tx", "all"), "current_tx_positions"),
            ("rx", getattr(state, "selected_rx", "all"), "current_rx_positions"),
        ):
            if selection == "all" or not isinstance(selection, int):
                continue
            positions = getattr(viz, positions_attr, [])
            if selection >= len(positions):
                continue
            position = np.asarray(positions[selection], dtype=np.float64)
            orientation = self._get_node_orientation(kind, selection)
            contexts.append((kind, selection, position, orientation))

        return contexts

    def _update_angular_references(self, *, local: bool) -> None:
        """Create global or local angular reference overlays at selected nodes."""
        state = self.visualizer.app_state
        mode = "local" if local else "global"
        for kind, index, position, orientation in self._iter_selected_node_contexts():
            geom_name = f"angular_reference_{mode}_{kind}_{index}"
            line_payload = create_angular_reference_line_payload(
                center=position,
                radius=state.aperture_radius_m,
                orientation=orientation,
                local=local,
            )
            if line_payload is not None and len(line_payload.points) > 0:
                self._add_reference_geometry(line_payload, geom_name)

            self._add_angular_reference_labels(
                kind=kind,
                index=index,
                position=position,
                orientation=orientation,
                local=local,
            )

    def _add_angular_reference_labels(
        self,
        *,
        kind: str,
        index: int,
        position: np.ndarray,
        orientation: tuple[float, float, float] | None,
        local: bool,
    ) -> None:
        """Add optional text labels for angular reference axes."""
        renderer = getattr(self.visualizer, "renderer", None)
        if renderer is None:
            return

        mode = "local" if local else "global"
        labels = angular_reference_label_positions(
            center=position,
            radius=self.visualizer.app_state.aperture_radius_m,
            orientation=orientation,
            local=local,
        )
        for label_index, (text, label_position, color) in enumerate(labels):
            label_name = f"angular_reference_{mode}_{kind}_{index}_label_{label_index}"
            label = create_text_label(
                label_name,
                text,
                color,
                font_size=0.16,
                position=label_position,
            )
            self._add_reference_geometry(label, label_name)

    def _update_aoa_aperture(self) -> None:
        """Create/update AOA aperture visualization at selected RX position."""
        viz = self.visualizer
        state = viz.app_state

        # Don't show if "all" is selected (too cluttered)
        if state.selected_rx == "all":
            logger.info("AOA aperture: skipping, selected_rx='all' - select a specific RX")
            return

        rx_idx = state.selected_rx
        if not isinstance(rx_idx, int):
            logger.info(f"AOA aperture: selected_rx={rx_idx} is not an int")
            return

        rx_positions = getattr(viz, "current_rx_positions", [])
        logger.info(f"AOA aperture: rx_idx={rx_idx}, num_rx_positions={len(rx_positions)}")
        if rx_idx >= len(rx_positions):
            logger.info(
                f"AOA aperture: RX index {rx_idx} out of range (only {len(rx_positions)} positions)"
            )
            return

        rx_pos = np.asarray(rx_positions[rx_idx])

        # Try multiple sources: frame cache (normal loading) or current_rx_orientations (override)
        rx_orientation = None
        rx_orientations = []

        # First try cached frame data (used during normal frame loading)
        current_step = state.step
        frame_data = viz.cache_service.get_frame(current_step)
        if frame_data is not None:
            rx_orientations = frame_data.get("rx_orientations", [])
            if hasattr(rx_orientations, "tolist"):
                rx_orientations = rx_orientations.tolist()

        # Fallback to current_rx_orientations (used by override service)
        if not rx_orientations:
            rx_orientations = getattr(viz, "current_rx_orientations", [])

        if rx_idx < len(rx_orientations):
            ori = rx_orientations[rx_idx]
            # Ensure it's a tuple of (yaw, pitch, roll) in radians
            if hasattr(ori, "__len__") and len(ori) == 3:
                rx_orientation = (float(ori[0]), float(ori[1]), float(ori[2]))
                logger.info(
                    f"AOA aperture: using RX orientation (yaw={np.degrees(ori[0]):.1f}°, "
                    f"pitch={np.degrees(ori[1]):.1f}°, roll={np.degrees(ori[2]):.1f}°)"
                )
        else:
            logger.info(
                f"AOA aperture: no orientation data for RX{rx_idx} "
                f"(rx_orientations has {len(rx_orientations)} entries)"
            )

        # Get drawable angle bounds. Natural filter limits are stored as None,
        # but the preview should still draw the complete aperture.
        az_min, az_max, el_min, el_max = self._resolve_preview_bounds(state, "aoa")

        logger.info(
            f"AOA aperture: az=[{az_min}, {az_max}], el=[{el_min}, {el_max}], "
            f"radius={state.aperture_radius_m}, pos={rx_pos}"
        )

        self._add_aperture_patch(
            geom_name=f"aperture_aoa_{rx_idx}_patch",
            center=rx_pos,
            az_min=az_min,
            az_max=az_max,
            el_min=el_min,
            el_max=el_max,
            radius=state.aperture_radius_m,
            color=AOA_APERTURE_COLOR,
            orientation=rx_orientation,
        )

        line_payload = create_aperture_line_payload(
            center=rx_pos,
            az_min_deg=az_min,
            az_max_deg=az_max,
            el_min_deg=el_min,
            el_max_deg=el_max,
            radius=state.aperture_radius_m,
            color=AOA_APERTURE_COLOR,
            orientation=rx_orientation,
        )

        if line_payload is None:
            logger.info("AOA aperture: line payload creation returned None")
            return

        if len(line_payload.points) == 0:
            logger.info("AOA aperture: line payload has no points")
            return

        geom_name = f"aperture_aoa_{rx_idx}"
        self._add_line_payload_to_renderer(line_payload, geom_name, is_aoa=True)
        logger.info(f"AOA aperture created at RX {rx_idx} with {len(line_payload.points)} points")

    def _update_aod_aperture(self) -> None:
        """Create/update AOD aperture visualization at selected TX position."""
        viz = self.visualizer
        state = viz.app_state

        # Don't show if "all" is selected (too cluttered)
        if state.selected_tx == "all":
            logger.info("AOD aperture: skipping, selected_tx='all' - select a specific TX")
            return

        tx_idx = state.selected_tx
        if not isinstance(tx_idx, int):
            logger.info(f"AOD aperture: selected_tx={tx_idx} is not an int")
            return

        tx_positions = getattr(viz, "current_tx_positions", [])
        logger.info(f"AOD aperture: tx_idx={tx_idx}, num_tx_positions={len(tx_positions)}")
        if tx_idx >= len(tx_positions):
            logger.info(
                f"AOD aperture: TX index {tx_idx} out of range (only {len(tx_positions)} positions)"
            )
            return

        tx_pos = np.asarray(tx_positions[tx_idx])

        # Try multiple sources: frame cache (normal loading) or current_tx_orientations (override)
        tx_orientation = None
        tx_orientations = []

        # First try cached frame data (used during normal frame loading)
        current_step = state.step
        frame_data = viz.cache_service.get_frame(current_step)
        if frame_data is not None:
            tx_orientations = frame_data.get("tx_orientations", [])
            if hasattr(tx_orientations, "tolist"):
                tx_orientations = tx_orientations.tolist()

        # Fallback to current_tx_orientations (used by override service)
        if not tx_orientations:
            tx_orientations = getattr(viz, "current_tx_orientations", [])

        if tx_idx < len(tx_orientations):
            ori = tx_orientations[tx_idx]
            # Ensure it's a tuple of (yaw, pitch, roll) in radians
            if hasattr(ori, "__len__") and len(ori) == 3:
                tx_orientation = (float(ori[0]), float(ori[1]), float(ori[2]))
                logger.info(
                    f"AOD aperture: using TX orientation (yaw={np.degrees(ori[0]):.1f}°, "
                    f"pitch={np.degrees(ori[1]):.1f}°, roll={np.degrees(ori[2]):.1f}°)"
                )
        else:
            logger.info(
                f"AOD aperture: no orientation data for TX{tx_idx} "
                f"(tx_orientations has {len(tx_orientations)} entries)"
            )

        # Get drawable angle bounds. Natural filter limits are stored as None,
        # but the preview should still draw the complete aperture.
        az_min, az_max, el_min, el_max = self._resolve_preview_bounds(state, "aod")

        logger.info(
            f"AOD aperture: az=[{az_min}, {az_max}], el=[{el_min}, {el_max}], "
            f"radius={state.aperture_radius_m}, pos={tx_pos}"
        )

        self._add_aperture_patch(
            geom_name=f"aperture_aod_{tx_idx}_patch",
            center=tx_pos,
            az_min=az_min,
            az_max=az_max,
            el_min=el_min,
            el_max=el_max,
            radius=state.aperture_radius_m,
            color=AOD_APERTURE_COLOR,
            orientation=tx_orientation,
        )

        line_payload = create_aperture_line_payload(
            center=tx_pos,
            az_min_deg=az_min,
            az_max_deg=az_max,
            el_min_deg=el_min,
            el_max_deg=el_max,
            radius=state.aperture_radius_m,
            color=AOD_APERTURE_COLOR,
            orientation=tx_orientation,
        )

        if line_payload is None:
            logger.info("AOD aperture: line payload creation returned None")
            return

        if len(line_payload.points) == 0:
            logger.info("AOD aperture: line payload has no points")
            return

        geom_name = f"aperture_aod_{tx_idx}"
        self._add_line_payload_to_renderer(line_payload, geom_name, is_aoa=False)
        logger.info(f"AOD aperture created at TX {tx_idx} with {len(line_payload.points)} points")

    def _add_aperture_patch(
        self,
        *,
        geom_name: str,
        center: np.ndarray,
        az_min: float,
        az_max: float,
        el_min: float,
        el_max: float,
        radius: float,
        color: list[float],
        orientation: tuple[float, float, float] | None,
    ) -> None:
        """Add the pygfx filled accepted-zone angular sector."""
        patch = create_aperture_mesh_payload(
            center=center,
            az_min_deg=az_min,
            az_max_deg=az_max,
            el_min_deg=el_min,
            el_max_deg=el_max,
            radius=radius,
            orientation=orientation,
        )
        if patch is None or len(patch.triangles) == 0:
            return
        material = MaterialPayload(
            base_color=(float(color[0]), float(color[1]), float(color[2]), 0.42),
            roughness=1.0,
            metallic=0.0,
            shader="transparent",
        )
        self._add_reference_geometry(patch, geom_name, material=material)

    def _add_line_payload_to_renderer(
        self,
        line_payload,
        name: str,
        is_aoa: bool = True,
    ) -> None:
        """Add an aperture line payload to the renderer."""
        active = self._active_aoa_geometries if is_aoa else self._active_aod_geometries
        self._add_geometry_to_renderer(line_payload, name, active)

    def _add_reference_geometry(
        self, geometry, name: str, material: MaterialPayload | None = None
    ) -> None:
        """Add a global/local angular reference geometry to the renderer."""
        self._add_geometry_to_renderer(
            geometry, name, self._active_reference_geometries, material=material
        )

    def _add_geometry_to_renderer(
        self,
        geometry,
        name: str,
        active_geometries: dict[str, Any],
        material: MaterialPayload | None = None,
    ) -> bool:
        """Stage one complete preview snapshot, or synchronize it immediately."""
        if isinstance(geometry, RenderObjectState):
            if geometry.id != name:
                return False
            state = geometry
            if material is not None:
                state.material = material
        else:
            state = RenderObjectState(
                id=name,
                payload=geometry,
                material=material or MaterialPayload(),
            )
        state.visible = True

        if self._collecting_desired_geometry:
            active_geometries[name] = state
            return True

        renderer = getattr(self.visualizer, "renderer", None)
        ensure_object = getattr(renderer, "ensure_object", None)
        try:
            added = bool(callable(ensure_object) and ensure_object(state.to_render_object()))
        except (RuntimeError, TypeError, ValueError):
            logger.warning("Failed to add aperture geometry '%s'", name, exc_info=True)
            return False
        if added:
            active_geometries[name] = state
        return added

    def _sync_geometry_collection(
        self,
        active: dict[str, Any],
        desired: dict[str, Any],
    ) -> tuple[bool, bool]:
        """Converge one aperture category without forgetting failed operations."""
        all_synced = True
        changed = False
        for name, geometry in tuple(active.items()):
            if name in desired:
                continue
            if self._remove_geometry_from_renderer(name, geometry):
                active.pop(name, None)
                changed = True
            else:
                all_synced = False

        renderer = getattr(self.visualizer, "renderer", None)
        ensure_object = getattr(renderer, "ensure_object", None)
        for name, desired_state in desired.items():
            current = active.get(name)
            if (
                isinstance(current, RenderObjectState)
                and isinstance(desired_state, RenderObjectState)
                and _same_render_state(current, desired_state)
            ):
                continue
            try:
                ensured = bool(
                    isinstance(desired_state, RenderObjectState)
                    and callable(ensure_object)
                    and ensure_object(desired_state.to_render_object())
                )
            except (RuntimeError, TypeError, ValueError):
                logger.warning("Failed to synchronize aperture geometry '%s'", name, exc_info=True)
                ensured = False
            if ensured:
                active[name] = desired_state
                changed = True
            else:
                all_synced = False
        return all_synced, changed

    def _remove_geometry_from_renderer(self, name: str, geometry) -> bool:
        """Remove a geometry object from the renderer."""
        viz = self.visualizer
        renderer = getattr(viz, "renderer", None)
        if renderer is None:
            return False

        try:
            remove_object = getattr(renderer, "remove_object", None)
            removed = bool(callable(remove_object) and remove_object(name))
            if removed:
                logger.debug("ApertureService: Removed named geometry '%s'", name)
            return removed
        except (RuntimeError, TypeError, ValueError) as e:
            logger.debug(f"Failed to remove aperture geometry: {e}")
            return False

    def _clear_apertures(self) -> bool:
        """Remove all active aperture geometries from the scene."""
        all_removed = True
        for name, geometry in list(self._active_aoa_geometries.items()):
            if self._remove_geometry_from_renderer(name, geometry):
                self._active_aoa_geometries.pop(name, None)
            else:
                all_removed = False

        for name, geometry in list(self._active_aod_geometries.items()):
            if self._remove_geometry_from_renderer(name, geometry):
                self._active_aod_geometries.pop(name, None)
            else:
                all_removed = False
        return all_removed

    def _clear_references(self) -> bool:
        """Remove all active angular reference geometries from the scene."""
        all_removed = True
        for name, geometry in list(self._active_reference_geometries.items()):
            if self._remove_geometry_from_renderer(name, geometry):
                self._active_reference_geometries.pop(name, None)
            else:
                all_removed = False
        return all_removed

    def clear_all(self) -> bool:
        """Public method to clear all aperture visualizations."""
        renderer = getattr(self.visualizer, "renderer", None)
        batch_updates = getattr(renderer, "batch_updates", None)
        batch = batch_updates() if callable(batch_updates) else nullcontext()
        before = (
            len(self._active_aoa_geometries)
            + len(self._active_aod_geometries)
            + len(self._active_reference_geometries)
        )
        with batch:
            apertures_removed = self._clear_apertures()
            references_removed = self._clear_references()
            after = (
                len(self._active_aoa_geometries)
                + len(self._active_aod_geometries)
                + len(self._active_reference_geometries)
            )
            if after < before:
                request_redraw = getattr(renderer, "request_redraw", None)
                if callable(request_redraw):
                    request_redraw()
        return apertures_removed and references_removed
