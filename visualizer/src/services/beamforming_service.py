"""Build renderer-neutral beamforming surfaces and frame annotations.

The service accepts beamforming metadata from the frame pipeline and can also
derive standalone antenna-array patterns. It owns mesh generation, display
units, cache keys, and renderer reconciliation for both paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from shared.logging import get_logger

from ..beamforming.extensions import get_beamforming_mode
from ..beamforming.visualization import (
    bounded_pattern_sample_counts,
    compute_pattern_metrics,
    generate_beamforming_mesh,
)
from ..scene.surface_payloads import BeamformingSurface, compute_vertex_normals
from ..types.render_payloads import MeshPayload, SurfaceColorSource
from ..utils.antenna_utils import normalize_visual_element_pattern
from .base import BaseService
from .object_key_service import make_beamforming_key

if TYPE_CHECKING:
    from ..metrics.mpc_canon import CanonicalStepData

logger = get_logger(__name__)

_C = 299_792_458.0  # speed of light (m/s)


def _plan_beamforming_surface_ids(
    pair_entries: list[dict[str, Any]],
    *,
    resolved_tx: str | None,
    resolved_rx: str | None,
) -> dict[tuple[int, str], str]:
    """Plan order-independent surface IDs from stable pair metadata.

    Common, already-unique IDs retain their compact form. When normalized IDs
    collide, every member receives a suffix derived from TX/RX indices and the
    source pair index. Exact duplicate metadata is rejected instead of falling
    back to encounter-order suffixes that can swap object identity by frame.
    """
    groups: dict[
        str,
        list[tuple[tuple[int, str], str, object, object, object]],
    ] = {}
    for pair_data in pair_entries:
        if resolved_tx and pair_data["tx_name"] != resolved_tx:
            continue
        if resolved_rx and pair_data["rx_name"] != resolved_rx:
            continue

        pair = pair_data["pair"]
        pair_tx = pair_data["tx_index"]
        pair_rx = pair_data["rx_index"]
        unique_pair_id = pair.get("unique_pair_id") or f"{pair_tx}_{pair_rx}"
        for role in ("tx", "rx"):
            entry = pair.get(role)
            if not entry:
                continue
            device_index = entry.get("device_index")
            if device_index is None:
                continue
            mesh_id = f"{device_index}_{role}_{unique_pair_id}"
            base_surface_id = make_beamforming_key(mesh_id)
            groups.setdefault(base_surface_id, []).append(
                (
                    (id(pair), role),
                    mesh_id,
                    pair_tx,
                    pair_rx,
                    pair.get("pair_index"),
                )
            )

    proposals: list[tuple[tuple[int, str], str]] = []
    for base_surface_id, candidates in groups.items():
        if len(candidates) == 1:
            proposals.append((candidates[0][0], base_surface_id))
            continue
        for candidate_key, mesh_id, pair_tx, pair_rx, pair_index in candidates:
            stable_suffix = f"pair_{pair_tx}_{pair_rx}_{pair_index}"
            proposals.append(
                (
                    candidate_key,
                    make_beamforming_key(f"{mesh_id}_{stable_suffix}"),
                )
            )

    planned_ids = [surface_id for _, surface_id in proposals]
    candidate_keys = [candidate_key for candidate_key, _ in proposals]
    if len(set(planned_ids)) != len(planned_ids) or len(set(candidate_keys)) != len(candidate_keys):
        raise ValueError(
            "Beamforming pairs must have unique stable metadata "
            "(unique_pair_id, TX/RX indices, and pair_index)"
        )
    return dict(proposals)


class BeamformingService(BaseService):
    """Manages beamforming pattern mesh generation for the visualizer."""

    def __init__(self, visualizer: Any) -> None:
        """Bind beamforming mesh generation to the visualizer and optional loaders."""
        super().__init__()
        self._visualizer = visualizer
        self._external_beamforming_loader: Optional[Any] = None

    # Public entry point

    def build_meshes(
        self,
        raw_frame: dict[str, Any],
        tx_positions_all: np.ndarray,
        rx_positions_all: np.ndarray,
        tx_orientations_all: np.ndarray,
        rx_orientations_all: np.ndarray,
        *,
        canonical_data: CanonicalStepData | None = None,
        beamforming_tx_node: str,
        beamforming_rx_node: str,
        selected_tx: Union[int, str],
        selected_rx: Union[int, str],
        step: int,
        beamforming_azimuth_samples: int,
        beamforming_elevation_samples: int,
        beamforming_tx_scale: float,
        beamforming_rx_scale: float,
        beamforming_db_scale: bool = False,
        beamforming_dynamic_range_db: float = 40.0,
        beamforming_colormap: str = "jet",
        beamforming_element_pattern: str = "isotropic",
        beamforming_tx_element_pattern: Optional[str] = None,
        beamforming_rx_element_pattern: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Build beamforming meshes for all modes.

        Returns ``{"meshes": tuple[BeamformingSurface, ...], "info": {...}}``
        or *None*.
        """
        standalone_mode = raw_frame.get("standalone_beamforming_mode", "frame")

        if standalone_mode != "frame":
            standalone_params = dict(raw_frame.get("standalone_beamforming_params", {}))
            standalone_params["mode"] = standalone_mode
            frame_step = raw_frame.get("_source", {}).get("step", step)
            frame_duration = raw_frame.get("_source", {}).get("duration", None)
            num_steps = raw_frame.get("_source", {}).get("num_steps", None)
            if self._visualizer:
                if frame_duration is None:
                    frame_duration = getattr(self._visualizer, "_frame_duration", None)
                if num_steps is None:
                    num_steps = getattr(self._visualizer, "total_animation_steps", None)
            return self._build_standalone_meshes(
                standalone_params,
                tx_positions_all,
                rx_positions_all,
                tx_orientations_all,
                rx_orientations_all,
                beamforming_tx_node,
                beamforming_rx_node,
                canonical_data=canonical_data,
                selected_tx=selected_tx,
                selected_rx=selected_rx,
                step=frame_step,
                frame_duration=frame_duration,
                num_steps=num_steps,
                beamforming_azimuth_samples=beamforming_azimuth_samples,
                beamforming_elevation_samples=beamforming_elevation_samples,
                beamforming_tx_scale=beamforming_tx_scale,
                beamforming_rx_scale=beamforming_rx_scale,
                beamforming_db_scale=beamforming_db_scale,
                beamforming_dynamic_range_db=beamforming_dynamic_range_db,
                beamforming_colormap=beamforming_colormap,
                beamforming_element_pattern=beamforming_element_pattern,
                beamforming_tx_element_pattern=beamforming_tx_element_pattern,
                beamforming_rx_element_pattern=beamforming_rx_element_pattern,
            )

        # Frame-based beamforming
        return self._build_frame_meshes(
            raw_frame.get("beamforming"),
            tx_positions_all,
            rx_positions_all,
            tx_orientations_all,
            rx_orientations_all,
            beamforming_tx_node,
            beamforming_rx_node,
            selected_tx=selected_tx,
            selected_rx=selected_rx,
            beamforming_azimuth_samples=beamforming_azimuth_samples,
            beamforming_elevation_samples=beamforming_elevation_samples,
            beamforming_tx_scale=beamforming_tx_scale,
            beamforming_rx_scale=beamforming_rx_scale,
            beamforming_db_scale=beamforming_db_scale,
            beamforming_dynamic_range_db=beamforming_dynamic_range_db,
            beamforming_colormap=beamforming_colormap,
            beamforming_element_pattern=beamforming_element_pattern,
            beamforming_tx_element_pattern=beamforming_tx_element_pattern,
            beamforming_rx_element_pattern=beamforming_rx_element_pattern,
        )

    # Frame-based mesh generation (shared by all modes)

    def _build_frame_meshes(
        self,
        beamforming_data: Optional[dict[str, Any]],
        tx_positions_all: np.ndarray,
        rx_positions_all: np.ndarray,
        tx_orientations_all: np.ndarray,
        rx_orientations_all: np.ndarray,
        beamforming_tx_node: str,
        beamforming_rx_node: str,
        *,
        selected_tx: Union[int, str, None] = None,
        selected_rx: Union[int, str, None] = None,
        beamforming_azimuth_samples: int,
        beamforming_elevation_samples: int,
        beamforming_tx_scale: float,
        beamforming_rx_scale: float,
        beamforming_db_scale: bool = False,
        beamforming_dynamic_range_db: float = 40.0,
        beamforming_colormap: str = "jet",
        beamforming_element_pattern: str = "isotropic",
        beamforming_tx_element_pattern: Optional[str] = None,
        beamforming_rx_element_pattern: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Convert beamforming pair metadata into renderable meshes."""
        if not beamforming_data or "pairs" not in beamforming_data:
            return {
                "meshes": (),
                "info": {
                    "available_tx_nodes": [],
                    "available_rx_nodes": [],
                    "resolved_tx_node": None,
                    "resolved_rx_node": None,
                    "pairs": [],
                    "status": "Frame Data unavailable: loaded frame has no beamforming metadata",
                },
            }

        pairs = beamforming_data.get("pairs") or []
        if not pairs:
            return {
                "meshes": (),
                "info": {
                    "available_tx_nodes": [],
                    "available_rx_nodes": [],
                    "resolved_tx_node": None,
                    "resolved_rx_node": None,
                    "pairs": [],
                    "status": "Frame Data unavailable: no beamforming pairs in this frame",
                },
            }

        tx_nodes: dict[str, dict[str, Any]] = {}
        rx_nodes: dict[str, dict[str, Any]] = {}
        pair_entries: list[dict[str, Any]] = []

        for pair in pairs:
            pair_tx_idx = pair.get("tx_index")
            pair_rx_idx = pair.get("rx_index")
            tx_entry = pair.get("tx")
            rx_entry = pair.get("rx")

            if pair_tx_idx is None or pair_rx_idx is None or tx_entry is None or rx_entry is None:
                continue

            tx_name = tx_entry.get("device_name") or f"tx_{pair_tx_idx + 1}"
            rx_name = rx_entry.get("device_name") or f"rx_{pair_rx_idx + 1}"

            tx_nodes.setdefault(tx_name, {"index": pair_tx_idx})
            rx_nodes.setdefault(rx_name, {"index": pair_rx_idx})

            pair_entries.append(
                {
                    "pair": pair,
                    "pair_index": pair.get("pair_index"),
                    "tx_index": pair_tx_idx,
                    "rx_index": pair_rx_idx,
                    "tx_name": tx_name,
                    "rx_name": rx_name,
                    "tx_entry": tx_entry,
                    "rx_entry": rx_entry,
                }
            )

        available_tx_nodes = sorted(tx_nodes.keys())
        available_rx_nodes = sorted(rx_nodes.keys())

        selection_supplied = selected_tx is not None or selected_rx is not None
        render_pair_entries = pair_entries
        selection_status: Optional[str] = None
        requested_tx_index: Optional[int] = None
        requested_rx_index: Optional[int] = None

        def _selection_index(value: Union[int, str, None]) -> Optional[int]:
            """Return a concrete global selection index, or ``None`` for All."""
            if value in {None, "all"}:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        if selection_supplied:
            requested_tx_index = _selection_index(selected_tx)
            requested_rx_index = _selection_index(selected_rx)
            if requested_tx_index is None or requested_rx_index is None:
                resolved_tx = None
                resolved_rx = None
                render_pair_entries = []
                selection_status = "Select one TX and one RX to render beam patterns"
            else:
                matching_pair = next(
                    (
                        entry
                        for entry in pair_entries
                        if int(entry["tx_index"]) == requested_tx_index
                        and int(entry["rx_index"]) == requested_rx_index
                    ),
                    None,
                )
                if matching_pair is None:
                    resolved_tx = f"tx_{requested_tx_index + 1}"
                    resolved_rx = f"rx_{requested_rx_index + 1}"
                    render_pair_entries = []
                    selection_status = (
                        "No beamforming data for "
                        f"TX{requested_tx_index + 1} -> RX{requested_rx_index + 1}"
                    )
                else:
                    resolved_tx = matching_pair["tx_name"]
                    resolved_rx = matching_pair["rx_name"]
                    render_pair_entries = [matching_pair]
        else:
            # Compatibility path for direct service callers that predate the
            # global TX/RX scope. Production calls always provide selection.
            resolved_tx = beamforming_tx_node or "auto"
            if resolved_tx in {"auto", "all"} or (resolved_tx and resolved_tx not in tx_nodes):
                resolved_tx = available_tx_nodes[0] if available_tx_nodes else None

            resolved_rx = beamforming_rx_node or "auto"
            if resolved_rx in {"auto", "all"} or (resolved_rx and resolved_rx not in rx_nodes):
                resolved_rx = available_rx_nodes[0] if available_rx_nodes else None

            pair_found = any(
                entry["tx_name"] == resolved_tx and entry["rx_name"] == resolved_rx
                for entry in pair_entries
            )
            if not pair_found and pair_entries:
                matching_tx = next(
                    (entry for entry in pair_entries if entry["tx_name"] == resolved_tx),
                    None,
                )
                matching_rx = next(
                    (entry for entry in pair_entries if entry["rx_name"] == resolved_rx),
                    None,
                )
                fallback_pair = matching_tx or matching_rx or pair_entries[0]
                resolved_tx = fallback_pair["tx_name"]
                resolved_rx = fallback_pair["rx_name"]

        planned_surface_ids = _plan_beamforming_surface_ids(
            render_pair_entries,
            resolved_tx=resolved_tx,
            resolved_rx=resolved_rx,
        )
        surfaces: list[BeamformingSurface] = []
        pattern_notes: list[str] = []
        pattern_by_role: dict[str, str] = {}
        sampling_by_role: dict[str, dict[str, int]] = {}
        gain_by_role: dict[str, float] = {}
        metrics_by_role: dict[str, dict[str, float]] = {}
        rendered_roles: set[str] = set()

        for pair_data in render_pair_entries:
            if resolved_tx and pair_data["tx_name"] != resolved_tx:
                continue
            if resolved_rx and pair_data["rx_name"] != resolved_rx:
                continue

            pair = pair_data["pair"]

            for role, positions_all, orientations_all, scale in (
                (
                    "tx",
                    tx_positions_all,
                    tx_orientations_all,
                    float(beamforming_tx_scale),
                ),
                (
                    "rx",
                    rx_positions_all,
                    rx_orientations_all,
                    float(beamforming_rx_scale),
                ),
            ):
                entry = pair.get(role)
                if not entry:
                    continue

                weights = entry.get("weights")
                elem_positions = entry.get("element_positions")
                freq = entry.get("carrier_frequency_hz", 0.0)
                device_index = entry.get("device_index")

                if weights is None or elem_positions is None or freq <= 0.0 or device_index is None:
                    continue

                if device_index < 0 or device_index >= len(positions_all):
                    continue

                origin = positions_all[device_index]
                if len(orientations_all) > device_index:
                    orientation = orientations_all[device_index]
                else:
                    orientation = (0.0, 0.0, 0.0)

                try:
                    orient_array = np.asarray(orientation, dtype=np.float64).flatten()
                    if orient_array.size < 3:
                        orient_array = np.pad(orient_array, (0, 3 - orient_array.size))
                    orientation_tuple = tuple(orient_array[:3])
                except (ValueError, TypeError):
                    orientation_tuple = (0.0, 0.0, 0.0)

                try:
                    requested_pattern = (
                        beamforming_tx_element_pattern
                        if role == "tx"
                        else beamforming_rx_element_pattern
                    )
                    requested_pattern = requested_pattern or beamforming_element_pattern
                    visual_pattern, pattern_note = normalize_visual_element_pattern(
                        requested_pattern
                    )
                    pattern_by_role[role] = visual_pattern
                    if pattern_note and pattern_note not in pattern_notes:
                        pattern_notes.append(pattern_note)
                    requested_azimuth = max(12, int(beamforming_azimuth_samples))
                    requested_elevation = max(9, int(beamforming_elevation_samples))
                    mesh_azimuth, mesh_elevation = bounded_pattern_sample_counts(
                        weights,
                        requested_azimuth,
                        requested_elevation,
                    )
                    metric_azimuth, metric_elevation = bounded_pattern_sample_counts(
                        weights,
                        max(72, requested_azimuth),
                        max(37, requested_elevation),
                    )
                    sampling_by_role[role] = {
                        "azimuth": mesh_azimuth,
                        "elevation": mesh_elevation,
                    }
                    if (mesh_azimuth, mesh_elevation) != (
                        requested_azimuth,
                        requested_elevation,
                    ):
                        pattern_notes.append(
                            f"{role.upper()} sampling limited to "
                            f"{mesh_azimuth}x{mesh_elevation} for memory safety"
                        )
                    vertices, triangles, colors = generate_beamforming_mesh(
                        weights=weights,
                        element_positions=elem_positions,
                        carrier_frequency_hz=float(freq),
                        orientation_radians=orientation_tuple,
                        origin=origin,
                        scale=1.0,
                        azimuth_samples=mesh_azimuth,
                        elevation_samples=mesh_elevation,
                        db_scale=bool(beamforming_db_scale),
                        dynamic_range_db=float(beamforming_dynamic_range_db),
                        colormap=str(beamforming_colormap),
                        element_pattern=visual_pattern,
                    )
                    metrics_by_role[role] = compute_pattern_metrics(
                        weights=weights,
                        element_positions=elem_positions,
                        carrier_frequency_hz=float(freq),
                        element_pattern=visual_pattern,
                        azimuth_samples=metric_azimuth,
                        elevation_samples=metric_elevation,
                    )
                    max_radius = (
                        np.linalg.norm(vertices - origin, axis=1).max() if vertices.size else 0.0
                    )
                    logger.debug(
                        "Beamforming mesh %s base radius %.3f (pending scale %.3f)",
                        role,
                        max_radius,
                        scale,
                    )
                except (ValueError, TypeError, RuntimeError) as exc:
                    logger.debug(
                        "Beamforming mesh failed for %s pair %s: %s",
                        role,
                        pair.get("pair_index"),
                        exc,
                    )
                    continue

                surface_id = planned_surface_ids[(id(pair), role)]
                beam_gain = entry.get("beam_gain")
                if beam_gain is not None:
                    try:
                        gain_by_role[role] = float(beam_gain)
                    except (TypeError, ValueError):
                        pass

                vertices_array = np.asarray(vertices, dtype=np.float32)
                origin_array = np.asarray(origin, dtype=np.float32)
                scaled_vertices = origin_array + (vertices_array - origin_array) * max(
                    0.01,
                    float(scale),
                )
                triangles_array = np.asarray(triangles, dtype=np.int32)
                colors_array = np.asarray(colors, dtype=np.float32)
                vertex_colors = (
                    np.clip(colors_array, 0.0, 1.0)
                    if colors_array.shape == scaled_vertices.shape
                    else None
                )
                surfaces.append(
                    BeamformingSurface(
                        id=surface_id,
                        payload=MeshPayload(
                            vertices=scaled_vertices,
                            triangles=triangles_array,
                            normals=compute_vertex_normals(scaled_vertices, triangles_array),
                            vertex_colors=vertex_colors,
                            color_source=SurfaceColorSource.VERTEX,
                        ),
                    )
                )
                rendered_roles.add(role)

        missing_roles = sorted({"tx", "rx"} - rendered_roles) if surfaces else []
        if selection_status:
            status = selection_status
        elif not surfaces:
            status = (
                "Beam pattern unavailable: array metadata is incomplete or could not be rendered"
                if resolved_tx and resolved_rx
                else "Select one TX and one RX to render beam patterns"
            )
        elif missing_roles:
            missing_text = "/".join(role.upper() for role in missing_roles)
            status = f"Beam pattern partial: missing {missing_text} surface"
        else:
            status = f"Beam patterns: {resolved_tx} -> {resolved_rx}"
            if pattern_notes:
                status = f"{status}. {'; '.join(pattern_notes)}"

        summary = {
            "available_tx_nodes": available_tx_nodes,
            "available_rx_nodes": available_rx_nodes,
            "resolved_tx_node": resolved_tx,
            "resolved_rx_node": resolved_rx,
            "requested_tx_index": requested_tx_index,
            "requested_rx_index": requested_rx_index,
            "status": status,
            "element_patterns": pattern_by_role,
            "sampling_by_role": sampling_by_role,
            "gain_by_role": gain_by_role,
            "metrics_by_role": metrics_by_role,
            "pairs": [
                {
                    "pair_index": e["pair_index"],
                    "tx_index": e["tx_index"],
                    "rx_index": e["rx_index"],
                    "tx_name": e["tx_name"],
                    "rx_name": e["rx_name"],
                }
                for e in pair_entries
            ],
        }

        return {
            "meshes": tuple(surfaces),
            "info": summary,
        }

    # Standalone mesh generation

    def _build_standalone_meshes(
        self,
        standalone_params: dict[str, Any],
        tx_positions_all: np.ndarray,
        rx_positions_all: np.ndarray,
        tx_orientations_all: np.ndarray,
        rx_orientations_all: np.ndarray,
        beamforming_tx_node: str,
        beamforming_rx_node: str,
        selected_tx: Union[int, str],
        selected_rx: Union[int, str],
        step: Optional[int] = None,
        frame_duration: Optional[float] = None,
        num_steps: Optional[int] = None,
        *,
        canonical_data: CanonicalStepData | None = None,
        beamforming_azimuth_samples: int,
        beamforming_elevation_samples: int,
        beamforming_tx_scale: float,
        beamforming_rx_scale: float,
        beamforming_db_scale: bool = False,
        beamforming_dynamic_range_db: float = 40.0,
        beamforming_colormap: str = "jet",
        beamforming_element_pattern: str = "isotropic",
        beamforming_tx_element_pattern: Optional[str] = None,
        beamforming_rx_element_pattern: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Build beamforming meshes from standalone computation."""
        mode = standalone_params.get("mode", "standalone")
        if mode != "standalone":
            extension = get_beamforming_mode(str(mode))
            if step is not None and frame_duration is not None and num_steps is not None:
                if extension is not None:
                    result = extension.builder(
                        service=self,
                        standalone_params=standalone_params,
                        step=step,
                        frame_duration=frame_duration,
                        num_steps=num_steps,
                        tx_positions_all=tx_positions_all,
                        rx_positions_all=rx_positions_all,
                        tx_orientations_all=tx_orientations_all,
                        rx_orientations_all=rx_orientations_all,
                        beamforming_tx_node=beamforming_tx_node,
                        beamforming_rx_node=beamforming_rx_node,
                        selected_tx=selected_tx,
                        selected_rx=selected_rx,
                        beamforming_azimuth_samples=beamforming_azimuth_samples,
                        beamforming_elevation_samples=beamforming_elevation_samples,
                        beamforming_tx_scale=beamforming_tx_scale,
                        beamforming_rx_scale=beamforming_rx_scale,
                        beamforming_db_scale=beamforming_db_scale,
                        beamforming_dynamic_range_db=beamforming_dynamic_range_db,
                        beamforming_colormap=beamforming_colormap,
                        beamforming_element_pattern=beamforming_element_pattern,
                        beamforming_tx_element_pattern=beamforming_tx_element_pattern,
                        beamforming_rx_element_pattern=beamforming_rx_element_pattern,
                    )
                    if result is not None:
                        return result
                logger.warning(
                    "Optional beamforming mode '%s' is not available, falling back to standalone",
                    mode,
                )
            else:
                logger.warning(
                    "Optional beamforming mode '%s' requested without frame metadata, "
                    "falling back to standalone",
                    mode,
                )

        try:
            from ..beamforming.standalone import compute_standalone_beamforming

            num_rows = standalone_params.get("antenna_rows", 8)
            num_cols = standalone_params.get("antenna_cols", 8)
            h_spacing = standalone_params.get("horizontal_spacing_m", 0.00536)
            v_spacing = standalone_params.get("vertical_spacing_m", 0.00536)
            freq_ghz = standalone_params.get("carrier_frequency_ghz", 28.0)
            strategy = standalone_params.get("steering_strategy", "manual")
            azimuth = standalone_params.get("azimuth_deg", 0.0)
            elevation = standalone_params.get("elevation_deg", 0.0)

            if len(tx_positions_all) == 0 or len(rx_positions_all) == 0:
                logger.warning("No TX/RX positions available for standalone beamforming")
                return None

            if selected_tx in {"all", None} or selected_rx in {"all", None}:
                return {
                    "meshes": (),
                    "info": {
                        "available_tx_nodes": [
                            f"tx_{index + 1}" for index in range(len(tx_positions_all))
                        ],
                        "available_rx_nodes": [
                            f"rx_{index + 1}" for index in range(len(rx_positions_all))
                        ],
                        "resolved_tx_node": None,
                        "resolved_rx_node": None,
                        "pairs": [],
                        "status": "Select one TX and one RX to render beam patterns",
                    },
                }

            if len(tx_orientations_all) == 0:
                tx_orientations_all = np.zeros((len(tx_positions_all), 3), dtype=np.float64)
            if len(rx_orientations_all) == 0:
                rx_orientations_all = np.zeros((len(rx_positions_all), 3), dtype=np.float64)

            def _resolve_single_index(
                node_name: str,
                selection: Union[int, str],
                length: int,
                prefix: str,
            ) -> int | None:
                """Resolve UI node labels or app selections to one valid array index."""
                if length <= 0:
                    return None
                if selection not in {"all", None}:
                    try:
                        idx = int(selection)
                    except (ValueError, TypeError):
                        return None
                    return max(0, min(idx, length - 1))
                candidate = node_name
                if candidate not in {"", "auto", "all", None}:
                    try:
                        if isinstance(candidate, str) and candidate.startswith(f"{prefix}_"):
                            idx = int(candidate.rsplit("_", 1)[1]) - 1
                        else:
                            idx = int(candidate)
                    except (ValueError, TypeError, IndexError):
                        idx = None
                    if idx is not None:
                        return max(0, min(idx, length - 1))
                return None

            tx_idx = _resolve_single_index(
                beamforming_tx_node,
                selected_tx,
                len(tx_positions_all),
                "tx",
            )
            rx_idx = _resolve_single_index(
                beamforming_rx_node,
                selected_rx,
                len(rx_positions_all),
                "rx",
            )
            tx_indices = [] if tx_idx is None else [tx_idx]
            rx_indices = [] if rx_idx is None else [rx_idx]
            if not tx_indices or not rx_indices:
                logger.debug(
                    "Standalone beamforming: no TX/RX after filter (tx=%s, rx=%s)",
                    selected_tx,
                    selected_rx,
                )
                return None

            pairs: list[dict[str, Any]] = []
            pair_index = 0

            mpc_paths_by_pair = None
            if canonical_data is not None and strategy == "svd":
                mpc_paths_by_pair = self.extract_mpc_paths(
                    canonical_data,
                    carrier_frequency_hz=freq_ghz * 1e9,
                    selected_pair=(tx_idx, rx_idx),
                )

            for tx_idx in tx_indices:
                tx_pos = tx_positions_all[tx_idx]
                tx_pos_tuple = tuple(tx_pos)
                tx_ori = (
                    tuple(tx_orientations_all[tx_idx])
                    if tx_idx < len(tx_orientations_all)
                    else (0.0, 0.0, 0.0)
                )

                for rx_idx in rx_indices:
                    rx_pos = rx_positions_all[rx_idx]
                    rx_pos_tuple = tuple(rx_pos)
                    rx_ori = (
                        tuple(rx_orientations_all[rx_idx])
                        if rx_idx < len(rx_orientations_all)
                        else (0.0, 0.0, 0.0)
                    )

                    mpc_paths = None
                    if mpc_paths_by_pair is not None:
                        mpc_paths = mpc_paths_by_pair.get((tx_idx, rx_idx))

                    pair_data = compute_standalone_beamforming(
                        num_rows=num_rows,
                        num_cols=num_cols,
                        horizontal_spacing_m=h_spacing,
                        vertical_spacing_m=v_spacing,
                        carrier_frequency_hz=freq_ghz * 1e9,
                        tx_position=tx_pos_tuple,
                        rx_position=rx_pos_tuple,
                        tx_orientation=tx_ori,
                        rx_orientation=rx_ori,
                        steering_strategy=strategy,
                        azimuth_deg=azimuth if strategy == "manual" else None,
                        elevation_deg=elevation if strategy == "manual" else None,
                        mpc_paths=mpc_paths,
                    )

                    if pair_data and "pairs" in pair_data and len(pair_data["pairs"]) > 0:
                        pair_entry = pair_data["pairs"][0].copy()
                        pair_entry["pair_index"] = pair_index
                        pair_entry["tx_index"] = tx_idx
                        pair_entry["rx_index"] = rx_idx
                        tx_name = f"tx_{tx_idx + 1}"
                        rx_name = f"rx_{rx_idx + 1}"
                        pair_entry["tx_name"] = tx_name
                        pair_entry["rx_name"] = rx_name
                        pair_entry["tx"]["device_index"] = tx_idx
                        pair_entry["rx"]["device_index"] = rx_idx
                        pair_entry["tx"]["device_name"] = tx_name
                        pair_entry["rx"]["device_name"] = rx_name
                        pairs.append(pair_entry)
                        pair_index += 1

            if not pairs:
                logger.warning("No beamforming pairs computed")
                return None

            # Delegate filtering and mesh generation to _build_frame_meshes
            beamforming_data = {"pairs": pairs}
            return self._build_frame_meshes(
                beamforming_data,
                tx_positions_all,
                rx_positions_all,
                tx_orientations_all,
                rx_orientations_all,
                f"tx_{tx_indices[0] + 1}",
                f"rx_{rx_indices[0] + 1}",
                selected_tx=tx_indices[0],
                selected_rx=rx_indices[0],
                beamforming_azimuth_samples=beamforming_azimuth_samples,
                beamforming_elevation_samples=beamforming_elevation_samples,
                beamforming_tx_scale=beamforming_tx_scale,
                beamforming_rx_scale=beamforming_rx_scale,
                beamforming_db_scale=beamforming_db_scale,
                beamforming_dynamic_range_db=beamforming_dynamic_range_db,
                beamforming_colormap=beamforming_colormap,
                beamforming_element_pattern=beamforming_element_pattern,
                beamforming_tx_element_pattern=beamforming_tx_element_pattern,
                beamforming_rx_element_pattern=beamforming_rx_element_pattern,
            )
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as exc:
            logger.exception("Standalone beamforming mesh generation failed: %s", exc)
            return None

    # MPC path extraction (for SVD)

    def extract_mpc_paths(
        self,
        canonical_data: CanonicalStepData,
        carrier_frequency_hz: float = 28e9,
        selected_pair: Optional[tuple[int, int]] = None,
    ) -> Optional[dict[tuple[int, int], list[tuple[np.ndarray, float]]]]:
        """Borrow canonical MPC paths for SVD channel-matrix construction.

        Args:
            canonical_data: Full-path visualizer canonical arrays for one frame.
            carrier_frequency_hz: Carrier frequency for free-space path loss fallback.
            selected_pair: Optional ``(tx_idx, rx_idx)`` filter. When omitted,
                paths for every available pair are extracted.

        Returns:
            Mapping ``(tx_idx, rx_idx) -> [(path_vertices, path_loss_db), ...]``
            or *None*.
        """
        from ..metrics.mpc_path_catalog import MpcPathCatalog

        try:
            catalog = MpcPathCatalog(canonical_data)
            if catalog.path_count == 0:
                return None

            requested_pair = (
                None if selected_pair is None else (int(selected_pair[0]), int(selected_pair[1]))
            )

            wavelength = _C / float(carrier_frequency_hz)
            path_tx = catalog.tx_ids
            path_rx = catalog.rx_ids
            path_losses = catalog.path_losses_db
            path_loss_provenance = catalog.path_loss_provenance_codes
            mpc_paths_by_pair: dict[tuple[int, int], list[tuple[np.ndarray, float]]] = {}

            fallback_pairs: set[tuple[int, int]] = set()
            for path_idx in range(catalog.path_count):
                tx_idx = int(path_tx[path_idx])
                rx_idx = int(path_rx[path_idx])
                if tx_idx < 0 or rx_idx < 0:
                    continue
                pair = (tx_idx, rx_idx)
                if requested_pair is not None and pair != requested_pair:
                    continue

                vertices = catalog.path_points(path_idx)
                if (
                    vertices.ndim != 2
                    or vertices.shape[0] < 2
                    or vertices.shape[1] != 3
                    or not np.all(np.isfinite(vertices))
                ):
                    continue

                path_loss_db = float(path_losses[path_idx])
                use_geometric_fallback = (
                    not np.isfinite(path_loss_db) or int(path_loss_provenance[path_idx]) == 1
                )
                if use_geometric_fallback:
                    segments = np.diff(np.asarray(vertices, dtype=np.float64), axis=0)
                    geometric_distance = float(np.sum(np.linalg.norm(segments, axis=1)))
                    if geometric_distance > 0.0:
                        # Positive dB (matching frame convention: higher = more loss).
                        path_loss_db = 20.0 * np.log10(
                            4.0 * np.pi * geometric_distance / wavelength
                        )
                    else:
                        path_loss_db = 100.0
                    fallback_pairs.add(pair)

                mpc_paths_by_pair.setdefault(pair, []).append((vertices, path_loss_db))

            for tx_idx, rx_idx in sorted(fallback_pairs):
                logger.warning(
                    "SVD pair (%d,%d): using geometric free-space path loss "
                    "for paths without an exported finite value",
                    tx_idx,
                    rx_idx,
                )

            return mpc_paths_by_pair if mpc_paths_by_pair else None

        except (KeyError, IndexError, ValueError, TypeError) as exc:
            logger.warning("Failed to extract MPC paths for SVD beamforming: %s", exc)
            return None
