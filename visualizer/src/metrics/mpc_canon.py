"""Visualizer canonical arrays, filtering, and color mapping.

Providers construct ``CanonicalStepData`` from compact frame projections before
the renderer pipeline receives a payload. This module owns the resulting
point-, segment-, and path-aligned filtering and color operations.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from shared.logging import get_logger

from ..services.mpc_interaction_style_service import colorize_mpc_interaction_types
from ..utils.colors import (
    ensure_continuous_lut,
    ensure_viridis_lut,
    map_values_to_lut,
)

logger = get_logger(__name__)

MPC_MATERIAL_FALLBACK_GRAY = 0.7


def _bare_material_type(name: str) -> str:
    """Strip ``mat-itu_`` / ``mat-`` / ``itu_`` prefix to get bare material type.

    Used for flexible matching between scene BSDF IDs (e.g. ``"marble"``) and
    frame material names (e.g. ``"mat-itu_marble"``).
    """
    if name.startswith("mat-itu_"):
        return name[8:]
    if name.startswith("mat-"):
        return name[4:]
    if name.startswith("itu_"):
        return name[4:]
    if name.startswith("ground_") and len(name) > len("ground_"):
        return name[len("ground_") :]
    return name


def _material_color_aliases(name: str) -> tuple[str, ...]:
    """Return compatible material-key aliases without replacing exact keys."""
    aliases: list[str] = []
    bare = _bare_material_type(name)
    if bare and bare != name:
        aliases.extend((bare, f"itu_{bare}"))
        surface = _bare_material_type(bare)
        if surface and surface != bare:
            aliases.extend(
                (
                    surface,
                    f"ground_{surface}",
                    f"mat-ground_{surface}",
                    f"itu_{surface}",
                    f"mat-itu_{surface}",
                )
            )
    elif name:
        aliases.extend(
            (
                f"mat-itu_{name}",
                f"itu_{name}",
                f"ground_{name}",
                f"mat-ground_{name}",
            )
        )
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _expanded_material_colors(
    material_colors: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Expand material color aliases while preserving exact-name precedence."""
    expanded: dict[str, np.ndarray] = {}
    pending_aliases: list[tuple[str, np.ndarray]] = []
    for name, color in material_colors.items():
        color_arr = np.asarray(color, dtype=np.float32)
        expanded[name] = color_arr
        pending_aliases.extend((alias, color_arr) for alias in _material_color_aliases(name))
    for alias, color_arr in pending_aliases:
        expanded.setdefault(alias, color_arr)
    return expanded


def _material_lookup_stem(name: object) -> str:
    """Return a lowercase material stem for family/suffix comparisons."""
    if isinstance(name, bytes):
        try:
            name = name.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    text = str(name or "").strip().strip("\x00").replace(" ", "_").lower()
    prefixes = (
        "mat-itu_",
        "mat_itu_",
        "mat-itu-",
        "mat-ground_",
        "mat_ground_",
        "itu_",
        "itu-",
        "ground_",
        "mat-",
    )
    previous = None
    while text and text != previous:
        previous = text
        for prefix in prefixes:
            if text.startswith(prefix) and len(text) > len(prefix):
                text = text[len(prefix) :]
                break
        else:
            return text
    return text


def _lookup_family_material_color(
    expanded_colors: dict[str, np.ndarray],
    family: str,
) -> Optional[np.ndarray]:
    """Resolve a material-family color from common palette aliases."""
    if not family:
        return None
    candidates = (
        family,
        f"mat-itu_{family}",
        f"itu_{family}",
        f"mat-{family}",
        f"ground_{family}",
        f"mat-ground_{family}",
    )
    for candidate in candidates:
        matched = expanded_colors.get(candidate)
        if matched is not None:
            return matched
    return None


def _lookup_material_color_for_id(
    expanded_colors: dict[str, np.ndarray],
    name: object,
    bare: object,
    itu: object = "",
) -> Optional[np.ndarray]:
    """Resolve MPC material color, preferring ITU family over target suffix colors."""
    family = _material_lookup_stem(itu)
    if family:
        for value in (name, bare):
            stem = _material_lookup_stem(value)
            if stem == family or stem.startswith(f"{family}_"):
                matched = _lookup_family_material_color(expanded_colors, family)
                if matched is not None:
                    return matched

    for value in (name, bare):
        key = str(value or "")
        if not key:
            continue
        matched = expanded_colors.get(key)
        if matched is not None:
            return matched

    if family:
        return _lookup_family_material_color(expanded_colors, family)
    return None


@dataclass
class CanonicalStepData:
    """Aligned canonical arrays for one frame of MPC data.

    Point arrays are renderer-facing and include TX/RX endpoints plus bounce
    points. Segment arrays are line-facing and align with ``lines``. Path arrays
    are filter/statistics-facing and align with canonical path IDs. Material ID
    0 is reserved for TX/RX or no-material points.
    """

    # Geometry (unfiltered)
    points: np.ndarray  # float32 or float64 [N,3] C-contig (renderer-dependent)
    lines: np.ndarray  # int32   [M,2] C-contig

    # Per-point attributes (parallel to points)
    order: np.ndarray  # uint8   [N]
    itype: np.ndarray  # uint8   [N]
    delay: np.ndarray  # float32 [N]
    loss: np.ndarray  # float32 [N]

    # Optional (for selection filters)
    tx_id: Optional[np.ndarray] = None  # int16 [N]
    rx_id: Optional[np.ndarray] = None  # int16 [N]

    # Per-point path identity
    path_id: Optional[np.ndarray] = None  # int32 [N]

    # Per-path aggregates (aligned with path indices)
    path_start_indices: Optional[np.ndarray] = None  # int32 [P]
    path_orders: Optional[np.ndarray] = None  # uint8 [P]
    path_delays: Optional[np.ndarray] = None  # float32 [P]
    path_losses: Optional[np.ndarray] = None  # float32 [P]
    path_tx: Optional[np.ndarray] = None  # int16 [P]
    path_rx: Optional[np.ndarray] = None  # int16 [P]
    # True when a value is a geometric fallback rather than an exported RF
    # metric. ``None`` means the payload does not identify metric provenance.
    path_delay_is_estimated: Optional[np.ndarray] = None  # bool [P]
    path_loss_is_estimated: Optional[np.ndarray] = None  # bool [P]

    # Optional direct per-path angles. The Explorer can otherwise gather them
    # lazily from the point-angle broadcasts.
    # Missing measurements are represented by NaN, never by synthetic 0 degrees.
    path_aoa_az: Optional[np.ndarray] = None  # float32 [P]
    path_aoa_el: Optional[np.ndarray] = None  # float32 [P]
    path_aod_az: Optional[np.ndarray] = None  # float32 [P]
    path_aod_el: Optional[np.ndarray] = None  # float32 [P]

    # Per-segment aggregates (aligned with lines)
    segment_start_indices: Optional[np.ndarray] = None  # int32 [S]
    segment_end_indices: Optional[np.ndarray] = None  # int32 [S]
    segment_order: Optional[np.ndarray] = None  # uint8 [S]
    segment_itype: Optional[np.ndarray] = None  # uint8 [S]
    segment_delay: Optional[np.ndarray] = None  # float32 [S]
    segment_loss: Optional[np.ndarray] = None  # float32 [S]
    segment_tx_id: Optional[np.ndarray] = None  # int16 [S]
    segment_rx_id: Optional[np.ndarray] = None  # int16 [S]
    segment_path_id: Optional[np.ndarray] = None  # int32 [S]
    segment_material_ids: Optional[np.ndarray] = None  # int16 [S]

    # Optional material mapping (per POINT). TX/RX carry empty strings / id=0.
    material_names: Optional[np.ndarray] = None  # object [N] - material names per point
    material_ids: Optional[np.ndarray] = None  # int16 [N] - material IDs per point
    material_itu_types: Optional[np.ndarray] = None  # object [N] - ITU types per point
    # Reverse mapping: material_id -> (name, itu_type, bare_type). ID 0 = no material.
    material_id_to_name: Optional[dict[int, str]] = None
    material_id_to_itu: Optional[dict[int, str]] = None
    material_id_to_bare: Optional[dict[int, str]] = None

    # Ranges (stable color scaling)
    delay_min: float = 0.0
    delay_max: float = 1.0
    loss_min: float = 0.0
    loss_max: float = 1.0

    # Angle data populated from projected path metrics (per point, in degrees).
    aoa_az: Optional[np.ndarray] = None  # float32 [N] - Azimuth angle of arrival
    aoa_el: Optional[np.ndarray] = None  # float32 [N] - Elevation angle of arrival
    aod_az: Optional[np.ndarray] = None  # float32 [N] - Azimuth angle of departure
    aod_el: Optional[np.ndarray] = None  # float32 [N] - Elevation angle of departure

    # Angle ranges for UI bounds
    aoa_az_min: float = 0.0
    aoa_az_max: float = 360.0
    aoa_el_min: float = -90.0
    aoa_el_max: float = 90.0
    aod_az_min: float = 0.0
    aod_az_max: float = 360.0
    aod_el_min: float = -90.0
    aod_el_max: float = 90.0
    profile_ms: Optional[dict[str, float]] = None


def _canon_profile_enabled() -> bool:
    """Return True when per-stage canonicalization timing is requested."""
    for env_name in ("ORCHAV_CANON_PROFILE", "ORCHAV_BENCH_PROFILE_DETAIL"):
        if os.environ.get(env_name, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return True
    return False


def make_lookup_mask(max_value: int, allowed: np.ndarray) -> np.ndarray:
    """Build a boolean LUT covering both present values and allowed indices.

    Prevents out-of-bounds when user-specified allowed set includes values that
    exceed what is present in the current frame (e.g., allow order 6+ while
    the frame only contains orders up to 4).
    """
    allowed = np.asarray(allowed, dtype=np.int32)
    max_idx = int(max_value)
    if allowed.size:
        max_idx = int(max(max_idx, int(allowed.max())))
    if max_idx < 0:
        max_idx = 0
    lut = np.zeros(max_idx + 1, dtype=bool)
    if allowed.size:
        # Only set within bounds
        valid = allowed[(allowed >= 0) & (allowed <= max_idx)]
        if valid.size:
            lut[valid] = True
    return lut


def normalize_angle_deg(angle_deg: np.ndarray) -> np.ndarray:
    """Normalize angle(s) to [-180, 180] degree range.

    Args:
        angle_deg: Angle(s) in degrees (scalar or array)

    Returns:
        Normalized angle(s) in [-180, 180] range
    """
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _yaw_pitch_roll_to_rotation_matrix(
    orientation: tuple[float, float, float],
) -> np.ndarray:
    """Return the ZYX yaw/pitch/roll rotation matrix used by aperture previews."""
    yaw, pitch, roll = orientation
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _world_angles_to_local_angles(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    orientation: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Transform world azimuth/elevation angles into a device-local frame."""
    az_rad = np.radians(np.asarray(azimuth_deg, dtype=np.float64))
    el_rad = np.radians(np.asarray(elevation_deg, dtype=np.float64))
    cos_el = np.cos(el_rad)
    world_dirs = np.column_stack(
        (
            cos_el * np.cos(az_rad),
            cos_el * np.sin(az_rad),
            np.sin(el_rad),
        )
    )

    rotation = _yaw_pitch_roll_to_rotation_matrix(orientation)
    local_dirs = world_dirs @ rotation
    local_az = np.degrees(np.arctan2(local_dirs[:, 1], local_dirs[:, 0]))
    local_norm = np.linalg.norm(local_dirs, axis=1)
    local_el = np.degrees(
        np.arcsin(np.clip(local_dirs[:, 2] / np.maximum(local_norm, 1e-12), -1.0, 1.0))
    )
    return normalize_angle_deg(local_az), local_el


def _has_specific_node_selection(selection: int | str) -> bool:
    """Return True when a TX/RX selector targets one concrete node."""
    if selection is None:
        return False
    if isinstance(selection, str):
        return selection not in {"", "all"}
    return True


def _requires_full_angle_transform(
    orientation: Optional[tuple[float, float, float]],
) -> bool:
    """Return True when pitch/roll require full vector-based angle conversion."""
    if orientation is None:
        return False
    return abs(float(orientation[1])) > 1e-9 or abs(float(orientation[2])) > 1e-9


def _angle_range_mask(
    angle_deg: np.ndarray,
    min_deg: Optional[float],
    max_deg: Optional[float],
) -> np.ndarray:
    """Return an inclusive angular range mask with azimuth wrap handling."""
    angles = normalize_angle_deg(np.asarray(angle_deg, dtype=np.float64))
    mask = np.ones(angles.shape, dtype=bool)
    if min_deg is None and max_deg is None:
        return mask

    # A span covering the full circle does not restrict the selected angles.
    if (
        min_deg is not None
        and max_deg is not None
        and abs(float(max_deg) - float(min_deg)) >= 359.9
    ):
        return mask

    lower = None if min_deg is None else normalize_angle_deg(np.asarray(float(min_deg))).item()
    upper = None if max_deg is None else normalize_angle_deg(np.asarray(float(max_deg))).item()

    # Preserve +180 as an inclusive upper endpoint instead of normalizing it
    # to -180, otherwise ranges such as [-179, 180] drop the 180-degree side.
    if max_deg is not None and float(max_deg) >= 180.0:
        upper = 180.0
        angles = np.where(np.isclose(angles, -180.0), 180.0, angles)
    if min_deg is not None and float(min_deg) <= -180.0:
        lower = -180.0

    if lower is not None and upper is not None and lower > upper:
        return (angles >= lower) | (angles <= upper)
    if lower is not None:
        mask &= angles >= lower
    if upper is not None:
        mask &= angles <= upper
    return mask


def _point_mask_from_segment_mask(
    canon_lines: np.ndarray,
    segment_mask: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Promote a segment mask to the endpoints needed by renderer point arrays."""
    point_mask = np.zeros(n_points, dtype=bool)
    if canon_lines.size and np.any(segment_mask):
        keep_lines = canon_lines[segment_mask]
        point_mask[keep_lines[:, 0]] = True
        point_mask[keep_lines[:, 1]] = True
    return point_mask


def build_filter_mask(
    canon: CanonicalStepData,
    allowed_orders: Sequence[int],
    allowed_types: Sequence[int],
    selected_tx: int | str = "all",
    selected_rx: int | str = "all",
    allowed_materials: Optional[Sequence[str]] = None,
    show_tx_segments: bool = True,
    # Delay/Power range filters
    delay_min_ns: Optional[float] = None,
    delay_max_ns: Optional[float] = None,
    power_min_db: Optional[float] = None,
    power_max_db: Optional[float] = None,
    # Angle filters (AoA/AoD in degrees)
    aoa_az_min_deg: Optional[float] = None,
    aoa_az_max_deg: Optional[float] = None,
    aoa_el_min_deg: Optional[float] = None,
    aoa_el_max_deg: Optional[float] = None,
    aod_az_min_deg: Optional[float] = None,
    aod_az_max_deg: Optional[float] = None,
    aod_el_min_deg: Optional[float] = None,
    aod_el_max_deg: Optional[float] = None,
    material_filter_scope: str = "segment",
    # Device orientations for angle filtering (yaw, pitch, roll) in radians
    aoa_device_orientation: Optional[tuple[float, float, float]] = None,
    aod_device_orientation: Optional[tuple[float, float, float]] = None,
    profile: Optional[dict[str, float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build point and segment masks for the active MPC filter state.

    Order/type, TX/RX, material, delay/power, and angle filters are applied at
    segment level first. The returned point mask includes only endpoints used by
    kept segments, which keeps filtered line geometry compact after remap.
    """
    profile_start = time.perf_counter()
    if profile is not None:
        profile["mpc_filter_fast_path"] = 0.0

    apply_aoa_angle_filters = _has_specific_node_selection(selected_rx)
    apply_aod_angle_filters = _has_specific_node_selection(selected_tx)
    eff_aoa_az_min_deg = aoa_az_min_deg if apply_aoa_angle_filters else None
    eff_aoa_az_max_deg = aoa_az_max_deg if apply_aoa_angle_filters else None
    eff_aoa_el_min_deg = aoa_el_min_deg if apply_aoa_angle_filters else None
    eff_aoa_el_max_deg = aoa_el_max_deg if apply_aoa_angle_filters else None
    eff_aod_az_min_deg = aod_az_min_deg if apply_aod_angle_filters else None
    eff_aod_az_max_deg = aod_az_max_deg if apply_aod_angle_filters else None
    eff_aod_el_min_deg = aod_el_min_deg if apply_aod_angle_filters else None
    eff_aod_el_max_deg = aod_el_max_deg if apply_aod_angle_filters else None

    orders = np.asarray(allowed_orders, dtype=np.int32)
    types = np.asarray(allowed_types, dtype=np.int32)
    raw_orders = canon.order.astype(np.int32, copy=False)
    raw_types = canon.itype.astype(np.int32, copy=False)
    segment_starts = (
        canon.segment_start_indices
        if canon.segment_start_indices is not None
        else (canon.lines[:, 0] if canon.lines.size else np.empty((0,), dtype=np.int32))
    )
    segment_orders = (
        canon.segment_order.astype(np.int32, copy=False)
        if canon.segment_order is not None
        else raw_orders[segment_starts]
    )
    segment_types = (
        canon.segment_itype.astype(np.int32, copy=False)
        if canon.segment_itype is not None
        else raw_types[segment_starts]
    )
    segment_tx_ids = canon.segment_tx_id
    segment_rx_ids = canon.segment_rx_id
    segment_material_ids = canon.segment_material_ids
    _use_id_path = (
        allowed_materials is not None
        and canon.material_ids is not None
        and canon.material_id_to_name is not None
        and canon.material_id_to_bare is not None
        and canon.lines.size
    )
    _use_string_path = (
        not _use_id_path
        and allowed_materials is not None
        and canon.material_names is not None
        and canon.lines.size
    )
    material_filter_mode = "none"
    allowed_materials_set: set[str] | None = None
    canonical_allowed: set[str] | None = None
    has_no_material = False
    if allowed_materials is not None and canon.lines.size:
        allowed_materials_set = set(allowed_materials)
        has_no_material = "no-material" in allowed_materials_set
        canonical_allowed = {_bare_material_type(m) for m in allowed_materials}
        if _use_id_path:
            all_known_ids_allowed = True
            for mid, name in canon.material_id_to_name.items():
                if mid == 0:
                    continue
                if name in allowed_materials_set:
                    continue
                if canon.material_id_to_bare.get(mid, "") in canonical_allowed:
                    continue
                if canon.material_id_to_itu is not None:
                    itu_name = canon.material_id_to_itu.get(mid, "")
                    if itu_name and itu_name.lower() in canonical_allowed:
                        continue
                all_known_ids_allowed = False
                break
            if all_known_ids_allowed:
                material_filter_mode = "none" if has_no_material else "hide_no_material"
            else:
                material_filter_mode = "full_id"
        elif _use_string_path:
            material_filter_mode = "full_string"
    max_o = int(segment_orders.max()) if segment_orders.size else 0
    o_lut = make_lookup_mask(max_o, orders)
    if profile is not None:
        profile["mpc_filter_setup_ms"] = (time.perf_counter() - profile_start) * 1000.0
        profile["mpc_filter_material_mode"] = float(
            {"none": 0, "hide_no_material": 1, "full_id": 2, "full_string": 3}.get(
                material_filter_mode, 0
            )
        )

    # Fast path: when no real filtering is active, return all-true masks directly.
    fast_path_start = time.perf_counter()
    _no_range_filter = (
        delay_min_ns is None
        and delay_max_ns is None
        and power_min_db is None
        and power_max_db is None
        and eff_aoa_az_min_deg is None
        and eff_aoa_az_max_deg is None
        and eff_aoa_el_min_deg is None
        and eff_aoa_el_max_deg is None
        and eff_aod_az_min_deg is None
        and eff_aod_az_max_deg is None
        and eff_aod_el_min_deg is None
        and eff_aod_el_max_deg is None
    )
    _no_selection_filter = (
        selected_tx == "all" and selected_rx == "all" and material_filter_mode == "none"
    )
    if _no_range_filter and _no_selection_filter and canon.lines.size:
        _all_orders = False
        if segment_orders.size:
            valid_orders = orders[(orders >= 0) & (orders <= max_o)]
            expected_order_sum = max_o * (max_o + 1) // 2
            _all_orders = (
                valid_orders.size == max_o + 1 and int(valid_orders.sum()) == expected_order_sum
            )

        # Interaction types include native MPC codes 0/1/2/4/8 and virtual 99.
        # Only do the exact coverage check if the order set already qualifies.
        _all_types = False
        if _all_orders and segment_types.size:
            observed_types = np.unique(segment_types)
            allowed_type_vals = types[types >= 0]
            _all_types = bool(np.isin(observed_types, allowed_type_vals).all())
        if _all_orders and _all_types:
            if profile is not None:
                profile["mpc_filter_fast_path_check_ms"] = (
                    time.perf_counter() - fast_path_start
                ) * 1000.0
                profile["mpc_filter_fast_path"] = 1.0
                point_mask_start = time.perf_counter()
            point_mask = np.ones(canon.points.shape[0], dtype=bool)
            segment_mask = np.ones(canon.lines.shape[0], dtype=bool)
            if profile is not None:
                profile["mpc_filter_point_mask_ms"] = (
                    time.perf_counter() - point_mask_start
                ) * 1000.0
            return point_mask, segment_mask
    if profile is not None:
        profile["mpc_filter_fast_path_check_ms"] = (time.perf_counter() - fast_path_start) * 1000.0

    def _segment_mask_from_path_material_ids(
        allowed_ids_arr: Optional[np.ndarray],
        include_no_material_paths: bool,
        require_material_paths: bool = False,
    ) -> Optional[np.ndarray]:
        """Return a segment mask based on whole-path material membership.

        Material filtering is an MPC/path-level operation: selecting a
        material should keep the complete path if any bounce on that path has
        the material. ``no-material`` means paths with no material-bearing
        bounce, not the TX/RX endpoints of reflected paths.
        """
        if (
            canon.material_ids is None
            or canon.path_id is None
            or canon.segment_path_id is None
            or canon.segment_path_id.size == 0
        ):
            return None

        point_path_ids = canon.path_id.astype(np.int32, copy=False)
        segment_path_ids = canon.segment_path_id.astype(np.int32, copy=False)
        valid_point_paths = point_path_ids >= 0
        valid_segment_paths = segment_path_ids >= 0
        if not np.any(valid_point_paths) or not np.any(valid_segment_paths):
            return None

        n_paths = (
            max(
                int(np.max(point_path_ids[valid_point_paths])),
                int(np.max(segment_path_ids[valid_segment_paths])),
            )
            + 1
        )
        material_ids = canon.material_ids.astype(np.int16, copy=False)
        path_has_material = np.zeros(n_paths, dtype=bool)
        np.logical_or.at(
            path_has_material,
            point_path_ids[valid_point_paths],
            material_ids[valid_point_paths] != 0,
        )

        path_allowed = np.zeros(n_paths, dtype=bool)
        if allowed_ids_arr is not None and allowed_ids_arr.size:
            point_has_allowed_material = np.isin(material_ids, allowed_ids_arr)
            np.logical_or.at(
                path_allowed,
                point_path_ids[valid_point_paths],
                point_has_allowed_material[valid_point_paths],
            )
        if require_material_paths:
            path_allowed |= path_has_material
        if include_no_material_paths:
            path_allowed |= ~path_has_material

        mask = np.zeros(canon.lines.shape[0], dtype=bool)
        in_range = (segment_path_ids >= 0) & (segment_path_ids < n_paths)
        mask[in_range] = path_allowed[segment_path_ids[in_range]]
        return mask

    use_path_material_filter = str(material_filter_scope).lower() in {
        "path",
        "mpc",
        "whole_path",
        "whole-mpc",
    }

    if _no_range_filter and selected_tx == "all" and selected_rx == "all" and canon.lines.size:
        if material_filter_mode == "hide_no_material":
            material_start = time.perf_counter()
            segment_mask = None
            if use_path_material_filter:
                segment_mask = _segment_mask_from_path_material_ids(
                    allowed_ids_arr=None,
                    include_no_material_paths=False,
                    require_material_paths=True,
                )
            if segment_mask is None:
                if segment_material_ids is not None:
                    segment_mask = segment_material_ids != 0
                elif canon.material_ids is not None:
                    segment_mask = canon.material_ids[segment_starts] != 0
                else:
                    segment_mask = segment_types != 0
            if profile is not None:
                profile["mpc_filter_material_ms"] = (time.perf_counter() - material_start) * 1000.0
                point_mask_start = time.perf_counter()
            point_mask = _point_mask_from_segment_mask(
                canon.lines, segment_mask, canon.points.shape[0]
            )
            if profile is not None:
                profile["mpc_filter_point_mask_ms"] = (
                    time.perf_counter() - point_mask_start
                ) * 1000.0
            return point_mask, segment_mask

    # Interaction types are discrete segment labels, not numeric ranges.
    order_type_start = time.perf_counter()
    if canon.lines.size:
        allowed_type_vals = types[types >= 0]
        if allowed_type_vals.size:
            type_allowed = np.isin(segment_types, allowed_type_vals)
        else:
            type_allowed = np.zeros_like(segment_types, dtype=bool)

        segment_allowed = o_lut[segment_orders] & type_allowed
    else:
        segment_allowed = np.empty((0,), dtype=bool)
    if profile is not None:
        profile["mpc_filter_order_type_ms"] = (time.perf_counter() - order_type_start) * 1000.0

    tx_rx_start = time.perf_counter()
    if selected_tx != "all" and segment_tx_ids is not None:
        segment_allowed &= segment_tx_ids == int(selected_tx)

    if selected_rx != "all" and segment_rx_ids is not None:
        segment_allowed &= segment_rx_ids == int(selected_rx)
    if profile is not None:
        profile["mpc_filter_tx_rx_ms"] = (time.perf_counter() - tx_rx_start) * 1000.0

    # Material filter (optional, segment-based). If allowed_materials is None => no filter.
    # If allowed_materials is an empty list => hide all material-based segments (subject to TX setting).
    material_start = time.perf_counter()
    if material_filter_mode == "hide_no_material":
        if segment_material_ids is not None:
            segment_allowed &= segment_material_ids != 0
        elif canon.material_ids is not None:
            segment_allowed &= canon.material_ids[segment_starts] != 0
        else:
            segment_allowed &= segment_types != 0
    elif material_filter_mode == "full_id":
        # Fast integer-based material filtering using precomputed IDs
        allowed_ids = set()
        for mid, name in canon.material_id_to_name.items():
            if mid == 0:
                continue  # ID 0 = no-material, handled separately
            if name in allowed_materials_set:
                allowed_ids.add(mid)
            elif canon.material_id_to_bare[mid] in canonical_allowed:
                allowed_ids.add(mid)
            elif canon.material_id_to_itu is not None:
                itu = canon.material_id_to_itu.get(mid, "")
                if itu and itu.lower() in canonical_allowed:
                    allowed_ids.add(mid)

        if allowed_ids:
            allowed_ids_arr = np.array(sorted(allowed_ids), dtype=np.int16)
        else:
            allowed_ids_arr = np.empty((0,), dtype=np.int16)

        material_segment_allowed = None
        if use_path_material_filter:
            material_segment_allowed = _segment_mask_from_path_material_ids(
                allowed_ids_arr=allowed_ids_arr,
                include_no_material_paths=has_no_material,
            )
        if material_segment_allowed is None:
            segment_mat_ids = (
                segment_material_ids
                if segment_material_ids is not None
                else canon.material_ids[segment_starts]
            )
            if allowed_ids_arr.size:
                material_segment_allowed = np.isin(segment_mat_ids, allowed_ids_arr)
            else:
                material_segment_allowed = np.zeros(canon.lines.shape[0], dtype=bool)
            if has_no_material:
                material_segment_allowed |= segment_mat_ids == 0

    elif material_filter_mode == "full_string":
        # Fallback: string-based filtering (slower, for data without material IDs)
        logger.debug("build_filter_mask: using slow string-based material filtering")
        valid_starts = segment_starts < len(canon.material_names)
        material_segment_allowed = np.zeros(canon.lines.shape[0], dtype=bool)

        if np.any(valid_starts):
            valid_indices = segment_starts[valid_starts]
            segment_materials = canon.material_names[valid_indices]
            segment_itu_types = (
                canon.material_itu_types[valid_indices]
                if canon.material_itu_types is not None and len(canon.material_itu_types) > 0
                else None
            )

            def _normalize_name(name) -> str:
                """Normalize raw material names from string/object arrays."""
                if isinstance(name, bytes):
                    try:
                        name = name.decode("utf-8")
                    except UnicodeDecodeError:
                        return ""
                return str(name).strip() if name else ""

            normalized = np.array([_normalize_name(m) for m in segment_materials], dtype=object)
            normalized_itu = (
                np.array([_normalize_name(m) for m in segment_itu_types], dtype=object)
                if segment_itu_types is not None
                else None
            )

            def _is_material_allowed(idx: int, name: str) -> bool:
                """Return whether a material name or its ITU fallback is selected."""
                if name in allowed_materials_set:
                    return True
                if _bare_material_type(name) in canonical_allowed:
                    return True
                if normalized_itu is not None and idx < len(normalized_itu):
                    itu = str(normalized_itu[idx]).strip().lower()
                    if itu and itu in canonical_allowed:
                        return True
                return False

            is_allowed = np.array(
                [_is_material_allowed(i, n) for i, n in enumerate(normalized)],
                dtype=bool,
            )
            if has_no_material:
                is_empty = normalized == ""
                is_allowed |= is_empty
            material_segment_allowed[valid_starts] = is_allowed

    if material_filter_mode in {"full_id", "full_string"}:
        # Combine with type/order filtering at the SEGMENT level to keep logic consistent
        segment_allowed &= material_segment_allowed
    if profile is not None:
        profile["mpc_filter_material_ms"] = (time.perf_counter() - material_start) * 1000.0

    # Range filters use segment-start values so line and point masks agree.
    range_angle_start = time.perf_counter()
    if canon.lines.size and np.any(segment_allowed):
        seg_starts = segment_starts
        aoa_local_angles = None
        aod_local_angles = None
        if (
            apply_aoa_angle_filters
            and (
                eff_aoa_az_min_deg is not None
                or eff_aoa_az_max_deg is not None
                or eff_aoa_el_min_deg is not None
                or eff_aoa_el_max_deg is not None
            )
            and _requires_full_angle_transform(aoa_device_orientation)
            and canon.aoa_az is not None
            and canon.aoa_az.size
            and canon.aoa_el is not None
            and canon.aoa_el.size
        ):
            aoa_local_angles = _world_angles_to_local_angles(
                canon.aoa_az[seg_starts],
                canon.aoa_el[seg_starts],
                aoa_device_orientation,
            )
        if (
            apply_aod_angle_filters
            and (
                eff_aod_az_min_deg is not None
                or eff_aod_az_max_deg is not None
                or eff_aod_el_min_deg is not None
                or eff_aod_el_max_deg is not None
            )
            and _requires_full_angle_transform(aod_device_orientation)
            and canon.aod_az is not None
            and canon.aod_az.size
            and canon.aod_el is not None
            and canon.aod_el.size
        ):
            aod_local_angles = _world_angles_to_local_angles(
                canon.aod_az[seg_starts],
                canon.aod_el[seg_starts],
                aod_device_orientation,
            )

        # Delay range filter (index once, reuse for min and max)
        if (delay_min_ns is not None or delay_max_ns is not None) and canon.delay.size:
            segment_delays = (
                canon.segment_delay if canon.segment_delay is not None else canon.delay[seg_starts]
            )
            if delay_min_ns is not None:
                segment_allowed &= segment_delays >= delay_min_ns
            if delay_max_ns is not None:
                segment_allowed &= segment_delays <= delay_max_ns

        # Power (path loss) range filter (index once, reuse for min and max)
        if (power_min_db is not None or power_max_db is not None) and canon.loss.size:
            segment_loss = (
                canon.segment_loss if canon.segment_loss is not None else canon.loss[seg_starts]
            )
            if power_min_db is not None:
                segment_allowed &= segment_loss >= power_min_db
            if power_max_db is not None:
                segment_allowed &= segment_loss <= power_max_db

        # AoA Azimuth filter (transformed to device-local coordinates if orientation provided)
        if apply_aoa_angle_filters and canon.aoa_az is not None and canon.aoa_az.size:
            segment_aoa_az = canon.aoa_az[seg_starts]
            if aoa_local_angles is not None:
                segment_aoa_az_local = aoa_local_angles[0]
            elif aoa_device_orientation is not None:
                device_yaw_deg = np.degrees(aoa_device_orientation[0])
                segment_aoa_az_local = normalize_angle_deg(segment_aoa_az - device_yaw_deg)
            else:
                segment_aoa_az_local = segment_aoa_az

            if eff_aoa_az_min_deg is not None or eff_aoa_az_max_deg is not None:
                segment_allowed &= _angle_range_mask(
                    segment_aoa_az_local,
                    eff_aoa_az_min_deg,
                    eff_aoa_az_max_deg,
                )

        # AoA Elevation filter (transformed to device-local coordinates if orientation provided)
        if apply_aoa_angle_filters and canon.aoa_el is not None and canon.aoa_el.size:
            segment_aoa_el = canon.aoa_el[seg_starts]
            if aoa_local_angles is not None:
                segment_aoa_el_local = aoa_local_angles[1]
            elif aoa_device_orientation is not None:
                device_pitch_deg = np.degrees(aoa_device_orientation[1])
                segment_aoa_el_local = segment_aoa_el + device_pitch_deg
                segment_aoa_el_local = np.clip(segment_aoa_el_local, -90.0, 90.0)
            else:
                segment_aoa_el_local = segment_aoa_el

            if eff_aoa_el_min_deg is not None:
                segment_allowed &= segment_aoa_el_local >= eff_aoa_el_min_deg
            if eff_aoa_el_max_deg is not None:
                segment_allowed &= segment_aoa_el_local <= eff_aoa_el_max_deg

        # AoD Azimuth filter (transformed to device-local coordinates if orientation provided)
        if apply_aod_angle_filters and canon.aod_az is not None and canon.aod_az.size:
            segment_aod_az = canon.aod_az[seg_starts]
            if aod_local_angles is not None:
                segment_aod_az_local = aod_local_angles[0]
            elif aod_device_orientation is not None:
                device_yaw_deg = np.degrees(aod_device_orientation[0])
                segment_aod_az_local = normalize_angle_deg(segment_aod_az - device_yaw_deg)
            else:
                segment_aod_az_local = segment_aod_az

            if eff_aod_az_min_deg is not None or eff_aod_az_max_deg is not None:
                segment_allowed &= _angle_range_mask(
                    segment_aod_az_local,
                    eff_aod_az_min_deg,
                    eff_aod_az_max_deg,
                )

        # AoD Elevation filter (transformed to device-local coordinates if orientation provided)
        if apply_aod_angle_filters and canon.aod_el is not None and canon.aod_el.size:
            segment_aod_el = canon.aod_el[seg_starts]
            if aod_local_angles is not None:
                segment_aod_el_local = aod_local_angles[1]
            elif aod_device_orientation is not None:
                device_pitch_deg = np.degrees(aod_device_orientation[1])
                segment_aod_el_local = segment_aod_el + device_pitch_deg
                segment_aod_el_local = np.clip(segment_aod_el_local, -90.0, 90.0)
            else:
                segment_aod_el_local = segment_aod_el

            if eff_aod_el_min_deg is not None:
                segment_allowed &= segment_aod_el_local >= eff_aod_el_min_deg
            if eff_aod_el_max_deg is not None:
                segment_allowed &= segment_aod_el_local <= eff_aod_el_max_deg
    if profile is not None:
        profile["mpc_filter_range_angle_ms"] = (time.perf_counter() - range_angle_start) * 1000.0

    point_mask_start = time.perf_counter()
    point_mask = _point_mask_from_segment_mask(canon.lines, segment_allowed, canon.points.shape[0])

    # Fallback selection path for manually-constructed canonical fixtures that
    # do not populate per-segment TX/RX IDs.
    if selected_tx != "all" and canon.tx_id is not None and segment_tx_ids is None:
        tx_mask = canon.tx_id == int(selected_tx)
        point_mask &= tx_mask

    if selected_rx != "all" and canon.rx_id is not None and segment_rx_ids is None:
        rx_mask = canon.rx_id == int(selected_rx)
        point_mask &= rx_mask

    if canon.lines.size and (
        (selected_tx != "all" and segment_tx_ids is None)
        or (selected_rx != "all" and segment_rx_ids is None)
    ):
        segment_allowed &= point_mask[canon.lines[:, 0]] & point_mask[canon.lines[:, 1]]
    if profile is not None:
        profile["mpc_filter_point_mask_ms"] = (time.perf_counter() - point_mask_start) * 1000.0

    return point_mask, segment_allowed


def remap_lines(canon_lines: np.ndarray, point_mask: np.ndarray) -> np.ndarray:
    """Remap canonical line indices after applying a point mask."""
    N = int(point_mask.shape[0])
    new_index = np.full(N, -1, dtype=np.int32)
    new_index[point_mask] = np.arange(int(point_mask.sum()), dtype=np.int32)
    l0 = new_index[canon_lines[:, 0]]
    l1 = new_index[canon_lines[:, 1]]
    keep = (l0 >= 0) & (l1 >= 0)
    if not np.any(keep):
        return np.empty((0, 2), dtype=np.int32)
    return np.stack((l0[keep], l1[keep]), axis=1)


def colorize(
    canon: CanonicalStepData,
    mask: np.ndarray,
    mode: str,
    order_palette: np.ndarray,  # float32 [max_order+1,3]
    type_palette: np.ndarray,  # float32 [max_type+1, 3]
    viridis256: np.ndarray,  # float32 [256,3]
    material_colors: Optional[dict[str, np.ndarray]] = None,  # material_name -> [r,g,b]
) -> np.ndarray:
    """Return per-point RGB colors for the selected color mode."""
    if mode == "reflection_order":
        idx = canon.order[mask]
        idx = np.clip(idx, 0, len(order_palette) - 1)
        return order_palette[idx]
    if mode == "mpc_type":
        return colorize_mpc_interaction_types(canon.itype[mask], type_palette)
    if mode == "reconstruction_type":
        raw = canon.itype[mask].astype(np.int32, copy=False)
        colors = np.zeros((len(raw), 3), dtype=np.float32)
        colors[raw == 2] = np.array([0.2, 0.8, 0.3], dtype=np.float32)
        colors[raw == 99] = np.array([1.0, 0.6, 0.2], dtype=np.float32)
        colors[(raw == 0) | (raw == 1)] = np.array([0.3, 0.5, 0.9], dtype=np.float32)
        other_mask = (raw != 2) & (raw != 99) & (raw != 0) & (raw != 1)
        colors[other_mask] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        return colors
    if mode == "delay":
        lut = ensure_continuous_lut()
        return map_values_to_lut(canon.delay[mask], canon.delay_min, canon.delay_max, lut)
    if mode == "path_loss":
        lut = ensure_continuous_lut()
        return map_values_to_lut(canon.loss[mask], canon.loss_min, canon.loss_max, lut)
    if mode == "material":
        # Colors are per point; rendering samples each segment's start node.
        n_points_masked = int(mask.sum())
        point_colors = np.full(
            (n_points_masked, 3),
            MPC_MATERIAL_FALLBACK_GRAY,
            dtype=np.float32,
        )
        if n_points_masked == 0:
            return point_colors

        # Determine which original points are path starts (global, then masked)
        N = canon.points.shape[0]
        is_start = np.ones(N, dtype=bool)
        if canon.lines.size:
            # Any node that appears as a line end is not a start
            is_start[canon.lines[:, 1]] = False
        masked_is_start = is_start[mask]

        _use_id_color = (
            material_colors is not None
            and canon.material_ids is not None
            and canon.material_id_to_name is not None
            and canon.material_id_to_bare is not None
        )
        _use_str_color = (
            not _use_id_color and material_colors is not None and canon.material_names is not None
        )
        if _use_id_color:
            # Fast path: integer-based coloring using precomputed material IDs
            mat_ids_masked = canon.material_ids[mask]

            # Build expanded color lookup (name variants → color).
            expanded_colors = _expanded_material_colors(material_colors)

            # Build color LUT indexed by material_id
            max_id = int(mat_ids_masked.max()) if mat_ids_masked.size else 0
            color_lut = np.full(
                (max_id + 1, 3),
                MPC_MATERIAL_FALLBACK_GRAY,
                dtype=np.float32,
            )
            for mid in range(1, max_id + 1):
                name = canon.material_id_to_name.get(mid, "")
                bare = canon.material_id_to_bare.get(mid, "")
                itu = canon.material_id_to_itu.get(mid, "") if canon.material_id_to_itu else ""
                matched = _lookup_material_color_for_id(expanded_colors, name, bare, itu)
                if matched is not None:
                    color_lut[mid] = matched

            ids_clipped = np.clip(mat_ids_masked, 0, max_id)
            point_colors = color_lut[ids_clipped]

            no_material = mat_ids_masked == 0

        elif _use_str_color:
            # Slow fallback: string-based coloring
            mat_all = np.asarray(canon.material_names)
            mat_m = mat_all[mask]

            expanded_colors_str = _expanded_material_colors(material_colors)

            def _normalize_material_name(name) -> str:
                """Normalize material labels for string-based color lookup."""
                if isinstance(name, bytes):
                    try:
                        name = name.decode("utf-8")
                    except UnicodeDecodeError:
                        return ""
                if not name:
                    return ""
                return str(name).strip().strip("\x00").replace(" ", "_").lower()

            normalized_names = np.array([_normalize_material_name(m) for m in mat_m], dtype=object)
            unique_names = set(normalized_names)
            unique_names.discard("")

            # Build ITU type lookup for fallback matching
            itu_by_name: dict[str, str] = {}
            if canon.material_itu_types is not None:
                for mat_n, itu_t in zip(canon.material_names, canon.material_itu_types):
                    n = _normalize_material_name(mat_n)
                    i = str(itu_t).strip() if itu_t else ""
                    if n and i and n not in itu_by_name:
                        itu_by_name[n] = i

            name_to_idx: dict[str, int] = {"": 0}
            color_lut_list = [np.full(3, MPC_MATERIAL_FALLBACK_GRAY, dtype=np.float32)]

            for uname in unique_names:
                matched = _lookup_material_color_for_id(
                    expanded_colors_str,
                    uname,
                    _bare_material_type(uname),
                    itu_by_name.get(uname, ""),
                )
                if matched is not None:
                    name_to_idx[uname] = len(color_lut_list)
                    color_lut_list.append(matched)
                else:
                    name_to_idx[uname] = 0

            color_lut_arr = np.array(color_lut_list, dtype=np.float32)
            indices = np.array([name_to_idx.get(n, 0) for n in normalized_names], dtype=np.int32)
            point_colors = color_lut_arr[indices]
            no_material = normalized_names == ""
        else:
            no_material = np.ones(n_points_masked, dtype=bool)

        # Set black where point is a path start and has no material color
        to_black = masked_is_start & no_material
        point_colors[to_black] = [0.0, 0.0, 0.0]

        return point_colors.astype(np.float32)

    return np.full(
        (int(mask.sum()), 3),
        MPC_MATERIAL_FALLBACK_GRAY,
        dtype=np.float32,
    )


def colorize_segments(
    canon: CanonicalStepData,
    segment_mask: np.ndarray,
    mode: str,
    order_palette: np.ndarray,
    type_palette: np.ndarray,
    viridis256: np.ndarray,
    material_colors: Optional[dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Color line segments directly from per-segment canonical attributes."""
    if canon.lines.size == 0 or segment_mask.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    segment_mask = np.asarray(segment_mask, dtype=bool)
    segment_count = int(np.count_nonzero(segment_mask))
    if segment_count == 0:
        return np.empty((0, 3), dtype=np.float32)
    all_segments_visible = bool(
        segment_mask.size == canon.lines.shape[0] and segment_count == canon.lines.shape[0]
    )

    if canon.segment_start_indices is not None:
        starts = (
            canon.segment_start_indices
            if all_segments_visible
            else canon.segment_start_indices[segment_mask]
        )
    else:
        starts = canon.lines[:, 0] if all_segments_visible else canon.lines[segment_mask, 0]

    if mode == "reflection_order":
        if canon.segment_order is not None:
            idx = canon.segment_order if all_segments_visible else canon.segment_order[segment_mask]
        else:
            idx = canon.order[starts]
        idx = np.clip(idx, 0, len(order_palette) - 1)
        return order_palette[idx.astype(np.int32, copy=False)]

    if mode == "mpc_type":
        if canon.segment_itype is not None:
            raw = canon.segment_itype if all_segments_visible else canon.segment_itype[segment_mask]
        else:
            raw = canon.itype[starts]
        return colorize_mpc_interaction_types(raw, type_palette)

    if mode == "reconstruction_type":
        if canon.segment_itype is not None:
            raw = canon.segment_itype if all_segments_visible else canon.segment_itype[segment_mask]
        else:
            raw = canon.itype[starts]
        raw = raw.astype(np.int32, copy=False)
        colors = np.zeros((len(raw), 3), dtype=np.float32)
        colors[raw == 2] = np.array([0.2, 0.8, 0.3], dtype=np.float32)
        colors[raw == 99] = np.array([1.0, 0.6, 0.2], dtype=np.float32)
        colors[(raw == 0) | (raw == 1)] = np.array([0.3, 0.5, 0.9], dtype=np.float32)
        other_mask = (raw != 2) & (raw != 99) & (raw != 0) & (raw != 1)
        colors[other_mask] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        return colors

    if mode == "delay":
        if canon.segment_delay is not None:
            values = (
                canon.segment_delay if all_segments_visible else canon.segment_delay[segment_mask]
            )
        else:
            values = canon.delay[starts]
        return map_values_to_lut(values, canon.delay_min, canon.delay_max, ensure_continuous_lut())

    if mode == "path_loss":
        if canon.segment_loss is not None:
            values = (
                canon.segment_loss if all_segments_visible else canon.segment_loss[segment_mask]
            )
        else:
            values = canon.loss[starts]
        return map_values_to_lut(values, canon.loss_min, canon.loss_max, ensure_continuous_lut())

    if mode != "material":
        return np.full(
            (segment_count, 3),
            MPC_MATERIAL_FALLBACK_GRAY,
            dtype=np.float32,
        )

    colors = np.full(
        (segment_count, 3),
        MPC_MATERIAL_FALLBACK_GRAY,
        dtype=np.float32,
    )
    no_material = np.ones(segment_count, dtype=bool)
    if (
        material_colors is not None
        and canon.material_ids is not None
        and canon.material_id_to_name is not None
        and canon.material_id_to_bare is not None
    ):
        if canon.segment_material_ids is not None:
            mat_ids = (
                canon.segment_material_ids
                if all_segments_visible
                else canon.segment_material_ids[segment_mask]
            )
        else:
            mat_ids = canon.material_ids[starts]
        mat_ids = mat_ids.astype(np.int16, copy=False)
        max_id = int(mat_ids.max()) if mat_ids.size else 0
        color_lut = np.full(
            (max_id + 1, 3),
            MPC_MATERIAL_FALLBACK_GRAY,
            dtype=np.float32,
        )
        expanded_colors = _expanded_material_colors(material_colors)
        for mid in range(1, max_id + 1):
            name = canon.material_id_to_name.get(mid, "")
            bare = canon.material_id_to_bare.get(mid, "")
            itu = canon.material_id_to_itu.get(mid, "") if canon.material_id_to_itu else ""
            matched = _lookup_material_color_for_id(expanded_colors, name, bare, itu)
            if matched is not None:
                color_lut[mid] = matched
        colors = color_lut[np.clip(mat_ids, 0, max_id)]
        no_material = mat_ids == 0
    elif material_colors is not None and canon.material_names is not None:
        raw_names = np.asarray(canon.material_names, dtype=object)[starts]
        expanded_colors_str = _expanded_material_colors(material_colors)

        def _normalize_material_name(name) -> str:
            """Normalize segment material labels for string-based color lookup."""
            if isinstance(name, bytes):
                try:
                    name = name.decode("utf-8")
                except UnicodeDecodeError:
                    return ""
            if not name:
                return ""
            return str(name).strip().strip("\x00").replace(" ", "_").lower()

        normalized = np.array([_normalize_material_name(m) for m in raw_names], dtype=object)
        no_material = normalized == ""
        itu_by_name: dict[str, str] = {}
        if canon.material_itu_types is not None:
            for mat_n, itu_t in zip(canon.material_names, canon.material_itu_types):
                n = _normalize_material_name(mat_n)
                i = str(itu_t).strip() if itu_t else ""
                if n and i and n not in itu_by_name:
                    itu_by_name[n] = i
        unique_names = set(normalized)
        unique_names.discard("")
        name_to_idx: dict[str, int] = {"": 0}
        color_lut_list = [np.full(3, MPC_MATERIAL_FALLBACK_GRAY, dtype=np.float32)]
        for uname in unique_names:
            matched = _lookup_material_color_for_id(
                expanded_colors_str,
                uname,
                _bare_material_type(uname),
                itu_by_name.get(uname, ""),
            )
            if matched is not None:
                name_to_idx[uname] = len(color_lut_list)
                color_lut_list.append(matched)
            else:
                name_to_idx[uname] = 0
        color_lut_arr = np.array(color_lut_list, dtype=np.float32)
        indices = np.array([name_to_idx.get(n, 0) for n in normalized], dtype=np.int32)
        colors = color_lut_arr[indices]

    if canon.path_start_indices is not None and canon.path_start_indices.size:
        is_path_start = np.zeros(canon.points.shape[0], dtype=bool)
        valid_starts = canon.path_start_indices[
            (canon.path_start_indices >= 0) & (canon.path_start_indices < canon.points.shape[0])
        ]
        is_path_start[valid_starts.astype(np.int32, copy=False)] = True
        colors[is_path_start[starts] & no_material] = [0.0, 0.0, 0.0]
    return colors.astype(np.float32, copy=False)


def ensure_luts() -> np.ndarray:
    """Return the shared Viridis LUT used by MPC color modes."""
    return ensure_viridis_lut()
