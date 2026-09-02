#!/usr/bin/env python3
"""Build compact ``StandardMPCFrame`` objects from Sionna RT outputs.

This module is the Sionna-specific half of the I/O boundary. It reads Sionna RT
path tensors, device scene objects, target managers, material metadata, and
optional sensing output, then emits the canonical source-neutral frame.

Key vocabulary:
- Sionna path tensors: raw ``paths.valid``, ``paths.vertices``, and
  ``paths.interactions`` arrays from a ray-tracing step.
- Retained path indices: valid per-pair Sionna indices after optional power and
  Top-K filtering. Geometry and materials are expanded only for these paths.
- ``StandardMPCFrame``: the validated ORCHAV frame contract returned by
  ``process_frame_data`` and ``process_cached_frame_data``.
"""

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from generator.core.materials.path_metadata import MaterialMapping
from generator.core.sionna_integration import SIONNA_VERSION, version_greater_equal
from generator.core.utils import to_float
from shared.frames.contracts import (
    PATH_METRIC_VALIDITY_BITS,
    PathMetric,
)
from shared.frames.sionna_metadata import (
    SIONNA_INTERACTION_DIFFRACTION,
    SIONNA_INTERACTION_DIFFUSE,
    SIONNA_INTERACTION_LABELS,
    SIONNA_INTERACTION_REFRACTION,
    SIONNA_INTERACTION_SPECULAR,
    SIONNA_NON_LOS_INTERACTION_TYPES,
)
from shared.frames.types import StandardMPCFrame
from shared.logging import get_logger

from .filtering import (
    DEFAULT_PATH_FILTER_CONFIG,
    select_path_indices_by_power,
)

logger = get_logger(__name__)

AMPLITUDE_TO_DB_SCALE = 20.0
PATH_LOSS_EPSILON = 1e-12
SECONDS_TO_NANOSECONDS = 1e9

__all__ = [
    "process_frame_data",
    "process_cached_frame_data",
]


def _format_sionna_interaction_types(unique_interactions: np.ndarray) -> str:
    return ", ".join(
        f"{int_type}={SIONNA_INTERACTION_LABELS.get(int(int_type), f'Unknown({int(int_type)})')}"
        for int_type in unique_interactions
    )


def _azimuth_degrees_float32(radians: np.ndarray) -> np.ndarray:
    """Convert radians to float32 degrees while preserving ``[0, 360)``."""
    azimuth = np.mod(np.degrees(radians), 360.0).astype(np.float32, copy=False)
    # A value just below 360 can round to exactly 360 during the float32 cast.
    # Normalize once more at storage precision so the half-open range holds.
    return np.mod(azimuth, np.float32(360.0))


def process_material_mapping(
    material_mapping: MaterialMapping | None,
    num_tx: int,
    num_rx: int,
    retained_indices_by_pair: list[np.ndarray],
    path_lengths_by_pair: list[np.ndarray],
    total_bounces: int,
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Factor retained per-bounce materials into one compact catalog."""

    catalog: list[tuple[str, str]] = [("", "")]
    catalog_index = {catalog[0]: 0}
    material_ids = np.zeros((total_bounces,), dtype=np.uint16)
    cursor = 0

    if material_mapping is None:
        logger.warning("      No material mapping provided")

    for rx_idx in range(num_rx):
        for tx_idx in range(num_tx):
            pair_index = rx_idx * num_tx + tx_idx
            pair_materials = (
                material_mapping.get((tx_idx, rx_idx), {}) if material_mapping is not None else {}
            )
            retained = retained_indices_by_pair[pair_index]
            lengths = path_lengths_by_pair[pair_index]
            for raw_path_index, bounce_count in zip(retained, lengths, strict=True):
                path_materials = pair_materials.get(int(raw_path_index), [])
                for bounce_index in range(int(bounce_count)):
                    material = (
                        path_materials[bounce_index] if bounce_index < len(path_materials) else None
                    )
                    if material is None:
                        entry = ("", "")
                    else:
                        entry = (
                            str(material.get("name") or "unknown"),
                            str(material.get("itu_type") or ""),
                        )
                    material_id = catalog_index.get(entry)
                    if material_id is None:
                        material_id = len(catalog)
                        if material_id > np.iinfo(np.uint16).max:
                            raise ValueError("Material catalog exceeds the uint16 ID range")
                        catalog_index[entry] = material_id
                        catalog.append(entry)
                    material_ids[cursor] = material_id
                    cursor += 1

    if cursor != total_bounces:
        raise ValueError(
            f"Material alignment produced {cursor} entries for {total_bounces} bounces"
        )
    return (
        material_ids,
        tuple(name for name, _ in catalog),
        tuple(itu_type for _, itu_type in catalog),
    )


def _extract_device_positions_and_orientations(
    tx_list: list[Any], rx_list: list[Any], frame_idx: int
) -> tuple:
    """Extract current positions and orientations from Sionna device objects.

    Args:
        tx_list: List of transmitter objects.
        rx_list: List of receiver objects.
        frame_idx: Current frame index for logging.

    Returns:
        Tuple of ``(tx_positions, rx_positions, tx_orientations,
        rx_orientations, tx_rx_pairs)`` where positions/orientations are
        unique per-device arrays and ``tx_rx_pairs`` is a list of ``(tx_idx, rx_idx)``
        tuples in RX-major order.
    """
    tx_positions = np.asarray(
        [np.array(tx.position, dtype=np.float64).flatten() for tx in tx_list],
        dtype=np.float64,
    ).reshape((-1, 3))
    rx_positions = np.asarray(
        [np.array(rx.position, dtype=np.float64).flatten() for rx in rx_list],
        dtype=np.float64,
    ).reshape((-1, 3))
    tx_orientations = np.asarray(
        [
            np.array(
                [
                    float(np.asarray(tx.orientation.x).flat[0]),
                    float(np.asarray(tx.orientation.y).flat[0]),
                    float(np.asarray(tx.orientation.z).flat[0]),
                ],
                dtype=np.float64,
            )
            for tx in tx_list
        ],
        dtype=np.float64,
    ).reshape((-1, 3))
    rx_orientations = np.asarray(
        [
            np.array(
                [
                    float(np.asarray(rx.orientation.x).flat[0]),
                    float(np.asarray(rx.orientation.y).flat[0]),
                    float(np.asarray(rx.orientation.z).flat[0]),
                ],
                dtype=np.float64,
            )
            for rx in rx_list
        ],
        dtype=np.float64,
    ).reshape((-1, 3))

    tx_rx_pairs = np.asarray(
        np.array(
            [(tx_idx, rx_idx) for rx_idx in range(len(rx_list)) for tx_idx in range(len(tx_list))],
            dtype=np.int32,
        )
    ).reshape((-1, 2))

    logger.debug("      Current TX/RX positions and orientations for frame %d:", frame_idx + 1)
    for rx_idx in range(len(rx_list)):
        for tx_idx in range(len(tx_list)):
            tx_pos = tx_positions[tx_idx]
            rx_pos = rx_positions[rx_idx]
            tx_orient = tx_orientations[tx_idx]
            rx_orient = rx_orientations[rx_idx]
            logger.debug("         TX%d->RX%d: %s -> %s", tx_idx + 1, rx_idx + 1, tx_pos, rx_pos)
            logger.debug(
                "         TX%d->RX%d: orient %s -> %s",
                tx_idx + 1,
                rx_idx + 1,
                tx_orient,
                rx_orient,
            )
            logger.debug("           (Data index: TX%d->RX%d)", tx_idx, rx_idx)

    logger.debug("      [STATS] Total TX-RX pairs: %d", len(tx_rx_pairs))
    logger.debug("      TX indices: %s", [f"TX{i+1}(idx={i})" for i in range(len(tx_list))])
    logger.debug("      RX indices: %s", [f"RX{i+1}(idx={i})" for i in range(len(rx_list))])

    return (
        tx_positions,
        rx_positions,
        tx_orientations,
        rx_orientations,
        tx_rx_pairs,
    )


def _device_names(devices: list[Any], prefix: str) -> tuple[str, ...]:
    """Return source device names with a stable indexed fallback."""

    names: list[str] = []
    for index, device in enumerate(devices):
        raw_name = getattr(device, "name", None)
        name = str(raw_name).strip() if raw_name is not None else ""
        names.append(name or f"{prefix}_{index}")
    return tuple(names)


def _normalize_sionna_path_tensors(
    valid: np.ndarray,
    vertices: np.ndarray,
    interactions: np.ndarray,
    num_tx: int,
    num_rx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restore Sionna's collapsed single-device axes without copying data."""

    valid = np.asarray(valid)
    vertices = np.asarray(vertices)
    interactions = np.asarray(interactions)

    if valid.ndim == 3:
        if valid.shape[:2] != (num_rx, num_tx):
            raise ValueError(
                f"valid shape {valid.shape} does not match {num_rx} RX and {num_tx} TX"
            )
        num_paths = int(valid.shape[2])
    elif num_tx == 1 and num_rx == 1 and valid.ndim in (1, 2):
        num_paths = int(valid.size)
        valid = valid.reshape((1, 1, num_paths))
    else:
        raise ValueError(f"Unexpected valid shape {valid.shape} for {num_tx} TX and {num_rx} RX")

    if vertices.ndim == 5:
        expected_tail = (num_rx, num_tx, num_paths, 3)
        if vertices.shape[1:] != expected_tail:
            raise ValueError(f"vertices shape {vertices.shape} does not match (*, {expected_tail})")
    elif num_tx == 1 and num_rx == 1 and vertices.ndim in (3, 4):
        if vertices.shape[-1] != 3:
            raise ValueError(f"vertices must end in XYZ coordinates, got {vertices.shape}")
        max_depth = int(vertices.shape[0])
        if vertices.size != max_depth * num_paths * 3:
            raise ValueError(f"vertices shape {vertices.shape} does not match {num_paths} paths")
        vertices = vertices.reshape((max_depth, 1, 1, num_paths, 3))
    else:
        raise ValueError(
            f"Unexpected vertices shape {vertices.shape} for {num_tx} TX and {num_rx} RX"
        )

    max_depth = int(vertices.shape[0])
    if interactions.ndim == 4:
        expected = (max_depth, num_rx, num_tx, num_paths)
        if interactions.shape != expected:
            raise ValueError(f"interactions shape {interactions.shape} does not match {expected}")
    elif num_tx == 1 and num_rx == 1 and interactions.ndim in (2, 3):
        if interactions.size != max_depth * num_paths:
            raise ValueError(
                f"interactions shape {interactions.shape} does not match "
                f"depth {max_depth} and {num_paths} paths"
            )
        interactions = interactions.reshape((max_depth, 1, 1, num_paths))
    else:
        raise ValueError(
            f"Unexpected interactions shape {interactions.shape} "
            f"for {num_tx} TX and {num_rx} RX"
        )

    return valid.astype(np.bool_, copy=False), vertices, interactions


def _valid_path_indices_by_pair(
    valid: np.ndarray,
    num_tx: int,
    num_rx: int,
) -> list[np.ndarray]:
    """Return raw valid-path indices in the canonical RX-major pair order."""

    return [
        np.flatnonzero(valid[rx_idx, tx_idx]).astype(np.int64, copy=False)
        for rx_idx in range(num_rx)
        for tx_idx in range(num_tx)
    ]


def _pack_selected_geometry(
    vertices: np.ndarray,
    interactions: np.ndarray,
    retained_indices_by_pair: list[np.ndarray],
    num_tx: int,
    num_rx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Pack retained Sionna paths directly into path/bounce offset arrays."""

    pair_path_offsets = np.zeros((len(retained_indices_by_pair) + 1,), dtype=np.int64)
    lengths_by_pair: list[np.ndarray] = []
    xyz_parts: list[np.ndarray] = []
    interaction_parts: list[np.ndarray] = []

    pair_index = 0
    for rx_idx in range(num_rx):
        for tx_idx in range(num_tx):
            retained = retained_indices_by_pair[pair_index]
            pair_path_offsets[pair_index + 1] = pair_path_offsets[pair_index] + len(retained)
            if retained.size == 0:
                lengths_by_pair.append(np.empty((0,), dtype=np.int64))
                pair_index += 1
                continue

            pair_interactions = np.asarray(interactions[:, rx_idx, tx_idx, retained])
            bounce_mask = pair_interactions > 0
            lengths = np.sum(bounce_mask, axis=0, dtype=np.int64)
            lengths_by_pair.append(lengths)

            physical_interactions = pair_interactions.T[bounce_mask.T]
            if physical_interactions.size and (
                np.any(physical_interactions <= 0)
                or np.any(physical_interactions > np.iinfo(np.uint8).max)
            ):
                raise ValueError("Physical interaction codes must be in [1, 255]")
            interaction_parts.append(physical_interactions.astype(np.uint8, copy=False))

            pair_vertices = np.asarray(vertices[:, rx_idx, tx_idx, retained, :])
            physical_xyz = pair_vertices.transpose((1, 0, 2))[bounce_mask.T].astype(
                np.float32,
                copy=False,
            )
            if np.any(~np.isfinite(physical_xyz)):
                raise ValueError("Physical bounce vertices must contain finite coordinates")
            xyz_parts.append(physical_xyz)
            pair_index += 1

    total_paths = int(pair_path_offsets[-1])
    path_lengths = (
        np.concatenate(lengths_by_pair) if total_paths else np.empty((0,), dtype=np.int64)
    )
    bounce_offsets = np.zeros((total_paths + 1,), dtype=np.int64)
    if total_paths:
        np.cumsum(path_lengths, dtype=np.int64, out=bounce_offsets[1:])

    total_bounces = int(bounce_offsets[-1])
    bounce_xyz_m = (
        np.concatenate(xyz_parts, axis=0) if total_bounces else np.empty((0, 3), dtype=np.float32)
    )
    packed_interactions = (
        np.concatenate(interaction_parts) if total_bounces else np.empty((0,), dtype=np.uint8)
    )
    return (
        pair_path_offsets,
        bounce_offsets,
        bounce_xyz_m,
        packed_interactions,
        lengths_by_pair,
    )


_PAIR_METRIC_FIELDS: dict[PathMetric, str] = {
    PathMetric.DELAY_NS: "all_pair_delays_ns",
    PathMetric.PATH_LOSS_DB: "all_pair_path_loss_db",
    PathMetric.AOA_AZ_DEG: "all_pair_aoa_az_deg",
    PathMetric.AOA_EL_DEG: "all_pair_aoa_el_deg",
    PathMetric.AOD_AZ_DEG: "all_pair_aod_az_deg",
    PathMetric.AOD_EL_DEG: "all_pair_aod_el_deg",
}

_COMPACT_METRIC_FIELDS: dict[PathMetric, str] = {
    PathMetric.DELAY_NS: "delays_ns",
    PathMetric.PATH_LOSS_DB: "path_loss_db",
    PathMetric.AOA_AZ_DEG: "aoa_az_deg",
    PathMetric.AOA_EL_DEG: "aoa_el_deg",
    PathMetric.AOD_AZ_DEG: "aod_az_deg",
    PathMetric.AOD_EL_DEG: "aod_el_deg",
}


def _pack_path_metrics(
    metrics_by_pair: dict[str, list[np.ndarray]] | None,
    pair_path_offsets: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Flatten optional metrics and encode validity without sentinel values."""

    total_paths = int(pair_path_offsets[-1])
    metric_valid_bits = np.zeros((total_paths,), dtype=np.uint8)
    compact: dict[str, np.ndarray] = {}
    metrics = metrics_by_pair or {}

    for metric, source_field in _PAIR_METRIC_FIELDS.items():
        values = np.full((total_paths,), np.nan, dtype=np.float32)
        pair_columns = metrics.get(source_field, [])
        for pair_index in range(len(pair_path_offsets) - 1):
            start = int(pair_path_offsets[pair_index])
            end = int(pair_path_offsets[pair_index + 1])
            path_count = end - start
            if pair_index >= len(pair_columns):
                continue
            pair_values = np.asarray(pair_columns[pair_index], dtype=np.float32).reshape(-1)
            if pair_values.size == 0:
                continue
            if pair_values.size != path_count:
                raise ValueError(
                    f"{source_field}[{pair_index}] has {pair_values.size} values "
                    f"for {path_count} retained paths"
                )
            finite = np.isfinite(pair_values)
            pair_output = values[start:end]
            pair_output[finite] = pair_values[finite]
            pair_validity = metric_valid_bits[start:end]
            pair_validity[finite] |= np.uint8(PATH_METRIC_VALIDITY_BITS[metric])
        compact[_COMPACT_METRIC_FIELDS[metric]] = values

    return compact, metric_valid_bits


def _process_target_metadata(target_objects, target_managers, frame_idx):
    """Return target positions and metadata for one output frame."""
    # Positions are stored as an (n_targets, 3) array.
    if target_objects:
        positions = []
        for target_obj in target_objects:
            pos = target_obj.position
            positions.append([to_float(pos.x), to_float(pos.y), to_float(pos.z)])
        target_positions = np.array(positions, dtype=np.float64)
        logger.debug(f"[TARGET EXPORT] Frame {frame_idx}: target_pos: {target_positions}")
    else:
        target_positions = np.zeros((0, 3), dtype=np.float64)
        logger.debug(f"[TARGET EXPORT] Frame {frame_idx}: No targets")

    targets_metadata = []
    if target_objects:
        logger.debug(
            f"[TARGET METADATA] Frame {frame_idx}: Found {len(target_objects)} target objects"
        )
        for i, target_obj in enumerate(target_objects):
            logger.debug(
                f"[TARGET METADATA] Frame {frame_idx}: Processing target object {i}: {target_obj}"
            )
            target_manager = None
            for tm in target_managers:
                if tm.target_object == target_obj:
                    target_manager = tm
                    logger.debug(
                        f"[TARGET METADATA] Frame {frame_idx}: Found target manager for object {i}"
                    )
                    break

            if target_manager:
                current_mesh_idx = target_manager.current_mesh_idx
                current_mesh_file = (
                    target_manager.meshes[current_mesh_idx] if target_manager.meshes else "unknown"
                )
                logger.info(
                    f"[TARGET METADATA] Frame {frame_idx} (display={frame_idx+1}): Extracting mesh_file from target_manager: mesh_idx={current_mesh_idx}, mesh_file={os.path.basename(current_mesh_file)}"
                )

                # Metadata requires a position even when the runtime object does
                # not expose one, so the declared initial position is the fallback.
                try:
                    if hasattr(target_obj, "position") and target_obj.position is not None:
                        current_pos = [
                            to_float(target_obj.position.x),
                            to_float(target_obj.position.y),
                            to_float(target_obj.position.z),
                        ]
                        logger.debug(
                            f"[TARGET METADATA] Frame {frame_idx}: Got position from target object: {current_pos}"
                        )
                    else:
                        current_pos = list(target_manager.config.initial_position)
                        logger.warning(
                            f"[TARGET METADATA] Frame {frame_idx}: Using initial position from config: {current_pos}"
                        )
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(
                        f"[TARGET METADATA] Frame {frame_idx}: Error getting position from target object: {e}"
                    )
                    current_pos = list(target_manager.config.initial_position)
                    logger.debug(
                        f"[TARGET METADATA] Frame {frame_idx}: Fallback to initial position: {current_pos}"
                    )

                if hasattr(target_obj, "orientation") and target_obj.orientation is not None:
                    orientation_values = [
                        to_float(target_obj.orientation.x),
                        to_float(target_obj.orientation.y),
                        to_float(target_obj.orientation.z),
                    ]
                else:
                    # Missing runtime orientation is represented by zero Euler angles.
                    logger.warning(
                        f"[TARGET METADATA] Frame {frame_idx}: Target object {i} has no orientation, using zero orientation as fallback"
                    )
                    orientation_values = [0.0, 0.0, 0.0]

                # A portable mesh reference combines the basename with its
                # scenario-relative directory.
                target_meta = {
                    "name": target_manager.config.name,
                    "mesh_file": os.path.basename(current_mesh_file),
                    "mesh_directory": target_manager.relative_mesh_directory,
                    "mesh_index": current_mesh_idx,
                    "scale": to_float(target_manager.config.scale),
                    "orientation": orientation_values,
                    "material_type": target_manager.config.material_type,
                    "material_id": (
                        f"mat-itu_{target_manager.config.material_type}_"
                        f"{target_manager.config.name}"
                    ),
                    "use_ply_position": target_manager.config.use_ply_position,
                    "current_position": current_pos,
                    "mobility_type": (
                        type(target_manager.config.mobility).__name__
                        if target_manager.config.mobility
                        else "Stationary"
                    ),
                }
                targets_metadata.append(target_meta)

                logger.info(
                    f"[TARGET METADATA] Frame {frame_idx} (0-indexed, display={frame_idx+1}): {target_meta['name']} - {target_meta['mesh_file']} at {target_meta['current_position']}, orient={orientation_values}"
                )
                logger.debug(f"[TARGET METADATA] Frame {frame_idx}: Full metadata: {target_meta}")
            else:
                logger.error(
                    f"[TARGET METADATA] Frame {frame_idx}: Could not find target manager for target object {i}"
                )
                logger.debug(
                    f"[TARGET METADATA] Frame {frame_idx}: Available target managers: {[tm.config.name for tm in target_managers]}"
                )
                logger.debug(
                    f"[TARGET METADATA] Frame {frame_idx}: Target object type: {type(target_obj)}"
                )
    else:
        logger.debug(f"[TARGET METADATA] Frame {frame_idx}: No target objects found")

    return target_positions, targets_metadata


def _extract_optional_path_metrics(paths, valid, num_tx, num_rx, simulation_config):
    """Return optional per-path delay, path-loss, AOA, and AOD arrays.

    Metrics are exported only when enabled on ``simulation_config``. Returned
    arrays are RX-major and filtered with the same valid-path mask used for
    bounce geometry.
    """
    export_metrics = bool(getattr(simulation_config, "export_path_metrics", False))
    all_pair_metrics = None
    if export_metrics:
        try:

            def _to_numpy_array(value):
                if value is None:
                    return None
                try:
                    if hasattr(value, "numpy"):
                        value = value.numpy()
                except (ValueError, TypeError, RuntimeError):
                    return None
                try:
                    return np.asarray(value)
                except (ValueError, TypeError):
                    return None

            def _path_coefficients_from_attr():
                a_attr = _to_numpy_array(getattr(paths, "a", None))
                if a_attr is None:
                    return None

                # Handle the common real/imag stacked representation.
                if a_attr.ndim >= 1 and a_attr.shape[0] == 2 and not np.iscomplexobj(a_attr):
                    return a_attr[0] + 1j * a_attr[1]
                return a_attr

            # Sionna delays are seconds; compact frame delays are nanoseconds.
            tau_full = getattr(paths, "tau", None)
            tau_full = (
                tau_full.numpy() if tau_full is not None and hasattr(tau_full, "numpy") else None
            )

            # Path loss is derived from complex CIR amplitude below.
            a_full = None
            cir_fn = getattr(paths, "cir", None)
            if callable(cir_fn):
                try:
                    cir_result = cir_fn(normalize_delays=False, out_type="numpy", num_time_steps=1)
                    a_full = cir_result[0] if isinstance(cir_result, tuple) else cir_result
                except (ValueError, TypeError, RuntimeError):
                    a_full = _path_coefficients_from_attr()
            else:
                a_full = _path_coefficients_from_attr()

            # Sionna exposes AOA/AOD as spherical-coordinate angles.
            theta_r_full = getattr(paths, "theta_r", None)  # Zenith angle at RX
            phi_r_full = getattr(paths, "phi_r", None)  # Azimuth angle at RX
            theta_t_full = getattr(paths, "theta_t", None)  # Zenith angle at TX
            phi_t_full = getattr(paths, "phi_t", None)  # Azimuth angle at TX

            theta_r_full = (
                theta_r_full.numpy()
                if theta_r_full is not None and hasattr(theta_r_full, "numpy")
                else None
            )
            phi_r_full = (
                phi_r_full.numpy()
                if phi_r_full is not None and hasattr(phi_r_full, "numpy")
                else None
            )
            theta_t_full = (
                theta_t_full.numpy()
                if theta_t_full is not None and hasattr(theta_t_full, "numpy")
                else None
            )
            phi_t_full = (
                phi_t_full.numpy()
                if phi_t_full is not None and hasattr(phi_t_full, "numpy")
                else None
            )

            def sel_pair_1d(arr, rx, tx, kind="tau"):
                if arr is None:
                    return None
                if arr.ndim == 3:
                    return arr[rx, tx, :]
                if arr.ndim == 5:
                    # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
                    return arr[rx, 0, tx, 0, :]
                logger.warning(f"Unexpected {kind} dims: {arr.shape}")
                return None

            def sel_pair_a(arr, rx, tx, expected_paths):
                if arr is None:
                    return None
                pair = None
                if arr.ndim >= 5 and arr.shape[0] > rx and arr.shape[2] > tx:
                    # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths(, num_time_steps)]
                    pair = arr[rx, 0, tx, 0, ...]
                elif arr.ndim >= 3 and arr.shape[0] > rx and arr.shape[1] > tx:
                    # [num_rx, num_tx, num_paths(, num_time_steps)]
                    pair = arr[rx, tx, ...]
                else:
                    logger.warning(f"Unexpected a dims: {arr.shape}")
                    return None

                pair = np.asarray(pair)
                candidate_axes = [
                    axis for axis, size in enumerate(pair.shape) if size == expected_paths
                ]
                if not candidate_axes:
                    logger.warning(
                        "Could not identify path axis for a pair slice: shape=%s expected_paths=%d",
                        pair.shape,
                        expected_paths,
                    )
                    return None

                # Move the path axis to the front, then collapse any remaining
                # antenna/time axes by taking their first entry.
                pair = np.moveaxis(pair, candidate_axes[-1], 0)
                while pair.ndim > 1:
                    pair = np.take(pair, 0, axis=-1)
                return pair

            pair_delays_ns = []
            pair_path_loss_db = []
            pair_aoa_az_deg = []
            pair_aoa_el_deg = []
            pair_aod_az_deg = []
            pair_aod_el_deg = []

            # Keep the per-pair metric vectors aligned with the valid raw path
            # indices that feed compact path selection.
            for rx_idx in range(num_rx):
                for tx_idx in range(num_tx):
                    vm = valid[rx_idx, tx_idx, :].astype(bool)

                    t_pair = sel_pair_1d(tau_full, rx_idx, tx_idx, kind="tau")
                    d_ns = (
                        (t_pair * SECONDS_TO_NANOSECONDS)[vm]
                        if t_pair is not None
                        else np.array([], dtype=float)
                    )

                    a_pair = sel_pair_a(a_full, rx_idx, tx_idx, expected_paths=vm.size)
                    if a_pair is not None:
                        gain = np.abs(a_pair)[vm]
                        pl_db = -AMPLITUDE_TO_DB_SCALE * np.log10(
                            np.maximum(gain, PATH_LOSS_EPSILON)
                        )
                    else:
                        pl_db = np.array([], dtype=float)

                    th_r = sel_pair_1d(theta_r_full, rx_idx, tx_idx, kind="theta_r")
                    ph_r = sel_pair_1d(phi_r_full, rx_idx, tx_idx, kind="phi_r")
                    th_t = sel_pair_1d(theta_t_full, rx_idx, tx_idx, kind="theta_t")
                    ph_t = sel_pair_1d(phi_t_full, rx_idx, tx_idx, kind="phi_t")

                    # Sionna theta is a zenith angle; stored elevation is
                    # 90 degrees minus theta.
                    if th_r is not None and ph_r is not None:
                        az_aoa = _azimuth_degrees_float32(ph_r[vm])
                        el_aoa = 90.0 - np.degrees(th_r)[vm]
                    else:
                        az_aoa = np.array([], dtype=float)
                        el_aoa = np.array([], dtype=float)

                    if th_t is not None and ph_t is not None:
                        az_aod = _azimuth_degrees_float32(ph_t[vm])
                        el_aod = 90.0 - np.degrees(th_t)[vm]
                    else:
                        az_aod = np.array([], dtype=float)
                        el_aod = np.array([], dtype=float)

                    pair_delays_ns.append(d_ns.astype(np.float32))
                    pair_path_loss_db.append(pl_db.astype(np.float32))
                    pair_aoa_az_deg.append(az_aoa.astype(np.float32))
                    pair_aoa_el_deg.append(el_aoa.astype(np.float32))
                    pair_aod_az_deg.append(az_aod.astype(np.float32))
                    pair_aod_el_deg.append(el_aod.astype(np.float32))

            all_pair_metrics = {
                "all_pair_delays_ns": pair_delays_ns,
                "all_pair_path_loss_db": pair_path_loss_db,
                "all_pair_aoa_az_deg": pair_aoa_az_deg,
                "all_pair_aoa_el_deg": pair_aoa_el_deg,
                "all_pair_aod_az_deg": pair_aod_az_deg,
                "all_pair_aod_el_deg": pair_aod_el_deg,
            }
        except (ValueError, TypeError, IndexError, AttributeError) as e:
            logger.warning(f"Failed to export per-path metrics: {e}")
            all_pair_metrics = None

    return all_pair_metrics


_SIONNA_12 = version_greater_equal(SIONNA_VERSION, "1.2")


def _snapshot_xyz(
    snapshots: Mapping[str, Any] | None,
    key: str,
    fallback: np.ndarray,
) -> np.ndarray:
    """Use a valid stored snapshot, otherwise retain the extracted values."""

    if snapshots is None or snapshots.get(key) is None:
        return fallback
    try:
        values = np.asarray(snapshots[key], dtype=np.float64)
        if values.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        return values.reshape((-1, 3))
    except (TypeError, ValueError):
        return fallback


def _apply_snapshot_overrides(
    snapshots: Mapping[str, Any] | None,
    *,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    tx_orientations: np.ndarray,
    rx_orientations: np.ndarray,
    target_positions: np.ndarray,
    targets_metadata: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Bind stored pose snapshots to the frame before its single validation."""

    tx_positions = _snapshot_xyz(snapshots, "tx_positions_snapshot", tx_positions)
    rx_positions = _snapshot_xyz(snapshots, "rx_positions_snapshot", rx_positions)
    tx_orientations = _snapshot_xyz(
        snapshots,
        "tx_orientations_snapshot",
        tx_orientations,
    )
    rx_orientations = _snapshot_xyz(
        snapshots,
        "rx_orientations_snapshot",
        rx_orientations,
    )
    target_positions = _snapshot_xyz(
        snapshots,
        "target_positions_snapshot",
        target_positions,
    )
    target_orientations = _snapshot_xyz(
        snapshots,
        "target_orientations_snapshot",
        np.empty((0, 3), dtype=np.float64),
    )

    updated_metadata: list[dict[str, Any]] = []
    for index, metadata in enumerate(targets_metadata):
        updated = dict(metadata)
        if index < len(target_positions):
            updated["current_position"] = target_positions[index].tolist()
        if index < len(target_orientations):
            updated["orientation"] = target_orientations[index].tolist()
        updated_metadata.append(updated)

    return (
        tx_positions,
        rx_positions,
        tx_orientations,
        rx_orientations,
        target_positions,
        updated_metadata,
    )


def _log_path_type_stats(
    interactions: np.ndarray,
    bounce_offsets: np.ndarray,
) -> None:
    """Log MPC path type statistics after filtering.

    Classification uses the first physical interaction of every non-LoS path,
    matching the visualizer without recreating padded per-path matrices.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    lengths = np.diff(bounce_offsets)
    los = int(np.count_nonzero(lengths == 0))
    non_los = lengths > 0
    first_types = np.asarray(interactions)[bounce_offsets[:-1][non_los]].astype(
        np.int32,
        copy=False,
    )
    specular = int(np.count_nonzero(first_types == SIONNA_INTERACTION_SPECULAR))
    diffuse = int(np.count_nonzero(first_types == SIONNA_INTERACTION_DIFFUSE))
    refraction = int(np.count_nonzero(first_types == SIONNA_INTERACTION_REFRACTION))
    diffraction = int(np.count_nonzero(first_types == SIONNA_INTERACTION_DIFFRACTION))
    known = np.isin(first_types, SIONNA_NON_LOS_INTERACTION_TYPES)
    los += int(np.count_nonzero(~known))

    total = los + specular + diffuse + refraction + diffraction
    logger.debug("[STATS] MPC PATH TYPE ANALYSIS:")
    logger.debug("   LoS paths: %d", los)
    logger.debug("   Specular paths: %d", specular)
    logger.debug("   Diffuse paths: %d", diffuse)
    if _SIONNA_12:
        logger.debug("   Refraction paths: %d", refraction)
        logger.debug("   Diffraction paths: %d", diffraction)
    logger.debug("   Total paths: %d", total)


def process_frame_data(
    frame_idx: int,
    tx_list: list[Any],
    rx_list: list[Any],
    paths: Any,
    target_objects: list[Any],
    target_managers: list[Any],
    simulation_config: Any,
    material_mapping: MaterialMapping | None = None,
    sensing_data: dict[str, Any] | None = None,
    path_filter_config: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    bandwidth_hz: float | None = None,
    snapshot_overrides: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    timestamp_s: float | None = None,
    beamforming: Mapping[str, Any] | None = None,
    recomputed_from_stored_positions: bool = False,
) -> StandardMPCFrame:
    """Convert one Sionna RT ray-tracing result into ``StandardMPCFrame``.

    Filtering selects raw Sionna indices before bounce geometry and material
    IDs are materialized. The returned frame is already compact and validated;
    storage and transport code do not perform another complete-frame packing
    pass.
    """
    num_tx = len(tx_list)
    num_rx = len(rx_list)
    logger.info("      Processing %d TX-RX pairs from computed paths...", num_tx * num_rx)

    valid, vertices, raw_interactions = _normalize_sionna_path_tensors(
        paths.valid.numpy(),
        paths.vertices.numpy(),
        paths.interactions.numpy(),
        num_tx,
        num_rx,
    )
    (
        tx_positions,
        rx_positions,
        tx_orientations,
        rx_orientations,
        tx_rx_pairs,
    ) = _extract_device_positions_and_orientations(tx_list, rx_list, frame_idx)
    metrics_by_pair = _extract_optional_path_metrics(
        paths, valid, num_tx, num_rx, simulation_config
    )
    retained_indices = _valid_path_indices_by_pair(valid, num_tx, num_rx)
    if path_filter_config is not None or metrics_by_pair is not None:
        filter_config = (
            path_filter_config if path_filter_config is not None else DEFAULT_PATH_FILTER_CONFIG
        )
        retained_indices, metrics_by_pair = select_path_indices_by_power(
            retained_indices,
            metrics_by_pair,
            filter_config,
            output_dir=output_dir,
            bandwidth_hz=bandwidth_hz,
        )

    (
        pair_path_offsets,
        bounce_offsets,
        bounce_xyz_m,
        packed_interactions,
        path_lengths_by_pair,
    ) = _pack_selected_geometry(
        vertices,
        raw_interactions,
        retained_indices,
        num_tx,
        num_rx,
    )
    material_ids, material_names, material_itu_types = process_material_mapping(
        material_mapping,
        num_tx,
        num_rx,
        retained_indices,
        path_lengths_by_pair,
        int(bounce_offsets[-1]),
    )
    metric_arrays, metric_valid_bits = _pack_path_metrics(
        metrics_by_pair,
        pair_path_offsets,
    )
    target_positions, targets_metadata = _process_target_metadata(
        target_objects, target_managers, frame_idx
    )
    (
        tx_positions,
        rx_positions,
        tx_orientations,
        rx_orientations,
        target_positions,
        targets_metadata,
    ) = _apply_snapshot_overrides(
        snapshot_overrides,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        tx_orientations=tx_orientations,
        rx_orientations=rx_orientations,
        target_positions=target_positions,
        targets_metadata=targets_metadata,
    )

    _log_path_type_stats(packed_interactions, bounce_offsets)
    source = dict(provenance or {"provider": "generator", "frame_idx": int(frame_idx)})
    return StandardMPCFrame(
        frame_index=int(frame_idx),
        timestamp_s=timestamp_s,
        tx_rx_pairs=np.asarray(tx_rx_pairs, dtype=np.int32).reshape((-1, 2)),
        pair_path_offsets=pair_path_offsets,
        bounce_offsets=bounce_offsets,
        tx_positions=np.asarray(tx_positions, dtype=np.float64).reshape((-1, 3)),
        rx_positions=np.asarray(rx_positions, dtype=np.float64).reshape((-1, 3)),
        tx_orientations=np.asarray(tx_orientations, dtype=np.float64).reshape((-1, 3)),
        rx_orientations=np.asarray(rx_orientations, dtype=np.float64).reshape((-1, 3)),
        tx_names=_device_names(tx_list, "tx"),
        rx_names=_device_names(rx_list, "rx"),
        bounce_xyz_m=bounce_xyz_m,
        interactions=packed_interactions,
        material_ids=material_ids,
        material_names=material_names,
        material_itu_types=material_itu_types,
        metric_valid_bits=metric_valid_bits,
        target_positions_m=np.asarray(target_positions, dtype=np.float64).reshape((-1, 3)),
        targets_metadata=tuple(targets_metadata),
        sensing=sensing_data,
        beamforming=beamforming,
        provenance=source,
        recomputed_from_stored_positions=bool(recomputed_from_stored_positions),
        **metric_arrays,
    )


def process_cached_frame_data(
    frame_idx: int,
    tx_list: list[Any],
    rx_list: list[Any],
    target_objects: list[Any],
    target_managers: list[Any],
    sensing_data: dict[str, Any] | None = None,
    snapshot_overrides: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    timestamp_s: float | None = None,
    beamforming: Mapping[str, Any] | None = None,
    recomputed_from_stored_positions: bool = False,
) -> StandardMPCFrame:
    """Build a lightweight frame payload for coherent cached steps.

    Cached coherent steps reuse the last RT geometry. A zero-path frame records
    the current device and target state without duplicating unchanged paths.
    """
    (
        tx_positions,
        rx_positions,
        tx_orientations,
        rx_orientations,
        tx_rx_pairs,
    ) = _extract_device_positions_and_orientations(tx_list, rx_list, frame_idx)

    target_positions, targets_metadata = _process_target_metadata(
        target_objects, target_managers, frame_idx
    )

    num_pairs = len(tx_rx_pairs)
    (
        tx_positions,
        rx_positions,
        tx_orientations,
        rx_orientations,
        target_positions,
        targets_metadata,
    ) = _apply_snapshot_overrides(
        snapshot_overrides,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        tx_orientations=tx_orientations,
        rx_orientations=rx_orientations,
        target_positions=target_positions,
        targets_metadata=targets_metadata,
    )

    empty_metrics = {
        field: np.empty((0,), dtype=np.float32) for field in _COMPACT_METRIC_FIELDS.values()
    }
    return StandardMPCFrame(
        frame_index=int(frame_idx),
        timestamp_s=timestamp_s,
        tx_rx_pairs=np.asarray(tx_rx_pairs, dtype=np.int32).reshape((-1, 2)),
        pair_path_offsets=np.zeros((num_pairs + 1,), dtype=np.int64),
        bounce_offsets=np.zeros((1,), dtype=np.int64),
        tx_positions=np.asarray(tx_positions, dtype=np.float64).reshape((-1, 3)),
        rx_positions=np.asarray(rx_positions, dtype=np.float64).reshape((-1, 3)),
        tx_orientations=np.asarray(tx_orientations, dtype=np.float64).reshape((-1, 3)),
        rx_orientations=np.asarray(rx_orientations, dtype=np.float64).reshape((-1, 3)),
        tx_names=_device_names(tx_list, "tx"),
        rx_names=_device_names(rx_list, "rx"),
        bounce_xyz_m=np.empty((0, 3), dtype=np.float32),
        interactions=np.empty((0,), dtype=np.uint8),
        material_ids=np.empty((0,), dtype=np.uint16),
        material_names=("",),
        material_itu_types=("",),
        metric_valid_bits=np.empty((0,), dtype=np.uint8),
        target_positions_m=np.asarray(target_positions, dtype=np.float64).reshape((-1, 3)),
        targets_metadata=tuple(targets_metadata),
        sensing=sensing_data,
        beamforming=beamforming,
        provenance=dict(provenance or {"provider": "generator", "frame_idx": int(frame_idx)}),
        recomputed_from_stored_positions=bool(recomputed_from_stored_positions),
        **empty_metrics,
    )
