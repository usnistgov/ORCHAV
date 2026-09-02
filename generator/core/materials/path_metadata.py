"""Extract per-path material metadata from Sionna RT path results.

Sionna path tensors report, for each interaction depth, the scene object ID that
was hit and the interaction type. This module maps those object IDs back to
scene radio-material labels so frame writers and analysis tools can attach
per-bounce material metadata to each multipath component.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from shared.frames.sionna_metadata import (
    SIONNA_INTERACTION_DIFFRACTION,
    SIONNA_INTERACTION_DIFFUSE,
    SIONNA_INTERACTION_REFRACTION,
    SIONNA_INTERACTION_SPECULAR,
    SIONNA_INVALID_OBJECT_ID,
)
from shared.logging import get_logger

from ..sionna_integration import SIONNA_VERSION, version_greater_equal

logger = get_logger(__name__)

MaterialInfo = dict[str, str | None]
PathMaterialList = list[MaterialInfo | None]
MaterialPairKey = tuple[int, int]
MaterialMapping = dict[MaterialPairKey, dict[int, PathMaterialList]]

UNKNOWN_MATERIAL: MaterialInfo = {"name": "unknown", "itu_type": None}
NO_MATERIAL: MaterialInfo = {"name": "no-material", "itu_type": None}


def _safe_str(value: Any, fallback: str | None = None) -> str | None:
    """Stringify dynamic Sionna objects without letting bad reprs abort metadata."""
    if value is None:
        return fallback
    try:
        return str(value)
    except (TypeError, ValueError):
        return fallback


def _display_material_name(material_name: str, itu_type: str | None) -> str:
    """Normalize material names for stable frame metadata and plots."""
    if itu_type:
        # Target materials use a unique Sionna RT suffix such as
        # "mat-itu_glass_drone"; scene materials keep their raw XML BSDF IDs.
        target_prefix = f"mat-itu_{itu_type}_"
        if material_name.startswith(target_prefix):
            return f"mat-itu_{itu_type}"
        return material_name
    if material_name.startswith("itu_"):
        return f"mat-{material_name}"
    return material_name


def _material_info_for_scene_object(scene_object: Any) -> MaterialInfo:
    """Return frame-facing material metadata for one Sionna scene object."""
    material = getattr(scene_object, "radio_material", None)
    if material is None:
        return dict(NO_MATERIAL)

    itu_type = _safe_str(getattr(material, "itu_type", None))
    material_name = _safe_str(getattr(material, "name", material), fallback="unknown")
    material_name = material_name or "unknown"
    return {
        "name": _display_material_name(material_name, itu_type),
        "itu_type": itu_type,
    }


def _object_id_key(value: Any) -> Any:
    """Convert NumPy scalar object IDs to dictionary-compatible Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _material_lookup(scene: Any, used_ids: np.ndarray) -> dict[Any, MaterialInfo]:
    """Build an object-id to material-info lookup for IDs present in path tensors."""
    used_id_set = {_object_id_key(obj_id) for obj_id in np.asarray(used_ids).flat}
    id_to_material: dict[Any, MaterialInfo] = {}
    for scene_object in scene.objects.values():
        object_id = getattr(scene_object, "object_id", None)
        object_key = _object_id_key(object_id)
        if object_key in used_id_set:
            id_to_material[object_key] = _material_info_for_scene_object(scene_object)
    return id_to_material


def _material_interaction_mask(interactions: np.ndarray, obj_ids: np.ndarray) -> np.ndarray:
    """Return real interactions whose scene object can be looked up."""
    return (interactions > 0) & (obj_ids != SIONNA_INVALID_OBJECT_ID)


def _material_info_for_object_id(
    id_to_material: dict[Any, MaterialInfo], obj_id: Any
) -> MaterialInfo:
    """Return material metadata for an object ID, falling back to ``unknown``."""
    material_info = id_to_material.get(_object_id_key(obj_id))
    if material_info is None:
        return dict(UNKNOWN_MATERIAL)
    return material_info


def materials_per_bounce(scene: Any, paths: Any) -> MaterialMapping:
    """Extract material information for each bounce point from Sionna RT paths.

    The returned mapping is keyed by ``(tx_idx, rx_idx)`` and then by integer
    path index. ``generator.io.frames.builder`` aligns those raw path indices
    with the sanitized path arrays that are written to frame outputs.

    Each returned path list is compacted to physical interaction order. It
    therefore has the same length and ordering as the packed bounce geometry,
    rather than retaining Sionna's unused ``max_depth`` slots.

    The expected Sionna tensor shape is ``(max_depth, num_rx, num_tx,
    num_paths)`` for both ``paths.objects`` and ``paths.interactions``.
    """
    obj_shape = None
    interaction_shape = None
    try:
        obj_ids = paths.objects.numpy()
        interactions = paths.interactions.numpy()
        obj_shape = getattr(obj_ids, "shape", None)
        interaction_shape = getattr(interactions, "shape", None)
        max_depth, num_rx, num_tx, num_paths = obj_ids.shape
        interaction_mask = interactions > 0
        # Invalid object IDs still occupy a physical bounce slot, but there is
        # no scene object to include in the lookup.
        material_mask = _material_interaction_mask(interactions, obj_ids)
        used_ids = np.unique(obj_ids[material_mask])
        id2mat = _material_lookup(scene, used_ids)

        material_mapping: MaterialMapping = {}

        if logger.isEnabledFor(logging.DEBUG):
            _sionna_12 = version_greater_equal(SIONNA_VERSION, "1.2")
            logger.debug("[STATS] MPC PATH TYPE ANALYSIS (pre-filter):")
            logger.debug("   Interactions shape: %s", interactions.shape)
            logger.debug("   Objects shape: %s", obj_ids.shape)

            if max_depth > 0:
                has_bounce = np.any(interaction_mask, axis=0)
                first_depth = np.argmax(interaction_mask, axis=0)
                first_interactions = np.take_along_axis(
                    interactions,
                    first_depth[np.newaxis, ...],
                    axis=0,
                )[0]
                first_bounce_types = first_interactions[has_bounce]
            else:
                first_bounce_types = np.empty(0, dtype=interactions.dtype)

            specular_paths = int(
                np.count_nonzero(first_bounce_types == SIONNA_INTERACTION_SPECULAR)
            )
            diffuse_paths = int(np.count_nonzero(first_bounce_types == SIONNA_INTERACTION_DIFFUSE))
            refraction_paths = int(
                np.count_nonzero(first_bounce_types == SIONNA_INTERACTION_REFRACTION)
            )
            diffraction_paths = int(
                np.count_nonzero(first_bounce_types == SIONNA_INTERACTION_DIFFRACTION)
            )
            known_non_los = specular_paths + diffuse_paths + refraction_paths + diffraction_paths
            los_paths = int(num_rx * num_tx * num_paths - known_non_los)

            logger.debug("   LoS paths: %d", los_paths)
            logger.debug("   Specular paths: %d", specular_paths)
            logger.debug("   Diffuse paths: %d", diffuse_paths)
            if _sionna_12:
                logger.debug("   Refraction paths: %d", refraction_paths)
                logger.debug("   Diffraction paths: %d", diffraction_paths)
            total = (
                los_paths + specular_paths + diffuse_paths + refraction_paths + diffraction_paths
            )
            logger.debug("   Total paths: %d", total)

        for rx_idx in range(num_rx):
            for tx_idx in range(num_tx):
                pair_key = (tx_idx, rx_idx)
                pair_materials: dict[int, PathMaterialList] = {
                    path_idx: [] for path_idx in range(num_paths)
                }
                pair_mask = interaction_mask[:, rx_idx, tx_idx, :]
                pair_obj_ids = obj_ids[:, rx_idx, tx_idx, :]
                depth_indices, path_indices = np.nonzero(pair_mask)
                for depth_idx, path_idx in zip(depth_indices, path_indices):
                    obj_id = pair_obj_ids[depth_idx, path_idx]
                    pair_materials[int(path_idx)].append(
                        _material_info_for_object_id(id2mat, obj_id)
                    )
                material_mapping[pair_key] = pair_materials

        logger.debug(
            "Material mapping extracted: %d TX-RX pairs, %d total paths",
            len(material_mapping),
            sum(len(paths) for paths in material_mapping.values()),
        )
        return material_mapping

    except Exception:  # noqa: BLE001 - material diagnostics must not stop generation.
        logger.exception(
            "Error extracting material mapping (objects_shape=%s, interactions_shape=%s)",
            obj_shape,
            interaction_shape,
        )
        return {}
