"""Normalize convenient per-pair path data into ``StandardMPCFrame``.

The direct ``StandardMPCFrame`` constructor is the low-copy entry point for
producers that already own compact arrays. This module provides the single
higher-level alternative for importers whose source naturally uses one padded
array per TX/RX pair or one variable-length array per path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import PATH_METRIC_ORDER, PATH_METRIC_VALIDITY_BITS, PathMetric
from .types import PATH_METRIC_ARRAY_FIELDS, StandardMPCFrame

_EMPTY_MATERIAL = ("", "")


@dataclass(frozen=True, slots=True)
class _PairGeometry:
    vertices: np.ndarray | tuple[np.ndarray, ...]
    interactions: np.ndarray | tuple[np.ndarray, ...]
    lengths: np.ndarray
    padded: bool

    @property
    def path_count(self) -> int:
        return int(len(self.lengths))

    @property
    def bounce_count(self) -> int:
        return int(np.sum(self.lengths, dtype=np.int64))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _sequence_length(values: Any, name: str) -> int:
    try:
        return len(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc


def _pair_value(values: Sequence[Any] | None, pair_index: int, name: str) -> Any | None:
    if values is None:
        return None
    try:
        return values[pair_index]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"{name} must contain one entry per TX/RX pair") from exc


def _canonical_pairs(values: Any) -> np.ndarray:
    raw = np.asarray(values)
    if raw.size == 0 and raw.shape == (0,):
        return np.empty((0, 2), dtype=np.int32)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(f"tx_rx_pairs must have shape (N, 2), got {raw.shape}")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("tx_rx_pairs must use an integer dtype")
    if raw.size and (np.any(raw < 0) or np.any(raw > np.iinfo(np.int32).max)):
        raise ValueError("tx_rx_pairs values must fit the non-negative int32 range")
    return raw.astype(np.int32, copy=False)


def _canonical_xyz(values: Any, name: str) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric coordinates") from exc
    if raw.size == 0 and raw.shape == (0,):
        raw = raw.reshape((0, 3))
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {raw.shape}")
    if np.any(~np.isfinite(raw)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return raw


def _canonical_orientations(values: Any | None, count: int, name: str) -> np.ndarray:
    if values is None:
        return np.zeros((count, 3), dtype=np.float64)
    orientations = _canonical_xyz(values, name)
    if len(orientations) != count:
        raise ValueError(f"{name} count must match its device positions")
    return orientations


def _canonical_names(values: Sequence[Any] | None, count: int, prefix: str) -> tuple[str, ...]:
    if values is None:
        return tuple(f"{prefix}_{index}" for index in range(count))
    names = tuple(_as_text(value) for value in values)
    if len(names) != count:
        raise ValueError(f"{prefix}_names count must match {prefix}_positions")
    return names


def _canonical_lengths(values: Any, path_count: int, padded_depth: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (path_count,):
        raise ValueError(f"{name} must have shape {(path_count,)}, got {raw.shape}")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must use an integer dtype")
    if np.any(raw < 0) or np.any(raw > padded_depth):
        raise ValueError(f"{name} contains a value outside [0, {padded_depth}]")
    return raw.astype(np.int64, copy=False)


def _pair_geometry(
    vertices_value: Any,
    interactions_value: Any,
    lengths_value: Any | None,
    pair_index: int,
) -> _PairGeometry:
    vertices = vertices_value if isinstance(vertices_value, np.ndarray) else None
    if vertices is not None and vertices.ndim == 3:
        if vertices.shape[2] != 3:
            raise ValueError(
                f"vertices_by_pair[{pair_index}] must have shape (P, B, 3), "
                f"got {vertices.shape}"
            )
        if not np.issubdtype(vertices.dtype, np.number):
            raise ValueError(f"vertices_by_pair[{pair_index}] must be numeric")
        interactions = np.asarray(interactions_value)
        if interactions.shape != vertices.shape[:2]:
            raise ValueError(
                f"interactions_by_pair[{pair_index}] must have shape "
                f"{vertices.shape[:2]}, got {interactions.shape}"
            )
        if not np.issubdtype(interactions.dtype, np.integer):
            raise ValueError(f"interactions_by_pair[{pair_index}] must use an integer dtype")
        path_count, padded_depth = vertices.shape[:2]
        lengths = (
            np.full((path_count,), padded_depth, dtype=np.int64)
            if lengths_value is None
            else _canonical_lengths(
                lengths_value,
                path_count,
                padded_depth,
                f"path_lengths_by_pair[{pair_index}]",
            )
        )
        return _PairGeometry(vertices, interactions, lengths, padded=True)

    try:
        vertex_paths = tuple(np.asarray(path) for path in vertices_value)
        interaction_paths = tuple(np.asarray(path) for path in interactions_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"vertices_by_pair[{pair_index}] and interactions_by_pair[{pair_index}] "
            "must be padded arrays or per-path sequences"
        ) from exc
    if len(interaction_paths) != len(vertex_paths):
        raise ValueError(f"interactions_by_pair[{pair_index}] must contain one entry per path")

    inferred_lengths = np.empty((len(vertex_paths),), dtype=np.int64)
    for path_index, (path_vertices, path_interactions) in enumerate(
        zip(vertex_paths, interaction_paths, strict=True)
    ):
        if path_vertices.ndim != 2 or path_vertices.shape[1] != 3:
            raise ValueError(
                f"vertices_by_pair[{pair_index}][{path_index}] must have shape "
                f"(B, 3), got {path_vertices.shape}"
            )
        if not np.issubdtype(path_vertices.dtype, np.number):
            raise ValueError(f"vertices_by_pair[{pair_index}][{path_index}] must be numeric")
        if path_interactions.shape != (len(path_vertices),):
            raise ValueError(
                f"interactions_by_pair[{pair_index}][{path_index}] must have shape "
                f"{(len(path_vertices),)}, got {path_interactions.shape}"
            )
        if not np.issubdtype(path_interactions.dtype, np.integer):
            raise ValueError(
                f"interactions_by_pair[{pair_index}][{path_index}] must use an integer dtype"
            )
        inferred_lengths[path_index] = len(path_vertices)

    if lengths_value is not None:
        supplied = _canonical_lengths(
            lengths_value,
            len(vertex_paths),
            int(np.max(inferred_lengths, initial=0)),
            f"path_lengths_by_pair[{pair_index}]",
        )
        if not np.array_equal(supplied, inferred_lengths):
            raise ValueError(f"path_lengths_by_pair[{pair_index}] disagrees with the ragged paths")
    return _PairGeometry(vertex_paths, interaction_paths, inferred_lengths, padded=False)


def _write_physical_geometry(
    pair: _PairGeometry,
    xyz: np.ndarray,
    interactions: np.ndarray,
    start: int,
    pair_index: int,
) -> None:
    cursor = start
    if pair.padded:
        assert isinstance(pair.vertices, np.ndarray)
        assert isinstance(pair.interactions, np.ndarray)
        depth = pair.vertices.shape[1]
        mask = np.arange(depth, dtype=np.int64)[None, :] < pair.lengths[:, None]
        with np.errstate(over="ignore", invalid="ignore"):
            physical_xyz = np.asarray(pair.vertices[mask], dtype=np.float32)
        physical_interactions = pair.interactions[mask]
        end = cursor + len(physical_xyz)
        xyz[cursor:end] = physical_xyz
        _write_interactions(
            physical_interactions,
            interactions[cursor:end],
            f"interactions_by_pair[{pair_index}]",
        )
        if np.any(~np.isfinite(xyz[cursor:end])):
            raise ValueError(f"vertices_by_pair[{pair_index}] has non-finite physical bounces")
        return

    assert isinstance(pair.vertices, tuple)
    assert isinstance(pair.interactions, tuple)
    for path_index, (path_vertices, path_interactions) in enumerate(
        zip(pair.vertices, pair.interactions, strict=True)
    ):
        end = cursor + len(path_vertices)
        with np.errstate(over="ignore", invalid="ignore"):
            xyz[cursor:end] = np.asarray(path_vertices, dtype=np.float32)
        _write_interactions(
            path_interactions,
            interactions[cursor:end],
            f"interactions_by_pair[{pair_index}][{path_index}]",
        )
        if np.any(~np.isfinite(xyz[cursor:end])):
            raise ValueError(
                f"vertices_by_pair[{pair_index}][{path_index}] has non-finite coordinates"
            )
        cursor = end


def _write_interactions(source: np.ndarray, destination: np.ndarray, name: str) -> None:
    if source.size and (np.any(source <= 0) or np.any(source > np.iinfo(np.uint8).max)):
        raise ValueError(f"{name} physical interaction codes must be in [1, 255]")
    destination[:] = source


def _physical_text_values(
    values_by_pair: Sequence[Any] | None,
    pair: _PairGeometry,
    pair_index: int,
    name: str,
) -> np.ndarray | None:
    value = _pair_value(values_by_pair, pair_index, name)
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.size == 0:
        return None
    if pair.padded:
        assert isinstance(pair.vertices, np.ndarray)
        if raw.shape != pair.vertices.shape[:2]:
            raise ValueError(
                f"{name}[{pair_index}] must have shape {pair.vertices.shape[:2]}, "
                f"got {raw.shape}"
            )
        mask = np.arange(raw.shape[1], dtype=np.int64)[None, :] < pair.lengths[:, None]
        physical = raw[mask]
    else:
        try:
            paths = tuple(np.asarray(path) for path in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{pair_index}] must contain one entry per path") from exc
        if len(paths) != pair.path_count:
            raise ValueError(f"{name}[{pair_index}] must contain one entry per path")
        parts: list[np.ndarray] = []
        for path_index, (path, length) in enumerate(zip(paths, pair.lengths, strict=True)):
            if path.shape != (int(length),):
                raise ValueError(
                    f"{name}[{pair_index}][{path_index}] must have shape "
                    f"{(int(length),)}, got {path.shape}"
                )
            parts.append(path)
        physical = (
            np.concatenate(parts) if parts and pair.bounce_count else np.empty((0,), dtype=object)
        )
    return np.fromiter((_as_text(item) for item in physical), dtype=object, count=len(physical))


def _material_ids_for_pair(
    names: np.ndarray | None,
    itu_types: np.ndarray | None,
    bounce_count: int,
    catalog: list[tuple[str, str]],
    catalog_index: dict[tuple[str, str], int],
) -> np.ndarray:
    if names is None and itu_types is None:
        return np.zeros((bounce_count,), dtype=np.uint16)
    if names is None:
        names = np.full((bounce_count,), "", dtype=object)
    if itu_types is None:
        itu_types = np.full((bounce_count,), "", dtype=object)
    if len(names) != bounce_count or len(itu_types) != bounce_count:
        raise ValueError("Material axes must align with physical bounces")
    if bounce_count == 0:
        return np.empty((0,), dtype=np.uint16)

    unique_names, name_inverse = np.unique(names, return_inverse=True)
    unique_itu, itu_inverse = np.unique(itu_types, return_inverse=True)
    pair_codes = name_inverse.astype(np.int64, copy=False) * max(
        1, len(unique_itu)
    ) + itu_inverse.astype(np.int64, copy=False)
    _, first_indices, inverse = np.unique(
        pair_codes,
        return_index=True,
        return_inverse=True,
    )
    ids = np.empty((len(first_indices),), dtype=np.uint16)
    for unique_index in np.argsort(first_indices, kind="stable"):
        first_index = int(first_indices[unique_index])
        material = (str(names[first_index]), str(itu_types[first_index]))
        material_id = catalog_index.get(material)
        if material_id is None:
            if len(catalog) > np.iinfo(np.uint16).max:
                raise ValueError("Material catalog exceeds the uint16 material ID range")
            material_id = len(catalog)
            catalog.append(material)
            catalog_index[material] = material_id
        ids[unique_index] = material_id
    return ids[inverse]


def _normalize_metric_sources(
    metrics_by_pair: Mapping[PathMetric | str, Sequence[Any]] | None,
    pair_count: int,
) -> dict[PathMetric, Sequence[Any]]:
    normalized: dict[PathMetric, Sequence[Any]] = {}
    array_field_metrics = {name: metric for metric, name in PATH_METRIC_ARRAY_FIELDS.items()}
    for raw_metric, values in (metrics_by_pair or {}).items():
        try:
            field_metric = array_field_metrics.get(str(raw_metric))
            metric = field_metric if field_metric is not None else PathMetric(raw_metric)
        except ValueError as exc:
            raise ValueError(f"Unknown path metric {raw_metric!r}") from exc
        if metric in normalized:
            raise ValueError(f"Path metric {metric.value!r} was supplied more than once")
        if _sequence_length(values, f"metrics_by_pair[{metric.value!r}]") != pair_count:
            raise ValueError(
                f"metrics_by_pair[{metric.value!r}] must contain one entry per TX/RX pair"
            )
        normalized[metric] = values
    return normalized


def _normalize_targets(
    target_positions_m: Any | None,
    targets_metadata: Sequence[Mapping[str, Any]] | None,
) -> tuple[np.ndarray, tuple[Mapping[str, Any], ...]]:
    positions = (
        np.empty((0, 3), dtype=np.float64)
        if target_positions_m is None
        else np.asarray(target_positions_m, dtype=np.float64)
    )
    if positions.shape == (3,):
        positions = positions.reshape((1, 3))
    elif positions.size == 0:
        positions = np.empty((0, 3), dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"target_positions_m must have shape (N, 3), got {positions.shape}")
    if np.any(~np.isfinite(positions)):
        raise ValueError("target_positions_m must contain only finite coordinates")

    if targets_metadata is None:
        metadata: tuple[Mapping[str, Any], ...] = tuple({} for _ in positions)
    else:
        if len(targets_metadata) != len(positions):
            raise ValueError("targets_metadata count must match target_positions_m")
        if any(not isinstance(item, Mapping) for item in targets_metadata):
            raise ValueError("Each targets_metadata item must be a mapping")
        metadata = tuple(dict(item) for item in targets_metadata)
    return positions, metadata


def standard_mpc_frame_from_pair_data(
    *,
    frame_index: int,
    tx_rx_pairs: Any,
    tx_positions: Any,
    rx_positions: Any,
    vertices_by_pair: Sequence[Any],
    interactions_by_pair: Sequence[Any],
    path_lengths_by_pair: Sequence[Any] | None = None,
    tx_orientations: Any | None = None,
    rx_orientations: Any | None = None,
    tx_names: Sequence[Any] | None = None,
    rx_names: Sequence[Any] | None = None,
    material_names_by_pair: Sequence[Any] | None = None,
    material_itu_types_by_pair: Sequence[Any] | None = None,
    metrics_by_pair: Mapping[PathMetric | str, Sequence[Any]] | None = None,
    target_positions_m: Any | None = None,
    targets_metadata: Sequence[Mapping[str, Any]] | None = None,
    timestamp_s: float | None = None,
    sensing: Mapping[str, Any] | None = None,
    beamforming: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    recomputed_from_stored_positions: bool = False,
) -> StandardMPCFrame:
    """Build one compact frame from per-pair padded or ragged path arrays.

    A padded pair uses ``(paths, depth, 3)`` vertices, ``(paths, depth)``
    interactions, and an explicit path-length vector. A ragged pair uses one
    ``(bounces, 3)`` vertex array and one ``(bounces,)`` interaction array per
    path; lengths are inferred and any supplied lengths must agree. Rectangular
    arrays without lengths are interpreted as paths whose full depth is
    physical.

    Vertices contain physical interaction points in meters, not TX/RX
    endpoints; the device positions supply those endpoints. A direct path has
    length zero. Interaction codes and optional material values align with the
    physical vertices, and interaction code zero is not accepted as a bounce.
    Device orientations are ``(yaw, pitch, roll)`` in radians.

    Optional material axes follow the same padded/ragged organization. Metric
    values use one one-dimensional vector per pair; absent and non-finite
    entries become NaN with their validity bit clear.
    """

    pairs = _canonical_pairs(tx_rx_pairs)
    pair_count = len(pairs)
    for name, values in (
        ("vertices_by_pair", vertices_by_pair),
        ("interactions_by_pair", interactions_by_pair),
    ):
        if _sequence_length(values, name) != pair_count:
            raise ValueError(f"{name} must contain one entry per TX/RX pair")
    for name, values in (
        ("path_lengths_by_pair", path_lengths_by_pair),
        ("material_names_by_pair", material_names_by_pair),
        ("material_itu_types_by_pair", material_itu_types_by_pair),
    ):
        if values is not None and _sequence_length(values, name) != pair_count:
            raise ValueError(f"{name} must contain one entry per TX/RX pair")

    pair_geometry = tuple(
        _pair_geometry(
            vertices_by_pair[pair_index],
            interactions_by_pair[pair_index],
            _pair_value(path_lengths_by_pair, pair_index, "path_lengths_by_pair"),
            pair_index,
        )
        for pair_index in range(pair_count)
    )

    pair_path_offsets = np.zeros((pair_count + 1,), dtype=np.int64)
    for pair_index, pair in enumerate(pair_geometry):
        pair_path_offsets[pair_index + 1] = pair_path_offsets[pair_index] + pair.path_count
    total_paths = int(pair_path_offsets[-1])
    path_lengths = np.empty((total_paths,), dtype=np.int64)
    for pair_index, pair in enumerate(pair_geometry):
        start = int(pair_path_offsets[pair_index])
        end = int(pair_path_offsets[pair_index + 1])
        path_lengths[start:end] = pair.lengths
    bounce_offsets = np.zeros((total_paths + 1,), dtype=np.int64)
    if total_paths:
        np.cumsum(path_lengths, dtype=np.int64, out=bounce_offsets[1:])
    total_bounces = int(bounce_offsets[-1])

    bounce_xyz_m = np.empty((total_bounces, 3), dtype=np.float32)
    interactions = np.empty((total_bounces,), dtype=np.uint8)
    material_ids = np.empty((total_bounces,), dtype=np.uint16)
    catalog = [_EMPTY_MATERIAL]
    catalog_index = {_EMPTY_MATERIAL: 0}
    for pair_index, pair in enumerate(pair_geometry):
        path_start = int(pair_path_offsets[pair_index])
        path_end = int(pair_path_offsets[pair_index + 1])
        bounce_start = int(bounce_offsets[path_start])
        bounce_end = int(bounce_offsets[path_end])
        _write_physical_geometry(
            pair,
            bounce_xyz_m,
            interactions,
            bounce_start,
            pair_index,
        )
        names = _physical_text_values(
            material_names_by_pair,
            pair,
            pair_index,
            "material_names_by_pair",
        )
        itu_types = _physical_text_values(
            material_itu_types_by_pair,
            pair,
            pair_index,
            "material_itu_types_by_pair",
        )
        material_ids[bounce_start:bounce_end] = _material_ids_for_pair(
            names,
            itu_types,
            pair.bounce_count,
            catalog,
            catalog_index,
        )

    metric_sources = _normalize_metric_sources(metrics_by_pair, pair_count)
    metric_arrays = {
        metric: np.full((total_paths,), np.nan, dtype=np.float32) for metric in PATH_METRIC_ORDER
    }
    metric_valid_bits = np.zeros((total_paths,), dtype=np.uint8)
    for metric, values_by_pair in metric_sources.items():
        destination = metric_arrays[metric]
        for pair_index, pair in enumerate(pair_geometry):
            start = int(pair_path_offsets[pair_index])
            end = int(pair_path_offsets[pair_index + 1])
            raw = np.asarray(values_by_pair[pair_index])
            if raw.size == 0:
                continue
            if raw.shape != (pair.path_count,):
                raise ValueError(
                    f"metrics_by_pair[{metric.value!r}][{pair_index}] must have shape "
                    f"{(pair.path_count,)}, got {raw.shape}"
                )
            try:
                numeric = raw.astype(np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"metrics_by_pair[{metric.value!r}][{pair_index}] must be numeric"
                ) from exc
            finite = np.isfinite(numeric)
            with np.errstate(over="ignore", invalid="ignore"):
                destination[start:end][finite] = numeric[finite].astype(
                    np.float32,
                    copy=False,
                )
            valid = finite & np.isfinite(destination[start:end])
            destination[start:end][~valid] = np.nan
            metric_valid_bits[start:end][valid] |= np.uint8(PATH_METRIC_VALIDITY_BITS[metric])

    canonical_tx_positions = _canonical_xyz(tx_positions, "tx_positions")
    canonical_rx_positions = _canonical_xyz(rx_positions, "rx_positions")
    targets, target_metadata = _normalize_targets(target_positions_m, targets_metadata)
    metric_kwargs = {
        PATH_METRIC_ARRAY_FIELDS[metric]: metric_arrays[metric] for metric in PATH_METRIC_ORDER
    }
    return StandardMPCFrame(
        frame_index=frame_index,
        timestamp_s=None if timestamp_s is None else float(timestamp_s),
        tx_rx_pairs=pairs,
        pair_path_offsets=pair_path_offsets,
        bounce_offsets=bounce_offsets,
        tx_positions=canonical_tx_positions,
        rx_positions=canonical_rx_positions,
        tx_orientations=_canonical_orientations(
            tx_orientations,
            len(canonical_tx_positions),
            "tx_orientations",
        ),
        rx_orientations=_canonical_orientations(
            rx_orientations,
            len(canonical_rx_positions),
            "rx_orientations",
        ),
        tx_names=_canonical_names(tx_names, len(canonical_tx_positions), "tx"),
        rx_names=_canonical_names(rx_names, len(canonical_rx_positions), "rx"),
        bounce_xyz_m=bounce_xyz_m,
        interactions=interactions,
        material_ids=material_ids,
        material_names=tuple(name for name, _ in catalog),
        material_itu_types=tuple(itu for _, itu in catalog),
        metric_valid_bits=metric_valid_bits,
        target_positions_m=targets,
        targets_metadata=target_metadata,
        sensing=sensing,
        beamforming=beamforming,
        provenance=provenance,
        recomputed_from_stored_positions=recomputed_from_stored_positions,
        **metric_kwargs,
    )


__all__ = ["standard_mpc_frame_from_pair_data"]
