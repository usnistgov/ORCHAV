"""Shared loader for optional sensing payloads stored in packed frame sets.

``load_frames_sensing()`` reads only sensing, device, and target projections
through the shared provider contract. ``load_saved_results()`` reads only the
sensing projection needed to reconstruct postprocessor results. Both entry
points require the authoritative packed-v2 ``frames_manifest.json`` and never
inspect physical HDF5 datasets directly.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from shared.extensions.sensing import (
    assert_sensing_contract,
    assert_supported_sensing_schema,
)
from shared.frames.contracts import FrameComponent, FrameReadRequest
from shared.frames.manifest import (
    FRAMES_MANIFEST_FILENAME,
    FrameSetManifest,
    load_frame_manifest,
)
from shared.frames.packed import FrameProjection
from shared.frames.providers import Hdf5Provider
from shared.sensing.angle import select_range_doppler_angle_product
from shared.sensing.evaluation_truth import extract_recorded_evaluation_truth
from shared.sensing.products import select_product_by_contract

_SENSING_WITH_CONTEXT_REQUEST = FrameReadRequest(
    components=frozenset(
        {
            FrameComponent.DEVICES,
            FrameComponent.PROVENANCE,
            FrameComponent.TARGETS,
            FrameComponent.SENSING,
        }
    )
)
_SENSING_ONLY_REQUEST = FrameReadRequest(
    components=frozenset({FrameComponent.PROVENANCE, FrameComponent.SENSING})
)


def _resolve_frames_dir(source: Path) -> Path:
    """Resolve either a frame directory or a scenario/output root."""

    candidate = Path(source)
    if (candidate / FRAMES_MANIFEST_FILENAME).is_file():
        return candidate
    nested = candidate / "frames"
    if (nested / FRAMES_MANIFEST_FILENAME).is_file():
        return nested
    raise FileNotFoundError(f"{FRAMES_MANIFEST_FILENAME} not found in {candidate} or {nested}")


def _open_provider(
    source: Path,
) -> tuple[Path, FrameSetManifest, Hdf5Provider]:
    """Open a manifest-validated packed provider for ``source``."""

    frames_dir = _resolve_frames_dir(source)
    manifest = load_frame_manifest(frames_dir)
    provider = Hdf5Provider(
        frames_dir.parent,
        frames_subdir=frames_dir.name,
    )
    return frames_dir, manifest, provider


def _copy_target_metadata(
    metadata: tuple[Mapping[str, Any], ...] | None,
) -> list[dict[str, Any]]:
    """Preserve the legacy analysis-friendly target metadata representation."""

    copied = deepcopy(list(metadata or ()))
    for target in copied:
        for key, value in tuple(target.items()):
            if isinstance(value, list) and value and isinstance(value[0], (int, float)):
                target[key] = np.asarray(value, dtype=np.float64)
    return copied


def _require_resolved_config(sensing: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact persisted schema-v4 runtime configuration."""

    config = sensing.get("config_resolved")
    if not isinstance(config, Mapping):
        raise ValueError("Native sensing-v4 payload requires exact config_resolved")
    return deepcopy(dict(config))


def _acquisition_sample_times(sensing: Mapping[str, Any]) -> np.ndarray:
    """Return exact persisted acquisition sample epochs in authored order."""

    temporal = sensing.get("temporal")
    acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
    samples = acquisition.get("samples") if isinstance(acquisition, Mapping) else None
    if not isinstance(samples, list):
        return np.empty(0, dtype=np.float64)
    times = np.asarray(
        [
            sample.get("sample_time_s") if isinstance(sample, Mapping) else None
            for sample in samples
        ],
        dtype=np.float64,
    )
    if times.ndim != 1 or not np.all(np.isfinite(times)):
        raise ValueError("Native sensing-v4 acquisition sample times must be finite")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("Native sensing-v4 acquisition sample times must increase exactly")
    return times


def _select_primary_range_doppler(
    sensing: Mapping[str, Any],
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]] | None:
    """Select the exact primary single-pair range-Doppler product when present."""

    products = sensing.get("products")
    candidates = (
        [
            descriptor
            for descriptor in products.values()
            if isinstance(descriptor, Mapping)
            and descriptor.get("kind") == "range_doppler"
            and descriptor.get("variant") == "primary"
        ]
        if isinstance(products, Mapping)
        else []
    )
    if not candidates:
        return None
    try:
        data_key, values, coordinates, descriptor = select_product_by_contract(
            sensing,
            kind="range_doppler",
            variant="primary",
            axis_names=("pair", "doppler_frequency", "range"),
            representations=("power_linear", "complex_voltage"),
        )
    except KeyError as exc:
        raise ValueError(
            "Persisted primary range_doppler does not satisfy the exact "
            "pair/Doppler-Hz/range-m contract"
        ) from exc
    if values.ndim != 3 or values.shape[0] != 1:
        raise ValueError(
            f"Typed sensing product {data_key!r} requires shape [1, Doppler, range]; "
            f"received {values.shape}"
        )
    pair_coordinates = coordinates[0]
    if not isinstance(pair_coordinates, list) or len(pair_coordinates) != 1:
        raise ValueError(f"Typed sensing product {data_key!r} requires one exact pair")
    doppler_axis = np.asarray(coordinates[1], dtype=np.float64)
    range_axis = np.asarray(coordinates[2], dtype=np.float64)
    if (
        doppler_axis.shape != (values.shape[1],)
        or range_axis.shape != (values.shape[2],)
        or not np.all(np.isfinite(doppler_axis))
        or not np.all(np.isfinite(range_axis))
    ):
        raise ValueError(f"Typed sensing product {data_key!r} has malformed exact coordinates")
    if (doppler_axis.size > 1 and not np.all(np.diff(doppler_axis) > 0.0)) or (
        range_axis.size > 1 and not np.all(np.diff(range_axis) > 0.0)
    ):
        raise ValueError(f"Typed sensing product {data_key!r} coordinates must increase")
    return data_key, values, doppler_axis, range_axis, descriptor


def _sensing_entry(
    projection: FrameProjection,
    *,
    path: Path,
    source_generation_id: str,
    source_frame_set_id: str,
) -> dict[str, Any] | None:
    """Convert one packed projection to the sensing-analysis dictionary."""

    frame = projection.frame
    sensing = dict(frame.sensing or {})
    if sensing:
        assert_supported_sensing_schema(sensing)
        assert_sensing_contract(
            {"sensing": sensing},
            require_schema_version=True,
            require_described_products=True,
        )
    products = {
        key: np.array(value, copy=True)
        for key, value in sensing.items()
        if isinstance(value, np.ndarray)
    }
    temporal_value = sensing.get("temporal")
    has_temporal_acquisition = isinstance(temporal_value, Mapping) and isinstance(
        temporal_value.get("acquisition"), Mapping
    )
    # Static config can be present for a row with no sensing payload. Metadata-only
    # acquisitions are nevertheless first-class records (for example a retained
    # incomplete CPI with no normal coherent result).
    if not products and not has_temporal_acquisition:
        return None

    metadata = {
        key: deepcopy(value) for key, value in sensing.items() if not isinstance(value, np.ndarray)
    }
    typed_sensing = {**metadata, **products}
    config = _require_resolved_config(typed_sensing)
    sample_times_s = _acquisition_sample_times(typed_sensing)
    primary_rd = _select_primary_range_doppler(typed_sensing)
    angular_product = select_range_doppler_angle_product(typed_sensing)
    temporal = metadata.get("temporal")
    if isinstance(temporal, dict):
        acquisition = temporal.get("acquisition")
        if isinstance(acquisition, dict):
            for sample in acquisition.get("samples", ()):
                if not isinstance(sample, dict):
                    continue
                source = sample.get("source")
                if isinstance(source, dict):
                    source.setdefault("source_frame_set_id", source_frame_set_id)
    entry: dict[str, Any] = {
        "_path": str(path),
        "_frame_index": int(frame.frame_index),
        "_frame_timestamp_s": frame.timestamp_s,
        "_source_generation_id": source_generation_id,
        "_source_frame_set_id": source_frame_set_id,
        "_sensing_json": metadata,
        # Full logical sensing payload used by schema-v4 consumers.  Numeric
        # arrays are shared with the analysis-friendly projections below, so
        # this view does not add another in-memory copy.
        "_typed_sensing": typed_sensing,
        "_angular_product": angular_product,
        "config": config,
        "sample_times_s": sample_times_s,
        **products,
    }
    if primary_rd is not None:
        entry["range_axis_m"] = np.array(primary_rd[3], copy=True)

    ground_truth = extract_recorded_evaluation_truth(metadata)
    if ground_truth is not None:
        entry["_gt"] = ground_truth

    targets_metadata = _copy_target_metadata(frame.targets_metadata)
    target_positions = np.array(frame.target_positions_m, copy=True)
    entry.update(
        num_targets=len(target_positions),
        target_pos=target_positions,
        targets_metadata=targets_metadata,
        tx_positions=np.array(frame.tx_positions, copy=True),
        rx_positions=np.array(frame.rx_positions, copy=True),
        tx_orientations=np.array(frame.tx_orientations, copy=True),
        rx_orientations=np.array(frame.rx_orientations, copy=True),
    )
    return entry


def iter_frames_sensing(
    frames_dir: Path,
    *,
    frame_ids: Sequence[int] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Yield sensing analysis inputs from a packed HDF5 v2 frame set.

    ``frames_dir`` may point at the directory containing
    ``frames_manifest.json`` or at its scenario/output parent. Rows with a
    stored numeric sensing product or retained acquisition lineage are yielded.
    The provider is closed after normal exhaustion, an exception, or an explicit
    :meth:`~collections.abc.Generator.close` on the returned generator. Optional
    ``frame_ids`` selects explicit manifest rows before any numeric reads. Callers
    that stop early must close the iterator (a ``contextlib.closing`` block is
    one convenient option).
    """

    resolved_dir, manifest, provider = _open_provider(Path(frames_dir))
    try:
        locations = manifest.frame_locations()
        # Request one projection at a time. The packed provider's batch path
        # intentionally materializes a complete bounded HDF5 chunk before it
        # yields, which is ideal for path-statistics scans but would defeat the
        # retention guarantee of this streaming analysis API.
        selected_ids = manifest.frame_ids if frame_ids is None else tuple(frame_ids)
        if len(set(selected_ids)) != len(selected_ids) or any(
            step not in locations for step in selected_ids
        ):
            raise ValueError("Selected sensing frame IDs must be unique manifest rows")
        for frame_id in selected_ids:
            projection = provider.load_frame_projection(
                frame_id,
                _SENSING_WITH_CONTEXT_REQUEST,
            )
            chunk, _row = locations[int(projection.frame.frame_index)]
            entry = _sensing_entry(
                projection,
                path=resolved_dir / chunk.file,
                source_generation_id=manifest.generation_id,
                source_frame_set_id=manifest.frame_set_id,
            )
            if entry is not None:
                yield entry
    finally:
        provider.close()


def load_frames_sensing(
    frames_dir: Path,
) -> list[dict[str, Any]]:
    """Load sensing analysis inputs using the existing eager batch path.

    The explicit iterator above trades batch throughput for bounded retention.
    Keep eager callers on the provider's chunked projection path so opting into
    streaming does not quietly slow the default postprocessor workflow.
    """

    resolved_dir, manifest, provider = _open_provider(Path(frames_dir))
    locations = manifest.frame_locations()
    try:
        results: list[dict[str, Any]] = []
        projections = provider.iter_frame_projections(
            manifest.frame_ids,
            _SENSING_WITH_CONTEXT_REQUEST,
        )
        for projection in projections:
            chunk, _row = locations[int(projection.frame.frame_index)]
            entry = _sensing_entry(
                projection,
                path=resolved_dir / chunk.file,
                source_generation_id=manifest.generation_id,
                source_frame_set_id=manifest.frame_set_id,
            )
            if entry is not None:
                results.append(entry)
        return results
    finally:
        provider.close()


def load_saved_results(output_dir: Path) -> list[dict[str, Any]]:
    """Load typed postprocessor results through a sensing-only projection.

    Schema-v4 consumers use the producer-authored representation and exact
    coordinate arrays.  A legacy ``range_doppler_map`` is never interpreted by
    dtype or reconstructed from configuration here.
    """

    output_dir = Path(output_dir)
    if (
        not (output_dir / "frames").exists()
        and not (output_dir / FRAMES_MANIFEST_FILENAME).exists()
    ):
        return []
    _frames_dir, manifest, provider = _open_provider(output_dir)

    try:
        saved_results: list[dict[str, Any]] = []
        projections = provider.iter_frame_projections(
            manifest.frame_ids,
            _SENSING_ONLY_REQUEST,
        )
        for projection in projections:
            frame_index = int(projection.frame.frame_index)
            sensing = dict(projection.frame.sensing or {})
            if sensing:
                assert_supported_sensing_schema(sensing)
                assert_sensing_contract(
                    {"sensing": sensing},
                    require_schema_version=True,
                    require_described_products=True,
                )
            selected_rd = _select_primary_range_doppler(sensing)
            if selected_rd is None:
                continue

            typed_key, values, doppler_axis, range_axis, descriptor = selected_rd
            representation = descriptor.get("representation")
            if representation == "power_linear":
                rd_map = np.asarray(values[0])
            elif representation == "complex_voltage":
                rd_map = np.abs(values[0]) ** 2
            else:  # The exact selector should make this unreachable.
                raise ValueError(
                    f"Typed sensing product {typed_key!r} has unsupported representation "
                    f"{representation!r}"
                )
            config = _require_resolved_config(sensing)
            range_mode = config.get("range_mode")
            if not isinstance(range_mode, str) or not range_mode:
                raise ValueError("Typed sensing result requires config_resolved.range_mode")
            angular_product = select_range_doppler_angle_product(sensing)
            saved_results.append(
                {
                    "_frame_index": frame_index,
                    "_frame_timestamp_s": projection.frame.timestamp_s,
                    "_source_generation_id": manifest.generation_id,
                    "_source_frame_set_id": manifest.frame_set_id,
                    "_typed_sensing": sensing,
                    "_angular_product": angular_product,
                    "_product_data_key": typed_key,
                    "_product_id": descriptor.get("product_id"),
                    "temporal": deepcopy(sensing.get("temporal")),
                    "sample_times_s": _acquisition_sample_times(sensing),
                    "rd_map": np.array(rd_map, copy=True),
                    "range_axis": np.array(range_axis, copy=True),
                    "range_axis_m": np.array(range_axis, copy=True),
                    "doppler_axis": np.array(doppler_axis, copy=True),
                    "doppler_axis_hz": np.array(doppler_axis, copy=True),
                    "detections": deepcopy(sensing.get("detections") or []),
                    "config": config,
                    "_range_mode": range_mode,
                    "detection_velocity_axis_label": sensing.get("detection_velocity_axis_label"),
                    "detection_velocity_is_approximate": sensing.get(
                        "detection_velocity_is_approximate"
                    ),
                }
            )
        return saved_results
    finally:
        provider.close()


__all__ = ["iter_frames_sensing", "load_frames_sensing", "load_saved_results"]
