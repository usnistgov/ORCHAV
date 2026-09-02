"""Command-line inspection for generated HDF5 outputs.

``orchav-inspect`` is a lightweight reader for two shared contracts:
``StandardMPCFrame`` chunks under a frames directory and compact coverage HDF5
files. The public entry point is ``run()``; it returns a process-style exit code
so tests and wrapper scripts can call it without spawning a subprocess.

Frame inspection goes through ``shared.frames.providers.Hdf5Provider``.
Coverage inspection reports both materialized datasets and recipe-derived
metric ranges, matching what the visualizer can load lazily.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from shared.coverage.schema import (
    COVERAGE_CANONICAL_VALUE_METRICS,
    coverage_available_metrics,
    derive_coverage_metric_layer,
    normalise_coverage_tx_names,
    validate_coverage_hdf5_contract,
)
from shared.frames.contracts import PathMetric
from shared.frames.manifest import (
    FRAMES_MANIFEST_FILENAME,
    FrameManifestError,
    load_frame_manifest,
)
from shared.frames.providers import Hdf5Provider
from shared.frames.schema import count_frame_mpcs, summarize_frame
from shared.frames.sionna_metadata import SIONNA_INTERACTION_LABELS_LOWER
from shared.frames.types import StandardMPCFrame
from shared.scenarios import load_scenario


class FrameInspectionError(RuntimeError):
    """Raised when generated frame data cannot be inspected."""


class CoverageInspectionError(RuntimeError):
    """Raised when generated coverage data cannot be inspected."""


def _looks_like_frames_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / FRAMES_MANIFEST_FILENAME).exists():
        return True
    return False


def _scenario_frames_subdir(scenario_yaml: Path) -> str:
    """Return the validated frame directory declared by one scenario YAML."""
    try:
        scenario = load_scenario(scenario_yaml)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise FrameInspectionError(
            f"Could not load frame directory from {scenario_yaml}: {exc}"
        ) from exc
    return scenario.data.files.directory


def _resolve_frame_source(path: Path, frames_subdir: str | None) -> tuple[Path, str]:
    """Return ``(scenario_root, frames_subdir)`` for scenario or frames paths."""
    expanded = path.expanduser()

    if expanded.is_file() and expanded.name == "scenario.yaml":
        return expanded.parent, frames_subdir or _scenario_frames_subdir(expanded)
    if expanded.is_file() and expanded.suffix.lower() in {".h5", ".hdf5"}:
        return expanded.parent, "."
    if frames_subdir:
        return expanded, frames_subdir
    if _looks_like_frames_dir(expanded):
        return expanded.parent, expanded.name
    scenario_yaml = expanded / "scenario.yaml"
    if scenario_yaml.is_file():
        return expanded, _scenario_frames_subdir(scenario_yaml)
    return expanded, "frames"


def _exact_chunk_frame_ids(path: Path) -> tuple[int, ...] | None:
    """Return the advertised frame scope when *path* names one HDF5 chunk."""
    expanded = path.expanduser()
    if not expanded.is_file() or expanded.suffix.lower() not in {".h5", ".hdf5"}:
        return None

    manifest_path = expanded.parent / FRAMES_MANIFEST_FILENAME
    try:
        manifest = load_frame_manifest(expanded.parent, verify_files=False)
    except FrameManifestError as exc:
        raise FrameInspectionError(
            f"Could not verify {expanded.name} against {manifest_path}: {exc}"
        ) from exc

    chunk = next((item for item in manifest.chunks if item.file == expanded.name), None)
    if chunk is None:
        raise FrameInspectionError(f"{expanded.name} is not advertised by {manifest_path}")
    return chunk.frame_ids


def _resolve_coverage_source(path: Path) -> Path:
    """Resolve a scenario, coverage directory, or coverage file to one HDF5 file."""
    expanded = path.expanduser()
    if expanded.is_file() and expanded.suffix.lower() in {".h5", ".hdf5"}:
        return expanded
    if expanded.is_file() and expanded.name == "scenario.yaml":
        expanded = expanded.parent
    candidates = [
        expanded / "coverage" / "coverage_maps.h5",
        expanded / "coverage" / "coverage_maps.hdf5",
        expanded / "coverage_maps.h5",
        expanded / "coverage_maps.hdf5",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise CoverageInspectionError(f"No coverage v2 HDF5 file found under {expanded}")


def _format_frame_indices(frames: Sequence[int], max_ranges: int = 12) -> str:
    if not frames:
        return "none"

    sorted_frames = sorted(int(frame) for frame in frames)
    ranges: list[str] = []
    start = previous = sorted_frames[0]
    for frame in sorted_frames[1:]:
        if frame == previous + 1:
            previous = frame
            continue
        ranges.append(f"{start}" if start == previous else f"{start}-{previous}")
        start = previous = frame
    ranges.append(f"{start}" if start == previous else f"{start}-{previous}")

    if len(ranges) <= max_ranges:
        return ", ".join(ranges)
    displayed = ", ".join(ranges[:max_ranges])
    return f"{displayed}, ... ({len(sorted_frames)} frames total)"


def _array_rows(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        rows = value.tolist()
    else:
        rows = list(value)
    if not rows:
        return []
    if isinstance(rows[0], (int, float)):
        return [rows]
    return rows


def _position_rows(values: Any) -> list[list[float]]:
    rows: list[list[float]] = []
    for row in _array_rows(values):
        if len(row) < 3:
            continue
        rows.append([float(row[0]), float(row[1]), float(row[2])])
    return rows


def _format_positions(
    label: str, positions: Sequence[Sequence[float]], limit: int = 5
) -> list[str]:
    if not positions:
        return []

    lines = [f"{label} positions:"]
    for idx, position in enumerate(positions[:limit], start=1):
        x, y, z = position[:3]
        lines.append(f"  {label}{idx}: ({x:.3f}, {y:.3f}, {z:.3f})")
    remaining = len(positions) - limit
    if remaining > 0:
        lines.append(f"  ... {remaining} more")
    return lines


def _path_counts(frame: StandardMPCFrame) -> list[int]:
    """Return physical-path counts in canonical TX/RX pair order."""
    return [int(value) for value in np.diff(frame.pair_path_offsets)]


def _interaction_label(code: int) -> str:
    label = SIONNA_INTERACTION_LABELS_LOWER.get(code, "unknown")
    if code == 0:
        label = "direct/no interaction"
    return f"code {code} ({label})"


def _format_count_map(counts: dict[int, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(
        f"{_interaction_label(code)}={count}"
        for code, count in sorted(counts.items(), key=lambda item: item[0])
    )


def _interaction_counts(frame: StandardMPCFrame) -> dict[str, dict[int, int]]:
    """Count physical interaction codes by bounce and by containing path."""
    segment_counts: dict[int, int] = {}
    path_counts: dict[int, int] = {}

    if frame.interactions.size:
        values, counts = np.unique(frame.interactions, return_counts=True)
        segment_counts = {
            int(value): int(count) for value, count in zip(values, counts, strict=True)
        }
    for path_index in range(int(frame.num_paths)):
        start = int(frame.bounce_offsets[path_index])
        end = int(frame.bounce_offsets[path_index + 1])
        for value in np.unique(frame.interactions[start:end]):
            code = int(value)
            path_counts[code] = path_counts.get(code, 0) + 1

    return {
        "segments": segment_counts,
        "paths": path_counts,
    }


def _frame_summary_data(frame: StandardMPCFrame) -> dict[str, Any]:
    metric_fields = (
        (PathMetric.DELAY_NS, "delays_ns"),
        (PathMetric.PATH_LOSS_DB, "path_loss_db"),
        (PathMetric.AOA_AZ_DEG, "aoa_az"),
        (PathMetric.AOA_EL_DEG, "aoa_el"),
        (PathMetric.AOD_AZ_DEG, "aod_az"),
        (PathMetric.AOD_EL_DEG, "aod_el"),
    )
    metrics = [label for metric, label in metric_fields if np.any(frame.metric_is_valid(metric))]

    features: list[str] = []
    if frame.beamforming:
        features.append("beamforming")
    if frame.sensing:
        features.append("sensing")

    summary = {
        "num_tx": frame.num_tx,
        "num_rx": frame.num_rx,
        "num_targets": frame.num_targets,
        "num_pairs": int(frame.num_pairs),
        "total_mpcs": int(count_frame_mpcs(frame)),
        "paths_per_pair": _path_counts(frame),
        "interaction_counts": _interaction_counts(frame),
        "metrics": metrics,
        "features": features,
    }
    sensing = _sensing_summary_data(frame.sensing)
    if sensing:
        summary["sensing"] = sensing
    return summary


def _dataset_shape(value: Any) -> list[int] | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return [int(dim) for dim in arr.shape]


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(scalar):
        return None
    return scalar


def _sensing_summary_data(sensing: Any) -> dict[str, Any]:
    """Return display-oriented metadata for an optional frame extension block."""
    if not isinstance(sensing, dict) or not sensing:
        return {}

    config = sensing.get("config_resolved") or sensing.get("config") or {}
    processing_mode = str(config.get("processing_mode") or "unknown")
    if processing_mode == "unknown":
        strategy = str(sensing.get("processing_strategy", ""))
        if strategy == "coherent_cpi_fft":
            processing_mode = "coherent_cpi"
        elif strategy == "sequential_stft":
            processing_mode = "sequential_stft"
        elif strategy == "snapshot_direct_rd":
            processing_mode = "snapshot_direct_rd"
    detections = sensing.get("detections") or []
    gt = sensing.get("gt_sensing") or sensing.get("gt") or sensing.get("gt_state") or {}
    target_ranges = gt.get("target_ranges") if isinstance(gt, dict) else None

    summary: dict[str, Any] = {
        "processing_mode": processing_mode,
        "frame_role": str(sensing.get("frame_role", "unknown")),
        "range_doppler_shape": _dataset_shape(sensing.get("range_doppler_map")),
        "range_profile_shape": _dataset_shape(sensing.get("range_profile")),
        "raw_range_doppler_shape": _dataset_shape(sensing.get("range_doppler_map_raw")),
        "aoa_azimuth_shape": _dataset_shape(sensing.get("range_doppler_aoa_az_map")),
        "aoa_elevation_shape": _dataset_shape(sensing.get("range_doppler_aoa_el_map")),
        "azimuth_cube_shape": _dataset_shape(sensing.get("range_doppler_azimuth_cube")),
        "azimuth_axis_shape": _dataset_shape(sensing.get("azimuth_axis_deg")),
        "elevation_axis_shape": _dataset_shape(sensing.get("elevation_axis_deg")),
        "detection_count": len(detections) if isinstance(detections, list) else 0,
        "gt_target_count": len(target_ranges) if isinstance(target_ranges, list) else 0,
        "range_resolution_m": _finite_float(config.get("range_resolution_m")),
        "velocity_resolution_m_s": _finite_float(config.get("velocity_resolution_m_s")),
        "prf_hz": _finite_float(config.get("prf_hz")),
    }
    return {key: value for key, value in summary.items() if value not in (None, [])}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _decode_hdf5_strings(values: Any) -> list[str]:
    arr = np.asarray(values)
    result: list[str] = []
    for item in arr.tolist():
        if isinstance(item, bytes):
            result.append(item.decode("utf-8"))
        else:
            result.append(str(item))
    return result


def _range_for_array(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values)
    valid = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    if valid.size == 0:
        return {"shape": list(arr.shape), "min": None, "max": None}
    return {"shape": list(arr.shape), "min": float(np.min(valid)), "max": float(np.max(valid))}


def _json_attr(attrs: Any, key: str, default: Any) -> Any:
    value = attrs.get(key)
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _h5_has(parent: Any, name: str) -> bool:
    return name in parent


def _h5_child(parent: Any, name: str) -> Any:
    return parent[name]


def _h5_array(parent: Any, name: str, dtype: Any = None) -> np.ndarray:
    return np.asarray(parent[name], dtype=dtype)


def inspect_coverage(path: Path) -> dict[str, Any]:
    """Inspect generated v2 coverage HDF5 data and return structured summary.

    Metric ranges are computed from the advertised logical metric list, not only
    from datasets physically present under ``/values`` or ``/derived``. This is
    what makes compact coverage files inspectable without inflating storage.
    """
    if not path.expanduser().exists():
        raise CoverageInspectionError(f"Path does not exist: {path}")

    coverage_file = _resolve_coverage_source(path)
    try:
        import h5py
    except ImportError as exc:
        raise CoverageInspectionError("h5py is required to inspect coverage files") from exc

    with h5py.File(coverage_file, "r") as f:
        schema_version = int(f.attrs.get("coverage_schema_version", 0))
        try:
            validate_coverage_hdf5_contract(
                schema_version,
                f.attrs.get("coverage_storage_layout"),
            )
        except ValueError as exc:
            raise CoverageInspectionError(str(exc)) from exc

        grid = _h5_child(f, "grid")
        tx = _h5_child(f, "tx")
        rx = _h5_child(f, "rx")
        values = _h5_child(f, "values")
        derived = _h5_child(f, "derived") if _h5_has(f, "derived") else None
        solver_attrs = dict(_h5_child(f, "solver").attrs.items()) if _h5_has(f, "solver") else {}
        metadata_attrs = (
            dict(_h5_child(f, "metadata").attrs.items()) if _h5_has(f, "metadata") else {}
        )
        path_gain = _h5_array(values, "path_gain_linear", np.float32)
        tx_count = int(path_gain.shape[2])
        tx_names = normalise_coverage_tx_names(
            _decode_hdf5_strings(_h5_child(tx, "names")[()]),
            tx_count,
        )
        tx_power_dbm = _h5_array(tx, "powers_dbm", np.float32)
        materialized = (
            {key: _h5_array(derived, key) for key in derived.keys()} if derived is not None else {}
        )
        metrics_store = _json_attr(metadata_attrs, "metrics_store", [])
        metrics_derived = _json_attr(metadata_attrs, "metrics_derived", [])
        available_metrics = _json_attr(metadata_attrs, "available_metrics", [])
        if not available_metrics:
            available_metrics = coverage_available_metrics(
                tx_names=tx_names,
                tx_count=tx_count,
                metrics_store=metrics_store or COVERAGE_CANONICAL_VALUE_METRICS,
                metrics_derived=metrics_derived,
                primary_metric=str(f.attrs.get("metric_name", "best_path_loss_db")),
                materialized_values=COVERAGE_CANONICAL_VALUE_METRICS,
                materialized_derived=materialized.keys(),
            )
        # ``available_metrics`` can include recipe-backed names such as
        # ``path_loss_db/TX1``. Derive each range from canonical path gain so the
        # CLI summarizes what a visualizer user can actually select.
        metric_ranges: dict[str, Any] = {}
        for key in available_metrics:
            try:
                metric_ranges[key] = _range_for_array(
                    derive_coverage_metric_layer(
                        path_gain,
                        str(key),
                        tx_power_dbm=tx_power_dbm,
                        noise_power_w=float(metadata_attrs.get("noise_power_w", 0.0)),
                        tx_names=tx_names,
                        materialized=materialized,
                    )
                )
            except ValueError:
                continue
        return {
            "coverage_file": str(coverage_file),
            "schema_version": schema_version,
            "metric_name": str(f.attrs.get("metric_name", "unknown")),
            "value_range": {
                "min": float(f.attrs.get("value_min", 0.0)),
                "max": float(f.attrs.get("value_max", 0.0)),
            },
            "grid": {
                "shape_xyz": _h5_array(grid, "shape_xyz").tolist(),
                "shape_yx": _h5_array(grid, "shape_yx").tolist(),
                "spacing_xy_m": _h5_array(grid, "spacing_xy").tolist(),
                "heights_m": _h5_array(grid, "heights_m").tolist(),
            },
            "tx": {
                "count": int(_h5_child(tx, "positions_m").shape[0]),
                "names": tx_names,
                "positions_m": _h5_array(tx, "positions_m").tolist(),
                "powers_dbm": _h5_array(tx, "powers_dbm").tolist(),
            },
            "rx": {
                "count": int(_h5_child(rx, "positions_m").shape[0]),
                "names": _decode_hdf5_strings(_h5_child(rx, "names")[()]),
                "positions_m": _h5_array(rx, "positions_m").tolist(),
            },
            "solver": solver_attrs,
            "metadata": metadata_attrs,
            "metrics": metric_ranges,
        }


def inspect_frames(
    path: Path,
    *,
    frame_idx: int | None = None,
    frames_subdir: str | None = None,
    include_frame: bool = True,
    include_materials: bool = False,
) -> dict[str, Any]:
    """Inspect generated HDF5 frame chunks and return structured summary data.

    The returned dictionary is intentionally close to the CLI output: callers
    can print it as text, emit it as JSON, or use it in tests without opening
    the HDF5 files directly.
    """
    if not path.expanduser().exists():
        raise FrameInspectionError(f"Path does not exist: {path}")

    exact_chunk_frames = _exact_chunk_frame_ids(path)
    source, resolved_subdir = _resolve_frame_source(path, frames_subdir)
    frames_dir = source / resolved_subdir

    try:
        provider = Hdf5Provider(source, frames_subdir=resolved_subdir)
    except (FileNotFoundError, FrameManifestError) as exc:
        raise FrameInspectionError(str(exc)) from exc

    try:
        frames = (
            list(exact_chunk_frames) if exact_chunk_frames is not None else provider.list_frames()
        )
        if not frames:
            raise FrameInspectionError(f"No HDF5 frames found under {frames_dir}")

        selected_frame = int(frame_idx) if frame_idx is not None else int(frames[0])
        result: dict[str, Any] = {
            "source": str(source),
            "frames_dir": str(frames_dir),
            "frame_count": len(frames),
            "available_frames": [int(frame) for frame in frames],
            "available_frames_label": _format_frame_indices(frames),
            "selected_frame": selected_frame if include_frame else None,
        }

        if include_materials:
            try:
                manifest = load_frame_manifest(frames_dir, verify_files=False)
            except FrameManifestError as exc:
                raise FrameInspectionError(str(exc)) from exc
            material_block = manifest.provenance.get("material_properties")
            result["materials"] = _material_scattering_data(material_block)

        if not include_frame:
            return result

        if selected_frame not in frames:
            raise FrameInspectionError(
                f"Frame {selected_frame} is not available. "
                f"Available frames: {_format_frame_indices(frames)}"
            )

        frame = provider.load_frame(selected_frame)
        result["summary_text"] = summarize_frame(frame)
        result["summary"] = _frame_summary_data(frame)
        result["positions"] = {
            "tx": _position_rows(frame.tx_positions),
            "rx": _position_rows(frame.rx_positions),
            "targets": _position_rows(frame.target_positions_m),
        }
        source_info = frame.provenance
        if source_info:
            result["frame_source"] = source_info
        return result
    finally:
        provider.close()


def _format_text(result: dict[str, Any]) -> str:
    lines = [
        f"Source: {result['source']}",
        f"Frames directory: {result['frames_dir']}",
        f"Frame count: {result['frame_count']}",
        f"Available frames: {result['available_frames_label']}",
    ]

    selected_frame = result.get("selected_frame")
    if selected_frame is None:
        material_lines = _format_material_scattering_lines(result.get("materials"))
        if material_lines:
            lines.extend(["", *material_lines])
        return "\n".join(lines)

    lines.extend(["", f"Selected frame: {selected_frame}", ""])
    lines.extend(str(result["summary_text"]).splitlines())

    interaction_counts = result.get("summary", {}).get("interaction_counts", {})
    segment_counts = interaction_counts.get("segments", {})
    path_counts = interaction_counts.get("paths", {})
    if segment_counts or path_counts:
        lines.extend(
            [
                f"Interaction segments: {_format_count_map(segment_counts)}",
                f"Paths with interaction: {_format_count_map(path_counts)}",
            ]
        )

    material_lines = _format_material_scattering_lines(result.get("materials"))
    if material_lines:
        lines.extend(["", *material_lines])

    sensing_lines = _format_sensing_lines(result.get("summary", {}).get("sensing"))
    if sensing_lines:
        lines.extend(["", *sensing_lines])

    positions = result.get("positions", {})
    for label, key in (("TX", "tx"), ("RX", "rx"), ("Target", "targets")):
        position_lines = _format_positions(label, positions.get(key, []))
        if position_lines:
            lines.extend(["", *position_lines])

    frame_source = result.get("frame_source", {})
    source_file = frame_source.get("file")
    if source_file:
        lines.extend(["", f"HDF5 chunk: {source_file}"])

    return "\n".join(lines)


def _material_scattering_data(value: Any) -> dict[str, Any]:
    """Return manifest-recorded scene-material scattering coefficients."""
    if not isinstance(value, dict):
        return {"source": None, "scattering_coefficients": {}}

    raw_properties = value.get("properties")
    coefficients: dict[str, float] = {}
    if isinstance(raw_properties, dict):
        for raw_name, raw_values in raw_properties.items():
            if not isinstance(raw_values, dict):
                continue
            coefficient = _finite_float(raw_values.get("scattering_coefficient"))
            if coefficient is not None:
                coefficients[str(raw_name)] = coefficient

    source = value.get("source")
    return {
        "source": str(source) if source else None,
        "scattering_coefficients": dict(sorted(coefficients.items())),
    }


def _format_material_scattering_lines(value: Any) -> list[str]:
    """Format optional material coefficients for human-readable output."""
    if not isinstance(value, dict):
        return []
    coefficients = value.get("scattering_coefficients")
    if not isinstance(coefficients, dict) or not coefficients:
        return ["Scene material scattering coefficients: not recorded"]
    return [
        "Scene material scattering coefficients:",
        *(f"  {name}: {float(coefficient):.4g}" for name, coefficient in coefficients.items()),
    ]


def _format_shape(shape: Any) -> str:
    if not shape:
        return "none"
    return "x".join(str(int(dim)) for dim in shape)


def _format_optional_float(value: Any, suffix: str = "") -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value):.4g}{suffix}"
    except (TypeError, ValueError):
        return "unknown"


def _format_sensing_lines(sensing: Any) -> list[str]:
    if not isinstance(sensing, dict) or not sensing:
        return []

    lines = ["Frame extension: sensing"]
    mode = sensing.get("processing_mode", "unknown")
    role = sensing.get("frame_role", "unknown")
    lines.append(f"  Mode: {mode} ({role})")
    lines.append(
        "  Products: "
        f"RD {_format_shape(sensing.get('range_doppler_shape'))}, "
        f"range profile {_format_shape(sensing.get('range_profile_shape'))}"
    )
    raw_shape = sensing.get("raw_range_doppler_shape")
    if raw_shape:
        lines.append(f"  Raw RD: {_format_shape(raw_shape)}")
    angular_parts = []
    az_cube_shape = sensing.get("azimuth_cube_shape")
    if az_cube_shape:
        angular_parts.append(f"azimuth cube {_format_shape(az_cube_shape)}")
    aoa_az_shape = sensing.get("aoa_azimuth_shape")
    if aoa_az_shape:
        angular_parts.append(f"AoA az map {_format_shape(aoa_az_shape)}")
    aoa_el_shape = sensing.get("aoa_elevation_shape")
    if aoa_el_shape:
        angular_parts.append(f"AoA el map {_format_shape(aoa_el_shape)}")
    az_axis_shape = sensing.get("azimuth_axis_shape")
    if az_axis_shape:
        angular_parts.append(f"az axis {_format_shape(az_axis_shape)}")
    el_axis_shape = sensing.get("elevation_axis_shape")
    if el_axis_shape:
        angular_parts.append(f"el axis {_format_shape(el_axis_shape)}")
    if angular_parts:
        lines.append("  Angular: " + ", ".join(angular_parts))
    lines.append(
        "  Detections: "
        f"{int(sensing.get('detection_count', 0))} "
        f"(GT targets: {int(sensing.get('gt_target_count', 0))})"
    )
    lines.append(
        "  Resolution: "
        f"range {_format_optional_float(sensing.get('range_resolution_m'), ' m')}, "
        f"velocity {_format_optional_float(sensing.get('velocity_resolution_m_s'), ' m/s')}, "
        f"PRF {_format_optional_float(sensing.get('prf_hz'), ' Hz')}"
    )
    return lines


def _format_coverage_text(result: dict[str, Any]) -> str:
    grid = result["grid"]
    tx = result["tx"]
    rx = result["rx"]
    lines = [
        f"Coverage file: {result['coverage_file']}",
        f"Coverage schema: v{result['schema_version']}",
        f"Default metric: {result['metric_name']}",
        (
            f"Grid: {grid['shape_xyz'][0]}×{grid['shape_xyz'][1]}×{grid['shape_xyz'][2]} "
            f"at {grid['spacing_xy_m'][0]:.3g}×{grid['spacing_xy_m'][1]:.3g} m"
        ),
        f"Heights: {', '.join(f'{float(h):.2f} m' for h in grid['heights_m'])}",
        f"TX: {tx['count']} ({', '.join(tx['names']) or 'none'})",
        f"RX: {rx['count']} ({', '.join(rx['names']) or 'none'})",
    ]
    if tx["powers_dbm"]:
        lines.append(
            "TX powers: "
            + ", ".join(
                f"{name}={float(power):.1f} dBm"
                for name, power in zip(tx["names"], tx["powers_dbm"])
            )
        )
    lines.append("")
    lines.append("Metrics:")
    for name, info in sorted(result["metrics"].items()):
        range_text = (
            "no finite values" if info["min"] is None else f"{info['min']:.3g} to {info['max']:.3g}"
        )
        lines.append(f"  {name}: shape {info['shape']}, range {range_text}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``orchav-inspect``."""

    parser = argparse.ArgumentParser(
        prog="orchav-inspect",
        description="Print a human-readable summary of generated ORCHAV HDF5 frames.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Scenario directory, scenario.yaml, frames directory, or HDF5 chunk path.",
    )
    parser.add_argument(
        "--frame",
        "-f",
        type=int,
        default=None,
        help="Frame index to inspect. Defaults to the first available frame.",
    )
    parser.add_argument(
        "--frames-subdir",
        default=None,
        help="Frames directory relative to the scenario root. Defaults to 'frames'.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list available frame indices; do not load a frame.",
    )
    parser.add_argument(
        "--materials",
        action="store_true",
        help="Show scene-material scattering coefficients recorded in the frame manifest.",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Inspect coverage/coverage_maps.h5 instead of generated MPC frames.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit structured JSON instead of text.",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    """Run the inspect CLI and return a process-style exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.coverage or (
        args.path.suffix.lower() in {".h5", ".hdf5"} and args.path.name.startswith("coverage")
    ):
        try:
            result = inspect_coverage(args.path)
        except CoverageInspectionError as exc:
            sys.stderr.write(f"error: {exc}\n")
            return 1
        if args.as_json:
            sys.stdout.write(json.dumps(result, indent=2, default=_json_default))
        else:
            sys.stdout.write(_format_coverage_text(result))
        sys.stdout.write("\n")
        return 0

    try:
        result = inspect_frames(
            args.path,
            frame_idx=args.frame,
            frames_subdir=args.frames_subdir,
            include_frame=not args.list,
            include_materials=args.materials,
        )
    except FrameInspectionError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1

    if args.as_json:
        sys.stdout.write(json.dumps(result, indent=2, default=_json_default))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(_format_text(result))
        sys.stdout.write("\n")
    return 0


def main() -> None:
    """Entry point for the orchav-inspect CLI."""
    sys.exit(run())


if __name__ == "__main__":
    main()
