"""Profile generated benchmark scenarios for frame/export health.

The helpers read scenario YAML and HDF5 frame files without launching the
visualizer. They summarize MPC and target counts so benchmark-pack scripts can
flag empty exports, missing targets, and suspiciously constant frame counts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from shared.frames.contracts import MPC_HDF5_LAYOUT, MPC_HDF5_SCHEMA_VERSION
from shared.frames.manifest import (
    FRAMES_MANIFEST_FILENAME,
    FrameChunkManifest,
    load_frame_manifest,
)
from shared.scenarios.frame_paths import (
    DEFAULT_FRAMES_DIRECTORY,
    validate_frames_directory,
)
from shared.scenarios.paths import find_project_root, normalize_path


def _longest_run(values: list[int], predicate) -> int:
    """Return the longest contiguous run matching ``predicate``."""
    longest = 0
    current = 0
    for value in values:
        if predicate(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _longest_constant_run(values: list[int]) -> int:
    """Return the longest contiguous run of identical integer values."""
    if not values:
        return 0
    longest = 1
    current = 1
    prev = values[0]
    for value in values[1:]:
        if value == prev:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
            prev = value
    return longest


def _resolve_scenario_yaml(scenario: str | Path) -> Path:
    """Resolve either a scenario directory or YAML path to ``scenario.yaml``."""
    path = Path(scenario).expanduser().resolve()
    if path.is_dir():
        return path / "scenario.yaml"
    return path


def _frames_directory(scenario_yaml: Path) -> Path:
    """Resolve the configured v2 frame-set directory."""
    with scenario_yaml.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    data_files = ((cfg.get("data") or {}).get("files")) or {}
    directory = validate_frames_directory(data_files.get("directory", DEFAULT_FRAMES_DIRECTORY))
    return normalize_path(
        directory,
        base=scenario_yaml.parent,
        project_root=find_project_root(scenario_yaml.parent),
    )


def _discover_frame_chunks(
    scenario_yaml: Path,
) -> list[tuple[Path, FrameChunkManifest]]:
    """Return the exact ordered chunks advertised by the v2 manifest."""
    frames_dir = _frames_directory(scenario_yaml)
    if not (frames_dir / FRAMES_MANIFEST_FILENAME).is_file():
        return []
    manifest = load_frame_manifest(frames_dir)
    return [(frames_dir / chunk.file, chunk) for chunk in manifest.chunks]


def _load_scenario_metadata(scenario_yaml: Path) -> dict[str, Any]:
    """Load benchmark-relevant scenario metadata from YAML."""
    with scenario_yaml.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    actors = cfg.get("actors") or {}
    return {
        "expected_targets": len(actors.get("targets") or []),
        "raytracing_enabled": bool(((cfg.get("raytracing") or {}).get("enabled", False))),
    }


def _attribute_text(value: Any) -> str:
    """Return an HDF5 text attribute as a regular string."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _required_dataset(handle: h5py.File, path: str) -> h5py.Dataset:
    """Return one required packed-v2 dataset with a useful failure message."""
    dataset = handle.get(path)
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"{handle.filename} is missing required dataset /{path}")
    return dataset


def _profile_hdf5(path: Path, chunk: FrameChunkManifest) -> dict[str, Any]:
    """Extract cheap per-frame counts from one packed HDF5 v2 chunk."""
    with h5py.File(path, "r") as handle:
        if _attribute_text(handle.attrs.get("file_kind", "")) != "mpc_frames":
            raise ValueError(f"{path} is not an MPC frame chunk")
        if int(handle.attrs.get("schema_version", -1)) != MPC_HDF5_SCHEMA_VERSION:
            raise ValueError(f"{path} has an unsupported HDF5 schema version")
        if _attribute_text(handle.attrs.get("storage_layout", "")) != MPC_HDF5_LAYOUT:
            raise ValueError(f"{path} has an unsupported HDF5 storage layout")

        frame_ids = np.asarray(_required_dataset(handle, "frames/id")[:], dtype=np.int64)
        expected_ids = np.asarray(chunk.frame_ids, dtype=np.int64)
        if frame_ids.shape != expected_ids.shape or not np.array_equal(frame_ids, expected_ids):
            raise ValueError(f"{path} frame IDs do not match frames_manifest.json")
        frame_count = chunk.count

        pair_offsets = _required_dataset(handle, "index/frame_pair_path_offsets")
        if (
            pair_offsets.ndim != 2
            or pair_offsets.shape[0] != frame_count
            or pair_offsets.shape[1] < 1
        ):
            raise ValueError(f"{path} has invalid frame/pair path offsets")
        path_starts = np.asarray(pair_offsets[:, 0], dtype=np.int64)
        path_ends = np.asarray(pair_offsets[:, -1], dtype=np.int64)
        if (
            np.any(path_ends < path_starts)
            or path_starts[0] != 0
            or np.any(path_starts[1:] != path_ends[:-1])
        ):
            raise ValueError(f"{path} has non-contiguous frame path intervals")
        total_mpcs = path_ends - path_starts

        target_offsets = np.asarray(
            _required_dataset(handle, "index/frame_target_offsets")[:],
            dtype=np.int64,
        )
        if (
            target_offsets.shape != (frame_count + 1,)
            or target_offsets[0] != 0
            or np.any(target_offsets[1:] < target_offsets[:-1])
        ):
            raise ValueError(f"{path} has invalid frame target offsets")
        num_targets = np.diff(target_offsets)

        return {
            "file": str(path),
            "frame_count": frame_count,
            "num_mpcs": total_mpcs.tolist(),
            "num_targets": num_targets.tolist(),
        }


def profile_generated_scenario(scenario: str | Path) -> dict[str, Any]:
    """Return a health summary for generated benchmark scenario frames."""
    scenario_yaml = _resolve_scenario_yaml(scenario)
    frame_chunks = _discover_frame_chunks(scenario_yaml)
    scenario_metadata = _load_scenario_metadata(scenario_yaml)
    expected_targets = int(scenario_metadata["expected_targets"])
    raytracing_enabled = bool(scenario_metadata["raytracing_enabled"])

    all_num_mpcs: list[int] = []
    all_num_targets: list[int] = []
    file_summaries: list[dict[str, Any]] = []
    for frame_file, chunk in frame_chunks:
        file_profile = _profile_hdf5(frame_file, chunk)
        file_summaries.append(
            {
                "file": file_profile["file"],
                "frame_count": file_profile["frame_count"],
            }
        )
        all_num_mpcs.extend(file_profile["num_mpcs"])
        all_num_targets.extend(file_profile["num_targets"])

    frame_count = len(all_num_mpcs)
    zero_mpc_frames = sum(1 for value in all_num_mpcs if value == 0)
    nonzero_mpc_frames = frame_count - zero_mpc_frames
    longest_zero_run = _longest_run(all_num_mpcs, lambda value: value == 0)
    longest_constant_run = _longest_constant_run(all_num_mpcs)

    status = "ok"
    issues: list[str] = []
    if frame_count == 0:
        status = "fail"
        issues.append("no_frames")
    else:
        if expected_targets > 0 and (not all_num_targets or max(all_num_targets) == 0):
            status = "fail"
            issues.append("targets_missing_from_export")
        if nonzero_mpc_frames == 0:
            status = "fail"
            issues.append("all_zero_mpcs")
        elif longest_zero_run >= max(5, frame_count // 3):
            status = "warn"
            issues.append("long_zero_mpc_run")
        if raytracing_enabled and longest_constant_run >= max(8, frame_count // 2):
            status = "warn" if status == "ok" else status
            issues.append("suspicious_constant_mpc_count")

    return {
        "scenario": str(scenario_yaml),
        "raytracing_enabled": raytracing_enabled,
        "frame_files": file_summaries,
        "frame_count": frame_count,
        "expected_targets": expected_targets,
        "num_targets": {
            "min": int(min(all_num_targets)) if all_num_targets else 0,
            "max": int(max(all_num_targets)) if all_num_targets else 0,
            "mean": float(np.mean(all_num_targets)) if all_num_targets else 0.0,
        },
        "num_mpcs": {
            "min": int(min(all_num_mpcs)) if all_num_mpcs else 0,
            "max": int(max(all_num_mpcs)) if all_num_mpcs else 0,
            "mean": float(np.mean(all_num_mpcs)) if all_num_mpcs else 0.0,
            "zero_frames": int(zero_mpc_frames),
            "nonzero_frames": int(nonzero_mpc_frames),
            "zero_fraction": (float(zero_mpc_frames) / float(frame_count)) if frame_count else 1.0,
            "longest_zero_run": int(longest_zero_run),
            "longest_constant_run": int(longest_constant_run),
            "sample": [int(value) for value in all_num_mpcs[:12]],
        },
        "status": status,
        "issues": issues,
    }


def write_scenario_profile(scenario: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Write a scenario health profile to JSON and return the profile dict."""
    profile = profile_generated_scenario(scenario)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(profile, handle, indent=2)
    return profile
