"""Prepare supplied root-local poses on the authored Paris walking route."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from shared.animated_target import load_animated_target
from shared.animated_target_export import export_animated_target

WALK_START = np.array([-32.0, -55.5, 0.15])
WALK_V = np.array([0.2619151635, -0.9650909010, 0.0]) * 1.25


def prepare_walking(
    source: Path,
    output: Path,
    duration: float,
    *,
    start_s: float = 0.0,
    pose_period_s: float | None = None,
    offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Path:
    """Preserve supplied articulation, replacing only root travel and heading.

    A period is an explicit user policy, never inferred from clip length.
    It must exceed the final pose time so the last pose retains its sample slot.
    Without a period the source must cover the complete requested interval.
    """
    asset = load_animated_target(source)
    source_times = np.array([pose.time_s for pose in asset.pose_samples])
    relative = source_times - source_times[0]
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(start_s) or start_s < 0:
        raise ValueError("duration must be positive and route offset nonnegative")
    offset = np.asarray(offset_m, dtype=float)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("walking offset must have three finite metre values")
    if pose_period_s is None:
        if relative[-1] < start_s + duration:
            raise ValueError(
                "walking asset is too short; supply a longer clip or an explicit pose period"
            )
        boundaries = relative - start_s
    else:
        if not np.isfinite(pose_period_s) or pose_period_s <= relative[-1]:
            raise ValueError(
                "pose period must be finite and greater than the final relative pose time"
            )
        cycles = range(
            int(np.floor(start_s / pose_period_s)),
            int(np.floor((start_s + duration) / pose_period_s)) + 1,
        )
        boundaries = np.concatenate(
            [relative + cycle * pose_period_s - start_s for cycle in cycles]
        )
    times = np.unique(np.r_[0.0, boundaries[(boundaries > 0) & (boundaries < duration)], duration])
    motion_times = times + start_s
    cycles = np.zeros(len(times), dtype=int)
    if pose_period_s is not None:
        cycles = np.floor(motion_times / pose_period_s + 1e-12).astype(int)
        motion_times = np.maximum(0.0, motion_times - cycles * pose_period_s)
    indices = np.searchsorted(relative, motion_times + 1e-12, side="right") - 1
    indices = np.clip(indices, 0, len(relative) - 1)
    linear = asset.source_to_orchav_affine_4x4[:3, :3]
    vertices = np.stack([asset.pose_samples[index].vertices @ linear.T for index in indices])
    yaw = np.arctan2(WALK_V[1], WALK_V[0])
    mapping = [
        dict(
            time_s=float(time),
            source_pose_id=asset.pose_samples[index].sample_id,
            source_time_s=float(source_times[index]),
            cycle=int(cycle),
        )
        for time, index, cycle in zip(times, indices, cycles)
    ]
    result = export_animated_target(
        output,
        asset_id="opera_multitarget_walker",
        vertices=vertices,
        faces=asset.canonical_faces,
        times_s=times,
        root_translation=WALK_START + offset + (times[:, None] + start_s) * WALK_V,
        root_rotation_xyzw=np.tile([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)], (len(times), 1)),
        source_type="supplied_animated_surface",
        front_axis="+X",
        up_axis="+Z",
        source_records={
            "manifest": {
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "size_bytes": source.stat().st_size,
                "asset_revision_sha256": asset.revision_sha256,
            }
        },
        adapter_metadata={
            "source_asset_id": asset.asset_id,
            "source_license": dict(asset.license),
            "pose_policy": (
                "previous source pose; explicit finite repetition"
                if pose_period_s
                else "previous source pose; no repetition"
            ),
            "pose_period_s": pose_period_s,
            "source_time_origin_s": float(source_times[0]),
            "source_to_output_pose_mapping": mapping,
            "root_policy": "source root discarded; authored Paris route and heading applied once",
            "geometry_policy": "declared linear basis and scale; no per-pose recentering or grounding",
            "root_offset_m": offset.tolist(),
            "route_offset_s": start_s,
            "root_speed_m_s": 1.25,
        },
    )
    (output / "pose_mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    return result
