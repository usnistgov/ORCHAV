"""Offline authoring helpers for source-neutral, locally supplied motion assets.

Only NumPy and the neutral animation contract are required. Source frameworks,
body models and inference environments never enter this module.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from shared.animated_target import (
    animated_target_revision_sha256,
    canonical_json_bytes,
    canonical_sha256,
    load_animated_target,
)
from shared.animated_target_workflow import _prepare_destination, _promote_directory_noreplace

# Half-turn intervals have two equally short SLERP paths. Canonical per-sample
# quaternion signs cannot retain their branch through a body-basis change.
_ROOT_HALF_TURN_DOT_TOLERANCE = 1e-12


def load_numeric_npz(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and hash the same NPZ bytes, rejecting every object-valued member."""
    payload = Path(path).read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays, {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def sample_times(
    frame_count: int,
    *,
    times_s: Any = None,
    source_fps: float | None = None,
) -> np.ndarray:
    """Require explicit physical timing; retain irregular input timestamps."""
    if frame_count < 1:
        raise ValueError("motion must contain at least one frame")
    if source_fps is not None and (
        not np.isfinite(source_fps) or isinstance(source_fps, bool) or source_fps <= 0
    ):
        raise ValueError("source_fps must be finite and positive")
    if times_s is None:
        if source_fps is None:
            raise ValueError("motion requires time_s or an explicit positive source_fps")
        result = np.arange(frame_count, dtype=np.float64) / source_fps
    else:
        result = np.asarray(times_s, dtype=np.float64)
        if result.shape != (frame_count,):
            raise ValueError("time_s must have one timestamp per motion frame")
        if source_fps is not None and frame_count > 1:
            if not np.allclose(np.diff(result), 1.0 / source_fps, rtol=1e-9, atol=1e-12):
                raise ValueError("time_s and source_fps disagree")
    if not np.all(np.isfinite(result)) or np.any(result < 0) or np.any(np.diff(result) <= 0):
        raise ValueError("time_s must be finite, nonnegative and strictly increasing")
    return result


def selected_indices(frame_count: int, stride: int) -> np.ndarray:
    """Select native source samples without retiming or fabricated interpolation."""
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer")
    return np.arange(0, frame_count, stride, dtype=np.int64)


def source_affine(front_axis: str, up_axis: str, *, meters_per_unit: float = 1.0) -> np.ndarray:
    """Map a declared right-handed source basis to +X front / +Z up."""
    axes = {
        f"{sign}{axis}": value * np.eye(3)[index]
        for index, axis in enumerate("XYZ")
        for sign, value in (("+", 1.0), ("-", -1.0))
    }
    if front_axis not in axes or up_axis not in axes:
        raise ValueError("front/up axes must be one of +X, -X, +Y, -Y, +Z, -Z")
    front, up = axes[front_axis], axes[up_axis]
    if np.dot(front, up) != 0:
        raise ValueError("front_axis and up_axis must be perpendicular")
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("meters_per_unit must be finite and positive")
    result = np.eye(4)
    result[:3, :3] = meters_per_unit * np.stack((front, np.cross(up, front), up))
    return result


def canonical_quaternions(rotations_xyzw: Any) -> np.ndarray:
    """Validate active unit XYZW quaternions and choose deterministic antipodes."""
    rotations = np.array(rotations_xyzw, dtype=np.float64, copy=True)
    if rotations.ndim != 2 or rotations.shape[1] != 4 or not np.all(np.isfinite(rotations)):
        raise ValueError("root rotations must have finite shape (frames, 4)")
    norms = np.linalg.norm(rotations, axis=1)
    if not np.allclose(norms, 1.0, rtol=0, atol=1e-8):
        raise ValueError("root rotations must be unit quaternions")
    rotations /= norms[:, None]
    for quaternion in rotations:
        for index in (3, 0, 1, 2):
            if abs(quaternion[index]) > 1e-12:
                if quaternion[index] < 0:
                    quaternion *= -1
                break
    return rotations


def rotation_matrices(rotations_xyzw: Any) -> np.ndarray:
    """Convert active unit XYZW root rotations without a source-framework import."""
    q = canonical_quaternions(rotations_xyzw)
    x, y, z, w = q.T
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(-1, 3, 3)


def matrices_to_quaternions(matrices: np.ndarray) -> np.ndarray:
    """Convert proper active rotation matrices to canonical XYZW quaternions."""
    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (3, 3) or not np.all(np.isfinite(values)):
        raise ValueError("root rotations must have finite shape (frames, 3, 3)")
    if not np.allclose(np.swapaxes(values, -1, -2) @ values, np.eye(3), rtol=0, atol=1e-5):
        raise ValueError("root rotations must contain orthonormal rotations")
    if not np.allclose(np.linalg.det(values), 1, rtol=0, atol=1e-5):
        raise ValueError("root rotations must contain proper right-handed rotations")
    result = []
    for matrix in values:
        # The largest diagonal of the symmetric quaternion outer product
        # chooses a numerically stable component, including 180-degree turns.
        diagonal = (
            np.array(
                (
                    1 + 2 * matrix[0, 0] - np.trace(matrix),
                    1 + 2 * matrix[1, 1] - np.trace(matrix),
                    1 + 2 * matrix[2, 2] - np.trace(matrix),
                    1 + np.trace(matrix),
                )
            )
            / 4
        )
        k = int(np.argmax(diagonal))
        q = np.zeros(4)
        q[k] = np.sqrt(max(diagonal[k], 0))
        d = 4 * q[k]
        if k == 3:
            q[:3] = (
                (matrix[2, 1] - matrix[1, 2]) / d,
                (matrix[0, 2] - matrix[2, 0]) / d,
                (matrix[1, 0] - matrix[0, 1]) / d,
            )
        else:
            others = [index for index in range(3) if index != k]
            for j in others:
                q[j] = (matrix[k, j] + matrix[j, k]) / d
            i, j = (k + 1) % 3, (k + 2) % 3
            q[3] = (matrix[j, i] - matrix[i, j]) / d
        q /= np.linalg.norm(q)
        result.append(q)
    return canonical_quaternions(result)


def export_animated_target(
    output_dir: str | Path,
    *,
    asset_id: str,
    vertices: Any,
    faces: Any,
    times_s: Any,
    root_translation: Any,
    root_rotation_xyzw: Any,
    source_type: str,
    source_records: Mapping[str, Any],
    adapter_metadata: Mapping[str, Any],
    front_axis: str = "+Z",
    up_axis: str = "+Y",
    meters_per_unit: float = 1.0,
    root_local_front_axis: str | None = None,
    root_local_up_axis: str | None = None,
    closed_surface: bool = False,
    out_of_range: str = "error",
) -> Path:
    """Validate and atomically publish a complete V1 asset without overwriting.

    ``front_axis``/``up_axis`` describe the source world containing root motion.
    Input poses are root-local geometry; root motion is applied exactly once by
    the V1 binder. Supply both ``root_local_*_axis`` values when the body's local
    basis differs from that source world. Geometry is then re-expressed in the
    source-world basis and root rotations are compensated, preserving the full
    animation, including between root samples. A nonidentity basis change
    rejects adjacent root rotations with an ambiguous half-turn SLERP path
    (absolute quaternion dot <= 1e-12); supply an intermediate source sample.
    Omitting both retains the input
    local basis and rotations unchanged. The source is right handed with outward
    counterclockwise triangles. Generated data stays ``user_supplied_only``.
    An authenticated provenance sidecar contains source hashes and transforms;
    paths, prompts and environment-specific identifiers are not added here.
    """
    poses = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces)
    if poses.ndim != 3 or poses.shape[2] != 3 or not np.all(np.isfinite(poses)):
        raise ValueError("vertices must have finite shape (frames, vertices, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in "iu":
        raise ValueError("faces must be integer indexed triangles")
    times = sample_times(len(poses), times_s=times_s)
    roots = np.asarray(root_translation, dtype=np.float64)
    rotations = canonical_quaternions(root_rotation_xyzw)
    if roots.shape != (len(poses), 3) or not np.all(np.isfinite(roots)):
        raise ValueError("root_translation must have finite shape (frames, 3)")
    if len(rotations) != len(poses):
        raise ValueError("root rotations and poses must have matching frame counts")
    affine = source_affine(front_axis, up_axis, meters_per_unit=meters_per_unit)
    if (root_local_front_axis is None) != (root_local_up_axis is None):
        raise ValueError("root_local_front_axis and root_local_up_axis must be supplied together")
    basis_operation: dict[str, Any] = {"policy": "preserve_unspecified_input_basis"}
    if root_local_front_axis is not None and root_local_up_axis is not None:
        local_basis = source_affine(root_local_front_axis, root_local_up_axis)[:3, :3]
        world_basis = source_affine(front_axis, up_axis)[:3, :3]
        # Both matrices are proper unit rotations: scale belongs only to the
        # final world affine. B changes body coordinates, not the physical pose.
        basis = world_basis.T @ local_basis
        if not np.array_equal(basis, np.eye(3)):
            adjacent_dots = np.einsum("ij,ij->i", rotations[:-1], rotations[1:])
            if np.any(np.abs(adjacent_dots) <= _ROOT_HALF_TURN_DOT_TOLERANCE):
                raise ValueError(
                    "root-local basis re-expression cannot preserve an ambiguous 180-degree "
                    "root-rotation interval; supply an explicit intermediate source sample"
                )
            poses = poses @ basis.T
            rotations = matrices_to_quaternions(rotation_matrices(rotations) @ basis.T)
        basis_operation = {
            "policy": "reexpress_in_source_world_basis",
            "input_front_axis": root_local_front_axis,
            "input_up_axis": root_local_up_axis,
            "output_front_axis": front_axis,
            "output_up_axis": up_axis,
            "input_to_output_rotation_3x3": basis.tolist(),
            "root_rotation_compensation": "R_output = R_input @ B.T",
            "world_geometry": "unchanged; A @ R_output @ B @ v = A @ R_input @ v",
        }
    if out_of_range not in ("error", "hold"):
        raise ValueError("out_of_range must be error or hold; automatic looping is unsupported")
    if not isinstance(closed_surface, bool):
        raise ValueError("closed_surface must be a boolean")
    provenance = {
        "adapter_contract_version": 1,
        "exporter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "source_type": source_type,
        "sources": dict(source_records),
        "operations": {**adapter_metadata, "root_local_basis": basis_operation},
        "coordinates": {
            "front_axis": front_axis,
            "up_axis": up_axis,
            "source_to_orchav_affine_4x4": affine.tolist(),
        },
        "sample_times_s": times.tolist(),
        "root_ownership": "root_local_pose_then_source_root_then_source_to_orchav",
        "pose_interpolation": "step",
        "root_interpolation": "linear_translation_shortest_slerp_rotation",
        "redistribution": "user_supplied_only",
    }
    provenance_bytes = canonical_json_bytes(provenance)
    provenance_hash = hashlib.sha256(provenance_bytes).hexdigest()
    destination = _prepare_destination(output_dir)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        (stage / "poses").mkdir()
        (stage / "adapter_provenance.json").write_bytes(provenance_bytes)
        resources: dict[str, Any] = {}
        pose_samples, root_samples = [], []
        for pose, time, translation, rotation in zip(poses, times, roots, rotations):
            payload = canonical_json_bytes({"vertices": pose.tolist(), "faces": triangles.tolist()})
            digest = hashlib.sha256(payload).hexdigest()
            resource_id = f"mesh:sha256:{digest}"
            relative = f"poses/{digest}.json"
            if resource_id not in resources:
                (stage / relative).write_bytes(payload)
                resources[resource_id] = {
                    "resource_id": resource_id,
                    "path": relative,
                    "sha256": digest,
                    "size_bytes": len(payload),
                    "media_type": "application/vnd.orchav.animated-pose+json",
                }
            pose_identity = {"time_s": float(time), "resource_id": resource_id}
            pose_samples.append(
                {"sample_id": f"pose:sha256:{canonical_sha256(pose_identity)}", **pose_identity}
            )
            root_identity = {
                "time_s": float(time),
                "translation": translation.tolist(),
                "rotation_xyzw": rotation.tolist(),
            }
            root_samples.append(
                {"sample_id": f"root:sha256:{canonical_sha256(root_identity)}", **root_identity}
            )
        topology_hash = canonical_sha256(
            {
                "representation": "indexed_triangles",
                "vertex_count": poses.shape[1],
                "faces": triangles.tolist(),
            }
        )
        policy = {"out_of_range": out_of_range, "cyclic": False, "period_s": None}
        manifest = {
            "manifest_version": 1,
            "asset": {"asset_id": asset_id, "revision_sha256": ""},
            "resources": list(resources.values()),
            "coordinates": {
                "source_frame_id": f"{source_type}:source_coordinates",
                "source_handedness": "right",
                "front_axis": front_axis,
                "up_axis": up_axis,
                "length_unit": "m" if meters_per_unit == 1 else "source_unit",
                "destination_frame_id": "orchav:world_rh_x_front_z_up_m",
                "source_to_orchav_affine_4x4": affine.tolist(),
                "scale_ownership": "uniform_scale_in_affine_m_per_source_unit",
                "translation_ownership": "translation_in_affine_destination_m",
            },
            "topology": {
                "representation": "indexed_triangles",
                "mode": "fixed_across_poses",
                "vertex_count": poses.shape[1],
                "triangle_count": len(triangles),
                "topology_sha256": topology_hash,
                "closed_surface": closed_surface,
                "source_face_winding": "counter_clockwise_from_outside",
                "reflection_winding_policy": "unchanged_if_no_reflection",
                "normal_policy": "recompute",
            },
            "motion": {
                "time_axis": "asset_time_s",
                "pose": {"interpolation": "step", **policy, "samples": pose_samples},
                "root": {
                    "translation_interpolation": "linear",
                    "rotation_interpolation": "slerp_shortest",
                    "rotation_representation": "active_quaternion_xyzw",
                    **policy,
                    "samples": root_samples,
                },
            },
            "provenance": {
                "creator": "ORCHAV offline adapter",
                "source_type": source_type,
                "source_ids": [f"adapter_provenance:sha256:{provenance_hash}"],
                "description": "User-supplied source; provenance sidecar is pinned by its SHA256 source ID.",
            },
            "license": {
                "spdx_id": "LicenseRef-User-Supplied-Motion",
                "redistribution": "user_supplied_only",
                "notice": "Source model, motion and derived geometry retain their separate terms. No redistribution authorization is supplied.",
            },
        }
        manifest["asset"]["revision_sha256"] = animated_target_revision_sha256(manifest)
        (stage / "animated_target.json").write_bytes(canonical_json_bytes(manifest))
        load_animated_target(stage / "animated_target.json")
        _promote_directory_noreplace(stage, destination)
        return destination / "animated_target.json"
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_motion_source_metadata(path: str | Path) -> dict[str, Any]:
    """Read optional source lineage containing IDs, revisions, hashes and a seed.

    Raw prompts, local paths, account names and arbitrary metadata are not
    accepted by this interchange. A caller can supply a redacted prompt hash.
    """
    import json
    import re

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate source-metadata key: {key}")
            result[key] = value
        return result

    payload = Path(path).read_bytes()
    metadata = json.loads(payload, object_pairs_hook=unique_object)
    allowed = {
        "source_id",
        "upstream_revision",
        "checkpoint_sha256",
        "redacted_prompt_sha256",
        "seed",
        "generator_version",
    }
    if not isinstance(metadata, dict) or not metadata or set(metadata) - allowed:
        raise ValueError(
            "source metadata must contain only supported source IDs, revisions, hashes, seed, or generator_version"
        )
    for key, value in metadata.items():
        if key == "seed":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("source metadata seed must be a nonnegative integer")
        elif key in {"checkpoint_sha256", "redacted_prompt_sha256", "upstream_revision"}:
            sizes = (40, 64) if key == "upstream_revision" else (64,)
            if (
                not isinstance(value, str)
                or len(value) not in sizes
                or not re.fullmatch(r"[0-9a-f]+", value)
            ):
                raise ValueError(f"source metadata {key} must be a lowercase hexadecimal digest")
        elif (
            not isinstance(value, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._+-]{0,127}", value)
            or re.match(r"^[A-Za-z]:", value)
        ):
            raise ValueError(
                f"source metadata {key} must be a portable semantic identifier or version"
            )
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "metadata": metadata,
    }
