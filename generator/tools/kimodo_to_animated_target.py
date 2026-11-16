"""Offline Kimodo/SOMA motion-and-skin adapter; no inference runtime is loaded.

Inputs are user-supplied numeric NPZ/NPY exports. SOMA-77 motion can be skinned
directly. SOMA-30 motion additionally needs numeric neutral joints and relaxed
local rotations for its SOMA-77 skin. No checkpoints, prompts, skin assets or
generated motion are bundled with this code.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np

from shared.animated_target_export import (
    export_animated_target,
    load_motion_source_metadata,
    load_numeric_npz,
    matrices_to_quaternions,
    rotation_matrices,
    sample_times,
    selected_indices,
)

SOMA30_NAMES = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumbEnd",
    "LeftHandMiddleEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumbEnd",
    "RightHandMiddleEnd",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
)
SOMA30_PARENTS = np.array(
    (
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        6,
        6,
        3,
        10,
        11,
        12,
        13,
        13,
        3,
        16,
        17,
        18,
        19,
        19,
        0,
        22,
        23,
        24,
        0,
        26,
        27,
        28,
    )
)


def _rotations(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    matrices = np.asarray(values, dtype=np.float64)
    if matrices.shape != (*shape, 3, 3) or not np.all(np.isfinite(matrices)):
        raise ValueError(f"{name} must have finite shape {(*shape, 3, 3)}")
    if not np.allclose(np.swapaxes(matrices, -1, -2) @ matrices, np.eye(3), rtol=0, atol=1e-5):
        raise ValueError(f"{name} must contain orthonormal rotations")
    if not np.allclose(np.linalg.det(matrices), 1, rtol=0, atol=1e-5):
        raise ValueError(f"{name} must contain proper right-handed rotations")
    return matrices


def _skeleton(connections: Any, joint_count: int) -> tuple[np.ndarray, list[int], int]:
    edges = np.asarray(connections)
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("rig_joint_connections must be integer (parent, child) edges")
    if np.any(edges < 0) or np.any(edges >= joint_count):
        raise ValueError("skeleton edge contains an out-of-range joint index")
    parents = np.full(joint_count, -1, dtype=np.int64)
    for parent, child in edges:
        if parent == child or parents[child] >= 0:
            raise ValueError("skeleton joints must have one parent and no self edge")
        parents[child] = parent
    roots = np.flatnonzero(parents < 0)
    if len(roots) != 1:
        raise ValueError("skeleton must contain exactly one root")
    order = [int(roots[0])]
    for parent in order:
        order.extend(np.flatnonzero(parents == parent).tolist())
    if len(order) != joint_count or len(set(order)) != joint_count:
        raise ValueError("skeleton must be a connected acyclic tree")
    return parents, order, int(roots[0])


def _load_npy(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = Path(path).read_bytes()
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "iuf":
        raise ValueError("skeleton inputs must be numeric NPY arrays")
    return value, {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _expand_soma30(
    positions: np.ndarray,
    rotations: np.ndarray,
    names: list[str],
    parents: np.ndarray,
    order: list[int],
    neutral: np.ndarray,
    relaxed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if neutral.shape != (77, 3) or not np.all(np.isfinite(neutral)):
        raise ValueError("neutral joints must have finite shape (77, 3)")
    relaxed = _rotations(relaxed, (77,), "relaxed rotations")
    missing = set(SOMA30_NAMES) - set(names)
    if missing:
        raise ValueError(f"SOMA-77 skin lacks SOMA-30 joints: {sorted(missing)}")
    locals30 = rotations.copy()
    for joint, parent in enumerate(SOMA30_PARENTS):
        if parent >= 0:
            locals30[:, joint] = np.swapaxes(rotations[:, parent], -1, -2) @ rotations[:, joint]
    local = np.broadcast_to(relaxed, (len(positions), 77, 3, 3)).copy()
    for source, name in enumerate(SOMA30_NAMES):
        local[:, names.index(name)] = locals30[:, source]
    global_rotations = np.empty_like(local)
    global_positions = np.empty((len(positions), 77, 3))
    for joint in order:
        parent = parents[joint]
        if parent < 0:
            global_rotations[:, joint] = local[:, joint]
            global_positions[:, joint] = positions[:, 0]
        else:
            global_rotations[:, joint] = global_rotations[:, parent] @ local[:, joint]
            global_positions[:, joint] = global_positions[:, parent] + np.einsum(
                "nij,j->ni", global_rotations[:, parent], neutral[joint] - neutral[parent]
            )
    return global_positions, global_rotations


def export_kimodo_motion(
    motion_npz: str | Path,
    skin_npz: str | Path,
    output_dir: str | Path,
    *,
    asset_id: str,
    source_fps: float | None = None,
    stride: int = 1,
    source_metadata: str | Path | None = None,
    neutral_joints: str | Path | None = None,
    relaxed_pose: str | Path | None = None,
    root_local_front_axis: str | None = None,
    root_local_up_axis: str | None = None,
    closed_surface: bool = False,
    out_of_range: str = "error",
) -> Path:
    """Skin SOMA motion, preserve exact times, and publish a local V1 asset."""
    motion, motion_record = load_numeric_npz(motion_npz)
    skin, skin_record = load_numeric_npz(skin_npz)
    positions = np.asarray(motion["posed_joints"], dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1] not in (30, 77) or positions.shape[2] != 3:
        raise ValueError("posed_joints must have shape (frames, 30 or 77, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("posed_joints contains nonfinite values")
    source_joint_count = positions.shape[1]
    rotations = _rotations(motion["global_rot_mats"], positions.shape[:2], "global_rot_mats")
    times = sample_times(len(positions), times_s=motion.get("time_s"), source_fps=source_fps)
    selected = selected_indices(len(positions), stride)
    names_array = np.asarray(skin["rig_joint_names"])
    if names_array.shape != (77,) or names_array.dtype.kind not in "SU":
        raise ValueError("rig_joint_names must contain 77 strings")
    names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names_array]
    if len(set(names)) != 77:
        raise ValueError("rig_joint_names must be unique")
    parents, order, root_index = _skeleton(skin["rig_joint_connections"], 77)
    if names[root_index] != "Hips":
        raise ValueError("the SOMA skin root must be named Hips")
    bind = np.asarray(skin["bind_rig_transform"], dtype=np.float64)
    if bind.shape != (77, 4, 4) or not np.all(np.isfinite(bind)):
        raise ValueError("bind_rig_transform must have finite shape (77, 4, 4)")
    if not np.allclose(bind[:, 3], (0, 0, 0, 1), rtol=0, atol=1e-10):
        raise ValueError("bind transforms must be affine rigid transforms")
    _rotations(bind[:, :3, :3], (77,), "bind rotations")
    bind_vertices = np.asarray(skin["bind_vertices"], dtype=np.float64)
    if (
        bind_vertices.ndim != 2
        or bind_vertices.shape[1] != 3
        or not np.all(np.isfinite(bind_vertices))
    ):
        raise ValueError("bind_vertices must have finite shape (vertices, 3)")
    indices = np.asarray(skin["lbs_indices"])
    weights = np.asarray(skin["lbs_weights"], dtype=np.float64)
    if (
        indices.ndim != 2
        or indices.shape[0] != len(bind_vertices)
        or indices.dtype.kind not in "iu"
    ):
        raise ValueError("lbs_indices must be an integer per-vertex matrix")
    if weights.shape != indices.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("lbs_weights must be finite nonnegative per-vertex weights")
    if (
        np.any(indices < 0)
        or np.any(indices >= 77)
        or not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError("LBS joint indices must be valid and weights must sum to one")
    sources = {"motion_npz": motion_record, "skin_npz": skin_record}
    if source_metadata is not None:
        sources["source_metadata"] = load_motion_source_metadata(source_metadata)
    if source_joint_count == 30:
        if neutral_joints is None or relaxed_pose is None:
            raise ValueError("SOMA-30 motion requires numeric --neutral-joints and --relaxed-pose")
        neutral, sources["neutral_joints"] = _load_npy(neutral_joints)
        relaxed, sources["relaxed_pose"] = _load_npy(relaxed_pose)
        positions, rotations = _expand_soma30(
            positions, rotations, names, parents, order, neutral, relaxed
        )
    elif neutral_joints is not None or relaxed_pose is not None:
        raise ValueError("SOMA-77 motion does not use neutral-joint or relaxed-pose inputs")
    root_positions = positions[selected, root_index]
    root_quaternions = matrices_to_quaternions(rotations[selected, root_index])
    root_rotations = rotation_matrices(root_quaternions)
    inverse_bind = np.linalg.inv(bind)
    homogeneous = np.column_stack((bind_vertices, np.ones(len(bind_vertices))))
    poses = []
    for source_index, root, root_rotation in zip(selected, root_positions, root_rotations):
        posed = np.broadcast_to(np.eye(4), (77, 4, 4)).copy()
        posed[:, :3, :3] = rotations[source_index]
        posed[:, :3, 3] = positions[source_index]
        transforms = posed @ inverse_bind
        influences = np.einsum("vwij,vj->vwi", transforms[indices, :3, :], homogeneous)
        world = np.einsum("vwi,vw->vi", influences, weights)
        poses.append((world - root) @ root_rotation)
    return export_animated_target(
        output_dir,
        asset_id=asset_id,
        vertices=poses,
        faces=skin["faces"],
        times_s=times[selected],
        root_translation=root_positions,
        root_rotation_xyzw=root_quaternions,
        source_type="kimodo_soma",
        source_records=sources,
        adapter_metadata={
            "adapter": "orchav.kimodo_to_animated_target",
            "version": 1,
            "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_joint_count": source_joint_count,
            "skin_joint_count": 77,
            "stride": stride,
            "selected_source_indices": selected.tolist(),
            "skinning": "linear_blend_skinning",
            "root_reference": "Hips",
            "root_rotation_canonicalization": "proper matrix to normalized active quaternion",
            "pose_interpolation": "none",
            "grounding": "none",
            "inference": "external; this adapter consumes exported motion",
        },
        front_axis="+Z",
        up_axis="+Y",
        root_local_front_axis=root_local_front_axis,
        root_local_up_axis=root_local_up_axis,
        closed_surface=closed_surface,
        out_of_range=out_of_range,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-npz", type=Path, required=True)
    parser.add_argument("--skin-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument(
        "--source-fps",
        type=float,
        help="Required if the motion has no time_s array; native Kimodo exports commonly use 30 Hz",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--source-metadata",
        type=Path,
        help="Optional JSON of source IDs, revision/hash fields, seed and generator_version",
    )
    parser.add_argument(
        "--neutral-joints", type=Path, help="Numeric (77,3) NPY for SOMA-30 expansion"
    )
    parser.add_argument(
        "--relaxed-pose",
        type=Path,
        help="Numeric (77,3,3) local-rotation NPY for SOMA-30 expansion",
    )
    parser.add_argument(
        "--root-local-front-axis",
        help="Optional declared body-local forward axis before the Hips root rotation",
    )
    parser.add_argument(
        "--root-local-up-axis",
        help="Optional body-local up axis; supply together with --root-local-front-axis",
    )
    parser.add_argument("--closed-surface", action="store_true")
    parser.add_argument("--out-of-range", choices=("error", "hold"), default="error")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        path = export_kimodo_motion(**vars(args))
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    sys.stdout.write(f"{path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
