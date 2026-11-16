"""Prepare explicitly sampled SMPL articulation with continuous root approach.

User-supplied AMASS and SMPL inputs remain outside public releases. This helper
uses ORCHAV's existing SMPL SLERP evaluator and animated-target exporter; it does
not synthesize additional motion information or change the sensing estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _linear_root_motion(
    times_s: np.ndarray,
    *,
    floor_offset_m: float,
    approach_speed_m_s: float,
    start_range_m: float,
) -> np.ndarray:
    """Place a continuous approach in the manifest's -Y-front source basis.

    The asset basis maps source -Y to world +X, so increasing source Y gives
    world velocity (-speed, 0, 0). Surface articulation never enters this root.
    """
    times = np.asarray(times_s, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.all(np.isfinite(times))
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Root times must start at zero and increase strictly")
    if not np.isfinite(approach_speed_m_s) or approach_speed_m_s < 0:
        raise ValueError("Approach speed must be finite and nonnegative")
    if not np.isfinite(start_range_m) or start_range_m <= 0:
        raise ValueError("Start range must be finite and positive")
    if not np.isfinite(floor_offset_m):
        raise ValueError("The fixed floor offset must be finite")
    ranges = start_range_m - approach_speed_m_s * times
    if ranges[-1] <= 0:
        raise ValueError("The approach must end before the root reaches the radar")
    return np.column_stack((np.zeros(len(times)), -ranges, np.full(len(times), floor_offset_m)))


def prepare(
    motion_path: Path,
    model_folder: Path,
    output: Path,
    *,
    samples: int = 8192,
    sample_rate_hz: float = 4000.0,
    approach_speed_m_s: float = 1.4,
    start_range_m: float = 8.0,
    trusted_model: bool = False,
) -> Path:
    """Evaluate one pose per pulse and apply authored translation exactly once."""
    if not trusted_model:
        raise ValueError("--trusted-model is required to load the user-supplied SMPL pickle")
    if samples < 2 or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("At least two samples and a positive sample rate are required")
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = {key: archive[key] for key in archive.files}
    rate = float(motion["mocap_framerate"])
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("The source sample rate must be finite and positive")
    gender = str(motion["gender"].item()).lower()
    if gender not in {"male", "female", "neutral"}:
        raise ValueError("The source must declare a supported SMPL gender")
    model_path = model_folder / "smpl" / f"SMPL_{gender.upper()}.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    times = np.arange(samples, dtype=float) / sample_rate_hz
    if times[-1] > (len(motion["poses"]) - 1) / rate:
        raise ValueError("Requested output extends beyond the source motion samples")
    from generator.tools.smpl_to_ply import compute_frame_indices
    from generator.tools.smpl_to_ply import main as evaluate_smpl

    source_indices = compute_frame_indices(
        total_frames=len(motion["poses"]),
        src_fps=rate,
        tgt_fps=sample_rate_hz,
        start_frame=0,
        end_frame=None,
        max_frames=samples,
    )
    # The shared evaluator preserves integer source frames when downsampling
    # or using a near-native rate. Reject grids whose actual pose times differ
    # from this experiment's declared pulse times, before loading the model.
    if len(source_indices) != samples or not np.allclose(
        source_indices / rate, times, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "The SMPL evaluator cannot match the requested uniform pulse times at this "
            "sample rate; choose the source rate, an exact integer downsampling ratio, "
            "or an interpolated rate above the source rate"
        )
    # Validate the full trajectory before creating outputs or loading the model.
    roots = _linear_root_motion(
        times,
        floor_offset_m=0.0,
        approach_speed_m_s=approach_speed_m_s,
        start_range_m=start_range_m,
    )
    output.mkdir(parents=True, exist_ok=False)
    normalized_motion = output / "root_local_motion.npz"
    # This is an explicit experimental intervention: retain joint articulation,
    # remove source root travel, and author one continuous approach separately.
    motion["trans"] = np.zeros_like(motion["trans"])
    np.savez(normalized_motion, **motion)

    meshes = output / "meshes"
    evaluate_smpl(
        [
            "--amass-npz",
            str(normalized_motion),
            "--model-folder",
            str(model_folder),
            "--output-dir",
            str(meshes),
            "--name",
            "march",
            "--gender",
            gender,
            "--fps",
            str(sample_rate_hz),
            "--max-frames",
            str(samples),
            "--strip-global-orient",
            "--center-xy",
            "--xy-anchor",
            "pelvis",
            "--no-ground-feet",
        ]
    )
    import trimesh

    from shared.animated_target_export import export_animated_target

    paths = sorted(meshes.glob("march_*.ply"))
    if len(paths) != samples:
        raise ValueError(f"Expected {samples} explicit surfaces, found {len(paths)}")
    vertices, faces = [], None
    for path in paths:
        mesh = trimesh.load(path, process=False)
        current_faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces is None:
            faces = current_faces
        elif not np.array_equal(faces, current_faces):
            raise ValueError("SMPL surfaces changed ordered topology")
        vertices.append(np.asarray(mesh.vertices, dtype=np.float64))
    poses = np.asarray(vertices)
    # Release the per-mesh arrays before the exporter authenticates the asset.
    del vertices
    # smpl_to_ply emits -Y-front, +Z-up geometry. Its front axis maps to +X.
    # A fixed 180-degree yaw then faces the person back toward a sensor at x=0.
    roots[:, 2] = -float(poses[0, :, 2].min())
    rotations = np.repeat([[0.0, 0.0, 1.0, 0.0]], samples, axis=0)
    source_records = {
        "amass_motion": {"sha256": _sha256(motion_path), "size_bytes": motion_path.stat().st_size},
        "smpl_model": {"sha256": _sha256(model_path), "size_bytes": model_path.stat().st_size},
        "root_local_motion": {
            "sha256": _sha256(normalized_motion),
            "size_bytes": normalized_motion.stat().st_size,
        },
    }
    operations = {
        "source_sample_rate_hz": rate,
        "output_sample_rate_hz": sample_rate_hz,
        "output_sample_count": samples,
        "source_interval_s": [0.0, float(times[-1])],
        "joint_interpolation": "shortest-quaternion SLERP in existing smpl_to_ply",
        "source_translation": "set to zero in a new local input copy",
        "source_global_orientation": "removed before body evaluation",
        "horizontal_anchor": "pelvis; no per-frame AABB recentering",
        "grounding": "one fixed root Z offset derived from first surface minimum Z",
        "root_translation_source_m": roots[0].tolist(),
        "root_translation_final_source_m": roots[-1].tolist(),
        "authored_root_trajectory": {
            "model": "constant_velocity_linear_approach",
            "start_world_x_m": start_range_m,
            "end_world_x_m": float(start_range_m - approach_speed_m_s * times[-1]),
            "world_y_m": 0.0,
            "world_z_m": float(roots[0, 2]),
            "approach_speed_m_s": approach_speed_m_s,
            "world_velocity_m_s": [-approach_speed_m_s, 0.0, 0.0],
            "translation_owner": "animated_target_manifest_only",
        },
        "root_rotation_xyzw": rotations[0].tolist(),
        "cyclic": False,
        "new_independent_motion_samples": False,
        "body_model_mapping": "SMPL-H first21 body joints; two SMPL hand joints zero; finger articulation omitted",
        "shape_coefficients_used": 10,
        "dynamic_soft_tissue_coefficients": "not evaluated by the SMPL exporter",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "smplx", "torch", "trimesh")
        },
        "smpl_evaluator_sha256": _sha256(Path("generator/tools/smpl_to_ply.py")),
        "preparation_source_sha256": _sha256(Path(__file__)),
    }
    manifest = export_animated_target(
        output / "asset",
        asset_id="person:marching:linear_approach",
        vertices=poses,
        faces=faces,
        times_s=times,
        root_translation=roots,
        root_rotation_xyzw=rotations,
        source_type="smpl_interpolated_linear_root_control",
        source_records=source_records,
        adapter_metadata=operations,
        front_axis="-Y",
        up_axis="+Z",
        out_of_range="error",
    )
    (output / "preparation.json").write_text(
        json.dumps(
            {
                "sources": source_records,
                "operations": operations,
                "manifest_sha256": _sha256(manifest),
            },
            indent=2,
        )
        + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amass-npz", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--sample-rate", type=float, default=4000.0)
    parser.add_argument(
        "--approach-speed", type=float, default=1.4, help="Root speed toward the radar (m/s)"
    )
    parser.add_argument(
        "--start-range", type=float, default=8.0, help="Initial world X coordinate (m)"
    )
    parser.add_argument("--trusted-model", action="store_true")
    args = parser.parse_args()
    print(
        prepare(
            args.amass_npz,
            args.model_folder,
            args.output,
            samples=args.samples,
            sample_rate_hz=args.sample_rate,
            approach_speed_m_s=args.approach_speed,
            start_range_m=args.start_range,
            trusted_model=args.trusted_model,
        )
    )


if __name__ == "__main__":
    main()
