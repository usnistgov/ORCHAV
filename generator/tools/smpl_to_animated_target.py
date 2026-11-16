"""Convert user-supplied SMPL surfaces or AMASS motion into an animated asset.

The numeric surface route needs only NumPy. The AMASS route loads a separately
obtained SMPL model through the optional ``smplx``/Torch environment. Model
pickle loading requires the explicit ``--trusted-model`` option; motion NPZ
files never permit object deserialization.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from shared.animated_target_export import (
    canonical_quaternions,
    export_animated_target,
    load_motion_source_metadata,
    load_numeric_npz,
    rotation_matrices,
    sample_times,
    selected_indices,
)


def axis_angle_quaternions(axis_angles: Any) -> np.ndarray:
    """Convert finite root axis-angle vectors to canonical active XYZW."""
    values = np.asarray(axis_angles, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("root axis angles must have finite shape (frames, 3)")
    angles = np.linalg.norm(values, axis=1)
    if not np.all(np.isfinite(angles)):
        raise ValueError("root axis-angle magnitude is not finite")
    scale = 0.5 * np.sinc(angles / (2 * np.pi))
    return canonical_quaternions(np.column_stack((values * scale[:, None], np.cos(angles / 2))))


def _source_rate(data: dict[str, np.ndarray], source_fps: float | None) -> float | None:
    rates = []
    for key in ("mocap_framerate", "mocap_frame_rate", "frame_rate"):
        if key in data:
            value = np.asarray(data[key])
            if value.shape != () or value.dtype.kind not in "iuf":
                raise ValueError(f"{key} must be one numeric scalar")
            rates.append(float(value))
    if source_fps is not None:
        rates.append(source_fps)
    if rates and (
        not all(np.isfinite(rate) and rate > 0 for rate in rates)
        or not np.allclose(rates, rates[0], rtol=1e-9, atol=1e-12)
    ):
        raise ValueError("source frame-rate fields must be positive and agree")
    return rates[0] if rates else None


def export_smpl_surfaces(
    surface_npz: str | Path,
    output_dir: str | Path,
    *,
    asset_id: str,
    geometry_space: str,
    source_fps: float | None = None,
    stride: int = 1,
    source_metadata: str | Path | None = None,
    front_axis: str,
    up_axis: str,
    root_local_front_axis: str | None = None,
    root_local_up_axis: str | None = None,
    closed_surface: bool = False,
    out_of_range: str = "error",
) -> Path:
    """Adapt numeric deformed geometry with an explicit root ownership policy.

    NPZ fields: vertices[N,V,3], faces[F,3], root_translation[N,3],
    root_rotation_xyzw[N,4], optional time_s[N] or a source frame-rate scalar.
    ``world`` removes the supplied root transform once; ``root_local`` already
    excludes it. Optional paired body-local axes re-express geometry and
    compensate root rotations while preserving world animation. Without those
    axes, root-local vertices remain unchanged. Neither mode centers, grounds,
    removes world orientation or fits duration. Coordinates and roots use meters.
    """
    data, record = load_numeric_npz(surface_npz)
    if geometry_space not in ("world", "root_local"):
        raise ValueError("geometry_space must be world or root_local")
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError("vertices must have shape (frames, vertices, 3)")
    roots = np.asarray(data["root_translation"], dtype=np.float64)
    rotations = canonical_quaternions(data["root_rotation_xyzw"])
    if roots.shape != (len(vertices), 3) or len(rotations) != len(vertices):
        raise ValueError("surface pose and root arrays must have matching frame counts")
    times = sample_times(
        len(vertices), times_s=data.get("time_s"), source_fps=_source_rate(data, source_fps)
    )
    selected = selected_indices(len(vertices), stride)
    if geometry_space == "world":
        vertices = np.einsum(
            "nvi,nij->nvj", vertices - roots[:, None], rotation_matrices(rotations)
        )
    return export_animated_target(
        output_dir,
        asset_id=asset_id,
        vertices=vertices[selected],
        faces=data["faces"],
        times_s=times[selected],
        root_translation=roots[selected],
        root_rotation_xyzw=rotations[selected],
        source_type="smpl_surface",
        source_records={
            "surface_npz": record,
            **(
                {"source_metadata": load_motion_source_metadata(source_metadata)}
                if source_metadata
                else {}
            ),
        },
        adapter_metadata={
            "adapter": "orchav.smpl_to_animated_target",
            "version": 1,
            "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "geometry_space": geometry_space,
            "stride": stride,
            "selected_source_indices": selected.tolist(),
            "centering": "none",
            "grounding": "none",
            "orientation_removal": "none",
        },
        front_axis=front_axis,
        up_axis=up_axis,
        root_local_front_axis=root_local_front_axis,
        root_local_up_axis=root_local_up_axis,
        closed_surface=closed_surface,
        out_of_range=out_of_range,
    )


def export_amass_motion(
    amass_npz: str | Path,
    model_file: str | Path,
    output_dir: str | Path,
    *,
    asset_id: str,
    trusted_model: bool = False,
    source_fps: float | None = None,
    stride: int = 1,
    source_metadata: str | Path | None = None,
    front_axis: str,
    up_axis: str,
    closed_surface: bool = False,
    out_of_range: str = "error",
) -> Path:
    """Evaluate a trusted, separately obtained SMPL model at native AMASS times.

    Supports SMPL 72-parameter poses and AMASS SMPL-H 156-parameter poses.
    The latter retain 21 body joints and record that finger articulation is
    unavailable in the SMPL model. No implicit pose interpolation is performed.
    """
    if not trusted_model:
        raise ValueError("SMPL model loading requires --trusted-model for the supplied model file")
    data, motion_record = load_numeric_npz(amass_npz)
    poses = np.asarray(data["poses"], dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] not in (72, 156) or not np.all(np.isfinite(poses)):
        raise ValueError("AMASS poses must be finite (frames, 72) or (frames, 156)")
    trans = np.asarray(data["trans"], dtype=np.float64)
    betas = np.asarray(data["betas"], dtype=np.float64)
    if trans.shape != (len(poses), 3) or not np.all(np.isfinite(trans)):
        raise ValueError("AMASS trans must have finite shape (frames, 3)")
    if betas.ndim == 2 and betas.shape[0] == 1:
        betas = betas[0]
    if betas.ndim != 1 or len(betas) < 10 or not np.all(np.isfinite(betas)):
        raise ValueError("AMASS betas must contain at least ten finite coefficients")
    times = sample_times(
        len(poses), times_s=data.get("time_s"), source_fps=_source_rate(data, source_fps)
    )
    selected = selected_indices(len(poses), stride)
    raw_gender = np.asarray(data["gender"])
    if raw_gender.shape != ():
        raise ValueError("gender must be a scalar string")
    gender = raw_gender.item()
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    if gender not in ("neutral", "male", "female"):
        raise ValueError("gender must be neutral, male, or female")
    model_path = Path(model_file).resolve(strict=True)
    model_bytes = model_path.read_bytes()
    model_record = {
        "sha256": hashlib.sha256(model_bytes).hexdigest(),
        "size_bytes": len(model_bytes),
        "trusted_pickle": True,
    }
    try:
        import smplx
        import torch
    except ImportError as exc:
        raise RuntimeError(
            'AMASS conversion requires the optional environment: pip install "orchav[smpl]"'
        ) from exc
    model = smplx.SMPL(model_path=str(model_path), gender=gender, batch_size=1, num_betas=10)
    model.requires_grad_(False)
    model = model.to("cpu")
    # Authenticate the exact path again after the source library consumed it.
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != model_record["sha256"]:
        raise ValueError("SMPL model changed while being loaded")
    quaternions = axis_angle_quaternions(poses[selected, :3])
    matrices = rotation_matrices(quaternions)
    local_vertices, root_positions = [], []
    with torch.no_grad():
        for index, matrix in zip(selected, matrices):
            body = (
                poses[index, 3:72] if poses.shape[1] == 72 else np.pad(poses[index, 3:66], (0, 6))
            )
            output = model(
                betas=torch.tensor(betas[:10][None], dtype=torch.float32),
                body_pose=torch.tensor(body[None], dtype=torch.float32),
                global_orient=torch.tensor(poses[index, :3][None], dtype=torch.float32),
                transl=torch.tensor(trans[index][None], dtype=torch.float32),
                return_verts=True,
            )
            world = output.vertices.detach().cpu().numpy()[0].astype(np.float64)
            root = output.joints.detach().cpu().numpy()[0, 0].astype(np.float64)
            local_vertices.append((world - root) @ matrix)
            root_positions.append(root)
    return export_animated_target(
        output_dir,
        asset_id=asset_id,
        vertices=np.asarray(local_vertices),
        faces=np.asarray(model.faces),
        times_s=times[selected],
        root_translation=np.asarray(root_positions),
        root_rotation_xyzw=quaternions,
        source_type="smpl_amass",
        source_records={
            "motion_npz": motion_record,
            "smpl_model": model_record,
            **(
                {"source_metadata": load_motion_source_metadata(source_metadata)}
                if source_metadata
                else {}
            ),
        },
        adapter_metadata={
            "adapter": "orchav.smpl_to_animated_target",
            "version": 1,
            "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_pose_parameters": poses.shape[1],
            "body_model": "SMPL",
            "gender": gender,
            "shape_coefficients_used": 10,
            "shape_coefficients_supplied": len(betas),
            "shape_coefficients_omitted": max(0, len(betas) - 10),
            "stride": stride,
            "selected_source_indices": selected.tolist(),
            "finger_pose_policy": (
                "SMPL-H fingers omitted; two SMPL hand joints zero"
                if poses.shape[1] == 156
                else "native SMPL pose"
            ),
            "root_reference": "evaluated pelvis joint",
            "centering": "root decomposition only",
            "grounding": "none",
            "orientation_removal": "none",
            "pose_interpolation": "none",
        },
        front_axis=front_axis,
        up_axis=up_axis,
        # SMPL's documented rest-body frame differs from AMASS world axes.
        # The shared exporter compensates the root so motion stays unchanged.
        root_local_front_axis="+Z",
        root_local_up_axis="+Y",
        closed_surface=closed_surface,
        out_of_range=out_of_range,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--surface-npz", type=Path)
    inputs.add_argument("--amass-npz", type=Path)
    parser.add_argument(
        "--model-file", type=Path, help="Separately obtained SMPL model file for AMASS input"
    )
    parser.add_argument(
        "--trusted-model",
        action="store_true",
        help="Explicitly trust deserialization of the supplied SMPL model",
    )
    parser.add_argument(
        "--geometry-space",
        choices=("world", "root_local"),
        help="Required ownership declaration for --surface-npz",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument(
        "--source-fps",
        type=float,
        help="Required when input contains no physical timestamps or frame rate",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--source-metadata",
        type=Path,
        help="Optional JSON of source IDs, revision/hash fields, seed and generator_version",
    )
    parser.add_argument(
        "--front-axis",
        required=True,
        help="Declared source-world forward axis (+X/-X/+Y/-Y/+Z/-Z)",
    )
    parser.add_argument("--up-axis", required=True, help="Declared source-world upward axis")
    parser.add_argument(
        "--root-local-front-axis",
        help="For numeric surfaces: body-local forward axis before the supplied root rotation",
    )
    parser.add_argument(
        "--root-local-up-axis",
        help="For numeric surfaces: body-local up axis; supply together with --root-local-front-axis",
    )
    parser.add_argument("--closed-surface", action="store_true")
    parser.add_argument("--out-of-range", choices=("error", "hold"), default="error")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    common = {
        key: getattr(args, key)
        for key in (
            "asset_id",
            "source_fps",
            "stride",
            "source_metadata",
            "front_axis",
            "up_axis",
            "closed_surface",
            "out_of_range",
        )
    }
    try:
        if args.surface_npz:
            if args.geometry_space is None or args.model_file or args.trusted_model:
                parser.error(
                    "--surface-npz requires --geometry-space and does not use model options"
                )
            path = export_smpl_surfaces(
                args.surface_npz,
                args.output_dir,
                geometry_space=args.geometry_space,
                root_local_front_axis=args.root_local_front_axis,
                root_local_up_axis=args.root_local_up_axis,
                **common,
            )
        else:
            if args.root_local_front_axis is not None or args.root_local_up_axis is not None:
                parser.error(
                    "AMASS uses the SMPL model's known root-local basis; local-axis overrides are for --surface-npz"
                )
            if args.model_file is None or args.geometry_space:
                parser.error("--amass-npz requires --model-file and does not use --geometry-space")
            path = export_amass_motion(
                args.amass_npz,
                args.model_file,
                args.output_dir,
                trusted_model=args.trusted_model,
                **common,
            )
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    sys.stdout.write(f"{path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
