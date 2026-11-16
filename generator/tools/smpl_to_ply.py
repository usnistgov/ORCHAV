#!/usr/bin/env python3
"""
SMPL/SMPL-X Body Model to PLY Mesh Sequence Converter

Converts AMASS motion capture data (.npz) into numbered PLY mesh files
compatible with ORCHAV's TargetConfig mesh target system.

Requirements:
    pip install smplx trimesh torch

License note:
    This script is part of ORCHAV and does NOT bundle any SMPL model data.
    SMPL model files must be downloaded separately from https://smpl.is.tue.mpg.de/
    and are subject to their own license terms (MPI license, academic use).
    AMASS motion data from https://amass.is.tue.mpg.de/ has its own license terms.
    Users are responsible for complying with the respective licenses.

Usage:
    python -m generator.tools.smpl_to_ply \\
        --amass-npz motions/CMU/01/01_01_poses.npz \\
        --model-folder models/ \\
        --output-dir libraries/targets/smpl_walk_01 \\
        --fps 10.0

Output PLY files use binary little-endian format with float vertex colors,
matching the format expected by Mitsuba/Sionna RT (same as
``generator.core.target.mesh._write_float_vertex_color_ply``).
"""

import argparse
import inspect
import logging
import struct
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# SMPL uses Y-up; ORCHAV uses Z-up.
# Transform: new_x = x, new_y = -z, new_z = y
_Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _install_chumpy_compat() -> None:
    """Install runtime aliases needed by old SMPL v1.0.0 chumpy pickles.

    SMPL for Python v1.0.0 stores model arrays in pickles that import chumpy.
    Chumpy still references ``inspect.getargspec`` and old NumPy scalar aliases
    removed by current Python/NumPy releases. Patching those names before
    importing smplx lets smplx load the licensed model files without modifying
    them in place.
    """
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]

    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def check_dependencies() -> None:
    """Verify that smplx, trimesh, and torch are installed."""
    missing = []
    for pkg in ("smplx", "trimesh", "torch"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        pkgs = " ".join(missing)
        logger.error(
            "Missing dependencies: %s\n"
            "Install with:  pip install %s\n"
            "Or:            pip install -e '.[smpl]'",
            pkgs,
            pkgs,
        )
        sys.exit(1)


def load_amass_motion(npz_path: Path) -> dict:
    """Load an AMASS .npz motion file and return a normalised dict.

    AMASS sub-datasets use slightly different key names. This function
    normalises them to a consistent set:
        poses, betas, trans, gender, mocap_framerate

    Returns:
        Dict with keys: poses (N, J*3), betas (1, B) or (B,),
        trans (N, 3), gender (str), mocap_framerate (float).
    """
    with np.load(npz_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}

    # Normalise pose key: 'poses' or 'pose_body' + 'root_orient'
    if "poses" not in data:
        if "pose_body" in data:
            root = data.get("root_orient", np.zeros((len(data["pose_body"]), 3)))
            data["poses"] = np.concatenate([root, data["pose_body"]], axis=-1)
        else:
            raise KeyError(
                f"Cannot find pose data in {npz_path}. " f"Available keys: {sorted(data.keys())}"
            )

    # Normalise translation
    if "trans" not in data:
        data["trans"] = np.zeros((len(data["poses"]), 3))

    # Normalise betas
    if "betas" not in data:
        data["betas"] = np.zeros(10)

    poses = np.asarray(data["poses"])
    trans = np.asarray(data["trans"])
    betas = np.asarray(data["betas"])
    if (
        poses.ndim != 2
        or poses.shape[0] < 1
        or poses.shape[1] < 3
        or poses.shape[1] % 3
        or poses.dtype.kind not in "iuf"
        or not np.all(np.isfinite(poses))
    ):
        raise ValueError("poses must be a nonempty finite numeric (frames, joints*3) matrix")
    if (
        trans.shape != (len(poses), 3)
        or trans.dtype.kind not in "iuf"
        or not np.all(np.isfinite(trans))
    ):
        raise ValueError("trans must have finite numeric shape (frames, 3)")
    if betas.dtype.kind not in "iuf" or not np.all(np.isfinite(betas)):
        raise ValueError("betas must contain finite numeric coefficients")

    # Normalise gender
    gender = data.get("gender", "neutral")
    if isinstance(gender, np.ndarray):
        if gender.shape != ():
            raise ValueError("gender must be a scalar string")
        gender = gender.item()
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    data["gender"] = str(gender).lower()
    if data["gender"] not in ("neutral", "male", "female"):
        raise ValueError("gender must be neutral, male, or female")

    # Normalise framerate
    for key in ("mocap_framerate", "mocap_frame_rate", "frame_rate"):
        if key in data:
            rate = data[key]
            if isinstance(rate, np.ndarray):
                if rate.shape != () or rate.dtype.kind not in "iuf":
                    raise ValueError(f"{key} must be one numeric scalar")
                rate = float(rate.item())
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError(f"{key} must be finite and positive")
            data["mocap_framerate"] = float(rate)
            break
    if "mocap_framerate" not in data:
        raise ValueError(
            "Motion has no physical source frame rate; supply mocap_framerate, "
            "mocap_frame_rate, or frame_rate in the numeric NPZ"
        )

    return data


def create_body_model(
    model_folder: str,
    model_type: str = "smpl",
    gender: str = "neutral",
    num_betas: int = 10,
):
    """Create an SMPL/SMPL-X body model.

    Args:
        model_folder: Path to directory containing SMPL model files.
        model_type: ``"smpl"`` or ``"smplx"``.
        gender: ``"neutral"``, ``"male"``, or ``"female"``.
        num_betas: Number of shape coefficients.

    Returns:
        smplx body model instance.
    """
    _install_chumpy_compat()

    import smplx
    import torch

    model_path = Path(model_folder)
    if not model_path.exists():
        logger.error(
            "Model folder not found: %s\n"
            "Download SMPL models from https://smpl.is.tue.mpg.de/\n"
            "Expected structure:\n"
            "  %s/\n"
            "    SMPL_NEUTRAL.pkl  (or smpl/ subfolder with *.pkl)",
            model_folder,
            model_folder,
        )
        sys.exit(1)

    model = smplx.create(
        model_path=str(model_path),
        model_type=model_type,
        gender=gender,
        num_betas=num_betas,
        batch_size=1,
    )
    model.requires_grad_(False)

    device = torch.device("cpu")
    model = model.to(device)
    return model


def compute_frame_indices(
    total_frames: int,
    src_fps: float,
    tgt_fps: float | None,
    start_frame: int,
    end_frame: int | None,
    max_frames: int | None,
) -> np.ndarray:
    """Compute which source frames to extract (supports up- and down-sampling).

    When ``tgt_fps > src_fps``, returns **float** indices representing
    fractional positions between source frames.  The caller must
    interpolate between adjacent source frames for these fractional values.

    Args:
        total_frames: Total frames in source motion.
        src_fps: Source mocap framerate.
        tgt_fps: Target output framerate (None = use source rate).
        start_frame: First frame index.
        end_frame: Last frame index (exclusive, None = all).
        max_frames: Cap on number of output frames.

    Returns:
        1-D array of source frame indices (int for native/downsample,
        float64 for upsample).
    """
    if end_frame is None:
        end_frame = total_frames
    end_frame = min(end_frame, total_frames)

    if tgt_fps is None or abs(tgt_fps - src_fps) < 0.01:
        # Native rate — emit all frames
        indices = np.arange(start_frame, end_frame)
    else:
        step = src_fps / tgt_fps
        indices = np.arange(start_frame, end_frame - 1 + 1e-9, step)
        if tgt_fps < src_fps:
            indices = indices.astype(int)
        # else: keep as float for interpolation

    if max_frames is not None:
        indices = indices[:max_frames]

    return indices


def _axis_angle_to_quaternion(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3,) to unit quaternion (4,) [w, x, y, z]."""
    angle = np.linalg.norm(aa)
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / angle
    half = angle / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


def _quaternion_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion (4,) [w, x, y, z] to axis-angle (3,)."""
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(abs(w))
    if angle < 1e-10:
        return np.zeros(3)
    axis = q[1:4]
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return np.zeros(3)
    axis = axis / axis_norm
    if w < 0:
        axis = -axis
    return axis * angle


def _slerp_quaternion(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        # Nearly identical — use linear interpolation
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w0 = np.sin((1.0 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta
    return w0 * q0 + w1 * q1


def interpolate_smpl_params(
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    trans_a: np.ndarray,
    trans_b: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate SMPL pose (via SLERP) and translation (linear).

    Each pose is a flat array of concatenated axis-angle rotations
    (groups of 3).  Each 3-element group is converted to a quaternion,
    SLERPed, and converted back.

    Args:
        pose_a: Pose at frame A (N*3,).
        pose_b: Pose at frame B (N*3,).
        trans_a: Translation at frame A (3,).
        trans_b: Translation at frame B (3,).
        t: Interpolation weight in [0, 1].

    Returns:
        (interpolated_pose, interpolated_translation).
    """
    n_joints = len(pose_a) // 3
    interp_pose = np.zeros_like(pose_a)

    for j in range(n_joints):
        aa_a = pose_a[j * 3 : j * 3 + 3]
        aa_b = pose_b[j * 3 : j * 3 + 3]
        q_a = _axis_angle_to_quaternion(aa_a)
        q_b = _axis_angle_to_quaternion(aa_b)
        q_interp = _slerp_quaternion(q_a, q_b, t)
        interp_pose[j * 3 : j * 3 + 3] = _quaternion_to_axis_angle(q_interp)

    interp_trans = trans_a + t * (trans_b - trans_a)
    return interp_pose, interp_trans


def generate_mesh_with_joints(
    model,
    betas,
    body_pose,
    global_orient,
    transl,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a single-frame SMPL forward pass.

    Args:
        model: smplx body model.
        betas: Shape parameters (1, B) tensor.
        body_pose: Body pose (1, J*3) tensor.
        global_orient: Root orientation (1, 3) tensor.
        transl: Translation (1, 3) tensor.

    Returns:
        ``(vertices, faces, joints)`` as numpy arrays.
        vertices shape: (V, 3), faces shape: (F, 3), joints shape: (J, 3).
    """
    output = model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        return_verts=True,
    )
    vertices = output.vertices.detach().cpu().numpy().squeeze(0).astype(np.float64)
    joints = output.joints.detach().cpu().numpy().squeeze(0).astype(np.float64)
    faces = model.faces.astype(np.int32) if isinstance(model.faces, np.ndarray) else model.faces
    faces = np.asarray(faces, dtype=np.int32)
    return vertices, faces, joints


def generate_mesh(
    model,
    betas,
    body_pose,
    global_orient,
    transl,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a single-frame SMPL forward pass and return vertices + faces only."""
    vertices, faces, _joints = generate_mesh_with_joints(
        model,
        betas,
        body_pose,
        global_orient,
        transl,
    )
    return vertices, faces


def transform_points_to_orchav(points: np.ndarray) -> np.ndarray:
    """Convert arbitrary SMPL Y-up points to ORCHAV Z-up coordinates."""
    return np.asarray(points, dtype=np.float64) @ _Y_UP_TO_Z_UP.T


def transform_to_orchav(
    vertices: np.ndarray,
    center_xy: bool = True,
    ground_feet: bool = True,
    xy_reference: np.ndarray | None = None,
) -> np.ndarray:
    """Convert SMPL Y-up coordinates to ORCHAV Z-up coordinates.

    Transform: ``(x, y, z) -> (x, -z, y)``

    Then optionally center XY at origin and translate Z so feet sit at z=0.

    Args:
        vertices: (V, 3) array in SMPL Y-up space.
        center_xy: Center the mesh XY at origin.
        ground_feet: Translate Z so minimum z = 0.
        xy_reference: Optional (3,) reference point in SMPL coordinates to use
            for XY centering instead of the per-frame vertex centroid.

    Returns:
        (V, 3) array in ORCHAV Z-up space.
    """
    verts = transform_points_to_orchav(vertices).copy()

    if center_xy:
        if xy_reference is None:
            verts[:, 0] -= verts[:, 0].mean()
            verts[:, 1] -= verts[:, 1].mean()
        else:
            ref_xy = transform_points_to_orchav(np.asarray(xy_reference).reshape(1, 3))[0, :2]
            verts[:, 0] -= ref_xy[0]
            verts[:, 1] -= ref_xy[1]

    if ground_feet:
        verts[:, 2] -= verts[:, 2].min()

    return verts


def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals from mesh topology.

    Args:
        vertices: (V, 3) vertex positions.
        faces: (F, 3) triangle indices.

    Returns:
        (V, 3) normalised vertex normals.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)

    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return vertex_normals / norms


def write_ply_binary(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    color: Sequence[float] = (0.72, 0.62, 0.52),
) -> None:
    """Write a binary PLY file with float vertex colors.

    Format matches ``generator.core.target.mesh._write_float_vertex_color_ply``:
    vertices as ``<6d3f`` (6 doubles + 3 floats), faces as ``<B3I``.

    Args:
        path: Output file path.
        vertices: (V, 3) float64 positions.
        faces: (F, 3) int32 triangle indices.
        normals: (V, 3) float64 vertex normals.
        color: RGB tuple in [0, 1] range.
    """
    n_verts = len(vertices)
    n_faces = len(faces)

    r, g, b = (np.float32(c) for c in color)

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment Generated by ORCHAV smpl_to_ply",
            f"element vertex {n_verts}",
            "property double x",
            "property double y",
            "property double z",
            "property double nx",
            "property double ny",
            "property double nz",
            "property float red",
            "property float green",
            "property float blue",
            f"element face {n_faces}",
            "property list uchar uint vertex_indices",
            "end_header\n",
        ]
    ).encode("ascii")

    with open(path, "wb") as handle:
        handle.write(header)
        for vert, norm in zip(vertices, normals):
            handle.write(
                struct.pack(
                    "<6d3f",
                    float(vert[0]),
                    float(vert[1]),
                    float(vert[2]),
                    float(norm[0]),
                    float(norm[1]),
                    float(norm[2]),
                    float(r),
                    float(g),
                    float(b),
                )
            )
        for face in faces:
            handle.write(
                struct.pack(
                    "<B3I",
                    3,
                    int(face[0]),
                    int(face[1]),
                    int(face[2]),
                )
            )


def detect_gait_cycle(
    all_vertices: list[np.ndarray],
    min_cycle_frames: int = 4,
) -> int | None:
    """Find the best loop point for cyclic gait extraction.

    Compares each frame with frame 0 (skipping the first few frames)
    and returns the index where the pose best matches frame 0, i.e.
    the end of one complete gait cycle.

    Args:
        all_vertices: List of (V, 3) vertex arrays, one per frame.
        min_cycle_frames: Minimum cycle length to consider.

    Returns:
        Frame index of the best loop point (exclusive end), or None
        if no good cycle is found.
    """
    if len(all_vertices) < min_cycle_frames + 1:
        return None

    ref = all_vertices[0]
    rms_distances = []
    for i in range(1, len(all_vertices)):
        d = np.sqrt(((all_vertices[i] - ref) ** 2).sum(axis=1).mean())
        rms_distances.append(d)

    # Find the minimum RMS after the minimum cycle length
    search_start = min_cycle_frames - 1  # 0-indexed in rms_distances
    if search_start >= len(rms_distances):
        return None

    search_slice = rms_distances[search_start:]
    best_local_idx = int(np.argmin(search_slice))
    best_idx = search_start + best_local_idx
    best_frame = best_idx + 1  # +1 because rms_distances is offset by 1

    best_rms = rms_distances[best_idx]
    avg_consecutive = np.mean(
        [
            np.sqrt(((all_vertices[i + 1] - all_vertices[i]) ** 2).sum(axis=1).mean())
            for i in range(len(all_vertices) - 1)
        ]
    )

    logger.info(
        "Cycle detection: best loop at frame %d (RMS=%.4f m, avg step=%.4f m, ratio=%.1fx)",
        best_frame,
        best_rms,
        avg_consecutive,
        best_rms / avg_consecutive if avg_consecutive > 0 else float("inf"),
    )

    # Accept if the loop discontinuity is within 2x of normal frame-to-frame motion
    if best_rms > avg_consecutive * 2.0:
        logger.warning(
            "No clean cycle found (loop RMS %.1fx avg step). "
            "Try generating more frames or a different motion clip.",
            best_rms / avg_consecutive,
        )
        return None

    return best_frame


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert AMASS motion data to PLY mesh sequences via SMPL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Basic walk sequence at 10 fps\n"
            "  python -m generator.tools.smpl_to_ply \\\n"
            "      --amass-npz motions/CMU/01/01_01_poses.npz \\\n"
            "      --model-folder models/ \\\n"
            "      --output-dir libraries/targets/smpl_walk \\\n"
            "      --fps 10.0\n"
            "\n"
            "  # First 50 frames, male body, custom color\n"
            "  python -m generator.tools.smpl_to_ply \\\n"
            "      --amass-npz motions/CMU/02/02_01_poses.npz \\\n"
            "      --model-folder models/ \\\n"
            "      --gender male --max-frames 50 \\\n"
            "      --color 0.6 0.5 0.4\n"
        ),
    )

    parser.add_argument(
        "--amass-npz",
        required=True,
        type=Path,
        help="Path to AMASS .npz motion file",
    )
    parser.add_argument(
        "--model-folder",
        required=True,
        type=Path,
        help="Parent directory containing smpl/ or smplx/ model subfolder",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: libraries/targets/smpl_<name>/)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Filename prefix (default: derived from npz filename)",
    )
    parser.add_argument(
        "--model-type",
        choices=["smpl", "smplx"],
        default="smpl",
        help="Body model type (default: smpl)",
    )
    parser.add_argument(
        "--gender",
        choices=["neutral", "male", "female", "auto"],
        default="neutral",
        help="Body gender (default: neutral, auto=read from npz)",
    )
    parser.add_argument(
        "--num-betas",
        type=int,
        default=10,
        help="Number of shape coefficients (default: 10)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target output frame rate (default: source mocap rate)",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First source frame index (default: 0)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last source frame index, exclusive (default: all)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of output frames",
    )
    parser.add_argument(
        "--color",
        type=float,
        nargs=3,
        default=[0.72, 0.62, 0.52],
        metavar=("R", "G", "B"),
        help="Vertex color in 0-1 range (default: 0.72 0.62 0.52)",
    )
    parser.add_argument(
        "--center-xy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center mesh XY at origin (default: yes)",
    )
    parser.add_argument(
        "--xy-anchor",
        choices=["centroid", "pelvis"],
        default="centroid",
        help=(
            "When centering XY, subtract the per-frame mesh centroid " "(default) or pelvis joint"
        ),
    )
    parser.add_argument(
        "--ground-feet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Translate Z so feet sit at z=0 (default: yes)",
    )
    parser.add_argument(
        "--strip-global-orient",
        action="store_true",
        default=False,
        help="Zero out root rotation (keep canonical facing direction)",
    )
    parser.add_argument(
        "--cyclic",
        action="store_true",
        default=False,
        help="Auto-detect one gait cycle and trim output for seamless looping",
    )
    parser.add_argument(
        "--render-preview",
        action="store_true",
        default=False,
        help="Render preview/contact_sheet.png and preview/front.gif after mesh generation",
    )
    parser.add_argument(
        "--preview-contact-samples",
        type=int,
        default=6,
        help="Frames sampled into preview/contact_sheet.png (default: 6)",
    )
    parser.add_argument(
        "--preview-gif-frames",
        type=int,
        default=60,
        help="Frames sampled into preview/front.gif (default: 60)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the SMPL-to-PLY converter."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    check_dependencies()
    import torch

    # --- Load motion data ---
    if not args.amass_npz.exists():
        logger.error("Motion file not found: %s", args.amass_npz)
        sys.exit(1)

    logger.info("Loading motion: %s", args.amass_npz)
    motion = load_amass_motion(args.amass_npz)

    poses = motion["poses"]
    betas = motion["betas"]
    trans = motion["trans"]
    src_fps = motion["mocap_framerate"]
    gender = args.gender if args.gender != "auto" else motion["gender"]

    total_frames = len(poses)
    logger.info(
        "Motion: %d frames at %.1f Hz, gender=%s, betas=%s",
        total_frames,
        src_fps,
        gender,
        betas.shape,
    )

    # --- Resolve output paths ---
    name = args.name or args.amass_npz.stem
    output_dir = args.output_dir or Path(f"libraries/targets/smpl_{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Create body model ---
    logger.info("Creating %s model (gender=%s)", args.model_type, gender)
    model = create_body_model(
        model_folder=str(args.model_folder),
        model_type=args.model_type,
        gender=gender,
        num_betas=args.num_betas,
    )
    faces_np = np.asarray(model.faces, dtype=np.int32)

    # --- Determine which frames to extract ---
    frame_indices = compute_frame_indices(
        total_frames=total_frames,
        src_fps=src_fps,
        tgt_fps=args.fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        max_frames=args.max_frames,
    )
    n_output = len(frame_indices)
    tgt_fps = args.fps or src_fps
    logger.info(
        "Extracting %d frames (%.1f Hz -> %.1f Hz)",
        n_output,
        src_fps,
        tgt_fps,
    )
    if args.center_xy:
        logger.info("XY anchor: %s", args.xy_anchor)

    # --- Prepare model inputs ---
    betas_flat = betas.flatten()[: args.num_betas]
    betas_t = torch.tensor(betas_flat, dtype=torch.float32).unsqueeze(0)

    # Determine the expected body_pose size from the model
    n_body_joints = model.NUM_BODY_JOINTS
    body_pose_dim = n_body_joints * 3

    # AMASS stores poses in SMPL-H format (52 joints = 156 params):
    #   [0:3]     global_orient (root rotation)
    #   [3:66]    21 SMPL-H body joints (63 params)
    #   [66:111]  15 left hand joints
    #   [111:156] 15 right hand joints
    # SMPL expects 23 body joints (69 params). The first 21 map directly;
    # joints 22-23 (L_Hand, R_Hand) have no SMPL-H equivalent, set to zero.
    n_pose_params = poses.shape[1] if poses.ndim == 2 else len(poses[0])
    is_smplh_format = n_pose_params >= 156
    smplh_body_end = 66  # pose[3:66] = 21 SMPL-H body joints

    # --- Generate mesh vertices ---
    color = tuple(np.clip(args.color, 0.0, 1.0))

    all_frame_data: list[tuple[np.ndarray, np.ndarray]] = []  # (vertices, normals)

    is_upsampled = frame_indices.dtype == np.float64 and np.any(
        frame_indices != frame_indices.astype(int)
    )
    if is_upsampled:
        logger.info(
            "Upsampling: interpolating poses via SLERP (%.1f Hz -> %.1f Hz)",
            src_fps,
            tgt_fps,
        )

    def _extract_pose_and_translation(src_idx_int: int):
        """Extract pose components for a given source frame index."""
        pose = poses[src_idx_int]
        go = pose[:3].copy()
        if args.strip_global_orient:
            go = np.zeros(3)
        if is_smplh_format and body_pose_dim > 63:
            bp = np.zeros(body_pose_dim, dtype=np.float64)
            bp[:63] = pose[3:smplh_body_end]
        else:
            bp = pose[3 : 3 + body_pose_dim].copy()
            if len(bp) < body_pose_dim:
                bp = np.pad(bp, (0, body_pose_dim - len(bp)))
        tr = trans[src_idx_int].copy() if src_idx_int < len(trans) else np.zeros(3)
        return go, bp, tr

    logger.info("Generating %d mesh frames...", n_output)
    for out_idx, src_idx_f in enumerate(frame_indices):
        src_idx_int = int(src_idx_f)
        frac = float(src_idx_f) - src_idx_int

        if frac > 1e-6 and src_idx_int + 1 < total_frames:
            # Interpolate between adjacent source frames
            go_a, bp_a, tr_a = _extract_pose_and_translation(src_idx_int)
            go_b, bp_b, tr_b = _extract_pose_and_translation(src_idx_int + 1)

            global_orient, _ = interpolate_smpl_params(go_a, go_b, tr_a, tr_b, frac)
            body_pose, translation = interpolate_smpl_params(bp_a, bp_b, tr_a, tr_b, frac)
            # global_orient from SLERP of the two orient params
            global_orient = _quaternion_to_axis_angle(
                _slerp_quaternion(
                    _axis_angle_to_quaternion(go_a),
                    _axis_angle_to_quaternion(go_b),
                    frac,
                )
            )
        else:
            global_orient, body_pose, translation = _extract_pose_and_translation(src_idx_int)

        global_orient_t = torch.tensor(global_orient, dtype=torch.float32).unsqueeze(0)
        body_pose_t = torch.tensor(body_pose, dtype=torch.float32).unsqueeze(0)
        transl_t = torch.tensor(translation, dtype=torch.float32).unsqueeze(0)

        vertices, _, joints = generate_mesh_with_joints(
            model,
            betas_t,
            body_pose_t,
            global_orient_t,
            transl_t,
        )

        xy_reference = None
        if args.center_xy and args.xy_anchor == "pelvis":
            xy_reference = joints[0]

        # Coordinate transform: Y-up -> Z-up, center, ground
        vertices = transform_to_orchav(
            vertices,
            center_xy=args.center_xy,
            ground_feet=args.ground_feet,
            xy_reference=xy_reference,
        )
        normals = compute_normals(vertices, faces_np)
        all_frame_data.append((vertices, normals))

        if (out_idx + 1) % 50 == 0 or out_idx == 0:
            logger.info("  Generated %d/%d frames", out_idx + 1, n_output)

    # --- Cyclic trim ---
    if args.cyclic:
        all_verts = [v for v, _n in all_frame_data]
        cycle_end = detect_gait_cycle(all_verts)
        if cycle_end is not None:
            all_frame_data = all_frame_data[:cycle_end]
            logger.info(
                "Cyclic mode: trimmed to %d frames (1 gait cycle, %.2f s at %.1f Hz)",
                cycle_end,
                cycle_end / tgt_fps,
                tgt_fps,
            )
        else:
            logger.warning("Cyclic mode: keeping all %d frames (no clean cycle found)", n_output)

    # --- Write PLY files ---
    n_final = len(all_frame_data)
    height_min, height_max = float("inf"), float("-inf")

    for out_idx, (vertices, normals) in enumerate(all_frame_data):
        h = vertices[:, 2].max() - vertices[:, 2].min()
        height_min = min(height_min, h)
        height_max = max(height_max, h)

        out_path = output_dir / f"{name}_{out_idx + 1:05d}.ply"
        write_ply_binary(out_path, vertices, faces_np, normals, color)

        if (out_idx + 1) % 50 == 0 or out_idx == 0 or out_idx == n_final - 1:
            logger.info(
                "  [%d/%d] %s  (height: %.3f m)",
                out_idx + 1,
                n_final,
                out_path.name,
                h,
            )

    # --- Summary ---
    logger.info("--- Summary ---")
    logger.info("Output directory : %s", output_dir.resolve())
    logger.info("Files            : %d PLY meshes%s", n_final, " (cyclic)" if args.cyclic else "")
    logger.info("Vertices/frame   : %d", len(all_frame_data[0][0]))
    logger.info("Faces/frame      : %d", len(faces_np))
    logger.info("Height range     : %.3f - %.3f m", height_min, height_max)
    logger.info("Mesh pattern     : %s_*.ply", name)
    if args.cyclic and n_final < n_output:
        logger.info(
            "Cycle duration   : %.3f s (%.1f Hz × %d frames)", n_final / tgt_fps, tgt_fps, n_final
        )
    logger.info(
        'TargetConfig     : mesh_pattern="%s_*.ply", mesh_directory="%s"',
        name,
        output_dir,
    )

    if args.render_preview and n_final > 1:
        from generator.tools.render_mesh_previews import render_library_preview

        logger.info("Rendering preview assets into %s", output_dir / "preview")
        render_library_preview(
            output_dir,
            output_subdir="preview",
            contact_samples=args.preview_contact_samples,
            gif_frames=args.preview_gif_frames,
        )


if __name__ == "__main__":
    main()
