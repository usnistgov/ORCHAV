"""CPU-only evidence workflow for source-neutral animated-target assets.

This module turns the contracts in :mod:`shared.animated_target` into a small,
runnable experiment. It can author an ORCHAV-owned asymmetric fixture or bind
an existing manifest, then atomically publish machine-readable binding
evidence, a tabular geometry summary, and a timeline/geometry figure.

Matplotlib is imported only when publication is requested. Source-specific
motion runtimes remain outside this neutral workflow.
"""

from __future__ import annotations

import csv
import errno
import hashlib
import io
import math
import os
import platform
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shared.animated_target import (
    ANIMATED_TARGET_MANIFEST_VERSION,
    AnimatedTargetAsset,
    AnimatedTargetBinding,
    AnimatedTargetBindingContext,
    AnimatedTargetBindingTime,
    animated_target_revision_sha256,
    canonical_json_bytes,
    canonical_sha256,
    load_animated_target,
)

ANIMATED_TARGET_WORKFLOW_VERSION = 1

_POSE_MEDIA_TYPE = "application/vnd.orchav.animated-pose+json"
_SENSING_INSTALL_GUIDANCE = "Install ORCHAV with: python -m pip install orchav"
_CSV_COLUMNS = (
    "target_id",
    "scenario_clock_id",
    "motion_clock_id",
    "pulse_clock_id",
    "playback_rate",
    "playback_offset_s",
    "retiming_policy",
    "sample_id",
    "scenario_time_s",
    "motion_time_s",
    "pulse_time_s",
    "evaluated_pose_time_s",
    "pose_cycle",
    "pose_sample_id",
    "evaluated_root_time_s",
    "root_cycle",
    "root_lower_sample_id",
    "root_upper_sample_id",
    "root_fraction",
    "valid",
    "resolved_root_affine_sha256",
    "vertex_count",
    "centroid_x_m",
    "centroid_y_m",
    "centroid_z_m",
    "minimum_x_m",
    "minimum_y_m",
    "minimum_z_m",
    "maximum_x_m",
    "maximum_y_m",
    "maximum_z_m",
    "vertices_sha256",
    "normals_sha256",
)


class AnimatedTargetWorkflowError(RuntimeError):
    """Raised when an animated-target workflow cannot publish safely."""


@dataclass(frozen=True, slots=True)
class AnimatedTargetWorkflowArtifact:
    """One exact output-file inventory entry."""

    path: str
    role: str
    size_bytes: int
    sha256: str
    portability: str

    def to_dict(self) -> dict[str, Any]:
        """Return this entry in artifact-manifest form."""

        return {
            "path": self.path,
            "role": self.role,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "portability": self.portability,
        }


@dataclass(frozen=True, slots=True)
class AnimatedTargetWorkflowResult:
    """Identity and paths for one successfully published workflow."""

    output_dir: Path
    asset_id: str
    asset_revision_sha256: str
    binding_id: str
    workflow_id: str
    artifact_manifest_sha256: str
    artifacts: tuple[AnimatedTargetWorkflowArtifact, ...]


@dataclass(frozen=True, slots=True)
class _PlottingRuntime:
    matplotlib_version: str
    pyplot: Any
    poly3d_collection: Any


@dataclass(frozen=True, slots=True)
class _StagedResult:
    asset_id: str
    asset_revision_sha256: str
    binding_id: str
    workflow_id: str
    artifact_manifest_sha256: str
    artifacts: tuple[AnimatedTargetWorkflowArtifact, ...]


def default_synthetic_binding_times() -> tuple[AnimatedTargetBindingTime, ...]:
    """Return the exact scenario/motion/pulse clock map for the built-in fixture."""

    semantic_names = (
        "leading_edge",
        "early_rise",
        "first_quarter",
        "pre_transition",
        "pose_transition",
        "post_transition",
        "third_quarter",
        "late_fall",
        "trailing_edge",
    )
    motion_times = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
    return tuple(
        AnimatedTargetBindingTime(
            sample_id=f"acquisition:synthetic:{name}",
            scenario_time_s=2.0 + motion_time,
            motion_time_s=motion_time,
            pulse_time_s=100.0 + motion_time,
        )
        for name, motion_time in zip(semantic_names, motion_times)
    )


def default_synthetic_binding_context() -> AnimatedTargetBindingContext:
    """Return the immutable target and clock relationship for the fixture."""

    return AnimatedTargetBindingContext(
        target_id="target:synthetic_asymmetric_subject",
        scenario_clock_id="clock:scenario_reference",
        motion_clock_id="clock:asset_motion",
        pulse_clock_id="clock:sensing_pulse",
        playback_rate=1.0,
        playback_offset_s=-2.0,
        retiming_policy="affine_scenario_to_motion",
    )


def publish_synthetic_animated_target_workflow(
    output_dir: str | Path,
    *,
    binding_context: AnimatedTargetBindingContext | None = None,
    binding_times: Sequence[AnimatedTargetBindingTime] | None = None,
) -> AnimatedTargetWorkflowResult:
    """Atomically publish the ORCHAV-authored asymmetric reference workflow.

    The output contains the redistributable V1 asset manifest and its three
    pose resources in addition to the same binding evidence produced for an
    existing asset. ``output_dir`` must not already exist.
    """

    output = _prepare_destination(output_dir)
    plotting = _load_plotting_runtime()
    context = binding_context or default_synthetic_binding_context()
    requested_times = tuple(
        default_synthetic_binding_times() if binding_times is None else binding_times
    )

    def build(staging: Path) -> _StagedResult:
        manifest_path, asset_paths = _write_synthetic_asset(staging / "asset")
        asset = load_animated_target(manifest_path)
        binding = asset.bind(context, requested_times)
        return _write_workflow_evidence(
            staging,
            asset,
            binding,
            plotting,
            source_policy="included_orchav_authored_synthetic",
            source_artifact_paths=asset_paths,
        )

    return _publish_directory(output, build)


def publish_animated_target_workflow(
    manifest_path: str | Path,
    binding_context: AnimatedTargetBindingContext,
    binding_times: Sequence[AnimatedTargetBindingTime],
    output_dir: str | Path,
) -> AnimatedTargetWorkflowResult:
    """Bind an existing V1 asset and atomically publish equivalent evidence.

    Existing manifest/resources are validated and referenced by semantic
    identity but are not copied. Assets marked ``prohibited`` cannot publish
    derived evidence; ``user_supplied_only`` outputs are explicitly labelled
    local-only and non-redistributable.
    """

    asset = load_animated_target(manifest_path)
    redistribution = asset.license["redistribution"]
    if redistribution == "prohibited":
        raise AnimatedTargetWorkflowError(
            "asset license prohibits publication of derived geometry evidence"
        )
    output = _prepare_destination(output_dir)
    binding = asset.bind(binding_context, tuple(binding_times))
    plotting = _load_plotting_runtime()

    def build(staging: Path) -> _StagedResult:
        return _write_workflow_evidence(
            staging,
            asset,
            binding,
            plotting,
            source_policy="validated_existing_input_not_copied",
            source_artifact_paths=(),
        )

    return _publish_directory(output, build)


def _prepare_destination(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    if not output.name:
        raise AnimatedTargetWorkflowError("output_dir must name a new directory")
    if output.exists() or output.is_symlink():
        raise AnimatedTargetWorkflowError(f"output directory already exists: {output}")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AnimatedTargetWorkflowError(
            f"cannot create output parent directory {output.parent}: {exc}"
        ) from exc
    if not output.parent.is_dir():
        raise AnimatedTargetWorkflowError(f"output parent is not a directory: {output.parent}")
    return output


def _publish_directory(
    output: Path,
    build: Callable[[Path], _StagedResult],
) -> AnimatedTargetWorkflowResult:
    lock_path = output.parent / f".{output.name}.publish.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise AnimatedTargetWorkflowError(
            f"another publication owns the output lock: {lock_path}"
        ) from exc
    except OSError as exc:
        raise AnimatedTargetWorkflowError(f"cannot create output lock {lock_path}: {exc}") from exc

    staging: Path | None = None
    try:
        os.close(lock_descriptor)
        if output.exists() or output.is_symlink():
            raise AnimatedTargetWorkflowError(f"output directory already exists: {output}")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.staging-",
                dir=output.parent,
            )
        )
        staged_result = build(staging)
        _promote_directory_noreplace(staging, output)
        staging = None
        return AnimatedTargetWorkflowResult(
            output_dir=output.resolve(),
            asset_id=staged_result.asset_id,
            asset_revision_sha256=staged_result.asset_revision_sha256,
            binding_id=staged_result.binding_id,
            workflow_id=staged_result.workflow_id,
            artifact_manifest_sha256=staged_result.artifact_manifest_sha256,
            artifacts=staged_result.artifacts,
        )
    except AnimatedTargetWorkflowError:
        raise
    except Exception as exc:
        raise AnimatedTargetWorkflowError(f"animated-target publication failed: {exc}") from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _promote_directory_noreplace(source: Path, destination: Path) -> None:
    """Promote a sibling directory atomically without replacing a destination."""

    if os.name == "posix":
        try:
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            if platform.system() == "Darwin":
                # Apple's rename(2) contract and <sys/stdio.h> define
                # renamex_np(source, destination, RENAME_EXCL=0x00000004).
                # https://github.com/apple-oss-distributions/xnu/blob/main/bsd/man/man2/rename.2
                rename_exclusive = libc.renamex_np
                rename_exclusive.argtypes = (
                    ctypes.c_char_p,
                    ctypes.c_char_p,
                    ctypes.c_uint,
                )
                arguments = (os.fsencode(source), os.fsencode(destination), 0x00000004)
            else:
                rename_exclusive = libc.renameat2
                rename_exclusive.argtypes = (
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_char_p,
                    ctypes.c_uint,
                )
                arguments = (
                    -100,  # AT_FDCWD
                    os.fsencode(source),
                    -100,
                    os.fsencode(destination),
                    1,  # RENAME_NOREPLACE
                )
            rename_exclusive.restype = ctypes.c_int
        except (AttributeError, OSError) as exc:
            raise AnimatedTargetWorkflowError(
                "atomic no-replace directory promotion is unavailable; "
                "refusing an unsafe fallback"
            ) from exc

        result = rename_exclusive(*arguments)
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise AnimatedTargetWorkflowError(f"output directory already exists: {destination}")
        unavailable_errors = {
            errno.ENOSYS,
            errno.EINVAL,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if error_number in unavailable_errors:
            raise AnimatedTargetWorkflowError(
                "atomic no-replace directory promotion is unavailable; "
                "refusing an unsafe fallback"
            )
        raise OSError(error_number, os.strerror(error_number), str(destination))

    if os.name == "nt":
        _promote_windows_directory_noreplace(source, destination)
        return

    raise AnimatedTargetWorkflowError(
        "atomic no-replace directory promotion is unavailable; " "refusing an unsafe fallback"
    )


def _promote_windows_directory_noreplace(source: Path, destination: Path) -> None:
    """Tolerate brief Windows scanner/sync locks while retaining exclusive rename.

    Windows can deny a rename temporarily even after all writer handles are closed.
    Retry only its access/sharing/lock violations, for at most 1.55 seconds in total.
    Every attempt uses the same no-replace operation. Existing destinations and all
    other errors fail immediately; permanent permission errors retain the OS cause.
    """
    delays = (0.05, 0.1, 0.2, 0.4, 0.8)
    for attempt in range(len(delays) + 1):
        try:
            os.rename(source, destination)
            return
        except OSError as exc:
            # Windows sometimes reports access denied rather than exists when a
            # competing publisher creates a directory. Never retry that collision.
            if isinstance(exc, FileExistsError) or os.path.lexists(destination):
                raise AnimatedTargetWorkflowError(
                    f"output directory already exists: {destination}"
                ) from exc
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def _load_plotting_runtime() -> _PlottingRuntime:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        raise AnimatedTargetWorkflowError(
            f"Matplotlib is required for animated-target figures. {_SENSING_INSTALL_GUIDANCE}"
        ) from exc

    matplotlib.use("Agg", force=True)
    try:
        import matplotlib.pyplot as pyplot
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ModuleNotFoundError as exc:
        # Preserve tracebacks for broken installed packages. Only an exact
        # missing top-level Matplotlib package becomes installation guidance.
        if exc.name != "matplotlib":
            raise
        raise AnimatedTargetWorkflowError(
            f"Matplotlib is required for animated-target figures. {_SENSING_INSTALL_GUIDANCE}"
        ) from exc
    return _PlottingRuntime(
        matplotlib_version=str(matplotlib.__version__),
        pyplot=pyplot,
        poly3d_collection=Poly3DCollection,
    )


def _write_synthetic_asset(asset_dir: Path) -> tuple[Path, tuple[str, ...]]:
    poses_dir = asset_dir / "poses"
    poses_dir.mkdir(parents=True, exist_ok=False)
    faces = np.array(
        (
            (1, 2, 3),
            (0, 3, 2),
            (0, 1, 3),
            (0, 2, 1),
        ),
        dtype=np.int64,
    )
    poses = (
        (
            "mesh:neutral_stance",
            "poses/z_neutral_stance.json",
            np.array(
                (
                    (0.0, 0.0, 0.0),
                    (0.8, 0.1, 0.0),
                    (0.0, 1.2, 0.2),
                    (-0.2, 0.3, 1.1),
                ),
                dtype=np.float64,
            ),
        ),
        (
            "mesh:left_reach",
            "poses/m_left_reach.json",
            np.array(
                (
                    (0.0, 0.0, 0.0),
                    (1.0, 0.2, 0.1),
                    (0.1, 1.4, 0.25),
                    (-0.3, 0.4, 1.3),
                ),
                dtype=np.float64,
            ),
        ),
        (
            "mesh:asymmetric_settle",
            "poses/a_asymmetric_settle.json",
            np.array(
                (
                    (0.0, 0.0, 0.0),
                    (0.7, 0.3, 0.15),
                    (-0.1, 1.25, 0.25),
                    (-0.25, 0.35, 1.2),
                ),
                dtype=np.float64,
            ),
        ),
    )
    resources: list[dict[str, Any]] = []
    resource_paths: list[str] = []
    for resource_id, relative_path, vertices in poses:
        payload = canonical_json_bytes(
            {
                "vertices": vertices.tolist(),
                "faces": faces.tolist(),
            }
        )
        destination = asset_dir / relative_path
        destination.write_bytes(payload)
        resources.append(
            {
                "resource_id": resource_id,
                "path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "media_type": _POSE_MEDIA_TYPE,
            }
        )
        resource_paths.append(f"asset/{relative_path}")

    topology_sha256 = canonical_sha256(
        {
            "representation": "indexed_triangles",
            "vertex_count": 4,
            "faces": faces.tolist(),
        }
    )
    sine_22_5 = math.sin(math.radians(22.5))
    cosine_22_5 = math.cos(math.radians(22.5))
    sine_45 = math.sin(math.radians(45.0))
    cosine_45 = math.cos(math.radians(45.0))
    manifest: dict[str, Any] = {
        "manifest_version": ANIMATED_TARGET_MANIFEST_VERSION,
        "asset": {
            "asset_id": "orchav:synthetic:asymmetric_motion_workflow_v1",
            "revision_sha256": "0" * 64,
        },
        # Declaration order is intentionally physical, not filename-sorted.
        "resources": resources,
        "coordinates": {
            "source_frame_id": "synthetic:y_up_z_front",
            "source_handedness": "right",
            "front_axis": "+Z",
            "up_axis": "+Y",
            "length_unit": "m",
            "destination_frame_id": "orchav:world_rh_x_front_z_up_m",
            "source_to_orchav_affine_4x4": (
                (0.0, 0.0, 1.0, 0.25),
                (1.0, 0.0, 0.0, -0.1),
                (0.0, 1.0, 0.0, 0.05),
                (0.0, 0.0, 0.0, 1.0),
            ),
            "scale_ownership": "uniform_scale_in_affine_m_per_source_unit",
            "translation_ownership": "translation_in_affine_destination_m",
        },
        "topology": {
            "representation": "indexed_triangles",
            "mode": "fixed_across_poses",
            "vertex_count": 4,
            "triangle_count": 4,
            "topology_sha256": topology_sha256,
            "closed_surface": True,
            "source_face_winding": "counter_clockwise_from_outside",
            "reflection_winding_policy": "reverse",
            "normal_policy": "recompute",
        },
        "motion": {
            "time_axis": "asset_time_s",
            "pose": {
                "interpolation": "step",
                "out_of_range": "error",
                "cyclic": False,
                "period_s": None,
                "samples": (
                    {
                        "sample_id": "pose:neutral_stance",
                        "time_s": 0.0,
                        "resource_id": "mesh:neutral_stance",
                    },
                    {
                        "sample_id": "pose:left_reach",
                        "time_s": 0.5,
                        "resource_id": "mesh:left_reach",
                    },
                    {
                        "sample_id": "pose:asymmetric_settle",
                        "time_s": 1.0,
                        "resource_id": "mesh:asymmetric_settle",
                    },
                ),
            },
            "root": {
                "translation_interpolation": "linear",
                "rotation_interpolation": "slerp_shortest",
                "rotation_representation": "active_quaternion_xyzw",
                "out_of_range": "error",
                "cyclic": False,
                "period_s": None,
                "samples": (
                    {
                        "sample_id": "root:origin_facing_front",
                        "time_s": 0.0,
                        "translation": (0.0, 0.0, 0.0),
                        "rotation_xyzw": (0.0, 0.0, 0.0, 1.0),
                    },
                    {
                        "sample_id": "root:mid_stride_turn",
                        "time_s": 0.5,
                        "translation": (0.05, 0.0, 0.6),
                        "rotation_xyzw": (0.0, sine_22_5, 0.0, cosine_22_5),
                    },
                    {
                        "sample_id": "root:forward_offset_turn",
                        "time_s": 1.0,
                        "translation": (0.15, 0.1, 1.2),
                        "rotation_xyzw": (0.0, sine_45, 0.0, cosine_45),
                    },
                ),
            },
        },
        "provenance": {
            "creator": "ORCHAV contributors",
            "source_type": "orchav_authored_synthetic",
            "source_ids": ("orchav:workflow:asymmetric_pose_root_v1",),
            "description": (
                "Hand-authored asymmetric closed mesh, three poses, and independent root track."
            ),
        },
        "license": {
            "spdx_id": "GPL-3.0-only",
            "redistribution": "allowed",
            "notice": "ORCHAV-authored synthetic data; no third-party motion input.",
        },
    }
    manifest["asset"]["revision_sha256"] = animated_target_revision_sha256(manifest)
    manifest_path = asset_dir / "animated_target.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest_path, ("asset/animated_target.json", *resource_paths)


def _write_workflow_evidence(
    staging: Path,
    asset: AnimatedTargetAsset,
    binding: AnimatedTargetBinding,
    plotting: _PlottingRuntime,
    *,
    source_policy: str,
    source_artifact_paths: Sequence[str],
) -> _StagedResult:
    binding_request_path = staging / "binding_request.json"
    binding_request_path.write_bytes(
        canonical_json_bytes(
            {
                "contract": "orchav.animated_target_binding_request",
                "contract_version": 1,
                "context": binding.context.to_dict(),
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "scenario_time_s": sample.scenario_time_s,
                        "motion_time_s": sample.motion_time_s,
                        "pulse_time_s": sample.pulse_time_s,
                    }
                    for sample in binding.samples
                ],
            }
        )
    )
    binding_path = staging / "binding.json"
    binding_path.write_bytes(binding.to_canonical_json().encode("utf-8"))

    csv_path = staging / "binding_samples.csv"
    csv_path.write_bytes(_binding_csv_bytes(binding))

    figure_path = staging / "alignment_and_geometry.png"
    _write_figure(figure_path, asset, binding, plotting)

    roles = {
        "binding_request.json": "portable explicit target and clock binding request",
        "binding.json": "canonical binding identity and geometry hashes",
        "binding_samples.csv": "physical clock, interpolation, and geometry summary",
        "alignment_and_geometry.png": "pose/root/pulse alignment and geometry snapshots",
        "asset/animated_target.json": "ORCHAV-authored V1 animated-target manifest",
    }
    for relative_path in source_artifact_paths:
        if relative_path.startswith("asset/poses/"):
            roles[relative_path] = "ORCHAV-authored asymmetric pose resource"

    artifact_paths = tuple(source_artifact_paths) + (
        "binding_request.json",
        "binding.json",
        "binding_samples.csv",
        "alignment_and_geometry.png",
    )
    artifacts = tuple(
        _artifact(
            staging,
            relative_path,
            role=roles[relative_path],
            portability=(
                "runtime_qualified"
                if relative_path == "alignment_and_geometry.png"
                else "portable_exact"
            ),
        )
        for relative_path in sorted(artifact_paths)
    )
    runtime = {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "matplotlib": plotting.matplotlib_version,
        "matplotlib_backend": "Agg",
        "png_portability": (
            "The PNG SHA-256 is exact only for the declared runtime versions; "
            "binding JSON and CSV are the portable semantic evidence."
        ),
    }
    redistribution = asset.license["redistribution"]
    output_governance = {
        "source_redistribution": redistribution,
        "derived_artifact_distribution": (
            "manifest_allows_redistribution_pending_release_review"
            if redistribution == "allowed"
            else "local_user_controlled_only_not_redistributable"
        ),
        "release_review_required": True,
    }
    identity_basis = {
        "workflow_manifest_version": ANIMATED_TARGET_WORKFLOW_VERSION,
        "asset": {
            "asset_id": asset.asset_id,
            "revision_sha256": asset.revision_sha256,
        },
        "binding_id": binding.binding_id,
        "binding_context": binding.context.to_dict(),
        "source_policy": source_policy,
        "output_governance": output_governance,
        "artifacts": [artifact.to_dict() for artifact in artifacts],
        "runtime": runtime,
    }
    workflow_id = canonical_sha256(identity_basis)
    artifact_manifest = {
        **identity_basis,
        "workflow_id": workflow_id,
        "license": dict(asset.license),
        "provenance": dict(asset.provenance),
        "artifact_manifest_self_hash": (
            "Intentionally omitted from its own file inventory; callers may hash this file directly."
        ),
    }
    artifact_manifest_bytes = canonical_json_bytes(artifact_manifest)
    (staging / "artifact_manifest.json").write_bytes(artifact_manifest_bytes)
    return _StagedResult(
        asset_id=asset.asset_id,
        asset_revision_sha256=asset.revision_sha256,
        binding_id=binding.binding_id,
        workflow_id=workflow_id,
        artifact_manifest_sha256=hashlib.sha256(artifact_manifest_bytes).hexdigest(),
        artifacts=artifacts,
    )


def _binding_csv_bytes(binding: AnimatedTargetBinding) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for sample in binding.samples:
        vertices = sample.vertices_m
        evidence = sample.evidence_dict()
        writer.writerow(
            {
                "target_id": binding.context.target_id,
                "scenario_clock_id": binding.context.scenario_clock_id,
                "motion_clock_id": binding.context.motion_clock_id,
                "pulse_clock_id": binding.context.pulse_clock_id or "",
                "playback_rate": _float_text(binding.context.playback_rate),
                "playback_offset_s": _float_text(binding.context.playback_offset_s),
                "retiming_policy": binding.context.retiming_policy,
                "sample_id": sample.sample_id,
                "scenario_time_s": _float_text(sample.scenario_time_s),
                "motion_time_s": _float_text(sample.motion_time_s),
                "pulse_time_s": _float_text(sample.pulse_time_s),
                "evaluated_pose_time_s": _float_text(sample.evaluated_pose_time_s),
                "pose_cycle": str(sample.pose_cycle),
                "pose_sample_id": sample.pose_sample_id,
                "evaluated_root_time_s": _float_text(sample.evaluated_root_time_s),
                "root_cycle": str(sample.root_cycle),
                "root_lower_sample_id": sample.root_lower_sample_id,
                "root_upper_sample_id": sample.root_upper_sample_id,
                "root_fraction": _float_text(sample.root_fraction),
                "valid": "true" if sample.valid else "false",
                "resolved_root_affine_sha256": evidence["root"]["resolved_affine_sha256"],
                "vertex_count": str(vertices.shape[0]),
                "centroid_x_m": _float_text(np.mean(vertices[:, 0])),
                "centroid_y_m": _float_text(np.mean(vertices[:, 1])),
                "centroid_z_m": _float_text(np.mean(vertices[:, 2])),
                "minimum_x_m": _float_text(np.min(vertices[:, 0])),
                "minimum_y_m": _float_text(np.min(vertices[:, 1])),
                "minimum_z_m": _float_text(np.min(vertices[:, 2])),
                "maximum_x_m": _float_text(np.max(vertices[:, 0])),
                "maximum_y_m": _float_text(np.max(vertices[:, 1])),
                "maximum_z_m": _float_text(np.max(vertices[:, 2])),
                "vertices_sha256": evidence["vertices_sha256"],
                "normals_sha256": evidence["normals_sha256"],
            }
        )
    return stream.getvalue().encode("utf-8")


def _float_text(value: float | np.floating[Any] | None) -> str:
    if value is None:
        return ""
    return format(float(value), ".17g")


def _write_figure(
    destination: Path,
    asset: AnimatedTargetAsset,
    binding: AnimatedTargetBinding,
    plotting: _PlottingRuntime,
) -> None:
    pyplot = plotting.pyplot
    with pyplot.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        figure = pyplot.figure(figsize=(12.0, 7.2))
        grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 2.0), hspace=0.34, wspace=0.22)
        timeline = figure.add_subplot(grid[0, :])
        _draw_timeline(timeline, asset, binding)

        selected_indices = _snapshot_indices(len(binding.samples))
        selected_vertices = [binding.samples[index].vertices_m for index in selected_indices]
        lower, upper = _common_limits(selected_vertices)
        colors = ("#3B82F6", "#F59E0B", "#10B981")
        for column in range(3):
            axis = figure.add_subplot(grid[1, column], projection="3d")
            if column >= len(selected_indices):
                axis.set_axis_off()
                continue
            sample = binding.samples[selected_indices[column]]
            triangles = _bounded_triangles(sample.vertices_m, binding.faces)
            collection = plotting.poly3d_collection(
                triangles,
                facecolor=colors[column],
                edgecolor="#1F2937",
                linewidth=0.55,
                alpha=0.72,
            )
            axis.add_collection3d(collection)
            axis.scatter(
                sample.vertices_m[:, 0],
                sample.vertices_m[:, 1],
                sample.vertices_m[:, 2],
                color="#111827",
                s=10,
                depthshade=False,
            )
            axis.set_xlim(lower[0], upper[0])
            axis.set_ylim(lower[1], upper[1])
            axis.set_zlim(lower[2], upper[2])
            axis.set_box_aspect((1.0, 1.0, 1.0))
            axis.view_init(elev=23.0, azim=-58.0)
            axis.set_xlabel("X [m]")
            axis.set_ylabel("Y [m]")
            axis.set_zlabel("Z [m]")
            # Bound display labels to the subplot; JSON/CSV retain the full IDs.
            pose_label = sample.pose_sample_id
            if len(pose_label) > 28:
                pose_label = f"{pose_label[:25]}..."
            axis.set_title(
                f"{sample.sample_id.split(':')[-1]}\n"
                f"motion={_float_text(sample.motion_time_s)} s\n{pose_label}"
            )
        figure.suptitle(
            "Animated-target timing and transformed geometry\n"
            f"{asset.asset_id} · {binding.context.target_id} · binding {binding.binding_id[:12]}",
            fontsize=12,
        )
        figure.savefig(
            destination,
            format="png",
            dpi=140,
            bbox_inches="tight",
            metadata={"Software": f"ORCHAV / Matplotlib {plotting.matplotlib_version}"},
        )
        pyplot.close(figure)


def _draw_timeline(axis: Any, asset: AnimatedTargetAsset, binding: AnimatedTargetBinding) -> None:
    pose_times = [sample.time_s for sample in asset.pose_samples]
    root_times = [sample.time_s for sample in asset.root_samples]
    request_times = [sample.motion_time_s for sample in binding.samples]
    evaluated_pose_times = [sample.evaluated_pose_time_s for sample in binding.samples]

    axis.hlines(3.0, min(pose_times), max(pose_times), color="#2563EB", linewidth=2.0)
    axis.scatter(pose_times, [3.0] * len(pose_times), color="#2563EB", marker="s", zorder=3)
    axis.hlines(2.0, min(root_times), max(root_times), color="#D97706", linewidth=2.0)
    axis.scatter(root_times, [2.0] * len(root_times), color="#D97706", marker="o", zorder=3)
    axis.scatter(request_times, [1.0] * len(request_times), color="#059669", marker="|", s=180)
    for requested, evaluated in zip(request_times, evaluated_pose_times):
        axis.plot((requested, evaluated), (1.08, 2.92), color="#9CA3AF", linewidth=0.6)
    for sample in binding.samples:
        if sample.pulse_time_s is not None:
            axis.annotate(
                sample.sample_id.split(":")[-1],
                (sample.motion_time_s, 1.04),
                xytext=(2, 3),
                textcoords="offset points",
                rotation=55,
                ha="left",
                va="bottom",
                fontsize=6.5,
                color="#065F46",
            )
    axis.set_yticks((1.0, 2.0, 3.0), ("bound pulses", "root samples", "step poses"))
    axis.set_xlabel("asset motion time [s] (scenario and pulse times remain exact in JSON/CSV)")
    axis.set_ylim(0.25, 3.45)
    axis.set_title("Explicit clock binding: no ordinal or count-ratio synchronization")


def _snapshot_indices(count: int) -> tuple[int, ...]:
    candidates = (0, count // 2, count - 1)
    return tuple(dict.fromkeys(candidates))


def _common_limits(vertices: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack(vertices)
    minimum = np.min(combined, axis=0)
    maximum = np.max(combined, axis=0)
    center = (minimum + maximum) * 0.5
    extent = max(float(np.max(maximum - minimum)), 1e-3) * 0.62
    return center - extent, center + extent


def _bounded_triangles(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    maximum_faces: int = 2_000,
) -> np.ndarray:
    stride = max(1, math.ceil(faces.shape[0] / maximum_faces))
    return vertices[faces[::stride]]


def _artifact(
    root: Path,
    relative_path: str,
    *,
    role: str,
    portability: str,
) -> AnimatedTargetWorkflowArtifact:
    path = root / relative_path
    payload = path.read_bytes()
    return AnimatedTargetWorkflowArtifact(
        path=relative_path,
        role=role,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        portability=portability,
    )


__all__ = [
    "ANIMATED_TARGET_WORKFLOW_VERSION",
    "AnimatedTargetWorkflowArtifact",
    "AnimatedTargetWorkflowError",
    "AnimatedTargetWorkflowResult",
    "default_synthetic_binding_context",
    "default_synthetic_binding_times",
    "publish_animated_target_workflow",
    "publish_synthetic_animated_target_workflow",
]
