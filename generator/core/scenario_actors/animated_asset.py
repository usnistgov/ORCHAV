"""Opt-in V1 binding and persistent, source-neutral RT geometry preparation.

No source motion runtime is imported here. PLY coordinates are centered only for
Sionna's center-based placement; the manifest root is already baked into each
bound geometry. Coherent Doppler uses the explicit root translation derivative,
while articulation and root rotation remain frozen within one path solve.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from shared.animated_target import (
    AnimatedTargetAsset,
    AnimatedTargetBinding,
    AnimatedTargetBindingContext,
    AnimatedTargetBindingTime,
    canonical_json_bytes,
    load_animated_target,
)
from shared.scenarios.paths import resolve_actor_resource

from ._adapters import value
from .mobility import prepare_sampled_mobility
from .types import PreparedMobility, Timeline


@dataclass(frozen=True)
class PreparedAnimatedAsset:
    asset: AnimatedTargetAsset
    binding: AnimatedTargetBinding
    centers_m: tuple[tuple[float, float, float], ...]
    root_velocities_mps: tuple[tuple[float, float, float], ...]

    def mobility(self, timeline: Timeline) -> PreparedMobility:
        centers = prepare_sampled_mobility(self.centers_m, timeline)
        return PreparedMobility(
            centers.positions_m,
            self.root_velocities_mps,
            centers.forward_vectors,
            has_physical_velocity=any(any(v) for v in self.root_velocities_mps),
        )

    def catalog_metadata(self) -> dict:
        return {
            "contract_version": 1,
            "asset_id": self.asset.asset_id,
            "asset_revision_sha256": self.asset.revision_sha256,
            "binding_id": self.binding.binding_id,
            "binding_context": self.binding.context.to_dict(),
            "geometry_transform_owner": "animated_asset_manifest",
            "rt_velocity_source": "piecewise_linear_root_translation",
            "coherent_geometry_policy": "frozen_at_solve_epoch",
            "license": json.loads(canonical_json_bytes(dict(self.asset.license))),
            "provenance": json.loads(canonical_json_bytes(dict(self.asset.provenance))),
        }

    def sample_metadata(self, index: int) -> dict:
        if not 0 <= index < len(self.binding.samples):
            raise IndexError("animated target sample index is outside the bound timeline")
        sample = self.binding.samples[index]
        return {
            "binding_id": self.binding.binding_id,
            "asset_revision_sha256": self.asset.revision_sha256,
            "sample": sample.evidence_dict(),
            "geometry_evidence_policy": "source_solve_sample_not_interpolated",
            "geometry_scene_step": index,
            "rt_mesh_center_m": list(self.centers_m[index]),
            "root_velocity_m_s": list(self.root_velocities_mps[index]),
        }


def prepare_animated_asset(
    spec: object,
    timeline: Timeline,
    *,
    target_name: str,
    base_dir: str | Path,
    project_root: str | Path | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> PreparedAnimatedAsset:
    """Authenticate an asset and bind exact physical scenario timestamps."""
    path = resolve_actor_resource(
        str(value(spec, "path")),
        scenario_root=Path(base_dir),
        project_root=Path(project_root) if project_root is not None else Path(base_dir),
    )

    def report(stage: str, done: int, total: int) -> None:
        if on_progress is not None:
            on_progress(f"{target_name}: {stage}", done, total)

    progress = report if on_progress is not None else None
    asset = load_animated_target(path, on_progress=progress)
    playback = value(spec, "playback")
    context = AnimatedTargetBindingContext(
        target_id=target_name,
        scenario_clock_id=str(value(playback, "scenario_clock_id")),
        motion_clock_id=str(value(playback, "motion_clock_id")),
        pulse_clock_id=None,
        playback_rate=float(value(playback, "playback_rate")),
        playback_offset_s=float(value(playback, "playback_offset_s")),
        retiming_policy="affine_scenario_to_motion",
    )
    times = tuple(
        AnimatedTargetBindingTime(
            sample_id=f"{context.scenario_clock_id}:time:{float(t).hex()}",
            scenario_time_s=t,
            motion_time_s=context.playback_rate * t + context.playback_offset_s,
        )
        for t in timeline.timestamps_s
    )
    binding = asset.bind(context, times, on_progress=progress)
    centers = tuple(
        tuple(float(v) for v in (np.min(s.vertices_m, axis=0) + np.max(s.vertices_m, axis=0)) * 0.5)
        for s in binding.samples
    )
    velocities = []
    total = len(binding.samples)
    report("Computing motion velocities", 0, total)
    for index, sample in enumerate(binding.samples):
        velocities.append(
            _root_velocity(
                asset, sample.motion_time_s, sample.evaluated_root_time_s, context.playback_rate
            )
        )
        report("Computing motion velocities", index + 1, total)
    return PreparedAnimatedAsset(asset, binding, centers, tuple(velocities))


def _root_velocity(
    asset: AnimatedTargetAsset, requested: float, evaluated: float, rate: float
) -> tuple[float, float, float]:
    samples = asset.root_samples
    if (
        len(samples) < 2
        or evaluated > samples[-1].time_s
        or (
            asset.root_out_of_range == "hold"
            and (requested < samples[0].time_s or requested > samples[-1].time_s)
        )
    ):
        return (0.0, 0.0, 0.0)
    # At a knot use its right derivative, except the final knot's left derivative.
    upper = min(
        int(np.searchsorted([s.time_s for s in samples], evaluated, side="right")), len(samples) - 1
    )
    lower = max(upper - 1, 0)
    if upper == lower:
        return (0.0, 0.0, 0.0)
    derivative = (samples[upper].translation - samples[lower].translation) / (
        samples[upper].time_s - samples[lower].time_s
    )
    transformed = asset.source_to_orchav_affine_4x4[:3, :3] @ derivative * rate
    return tuple(float(v) for v in transformed)


def materialize_animated_meshes(
    prepared: PreparedAnimatedAsset,
    scenario_root: Path,
    *,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Persist deterministic centered PLYs, authenticating any existing cache.

    Derived assets stay under the user's scenario. No temporary file paths enter
    frame metadata; a conflicting/corrupted cache is reported rather than reused.
    """
    root = scenario_root.resolve()
    destination = root
    # Encoding is a derived-cache choice, not part of motion/geometry identity.
    # A separate namespace leaves existing ASCII resources referenced by saved
    # acquisitions intact and retains strict immutable-artifact verification.
    for component in (
        ".orchav",
        "animated_targets",
        prepared.binding.binding_id + "-ply-binary-v1",
    ):
        destination = destination / component
        _require_regular_cache_path(destination, root)
        destination.mkdir(exist_ok=True)
        _require_regular_cache_path(destination, root)
    expected_names = {f"sample_{index:09d}.ply" for index in range(len(prepared.binding.samples))}
    unexpected = {p.name for p in destination.glob("sample_*.ply")} - expected_names
    if unexpected:
        raise ValueError(
            f"animated target RT cache contains unexpected meshes: {sorted(unexpected)}"
        )
    artifacts = []
    total = len(prepared.binding.samples)
    stage = f"{prepared.binding.context.target_id}: Preparing ray-tracing mesh cache"
    if on_progress is not None:
        on_progress(stage, 0, total)
    for index, sample in enumerate(prepared.binding.samples):
        local = sample.vertices_m - np.asarray(prepared.centers_m[index])
        payload = _binary_triangle_ply(local, prepared.binding.faces)
        filename = f"sample_{index:09d}.ply"
        _write_verified(destination / filename, payload)
        artifacts.append(
            {
                "path": filename,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "sample_id": sample.sample_id,
                "center_m": list(prepared.centers_m[index]),
            }
        )
        if on_progress is not None and index + 1 < total:
            on_progress(stage, index + 1, total)
    _write_verified(destination / "binding.json", canonical_json_bytes(prepared.binding.to_dict()))
    _write_verified(
        destination / "rt_geometry.json",
        canonical_json_bytes(
            {
                "geometry_version": 1,
                "ply_encoding": "binary_little_endian",
                "vertex_precision": "float64",
                "binding_id": prepared.binding.binding_id,
                "placement": "centered_mesh_plus_exact_bound_aabb_center",
                "license": dict(prepared.asset.license),
                "provenance": dict(prepared.asset.provenance),
                "artifacts": artifacts,
            }
        ),
    )
    if on_progress is not None:
        on_progress(stage, total, total)
    return destination


def _binary_triangle_ply(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Serialize bound geometry without decimal parsing or precision reduction.

    Normals are intentionally not added: the RT input previously had no vertex
    normals, and adding them can change surface shading. The visualizer computes
    its display normals separately. Explicit byte order makes artifacts portable.
    """
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment ORCHAV animated asset RT geometry v1",
            f"element vertex {len(vertices)}",
            "property double x",
            "property double y",
            "property double z",
            f"element face {len(faces)}",
            "property list uchar int vertex_indices",
            "end_header",
            "",
        ]
    ).encode("ascii")
    face_records = np.empty(len(faces), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
    face_records["count"] = 3
    face_records["indices"] = faces
    return header + np.asarray(vertices, dtype="<f8").tobytes(order="C") + face_records.tobytes()


def _require_regular_cache_path(path: Path, root: Path) -> None:
    # Inspect every component below the explicit scenario root; Windows junctions
    # need a separate check because is_symlink() does not identify them.
    relative = path.relative_to(root)
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise ValueError(f"animated target RT cache may not use links or junctions: {current}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"animated target RT cache escapes scenario root: {path}")


def _write_verified(path: Path, payload: bytes) -> None:
    # Four ancestors are binding-directory, animated_targets, .orchav, scenario.
    root = path.parents[3]
    _require_regular_cache_path(path, root)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"animated target RT cache content mismatch: {path}")
        return
    staging = path.with_name(".tmp-" + uuid4().hex[:16])
    try:
        _require_regular_cache_path(staging, root)
        with staging.open("xb") as stream:
            stream.write(payload)
        _require_regular_cache_path(path, root)
        try:
            # Create-if-absent publication: unlike replace(), a racing writer
            # cannot overwrite an already published immutable artifact.
            try:
                os.link(staging, path)
            except PermissionError:
                if os.name != "nt":
                    raise
                # Windows rename also fails if the destination exists. This
                # permits filesystems or policies that disallow hard links
                # without weakening the no-overwrite publication contract.
                os.rename(staging, path)
        except FileExistsError:
            _require_regular_cache_path(path, root)
            if not path.is_file() or path.read_bytes() != payload:
                raise ValueError(f"animated target RT cache content mismatch: {path}") from None
    finally:
        staging.unlink(missing_ok=True)
