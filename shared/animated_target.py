"""Source-neutral animated-target assets and deterministic time binding.

The V1 contract in this module deliberately stops at an interchange boundary:
licensed or restricted motion systems may export the JSON manifest and pose
resources, but loading and binding those assets never imports their runtimes.
The manifest owns coordinates, topology, clocks, content identity, provenance,
and licensing so callers do not have to infer semantics from filenames.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

ANIMATED_TARGET_MANIFEST_VERSION = 1
ANIMATED_TARGET_BINDING_VERSION = 1

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_AXIS_PATTERN = re.compile(r"[+-][XYZ]")
_VECTOR_EPSILON = 1e-12
_UNIT_QUATERNION_TOLERANCE = 1e-6


class AnimatedTargetError(ValueError):
    """Raised when an animated-target asset or binding violates its contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one JSON-compatible value using the contract's canonical form."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise AnimatedTargetError(f"value is not canonical JSON: {exc}") from exc
    return encoded.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 digest of ``canonical_json_bytes(value)``."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def animated_target_revision_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the canonical manifest revision, excluding its self-reference."""

    normalized = _json_clone(manifest)
    asset = normalized.get("asset")
    if not isinstance(asset, dict):
        raise AnimatedTargetError("asset must be a JSON object")
    if "revision_sha256" not in asset:
        raise AnimatedTargetError("asset.revision_sha256 is required")
    del asset["revision_sha256"]
    return canonical_sha256(normalized)


@dataclass(frozen=True, slots=True)
class AnimatedTargetBindingTime:
    """One explicit relationship among scenario, motion, and optional pulse time.

    ``sample_id`` is supplied by the caller and must be stable in its domain;
    no ordinal or filename-derived identity is synthesized. ``motion_time_s``
    is the time at which pose and root tracks are evaluated. The other clocks
    are preserved verbatim in the binding evidence.
    """

    sample_id: str
    scenario_time_s: float
    motion_time_s: float
    pulse_time_s: float | None = None


@dataclass(frozen=True, slots=True)
class AnimatedTargetBindingContext:
    """Immutable identity and retiming policy for one target-clock binding."""

    target_id: str
    scenario_clock_id: str
    motion_clock_id: str
    pulse_clock_id: str | None
    playback_rate: float
    playback_offset_s: float
    retiming_policy: Literal["explicit_per_sample", "affine_scenario_to_motion"]

    def __post_init__(self) -> None:
        for field_name in ("target_id", "scenario_clock_id", "motion_clock_id"):
            _stable_id(getattr(self, field_name), f"binding_context.{field_name}")
        if self.pulse_clock_id is not None:
            _stable_id(self.pulse_clock_id, "binding_context.pulse_clock_id")
        playback_rate = _finite_number(self.playback_rate, "binding_context.playback_rate")
        if playback_rate <= 0.0:
            raise AnimatedTargetError("binding_context.playback_rate must be positive")
        object.__setattr__(self, "playback_rate", playback_rate)
        object.__setattr__(
            self,
            "playback_offset_s",
            _finite_number(
                self.playback_offset_s,
                "binding_context.playback_offset_s",
            ),
        )
        if self.retiming_policy not in (
            "explicit_per_sample",
            "affine_scenario_to_motion",
        ):
            raise AnimatedTargetError(
                "binding_context.retiming_policy must be 'explicit_per_sample' or "
                "'affine_scenario_to_motion'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical binding-context representation."""

        return {
            "target_id": self.target_id,
            "scenario_clock_id": self.scenario_clock_id,
            "motion_clock_id": self.motion_clock_id,
            "pulse_clock_id": self.pulse_clock_id,
            "playback_rate": self.playback_rate,
            "playback_offset_s": self.playback_offset_s,
            "retiming_policy": self.retiming_policy,
        }


@dataclass(frozen=True, slots=True)
class AnimatedTargetResource:
    """One hash-pinned pose resource declared by a V1 manifest."""

    resource_id: str
    path: str
    sha256: str
    size_bytes: int
    media_type: str


@dataclass(frozen=True, slots=True)
class AnimatedTargetPoseSample:
    """One timestamped pose mesh with stable semantic identity."""

    sample_id: str
    time_s: float
    resource_id: str
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None


@dataclass(frozen=True, slots=True)
class AnimatedTargetRootSample:
    """One timestamped root transform in source coordinates."""

    sample_id: str
    time_s: float
    translation: np.ndarray
    rotation_xyzw: np.ndarray


@dataclass(frozen=True, slots=True)
class _ResolvedAnimatedPose:
    """Geometry and clock evidence shared by binding and trajectory inspection."""

    pose: AnimatedTargetPoseSample
    pose_time_s: float
    root_time_s: float
    pose_cycle: int
    root_cycle: int
    root_lower: AnimatedTargetRootSample
    root_upper: AnimatedTargetRootSample
    root_fraction: float
    rotation_xyzw: np.ndarray
    root_affine: np.ndarray
    vertices_m: np.ndarray


@dataclass(frozen=True, slots=True)
class BoundAnimatedTargetSample:
    """Geometry and trace evidence bound to one explicit requested time."""

    sample_id: str
    scenario_time_s: float
    motion_time_s: float
    pulse_time_s: float | None
    evaluated_pose_time_s: float
    evaluated_root_time_s: float
    pose_cycle: int
    root_cycle: int
    pose_sample_id: str
    root_lower_sample_id: str
    root_upper_sample_id: str
    root_fraction: float
    valid: bool
    resolved_root_affine_4x4: np.ndarray
    vertices_m: np.ndarray
    normals: np.ndarray

    def evidence_dict(self) -> dict[str, Any]:
        """Return the canonical, geometry-hashed evidence for this sample."""

        return {
            "sample_id": self.sample_id,
            "scenario_time_s": self.scenario_time_s,
            "motion_time_s": self.motion_time_s,
            "pulse_time_s": self.pulse_time_s,
            "evaluated_motion": {
                "pose_time_s": self.evaluated_pose_time_s,
                "root_time_s": self.evaluated_root_time_s,
                "pose_cycle": self.pose_cycle,
                "root_cycle": self.root_cycle,
            },
            "pose_sample_id": self.pose_sample_id,
            "root": {
                "lower_sample_id": self.root_lower_sample_id,
                "upper_sample_id": self.root_upper_sample_id,
                "fraction": self.root_fraction,
                "resolved_affine_4x4": self.resolved_root_affine_4x4.tolist(),
                "resolved_affine_sha256": _array_sha256(
                    self.resolved_root_affine_4x4,
                    dtype="<f8",
                ),
            },
            "validity": {"valid": self.valid, "reason": None if self.valid else "invalid"},
            "vertices_sha256": _array_sha256(self.vertices_m, dtype="<f8"),
            "normals_sha256": _array_sha256(self.normals, dtype="<f8"),
        }


@dataclass(frozen=True, slots=True)
class AnimatedTargetBinding:
    """Deterministic result of binding an asset to explicit physical clocks."""

    asset_id: str
    asset_revision_sha256: str
    context: AnimatedTargetBindingContext
    faces: np.ndarray
    samples: tuple[BoundAnimatedTargetSample, ...]
    binding_id: str

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "binding_version": ANIMATED_TARGET_BINDING_VERSION,
            "asset": {
                "asset_id": self.asset_id,
                "revision_sha256": self.asset_revision_sha256,
            },
            "context": self.context.to_dict(),
            "topology": {
                "faces_sha256": _array_sha256(self.faces, dtype="<i8"),
                "triangle_count": int(self.faces.shape[0]),
            },
            "samples": [sample.evidence_dict() for sample in self.samples],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical-JSON-compatible evidence document."""

        payload = self._identity_payload()
        payload["binding_id"] = self.binding_id
        return payload

    def to_canonical_json(self) -> str:
        """Serialize the binding without platform- or insertion-order variance."""

        return canonical_json_bytes(self.to_dict()).decode("utf-8")


@dataclass(frozen=True, slots=True)
class AnimatedTargetAsset:
    """Validated V1 asset with source pose meshes and an independent root track."""

    manifest_path: Path
    asset_id: str
    revision_sha256: str
    resources: tuple[AnimatedTargetResource, ...]
    source_frame_id: str
    source_handedness: Literal["right", "left"]
    front_axis: str
    up_axis: str
    length_unit: str
    destination_frame_id: str
    scale_ownership: str
    translation_ownership: str
    meters_per_source_unit: float
    source_to_orchav_basis: np.ndarray
    source_to_orchav_translation_m: np.ndarray
    source_to_orchav_affine_4x4: np.ndarray
    source_face_winding: str
    reflection_winding_policy: str
    normal_policy: str
    topology_sha256: str
    faces: np.ndarray
    canonical_faces: np.ndarray
    pose_samples: tuple[AnimatedTargetPoseSample, ...]
    pose_out_of_range: Literal["error", "hold", "loop"]
    pose_cyclic: bool
    pose_period_s: float | None
    root_samples: tuple[AnimatedTargetRootSample, ...]
    root_out_of_range: Literal["error", "hold", "loop"]
    root_cyclic: bool
    root_period_s: float | None
    provenance: Mapping[str, Any]
    license: Mapping[str, Any]

    def bind(
        self,
        context: AnimatedTargetBindingContext,
        times: Sequence[AnimatedTargetBindingTime],
        *,
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> AnimatedTargetBinding:
        """Bind this asset to caller-supplied scenario/motion/pulse timestamps.

        Requests must already be in strictly increasing scenario time. This
        function never sorts them and never maps clocks by count ratio. Pose is
        sampled as an explicit step function; root translation is linear and
        root rotation uses shortest-path quaternion SLERP. Optional progress
        reports completed poses, with completion after binding identity creation.
        """

        requests = _validate_binding_times(times)
        _validate_binding_context(context, requests)
        total = len(requests)
        if on_progress is not None:
            on_progress("Binding animation poses", 0, total)
        samples: list[BoundAnimatedTargetSample] = []
        for index, request in enumerate(requests):
            samples.append(self._bind_one(request))
            # The final update also covers canonical binding identity creation.
            if on_progress is not None and index + 1 < total:
                on_progress("Binding animation poses", index + 1, total)
        bound_samples = tuple(samples)
        provisional = AnimatedTargetBinding(
            asset_id=self.asset_id,
            asset_revision_sha256=self.revision_sha256,
            context=context,
            faces=self.canonical_faces,
            samples=bound_samples,
            binding_id="",
        )
        binding_id = canonical_sha256(provisional._identity_payload())
        if on_progress is not None:
            on_progress("Binding animation poses", total, total)
        return AnimatedTargetBinding(
            asset_id=self.asset_id,
            asset_revision_sha256=self.revision_sha256,
            context=context,
            faces=self.canonical_faces,
            samples=bound_samples,
            binding_id=binding_id,
        )

    def sample_centers(
        self,
        context: AnimatedTargetBindingContext,
        times: Sequence[AnimatedTargetBindingTime],
    ) -> np.ndarray:
        """Return exact world AABB centers without normals or binding evidence.

        Inspection uses the same clock policies, articulation and root transform
        as ``bind``. Only one timestamp's vertices are retained at a time. Full
        binding still validates normals and produces canonical geometry hashes.
        """
        requests = _validate_binding_times(times)
        _validate_binding_context(context, requests)
        centers = np.empty((len(requests), 3), dtype=np.float64)
        for index, request in enumerate(requests):
            vertices = self._resolve_pose(request).vertices_m
            centers[index] = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
        return _readonly(centers, dtype=np.float64)

    def _resolve_pose(self, request: AnimatedTargetBindingTime) -> _ResolvedAnimatedPose:
        """Evaluate the shared physical-time and world-geometry contract."""
        pose_time, pose_cycle = _apply_time_policy(
            request.motion_time_s,
            self.pose_samples[0].time_s,
            self.pose_samples[-1].time_s,
            policy=self.pose_out_of_range,
            cyclic=self.pose_cyclic,
            period_s=self.pose_period_s,
            path=f"binding[{request.sample_id!r}].motion_time_s for pose",
        )
        root_time, root_cycle = _apply_time_policy(
            request.motion_time_s,
            self.root_samples[0].time_s,
            self.root_samples[-1].time_s,
            policy=self.root_out_of_range,
            cyclic=self.root_cyclic,
            period_s=self.root_period_s,
            path=f"binding[{request.sample_id!r}].motion_time_s for root",
        )

        pose_index = _step_sample_index(self.pose_samples, pose_time)
        pose = self.pose_samples[pose_index]
        root_lower, root_upper, root_fraction = _root_bracket(self.root_samples, root_time)
        translation = _linear(
            root_lower.translation,
            root_upper.translation,
            root_fraction,
        )
        rotation = _slerp_xyzw(
            root_lower.rotation_xyzw,
            root_upper.rotation_xyzw,
            root_fraction,
        )

        source_root_affine = np.eye(4, dtype=np.float64)
        source_root_affine[:3, :3] = _rotation_matrix_xyzw(rotation)
        source_root_affine[:3, 3] = translation
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                resolved_root_affine = self.source_to_orchav_affine_4x4 @ source_root_affine
                vertices_m = (
                    pose.vertices @ resolved_root_affine[:3, :3].T + resolved_root_affine[:3, 3]
                )
        except FloatingPointError as exc:
            raise AnimatedTargetError(
                f"binding[{request.sample_id!r}] produced non-finite geometry"
            ) from exc
        if not np.all(np.isfinite(resolved_root_affine)) or not np.all(np.isfinite(vertices_m)):
            raise AnimatedTargetError(
                f"binding[{request.sample_id!r}] produced non-finite geometry"
            )

        return _ResolvedAnimatedPose(
            pose=pose,
            pose_time_s=pose_time,
            root_time_s=root_time,
            pose_cycle=pose_cycle,
            root_cycle=root_cycle,
            root_lower=root_lower,
            root_upper=root_upper,
            root_fraction=root_fraction,
            rotation_xyzw=rotation,
            root_affine=resolved_root_affine,
            vertices_m=vertices_m,
        )

    def _bind_one(self, request: AnimatedTargetBindingTime) -> BoundAnimatedTargetSample:
        geometry = self._resolve_pose(request)
        pose = geometry.pose
        vertices_m = geometry.vertices_m

        if self.normal_policy == "recompute":
            normals = _vertex_normals(vertices_m, self.canonical_faces)
        else:
            if pose.normals is None:  # guarded while loading; defensive for constructed assets
                raise AnimatedTargetError(
                    f"pose sample {pose.sample_id!r} has no normals for transform policy"
                )
            source_world_normals = _rotate_vectors_xyzw(pose.normals, geometry.rotation_xyzw)
            normals = _normalize_rows(source_world_normals @ self.source_to_orchav_basis.T)

        return BoundAnimatedTargetSample(
            sample_id=request.sample_id,
            scenario_time_s=request.scenario_time_s,
            motion_time_s=request.motion_time_s,
            pulse_time_s=request.pulse_time_s,
            evaluated_pose_time_s=geometry.pose_time_s,
            evaluated_root_time_s=geometry.root_time_s,
            pose_cycle=geometry.pose_cycle,
            root_cycle=geometry.root_cycle,
            pose_sample_id=pose.sample_id,
            root_lower_sample_id=geometry.root_lower.sample_id,
            root_upper_sample_id=geometry.root_upper.sample_id,
            root_fraction=geometry.root_fraction,
            valid=True,
            resolved_root_affine_4x4=_readonly(
                geometry.root_affine,
                dtype=np.float64,
            ),
            vertices_m=_readonly(vertices_m, dtype=np.float64),
            normals=_readonly(normals, dtype=np.float64),
        )


def load_animated_target(
    manifest_path: str | Path,
    *,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> AnimatedTargetAsset:
    """Load and fully validate an asset, optionally reporting completed stage counts."""

    path = Path(manifest_path)
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise AnimatedTargetError(f"cannot read animated-target manifest {path}: {exc}") from exc
    try:
        raw = json.loads(raw_bytes, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnimatedTargetError(
            f"invalid JSON in animated-target manifest {path}: {exc}"
        ) from exc
    manifest = _object(raw, "manifest")
    _exact_keys(
        manifest,
        {
            "manifest_version",
            "asset",
            "resources",
            "coordinates",
            "topology",
            "motion",
            "provenance",
            "license",
        },
        "manifest",
    )
    version = _integer(manifest["manifest_version"], "manifest_version", minimum=1)
    if version != ANIMATED_TARGET_MANIFEST_VERSION:
        raise AnimatedTargetError(
            f"unsupported manifest_version={version}; expected {ANIMATED_TARGET_MANIFEST_VERSION}"
        )

    asset_raw = _object(manifest["asset"], "asset")
    _exact_keys(asset_raw, {"asset_id", "revision_sha256"}, "asset")
    asset_id = _stable_id(asset_raw["asset_id"], "asset.asset_id")
    revision = _sha256(asset_raw["revision_sha256"], "asset.revision_sha256")

    resources, authenticated_resource_bytes = _parse_resources(
        manifest["resources"],
        path.parent,
        on_progress=on_progress,
    )
    resource_by_id = {resource.resource_id: resource for resource in resources}
    coordinate_values = _parse_coordinates(manifest["coordinates"])
    topology_values = _parse_topology(manifest["topology"])
    motion_values = _parse_motion(manifest["motion"])
    provenance = _parse_provenance(manifest["provenance"])
    license_values = _parse_license(manifest["license"])

    expected_revision = animated_target_revision_sha256(manifest)
    if revision != expected_revision:
        raise AnimatedTargetError(
            "asset.revision_sha256 does not match the canonical manifest revision: "
            f"expected {expected_revision}, got {revision}"
        )

    _validate_adapter_provenance(provenance, path.parent, license_values)

    pose_samples, faces = _load_pose_samples(
        motion_values["pose_samples"],
        resource_by_id,
        authenticated_resource_bytes,
        vertex_count=topology_values["vertex_count"],
        triangle_count=topology_values["triangle_count"],
        normal_policy=topology_values["normal_policy"],
        closed_surface=topology_values["closed_surface"],
        on_progress=on_progress,
    )
    used_resources = {sample.resource_id for sample in pose_samples}
    declared_resources = set(resource_by_id)
    if used_resources != declared_resources:
        missing = sorted(declared_resources - used_resources)
        unknown = sorted(used_resources - declared_resources)
        raise AnimatedTargetError(
            "resources must be the exact hash closure of pose sample references; "
            f"unreferenced={missing}, undeclared={unknown}"
        )

    computed_topology_sha256 = canonical_sha256(
        {
            "representation": "indexed_triangles",
            "vertex_count": topology_values["vertex_count"],
            "faces": faces.astype(np.int64).tolist(),
        }
    )
    if topology_values["topology_sha256"] != computed_topology_sha256:
        raise AnimatedTargetError(
            "topology.topology_sha256 does not match the fixed per-pose topology: "
            f"expected {computed_topology_sha256}, got {topology_values['topology_sha256']}"
        )

    basis = coordinate_values["basis"]
    reflected = float(np.linalg.det(basis)) < 0.0
    if reflected and topology_values["reflection_winding_policy"] != "reverse":
        raise AnimatedTargetError(
            "a reflecting source_to_orchav_basis requires "
            "topology.reflection_winding_policy='reverse'"
        )
    canonical_faces = faces[:, [0, 2, 1]] if reflected else faces.copy()

    return AnimatedTargetAsset(
        manifest_path=path.resolve(),
        asset_id=asset_id,
        revision_sha256=revision,
        resources=resources,
        source_frame_id=coordinate_values["source_frame_id"],
        source_handedness=coordinate_values["source_handedness"],
        front_axis=coordinate_values["front_axis"],
        up_axis=coordinate_values["up_axis"],
        length_unit=coordinate_values["length_unit"],
        destination_frame_id=coordinate_values["destination_frame_id"],
        scale_ownership=coordinate_values["scale_ownership"],
        translation_ownership=coordinate_values["translation_ownership"],
        meters_per_source_unit=coordinate_values["meters_per_source_unit"],
        source_to_orchav_basis=_readonly(basis, dtype=np.float64),
        source_to_orchav_translation_m=_readonly(
            coordinate_values["translation_m"],
            dtype=np.float64,
        ),
        source_to_orchav_affine_4x4=_readonly(
            coordinate_values["affine"],
            dtype=np.float64,
        ),
        source_face_winding=topology_values["source_face_winding"],
        reflection_winding_policy=topology_values["reflection_winding_policy"],
        normal_policy=topology_values["normal_policy"],
        topology_sha256=topology_values["topology_sha256"],
        faces=_readonly(faces, dtype=np.int64),
        canonical_faces=_readonly(canonical_faces, dtype=np.int64),
        pose_samples=pose_samples,
        pose_out_of_range=motion_values["pose_out_of_range"],
        pose_cyclic=motion_values["pose_cyclic"],
        pose_period_s=motion_values["pose_period_s"],
        root_samples=motion_values["root_samples"],
        root_out_of_range=motion_values["root_out_of_range"],
        root_cyclic=motion_values["root_cyclic"],
        root_period_s=motion_values["root_period_s"],
        provenance=provenance,
        license=license_values,
    )


def _validate_adapter_provenance(
    provenance: Mapping[str, Any], root: Path, license_values: Mapping[str, Any]
) -> None:
    """Authenticate the reserved offline-adapter sidecar from the same bundle.

    Generic V1 assets do not need an adapter sidecar. Once the reserved source
    identity is declared, its sibling JSON is part of the required authenticated
    closure; a missing, replaced or redirected sidecar is not a valid asset.
    """
    reserved = [
        source_id
        for source_id in provenance["source_ids"]
        if source_id == "adapter_provenance" or source_id.startswith("adapter_provenance:")
    ]
    if not reserved:
        return
    if len(reserved) != 1:
        raise AnimatedTargetError("provenance must contain exactly one adapter_provenance identity")
    prefix = "adapter_provenance:sha256:"
    if not reserved[0].startswith(prefix):
        raise AnimatedTargetError(
            "adapter_provenance identity must use adapter_provenance:sha256:<digest>"
        )
    digest = _sha256(reserved[0][len(prefix) :], "adapter_provenance identity digest")
    sibling = root / "adapter_provenance.json"
    if sibling.is_symlink() or getattr(sibling, "is_junction", lambda: False)():
        raise AnimatedTargetError("adapter_provenance.json must not be a symlink or junction")
    sidecar_path = _resolve_resource(root, "adapter_provenance.json")
    try:
        payload = sidecar_path.read_bytes()
    except OSError as exc:
        raise AnimatedTargetError(f"cannot read adapter_provenance.json: {exc}") from exc
    if hashlib.sha256(payload).hexdigest() != digest:
        raise AnimatedTargetError("adapter_provenance.json SHA-256 mismatch")
    try:
        raw = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnimatedTargetError(f"invalid JSON in adapter_provenance.json: {exc}") from exc
    sidecar = _object(raw, "adapter_provenance")
    version = _integer(
        sidecar.get("adapter_contract_version"),
        "adapter_provenance.adapter_contract_version",
        minimum=1,
    )
    if version != 1:
        raise AnimatedTargetError("unsupported adapter_provenance.adapter_contract_version")
    if sidecar.get("source_type") != provenance["source_type"]:
        raise AnimatedTargetError("adapter_provenance.source_type must match manifest provenance")
    if sidecar.get("redistribution") != license_values["redistribution"]:
        raise AnimatedTargetError("adapter_provenance.redistribution must match manifest license")
    sources = _object(sidecar.get("sources"), "adapter_provenance.sources")
    if not sources:
        raise AnimatedTargetError("adapter_provenance.sources must not be empty")
    for name, raw_source in sources.items():
        record = _object(raw_source, f"adapter_provenance.sources.{name}")
        _sha256(record.get("sha256"), f"adapter_provenance.sources.{name}.sha256")
        _integer(
            record.get("size_bytes"), f"adapter_provenance.sources.{name}.size_bytes", minimum=1
        )
    _object(sidecar.get("operations"), "adapter_provenance.operations")
    # Reject non-JSON numeric values anywhere in authenticated descriptive fields.
    canonical_json_bytes(sidecar)


def _parse_resources(
    raw: Any,
    root: Path,
    *,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[tuple[AnimatedTargetResource, ...], Mapping[str, bytes]]:
    entries = _list(raw, "resources")
    if not entries:
        raise AnimatedTargetError("resources must contain at least one pose resource")
    result: list[AnimatedTargetResource] = []
    authenticated_bytes: dict[str, bytes] = {}
    ids: set[str] = set()
    paths: set[str] = set()
    if on_progress is not None:
        on_progress("Verifying pose resources", 0, len(entries))
    for index, value in enumerate(entries):
        item_path = f"resources[{index}]"
        item = _object(value, item_path)
        _exact_keys(
            item,
            {"resource_id", "path", "sha256", "size_bytes", "media_type"},
            item_path,
        )
        resource_id = _stable_id(item["resource_id"], f"{item_path}.resource_id")
        relative_path = _safe_resource_path(item["path"], f"{item_path}.path")
        digest = _sha256(item["sha256"], f"{item_path}.sha256")
        size_bytes = _integer(item["size_bytes"], f"{item_path}.size_bytes", minimum=1)
        media_type = _string(item["media_type"], f"{item_path}.media_type")
        if media_type != "application/vnd.orchav.animated-pose+json":
            raise AnimatedTargetError(
                f"{item_path}.media_type must be " "'application/vnd.orchav.animated-pose+json'"
            )
        if resource_id in ids:
            raise AnimatedTargetError(f"duplicate resource_id {resource_id!r}")
        if relative_path in paths:
            raise AnimatedTargetError(f"duplicate resource path {relative_path!r}")
        ids.add(resource_id)
        paths.add(relative_path)

        resource_path = _resolve_resource(root, relative_path)
        try:
            payload = resource_path.read_bytes()
        except OSError as exc:
            raise AnimatedTargetError(
                f"cannot read resource {resource_id!r} at {relative_path!r}: {exc}"
            ) from exc
        if len(payload) != size_bytes:
            raise AnimatedTargetError(
                f"resource {resource_id!r} size mismatch: expected {size_bytes}, "
                f"got {len(payload)}"
            )
        actual_digest = hashlib.sha256(payload).hexdigest()
        if actual_digest != digest:
            raise AnimatedTargetError(
                f"resource {resource_id!r} SHA-256 mismatch: expected {digest}, "
                f"got {actual_digest}"
            )
        result.append(
            AnimatedTargetResource(
                resource_id=resource_id,
                path=relative_path,
                sha256=digest,
                size_bytes=size_bytes,
                media_type=media_type,
            )
        )
        authenticated_bytes[resource_id] = payload
        if on_progress is not None:
            on_progress("Verifying pose resources", index + 1, len(entries))
    return tuple(result), authenticated_bytes


def _parse_coordinates(raw: Any) -> dict[str, Any]:
    coordinates = _object(raw, "coordinates")
    _exact_keys(
        coordinates,
        {
            "source_frame_id",
            "source_handedness",
            "front_axis",
            "up_axis",
            "length_unit",
            "destination_frame_id",
            "source_to_orchav_affine_4x4",
            "scale_ownership",
            "translation_ownership",
        },
        "coordinates",
    )
    source_frame_id = _stable_id(coordinates["source_frame_id"], "coordinates.source_frame_id")
    handedness = coordinates["source_handedness"]
    if handedness not in ("right", "left"):
        raise AnimatedTargetError("coordinates.source_handedness must be 'right' or 'left'")
    front_axis = _axis(coordinates["front_axis"], "coordinates.front_axis")
    up_axis = _axis(coordinates["up_axis"], "coordinates.up_axis")
    if abs(float(np.dot(_axis_vector(front_axis), _axis_vector(up_axis)))) > 1e-12:
        raise AnimatedTargetError("coordinates.front_axis and up_axis must be orthogonal")
    length_unit = _string(coordinates["length_unit"], "coordinates.length_unit")
    destination_frame_id = _stable_id(
        coordinates["destination_frame_id"],
        "coordinates.destination_frame_id",
    )
    if destination_frame_id != "orchav:world_rh_x_front_z_up_m":
        raise AnimatedTargetError(
            "coordinates.destination_frame_id must be " "'orchav:world_rh_x_front_z_up_m'"
        )
    scale_ownership = coordinates["scale_ownership"]
    if scale_ownership != "uniform_scale_in_affine_m_per_source_unit":
        raise AnimatedTargetError(
            "coordinates.scale_ownership must be " "'uniform_scale_in_affine_m_per_source_unit'"
        )
    translation_ownership = coordinates["translation_ownership"]
    if translation_ownership != "translation_in_affine_destination_m":
        raise AnimatedTargetError(
            "coordinates.translation_ownership must be " "'translation_in_affine_destination_m'"
        )
    affine = _matrix4(
        coordinates["source_to_orchav_affine_4x4"],
        "coordinates.source_to_orchav_affine_4x4",
    )
    if not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12, rtol=0.0):
        raise AnimatedTargetError(
            "coordinates.source_to_orchav_affine_4x4 must have final row [0, 0, 0, 1]"
        )
    linear = affine[:3, :3]
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            column_scales = np.linalg.norm(linear, axis=0)
    except FloatingPointError as exc:
        raise AnimatedTargetError(
            "coordinates.source_to_orchav_affine_4x4 scale is not finite"
        ) from exc
    if not np.all(np.isfinite(column_scales)):
        raise AnimatedTargetError("coordinates.source_to_orchav_affine_4x4 scale is not finite")
    meters_per_source_unit = float(column_scales[0])
    if meters_per_source_unit <= 0.0 or not np.allclose(
        column_scales,
        meters_per_source_unit,
        atol=1e-12,
        rtol=0.0,
    ):
        raise AnimatedTargetError(
            "coordinates.source_to_orchav_affine_4x4 must own one positive uniform scale"
        )
    basis = linear / meters_per_source_unit
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-12, rtol=0.0):
        raise AnimatedTargetError(
            "coordinates.source_to_orchav_affine_4x4 linear basis must be orthonormal"
        )
    determinant = float(np.linalg.det(basis))
    expected_sign = 1.0 if handedness == "right" else -1.0
    if not math.isclose(determinant, expected_sign, abs_tol=1e-12, rel_tol=0.0):
        raise AnimatedTargetError(
            "coordinates.source_to_orchav_affine_4x4 basis determinant must match "
            "source_handedness; "
            f"expected {expected_sign:+.0f}, got {determinant:+.12g}"
        )
    if not np.allclose(basis @ _axis_vector(front_axis), (1.0, 0.0, 0.0), atol=1e-12):
        raise AnimatedTargetError(
            "coordinates.source_to_orchav_affine_4x4 must map front_axis to ORCHAV +X"
        )
    if not np.allclose(basis @ _axis_vector(up_axis), (0.0, 0.0, 1.0), atol=1e-12):
        raise AnimatedTargetError(
            "coordinates.source_to_orchav_affine_4x4 must map up_axis to ORCHAV +Z"
        )
    return {
        "source_frame_id": source_frame_id,
        "source_handedness": handedness,
        "front_axis": front_axis,
        "up_axis": up_axis,
        "length_unit": length_unit,
        "destination_frame_id": destination_frame_id,
        "scale_ownership": scale_ownership,
        "translation_ownership": translation_ownership,
        "meters_per_source_unit": meters_per_source_unit,
        "basis": basis,
        "translation_m": affine[:3, 3],
        "affine": affine,
    }


def _parse_topology(raw: Any) -> dict[str, Any]:
    topology = _object(raw, "topology")
    _exact_keys(
        topology,
        {
            "representation",
            "mode",
            "vertex_count",
            "triangle_count",
            "topology_sha256",
            "closed_surface",
            "source_face_winding",
            "reflection_winding_policy",
            "normal_policy",
        },
        "topology",
    )
    if topology["representation"] != "indexed_triangles":
        raise AnimatedTargetError("topology.representation must be 'indexed_triangles'")
    if topology["mode"] != "fixed_across_poses":
        raise AnimatedTargetError("topology.mode must be 'fixed_across_poses' in V1")
    source_face_winding = topology["source_face_winding"]
    if source_face_winding != "counter_clockwise_from_outside":
        raise AnimatedTargetError(
            "topology.source_face_winding must be 'counter_clockwise_from_outside'"
        )
    reflection_policy = topology["reflection_winding_policy"]
    if reflection_policy not in ("reverse", "unchanged_if_no_reflection"):
        raise AnimatedTargetError(
            "topology.reflection_winding_policy must be 'reverse' or "
            "'unchanged_if_no_reflection'"
        )
    normal_policy = topology["normal_policy"]
    if normal_policy not in ("recompute", "transform"):
        raise AnimatedTargetError("topology.normal_policy must be 'recompute' or 'transform'")
    if not isinstance(topology["closed_surface"], bool):
        raise AnimatedTargetError("topology.closed_surface must be a boolean")
    return {
        "vertex_count": _integer(topology["vertex_count"], "topology.vertex_count", minimum=3),
        "triangle_count": _integer(
            topology["triangle_count"], "topology.triangle_count", minimum=1
        ),
        "topology_sha256": _sha256(topology["topology_sha256"], "topology.topology_sha256"),
        "closed_surface": topology["closed_surface"],
        "source_face_winding": source_face_winding,
        "reflection_winding_policy": reflection_policy,
        "normal_policy": normal_policy,
    }


def _parse_motion(raw: Any) -> dict[str, Any]:
    motion = _object(raw, "motion")
    _exact_keys(motion, {"time_axis", "pose", "root"}, "motion")
    if motion["time_axis"] != "asset_time_s":
        raise AnimatedTargetError("motion.time_axis must be 'asset_time_s'")
    pose = _object(motion["pose"], "motion.pose")
    _exact_keys(
        pose,
        {"interpolation", "out_of_range", "cyclic", "period_s", "samples"},
        "motion.pose",
    )
    if pose["interpolation"] != "step":
        raise AnimatedTargetError("motion.pose.interpolation must be 'step' in V1")
    pose_policy, pose_cyclic, pose_period = _playback_policy(pose, "motion.pose")
    pose_entries = _list(pose["samples"], "motion.pose.samples")
    if not pose_entries:
        raise AnimatedTargetError("motion.pose.samples must not be empty")
    pose_samples: list[tuple[str, float, str]] = []
    pose_ids: set[str] = set()
    pose_times: list[float] = []
    for index, value in enumerate(pose_entries):
        label = f"motion.pose.samples[{index}]"
        item = _object(value, label)
        _exact_keys(item, {"sample_id", "time_s", "resource_id"}, label)
        sample_id = _stable_id(item["sample_id"], f"{label}.sample_id")
        if sample_id in pose_ids:
            raise AnimatedTargetError(f"duplicate pose sample_id {sample_id!r}")
        pose_ids.add(sample_id)
        time_s = _time(item["time_s"], f"{label}.time_s")
        resource_id = _stable_id(item["resource_id"], f"{label}.resource_id")
        pose_samples.append((sample_id, time_s, resource_id))
        pose_times.append(time_s)
    _strictly_increasing(pose_times, "motion.pose.samples time_s")
    _validate_period(pose_times, pose_cyclic, pose_period, "motion.pose")

    root = _object(motion["root"], "motion.root")
    _exact_keys(
        root,
        {
            "translation_interpolation",
            "rotation_interpolation",
            "rotation_representation",
            "out_of_range",
            "cyclic",
            "period_s",
            "samples",
        },
        "motion.root",
    )
    if root["translation_interpolation"] != "linear":
        raise AnimatedTargetError("motion.root.translation_interpolation must be 'linear'")
    if root["rotation_interpolation"] != "slerp_shortest":
        raise AnimatedTargetError("motion.root.rotation_interpolation must be 'slerp_shortest'")
    if root["rotation_representation"] != "active_quaternion_xyzw":
        raise AnimatedTargetError(
            "motion.root.rotation_representation must be 'active_quaternion_xyzw'"
        )
    root_policy, root_cyclic, root_period = _playback_policy(root, "motion.root")
    root_entries = _list(root["samples"], "motion.root.samples")
    if not root_entries:
        raise AnimatedTargetError("motion.root.samples must not be empty")
    root_samples: list[AnimatedTargetRootSample] = []
    root_ids: set[str] = set()
    root_times: list[float] = []
    for index, value in enumerate(root_entries):
        label = f"motion.root.samples[{index}]"
        item = _object(value, label)
        _exact_keys(
            item,
            {"sample_id", "time_s", "translation", "rotation_xyzw"},
            label,
        )
        sample_id = _stable_id(item["sample_id"], f"{label}.sample_id")
        if sample_id in root_ids:
            raise AnimatedTargetError(f"duplicate root sample_id {sample_id!r}")
        root_ids.add(sample_id)
        time_s = _time(item["time_s"], f"{label}.time_s")
        translation = _vector3(item["translation"], f"{label}.translation")
        quaternion = _quaternion_xyzw(item["rotation_xyzw"], f"{label}.rotation_xyzw")
        root_samples.append(
            AnimatedTargetRootSample(
                sample_id=sample_id,
                time_s=time_s,
                translation=_readonly(translation, dtype=np.float64),
                rotation_xyzw=_readonly(quaternion, dtype=np.float64),
            )
        )
        root_times.append(time_s)
    _strictly_increasing(root_times, "motion.root.samples time_s")
    _validate_period(root_times, root_cyclic, root_period, "motion.root")
    if pose_cyclic:
        first_resource = pose_samples[0][2]
        last_resource = pose_samples[-1][2]
        if first_resource != last_resource:
            raise AnimatedTargetError(
                "cyclic motion.pose must repeat the first resource at the final sample"
            )
    if root_cyclic:
        first = root_samples[0]
        last = root_samples[-1]
        if not np.array_equal(first.translation, last.translation) or not _same_rotation(
            first.rotation_xyzw, last.rotation_xyzw
        ):
            raise AnimatedTargetError(
                "cyclic motion.root must repeat its first transform at the final sample"
            )
    return {
        "pose_samples": tuple(pose_samples),
        "pose_out_of_range": pose_policy,
        "pose_cyclic": pose_cyclic,
        "pose_period_s": pose_period,
        "root_samples": tuple(root_samples),
        "root_out_of_range": root_policy,
        "root_cyclic": root_cyclic,
        "root_period_s": root_period,
    }


def _parse_provenance(raw: Any) -> Mapping[str, Any]:
    provenance = _object(raw, "provenance")
    _exact_keys(
        provenance,
        {"creator", "source_type", "source_ids", "description"},
        "provenance",
    )
    creator = _string(provenance["creator"], "provenance.creator")
    source_type = _stable_id(provenance["source_type"], "provenance.source_type")
    description = _string(provenance["description"], "provenance.description")
    raw_ids = _list(provenance["source_ids"], "provenance.source_ids")
    source_ids = [
        _stable_id(value, f"provenance.source_ids[{index}]") for index, value in enumerate(raw_ids)
    ]
    if len(source_ids) != len(set(source_ids)):
        raise AnimatedTargetError("provenance.source_ids must be unique")
    return MappingProxyType(
        {
            "creator": creator,
            "source_type": source_type,
            "source_ids": tuple(source_ids),
            "description": description,
        }
    )


def _parse_license(raw: Any) -> Mapping[str, Any]:
    license_values = _object(raw, "license")
    _exact_keys(license_values, {"spdx_id", "redistribution", "notice"}, "license")
    redistribution = license_values["redistribution"]
    if redistribution not in ("allowed", "user_supplied_only", "prohibited"):
        raise AnimatedTargetError(
            "license.redistribution must be 'allowed', 'user_supplied_only', or 'prohibited'"
        )
    return MappingProxyType(
        {
            "spdx_id": _string(license_values["spdx_id"], "license.spdx_id"),
            "redistribution": redistribution,
            "notice": _string(license_values["notice"], "license.notice"),
        }
    )


def _load_pose_samples(
    specifications: Sequence[tuple[str, float, str]],
    resources: Mapping[str, AnimatedTargetResource],
    authenticated_resource_bytes: Mapping[str, bytes],
    *,
    vertex_count: int,
    triangle_count: int,
    normal_policy: str,
    closed_surface: bool,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> tuple[tuple[AnimatedTargetPoseSample, ...], np.ndarray]:
    samples: list[AnimatedTargetPoseSample] = []
    fixed_faces: np.ndarray | None = None
    decoded: dict[str, AnimatedTargetPoseSample] = {}
    if on_progress is not None:
        on_progress("Validating pose geometry", 0, len(specifications))
    for sample_id, time_s, resource_id in specifications:
        resource = resources.get(resource_id)
        if resource is None:
            raise AnimatedTargetError(
                f"pose sample {sample_id!r} references undeclared resource {resource_id!r}"
            )
        # A pose sequence may reuse a mesh at many timestamps. Its authenticated
        # bytes and validation rules are identical within this load; share the
        # immutable arrays while retaining each sample's identity and time.
        cached = decoded.get(resource_id)
        if cached is not None:
            samples.append(replace(cached, sample_id=sample_id, time_s=time_s))
            if on_progress is not None:
                on_progress("Validating pose geometry", len(samples), len(specifications))
            continue
        try:
            raw = json.loads(
                authenticated_resource_bytes[resource_id],
                object_pairs_hook=_unique_json_object,
            )
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AnimatedTargetError(
                f"invalid pose resource {resource_id!r} at {resource.path!r}: {exc}"
            ) from exc
        payload = _object(raw, f"resource[{resource_id!r}]")
        expected_keys = {"vertices", "faces"}
        if normal_policy == "transform":
            expected_keys.add("normals")
        _exact_keys(payload, expected_keys, f"resource[{resource_id!r}]")
        vertices = _numeric_matrix(payload["vertices"], f"resource[{resource_id!r}].vertices", 3)
        if vertices.shape != (vertex_count, 3):
            raise AnimatedTargetError(
                f"resource {resource_id!r} vertices shape must be ({vertex_count}, 3), "
                f"got {vertices.shape}"
            )
        faces = _integer_matrix(payload["faces"], f"resource[{resource_id!r}].faces", 3)
        if faces.shape != (triangle_count, 3):
            raise AnimatedTargetError(
                f"resource {resource_id!r} faces shape must be ({triangle_count}, 3), "
                f"got {faces.shape}"
            )
        if np.any(faces < 0) or np.any(faces >= vertex_count):
            raise AnimatedTargetError(f"resource {resource_id!r} has out-of-range face indices")
        if np.any(
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 2] == faces[:, 0])
        ):
            raise AnimatedTargetError(f"resource {resource_id!r} has repeated triangle indices")
        if closed_surface:
            _validate_closed_surface_edges(faces, resource_id=resource_id)
        if fixed_faces is None:
            fixed_faces = faces
        elif not np.array_equal(faces, fixed_faces):
            raise AnimatedTargetError(
                f"resource {resource_id!r} changes topology; V1 requires exact fixed face order"
            )
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                face_vectors = np.cross(
                    vertices[faces[:, 1]] - vertices[faces[:, 0]],
                    vertices[faces[:, 2]] - vertices[faces[:, 0]],
                )
                face_lengths = np.linalg.norm(face_vectors, axis=1)
        except FloatingPointError as exc:
            raise AnimatedTargetError(
                f"resource {resource_id!r} derived face geometry is not finite"
            ) from exc
        if not np.all(np.isfinite(face_vectors)) or not np.all(np.isfinite(face_lengths)):
            raise AnimatedTargetError(
                f"resource {resource_id!r} derived face geometry is not finite"
            )
        if np.any(face_lengths <= _VECTOR_EPSILON):
            raise AnimatedTargetError(f"resource {resource_id!r} has a degenerate triangle")
        if closed_surface:
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    signed_volume = float(
                        np.sum(
                            np.einsum(
                                "ij,ij->i",
                                vertices[faces[:, 0]],
                                np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
                            )
                        )
                        / 6.0
                    )
            except FloatingPointError as exc:
                raise AnimatedTargetError(
                    f"resource {resource_id!r} signed volume is not finite"
                ) from exc
            if not math.isfinite(signed_volume):
                raise AnimatedTargetError(f"resource {resource_id!r} signed volume is not finite")
            # Indexed triangle winding is defined geometrically using the
            # right-handed cross product, independently of the named source
            # axes. A reflecting basis is handled later by reversing indices.
            if signed_volume <= _VECTOR_EPSILON:
                raise AnimatedTargetError(
                    f"resource {resource_id!r} face winding is inconsistent with "
                    "outward counter-clockwise topology"
                )

        normals: np.ndarray | None = None
        if normal_policy == "transform":
            normals = _numeric_matrix(payload["normals"], f"resource[{resource_id!r}].normals", 3)
            if normals.shape != (vertex_count, 3):
                raise AnimatedTargetError(
                    f"resource {resource_id!r} normals shape must be ({vertex_count}, 3), "
                    f"got {normals.shape}"
                )
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    lengths = np.linalg.norm(normals, axis=1)
            except FloatingPointError as exc:
                raise AnimatedTargetError(
                    f"resource {resource_id!r} normals must be finite unit vectors"
                ) from exc
            if not np.all(np.isfinite(lengths)) or not np.allclose(
                lengths,
                1.0,
                atol=1e-6,
                rtol=0.0,
            ):
                raise AnimatedTargetError(
                    f"resource {resource_id!r} normals must be finite unit vectors"
                )
            expected_normals = _vertex_normals(vertices, faces)
            if np.any(np.einsum("ij,ij->i", normals, expected_normals) <= 0.0):
                raise AnimatedTargetError(
                    f"resource {resource_id!r} normals contradict face winding"
                )
        samples.append(
            AnimatedTargetPoseSample(
                sample_id=sample_id,
                time_s=time_s,
                resource_id=resource_id,
                vertices=_readonly(vertices, dtype=np.float64),
                faces=_readonly(faces, dtype=np.int64),
                normals=(_readonly(normals, dtype=np.float64) if normals is not None else None),
            )
        )
        decoded[resource_id] = samples[-1]
        if on_progress is not None:
            on_progress("Validating pose geometry", len(samples), len(specifications))
    if fixed_faces is None:  # specifications are validated as non-empty
        raise AnimatedTargetError("motion.pose.samples must not be empty")
    return tuple(samples), fixed_faces


def _validate_binding_times(
    times: Sequence[AnimatedTargetBindingTime],
) -> tuple[AnimatedTargetBindingTime, ...]:
    if isinstance(times, (str, bytes)):
        raise AnimatedTargetError("binding times must be a sequence of binding records")
    result: list[AnimatedTargetBindingTime] = []
    ids: set[str] = set()
    scenario_times: list[float] = []
    pulse_times: list[float] = []
    for index, raw in enumerate(times):
        if not isinstance(raw, AnimatedTargetBindingTime):
            raise AnimatedTargetError(
                f"binding times[{index}] must be AnimatedTargetBindingTime, got {type(raw).__name__}"
            )
        sample_id = _stable_id(raw.sample_id, f"binding times[{index}].sample_id")
        if sample_id in ids:
            raise AnimatedTargetError(f"duplicate binding sample_id {sample_id!r}")
        ids.add(sample_id)
        scenario_time = _time(raw.scenario_time_s, f"binding times[{index}].scenario_time_s")
        motion_time = _finite_number(raw.motion_time_s, f"binding times[{index}].motion_time_s")
        pulse_time = (
            None
            if raw.pulse_time_s is None
            else _time(raw.pulse_time_s, f"binding times[{index}].pulse_time_s")
        )
        if pulse_time is not None:
            pulse_times.append(pulse_time)
        result.append(
            AnimatedTargetBindingTime(
                sample_id=sample_id,
                scenario_time_s=scenario_time,
                motion_time_s=motion_time,
                pulse_time_s=pulse_time,
            )
        )
        scenario_times.append(scenario_time)
    if not result:
        raise AnimatedTargetError("binding times must not be empty")
    _strictly_increasing(scenario_times, "binding scenario_time_s")
    _strictly_increasing(pulse_times, "binding pulse_time_s")
    return tuple(result)


def _validate_binding_context(
    context: AnimatedTargetBindingContext,
    requests: Sequence[AnimatedTargetBindingTime],
) -> None:
    if not isinstance(context, AnimatedTargetBindingContext):
        raise AnimatedTargetError("binding context must be AnimatedTargetBindingContext")
    if any(request.pulse_time_s is not None for request in requests) and (
        context.pulse_clock_id is None
    ):
        raise AnimatedTargetError(
            "binding_context.pulse_clock_id is required when pulse times are present"
        )
    if context.retiming_policy != "affine_scenario_to_motion":
        return
    for request in requests:
        expected = context.playback_rate * request.scenario_time_s + context.playback_offset_s
        if not math.isclose(
            request.motion_time_s,
            expected,
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            raise AnimatedTargetError(
                f"binding sample {request.sample_id!r} motion_time_s does not match "
                "the affine scenario-to-motion retiming policy: "
                f"expected {expected}, got {request.motion_time_s}"
            )


def _playback_policy(
    track: Mapping[str, Any], path: str
) -> tuple[Literal["error", "hold", "loop"], bool, float | None]:
    policy = track["out_of_range"]
    if policy not in ("error", "hold", "loop"):
        raise AnimatedTargetError(f"{path}.out_of_range must be 'error', 'hold', or 'loop'")
    if not isinstance(track["cyclic"], bool):
        raise AnimatedTargetError(f"{path}.cyclic must be a boolean")
    cyclic = track["cyclic"]
    period_value = track["period_s"]
    period = None if period_value is None else _finite_number(period_value, f"{path}.period_s")
    if policy == "loop" and not cyclic:
        raise AnimatedTargetError(f"{path}.out_of_range='loop' requires cyclic=true")
    if cyclic and policy != "loop":
        raise AnimatedTargetError(f"{path}.cyclic=true requires out_of_range='loop'")
    if cyclic and (period is None or period <= 0.0):
        raise AnimatedTargetError(f"{path}.period_s must be positive for cyclic playback")
    if not cyclic and period is not None:
        raise AnimatedTargetError(f"{path}.period_s must be null when cyclic=false")
    return policy, cyclic, period


def _validate_period(times: Sequence[float], cyclic: bool, period: float | None, path: str) -> None:
    if not cyclic:
        return
    assert period is not None
    expected = times[-1] - times[0]
    if len(times) < 2 or not math.isclose(period, expected, abs_tol=1e-12, rel_tol=0.0):
        raise AnimatedTargetError(
            f"{path}.period_s must exactly equal final minus first sample time ({expected})"
        )


def _apply_time_policy(
    value: float,
    start: float,
    end: float,
    *,
    policy: Literal["error", "hold", "loop"],
    cyclic: bool,
    period_s: float | None,
    path: str,
) -> tuple[float, int]:
    if start <= value <= end:
        return value, 0
    if policy == "error":
        raise AnimatedTargetError(f"{path}={value} is outside [{start}, {end}]")
    if policy == "hold":
        return min(max(value, start), end), 0
    if not cyclic or period_s is None:  # validated at load; defensive for constructed assets
        raise AnimatedTargetError(f"{path} requests undeclared cyclic playback")
    wrapped = start + ((value - start) % period_s)
    evaluated = start if math.isclose(wrapped, end, abs_tol=1e-12, rel_tol=0.0) else wrapped
    return evaluated, math.floor((value - start) / period_s)


def _step_sample_index(samples: Sequence[AnimatedTargetPoseSample], time_s: float) -> int:
    index = 0
    for candidate, sample in enumerate(samples):
        if sample.time_s > time_s:
            break
        index = candidate
    return index


def _root_bracket(
    samples: Sequence[AnimatedTargetRootSample], time_s: float
) -> tuple[AnimatedTargetRootSample, AnimatedTargetRootSample, float]:
    if time_s <= samples[0].time_s:
        return samples[0], samples[0], 0.0
    if time_s >= samples[-1].time_s:
        return samples[-1], samples[-1], 0.0
    for upper_index in range(1, len(samples)):
        upper = samples[upper_index]
        if time_s == upper.time_s:
            return upper, upper, 0.0
        if time_s < upper.time_s:
            lower = samples[upper_index - 1]
            fraction = (time_s - lower.time_s) / (upper.time_s - lower.time_s)
            return lower, upper, float(fraction)
    return samples[-1], samples[-1], 0.0


def _linear(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = left + float(fraction) * (right - left)
    except FloatingPointError as exc:
        raise AnimatedTargetError("linear interpolation produced non-finite values") from exc
    if not np.all(np.isfinite(result)):
        raise AnimatedTargetError("linear interpolation produced non-finite values")
    return result


def _slerp_xyzw(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    amount = min(max(float(fraction), 0.0), 1.0)
    end = right
    dot = float(np.dot(left, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        return _normalize_vector(left + amount * (end - left))
    theta = math.acos(dot)
    sine = math.sin(theta)
    return _normalize_vector(
        (math.sin((1.0 - amount) * theta) / sine) * left + (math.sin(amount * theta) / sine) * end
    )


def _rotate_vectors_xyzw(vectors: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    vector_part = quaternion[:3]
    scalar = float(quaternion[3])
    first_cross = np.cross(np.broadcast_to(vector_part, vectors.shape), vectors)
    second_cross = np.cross(np.broadcast_to(vector_part, vectors.shape), first_cross)
    return vectors + 2.0 * (scalar * first_cross + second_cross)


def _rotation_matrix_xyzw(quaternion: np.ndarray) -> np.ndarray:
    """Return the active 3x3 rotation represented by a unit xyzw quaternion."""

    x, y, z, w = (float(component) for component in quaternion)
    return np.array(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            face_normals = np.cross(
                vertices[faces[:, 1]] - vertices[faces[:, 0]],
                vertices[faces[:, 2]] - vertices[faces[:, 0]],
            )
            # Accumulate in the same corner-major order as three add.at calls.
            # bincount avoids the slower multidimensional scatter while keeping
            # each vertex's floating-point addition order (and evidence hashes).
            indices = faces.T.ravel()
            normals = np.column_stack(
                [
                    np.bincount(
                        indices,
                        weights=np.tile(face_normals[:, axis], 3),
                        minlength=len(vertices),
                    )
                    for axis in range(3)
                ]
            )
    except FloatingPointError as exc:
        raise AnimatedTargetError("derived vertex normals are not finite") from exc
    if not np.all(np.isfinite(face_normals)) or not np.all(np.isfinite(normals)):
        raise AnimatedTargetError("derived vertex normals are not finite")
    return _normalize_rows(normals)


def _validate_closed_surface_edges(faces: np.ndarray, *, resource_id: str) -> None:
    """Require every closed-mesh edge exactly twice with opposite directions."""

    edge_directions: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face in faces:
        for start, end in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            key = (min(start, end), max(start, end))
            edge_directions.setdefault(key, []).append((start, end))
    invalid = [
        edge
        for edge, directions in edge_directions.items()
        if len(directions) != 2 or directions[0] != tuple(reversed(directions[1]))
    ]
    if invalid:
        raise AnimatedTargetError(
            f"resource {resource_id!r} is not a consistently wound closed surface; "
            f"invalid edges={invalid[:5]}"
        )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise AnimatedTargetError("cannot normalize non-finite vertex normals")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            lengths = np.linalg.norm(values, axis=1, keepdims=True)
    except FloatingPointError as exc:
        raise AnimatedTargetError("cannot normalize non-finite vertex normals") from exc
    if not np.all(np.isfinite(lengths)):
        raise AnimatedTargetError("cannot normalize non-finite vertex normals")
    if np.any(lengths <= _VECTOR_EPSILON):
        raise AnimatedTargetError("cannot normalize zero-length vertex normals")
    result = values / lengths
    if not np.all(np.isfinite(result)):
        raise AnimatedTargetError("cannot normalize non-finite vertex normals")
    return result


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(value)):
        raise AnimatedTargetError("cannot normalize a non-finite quaternion")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            norm = float(np.linalg.norm(value))
    except FloatingPointError as exc:
        raise AnimatedTargetError("cannot normalize a non-finite quaternion") from exc
    if not math.isfinite(norm):
        raise AnimatedTargetError("cannot normalize a non-finite quaternion")
    if norm <= _VECTOR_EPSILON:
        raise AnimatedTargetError("cannot normalize a zero-length quaternion")
    return value / norm


def _same_rotation(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        np.allclose(left, right, atol=1e-12, rtol=0.0)
        or np.allclose(left, -right, atol=1e-12, rtol=0.0)
    )


def _array_sha256(value: np.ndarray, *, dtype: str) -> str:
    array = np.ascontiguousarray(value, dtype=np.dtype(dtype))
    header = canonical_json_bytes({"dtype": dtype, "shape": list(array.shape)})
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _readonly(value: Any, *, dtype: Any) -> np.ndarray:
    owned = np.array(value, dtype=dtype, order="C", copy=True)
    # A NumPy-owned array can make itself writable again with
    # ``setflags(write=True)``.  Reconstructing from immutable ``bytes`` makes
    # the authenticated value structurally read-only for every caller.
    return np.frombuffer(owned.tobytes(order="C"), dtype=owned.dtype).reshape(owned.shape)


def _resolve_resource(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    current = root
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise AnimatedTargetError(
                f"resource path must not traverse a symlink: {relative_path!r}"
            )
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise AnimatedTargetError(
            f"resource path escapes the manifest directory: {relative_path!r}"
        ) from exc
    if not resolved.is_file():
        raise AnimatedTargetError(f"resource path is not a regular file: {relative_path!r}")
    return resolved


def _safe_resource_path(value: Any, path: str) -> str:
    text = _string(value, path)
    if "\\" in text:
        raise AnimatedTargetError(f"{path} must use forward slashes")
    pure = PurePosixPath(text)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise AnimatedTargetError(f"{path} must be a safe relative path")
    return pure.as_posix()


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise AnimatedTargetError(f"manifest is not JSON-compatible: {exc}") from exc


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting ambiguous duplicate keys."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnimatedTargetError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AnimatedTargetError(f"{path} must be a JSON object")
    return dict(value)


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise AnimatedTargetError(f"{path} must be a JSON list")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    keys = set(value)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        raise AnimatedTargetError(f"{path} fields mismatch; missing={missing}, extra={extra}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnimatedTargetError(f"{path} must be a non-empty string")
    return value


def _stable_id(value: Any, path: str) -> str:
    text = _string(value, path)
    if text != text.strip() or any(character.isspace() for character in text):
        raise AnimatedTargetError(f"{path} must be a whitespace-free stable ID")
    if (
        "/" in text
        or "\\" in text
        or text.casefold().startswith("file:")
        or re.match(r"^[A-Za-z]:", text) is not None
        or text.startswith("~")
        or text in {".", ".."}
    ):
        raise AnimatedTargetError(f"{path} must be a semantic ID, not a path: {text!r}")
    final_segment = re.split(r"[:/]", text)[-1]
    if final_segment.isdigit() or re.fullmatch(
        r"(?:frame|sample|pose|root|pulse|index)[_-]?\d+",
        final_segment,
        flags=re.IGNORECASE,
    ):
        raise AnimatedTargetError(f"{path} must be semantic, not an ordinal-derived ID: {text!r}")
    if final_segment.lower() in {"frame", "sample", "pose", "root", "pulse", "index"}:
        raise AnimatedTargetError(f"{path} must include a semantic suffix: {text!r}")
    if final_segment.lower().endswith((".json", ".ply", ".obj", ".npz", ".npy")):
        raise AnimatedTargetError(f"{path} must not derive identity from a filename: {text!r}")
    return text


def _sha256(value: Any, path: str) -> str:
    text = _string(value, path)
    if not _SHA256_PATTERN.fullmatch(text):
        raise AnimatedTargetError(f"{path} must be a lowercase 64-character SHA-256 digest")
    return text


def _integer(value: Any, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnimatedTargetError(f"{path} must be an integer")
    if value < minimum:
        raise AnimatedTargetError(f"{path} must be at least {minimum}")
    return value


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnimatedTargetError(f"{path} must be a number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise AnimatedTargetError(f"{path} must be finite") from exc
    if not math.isfinite(result):
        raise AnimatedTargetError(f"{path} must be finite")
    return result


def _time(value: Any, path: str) -> float:
    result = _finite_number(value, path)
    if result < 0.0:
        raise AnimatedTargetError(f"{path} must be non-negative")
    return result


def _axis(value: Any, path: str) -> str:
    text = _string(value, path)
    if not _AXIS_PATTERN.fullmatch(text):
        raise AnimatedTargetError(f"{path} must be one of +X, -X, +Y, -Y, +Z, -Z")
    return text


def _axis_vector(axis: str) -> np.ndarray:
    result = np.zeros(3, dtype=np.float64)
    result["XYZ".index(axis[1])] = 1.0 if axis[0] == "+" else -1.0
    return result


def _matrix3(value: Any, path: str) -> np.ndarray:
    matrix = _numeric_matrix(value, path, 3)
    if matrix.shape != (3, 3):
        raise AnimatedTargetError(f"{path} must have shape (3, 3), got {matrix.shape}")
    return matrix


def _matrix4(value: Any, path: str) -> np.ndarray:
    matrix = _numeric_matrix(value, path, 4)
    if matrix.shape != (4, 4):
        raise AnimatedTargetError(f"{path} must have shape (4, 4), got {matrix.shape}")
    return matrix


def _vector3(value: Any, path: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 3:
        raise AnimatedTargetError(f"{path} must be a three-number JSON list")
    return np.array([_finite_number(item, f"{path}[{index}]") for index, item in enumerate(value)])


def _quaternion_xyzw(value: Any, path: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 4:
        raise AnimatedTargetError(f"{path} must be a four-number JSON list")
    result = np.array(
        [_finite_number(item, f"{path}[{index}]") for index, item in enumerate(value)],
        dtype=np.float64,
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            norm = float(np.linalg.norm(result))
    except FloatingPointError as exc:
        raise AnimatedTargetError(f"{path} must be a finite unit quaternion") from exc
    if not math.isfinite(norm):
        raise AnimatedTargetError(f"{path} must be a finite unit quaternion")
    if not math.isclose(norm, 1.0, abs_tol=_UNIT_QUATERNION_TOLERANCE, rel_tol=0.0):
        raise AnimatedTargetError(f"{path} must be a unit quaternion, got norm {norm}")
    result = result / norm
    # q and -q encode the same rotation, but the manifest revision authenticates
    # the authored JSON bytes before this parser runs.  Require adapters to emit
    # the canonical antipode so equivalent rotations cannot acquire distinct
    # authenticated asset and binding identities.
    for component_index in (3, 0, 1, 2):
        component = float(result[component_index])
        if abs(component) <= _VECTOR_EPSILON:
            continue
        if component < 0.0:
            raise AnimatedTargetError(
                f"{path} must use the canonical quaternion antipode: the first "
                "significant component in w/x/y/z order must be positive"
            )
        break
    return result


def _numeric_matrix(value: Any, path: str, columns: int) -> np.ndarray:
    rows = _list(value, path)
    parsed: list[list[float]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != columns:
            raise AnimatedTargetError(f"{path}[{row_index}] must be a {columns}-number JSON list")
        parsed.append(
            [
                _finite_number(item, f"{path}[{row_index}][{column_index}]")
                for column_index, item in enumerate(row)
            ]
        )
    return np.asarray(parsed, dtype=np.float64)


def _integer_matrix(value: Any, path: str, columns: int) -> np.ndarray:
    rows = _list(value, path)
    parsed: list[list[int]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != columns:
            raise AnimatedTargetError(f"{path}[{row_index}] must be a {columns}-integer JSON list")
        parsed.append(
            [
                _integer(item, f"{path}[{row_index}][{column_index}]", minimum=0)
                for column_index, item in enumerate(row)
            ]
        )
    try:
        return np.asarray(parsed, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnimatedTargetError(f"{path} values must fit signed 64-bit integer indices") from exc


def _strictly_increasing(values: Sequence[float], path: str) -> None:
    if any(right <= left for left, right in zip(values, values[1:])):
        raise AnimatedTargetError(f"{path} must be strictly increasing in manifest order")


__all__ = [
    "ANIMATED_TARGET_BINDING_VERSION",
    "ANIMATED_TARGET_MANIFEST_VERSION",
    "AnimatedTargetAsset",
    "AnimatedTargetBinding",
    "AnimatedTargetBindingContext",
    "AnimatedTargetBindingTime",
    "AnimatedTargetError",
    "BoundAnimatedTargetSample",
    "animated_target_revision_sha256",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_animated_target",
]
