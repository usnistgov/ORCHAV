"""Renderer-neutral preview assets compiled from authoring documents.

This module produces immutable mesh, material, and transform payloads. Qt and
renderer code consume these values but do not resolve files,
choose target animation frames, or reproduce generator placement math.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Generic, Mapping, Protocol, TypeVar
from uuid import UUID

import numpy as np

from visualizer.src.materials.catalog import (
    material_preset,
    normalize_material_type_name,
    resolve_pbr_material,
)
from visualizer.src.scene.assembly import build_texture_cache, mesh_entry_to_payload
from visualizer.src.scene.geometry_payload_factory import load_mesh_payload
from visualizer.src.scene.io import XMLSceneHandler, build_scene_from_root
from visualizer.src.scene.target_transforms import (
    build_sionna_rotation_matrix,
    rotated_aabb_center,
    target_transform_matrix,
)
from visualizer.src.types.render_payloads import MaterialPayload, MeshPayload

from .domain import AuthoringActor, SceneReference, Vector3


class PreparedSamples(Protocol):
    """Structural view of generator-prepared samples used by asset compilation."""

    @property
    def positions(self) -> tuple[Vector3, ...]:
        """Return generator-prepared positions."""
        ...

    @property
    def orientations(self) -> tuple[Vector3, ...]:
        """Return generator-prepared degree-valued orientations."""
        ...


@dataclass(frozen=True, slots=True)
class FileRevision:
    """Canonical file identity and the stat fields used for cache invalidation."""

    canonical_path: str
    size_bytes: int
    modified_ns: int
    changed_ns: int
    device_id: int
    file_id: int

    @classmethod
    def capture(cls, path: str | Path) -> "FileRevision":
        """Capture one concrete filesystem revision."""
        source = Path(path).resolve(strict=True)
        stat = source.stat()
        return cls(
            canonical_path=os.path.normcase(str(source)),
            size_bytes=int(stat.st_size),
            modified_ns=int(stat.st_mtime_ns),
            changed_ns=int(stat.st_ctime_ns),
            device_id=int(getattr(stat, "st_dev", 0)),
            file_id=int(getattr(stat, "st_ino", 0)),
        )

    @property
    def token(self) -> str:
        """Return a compact stable token suitable for renderer cache keys."""
        return _digest(
            (
                self.canonical_path,
                self.size_bytes,
                self.modified_ns,
                self.changed_ns,
                self.device_id,
                self.file_id,
            )
        )


@dataclass(frozen=True, slots=True)
class ScenePreviewAsset:
    """One material-bound scene mesh prepared without renderer dependencies."""

    cache_key: str
    name: str
    scene_source: str
    scene_id: str
    scene_path: str
    mesh_path: str | None
    mesh: MeshPayload
    material: MaterialPayload


@dataclass(frozen=True, slots=True)
class TargetPreviewAsset:
    """One target animation frame and its generator-matching local transform.

    Consumers compose ``prepared_actor_pose(position, orientation)`` with
    ``local_to_actor``. The resulting matrix follows the target placement
    contract: the AABB center after rotation lands at the generator-prepared
    position.
    """

    cache_key: str
    actor_id: UUID
    step_index: int
    mesh_path: str
    scale: float
    mesh: MeshPayload
    material: MaterialPayload
    local_to_actor: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", UUID(str(self.actor_id)))
        if self.step_index < 0:
            raise ValueError("Target preview step index must be nonnegative")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("Target preview scale must be finite and positive")
        matrix = np.asarray(self.local_to_actor, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("Target local-to-actor transform must be a finite 4x4 matrix")
        immutable: np.ndarray = np.frombuffer(matrix.tobytes(order="C"), dtype=np.float64).reshape(
            (4, 4)
        )
        object.__setattr__(self, "local_to_actor", immutable)


def prepared_actor_pose(
    position: tuple[float, float, float],
    orientation_degrees: tuple[float, float, float],
) -> np.ndarray:
    """Return the pose matrix for generator-prepared position and Sionna YPR."""
    position_array: np.ndarray = np.asarray(position, dtype=np.float64)
    orientation_array: np.ndarray = np.asarray(orientation_degrees, dtype=np.float64)
    if (
        position_array.shape != (3,)
        or orientation_array.shape != (3,)
        or not np.all(np.isfinite(position_array))
        or not np.all(np.isfinite(orientation_array))
    ):
        raise ValueError("Prepared target pose requires finite three-component values")
    rotation = build_sionna_rotation_matrix(*np.radians(orientation_array))
    pose: np.ndarray = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = position_array
    return pose


_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


class _BoundedCache(Generic[_Key, _Value]):
    """Small access-ordered cache owned by one compiler instance."""

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max(1, int(max_entries))
        self._values: OrderedDict[_Key, _Value] = OrderedDict()

    def get(self, key: _Key) -> _Value | None:
        value = self._values.get(key)
        if value is not None:
            self._values.move_to_end(key)
        return value

    def put(self, key: _Key, value: _Value) -> _Value:
        self._values.pop(key, None)
        self._values[key] = value
        while len(self._values) > self._max_entries:
            self._values.popitem(last=False)
        return value


@dataclass(frozen=True, slots=True)
class _TargetMeshKey:
    revision: FileRevision
    scale: float


@dataclass(frozen=True, slots=True)
class _TargetMaterialKey:
    material_type: str
    properties_signature: str
    texture_revisions: tuple[FileRevision, ...]


@dataclass(frozen=True, slots=True)
class _SceneCacheRecord:
    xml_revision: FileRevision
    dependency_revisions: tuple[FileRevision, ...]
    asset_signature: str


class PreviewAssetCompiler:
    """Resolve and cache neutral scene and target preview payloads."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self._scene_assets = _BoundedCache[str, tuple[ScenePreviewAsset, ...]](4)
        self._scene_records = _BoundedCache[tuple[str, str, str], _SceneCacheRecord](4)
        self._target_meshes = _BoundedCache[_TargetMeshKey, MeshPayload](64)
        self._target_materials = _BoundedCache[_TargetMaterialKey, MaterialPayload](32)

    def build_scene_assets(
        self,
        scene: SceneReference,
        scene_path: str | Path,
    ) -> tuple[ScenePreviewAsset, ...]:
        """Build scene payloads using the shared XML and assembly boundaries."""
        path = Path(scene_path).resolve(strict=True)
        revision = FileRevision.capture(path)
        record_key = (scene.source, scene.id, revision.canonical_path)
        record = self._scene_records.get(record_key)
        if (
            record is not None
            and record.xml_revision == revision
            and self._revisions_are_current(record.dependency_revisions)
        ):
            cached = self._scene_assets.get(record.asset_signature)
            if cached is not None:
                return cached

        xml_root = XMLSceneHandler.load_xml_scene(str(path))
        entries = build_scene_from_root(xml_root, str(path))
        texture_cache = build_texture_cache(self.project_root)
        dependencies = self._scene_dependencies(entries, texture_cache)
        signature = self._scene_signature(revision, entries, dependencies)
        cached = self._scene_assets.get(signature)
        if cached is not None:
            self._scene_records.put(
                record_key,
                _SceneCacheRecord(revision, dependencies, signature),
            )
            return cached

        assets: list[ScenePreviewAsset] = []
        for index, entry in enumerate(entries):
            converted = mesh_entry_to_payload(entry, texture_cache, index=index)
            if converted is None:
                continue
            name, mesh, material = converted
            source_signature = entry.get("_source_signature")
            mesh_path = None
            if isinstance(source_signature, Mapping):
                value = source_signature.get("path")
                mesh_path = str(value) if value is not None else None
            assets.append(
                ScenePreviewAsset(
                    cache_key=f"authoring-scene:{signature}:{index}",
                    name=name,
                    scene_source=scene.source,
                    scene_id=scene.id,
                    scene_path=str(path),
                    mesh_path=mesh_path,
                    mesh=mesh,
                    material=material,
                )
            )
        compiled = self._scene_assets.put(signature, tuple(assets))
        self._scene_records.put(
            record_key,
            _SceneCacheRecord(revision, dependencies, signature),
        )
        return compiled

    def build_target_assets(
        self,
        actor: AuthoringActor,
        mesh_paths: tuple[Path, ...],
        samples: PreparedSamples,
    ) -> tuple[TargetPreviewAsset, ...]:
        """Build one exact mesh/material/transform record per timeline step."""
        target = actor.target
        if target is None:
            raise ValueError(f"Target {actor.name!r} has no catalog asset")
        if not math.isfinite(target.scale) or target.scale <= 0.0:
            raise ValueError(f"Target {actor.name!r} scale must be finite and positive")
        if not mesh_paths:
            raise FileNotFoundError(f"Target {actor.name!r} has no matching mesh files")
        if len(samples.positions) != len(samples.orientations):
            raise ValueError(f"Target {actor.name!r} prepared sample lengths do not match")

        material = self._target_material(target.material, context=f"target {actor.name}")
        frames: list[TargetPreviewAsset] = []
        for step_index, (position, orientation) in enumerate(
            zip(samples.positions, samples.orientations)
        ):
            mesh_index = step_index % len(mesh_paths) if target.mesh_animation else 0
            mesh_path = Path(mesh_paths[mesh_index]).resolve(strict=True)
            revision = FileRevision.capture(mesh_path)
            mesh = self._target_mesh(revision, target.scale)
            pose = prepared_actor_pose(position, orientation)
            rotation = pose[:3, :3]
            rotated_center = rotated_aabb_center(mesh.vertices, rotation)
            if rotated_center is None or not np.all(np.isfinite(rotated_center)):
                raise ValueError(f"Target mesh has no finite AABB center: {mesh_path}")
            world_transform = target_transform_matrix(
                position=position,
                rotation_matrix=rotation,
                rotated_center=rotated_center,
            )
            if world_transform is None or not np.all(np.isfinite(world_transform)):
                raise ValueError(f"Target transform could not be prepared: {actor.name}")
            local_to_actor = np.linalg.solve(pose, np.asarray(world_transform, dtype=np.float64))
            frames.append(
                TargetPreviewAsset(
                    cache_key=(
                        f"authoring-target:{revision.token}:"
                        f"{_digest((target.material, target.scale))}"
                    ),
                    actor_id=actor.id,
                    step_index=step_index,
                    mesh_path=str(mesh_path),
                    scale=target.scale,
                    mesh=mesh,
                    material=material,
                    local_to_actor=local_to_actor,
                )
            )
        return tuple(frames)

    def _target_mesh(self, revision: FileRevision, scale: float) -> MeshPayload:
        key = _TargetMeshKey(revision, float(scale))
        cached = self._target_meshes.get(key)
        if cached is not None:
            return cached

        mesh = load_mesh_payload(revision.canonical_path)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
            raise ValueError(f"Target mesh has no triangle vertices: {revision.canonical_path}")
        if not np.all(np.isfinite(vertices)):
            raise ValueError(f"Target mesh contains non-finite vertices: {revision.canonical_path}")
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
        scaled = center + float(scale) * (vertices - center)
        prepared = replace(
            mesh,
            vertices=scaled,
            cache_key=f"authoring-target-mesh:{revision.token}:scale={float(scale):.17g}",
        )
        return self._target_meshes.put(key, prepared)

    def _target_material(self, material_type: str, *, context: str) -> MaterialPayload:
        normalized = normalize_material_type_name(material_type)
        properties = dict(material_preset(normalized))
        texture_revisions = tuple(
            revision
            for key, value in sorted(properties.items())
            if key.endswith("_path")
            and value
            and (revision := _optional_revision(Path(str(value)))) is not None
        )
        key = _TargetMaterialKey(
            material_type=normalized,
            properties_signature=_digest(properties),
            texture_revisions=texture_revisions,
        )
        cached = self._target_materials.get(key)
        if cached is not None:
            return cached
        color = properties.get("color", [0.8, 0.6, 0.5])
        material = resolve_pbr_material(color, properties, context=context).payload
        return self._target_materials.put(key, material)

    @staticmethod
    def _scene_signature(
        xml_revision: FileRevision,
        entries: list[dict[str, Any]],
        dependencies: tuple[FileRevision, ...],
    ) -> str:
        entry_signatures = []
        for entry in entries:
            entry_signatures.append(
                {
                    "name": entry.get("name"),
                    "shape_index": entry.get("shape_index"),
                    "source": entry.get("_source_signature"),
                    "material_id": entry.get("material_id"),
                    "material_type": entry.get("material_type"),
                    "color": entry.get("color"),
                    "pbr": entry.get("pbr_properties"),
                    "visible": entry.get("visible", True),
                }
            )
        return _digest((xml_revision, entry_signatures, dependencies))

    @staticmethod
    def _scene_dependencies(
        entries: list[dict[str, Any]],
        texture_cache: Mapping[str, str],
    ) -> tuple[FileRevision, ...]:
        revisions: dict[str, FileRevision] = {}
        for entry in entries:
            signature = entry.get("_source_signature")
            if not isinstance(signature, Mapping):
                continue
            value = signature.get("path")
            if value is None:
                continue
            revision = _optional_revision(Path(str(value)))
            if revision is not None:
                revisions[revision.canonical_path] = revision
        for value in texture_cache.values():
            revision = _optional_revision(Path(value))
            if revision is not None:
                revisions[revision.canonical_path] = revision
        return tuple(revisions[key] for key in sorted(revisions))

    @staticmethod
    def _revisions_are_current(revisions: tuple[FileRevision, ...]) -> bool:
        for revision in revisions:
            try:
                if FileRevision.capture(revision.canonical_path) != revision:
                    return False
            except OSError:
                return False
        return True


def _optional_revision(path: Path) -> FileRevision | None:
    try:
        return FileRevision.capture(path)
    except OSError:
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, FileRevision):
        return {
            "path": value.canonical_path,
            "size": value.size_bytes,
            "mtime_ns": value.modified_ns,
            "ctime_ns": value.changed_ns,
            "device": value.device_id,
            "file_id": value.file_id,
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
