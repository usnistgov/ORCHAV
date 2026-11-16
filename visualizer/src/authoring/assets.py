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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Generic, Mapping, Protocol, TypeVar
from uuid import UUID

import numpy as np

from generator.core.target.representation import NativeTargetConfig
from shared.scenarios.paths import resolve_actor_resource
from visualizer.src.materials.texture_policy import TEXTURE_MAP_KEYS, textures_globally_enabled
from visualizer.src.model import RenderObjectState
from visualizer.src.scene.assembly import build_texture_cache, mesh_entry_to_payload
from visualizer.src.scene.geometry_payload_factory import load_mesh_payload
from visualizer.src.scene.io import XMLSceneHandler, build_scene_from_root, finalize_uv_cache_stores
from visualizer.src.scene.native_target_geometry import (
    native_proxy_payload,
    native_scattering_points,
)
from visualizer.src.scene.target_transforms import (
    TargetGeometryMeta,
    build_sionna_rotation_matrix,
    rotated_aabb_center,
    target_transform_matrix,
)
from visualizer.src.services.material_context import MaterialContext
from visualizer.src.services.material_pbr_service import MaterialPBRService
from visualizer.src.services.target_asset_cache import TargetAsset, TargetAssetCache
from visualizer.src.services.visual_profile_service import VisualProfileService
from visualizer.src.types.render_payloads import MaterialPayload, MeshPayload

from .domain import AuthoringActor, SceneReference, Vector3, vector3
from .frame_geometry import FrameGeometry


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
    attribution: str | None = None
    material_type: str = ""
    ground: bool = False


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
    scattering_points_local_m: tuple[Vector3, ...] = ()
    scattering_point_ids: tuple[int, ...] = ()
    representation: str = "mesh_material"
    metadata_source: str = "mesh_asset"

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
class _SceneCacheRecord:
    xml_revision: FileRevision
    dependency_revisions: tuple[FileRevision, ...]
    asset_signature: str
    visual_sidecar_revision: FileRevision | None = None
    visual_texture_revisions: tuple[tuple[str, FileRevision | None], ...] = ()


class PreviewAssetCompiler:
    """Resolve and cache neutral scene and target preview payloads."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self._scene_assets = _BoundedCache[str, tuple[ScenePreviewAsset, ...]](4)
        self._scene_records = _BoundedCache[tuple, _SceneCacheRecord](4)
        self.target_cache = TargetAssetCache()
        self._surfaces = _BoundedCache[FileRevision, Any](32)
        self._scenario_root: Path = self.project_root
        self.configure_appearance({}, self.project_root)

    def configure_appearance(self, defaults: Mapping[str, Any], scenario_root: Path) -> None:
        """Capture scenario appearance using the normal visualizer services."""
        key = _digest((defaults, textures_globally_enabled()))
        if getattr(self, "_appearance_key", None) == key and self._scenario_root == scenario_root:
            return
        self._appearance = MaterialContext.from_defaults(defaults)
        self._appearance_key = key
        self._scenario_root = scenario_root
        self._materials = MaterialPBRService(self._appearance)
        self._profiles = VisualProfileService(self._appearance)
        self._profiles.load_scenario_rules(defaults.get("visual_profiles", []))

    def build_scene_assets(
        self,
        scene: SceneReference,
        scene_path: str | Path,
        *,
        authored_surface_textures: bool = True,
        geometry_ready: Callable[[tuple[ScenePreviewAsset, ...]], None] | None = None,
    ) -> tuple[ScenePreviewAsset, ...]:
        """Build scene payloads using the shared XML and assembly boundaries."""
        path = Path(scene_path).resolve(strict=True)
        revision = FileRevision.capture(path)
        metadata_revisions = tuple(
            _optional_revision(path.with_name(name))
            for name in ("building_visuals.json", "osm_metadata.json")
        )
        record_key = (
            scene.source,
            scene.id,
            revision.canonical_path,
            authored_surface_textures,
            self._appearance_key,
            metadata_revisions,
        )
        record = self._scene_records.get(record_key)
        visual_sidecar = path.with_name("visual_surfaces.json")
        visual_revision = _optional_revision(visual_sidecar)
        if (
            record is not None
            and record.xml_revision == revision
            and record.visual_sidecar_revision == visual_revision
            and self._revisions_are_current(record.dependency_revisions)
            and all(
                _optional_revision(Path(texture_path)) == texture_revision
                for texture_path, texture_revision in record.visual_texture_revisions
            )
        ):
            cached = self._scene_assets.get(record.asset_signature)
            if cached is not None:
                return cached

        xml_root = XMLSceneHandler.load_xml_scene(str(path))
        entries = build_scene_from_root(xml_root, str(path))
        texture_cache = build_texture_cache(self.project_root)
        dependencies = self._scene_dependencies(entries, texture_cache)
        signature = _digest(
            (
                self._scene_signature(revision, entries, dependencies),
                authored_surface_textures,
                self._appearance_key,
                metadata_revisions,
            )
        )
        visual_textures = (
            self._visual_texture_revisions(visual_sidecar) if visual_revision is not None else ()
        )
        if visual_revision is not None:
            signature = _digest((signature, visual_revision, visual_textures))
        cache_record = _SceneCacheRecord(
            revision, dependencies, signature, visual_revision, visual_textures
        )
        cached = self._scene_assets.get(signature)
        if cached is not None:
            self._scene_records.put(
                record_key,
                cache_record,
            )
            return cached

        def assemble(materials, asset_signature) -> tuple[ScenePreviewAsset, ...]:
            assets: list[ScenePreviewAsset] = []
            for index, source_entry in enumerate(entries):
                entry = dict(source_entry)
                if not authored_surface_textures and "_visual_surface_base_properties" in entry:
                    entry = dict(entry)
                    entry["pbr_properties"] = dict(entry["_visual_surface_base_properties"])
                    entry["color"] = entry.get("_visual_surface_base_color")
                    entry.pop("texture_path", None)
                    entry.pop("_resolved_texture_maps", None)
                binding = self._profiles.resolve_binding(
                    str(entry.get("name", "")), str(entry.get("material_type", "default")), "mesh"
                )
                if binding is not None:
                    materials.set_visual_binding(entry, binding)
                converted = mesh_entry_to_payload(
                    entry, texture_cache, index=index, material_service=materials
                )
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
                        cache_key=f"authoring-scene:{asset_signature}:{index}",
                        name=name,
                        scene_source=scene.source,
                        scene_id=scene.id,
                        scene_path=str(path),
                        mesh_path=mesh_path,
                        mesh=mesh,
                        material=material,
                        material_type=str(entry.get("material_type", "")),
                        ground=bool(len(mesh.vertices) and np.ptp(mesh.vertices[:, 2]) < 1.0),
                    )
                )
            return tuple(assets)

        if geometry_ready is not None and (
            self._appearance.building_appearance != "original"
            or self._appearance.road_appearance != "original"
        ):
            initial_context = replace(
                self._appearance, building_appearance="original", road_appearance="original"
            )
            geometry_ready(assemble(MaterialPBRService(initial_context), signature + ":base"))
        assets = assemble(self._materials, signature)
        finalize_uv_cache_stores()
        compiled = self._scene_assets.put(signature, tuple(assets))
        self._scene_records.put(
            record_key,
            cache_record,
        )
        return compiled

    def build_target_assets(
        self,
        actor: AuthoringActor,
        mesh_paths: tuple[Path, ...],
        samples: PreparedSamples,
    ) -> Sequence[TargetPreviewAsset]:
        """Prepare the displayed mesh now and resolve later frames on demand."""
        if actor.representation != "mesh_material":
            return self._build_native_target_assets(actor, mesh_paths, samples)
        target = actor.target
        if target is None:
            raise ValueError(f"Target {actor.name!r} has no catalog asset")
        if not math.isfinite(target.scale) or target.scale <= 0.0:
            raise ValueError(f"Target {actor.name!r} scale must be finite and positive")
        if not mesh_paths:
            raise FileNotFoundError(f"Target {actor.name!r} has no matching mesh files")
        if len(samples.positions) != len(samples.orientations):
            raise ValueError(f"Target {actor.name!r} prepared sample lengths do not match")

        materials = self._materials
        binding = self._profiles.resolve_binding(actor.name, target.material, "target")
        surface = self._target_surface(actor)
        namespace = self._target_namespace(mesh_paths[0], target.scale, surface)
        self.target_cache.register_sequence(
            namespace, mesh_paths if target.mesh_animation else mesh_paths[:1]
        )

        def prepare(step_index: int) -> TargetPreviewAsset:
            position = samples.positions[step_index]
            orientation = samples.orientations[step_index]
            mesh_index = step_index % len(mesh_paths) if target.mesh_animation else 0
            if target.mesh_animation and target.mesh_end_behavior == "hold_last":
                mesh_index = min(step_index, len(mesh_paths) - 1)
            mesh_path = Path(mesh_paths[mesh_index]).resolve(strict=True)
            revision = FileRevision.capture(mesh_path)
            mesh = self._target_mesh(revision, target.scale, surface=surface)
            entry = {
                "mesh": mesh,
                "material_type": target.material,
                "type": "target",
                "_target_surface": surface,
            }
            if binding is not None:
                materials.set_visual_binding(entry, binding)
            material = materials.resolve_entry_material(entry).payload
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
            return TargetPreviewAsset(
                cache_key=(
                    f"authoring-target:{revision.token}:"
                    f"{_digest((target.material, target.scale, binding, getattr(surface, 'fingerprint', None)))}"
                ),
                actor_id=actor.id,
                step_index=step_index,
                mesh_path=str(mesh_path),
                scale=target.scale,
                mesh=mesh,
                material=material,
                local_to_actor=local_to_actor,
            )

        frames = FrameGeometry(len(samples.positions), prepare)
        if frames:
            frames[0]  # Surface malformed first assets while other actors still load.
        return frames

    def build_animated_target_assets(
        self, actor: AuthoringActor, animation: Any
    ) -> Sequence[TargetPreviewAsset]:
        """Display the canonical time-bound geometry without writing RT meshes."""
        assert actor.target is not None
        target = actor.target
        materials = self._materials
        binding = self._profiles.resolve_binding(actor.name, actor.target.material, "target")
        surface = self._target_surface(actor)
        faces = np.asarray(animation.binding.faces, dtype=np.int32)

        def prepare(index: int) -> TargetPreviewAsset:
            sample = animation.binding.samples[index]
            key = f"authoring-animation:{animation.binding.binding_id}:{index}"
            # Root motion and source transforms are already baked by the asset
            # binder. The actor pose supplies its AABB center exactly once.
            vertices = np.asarray(sample.vertices_m - animation.centers_m[index], dtype=np.float32)
            mesh = MeshPayload(vertices=vertices, triangles=faces, cache_key=key)
            if surface is not None:
                mesh = surface.apply(mesh)
            entry = {
                "mesh": mesh,
                "material_type": target.material,
                "type": "target",
                "_target_surface": surface,
            }
            if binding is not None:
                materials.set_visual_binding(entry, binding)
            material = materials.resolve_entry_material(entry).payload
            return TargetPreviewAsset(
                cache_key=key,
                actor_id=actor.id,
                step_index=index,
                mesh_path=str(animation.asset.manifest_path),
                scale=1.0,
                mesh=mesh,
                material=material,
                local_to_actor=np.eye(4),
                metadata_source="animated_asset_manifest",
            )

        return FrameGeometry(len(animation.binding.samples), prepare)

    def _target_surface(self, actor: AuthoringActor) -> Any:
        """Resolve authored skins through the same profile and path services."""
        assert actor.target is not None
        path = self._profiles.resolve_surface_path(actor.name, actor.target.material, "target")
        if not path:
            return None
        from visualizer.src.scene.target_surface import load_target_surface

        resolved = resolve_actor_resource(
            path, scenario_root=self._scenario_root, project_root=self.project_root
        )
        revision = FileRevision.capture(resolved)
        surface = self._surfaces.get(revision)
        if surface is None:
            surface = self._surfaces.put(revision, load_target_surface(resolved))
        return surface

    def _build_native_target_assets(
        self, actor: AuthoringActor, mesh_paths: tuple[Path, ...], samples: PreparedSamples
    ) -> Sequence[TargetPreviewAsset]:
        """Use generator-owned preset metadata and the shared prepared actor pose."""
        if actor.geometry is None:
            raise ValueError("Native target requires occlusion geometry")
        mesh_path = mesh_paths[0] if mesh_paths else None
        config = NativeTargetConfig(
            name=actor.name,
            representation=actor.representation,
            geometry=actor.geometry,
            constant_rcs=actor.constant_rcs,
            tr38901=actor.tr38901,
            resolved_mesh_path=mesh_path,
        )
        metadata = config.preview_metadata()
        mesh = native_proxy_payload(metadata, mesh_path)
        point_ids, points = native_scattering_points(metadata, world=False)
        local_points = tuple(vector3(row) for row in points)
        material = MaterialPayload(base_color=(0.2, 0.4, 1.0, 0.25))
        revision = FileRevision.capture(mesh_path).token if mesh_path is not None else "box"
        cache_key = f"authoring-native:{revision}:{_digest(actor.target_representation_mapping())}"

        def prepare(step: int) -> TargetPreviewAsset:
            position = samples.positions[step]
            orientation = samples.orientations[step]
            pose = prepared_actor_pose(position, orientation)
            rotation = pose[:3, :3]
            center = rotated_aabb_center(mesh.vertices, rotation)
            world_transform = target_transform_matrix(
                position=position, rotation_matrix=rotation, rotated_center=center
            )
            if world_transform is None:
                raise ValueError("Native target preview pose could not be prepared")
            return TargetPreviewAsset(
                cache_key=cache_key,
                actor_id=actor.id,
                step_index=step,
                mesh_path=str(mesh_path or ""),
                scale=1.0,
                mesh=mesh,
                material=material,
                local_to_actor=np.linalg.solve(pose, world_transform),
                scattering_points_local_m=local_points,
                scattering_point_ids=point_ids,
                representation=actor.representation,
                metadata_source=str(metadata["metadata_source"]),
            )

        return FrameGeometry(len(samples.positions), prepare)

    @staticmethod
    def _target_namespace(path: str | Path, scale: float, surface: Any) -> str:
        return (
            f"{Path(path).parent}:scale={scale:.17g}:surface={getattr(surface, 'fingerprint', '')}"
        )

    def _target_mesh(
        self, revision: FileRevision, scale: float, *, surface: Any = None
    ) -> MeshPayload:
        namespace = self._target_namespace(revision.canonical_path, scale, surface)
        source = self.target_cache.register_source(namespace, revision.canonical_path)

        def load(resolved):
            mesh = load_mesh_payload(resolved.canonical_path)
            if surface is not None:
                mesh = surface.apply(mesh)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
                raise ValueError(f"Target mesh has no triangle vertices: {resolved.canonical_path}")
            if not np.all(np.isfinite(vertices)):
                raise ValueError(
                    f"Target mesh contains non-finite vertices: {resolved.canonical_path}"
                )
            center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
            scaled = center + float(scale) * (vertices - center)
            prepared = replace(
                mesh,
                vertices=scaled,
                cache_key=f"authoring-target-mesh:{_digest(resolved.key)}:scale={scale:.17g}",
            )
            return TargetAsset(
                source=resolved,
                mesh=RenderObjectState(id=namespace, payload=prepared),
                original_vertices=vertices,
                scaled_vertices=scaled,
                geometry_meta=TargetGeometryMeta(scaled_aabb_center=center),
            )

        asset = self.target_cache.get_or_load(namespace, source.mesh_filename, load)
        self.target_cache.prefetch_after(namespace, source.mesh_filename, count=2, loader=load)
        return asset.mesh.payload

    @staticmethod
    def _scene_signature(
        xml_revision: FileRevision,
        entries: list[dict[str, Any]],
        dependencies: tuple[FileRevision, ...],
    ) -> str:
        entry_signatures = []
        for entry in entries:
            entry_signature = {
                "name": entry.get("name"),
                "shape_index": entry.get("shape_index"),
                "source": entry.get("_source_signature"),
                "material_id": entry.get("material_id"),
                "material_type": entry.get("material_type"),
                "color": entry.get("color"),
                "pbr": entry.get("pbr_properties"),
                "visible": entry.get("visible", True),
            }
            if entry.get("_visual_surface_key"):
                entry_signature["visual_surface"] = entry["_visual_surface_key"]
            entry_signatures.append(entry_signature)
        return _digest((xml_revision, entry_signatures, dependencies))

    @staticmethod
    def _visual_texture_revisions(
        sidecar: Path,
    ) -> tuple[tuple[str, FileRevision | None], ...]:
        """Track authored maps even when a missing file rejects the mapping.

        Remember absent files so their later creation invalidates a cached
        fallback preview. This runs only after the early scene cache misses;
        normal repeated compilation checks stored file revisions without
        reparsing the sidecar. Schema diagnostics remain owned by the loader.
        """
        try:
            document = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ()
        records = document.get("meshes") if isinstance(document, dict) else None
        if not isinstance(records, dict):
            return ()
        revisions: dict[str, FileRevision | None] = {}
        for record in records.values():
            surfaces = record.get("surfaces") if isinstance(record, dict) else None
            if not isinstance(surfaces, list):
                continue
            for surface in surfaces:
                material = surface.get("material") if isinstance(surface, dict) else None
                if not isinstance(material, dict):
                    continue
                for key in TEXTURE_MAP_KEYS:
                    value = material.get(key)
                    if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
                        continue
                    # Retain the lookup path so retargeting a texture symlink
                    # changes its captured revision on the next compilation.
                    path = sidecar.parent / value
                    lookup_path = os.path.normcase(str(path))
                    revisions[lookup_path] = _optional_revision(path)
        return tuple((key, revisions[key]) for key in sorted(revisions))

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
