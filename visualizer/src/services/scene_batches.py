"""Typed ownership model for static scene presentation batches.

Scene entries remain the semantic source of truth.  This module owns only the
renderer presentation plan: which compatible entries share an aggregate,
which entries are temporary individual exceptions, and which entries have no
native owner because they are hidden.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from ..materials.catalog import ResolvedMaterial
from ..model import RenderObjectState
from ..types.render_payloads import MaterialPayload, MeshPayload


@dataclass(frozen=True, slots=True)
class PreparedSceneMember:
    """One scene entry after its material, UVs, and visibility are resolved.

    Both the individual and aggregate publishers consume this object.  Keeping
    preparation outside those publishers prevents their renderer-owner policy
    from accidentally growing separate material and texture semantics.
    """

    state: RenderObjectState
    entry: dict[str, Any] | None
    visible: bool
    material_type: str
    resolved_material: ResolvedMaterial | None

    @property
    def material(self) -> MaterialPayload:
        """Return the renderer-neutral material carried by this member."""
        if self.resolved_material is not None:
            return self.resolved_material.payload
        return self.state.material

    @property
    def material_signature(self) -> tuple[str, MaterialPayload]:
        """Return the complete compatibility key for static batching."""
        return self.material_type, self.material


@dataclass(frozen=True, slots=True)
class SceneBatchPartition:
    """One resolved aggregate/exception partition for a scene batch."""

    aggregate_member_ids: tuple[int, ...] = ()
    individual_member_ids: tuple[int, ...] = ()
    hidden_member_ids: tuple[int, ...] = ()
    aggregate_material: MaterialPayload | None = None

    @property
    def visible_member_ids(self) -> tuple[int, ...]:
        """Return every member that currently needs a native presentation."""
        return self.aggregate_member_ids + self.individual_member_ids


@dataclass(slots=True)
class SceneBatch(MutableMapping[str, Any]):
    """Typed scene batch with a small compatibility mapping for presentation state.

    ``presentation`` contains backend-neutral render handles such as the
    aggregate ``RenderObjectState`` and outline state.  Canonical membership,
    baseline geometry, source identity, and partitioning are explicit fields so
    they cannot silently diverge across parallel dictionaries.
    """

    name: str
    material_signature: tuple[str, MaterialPayload]
    member_mesh_ids: list[int] = field(default_factory=list)
    baseline_geometry: MeshPayload | None = None
    baseline_sources: tuple[tuple[str, MeshPayload], ...] = ()
    current_partition: SceneBatchPartition = field(default_factory=SceneBatchPartition)
    presentation: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.presentation[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.presentation[key] = value

    def __delitem__(self, key: str) -> None:
        del self.presentation[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.presentation)

    def __len__(self) -> int:
        return len(self.presentation)

    def resolve_partition(self, resolved_by_mesh_id: dict[int, Any]) -> SceneBatchPartition:
        """Resolve one partition from already-resolved member appearances.

        A whole-batch highlight stays aggregated when all visible members share
        the same highlighted material.  Mixed highlighted members become
        individual exceptions while normal members retain the aggregate.
        Hidden members never receive a native owner.
        """
        normal: list[int] = []
        highlighted: list[int] = []
        hidden: list[int] = []
        highlight_materials: list[MaterialPayload] = []

        for mesh_id in self.member_mesh_ids:
            resolved = resolved_by_mesh_id.get(mesh_id)
            if resolved is None or not bool(resolved.visible):
                hidden.append(mesh_id)
                continue
            if bool(resolved.highlighted):
                highlighted.append(mesh_id)
                highlight_materials.append(resolved.material)
            else:
                normal.append(mesh_id)

        aggregate_material: MaterialPayload | None = None
        individual = highlighted
        aggregate = normal
        if not normal and highlighted:
            candidate = highlight_materials[0]
            if all(material == candidate for material in highlight_materials[1:]):
                aggregate = highlighted
                individual = []
                aggregate_material = candidate

        partition = SceneBatchPartition(
            aggregate_member_ids=tuple(aggregate),
            individual_member_ids=tuple(individual),
            hidden_member_ids=tuple(hidden),
            aggregate_material=aggregate_material,
        )
        self.current_partition = partition
        return partition


@dataclass(slots=True)
class SceneBatchRegistry:
    """Sole owner of scene batch membership and renderer-owner bookkeeping."""

    batches: dict[str, SceneBatch] = field(default_factory=dict)
    mesh_to_batch: dict[int, str] = field(default_factory=dict)
    entries_by_mesh_id: dict[int, dict[str, Any]] = field(default_factory=dict)
    individual_owner_ids: set[str] = field(default_factory=set)
    pending_batch_ids: set[str] = field(default_factory=set)
    pending_entry_sources: dict[int, str] = field(default_factory=dict)

    def clear(self) -> None:
        """Clear all ownership and retry state for a new scene plan."""
        self.clear_presentations()
        self.entries_by_mesh_id.clear()

    def clear_presentations(self) -> None:
        """Clear native-owner planning while retaining canonical entry indexes."""
        self.batches.clear()
        self.mesh_to_batch.clear()
        self.individual_owner_ids.clear()
        self.pending_batch_ids.clear()
        self.pending_entry_sources.clear()

    def register_entry(self, mesh_id: int, entry: dict[str, Any]) -> None:
        """Register the canonical entry for one persistent mesh identity."""
        self.entries_by_mesh_id[mesh_id] = entry

    def add_batch(self, batch: SceneBatch) -> None:
        """Register a new batch and its canonical reverse membership."""
        if batch.name in self.batches:
            raise ValueError(f"Scene batch already exists: {batch.name}")
        member_ids = set(batch.member_mesh_ids)
        if len(member_ids) != len(batch.member_mesh_ids):
            raise ValueError(f"Scene batch contains duplicate members: {batch.name}")
        for mesh_id in member_ids:
            previous = self.mesh_to_batch.get(mesh_id)
            if previous is not None:
                raise ValueError(f"Mesh {mesh_id} already belongs to scene batch {previous}")

        self.batches[batch.name] = batch
        for mesh_id in batch.member_mesh_ids:
            self.mesh_to_batch[mesh_id] = batch.name

    def remove_batch(self, batch_name: str) -> SceneBatch | None:
        """Remove a batch and every reverse membership owned by it."""
        batch = self.batches.pop(batch_name, None)
        if batch is None:
            return None
        for mesh_id in batch.member_mesh_ids:
            if self.mesh_to_batch.get(mesh_id) == batch_name:
                self.mesh_to_batch.pop(mesh_id, None)
        self.pending_batch_ids.discard(batch_name)
        return batch

    def attach(self, batch_name: str, mesh_id: int, entry: dict[str, Any]) -> None:
        """Attach one canonical entry to an existing compatible batch."""
        previous = self.mesh_to_batch.get(mesh_id)
        if previous is not None and previous != batch_name:
            raise ValueError(f"Mesh {mesh_id} already belongs to scene batch {previous}")
        batch = self.batches[batch_name]
        if mesh_id not in batch.member_mesh_ids:
            batch.member_mesh_ids.append(mesh_id)
        self.mesh_to_batch[mesh_id] = batch_name
        self.entries_by_mesh_id[mesh_id] = entry

    def detach(self, batch_name: str, mesh_id: int) -> bool:
        """Detach one member and return whether the batch existed."""
        batch = self.batches.get(batch_name)
        if batch is None:
            self.mesh_to_batch.pop(mesh_id, None)
            return False
        batch.member_mesh_ids[:] = [member for member in batch.member_mesh_ids if member != mesh_id]
        if self.mesh_to_batch.get(mesh_id) == batch_name:
            self.mesh_to_batch.pop(mesh_id, None)
        self.pending_batch_ids.add(batch_name)
        return True

    def batch_for_mesh(self, mesh_id: int) -> SceneBatch | None:
        """Return the batch currently owning *mesh_id*, if any."""
        batch_name = self.mesh_to_batch.get(mesh_id)
        return self.batches.get(batch_name) if batch_name is not None else None
