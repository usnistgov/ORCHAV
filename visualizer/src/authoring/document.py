"""Mutable document authority for immutable authoring scenario values."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable
from uuid import UUID

from shared.scenarios.actors import MeshSequenceMobilitySpec, NetworkRouteMobilitySpec

from .domain import (
    ActorRole,
    AuthoringActor,
    AuthoringGroup,
    AuthoringResource,
    AuthoringScenario,
    AuthoringSubject,
    ScenarioDependency,
    SceneReference,
    SubjectKind,
    TimelineSettings,
)
from .undo import CommandStack, UndoStack


class DocumentOwnership(str, Enum):
    """How an authoring document relates to a YAML file."""

    NEW = "new"
    OWNED = "owned"
    COPIED = "copied"
    READ_ONLY = "read_only"


class DocumentEventKind(str, Enum):
    """Observable document changes consumed by workspace adapters."""

    CONTENT = "content"
    SELECTION = "selection"
    TRANSIENT = "transient"
    SAVED = "saved"


@dataclass(frozen=True, slots=True)
class DocumentEvent:
    """One immutable notification emitted after document state changes."""

    kind: DocumentEventKind
    revision: int
    dirty: bool
    selected_actor_id: UUID | None
    selected_group_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class _DocumentState:
    scenario: AuthoringScenario
    selection: AuthoringSubject | None


class _ReplaceStateCommand:
    """Undoable before/after document state replacement."""

    def __init__(
        self,
        document: "ScenarioDocument",
        before: _DocumentState,
        after: _DocumentState,
        text: str,
    ) -> None:
        self._document = document
        self._before = before
        self._after = after
        self._text = text

    @property
    def text(self) -> str:
        return self._text

    def redo(self) -> None:
        self._document._apply_state(self._after)

    def undo(self) -> None:
        self._document._apply_state(self._before)


class ScenarioDocument:
    """Sole mutable authority for one authoring scenario and its selection.

    Content values are immutable.  Every durable mutation is represented by
    one command on the supplied undo stack.  During direct manipulation, a
    transient edit may publish intermediate values and then commit as one
    command or cancel back to its exact starting state.
    """

    def __init__(
        self,
        scenario: AuthoringScenario | None = None,
        *,
        path: Path | None = None,
        ownership: DocumentOwnership = DocumentOwnership.NEW,
        undo_stack: UndoStack | None = None,
        saved: bool = False,
    ) -> None:
        self._scenario = scenario or AuthoringScenario()
        self._selection: AuthoringSubject | None = None
        self._revision = 0
        self._path = Path(path).resolve() if path is not None else None
        self._ownership = DocumentOwnership(ownership)
        self._undo_stack = undo_stack or CommandStack()
        self._listeners: list[Callable[[DocumentEvent], None]] = []
        self._saved_scenario = self._scenario if saved else None
        self._transient_before: _DocumentState | None = None
        self._transient_label = ""
        if saved:
            self._undo_stack.set_clean()

    @classmethod
    def new(cls, *, undo_stack: UndoStack | None = None) -> "ScenarioDocument":
        """Create an unsaved document on the explicit Empty library scene."""
        return cls(
            AuthoringScenario(scene=SceneReference("library", "empty/empty.xml")),
            undo_stack=undo_stack,
        )

    @classmethod
    def loaded(
        cls,
        scenario: AuthoringScenario,
        path: Path,
        *,
        undo_stack: UndoStack | None = None,
    ) -> "ScenarioDocument":
        """Create a clean builder-owned document loaded from disk."""
        return cls(
            scenario,
            path=path,
            ownership=DocumentOwnership.OWNED,
            undo_stack=undo_stack,
            saved=True,
        )

    @property
    def scenario(self) -> AuthoringScenario:
        return self._scenario

    @property
    def actors(self) -> tuple[AuthoringActor, ...]:
        return self._scenario.actors

    @property
    def groups(self) -> tuple[AuthoringGroup, ...]:
        return self._scenario.groups

    @property
    def selected_subject(self) -> AuthoringSubject | None:
        return self._selection

    @property
    def selected_actor_id(self) -> UUID | None:
        if self._selection is None or self._selection.kind is not SubjectKind.ACTOR:
            return None
        return self._selection.id

    @property
    def selected_group_id(self) -> UUID | None:
        if self._selection is None or self._selection.kind is not SubjectKind.GROUP:
            return None
        return self._selection.id

    @property
    def selected_actor(self) -> AuthoringActor | None:
        actor_id = self.selected_actor_id
        return self._scenario.actor(actor_id) if actor_id is not None else None

    @property
    def selected_group(self) -> AuthoringGroup | None:
        group_id = self.selected_group_id
        return self._scenario.group(group_id) if group_id is not None else None

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def ownership(self) -> DocumentOwnership:
        return self._ownership

    @property
    def read_only(self) -> bool:
        return self._ownership is DocumentOwnership.READ_ONLY

    @property
    def dirty(self) -> bool:
        return self._saved_scenario is None or self._scenario != self._saved_scenario

    @property
    def undo_stack(self) -> UndoStack:
        return self._undo_stack

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack.can_undo())

    @property
    def can_redo(self) -> bool:
        return bool(self._undo_stack.can_redo())

    def subscribe(self, listener: Callable[[DocumentEvent], None]) -> Callable[[], None]:
        """Subscribe to state notifications and return an unsubscribe callback."""
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def _emit(self, kind: DocumentEventKind) -> None:
        event = DocumentEvent(
            kind,
            self._revision,
            self.dirty,
            self.selected_actor_id,
            self.selected_group_id,
        )
        for listener in tuple(self._listeners):
            listener(event)

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise PermissionError("the authoring document is read-only")
        if self._transient_before is not None:
            raise RuntimeError("finish or cancel the transient edit first")

    def _state(self) -> _DocumentState:
        return _DocumentState(self._scenario, self._selection)

    def _apply_state(self, state: _DocumentState) -> None:
        content_changed = state.scenario != self._scenario
        selection_changed = state.selection != self._selection
        self._scenario = state.scenario
        self._selection = (
            state.selection
            if state.selection is None or state.scenario.subject(state.selection) is not None
            else None
        )
        if content_changed:
            self._revision += 1
            self._emit(DocumentEventKind.CONTENT)
        elif selection_changed:
            self._emit(DocumentEventKind.SELECTION)

    def _push(
        self,
        scenario: AuthoringScenario,
        selection: AuthoringSubject | UUID | None,
        text: str,
    ) -> None:
        self._ensure_writable()
        scenario = self._with_referenced_resources(scenario)
        before = self._state()
        normalized = (
            AuthoringSubject(SubjectKind.ACTOR, selection)
            if isinstance(selection, UUID)
            else selection
        )
        after = _DocumentState(scenario, normalized)
        if before == after:
            return
        self._undo_stack.push(_ReplaceStateCommand(self, before, after, text))

    @staticmethod
    def _with_referenced_resources(
        scenario: AuthoringScenario,
    ) -> AuthoringScenario:
        """Discard resource registrations no mobility value references."""

        referenced: set[str] = set()
        mobilities = (
            *(actor.mobility for actor in scenario.actors),
            *(group.mobility for group in scenario.groups),
        )
        for mobility in mobilities:
            if isinstance(mobility, MeshSequenceMobilitySpec):
                referenced.add(mobility.positions_path)
            elif isinstance(mobility, NetworkRouteMobilitySpec) and mobility.graph_path is not None:
                referenced.add(mobility.graph_path)
        resources = tuple(
            resource for resource in scenario.resources if resource.relative_path in referenced
        )
        return (
            scenario
            if resources == scenario.resources
            else replace(
                scenario,
                resources=resources,
            )
        )

    def select(self, actor_id: UUID | str | None) -> None:
        """Change selection without changing content, revision, or undo history."""
        selection = (
            AuthoringSubject(SubjectKind.ACTOR, UUID(str(actor_id)))
            if actor_id is not None
            else None
        )
        self.select_subject(selection)

    def select_group(self, group_id: UUID | str | None) -> None:
        """Select a group without changing content or undo history."""

        selection = (
            AuthoringSubject(SubjectKind.GROUP, UUID(str(group_id)))
            if group_id is not None
            else None
        )
        self.select_subject(selection)

    def select_subject(self, selection: AuthoringSubject | None) -> None:
        """Select an actor or group by immutable document identity."""

        if selection is not None and self._scenario.subject(selection) is None:
            raise KeyError(f"unknown {selection.kind.value} id: {selection.id}")
        if selection == self._selection:
            return
        self._selection = selection
        self._emit(DocumentEventKind.SELECTION)

    def add_actor(self, actor: AuthoringActor, *, select: bool = True) -> None:
        """Append an actor as one undoable command."""
        if self._scenario.actor(actor.id) is not None:
            raise ValueError(f"duplicate actor id: {actor.id}")
        scenario = replace(self._scenario, actors=(*self._scenario.actors, actor))
        selection = AuthoringSubject(SubjectKind.ACTOR, actor.id) if select else self._selection
        self._push(scenario, selection, f"Add {actor.name}")

    def add_default_actor(
        self,
        role: ActorRole | str,
        name: str | None = None,
    ) -> AuthoringActor:
        """Create and add a role-appropriate incomplete/default actor."""
        normalized_role = ActorRole(role)
        if name is None:
            prefix = {
                ActorRole.TX: "TX",
                ActorRole.RX: "RX",
                ActorRole.TARGET: "Target",
            }[normalized_role]
            used = {actor.name for actor in self.actors}
            suffix = 1
            while f"{prefix}{suffix}" in used:
                suffix += 1
            name = f"{prefix}{suffix}"
        actor = AuthoringActor.create(normalized_role, name)
        self.add_actor(actor)
        return actor

    def remove_actor(self, actor_id: UUID | str) -> AuthoringActor:
        """Remove one actor while leaving UUID references explicitly dangling."""
        wanted = UUID(str(actor_id))
        actor = self._scenario.actor(wanted)
        if actor is None:
            raise KeyError(f"unknown actor id: {wanted}")
        capability = self._scenario.capability("identity", wanted)
        if not capability.editable:
            raise PermissionError(capability.reason)
        scenario = replace(
            self._scenario,
            actors=tuple(item for item in self._scenario.actors if item.id != wanted),
        )
        selection = (
            None
            if self._selection == AuthoringSubject(SubjectKind.ACTOR, wanted)
            else self._selection
        )
        self._push(scenario, selection, f"Delete {actor.name}")
        return actor

    def add_group(self, group: AuthoringGroup, *, select: bool = True) -> None:
        """Append a group as one undoable command."""

        if self._scenario.group(group.id) is not None:
            raise ValueError(f"duplicate group id: {group.id}")
        scenario = replace(self._scenario, groups=(*self._scenario.groups, group))
        selection = AuthoringSubject(SubjectKind.GROUP, group.id) if select else self._selection
        self._push(scenario, selection, f"Add {group.name}")

    def add_group_with_members(
        self,
        group: AuthoringGroup,
        members: tuple[AuthoringActor, ...],
    ) -> None:
        """Add a group and its member actor updates as one command."""

        if self._scenario.group(group.id) is not None:
            raise ValueError(f"duplicate group id: {group.id}")
        replacements = {member.id: member for member in members}
        if len(replacements) != len(members):
            raise ValueError("group members must have unique actor ids")
        for member in members:
            existing = self._scenario.actor(member.id)
            if existing is None:
                raise KeyError(f"unknown actor id: {member.id}")
            if existing.role != member.role:
                raise ValueError("actor role is immutable")
        scenario = replace(
            self._scenario,
            actors=tuple(replacements.get(actor.id, actor) for actor in self._scenario.actors),
            groups=(*self._scenario.groups, group),
        )
        self._push(
            scenario,
            AuthoringSubject(SubjectKind.GROUP, group.id),
            f"Create {group.name}",
        )

    def add_default_group(self, name: str | None = None) -> AuthoringGroup:
        """Create and add a stationary group."""

        if name is None:
            used = {group.name for group in self.groups}
            suffix = 1
            while f"Group{suffix}" in used:
                suffix += 1
            name = f"Group{suffix}"
        group = AuthoringGroup.create(name)
        self.add_group(group)
        return group

    def remove_group(self, group_id: UUID | str) -> AuthoringGroup:
        """Remove one group while leaving member references diagnosable."""

        wanted = UUID(str(group_id))
        group = self._scenario.group(wanted)
        if group is None:
            raise KeyError(f"unknown group id: {wanted}")
        scenario = replace(
            self._scenario,
            groups=tuple(item for item in self._scenario.groups if item.id != wanted),
        )
        selection = (
            None
            if self._selection == AuthoringSubject(SubjectKind.GROUP, wanted)
            else self._selection
        )
        self._push(scenario, selection, f"Delete {group.name}")
        return group

    def replace_actor(self, actor: AuthoringActor, *, text: str | None = None) -> None:
        """Replace an actor without allowing its identity or role to change."""
        existing = self._scenario.actor(actor.id)
        if existing is None:
            raise KeyError(f"unknown actor id: {actor.id}")
        if actor.role != existing.role:
            raise ValueError("actor role is immutable")
        if actor.name != existing.name:
            capability = self._scenario.capability("identity", actor.id)
            if not capability.editable:
                raise PermissionError(capability.reason)
        if actor.mobility != existing.mobility:
            capability = self._scenario.capability("mobility", actor.id)
            if not capability.editable:
                raise PermissionError(capability.reason)
        if actor.orientation != existing.orientation:
            capability = self._scenario.capability("orientation", actor.id)
            if not capability.editable:
                raise PermissionError(capability.reason)
        if actor.target != existing.target:
            capability = self._scenario.capability("target_asset", actor.id)
            if not capability.editable and actor.target is not None and existing.target is not None:
                locked_locator = (
                    actor.target.source,
                    actor.target.path,
                    actor.target.asset_id,
                    actor.target.mesh_directory,
                    actor.target.mesh_pattern,
                    actor.target.start_index,
                    actor.target.frame_stride,
                )
                existing_locator = (
                    existing.target.source,
                    existing.target.path,
                    existing.target.asset_id,
                    existing.target.mesh_directory,
                    existing.target.mesh_pattern,
                    existing.target.start_index,
                    existing.target.frame_stride,
                )
                if locked_locator != existing_locator:
                    raise PermissionError(capability.reason)
        self._push(
            self._scenario.replace_actor(actor),
            self._selection,
            text or f"Edit {actor.name}",
        )

    @staticmethod
    def _merged_resources(
        scenario: AuthoringScenario,
        resources: tuple[AuthoringResource, ...],
    ) -> tuple[AuthoringResource, ...]:
        merged = {resource.relative_path: resource for resource in scenario.resources}
        merged.update({resource.relative_path: resource for resource in resources})
        return tuple(merged.values())

    def replace_actor_with_resources(
        self,
        actor: AuthoringActor,
        resources: tuple[AuthoringResource, ...],
        *,
        text: str | None = None,
    ) -> None:
        """Replace an actor and register its external resources atomically."""

        scenario = self._scenario.replace_actor(actor)
        scenario = replace(
            scenario,
            resources=self._merged_resources(scenario, resources),
        )
        self._push(scenario, self._selection, text or f"Edit {actor.name}")

    def update_actor(self, actor_id: UUID | str, **changes: object) -> AuthoringActor:
        """Update fields on an actor as one undoable command."""
        actor = self._scenario.actor(actor_id)
        if actor is None:
            raise KeyError(f"unknown actor id: {actor_id}")
        updated = actor.with_changes(**changes)
        self.replace_actor(updated)
        return updated

    def rename_actor(self, actor_id: UUID | str, name: str) -> AuthoringActor:
        """Rename an actor; UUID look-at references remain unchanged."""
        actor = self._scenario.actor(actor_id)
        if actor is None:
            raise KeyError(f"unknown actor id: {actor_id}")
        renamed = actor.with_changes(name=name)
        self.replace_actor(renamed, text=f"Rename {actor.name}")
        return renamed

    def replace_group(self, group: AuthoringGroup, *, text: str | None = None) -> None:
        """Replace a group without changing its identity."""

        existing = self._scenario.group(group.id)
        if existing is None:
            raise KeyError(f"unknown group id: {group.id}")
        capability = self._scenario.capability("group", group.id)
        if not capability.editable and group != existing:
            raise PermissionError(capability.reason)
        self._push(
            self._scenario.replace_group(group),
            self._selection,
            text or f"Edit {group.name}",
        )

    def replace_group_with_resources(
        self,
        group: AuthoringGroup,
        resources: tuple[AuthoringResource, ...],
        *,
        text: str | None = None,
    ) -> None:
        """Replace a group and register its external resources atomically."""

        scenario = self._scenario.replace_group(group)
        scenario = replace(
            scenario,
            resources=self._merged_resources(scenario, resources),
        )
        self._push(scenario, self._selection, text or f"Edit {group.name}")

    def update_group(self, group_id: UUID | str, **changes: object) -> AuthoringGroup:
        """Apply validated group field changes as one undoable command."""

        group = self._scenario.group(group_id)
        if group is None:
            raise KeyError(f"unknown group id: {group_id}")
        updated = group.with_changes(**changes)
        self.replace_group(updated)
        return updated

    def rename_group(self, group_id: UUID | str, name: str) -> AuthoringGroup:
        """Rename a group without changing its UUID-based member references."""

        group = self._scenario.group(group_id)
        if group is None:
            raise KeyError(f"unknown group id: {group_id}")
        renamed = group.with_changes(name=name)
        self.replace_group(renamed, text=f"Rename {group.name}")
        return renamed

    def set_scene(self, scene: SceneReference | None) -> None:
        """Set the exact scene reference as one undoable command."""
        capability = self._scenario.capability("scene")
        if not capability.editable:
            raise PermissionError(capability.reason)
        self._push(replace(self._scenario, scene=scene), self._selection, "Change scene")

    def set_timeline(
        self,
        timeline: TimelineSettings,
        *,
        replace_source_quality: bool = False,
    ) -> None:
        """Replace timeline settings and any explicitly superseded quality block."""

        source_snapshot = self._scenario.source_snapshot
        if replace_source_quality:
            source_snapshot = source_snapshot.without_path("raytracing.quality.custom")
        self._push(
            replace(
                self._scenario,
                timeline=timeline,
                source_snapshot=source_snapshot,
            ),
            self._selection,
            "Change timeline",
        )

    def undo(self) -> None:
        """Undo one durable content command."""
        if self._transient_before is not None:
            raise RuntimeError("finish or cancel the transient edit first")
        self._undo_stack.undo()

    def redo(self) -> None:
        """Redo one durable content command."""
        if self._transient_before is not None:
            raise RuntimeError("finish or cancel the transient edit first")
        self._undo_stack.redo()

    def begin_transient_edit(self, label: str = "Move actor") -> None:
        """Start a direct-manipulation edit that will become one undo item."""
        self._ensure_writable()
        self._transient_before = self._state()
        self._transient_label = label

    def update_transient_actor(self, actor: AuthoringActor) -> None:
        """Publish one intermediate actor value without changing revision."""
        if self._transient_before is None:
            raise RuntimeError("no transient edit is active")
        existing = self._scenario.actor(actor.id)
        if existing is None:
            raise KeyError(f"unknown actor id: {actor.id}")
        if existing.role != actor.role:
            raise ValueError("actor role is immutable")
        self._scenario = self._scenario.replace_actor(actor)
        self._emit(DocumentEventKind.TRANSIENT)

    def update_transient_group(self, group: AuthoringGroup) -> None:
        """Publish one intermediate group value without changing revision."""

        if self._transient_before is None:
            raise RuntimeError("no transient edit is active")
        if self._scenario.group(group.id) is None:
            raise KeyError(f"unknown group id: {group.id}")
        self._scenario = self._scenario.replace_group(group)
        self._emit(DocumentEventKind.TRANSIENT)

    def commit_transient_edit(self) -> None:
        """Commit all transient updates as exactly one undo command."""
        if self._transient_before is None:
            raise RuntimeError("no transient edit is active")
        before = self._transient_before
        after = self._state()
        label = self._transient_label
        self._transient_before = None
        self._transient_label = ""
        if before == after:
            return
        self._scenario = before.scenario
        self._selection = before.selection
        self._undo_stack.push(_ReplaceStateCommand(self, before, after, label))

    def cancel_transient_edit(self) -> None:
        """Restore the exact state from before a transient edit."""
        if self._transient_before is None:
            return
        before = self._transient_before
        self._transient_before = None
        self._transient_label = ""
        self._scenario = before.scenario
        self._selection = before.selection
        self._emit(DocumentEventKind.TRANSIENT)

    def mark_saved(self, path: Path) -> None:
        """Record a successful atomic save and establish a new clean state."""
        self._ensure_writable()
        resolved = Path(path).resolve()
        if resolved.name != "scenario.yaml":
            raise ValueError("authoring documents must be saved as scenario.yaml")
        owned_resources = tuple(
            AuthoringResource(
                kind=resource.kind,
                source_path=resolved.parent / resource.relative_path,
                relative_path=resource.relative_path,
            )
            for resource in self._scenario.resources
        )
        owned_dependencies = tuple(
            (
                dependency
                if dependency.external
                else ScenarioDependency(
                    source_path=resolved.parent / dependency.relative_path,
                    relative_path=dependency.relative_path,
                    kind=dependency.kind,
                    origin_path=dependency.origin_path,
                )
            )
            for dependency in self._scenario.dependencies
        )
        self._scenario = replace(
            self._scenario,
            resources=owned_resources,
            dependencies=owned_dependencies,
        )
        self._path = resolved
        self._ownership = DocumentOwnership.OWNED
        self._saved_scenario = self._scenario
        self._undo_stack.set_clean()
        self._emit(DocumentEventKind.SAVED)
