"""Renderer-neutral handoff helpers for persistent scene objects.

SceneService owns scene-loading and aggregate policy. This module translates
its stable :class:`RenderObjectState` instances to the common renderer object
contract without exposing backend-named geometry operations to scene code.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..model import RenderObjectState
from ..scene.geometry_payload_factory import merge_mesh_payloads
from ..services.object_identity import make_target_entry_geometry_name
from ..types.render_payloads import MaterialPayload, MeshPayload, material_payload_from_mapping

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer
    from ..renderers.protocol import RendererProtocol


def scene_mesh_payload(mesh: Any) -> MeshPayload | None:
    """Return a neutral mesh payload from scene-owned geometry."""
    if isinstance(mesh, RenderObjectState) and isinstance(mesh.payload, MeshPayload):
        return mesh.payload
    if isinstance(mesh, MeshPayload):
        return mesh
    return None


def merge_scene_mesh_payloads(meshes: list[Any]) -> MeshPayload | None:
    """Merge scene mesh payloads without entering a renderer backend."""
    payloads = [payload for mesh in meshes if (payload := scene_mesh_payload(mesh)) is not None]
    if not payloads:
        return None
    return merge_mesh_payloads(payloads)


def scene_mesh_has_triangle_uvs(mesh: Any) -> bool:
    """Return whether renderer-neutral scene mesh geometry has triangle UVs."""
    payload = scene_mesh_payload(mesh)
    if payload is not None:
        uvs = payload.triangle_uvs
        return uvs is not None and len(uvs) > 0
    return False


class SceneRenderSync:
    """Synchronize scene-owned states through the common object contract."""

    def __init__(self, visualizer: OrchavVisualizer) -> None:
        """Bind scene renderer synchronization to the active visualizer renderer."""
        self.visualizer = visualizer

    @property
    def renderer(self) -> RendererProtocol:
        """Return the active renderer through its explicit shared contract."""
        return self.visualizer.renderer

    def sync_label_geometry(self, *, name: str, geometry: Any, visible: bool) -> bool:
        """Synchronize a persistent text label through ``ensure_object``."""
        if not isinstance(geometry, RenderObjectState) or geometry.id != name:
            return False
        return self.ensure_object(geometry, effective_visible=bool(visible))

    def ensure_object(
        self,
        state: RenderObjectState,
        *,
        effective_visible: bool | None = None,
        material: dict[str, Any] | MaterialPayload | None = None,
        snapshot_material: dict[str, Any] | MaterialPayload | None = None,
    ) -> bool:
        """Synchronize one persistent render state through ``ensure_object``."""
        if material is not None and snapshot_material is not None:
            raise ValueError("material and snapshot_material are mutually exclusive")
        if material is not None:
            state.material = (
                material
                if isinstance(material, MaterialPayload)
                else material_payload_from_mapping(material)
            )
        snapshot = state.to_render_object(effective_visible=effective_visible)
        if snapshot_material is not None:
            effective_snapshot_material = (
                snapshot_material
                if isinstance(snapshot_material, MaterialPayload)
                else material_payload_from_mapping(snapshot_material)
            )
            snapshot = replace(snapshot, material=effective_snapshot_material)
        return bool(self.renderer.ensure_object(snapshot))

    def set_object_visibility(self, object_id: str, visible: bool) -> bool:
        """Set persistent-object visibility through the common contract."""
        return bool(self.renderer.set_visible(object_id, bool(visible)))

    def remove_object(self, state: RenderObjectState) -> bool:
        """Remove one persistent render state through the common contract."""
        return bool(self.renderer.remove_object(state.id))

    def set_object_material(
        self,
        state: RenderObjectState,
        material: dict[str, Any] | MaterialPayload,
        *,
        effective_visible: bool | None = None,
    ) -> bool:
        """Apply persistent-object material, ensuring the object if absent."""
        state.material = (
            material
            if isinstance(material, MaterialPayload)
            else material_payload_from_mapping(material)
        )
        if self.renderer.set_material(state.id, state.material):
            return True
        return self.ensure_object(state, effective_visible=effective_visible)

    def sync_target_label(
        self,
        *,
        target_entry: dict[str, Any],
        index: int,
        label: Any,
        visible: bool,
    ) -> None:
        """Sync a target label through the common object surface."""
        label_name = make_target_entry_geometry_name(target_entry, "label")
        self.sync_label_geometry(name=label_name, geometry=label, visible=visible)
