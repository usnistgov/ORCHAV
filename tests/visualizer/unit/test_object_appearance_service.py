from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np

from visualizer.src.materials.appearance import MaterialDisplayMode
from visualizer.src.model import RenderObjectState, make_text_label_state
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.material_mode_commands import MaterialModeCommandService
from visualizer.src.services.material_modes import MaterialModeService
from visualizer.src.services.object_appearance_service import ObjectAppearanceService
from visualizer.src.types.render_payloads import MaterialPayload, MeshPayload


def _mesh_payload() -> MeshPayload:
    return MeshPayload(
        vertices=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        triangles=np.empty((0, 3), dtype=np.int32),
    )


def _scene_state(name: str) -> RenderObjectState:
    return RenderObjectState(
        id=f"scene:{name}::mesh",
        payload=_mesh_payload(),
        metadata={"type": "scene_mesh"},
    )


def _target_state(name: str) -> RenderObjectState:
    return RenderObjectState(
        id=f"target:{name}::mesh",
        payload=_mesh_payload(),
        metadata={"type": "target_mesh"},
    )


class _Renderer:
    capabilities = RendererCapabilities(pbr=True)

    def __init__(self) -> None:
        self.named: set[str] = set()
        self.hidden: set[str] = set()
        self.visibility_calls: list[tuple[str, bool]] = []
        self.ensure_calls: list[dict] = []
        self.object_ensure_calls: list[Any] = []
        self.material_calls: list[tuple[str, MaterialPayload]] = []
        self.object_material_calls: list[tuple[str, MaterialPayload]] = []
        self.update_count = 0
        self.batch_depth = 0
        self.outer_batch_count = 0

    @contextmanager
    def batch_updates(self):
        outermost = self.batch_depth == 0
        if outermost:
            self.outer_batch_count += 1
        self.batch_depth += 1
        try:
            yield
        finally:
            self.batch_depth -= 1

    def has_named_geometry(self, name: str) -> bool:
        return name in self.named

    def is_named_visible(self, name: str):
        if name not in self.named:
            return None
        return name not in self.hidden

    def set_named_visibility(self, name: str, visible: bool) -> bool:
        self.visibility_calls.append((name, bool(visible)))
        self.named.add(name)
        if visible:
            self.hidden.discard(name)
        else:
            self.hidden.add(name)
        return True

    def set_visible(self, name: str, visible: bool) -> bool:
        return self.set_named_visibility(name, visible)

    def ensure_named_geometry(self, **kwargs) -> bool:
        self.ensure_calls.append(kwargs)
        self.named.add(kwargs["name"])
        if kwargs.get("visible", True):
            self.hidden.discard(kwargs["name"])
        else:
            self.hidden.add(kwargs["name"])
        return True

    def set_named_material(self, name: str, material: MaterialPayload) -> bool:
        self.material_calls.append((name, material))
        self.named.add(name)
        return True

    def ensure_object(self, obj: Any) -> bool:
        self.object_ensure_calls.append(obj)
        self.named.add(obj.id)
        return True

    def set_material(self, name: str, material: MaterialPayload) -> bool:
        self.object_material_calls.append((name, material))
        return name in self.named

    def update_renderer(self) -> None:
        self.update_count += 1


class _PbrService:
    def __init__(self) -> None:
        self.applied: list[tuple[str, bool]] = []

    def get_entry_geometry_name(self, entry: dict) -> str:
        return entry["geometry_name"]

    def get_effective_entry_properties(self, entry: dict) -> dict:
        props = dict(entry.get("pbr_properties", {}))
        props.setdefault("color", entry.get("color", [0.3, 0.4, 0.5]))
        props.setdefault("roughness", 0.7)
        props.setdefault("metallic", 0.1)
        props.setdefault("reflectance", 0.2)
        props.setdefault("alpha", 1.0)
        return props

    def apply_entry_material(
        self,
        entry: dict,
        *,
        highlighted: bool = False,
        effective_visible: bool | None = None,
    ) -> bool:
        self.applied.append((entry["geometry_name"], highlighted))
        return False


def _viz(**overrides):
    renderer = overrides.pop("renderer", _Renderer())
    values = {
        "renderer": renderer,
        "vis_initialized": True,
        "vis": object(),
        "mesh_entries": [],
        "target_entries": [],
        "tx_entries": [],
        "rx_entries": [],
        "material_pbr_service": _PbrService(),
        "scene_service": SimpleNamespace(),
        "scene_appearance_service": None,
        "target_service": None,
        "app_state": SimpleNamespace(camera_mode="overview", pov_hidden_node=None),
        "outlines_enabled": False,
        "target_outlines_enabled": False,
        "building_labels": [],
    }
    values.update(overrides)
    viz = SimpleNamespace(**values)
    viz.object_appearance_service = ObjectAppearanceService(viz)
    return viz


def test_resolve_canonical_entry_across_mesh_target_and_nodes() -> None:
    mesh = {"name": "Wall", "object_key": "scene:wall::mesh"}
    target = {"name": "Ped", "object_id": "target:ped"}
    tx = {"name": "TX1", "entry_type": "tx", "node_index": 0}
    rx = {"name": "RX1", "entry_type": "rx", "node_index": 0}
    viz = _viz(mesh_entries=[mesh], target_entries=[target], tx_entries=[tx], rx_entries=[rx])
    service = viz.object_appearance_service

    assert service.resolve_canonical_entry({"object_key": "scene:wall::mesh"}) is mesh
    assert service.resolve_canonical_entry({"object_id": "target:ped"}) is target
    assert service.resolve_canonical_entry({"entry_type": "tx", "node_index": 0}) is tx
    assert service.resolve_canonical_entry({"entry_type": "rx", "node_index": 0}) is rx


def test_visibility_toggle_updates_entry_and_common_renderer_visibility() -> None:
    mesh = _scene_state("wall")
    entry = {
        "name": "Wall",
        "entry_type": "mesh",
        "mesh": mesh,
        "visible": True,
        "geometry_name": "scene:wall::mesh",
    }
    renderer = _Renderer()
    renderer.named.add("scene:wall::mesh")
    sync_batch = Mock(return_value=True)
    viz = _viz(
        renderer=renderer,
        mesh_entries=[entry],
        scene_service=SimpleNamespace(sync_scene_entry_visibility_batch=sync_batch),
    )

    viz.object_appearance_service.set_object_visibility(entry, False)

    assert entry["visible"] is False
    sync_batch.assert_called_once_with([(entry, False)])
    assert renderer.visibility_calls == []
    assert renderer.update_count == 1


def test_outline_sync_has_no_renderer_fallback_without_scene_owner() -> None:
    renderer = _Renderer()
    outline = RenderObjectState(
        id="scene:wall::outline",
        payload=_mesh_payload(),
        visible=True,
    )
    entry = {
        "name": "Wall",
        "entry_type": "mesh",
        "mesh": _scene_state("wall"),
        "visible": False,
        "outline_geometry": outline,
        "outline_visible": True,
    }
    viz = _viz(
        renderer=renderer,
        mesh_entries=[entry],
        outlines_enabled=True,
        scene_appearance_service=None,
    )

    viz.object_appearance_service._sync_outline_visibility(
        entry,
        effective_visible=False,
    )

    assert outline.visible is True
    assert entry["outline_visible"] is True
    assert renderer.object_ensure_calls == []


def test_visibility_batch_delegates_scene_aggregation_once_and_redraws_once() -> None:
    states = [_scene_state("wall_a"), _scene_state("wall_b")]
    entries = [
        {
            "name": "Wall A",
            "entry_type": "mesh",
            "mesh": states[0],
            "visible": False,
        },
        {
            "name": "Wall B",
            "entry_type": "mesh",
            "mesh": states[1],
            "visible": True,
        },
    ]
    renderer = _Renderer()
    sync_batch = Mock(side_effect=lambda _batch: renderer.batch_depth == 1)
    viz = _viz(
        renderer=renderer,
        mesh_entries=entries,
        scene_service=SimpleNamespace(sync_scene_entry_visibility_batch=sync_batch),
    )

    assert viz.object_appearance_service.refresh_object_visibility_batch(entries)

    assert states[0].visible is False
    assert states[1].visible is True
    assert sync_batch.call_args.args[0] == [
        (entries[0], False),
        (entries[1], True),
    ]
    assert renderer.outer_batch_count == 1
    assert renderer.update_count == 1


def test_visibility_batch_syncs_target_once_without_scene_batch() -> None:
    state = _target_state("ped")
    entry = {
        "name": "Ped",
        "entry_type": "target",
        "node_index": 0,
        "mesh": state,
        "visible": False,
        "_frame_visible": True,
    }
    renderer = _Renderer()
    sync_target = Mock(side_effect=lambda _entry, **_kwargs: renderer.batch_depth == 1)
    sync_scene = Mock(return_value=True)
    viz = _viz(
        renderer=renderer,
        target_entries=[entry],
        target_service=SimpleNamespace(sync_target_entry_snapshot=sync_target),
        scene_service=SimpleNamespace(sync_scene_entry_visibility_batch=sync_scene),
    )

    assert viz.object_appearance_service.refresh_object_visibility_batch([entry])

    assert state.visible is False
    sync_target.assert_called_once_with(entry, effective_visible=False)
    sync_scene.assert_not_called()
    assert renderer.outer_batch_count == 1
    assert renderer.update_count == 1


def test_mixed_visibility_batch_stages_each_owner_and_redraws_once() -> None:
    scene_state = _scene_state("wall")
    target_state = _target_state("ped")
    scene_entry = {
        "name": "Wall",
        "entry_type": "mesh",
        "mesh": scene_state,
        "visible": True,
    }
    target_entry = {
        "name": "Ped",
        "entry_type": "target",
        "node_index": 0,
        "mesh": target_state,
        "visible": True,
        "_frame_visible": True,
    }
    tx_entry = {
        "name": "TX1",
        "entry_type": "tx",
        "node_index": 0,
        "visible": True,
    }
    rx_entry = {
        "name": "RX1",
        "entry_type": "rx",
        "node_index": 0,
        "visible": True,
    }
    renderer = _Renderer()

    def _assert_batched(*_args, **_kwargs) -> bool:
        return renderer.batch_depth == 1

    sync_scene = Mock(side_effect=_assert_batched)
    sync_target = Mock(side_effect=_assert_batched)
    sync_nodes = Mock(side_effect=_assert_batched)
    viz = _viz(
        renderer=renderer,
        mesh_entries=[scene_entry],
        target_entries=[target_entry],
        tx_entries=[tx_entry],
        rx_entries=[rx_entry],
        scene_service=SimpleNamespace(sync_scene_entry_visibility_batch=sync_scene),
        target_service=SimpleNamespace(sync_target_entry_snapshot=sync_target),
        node_service=SimpleNamespace(sync_node_visibility_snapshot=sync_nodes),
    )
    stale_scene_copy = {"name": "Wall", "entry_type": "mesh", "visible": True}
    stale_target_copy = {
        "name": "Ped",
        "entry_type": "target",
        "node_index": 0,
        "visible": True,
    }

    assert viz.object_appearance_service.set_object_visibility_batch(
        [
            stale_scene_copy,
            scene_entry,
            stale_target_copy,
            {"entry_type": "tx", "node_index": 0, "visible": True},
            {"entry_type": "rx", "node_index": 0, "visible": True},
        ],
        False,
    )

    assert sync_scene.call_args.args[0] == [(scene_entry, False)]
    sync_target.assert_called_once_with(target_entry, effective_visible=False)
    sync_nodes.assert_called_once_with()
    assert all(
        entry["visible"] is False for entry in (scene_entry, target_entry, tx_entry, rx_entry)
    )
    assert stale_scene_copy["visible"] is False
    assert stale_target_copy["visible"] is False
    assert scene_state.visible is False
    assert target_state.visible is False
    assert renderer.outer_batch_count == 1
    assert renderer.update_count == 1


def test_individual_node_visibility_uses_the_node_snapshot_path() -> None:
    tx_entry = {
        "name": "TX1",
        "entry_type": "tx",
        "node_index": 0,
        "visible": True,
    }
    stale_copy = {"entry_type": "tx", "node_index": 0, "visible": True}
    renderer = _Renderer()
    sync_nodes = Mock(side_effect=lambda: renderer.batch_depth == 1)
    viz = _viz(
        renderer=renderer,
        tx_entries=[tx_entry],
        node_service=SimpleNamespace(sync_node_visibility_snapshot=sync_nodes),
    )

    viz.object_appearance_service.set_object_visibility(stale_copy, False)

    assert stale_copy["visible"] is False
    assert tx_entry["visible"] is False
    sync_nodes.assert_called_once_with()
    assert renderer.outer_batch_count == 1
    assert renderer.update_count == 1


def test_semantic_visibility_toggle_updates_render_handle_intent() -> None:
    state = RenderObjectState(
        id="target:ped::mesh",
        payload=MeshPayload(
            vertices=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            triangles=np.empty((0, 3), dtype=np.int32),
        ),
    )
    entry = {
        "name": "Ped",
        "entry_type": "target",
        "mesh": state,
        "visible": True,
        "geometry_name": state.id,
    }
    renderer = _Renderer()
    renderer.named.add(state.id)
    sync_snapshot = Mock(return_value=True)
    viz = _viz(
        renderer=renderer,
        target_entries=[entry],
        target_service=SimpleNamespace(sync_target_entry_snapshot=sync_snapshot),
    )

    viz.object_appearance_service.set_object_visibility(entry, False)

    assert entry["visible"] is False
    assert state.visible is False
    sync_snapshot.assert_called_once_with(entry, effective_visible=False)
    assert renderer.visibility_calls == []


def test_semantic_target_show_keeps_frame_and_pov_hidden_object_hidden() -> None:
    state = RenderObjectState(
        id="target:ped::mesh",
        payload=MeshPayload(
            vertices=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            triangles=np.empty((0, 3), dtype=np.int32),
        ),
        visible=False,
    )
    entry = {
        "name": "Ped",
        "entry_type": "target",
        "node_index": 0,
        "mesh": state,
        "visible": False,
        "_frame_visible": False,
        "geometry_name": state.id,
    }
    renderer = _Renderer()
    renderer.named.add(state.id)
    hidden_for_pov = Mock(return_value=False)
    sync_snapshot = Mock(return_value=True)
    viz = _viz(
        renderer=renderer,
        target_entries=[entry],
        node_service=SimpleNamespace(_is_hidden_for_pov=hidden_for_pov),
        target_service=SimpleNamespace(sync_target_entry_snapshot=sync_snapshot),
    )

    viz.object_appearance_service.set_object_visibility(entry, True)

    assert entry["visible"] is True
    assert state.visible is True
    assert sync_snapshot.call_args.kwargs["effective_visible"] is False

    entry["_frame_visible"] = True
    viz.app_state = SimpleNamespace(
        camera_mode="pov",
        pov_hidden_node=("target", 0),
    )
    viz.object_appearance_service.set_object_visibility(entry, True)

    assert sync_snapshot.call_args.kwargs["effective_visible"] is False
    hidden_for_pov.assert_not_called()


def test_target_material_refresh_delegates_without_mutating_render_intent() -> None:
    state = RenderObjectState(
        id="target:ped::mesh",
        payload=MeshPayload(
            vertices=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            triangles=np.empty((0, 3), dtype=np.int32),
        ),
        visible=True,
    )
    set_material = Mock(return_value=False)
    ensure_object = Mock(return_value=True)
    renderer = SimpleNamespace(
        set_material=set_material,
        ensure_object=ensure_object,
        request_redraw=Mock(),
    )
    entry = {
        "name": "Ped",
        "entry_type": "target",
        "mesh": state,
        "visible": True,
        "_frame_visible": False,
        "node_index": 0,
        "geometry_name": state.id,
    }
    hidden_for_pov = Mock(return_value=False)
    refresh_target = Mock(return_value=True)
    viz = _viz(
        renderer=renderer,
        target_entries=[entry],
        node_service=SimpleNamespace(_is_hidden_for_pov=hidden_for_pov),
        target_service=SimpleNamespace(refresh_target_entry_material=refresh_target),
    )

    assert viz.object_appearance_service.refresh_entry_material(entry)
    resolved = refresh_target.call_args.kwargs["resolved_appearance"]
    assert refresh_target.call_args.args == (entry,)
    assert resolved.visible is False
    assert entry["visible"] is True
    assert state.visible is True
    set_material.assert_not_called()
    ensure_object.assert_not_called()
    hidden_for_pov.assert_not_called()


def test_target_material_refresh_never_bypasses_target_service() -> None:
    renderer = _Renderer()
    renderer.set_material = Mock(return_value=False)
    renderer.ensure_object = Mock(return_value=True)
    renderer.set_named_material = Mock(return_value=True)
    renderer.ensure_named_geometry = Mock(return_value=True)
    mesh = _target_state("ped")
    entry = {
        "name": "Ped",
        "entry_type": "target",
        "mesh": mesh,
        "visible": True,
        "_frame_visible": False,
        "node_index": 0,
        "geometry_name": "target:ped::mesh",
    }
    refresh_target = Mock(return_value=True)
    viz = _viz(
        renderer=renderer,
        target_entries=[entry],
        node_service=SimpleNamespace(_is_hidden_for_pov=Mock(return_value=False)),
        target_service=SimpleNamespace(refresh_target_entry_material=refresh_target),
    )

    assert viz.object_appearance_service.refresh_entry_material(entry)

    resolved = refresh_target.call_args.kwargs["resolved_appearance"]
    assert refresh_target.call_args.args == (entry,)
    assert resolved.visible is False
    renderer.set_material.assert_not_called()
    renderer.ensure_object.assert_not_called()
    renderer.set_named_material.assert_not_called()
    renderer.ensure_named_geometry.assert_not_called()


def test_highlight_refresh_uses_effective_material_payload(tmp_path, monkeypatch) -> None:
    albedo = tmp_path / "albedo.png"
    albedo.write_bytes(b"texture")
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    entry = {
        "name": "Ped",
        "target_name": "Ped",
        "entry_type": "target",
        "mesh": _target_state("ped"),
        "geometry_name": "target:ped::mesh",
        "color": [0.2, 0.3, 0.4],
        "pbr_properties": {"texture_path": str(albedo)},
    }
    renderer = _Renderer()
    refresh_target = Mock(return_value=True)
    viz = _viz(
        renderer=renderer,
        target_entries=[entry],
        target_service=SimpleNamespace(refresh_target_entry_material=refresh_target),
    )

    viz.object_appearance_service.set_object_highlight(entry, True)

    assert entry["highlighted"] is True
    resolved = refresh_target.call_args.kwargs["resolved_appearance"]
    assert refresh_target.call_args.args == (entry,)
    assert resolved.visible is True
    assert entry["mesh"].material.texture_path == str(albedo)
    assert entry["mesh"].material.color_multiplier == (1.0, 1.0, 1.0)
    assert resolved.material.texture_path == str(albedo)
    assert resolved.material.color_multiplier == (1.0, 0.3, 0.3)
    assert renderer.object_material_calls == []


def test_scene_material_refresh_delegates_to_scene_snapshot_owner() -> None:
    state = _scene_state("wall")
    entry = {
        "name": "Wall",
        "entry_type": "mesh",
        "stable_mesh_id": "wall",
        "mesh": state,
        "visible": True,
        "color": [0.2, 0.3, 0.4],
    }
    sync_resolved = Mock(return_value=True)
    renderer = SimpleNamespace(
        set_material=Mock(return_value=True),
        ensure_object=Mock(return_value=True),
        update_renderer=Mock(),
    )
    pbr_service = _PbrService()
    viz = _viz(
        renderer=renderer,
        mesh_entries=[entry],
        material_pbr_service=pbr_service,
        scene_service=SimpleNamespace(sync_scene_resolved_appearance=sync_resolved),
    )

    assert viz.object_appearance_service.refresh_entry_material(entry)

    sync_resolved.assert_called_once()
    call = sync_resolved.call_args
    assert call.args[0] is entry
    assert call.args[1].visible is True
    assert pbr_service.applied == []
    renderer.set_material.assert_not_called()
    renderer.ensure_object.assert_not_called()


def test_scene_highlight_failure_does_not_retry_generic_material_path() -> None:
    entry = {
        "name": "Wall",
        "entry_type": "mesh",
        "mesh": _scene_state("wall"),
        "geometry_name": "scene:wall::mesh",
    }
    sync_resolved = Mock(return_value=False)
    pbr_service = _PbrService()
    viz = _viz(
        mesh_entries=[entry],
        material_pbr_service=pbr_service,
        scene_service=SimpleNamespace(
            sync_scene_resolved_appearance=sync_resolved,
        ),
    )

    viz.object_appearance_service.set_object_highlight(entry, True)

    sync_resolved.assert_called_once()
    assert sync_resolved.call_args.args[1].highlighted is True
    assert pbr_service.applied == []


def test_building_label_visibility_routes_render_state_through_common_sync() -> None:
    entry = {
        "name": "Building",
        "entry_type": "mesh",
        "mesh": _scene_state("building"),
        "geometry_name": "scene:building::mesh",
    }
    label = make_text_label_state("bldg_label_0", "Building", [0.8, 0.8, 0.8])
    renderer = SimpleNamespace(
        ensure_object=Mock(return_value=True),
        ensure_named_geometry=Mock(return_value=True),
        set_visible=Mock(return_value=True),
        set_named_visibility=Mock(return_value=True),
    )
    viz = _viz(
        renderer=renderer,
        mesh_entries=[entry],
        building_labels=[label],
    )

    viz.object_appearance_service.set_building_label_visibility(entry, True)
    viz.object_appearance_service.set_building_label_visibility(entry, False)

    assert [call.args[0].visible for call in renderer.ensure_object.call_args_list] == [True, False]
    renderer.ensure_named_geometry.assert_not_called()
    renderer.set_visible.assert_not_called()
    renderer.set_named_visibility.assert_not_called()


def test_failed_building_label_common_sync_does_not_bypass_object_cache() -> None:
    entry = {
        "name": "Building",
        "entry_type": "mesh",
        "mesh": _scene_state("building"),
        "geometry_name": "scene:building::mesh",
    }
    label = make_text_label_state("bldg_label_0", "Building", [0.8, 0.8, 0.8])
    renderer = SimpleNamespace(
        ensure_object=Mock(return_value=False),
        ensure_named_geometry=Mock(return_value=True),
    )
    viz = _viz(renderer=renderer, mesh_entries=[entry], building_labels=[label])

    viz.object_appearance_service.set_building_label_visibility(entry, True)

    renderer.ensure_object.assert_called_once()
    renderer.ensure_named_geometry.assert_not_called()


def test_building_label_visibility_follows_parent_visibility() -> None:
    entry = {
        "name": "Building",
        "entry_type": "mesh",
        "mesh": _scene_state("building"),
        "visible": False,
        "geometry_name": "scene:building::mesh",
    }
    label = make_text_label_state("bldg_label_0", "Building", [0.8, 0.8, 0.8])
    renderer = SimpleNamespace(
        ensure_object=Mock(return_value=True),
        has_named_geometry=Mock(return_value=True),
        is_named_visible=Mock(return_value=False),
        set_visible=Mock(return_value=True),
        request_redraw=Mock(),
    )
    viz = _viz(renderer=renderer, mesh_entries=[entry], building_labels=[label])

    viz.object_appearance_service.set_building_label_visibility(entry, True)
    assert renderer.ensure_object.call_args.args[0].visible is False

    viz.scene_service = SimpleNamespace(
        sync_scene_entry_visibility_batch=Mock(return_value=True),
    )
    viz.object_appearance_service.set_object_visibility(entry, True)
    assert renderer.ensure_object.call_args.args[0].visible is True


def test_target_label_visibility_routes_to_target_service() -> None:
    entry = {"entry_type": "target", "show_label": False}
    node_service = SimpleNamespace(update_target_label_visibility=Mock())
    sync_snapshot = Mock(return_value=True)
    viz = _viz(
        target_entries=[entry],
        node_service=node_service,
        target_service=SimpleNamespace(sync_target_entry_snapshot=sync_snapshot),
    )

    viz.object_appearance_service.set_building_label_visibility(entry, True)

    sync_snapshot.assert_called_once_with(entry)
    node_service.update_target_label_visibility.assert_not_called()
    assert entry["show_label"] is True


def test_material_mode_command_uses_one_appearance_batch_for_scene_and_targets() -> None:
    material_mode_service = MaterialModeService()
    material_mode_command_service = MaterialModeCommandService(material_mode_service)
    material_mode_service.set_mode("brick", MaterialDisplayMode.HIDDEN)
    material_mode_service.set_mode("pedestrian", MaterialDisplayMode.HIDDEN)
    scene_entry = {
        "name": "Wall",
        "entry_type": "mesh",
        "mesh": _scene_state("wall"),
        "geometry_name": "scene:wall::mesh",
        "material_id": "brick",
    }
    target_entry = {
        "name": "Ped",
        "entry_type": "target",
        "mesh": _target_state("ped"),
        "geometry_name": "target:ped::mesh",
        "material_id": "pedestrian",
    }
    renderer = _Renderer()
    renderer.named.add("scene:wall::mesh")
    sync_scene_batch = Mock(return_value=True)
    sync_target = Mock(return_value=True)
    viz = _viz(
        renderer=renderer,
        mesh_entries=[scene_entry],
        target_entries=[target_entry],
        scene_service=SimpleNamespace(
            sync_scene_resolved_appearance_batch=sync_scene_batch,
        ),
        target_service=SimpleNamespace(sync_target_entry_snapshot=sync_target),
    )
    appearance = viz.object_appearance_service
    viz.material_mode_service = material_mode_service

    material_mode_command_service.apply_material_modes(
        [scene_entry],
        [target_entry],
        appearance.refresh_entry_appearance_batch,
    )

    assert scene_entry.get("visible", True) is True
    assert target_entry.get("visible", True) is True
    scene_pairs = sync_scene_batch.call_args.args[0]
    assert scene_pairs[0][0] is scene_entry
    assert scene_pairs[0][1].visible is False
    assert sync_scene_batch.call_args.kwargs == {"materials_changed": False}
    target_resolved = sync_target.call_args.kwargs["resolved_appearance"]
    assert sync_target.call_args.args == (target_entry,)
    assert target_resolved.visible is False
    assert renderer.visibility_calls == []
    assert renderer.outer_batch_count == 1
    assert renderer.update_count == 1
