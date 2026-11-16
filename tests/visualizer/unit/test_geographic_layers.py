"""Geographic switches preserve geometry, textures and independent visibility intent."""

import json
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tests.visualizer.unit.test_visual_surfaces import surface_scene as surface_scene
from visualizer.src.scene.geographic_layers import (
    annotate_geographic_layers,
    partition_geographic_layers,
)
from visualizer.src.scene.visibility_policy import (
    effective_entry_label_visibility,
    effective_entry_visibility,
)
from visualizer.src.scene.visual_surfaces import apply_visual_surfaces
from visualizer.src.services.geographic_layer_service import GeographicLayerService
from visualizer.src.services.session_service import SessionService
from visualizer.src.services.source_edit_policy import visual_surface_source_edit_message


def _metadata(scene, *, explicit=True):
    rel = scene.entry["rel_path"]
    value = {
        "artifacts": {"mesh_files": [rel]},
        "osm_config": {"barriers": True, "green_vegetation": True},
        "visual_building_faces": {rel: {"building_triangles": 6, "total_triangles": 8}},
    }
    if explicit:
        value["visual_geographic_layers"] = {
            rel: {
                "total_triangles": 8,
                "source_sha256": sha256(scene.mesh_path.read_bytes()).hexdigest(),
                "layers": {"buildings": [[0, 6]], "barriers": [[6, 2]]},
            }
        }
    return value


def _classify(scene, metadata):
    (scene.directory / "osm_metadata.json").write_text(json.dumps(metadata))
    entries = [deepcopy(scene.entry)]
    annotate_geographic_layers(entries, str(scene.xml_path))
    return partition_geographic_layers(apply_visual_surfaces(entries, str(scene.xml_path)))


def _triangles(entries):
    return Counter(
        tuple(mesh.vertices[triangle].ravel())
        for entry in entries
        for mesh in [entry["mesh"]]
        for triangle in mesh.triangles
    )


def test_partition_preserves_all_corners_uvs_and_authored_facade(surface_scene):
    scene = surface_scene
    scene.sidecar.write_text(json.dumps(scene.document))
    original = scene.mesh_path.read_bytes(), scene.xml_path.read_bytes()
    entries = _classify(scene, _metadata(scene))
    assert _triangles(entries) == _triangles([scene.entry])
    assert len({entry["object_id"] for entry in entries}) == len(entries) == 3
    facade = next(entry for entry in entries if entry.get("_visual_surface_id") == "front")
    assert facade["geographic_layer"] == "buildings"
    assert facade["texture_path"].endswith("front.png")
    np.testing.assert_array_equal(facade["mesh"].triangle_uvs, scene.surface["triangle_uvs"])
    barrier = next(entry for entry in entries if entry["geographic_layer"] == "barriers")
    assert "_osm_building_faces" not in barrier
    assert visual_surface_source_edit_message(barrier)
    assert (scene.mesh_path.read_bytes(), scene.xml_path.read_bytes()) == original


@pytest.mark.parametrize("damage", ["hash", "count", "overlap", "bounds"])
def test_stale_or_bad_metadata_keeps_original_mesh(surface_scene, damage):
    metadata = _metadata(surface_scene)
    record = next(iter(metadata["visual_geographic_layers"].values()))
    if damage == "hash":
        record["source_sha256"] = "0" * 64
    if damage == "count":
        record["total_triangles"] = 9
    if damage == "overlap":
        record["layers"]["barriers"] = [[5, 2]]
    if damage == "bounds":
        record["layers"]["barriers"] = [[7, 2]]
    entries = _classify(surface_scene, metadata)
    assert len(entries) == 1 and "geographic_layer" not in entries[0]
    assert _triangles(entries) == _triangles([surface_scene.entry])


@pytest.mark.parametrize(
    ("material", "green", "tail"),
    [
        ("concrete", True, "barriers"),
        ("metal", True, "barriers"),
        ("wood", False, None),
        ("wood", True, "barriers"),
    ],
)
def test_legacy_building_prefix_never_misclassifies_mixed_wood(
    surface_scene, material, green, tail
):
    scene = surface_scene
    scene.entry["rel_path"] = f"mesh/buildings_{material}.ply"
    metadata = _metadata(scene, explicit=False)
    metadata["osm_config"]["green_vegetation"] = green
    entries = _classify(scene, metadata)
    assert [entry.get("geographic_layer") for entry in entries] == (
        [None, "buildings"] if tail is None else ["buildings", tail]
    )
    assert _triangles(entries) == _triangles([scene.entry])


def test_switch_preserves_manual_visibility_and_workspace_state(qapp):
    from visualizer.src.panels.object_panel import ObjectManagementPanel

    entries = [
        {"name": "wall", "geographic_layer": "barriers", "visible": True, "show_label": True},
        {"name": "house", "geographic_layer": "buildings", "visible": False},
    ]
    publish = Mock()
    viz = SimpleNamespace(
        mesh_entries=entries,
        object_appearance_service=SimpleNamespace(refresh_object_visibility_batch=publish),
    )
    service = GeographicLayerService(viz)
    viz.scene_service = SimpleNamespace(geographic_layers=service)
    service.set_scene()
    panel = ObjectManagementPanel(viz)
    widget = panel.create_panel()
    viz.ui_manager = SimpleNamespace(panels={"objects": panel})
    panel.sync_geographic_layers()
    assert not panel.widgets["geographic_layers"].isHidden()
    assert not panel.widgets["geographic_layers_toggle"].isChecked()
    assert panel.widgets["geo_vegetation"].isHidden()
    panel.widgets["geo_barriers"].click()
    assert entries[0]["visible"] is True
    assert not effective_entry_visibility(entries[0])
    assert not effective_entry_label_visibility(entries[0])
    publish.assert_called_once_with([entries[0]])
    session = SessionService(viz)
    saved = session._get_rendering_state()
    assert saved["geographic_layers"]["barriers"] is False
    service.restore({})
    assert effective_entry_visibility(entries[0])
    session._restore_rendering_state(saved)
    assert not effective_entry_visibility(entries[0])
    assert not effective_entry_visibility(entries[1])
    service.set_scene()
    assert effective_entry_visibility(entries[0])
    viz.mesh_entries = []
    service.set_scene()
    panel.sync_geographic_layers()
    assert panel.widgets["geographic_layers"].isHidden()
    widget.deleteLater()


def test_old_workspace_source_intent_applies_to_each_partition():
    entries = [
        {"stable_mesh_id": f"city_geo_{kind}", "_geographic_source_mesh_id": "city"}
        for kind in ("buildings", "barriers")
    ]
    service = SessionService(SimpleNamespace(mesh_entries=entries))
    service._restore_entry_state({"scene:city": {"visible": False}})
    assert all(entry["visible"] is False for entry in entries)


def test_renderer_gate_composes_with_material_and_manual_visibility():
    from tests.visualizer.unit.test_object_appearance_service import _scene_state, _viz
    from visualizer.src.materials.appearance import MaterialDisplayMode
    from visualizer.src.services.material_modes import MaterialModeService

    entry = {
        "name": "wall",
        "mesh": _scene_state("wall"),
        "entry_type": "mesh",
        "visible": True,
        "material_id": "concrete",
        "_geographic_layer_visible": False,
    }
    sync = Mock(return_value=True)
    viz = _viz(
        mesh_entries=[entry], scene_service=SimpleNamespace(sync_scene_entry_visibility_batch=sync)
    )
    appearance = viz.object_appearance_service
    appearance.refresh_object_visibility_batch([entry])
    assert sync.call_args.args[0] == [(entry, False)]
    assert entry["visible"] is True
    entry["_geographic_layer_visible"] = True
    entry["_global_visible"] = False
    assert not appearance.resolve_entry_appearance(entry).visible
    entry["_global_visible"] = True
    viz.material_mode_service = MaterialModeService()
    viz.material_mode_service.set_mode("concrete", MaterialDisplayMode.HIDDEN)
    assert not appearance.resolve_entry_appearance(entry).visible
    viz.material_mode_service.set_mode("concrete", MaterialDisplayMode.NORMAL)
    assert appearance.resolve_entry_appearance(entry).visible


def test_partition_buffer_identity_cannot_alias_other_category(surface_scene):
    from visualizer.src.services.scene_service import SceneService

    scene = surface_scene
    entries = _classify(scene, _metadata(scene))
    service = SceneService(SimpleNamespace(xml_path=str(scene.xml_path)))
    service.scene_path = str(scene.xml_path)
    keys = [service._pygfx_mesh_buffer_cache_key(entry, effective_props={}) for entry in entries]
    assert all(keys) and len(set(keys)) == 2
