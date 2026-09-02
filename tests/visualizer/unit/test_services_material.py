from __future__ import annotations

from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np

from visualizer.src.materials.appearance import MaterialDisplayMode
from visualizer.src.services.material_entry_editor import MaterialEntryEditService
from visualizer.src.services.material_mode_commands import MaterialModeCommandService
from visualizer.src.services.material_modes import MaterialModeService
from visualizer.src.services.material_properties import (
    sync_entry_pbr_properties_from_catalog,
)


def test_register_and_get_mode_defaults():
    service = MaterialModeService()
    service.register_materials(["mat-1", "mat-2"])
    assert service.get_mode("mat-1") is MaterialDisplayMode.NORMAL
    assert service.get_mode("unknown") is MaterialDisplayMode.NORMAL


def test_set_mode_validates():
    service = MaterialModeService()
    service.set_mode("mat-1", MaterialDisplayMode.HIDDEN)
    assert service.get_mode("mat-1") is MaterialDisplayMode.HIDDEN


def test_set_mode_invalid():
    service = MaterialModeService()
    try:
        service.set_mode("mat-1", "foo")
    except ValueError as exc:
        assert "Invalid material display mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_resolve_toggled_mode_for_hide_and_highlight():
    service = MaterialModeService()

    assert (
        service.resolve_toggled_mode("mat-1", MaterialDisplayMode.HIDDEN)
        is MaterialDisplayMode.HIDDEN
    )
    service.set_mode("mat-1", MaterialDisplayMode.HIDDEN)
    assert (
        service.resolve_toggled_mode("mat-1", MaterialDisplayMode.HIDDEN)
        is MaterialDisplayMode.NORMAL
    )

    service.set_mode("mat-1", MaterialDisplayMode.NORMAL)
    assert (
        service.resolve_toggled_mode("mat-1", MaterialDisplayMode.HIGHLIGHTED)
        is MaterialDisplayMode.HIGHLIGHTED
    )
    service.set_mode("mat-1", MaterialDisplayMode.HIGHLIGHTED)
    assert (
        service.resolve_toggled_mode("mat-1", MaterialDisplayMode.HIGHLIGHTED)
        is MaterialDisplayMode.NORMAL
    )

    assert (
        service.resolve_toggled_mode("mat-1", MaterialDisplayMode.NORMAL)
        is MaterialDisplayMode.NORMAL
    )


def test_apply_material_modes_republishes_scene_and_targets_without_mutating_manual_state():
    modes = MaterialModeService()
    service = MaterialModeCommandService(modes)
    modes.set_mode("scene-mat", MaterialDisplayMode.HIDDEN)
    modes.set_mode("mat-itu_glass_Walker", MaterialDisplayMode.HIDDEN)

    scene_entry = {"material_id": "scene-mat", "material_type": "brick"}
    target_entry = {"material_id": "mat-itu_glass_Walker", "material_type": "glass"}
    scene_entry.update(visible=False, highlighted=True)
    target_entry.update(visible=True, highlighted=False)
    batches = []

    def _refresh(entries, *, materials_changed=True, update_renderer=True):
        batches.append((list(entries), materials_changed, update_renderer))
        return True

    service.apply_material_modes(
        [scene_entry],
        [target_entry],
        _refresh,
        update_renderer=True,
    )

    assert batches == [([scene_entry, target_entry], False, True)]
    assert scene_entry["visible"] is False
    assert scene_entry["highlighted"] is True
    assert target_entry["visible"] is True
    assert target_entry["highlighted"] is False


def test_apply_material_modes_registers_material_id_and_type_keys():
    modes = MaterialModeService()
    service = MaterialModeCommandService(modes)
    modes.set_mode("brick", MaterialDisplayMode.HIDDEN)

    entry = {
        "material_id": "mat-itu_brick",
        "material_type": "brick",
        "highlighted": True,
    }
    calls = []

    service.apply_material_modes(
        [entry],
        [],
        lambda entries, *, materials_changed=True, update_renderer=True: calls.append(
            (list(entries), materials_changed, update_renderer)
        )
        or True,
        update_renderer=False,
    )

    assert calls == [([entry], False, False)]
    assert modes.get_mode("mat-itu_brick") is MaterialDisplayMode.NORMAL
    assert modes.get_mode("brick") is MaterialDisplayMode.HIDDEN
    assert entry["highlighted"] is True


def test_apply_material_modes_uses_one_batch_callback_for_many_entries():
    modes = MaterialModeService()
    service = MaterialModeCommandService(modes)
    modes.set_mode("brick", MaterialDisplayMode.HIDDEN)

    mesh_entries = [
        {"material_id": "brick_a", "material_type": "brick", "mesh": object()},
        {"material_id": "brick_b", "material_type": "brick", "mesh": object()},
    ]

    calls = []

    service.apply_material_modes(
        mesh_entries,
        [],
        lambda entries, *, materials_changed=True, update_renderer=True: calls.append(
            (list(entries), materials_changed)
        )
        or True,
    )

    assert calls == [(mesh_entries, False)]


def test_apply_material_modes_republishes_only_the_changed_material_key():
    modes = MaterialModeService()
    service = MaterialModeCommandService(modes)
    brick = {"material_id": "brick_a", "material_type": "brick"}
    glass = {"material_id": "glass_a", "material_type": "glass"}
    calls = []

    service.apply_material_modes(
        [brick, glass],
        [],
        lambda entries, **kwargs: calls.append((list(entries), kwargs)) or True,
        material_key="brick",
    )

    assert calls == [
        (
            [brick],
            {"materials_changed": False, "update_renderer": True},
        )
    ]


def test_apply_material_modes_uses_em_independent_visual_key():
    modes = MaterialModeService()
    service = MaterialModeCommandService(modes)
    entry = {"material_id": "mat-itu_glass", "material_type": "glass"}
    calls = []

    service.apply_material_modes(
        [entry],
        [],
        lambda entries, **_kwargs: calls.append(list(entries)) or True,
        material_key="manual-brick",
        visual_material_key=lambda _entry: "manual-brick",
    )

    assert calls == [[entry]]
    assert modes.get_mode("manual-brick") is MaterialDisplayMode.NORMAL


def test_update_material_color_updates_matching_entries_and_continues_after_xml_error(
    monkeypatch,
):
    service = MaterialEntryEditService()
    xml_a = ET.Element("bsdf", id="a")
    xml_b = ET.Element("bsdf", id="b")
    entry = {"name": "Wall A", "material_id": "mat-a", "xml_bsdf": xml_a}
    sibling = {"name": "Wall B", "material_id": "mat-a", "xml_bsdf": xml_b}
    unrelated = {"name": "Wall C", "material_id": "mat-c", "xml_bsdf": None}
    calls = []

    def _update_xml(xml_bsdf, color):
        calls.append((xml_bsdf, list(color)))
        if xml_bsdf is xml_a:
            raise ValueError("bad xml")

    monkeypatch.setattr(
        "visualizer.src.services.material_entry_editor.MaterialHandler.update_material_color",
        _update_xml,
    )

    updated = service.update_material_color(
        entry,
        [0.1, 0.2, 0.3],
        [entry, sibling, unrelated],
        [],
    )

    assert updated == [entry, sibling]
    assert entry["color"] == [0.1, 0.2, 0.3]
    assert sibling["color"] == [0.1, 0.2, 0.3]
    assert "color" not in unrelated
    assert calls == [(xml_a, [0.1, 0.2, 0.3]), (xml_b, [0.1, 0.2, 0.3])]


def test_change_material_id_exact_bsdf_updates_entry_shape_ref_and_pbr_props():
    service = MaterialEntryEditService()
    root = ET.fromstring("""
        <scene>
          <bsdf id="mat-itu_concrete" type="diffuse">
            <rgb name="reflectance" value="0.1 0.2 0.3" />
          </bsdf>
        </scene>
        """)
    shape = ET.fromstring('<shape><ref name="bsdf" id="old" /></shape>')
    entry = {
        "name": "Wall",
        "entry_type": "mesh",
        "material_id": "old",
        "color": [0.7, 0.7, 0.7],
        "xml_shape": shape,
    }

    actual_id, color, bsdf = service.change_material_id(
        entry,
        "mat-itu_concrete",
        root,
        [entry],
        [],
    )

    assert actual_id == "mat-itu_concrete"
    assert color == [0.1, 0.2, 0.3]
    assert bsdf is root.find("bsdf")
    assert entry["material_id"] == "mat-itu_concrete"
    assert entry["material_type"] == "concrete"
    assert entry["pbr_properties"]["roughness"] == 0.8
    assert shape.find("ref").get("id") == "mat-itu_concrete"


def test_change_material_id_normalized_bsdf_uses_actual_id():
    service = MaterialEntryEditService()
    root = ET.fromstring("""
        <scene>
          <bsdf id="mat-itu_glass" type="diffuse">
            <rgb name="color" value="0.3 0.4 0.5" />
          </bsdf>
        </scene>
        """)
    entry = {"name": "Window", "entry_type": "mesh", "material_id": "old"}

    actual_id, color, bsdf = service.change_material_id(
        entry,
        "mat_itu_glass",
        root,
        [entry],
        [],
    )

    assert actual_id == "mat-itu_glass"
    assert color == [0.3, 0.4, 0.5]
    assert bsdf is root.find("bsdf")
    assert entry["material_type"] == "glass"


def test_change_material_id_falls_back_to_other_entries_when_xml_missing():
    service = MaterialEntryEditService()
    root = ET.Element("scene")
    fallback_bsdf = ET.Element("bsdf", id="mat-custom")
    entry = {"name": "Wall", "entry_type": "mesh", "material_id": "old"}
    other = {
        "name": "Other Wall",
        "entry_type": "mesh",
        "material_id": "mat-custom",
        "color": [0.2, 0.4, 0.6],
        "xml_bsdf": fallback_bsdf,
    }

    actual_id, color, bsdf = service.change_material_id(
        entry,
        "mat-custom",
        root,
        [entry, other],
        [],
    )

    assert actual_id == "mat-custom"
    assert color == [0.2, 0.4, 0.6]
    assert bsdf is fallback_bsdf
    assert entry["xml_bsdf"] is fallback_bsdf


def test_change_material_id_falls_back_to_mpc_core_colors():
    service = MaterialEntryEditService()
    root = ET.Element("scene")
    entry = {"name": "Wood Wall", "entry_type": "mesh", "material_id": "old"}
    mpc_core = SimpleNamespace(
        _get_material_colors=lambda: {"mat_itu_wood": np.array([0.4, 0.3, 0.2])}
    )

    actual_id, color, bsdf = service.change_material_id(
        entry,
        "mat-itu-wood",
        root,
        [entry],
        [],
        mpc_core=mpc_core,
    )

    assert actual_id == "mat-itu-wood"
    assert color == [0.4, 0.3, 0.2]
    assert bsdf is None
    assert entry["color"] == [0.4, 0.3, 0.2]


def test_sync_entry_pbr_properties_from_catalog_carries_advanced_fields() -> None:
    entry: dict = {}
    sync_entry_pbr_properties_from_catalog(entry, "marble")
    props = entry["pbr_properties"]
    assert props["clearcoat"] == 0.6
    assert props["clearcoat_roughness"] == 0.1

    entry = {}
    sync_entry_pbr_properties_from_catalog(entry, "glass")
    props = entry["pbr_properties"]
    assert props["transmission"] == 0.9
    assert props["glass_thickness"] == 0.5

    entry = {}
    sync_entry_pbr_properties_from_catalog(entry, "metal")
    props = entry["pbr_properties"]
    assert props["anisotropy"] == 0.15

    entry = {}
    sync_entry_pbr_properties_from_catalog(entry, "concrete")
    props = entry["pbr_properties"]
    assert props["clearcoat"] == 0.0
    assert props["anisotropy"] == 0.0
    assert props["transmission"] == 0.0
