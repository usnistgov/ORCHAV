"""Reversible in-app texture opt-in without changes to launch or scientific state."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from visualizer.src.materials import texture_policy
from visualizer.src.model import RenderObjectState
from visualizer.src.panels.materials_pbr_panel import MaterialsPanel
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.material_modes import MaterialModeService
from visualizer.src.services.material_pbr_service import MaterialPBRService
from visualizer.src.state import create_initial_state
from visualizer.src.types.render_payloads import MeshPayload, SurfaceColorSource

_APP = QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def clean_texture_choice(monkeypatch):
    monkeypatch.setattr(texture_policy, "_TEXTURE_MAPS_OVERRIDE", None)
    monkeypatch.delenv("ORCHAV_ENABLE_TEXTURES", raising=False)
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)


@pytest.fixture
def viz(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    parent = SimpleNamespace(
        renderer=SimpleNamespace(capabilities=RendererCapabilities(pbr=True)),
        mesh_entries=[],
        target_entries=[],
        object_appearance_service=SimpleNamespace(
            refresh_entry_appearance_batch=Mock(return_value=True)
        ),
        material_mode_service=MaterialModeService(),
        app_state=create_initial_state(),
    )
    parent.material_pbr_service = MaterialPBRService(parent)
    return parent


@pytest.mark.parametrize("launch_enabled", [False, True])
def test_on_off_reset_retains_launch_environment(monkeypatch, launch_enabled):
    if launch_enabled:
        monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    before = dict(texture_policy.os.environ)
    for choice in [True, True, False, False, True, None, False, None]:
        assert texture_policy.set_texture_maps_override(choice)
        assert texture_policy.textures_globally_enabled() is (
            launch_enabled if choice is None else choice
        )
    assert dict(texture_policy.os.environ) == before


def test_launch_opt_out_cannot_be_overridden(monkeypatch):
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    assert not texture_policy.set_texture_maps_override(True)
    assert not texture_policy.textures_globally_enabled()
    assert texture_policy.texture_maps_override() is None


def test_new_launch_policy_clears_session_choice():
    texture_policy.set_texture_maps_override(True)
    texture_policy.apply_texture_launch_policy()
    assert texture_policy.texture_maps_override() is None
    assert not texture_policy.textures_globally_enabled()


@pytest.mark.parametrize("material", ["concrete", "metal", "glass"])
def test_maps_round_trip_without_destroying_authored_values(viz, tmp_path, material):
    albedo = tmp_path / "albedo.png"
    albedo.write_bytes(b"map")
    normal = tmp_path / "normal.png"
    normal.write_bytes(b"map")
    authored = {
        "texture_path": str(albedo),
        "normal_map_path": str(normal),
        "color": [0.2, 0.3, 0.4],
        "roughness": 0.42,
    }
    entry = {"name": "Wall", "material_type": material, "pbr_properties": authored.copy()}
    target = {"name": "Target", "entry_type": "target", "material_type": material}
    viz.mesh_entries = [entry]
    viz.target_entries = [target]
    service = viz.material_pbr_service
    for choice in [True, False, True, False, None]:
        assert service.set_texture_maps_enabled(choice)
        resolved = service.resolve_entry_material(entry)
        assert resolved.texture_policy.has_active_maps is (choice is True)
        assert resolved.texture_policy.color_editable is (choice is not True)
        assert resolved.payload.roughness == pytest.approx(0.42)
        assert resolved.payload.base_color[:3] == ((1.0, 1.0, 1.0) if choice else (0.2, 0.3, 0.4))
        assert entry["pbr_properties"] == authored
        assert service.overrides == {}
    refresh = viz.object_appearance_service.refresh_entry_appearance_batch
    assert refresh.call_count == 4
    assert refresh.call_args.args[0] == [entry, target]


@pytest.mark.parametrize("failure", [False, RuntimeError("renderer unavailable")])
def test_retry_after_incomplete_publication(viz, failure):
    viz.mesh_entries = [{"name": "Wall", "material_type": "concrete"}]
    refresh = viz.object_appearance_service.refresh_entry_appearance_batch
    refresh.side_effect = [failure, True]
    assert not viz.material_pbr_service.set_texture_maps_enabled(True)
    assert viz.material_pbr_service.set_texture_maps_enabled(True)
    assert refresh.call_count == 2
    assert viz.material_pbr_service.set_texture_maps_enabled(True)
    assert refresh.call_count == 2


def test_vertex_and_scientific_overlay_colors_are_not_replaced(viz, tmp_path):
    albedo = tmp_path / "albedo.png"
    albedo.write_bytes(b"map")
    colors = np.asarray([[0.1, 0.3, 0.9], [0.2, 0.8, 0.2], [0.9, 0.2, 0.1]])
    payload = MeshPayload(
        vertices=np.eye(3),
        triangles=np.asarray([[0, 1, 2]]),
        vertex_colors=colors,
        color_source=SurfaceColorSource.VERTEX,
    )
    state = RenderObjectState(id="scene:wall::mesh", payload=payload)
    entry = {
        "name": "Wall",
        "mesh": state,
        "material_type": "concrete",
        "pbr_properties": {"texture_path": str(albedo)},
    }
    overlay = RenderObjectState(id="coverage:metric", payload=replace(payload))
    viz.mesh_entries = [entry]
    viz.coverage_mesh = overlay
    viz.color_by = "received_power"
    viz.rf_xray_enabled = True
    for enabled in [True, False, True, None]:
        assert viz.material_pbr_service.set_texture_maps_enabled(enabled)
        resolved = viz.material_pbr_service.resolve_entry_material(entry)
        assert resolved.payload.texture_path is None
        np.testing.assert_array_equal(state.payload.vertex_colors, colors)
        assert viz.coverage_mesh is overlay
        np.testing.assert_array_equal(overlay.payload.vertex_colors, colors)
        assert viz.color_by == "received_power"
        assert viz.rf_xray_enabled
    assert viz.object_appearance_service.refresh_entry_appearance_batch.call_args.args == ([entry],)


def test_panel_repeated_switches_reset_and_recreation(viz):
    panel = MaterialsPanel(viz)
    group = panel.create_panel()
    panel.update_ui_from_viz()
    combo = panel.widgets["texture_maps"]
    for choice in ["on", "off", "on", "default"]:
        combo.setCurrentIndex(combo.findData(choice))
        assert texture_policy.textures_globally_enabled() is (choice == "on")
    combo.setCurrentIndex(combo.findData("on"))
    group.close()
    replacement = MaterialsPanel(viz)
    replacement_group = replacement.create_panel()
    replacement.update_ui_from_viz()
    assert replacement.widgets["texture_maps"].currentData() == "on"
    replacement.widgets["texture_maps"].setCurrentIndex(0)
    assert texture_policy.texture_maps_override() is None
    assert not texture_policy.textures_globally_enabled()
    group.deleteLater()
    replacement_group.deleteLater()


def test_panel_hard_opt_out_and_non_pbr_disable_control(viz, monkeypatch):
    panel = MaterialsPanel(viz)
    group = panel.create_panel()
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    panel.update_ui_from_viz()
    assert not panel.widgets["texture_maps"].isEnabled()
    assert "Locked off" in panel.widgets["texture_maps_status"].text()
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES")
    viz.renderer.capabilities = RendererCapabilities(pbr=False)
    panel.update_ui_from_viz()
    assert not panel.widgets["texture_maps"].isEnabled()
    group.deleteLater()


@pytest.mark.parametrize("auto_uvs", [True, False])
def test_first_in_app_enable_discovers_material_id_maps(viz, tmp_path, monkeypatch, auto_uvs):
    from visualizer.src.services import scene_service

    library = tmp_path / "textures"
    library.mkdir()
    albedo = library / "local_brick.png"
    albedo.write_bytes(b"map")
    monkeypatch.setattr(scene_service, "TEXTURE_LIBRARY_PATH", library)
    viz.scene_service = scene_service.SceneService(viz)
    viz.scenario = SimpleNamespace(view_defaults={"auto_generate_uvs": auto_uvs})
    entry = {
        "name": "Wall",
        "material_id": "mat-local_brick",
        "material_type": "wood",
        "_resolved_texture_maps": None,
    }
    viz.mesh_entries = [entry]
    service = viz.material_pbr_service
    assert service.resolve_entry_material(entry).texture_policy.active_albedo_path is None
    for enabled in [True, False, True, None]:
        assert service.set_texture_maps_enabled(enabled)
        active = service.resolve_entry_material(entry).texture_policy.active_albedo_path
        assert active == (str(albedo) if enabled and auto_uvs else None)
    assert entry.get("texture_path") is None
    assert service.overrides == {}


@pytest.mark.parametrize("failure", [False, RuntimeError("renderer unavailable")])
def test_panel_failed_choice_can_be_retried_without_switching(viz, failure):
    viz.mesh_entries = [{"name": "Wall", "material_type": "concrete"}]
    refresh = viz.object_appearance_service.refresh_entry_appearance_batch
    refresh.side_effect = [failure, True]
    panel = MaterialsPanel(viz)
    group = panel.create_panel()
    combo = panel.widgets["texture_maps"]
    combo.setCurrentIndex(combo.findData("on"))
    assert combo.currentData() == "on"
    assert not panel.widgets["texture_maps_retry"].isHidden()
    assert "incomplete" in panel.widgets["texture_maps_status"].text()
    panel.update_ui_from_viz()
    assert "incomplete" in panel.widgets["texture_maps_status"].text()
    replacement = MaterialsPanel(viz)
    replacement_group = replacement.create_panel()
    assert not replacement.widgets["texture_maps_retry"].isHidden()
    replacement.widgets["texture_maps_retry"].click()
    assert replacement.widgets["texture_maps"].currentData() == "on"
    assert refresh.call_count == 2
    assert replacement.widgets["texture_maps_retry"].isHidden()
    assert replacement.widgets["texture_maps_status"].text() == "Active"
    assert not viz.material_pbr_service.texture_maps_refresh_pending
    group.deleteLater()
    replacement_group.deleteLater()
