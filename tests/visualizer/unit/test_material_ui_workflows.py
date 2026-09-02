"""Focused behavioral tests for visual-material UI workflows."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from visualizer.src.controllers.material_ui_controller import MaterialUIController
from visualizer.src.materials.appearance import (
    VisualMaterialBinding,
    VisualMaterialSource,
)
from visualizer.src.panels.materials_pbr_panel import MaterialsPanel
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.material_modes import MaterialModeService
from visualizer.src.state import create_initial_state

_APP = QApplication.instance() or QApplication([])


def test_object_color_edit_uses_stable_visual_material_key() -> None:
    """An EM ID change must not redirect an explicit visual-group color edit."""
    pbr_service = SimpleNamespace(
        get_visual_material_key=Mock(return_value="concrete"),
        resolve_entry_material=Mock(
            return_value=SimpleNamespace(
                properties_copy=lambda: {"color": [0.2, 0.3, 0.4]},
                texture_policy=SimpleNamespace(color_editable=True),
            )
        ),
        set_property=Mock(return_value=True),
    )
    visualizer = SimpleNamespace(
        dialog_manager=SimpleNamespace(
            pick_color=Mock(return_value=QColor.fromRgbF(0.1, 0.6, 0.9))
        ),
        material_pbr_service=pbr_service,
    )
    parent = SimpleNamespace(
        visualizer=visualizer,
        material_mode_command_service=None,
        material_entry_edit_service=None,
        populate_controls=Mock(),
    )
    entry = {"name": "Wall", "material_type": "glass"}

    MaterialUIController(parent).handle_material_color_changed(entry)

    pbr_service.get_visual_material_key.assert_called_once_with(entry)
    material_key, property_name, color = pbr_service.set_property.call_args.args
    assert material_key == "concrete"
    assert property_name == "color"
    assert color == pytest.approx([0.1, 0.6, 0.9], abs=1 / 255)
    parent.populate_controls.assert_called_once_with()


def _make_panel_parent(entry: dict, binding_ref: list[VisualMaterialBinding]) -> SimpleNamespace:
    pbr_service = SimpleNamespace(
        get_visual_material_key=lambda _entry: "concrete",
        get_visual_binding=lambda _entry: binding_ref[0],
        get_material_types_in_scene=lambda: ["concrete"],
        get_preset_properties=lambda _name: {},
    )
    return SimpleNamespace(
        material_mode_service=MaterialModeService(),
        app_state=create_initial_state(),
        renderer=SimpleNamespace(capabilities=RendererCapabilities(pbr=True, rf_xray_overlay=True)),
        ui_controller=SimpleNamespace(apply_material_modes=lambda _key: None),
        material_pbr_service=pbr_service,
        mesh_entries=[entry],
        target_entries=[],
    )


def test_profile_display_is_derived_from_active_visual_binding() -> None:
    """Profile text and Clear visibility follow binding state, not compiled rules."""
    entry = {"name": "Wall", "material_type": "glass"}
    binding_ref = [
        VisualMaterialBinding(
            source=VisualMaterialSource.PROFILE,
            material_type="concrete",
            preset="Gold",
        )
    ]
    parent = _make_panel_parent(entry, binding_ref)
    panel = MaterialsPanel(parent)
    group = panel.create_panel()

    panel._update_profile_info("concrete")
    assert panel.widgets["em_material_label"].text() == "glass"
    assert panel.widgets["visual_preset_label"].text() == "Gold (profile)"
    assert panel.widgets["clear_profile_btn"].isHidden() is False

    binding_ref[0] = VisualMaterialBinding(
        source=VisualMaterialSource.MANUAL,
        material_type="concrete",
        preset="Brick",
    )
    panel._update_profile_info("concrete")
    assert panel.widgets["visual_preset_label"].text() == "Brick (manual)"

    binding_ref[0] = VisualMaterialBinding()
    panel._update_profile_info("concrete")
    assert panel.widgets["visual_preset_label"].text() == "Follow EM"
    assert panel.widgets["clear_profile_btn"].isHidden() is True
    group.deleteLater()


def test_profile_display_reports_mixed_visual_bindings() -> None:
    """A grouped panel row must not hide a profile behind a Follow EM representative."""
    follow_entry = {"name": "Wall A", "material_type": "glass"}
    profile_entry = {"name": "Wall B", "material_type": "brick"}
    bindings = {
        id(follow_entry): VisualMaterialBinding(),
        id(profile_entry): VisualMaterialBinding(
            source=VisualMaterialSource.PROFILE,
            material_type="concrete",
            preset="Gold",
        ),
    }
    parent = _make_panel_parent(follow_entry, [VisualMaterialBinding()])
    parent.mesh_entries = [follow_entry, profile_entry]
    parent.material_pbr_service.get_visual_binding = lambda entry: bindings[id(entry)]
    panel = MaterialsPanel(parent)
    group = panel.create_panel()

    panel._update_profile_info("concrete")

    assert panel.widgets["em_material_label"].text() == "Mixed"
    assert panel.widgets["visual_preset_label"].text() == "Gold (profile) + Follow EM"
    assert panel.widgets["clear_profile_btn"].isHidden() is False
    group.deleteLater()


def test_clear_profile_button_uses_clear_visual_assignment() -> None:
    """The Clear button invokes assignment clearing rather than override reset."""
    entry = {"name": "Wall", "material_type": "concrete"}
    binding_ref = [
        VisualMaterialBinding(
            source=VisualMaterialSource.PROFILE,
            material_type="concrete",
            preset="Gold",
        )
    ]
    parent = _make_panel_parent(entry, binding_ref)

    def clear_assignment(material_type: str) -> bool:
        assert material_type == "concrete"
        binding_ref[0] = VisualMaterialBinding()
        return True

    parent.material_pbr_service.clear_visual_assignment = Mock(side_effect=clear_assignment)
    panel = MaterialsPanel(parent)
    group = panel.create_panel()
    material_combo = panel.widgets["material_combo"]
    material_combo.blockSignals(True)
    material_combo.addItem("concrete")
    material_combo.blockSignals(False)
    panel.refresh_material_list = Mock()

    panel._on_clear_profile()

    parent.material_pbr_service.clear_visual_assignment.assert_called_once_with("concrete")
    panel.refresh_material_list.assert_called_once_with()
    group.deleteLater()
