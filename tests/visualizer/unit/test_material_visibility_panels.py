import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from visualizer.src.materials.appearance import MaterialDisplayMode
from visualizer.src.panels.materials_pbr_panel import MaterialsPanel
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.material_modes import MaterialModeService
from visualizer.src.state import create_initial_state

_APP = QApplication.instance() or QApplication([])


def _make_parent():
    parent = SimpleNamespace()
    parent.material_mode_service = MaterialModeService()
    parent.apply_calls = 0
    parent.app_state = create_initial_state()
    parent.renderer = SimpleNamespace(
        capabilities=RendererCapabilities(pbr=True, rf_xray_overlay=True)
    )

    def _apply(_material_key):
        parent.apply_calls += 1

    parent.ui_controller = SimpleNamespace(apply_material_modes=_apply)
    return parent


def test_materials_panel_hide_toggles_back_to_normal():
    parent = _make_parent()
    panel = MaterialsPanel(parent)
    group = panel.create_panel()

    combo = panel.widgets["material_combo"]
    combo.addItem("brick")
    combo.setCurrentText("brick")

    panel._on_visibility_changed(MaterialDisplayMode.HIDDEN)
    assert parent.material_mode_service.get_mode("brick") is MaterialDisplayMode.HIDDEN
    assert panel.widgets["hide_btn"].isChecked() is True
    assert panel.widgets["highlight_btn"].isChecked() is False
    assert "normal_btn" not in panel.widgets

    panel._on_visibility_changed(MaterialDisplayMode.HIDDEN)
    assert parent.material_mode_service.get_mode("brick") is MaterialDisplayMode.NORMAL
    assert panel.widgets["hide_btn"].isChecked() is False
    assert panel.widgets["highlight_btn"].isChecked() is False
    assert parent.apply_calls == 2
    assert group.title() == "Materials"


def test_materials_panel_highlight_toggles_back_to_normal():
    parent = _make_parent()
    panel = MaterialsPanel(parent)
    group = panel.create_panel()

    combo = panel.widgets["material_combo"]
    combo.addItem("metal")
    combo.setCurrentText("metal")

    panel._on_visibility_changed(MaterialDisplayMode.HIGHLIGHTED)
    assert parent.material_mode_service.get_mode("metal") is MaterialDisplayMode.HIGHLIGHTED
    assert panel.widgets["highlight_btn"].isChecked() is True
    assert panel.widgets["hide_btn"].isChecked() is False
    assert "normal_btn" not in panel.widgets

    panel._on_visibility_changed(MaterialDisplayMode.HIGHLIGHTED)
    assert parent.material_mode_service.get_mode("metal") is MaterialDisplayMode.NORMAL
    assert panel.widgets["highlight_btn"].isChecked() is False
    assert panel.widgets["hide_btn"].isChecked() is False
    assert parent.apply_calls == 2
    assert group.title() == "Materials"


def test_materials_panel_display_toggles_are_mutually_exclusive():
    parent = _make_parent()
    panel = MaterialsPanel(parent)
    group = panel.create_panel()

    combo = panel.widgets["material_combo"]
    combo.addItem("glass")
    combo.setCurrentText("glass")

    panel._on_visibility_changed(MaterialDisplayMode.HIDDEN)
    panel._on_visibility_changed(MaterialDisplayMode.HIGHLIGHTED)

    assert parent.material_mode_service.get_mode("glass") is MaterialDisplayMode.HIGHLIGHTED
    assert panel.widgets["hide_btn"].isChecked() is False
    assert panel.widgets["highlight_btn"].isChecked() is True
    assert group.title() == "Materials"


def test_materials_panel_rf_xray_controls_include_property_and_no_bounces():
    parent = _make_parent()
    parent.app_state = create_initial_state(rf_xray_mode="material_properties")
    panel = MaterialsPanel(parent)
    materials_group = panel.create_panel()
    assert "rf_xray_mode_combo" not in panel.widgets

    group = panel.create_rf_xray_panel()

    mode_combo = panel.widgets["rf_xray_mode_combo"]
    property_combo = panel.widgets["rf_xray_property_combo"]

    assert mode_combo.findData("material_properties") >= 0
    assert property_combo.findData("scattering_coefficient") >= 0
    assert property_combo.isEnabled()
    assert "rf_xray_bounces_cb" not in panel.widgets

    group.deleteLater()
    materials_group.deleteLater()
