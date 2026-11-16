"""Surface controls keep saved values and callbacks while sharing narrow rows."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QCheckBox, QVBoxLayout, QWidget

from visualizer.src.materials import texture_policy
from visualizer.src.panels.appearance_panel import AppearancePanel
from visualizer.src.panels.materials_pbr_panel import MaterialsPanel
from visualizer.src.renderers.protocol import RendererCapabilities


class _AppearanceService(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.active_name = "Custom"
        self.previous = None
        self.notice = ""
        self.applying = False
        self.surface = "original"
        self.unlit = False
        self.restored = []
        self.mark_if_custom = Mock()

    def names(self):
        return []

    def capture(self):
        return {"surface": self.surface, "unlit": self.unlit, "wireframe": False}

    def restore(self, data):
        self.restored.append(data)
        self.surface, self.unlit = data["surface"], data["unlit"]
        self.changed.emit()


@pytest.fixture
def controls(qtbot, monkeypatch):
    monkeypatch.setattr(texture_policy, "_TEXTURE_MAPS_OVERRIDE", True)
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    viz = SimpleNamespace(
        renderer=SimpleNamespace(
            capabilities=RendererCapabilities(
                pbr=True, screen_space_labels=True, label_callouts=True
            ),
            get_wireframe=lambda: False,
            get_bloom_settings=lambda: {"enabled": False, "bloom_strength": 0.1},
        ),
        mesh_entries=[
            {
                "_osm_building_faces": 12,
                "_osm_building_parts": [(0, 12)],
                "_osm_road_segments": [{}],
                "_visual_surface_partition": {},
            }
        ],
        target_entries=[],
        building_appearance="varied",
        facade_theme="haussmann",
        road_appearance="marked",
        appearance_preset_service=_AppearanceService(),
        demo_service=_AppearanceService(),
        scene_service=SimpleNamespace(
            street_names=SimpleNamespace(
                enabled=True,
                available=True,
                style_mode="card",
                set_style=Mock(),
                set_enabled=Mock(),
            )
        ),
    )
    materials = MaterialsPanel(viz)
    materials_group = materials.create_panel()
    qtbot.addWidget(materials_group)
    render = SimpleNamespace(
        widgets={"texture_mipmaps_cb": QCheckBox("Texture mipmaps")},
        _sync_from_visualizer=Mock(),
    )
    section = QWidget()
    section_layout = QVBoxLayout(section)
    surface_panel = QWidget()
    QVBoxLayout(surface_panel)
    section_layout.addWidget(surface_panel)
    section.content_layout = lambda: section_layout
    lighting = QWidget()
    qtbot.addWidget(section)
    qtbot.addWidget(lighting)
    manager = SimpleNamespace(
        panels={"materials": materials, "render": render},
        sections={"scene_style": section, "lighting": lighting},
    )
    appearance = AppearancePanel(viz, manager)
    group = appearance.create_panel()
    qtbot.addWidget(group)
    appearance.connect_customization()
    section.resize(380, 600)
    section.show()
    yield viz, materials, appearance, render, section
    appearance.cleanup()


def test_surface_color_and_unlit_edits_keep_semantic_values(controls):
    viz, _materials, appearance, _render, _section = controls
    assert appearance.widgets["surface_colors_label"].buddy() is appearance.surface
    appearance.surface.setCurrentIndex(appearance.surface.findData("radio"))
    appearance.surface.activated.emit(appearance.surface.currentIndex())
    assert viz.appearance_preset_service.restored[-1] == {
        "surface": "radio",
        "unlit": False,
        "wireframe": False,
    }
    appearance.unlit.click()
    assert viz.appearance_preset_service.restored[-1]["unlit"] is True


def test_demo_changes_refresh_render_controls_without_customizing_the_saved_look(controls):
    viz, _materials, appearance, render, _section = controls
    render._sync_from_visualizer.reset_mock()
    viz.demo_service.changed.emit()
    render._sync_from_visualizer.assert_called_once_with()
    viz.appearance_preset_service.mark_if_custom.assert_not_called()
    assert not appearance._edit_timer.isActive()


def test_building_theme_restore_and_availability_survive_new_labels(controls):
    viz, materials, _appearance, _render, _section = controls
    widgets = materials.widgets
    assert widgets["building_appearance"].currentData() == "varied"
    assert widgets["facade_theme"].currentData() == "haussmann"
    assert widgets["road_appearance"].currentData() == "marked"
    assert not widgets["facade_theme"].isHidden()
    viz.building_appearance = "procedural"
    materials.sync_procedural_controls()
    assert widgets["building_appearance"].currentData() == "procedural"
    assert widgets["facade_theme"].isHidden()
    viz.mesh_entries[0].pop("_osm_building_parts")
    materials.sync_procedural_controls()
    combo = widgets["building_appearance"]
    assert not combo.model().item(combo.findData("varied")).isEnabled()


def test_mipmaps_reuses_one_control_beside_texture_choice(controls):
    _viz, materials, _appearance, render, _section = controls
    mipmaps = render.widgets["texture_mipmaps_cb"]
    toggled = Mock()
    mipmaps.toggled.connect(toggled)
    assert materials.texture_maps_row.indexOf(mipmaps) >= 0
    assert mipmaps.parentWidget() is materials.surface_controls
    mipmaps.click()
    toggled.assert_called_once_with(True)


def test_related_controls_fit_380_pixel_panel_without_overlap(controls, qapp):
    _viz, materials, appearance, render, section = controls
    qapp.processEvents()
    assert section.minimumSizeHint().width() <= 380
    for left, right in (
        (appearance.surface, appearance.unlit),
        (materials.widgets["texture_maps"], render.widgets["texture_mipmaps_cb"]),
        (materials.widgets["street_names"], materials.widgets["street_name_style"]),
    ):
        assert left.geometry().right() < right.geometry().left()
        assert left.geometry().center().y() == right.geometry().center().y()
        assert left.width() >= left.minimumSizeHint().width()
        assert right.width() >= right.minimumSizeHint().width()


def test_street_name_style_is_conditional_and_sync_does_not_republish(controls):
    viz, materials, _appearance, _render, _section = controls
    service = viz.scene_service.street_names
    style = materials.widgets["street_name_style"]
    assert not style.isHidden() and style.isEnabled()
    assert [style.itemText(i) for i in range(style.count())] == ["Plain", "Box", "Card"]
    service.set_style.side_effect = lambda mode: setattr(service, "style_mode", mode)
    style.setCurrentIndex(style.findData("box"))
    service.set_style.assert_called_once_with("box")
    materials.sync_procedural_controls()
    service.set_style.assert_called_once()
    service.enabled = False
    materials.sync_procedural_controls()
    assert style.isHidden() and style.currentData() == "box"
    service.enabled = True
    materials.sync_procedural_controls()
    assert not style.isHidden() and style.currentData() == "box"
