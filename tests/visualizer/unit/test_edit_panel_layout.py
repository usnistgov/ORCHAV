"""Editing controls fit the same narrow sidebar as the Scene tab."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from visualizer.src.panels.nodes_panel import NodesSelectionPanel
from visualizer.src.services.raytracing_settings_service import RaytracingSettingsService


@pytest.mark.parametrize("mode", ["files", "live_grpc"])
def test_edit_controls_fit_narrow_sidebar(qtbot, mode):
    owner = SimpleNamespace(
        live_preview_service=SimpleNamespace(
            is_available=lambda: True,
            edit_mode=lambda: mode,
            enabled=True,
            dirty_edit_count=lambda: 1,
        ),
        raytracing_settings_service=RaytracingSettingsService(),
        scenario=SimpleNamespace(raytracing={"quality": {"preset": "custom"}}),
    )
    panel = NodesSelectionPanel(owner)
    group = panel.create_interactive_preview_panel()
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.addWidget(group)
    layout.addStretch()
    scroll = QScrollArea()
    qtbot.addWidget(scroll)
    scroll.setWidgetResizable(True)
    scroll.setWidget(page)
    scroll.resize(380, 850)
    scroll.show()
    qtbot.waitUntil(lambda: group.isVisible())

    assert scroll.horizontalScrollBar().maximum() == 0
    assert page.minimumSizeHint().width() <= scroll.viewport().width()
    for key in (
        "live_preview_cb",
        "live_preview_recompute_btn",
        "live_preview_reset_selected_btn",
        "live_preview_reset_all_btn",
        "rt_max_depth",
        "rt_samples_per_src",
        "rt_max_num_paths_per_src",
        "rt_seed",
        "rt_los",
        "rt_specular",
        "rt_diffuse",
        "rt_refraction",
        "rt_diffraction",
    ):
        widget = panel.widgets[key]
        if widget.isVisible():
            left = widget.mapTo(page, QPoint()).x()
            assert left >= 0
            assert left + widget.width() <= page.width(), key

    panel.widgets["rt_max_depth"].setValue(6)
    panel.widgets["rt_diffraction"].setChecked(True)
    settings = owner.raytracing_settings_service.release_settings()
    assert settings["max_depth"] == 6
    assert settings["diffraction"] is True
