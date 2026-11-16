"""MPC layers reveal their own sizing controls without changing saved choices."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QPoint

from shared.statistics.themes import theme_manager
from visualizer.src.panels.mpc_panel import MPCVisualizationPanel
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.state import MpcVisibility, create_initial_state


@pytest.fixture
def panel(qtbot):
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(mpc_type_markers=True),
        set_line_width=Mock(),
        set_point_size=Mock(),
    )
    owner = SimpleNamespace(renderer=renderer, app_state=create_initial_state())
    panel = MPCVisualizationPanel(owner)
    group = panel.create_panel()
    qtbot.addWidget(group)
    filters = panel.create_filtering_panel()
    qtbot.addWidget(filters)
    panel._test_filters = filters
    yield panel, group, owner
    theme_manager.remove_listener(panel._on_theme_change)


def test_layer_controls_hide_and_restore_without_losing_sizes_or_marker_choice(panel):
    controls, _group, owner = panel
    widgets = controls.widgets
    widgets["line_width_spin"].setValue(3.5)
    widgets["point_size_spin"].setValue(8.5)
    widgets["mpc_interaction_markers_cb"].setChecked(True)
    owner.renderer.set_line_width.reset_mock()
    owner.renderer.set_point_size.reset_mock()

    for checkbox, details in (
        ("mpc_paths_cb", "line_width_controls"),
        ("mpc_bounce_points_cb", "point_size_controls"),
    ):
        widgets[checkbox].setChecked(False)
        assert widgets[details].isHidden()
        if checkbox == "mpc_bounce_points_cb":
            assert widgets["mpc_interaction_markers_cb"].isHidden()
        widgets[checkbox].setChecked(True)
        assert not widgets[details].isHidden()
    assert widgets["line_width_spin"].value() == 3.5
    assert widgets["point_size_spin"].value() == 8.5
    assert widgets["mpc_interaction_markers_cb"].isChecked()
    assert not widgets["mpc_interaction_markers_cb"].isHidden()
    owner.renderer.set_line_width.assert_not_called()
    owner.renderer.set_point_size.assert_not_called()


def test_external_visibility_sync_updates_subcontrols_without_replaying_actions(panel):
    controls, _group, owner = panel
    changed = Mock()
    controls.widgets["mpc_paths_cb"].toggled.connect(changed)
    controls.widgets["mpc_bounce_points_cb"].toggled.connect(changed)
    state = replace(owner.app_state, mpc_visibility=MpcVisibility(enabled=False))
    controls.sync_visibility_from_state(state)
    assert controls.widgets["line_width_controls"].isHidden()
    assert controls.widgets["point_size_controls"].isHidden()
    assert controls.widgets["mpc_interaction_markers_cb"].isHidden()
    controls.sync_visibility_from_state(owner.app_state)
    assert not controls.widgets["line_width_controls"].isHidden()
    assert not controls.widgets["point_size_controls"].isHidden()
    changed.assert_not_called()


def test_visibility_rows_fit_400_pixels_and_align_controls(panel, qapp):
    controls, _group, _owner = panel
    assert _group.minimumSizeHint().width() <= 400
    visibility = controls.widgets["mpc_paths_cb"].parentWidget().parentWidget()
    visibility.setParent(None)
    visibility.resize(400, visibility.sizeHint().height())
    visibility.show()
    qapp.processEvents()
    assert visibility.minimumSizeHint().width() <= 400
    rows = controls.widgets["mpc_paths_cb"].parentWidget().layout()
    assert rows.getItemPosition(rows.indexOf(controls.widgets["line_width_controls"]))[:2] == (0, 1)
    assert rows.getItemPosition(rows.indexOf(controls.widgets["point_size_controls"]))[:2] == (1, 1)
    assert rows.getItemPosition(rows.indexOf(controls.widgets["mpc_interaction_markers_cb"]))[
        :2
    ] == (1, 2)
    width = controls.widgets["line_width_spin"]
    size = controls.widgets["point_size_spin"]
    assert width.mapTo(visibility, QPoint()).x() == size.mapTo(visibility, QPoint()).x()
    assert width.width() == size.width()
    visibility.close()
    visibility.deleteLater()


@pytest.mark.parametrize("mode", ["reflection_order", "mpc_type", "delay", "path_loss", "material"])
def test_color_mode_reveals_only_its_applicable_palette(panel, mode):
    controls, _group, _owner = panel
    controls.update_color_legend(mode)
    assert controls.widgets["categorical_cmap_combo"].isHidden() is (
        mode not in {"reflection_order", "mpc_type"}
    )
    assert controls.widgets["continuous_cmap_combo"].isHidden() is (
        mode not in {"delay", "path_loss"}
    )


def test_render_cap_details_follow_external_and_clicked_state(panel):
    controls, _group, owner = panel
    assert controls.widgets["topk_render_presets"].isHidden()
    controls.sync_visibility_from_state(replace(owner.app_state, topk_render_enabled=True))
    assert not controls.widgets["topk_render_presets"].isHidden()
    assert not controls.widgets["topk_render_max_spin"].isHidden()
    controls.widgets["topk_render_cb"].click()
    assert controls.widgets["topk_render_presets"].isHidden()
    assert controls.widgets["topk_render_max_spin"].isHidden()


def test_filtering_fits_380_pixels(panel, qapp):
    controls, _group, _owner = panel
    filters = controls._test_filters
    controls.widgets["path_properties_section"].expand()
    filters.resize(380, 700)
    filters.show()
    qapp.processEvents()
    assert filters.minimumSizeHint().width() <= 380
