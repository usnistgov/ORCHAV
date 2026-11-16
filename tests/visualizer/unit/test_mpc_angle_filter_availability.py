"""Angle controls require a reference endpoint and saved data for the selected paths."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from shared.statistics.themes import theme_manager
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from visualizer.src.io.packed_frame_payload import (
    standard_frame_to_visual_frame,
    visual_frame_read_request,
)
from visualizer.src.panels.mpc_panel import MPCVisualizationPanel

ANGLE_KEYS = ("aod_az", "aod_el", "aoa_az", "aoa_el")


@pytest.fixture
def panel(qtbot):
    owner = SimpleNamespace(
        app_state=SimpleNamespace(selected_tx="all", selected_rx="all"),
        ui_controller=SimpleNamespace(handle_range_filter_changed=Mock()),
    )
    controls = MPCVisualizationPanel(owner)
    group = controls.create_filtering_panel()
    qtbot.addWidget(group)
    yield controls, owner
    theme_manager.remove_listener(controls._on_theme_change)


@pytest.fixture
def canon():
    return standard_frame_to_visual_frame(
        build_standard_mpc_frame("baseline"), request=visual_frame_read_request()
    )["canonical_data"]


def _assert_available(panel, expected):
    for key, enabled in zip(ANGLE_KEYS, expected):
        for suffix in ("label", "min", "max"):
            assert panel.widgets[f"{key}_filter_{suffix}"].isEnabled() is enabled


def test_angle_filters_require_explicit_endpoint_even_when_saved_angles_exist(panel, canon):
    controls, owner = panel
    controls.sync_angle_filter_availability(canon)
    _assert_available(controls, (False, False, False, False))
    assert "All TX" in controls.widgets["aod_az_filter_min"].toolTip()
    assert "All RX" in controls.widgets["aoa_el_filter_max"].toolTip()
    assert "Context" in controls.widgets["angle_filter_status"].text()

    owner.app_state.selected_tx = 1
    controls.refresh_aperture_preview_state()
    _assert_available(controls, (True, True, False, False))

    owner.app_state.selected_rx = 0
    controls.refresh_aperture_preview_state()
    _assert_available(controls, (True, True, True, True))
    controls.widgets["aod_az_filter_min"].setValue(-45)
    owner.ui_controller.handle_range_filter_changed.reset_mock()

    owner.app_state.selected_tx = "all"
    controls.refresh_aperture_preview_state()
    _assert_available(controls, (False, False, True, True))
    owner.app_state.selected_tx = 1
    controls.refresh_aperture_preview_state()
    assert controls.widgets["aod_az_filter_min"].value() == -45
    owner.ui_controller.handle_range_filter_changed.assert_not_called()


@pytest.mark.parametrize(
    ("tx", "rx", "expected"),
    [
        (1, 0, (True, True, True, True)),
        (0, 1, (True, False, False, True)),
        (0, 0, (True, True, True, False)),
        (1, 1, (False, False, False, False)),
    ],
)
def test_each_angle_requires_valid_saved_values_on_selected_paths(panel, canon, tx, rx, expected):
    """The fixture's validity bits leave different angles missing on different pairs."""
    controls, owner = panel
    owner.app_state.selected_tx, owner.app_state.selected_rx = tx, rx
    controls.sync_angle_filter_availability(canon)
    _assert_available(controls, expected)
    for key, available in zip(ANGLE_KEYS, expected):
        if not available:
            assert "No valid saved" in controls.widgets[f"{key}_filter_min"].toolTip()


def test_frame_changes_disable_missing_and_nonfinite_angles_without_losing_limits(panel, canon):
    controls, owner = panel
    owner.app_state.selected_tx, owner.app_state.selected_rx = 1, 0
    controls.sync_angle_filter_availability(canon)
    controls.widgets["aoa_el_filter_max"].setValue(35)
    owner.ui_controller.handle_range_filter_changed.reset_mock()

    missing = replace(
        canon,
        aod_az=None,
        aod_el=np.full(len(canon.points), np.nan),
        aoa_az=np.full(len(canon.points), np.inf),
    )
    controls.sync_angle_filter_availability(missing)
    _assert_available(controls, (False, False, False, True))
    controls.sync_angle_filter_availability(None)
    _assert_available(controls, (False, False, False, False))
    controls.sync_angle_filter_availability(canon)
    _assert_available(controls, (True, True, True, True))
    assert controls.widgets["aoa_el_filter_max"].value() == 35
    owner.ui_controller.handle_range_filter_changed.assert_not_called()


def test_angle_availability_samples_paths_once_per_frame_and_selection(panel, canon, monkeypatch):
    controls, owner = panel
    finite = Mock(wraps=np.isfinite)
    monkeypatch.setattr(np, "isfinite", finite)
    controls.sync_angle_filter_availability(canon)
    assert finite.call_count == 4
    assert all(len(args[0]) == len(canon.path_start_indices) for args, _ in finite.call_args_list)

    for _ in range(3):
        controls.sync_angle_filter_availability(canon)
        controls.refresh_aperture_preview_state()
    assert finite.call_count == 4

    owner.app_state.selected_tx, owner.app_state.selected_rx = 0, 1
    controls.refresh_aperture_preview_state()
    assert finite.call_count == 8
    assert all(len(args[0]) == 1 for args, _ in finite.call_args_list[-4:])
