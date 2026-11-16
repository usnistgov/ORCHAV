"""Frame-detail opt-in and consistent MPC marker coloring."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from tests.visualizer.unit.test_semantic_mpc_parity import _visual_payload
from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.panels.mpc_panel import MPCVisualizationPanel
from visualizer.src.pipeline.core import MPCCore
from visualizer.src.pipeline.frame_pipeline import build_vm_cache_key
from visualizer.src.state import AppState, MpcVisibility, create_initial_state, update_state


def _view(core, mode, *, details):
    return core.create_view_model(
        0,
        _visual_payload(build_standard_mpc_frame()),
        mode,
        mpc_allowed_orders=(0, 1, 2, 3),
        mpc_allowed_types=(0, 1, 2, 4, 8, 37, 99),
        mpc_visibility=MpcVisibility(enabled=True, paths=True, bounce_points=True),
        include_frame_details=details,
    )


@pytest.mark.parametrize("mode", ["reflection_order", "mpc_type", "material", "delay", "path_loss"])
def test_bounce_colors_match_the_outgoing_segment_in_selected_mode(mode):
    core = MPCCore(logger=Mock())
    vm = _view(core, mode, details=False)
    assert vm is not None and len(vm.mpc_bounce_points)
    starts = vm.mpc_points[vm.mpc_lines[:, 0]]
    for point, color in zip(vm.mpc_bounce_points, vm.mpc_bounce_colors, strict=True):
        matches = np.flatnonzero(np.all(starts == point, axis=1))
        assert len(matches) == 1
        np.testing.assert_allclose(color, vm.mpc_colors[matches[0]])


@pytest.mark.parametrize("mode", ["delay", "path_loss"])
def test_compact_text_skips_histograms_and_opt_in_preserves_geometry(mode, monkeypatch):
    core = MPCCore(logger=Mock())
    original = core._build_range_summary
    summary = Mock(wraps=original)
    monkeypatch.setattr(core, "_build_range_summary", summary)
    compact = _view(core, mode, details=False)
    summary.assert_not_called()
    assert compact is not None and "\n" not in compact.stats_text

    detailed = _view(core, mode, details=True)
    summary.assert_called_once()
    assert detailed is not None and "\n" in detailed.stats_text
    for name in (
        "mpc_points",
        "mpc_lines",
        "mpc_colors",
        "mpc_bounce_points",
        "mpc_bounce_colors",
        "path_mask",
    ):
        np.testing.assert_array_equal(getattr(compact, name), getattr(detailed, name))


def test_type_legend_is_available_without_detailed_text():
    core = MPCCore(logger=Mock())
    compact = _view(core, "mpc_type", details=False)
    detailed = _view(core, "mpc_type", details=True)
    assert compact.mpc_line_itype_codes == detailed.mpc_line_itype_codes
    assert 37 in compact.mpc_line_itype_codes
    assert "Unknown (Type 37)" in detailed.stats_text
    assert "\n" not in compact.stats_text


def test_frame_details_default_and_session_roundtrip():
    compact = create_initial_state()
    detailed = replace(compact, show_mpc_frame_details=True)
    assert not compact.show_mpc_frame_details
    assert AppState.from_dict(detailed.to_dict()).show_mpc_frame_details
    legacy = detailed.to_dict()
    legacy.pop("show_mpc_frame_details")
    assert not AppState.from_dict(legacy).show_mpc_frame_details
    assert build_vm_cache_key(0, compact, None) != build_vm_cache_key(0, detailed, None)


def test_frame_details_widget_updates_state_and_restores_silently(qtbot):
    viz = SimpleNamespace(app_state=create_initial_state(), schedule_update=Mock())
    controller = UIController.__new__(UIController)
    controller.visualizer = viz
    viz.ui_controller = controller
    panel = MPCVisualizationPanel(viz)
    group = panel.create_panel()
    qtbot.addWidget(group)

    def set_state(**changes):
        viz.app_state = update_state(viz.app_state, **changes)
        panel.sync_visibility_from_state(viz.app_state)

    viz.set_state = set_state
    checkbox = panel.widgets["mpc_frame_details_cb"]
    assert not checkbox.isChecked()
    checkbox.click()
    assert viz.app_state.show_mpc_frame_details
    viz.schedule_update.assert_called_once_with()
    set_state(show_mpc_frame_details=False)
    assert not checkbox.isChecked()
    assert viz.schedule_update.call_count == 1
