"""Exercise the real Paths widgets against canonical visibility and UI actions."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QApplication

from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.panels.global_context_panel import GlobalContextPanel
from visualizer.src.panels.mpc_panel import MPCVisualizationPanel
from visualizer.src.services.cache_service import CacheInvalidationScope
from visualizer.src.services.session_service import SessionService
from visualizer.src.state import MpcVisibility, create_initial_state, update_state


@pytest.mark.parametrize("part", ["paths", "bounce_points"])
def test_clicking_hidden_part_enables_layer_and_schedules_paused_refresh(part):
    app = QApplication.instance() or QApplication([])
    viz = SimpleNamespace(
        app_state=create_initial_state(mpc_layer_enabled=False),
        schedule_update=Mock(),
        cache_service=SimpleNamespace(invalidate=Mock()),
    )
    panel = MPCVisualizationPanel(viz)
    group = panel.create_panel()
    context = GlobalContextPanel(viz)
    context_group = context.create_panel()
    controller = UIController.__new__(UIController)
    controller.visualizer = viz

    def set_state(**changes):
        viz.app_state = update_state(viz.app_state, **changes)
        panel.sync_visibility_from_state(viz.app_state)
        context.sync_from_state(viz.app_state)

    viz.set_state = set_state
    widget = panel.widgets[f"mpc_{part}_cb"]
    widget.toggled.connect(getattr(controller, f"handle_mpc_{part}_toggled"))
    master = context.widgets["mpc_layer_cb"]
    master.toggled.connect(controller.handle_mpc_layer_toggled)
    set_state()
    assert not master.isChecked()
    assert not widget.isChecked() and widget.isEnabled()

    # This used to be a checked, disabled widget; click() had no effect.
    widget.click()
    assert master.isChecked() and widget.isChecked()
    assert getattr(viz.app_state.mpc_visibility, f"effective_{part}")
    viz.schedule_update.assert_called_once_with()
    viz.cache_service.invalidate.assert_called_once_with(
        CacheInvalidationScope.MPC_RENDER_SETTINGS, reason=f"mpc_{part}"
    )

    # Hiding a part preserves the other part's choice and still refreshes.
    widget.click()
    assert not widget.isChecked()
    assert not getattr(viz.app_state.mpc_visibility, f"effective_{part}")
    assert viz.app_state.mpc_visibility.enabled
    assert viz.schedule_update.call_count == 2
    widget.click()
    master.click()
    assert not master.isChecked() and not widget.isChecked()
    assert widget.isEnabled()
    assert getattr(viz.app_state.mpc_visibility, part)  # preference retained
    widget.click()
    assert master.isChecked() and widget.isChecked()

    # Loading another scenario/session must synchronize silently and leave the
    # same one-click route back to visible paths.
    count = viz.schedule_update.call_count
    set_state(mpc_visibility=MpcVisibility(enabled=False))
    service = SessionService(viz)
    service._sync_mpc_panel({"mpc": panel}, viz.app_state)
    assert not widget.isChecked() and widget.isEnabled()
    assert viz.schedule_update.call_count == count
    widget.click()
    assert getattr(viz.app_state.mpc_visibility, f"effective_{part}")
    group.deleteLater()
    context_group.deleteLater()
    app.processEvents()
