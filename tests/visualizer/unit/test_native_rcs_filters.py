"""Native sensing remains visible by default and respects saved filter intent."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.metrics.mpc_canon import CanonicalStepData, build_filter_mask
from visualizer.src.panels.mpc_panel import MPCVisualizationPanel
from visualizer.src.state import DEFAULT_MPC_ALLOWED_TYPES, AppState, create_initial_state


def native_path():
    return CanonicalStepData(
        points=np.array([[0, 0, 0], [5, 0, 0], [0, 0, 0]], dtype=np.float32),
        lines=np.array([[0, 1], [1, 2]], dtype=np.int32),
        order=np.ones(3, dtype=np.uint8),
        itype=np.array([0, 16, 0], dtype=np.uint8),
        delay=np.zeros(3, dtype=np.float32),
        loss=np.zeros(3, dtype=np.float32),
        segment_itype=np.array([16, 16], dtype=np.uint8),
    )


def test_default_native_path_is_visible_and_sensing_toggle_controls_it(qtbot):
    viz = SimpleNamespace(app_state=create_initial_state(), schedule_update=lambda: None)
    viz.set_state = lambda **changes: setattr(viz, "app_state", replace(viz.app_state, **changes))
    controller = SimpleNamespace(visualizer=viz, _invalidate_cache=lambda *args, **kwargs: None)
    panel = MPCVisualizationPanel(None)
    controls = panel._create_discrete_filters()
    qtbot.addWidget(controls)
    checkbox = panel.widgets["type_16_cb"]
    assert checkbox.text() == "Sensing"
    assert checkbox.isChecked()
    checkbox.toggled.connect(
        lambda enabled: UIController.handle_mpc_type_filter_changed(controller, 16, enabled)
    )

    def segments():
        return build_filter_mask(
            native_path(),
            allowed_orders=list(viz.app_state.mpc_allowed_orders),
            allowed_types=list(viz.app_state.mpc_allowed_types),
        )[1]

    assert np.all(segments())
    checkbox.setChecked(False)
    assert not np.any(segments())
    checkbox.setChecked(True)
    assert np.all(segments())


@pytest.mark.parametrize("version,old", [(1, [0, 1, 2, 4, 8]), (2, [0, 1, 2, 4, 8, 99])])
def test_older_full_default_filters_migrate_to_include_sensing(version, old):
    data = create_initial_state().to_dict()
    data["mpc_type_filter_schema_version"] = version
    data["mpc_allowed_types"] = old
    restored = AppState.from_dict(data)
    assert restored.mpc_allowed_types == DEFAULT_MPC_ALLOWED_TYPES
    assert 16 in restored.mpc_allowed_types


@pytest.mark.parametrize(
    "version,selected", [(2, [0, 1, 2, 4, 8]), (2, [1, 8]), (2, []), (3, [0, 1, 2, 4, 8, 99])]
)
def test_custom_or_current_explicit_hidden_types_are_preserved(version, selected):
    data = create_initial_state().to_dict()
    data["mpc_type_filter_schema_version"] = version
    data["mpc_allowed_types"] = selected
    assert AppState.from_dict(data).mpc_allowed_types == frozenset(selected)


def test_sensing_disabled_choice_survives_save_reload():
    state = replace(create_initial_state(), mpc_allowed_types=DEFAULT_MPC_ALLOWED_TYPES - {16})
    encoded = json.loads(json.dumps(state.to_dict()))
    assert encoded["mpc_type_filter_schema_version"] == 3
    assert AppState.from_dict(encoded).mpc_allowed_types == state.mpc_allowed_types
