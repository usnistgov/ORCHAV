"""Path Analysis borrows path data without opening an unrequested table."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tests.visualizer.unit.test_cinematic_pulses import renderer
from tests.visualizer.unit.test_experimental_path_analysis import _catalog
from visualizer.src.experiments.service import ExperimentService
from visualizer.src.services.mpc_explorer_service import ensure_mpc_explorer


@pytest.fixture
def analysis(qtbot):
    viz = SimpleNamespace(
        renderer=renderer(),
        animation_running=False,
        schedule_update=Mock(),
        _mpc_presented_source_epoch=1,
    )
    service = ExperimentService(viz)
    viz.extension_services = {"experiments": service}
    yield service
    service.close()
    explorer = getattr(viz, "_mpc_explorer_session", None)
    if explorer is not None:
        explorer.shutdown()


def _present_paths(service, qtbot):
    canonical = _catalog([10, 20, 30], angles=[-20, 0, 20]).canonical_data
    view_model = SimpleNamespace(canonical_data=canonical, path_mask=np.ones(3, dtype=bool))
    packet = SimpleNamespace(canonical_data=canonical, segment_mask=np.ones(3, dtype=bool))
    service.present(0, view_model, packet)
    qtbot.waitUntil(lambda: service.data is not None, timeout=3000)
    assert service.data.path_count == 3
    return canonical


def test_analysis_plots_work_without_creating_explorer_and_release_the_session(analysis, qtbot):
    analysis.open()
    explorer = analysis._explorer
    assert analysis.window.isVisible()
    assert explorer.window is None

    canonical = _present_paths(analysis, qtbot)
    assert explorer.window is None
    np.testing.assert_array_equal(analysis.window.arrivals.xData, [10, 20, 30])
    analysis.gate(15, 25)
    np.testing.assert_array_equal(analysis.path_ids, [1])
    np.testing.assert_array_equal(explorer._matching_path_ids, [1])
    np.testing.assert_array_equal(canonical.path_delays, [10, 20, 30])
    assert explorer.window is None

    analysis.window.close()
    assert not analysis.active
    assert explorer._finished
    assert getattr(analysis.viz, "_mpc_explorer_session", None) is None
    assert not analysis.overlay.layers


def test_show_paths_opens_the_linked_table_without_reviving_it_on_analysis_open(analysis, qtbot):
    analysis.open()
    _present_paths(analysis, qtbot)
    analysis.gate(15, 25)
    explorer = analysis._explorer
    assert explorer.window is None

    analysis.window.paths_requested.emit()
    table = explorer.window
    assert table.isVisible()
    qtbot.waitUntil(lambda: explorer._model.total_row_count == 1, timeout=3000)
    assert explorer._model.path_id_for_row(0) == 1

    table.close()
    assert table.isHidden()
    assert analysis.active and explorer.is_active
    analysis.open()
    assert explorer.window is table and table.isHidden()
    assert analysis.active


def test_closing_analysis_preserves_an_explicitly_opened_explorer(analysis, qtbot):
    explorer = ensure_mpc_explorer(analysis.viz)
    explorer.open()
    table = explorer.window
    analysis.open()
    _present_paths(analysis, qtbot)
    assert explorer.window is table and table.isVisible()

    analysis.window.close()
    assert not analysis.active
    assert table.isVisible()
    assert explorer.is_active and not explorer._finished


def test_closing_and_reopening_analysis_preserves_live_plot_views(analysis, qtbot):
    from pyqtgraph import ViewBox
    from shiboken6 import isValid

    analysis.open()
    _present_paths(analysis, qtbot)
    window = analysis.window
    views = [
        plot.getViewBox() for plot in (window.arrival_plot, window.angle_plot, window.colorbar)
    ]
    window.close()
    assert all(isValid(view) and view in ViewBox.AllViews for view in views)
    analysis.open()
    assert analysis.window is window and window.isVisible()
    _present_paths(analysis, qtbot)
    analysis.gate(15, 25)
    np.testing.assert_array_equal(analysis.path_ids, [1])
    assert all(isValid(view) and view in ViewBox.AllViews for view in views)
