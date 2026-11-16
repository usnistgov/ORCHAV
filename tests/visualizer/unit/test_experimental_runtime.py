"""Lifecycle regressions for optional, pygfx-only recorded-data experiments."""

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from visualizer.src.experiments.runtime import enabled
from visualizer.src.experiments.service import ExperimentService
from visualizer.src.experiments.window import ExperimentWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_renderer_gate_never_activates_open3d():
    for renderer in ("open3d", "modern", "legacy", None):
        assert not enabled(SimpleNamespace(renderer=SimpleNamespace(renderer_id=renderer)))
    assert enabled(SimpleNamespace(renderer=SimpleNamespace(renderer_id="pygfx")))


def test_hidden_service_does_not_index_frame(app):
    service = ExperimentService(SimpleNamespace(_mpc_presented_source_epoch=2))
    packet = SimpleNamespace(canonical_data=object())
    service.present(7, object(), packet)
    assert service.token == (2, 7, id(packet.canonical_data))
    assert service.catalog is None and service.data is None and service._future is None
    service.close()


def test_late_worker_result_rejected_after_pause(app, qtbot):
    viz = SimpleNamespace()
    service = ExperimentService(viz)
    service.window = ExperimentWindow()
    qtbot.addWidget(service.window)
    service.window.show()
    service._serial = 4
    service.pause()
    service._completed(4, ("stale", "stale", 100), None)
    assert service.data is None and service.histogram is None
    assert not service._clock.isActive()
    service.close()


def test_options_clear_stale_plots_and_selection(app, qtbot):
    service = ExperimentService(SimpleNamespace())
    service.window = ExperimentWindow()
    qtbot.addWidget(service.window)
    service.window.show()
    service.window.arrivals.setData([1, 2], [3, 4])
    service.data = object()
    service.histogram = object()
    service._options_changed()
    assert service.data is None and service.histogram is None
    assert service.window.arrivals.xData is None
    assert len(service.path_ids) == 0
    service.close()


def test_window_updates_are_not_recursive_user_actions(app, qtbot):
    window = ExperimentWindow()
    qtbot.addWidget(window)
    events = []
    window.options_changed.connect(lambda: events.append("changed"))
    window.set_pairs([(0, 0), (1, 2)])
    assert not events
    assert window.settings()["pair"] == (0, 0)
    window.pair.setCurrentIndex(1)
    assert events == ["changed"]
    window.set_pairs([(3, 4), (1, 2), (0, 0)])
    assert window.settings()["pair"] == (1, 2)
    assert events == ["changed"]
    window.close()


def test_range_request_rejects_other_scene_frame(app):
    cell = object()
    dashboard = SimpleNamespace(
        isVisible=lambda: True,
        _last_request=SimpleNamespace(effective_display_step=3),
        _linked_selection=SimpleNamespace(cell=cell),
    )
    viz = SimpleNamespace(
        ui_manager=SimpleNamespace(panels={"sensing": SimpleNamespace(_sensing_window=dashboard)})
    )
    service = ExperimentService(viz)
    service.step = 4
    with pytest.raises(ValueError, match="selected result's scene"):
        service.open_cell_paths(dashboard)
    service.close()


def test_shutdown_idempotent(app, qtbot):
    service = ExperimentService(SimpleNamespace())
    service.window = ExperimentWindow()
    qtbot.addWidget(service.window)
    service.window.show()
    service._clock.start()
    service.close()
    service.close()
    assert service.window is None and service._retired
    assert not service._clock.isActive()


def test_retirement_unregisters_only_owned_plot_views_before_qt_deletion(app, qtbot):
    from pyqtgraph import ViewBox
    from PySide6.QtCore import QCoreApplication, QEvent
    from shiboken6 import isValid

    service = ExperimentService(SimpleNamespace())
    window = service.window = ExperimentWindow()
    qtbot.addWidget(window)
    owned = [
        plot.getViewBox() for plot in (window.arrival_plot, window.angle_plot, window.colorbar)
    ]
    unrelated = ViewBox()
    try:
        assert all(view in ViewBox.AllViews for view in owned)
        service.close()
        service.close()
        window.retire_plots()
        assert all(view not in ViewBox.AllViews for view in owned)
        assert unrelated in ViewBox.AllViews

        # Keep wrappers alive across actual Qt destruction, reproducing the
        # ordering that used to leave invalid objects in pyqtgraph's registry.
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert all(not isValid(view) for view in owned)
        assert all(view not in ViewBox.AllViews for view in owned)
        assert isValid(unrelated) and unrelated in ViewBox.AllViews
    finally:
        unrelated.close()


def test_edit_one_gate_bound_preserves_unrounded_other_bound(app, qtbot):
    window = ExperimentWindow()
    qtbot.addWidget(window)
    assert callable(window.lower)  # Preserve QWidget's stacking-order method.
    window.gate.setRegion((26.685127, 48.107296))
    window.lower_bound.setValue(30.0)
    window._boxes(0)
    assert window.gate.getRegion() == (30.0, 48.107296)
    window._boxes(1)
    assert window.gate.getRegion() == (30.0, 48.107296)
    window.close()


def test_analysis_service_retires_without_a_separate_launcher_panel(app):
    from visualizer.src.experiments.runtime import register
    from visualizer.src.extensions import (
        clear_runtime_extension_services,
        registered_runtime_extensions,
    )

    register()
    extension = next(e for e in registered_runtime_extensions() if e.key == "experiments")
    assert extension.panel_factory is None
    calls = []
    viz = SimpleNamespace(
        extension_services={"experiments": SimpleNamespace(close=lambda: calls.append("closed"))}
    )
    clear_runtime_extension_services(viz)
    assert calls == ["closed"]
    assert viz.extension_services == {}


def test_selected_plot_markers_use_aggregated_arrivals_and_wrapped_angles(app, qtbot):
    from dataclasses import replace

    import numpy as np

    from tests.visualizer.unit.test_experimental_integration_review import _catalog
    from visualizer.src.experiments.path_analysis import delay_angle_histogram, select_paths

    catalog = _catalog(4)
    catalog.canonical_data.path_delays[:] = [5, 5, 15, np.nan]
    catalog.canonical_data.path_aoa_az[:] = [180, 0, np.nan, 0]
    data = select_paths(catalog, tx_id=0, rx_id=0)
    window = ExperimentWindow()
    qtbot.addWidget(window)
    window.set_pairs([(0, 0)])
    window.set_analysis(data, delay_angle_histogram(data), reset_gate=True, clock_end=1)

    def select(path_id):
        window.set_selected_path(
            SimpleNamespace(identity=(0, path_id), canonical_path_id=path_id, tx_id=0, rx_id=0)
        )

    select(0)
    np.testing.assert_array_equal(window.selected_arrival.xData, [5])
    np.testing.assert_array_equal(window.selected_arrival.yData, [2])
    np.testing.assert_array_equal(window.selected_angle.xData, [-180])
    np.testing.assert_array_equal(window.selected_angle.yData, [5])
    select(2)
    assert window.selected_delay.isVisible()
    assert not window.selected_angle.isVisible()
    assert "angle unavailable" in window.selected_label.text()
    select(3)
    assert not window.selected_delay.isVisible()
    assert not window.selected_arrival.isVisible()
    assert "delay unavailable" in window.selected_label.text()
    select(100)
    assert not window.selected_angle.isVisible()
    assert "outside the displayed pair/scope" in window.selected_label.text()
    select(0)
    window.clear_analysis()
    assert not window.selected_angle.isVisible()
    assert window._analysis_data is None
    zero_data = replace(data, weights=np.zeros(4))
    window.set_analysis(zero_data, delay_angle_histogram(zero_data), reset_gate=True, clock_end=1)
    select(1)
    np.testing.assert_array_equal(window.selected_arrival.yData, [0])
    assert not window.selected_angle.isVisible()
    assert "zero weight" in window.selected_label.text()


def test_retiring_experiment_service_preserves_unrelated_button_connections(app, qtbot):
    """Closing a scenario service must not disconnect persistent controls."""
    import gc

    from PySide6.QtWidgets import QPushButton

    button = QPushButton("Next frame")
    qtbot.addWidget(button)
    calls = []
    button.clicked.connect(lambda _checked=False: calls.append("clicked"))
    service = ExperimentService(SimpleNamespace())
    button.click()
    assert calls == ["clicked"]

    service.close()
    del service
    gc.collect()
    button.click()
    assert calls == ["clicked", "clicked"]


def test_retiring_experiment_service_preserves_unrelated_timer_connections(app, qtbot):
    """Scene cleanup must leave other Qt-driven work able to run."""
    import gc

    from PySide6.QtCore import QTimer

    timer = QTimer()
    timer.setSingleShot(True)
    ticks = []
    timer.timeout.connect(lambda: ticks.append("tick"))
    service = ExperimentService(SimpleNamespace())
    timer.start(1)
    qtbot.waitUntil(lambda: len(ticks) == 1)

    service.close()
    del service
    gc.collect()
    timer.start(1)
    qtbot.waitUntil(lambda: len(ticks) == 2, timeout=500)
