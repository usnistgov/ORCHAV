"""Linked path views preserve frame identity through docking and analysis queries."""

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow

from tests.visualizer.unit.test_experimental_integration_review import _catalog, _overlay
from visualizer.src.experiments.service import ExperimentService, _analyze
from visualizer.src.renderers.pygfx import experiments
from visualizer.src.widgets.auxiliary_dock import DOCK_AREAS, AuxiliaryDock


@pytest.fixture
def workspace(qtbot, monkeypatch):
    viz = QMainWindow()
    viz.resize(1500, 950)
    viz.setCentralWidget(QLabel("Scene"))
    qtbot.addWidget(viz)
    viz.renderer = SimpleNamespace(renderer_id="pygfx")
    viz.schedule_update = lambda: None
    viz._mpc_presented_source_epoch = 1
    service = ExperimentService(viz)
    viz.extension_services = {"experiments": service}
    overlay = _overlay()
    monkeypatch.setattr(experiments, "ExperimentOverlay", lambda _renderer: overlay)
    viz.show()
    service.open()
    canonical = _catalog(2100).canonical_data
    canonical.path_delays[:] = np.arange(2100)
    service.present(
        4,
        SimpleNamespace(path_mask=None, mpc_visibility=SimpleNamespace(effective_paths=True)),
        SimpleNamespace(canonical_data=canonical, segment_mask=None),
    )
    service._debounce.stop()
    service._completed(service._serial, _analyze(service.catalog, service.window.settings()), None)
    session = service._explorer
    qtbot.waitUntil(lambda: session._model.total_row_count == 2100)
    yield viz, service, session
    service.close()
    session.shutdown()


def test_plot_selection_uses_one_catalog_and_keeps_all_matches(workspace, qtbot):
    _viz, analysis, session = workspace
    assert analysis.catalog is session.catalog
    analysis.gate(100, 2050)
    qtbot.waitUntil(lambda: session._model.total_row_count == 1951)
    assert len(analysis.path_ids) == 1951
    analysis.window.cinematic_limit.setCurrentIndex(analysis.window.cinematic_limit.findData(100))
    assert len(analysis.path_ids) == 100
    assert session._model.total_row_count == 1951
    assert "1,951" in session.window.matching_label.text()
    assert "100/1951" in analysis.window.cinematic_note.text()
    assert not hasattr(analysis.window, "path_choice")
    session._select_path(2000, origin="table")
    assert session._selection_port.selected_path_id == 2000
    assert "Selected path: 2000" in analysis.window.selected_label.text()
    analysis.clear_overlays()
    qtbot.waitUntil(lambda: session._model.total_row_count == 2100)
    assert session._selection_port.selected_path_id == 2000
    assert len(analysis.path_ids) == 0


def test_shared_scope_and_empty_plot_selection(workspace, qtbot):
    _viz, analysis, session = workspace
    session.window.scope_combo.setCurrentIndex(session.window.scope_combo.findData("filtered"))
    assert analysis.window.settings()["scope"] == "filtered"
    analysis.window.scope.setCurrentIndex(analysis.window.scope.findData("all"))
    assert session.scope == "all"
    analysis._debounce.stop()
    analysis._completed(
        analysis._serial, _analyze(analysis.catalog, analysis.window.settings()), None
    )
    analysis.gate(3000, 4000)
    qtbot.waitUntil(lambda: session._model.total_row_count == 0)
    assert session._matching_path_ids.size == 0
    assert "Matching paths: 0" in analysis.window.matching_label.text()
    session.window.matchingPathsCleared.emit()
    qtbot.waitUntil(lambda: session._model.total_row_count == 2100)
    assert analysis.window.matching_label.text() == "Matching paths: none"


def test_delay_angle_cell_populates_the_shared_table(workspace, qtbot):
    _viz, analysis, session = workspace
    delay, angle = np.argwhere(analysis.histogram.values > 0)[0]
    expected = analysis.histogram.path_ids_for_bin(int(delay), int(angle))
    analysis.select_bin(int(delay), int(angle))
    qtbot.waitUntil(lambda: session._model.total_row_count == len(expected))
    np.testing.assert_array_equal(session._matching_path_ids, expected)
    assert "Delay–angle cell" in session.window.matching_label.text()
    assert analysis.overlay.direction_ids.size == len(expected)


def test_docking_and_tabs_preserve_selection_and_models(workspace, qtbot):
    viz, analysis, session = workspace
    table, plots = session.window, analysis.window
    session._select_path(17, origin="table")
    catalog, model = session.catalog, session._model
    for view in (table, plots):
        assert view.parentWidget() is viz
        for name, area in DOCK_AREAS.items():
            view.place(name)
            assert not view.isFloating() and viz.dockWidgetArea(view) == area
        view.place("float")
        assert view.isFloating()
    plots.join(table, tabbed=True)
    assert plots in viz.tabifiedDockWidgets(table)
    for view in (table, plots, table):
        view.raise_()
        qtbot.wait(20)
        assert session.catalog is catalog and session._model is model
        assert session._selection_port.selected_path_id == 17
        assert analysis.active
    plots.join(table, tabbed=False)
    assert plots not in viz.tabifiedDockWidgets(table)
    assert session._selection_port.selected_path_id == 17


def test_table_can_close_and_reopen_while_analysis_keeps_selection(workspace, qtbot):
    _viz, analysis, session = workspace
    session._select_path(42, origin="table")
    model = session._model
    session.window.close()
    qtbot.wait(20)
    assert session.is_active and session._model is model
    analysis.show_paths()
    assert session.window.isVisible()
    assert session._selection_port.selected_path_id == 42
    analysis.window.close()
    assert analysis._explorer is None and session.is_active
    assert session._selection_port.selected_path_id == 42
    session.window.close()
    qtbot.waitUntil(lambda: session._finished)


def test_frame_change_rejects_previous_matching_set(workspace, qtbot):
    _viz, analysis, session = workspace
    old_token = session.frame_token
    session._select_path(1, origin="table")
    canonical = _catalog(8).canonical_data
    analysis.present(
        5,
        SimpleNamespace(path_mask=None),
        SimpleNamespace(canonical_data=canonical, segment_mask=None),
    )
    analysis._debounce.stop()
    session.set_matching_paths(old_token, np.array([1]), "stale")
    qtbot.waitUntil(lambda: session._model.total_row_count == 8)
    assert session._matching_path_ids is None
    assert session._selection_port.selected_path_id is None
    assert "Selected path: none" in analysis.window.selected_label.text()
    assert analysis.catalog is session.catalog


def test_table_alone_keeps_selection_when_moved(workspace, qtbot):
    viz, analysis, session = workspace
    analysis.window.close()
    session._select_path(24, origin="table")
    catalog, model = session.catalog, session._model
    peer = AuxiliaryDock("Actor view", viz)
    peer.setWidget(QLabel("Camera"))
    peer.attach()
    peer.show()
    for position in ("top", "right", "bottom", "float"):
        session.window.place(position)
        qtbot.wait(20)
        assert session.catalog is catalog and session._model is model
        assert session._selection_port.selected_path_id == 24
    peer.join(session.window, tabbed=True)
    peer.raise_()
    qtbot.wait(20)
    assert session.is_active and session._selection_port.selected_path_id == 24
    peer.close()


def test_placement_menu_can_join_live_views(workspace):
    viz, _analysis, session = workspace
    peer = AuxiliaryDock("Live figure", viz)
    peer.setWidget(QLabel("Plot"))
    peer.attach()
    peer.show()
    menu = session.window.placement_menu()
    actions = {action.text(): action for action in menu.actions()}
    assert {"Dock top", "Dock right", "Dock bottom", "Float"} <= actions.keys()
    actions["Dock top"].trigger()
    assert viz.dockWidgetArea(session.window) == Qt.TopDockWidgetArea
    join = actions["Tab with"].menu()
    next(action for action in join.actions() if action.text() == "Live figure").trigger()
    assert peer in viz.tabifiedDockWidgets(session.window)
    peer.close()
    menu.deleteLater()


@pytest.mark.parametrize("row", [0, 2])
def test_row_click_toggles_both_plot_markers_without_clearing_matches(workspace, qtbot, row):
    _viz, analysis, session = workspace
    table = session.window.table
    index = table.model().index(row, 0)
    path_id = table.model().path_id_for_row(row)
    table.scrollTo(index)
    matches = session._matching_path_ids.copy()
    gate = analysis.window.gate.getRegion()

    def click(column):
        cell = table.model().index(row, column)
        qtbot.mouseClick(table.viewport(), Qt.LeftButton, pos=table.visualRect(cell).center())

    click(0)
    assert session._selected_path_id == path_id
    assert analysis.window.selected_delay.isVisible()
    assert analysis.window.selected_delay.value() == path_id
    np.testing.assert_array_equal(analysis.window.selected_arrival.xData, [path_id])
    np.testing.assert_array_equal(analysis.window.selected_angle.yData, [path_id])
    assert analysis.window.selected_angle.isVisible()
    assert analysis.window.gate.getRegion() == gate
    click(1)  # Clicking another cell in the same selected row also deselects it.
    assert session._selection_port.selected_path_id is None
    assert not table.selectionModel().hasSelection()
    assert not table.currentIndex().isValid()
    assert not session.window.details.toPlainText()
    assert not analysis.window.selected_delay.isVisible()
    assert not analysis.window.selected_angle.isVisible()
    assert not analysis.window.selected_arrival.isVisible()
    np.testing.assert_array_equal(session._matching_path_ids, matches)
    assert session._selection_port.catalog is session.catalog
    click(1)
    assert session._selection_port.selected_path_id == path_id
    assert analysis.window.selected_angle.isVisible()
    session.window.select_path(path_id)
    session.window.select_path(path_id)
    assert session._selected_path_id == path_id  # Programmatic reveals never toggle.


def test_selection_follows_a_different_pair_and_waits_for_its_analysis(workspace):
    _viz, analysis, session = workspace
    canonical = _catalog(6).canonical_data
    canonical.path_delays[:] = np.arange(6)
    canonical.path_tx[3:] = 1
    canonical.tx_id[canonical.path_id >= 3] = 1
    analysis.present(5, SimpleNamespace(path_mask=None), SimpleNamespace(canonical_data=canonical))
    analysis._debounce.stop()
    analysis._completed(
        analysis._serial, _analyze(analysis.catalog, analysis.window.settings()), None
    )
    session._select_path(4, origin="table")
    assert analysis.window.settings()["pair"] == (1, 0)
    assert not analysis.window.selected_delay.isVisible()
    analysis._debounce.stop()
    analysis._completed(
        analysis._serial, _analyze(analysis.catalog, analysis.window.settings()), None
    )
    assert analysis.window.selected_delay.value() == 4
    assert analysis.window.selected_angle.isVisible()
    session.clear_selected_path()
    # A later plot rebuild must not restore the cleared selection.
    analysis.window.set_analysis(analysis.data, analysis.histogram, reset_gate=False, clock_end=1)
    assert not analysis.window.selected_delay.isVisible()
