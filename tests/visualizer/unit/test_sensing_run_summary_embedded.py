"""Summary figures use the workspace scroll area and shared graph height."""

from PySide6.QtWidgets import QScrollArea

from tests.visualizer.unit.test_sensing_run_summary_polish import _data
from visualizer.src.sensing.run_summary_panel import RunSummaryPanel


def _settle(qtbot, panel):
    previous, unchanged = None, 0

    def stable():
        nonlocal previous, unchanged
        state = (
            panel.size().toTuple(),
            tuple((plot.size().toTuple(), plot.minimumHeight()) for plot in panel.plots.values()),
        )
        idle = not panel._embedded_size_timer.isActive() and all(
            not plot._legend_timer.isActive() for plot in panel.plots.values()
        )
        unchanged = unchanged + 1 if idle and state == previous else 0
        previous = state
        return unchanged >= 4

    qtbot.waitUntil(stable)


def _host(qtbot, width=1267, height=660):
    scroll = QScrollArea()
    qtbot.addWidget(scroll)
    scroll.setWidgetResizable(True)
    panel = RunSummaryPanel()
    panel.set_embedded_graph_height(180)
    assert not panel.plots  # Opting into embedded sizing keeps plots lazy.
    panel.set_summary(_data())
    scroll.setWidget(panel)
    scroll.resize(width, height)
    scroll.show()
    _settle(qtbot, panel)
    return scroll, panel


def test_embedded_summary_first_row_fits_and_extra_host_height_does_not_stretch_it(qtbot):
    host, panel = _host(qtbot)
    card = panel.plot_cards["counts"]
    assert card.mapTo(host.viewport(), card.rect().bottomRight()).y() < host.viewport().height()
    assert host.verticalScrollBar().maximum() > 0
    host.resize(1267, 1500)
    _settle(qtbot, panel)
    original_height = card.height()
    host.resize(1267, 1800)
    _settle(qtbot, panel)
    assert card.height() == original_height
    assert panel.tabs.widget(0).verticalScrollBar().maximum() == 0


def test_embedded_graph_height_resizes_existing_figures_without_rendering_again(qtbot):
    _, panel = _host(qtbot)
    plot = panel.plots["counts"]
    item = plot.listDataItems()[0]
    plot.setRange(xRange=(0.5, 1.5), yRange=(-0.5, 2.5), padding=0)
    _settle(qtbot, panel)
    original_height = panel.plot_cards["counts"].height()
    original_ranges = plot.viewRange()
    panel.set_embedded_graph_height(340)
    _settle(qtbot, panel)
    assert panel.plot_cards["counts"].height() >= original_height + 160
    assert plot.listDataItems()[0] is item
    assert plot.viewRange() == original_ranges
    panel.set_embedded_graph_height(180)
    _settle(qtbot, panel)
    assert panel.plot_cards["counts"].height() == original_height


def test_embedded_summary_reflows_rows_and_current_tab_without_inner_scrolling(qtbot):
    host, panel = _host(qtbot, width=600)
    assert panel._columns == 1
    narrow_height = panel.height()
    host.resize(1267, 660)
    _settle(qtbot, panel)
    assert panel._columns == 2
    assert panel.height() < narrow_height
    overview_height = panel.height()
    panel.tabs.setCurrentIndex(2)
    _settle(qtbot, panel)
    assert panel.height() < overview_height
    panel.tabs.setCurrentIndex(0)
    _settle(qtbot, panel)
    assert panel.height() == overview_height
    for index, (_, keys) in enumerate(panel._page_grids):
        panel.tabs.setCurrentIndex(index)
        _settle(qtbot, panel)
        scroll = panel.tabs.widget(index)
        assert scroll.verticalScrollBar().maximum() == 0
        assert scroll.horizontalScrollBar().maximum() == 0
        for key in keys:
            card = panel.plot_cards[key]
            assert (
                card.mapTo(scroll.widget(), card.rect().bottomRight()).y()
                < scroll.widget().height()
            )
    assert not panel._embedded_size_timer.isActive()


def test_embedded_summary_grows_for_wrapped_legend_and_optional_note(qtbot):
    _, panel = _host(qtbot)
    plot = panel.plots["counts"]
    card = panel.plot_cards["counts"]
    original_height = card.height()
    for index in range(9):
        plot.plot([0, 1], [index, index], name=f"Additional saved target {index}")
    _settle(qtbot, panel)
    assert card.height() > original_height
    legend_height = card.height()
    note = panel.plot_notes["counts"]
    note.setText("No saved outputs for the selected interval.")
    note.show()
    _settle(qtbot, panel)
    assert card.height() > legend_height


def test_embedded_summary_drops_narrow_height_when_returning_from_another_workspace(qtbot):
    from visualizer.src.sensing.sensing_window import SensingWindow

    window = SensingWindow()
    qtbot.addWidget(window)
    window._graph_height_box.setValue(180)
    window.resize(420, 1000)
    panel = window.run_summary_panel
    panel.set_summary(_data())
    window.show()
    _settle(qtbot, panel)
    window._set_view_mode("current")
    window.resize(1267, 1000)
    _settle(qtbot, panel)
    window._set_view_mode("summary")
    _settle(qtbot, panel)

    viewport = window.evidence_scroll.viewport()
    display = window.figure_options
    display_bottom = display.mapTo(viewport, display.rect().bottomLeft()).y()
    assert 0 < panel.mapTo(viewport, panel.rect().topLeft()).y() - display_bottom < 30
    assert not window._grid.hasHeightForWidth()
    card = panel.plot_cards["counts"]
    assert card.mapTo(viewport, card.rect().bottomRight()).y() < viewport.height()
    assert not panel._embedded_size_timer.isActive()
