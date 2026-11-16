"""Panel/floating placement shares analysis state and releases all Qt owners."""

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QTimer
from PySide6.QtWidgets import QScrollArea, QTabWidget, QWidget
from shiboken6 import isValid

from tests.visualizer.unit.test_sensing_dashboard_recording import _request
from visualizer.src.app.theme import get_theme_manager
from visualizer.src.panels.sensing_panel import SensingPanel


@pytest.fixture
def embedded(qtbot, monkeypatch):
    QSettings().remove("layout/sections/sensing_results")
    QSettings().remove("layout/sections/sensing_options")
    panel = SensingPanel(SimpleNamespace(renderer=None))
    # Placement tests do not start a provider; feed a typed saved result directly.
    monkeypatch.setattr(panel, "_refresh_live_plots", lambda *args, **kwargs: None)
    monkeypatch.setattr(panel, "_on_linked_selection", lambda selection: None)
    tabs = QTabWidget()
    tabs.addTab(QWidget(), "Scene")
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(panel.create_panel())
    tabs.addTab(scroll, "Sensing")
    tabs.resize(460, 950)
    qtbot.addWidget(tabs)
    tabs.show()
    yield panel, tabs, scroll
    panel.cleanup()
    QSettings().remove("layout/sections/sensing_results")
    QSettings().remove("layout/sections/sensing_options")
    tabs.close()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_analysis_is_created_on_first_visit_and_reused_when_tabs_change(embedded, qtbot):
    panel, tabs, _ = embedded
    assert panel._sensing_window is None
    tabs.setCurrentIndex(1)
    qtbot.waitUntil(lambda: panel._dashboard_is_visible())
    dashboard = panel._sensing_window
    request = _request()
    dashboard.update_from_request(request)
    tabs.setCurrentIndex(0)
    assert not panel._dashboard_is_visible()
    assert isValid(dashboard)
    tabs.setCurrentIndex(1)
    assert panel._sensing_window is dashboard
    assert dashboard._last_request is request


def test_popout_return_and_window_close_preserve_data_selection_and_zoom(embedded, qtbot):
    panel, tabs, _ = embedded
    tabs.setCurrentIndex(1)
    dashboard = panel._sensing_window
    request = _request()
    dashboard.update_from_request(request)
    dashboard._select_inspection_coordinate(1, 1)
    selected = dashboard._linked_selection
    assert selected is not None
    dashboard.rd_plot.setRange(xRange=(0.5, 2.0), yRange=(-1, 1), padding=0)
    ranges = dashboard.rd_plot.viewRange()
    image = dashboard.rd_image.image
    for _ in range(2):
        dashboard.placement_button.click()
        shell = panel.dashboard_host.floating_window
        qtbot.waitUntil(shell.isVisible)
        assert panel._dashboard_is_visible()
        tabs.setCurrentIndex(0)
        assert panel._dashboard_is_visible()  # A visible pop-out keeps receiving results.
        assert dashboard._last_request is request
        assert dashboard.rd_image.image is image
        assert dashboard._linked_selection is selected
        np.testing.assert_allclose(dashboard.rd_plot.viewRange(), ranges)
        shell.close()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert isValid(dashboard) and not getattr(dashboard, "_plots_released", False)
        assert panel._sensing_window is dashboard
        tabs.setCurrentIndex(1)
        assert dashboard.parentWidget() is panel.dashboard_host
        np.testing.assert_allclose(dashboard.rd_plot.viewRange(), ranges)


def test_narrow_layout_stacks_plots_and_wide_layout_restores_columns(embedded, qtbot):
    panel, tabs, scroll = embedded
    tabs.setCurrentIndex(1)
    dashboard = panel._sensing_window
    dashboard._set_view_mode("analysis")
    dashboard.update_from_request(_request())
    qtbot.waitUntil(lambda: dashboard._compact)
    grid = dashboard._grid
    assert grid.getItemPosition(grid.indexOf(dashboard.rp_plot))[:2] == (1, 0)
    assert scroll.horizontalScrollBar().maximum() == 0
    dashboard.rd_plot.setTitle(
        "A long range–Doppler title must leave room for the power color scale"
    )
    qtbot.waitUntil(
        lambda: dashboard.rd_colorbar.sceneBoundingRect().right() <= dashboard.rd_plot.width()
    )
    tabs.resize(1150, 950)
    qtbot.waitUntil(lambda: not dashboard._compact)
    assert grid.getItemPosition(grid.indexOf(dashboard.rp_plot))[:2] == (0, 1)


@pytest.mark.parametrize("theme", [pytest.param(None, id="platform"), "light", "dark"])
def test_rd_legend_and_wrapped_title_stay_outside_data_when_resizing(embedded, qtbot, theme):
    panel, tabs, scroll = embedded
    tabs.setCurrentIndex(1)
    dashboard = panel._sensing_window
    dashboard._set_view_mode("analysis")
    dashboard.update_from_request(_request())
    plot = dashboard.rd_plot.getPlotItem()
    legend = plot.legend

    def check_layout():
        bounds = dashboard.rd_plot.sceneRect()
        title = plot.titleLabel.item.sceneBoundingRect()
        assert bounds.contains(title)
        assert title.bottom() <= legend.sceneBoundingRect().top()
        assert legend.sceneBoundingRect().bottom() <= plot.vb.sceneBoundingRect().top()
        assert plot.vb.height() >= 160
        for sample, label in legend.items:
            assert bounds.contains(sample.sceneBoundingRect())
            assert bounds.contains(label.item.sceneBoundingRect())
        assert scroll.horizontalScrollBar().maximum() == 0

    manager = get_theme_manager()
    original_theme = manager.mode
    try:
        if theme is not None:
            manager.set_mode(theme, persist=False)
        dashboard.rd_plot.setTitle(
            "Range–Velocity<br><span style='font-size: 9pt'>"
            "Range window: Hann · Doppler window: Hann</span>"
        )
        for width in (360, 1150, 360):
            tabs.resize(width, 950)
            qtbot.waitUntil(check_layout)
        assert legend.rowCount > 1
        dashboard._select_inspection_coordinate(1, 2)
        qtbot.waitUntil(check_layout)
    finally:
        if theme is not None:
            manager.set_mode(original_theme, persist=False)


@pytest.mark.parametrize(
    "mode, names, positions",
    [
        (
            "analysis",
            ("rd_plot", "rp_plot", "gt_range_plot", "gt_speed_plot"),
            ((0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1)),
        ),
        (
            "tracking",
            ("tracking_plot", "gt_range_plot", "gt_speed_plot"),
            ((0, 0, 1, 2), (1, 0, 1, 1), (1, 1, 1, 1)),
        ),
        (
            "angular",
            ("rd_plot", "range_azimuth_plot", "doppler_azimuth_plot", "angular_spectrum_plot"),
            ((0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1)),
        ),
    ],
)
def test_workspace_shows_all_figures_without_companion_selector(
    embedded, qtbot, mode, names, positions
):
    panel, tabs, scroll = embedded
    tabs.setCurrentIndex(1)
    dashboard = panel._sensing_window
    dashboard._set_view_mode(mode)
    tabs.resize(1150, 950)
    qtbot.waitUntil(lambda: not dashboard._compact)
    for name, position in zip(names, positions):
        plot = getattr(dashboard, name)
        assert plot.isVisible()
        assert dashboard._grid.getItemPosition(dashboard._grid.indexOf(plot)) == position
    assert dashboard._support_box.isHidden()
    tabs.resize(420, 950)
    qtbot.waitUntil(lambda: dashboard._compact)
    for row, name in enumerate(names):
        plot = getattr(dashboard, name)
        assert plot.isVisible()
        assert dashboard._grid.getItemPosition(dashboard._grid.indexOf(plot)) == (row, 0, 1, 1)
    qtbot.waitUntil(lambda: scroll.horizontalScrollBar().maximum() == 0)


def test_hidden_analysis_suspends_polling_without_resetting_session(embedded, monkeypatch):
    panel, tabs, _ = embedded
    tabs.setCurrentIndex(1)
    session = SimpleNamespace(source_identity="recording", provider_identity=lambda _: "recording")
    panel._dashboard_session = session
    panel._dashboard_poll_timer = QTimer(panel._container)
    panel._dashboard_poll_timer.start(25)
    monkeypatch.setattr(
        panel, "_retire_dashboard_session", lambda: pytest.fail("retired unchanged source")
    )
    tabs.setCurrentIndex(0)
    panel._poll_dashboard()
    assert not panel._dashboard_poll_timer.isActive()
    assert panel._dashboard_session is session
    panel._dashboard_session = None
    monkeypatch.undo()


def test_cleanup_closes_floating_shell_and_releases_plots(embedded):
    panel, tabs, _ = embedded
    tabs.setCurrentIndex(1)
    dashboard = panel._sensing_window
    panel.dashboard_host.detach()
    shell = panel.dashboard_host.floating_window
    panel.cleanup()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not isValid(dashboard)
    assert not isValid(shell)
    assert panel._sensing_window is None


@pytest.mark.parametrize("floating", [False, True])
def test_minimized_host_suspends_analysis_and_restores_existing_state(embedded, qtbot, floating):
    panel, tabs, _ = embedded
    tabs.setCurrentIndex(1)
    dashboard = panel._sensing_window
    request = _request()
    dashboard.update_from_request(request)
    dashboard._set_view_mode("micro-doppler")
    if floating:
        panel.dashboard_host.detach()
        shell = panel.dashboard_host.floating_window
    else:
        shell = tabs
    shell.showMinimized()
    assert not panel._dashboard_is_visible()
    assert dashboard._micro_doppler_defer_reason() == "deferred (window minimized)"
    shell.showNormal()
    qtbot.waitUntil(panel._dashboard_is_visible)
    assert dashboard._last_request is request


def test_result_graphs_can_collapse_without_losing_results_or_zoom(embedded, qtbot, monkeypatch):
    panel, tabs, _ = embedded
    tabs.setCurrentIndex(1)
    panel.options_section.expand()
    dashboard = panel._sensing_window
    request = _request()
    dashboard.update_from_request(request)
    dashboard.rd_plot.setRange(xRange=(0.5, 2), yRange=(-1, 1), padding=0)
    ranges = dashboard.rd_plot.viewRange()
    pause = []
    monkeypatch.setattr(panel, "_pause_dashboard_updates", lambda: pause.append(True))
    panel.results_section.collapse()
    assert not panel._dashboard_is_visible()
    assert pause == [True]
    assert panel.widgets["sensing_playback"].isVisible()
    panel.results_section.expand()
    qtbot.waitUntil(panel._dashboard_is_visible)
    assert panel._sensing_window is dashboard
    assert dashboard._last_request is request
    np.testing.assert_allclose(dashboard.rd_plot.viewRange(), ranges)


def test_collapsed_sensing_options_leave_result_graphs_visible_and_updating(
    embedded, qtbot, monkeypatch
):
    panel, tabs, _ = embedded
    tabs.setCurrentIndex(1)
    qtbot.waitUntil(panel._dashboard_is_visible)
    dashboard = panel._sensing_window
    assert not panel.options_section.is_expanded()
    assert panel.results_section.is_expanded()
    assert not panel.widgets["sensing_playback"].isVisible()
    assert not panel.widgets["show_sensing_detections"].isVisible()
    assert not panel.widgets["load_sensing_figures"].isVisible()
    panel.options_section.expand()
    assert panel.widgets["sensing_playback"].isVisible()
    assert panel.widgets["load_sensing_figures"].isVisible()
    dashboard.update_from_request(_request(0))
    paused = []
    monkeypatch.setattr(panel, "_pause_dashboard_updates", lambda: paused.append(True))
    panel.options_section.collapse()
    assert not paused
    assert panel._dashboard_is_visible()
    request = _request(1)
    dashboard.update_from_request(request)
    assert dashboard._last_request is request
    assert dashboard._rendered_request is request
    assert panel._sensing_window is dashboard


@pytest.mark.parametrize("expanded", [False, True])
def test_sensing_options_restore_independently_from_result_graphs(
    qtbot, monkeypatch, tmp_path, expanded
):
    from visualizer.src.panels import sensing_panel

    settings = QSettings(str(tmp_path / "sensing-options.ini"), QSettings.IniFormat)
    monkeypatch.setattr(sensing_panel, "QSettings", lambda: settings)
    panels = []
    try:
        for _ in range(2):
            panel = SensingPanel(SimpleNamespace(renderer=None))
            monkeypatch.setattr(panel, "_refresh_live_plots", lambda *args, **kwargs: None)
            widget = panel.create_panel()
            qtbot.addWidget(widget)
            panels.append(panel)
            if len(panels) == 1:
                panel.options_section.expand() if expanded else panel.options_section.collapse()
            else:
                assert panel.options_section.is_expanded() is expanded
            assert panel.results_section.is_expanded()
        assert settings.value("layout/sections/sensing_options", type=bool) is expanded
    finally:
        for panel in panels:
            panel.cleanup()


def test_initially_collapsed_results_do_not_create_plots(qtbot, monkeypatch):
    QSettings().setValue("layout/sections/sensing_results", False)
    panel = SensingPanel(SimpleNamespace(renderer=None))
    monkeypatch.setattr(panel, "_refresh_live_plots", lambda *args, **kwargs: None)
    widget = panel.create_panel()
    qtbot.addWidget(widget)
    try:
        widget.show()
        QCoreApplication.processEvents()
        assert panel._sensing_window is None
        assert not panel.results_section.is_expanded()
        panel._on_open_dashboard()
        assert panel.results_section.is_expanded()
        assert panel._sensing_window is not None
    finally:
        panel.cleanup()
        QSettings().remove("layout/sections/sensing_results")
