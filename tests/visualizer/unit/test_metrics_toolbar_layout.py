"""Metrics refresh and figure controls stay grouped and aligned in narrow panels."""

import pytest
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtWidgets import QLabel

from visualizer.src.app.theme import ThemeManager
from visualizer.src.metrics.viz_metrics import MetricsDashboard


@pytest.mark.parametrize("width", [380, 1100])
def test_live_metrics_controls_wrap_without_overflow_and_align_refresh_buttons(qtbot, width):
    dashboard = MetricsDashboard()
    dashboard.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    qtbot.addWidget(dashboard)
    dashboard.resize(width, 900)
    dashboard.show()
    qtbot.waitUntil(lambda: dashboard._refresh_controls_row.width() > 100)

    refresh_row = dashboard._refresh_controls_row
    plot_row = dashboard._plot_controls_row
    export_row = dashboard._export_controls_row
    refresh_controls = (dashboard.pause_cb, dashboard.refresh_btn, dashboard.update_rate_spin)
    plot_controls = (
        dashboard.auto_range_cb,
        dashboard.log_y_cb,
        dashboard.overview_grid_cb,
        dashboard.settings_toggle,
    )
    for row, controls in (
        (refresh_row, refresh_controls),
        (plot_row, plot_controls),
    ):
        for control in controls:
            assert row.isAncestorOf(control)
            top_left = control.mapTo(row, QPoint(0, 0))
            assert top_left.x() >= 0
            assert top_left.x() + control.width() <= row.width()
            assert top_left.y() >= 0
            assert top_left.y() + control.height() <= row.height()

    freeze_center = dashboard.pause_cb.mapTo(refresh_row, dashboard.pause_cb.rect().center())
    refresh_center = dashboard.refresh_btn.mapTo(refresh_row, dashboard.refresh_btn.rect().center())
    assert abs(freeze_center.y() - refresh_center.y()) <= 1
    assert refresh_row.geometry().bottom() < plot_row.geometry().top()
    assert dashboard.settings_drawer.isAncestorOf(export_row)
    assert export_row.isAncestorOf(dashboard.export_csv_btn)
    assert export_row.isAncestorOf(dashboard.export_png_btn)
    dashboard.close()


@pytest.mark.parametrize("width", [380, 1100])
def test_metrics_controls_resize_after_construction_and_headings_follow_first_line(
    qtbot, qapp, tmp_path, width
):
    previous_style, previous_palette = qapp.styleSheet(), qapp.palette()
    manager = ThemeManager(QSettings(str(tmp_path / "metrics-theme.ini"), QSettings.IniFormat))
    manager.install(qapp, mode="light")
    dashboard = MetricsDashboard()
    dashboard.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    qtbot.addWidget(dashboard)
    dashboard.resize(width, 900)
    dashboard.show()
    dashboard.settings_toggle.setChecked(True)
    dashboard._channel_pair_control.show()
    rows = (
        (
            dashboard._refresh_controls_row,
            (dashboard.pause_cb, dashboard.refresh_btn, dashboard.update_rate_spin.parentWidget()),
        ),
        (
            dashboard._plot_controls_row,
            (
                dashboard.auto_range_cb,
                dashboard.log_y_cb,
                dashboard.overview_grid_cb,
                dashboard._channel_pair_control,
                dashboard.settings_toggle,
            ),
        ),
        (dashboard._export_controls_row, (dashboard.export_csv_btn, dashboard.export_png_btn)),
    )
    try:
        for mode, scale in [("light", 1.0), ("dark", 1.3), ("light", 1.0)]:
            manager.set_mode(mode, persist=False)
            manager.set_interface_scale(scale, persist=False)
            qapp.processEvents()
            for row, controls in rows:
                heading = row.findChild(QLabel, "scope")
                first = controls[0]
                heading_y = heading.mapTo(row, heading.rect().center()).y()
                first_y = first.mapTo(row, first.rect().center()).y()
                assert abs(heading_y - first_y) <= 1
                for control in controls:
                    if control.isHidden():
                        continue
                    assert control.height() >= control.minimumSizeHint().height()
            for control in (dashboard.update_rate_spin, dashboard.pair_metric_combo):
                wrapper = control.parentWidget()
                position = control.mapTo(wrapper, QPoint())
                assert position.y() >= 0
                assert position.y() + control.height() <= wrapper.height()
                assert control.height() >= control.minimumSizeHint().height()
    finally:
        dashboard.close()
        qapp.removeEventFilter(manager)
        qapp.setStyleSheet(previous_style)
        qapp.setPalette(previous_palette)
