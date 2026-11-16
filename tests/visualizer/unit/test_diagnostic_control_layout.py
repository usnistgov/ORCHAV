"""Diagnostic captions and signed cutaway values fit after interface scaling."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QVBoxLayout, QWidget

from visualizer.src.app.theme import ThemeManager, generate_application_stylesheet
from visualizer.src.panels.collapsible_section import CollapsibleSection
from visualizer.src.panels.data_source_panel import DataSourcePanel
from visualizer.src.panels.render_panel import RenderPanel


def _style(workspace, scale):
    manager = ThemeManager()
    manager._scale = scale
    workspace.setStyleSheet(manager._scaled_stylesheet(generate_application_stylesheet()))


def test_live_headers_fit_theme_changes_without_resizing_for_new_rows(qtbot, qapp):
    workspace = QWidget()
    qtbot.addWidget(workspace)
    workspace.setProperty("workflowWorkspace", True)
    layout = QVBoxLayout(workspace)
    panel = DataSourcePanel(SimpleNamespace(frame_source=None))
    performance = panel._live_grpc_panel._performance
    layout.addWidget(performance._create_telemetry_group())
    layout.addWidget(performance._create_connection_diagnostics_group())
    for section in workspace.findChildren(CollapsibleSection):
        section.expand()
    tables = [panel.widgets[key] for key in ("request_history_table", "connection_history_table")]
    workspace.resize(700, 900)
    workspace.show()
    try:
        for scale in (1.0, 1.3):
            _style(workspace, scale)
            for table in tables:
                header = table.horizontalHeader()
                qtbot.waitUntil(
                    lambda h=header: all(
                        h.sectionSize(i) >= h.sectionSizeHint(i) for i in range(h.count())
                    )
                )
        tables[0].horizontalHeader().resizeSection(0, 300)
        _style(workspace, 1.0)
        qapp.processEvents()
        assert tables[0].horizontalHeader().sectionSize(0) == 300
        widths = [
            [table.columnWidth(column) for column in range(table.model().columnCount())]
            for table in tables
        ]
        panel.widgets["request_history_model"].set_requests(
            [dict(request_timestamp=1, frame_idx=2, success=True, generation_time_ms=3)]
        )
        panel.widgets["connection_history_model"].set_events(
            [(1, "Connected", "Success", "Local test fixture", "1 s")]
        )
        qapp.processEvents()
        assert all(table.model().rowCount() == 1 for table in tables)
        assert [
            [table.columnWidth(column) for column in range(table.model().columnCount())]
            for table in tables
        ] == widths
    finally:
        panel.cleanup()


@pytest.mark.parametrize("scale", [1.0, 1.3])
def test_cutaway_fields_show_full_signed_range_and_units(qtbot, qapp, scale):
    workspace = QWidget()
    qtbot.addWidget(workspace)
    workspace.setProperty("workflowWorkspace", True)
    _style(workspace, scale)
    layout = QVBoxLayout(workspace)
    panel = RenderPanel(SimpleNamespace(renderer=None))
    layout.addWidget(panel.create_scene_view_panel())
    workspace.resize(600, 800)
    workspace.show()
    qapp.processEvents()
    for axis in "xyz":
        spin = panel.widgets[f"clip_{axis}_spin"]
        for value in (spin.minimum(), spin.maximum()):
            with QSignalBlocker(spin):
                spin.setValue(value)
            qapp.processEvents()
            editor = spin.lineEdit()
            assert (
                editor.fontMetrics().horizontalAdvance(editor.text())
                <= editor.contentsRect().width()
            )
            assert spin.suffix() == " m"
