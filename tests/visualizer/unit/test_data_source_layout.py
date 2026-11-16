"""File-source diagnostics fit a sidebar and retain their wide-column layout."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from visualizer.src.app.theme import get_theme_manager
from visualizer.src.panels.data_source_panel import DataSourcePanel


@pytest.mark.parametrize("theme_mode", ["dark", "light"])
def test_file_diagnostics_stack_without_horizontal_overflow(qtbot, qapp, theme_mode):
    theme = get_theme_manager()
    previous_mode = theme.mode
    panel = DataSourcePanel(SimpleNamespace(frame_source=None))
    scroll = QScrollArea()
    qtbot.addWidget(scroll)
    try:
        theme.set_mode(theme_mode, persist=False)
        content = panel._create_file_mode_content()
        values = {
            "format_type": "HDF5 (Bulk) - 240 chunks",
            "file_count": "240",
            "chunk_info": "240 chunks, 0.8 MB, 0.2 MB, 245.3 MB...",
            "file_path": (
                "/example/research/recordings/long-directory-name/"
                "scenarios/public-example/multi-target-recording"
            ),
            "load_time": "On-demand",
            "preload_status": "In Progress (120/240, 50%)",
            "loading_mode": "Lazy Load",
            "memory_usage": "Not measured",
            "total_frames": "240",
            "loaded_frames": "120/240",
            "frame_range": "0 - 239",
            "file_size": "84.3 GB",
        }
        for key, value in values.items():
            panel.widgets[key].setText(value)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(content)
        layout.addStretch()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        scroll.show()
        qtbot.waitExposed(scroll)

        for width in (380, 900, 380):
            scroll.resize(width, 600)
            qtbot.waitUntil(lambda: scroll.width() == width)
            qapp.processEvents()
            assert scroll.width() == width
            assert scroll.horizontalScrollBar().maximum() == 0
            for key, value in values.items():
                label = panel.widgets[key]
                position = label.mapTo(content, QPoint(0, 0))
                assert label.text() == value
                assert label.isVisible()
                assert position.x() >= 0
                assert position.x() + label.width() <= content.width()
            format_position = panel.widgets["format_type"].mapTo(content, QPoint(0, 0))
            loading_position = panel.widgets["load_time"].mapTo(content, QPoint(0, 0))
            frames_position = panel.widgets["total_frames"].mapTo(content, QPoint(0, 0))
            if width == 380:
                assert format_position.y() < loading_position.y() < frames_position.y()
            else:
                assert format_position.x() < loading_position.x() < frames_position.x()
    finally:
        panel.cleanup()
        scroll.close()
        theme.set_mode(previous_mode, persist=False)
