"""Screenshot scale UI and actionable native-export failures."""

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QFileDialog, QWidget

from visualizer.src.panels.export_panel import ExportPanel
from visualizer.src.renderers.protocol import RendererCapabilities


@pytest.mark.parametrize("renderer_id", ["pygfx", "modern"])
def test_scale_tooltip_matches_backend_and_default_stays_one(qtbot, renderer_id):
    parent = QWidget()
    parent.renderer = SimpleNamespace(
        renderer_id=renderer_id,
        capabilities=RendererCapabilities(screenshot_export=True, viewport_hud=True),
    )
    qtbot.addWidget(parent)
    panel = ExportPanel(parent)
    group = panel.create_panel()
    qtbot.addWidget(group)
    spin = panel.widgets["screenshot_resolution_spin"]
    assert spin.value() == 1
    assert panel.widgets["include_hud_cb"].isChecked() is False
    if renderer_id == "pygfx":
        assert "additional scene detail" in spin.toolTip()
        assert "bloom" in spin.toolTip()
    else:
        assert "do not add scene detail" in spin.toolTip()


def test_cancel_does_not_export_and_failures_are_actionable(qtbot, monkeypatch):
    calls, messages = [], []
    parent = QWidget()
    parent.renderer = SimpleNamespace(
        renderer_id="pygfx",
        capabilities=RendererCapabilities(screenshot_export=True, viewport_hud=True),
        last_screenshot_error="Tiled export cannot preserve bloom; reduce the resolution",
        export_screenshot=lambda *args, **kw: calls.append((args, kw)) or False,
    )
    parent._set_status_message = lambda *args: messages.append(args)
    qtbot.addWidget(parent)
    panel = ExportPanel(parent)
    group = panel.create_panel()
    qtbot.addWidget(group)
    panel.widgets["screenshot_resolution_spin"].setValue(2)
    panel.widgets["include_hud_cb"].setChecked(True)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: ("", ""))
    panel._on_screenshot_clicked()
    assert not calls and not messages
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a: ("figure.png", ""))
    panel._on_screenshot_clicked()
    assert callable(calls[0][1].pop("progress_callback"))
    assert calls == [(("figure.png",), {"resolution_scale": 2.0, "include_hud": True})]
    assert messages[-1][0] == parent.renderer.last_screenshot_error


def test_scale_help_refreshes_after_renderer_initialization(qtbot):
    parent = QWidget()
    parent.renderer = None
    qtbot.addWidget(parent)
    panel = ExportPanel(parent)
    group = panel.create_panel()
    qtbot.addWidget(group)
    parent.renderer = SimpleNamespace(
        renderer_id="pygfx", capabilities=RendererCapabilities(screenshot_export=True)
    )
    panel.set_screenshot_export_enabled(True)
    assert "additional scene detail" in panel.widgets["screenshot_resolution_spin"].toolTip()


@pytest.mark.parametrize("outcome", [True, False, "exception"])
def test_progress_restores_controls_without_processing_events(qtbot, monkeypatch, outcome):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    parent = QWidget()
    qtbot.addWidget(parent)
    parent.renderer = SimpleNamespace(
        renderer_id="pygfx", capabilities=RendererCapabilities(screenshot_export=True)
    )
    panel = ExportPanel(parent)
    group = panel.create_panel()
    qtbot.addWidget(group)
    button = panel.widgets["screenshot_btn"]
    old_cursor = parent.cursor().shape()

    def disallow_events(*args):
        raise AssertionError("Must not process arbitrary events during frozen-frame export")

    def export(*args, **kwargs):
        assert not button.isEnabled()
        assert parent.cursor().shape() == Qt.WaitCursor
        kwargs["progress_callback"](0.5)
        assert button.text() == "Rendering 50%"
        if outcome == "exception":
            raise RuntimeError("injected export failure")
        return outcome

    parent.renderer.export_screenshot = export
    with monkeypatch.context() as patch:
        patch.setattr(QApplication, "processEvents", disallow_events)
        assert panel._export_high_resolution_screenshot(parent.renderer, "figure.png", 4, True) is (
            outcome is True
        )
    assert button.isEnabled()
    assert button.text() == "Screenshot"
    assert parent.cursor().shape() == old_cursor
