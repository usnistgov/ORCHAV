"""Tests for the persistent controls-and-renderer Qt workspace."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QVBoxLayout, QWidget

from visualizer.src.app.renderer_lifecycle import (
    boot_visualizer_empty,
    stop_renderer_session,
)
from visualizer.src.app.viewport_workspace import (
    EmbeddedViewportHost,
    ViewportState,
    VisualizationWorkspace,
)
from visualizer.src.renderers.protocol import RendererCapabilities

PROJECT_ROOT = Path(__file__).resolve().parents[3]


_CONSTRUCTOR_PROBE = textwrap.dedent("""
    import sys

    from PySide6.QtWidgets import QApplication

    from visualizer.visualizer import OrchavVisualizer

    renderer_type, layout_profile = sys.argv[1:3]
    app = QApplication.instance() or QApplication([])
    visualizer = OrchavVisualizer(
        renderer_type=renderer_type,
        layout_profile=layout_profile,
        viewport_mode="auto",
    )
    try:
        assert visualizer._viewport_mode == "detached"
    finally:
        visualizer.close()
        visualizer.deleteLater()
        app.processEvents()
    """)


def _run_constructor_probe(renderer_type: str, layout_profile: str = "auto") -> None:
    """Exercise QApplication-wide shell setup in a clean native process."""
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    result = subprocess.run(
        [sys.executable, "-c", _CONSTRUCTOR_PROBE, renderer_type, layout_profile],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"visualizer constructor probe exited with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


class _Settings:
    def __init__(self) -> None:
        self.values = {}

    def value(self, key):
        return self.values.get(key)

    def setValue(self, key, value):  # noqa: N802 - QSettings-compatible double
        self.values[key] = value


def test_embedded_host_exposes_explicit_lifecycle_pages(qapp) -> None:
    host = EmbeddedViewportHost()

    assert host.state is ViewportState.EMPTY
    assert host.canvas_parent.objectName() == "embeddedViewportCanvasParent"

    host.set_loading_progress(3, 8, "Loading geometry…")
    assert host.state is ViewportState.LOADING

    host.set_state(ViewportState.ACTIVE)
    assert host.state is ViewportState.ACTIVE

    host.set_state(ViewportState.ERROR, "Renderer failed")
    assert host.state is ViewportState.ERROR
    assert "Renderer failed" in host._error_label.text()

    host.deleteLater()


def test_embedded_workspace_persists_only_splitter_shell_state(qapp) -> None:
    settings = _Settings()
    workspace = VisualizationWorkspace(
        QWidget(),
        embedded=True,
        settings=settings,
    )
    workspace.splitter.setSizes((360, 900))

    workspace.save_settings()

    assert VisualizationWorkspace._SPLITTER_SETTINGS_KEY in settings.values
    assert all("scenario" not in key for key in settings.values)
    workspace.deleteLater()


def test_detached_workspace_keeps_renderer_host_out_of_layout(qapp) -> None:
    workspace = VisualizationWorkspace(QWidget(), embedded=False, settings=_Settings())

    assert workspace.splitter.count() == 1
    assert workspace.viewport_host.isHidden()
    workspace.deleteLater()


def test_programmatic_open3d_constructor_resolves_auto_to_detached() -> None:
    _run_constructor_probe("open3d")


@pytest.mark.parametrize("layout_profile", ["capture-renderer", "capture-workspace"])
def test_programmatic_capture_layout_resolves_auto_to_detached(layout_profile: str) -> None:
    _run_constructor_probe("pygfx", layout_profile)


def test_renderer_session_uses_final_embedded_parent_and_preserves_host(qapp) -> None:
    shell = QWidget()
    shell.resize(1200, 700)
    shell_layout = QVBoxLayout(shell)
    host = EmbeddedViewportHost(shell)
    shell_layout.addWidget(host)
    shell.show()
    qapp.processEvents()
    calls = []
    renderer = SimpleNamespace(
        renderer_id="test",
        capabilities=RendererCapabilities(embedded_viewport=True),
        initialize_visualizer=lambda _title, width, height, **kwargs: (
            calls.append((width, height, kwargs["host_parent"])),
            kwargs["host_parent"],
        )[-1],
        set_background_color=lambda _color: None,
        resize=lambda width, height: calls.append(("resize", width, height)),
        close=lambda: calls.append(("close",)),
    )
    viz = SimpleNamespace(
        _viewport_mode="embedded",
        _viewport_host=host,
        _window_layout=None,
        _pending_camera=None,
        renderer=renderer,
        _on_viewport_resized=lambda width, height: renderer.resize(width, height),
    )

    boot_visualizer_empty(viz)

    assert calls[0][2] is host.canvas_parent
    assert calls[0][:2] == (host.canvas_parent.width(), host.canvas_parent.height())
    assert host.state is ViewportState.ACTIVE
    stop_renderer_session(viz)
    assert host.state is ViewportState.EMPTY
    assert host.canvas_parent is not None
    shell.close()
    shell.deleteLater()
