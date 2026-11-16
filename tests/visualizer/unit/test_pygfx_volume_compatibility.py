"""Reject unsupported volume runtimes before loading data or allocating GPU resources."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pygfx
import pytest
from PySide6.QtWidgets import QWidget

from visualizer.src.coverage.volume_window import open_coverage_volume
from visualizer.src.panels.coverage_panel import CoverageMapPanel
from visualizer.src.panels.sensing_panel import SensingPanel
from visualizer.src.renderers.pygfx.renderer import PygfxRenderer
from visualizer.src.renderers.pygfx.volume import ensure_volume
from visualizer.src.sensing.volume_window import open_sensing_volume
from visualizer.src.types.render_payloads import VolumePayload


@pytest.mark.parametrize(
    "version,supported",
    [
        ("0.16.0", False),
        ("0.9.0", False),
        ("0.17.0rc1", False),
        ("0.17.0", True),
        ("0.17.1", True),
        ("0.17.0+local", True),
        ("0.18.0", False),
        ("1.0.0", False),
        ("unknown", False),
        ("", False),
    ],
)
def test_runtime_capabilities_follow_loaded_version(monkeypatch, version, supported):
    monkeypatch.setattr(pygfx, "__version__", version)
    renderer = PygfxRenderer(SimpleNamespace())
    assert renderer.capabilities.volume_rendering is supported
    reason = renderer.capabilities.volume_rendering_unavailable_reason
    assert bool(reason) is not supported
    if reason:
        assert ">=0.17,<0.18" in reason and (version or "unknown") in reason
    # Unrelated capabilities must remain accessible in older source environments.
    assert renderer.capabilities.trajectories and renderer.capabilities.pbr


@pytest.fixture
def old_renderer(monkeypatch):
    monkeypatch.setattr(pygfx, "__version__", "0.16.0")
    return PygfxRenderer(SimpleNamespace())


@pytest.mark.parametrize("kind", ["coverage", "sensing"])
def test_entry_points_reject_before_loading_or_closing_other_views(old_renderer, kind):
    old_window = SimpleNamespace(closed=False, close=Mock())
    parent = SimpleNamespace(renderer=old_renderer, coverage_volume_window=old_window)
    # No source metadata exists: rejection must precede source access.
    with pytest.raises(ValueError, match=r"pygfx >=0.17,<0.18.*0.16.0"):
        if kind == "coverage":
            open_coverage_volume(parent)
        else:
            open_sensing_volume(parent, "/does-not-exist")
    old_window.close.assert_not_called()


@pytest.mark.parametrize("mode", ["slice", "mip", "iso"])
def test_direct_renderer_rejects_before_native_allocation(old_renderer, mode, caplog):
    old_renderer._initialized = True
    old_renderer._scene = Mock()
    payload = VolumePayload(np.ones((2, 2, 2)), np.ones((2, 4)), (0, 1), mode=mode)
    assert not ensure_volume(old_renderer, "test", payload)
    assert "pygfx >=0.17,<0.18" in caplog.text and "0.16.0" in caplog.text
    old_renderer._scene.add.assert_not_called()
    assert not old_renderer._objects


@pytest.mark.parametrize("version,supported", [("0.16.0", False), ("0.17.0", True)])
def test_gui_controls_follow_volume_capability(monkeypatch, qtbot, version, supported):
    monkeypatch.setattr(pygfx, "__version__", version)
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.renderer = PygfxRenderer(SimpleNamespace())
    coverage = CoverageMapPanel(parent)
    coverage_group = coverage.create_panel()
    qtbot.addWidget(coverage_group)
    coverage.update_coverage_status(
        True,
        {"coverage_file": "saved.h5", "heights": [1, 2], "metric_name": "path_gain_db"},
    )
    parent.coverage_data = {"coverage_file": "saved.h5", "heights": [1, 2]}
    reason = coverage.volume_view.availability()
    assert (not reason) is supported
    if not supported:
        assert version in reason
    assert coverage.widgets["coverage_toggle"].isEnabled()

    sensing = SensingPanel(parent)
    sensing._start_auto_refresh = lambda: None
    sensing_group = sensing.create_panel()
    qtbot.addWidget(sensing_group)
    action = sensing.widgets["open_volume_action"]
    assert action.isEnabled() is supported
    if not supported:
        assert version in action.toolTip()
    assert sensing.widgets["show_sensing_detections"].isEnabled()


def test_sensing_menu_updates_after_deferred_renderer_init(monkeypatch, qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    panel = SensingPanel(parent)
    panel._start_auto_refresh = lambda: None
    group = panel.create_panel()
    qtbot.addWidget(group)
    assert not panel.widgets["open_volume_action"].isEnabled()
    monkeypatch.setattr(pygfx, "__version__", "0.17.0")
    parent.renderer = PygfxRenderer(SimpleNamespace())
    panel._update_volume_availability()
    assert panel.widgets["open_volume_action"].isEnabled()


def test_cli_rejects_unsupported_volume_before_opening_scenario(old_renderer):
    from visualizer.src.app.startup_workflow import _complete_cli_startup, parse_cli_args

    args = parse_cli_args(["--no-resume", "--sensing-volume", "/does-not-exist"])
    parent = SimpleNamespace(renderer=old_renderer, _deferred_init=Mock(), open_scenario=Mock())
    with pytest.raises(ValueError, match=r"pygfx >=0.17,<0.18.*0.16.0"):
        _complete_cli_startup(args=args, app=Mock(), reporter=Mock(), viz=parent)
    parent.open_scenario.assert_not_called()
