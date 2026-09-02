"""Performance contracts for the 2D trajectory preview."""

from __future__ import annotations

import time

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QVBoxLayout, QWidget

from visualizer.src.panels.trajectory_preview_panel import TrajectoryPreviewPanel
from visualizer.src.services.trajectory_load_service import (
    TrajectoryLoadCoordinator,
    TrajectorySnapshot,
)


def _snapshot(*, x_offset: float = 0.0) -> TrajectorySnapshot:
    return TrajectorySnapshot.from_mutable(
        {
            "tx_positions": {
                0: [
                    (0, x_offset + 0.0, 0.0, 1.0),
                    (1, x_offset + 1.0, 2.0, 1.0),
                    (2, x_offset + 2.0, 4.0, 1.0),
                ]
            },
            "rx_positions": {
                0: [
                    (0, x_offset + 10.0, 0.0, 1.0),
                    (1, x_offset + 11.0, 2.0, 1.0),
                    (2, x_offset + 12.0, 4.0, 1.0),
                ]
            },
            "target_positions": {
                "car": [
                    (0, x_offset + 20.0, 0.0, 0.0),
                    (1, x_offset + 21.0, 2.0, 0.0),
                    (2, x_offset + 22.0, 4.0, 0.0),
                ]
            },
            "frames_loaded": [0, 1, 2],
            "total_frames": 3,
        }
    )


def _process_events_until(condition, *, timeout: float = 1.0) -> None:
    """Pump Qt events until a UI condition holds without entering a nested loop."""
    deadline = time.monotonic() + timeout
    while not condition():
        QCoreApplication.processEvents()
        if time.monotonic() >= deadline:
            pytest.fail("Qt condition was not satisfied before the timeout")
        time.sleep(0.001)


@pytest.fixture
def visible_preview(qtbot):
    coordinator = TrajectoryLoadCoordinator()

    class _Parent:
        trajectory_load_coordinator = coordinator
        frame_source = None

    panel = TrajectoryPreviewPanel(_Parent())
    group = panel.create_panel()
    host = QWidget()
    QVBoxLayout(host).addWidget(group)
    qtbot.addWidget(host)
    host.show()
    _process_events_until(panel._is_plot_visible)
    if not panel._has_matplotlib:
        pytest.skip("Matplotlib is not available")

    panel._on_loading_complete(_snapshot())
    assert panel._static_plot_dirty is False
    yield panel, group, host

    panel.cleanup()
    coordinator.shutdown()


def _offsets(panel: TrajectoryPreviewPanel) -> np.ndarray:
    return np.asarray(panel._current_marker_artist.get_offsets(), dtype=float)


def test_frame_change_reuses_marker_without_clearing_axes(visible_preview, mocker) -> None:
    """Playback must not rebuild static lines, labels, or plot bounds."""
    panel, _group, _host = visible_preview
    marker = panel._current_marker_artist
    clear_spy = mocker.spy(panel._ax, "clear")
    rebuild_spy = mocker.spy(panel, "_rebuild_static_plot")
    panel._last_frame_update_at = 0.0

    panel.set_current_frame(1)

    assert panel._current_marker_artist is marker
    assert clear_spy.call_count == 0
    assert rebuild_spy.call_count == 0
    np.testing.assert_allclose(
        _offsets(panel),
        np.asarray([(1.0, 2.0), (11.0, 2.0), (21.0, 2.0)]),
    )


def test_trajectory_positions_are_indexed_by_frame(visible_preview) -> None:
    """Marker lookup should use the precomputed frame index, not scan tracks."""
    panel, _group, _host = visible_preview

    assert panel._frame_positions["tx"][2] == ((2.0, 4.0),)
    assert panel._frame_positions["rx"][1] == ((11.0, 2.0),)
    assert panel._frame_positions["target"][0] == ((20.0, 0.0),)


def test_hidden_preview_defers_frame_and_static_updates(visible_preview, mocker) -> None:
    """A hidden Analysis panel must not schedule Matplotlib repaint work."""
    panel, _group, host = visible_preview
    panel._last_frame_update_at = 0.0
    panel.set_current_frame(1)
    frame_one_offsets = _offsets(panel).copy()

    host.hide()
    _process_events_until(lambda: not panel._is_plot_visible())
    draw_spy = mocker.spy(panel.widgets["canvas"], "draw_idle")
    clear_spy = mocker.spy(panel._ax, "clear")

    panel.set_current_frame(2)
    panel.widgets["show_rx"].setChecked(False)

    assert draw_spy.call_count == 0
    assert clear_spy.call_count == 0
    assert panel._static_plot_dirty is True
    np.testing.assert_allclose(_offsets(panel), frame_one_offsets)

    host.show()
    _process_events_until(lambda: clear_spy.call_count == 1)

    assert panel._static_plot_dirty is False
    assert draw_spy.call_count == 1
    np.testing.assert_allclose(
        _offsets(panel),
        np.asarray([(2.0, 4.0), (22.0, 4.0)]),
    )


def test_hidden_partial_snapshots_index_only_latest_on_show(visible_preview, mocker) -> None:
    """Cumulative partial publications must not repeatedly index while hidden."""
    panel, _group, host = visible_preview
    host.hide()
    _process_events_until(lambda: not panel._is_plot_visible())
    index_spy = mocker.spy(panel, "_ensure_frame_position_index")

    panel._on_partial_update(_snapshot(x_offset=100.0))
    final_snapshot = _snapshot(x_offset=200.0)
    panel._on_loading_complete(final_snapshot)

    assert index_spy.call_count == 0
    assert panel._trajectories is final_snapshot
    assert panel._indexed_trajectories is not final_snapshot
    assert panel._static_plot_dirty is True

    host.show()
    _process_events_until(lambda: panel._indexed_trajectories is final_snapshot)

    assert index_spy.call_count == 1
    assert panel._static_plot_dirty is False
    assert panel._frame_positions["tx"][2] == ((202.0, 4.0),)
    assert panel._frame_positions["target"][0] == ((220.0, 0.0),)


def test_rapid_frame_changes_coalesce_to_latest_marker_update(visible_preview, mocker) -> None:
    """Playback faster than the plot budget should issue one latest-frame draw."""
    panel, _group, _host = visible_preview
    draw_spy = mocker.spy(panel.widgets["canvas"], "draw_idle")
    panel._last_frame_update_at = time.monotonic()

    panel.set_current_frame(1)
    panel.set_current_frame(2)

    assert draw_spy.call_count == 0
    assert panel._frame_update_timer.isActive()

    _process_events_until(lambda: draw_spy.call_count == 1)
    np.testing.assert_allclose(
        _offsets(panel),
        np.asarray([(2.0, 4.0), (12.0, 4.0), (22.0, 4.0)]),
    )
