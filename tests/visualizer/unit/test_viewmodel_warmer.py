"""Tests for ViewModelWarmer — verifies cross-thread signal delivery and cache warming."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from visualizer.src.pipeline.core import FrameRenderPacket, ViewModel
from visualizer.src.pipeline.frame_pipeline import build_vm_cache_key
from visualizer.src.services.cache_service import CacheService
from visualizer.src.services.viewmodel_warmer import ViewModelWarmer
from visualizer.src.state import MpcVisibility, create_initial_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_view_model(**overrides: Any) -> ViewModel:
    defaults = dict(
        tx_positions=np.array([[0.0, 0.0, 0.0]]),
        rx_positions=np.array([[1.0, 0.0, 0.0]]),
        tx_orientations=np.zeros((1, 3)),
        rx_orientations=np.zeros((1, 3)),
        mpc_points=np.zeros((0, 3), dtype=np.float32),
        mpc_lines=np.zeros((0, 2), dtype=np.int32),
        mpc_colors=np.zeros((0, 3), dtype=np.float32),
        colorbar=None,
        stats_text="stats",
        mpc_visibility=MpcVisibility(),
        target_positions=np.empty((0, 3)),
        target_orientations=np.empty((0, 3)),
        target_mesh_files=[],
        target_use_ply_positions=[],
        target_metadata=[],
    )
    defaults.update(overrides)
    return ViewModel(**defaults)


class DummyMpcCore:
    """Stand-in for MPCCore that returns a fixed ViewModel."""

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.raw_frames: list[dict[str, Any]] = []
        self.return_value: Optional[ViewModel] = _make_view_model()

    def create_view_model(self, **kwargs: Any) -> Optional[ViewModel]:
        self.calls.append(kwargs.get("step", -1))
        self.raw_frames.append(kwargs["raw_frame"])
        return self.return_value


def _make_raw_frame(step: int) -> dict[str, Any]:
    return {
        "step": step,
        "tx_positions": [[0, 0, 0]],
        "rx_positions": [[1, 0, 0]],
        "all_padded_vertices": [],
        "all_padded_interactions": [],
        "tx_rx_pairs": [],
    }


class _DummyObservableCache(OrderedDict):
    """Minimal observable cache matching the real _ObservableCache API."""

    def __init__(self) -> None:
        super().__init__()
        self._warmer: Optional[ViewModelWarmer] = None

    def set_warmer(self, warmer: ViewModelWarmer) -> None:
        self._warmer = warmer

    def clear(self) -> None:
        super().clear()
        if self._warmer is not None:
            self._warmer.invalidate(reason="cache cleared")


@pytest.fixture
def dummy_viz() -> SimpleNamespace:
    """Build a minimal visualizer stand-in with cache and mpc_core."""
    mpc_core = DummyMpcCore()
    state = create_initial_state(step=0)

    viz = SimpleNamespace(
        app_state=state,
        mpc_core=mpc_core,
        mpc_view_cache=_DummyObservableCache(),
        mpc_allowed_materials=None,
        show_tx_segments=True,
    )
    cache_service = CacheService(viz, max_frame_cache_size=100)
    viz.cache_service = cache_service
    return viz


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_vm_cache_key_deterministic(dummy_viz: SimpleNamespace) -> None:
    """build_vm_cache_key returns the same key for the same inputs."""
    state = dummy_viz.app_state
    k1 = build_vm_cache_key(5, state, None)
    k2 = build_vm_cache_key(5, state, None)
    assert k1 == k2
    # Different step → different key
    k3 = build_vm_cache_key(6, state, None)
    assert k1 != k3


def test_build_vm_cache_key_changes_for_topk_state(dummy_viz: SimpleNamespace) -> None:
    """Top-K state must participate in cache-key generation."""
    state = dummy_viz.app_state
    key_base = build_vm_cache_key(2, state, None)

    topk_on = replace(state, topk_render_enabled=True)
    key_topk_on = build_vm_cache_key(2, topk_on, None)
    assert key_topk_on != key_base

    topk_cap_changed = replace(topk_on, topk_render_max_paths=7777)
    key_topk_cap_changed = build_vm_cache_key(2, topk_cap_changed, None)
    assert key_topk_cap_changed != key_topk_on


def test_build_vm_cache_key_ignores_service_owned_label_visibility(
    dummy_viz: SimpleNamespace,
) -> None:
    """TX/RX label visibility must not invalidate frame-heavy ViewModels."""
    state = dummy_viz.app_state
    key_base = build_vm_cache_key(2, state, None)

    labels_hidden = replace(state, show_labels=not state.show_labels)
    key_labels_hidden = build_vm_cache_key(2, labels_hidden, None)

    assert key_labels_hidden == key_base


def test_warmer_enqueue_and_process(dummy_viz: SimpleNamespace) -> None:
    """Warmer creates ViewModels for enqueued steps when the timer ticks."""
    # Store some raw frames in the cache first
    for step in range(5):
        dummy_viz.cache_service.store_frame(step, _make_raw_frame(step))

    warmer = ViewModelWarmer(dummy_viz)
    dummy_viz.mpc_view_cache.set_warmer(warmer)

    # Enqueue steps
    for step in range(5):
        warmer.enqueue(step)

    assert warmer._active
    assert len(warmer._queue) > 0 or warmer._warmed_count > 0

    # Simulate timer ticks by calling _process_tick directly
    for _ in range(10):
        warmer._process_tick()
        if not warmer._queue:
            break

    # All 5 ViewModels should now be in the cache
    assert dummy_viz.mpc_core.calls == list(range(5))
    assert len(dummy_viz.mpc_view_cache) == 5

    # Session stays active after queue drains (more signals may be in-flight)
    assert warmer._active
    assert warmer._warmed_count == 5

    # Verify the cache keys match what _derive_view_model would produce
    state = dummy_viz.app_state
    for step in range(5):
        key = build_vm_cache_key(step, state, None)
        assert key in dummy_viz.mpc_view_cache, f"Step {step} not in cache"

    # Only stop() ends the session
    warmer.stop()
    assert not warmer._active


def test_warmer_skips_already_cached(dummy_viz: SimpleNamespace) -> None:
    """Warmer skips steps that are already in the ViewModel cache."""
    dummy_viz.cache_service.store_frame(0, _make_raw_frame(0))
    dummy_viz.cache_service.store_frame(1, _make_raw_frame(1))

    warmer = ViewModelWarmer(dummy_viz)

    # Pre-populate cache for step 0
    key0 = build_vm_cache_key(0, dummy_viz.app_state, None)
    dummy_viz.mpc_view_cache[key0] = _make_view_model()

    warmer.enqueue(0)
    warmer.enqueue(1)
    warmer._process_tick()
    warmer._process_tick()

    # Only step 1 should have called create_view_model (step 0 was cached)
    assert dummy_viz.mpc_core.calls == [1]


def test_warmer_prewarms_renderer_cache_for_new_and_cached_vms(
    dummy_viz: SimpleNamespace,
) -> None:
    """Renderer-side caches are warmed from both new and cached ViewModels."""
    dummy_viz.cache_service.store_frame(0, _make_raw_frame(0))
    calls: list[FrameRenderPacket] = []

    class _Renderer:
        def prewarm_mpc_line_cache(self, packet: FrameRenderPacket) -> bool:
            calls.append(packet)
            return True

    dummy_viz.renderer = _Renderer()
    warmer = ViewModelWarmer(dummy_viz)

    warmer.enqueue(0)
    warmer._process_tick()

    assert dummy_viz.mpc_core.calls == [0]
    assert len(calls) == 1
    assert calls[0].mpc_points is dummy_viz.mpc_core.return_value.mpc_points
    assert warmer._renderer_warmed_count == 1

    # Re-enqueueing the same step should skip ViewModel construction but still
    # give the renderer a chance to fill any cache that was cleared separately.
    warmer.enqueue(0)
    warmer._process_tick()

    assert dummy_viz.mpc_core.calls == [0]
    assert len(calls) == 2
    assert calls[1].mpc_points is dummy_viz.mpc_core.return_value.mpc_points
    assert warmer._renderer_warmed_count == 2


def test_warmer_does_not_write_view_options_into_raw_frame_cache(
    dummy_viz: SimpleNamespace,
) -> None:
    """Prewarming must not turn derived view settings into frame data."""
    raw_frame = _make_raw_frame(0)
    dummy_viz.cache_service.store_frame(0, raw_frame)
    cached_frame = dummy_viz.cache_service.get_frame(0)
    warmer = ViewModelWarmer(dummy_viz)

    warmer.enqueue(0)
    warmer._process_tick()

    assert "standalone_beamforming_mode" not in cached_frame
    assert dummy_viz.mpc_core.raw_frames[-1] is not cached_frame
    assert dummy_viz.mpc_core.raw_frames[-1]["standalone_beamforming_mode"] == "frame"


def test_warmer_skips_visible_standalone_beamforming(
    dummy_viz: SimpleNamespace,
) -> None:
    """Standalone beam results must be computed by the regular on-demand path."""
    dummy_viz.app_state = replace(
        dummy_viz.app_state,
        show_beamforming=True,
        standalone_beamforming_mode="standalone",
    )
    dummy_viz.cache_service.store_frame(0, _make_raw_frame(0))
    warmer = ViewModelWarmer(dummy_viz)

    warmer.enqueue(0)

    assert not warmer._active
    assert not warmer._queue
    assert dummy_viz.mpc_core.calls == []
    assert len(dummy_viz.mpc_view_cache) == 0


def test_warmer_keeps_visible_frame_beamforming(
    dummy_viz: SimpleNamespace,
) -> None:
    """Frame-backed beams remain eligible for safe ViewModel warming."""
    dummy_viz.app_state = replace(
        dummy_viz.app_state,
        show_beamforming=True,
        standalone_beamforming_mode="frame",
    )
    dummy_viz.cache_service.store_frame(0, _make_raw_frame(0))
    warmer = ViewModelWarmer(dummy_viz)

    warmer.enqueue(0)
    warmer._process_tick()

    key = build_vm_cache_key(0, dummy_viz.app_state, None)
    assert dummy_viz.mpc_core.calls == [0]
    assert key in dummy_viz.mpc_view_cache
    assert dummy_viz.mpc_core.raw_frames[-1]["standalone_beamforming_mode"] == "frame"


def test_warmer_invalidate_clears_queue(dummy_viz: SimpleNamespace) -> None:
    """invalidate() clears the queue and stops the timer."""
    for step in range(5):
        dummy_viz.cache_service.store_frame(step, _make_raw_frame(step))

    warmer = ViewModelWarmer(dummy_viz)
    for step in range(5):
        warmer.enqueue(step)

    assert warmer._active
    warmer.invalidate(reason="test")
    assert not warmer._active
    assert len(warmer._queue) == 0


def test_warmer_state_drift_invalidates(dummy_viz: SimpleNamespace) -> None:
    """If app_state changes between enqueue and tick, the warmer invalidates."""
    from dataclasses import replace

    dummy_viz.cache_service.store_frame(0, _make_raw_frame(0))

    warmer = ViewModelWarmer(dummy_viz)
    warmer.enqueue(0)

    # Change state to make snapshot stale
    dummy_viz.app_state = replace(dummy_viz.app_state, color_mode="delay")

    warmer._process_tick()

    # Warmer should have invalidated — no VM created
    assert not warmer._active
    assert dummy_viz.mpc_core.calls == []


def test_observable_cache_clear_invalidates_warmer(dummy_viz: SimpleNamespace) -> None:
    """Clearing the observable cache triggers warmer invalidation."""
    for step in range(3):
        dummy_viz.cache_service.store_frame(step, _make_raw_frame(step))

    warmer = ViewModelWarmer(dummy_viz)
    dummy_viz.mpc_view_cache.set_warmer(warmer)

    for step in range(3):
        warmer.enqueue(step)

    assert warmer._active

    # Simulate what happens when a filter/color change clears the VM cache
    dummy_viz.mpc_view_cache.clear()

    assert not warmer._active
    assert len(warmer._queue) == 0


def test_warmer_stop_resets_everything(dummy_viz: SimpleNamespace) -> None:
    """stop() fully resets the warmer state."""
    dummy_viz.cache_service.store_frame(0, _make_raw_frame(0))

    warmer = ViewModelWarmer(dummy_viz)
    warmer.enqueue(0)
    warmer._process_tick()

    # After processing, session is still active (paused, not ended)
    assert warmer._active
    assert warmer._warmed_count == 1

    warmer.stop()
    assert not warmer._active
    assert warmer._warmed_count == 0
    assert warmer._snapshot_key is None
    assert len(warmer._queue) == 0


def test_warmer_batches_across_interleaved_signals(dummy_viz: SimpleNamespace) -> None:
    """Simulate QueuedConnection delivery: enqueue one step, tick, repeat.

    With the old code, each enqueue→tick cycle would re-activate and reset
    warmed_count. With the fix, the session stays open and count accumulates.
    """
    for step in range(5):
        dummy_viz.cache_service.store_frame(step, _make_raw_frame(step))

    warmer = ViewModelWarmer(dummy_viz)
    dummy_viz.mpc_view_cache.set_warmer(warmer)

    # Simulate interleaved signal delivery (one enqueue, one tick, repeat)
    for step in range(5):
        warmer.enqueue(step)
        warmer._process_tick()

    # Single session, all 5 VMs warmed — no re-activation churn
    assert warmer._active
    assert warmer._warmed_count == 5
    assert len(dummy_viz.mpc_view_cache) == 5

    # stop() ends the session
    warmer.stop()
    assert not warmer._active
    assert warmer._warmed_count == 0


class _CrossThreadSignals(QObject):
    """Mimics _PreloadSignals for cross-thread testing."""

    step_ready = Signal(int)


def test_warmer_cross_thread_signal_delivery(dummy_viz: SimpleNamespace) -> None:
    """Verify that step_ready emitted from a background thread delivers
    enqueue() on the main thread via QueuedConnection.

    This is the critical test — the original bug was that enqueue() ran on
    the background thread, causing QTimer.start() to silently fail.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    for step in range(3):
        dummy_viz.cache_service.store_frame(step, _make_raw_frame(step))

    warmer = ViewModelWarmer(dummy_viz)
    dummy_viz.mpc_view_cache.set_warmer(warmer)

    signals = _CrossThreadSignals()
    from PySide6.QtCore import Qt

    signals.step_ready.connect(warmer.enqueue, Qt.QueuedConnection)

    # Emit from a background thread (mimicking _PreloadWorker)
    def _emit_from_bg():
        for step in range(3):
            signals.step_ready.emit(step)

    bg = threading.Thread(target=_emit_from_bg)
    bg.start()
    bg.join(timeout=2.0)

    # Process Qt events so queued signals are delivered on the main thread
    for _ in range(50):
        app.processEvents()
        time.sleep(0.01)
        if warmer._active:
            break

    # The warmer should have been activated by the queued signals
    assert warmer._active or warmer._warmed_count > 0, (
        "Warmer was not activated — step_ready signals were not delivered "
        "to the main thread. Check QueuedConnection and QObject inheritance."
    )

    # Process ticks to warm all frames
    for _ in range(10):
        warmer._process_tick()
        if not warmer._queue:
            break

    assert warmer._warmed_count == 3
    assert len(dummy_viz.mpc_view_cache) == 3
