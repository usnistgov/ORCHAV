"""Integration tests for visualizer data loading stack."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from shared.frames.contracts import PathMetric
from shared.frames.manifest import manifest_from_chunks, write_frame_manifest_atomic
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed_hdf5_writer import write_packed_mpc_frame_chunk
from shared.frames.providers import Hdf5Provider
from shared.statistics.frame_stats import FrameStats
from tests.visualizer.fixtures.sample_frames import save_sample_frame_hdf5
from visualizer.src.pipeline.core import ViewModel
from visualizer.src.pipeline.frame_pipeline import FramePipeline
from visualizer.src.services.cache_service import CacheService
from visualizer.src.state import MpcVisibility, create_initial_state


@pytest.mark.integration
class TestDataLoadingPipeline:
    """Test the complete data loading pipeline."""

    def test_load_frame_sequence(self, temp_test_dir):
        """Test loading a sequence of frames."""
        # Create test data
        hdf5_path = temp_test_dir / "frames.h5"
        n_frames = 10
        save_sample_frame_hdf5(hdf5_path, n_frames=n_frames, n_mpcs=100)

        # Verify all frames can be accessed

        with h5py.File(hdf5_path, "r") as f:
            frames_group = f["frames"]
            for i in range(n_frames):
                frame_key = f"frame_{i}"
                assert frame_key in frames_group
                frame = frames_group[frame_key]
                assert "mpc_data" in frame
                assert "tx_positions" in frame
                assert "rx_positions" in frame

    def test_end_to_end_frame_processing(self):
        """Test complete pipeline from file to ViewModel."""
        raw_frame = _make_raw_frame()
        visualizer = _make_visualizer(raw_frame)
        pipeline = FramePipeline(visualizer)

        pipeline.update(0)

        assert len(visualizer.renderer.applied) == 1
        packet = visualizer.renderer.applied[0]
        assert not hasattr(packet, "tx_positions")
        np.testing.assert_array_equal(
            visualizer.current_view_model.tx_positions,
            raw_frame.tx_positions,
        )
        assert visualizer.frame_source.load_count == 1
        assert visualizer.force_update_next_frame is False

    def test_caching_behavior(self):
        """Test that override cache bypasses frame source after first load."""
        raw_frame = _make_raw_frame()
        visualizer = _make_visualizer(raw_frame)
        pipeline = FramePipeline(visualizer)

        pipeline.update(0)
        assert visualizer.cache_service.has_frame(0)
        first_count = visualizer.frame_source.load_count

        cached = visualizer.cache_service.get_frame(0)
        visualizer.cache_service.store_frame(0, cached, source="override")
        visualizer.force_update_next_frame = True
        visualizer.frame_source.raise_on_load = True
        pipeline.update(0)

        assert visualizer.frame_source.load_count == first_count
        assert len(visualizer.renderer.applied) == 2

    def test_hdf5_provider_round_trip(self, tmp_path):
        """Verify Hdf5Provider can enumerate and load synthetic per-frame data."""
        root = tmp_path / "scenario"
        frames_dir = root / "frames"
        frames_dir.mkdir(parents=True)

        self._write_minimal_frame(frames_dir, frame_idx=0)

        provider = Hdf5Provider(str(root))

        assert provider.list_frames() == [0]
        assert provider.has_frame(0) is True

        frame = provider.load_frame(0)
        assert frame.num_tx == 1
        assert frame.num_rx == 1
        assert frame.tx_positions.shape == (1, 3)
        assert frame.rx_positions.shape == (1, 3)
        assert frame.bounce_xyz_m.shape == (2, 3)
        np.testing.assert_array_equal(frame.tx_rx_pairs, np.array([[0, 0]]))

    @staticmethod
    def _write_minimal_frame(frames_dir: Path, frame_idx: int) -> None:
        """Create a minimal manifest-driven packed HDF5 v2 frame set."""
        vertices = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], dtype=np.float32)
        interactions = np.ones((1, 2), dtype=np.uint8)
        materials = np.array([["mat-itu_air", "mat-itu_air"]])
        metrics = {
            metric: [np.asarray([value], dtype=np.float32)]
            for metric, value in zip(PathMetric, range(1, 7), strict=True)
        }
        frame = standard_mpc_frame_from_pair_data(
            frame_index=frame_idx,
            tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
            tx_positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            rx_positions=np.asarray([[10.0, 0.0, 0.0]], dtype=np.float64),
            tx_orientations=np.zeros((1, 3), dtype=np.float64),
            rx_orientations=np.zeros((1, 3), dtype=np.float64),
            vertices_by_pair=[vertices],
            interactions_by_pair=[interactions],
            path_lengths_by_pair=[np.asarray([2], dtype=np.int64)],
            material_names_by_pair=[materials],
            material_itu_types_by_pair=[materials],
            metrics_by_pair=metrics,
            target_positions_m=np.empty((0, 3), dtype=np.float64),
            targets_metadata=(),
            provenance={"provider": "test", "frame_idx": frame_idx},
        )
        chunk = write_packed_mpc_frame_chunk(
            frames_dir,
            [frame],
            generation_id="visualizer-data-loading-test-generation",
            compression=None,
        )
        manifest = manifest_from_chunks(
            generation_id="visualizer-data-loading-test-generation",
            frame_set_id="visualizer-data-loading-test-frame-set",
            chunks=[chunk],
            compression={"configured": None, "filter": "none", "shuffle": False},
            segmentation={"max_frames": 1},
            provenance={"fixture": "visualizer-data-loading"},
            created_utc="2026-07-29T00:00:00+00:00",
        )
        write_frame_manifest_atomic(frames_dir, manifest)


def _make_raw_frame():
    return standard_mpc_frame_from_pair_data(
        frame_index=0,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        tx_positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        rx_positions=np.asarray([[5.0, 0.0, 0.0]], dtype=np.float64),
        tx_orientations=np.zeros((1, 3), dtype=np.float64),
        rx_orientations=np.zeros((1, 3), dtype=np.float64),
        vertices_by_pair=[np.empty((0, 1, 3), dtype=np.float32)],
        interactions_by_pair=[np.empty((0, 1), dtype=np.uint8)],
        path_lengths_by_pair=[np.empty((0,), dtype=np.int64)],
    )


def _make_visualizer(raw_frame):
    view_model = _make_view_model(raw_frame)
    renderer = _DummyRenderer()
    frame_source = _DummyFrameSource(raw_frame)
    mpc_core = _StubMPCCore(view_model)

    vis = SimpleNamespace()
    vis.app_state = create_initial_state(step=0)
    vis.last_app_state = None
    vis.ready = True
    vis.renderer = renderer
    vis.frame_source = frame_source
    vis.mpc_core = mpc_core
    vis.cache_service = CacheService(vis)
    vis.use_preload_mode = False
    vis.force_update_next_frame = False
    vis.mpc_view_cache = OrderedDict()
    vis.metrics_window = None
    vis.target_entries = []
    vis.vis = None
    vis.frame_times = []
    vis._scene_boot_start = None
    vis._scene_boot_logged = True
    vis._material_filter_dirty = False
    vis._coverage_interpolation_dirty = False
    vis.coverage_data = None
    vis.coverage_heights = []
    vis.coverage_height_index = 0
    vis.coverage_opacity = 0.7
    vis.tx_markers = []
    vis.rx_markers = []
    vis.schedule_update = lambda: None
    vis.ui_controller = SimpleNamespace(
        update_frame_context=lambda *args, **kwargs: None,
        handle_frame_timing_update=lambda *args, **kwargs: None,
    )
    vis.beamforming_ui_controller = SimpleNamespace(
        apply_selector_state=lambda: None,
        set_frame_beamforming_available=lambda _available: None,
        update_node_options=lambda _info, _pairs=None: None,
        update_standalone_buttons_state=lambda: None,
    )
    vis._ensure_tx_rx_markers_created = lambda *args, **kwargs: None
    vis.set_state = lambda **changes: setattr(vis, "app_state", vis.app_state)
    vis.mpc_allowed_materials = None
    vis.show_tx_segments = True
    vis.mpc_core._last_reflection_order_counts = {}
    vis.mpc_core.current_delay_range = None
    vis.mpc_core.current_delay_range = None
    vis.mpc_core.current_path_loss_range = None
    vis.node_service = SimpleNamespace()
    vis.node_service.ensure_tx_rx_markers_created = lambda *args, **kwargs: None
    vis.node_service.update_available_tx_rx_from_frame = lambda tx, rx: setattr(
        vis, "_last_counts", (tx, rx)
    )
    vis.node_service.update_tx_rx_positions = lambda *_args, **_kwargs: True
    vis.target_service = SimpleNamespace(
        process_targets_from_view_model=lambda *_args, **_kwargs: True
    )
    vis.node_service.create_orientation_frames = lambda *_args, **_kwargs: True
    vis.node_service.retry_pending_node_syncs = lambda: True
    return vis


def _make_view_model(raw_frame):
    return ViewModel(
        tx_positions=raw_frame.tx_positions,
        rx_positions=raw_frame.rx_positions,
        tx_orientations=raw_frame.tx_orientations,
        rx_orientations=raw_frame.rx_orientations,
        mpc_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        mpc_lines=np.empty((0, 2), dtype=np.int32),
        mpc_colors=np.empty((0, 3), dtype=np.float32),
        colorbar=None,
        stats_text="stats",
        mpc_visibility=MpcVisibility(),
        target_positions=np.empty((0, 3)),
        target_orientations=np.empty((0, 3)),
        target_mesh_files=[],
        target_use_ply_positions=[],
        target_metadata=[],
    )


class _DummyFrameSource:
    def __init__(self, frame):
        self.frame = frame
        self.load_count = 0
        self.raise_on_load = False

    def load_frame(self, step: int):
        if self.raise_on_load:
            raise AssertionError("Frame source should not be called when cache is used")
        self.load_count += 1
        return self.frame


class _StubMPCCore:
    def __init__(self, view_model):
        self.view_model = view_model

    def create_view_model(self, step, raw_frame, **kwargs):
        return self.view_model

    def stats(self, payload):
        return FrameStats(total_paths=1, orders_hist={0: 1})


class _DummyRenderer:
    def __init__(self):
        self.applied = []

    def apply_frame(self, packet) -> bool:
        self.applied.append(packet)
        return True

    def begin_frame_update(self) -> None:
        pass

    def end_frame_update(self) -> bool:
        return True
