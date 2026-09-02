"""Integration tests covering generator→visualizer HDF5 compatibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shared.frames.loader import FrameLoaderService
from shared.frames.manifest import manifest_from_chunks, write_frame_manifest_atomic
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed_hdf5_writer import write_packed_mpc_frame_chunk
from shared.frames.providers import Hdf5Provider
from shared.frames.types import StandardMPCFrame


def _build_generator_frame() -> StandardMPCFrame:
    """Create a minimal canonical generator frame."""
    tx_pos = np.array([[0.0, 0.0, 1.5]], dtype=np.float64)
    rx_pos = np.array([[10.0, 0.0, 1.5]], dtype=np.float64)
    orientation = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    pair_lengths = np.array([2], dtype=np.int64)
    pair_vertices = np.array([[[0.0, 0.0, 1.5], [10.0, 0.0, 1.5]]], dtype=np.float32)
    pair_interactions = np.ones((1, 2), dtype=np.int32)

    return standard_mpc_frame_from_pair_data(
        frame_index=0,
        provenance={"provider": "test", "frame_idx": 0},
        tx_positions=tx_pos,
        rx_positions=rx_pos,
        tx_orientations=orientation,
        rx_orientations=orientation,
        tx_rx_pairs=np.array([[0, 0]], dtype=np.int32),
        target_positions_m=np.array([[1.0, 2.0, 1.0]], dtype=np.float64),
        targets_metadata=[{"name": "target-0", "type": "human"}],
        vertices_by_pair=[pair_vertices],
        interactions_by_pair=[pair_interactions],
        path_lengths_by_pair=[pair_lengths],
        material_names_by_pair=[np.array([["mat-default", "mat-default"]], dtype=object)],
        material_itu_types_by_pair=[np.array([["A", "A"]], dtype=object)],
        metrics_by_pair={
            "delays_ns": [np.array([12.3], dtype=np.float32)],
            "path_loss_db": [np.array([48.0], dtype=np.float32)],
            "aoa_az_deg": [np.array([5.0], dtype=np.float32)],
            "aoa_el_deg": [np.array([2.0], dtype=np.float32)],
            "aod_az_deg": [np.array([8.0], dtype=np.float32)],
            "aod_el_deg": [np.array([1.0], dtype=np.float32)],
        },
    )


def _write_single_frame_dataset(root: Path) -> Path:
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame = _build_generator_frame()
    chunk = write_packed_mpc_frame_chunk(
        frames_dir,
        [frame],
        generation_id="generator-visualizer-test-generation",
        compression=None,
    )
    manifest = manifest_from_chunks(
        generation_id="generator-visualizer-test-generation",
        frame_set_id="generator-visualizer-test-frame-set",
        chunks=[chunk],
        compression={"configured": None, "filter": "none", "shuffle": False},
        segmentation={"max_frames": 1},
        provenance={"fixture": "generator-visualizer"},
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)
    return frames_dir / chunk.file


def test_generator_hdf5_round_trip(tmp_path: Path):
    """Generator HDF5 output should load via shared provider/frame loader."""
    scenario_root = tmp_path / "scenario"
    _write_single_frame_dataset(scenario_root)

    provider = Hdf5Provider(scenario_root)
    loader = FrameLoaderService(provider)

    assert provider.list_frames() == [0]
    frame = loader.get_frame(0)
    assert frame is not None

    np.testing.assert_allclose(frame.tx_positions, np.array([[0.0, 0.0, 1.5]]))
    np.testing.assert_allclose(frame.rx_positions, np.array([[10.0, 0.0, 1.5]]))
    assert frame.num_tx == 1
    assert frame.num_rx == 1

    # Confirm per-pair metrics survived the round-trip.
    assert frame.delays_ns[0] == pytest.approx(12.3, rel=1e-5)
    assert frame.path_loss_db[0] == pytest.approx(48.0, rel=1e-5)
    material_idx = int(frame.material_ids[0])
    assert frame.material_names[material_idx] == "mat-default"
    assert frame.num_targets == 1
