"""Tests for visualizer.src.services.beamforming_service."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from visualizer.src.beamforming import standalone as standalone_module
from visualizer.src.beamforming.standalone import (
    compute_los_angles,
    compute_steering_weights,
    compute_svd_beamforming,
)
from visualizer.src.beamforming.visualization import MAX_BEAM_PATTERN_WORK_ITEMS
from visualizer.src.metrics.mpc_canon import CanonicalStepData
from visualizer.src.scene.surface_payloads import BeamformingSurface
from visualizer.src.services.beamforming_service import BeamformingService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_visualizer():
    """Minimal mock visualizer with app_state."""
    viz = MagicMock()
    viz.app_state = SimpleNamespace(
        standalone_carrier_frequency_ghz=28.0,
        standalone_horizontal_spacing_m=0.00536,
        standalone_vertical_spacing_m=0.00536,
        beamforming_tx_element_pattern="isotropic",
        beamforming_rx_element_pattern="isotropic",
    )
    viz._frame_duration = 1.0
    viz.total_animation_steps = 10
    return viz


@pytest.fixture()
def service(mock_visualizer):
    return BeamformingService(mock_visualizer)


def _make_raw_frame(*, mode: str = "standalone", strategy: str = "manual"):
    """Build a minimal raw_frame dict for testing."""
    return {
        "standalone_beamforming_mode": mode,
        "standalone_beamforming_params": {
            "antenna_rows": 2,
            "antenna_cols": 2,
            "horizontal_spacing_m": 0.00536,
            "vertical_spacing_m": 0.00536,
            "carrier_frequency_ghz": 28.0,
            "steering_strategy": strategy,
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
        },
        "tx_positions": np.array([[0.0, 0.0, 10.0]]),
        "rx_positions": np.array([[50.0, 0.0, 1.0]]),
        "_source": {"step": 0, "duration": 1.0, "num_steps": 10},
    }


def _canonical_paths(
    *paths: tuple[tuple[int, int], np.ndarray, float, bool],
) -> CanonicalStepData:
    """Build the minimal aligned canonical frame used by SVD service tests."""
    point_parts: list[np.ndarray] = []
    line_parts: list[np.ndarray] = []
    point_order: list[np.ndarray] = []
    point_itype: list[np.ndarray] = []
    point_delay: list[np.ndarray] = []
    point_loss: list[np.ndarray] = []
    point_tx: list[np.ndarray] = []
    point_rx: list[np.ndarray] = []
    point_path: list[np.ndarray] = []
    path_starts: list[int] = []
    path_orders: list[int] = []
    path_losses: list[float] = []
    path_tx: list[int] = []
    path_rx: list[int] = []
    path_loss_is_estimated: list[bool] = []
    point_offset = 0

    for path_id, (pair, vertices, loss_db, estimated) in enumerate(paths):
        path_points = np.asarray(vertices, dtype=np.float32)
        if path_points.ndim != 2 or path_points.shape[1] != 3 or len(path_points) < 2:
            raise ValueError("test canonical paths must have shape [N,3] with N >= 2")
        point_count = len(path_points)
        order = point_count - 2
        path_starts.append(point_offset)
        point_parts.append(path_points)
        line_parts.append(
            np.column_stack(
                (
                    np.arange(point_offset, point_offset + point_count - 1, dtype=np.int32),
                    np.arange(point_offset + 1, point_offset + point_count, dtype=np.int32),
                )
            )
        )
        point_order.append(np.full(point_count, order, dtype=np.uint8))
        point_itype.append(np.zeros(point_count, dtype=np.uint8))
        point_delay.append(np.zeros(point_count, dtype=np.float32))
        point_loss.append(np.full(point_count, loss_db, dtype=np.float32))
        point_tx.append(np.full(point_count, pair[0], dtype=np.int16))
        point_rx.append(np.full(point_count, pair[1], dtype=np.int16))
        point_path.append(np.full(point_count, path_id, dtype=np.int32))
        path_orders.append(order)
        path_losses.append(loss_db)
        path_tx.append(pair[0])
        path_rx.append(pair[1])
        path_loss_is_estimated.append(estimated)
        point_offset += point_count

    def _join(parts: list[np.ndarray], shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        if not parts:
            return np.empty(shape, dtype=dtype)
        return np.ascontiguousarray(np.concatenate(parts), dtype=dtype)

    return CanonicalStepData(
        points=_join(point_parts, (0, 3), np.dtype(np.float32)),
        lines=_join(line_parts, (0, 2), np.dtype(np.int32)),
        order=_join(point_order, (0,), np.dtype(np.uint8)),
        itype=_join(point_itype, (0,), np.dtype(np.uint8)),
        delay=_join(point_delay, (0,), np.dtype(np.float32)),
        loss=_join(point_loss, (0,), np.dtype(np.float32)),
        tx_id=_join(point_tx, (0,), np.dtype(np.int16)),
        rx_id=_join(point_rx, (0,), np.dtype(np.int16)),
        path_id=_join(point_path, (0,), np.dtype(np.int32)),
        path_start_indices=np.asarray(path_starts, dtype=np.int32),
        path_orders=np.asarray(path_orders, dtype=np.uint8),
        path_delays=np.zeros(len(paths), dtype=np.float32),
        path_losses=np.asarray(path_losses, dtype=np.float32),
        path_tx=np.asarray(path_tx, dtype=np.int16),
        path_rx=np.asarray(path_rx, dtype=np.int16),
        path_loss_is_estimated=np.asarray(path_loss_is_estimated, dtype=np.bool_),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildMeshes:
    """Integration tests for the top-level build_meshes entry point."""

    def test_standalone_manual(self, service):
        raw = _make_raw_frame(mode="standalone", strategy="manual")
        tx = np.array([[0.0, 0.0, 10.0]])
        rx = np.array([[50.0, 0.0, 1.0]])
        tx_ori = np.zeros((1, 3))
        rx_ori = np.zeros((1, 3))

        result = service.build_meshes(
            raw,
            tx,
            rx,
            tx_ori,
            rx_ori,
            beamforming_tx_node="auto",
            beamforming_rx_node="all",
            selected_tx=0,
            selected_rx=0,
            step=0,
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )
        assert result is not None
        assert "meshes" in result
        assert "info" in result
        meshes = result["meshes"]
        assert isinstance(meshes, tuple)
        assert len(meshes) > 0
        assert all(isinstance(surface, BeamformingSurface) for surface in meshes)
        assert all(surface.payload.normals is not None for surface in meshes)
        assert all(surface.payload.vertex_colors is not None for surface in meshes)

    def test_standalone_mode_does_not_mutate_cached_frame_parameters(self, service):
        raw = _make_raw_frame(mode="standalone", strategy="manual")
        original_params = dict(raw["standalone_beamforming_params"])
        service._build_standalone_meshes = MagicMock(return_value={"meshes": (), "info": {}})

        service.build_meshes(
            raw,
            np.array([[0.0, 0.0, 10.0]]),
            np.array([[50.0, 0.0, 1.0]]),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            beamforming_tx_node="auto",
            beamforming_rx_node="all",
            selected_tx=0,
            selected_rx=0,
            step=0,
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )

        assert raw["standalone_beamforming_params"] == original_params
        passed_params = service._build_standalone_meshes.call_args.args[0]
        assert passed_params is not raw["standalone_beamforming_params"]
        assert passed_params["mode"] == "standalone"

    def test_standalone_los(self, service):
        raw = _make_raw_frame(mode="standalone", strategy="los")
        tx = np.array([[0.0, 0.0, 10.0]])
        rx = np.array([[50.0, 0.0, 1.0]])
        result = service.build_meshes(
            raw,
            tx,
            rx,
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            beamforming_tx_node="auto",
            beamforming_rx_node="all",
            selected_tx=0,
            selected_rx=0,
            step=0,
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )
        assert result is not None
        assert result["meshes"] is not None

    def test_standalone_svd(self, service):
        raw = _make_raw_frame(mode="standalone", strategy="svd")
        tx = np.array([[0.0, 0.0, 10.0]])
        rx = np.array([[50.0, 0.0, 1.0]])
        canonical = _canonical_paths(
            (
                (0, 0),
                np.array([[0.0, 0.0, 10.0], [25.0, 5.0, 5.0], [50.0, 0.0, 1.0]]),
                80.0,
                False,
            )
        )
        result = service.build_meshes(
            raw,
            tx,
            rx,
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            canonical_data=canonical,
            beamforming_tx_node="auto",
            beamforming_rx_node="all",
            selected_tx=0,
            selected_rx=0,
            step=0,
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )
        assert result is not None
        assert result["meshes"] is not None

    def test_standalone_svd_passes_canonical_reflection_to_channel_builder(
        self,
        service,
        monkeypatch,
    ):
        raw = _make_raw_frame(mode="standalone", strategy="svd")
        tx = np.array([[0.0, 0.0, 10.0]])
        rx = np.array([[50.0, 0.0, 1.0]])
        reflected = np.array(
            [[0.0, 0.0, 10.0], [25.0, 5.0, 5.0], [50.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        canonical = _canonical_paths(((0, 0), reflected, 80.0, False))
        captured_paths: list[list[tuple[np.ndarray, float]] | None] = []
        original_compute = standalone_module.compute_standalone_beamforming

        def capture_paths(**kwargs):
            captured_paths.append(kwargs["mpc_paths"])
            return original_compute(**kwargs)

        monkeypatch.setattr(standalone_module, "compute_standalone_beamforming", capture_paths)

        result = service.build_meshes(
            raw,
            tx,
            rx,
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            canonical_data=canonical,
            beamforming_tx_node="auto",
            beamforming_rx_node="all",
            selected_tx=0,
            selected_rx=0,
            step=0,
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )

        assert result is not None
        assert len(captured_paths) == 1
        assert captured_paths[0] is not None
        np.testing.assert_array_equal(captured_paths[0][0][0], reflected)

    def test_standalone_svd_extracts_only_the_selected_pair(self, service):
        raw = _make_raw_frame(mode="standalone", strategy="svd")
        tx = np.array([[0.0, 0.0, 10.0], [5.0, 0.0, 10.0]])
        rx = np.array([[50.0, 0.0, 1.0], [60.0, 0.0, 1.0]])
        canonical = _canonical_paths(
            (
                (1, 0),
                np.array([[5.0, 0.0, 10.0], [50.0, 0.0, 1.0]]),
                90.0,
                False,
            )
        )
        service.extract_mpc_paths = MagicMock(return_value=None)

        result = service.build_meshes(
            raw,
            tx,
            rx,
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            canonical_data=canonical,
            beamforming_tx_node="auto",
            beamforming_rx_node="all",
            selected_tx=1,
            selected_rx=0,
            step=0,
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )

        assert result is not None
        service.extract_mpc_paths.assert_called_once()
        assert service.extract_mpc_paths.call_args.args == (canonical,)
        assert service.extract_mpc_paths.call_args.kwargs["selected_pair"] == (1, 0)

    def test_frame_mode_no_data(self, service):
        """Frame mode with no beamforming data returns None."""
        raw = {"standalone_beamforming_mode": "frame"}
        tx = np.array([[0.0, 0.0, 10.0]])
        rx = np.array([[50.0, 0.0, 1.0]])
        result = service.build_meshes(
            raw,
            tx,
            rx,
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            beamforming_tx_node="auto",
            beamforming_rx_node="all",
            selected_tx="all",
            selected_rx="all",
            step=0,
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )
        assert result is not None
        assert result["meshes"] == ()
        assert "unavailable" in result["info"]["status"].lower()

    def test_empty_positions_returns_none(self, service):
        raw = _make_raw_frame()
        result = service.build_meshes(
            raw,
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty((0, 3)),
            beamforming_tx_node="auto",
            beamforming_rx_node="all",
            selected_tx="all",
            selected_rx="all",
            step=0,
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )
        assert result is None


class TestExtractMpcPaths:
    """Test MPC path extraction for SVD beamforming."""

    def test_with_path_loss_data(self, service):
        vertices = np.array([[0, 0, 10], [25, 5, 5], [50, 0, 1]], dtype=np.float32)
        canonical = _canonical_paths(((0, 0), vertices, -80.0, False))

        result = service.extract_mpc_paths(canonical)

        assert result is not None
        assert (0, 0) in result
        paths = result[(0, 0)]
        assert len(paths) == 1
        verts, pl = paths[0]
        assert np.shares_memory(verts, canonical.points)
        np.testing.assert_array_equal(verts, vertices)
        assert pl == -80.0

    def test_selected_pair_filter_preserves_default_all_pair_behavior(self, service):
        canonical = _canonical_paths(
            ((0, 0), np.array([[0, 0, 10], [50, 0, 1]]), 80.0, False),
            ((0, 1), np.array([[0, 0, 10], [60, 0, 1]]), 90.0, False),
            ((1, 0), np.array([[5, 0, 10], [50, 0, 1]]), 100.0, False),
        )

        all_pairs = service.extract_mpc_paths(canonical)
        selected = service.extract_mpc_paths(canonical, selected_pair=(0, 1))

        assert all_pairs is not None
        assert set(all_pairs) == {(0, 0), (0, 1), (1, 0)}
        assert selected is not None
        assert set(selected) == {(0, 1)}
        assert selected[(0, 1)][0][1] == 90.0

    def test_reconstructs_full_paths_from_bounce_vertices(self, service):
        canonical = _canonical_paths(
            ((0, 0), np.array([[0, 0, 10], [50, 0, 1]]), 80.0, False),
            (
                (0, 0),
                np.array([[0, 0, 10], [25, 5, 5], [50, 0, 1]]),
                100.0,
                False,
            ),
            (
                (0, 0),
                np.array([[0, 0, 10], [20, 3, 4], [30, -2, 2], [50, 0, 1]]),
                120.0,
                False,
            ),
        )

        result = service.extract_mpc_paths(canonical)

        assert result is not None
        paths = result[(0, 0)]
        assert len(paths) == 3
        assert np.allclose(paths[0][0], np.array([[0.0, 0.0, 10.0], [50.0, 0.0, 1.0]]))
        assert np.allclose(
            paths[1][0],
            np.array([[0.0, 0.0, 10.0], [25.0, 5.0, 5.0], [50.0, 0.0, 1.0]]),
        )
        assert np.allclose(
            paths[2][0],
            np.array([[0.0, 0.0, 10.0], [20.0, 3.0, 4.0], [30.0, -2.0, 2.0], [50.0, 0.0, 1.0]]),
        )

    def test_keeps_bounce_vertex_at_origin(self, service):
        canonical = _canonical_paths(
            (
                (0, 0),
                np.array([[0, 0, 10], [0, 0, 0], [50, 0, 1]]),
                100.0,
                False,
            )
        )

        result = service.extract_mpc_paths(canonical)

        assert result is not None
        path_vertices, _ = result[(0, 0)][0]
        assert np.allclose(
            path_vertices,
            np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0], [50.0, 0.0, 1.0]]),
        )

    def test_reconstructed_hello_world_like_paths_keep_los_dominant(self, service):
        raw = {
            "all_padded_vertices": [
                np.array(
                    [
                        [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
                        [[-20.65, -52.78, 13.69], [np.nan, np.nan, np.nan]],
                        [[9.434, 0.0, 0.0], [np.nan, np.nan, np.nan]],
                        [[-107.2, 52.21, 13.68], [np.nan, np.nan, np.nan]],
                        [[-20.65, -52.78, 12.24], [6.654, -5.762, 0.0]],
                        [[-107.2, 52.21, 12.24], [-2.795, 5.702, 0.0]],
                        [[-1.708, -90.10, 17.63], [-70.68, -99.06, 11.95]],
                        [[-227.3, -101.1, 17.98], [-165.2, 206.4, 9.136]],
                    ],
                    dtype=np.float64,
                )
            ],
            "all_path_lengths": [np.array([0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)],
            "all_pair_path_loss_db": [
                np.array(
                    [
                        89.53508,
                        108.84572,
                        99.83474,
                        115.442116,
                        120.553474,
                        120.99946,
                        121.31837,
                        134.66043,
                    ],
                    dtype=np.float64,
                )
            ],
            "tx_rx_pairs": np.array([[0, 0]], dtype=np.int32),
        }
        tx_position = np.array([[0.0, 0.0, 25.0]])
        rx_position = np.array([[10.0, 0.0, 1.5]])
        canonical = _canonical_paths(
            *(
                (
                    (0, 0),
                    np.vstack(
                        (
                            tx_position[0],
                            raw["all_padded_vertices"][0][path_idx, :bounce_count],
                            rx_position[0],
                        )
                    ),
                    float(raw["all_pair_path_loss_db"][0][path_idx]),
                    False,
                )
                for path_idx, bounce_count in enumerate(raw["all_path_lengths"][0])
            )
        )

        extracted = service.extract_mpc_paths(canonical)

        assert extracted is not None
        mpc_paths = extracted[(0, 0)]
        assert len(mpc_paths) == 8
        assert np.allclose(mpc_paths[0][0], np.array([[0.0, 0.0, 25.0], [10.0, 0.0, 1.5]]))

        tx_weights, rx_weights, _ = compute_svd_beamforming(
            num_rows_tx=8,
            num_cols_tx=8,
            num_rows_rx=8,
            num_cols_rx=8,
            horizontal_spacing_m=0.00536,
            vertical_spacing_m=0.00536,
            carrier_frequency_hz=28e9,
            tx_position=(0.0, 0.0, 25.0),
            rx_position=(10.0, 0.0, 1.5),
            tx_orientation=(0.0, 0.0, 0.0),
            rx_orientation=(0.0, 0.0, 0.0),
            mpc_paths=mpc_paths,
        )

        tx_az, tx_el = compute_los_angles((0.0, 0.0, 25.0), (10.0, 0.0, 1.5), (0.0, 0.0, 0.0))
        rx_az, rx_el = compute_los_angles((10.0, 0.0, 1.5), (0.0, 0.0, 25.0), (0.0, 0.0, 0.0))
        los_tx_weights, _, _ = compute_steering_weights(8, 8, 0.00536, 0.00536, 28e9, tx_az, tx_el)
        los_rx_weights, _, _ = compute_steering_weights(8, 8, 0.00536, 0.00536, 28e9, rx_az, rx_el)

        tx_alignment = abs(np.vdot(tx_weights, los_tx_weights)) / (
            np.linalg.norm(tx_weights) * np.linalg.norm(los_tx_weights)
        )
        rx_alignment = abs(np.vdot(rx_weights, los_rx_weights)) / (
            np.linalg.norm(rx_weights) * np.linalg.norm(los_rx_weights)
        )

        assert tx_alignment > 0.999
        assert rx_alignment > 0.95

    def test_fallback_geometric_distance(self, service):
        vertices = np.array([[0, 0, 10], [50, 0, 1]], dtype=np.float32)
        canonical = _canonical_paths(((0, 0), vertices, np.nan, False))
        result = service.extract_mpc_paths(
            canonical,
            carrier_frequency_hz=28e9,
        )
        assert result is not None
        _, pl = result[(0, 0)][0]
        # Free-space path loss at ~50m, 28GHz -- positive dB (higher = more loss).
        assert 70 < pl < 120

    def test_estimated_canonical_loss_uses_frequency_specific_fallback(self, service):
        vertices = np.array([[0, 0, 10], [50, 0, 1]], dtype=np.float32)
        canonical = _canonical_paths(((0, 0), vertices, 5.0, True))

        low_frequency = service.extract_mpc_paths(canonical, carrier_frequency_hz=1e9)
        high_frequency = service.extract_mpc_paths(canonical, carrier_frequency_hz=28e9)

        assert low_frequency is not None
        assert high_frequency is not None
        low_loss = low_frequency[(0, 0)][0][1]
        high_loss = high_frequency[(0, 0)][0][1]
        assert high_loss > low_loss + 20.0
        assert low_loss != pytest.approx(5.0)

    def test_empty_vertices_returns_none(self, service):
        result = service.extract_mpc_paths(_canonical_paths())
        assert result is None

    def test_service_does_not_retain_raw_frame_state(self, service):
        raw = _make_raw_frame(mode="standalone", strategy="svd")
        tx = np.array([[0.0, 0.0, 10.0]])
        rx = np.array([[50.0, 0.0, 1.0]])
        canonical = _canonical_paths(((0, 0), np.vstack((tx[0], rx[0])), 80.0, False))
        service.build_meshes(
            raw,
            tx,
            rx,
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            canonical_data=canonical,
            beamforming_tx_node="auto",
            beamforming_rx_node="all",
            selected_tx="all",
            selected_rx="all",
            step=0,
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )
        assert not hasattr(service, "_last_raw_frame")


class TestFrameBasedMeshes:
    """Test frame-based beamforming with pre-computed weights."""

    def test_with_valid_pair(self, service):
        w = np.ones(4, dtype=complex) / 2.0
        w_stacked = np.stack([w.real, w.imag], axis=-1).astype(np.float32)
        pos = np.array(
            [[0, 0, 0], [0, 0.005, 0], [0, 0, 0.005], [0, 0.005, 0.005]], dtype=np.float32
        )

        beamforming_data = {
            "pairs": [
                {
                    "pair_index": 0,
                    "tx_index": 0,
                    "rx_index": 0,
                    "tx": {
                        "device_name": "tx_1",
                        "device_index": 0,
                        "role": "tx",
                        "weights": w_stacked,
                        "element_positions": pos,
                        "carrier_frequency_hz": 28e9,
                        "beam_gain": 32.0,
                    },
                    "rx": {
                        "device_name": "rx_1",
                        "device_index": 0,
                        "role": "rx",
                        "weights": w_stacked,
                        "element_positions": pos,
                        "carrier_frequency_hz": 28e9,
                        "beam_gain": 1.0,
                    },
                }
            ]
        }

        result = service._build_frame_meshes(
            beamforming_data,
            np.array([[0.0, 0.0, 10.0]]),
            np.array([[50.0, 0.0, 1.0]]),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            "auto",
            "all",
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )
        assert result is not None
        assert isinstance(result["meshes"], tuple)
        # Should have TX + RX meshes
        assert len(result["meshes"]) == 2
        assert {surface.id for surface in result["meshes"]} == {
            "beamforming:0_tx_0_0:mesh",
            "beamforming:0_rx_0_0:mesh",
        }
        assert all(
            surface.payload.vertices.flags.writeable is False for surface in result["meshes"]
        )
        assert result["info"]["gain_by_role"] == {"tx": 32.0, "rx": 1.0}
        assert set(result["info"]["metrics_by_role"]) == {"tx", "rx"}
        tx_metrics = result["info"]["metrics_by_role"]["tx"]
        assert tx_metrics["peak_gain_dbi"] > 0.0
        assert tx_metrics["hpbw_az_deg"] > 0.0
        assert tx_metrics["hpbw_el_deg"] > 0.0
        assert tx_metrics["sll_db"] <= 0.0

    def test_concrete_pair_with_invalid_array_metadata_is_unavailable(self, service):
        beamforming_data = {
            "pairs": [
                {
                    "pair_index": 0,
                    "tx_index": 0,
                    "rx_index": 0,
                    "tx": {
                        "device_name": "tx_1",
                        "device_index": 0,
                        "weights": None,
                        "element_positions": None,
                        "carrier_frequency_hz": 28e9,
                    },
                    "rx": {
                        "device_name": "rx_1",
                        "device_index": 0,
                        "weights": None,
                        "element_positions": None,
                        "carrier_frequency_hz": 28e9,
                    },
                }
            ]
        }

        result = service._build_frame_meshes(
            beamforming_data,
            np.array([[0.0, 0.0, 10.0]]),
            np.array([[50.0, 0.0, 1.0]]),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            "tx_1",
            "rx_1",
            selected_tx=0,
            selected_rx=0,
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )

        assert result["meshes"] == ()
        assert result["info"]["status"].startswith("Beam pattern unavailable")

    def test_one_valid_role_is_reported_as_partial(self, service):
        weights = np.array([[1.0, 0.0]], dtype=np.float32)
        positions = np.zeros((1, 3), dtype=np.float32)
        beamforming_data = {
            "pairs": [
                {
                    "pair_index": 0,
                    "tx_index": 0,
                    "rx_index": 0,
                    "tx": {
                        "device_name": "tx_1",
                        "device_index": 0,
                        "weights": weights,
                        "element_positions": positions,
                        "carrier_frequency_hz": 28e9,
                    },
                    "rx": {
                        "device_name": "rx_1",
                        "device_index": 0,
                        "weights": None,
                        "element_positions": None,
                        "carrier_frequency_hz": 28e9,
                    },
                }
            ]
        }

        result = service._build_frame_meshes(
            beamforming_data,
            np.zeros((1, 3)),
            np.ones((1, 3)),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            "tx_1",
            "rx_1",
            selected_tx=0,
            selected_rx=0,
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )

        assert len(result["meshes"]) == 1
        assert result["info"]["status"] == "Beam pattern partial: missing RX surface"

    def test_frame_metadata_large_arrays_are_sample_bounded(
        self,
        service,
        monkeypatch: pytest.MonkeyPatch,
    ):
        num_elements = 64 * 64
        weights = np.column_stack(
            (
                np.ones(num_elements, dtype=np.float32),
                np.zeros(num_elements, dtype=np.float32),
            )
        )
        positions = np.zeros((num_elements, 3), dtype=np.float32)
        mesh_calls: list[dict] = []
        metric_calls: list[dict] = []

        def fake_mesh(**kwargs):
            mesh_calls.append(kwargs)
            origin = np.asarray(kwargs["origin"], dtype=np.float64)
            return (
                origin + np.eye(3, dtype=np.float64),
                np.array([[0, 1, 2]], dtype=np.int32),
                np.ones((3, 3), dtype=np.float64),
            )

        def fake_metrics(**kwargs):
            metric_calls.append(kwargs)
            return {}

        monkeypatch.setattr(
            "visualizer.src.services.beamforming_service.generate_beamforming_mesh",
            fake_mesh,
        )
        monkeypatch.setattr(
            "visualizer.src.services.beamforming_service.compute_pattern_metrics",
            fake_metrics,
        )
        entry = {
            "device_index": 0,
            "weights": weights,
            "element_positions": positions,
            "carrier_frequency_hz": 28e9,
        }
        beamforming_data = {
            "pairs": [
                {
                    "pair_index": 0,
                    "tx_index": 0,
                    "rx_index": 0,
                    "tx": {**entry, "device_name": "tx_1"},
                    "rx": {**entry, "device_name": "rx_1"},
                }
            ]
        }

        result = service._build_frame_meshes(
            beamforming_data,
            np.zeros((1, 3)),
            np.ones((1, 3)),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            "tx_1",
            "rx_1",
            selected_tx=0,
            selected_rx=0,
            beamforming_azimuth_samples=180,
            beamforming_elevation_samples=91,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )

        assert len(result["meshes"]) == 2
        assert len(mesh_calls) == len(metric_calls) == 2
        for call in (*mesh_calls, *metric_calls):
            work_items = num_elements * call["azimuth_samples"] * call["elevation_samples"]
            assert work_items <= MAX_BEAM_PATTERN_WORK_ITEMS
        assert all(call["elevation_samples"] < 91 for call in mesh_calls)

    def test_builds_final_scaled_surfaces_from_explicit_display_options(
        self,
        service,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _AppStateMustNotBeRead:
            def __getattr__(self, name: str):
                raise AssertionError(f"unexpected app_state read: {name}")

        service._visualizer.app_state = _AppStateMustNotBeRead()
        mesh_calls: list[dict] = []

        def fake_generate_beamforming_mesh(**kwargs):
            mesh_calls.append(kwargs)
            origin = np.asarray(kwargs["origin"], dtype=np.float32)
            return (
                origin
                + np.asarray(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
                np.asarray([[0, 1, 2]], dtype=np.int32),
                np.asarray([[1.2, -0.2, 0.5]] * 3, dtype=np.float32),
            )

        monkeypatch.setattr(
            "visualizer.src.services.beamforming_service.generate_beamforming_mesh",
            fake_generate_beamforming_mesh,
        )
        monkeypatch.setattr(
            "visualizer.src.services.beamforming_service.compute_pattern_metrics",
            lambda **_kwargs: {},
        )
        weights = np.ones((1, 2), dtype=np.float32)
        element_positions = np.zeros((1, 3), dtype=np.float32)
        pair = {
            "pair_index": 0,
            "unique_pair_id": "Pair A",
            "tx_index": 0,
            "rx_index": 0,
            "tx": {
                "device_name": "tx_1",
                "device_index": 0,
                "weights": weights,
                "element_positions": element_positions,
                "carrier_frequency_hz": 28e9,
            },
            "rx": {
                "device_name": "rx_1",
                "device_index": 0,
                "weights": weights,
                "element_positions": element_positions,
                "carrier_frequency_hz": 28e9,
            },
        }

        result = service._build_frame_meshes(
            {"pairs": [pair]},
            np.asarray([[10.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[20.0, 0.0, 0.0]], dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
            "auto",
            "auto",
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=2.0,
            beamforming_rx_scale=3.0,
            beamforming_db_scale=True,
            beamforming_dynamic_range_db=27.0,
            beamforming_colormap="viridis",
            beamforming_element_pattern="isotropic",
            beamforming_tx_element_pattern="dipole",
            beamforming_rx_element_pattern="tr38901",
        )

        assert result is not None
        tx_surface, rx_surface = result["meshes"]
        assert tx_surface.id == "beamforming:0_tx_pair_a:mesh"
        assert rx_surface.id == "beamforming:0_rx_pair_a:mesh"
        np.testing.assert_allclose(tx_surface.payload.vertices[0], [12.0, 0.0, 0.0])
        np.testing.assert_allclose(rx_surface.payload.vertices[0], [23.0, 0.0, 0.0])
        np.testing.assert_allclose(tx_surface.payload.vertex_colors[0], [1.0, 0.0, 0.5])
        assert tx_surface.payload.normals is not None
        assert tx_surface.payload.vertices.flags.writeable is False
        assert [call["element_pattern"] for call in mesh_calls] == ["dipole", "tr38901"]
        assert all(call["db_scale"] is True for call in mesh_calls)
        assert all(call["dynamic_range_db"] == 27.0 for call in mesh_calls)
        assert all(call["colormap"] == "viridis" for call in mesh_calls)

    def test_colliding_surface_ids_use_order_independent_pair_metadata(
        self,
        service,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "visualizer.src.services.beamforming_service.generate_beamforming_mesh",
            lambda **kwargs: (
                np.asarray(kwargs["origin"], dtype=np.float32)
                + np.asarray(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
                np.asarray([[0, 1, 2]], dtype=np.int32),
                np.ones((3, 3), dtype=np.float32),
            ),
        )
        monkeypatch.setattr(
            "visualizer.src.services.beamforming_service.compute_pattern_metrics",
            lambda **_kwargs: {},
        )
        weights = np.ones((1, 2), dtype=np.float32)
        element_positions = np.zeros((1, 3), dtype=np.float32)

        def make_pair(pair_index: int) -> dict:
            def make_entry(role: str) -> dict:
                return {
                    "device_name": f"{role}_1",
                    "device_index": 0,
                    "weights": weights,
                    "element_positions": element_positions,
                    "carrier_frequency_hz": 28e9,
                }

            return {
                "pair_index": pair_index,
                "unique_pair_id": "shared",
                "tx_index": 0,
                "rx_index": 0,
                "tx": make_entry("tx"),
                "rx": make_entry("rx"),
            }

        def build_ids(pairs: list[dict]) -> set[str]:
            result = service._build_frame_meshes(
                {"pairs": pairs},
                np.zeros((1, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
                np.zeros((1, 3), dtype=np.float32),
                "auto",
                "auto",
                beamforming_azimuth_samples=12,
                beamforming_elevation_samples=9,
                beamforming_tx_scale=1.0,
                beamforming_rx_scale=1.0,
            )
            assert result is not None
            return {surface.id for surface in result["meshes"]}

        pair_3 = make_pair(3)
        pair_7 = make_pair(7)
        forward = build_ids([pair_7, pair_3])
        reverse = build_ids([pair_3, pair_7])

        assert (
            forward
            == reverse
            == {
                "beamforming:0_tx_shared_pair_0_0_3:mesh",
                "beamforming:0_rx_shared_pair_0_0_3:mesh",
                "beamforming:0_tx_shared_pair_0_0_7:mesh",
                "beamforming:0_rx_shared_pair_0_0_7:mesh",
            }
        )

        with pytest.raises(ValueError, match="unique stable metadata"):
            build_ids([make_pair(3), make_pair(3)])

    def test_frame_mode_uses_exact_global_pair(self, service):
        w = np.ones(4, dtype=complex) / 2.0
        w_stacked = np.stack([w.real, w.imag], axis=-1).astype(np.float32)
        pos = np.array(
            [[0, 0, 0], [0, 0.005, 0], [0, 0, 0.005], [0, 0.005, 0.005]],
            dtype=np.float32,
        )
        pairs = []
        for rx_idx in range(2):
            pairs.append(
                {
                    "pair_index": rx_idx,
                    "tx_index": 0,
                    "rx_index": rx_idx,
                    "tx": {
                        "device_name": "tx_1",
                        "device_index": 0,
                        "role": "tx",
                        "weights": w_stacked,
                        "element_positions": pos,
                        "carrier_frequency_hz": 28e9,
                    },
                    "rx": {
                        "device_name": f"rx_{rx_idx + 1}",
                        "device_index": rx_idx,
                        "role": "rx",
                        "weights": w_stacked,
                        "element_positions": pos,
                        "carrier_frequency_hz": 28e9,
                    },
                }
            )

        result = service._build_frame_meshes(
            {"pairs": pairs},
            np.array([[0.0, 0.0, 10.0]]),
            np.array([[50.0, 0.0, 1.0], [60.0, 0.0, 1.0]]),
            np.zeros((1, 3)),
            np.zeros((2, 3)),
            "auto",
            "all",
            selected_tx=0,
            selected_rx=1,
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )

        assert result is not None
        assert result["info"]["resolved_tx_node"] == "tx_1"
        assert result["info"]["resolved_rx_node"] == "rx_2"
        assert result["info"]["requested_tx_index"] == 0
        assert result["info"]["requested_rx_index"] == 1
        assert {surface.id for surface in result["meshes"]} == {
            "beamforming:0_tx_0_1:mesh",
            "beamforming:1_rx_0_1:mesh",
        }
        assert len(result["meshes"]) == 2

    def test_frame_mode_missing_global_pair_does_not_fallback(self, service):
        pair = {
            "pair_index": 0,
            "tx_index": 0,
            "rx_index": 0,
            "tx": {"device_name": "tx_1"},
            "rx": {"device_name": "rx_1"},
        }

        result = service._build_frame_meshes(
            {"pairs": [pair]},
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            "tx_2",
            "rx_2",
            selected_tx=1,
            selected_rx=1,
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )

        assert result is not None
        assert result["meshes"] == ()
        assert result["info"]["resolved_tx_node"] == "tx_2"
        assert result["info"]["resolved_rx_node"] == "rx_2"
        assert result["info"]["status"] == "No beamforming data for TX2 -> RX2"

    def test_standalone_global_all_requires_concrete_pair(self, service):
        raw = _make_raw_frame(mode="standalone", strategy="svd")
        tx = np.array([[0.0, 0.0, 10.0], [5.0, 0.0, 10.0]])
        rx = np.array([[50.0, 0.0, 1.0], [60.0, 0.0, 1.0]])

        result = service.build_meshes(
            raw,
            tx,
            rx,
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            beamforming_tx_node="auto",
            beamforming_rx_node="all",
            selected_tx="all",
            selected_rx="all",
            step=0,
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )

        assert result is not None
        assert result["info"]["resolved_tx_node"] is None
        assert result["info"]["resolved_rx_node"] is None
        assert result["meshes"] == ()
        assert result["info"]["status"] == "Select one TX and one RX to render beam patterns"

    def test_none_data_returns_none(self, service):
        result = service._build_frame_meshes(
            None,
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            "auto",
            "all",
            beamforming_azimuth_samples=12,
            beamforming_elevation_samples=9,
            beamforming_tx_scale=1.0,
            beamforming_rx_scale=1.0,
        )
        assert result is not None
        assert result["meshes"] == ()
        assert result["info"]["available_tx_nodes"] == []
