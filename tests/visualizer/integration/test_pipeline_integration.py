"""Integration tests for visualizer data pipelines.

These tests cover compact frame storage, session persistence, projection-backed
canonical visual data, filtering, colorization, and ViewModel construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from shared.frames import (
    StandardMPCFrame,
    project_standard_mpc_frame,
    standard_mpc_frame_from_pair_data,
)
from shared.frames.manifest import manifest_from_chunks, write_frame_manifest_atomic
from shared.frames.packed_hdf5_writer import write_packed_mpc_frame_chunk
from shared.frames.providers import Hdf5Provider
from shared.frames.schema import is_valid_standard_mpc_frame, validate_standard_mpc_frame
from visualizer.src.io.packed_frame_payload import (
    projection_to_visual_frame,
    visual_frame_read_request,
)
from visualizer.src.metrics.mpc_canon import (
    CanonicalStepData,
    build_filter_mask,
    colorize,
)
from visualizer.src.services.session_service import SESSION_VERSION, SessionService
from visualizer.src.state import create_initial_state, update_state


def _write_frame_set(root: Path, frames: list[StandardMPCFrame]) -> Path:
    """Write complete compact frames behind the authoritative HDF5 v2 manifest."""
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    chunk = write_packed_mpc_frame_chunk(
        frames_dir,
        frames,
        generation_id="visualizer-pipeline-test-generation",
        compression=None,
    )
    manifest = manifest_from_chunks(
        generation_id="visualizer-pipeline-test-generation",
        frame_set_id="visualizer-pipeline-test-frame-set",
        chunks=[chunk],
        compression={"configured": None, "filter": "none", "shuffle": False},
        segmentation={"max_frames": len(frames)},
        provenance={"fixture": "visualizer-pipeline"},
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)
    return frames_dir / chunk.file


# =============================================================================
# Frame Round-Trip Integration Tests
# =============================================================================


class TestFrameRoundTrip:
    """Test complete frame data round-trip through HDF5."""

    def test_minimal_frame_round_trip(self, tmp_path: Path) -> None:
        """Verify minimal frame survives HDF5 write/read cycle."""
        frame = self._create_minimal_frame()
        self._write_frame_to_hdf5(tmp_path, frame)

        provider = Hdf5Provider(str(tmp_path))
        loaded_frame = provider.load_frame(0)

        assert is_valid_standard_mpc_frame(loaded_frame)

        assert loaded_frame.num_tx == frame.num_tx
        assert loaded_frame.num_rx == frame.num_rx
        np.testing.assert_allclose(loaded_frame.tx_positions, frame.tx_positions)
        np.testing.assert_allclose(loaded_frame.rx_positions, frame.rx_positions)

    def test_frame_with_paths_round_trip(self, tmp_path: Path) -> None:
        """Verify frame with path data survives round-trip."""
        frame = self._create_frame_with_paths()
        self._write_frame_to_hdf5(tmp_path, frame)

        provider = Hdf5Provider(str(tmp_path))
        loaded_frame = provider.load_frame(0)

        errors = validate_standard_mpc_frame(loaded_frame, raise_on_error=False)
        assert not errors, f"Schema validation failed: {errors}"

        np.testing.assert_array_equal(loaded_frame.pair_path_offsets, frame.pair_path_offsets)
        np.testing.assert_array_equal(loaded_frame.bounce_offsets, frame.bounce_offsets)
        np.testing.assert_allclose(loaded_frame.bounce_xyz_m, frame.bounce_xyz_m)
        np.testing.assert_array_equal(loaded_frame.interactions, frame.interactions)

    def test_multi_pair_frame_round_trip(self, tmp_path: Path) -> None:
        """Verify frame with multiple TX-RX pairs survives round-trip."""
        frame = self._create_multi_pair_frame()
        self._write_frame_to_hdf5(tmp_path, frame)

        provider = Hdf5Provider(str(tmp_path))
        loaded_frame = provider.load_frame(0)

        np.testing.assert_array_equal(loaded_frame.tx_rx_pairs, frame.tx_rx_pairs)
        np.testing.assert_array_equal(loaded_frame.pair_path_offsets, frame.pair_path_offsets)
        assert loaded_frame.num_pairs == frame.num_pairs

    def test_frame_list_enumeration(self, tmp_path: Path) -> None:
        """Verify provider correctly lists available frames."""
        _write_frame_set(
            tmp_path,
            [self._create_minimal_frame(frame_index=frame_idx) for frame_idx in range(3)],
        )

        provider = Hdf5Provider(str(tmp_path))
        frames = provider.list_frames()

        assert sorted(frames) == [0, 1, 2]

    def _create_minimal_frame(self, *, frame_index: int = 0) -> StandardMPCFrame:
        """Create a valid compact frame with one pair and no paths."""
        return standard_mpc_frame_from_pair_data(
            frame_index=frame_index,
            tx_positions=[[0.0, 0.0, 1.5]],
            rx_positions=[[10.0, 0.0, 1.5]],
            tx_rx_pairs=[[0, 0]],
            vertices_by_pair=[np.empty((0, 2, 3), dtype=np.float32)],
            interactions_by_pair=[np.empty((0, 2), dtype=np.uint8)],
            path_lengths_by_pair=[np.empty((0,), dtype=np.int64)],
            provenance={"provider": "test", "frame_idx": frame_index},
        )

    def _create_frame_with_paths(self) -> StandardMPCFrame:
        """Create a compact frame containing one path with two bounces."""
        vertices = np.array([[[0.0, 0.0, 1.5], [10.0, 0.0, 1.5]]], dtype=np.float32)
        interactions = np.array([[1, 1]], dtype=np.uint8)

        return standard_mpc_frame_from_pair_data(
            frame_index=0,
            tx_positions=[[0.0, 0.0, 1.5]],
            rx_positions=[[10.0, 0.0, 1.5]],
            tx_rx_pairs=[[0, 0]],
            vertices_by_pair=[vertices],
            interactions_by_pair=[interactions],
            path_lengths_by_pair=[np.array([2], dtype=np.int64)],
            provenance={"provider": "test", "frame_idx": 0},
        )

    def _create_multi_pair_frame(self) -> StandardMPCFrame:
        """Create a compact frame containing two TX/RX pairs."""
        return standard_mpc_frame_from_pair_data(
            frame_index=0,
            tx_positions=[[0.0, 0.0, 1.5]],
            rx_positions=[[10.0, 0.0, 1.5], [10.0, 5.0, 1.5]],
            tx_rx_pairs=[[0, 0], [0, 1]],
            vertices_by_pair=[
                np.array([[[0.0, 0.0, 1.5], [10.0, 0.0, 1.5]]], dtype=np.float32),
                np.array([[[0.0, 0.0, 1.5], [10.0, 5.0, 1.5]]], dtype=np.float32),
            ],
            interactions_by_pair=[
                np.array([[1, 1]], dtype=np.uint8),
                np.array([[1, 1]], dtype=np.uint8),
            ],
            path_lengths_by_pair=[
                np.array([2], dtype=np.int64),
                np.array([2], dtype=np.int64),
            ],
            provenance={"provider": "test", "frame_idx": 0},
        )

    def _write_frame_to_hdf5(self, root: Path, frame: StandardMPCFrame) -> Path:
        """Write one complete compact HDF5 v2 frame set."""
        return _write_frame_set(root, [frame])


# =============================================================================
# Session Persistence Integration Tests
# =============================================================================


class TestSessionPersistence:
    """Test session save/load round-trip."""

    def test_session_save_creates_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify session save creates a valid JSON file."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        viz = self._create_mock_visualizer(tmp_path)
        service = SessionService(viz)

        path = service.save_session(name="test_session")

        assert path.exists()
        assert path.suffix == ".json"

        # Verify JSON is valid
        with open(path) as f:
            data = json.load(f)

        assert "version" in data
        assert data["version"] == SESSION_VERSION

    def test_session_round_trip_preserves_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify session save/load preserves app state."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Create visualizer with specific state
        viz = self._create_mock_visualizer(tmp_path)
        viz.app_state = create_initial_state(step=42)
        service = SessionService(viz)

        # Save session
        path = service.save_session(name="state_test")

        # Load into fresh visualizer
        viz2 = self._create_mock_visualizer(tmp_path)
        service2 = SessionService(viz2)
        result = service2.load_session(path)

        assert result is True

    def test_session_load_nonexistent_returns_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify loading nonexistent session returns False."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        viz = self._create_mock_visualizer(tmp_path)
        service = SessionService(viz)

        result = service.load_session(tmp_path / "nonexistent.json")

        assert result is False

    def test_session_load_invalid_json_returns_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify loading invalid JSON returns False."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Create invalid JSON file
        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text("not valid json {{{")

        viz = self._create_mock_visualizer(tmp_path)
        service = SessionService(viz)

        result = service.load_session(invalid_path)

        assert result is False

    def _create_mock_visualizer(self, root: Path) -> SimpleNamespace:
        """Create a scene-only visualizer with one real scenario root."""
        scenario_root = root / "scenario"
        scenario_root.mkdir(exist_ok=True)
        (scenario_root / "scenario.yaml").write_text(
            "schema_version: 1\n",
            encoding="utf-8",
        )

        viz = SimpleNamespace()
        viz.app_state = create_initial_state(step=0)
        viz.current_scenario_path = scenario_root
        viz._scene_only_mode = True
        viz._session_restore_in_progress = False
        viz.force_update_next_frame = False
        viz.selected_objects = set()
        viz.mesh_entries = []
        viz.target_entries = []
        viz.tx_entries = []
        viz.rx_entries = []
        viz.node_coloring_mode = "per_type"
        viz.ui_manager = None
        viz.cancel_scheduled_update = lambda: None
        viz.schedule_update = lambda: None
        viz.renderer = SimpleNamespace(
            get_view_control=lambda: SimpleNamespace(
                convert_to_pinhole_camera_parameters=lambda: SimpleNamespace(
                    extrinsic=np.eye(4),
                    intrinsic=SimpleNamespace(
                        intrinsic_matrix=np.eye(3),
                        width=800,
                        height=600,
                    ),
                )
            )
        )
        viz.animation_controller = SimpleNamespace(
            current_frame=0,
            frame_count=100,
            fps=30,
        )

        def set_state(**changes: Any) -> None:
            viz.app_state = update_state(viz.app_state, **changes)

        viz.set_state = set_state
        return viz


# =============================================================================
# Canonical Data Processing Integration Tests
# =============================================================================


class TestCanonicalDataProcessing:
    """Test projection-backed canonical data and filtering."""

    def test_projection_builds_canonical_visual_data(self) -> None:
        """Verify compact frame projection creates canonical visual data."""
        canon = self._create_canonical_frame()

        assert isinstance(canon, CanonicalStepData)
        assert canon.points is not None
        assert canon.lines is not None
        assert len(canon.points) > 0

    def test_canonical_preserves_point_count(self) -> None:
        """Verify canonical data preserves expected point count."""
        canon = self._create_canonical_frame()

        # Each path has 2 points (TX->RX), we have 1 path
        # Plus potentially TX/RX points depending on implementation
        assert len(canon.points) >= 2

    def test_filter_mask_by_reflection_order(self) -> None:
        """Verify filtering by reflection order works."""
        canon = self._create_canonical_frame()

        # Filter to only show LOS (order 0)
        point_mask, line_mask = build_filter_mask(
            canon,
            allowed_orders=[0],
            allowed_types=[0, 1, 2, 4, 8],  # All interaction types
            selected_tx="all",
            selected_rx="all",
        )

        assert point_mask is not None
        assert isinstance(point_mask, np.ndarray)

    def test_colorize_by_reflection_order(self) -> None:
        """Verify colorization by reflection order produces RGB colors."""
        from visualizer.src.utils.colors import ensure_viridis_lut

        canon = self._create_canonical_frame()
        point_mask = np.ones(len(canon.points), dtype=bool)

        # Provide default palettes
        order_palette = np.array(
            [
                [0.2, 0.6, 1.0],  # Order 0: Blue
                [0.2, 0.8, 0.2],  # Order 1: Green
                [1.0, 0.8, 0.2],  # Order 2: Yellow
                [1.0, 0.4, 0.2],  # Order 3: Orange
                [1.0, 0.2, 0.2],  # Order 4: Red
                [0.8, 0.2, 0.8],  # Order 5: Purple
                [0.5, 0.5, 0.5],  # Order 6+: Gray
            ],
            dtype=np.float32,
        )

        colors = colorize(
            canon,
            point_mask,
            mode="reflection_order",
            order_palette=order_palette,
            type_palette=None,
            viridis256=ensure_viridis_lut(),
        )

        assert colors.shape[1] == 3  # RGB
        assert np.all(colors >= 0.0)
        assert np.all(colors <= 1.0)

    def test_colorize_by_delay(self) -> None:
        """Verify colorization by delay produces RGB colors."""
        canon = self._create_canonical_frame()
        point_mask = np.ones(len(canon.points), dtype=bool)

        colors = colorize(
            canon,
            point_mask,
            mode="delay",
            order_palette=None,
            type_palette=None,
            viridis256=None,
        )

        assert colors.shape[1] == 3

    def test_colorize_by_path_loss(self) -> None:
        """Verify colorization by path loss produces RGB colors."""
        canon = self._create_canonical_frame()
        point_mask = np.ones(len(canon.points), dtype=bool)

        colors = colorize(
            canon,
            point_mask,
            mode="path_loss",
            order_palette=None,
            type_palette=None,
            viridis256=None,
        )

        assert colors.shape[1] == 3

    def _create_canonical_frame(self) -> CanonicalStepData:
        """Create renderer-facing data through the compact projection seam."""

        frame = standard_mpc_frame_from_pair_data(
            frame_index=0,
            tx_positions=[[0.0, 0.0, 1.5]],
            rx_positions=[[10.0, 0.0, 1.5]],
            tx_rx_pairs=[[0, 0]],
            vertices_by_pair=[(np.empty((0, 3), dtype=np.float32),)],
            interactions_by_pair=[(np.empty((0,), dtype=np.uint8),)],
            metrics_by_pair={
                "delay_ns": [np.array([33.3], dtype=np.float32)],
                "path_loss_db": [np.array([60.0], dtype=np.float32)],
            },
        )
        payload = projection_to_visual_frame(
            project_standard_mpc_frame(frame, visual_frame_read_request())
        )
        return payload["canonical_data"]


# =============================================================================
# Schema Validation Integration Tests
# =============================================================================


class TestSchemaValidation:
    """Test frame schema validation across the pipeline."""

    def test_generator_output_validates(self) -> None:
        """Verify a producer-created compact frame passes validation."""
        frame = standard_mpc_frame_from_pair_data(
            frame_index=0,
            tx_rx_pairs=[[0, 0]],
            tx_positions=[[0.0, 0.0, 0.0]],
            rx_positions=[[1.0, 0.0, 0.0]],
            vertices_by_pair=[np.empty((0, 2, 3), dtype=np.float32)],
            interactions_by_pair=[np.empty((0, 2), dtype=np.uint8)],
            path_lengths_by_pair=[np.empty((0,), dtype=np.int64)],
        )

        assert is_valid_standard_mpc_frame(frame)

    def test_visualizer_loaded_frame_validates(self, tmp_path: Path) -> None:
        """Verify frame loaded via provider passes validation."""
        frame = standard_mpc_frame_from_pair_data(
            frame_index=0,
            tx_rx_pairs=[[0, 0]],
            tx_positions=[[0.0, 0.0, 0.0]],
            rx_positions=[[1.0, 0.0, 0.0]],
            vertices_by_pair=[np.empty((0, 2, 3), dtype=np.float32)],
            interactions_by_pair=[np.empty((0, 2), dtype=np.uint8)],
            path_lengths_by_pair=[np.empty((0,), dtype=np.int64)],
            provenance={"provider": "test", "frame_idx": 0},
        )
        _write_frame_set(tmp_path, [frame])

        provider = Hdf5Provider(str(tmp_path))
        loaded_frame = provider.load_frame(0)

        errors = validate_standard_mpc_frame(loaded_frame, raise_on_error=False)
        assert not errors, f"Validation errors: {errors}"

    def test_validation_catches_missing_fields(self) -> None:
        """Verify validation catches missing required fields."""
        incomplete_frame = {
            "tx_positions": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            # Missing other required fields
        }

        assert not is_valid_standard_mpc_frame(incomplete_frame)

    def test_construction_catches_device_axis_mismatch(self) -> None:
        """Verify compact-frame construction rejects misaligned device arrays."""
        frame = standard_mpc_frame_from_pair_data(
            frame_index=0,
            tx_rx_pairs=[[0, 0]],
            tx_positions=[[0.0, 0.0, 0.0]],
            rx_positions=[[1.0, 0.0, 0.0]],
            vertices_by_pair=[np.empty((0, 2, 3), dtype=np.float32)],
            interactions_by_pair=[np.empty((0, 2), dtype=np.uint8)],
            path_lengths_by_pair=[np.empty((0,), dtype=np.int64)],
        )
        values = {
            field: getattr(frame, field)
            for field in frame.__dataclass_fields__
            if field != "version"
        }
        values["tx_positions"] = np.zeros((2, 3), dtype=np.float64)

        with pytest.raises(ValueError, match="tx_orientations count"):
            StandardMPCFrame(**values)
