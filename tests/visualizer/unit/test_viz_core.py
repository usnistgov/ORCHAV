from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from visualizer.src.controllers.material_ui_controller import MaterialUIController
from visualizer.src.metrics.mpc_canon import CanonicalStepData, colorize_segments, ensure_luts
from visualizer.src.pipeline.core import FrameRenderPacket, MPCCore, ViewModel
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.scene.surface_payloads import BeamformingSurface
from visualizer.src.state import MpcVisibility
from visualizer.src.types.render_payloads import MeshPayload


def _make_beamforming_surface(surface_id: str = "beamforming:tx0:mesh") -> BeamformingSurface:
    return BeamformingSurface(
        id=surface_id,
        payload=MeshPayload(
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )


def _make_basic_view_model(**overrides):
    defaults = dict(
        tx_positions=np.empty((0, 3), dtype=np.float32),
        rx_positions=np.empty((0, 3), dtype=np.float32),
        tx_orientations=np.empty((0, 3), dtype=np.float32),
        rx_orientations=np.empty((0, 3), dtype=np.float32),
        mpc_points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        mpc_lines=np.array([[0, 1]], dtype=np.int32),
        mpc_colors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        colorbar=None,
        stats_text="",
        mpc_visibility=MpcVisibility(),
        target_positions=np.empty((0, 3), dtype=np.float32),
        target_orientations=np.empty((0, 3), dtype=np.float32),
        target_mesh_files=[],
        target_use_ply_positions=[],
        target_metadata=[],
    )
    defaults.update(overrides)
    return ViewModel(**defaults)


def test_view_model_freezes_mpc_arrays_and_builds_revisions():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    view_model = _make_basic_view_model(mpc_points=points)

    assert view_model.mpc_line_revision[0] == "mpc-line-v1"
    assert view_model.mpc_point_revision[0] == "mpc-point-v1"
    assert view_model.mpc_points.flags.writeable is False
    assert points.flags.writeable is False
    with pytest.raises(ValueError):
        view_model.mpc_points[0, 0] = 2.0


def test_view_model_mpc_revisions_change_with_new_payload_arrays():
    first = _make_basic_view_model()
    second = _make_basic_view_model(
        mpc_points=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        mpc_bounce_points=np.array([[0.5, 0.0, 0.0]], dtype=np.float32),
    )

    assert second.mpc_line_revision != first.mpc_line_revision
    assert second.mpc_point_revision != first.mpc_point_revision


def test_view_model_render_packet_is_shallow_and_excludes_persistent_entities():
    surface = _make_beamforming_surface()
    view_model = _make_basic_view_model(
        mpc_bounce_points=np.array([[0.5, 0.0, 0.0]], dtype=np.float32),
        coverage_metadata={"metric": "rss"},
        beamforming_meshes=[surface],
    )

    packet = view_model.to_render_packet()

    assert isinstance(packet, FrameRenderPacket)
    assert packet.mpc_points is view_model.mpc_points
    assert packet.mpc_bounce_points is view_model.mpc_bounce_points
    assert packet.coverage_metadata is view_model.coverage_metadata
    assert packet.beamforming_meshes is view_model.beamforming_meshes
    assert packet.beamforming_meshes == (surface,)
    assert not hasattr(packet, "tx_positions")
    assert not hasattr(packet, "target_positions")
    assert not hasattr(packet, "labels")


def test_view_model_caches_sorted_rendered_interaction_codes_on_packet():
    """Legends reuse one compact type summary instead of rescanning line arrays."""
    view_model = _make_basic_view_model(
        mpc_line_itypes=np.array([99, 1, 37, 1], dtype=np.uint8),
        mpc_line_itype_codes=(99, 1, 37, 1),
    )

    packet = view_model.to_render_packet()

    assert view_model.mpc_line_itype_codes == (1, 37, 99)
    assert packet.mpc_line_itype_codes is view_model.mpc_line_itype_codes


class TestMPCCore:
    """Unit tests for MPCCore class."""

    @pytest.fixture
    def mpc_core(self):
        """Fixture to provide an MPCCore instance."""
        logger = MagicMock()
        return MPCCore(logger=logger)

    @pytest.fixture
    def sample_raw_frame(self):
        """Fixture to provide a sample raw frame."""
        return {
            "tx_positions": [[0, 0, 0], [10, 0, 0]],
            "rx_positions": [[5, 5, 0], [15, 5, 0]],
            "tx_orientations": [[0, 0, 0], [0, 0, 0]],
            "rx_orientations": [[0, 0, 0], [0, 0, 0]],
            "all_padded_vertices": np.zeros((4, 1, 3, 3)),  # 4 pairs (2x2), 1 path each, 3 vertices
            "all_path_lengths": [[1.0], [1.0], [1.0], [1.0]],
            "tx_rx_pairs": [[0, 0], [0, 1], [1, 0], [1, 1]],
            "reflection_order_counts": {0: 10, 1: 5},
            "delay_range": (0.0, 100.0),
            "path_loss_range": (50.0, 100.0),
        }

    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_beamforming_options_flow_explicitly_to_service(
        self,
        mock_get_canonical,
        mpc_core,
        sample_raw_frame,
        mock_canonical_data,
    ):
        mock_get_canonical.return_value = mock_canonical_data
        surface = _make_beamforming_surface()
        mpc_core.beamforming_service = MagicMock()
        mpc_core.beamforming_service.build_meshes.return_value = {
            "meshes": (surface,),
            "info": {
                "pairs": [{"tx_name": "tx_1", "rx_name": "rx_1"}],
                "status": "ready",
            },
        }

        view_model = mpc_core.create_view_model(
            step=0,
            raw_frame=sample_raw_frame,
            color_mode="reflection_order",
            show_beamforming=True,
            beamforming_db_scale=True,
            beamforming_dynamic_range_db=27.0,
            beamforming_colormap="viridis",
            beamforming_element_pattern="isotropic",
            beamforming_tx_element_pattern="dipole",
            beamforming_rx_element_pattern="tr38901",
        )

        assert view_model is not None
        assert view_model.beamforming_meshes == (surface,)
        assert view_model.beamforming_pairs == [{"tx_name": "tx_1", "rx_name": "rx_1"}]
        kwargs = mpc_core.beamforming_service.build_meshes.call_args.kwargs
        assert kwargs["canonical_data"] is mock_canonical_data
        assert kwargs["beamforming_db_scale"] is True
        assert kwargs["beamforming_dynamic_range_db"] == 27.0
        assert kwargs["beamforming_colormap"] == "viridis"
        assert kwargs["beamforming_element_pattern"] == "isotropic"
        assert kwargs["beamforming_tx_element_pattern"] == "dipole"
        assert kwargs["beamforming_rx_element_pattern"] == "tr38901"

    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_disabled_beamforming_does_not_call_service(
        self,
        mock_get_canonical,
        mpc_core,
        sample_raw_frame,
        mock_canonical_data,
    ):
        mock_get_canonical.return_value = mock_canonical_data
        mpc_core.beamforming_service = MagicMock()

        view_model = mpc_core.create_view_model(
            step=0,
            raw_frame=sample_raw_frame,
            color_mode="reflection_order",
            show_beamforming=False,
        )

        assert view_model is not None
        assert view_model.beamforming_meshes == ()
        mpc_core.beamforming_service.build_meshes.assert_not_called()

    @pytest.mark.parametrize(
        ("enabled", "paths", "bounce_points"),
        [
            (enabled, paths, bounce_points)
            for enabled in (False, True)
            for paths in (False, True)
            for bounce_points in (False, True)
        ],
    )
    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_create_view_model_applies_effective_mpc_visibility(
        self,
        mock_get_canonical,
        enabled,
        paths,
        bounce_points,
        mpc_core,
        sample_raw_frame,
        mock_canonical_data,
        monkeypatch,
    ):
        mock_get_canonical.return_value = mock_canonical_data
        monkeypatch.setattr(
            mpc_core,
            "_build_bounce_point_mask",
            lambda **_kwargs: np.array([False, True, False, False], dtype=bool),
        )
        visibility = MpcVisibility(
            enabled=enabled,
            paths=paths,
            bounce_points=bounce_points,
        )

        view_model = mpc_core.create_view_model(
            step=0,
            raw_frame=sample_raw_frame,
            color_mode="mpc_type",
            mpc_visibility=visibility,
        )

        assert view_model is not None
        assert view_model.mpc_visibility == visibility
        assert bool(view_model.mpc_lines.size) is visibility.effective_paths
        assert bool(view_model.mpc_bounce_points.size) is visibility.effective_bounce_points
        assert view_model.mpc_line_itype_codes == ((1,) if visibility.effective_paths else ())

    @pytest.fixture
    def mock_canonical_data(self):
        """Fixture to provide mock CanonicalStepData."""
        return CanonicalStepData(
            points=np.array(
                [
                    [0, 0, 0],  # TX0
                    [1, 1, 1],  # RX0
                    [2, 2, 2],  # TX1
                    [3, 3, 3],  # RX1
                ],
                dtype=np.float32,
            ),
            lines=np.array([[0, 1], [2, 3]], dtype=np.int32),
            order=np.array([0, 0, 0, 0], dtype=np.int32),
            itype=np.array([1, 1, 1, 1], dtype=np.uint8),
            delay=np.array([10.0, 10.0, 20.0, 20.0], dtype=np.float32),
            loss=np.array([80.0, 80.0, 70.0, 70.0], dtype=np.float32),
            tx_id=np.array([0, 0, 1, 1], dtype=np.int16),
            rx_id=np.array([0, 0, 1, 1], dtype=np.int16),
            path_id=np.array([0, 0, 1, 1], dtype=np.int32),
            path_start_indices=np.array([0, 2], dtype=np.int32),
            path_orders=np.array([0, 0], dtype=np.uint8),
            path_delays=np.array([10.0, 20.0], dtype=np.float32),
            path_losses=np.array([80.0, 70.0], dtype=np.float32),
            path_tx=np.array([0, 1], dtype=np.int16),
            path_rx=np.array([0, 1], dtype=np.int16),
            segment_start_indices=np.array([0, 2], dtype=np.int32),
            segment_end_indices=np.array([1, 3], dtype=np.int32),
            segment_order=np.array([0, 0], dtype=np.uint8),
            segment_itype=np.array([1, 1], dtype=np.uint8),
            segment_delay=np.array([10.0, 20.0], dtype=np.float32),
            segment_loss=np.array([80.0, 70.0], dtype=np.float32),
            segment_tx_id=np.array([0, 1], dtype=np.int16),
            segment_rx_id=np.array([0, 1], dtype=np.int16),
            segment_path_id=np.array([0, 1], dtype=np.int32),
        )

    def test_init(self):
        """Test MPCCore initialization."""
        core = MPCCore()
        assert core.logger is not None
        assert core._canon_cache == {}

    def test_discover_tx_rx(self, mpc_core, sample_raw_frame):
        """Test discover_tx_rx method."""
        # Case 1: From tx_positions/rx_positions (inferred via num_tx/num_rx if present, or tx_rx_pairs)
        # The sample frame doesn't have num_tx/num_rx, but has tx_rx_pairs
        tx, rx = mpc_core.discover_tx_rx(sample_raw_frame)
        assert len(tx) == 2
        assert len(rx) == 2
        assert tx == [0, 1]
        assert rx == [0, 1]

        # Case 2: Explicit num_tx/num_rx
        frame_explicit = {"num_tx": 3, "num_rx": 4}
        tx, rx = mpc_core.discover_tx_rx(frame_explicit)
        assert len(tx) == 3
        assert len(rx) == 4

    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_create_view_model_basic(
        self, mock_get_canonical, mpc_core, sample_raw_frame, mock_canonical_data
    ):
        """Test basic ViewModel creation."""
        mock_get_canonical.return_value = mock_canonical_data

        vm = mpc_core.create_view_model(
            step=0, raw_frame=sample_raw_frame, color_mode="reflection_order"
        )

        assert isinstance(vm, ViewModel)
        assert len(vm.tx_positions) == 2
        assert len(vm.rx_positions) == 2
        assert len(vm.mpc_points) == 4
        assert len(vm.mpc_lines) == 2
        assert vm.stats_text is not None

    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_zero_filter_result_keeps_an_explicit_empty_path_mask(
        self, mock_get_canonical, mpc_core, sample_raw_frame, mock_canonical_data
    ):
        """An empty filter result must not fall back to the implicit all-path scope."""
        mock_get_canonical.return_value = mock_canonical_data

        vm = mpc_core.create_view_model(
            step=0,
            raw_frame=sample_raw_frame,
            color_mode="reflection_order",
            mpc_allowed_orders=[9],
        )

        assert vm is not None
        assert vm.path_mask is not None
        assert vm.path_mask.shape == (2,)
        assert not vm.path_mask.any()
        assert not vm.segment_mask.any()

    def test_get_canonical_accepts_serialized_canonical_payload(
        self, mpc_core, mock_canonical_data
    ):
        """Renderer-specific sources may supply serialized canonical fields."""
        raw_frame = {"canonical_data": dict(mock_canonical_data.__dict__)}

        canon = mpc_core._get_canonical(3, raw_frame)

        assert isinstance(canon, CanonicalStepData)
        np.testing.assert_allclose(canon.points, mock_canonical_data.points)
        np.testing.assert_array_equal(canon.lines, mock_canonical_data.lines)
        assert mpc_core._last_canonical_cache_hit is False

    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_create_view_model_filtering(
        self, mock_get_canonical, mpc_core, sample_raw_frame, mock_canonical_data
    ):
        """Test ViewModel creation with filtering."""
        mock_get_canonical.return_value = mock_canonical_data

        # Filter by TX (select TX 0)
        vm = mpc_core.create_view_model(
            step=0, raw_frame=sample_raw_frame, color_mode="reflection_order", selected_tx=0
        )

        assert len(vm.tx_positions) == 2
        assert len(vm.tx_orientations) == 2
        np.testing.assert_array_equal(vm.tx_positions, sample_raw_frame["tx_positions"])
        np.testing.assert_array_equal(vm.tx_orientations, sample_raw_frame["tx_orientations"])
        # Mock data has tx_id=0, so it should remain
        assert len(vm.mpc_points) == 2

        # Selecting TX 1 changes MPC visibility without changing node identity arrays.
        vm_empty = mpc_core.create_view_model(
            step=0, raw_frame=sample_raw_frame, color_mode="reflection_order", selected_tx=1
        )
        assert len(vm_empty.tx_positions) == 2
        assert len(vm_empty.tx_orientations) == 2
        np.testing.assert_array_equal(vm_empty.tx_positions, sample_raw_frame["tx_positions"])
        np.testing.assert_array_equal(
            vm_empty.tx_orientations,
            sample_raw_frame["tx_orientations"],
        )
        assert len(vm_empty.mpc_points) == 2

    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_create_view_model_color_modes(
        self, mock_get_canonical, mpc_core, sample_raw_frame, mock_canonical_data
    ):
        """Test different color modes."""
        mock_get_canonical.return_value = mock_canonical_data

        modes = ["reflection_order", "mpc_type", "delay", "path_loss"]
        for mode in modes:
            vm = mpc_core.create_view_model(step=0, raw_frame=sample_raw_frame, color_mode=mode)
            assert vm is not None
            assert vm.mpc_colors.shape[1] == 3

    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_create_view_model_uses_coarse_profile_by_default(
        self, mock_get_canonical, mpc_core, sample_raw_frame, mock_canonical_data, monkeypatch
    ):
        """Detailed MPC profiling should stay disabled outside benchmark/detail mode."""
        monkeypatch.delenv("ORCHAV_BENCH_PROFILE_DETAIL", raising=False)
        mock_get_canonical.return_value = mock_canonical_data

        vm = mpc_core.create_view_model(
            step=0, raw_frame=sample_raw_frame, color_mode="reflection_order"
        )

        assert vm is not None
        breakdown = mpc_core.get_last_viewmodel_breakdown()
        assert "filter_ms" in breakdown
        assert "mpc_filter_setup_ms" not in breakdown
        assert "mpc_visible_segments_count" not in breakdown

    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_create_view_model_records_detailed_mpc_profile(
        self, mock_get_canonical, mpc_core, sample_raw_frame, mock_canonical_data, monkeypatch
    ):
        """Detail mode should expose MPC counts and per-phase ViewModel timings."""
        monkeypatch.setenv("ORCHAV_BENCH_PROFILE_DETAIL", "1")
        mock_get_canonical.return_value = mock_canonical_data

        vm = mpc_core.create_view_model(
            step=0, raw_frame=sample_raw_frame, color_mode="reflection_order"
        )

        assert vm is not None
        breakdown = mpc_core.get_last_viewmodel_breakdown()
        assert breakdown["mpc_raw_points_count"] == 4.0
        assert breakdown["mpc_raw_segments_count"] == 2.0
        assert breakdown["mpc_raw_paths_count"] == 2.0
        assert breakdown["mpc_visible_points_count"] == 4.0
        assert breakdown["mpc_visible_segments_count"] == 2.0
        assert breakdown["mpc_visible_paths_count"] == 2.0
        assert breakdown["mpc_filter_fast_path"] == 1.0
        assert "mpc_line_payload_build_ms" in breakdown
        assert "mpc_point_payload_build_ms" in breakdown
        assert "mpc_line_payload_bytes" in breakdown
        assert "mpc_viewmodel_construct_ms" in breakdown

    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_create_view_model_material_mode_with_id_backed_materials(
        self, mock_get_canonical, mpc_core, sample_raw_frame, mock_canonical_data
    ):
        """Material mode should work with ID-backed canonical material data."""
        mock_canonical_data.material_ids = np.array([0, 0, 1, 1], dtype=np.int16)
        mock_canonical_data.material_names = None
        mock_canonical_data.material_itu_types = None
        mock_canonical_data.material_id_to_name = {0: "", 1: "mat-itu_glass"}
        mock_canonical_data.material_id_to_bare = {0: "", 1: "glass"}
        mock_canonical_data.material_id_to_itu = {0: "", 1: "glass"}
        mock_get_canonical.return_value = mock_canonical_data

        vm = mpc_core.create_view_model(step=0, raw_frame=sample_raw_frame, color_mode="material")

        assert vm is not None
        assert "Material" in vm.stats_text

    def test_summarize_mpcs_uses_path_mask_without_unique(self, mpc_core):
        """Path-mask summaries should not rediscover unique path ids."""
        canon = CanonicalStepData(
            points=np.zeros((6, 3), dtype=np.float32),
            lines=np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32),
            order=np.array([0, 0, 1, 1, 2, 2], dtype=np.uint8),
            itype=np.ones((6,), dtype=np.uint8),
            delay=np.zeros((6,), dtype=np.float32),
            loss=np.zeros((6,), dtype=np.float32),
            path_orders=np.array([0, 1, 2], dtype=np.uint8),
            path_delays=np.array([10.0, 20.0, 30.0], dtype=np.float32),
            path_losses=np.array([70.0, 80.0, 90.0], dtype=np.float32),
            segment_path_id=np.array([0, 1, 2], dtype=np.int32),
        )
        path_mask = np.array([True, False, True])
        segment_mask = np.array([True, False, True])
        point_mask = np.ones((6,), dtype=bool)

        with patch("visualizer.src.pipeline.core.np.unique", side_effect=AssertionError):
            summary = mpc_core._summarize_mpcs(
                canon=canon,
                color_mode="reflection_order",
                point_mask=point_mask,
                segment_mask=segment_mask,
                path_mask=path_mask,
            )

        assert summary["total_mpcs"] == 2
        assert summary["all_mpcs"] == 3
        assert summary["total_segments"] == 2
        assert summary["header"].startswith("Reflection Order")
        assert summary["lines"][0].endswith("LoS: 1")
        assert summary["lines"][1].endswith("2nd Order: 1")

    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_frame_material_choices_do_not_enable_filter(
        self, mock_get_canonical, sample_raw_frame, mock_canonical_data
    ):
        """Detected frame material choices must not become an active filter."""
        mock_canonical_data.material_ids = np.array([0, 1, 0, 2], dtype=np.int16)
        mock_canonical_data.segment_material_ids = np.array([0, 0], dtype=np.int16)
        mock_canonical_data.material_names = None
        mock_canonical_data.material_itu_types = None
        mock_canonical_data.material_id_to_name = {
            0: "",
            1: "mat-itu_glass",
            2: "mat-itu_plasterboard",
        }
        mock_canonical_data.material_id_to_bare = {0: "", 1: "glass", 2: "plasterboard"}
        mock_canonical_data.material_id_to_itu = {0: "", 1: "glass", 2: "plasterboard"}
        mock_get_canonical.return_value = mock_canonical_data
        visualizer = SimpleNamespace(
            renderer=SimpleNamespace(
                capabilities=RendererCapabilities(prefer_float32_frame_data=True)
            ),
            mpc_allowed_materials=None,
            mpc_material_filter_scope="segment",
            _mpc_material_filter_choices=set(),
            ui_controller=SimpleNamespace(populate_material_filters=MagicMock()),
        )
        core = MPCCore(logger=MagicMock(), visualizer=visualizer)

        vm = core.create_view_model(step=0, raw_frame=sample_raw_frame, color_mode="material")

        assert vm is not None
        assert visualizer.mpc_allowed_materials is None
        assert visualizer._mpc_material_filter_choices == {
            "glass",
            "no-material",
            "plasterboard",
        }
        visualizer.ui_controller.populate_material_filters.assert_called_once()
        assert vm.stats_text.startswith("MPCs: 2 | Segments:")

    def test_frame_material_choices_prefer_surface_name_over_itu(self):
        """MPC material choices should use frame material labels before ITU classes."""
        canon = CanonicalStepData(
            points=np.zeros((2, 3), dtype=np.float32),
            lines=np.array([[0, 1]], dtype=np.int32),
            order=np.array([0, 0], dtype=np.uint8),
            itype=np.array([0, 0], dtype=np.uint8),
            delay=np.zeros(2, dtype=np.float32),
            loss=np.zeros(2, dtype=np.float32),
            material_ids=np.array([0, 1], dtype=np.int16),
            material_id_to_name={0: "", 1: "ground_asphalt"},
            material_id_to_bare={0: "", 1: "asphalt"},
            material_id_to_itu={0: "", 1: "concrete"},
        )
        core = MPCCore(logger=MagicMock())

        assert core._get_all_frame_materials(canon) == ["asphalt", "no-material"]

        visualizer = SimpleNamespace(
            _mpc_material_filter_choices={"asphalt", "no-material"},
            mesh_entries=[],
            target_entries=[],
        )
        core = MPCCore(logger=MagicMock(), visualizer=visualizer)
        assert [name for name, _ in core.material_legend_items()] == ["asphalt", "no-material"]

    def test_frame_material_choices_collapse_target_specific_itu_labels(self):
        """Target-specific frame material IDs should display as their material family."""
        canon = CanonicalStepData(
            points=np.zeros((4, 3), dtype=np.float32),
            lines=np.array([[0, 1], [2, 3]], dtype=np.int32),
            order=np.zeros(4, dtype=np.uint8),
            itype=np.ones(4, dtype=np.uint8),
            delay=np.zeros(4, dtype=np.float32),
            loss=np.zeros(4, dtype=np.float32),
            material_ids=np.array([0, 1, 0, 2], dtype=np.int16),
            material_id_to_name={
                0: "",
                1: "ground_asphalt",
                2: "itu_metal_ParkedCar",
            },
            material_id_to_bare={0: "", 1: "asphalt", 2: "metal_parkedcar"},
            material_id_to_itu={0: "", 1: "concrete", 2: "metal"},
        )
        core = MPCCore(logger=MagicMock())

        assert core._get_all_frame_materials(canon) == ["asphalt", "metal", "no-material"]

    def test_material_filter_population_uses_normalized_environment_materials(self):
        """Initial MPC material choices should come from scene metadata without frame I/O."""

        class _MpcPanel:
            def __init__(self) -> None:
                self.material_ids = []
                self.checked = set()

            def set_materials(self, material_ids, checked):
                self.material_ids = list(material_ids)
                self.checked = set(checked)

        class _FrameSource:
            def list_frames(self):
                raise AssertionError("material filter population must not list frames")

            def load_frame(self, _step):
                raise AssertionError("material filter population must not load frames")

        mpc_panel = _MpcPanel()
        viz = SimpleNamespace(
            animation_step=0,
            frame_source=_FrameSource(),
            mpc_allowed_materials=None,
            _mpc_material_filter_choices=set(),
            _last_material_keys=None,
            mesh_entries=[
                {"material_id": "mat-itu_concrete", "material_type": "itu_concrete"},
                {"material_id": "mat-ground_asphalt"},
                {"material_type": "mat_itu_Glass"},
            ],
            target_entries=[
                {"material_id": "mat_itu_marble"},
                {"material_id": "mat-itu_glass_Pedestrian", "material_type": "glass"},
                {"material_id": "mat-itu_metal_ParkedCar", "material_type": "metal"},
            ],
            ui_manager=SimpleNamespace(panels={"mpc": mpc_panel}),
        )
        viz.mpc_core = MPCCore(logger=MagicMock(), visualizer=viz)
        parent = SimpleNamespace(
            visualizer=viz,
            material_mode_command_service=None,
            material_entry_edit_service=None,
        )

        MaterialUIController(parent).populate_material_filters()

        assert mpc_panel.material_ids == [
            "asphalt",
            "concrete",
            "glass",
            "marble",
            "metal",
            "no-material",
        ]
        assert mpc_panel.checked == set(mpc_panel.material_ids)
        assert viz._mpc_material_filter_choices == set(mpc_panel.material_ids)

    def test_material_filter_population_keeps_environment_when_frame_choices_exist(self):
        """A partial frame-material cache must not replace the environment list."""

        class _MpcPanel:
            def __init__(self) -> None:
                self.material_ids = []
                self.checked = set()

            def set_materials(self, material_ids, checked):
                self.material_ids = list(material_ids)
                self.checked = set(checked)

        mpc_panel = _MpcPanel()
        viz = SimpleNamespace(
            mpc_allowed_materials=None,
            _mpc_material_filter_choices={"mat-itu_concrete"},
            _last_material_keys=None,
            mesh_entries=[
                {"material_id": "mat-itu_marble"},
            ],
            target_entries=[],
            ui_manager=SimpleNamespace(panels={"mpc": mpc_panel}),
        )
        viz.mpc_core = MPCCore(logger=MagicMock(), visualizer=viz)
        parent = SimpleNamespace(
            visualizer=viz,
            material_mode_command_service=None,
            material_entry_edit_service=None,
        )

        MaterialUIController(parent).populate_material_filters()

        assert mpc_panel.material_ids == ["concrete", "marble", "no-material"]
        assert mpc_panel.checked == {"concrete", "marble", "no-material"}
        assert viz._mpc_material_filter_choices == {"concrete", "marble", "no-material"}

    def test_material_filter_population_normalizes_complete_allow_list(self):
        """A restored allow-list covering every available material is inactive."""

        class _MpcPanel:
            def __init__(self) -> None:
                self.material_ids = []
                self.checked = set()

            def set_materials(self, material_ids, checked):
                self.material_ids = list(material_ids)
                self.checked = set(checked)

        mpc_panel = _MpcPanel()
        viz = SimpleNamespace(
            mpc_allowed_materials={"concrete", "no-material"},
            _mpc_material_filter_choices={"concrete", "no-material"},
            _last_material_keys=None,
            mesh_entries=[{"material_type": "concrete"}],
            target_entries=[],
            ui_manager=SimpleNamespace(panels={"mpc": mpc_panel}),
        )
        viz.mpc_core = MPCCore(logger=MagicMock(), visualizer=viz)
        parent = SimpleNamespace(
            visualizer=viz,
            material_mode_command_service=None,
            material_entry_edit_service=None,
        )

        MaterialUIController(parent).populate_material_filters()

        assert viz.mpc_allowed_materials is None
        assert mpc_panel.checked == {"concrete", "no-material"}

    def test_distinct_material_colors_share_frame_and_scene_aliases(self):
        visualizer = SimpleNamespace(
            _mpc_material_filter_choices={"asphalt"},
            mesh_entries=[
                {
                    "material_id": "mat-ground_asphalt",
                    "color": [0.18, 0.18, 0.18],
                }
            ],
            target_entries=[],
        )
        core = MPCCore(logger=MagicMock(), visualizer=visualizer)

        colors = core._resolve_material_colors(use_distinct=True)
        legend = dict(core.material_legend_items(use_distinct=True))

        assert colors is not None
        np.testing.assert_allclose(legend["asphalt"], colors["asphalt"])
        np.testing.assert_allclose(legend["asphalt"], colors["ground_asphalt"])
        np.testing.assert_allclose(legend["asphalt"], colors["mat-ground_asphalt"])

    def test_active_material_legend_items_expose_only_explicit_filter_colors(self):
        visualizer = SimpleNamespace(
            mpc_allowed_materials={"glass"},
            _mpc_material_filter_choices={"concrete", "glass"},
            mesh_entries=[
                {"material_id": "concrete", "color": [0.4, 0.4, 0.4]},
                {"material_id": "glass", "color": [0.1, 0.3, 0.8]},
            ],
            target_entries=[],
        )
        core = MPCCore(logger=MagicMock(), visualizer=visualizer)

        active_items = core.material_legend_items(active_only=True)

        assert [label for label, _color in active_items] == ["glass"]
        np.testing.assert_allclose(active_items[0][1], [0.1, 0.3, 0.8])

        visualizer.mpc_allowed_materials = None
        assert core.material_legend_items(active_only=True) == []

    @pytest.mark.parametrize("use_distinct", [False, True])
    def test_active_material_legend_keeps_unresolved_material_with_render_fallback(
        self,
        use_distinct,
    ):
        visualizer = SimpleNamespace(
            mpc_allowed_materials={"custom_coating"},
            _mpc_material_filter_choices={"custom_coating"},
            mesh_entries=[],
            target_entries=[],
        )
        core = MPCCore(logger=MagicMock(), visualizer=visualizer)
        material_colors = core._resolve_material_colors(use_distinct)
        legend = dict(
            core.material_legend_items(
                use_distinct,
                active_only=True,
            )
        )
        canon = CanonicalStepData(
            points=np.zeros((2, 3), dtype=np.float32),
            lines=np.array([[0, 1]], dtype=np.int32),
            order=np.zeros(2, dtype=np.uint8),
            itype=np.ones(2, dtype=np.uint8),
            delay=np.zeros(2, dtype=np.float32),
            loss=np.zeros(2, dtype=np.float32),
            material_ids=np.array([1, 0], dtype=np.int16),
            segment_material_ids=np.array([1], dtype=np.int16),
            material_id_to_name={0: "", 1: "custom_coating"},
            material_id_to_bare={0: "", 1: "custom_coating"},
            material_id_to_itu={0: "", 1: ""},
        )

        rendered = colorize_segments(
            canon,
            np.array([True]),
            "material",
            np.zeros((10, 3), dtype=np.float32),
            np.zeros((10, 3), dtype=np.float32),
            ensure_luts(),
            material_colors=material_colors,
        )

        assert list(legend) == ["custom_coating"]
        np.testing.assert_allclose(legend["custom_coating"], rendered[0])
        np.testing.assert_allclose(legend["custom_coating"], [0.7, 0.7, 0.7])

    def test_active_material_legend_uses_distinct_color_for_custom_material_when_available(
        self,
    ):
        visualizer = SimpleNamespace(
            mpc_allowed_materials={"custom_coating"},
            _mpc_material_filter_choices={"custom_coating", "glass"},
            mesh_entries=[
                {"material_id": "glass", "color": [0.1, 0.3, 0.8]},
            ],
            target_entries=[],
        )
        core = MPCCore(logger=MagicMock(), visualizer=visualizer)

        colors = core._resolve_material_colors(use_distinct=True)
        legend = dict(
            core.material_legend_items(
                use_distinct=True,
                active_only=True,
            )
        )

        assert colors is not None
        np.testing.assert_allclose(legend["custom_coating"], colors["custom_coating"])
        assert not np.allclose(legend["custom_coating"], [0.7, 0.7, 0.7])

    def test_mpc_type_summary_uses_canonical_virtual_and_unknown_labels(self):
        core = MPCCore(logger=MagicMock())

        lines = core._mpc_type_summary_lines(np.array([99, 8, 255, 1, 99, 42], dtype=np.uint8))

        assert lines == [
            "- Specular: 1",
            "- Diffraction: 1",
            "- Virtual: 2",
            "- Unknown (Type 42): 1",
            "- Unknown (Type 255): 1",
        ]

    def test_distinct_material_colors_collapse_target_specific_aliases(self):
        """Distinct MPC colors should match legend colors for target-specific materials."""
        visualizer = SimpleNamespace(
            _mpc_material_filter_choices={"asphalt", "metal", "no-material"},
            mesh_entries=[
                {
                    "material_id": "ground_asphalt",
                    "material_type": "asphalt",
                    "color": [0.18, 0.18, 0.18],
                }
            ],
            target_entries=[
                {
                    "material_id": "mat-itu_metal_ParkedCar",
                    "material_type": "metal",
                    "color": [0.22, 0.22, 0.25],
                }
            ],
        )
        core = MPCCore(logger=MagicMock(), visualizer=visualizer)
        colors = core._resolve_material_colors(use_distinct=True)
        legend = dict(core.material_legend_items(use_distinct=True))
        canon = CanonicalStepData(
            points=np.zeros((2, 3), dtype=np.float32),
            lines=np.array([[0, 1]], dtype=np.int32),
            order=np.zeros(2, dtype=np.uint8),
            itype=np.ones(2, dtype=np.uint8),
            delay=np.zeros(2, dtype=np.float32),
            loss=np.zeros(2, dtype=np.float32),
            material_ids=np.array([2, 0], dtype=np.int16),
            segment_material_ids=np.array([2], dtype=np.int16),
            material_id_to_name={0: "", 1: "ground_asphalt", 2: "itu_metal_ParkedCar"},
            material_id_to_bare={0: "", 1: "asphalt", 2: "metal_parkedcar"},
            material_id_to_itu={0: "", 1: "concrete", 2: "metal"},
        )

        assert colors is not None
        np.testing.assert_allclose(colors["mat-itu_metal_ParkedCar"], colors["metal"])

        segment_colors = colorize_segments(
            canon,
            np.array([True]),
            "material",
            np.zeros((10, 3), dtype=np.float32),
            np.zeros((10, 3), dtype=np.float32),
            ensure_luts(),
            material_colors=colors,
        )

        np.testing.assert_allclose(segment_colors[0], legend["metal"])

    def test_non_distinct_material_colors_prefer_family_over_target_visual_alias(self):
        """Non-distinct MPC colors should not inherit target visual-profile colors."""
        skin_color = np.array([0.82, 0.62, 0.49], dtype=np.float32)
        visualizer = SimpleNamespace(
            _mpc_material_filter_choices={"glass", "no-material"},
            mesh_entries=[],
            target_entries=[
                {
                    "material_id": "mat-itu_glass_Pedestrian",
                    "material_type": "glass",
                    "color": skin_color.tolist(),
                }
            ],
        )
        core = MPCCore(logger=MagicMock(), visualizer=visualizer)
        colors = core._resolve_material_colors(use_distinct=False)
        legend = dict(core.material_legend_items(use_distinct=False))
        canon = CanonicalStepData(
            points=np.zeros((2, 3), dtype=np.float32),
            lines=np.array([[0, 1]], dtype=np.int32),
            order=np.zeros(2, dtype=np.uint8),
            itype=np.ones(2, dtype=np.uint8),
            delay=np.zeros(2, dtype=np.float32),
            loss=np.zeros(2, dtype=np.float32),
            material_ids=np.array([1, 0], dtype=np.int16),
            segment_material_ids=np.array([1], dtype=np.int16),
            material_id_to_name={0: "", 1: "itu_glass_Pedestrian"},
            material_id_to_bare={0: "", 1: "glass_pedestrian"},
            material_id_to_itu={0: "", 1: "glass"},
        )

        assert colors is not None
        segment_colors = colorize_segments(
            canon,
            np.array([True]),
            "material",
            np.zeros((10, 3), dtype=np.float32),
            np.zeros((10, 3), dtype=np.float32),
            ensure_luts(),
            material_colors=colors,
        )

        np.testing.assert_allclose(segment_colors[0], legend["glass"])
        assert not np.allclose(segment_colors[0], skin_color)

    def test_material_filter_controls_bounce_markers(self):
        """Segment material filtering should also filter physical bounce markers."""
        canonical = CanonicalStepData(
            points=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            lines=np.array([[0, 1], [1, 2], [3, 4]], dtype=np.int32),
            order=np.array([1, 1, 1, 0, 0], dtype=np.uint8),
            itype=np.array([0, 1, 0, 0, 0], dtype=np.uint8),
            delay=np.zeros(5, dtype=np.float32),
            loss=np.zeros(5, dtype=np.float32),
            tx_id=np.zeros(5, dtype=np.int16),
            rx_id=np.zeros(5, dtype=np.int16),
            path_id=np.array([0, 0, 0, 1, 1], dtype=np.int32),
            path_start_indices=np.array([0, 3], dtype=np.int32),
            path_orders=np.array([1, 0], dtype=np.uint8),
            path_delays=np.zeros(2, dtype=np.float32),
            path_losses=np.zeros(2, dtype=np.float32),
            path_tx=np.zeros(2, dtype=np.int16),
            path_rx=np.zeros(2, dtype=np.int16),
            segment_start_indices=np.array([0, 1, 3], dtype=np.int32),
            segment_end_indices=np.array([1, 2, 4], dtype=np.int32),
            segment_order=np.array([1, 1, 0], dtype=np.uint8),
            segment_itype=np.array([1, 1, 0], dtype=np.uint8),
            segment_path_id=np.array([0, 0, 1], dtype=np.int32),
            material_ids=np.array([0, 1, 0, 0, 0], dtype=np.int16),
            segment_material_ids=np.array([0, 1, 0], dtype=np.int16),
            material_id_to_name={0: "", 1: "ground_asphalt"},
            material_id_to_itu={0: "", 1: "concrete"},
            material_id_to_bare={0: "", 1: "asphalt"},
        )
        raw_frame = {
            "canonical_data": canonical,
            "tx_positions": np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
            "rx_positions": np.array([[2.0, 0.0, 1.0]], dtype=np.float64),
            "tx_orientations": np.zeros((1, 3), dtype=np.float64),
            "rx_orientations": np.zeros((1, 3), dtype=np.float64),
            "tx_rx_pairs": np.array([[0, 0]], dtype=np.int32),
            "targets_metadata": [],
        }
        visualizer = SimpleNamespace(
            renderer=SimpleNamespace(
                capabilities=RendererCapabilities(prefer_float32_frame_data=True)
            ),
            mpc_allowed_materials=None,
            mpc_material_filter_scope="segment",
            _mpc_material_filter_choices=set(),
            ui_controller=SimpleNamespace(populate_material_filters=MagicMock()),
            mesh_entries=[],
            target_entries=[],
            beamforming_service=None,
        )
        core = MPCCore(logger=MagicMock(), visualizer=visualizer)
        common = {
            "step": 0,
            "raw_frame": raw_frame,
            "color_mode": "material",
            "selected_tx": "all",
            "selected_rx": "all",
            "mpc_allowed_orders": [0, 1],
            "mpc_allowed_types": [0, 1, 2, 4, 8],
            "mpc_visibility": MpcVisibility(),
        }

        asphalt_vm = core.create_view_model(**common, mpc_allowed_materials={"asphalt"})
        no_material_vm = core.create_view_model(**common, mpc_allowed_materials={"no-material"})

        assert asphalt_vm is not None
        assert no_material_vm is not None
        assert asphalt_vm.mpc_bounce_points.shape[0] == 1
        assert no_material_vm.mpc_bounce_points.shape[0] == 0

    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_create_view_model_topk_uses_strongest_paths(
        self, mock_get_canonical, mpc_core, sample_raw_frame, mock_canonical_data
    ):
        """Top-K render cap should keep the lowest-loss visible paths only."""
        mock_get_canonical.return_value = mock_canonical_data
        mock_canonical_data.segment_itype = np.array([2, 99], dtype=np.uint8)

        vm = mpc_core.create_view_model(
            step=0,
            raw_frame=sample_raw_frame,
            color_mode="mpc_type",
            mpc_allowed_types=[0, 1, 2, 99],
            topk_render_enabled=True,
            topk_render_max_paths=1,
        )

        assert vm is not None
        assert vm.mpc_lines.shape[0] == 1
        assert vm.mpc_points.shape[0] == 2
        # Path with lower loss (70 dB) is path_id=1 -> points [2,2,2] and [3,3,3].
        np.testing.assert_allclose(vm.mpc_points, np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))
        assert vm.stats_text.startswith("MPCs: 1/2")
        # Stats mask is pre-cap (Top-K must not reduce stats scope).
        assert vm.path_mask is not None
        assert int(vm.path_mask.sum()) == 2
        assert "Rendered MPCs: 1/2" in vm.stats_text
        assert vm.mpc_line_itype_codes == (99,)

    def test_stats_without_canonical_returns_empty(self, mpc_core, sample_raw_frame):
        """Test stats returns empty stats when no canonical_data present."""
        stats = mpc_core.stats(sample_raw_frame)
        assert stats.total_paths == 0
        assert stats.orders_hist == {}
        assert stats.delay_range_ns is None
        assert stats.path_loss_range is None

    def test_stats_canonical(self, mpc_core, mock_canonical_data):
        """Test stats computation with canonical data."""
        payload = {"canonical_data": mock_canonical_data, "metrics_visible": True}
        # We need to mock MPCStatsComputer since it's imported inside the method
        with patch("visualizer.src.metrics.mpc_stats.MPCStatsComputer") as MockComputer:
            mock_computer_instance = MockComputer.return_value
            mock_computer_instance.compute_frame_stats.return_value = "mock_stats"

            stats = mpc_core.stats(payload)
            assert stats == "mock_stats"
            mock_computer_instance.compute_frame_stats.assert_called_once_with(
                mock_canonical_data, True, path_mask=None
            )

    def test_canonical_cache_accounts_for_metric_provenance(self, mpc_core):
        """Retained provenance masks participate in the canonical LRU budget."""
        empty_f32 = np.empty((0,), dtype=np.float32)
        canonical = CanonicalStepData(
            points=np.empty((0, 3), dtype=np.float32),
            lines=np.empty((0, 2), dtype=np.int32),
            order=np.empty((0,), dtype=np.uint8),
            itype=np.empty((0,), dtype=np.uint8),
            delay=empty_f32,
            loss=empty_f32,
            path_delay_is_estimated=np.zeros(3, dtype=bool),
            path_loss_is_estimated=np.ones(5, dtype=bool),
        )

        assert mpc_core._estimate_canon_bytes(canonical) == 8

    def test_canonical_cache_accounts_for_angle_and_material_arrays(self, mpc_core):
        """Every ndarray in the canonical contract participates in the byte budget."""
        empty_f32 = np.empty((0,), dtype=np.float32)
        canonical = CanonicalStepData(
            points=np.empty((0, 3), dtype=np.float32),
            lines=np.empty((0, 2), dtype=np.int32),
            order=np.empty((0,), dtype=np.uint8),
            itype=np.empty((0,), dtype=np.uint8),
            delay=empty_f32,
            loss=empty_f32,
            path_aoa_az=np.arange(2, dtype=np.float32),
            path_aoa_el=np.arange(2, dtype=np.float32),
            path_aod_az=np.arange(2, dtype=np.float32),
            path_aod_el=np.arange(2, dtype=np.float32),
            material_names=np.asarray(("concrete", "glass"), dtype=object),
            material_ids=np.asarray((1, 2), dtype=np.int16),
            material_itu_types=np.asarray(("concrete", "glass"), dtype=object),
            aoa_az=np.arange(3, dtype=np.float32),
            aoa_el=np.arange(3, dtype=np.float32),
            aod_az=np.arange(3, dtype=np.float32),
            aod_el=np.arange(3, dtype=np.float32),
        )
        expected = sum(
            int(value.nbytes)
            for value in (
                canonical.path_aoa_az,
                canonical.path_aoa_el,
                canonical.path_aod_az,
                canonical.path_aod_el,
                canonical.material_names,
                canonical.material_ids,
                canonical.material_itu_types,
                canonical.aoa_az,
                canonical.aoa_el,
                canonical.aod_az,
                canonical.aod_el,
            )
        )

        assert mpc_core._estimate_canon_bytes(canonical) == expected

    def test_canonical_cache_counts_shared_backing_buffer_once(self, mpc_core):
        """Views charge their retained owner once, including bytes outside each view."""
        backing = np.arange(64, dtype=np.uint8)
        empty_f32 = np.empty((0,), dtype=np.float32)
        canonical = CanonicalStepData(
            points=np.empty((0, 3), dtype=np.float32),
            lines=np.empty((0, 2), dtype=np.int32),
            order=backing[:8],
            itype=backing[8:16],
            delay=empty_f32,
            loss=empty_f32,
            path_orders=backing[16:20],
            segment_order=backing[20:24],
        )

        assert mpc_core._estimate_canon_bytes(canonical) == backing.nbytes

    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_canonical_cache_invalidation(
        self, mock_get_canonical, mpc_core, sample_raw_frame, mock_canonical_data
    ):
        """Ensure canonical cache is reused and invalidated correctly."""
        mock_get_canonical.return_value = mock_canonical_data

        mpc_core.create_view_model(
            step=0,
            raw_frame=sample_raw_frame,
            color_mode="reflection_order",
        )
        mpc_core.create_view_model(
            step=0,
            raw_frame=sample_raw_frame,
            color_mode="reflection_order",
        )
        assert mock_get_canonical.call_count == 1  # cache hit

        mpc_core.invalidate_step(step=0)
        mpc_core.create_view_model(
            step=0,
            raw_frame=sample_raw_frame,
            color_mode="reflection_order",
        )
        assert mock_get_canonical.call_count == 2  # recomputed after invalidation

    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_create_view_model_with_targets_and_colorbar(
        self, mock_get_canonical, sample_raw_frame, mock_canonical_data
    ):
        """Verify target metadata extraction and colorbar generation."""
        mock_canonical_data.delay_min = 5.0
        mock_canonical_data.delay_max = 10.0
        mock_get_canonical.return_value = mock_canonical_data

        sample_raw_frame["targets_metadata"] = [
            {
                "name": "vehicle",
                "current_position": [1.0, 2.0, 3.0],
                "orientation": [0.0, 0.0, 90.0],
                "mesh_file": "car.ply",
                "mesh_index": 7,
                "use_ply_position": True,
            }
        ]

        core = MPCCore()
        view_model = core.create_view_model(
            step=3,
            raw_frame=sample_raw_frame,
            color_mode="delay",
        )

        assert view_model.colorbar == ("Delay (ns)", (5.0, 10.0))
        assert view_model.target_positions.shape == (1, 3)
        assert view_model.target_mesh_files == ["car.ply"]
        assert view_model.target_metadata[0]["name"] == "vehicle"
        assert view_model.target_metadata[0]["position_valid"] is True

    @patch("visualizer.src.pipeline.core.MPCCore._canonical_from_payload")
    def test_create_view_model_marks_invalid_missing_target_position(
        self, mock_get_canonical, sample_raw_frame, mock_canonical_data
    ):
        """Missing target positions should be flagged invalid instead of treated as real origin."""
        mock_get_canonical.return_value = mock_canonical_data
        sample_raw_frame["targets_metadata"] = [
            {
                "name": "vehicle",
                "current_position": None,
                "orientation": [0.0, 0.0, 0.0],
                "mesh_file": "car.ply",
                "use_ply_position": False,
            }
        ]
        sample_raw_frame["target_pos"] = np.zeros((0, 3), dtype=np.float32)

        core = MPCCore()
        view_model = core.create_view_model(
            step=3,
            raw_frame=sample_raw_frame,
            color_mode="delay",
        )

        np.testing.assert_allclose(view_model.target_positions[0], [0.0, 0.0, 0.0])
        assert view_model.target_metadata[0]["position_valid"] is False
