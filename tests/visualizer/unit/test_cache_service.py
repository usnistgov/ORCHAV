"""
Test suite for CacheService.

Tests cache clearing logic for MPC, renderer, and animation services.
"""

import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tests.visualizer.fixtures.mock_factories import make_mock_visualizer
from visualizer.src.model import RenderObjectState
from visualizer.src.scene.target_transforms import TargetGeometryMeta
from visualizer.src.services.cache_service import CacheInvalidationScope, CacheService
from visualizer.src.services.target_asset_cache import (
    ResolvedTargetAssetSource,
    TargetAsset,
    TargetAssetCache,
    TargetAssetKey,
    TargetRuntimeState,
    TargetSourceRevision,
)
from visualizer.src.types.render_payloads import MeshPayload


def _target_asset() -> TargetAsset:
    """Return one complete in-memory asset for invalidation tests."""
    logical_key = ("target", "mesh.ply")
    canonical_path = "memory://target/mesh.ply"
    source = ResolvedTargetAssetSource(
        target_name=logical_key[0],
        mesh_filename=logical_key[1],
        canonical_path=canonical_path,
        key=TargetAssetKey(
            canonical_path=canonical_path,
            revision=TargetSourceRevision(0, 0, 0, 0),
            target_name=logical_key[0],
            mesh_filename=logical_key[1],
        ),
    )
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    mesh = RenderObjectState(
        id="target:target::mesh",
        payload=MeshPayload(
            vertices=vertices.copy(),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )
    return TargetAsset(
        source=source,
        mesh=mesh,
        original_vertices=vertices.copy(),
        scaled_vertices=vertices.copy(),
        geometry_meta=TargetGeometryMeta(scaled_aabb_center=np.asarray([0.5, 0.5, 0.0])),
    )


class TestCacheService:
    """Test CacheService functionality."""

    def test_clear_local_frame_caches_preserves_reusable_assets(self):
        """Frame clearing must not evict coverage or other reusable assets."""
        mock_viz = make_mock_visualizer()

        # Setup caches
        mock_viz.mpc_view_cache = {0: "data"}
        mock_viz.last_app_state = "state"

        # Setup services
        mock_viz.animation_service = Mock()
        mock_viz.animation_service.clear_preload_data = Mock(return_value=1)
        mock_viz.coverage_service = Mock()

        # Setup MPCCore
        mock_viz.mpc_core = Mock()
        mock_viz.mpc_core._canon_cache = ["data"]

        service = CacheService(mock_viz)
        service.store_frame(0, {"data": "value"}, source="override")
        service.clear_local_frame_caches(reason="test")

        # Verify caches cleared
        assert service.frame_cache.size == 0
        assert len(mock_viz.mpc_view_cache) == 0
        assert mock_viz.last_app_state is None

        # Verify service calls
        mock_viz.animation_service.clear_preload_data.assert_called_once_with(reset_cache_size=True)
        mock_viz.coverage_service.clear.assert_not_called()
        mock_viz.mpc_core.invalidate_step.assert_called_once_with(None)

    def test_cache_log_messages_encode_with_windows_cp1252(self, caplog):
        """Routine cache logs must be safe for the supported Windows console."""
        mock_viz = make_mock_visualizer()
        mock_viz.mpc_view_cache = {0: "view"}
        mock_viz.last_app_state = "state"
        mock_viz.animation_service = Mock()
        mock_viz.animation_service.clear_preload_data = Mock(return_value=1)
        mock_viz.coverage_service = Mock()
        mock_viz.mpc_core = Mock()
        mock_viz.mpc_core._canon_cache = {0: "canon"}
        mock_viz.renderer = SimpleNamespace(last_frame_packet="packet", vis_initialized=False)

        service = CacheService(mock_viz, max_frame_cache_size=1)
        service.store_frame(0, {"data": "value"}, source="override")

        with caplog.at_level(logging.INFO, logger="orchav.cache_service"):
            service.ensure_frame_cache_capacity(2)
            service.clear_local_frame_caches(reason="cp1252")

        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == "orchav.cache_service"
        ]
        assert messages
        for message in messages:
            message.encode("cp1252", errors="strict")

    def test_clear_static_asset_caches_uses_explicit_asset_lifecycle(self):
        """Asset clearing leaves raw frames intact and clears each asset owner once."""
        mock_viz = make_mock_visualizer()
        mock_viz.mpc_view_cache = {0: "view"}
        mock_viz.renderer = SimpleNamespace(name="renderer")
        mock_viz.target_asset_cache = SimpleNamespace(
            clear_inactive_assets=Mock(return_value={"entries": 2, "bytes": 128, "pending": 1})
        )
        mock_viz.material_pbr_service = SimpleNamespace(invalidate_material_resolution_cache=Mock())
        service = CacheService(mock_viz)
        service.store_frame(0, {"data": "value"})

        with patch(
            "visualizer.src.services.cache_service.clear_reusable_asset_caches",
            return_value={"decoded_texture_memory": {"entries": 3}},
        ) as clear_assets:
            result = service.clear_static_asset_caches(reason="unit")

        assert service.frame_cache.size == 1
        assert mock_viz.mpc_view_cache == {0: "view"}
        mock_viz.material_pbr_service.invalidate_material_resolution_cache.assert_called_once_with()
        mock_viz.target_asset_cache.clear_inactive_assets.assert_called_once_with()
        clear_assets.assert_called_once_with(mock_viz.renderer, include_disk=True)
        assert result == {
            "target_assets": {"entries": 2, "bytes": 128, "pending": 1},
            "layers": {"decoded_texture_memory": {"entries": 3}},
        }

    def test_clear_local_frame_caches_resets_renderer(self):
        """Test that renderer state is reset."""
        mock_viz = make_mock_visualizer()

        # Disable mpc_core for this test to avoid len(Mock) error
        mock_viz.mpc_core = None

        mock_viz.mpc_view_cache = {}

        # Create proper mocks for services
        mock_viz.animation_service = Mock()
        mock_viz.animation_service.clear_preload_data = Mock(return_value=0)
        mock_viz.animation_service.clear_cache = Mock()
        mock_viz.coverage_service = Mock()
        mock_viz.coverage_service.clear = Mock()

        # Setup renderer
        mock_viz.renderer = Mock()
        mock_viz.renderer.last_frame_packet = "packet"
        mock_viz.renderer.mpc_lineset = Mock()
        mock_viz.renderer.mpc_pcd = Mock()
        mock_viz.renderer.vis_initialized = True

        service = CacheService(mock_viz)

        # Mock the renderer module imports inside the method
        with patch.dict(
            "sys.modules",
            {
                "src.renderer": Mock(
                    EMPTY_VECTOR3D=Mock(), EMPTY_VECTOR2I=Mock(), EMPTY_VECTOR3D_COLORS=Mock()
                )
            },
        ):
            service.clear_local_frame_caches()

        # Verify renderer reset
        assert mock_viz.renderer.last_frame_packet is None
        assert mock_viz.renderer.update_renderer.called

    def test_clear_local_frame_caches_skips_missing_renderer_geometries(self):
        """Cache clearing should tolerate renderers without Open3D MPC objects."""
        mock_viz = make_mock_visualizer()
        mock_viz.mpc_core = None
        mock_viz.mpc_view_cache = {}
        mock_viz.animation_service = Mock()
        mock_viz.animation_service.clear_preload_data = Mock(return_value=0)
        mock_viz.coverage_service = Mock()
        mock_viz.coverage_service.clear = Mock()
        mock_viz.renderer = SimpleNamespace(
            last_frame_packet="packet",
            mpc_lineset=None,
            mpc_pcd=None,
            vis_initialized=True,
            update_geometry_in_visualizer=Mock(),
            update_renderer=Mock(),
        )

        service = CacheService(mock_viz)
        service.clear_local_frame_caches(reason="PARAM_UPDATE")

        assert mock_viz.renderer.last_frame_packet is None
        mock_viz.renderer.update_geometry_in_visualizer.assert_not_called()
        mock_viz.renderer.update_renderer.assert_called_once()

    def test_reset_bounce_cache(self):
        """Test that bounce caches are reset to empty arrays."""
        mock_viz = make_mock_visualizer()

        # Setup bounce caches with data
        import numpy as np

        mock_viz._cached_bounce_points = np.array([[1, 2, 3]])
        mock_viz._cached_bounce_colors = np.array([[1, 0, 0]])

        service = CacheService(mock_viz)
        service._reset_bounce_cache("_cached_bounce_points", "test")

        # Verify reset to empty
        assert len(mock_viz._cached_bounce_points) == 0
        assert mock_viz._cached_bounce_points.shape == (0, 3)

    def test_scoped_material_invalidation_preserves_frame_cache(self):
        """Material/color invalidation clears derived caches but keeps raw frames."""
        mock_viz = make_mock_visualizer()
        mock_viz.mpc_view_cache = {0: "view"}
        mock_viz.last_app_state = "state"
        mock_viz.renderer = SimpleNamespace(
            last_frame_packet="packet",
            vis_initialized=False,
        )
        mock_viz.mpc_core = Mock()
        mock_viz.mpc_core._canon_cache = {0: "canon"}
        mock_viz.mpc_core.invalidate_material_colors_cache = Mock()

        service = CacheService(mock_viz)
        service.store_frame(0, {"data": "value"})

        result = service.invalidate(
            CacheInvalidationScope.MATERIALS_COLORS,
            reason="unit",
        )

        assert service.frame_cache.size == 1
        assert len(mock_viz.mpc_view_cache) == 0
        assert mock_viz.last_app_state is None
        assert mock_viz.renderer.last_frame_packet is None
        assert result["frame_cache"] == 0
        assert result["view_model_cache"] == 1
        mock_viz.mpc_core.invalidate_step.assert_not_called()
        mock_viz.mpc_core.invalidate_material_colors_cache.assert_called_once_with()

    def test_frame_data_invalidation_clears_derived_view_models(self):
        """Frame-data invalidation clears raw, canonical, and derived view caches."""
        mock_viz = make_mock_visualizer()
        mock_viz.mpc_view_cache = {0: "view"}
        mock_viz.last_app_state = "state"
        mock_viz.renderer = SimpleNamespace(last_frame_packet="packet", vis_initialized=False)
        mock_viz.mpc_core = Mock()
        mock_viz.mpc_core._canon_cache = {0: "canon"}
        mock_viz.animation_service = Mock()
        mock_viz.animation_service.clear_preload_data = Mock(return_value=0)

        service = CacheService(mock_viz)
        service.store_frame(0, {"data": "value"})

        result = service.invalidate(CacheInvalidationScope.FRAME_DATA, reason="unit")

        assert service.frame_cache.size == 0
        assert len(mock_viz.mpc_view_cache) == 0
        assert mock_viz.last_app_state is None
        assert mock_viz.renderer.last_frame_packet is None
        assert result["frame_cache"] == 1
        assert result["view_model_cache"] == 1
        assert result["canonical_cache"] == 1
        mock_viz.mpc_core.invalidate_step.assert_called_once_with(None)

    def test_target_geometry_invalidation_clears_target_runtime_caches(self):
        """Target geometry invalidation clears the one typed target owner."""
        mock_viz = make_mock_visualizer()
        mock_viz.mpc_view_cache = {}
        mock_viz.renderer = SimpleNamespace(last_frame_packet=None, vis_initialized=False)
        cache = mock_viz.target_asset_cache
        assert isinstance(cache, TargetAssetCache)
        cache.put(_target_asset())
        cache.runtime_states["target"] = TargetRuntimeState(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            mesh_filename="mesh.ply",
            position_valid=True,
            use_ply_position=False,
            runtime_visible=True,
        )

        service = CacheService(mock_viz)
        result = service.invalidate(CacheInvalidationScope.TARGET_GEOMETRY, reason="unit")

        assert cache.logical_keys() == ()
        assert cache.runtime_states == {}
        assert result["target_cache"] == 2

    def test_fallback_frame_data_invalidation_supports_notebook_core(self):
        """The lightweight fallback can invalidate notebook ``_mpc_core`` objects."""
        from visualizer.src.services.cache_service import invalidate_visualizer_cache

        mpc_core = Mock()
        notebook = SimpleNamespace(_mpc_core=mpc_core)

        invalidate_visualizer_cache(
            notebook,
            CacheInvalidationScope.FRAME_DATA,
            reason="unit",
        )

        mpc_core.invalidate_step.assert_called_once_with(step=None)

    def test_invalidate_canonical_step_preserves_frame_cache(self):
        """Single-step canonical invalidation does not remove stored raw frames."""
        mock_viz = make_mock_visualizer()
        mock_viz.mpc_view_cache = {0: "view"}
        mock_viz.last_app_state = "state"
        mock_viz.renderer = SimpleNamespace(last_frame_packet="packet", vis_initialized=False)
        mock_viz.mpc_core = Mock()
        mock_viz.mpc_core._canon_cache = {0: "canon"}

        service = CacheService(mock_viz)
        service.store_frame(0, {"data": "value"}, source="override")

        result = service.invalidate_canonical_step(0, reason="unit")

        assert service.frame_cache.size == 1
        assert len(mock_viz.mpc_view_cache) == 0
        assert mock_viz.last_app_state is None
        assert mock_viz.renderer.last_frame_packet is None
        assert result["frame_cache"] == 0
        assert result["canonical_cache"] == 1
        mock_viz.mpc_core.invalidate_step.assert_called_once_with(0)

    def test_cache_telemetry_reports_frame_and_renderer_counters(self):
        """Telemetry exposes hit/miss/eviction counters and renderer byte budget."""
        mock_viz = make_mock_visualizer()
        mock_viz.mpc_view_cache = {0: "view", 1: "view"}
        mock_viz.renderer = SimpleNamespace(
            get_runtime_stats=lambda: {
                "mpc_line_cache_bytes": 1024,
                "mpc_line_cache_max_bytes": 4096,
                "mpc_line_cache_hits": 3,
                "mpc_line_cache_misses": 1,
                "mpc_line_cache_evictions": 2,
            }
        )

        service = CacheService(mock_viz, max_frame_cache_size=1)
        service.store_frame(0, {"data": "first"})
        assert service.get_frame(0) is not None
        assert service.get_frame(1) is None
        service.store_frame(1, {"data": "second"})

        telemetry = service.get_cache_telemetry()

        assert telemetry["frame_cache_size"] == 1
        assert telemetry["frame_cache_max_size"] == 1
        assert telemetry["frame_cache_hits"] == 1
        assert telemetry["frame_cache_misses"] == 1
        assert telemetry["frame_cache_evictions"] == 1
        assert telemetry["view_model_cache_size"] == 2
        assert telemetry["mpc_line_cache_bytes"] == 1024
        assert telemetry["mpc_line_cache_max_bytes"] == 4096
        assert telemetry["mpc_line_cache_hits"] == 3
