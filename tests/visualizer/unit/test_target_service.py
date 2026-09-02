"""Unit tests for target runtime visibility behavior."""

from __future__ import annotations

import glob
import os
import tempfile
from dataclasses import replace
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tests.visualizer.fixtures.mock_factories import (
    RecordingBatchRenderer,
    make_mock_visualizer,
)
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from visualizer.src.materials.appearance import MaterialDisplayMode
from visualizer.src.materials.catalog import pbr_props_to_kwargs, resolve_pbr_material
from visualizer.src.model import (
    RenderObjectState,
    Transform,
    make_text_label_state,
    render_state_points,
)
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.scene.target_transforms import TargetGeometryMeta
from visualizer.src.services.cache_service import CacheService
from visualizer.src.services.material_modes import MaterialModeService
from visualizer.src.services.object_appearance_service import ObjectAppearanceService
from visualizer.src.services.object_identity import make_target_entry_geometry_name
from visualizer.src.services.target_asset_cache import (
    ResolvedTargetAssetSource,
    TargetAsset,
    TargetAssetCache,
    TargetAssetKey,
    TargetRuntimeState,
    TargetSourceRevision,
)
from visualizer.src.services.target_service import TargetService
from visualizer.src.types.render_payloads import MeshPayload, SurfaceColorSource

_MESH_ID_COUNTER = count()


def _make_target_entry(name: str, mesh: object, index: int) -> dict:
    entry = {
        "name": name,
        "target_name": name,
        "mesh": mesh,
        "mesh_file": "mesh.ply",
        "visible": True,
        "show_label": True,
        "node_index": index,
        "color": [0.7, 0.7, 0.7],
    }
    if isinstance(mesh, RenderObjectState):
        mesh.id = make_target_entry_geometry_name(entry, "mesh")
    return entry


def _make_mesh(center: list[float]) -> RenderObjectState:
    center_arr = np.asarray(center, dtype=np.float64)
    offsets = np.asarray(
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    return RenderObjectState(
        id=f"target:mock_{next(_MESH_ID_COUNTER)}::mesh",
        payload=MeshPayload(
            vertices=center_arr + offsets,
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )


def _make_mesh_handle(name: str = "target_1") -> RenderObjectState:
    return RenderObjectState(
        id=f"target:{name}::mesh",
        payload=MeshPayload(
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )


def _write_ascii_ply(path: str) -> np.ndarray:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 3\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "element face 1\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
            "0 0 0\n"
            "1 0 0\n"
            "0 1 0\n"
            "3 0 1 2\n"
        )
    return vertices


def _make_view_model(
    names: list[str],
    positions: list[list[float]],
    *,
    position_valid: list[bool],
    use_ply: list[bool],
) -> SimpleNamespace:
    return SimpleNamespace(
        target_positions=np.asarray(positions, dtype=np.float64),
        target_orientations=np.zeros((len(names), 3), dtype=np.float64),
        target_mesh_files=["mesh.ply"] * len(names),
        target_use_ply_positions=list(use_ply),
        target_metadata=[
            {
                "name": name,
                "mesh_file": "mesh.ply",
                "position_valid": bool(valid),
            }
            for name, valid in zip(names, position_valid)
        ],
    )


def _setup_target_service():
    mock_viz = make_mock_visualizer(tx_count=0, rx_count=0)
    mock_viz.target_scale_overrides = {}
    mock_viz.target_outlines_enabled = False
    mock_viz.outline_color = [0.0, 0.0, 0.0]
    # Visual profile service returns None (no profile match) by default
    mock_viz.visual_profile_service.resolve.return_value = None
    return mock_viz, TargetService(mock_viz)


def _setup_pygfx_target_service():
    mock_viz, target_service = _setup_target_service()
    mock_viz.renderer.capabilities = RendererCapabilities(pbr=True)
    mock_viz.renderer.set_transform = Mock(return_value=True)
    mock_viz.renderer.set_named_material = Mock(return_value=True)
    mock_viz.renderer.has_named_geometry = Mock(return_value=True)
    mock_viz.renderer.set_named_visibility = Mock(return_value=True)
    return mock_viz, target_service


def _seed_target_asset(
    visualizer: object,
    target_name: str,
    mesh_file: str,
    mesh: RenderObjectState,
    *,
    original_vertices: np.ndarray | None = None,
    scaled_vertices: np.ndarray | None = None,
) -> TargetAsset:
    """Insert one complete target asset into the typed owner."""
    cache = getattr(visualizer, "target_asset_cache", None)
    assert isinstance(cache, TargetAssetCache)
    payload = mesh.payload
    assert isinstance(payload, MeshPayload)
    original = np.asarray(
        payload.vertices if original_vertices is None else original_vertices,
        dtype=np.float64,
    )
    scaled = np.asarray(
        payload.vertices if scaled_vertices is None else scaled_vertices,
        dtype=np.float64,
    )
    canonical_path = f"memory://{target_name}/{mesh_file}"
    source = ResolvedTargetAssetSource(
        target_name=target_name,
        mesh_filename=mesh_file,
        canonical_path=canonical_path,
        key=TargetAssetKey(
            canonical_path=canonical_path,
            revision=TargetSourceRevision(
                size_bytes=0,
                modified_ns=0,
                changed_ns=0,
                file_id=0,
            ),
            target_name=target_name,
            mesh_filename=mesh_file,
        ),
    )
    center = (
        (scaled.min(axis=0) + scaled.max(axis=0)) / 2.0
        if scaled.size
        else np.zeros(3, dtype=np.float64)
    )
    asset = TargetAsset(
        source=source,
        mesh=mesh,
        original_vertices=original,
        scaled_vertices=scaled,
        geometry_meta=TargetGeometryMeta(scaled_aabb_center=center),
    )
    cache.put(asset)
    return asset


def test_target_frame_snapshot_keeps_material_hide_and_manual_visibility_latent():
    """Frame synchronization must consume the resolved parent snapshot."""
    mock_viz, target_service = _setup_target_service()
    mesh = _make_mesh_handle("person")
    entry = _make_target_entry("person", mesh, index=0)
    entry.update(entry_type="target", material_id="skin-id", material_type="skin")
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []
    mock_viz.material_pbr_service = None
    mock_viz.material_mode_service = MaterialModeService()
    mock_viz.object_appearance_service = ObjectAppearanceService(mock_viz)

    mock_viz.material_mode_service.set_mode("skin", MaterialDisplayMode.HIDDEN)
    assert target_service.sync_target_entry_snapshot(entry)
    hidden = mock_viz.renderer.ensure_object.call_args.args[0]
    assert hidden.visible is False

    # A later frame-present state cannot resurrect the material-hidden target.
    entry["_frame_visible"] = True
    assert target_service.sync_target_entry_snapshot(entry)
    assert mock_viz.renderer.ensure_object.call_args.args[0].visible is False

    # Normal removes only the overlay; manual visibility remains authoritative.
    entry["visible"] = False
    mock_viz.material_mode_service.set_mode("skin", "normal")
    assert target_service.sync_target_entry_snapshot(entry)
    assert mock_viz.renderer.ensure_object.call_args.args[0].visible is False


def test_material_mutation_is_application_owned_and_renderer_pure():
    mock_viz, target_service = _setup_target_service()
    mesh = _make_mesh_handle("car")
    material = pbr_props_to_kwargs(
        [0.22, 0.22, 0.254],
        {
            "color": [0.22, 0.22, 0.254],
            "roughness": 0.2,
            "metallic": 1.0,
            "reflectance": 0.8,
            "alpha": 1.0,
        },
    )

    target_service._set_handle_material(mesh, material)

    assert mesh.material.base_color == pytest.approx((0.22, 0.22, 0.254, 1.0))
    assert mesh.material.roughness == pytest.approx(0.2)
    assert mesh.material.metallic == pytest.approx(1.0)
    assert mesh.material.reflectance == pytest.approx(0.8)
    mock_viz.renderer.ensure_object.assert_not_called()
    mock_viz.renderer.set_material.assert_not_called()
    mock_viz.renderer.set_named_material.assert_not_called()


def test_vertex_color_source_preserves_authored_colors_and_handle_material():
    _, target_service = _setup_target_service()
    mesh = _make_mesh_handle("car")
    mesh.replace_payload(
        MeshPayload(
            vertices=mesh.payload.vertices,
            triangles=mesh.payload.triangles,
            vertex_colors=np.asarray(
                [[0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8]],
                dtype=np.float64,
            ),
        )
    )
    mesh.material = mesh.material.__class__(
        base_color=(0.22, 0.22, 0.254, 1.0),
        roughness=0.2,
        metallic=1.0,
        reflectance=0.8,
    )

    original_colors = np.asarray(mesh.payload.vertex_colors).copy()
    target_service._set_handle_vertex_color_source(mesh)

    assert mesh.material.base_color == pytest.approx((0.22, 0.22, 0.254, 1.0))
    assert mesh.material.roughness == pytest.approx(0.2)
    assert mesh.material.metallic == pytest.approx(1.0)
    assert mesh.material.reflectance == pytest.approx(0.8)
    np.testing.assert_allclose(mesh.payload.vertex_colors, original_colors)
    assert mesh.payload.color_source is SurfaceColorSource.VERTEX


def test_rich_vertex_color_detection_accepts_render_object_handles():
    _, target_service = _setup_target_service()
    colors = np.asarray([[i / 20.0, 0.1, 0.2] for i in range(20)], dtype=np.float64)
    mesh = RenderObjectState(
        id="target:colored::mesh",
        payload=MeshPayload(
            vertices=np.asarray([[float(i), 0.0, 0.0] for i in range(20)], dtype=np.float64),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
            vertex_colors=colors,
        ),
    )

    assert target_service._has_rich_vertex_colors(mesh)


@pytest.mark.parametrize(
    ("distinct_colors", "expected"),
    ((1, False), (16, False), (17, True)),
)
def test_vertex_color_detection_remains_exact_for_sparse_variation(
    distinct_colors: int,
    expected: bool,
) -> None:
    _, target_service = _setup_target_service()
    colors = np.zeros((1000, 3), dtype=np.float64)
    for index in range(1, distinct_colors):
        # Keep the variation sparse so the deterministic probe alone does not
        # see every authored color; the exact fallback remains authoritative.
        colors[-index] = [float(index), float(index + 1), float(index + 2)]
    mesh = RenderObjectState(
        id="target:sparse_colors::mesh",
        payload=MeshPayload(
            vertices=np.zeros((1000, 3), dtype=np.float64),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
            vertex_colors=colors,
        ),
    )

    assert target_service._has_rich_vertex_colors(mesh) is expected


def test_uniform_vertex_color_detection_skips_unique_sort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, target_service = _setup_target_service()
    colors = np.full((1000, 3), [0.25, 0.5, 0.75], dtype=np.float64)
    mesh = RenderObjectState(
        id="target:uniform_colors::mesh",
        payload=MeshPayload(
            vertices=np.zeros((1000, 3), dtype=np.float64),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
            vertex_colors=colors,
        ),
    )
    monkeypatch.setattr(
        np,
        "unique",
        Mock(side_effect=AssertionError("uniform colors must not be sorted")),
    )

    assert not target_service._has_rich_vertex_colors(mesh)


def test_vertex_replacement_defers_renderer_sync_until_state_is_complete():
    mock_viz, target_service = _setup_target_service()
    mesh = _make_mesh_handle("car")
    replacement = np.asarray(mesh.payload.vertices, dtype=np.float64) + 4.0
    original_payload = mesh.payload

    target_service._set_mesh_vertices(mesh, replacement)

    np.testing.assert_allclose(mesh.payload.vertices, replacement)
    assert mesh.payload is not original_payload
    mock_viz.renderer.ensure_object.assert_not_called()


def test_target_state_mutations_do_not_enter_external_backend_caches():
    mock_viz, target_service = _setup_pygfx_target_service()
    mock_viz.renderer.register_geometry_payload_cache_key = Mock()
    mock_viz.renderer.prime_geometry_buffer_cache = Mock()
    mesh = _make_mesh_handle("car")
    entry = _make_target_entry("car", mesh, index=0)
    entry["has_vertex_texture"] = True
    entry["_mesh_center"] = [0.0, 0.0, 0.0]

    target_service._set_mesh_vertices(mesh, mesh.payload.vertices + 1.0)

    mock_viz.renderer.register_geometry_payload_cache_key.assert_not_called()
    mock_viz.renderer.prime_geometry_buffer_cache.assert_not_called()
    mock_viz.renderer.ensure_object.assert_not_called()


def test_target_lookahead_schedules_only_for_mesh_or_direction_changes() -> None:
    mock_viz, target_service = _setup_target_service()
    cache = mock_viz.target_asset_cache
    cache.prefetch_after = Mock(return_value=2)
    entry = {
        "target_name": "person",
        "mesh_file": "frame_a.ply",
        "_target_asset_build_spec": target_service._target_asset_build_spec(
            scale=1.0,
            orientation=[0.0, 0.0, 0.0],
            use_ply_position=False,
            pbr_props={},
        ),
    }
    mock_viz.play_direction = 1

    target_service._schedule_target_lookahead(entry)
    target_service._schedule_target_lookahead(entry)
    mock_viz.play_direction = -1
    target_service._schedule_target_lookahead(entry)
    target_service._schedule_target_lookahead(entry)
    entry["mesh_file"] = "frame_b.ply"
    target_service._schedule_target_lookahead(entry)

    assert cache.prefetch_after.call_count == 3
    assert [call.kwargs["direction"] for call in cache.prefetch_after.call_args_list] == [
        1,
        -1,
        -1,
    ]


def test_target_lookahead_waves_prioritize_every_targets_next_frame() -> None:
    mock_viz, target_service = _setup_target_service()
    cache = mock_viz.target_asset_cache
    cache.prefetch_after = Mock(return_value=1)
    entries = [
        {
            "target_name": name,
            "mesh_file": "frame_a.ply",
            "_target_asset_build_spec": target_service._target_asset_build_spec(
                scale=1.0,
                orientation=[0.0, 0.0, 0.0],
                use_ply_position=False,
                pbr_props={},
            ),
        }
        for name in ("first", "second", "third")
    ]
    mock_viz.play_direction = 1

    target_service._schedule_target_lookahead_waves(entries)

    assert [call.args[0] for call in cache.prefetch_after.call_args_list] == [
        "first",
        "second",
        "third",
        "first",
        "second",
        "third",
    ]
    assert [call.kwargs["count"] for call in cache.prefetch_after.call_args_list] == [
        1,
        1,
        1,
        2,
        2,
        2,
    ]


def test_process_targets_hides_target_when_position_invalid():
    """Targets with invalid missing positions should be hidden for that frame."""
    mock_viz, target_service = _setup_target_service()
    mesh = _make_mesh_handle("target_1")

    entry = _make_target_entry("target_1", mesh, index=0)
    mock_viz.target_entries = [entry]
    _seed_target_asset(mock_viz, "target_1", "mesh.ply", mesh)
    mock_viz.target_labels = []

    view_model = _make_view_model(
        ["target_1"],
        [[0.0, 0.0, 0.0]],
        position_valid=[False],
        use_ply=[False],
    )

    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    assert entry["_frame_visible"] is False
    assert entry["visible"] is True
    assert mesh.visible is True
    assert mock_viz._named_visibility[mesh.id] is False
    mock_viz.renderer.set_visible.assert_not_called()
    mock_viz.renderer.set_named_visibility.assert_not_called()


def test_target_runtime_visibility_uses_one_complete_snapshot():
    """Per-frame visibility must not rewrite intent or use imperative setters."""
    mock_viz, target_service = _setup_target_service()
    mesh = _make_mesh_handle("target_1")
    entry = _make_target_entry("target_1", mesh, index=0)
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []
    target_service.sync_target_entry_snapshot(entry)

    assert mesh.visible is True
    assert mock_viz._named_visibility[mesh.id] is True
    assert mock_viz.renderer.ensure_object.call_count == 1
    mock_viz.renderer.set_visible.assert_not_called()
    mock_viz.renderer.set_named_visibility.assert_not_called()

    entry["_frame_visible"] = False
    target_service.sync_target_entry_snapshot(entry)

    assert mesh.visible is True
    assert mock_viz._named_visibility[mesh.id] is False
    assert mock_viz.renderer.ensure_object.call_count == 2
    mock_viz.renderer.set_visible.assert_not_called()
    mock_viz.renderer.set_named_visibility.assert_not_called()


def test_target_snapshot_resolves_sparse_label_list_by_stable_id():
    """A meshless earlier target must not steal a later target's label slot."""
    mock_viz, target_service = _setup_pygfx_target_service()
    missing_entry = {
        "name": "missing",
        "target_name": "missing",
        "mesh": None,
        "visible": True,
        "show_label": True,
        "node_index": 0,
    }
    mesh = _make_mesh_handle("valid")
    entry = _make_target_entry("valid", mesh, index=1)
    label_name = make_target_entry_geometry_name(entry, "label")
    label = make_text_label_state(label_name, "Valid", [0.8, 0.8, 0.8])
    mock_viz.target_entries = [missing_entry, entry]
    mock_viz.target_labels = [label]

    assert target_service.sync_target_entry_snapshot(entry)

    ensured_ids = [call.args[0].id for call in mock_viz.renderer.ensure_object.call_args_list]
    assert ensured_ids == [mesh.id, label_name]


def test_target_snapshot_batches_mesh_label_and_outline_without_noop_present():
    mock_viz, target_service = _setup_target_service()
    renderer = RecordingBatchRenderer()
    mock_viz.renderer = renderer
    mock_viz.target_outlines_enabled = True
    mesh = _make_mesh_handle("batched")
    entry = _make_target_entry("batched", mesh, index=0)
    label_name = make_target_entry_geometry_name(entry, "label")
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = [make_text_label_state(label_name, "Batched", [0.8, 0.8, 0.8])]

    assert target_service.sync_target_entry_snapshot(entry)

    expected_ids = {
        mesh.id,
        label_name,
        make_target_entry_geometry_name(entry, "outline"),
    }
    assert {object_id for object_id, _depth in renderer.operation_depths} == expected_ids
    assert all(depth >= 1 for _object_id, depth in renderer.operation_depths)
    assert renderer.outer_batch_count == 1
    assert renderer.update_renderer_calls == 1

    assert target_service.sync_target_entry_snapshot(entry)

    assert renderer.outer_batch_count == 2
    assert renderer.update_renderer_calls == 1


def test_process_targets_hides_stale_targets_not_in_frame():
    """Targets absent from current frame metadata should be hidden transiently."""
    mock_viz, target_service = _setup_target_service()

    mesh_1 = _make_mesh_handle("target_1")
    mesh_2 = _make_mesh_handle("target_2")

    entry_1 = _make_target_entry("target_1", mesh_1, index=0)
    entry_2 = _make_target_entry("target_2", mesh_2, index=1)
    mock_viz.target_entries = [entry_1, entry_2]
    _seed_target_asset(mock_viz, "target_1", "mesh.ply", mesh_1)
    _seed_target_asset(mock_viz, "target_2", "mesh.ply", mesh_2)
    mock_viz.target_labels = []

    view_model = _make_view_model(
        ["target_1"],
        [[1.0, 2.0, 3.0]],
        position_valid=[True],
        use_ply=[False],
    )

    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    assert entry_1["_frame_visible"] is True
    assert entry_2["_frame_visible"] is False
    assert entry_2["visible"] is True
    assert mock_viz._named_visibility[mesh_2.id] is False
    mock_viz.renderer.set_visible.assert_not_called()


def test_target_returning_with_same_pose_is_reshown_after_absent_frame():
    """An absent frame must invalidate the prior visible-state cache entry."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mesh = _make_mesh_handle("target_1")
    entry = _make_target_entry("target_1", mesh, index=0)
    mock_viz.target_entries = [entry]
    _seed_target_asset(mock_viz, "target_1", "mesh.ply", mesh)
    mock_viz.target_labels = []
    present = _make_view_model(
        ["target_1"],
        [[1.0, 2.0, 3.0]],
        position_valid=[True],
        use_ply=[False],
    )
    absent = _make_view_model([], [], position_valid=[], use_ply=[])

    target_service.process_targets_from_view_model(step=0, view_model=present)
    assert mock_viz._named_visibility[mesh.id] is True

    target_service.process_targets_from_view_model(step=1, view_model=absent)
    assert entry["_frame_visible"] is False
    assert mock_viz._named_visibility[mesh.id] is False
    assert "target_1" not in mock_viz.target_asset_cache.runtime_states

    mock_viz.renderer.ensure_object.reset_mock()
    target_service.process_targets_from_view_model(step=2, view_model=present)

    snapshots = [
        call.args[0]
        for call in mock_viz.renderer.ensure_object.call_args_list
        if call.args[0].id == mesh.id
    ]
    assert len(snapshots) == 1
    assert snapshots[0].visible is True
    assert mock_viz._named_visibility[mesh.id] is True


def test_process_targets_hides_invalid_target_even_without_cache_entries():
    """Frame visibility updates must run even when target cache is empty."""
    mock_viz, target_service = _setup_target_service()
    mesh = _make_mesh_handle("target_1")

    entry = _make_target_entry("target_1", mesh, index=0)
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []

    view_model = _make_view_model(
        ["target_1"],
        [[0.0, 0.0, 0.0]],
        position_valid=[False],
        use_ply=[False],
    )

    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    assert entry["_frame_visible"] is False
    assert mock_viz._named_visibility[mesh.id] is False
    mock_viz.renderer.set_visible.assert_not_called()


def test_process_targets_records_benchmark_handoff_metrics():
    """Benchmark mode should expose target handoff counters from the target loop."""
    mock_viz, target_service = _setup_target_service()
    mock_viz.pipeline = SimpleNamespace(benchmark_recorder=object())
    mesh = _make_mesh_handle("target_1")

    entry = _make_target_entry("target_1", mesh, index=0)
    mock_viz.target_entries = [entry]
    _seed_target_asset(mock_viz, "target_1", "mesh.ply", mesh)
    mock_viz.target_labels = []

    view_model = _make_view_model(
        ["target_1"],
        [[1.0, 2.0, 3.0]],
        position_valid=[True],
        use_ply=[False],
    )

    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    breakdown = target_service.get_last_runtime_breakdown()
    assert breakdown["target_state_changed_count"] == 1.0
    assert breakdown["target_batch_visible_update_count"] == 1.0
    assert breakdown["target_transform_snapshot_count"] == 1.0
    assert breakdown["target_runtime_visible_count"] == 1.0
    assert breakdown["target_handoff_geometry_sync_count"] >= 1.0
    assert breakdown["target_handoff_sync_entity_count"] >= 1.0
    assert breakdown["target_renderer_update_count"] == 1.0
    assert breakdown["process_targets_total_ms"] >= 0.0


def test_process_targets_cache_replays_visibility_after_pov_return_to_overview():
    """Cached target frames must still resync visibility when POV mode changes."""
    mock_viz, target_service = _setup_target_service()
    mesh = _make_mesh_handle("target_1")

    entry = _make_target_entry("target_1", mesh, index=0)
    mock_viz.target_entries = [entry]
    _seed_target_asset(mock_viz, "target_1", "mesh.ply", mesh)
    label_name = make_target_entry_geometry_name(entry, "label")
    mock_viz.target_labels = [make_text_label_state(label_name, "Target 1", [0.8, 0.8, 0.8])]
    view_model = _make_view_model(
        ["target_1"],
        [[1.0, 2.0, 3.0]],
        position_valid=[True],
        use_ply=[False],
    )

    mock_viz.app_state.camera_mode = "overview"
    mock_viz.app_state.pov_hidden_node = None
    target_service.process_targets_from_view_model(step=0, view_model=view_model)
    assert mock_viz.renderer.has_named_geometry(label_name)

    mock_viz.renderer.ensure_object.reset_mock()
    mock_viz.app_state.camera_mode = "pov"
    mock_viz.app_state.pov_hidden_node = ("target", 0)
    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    hidden_calls = [
        call.args[0]
        for call in mock_viz.renderer.ensure_object.call_args_list
        if call.args[0].id == mesh.id
    ]
    assert len(hidden_calls) == 1
    assert hidden_calls[0].visible is False
    assert mock_viz._named_visibility[label_name] is False

    mock_viz.renderer.ensure_object.reset_mock()
    mock_viz.app_state.camera_mode = "overview"
    mock_viz.app_state.pov_hidden_node = None
    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    assert mock_viz._named_visibility["target:target_1::mesh"] is True
    assert mock_viz._named_visibility[label_name] is True
    visible_calls = [
        call.args[0]
        for call in mock_viz.renderer.ensure_object.call_args_list
        if call.args[0].id == mesh.id
    ]
    assert len(visible_calls) == 1
    assert visible_calls[0].visible is True
    mock_viz.renderer.set_visible.assert_not_called()
    mock_viz.renderer.set_named_visibility.assert_not_called()
    assert mock_viz.target_asset_cache.runtime_states["target_1"].runtime_visible is True


def test_reversed_target_metadata_preserves_canonical_identity_and_pov_index():
    """Metadata order selects frame arrays, never target identity or POV ownership."""
    mock_viz, target_service = _setup_pygfx_target_service()
    alpha_mesh = _make_mesh_handle("alpha")
    beta_mesh = _make_mesh_handle("beta")
    alpha_entry = _make_target_entry("alpha", alpha_mesh, index=0)
    beta_entry = _make_target_entry("beta", beta_mesh, index=1)
    mock_viz.target_entries = [alpha_entry, beta_entry]
    _seed_target_asset(mock_viz, "alpha", "mesh.ply", alpha_mesh)
    _seed_target_asset(mock_viz, "beta", "mesh.ply", beta_mesh)
    mock_viz.target_labels = []
    mock_viz.app_state.camera_mode = "pov"
    mock_viz.app_state.pov_hidden_node = ("target", 0)
    view_model = _make_view_model(
        ["beta", "alpha"],
        [[20.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        position_valid=[True, True],
        use_ply=[False, False],
    )

    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    assert alpha_entry["node_index"] == 0
    assert beta_entry["node_index"] == 1
    assert alpha_entry["object_id"] == "target:alpha"
    assert beta_entry["object_id"] == "target:beta"
    np.testing.assert_allclose(alpha_entry["position"], [10.0, 0.0, 0.0])
    np.testing.assert_allclose(beta_entry["position"], [20.0, 0.0, 0.0])

    snapshots = {
        call.args[0].id: call.args[0]
        for call in mock_viz.renderer.ensure_object.call_args_list
        if call.args[0].id in {alpha_mesh.id, beta_mesh.id}
    }
    assert snapshots[alpha_mesh.id].visible is False
    assert snapshots[beta_mesh.id].visible is True


def test_sparse_first_target_uses_canonical_index_with_metadata_position_zero():
    """A missing first mesh slot must not make metadata index zero become identity zero."""
    mock_viz, target_service = _setup_pygfx_target_service()
    missing_entry = {
        "name": "missing",
        "target_name": "missing",
        "mesh": None,
        "visible": True,
        "show_label": True,
        "node_index": 0,
    }
    present_mesh = _make_mesh_handle("present")
    present_entry = _make_target_entry("present", present_mesh, index=1)
    mock_viz.target_entries = [missing_entry, present_entry]
    _seed_target_asset(mock_viz, "present", "mesh.ply", present_mesh)
    mock_viz.target_labels = []
    mock_viz.app_state.camera_mode = "pov"
    mock_viz.app_state.pov_hidden_node = ("target", 1)
    view_model = _make_view_model(
        ["present"],
        [[7.0, 8.0, 9.0]],
        position_valid=[True],
        use_ply=[False],
    )

    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    assert present_entry["node_index"] == 1
    assert present_entry["object_id"] == "target:present"
    np.testing.assert_allclose(present_entry["position"], [7.0, 8.0, 9.0])
    snapshots = [
        call.args[0]
        for call in mock_viz.renderer.ensure_object.call_args_list
        if call.args[0].id == present_mesh.id
    ]
    assert len(snapshots) == 1
    assert snapshots[0].visible is False


def test_mesh_switch_defers_render_until_unified_target_update_pass():
    """Mesh swaps should avoid intermediate renderer updates before transform pass."""
    mock_viz, target_service = _setup_target_service()
    old_mesh = _make_mesh([0.0, 0.0, 0.0])
    new_mesh = _make_mesh([0.0, 0.0, 0.0])

    entry = _make_target_entry("target_1", old_mesh, index=0)
    mesh_name = make_target_entry_geometry_name(entry, "mesh")
    old_mesh.id = mesh_name
    new_mesh.id = mesh_name
    entry["mesh_file"] = "mesh_old.ply"
    mock_viz.target_entries = [entry]
    _seed_target_asset(mock_viz, "target_1", "mesh_old.ply", old_mesh)
    _seed_target_asset(mock_viz, "target_1", "mesh_new.ply", new_mesh)
    mock_viz.target_labels = []

    view_model = SimpleNamespace(
        target_positions=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        target_orientations=np.zeros((1, 3), dtype=np.float64),
        target_mesh_files=["mesh_new.ply"],
        target_use_ply_positions=[False],
        target_metadata=[{"name": "target_1", "mesh_file": "mesh_new.ply", "position_valid": True}],
    )

    target_service.process_targets_from_view_model(step=32, view_model=view_model)

    assert entry["mesh"] is new_mesh
    assert mock_viz.renderer.update_renderer.call_count == 1


@pytest.mark.parametrize(
    ("old_source", "new_source", "resolved_color"),
    [
        (SurfaceColorSource.MATERIAL, SurfaceColorSource.VERTEX, [1.0, 1.0, 1.0]),
        (SurfaceColorSource.VERTEX, SurfaceColorSource.MATERIAL, [0.2, 0.7, 0.3]),
    ],
)
def test_mesh_switch_reresolves_material_when_color_ownership_changes(
    old_source: SurfaceColorSource,
    new_source: SurfaceColorSource,
    resolved_color: list[float],
) -> None:
    """A mixed sequence must not carry uniform or vertex color semantics across frames."""
    mock_viz, target_service = _setup_target_service()
    old_mesh = _make_mesh([0.0, 0.0, 0.0])
    new_mesh = _make_mesh([0.0, 0.0, 0.0])
    old_mesh.replace_payload(replace(old_mesh.payload, color_source=old_source))
    new_mesh.replace_payload(replace(new_mesh.payload, color_source=new_source))

    entry = _make_target_entry("target_1", old_mesh, index=0)
    mesh_name = make_target_entry_geometry_name(entry, "mesh")
    old_mesh.id = mesh_name
    new_mesh.id = mesh_name
    entry.update(
        mesh_file="mesh_old.ply",
        entry_type="target",
        material_type="custom",
        has_vertex_texture=old_source is SurfaceColorSource.VERTEX,
    )
    old_asset = _seed_target_asset(mock_viz, "target_1", "mesh_old.ply", old_mesh)
    new_asset = _seed_target_asset(mock_viz, "target_1", "mesh_new.ply", new_mesh)
    old_asset.has_vertex_texture = old_source is SurfaceColorSource.VERTEX
    new_asset.has_vertex_texture = new_source is SurfaceColorSource.VERTEX
    entry["_target_asset"] = old_asset
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []
    mock_viz.target_asset_cache.pin(old_asset)

    # Give the old frame a deliberately different material so a blind reuse
    # would make the ownership transition visible in this assertion.
    target_service._set_handle_material(
        old_mesh,
        pbr_props_to_kwargs(
            [0.8, 0.1, 0.2],
            {"color": [0.8, 0.1, 0.2], "roughness": 0.9},
        ),
    )
    resolved = resolve_pbr_material(
        resolved_color,
        {"color": resolved_color, "roughness": 0.35},
        context="mixed target test",
    )
    mock_viz.material_pbr_service.resolve_entry_material = Mock(return_value=resolved)
    view_model = SimpleNamespace(
        target_positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        target_orientations=np.zeros((1, 3), dtype=np.float64),
        target_mesh_files=["mesh_new.ply"],
        target_use_ply_positions=[False],
        target_metadata=[{"name": "target_1", "mesh_file": "mesh_new.ply", "position_valid": True}],
    )

    target_service.process_targets_from_view_model(step=1, view_model=view_model)

    assert entry["mesh"] is new_mesh
    assert new_mesh.material == resolved.payload
    mock_viz.material_pbr_service.resolve_entry_material.assert_called_once_with(entry)


def test_mesh_switch_hands_off_pin_when_only_one_asset_fits() -> None:
    mock_viz, target_service = _setup_target_service()
    mock_viz.current_scenario_policy = None
    cache = mock_viz.target_asset_cache

    with tempfile.TemporaryDirectory() as tmpdir:
        old_path = os.path.join(tmpdir, "mesh_old.ply")
        new_path = os.path.join(tmpdir, "mesh_new.ply")
        _write_ascii_ply(old_path)
        _write_ascii_ply(new_path)
        metadata = {
            "name": "target_1",
            "mesh_directory": tmpdir,
            "scale": 1.0,
            "orientation": [0.0, 0.0, 0.0],
            "use_ply_position": False,
            "material_type": "custom",
            "position_valid": True,
        }
        current = target_service._load_mesh_on_demand(
            "target_1",
            "mesh_old.ply",
            metadata,
            mock_viz,
        )
        assert isinstance(current, TargetAsset)
        entry = _make_target_entry("target_1", current.mesh, index=0)
        entry["mesh_file"] = "mesh_old.ply"
        entry["_target_asset"] = current
        # Avoid unrelated background lookahead in this byte-handoff test.
        entry["_target_lookahead_schedule"] = ("mesh_new.ply", 1)
        mock_viz.target_entries = [entry]
        mock_viz.target_labels = []
        cache.pin(current)
        cache.configure(max_bytes=current.estimated_bytes + 1)
        view_model = SimpleNamespace(
            target_positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            target_orientations=np.zeros((1, 3), dtype=np.float64),
            target_mesh_files=["mesh_new.ply"],
            target_use_ply_positions=[False],
            target_metadata=[{**metadata, "mesh_file": "mesh_new.ply"}],
        )

        target_service.process_targets_from_view_model(step=1, view_model=view_model)

        replacement = entry.get("_target_asset")
        assert isinstance(replacement, TargetAsset)
        assert replacement is not current
        assert entry["mesh_file"] == "mesh_new.ply"
        assert cache.asset_for_logical_key(current.logical_key) is None
        assert cache.asset_for_logical_key(replacement.logical_key) is replacement
        assert cache.telemetry()["pinned"] == 1
        assert cache.telemetry()["bytes"] <= cache.telemetry()["max_bytes"]


def test_mesh_switch_resyncs_common_object_with_pbr_material():
    """Mesh swaps carry material intent through backend-owned object sync."""
    mock_viz, target_service = _setup_target_service()
    old_mesh = _make_mesh([0.0, 0.0, 0.0])
    new_mesh = _make_mesh([0.0, 0.0, 0.0])

    entry = _make_target_entry("target_1", old_mesh, index=0)
    mesh_name = make_target_entry_geometry_name(entry, "mesh")
    old_mesh.id = mesh_name
    new_mesh.id = mesh_name
    entry["mesh_file"] = "mesh_old.ply"
    entry["material_type"] = "concrete"
    entry["pbr_roughness"] = 0.42
    entry["pbr_metallic"] = 0.13
    entry["pbr_reflectance"] = 0.51
    entry["pbr_alpha"] = 0.73
    mock_viz.target_entries = [entry]
    _seed_target_asset(mock_viz, "target_1", "mesh_old.ply", old_mesh)
    _seed_target_asset(mock_viz, "target_1", "mesh_new.ply", new_mesh)
    mock_viz.target_labels = []

    mock_viz.renderer.capabilities = RendererCapabilities(pbr=True)
    mock_viz.renderer.has_named_geometry = Mock(return_value=True)
    mock_viz.renderer.add_or_update_named_geometry = Mock()
    mock_viz.renderer.remove_named_geometry = Mock(return_value=True)
    mock_viz.renderer.set_material = Mock(return_value=False)
    mock_viz.renderer.set_named_material = Mock(return_value=False)
    mock_viz.renderer.ensure_named_geometry = Mock(return_value=True)
    mock_viz.renderer.ensure_object.reset_mock()
    target_service._set_handle_material(
        old_mesh,
        pbr_props_to_kwargs(
            [0.2, 0.4, 0.6],
            {
                "color": [0.2, 0.4, 0.6],
                "roughness": 0.42,
                "metallic": 0.13,
                "reflectance": 0.51,
                "alpha": 0.73,
            },
        ),
    )

    view_model = SimpleNamespace(
        target_positions=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        target_orientations=np.zeros((1, 3), dtype=np.float64),
        target_mesh_files=["mesh_new.ply"],
        target_use_ply_positions=[False],
        target_metadata=[{"name": "target_1", "mesh_file": "mesh_new.ply", "position_valid": True}],
    )

    target_service.process_targets_from_view_model(step=32, view_model=view_model)

    assert entry["mesh"] is new_mesh
    assert "_force_resync_mesh" not in entry
    mock_viz.renderer.remove_named_geometry.assert_not_called()
    object_calls = [
        call.args[0]
        for call in mock_viz.renderer.ensure_object.call_args_list
        if call.args[0].id == mesh_name
    ]
    assert len(object_calls) == 1
    render_object = object_calls[0]
    assert render_object.payload is new_mesh.payload
    material = render_object.material_payload
    assert material is not None
    assert material.base_color == pytest.approx((0.2, 0.4, 0.6, 0.73))
    assert material.roughness == pytest.approx(0.42)
    assert material.metallic == pytest.approx(0.13)
    assert material.reflectance == pytest.approx(0.51)
    mock_viz.renderer.ensure_named_geometry.assert_not_called()


def test_failed_mesh_switch_preserves_native_object_and_retries_identical_frame():
    """Failed replacement stays pending and retries without service-side removal."""
    mock_viz, target_service = _setup_pygfx_target_service()
    old_mesh = _make_mesh([0.0, 0.0, 0.0])
    new_mesh = _make_mesh([5.0, 0.0, 0.0])
    entry = _make_target_entry("target_1", old_mesh, index=0)
    mesh_name = make_target_entry_geometry_name(entry, "mesh")
    new_mesh.id = mesh_name
    entry["mesh_file"] = "mesh_old.ply"
    mock_viz.target_entries = [entry]
    _seed_target_asset(mock_viz, "target_1", "mesh_old.ply", old_mesh)
    _seed_target_asset(mock_viz, "target_1", "mesh_new.ply", new_mesh)
    mock_viz.target_labels = []

    old_native_snapshot = old_mesh.to_render_object()
    mock_viz._named_objects[mesh_name] = old_native_snapshot
    mock_viz._named_visibility[mesh_name] = True
    outcomes = iter([False, True])

    def ensure_object(snapshot):
        succeeded = next(outcomes)
        if succeeded:
            mock_viz._named_objects[snapshot.id] = snapshot
            mock_viz._named_visibility[snapshot.id] = bool(snapshot.visible)
        return succeeded

    mock_viz.renderer.ensure_object = Mock(side_effect=ensure_object)
    view_model = SimpleNamespace(
        target_positions=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        target_orientations=np.zeros((1, 3), dtype=np.float64),
        target_mesh_files=["mesh_new.ply"],
        target_use_ply_positions=[False],
        target_metadata=[{"name": "target_1", "mesh_file": "mesh_new.ply", "position_valid": True}],
    )

    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    assert entry["mesh"] is new_mesh
    assert entry["_renderer_sync_pending"] is True
    assert mock_viz._named_objects[mesh_name] is old_native_snapshot
    mock_viz.renderer.remove_object.assert_not_called()
    mock_viz.renderer.remove_named_geometry.assert_not_called()
    mock_viz.renderer.ensure_named_geometry.assert_not_called()

    target_service.process_targets_from_view_model(step=1, view_model=view_model)

    assert mock_viz.renderer.ensure_object.call_count == 2
    assert entry["_renderer_sync_pending"] is False
    applied = mock_viz._named_objects[mesh_name]
    assert applied.payload is new_mesh.payload
    mock_viz.renderer.remove_object.assert_not_called()
    mock_viz.renderer.remove_named_geometry.assert_not_called()


def test_pygfx_mesh_switch_resyncs_stable_object_without_external_remap():
    """A payload swap stays on the common object contract and uploads once."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mock_viz.pipeline = SimpleNamespace(benchmark_recorder=object())
    old_mesh = _make_mesh([0.0, 0.0, 0.0])
    new_mesh = _make_mesh([5.0, 0.0, 0.0])

    entry = _make_target_entry("target_1", old_mesh, index=0)
    mesh_name = make_target_entry_geometry_name(entry, "mesh")
    old_mesh.id = mesh_name
    new_mesh.id = mesh_name
    entry["mesh_file"] = "mesh_old.ply"
    entry["pbr_properties"] = {
        "color": [0.7, 0.7, 0.7],
        "roughness": 0.5,
        "metallic": 0.0,
        "reflectance": 0.5,
    }
    mock_viz.target_entries = [entry]
    _seed_target_asset(mock_viz, "target_1", "mesh_old.ply", old_mesh)
    _seed_target_asset(mock_viz, "target_1", "mesh_new.ply", new_mesh)
    mock_viz.target_labels = []
    mock_viz.renderer.set_named_material = Mock(return_value=True)
    mock_viz.renderer.ensure_named_geometry = Mock(return_value=True)
    mock_viz.renderer.remap_external_geometry_name = Mock(return_value=True)
    mock_viz.renderer.ensure_object.reset_mock()

    view_model = SimpleNamespace(
        target_positions=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        target_orientations=np.zeros((1, 3), dtype=np.float64),
        target_mesh_files=["mesh_new.ply"],
        target_use_ply_positions=[False],
        target_metadata=[{"name": "target_1", "mesh_file": "mesh_new.ply", "position_valid": True}],
    )

    target_service.process_targets_from_view_model(step=1, view_model=view_model)

    assert entry["mesh"] is new_mesh
    object_calls = [
        call.args[0]
        for call in mock_viz.renderer.ensure_object.call_args_list
        if call.args[0].id == mesh_name
    ]
    assert len(object_calls) == 1
    assert object_calls[0].payload is new_mesh.payload
    mock_viz.renderer.ensure_named_geometry.assert_not_called()
    mock_viz.renderer.remap_external_geometry_name.assert_not_called()
    breakdown = target_service.get_last_runtime_breakdown()
    assert breakdown["target_handoff_sync_entity_count"] == 1.0
    assert "target_handoff_ensure_named_geometry_count" not in breakdown


def test_process_targets_reprocesses_when_use_ply_position_changes():
    """Changing use_ply_position must bypass the target state skip gate."""
    mock_viz, target_service = _setup_target_service()
    mesh = _make_mesh([0.0, 0.0, 0.0])

    entry = _make_target_entry("target_1", mesh, index=0)
    entry["_frame_visible"] = False
    mock_viz.target_entries = [entry]
    _seed_target_asset(mock_viz, "target_1", "mesh.ply", mesh)
    mock_viz.target_labels = []
    mock_viz.target_asset_cache.runtime_states["target_1"] = TargetRuntimeState(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        mesh_filename="mesh.ply",
        position_valid=False,
        use_ply_position=False,
        runtime_visible=True,
    )

    view_model = _make_view_model(
        ["target_1"],
        [[0.0, 0.0, 0.0]],
        position_valid=[False],
        use_ply=[True],
    )

    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    assert entry["_frame_visible"] is True
    assert mock_viz.target_asset_cache.runtime_states["target_1"].use_ply_position is True


def test_process_targets_applies_scale_only_frame_changes() -> None:
    mock_viz, target_service = _setup_target_service()
    mesh = _make_mesh([0.0, 0.0, 0.0])
    entry = _make_target_entry("target_1", mesh, index=0)
    asset = _seed_target_asset(mock_viz, "target_1", "mesh.ply", mesh)
    entry["_target_asset"] = asset
    entry["scale"] = 1.0
    entry["_last_applied_scale"] = 1.0
    entry["_last_applied_scale_mesh_file"] = "mesh.ply"
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []
    cache = mock_viz.target_asset_cache
    cache.runtime_states["target_1"] = TargetRuntimeState(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        mesh_filename="mesh.ply",
        position_valid=True,
        use_ply_position=False,
        runtime_visible=True,
        scale=1.0,
    )
    original_vertices = np.asarray(asset.original_vertices).copy()
    center = (original_vertices.min(axis=0) + original_vertices.max(axis=0)) * 0.5
    view_model = SimpleNamespace(
        target_positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        target_orientations=np.zeros((1, 3), dtype=np.float64),
        target_mesh_files=["mesh.ply"],
        target_use_ply_positions=[False],
        target_metadata=[
            {
                "name": "target_1",
                "mesh_file": "mesh.ply",
                "position_valid": True,
                "scale": "2.0",
            }
        ],
    )

    target_service.process_targets_from_view_model(step=1, view_model=view_model)

    np.testing.assert_allclose(
        render_state_points(mesh),
        (original_vertices - center) * 2.0 + center,
    )
    assert entry["scale"] == pytest.approx(2.0)
    assert cache.runtime_states["target_1"].scale == pytest.approx(2.0)


def test_failed_runtime_scale_remains_pending_for_identical_frame_retry() -> None:
    """Desired scale metadata must not be cached as applied after a failed mutation."""
    mock_viz, target_service = _setup_target_service()
    mesh = _make_mesh([0.0, 0.0, 0.0])
    entry = _make_target_entry("target_1", mesh, index=0)
    entry["scale"] = 1.0
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []
    target_service.apply_target_scale_from_metadata = Mock(return_value=False)
    view_model = SimpleNamespace(
        target_positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        target_orientations=np.zeros((1, 3), dtype=np.float64),
        target_mesh_files=["mesh.ply"],
        target_use_ply_positions=[False],
        target_metadata=[
            {
                "name": "target_1",
                "mesh_file": "mesh.ply",
                "position_valid": True,
                "scale": 2.0,
            }
        ],
    )

    target_service.process_targets_from_view_model(step=1, view_model=view_model)
    target_service.process_targets_from_view_model(step=2, view_model=view_model)

    assert target_service.apply_target_scale_from_metadata.call_count == 2
    assert mock_viz.target_asset_cache.runtime_states["target_1"].scale == pytest.approx(1.0)


def test_ply_baked_orientation_resets_when_rotation_returns_to_zero():
    """Zero orientation must restore the scaled PLY baseline, not keep the last rotation."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mesh = _make_mesh([0.0, 0.0, 0.0])
    entry = _make_target_entry("target_1", mesh, index=0)
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []
    baseline = render_state_points(mesh).copy()
    _seed_target_asset(
        mock_viz,
        "target_1",
        "mesh.ply",
        mesh,
        original_vertices=baseline,
        scaled_vertices=baseline,
    )

    def _view_model(orientation):
        return SimpleNamespace(
            target_positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            target_orientations=np.asarray([orientation], dtype=np.float64),
            target_mesh_files=["mesh.ply"],
            target_use_ply_positions=[True],
            target_metadata=[
                {"name": "target_1", "mesh_file": "mesh.ply", "position_valid": False}
            ],
        )

    target_service.process_targets_from_view_model(0, _view_model([1.0, 0.0, 0.0]))
    assert not np.allclose(render_state_points(mesh), baseline)

    target_service.process_targets_from_view_model(1, _view_model([0.0, 0.0, 0.0]))

    np.testing.assert_allclose(render_state_points(mesh), baseline)
    assert "_rotation_matrix" not in entry
    np.testing.assert_allclose(mesh.world_transform.matrix, np.eye(4))


def test_switching_from_ply_to_transform_restores_local_payload_and_zero_rotation():
    """Transform mode must not apply a matrix on top of stale PLY-baked vertices."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mesh = _make_mesh([0.0, 0.0, 0.0])
    entry = _make_target_entry("target_1", mesh, index=0)
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []
    baseline = render_state_points(mesh).copy()
    _seed_target_asset(
        mock_viz,
        "target_1",
        "mesh.ply",
        mesh,
        original_vertices=baseline,
        scaled_vertices=baseline,
    )

    def _view_model(orientation, *, use_ply):
        return SimpleNamespace(
            target_positions=np.asarray([[4.0, 5.0, 6.0]], dtype=np.float64),
            target_orientations=np.asarray([orientation], dtype=np.float64),
            target_mesh_files=["mesh.ply"],
            target_use_ply_positions=[use_ply],
            target_metadata=[{"name": "target_1", "mesh_file": "mesh.ply", "position_valid": True}],
        )

    target_service.process_targets_from_view_model(
        0,
        _view_model([1.0, 0.0, 0.0], use_ply=True),
    )
    assert not np.allclose(render_state_points(mesh), baseline)

    target_service.process_targets_from_view_model(
        1,
        _view_model([1.0, 0.0, 0.0], use_ply=False),
    )
    np.testing.assert_allclose(render_state_points(mesh), baseline)
    assert "_rotation_matrix" in entry

    target_service.process_targets_from_view_model(
        2,
        _view_model([0.0, 0.0, 0.0], use_ply=False),
    )
    assert "_rotation_matrix" not in entry
    np.testing.assert_allclose(mesh.world_transform.matrix[:3, :3], np.eye(3))


def test_open3d_transform_playback_reuses_local_payload_across_positions():
    """Transform-managed Open3D playback must not rebuild target payloads."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mock_viz.renderer.renderer_type = "open3d"
    mesh = _make_mesh([0.0, 0.0, 0.0])
    original_points = render_state_points(mesh).copy()
    original_payload = mesh.payload

    entry = _make_target_entry("target_1", mesh, index=0)
    entry["pbr_properties"] = {
        "color": [0.7, 0.7, 0.7],
        "roughness": 0.5,
        "metallic": 0.0,
        "reflectance": 0.5,
    }
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []
    asset = _seed_target_asset(
        mock_viz,
        "target_1",
        "mesh.ply",
        mesh,
        scaled_vertices=np.asarray(
            [[-1.0, -0.5, 0.0], [1.0, -0.5, 0.0], [0.0, 0.5, 0.0]],
            dtype=np.float64,
        ),
    )
    entry["_target_asset"] = asset

    view_model = _make_view_model(
        ["target_1"],
        [[1.0, 2.0, 3.0]],
        position_valid=[True],
        use_ply=[False],
    )

    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    moved_view_model = _make_view_model(
        ["target_1"],
        [[4.0, 5.0, 6.0]],
        position_valid=[True],
        use_ply=[False],
    )
    mock_viz.renderer.ensure_object.reset_mock()
    target_service.process_targets_from_view_model(step=1, view_model=moved_view_model)

    assert mock_viz.renderer.set_geometry_vertices.call_count == 0
    assert mesh.payload is original_payload
    np.testing.assert_allclose(render_state_points(mesh), original_points)
    snapshots = [
        call.args[0]
        for call in mock_viz.renderer.ensure_object.call_args_list
        if call.args[0].id == mesh.id
    ]
    assert len(snapshots) == 1
    np.testing.assert_allclose(snapshots[0].transform.translation, [4.0, 5.0, 6.0])
    meta = asset.geometry_meta
    assert np.allclose(meta.scaled_aabb_center, np.array([0.0, 0.0, 0.0]))


def test_frame_processing_does_not_resolve_or_repaint_highlight_material():
    """The frame-hot path republishes the already-resolved parent material."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mock_viz.renderer.set_material = Mock(return_value=False)
    mesh = _make_mesh([0.0, 0.0, 0.0])

    entry = _make_target_entry("target_1", mesh, index=0)
    entry["pbr_properties"] = {
        "color": [0.7, 0.7, 0.7],
        "roughness": 0.5,
        "metallic": 0.0,
        "reflectance": 0.5,
    }
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []
    asset = _seed_target_asset(
        mock_viz,
        "target_1",
        "mesh.ply",
        mesh,
        scaled_vertices=np.asarray(
            [[-1.0, -0.5, 0.0], [1.0, -0.5, 0.0], [0.0, 0.5, 0.0]],
            dtype=np.float64,
        ),
    )
    entry["_target_asset"] = asset

    view_model = _make_view_model(
        ["target_1"],
        [[1.0, 2.0, 3.0]],
        position_valid=[True],
        use_ply=[False],
    )
    mock_viz.target_asset_cache.runtime_states["target_1"] = TargetRuntimeState(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        mesh_filename="mesh.ply",
        position_valid=True,
        use_ply_position=False,
        runtime_visible=True,
    )
    entry["highlighted"] = True

    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    snapshots = [
        call.args[0]
        for call in mock_viz.renderer.ensure_object.call_args_list
        if call.args[0].id == mesh.id
    ]
    assert len(snapshots) == 1
    assert snapshots[0].material_payload == mesh.material
    mock_viz.renderer.set_material.assert_not_called()
    mock_viz.renderer.set_named_material.assert_not_called()
    assert "_runtime_color_state" not in entry


def test_rotated_aabb_center_cache_reused_when_rotation_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    """Rotated AABB center should be reused when the mesh and rotation are unchanged."""
    mock_viz, target_service = _setup_target_service()
    cache_key = ("target_1", "mesh.ply")
    scaled_vertices = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    mesh = RenderObjectState(
        id="target:target_1::mesh",
        payload=MeshPayload(
            vertices=scaled_vertices.copy(),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )
    asset = _seed_target_asset(
        mock_viz,
        *cache_key,
        mesh,
        scaled_vertices=scaled_vertices,
    )

    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    first = target_service._resolve_rotated_aabb_center("target_1", "mesh.ply", rotation)
    assert first is not None

    recompute = Mock(side_effect=AssertionError("unchanged rotation should reuse metadata"))
    monkeypatch.setattr(
        "visualizer.src.services.target_service.compute_rotated_aabb_center",
        recompute,
    )
    second = target_service._resolve_rotated_aabb_center("target_1", "mesh.ply", rotation)

    assert np.allclose(second, first)
    assert asset.geometry_meta.rotated_aabb_center is not None
    recompute.assert_not_called()


def test_apply_target_scale_uses_aabb_center_not_vertex_mean():
    """Scale must stay anchored to the same AABB center used by runtime transforms."""
    mock_viz, target_service = _setup_target_service()
    original_vertices = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 4.0, 6.0], [2.0, 4.0, 6.0]],
        dtype=np.float64,
    )
    mesh = RenderObjectState(
        id="target:target_1::mesh",
        payload=MeshPayload(
            vertices=original_vertices.copy(),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )

    entry = _make_target_entry("target_1", mesh, index=0)
    entry["scale"] = 1.0
    asset = _seed_target_asset(
        mock_viz,
        "target_1",
        "mesh.ply",
        mesh,
        original_vertices=original_vertices,
        scaled_vertices=original_vertices,
    )
    entry["_target_asset"] = asset
    mock_viz.target_entries = [entry]

    assert target_service.apply_target_scale_from_metadata(entry, 2.0) is True

    aabb_center = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    expected = (original_vertices - aabb_center) * 2.0 + aabb_center
    np.testing.assert_allclose(render_state_points(mesh), expected)
    assert mock_viz.target_asset_cache.asset_for_logical_key(("target_1", "mesh.ply")) is asset
    np.testing.assert_allclose(asset.scaled_vertices, expected)


def test_scale_mutation_defers_one_complete_component_sync():
    """Frame assembly mutates scale first and publishes each component once."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mock_viz.target_outlines_enabled = True
    mesh = _make_mesh_handle("target_1")
    entry = _make_target_entry("target_1", mesh, index=0)
    entry["_target_position"] = [10.0, 20.0, 30.0]
    entry["_mesh_center"] = [0.5, 0.5, 0.0]
    label_name = make_target_entry_geometry_name(entry, "label")
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = [make_text_label_state(label_name, "Target 1", [0.8, 0.8, 0.8])]
    _seed_target_asset(mock_viz, "target_1", "mesh.ply", mesh)

    assert target_service.apply_target_scale_from_metadata(
        entry,
        2.0,
        sync_renderer=False,
    )
    assert entry["_renderer_sync_pending"] is True
    mock_viz.renderer.ensure_object.assert_not_called()

    assert target_service.sync_target_entry_snapshot(entry)

    calls_by_id: dict[str, list] = {}
    for call in mock_viz.renderer.ensure_object.call_args_list:
        snapshot = call.args[0]
        calls_by_id.setdefault(snapshot.id, []).append(snapshot)
    outline_name = make_target_entry_geometry_name(entry, "outline")
    assert set(calls_by_id) == {mesh.id, label_name, outline_name}
    assert all(len(calls) == 1 for calls in calls_by_id.values())
    assert entry["_renderer_sync_pending"] is False
    np.testing.assert_allclose(
        calls_by_id[mesh.id][0].transform.translation,
        [9.5, 19.5, 30.0],
    )


def test_immediate_scale_sync_preserves_effective_hidden_state():
    """Scaling a frame-hidden target must not transiently show it."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mesh = _make_mesh_handle("target_1")
    entry = _make_target_entry("target_1", mesh, index=0)
    entry["_frame_visible"] = False
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []
    _seed_target_asset(mock_viz, "target_1", "mesh.ply", mesh)

    assert target_service.apply_target_scale_from_metadata(
        entry,
        2.0,
        sync_renderer=True,
    )

    snapshots = [
        call.args[0]
        for call in mock_viz.renderer.ensure_object.call_args_list
        if call.args[0].id == mesh.id
    ]
    assert len(snapshots) == 1
    assert snapshots[0].visible is False
    assert mesh.visible is True


def test_process_targets_preserves_orientation_units_metadata():
    """Target orientation metadata remains yaw/pitch/roll radians plus degree display values."""
    mock_viz, target_service = _setup_target_service()
    mesh = _make_mesh([0.0, 0.0, 0.0])

    entry = _make_target_entry("target_1", mesh, index=0)
    mock_viz.target_entries = [entry]
    _seed_target_asset(mock_viz, "target_1", "mesh.ply", mesh)
    mock_viz.target_labels = []

    orientation = np.asarray([0.25, -0.5, 0.75], dtype=np.float64)
    view_model = SimpleNamespace(
        target_positions=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        target_orientations=np.asarray([orientation], dtype=np.float64),
        target_mesh_files=["mesh.ply"],
        target_use_ply_positions=[False],
        target_metadata=[{"name": "target_1", "mesh_file": "mesh.ply", "position_valid": True}],
    )

    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    np.testing.assert_allclose(entry["orientation_radians"], orientation)
    np.testing.assert_allclose(entry["orientation_degrees"], np.degrees(orientation))


def test_target_transform_matrix_uses_rotated_aabb_center():
    """Scene-graph transforms must follow Sionna's rotated-AABB placement convention."""
    mock_viz, target_service = _setup_target_service()
    cache_key = ("target_1", "mesh.ply")
    verts = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=np.float64,
    )
    angle = np.radians(45.0)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    target_pos = np.asarray([10.0, 20.0, 5.0], dtype=np.float64)
    entry = _make_target_entry("target_1", _make_mesh([0.0, 0.0, 0.0]), index=0)
    entry["_target_position"] = target_pos
    entry["_mesh_center"] = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    entry["_rotation_matrix"] = rotation
    entry["_target_asset"] = _seed_target_asset(
        mock_viz,
        *cache_key,
        entry["mesh"],
        scaled_vertices=verts,
    )

    transform = target_service._target_transform_matrix(entry)
    assert transform is not None

    rotated = (rotation @ verts.T).T
    final = rotated + transform[:3, 3]
    final_aabb_center = (final.min(axis=0) + final.max(axis=0)) / 2.0
    np.testing.assert_allclose(final_aabb_center, target_pos, atol=1e-6)


def test_transform_managed_target_publishes_mesh_without_imperative_fallback():
    """Transform-managed targets publish one snapshot and no outline alias."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mock_viz.renderer.set_transform = Mock(return_value=False)
    mesh = _make_mesh_handle("target_1")

    entry = _make_target_entry("target_1", mesh, index=0)
    entry["pbr_properties"] = {
        "color": [0.7, 0.7, 0.7],
        "roughness": 0.5,
        "metallic": 0.0,
        "reflectance": 0.5,
    }
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []
    _seed_target_asset(
        mock_viz,
        "target_1",
        "mesh.ply",
        mesh,
        scaled_vertices=np.asarray(
            [[-1.0, -0.5, 0.0], [1.0, -0.5, 0.0], [0.0, 0.5, 0.0]],
            dtype=np.float64,
        ),
    )

    view_model = _make_view_model(
        ["target_1"],
        [[1.0, 2.0, 3.0]],
        position_valid=[True],
        use_ply=[False],
    )

    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    assert mock_viz.renderer.update_geometry_in_visualizer.call_count == 0
    mesh_calls = [
        call.args[0]
        for call in mock_viz.renderer.ensure_object.call_args_list
        if call.args[0].id == mesh.id
    ]
    assert len(mesh_calls) == 1
    mock_viz.renderer.set_transform.assert_not_called()


def test_pygfx_fast_transform_success_does_not_log_fallback(caplog):
    """Successful fast transforms must not emit the slow-path fallback log."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mesh = _make_mesh([0.0, 0.0, 0.0])

    entry = _make_target_entry("target_1", mesh, index=0)
    entry["pbr_properties"] = {
        "color": [0.7, 0.7, 0.7],
        "roughness": 0.5,
        "metallic": 0.0,
        "reflectance": 0.5,
    }
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []
    _seed_target_asset(
        mock_viz,
        "target_1",
        "mesh.ply",
        mesh,
        scaled_vertices=np.asarray(
            [[-1.0, -0.5, 0.0], [1.0, -0.5, 0.0], [0.0, 0.5, 0.0]],
            dtype=np.float64,
        ),
    )

    view_model = _make_view_model(
        ["target_1"],
        [[1.0, 2.0, 3.0]],
        position_valid=[True],
        use_ply=[False],
    )

    with caplog.at_level("DEBUG", logger="orchav.target_service"):
        target_service.process_targets_from_view_model(step=0, view_model=view_model)

    assert "Fast transform failed, used slow path" not in caplog.text
    assert mock_viz.renderer.update_geometry_in_visualizer.call_count == 0


def test_transform_managed_target_publishes_one_idempotent_object_snapshot():
    """Backends, not services, decide whether a changed snapshot needs upload."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mock_viz.pipeline = SimpleNamespace(benchmark_recorder=object())
    mesh = _make_mesh([0.0, 0.0, 0.0])

    entry = _make_target_entry("target_1", mesh, index=0)
    entry["pbr_properties"] = {
        "color": [0.7, 0.7, 0.7],
        "roughness": 0.5,
        "metallic": 0.0,
        "reflectance": 0.5,
    }
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []
    _seed_target_asset(
        mock_viz,
        "target_1",
        "mesh.ply",
        mesh,
        scaled_vertices=np.asarray(
            [[-1.0, -0.5, 0.0], [1.0, -0.5, 0.0], [0.0, 0.5, 0.0]],
            dtype=np.float64,
        ),
    )

    view_model = _make_view_model(
        ["target_1"],
        [[1.0, 2.0, 3.0]],
        position_valid=[True],
        use_ply=[False],
    )

    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    breakdown = target_service.get_last_runtime_breakdown()
    assert breakdown["target_transform_snapshot_count"] == 1.0
    assert "target_handoff_runtime_mesh_visibility_only_count" not in breakdown
    assert breakdown["target_handoff_sync_entity_count"] == 1.0
    mesh_calls = [
        call.args[0]
        for call in mock_viz.renderer.ensure_object.call_args_list
        if call.args[0].id == mesh.id
    ]
    assert len(mesh_calls) == 1
    assert mock_viz.renderer.update_geometry_in_visualizer.call_count == 0


def test_target_edge_toggle_applies_object_transform():
    """Target edges added after mesh placement must inherit the target transform."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mock_viz.renderer.renderer_type = "pygfx"
    mock_viz.renderer.ensure_object = Mock(return_value=True)
    mock_viz.renderer.set_transform = Mock(return_value=True)
    mock_viz.renderer.set_visible = Mock(return_value=True)

    mesh = _make_mesh_handle()
    entry = _make_target_entry("target_1", mesh, index=0)
    entry["_target_position"] = [10.0, 20.0, 30.0]
    entry["_mesh_center"] = [1.0, 2.0, 3.0]
    mesh.world_transform = Transform.from_translation([9.0, 18.0, 27.0])
    mock_viz.target_entries = [entry]

    target_service.set_target_edge_visibility(True)

    mock_viz.renderer.ensure_object.assert_called()
    snapshot = mock_viz.renderer.ensure_object.call_args.args[0]
    assert snapshot.id == "target:target_1::outline"
    assert np.allclose(snapshot.transform_matrix[:3, 3], [9.0, 18.0, 27.0])
    mock_viz.renderer.set_transform.assert_not_called()
    mock_viz.renderer.set_visible.assert_not_called()
    assert entry["outline_visible"] is True


def test_target_outline_runtime_hiding_preserves_semantic_outline_intent():
    """Frame and POV hiding affect snapshots without changing outline intent."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mock_viz.renderer.ensure_object = Mock(return_value=True)
    mock_viz.renderer.set_visible = Mock(return_value=True)
    target_service._is_hidden_for_pov = Mock(return_value=False)

    mesh = _make_mesh_handle()
    entry = _make_target_entry("target_1", mesh, index=0)
    entry["_frame_visible"] = True
    mock_viz.target_entries = [entry]

    target_service.set_target_edge_visibility(True)
    outline = entry["outline_geometry"]
    assert outline.visible is True
    assert entry["outline_visible"] is True

    entry["_frame_visible"] = False
    target_service.sync_target_entry_edge_visibility(entry, update_renderer=False)
    frame_hidden = mock_viz.renderer.ensure_object.call_args.args[0]
    assert frame_hidden.visible is False
    assert outline.visible is True
    assert entry["outline_visible"] is False

    entry["_frame_visible"] = True
    target_service._is_hidden_for_pov.return_value = True
    target_service.sync_target_entry_edge_visibility(entry, update_renderer=False)
    pov_hidden = mock_viz.renderer.ensure_object.call_args.args[0]
    assert pov_hidden.visible is False
    assert outline.visible is True
    assert entry["outline_visible"] is False

    # The declarative snapshot already carries transform and visibility, so
    # the service must not issue duplicate imperative updates afterward.
    mock_viz.renderer.set_transform.assert_not_called()
    mock_viz.renderer.set_visible.assert_not_called()


def test_frame_hidden_target_processing_does_not_show_outline():
    """The per-frame target path must not recreate a visible orphan outline."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mock_viz.target_outlines_enabled = True
    mock_viz.renderer.ensure_object = Mock(return_value=True)
    mock_viz.renderer.set_visible = Mock(return_value=True)
    target_service._is_hidden_for_pov = Mock(return_value=False)

    mesh = _make_mesh_handle()
    entry = _make_target_entry("target_1", mesh, index=0)
    mock_viz.target_entries = [entry]
    _seed_target_asset(mock_viz, "target_1", "mesh.ply", mesh)
    mock_viz.target_labels = []

    view_model = _make_view_model(
        ["target_1"],
        [[0.0, 0.0, 0.0]],
        position_valid=[False],
        use_ply=[False],
    )

    target_service.process_targets_from_view_model(step=0, view_model=view_model)

    assert entry["_frame_visible"] is False
    assert entry["outline_visible"] is False
    assert entry.get("outline_geometry") is None


def test_target_outline_invalidation_defers_replacement_to_backend_owner():
    """Invalidation retains the stable handle without removing the native object."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mock_viz.renderer.renderer_type = "pygfx"
    mock_viz.renderer.ensure_object = Mock(return_value=True)
    mock_viz.renderer.remove_object = Mock(return_value=True)

    mesh = _make_mesh_handle()
    entry = _make_target_entry("target_1", mesh, index=0)
    entry["_target_position"] = [10.0, 20.0, 30.0]
    entry["_mesh_center"] = [1.0, 2.0, 3.0]
    mock_viz.target_entries = [entry]

    target_service.set_target_edge_visibility(True)
    outline = entry["outline_geometry"]
    target_service._invalidate_target_outline(entry)

    mock_viz.renderer.remove_object.assert_not_called()
    assert entry["outline_geometry"] is outline
    assert entry["outline_visible"] is False
    assert entry["_outline_payload_dirty"] is True
    assert "_outline_geometry_name" not in entry

    target_service.set_target_edge_visibility(True)
    replacement = entry["outline_geometry"]
    assert replacement is outline
    assert replacement.id == outline.id
    assert entry["_outline_payload_dirty"] is False
    mock_viz.renderer.remove_object.assert_not_called()


# ------------------------------------------------------------------
# Regression tests for preload glob and cache-miss fixes
# ------------------------------------------------------------------


def test_preload_glob_finds_all_ply_files_nonzero_start():
    """Preload with mesh_start_index != 0 should cache ALL PLY files in the directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 240 numbered PLY files (simulating smpl_hand_signals_00001..00240)
        filenames = []
        for i in range(1, 241):
            fname = f"smpl_hand_signals_{i:05d}.ply"
            filenames.append(fname)
            # Write minimal PLY so the file exists
            with open(os.path.join(tmpdir, fname), "w") as f:
                f.write("ply\n")

        # The glob pattern should be "*.ply" regardless of start index
        ext = ".ply"
        mesh_pattern = f"*{ext}"
        full_pattern = os.path.join(tmpdir, mesh_pattern)
        matching = sorted(glob.glob(full_pattern))

        assert len(matching) == 240
        # First file is 00001, last is 00240
        assert os.path.basename(matching[0]) == "smpl_hand_signals_00001.ply"
        assert os.path.basename(matching[-1]) == "smpl_hand_signals_00240.ply"


def test_preload_glob_crosses_00100_boundary():
    """Glob must find files with indices >= 00100 (old 000* pattern missed these)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files spanning the old boundary
        for i in [1, 50, 99, 100, 101, 150]:
            fname = f"mesh_{i:05d}.ply"
            with open(os.path.join(tmpdir, fname), "w") as f:
                f.write("ply\n")

        mesh_pattern = "*.ply"
        matching = sorted(glob.glob(os.path.join(tmpdir, mesh_pattern)))
        assert len(matching) == 6
        names = [os.path.basename(f) for f in matching]
        assert "mesh_00100.ply" in names
        assert "mesh_00150.ply" in names


def test_cache_miss_no_metadata_corruption():
    """Cache miss must NOT update target_entry['mesh_file'] to the missing mesh."""
    mock_viz, target_service = _setup_target_service()
    old_mesh = _make_mesh([0.0, 0.0, 0.0])

    entry = _make_target_entry("target_1", old_mesh, index=0)
    entry["mesh_file"] = "mesh_old.ply"
    mock_viz.target_entries = [entry]
    # Only the old mesh is in cache — new mesh is missing
    _seed_target_asset(mock_viz, "target_1", "mesh_old.ply", old_mesh)
    mock_viz.target_labels = []
    # Disable scenario policy so _load_mesh_on_demand uses mesh_directory directly
    mock_viz.current_scenario_policy = None

    view_model = SimpleNamespace(
        target_positions=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        target_orientations=np.zeros((1, 3), dtype=np.float64),
        target_mesh_files=["mesh_new.ply"],
        target_use_ply_positions=[False],
        target_metadata=[
            {
                "name": "target_1",
                "mesh_file": "mesh_new.ply",
                "mesh_directory": "/nonexistent",
                "position_valid": True,
            }
        ],
    )

    target_service.process_targets_from_view_model(step=5, view_model=view_model)

    # mesh_file must still be the OLD (actually rendered) file
    assert entry["mesh_file"] == "mesh_old.ply"
    # Mesh object must still be the old mesh
    assert entry["mesh"] is old_mesh
    # State cache must also record the old mesh file so change-detection retries
    state_cache = mock_viz.target_asset_cache.runtime_states
    assert state_cache["target_1"].mesh_filename == "mesh_old.ply"


def test_cache_miss_on_demand_loads_mesh():
    """Cache miss with file on disk should trigger on-demand load and cache it."""
    mock_viz, target_service = _setup_target_service()
    old_mesh = _make_mesh([0.0, 0.0, 0.0])

    entry = _make_target_entry("target_1", old_mesh, index=0)
    entry["mesh_file"] = "mesh_old.ply"
    mock_viz.target_entries = [entry]
    mock_viz.target_labels = []
    entry["_target_asset"] = _seed_target_asset(
        mock_viz,
        "target_1",
        "mesh_old.ply",
        old_mesh,
    )

    # Disable scenario policy so _load_mesh_on_demand uses mesh_directory directly
    mock_viz.current_scenario_policy = None

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the mesh file on disk
        mesh_path = os.path.join(tmpdir, "mesh_new.ply")
        _write_ascii_ply(mesh_path)

        view_model = SimpleNamespace(
            target_positions=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
            target_orientations=np.zeros((1, 3), dtype=np.float64),
            target_mesh_files=["mesh_new.ply"],
            target_use_ply_positions=[False],
            target_metadata=[
                {
                    "name": "target_1",
                    "mesh_file": "mesh_new.ply",
                    "mesh_directory": tmpdir,
                    "scale": 1.0,
                    "orientation": [0.0, 0.0, 0.0],
                    "material_type": "custom",
                    "position_valid": True,
                }
            ],
        )

        target_service.process_targets_from_view_model(step=5, view_model=view_model)

    # On-demand load should have populated the cache
    loaded = mock_viz.target_asset_cache.asset_for_logical_key(("target_1", "mesh_new.ply"))
    assert isinstance(loaded, TargetAsset)
    assert isinstance(loaded.mesh, RenderObjectState)


def test_load_target_models_skips_baked_orientation_for_transform_managed_targets():
    """Pygfx-style transform-managed targets must keep cached meshes unrotated."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mock_viz.cache_service = Mock()
    mock_viz.frame_source = Mock()
    mock_viz.frame_source.has_frame = Mock(return_value=True)
    mock_viz.frame_source.list_frames = Mock(return_value=[0])
    mock_viz._set_status_message = Mock()
    mock_viz.total_animation_steps = 0
    mock_viz.target_entries = []
    mock_viz.target_meshes = {}
    mock_viz.current_scenario_policy = None

    with tempfile.TemporaryDirectory() as tmpdir:
        mesh_path = os.path.join(tmpdir, "mesh_00001.ply")
        original_vertices = _write_ascii_ply(mesh_path)

        mock_viz.cache_service.get_frame = Mock(
            return_value={
                "targets_metadata": [
                    {
                        "name": "target_1",
                        "mesh_file": "mesh_00001.ply",
                        "mesh_directory": tmpdir,
                        "scale": 1.0,
                        "orientation": [0.2, -0.1, 0.4],
                        "use_ply_position": False,
                        "material_type": "custom",
                        "current_position": [0.0, 0.0, 0.0],
                    }
                ]
            }
        )

        target_service.load_target_models()

    loaded = mock_viz.target_asset_cache.asset_for_logical_key(("target_1", "mesh_00001.ply"))
    assert isinstance(loaded, TargetAsset)
    np.testing.assert_allclose(render_state_points(loaded.mesh), original_vertices)
    mock_viz.renderer.ensure_object.assert_not_called()
    mock_viz.renderer.ensure_named_geometry.assert_not_called()


def test_load_mesh_on_demand_skips_baked_orientation_for_transform_managed_targets():
    """Pygfx on-demand loads must not pre-rotate meshes that use scene transforms."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mock_viz.current_scenario_policy = None

    with tempfile.TemporaryDirectory() as tmpdir:
        mesh_path = os.path.join(tmpdir, "mesh_new.ply")
        original_vertices = _write_ascii_ply(mesh_path)

        loaded = target_service._load_mesh_on_demand(
            "target_1",
            "mesh_new.ply",
            {
                "mesh_directory": tmpdir,
                "scale": 1.0,
                "orientation": [0.2, -0.1, 0.4],
                "use_ply_position": False,
                "material_type": "custom",
            },
            mock_viz,
        )

    assert isinstance(loaded, TargetAsset)
    np.testing.assert_allclose(render_state_points(loaded.mesh), original_vertices)
    mock_viz.renderer.ensure_object.assert_not_called()
    mock_viz.renderer.ensure_named_geometry.assert_not_called()


def test_load_mesh_on_demand_bakes_orientation_when_using_ply_position():
    """Embedded-position targets still need load-time orientation baked in."""
    mock_viz, target_service = _setup_pygfx_target_service()
    mock_viz.current_scenario_policy = None

    with tempfile.TemporaryDirectory() as tmpdir:
        mesh_path = os.path.join(tmpdir, "mesh_new.ply")
        original_vertices = _write_ascii_ply(mesh_path)

        loaded = target_service._load_mesh_on_demand(
            "target_1",
            "mesh_new.ply",
            {
                "mesh_directory": tmpdir,
                "scale": 1.0,
                "orientation": [0.2, -0.1, 0.4],
                "use_ply_position": True,
                "material_type": "custom",
            },
            mock_viz,
        )

    assert isinstance(loaded, TargetAsset)
    assert not np.allclose(render_state_points(loaded.mesh), original_vertices)
    mock_viz.renderer.ensure_object.assert_not_called()
    mock_viz.renderer.ensure_named_geometry.assert_not_called()


def test_load_target_models_uses_cached_frame_prime():
    """Startup target loading should reuse the shared frame cache path."""
    mock_viz, target_service = _setup_target_service()
    mock_viz.cache_service = CacheService(mock_viz)
    mock_viz.frame_source = Mock()
    mock_viz.frame_source.has_frame = Mock(return_value=True)
    mock_viz.frame_source.load_frame = Mock(side_effect=AssertionError("direct load not expected"))
    mock_viz.animation_service = Mock()
    mock_viz.animation_service.ensure_step_cached = Mock(return_value={"targets_metadata": []})
    mock_viz._set_status_message = Mock()
    _seed_target_asset(
        mock_viz,
        "stale",
        "stale.ply",
        _make_mesh_handle("stale"),
    )
    mock_viz.target_entries = [{"name": "stale"}]

    target_service.load_target_models()

    mock_viz.animation_service.ensure_step_cached.assert_called_once_with(0)
    mock_viz.frame_source.load_frame.assert_not_called()
    assert mock_viz.target_asset_cache.logical_keys() == ()
    assert mock_viz.target_entries == []
    assert mock_viz.num_targets == 0


def test_load_target_models_projects_direct_standard_frame_fallback():
    """Direct source loading must enter the target mapping seam."""
    mock_viz, target_service = _setup_target_service()
    mock_viz.cache_service = CacheService(mock_viz)
    mock_viz.frame_source = Mock()
    mock_viz.frame_source.has_frame = Mock(return_value=True)
    mock_viz.frame_source.list_frames = Mock(return_value=[0])
    mock_viz.frame_source.load_frame = Mock(return_value=build_standard_mpc_frame())
    mock_viz.animation_service = None
    mock_viz.mpc_core.canon_points_dtype = np.float32
    mock_viz._set_status_message = Mock()
    mock_viz.target_entries = []
    mock_viz.target_meshes = {}

    target_service.load_target_models()

    cached = mock_viz.cache_service.get_frame(0)
    assert isinstance(cached, dict)
    assert "canonical_data" in cached
    assert cached["targets_metadata"] == []
    assert mock_viz.num_targets == 0


def test_transform_matches_sionna_aabb_convention():
    """The 4x4 transform must place the rotated mesh's AABB center at pos.

    Sionna RT positions meshes so that AABB_center(R @ vertices) == pos.
    The renderer must NOT use pos - R @ center (which differs because
    R @ AABB_center(v) != AABB_center(R @ v) for non-symmetric shapes).
    """
    # Create a deliberately asymmetric mesh (AABB center != geometric center)
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],  # asymmetric in Y
            [0.0, 0.0, 3.0],  # asymmetric in Z
        ],
        dtype=np.float64,
    )
    aabb_center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0

    # Rotation: 45 degrees around Z-axis
    angle = np.radians(45.0)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    target_pos = np.array([10.0, 20.0, 5.0])

    # Correct (Sionna convention): rotate then position AABB center
    rotated_verts = (R @ verts.T).T
    rotated_aabb_center = (rotated_verts.min(axis=0) + rotated_verts.max(axis=0)) / 2.0
    correct_translation = target_pos - rotated_aabb_center

    # Wrong (old formula): pos - R @ center
    wrong_translation = target_pos - R @ aabb_center

    # Verify they're different (this IS the bug condition)
    assert not np.allclose(
        correct_translation, wrong_translation, atol=1e-6
    ), "Test mesh is too symmetric — AABB center doesn't shift under rotation"

    # Verify correct transform places AABB center at target_pos
    final_verts = rotated_verts + correct_translation
    final_aabb = (final_verts.min(axis=0) + final_verts.max(axis=0)) / 2.0
    np.testing.assert_allclose(final_aabb, target_pos, atol=1e-10)

    # Verify wrong transform does NOT place AABB center at target_pos
    wrong_final = rotated_verts + wrong_translation
    wrong_aabb = (wrong_final.min(axis=0) + wrong_final.max(axis=0)) / 2.0
    assert not np.allclose(wrong_aabb, target_pos, atol=1e-3)
