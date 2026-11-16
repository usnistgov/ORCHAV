"""Native point replay, visibility, and packed sensing-path identity."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from shared.frames import FrameComponent, StandardMPCFrame, project_standard_mpc_frame
from shared.frames.provider_base import DataProvider
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from visualizer.src.io.packed_frame_payload import (
    standard_frame_to_visual_frame,
    try_load_packed_visual_frame,
    visual_frame_read_request,
)
from visualizer.src.metrics.mpc_path_catalog import MpcPathCatalog, interaction_type_bit
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.mpc_interaction_style_service import mpc_interaction_label
from visualizer.src.services.target_service import TargetService


class RecordingRenderer:
    capabilities = RendererCapabilities()

    def __init__(self):
        self.objects = {}

    def batch_updates(self):
        return nullcontext()

    def ensure_object(self, render_object):
        self.objects[render_object.id] = render_object
        return True


def metadata():
    return {
        "name": "car",
        "representation": "tr38901",
        "geometry": {"type": "box", "length_m": 5, "width_m": 2, "height_m": 1.6},
        "current_position": [10, 20, 3],
        "orientation": [0, 0, 0],
        "scattering_point_ids": [0, 1],
        "scattering_points_world_m": [[9, 20, 3], [11, 20, 3]],
        "metadata_source": "native_runtime",
    }


def _native_frame(step, source_step, *, representation="tr38901", embedded=False):
    target = metadata()
    target["representation"] = representation
    target["current_position"] = [10 + step, 20, 3]
    target["orientation"] = [0.01 * step, 0, 0]
    target["scattering_points_world_m"] = [[9 + step, 20, 3], [11 + step, 20, 3]]
    provenance = {"source_rt_frame_idx": source_step}
    if embedded:
        source = _native_frame(source_step, source_step, representation=representation)
        provenance["sionna_sensing"] = {
            "schema_version": 1,
            "bounce_indices": [],
            "target_indices": [],
            "scattering_point_indices": [],
            "target_state_at_solve": list(source.targets_metadata),
        }
    return replace(
        build_standard_mpc_frame("empty", frame_idx=step),
        target_positions_m=np.asarray([target["current_position"]], dtype=np.float64),
        targets_metadata=(target,),
        provenance=provenance,
    )


class _PoseProvider(DataProvider):
    def __init__(self, frames):
        self.frames = {frame.frame_index: frame for frame in frames}
        self.reads = []

    def list_frames(self):
        return sorted(self.frames)

    def has_frame(self, step):
        return step in self.frames

    def load_frame(self, step):
        pytest.fail("Native pose replay must keep selective frame loading")

    def load_frame_projection(self, step, request):
        self.reads.append((step, request))
        return project_standard_mpc_frame(self.frames[step], request)


@pytest.mark.parametrize("representation", ["constant_rcs", "tr38901"])
def test_native_pose_holds_within_cpi_and_advances_on_boundary_when_seeking(representation):
    steps = [0, 1, 32, 63, 64, 65, 127, 128]
    frames = [
        _native_frame(step, step // 64 * 64, representation=representation, embedded=step == 63)
        for step in steps
    ]
    provider = _PoseProvider(frames)
    original_metadata = [deepcopy(frame.targets_metadata) for frame in frames]
    original_positions = [frame.target_positions_m.copy() for frame in frames]
    renderer = RecordingRenderer()
    viz = SimpleNamespace(
        renderer=renderer, target_entries=[], target_labels=[], app_state=None, vis=None
    )
    service = TargetService(viz)
    entry = None
    # Include backwards/random seeks: holding the last displayed target is wrong.
    for step in [*steps, 32, 127, 1]:
        payload = try_load_packed_visual_frame(provider, step, request=visual_frame_read_request())
        expected = provider.frames[step // 64 * 64].targets_metadata[0]
        actual = payload["targets_metadata"][0]
        assert actual == expected
        np.testing.assert_array_equal(payload["target_pos"][0], expected["current_position"])
        if entry is None:
            entry = service._native_target_entry(actual, 0)
        else:
            service._update_native_target_entry(entry, actual)
        assert service.sync_target_entry_snapshot(entry)
        np.testing.assert_array_equal(entry["position"], expected["current_position"])
        np.testing.assert_array_equal(entry["orientation"], expected["orientation"])
        for point, position in zip(
            entry["_native_scattering_points"], expected["scattering_points_world_m"], strict=True
        ):
            np.testing.assert_array_equal(
                renderer.objects[point.id].transform.matrix[:3, 3], position
            )
    for frame, before_metadata, before_position in zip(
        frames, original_metadata, original_positions, strict=True
    ):
        assert frame.targets_metadata == before_metadata
        np.testing.assert_array_equal(frame.target_positions_m, before_position)
    reference_reads = [request for _, request in provider.reads if not request.metrics]
    assert reference_reads
    assert all(
        request.components == frozenset({FrameComponent.TARGETS, FrameComponent.PROVENANCE})
        for request in reference_reads
    )


def test_completed_native_cpi_uses_embedded_solve_pose_without_source_io():
    frame = _native_frame(63, 0, embedded=True)
    provider = _PoseProvider([frame])
    payload = try_load_packed_visual_frame(provider, 63, request=visual_frame_read_request())
    assert [step for step, _ in provider.reads] == [63]
    np.testing.assert_array_equal(payload["target_pos"], [[10, 20, 3]])
    direct = standard_frame_to_visual_frame(frame, request=visual_frame_read_request())
    assert direct["targets_metadata"] == payload["targets_metadata"]


@pytest.mark.parametrize("source_step", [None, True, -1, 65, 64])
def test_native_replay_requires_an_explicit_earlier_solve(source_step):
    frame = _native_frame(64, source_step)
    provider = _PoseProvider([frame])
    payload = try_load_packed_visual_frame(provider, 64, request=visual_frame_read_request())
    np.testing.assert_array_equal(payload["target_pos"], frame.target_positions_m)
    assert [step for step, _ in provider.reads] == [64]


def test_sequential_native_replay_keeps_every_pulse_pose():
    provider = _PoseProvider([_native_frame(step, step) for step in range(3)])
    for step in range(3):
        payload = try_load_packed_visual_frame(provider, step, request=visual_frame_read_request())
        np.testing.assert_array_equal(payload["target_pos"], [[10 + step, 20, 3]])
    assert [step for step, _ in provider.reads] == [0, 1, 2]


def test_partial_recording_without_source_keeps_recorded_pose():
    frame = _native_frame(32, 0)
    provider = _PoseProvider([frame])
    payload = try_load_packed_visual_frame(provider, 32, request=visual_frame_read_request())
    np.testing.assert_array_equal(payload["target_pos"], frame.target_positions_m)
    assert payload["targets_metadata"] == list(frame.targets_metadata)


def test_native_source_pose_matches_identity_in_a_mixed_target_catalog():
    source = _native_frame(0, 0)
    mesh = {"name": "mesh", "representation": "mesh_material", "current_position": [3, 4, 5]}
    frame = replace(
        _native_frame(32, 0),
        targets_metadata=(mesh, _native_frame(32, 0).targets_metadata[0]),
        target_positions_m=np.array([[3, 4, 5], [42, 20, 3]], dtype=np.float64),
    )
    provider = _PoseProvider([source, frame])
    payload = try_load_packed_visual_frame(provider, 32, request=visual_frame_read_request())
    assert payload["targets_metadata"][0] == mesh
    np.testing.assert_array_equal(payload["target_pos"], [[3, 4, 5], [10, 20, 3]])


def test_native_pose_agrees_in_preloaded_and_notebook_playback(qapp):
    from visualizer.src.notebook.frame_data import load_notebook_visual_frame
    from visualizer.src.services.animation_service import _PreloadWorker
    from visualizer.src.services.cache_service import CacheService

    provider = _PoseProvider([_native_frame(step, step // 64 * 64) for step in (0, 32, 64, 65)])
    cache = CacheService(SimpleNamespace())
    worker = _PreloadWorker(
        SimpleNamespace(provider=provider),
        None,
        cache,
        [32, 64, 65],
        is_current=lambda: True,
        run_if_current=lambda action: action() or True,
    )
    worker.run()
    for step in (32, 64, 65):
        preloaded = cache.get_frame(step)
        notebook = load_notebook_visual_frame(SimpleNamespace(provider=provider), step)
        np.testing.assert_array_equal(preloaded["target_pos"], [[10 + step // 64 * 64, 20, 3]])
        np.testing.assert_array_equal(preloaded["target_pos"], notebook["target_pos"])
        assert preloaded["targets_metadata"] == notebook["targets_metadata"]


def test_retained_markers_keep_identity_hide_with_parent_and_reappear():
    renderer = RecordingRenderer()
    viz = SimpleNamespace(
        renderer=renderer, target_entries=[], target_labels=[], app_state=None, vis=None
    )
    service = TargetService(viz)
    entry = service._native_target_entry(metadata(), 0)
    viz.target_entries.append(entry)
    assert service.sync_target_entry_snapshot(entry)
    original = entry["_native_scattering_points"]
    ids = tuple(point.id for point in original)
    np.testing.assert_array_equal(renderer.objects[ids[0]].transform.matrix[:3, 3], [9, 20, 3])
    entry["visible"] = False
    assert service.sync_target_entry_snapshot(entry)
    assert not any(renderer.objects[key].visible for key in ids)
    entry["visible"] = True
    changed = metadata()
    changed["scattering_points_world_m"] = [[90, 0, 0], [91, 0, 0]]
    service._update_native_target_entry(entry, changed)
    assert service.sync_target_entry_snapshot(entry)
    assert tuple(point.id for point in entry["_native_scattering_points"]) == ids
    assert entry["_native_scattering_points"][0] is original[0]
    assert renderer.objects[ids[0]].visible
    np.testing.assert_array_equal(renderer.objects[ids[0]].transform.matrix[:3, 3], [90, 0, 0])
    assert original[0].metadata["visibility_evidence"] == "not_recorded"


def test_missing_recorded_points_hide_previous_markers():
    viz = SimpleNamespace(
        renderer=RecordingRenderer(), target_entries=[], target_labels=[], app_state=None, vis=None
    )
    service = TargetService(viz)
    entry = service._native_target_entry(metadata(), 0)
    missing = metadata()
    missing.pop("scattering_point_ids")
    missing.pop("scattering_points_world_m")
    service._update_native_target_entry(entry, missing)
    assert all(not point.visible for point in entry["_native_scattering_points"])


def _native_appearance_services(tmp_path, monkeypatch, *, alpha=1.0):
    from pathlib import Path

    from visualizer.src.services.material_pbr_service import MaterialPBRService
    from visualizer.src.services.object_appearance_service import ObjectAppearanceService
    from visualizer.src.services.scene_appearance_service import SceneAppearanceService

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    renderer = RecordingRenderer()
    renderer.capabilities = RendererCapabilities(transparency=True)
    viz = SimpleNamespace(
        renderer=renderer,
        target_entries=[],
        target_labels=[],
        mesh_entries=[],
        app_state=None,
        vis=object(),
        vis_initialized=True,
        current_target_alpha=alpha,
    )
    viz.target_service = TargetService(viz)
    viz.object_appearance_service = ObjectAppearanceService(viz)
    viz.material_pbr_service = MaterialPBRService(viz)
    viz.scene_appearance_service = SceneAppearanceService(viz)
    entry = viz.target_service._native_target_entry(metadata(), 0)
    viz.target_entries.append(entry)
    assert viz.target_service.sync_target_entry_snapshot(entry)
    return viz, entry


def test_native_opacity_restore_and_material_refresh_preserve_authored_proxy(tmp_path, monkeypatch):
    viz, entry = _native_appearance_services(tmp_path, monkeypatch)
    mesh_id = entry["mesh"].id
    initial = viz.renderer.objects[mesh_id].material_payload
    point_ids = tuple(point.id for point in entry["_native_scattering_points"])
    point_materials = tuple(viz.renderer.objects[key].material_payload for key in point_ids)
    assert initial.base_color[3] == 0.25
    for slider_alpha, expected_alpha in ((0.0, 0.0), (0.4, 0.4), (1.0, 0.25)):
        viz.scene_appearance_service.set_target_transparency(slider_alpha)
        assert viz.renderer.objects[mesh_id].material_payload.base_color[3] == expected_alpha
        assert all(viz.renderer.objects[key].visible for key in point_ids)
        assert tuple(viz.renderer.objects[key].material_payload for key in point_ids) == (
            point_materials
        )
    assert viz.renderer.objects[mesh_id].material_payload == initial
    assert viz.object_appearance_service.refresh_entry_material(entry, highlighted=True)
    assert viz.renderer.objects[mesh_id].material_payload.base_color[3] == 0.25
    assert viz.object_appearance_service.refresh_entry_material(entry, highlighted=False)
    assert viz.renderer.objects[mesh_id].material_payload == initial
    changed = metadata()
    changed["current_position"] = [30, 20, 3]
    changed["scattering_points_world_m"] = [[29, 20, 3], [31, 20, 3]]
    viz.target_service._update_native_target_entry(entry, changed)
    assert viz.target_service.sync_target_entry_snapshot(entry)
    assert viz.renderer.objects[mesh_id].material_payload == initial
    assert tuple(point.id for point in entry["_native_scattering_points"]) == point_ids


@pytest.mark.parametrize("alpha,expected", [(0.0, 0.0), (0.4, 0.4), (1.0, 0.25)])
def test_new_native_proxy_inherits_existing_global_opacity(tmp_path, monkeypatch, alpha, expected):
    viz, entry = _native_appearance_services(tmp_path, monkeypatch, alpha=alpha)
    assert viz.renderer.objects[entry["mesh"].id].material_payload.base_color[3] == expected
    assert all(
        viz.renderer.objects[point.id].visible for point in entry["_native_scattering_points"]
    )


def test_native_asymmetric_mesh_uses_rotated_aabb_center(tmp_path):
    mesh_path = tmp_path / "proxy.obj"
    mesh_path.write_text("v 0 0 0\nv 4 0 0\nv 0 2 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n")
    recorded = metadata()
    recorded["geometry"] = {"type": "mesh", "path": str(mesh_path), "scaling": [1, 1, 1]}
    recorded["orientation"] = [np.pi / 4, 0, 0]
    viz = SimpleNamespace(
        renderer=RecordingRenderer(), target_entries=[], target_labels=[], app_state=None, vis=None
    )
    service = TargetService(viz)
    entry = service._native_target_entry(recorded, 0)
    transform = service._target_transform_matrix(entry)
    vertices = entry["mesh"].payload.vertices
    world = vertices @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(
        (world.min(axis=0) + world.max(axis=0)) / 2, recorded["current_position"]
    )


def test_packed_sensing_identity_maps_past_endpoints_and_regular_bounces():
    zeros = np.zeros((1, 3), dtype=np.float64)
    metrics = {
        name: np.full(2, np.nan, dtype=np.float32)
        for name in (
            "delays_ns",
            "path_loss_db",
            "aoa_az_deg",
            "aoa_el_deg",
            "aod_az_deg",
            "aod_el_deg",
        )
    }
    target = metadata()
    frame = StandardMPCFrame(
        frame_index=0,
        tx_rx_pairs=np.array([[0, 0]], dtype=np.int32),
        pair_path_offsets=np.array([0, 2], dtype=np.int64),
        bounce_offsets=np.array([0, 0, 2], dtype=np.int64),
        tx_positions=zeros,
        rx_positions=np.ones((1, 3), dtype=np.float64),
        tx_orientations=zeros,
        rx_orientations=zeros,
        tx_names=("tx",),
        rx_names=("rx",),
        bounce_xyz_m=np.array([[2, 0, 0], [9, 20, 3]], dtype=np.float32),
        interactions=np.array([1, 16], dtype=np.uint8),
        material_ids=np.zeros(2, dtype=np.uint16),
        material_names=("",),
        material_itu_types=("",),
        metric_valid_bits=np.zeros(2, dtype=np.uint8),
        target_positions_m=np.array([target["current_position"]], dtype=np.float64),
        targets_metadata=(target,),
        provenance={
            "sionna_sensing": {
                "schema_version": 1,
                "bounce_indices": [1],
                "target_indices": [0],
                "scattering_point_indices": [1],
            }
        },
        **metrics,
    )
    canonical = standard_frame_to_visual_frame(frame, request=visual_frame_read_request())[
        "canonical_data"
    ]
    assert canonical.native_sensing_points == {4: (0, "car", 1)}
    catalog = MpcPathCatalog(canonical)
    assert catalog.sensing_sequence(0) == ()
    assert catalog.sensing_sequence(1) == ((2, "car", 1),)
    np.testing.assert_array_equal(catalog.contains_interaction(16), [False, True])
    assert mpc_interaction_label(16) == "Sensing"
    assert interaction_type_bit(16) != interaction_type_bit(123)
