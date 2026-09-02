"""Focused safety tests for pygfx fixed-topology vertex streaming."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import visualizer.src.renderers.pygfx.geometry as pygfx_geometry_module
from visualizer.src.model import RenderObject, Transform
from visualizer.src.renderers.pygfx.geometry import PygfxGeometryMixin
from visualizer.src.types.render_payloads import (
    MaterialPayload,
    MeshPayload,
    SurfaceColorSource,
    mesh_payload_for_pbr_material,
)


class _Buffer:
    def __init__(self, data: np.ndarray, *, fail: bool = False) -> None:
        self.data = np.array(data, copy=True)
        self.fail = fail
        self.update_full_calls = 0

    def update_full(self) -> None:
        self.update_full_calls += 1
        if self.fail:
            self.fail = False
            raise RuntimeError("simulated vertex upload failure")


class _VertexStreamHarness(PygfxGeometryMixin):
    def __init__(self, payload: MeshPayload) -> None:
        self.name = "target:walker::mesh"
        self.material = MaterialPayload(base_color=(0.3, 0.4, 0.5, 1.0))
        installed_payload = (
            mesh_payload_for_pbr_material(payload)
            if payload.color_source is not SurfaceColorSource.VERTEX
            and payload.vertex_colors is not None
            else payload
        )
        prepared = self._prepare_geometry_buffers(installed_payload)
        assert prepared is not None
        geometry = SimpleNamespace(**{name: _Buffer(values) for name, values in prepared.items()})
        self._name_to_handle = {self.name: 1}
        self._objects = {
            self.name: SimpleNamespace(geometry=geometry, material=SimpleNamespace()),
        }
        self._topology = {
            self.name: self._get_buffer_layout_signature(
                installed_payload,
                buffers=prepared,
            ),
        }
        self._render_object_snapshots = {
            self.name: (payload, False, payload.color_source),
        }
        self._dirty_render_object_geometry: set[str] = set()
        self._geometry_upload_center = {self.name: np.zeros(3, dtype=np.float32)}
        self._materials = {self.name: self.material}
        self._material_apply_signatures = {self.name: ("installed",)}
        self._pick_metadata: dict[str, dict[str, object]] = {}
        self.material_calls = 0
        self.transform_calls = 0
        self.visibility_calls = 0
        self.last_transform: Transform | None = None
        self.last_visibility: bool | None = None
        self.metrics: dict[str, float] = {}

    def _record_profile_metric(self, name: str, value: float) -> None:
        self.metrics[name] = self.metrics.get(name, 0.0) + float(value)

    def _record_profile_bytes(self, *_args, **_kwargs) -> None:
        pass

    def set_named_material(self, _name: str, material: MaterialPayload) -> bool:
        self.material_calls += 1
        self._materials[self.name] = material
        self._material_apply_signatures[self.name] = ("updated",)
        return True

    def _apply_render_object_transform(self, obj: RenderObject) -> bool:
        self.transform_calls += 1
        self.last_transform = obj.transform
        return True

    def set_named_visibility(self, _name: str, visible: bool) -> bool:
        self.visibility_calls += 1
        self.last_visibility = visible
        return True


def _mesh(
    *,
    offset: float = 0.0,
    triangles: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    triangle_uvs: np.ndarray | None = None,
    color_source: SurfaceColorSource = SurfaceColorSource.MATERIAL,
) -> MeshPayload:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    vertices[:, 2] += float(offset)
    return MeshPayload(
        vertices=vertices,
        triangles=(
            np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32) if triangles is None else triangles
        ),
        normals=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (4, 1)),
        vertex_colors=colors,
        triangle_uvs=triangle_uvs,
        color_source=color_source,
    )


def test_vertex_stream_updates_only_positions_and_normals_and_commits_snapshot() -> None:
    initial = _mesh()
    desired = _mesh(offset=0.5)
    renderer = _VertexStreamHarness(initial)
    transform = Transform.from_translation([1.0, 2.0, 3.0])
    snapshot = RenderObject(
        id=renderer.name,
        payload=desired,
        material=renderer.material,
        transform=transform,
        visibility=False,
        metadata={"step": 1},
    )

    assert renderer.update_mesh_vertex_stream(snapshot) is True

    geometry = renderer._objects[renderer.name].geometry
    np.testing.assert_array_equal(geometry.positions.data, desired.vertices)
    np.testing.assert_array_equal(geometry.normals.data, desired.normals)
    assert geometry.positions.update_full_calls == 1
    assert geometry.normals.update_full_calls == 1
    assert geometry.indices.update_full_calls == 0
    assert renderer._render_object_snapshots[renderer.name][0] is desired
    assert renderer.name not in renderer._dirty_render_object_geometry
    assert renderer.transform_calls == 1
    assert renderer.visibility_calls == 1
    assert renderer._pick_metadata[renderer.name] == {"step": 1}
    np.testing.assert_array_equal(
        renderer._geometry_upload_center[renderer.name],
        np.zeros(3, dtype=np.float32),
    )


def test_unchanged_payload_falls_back_to_component_only_state_sync() -> None:
    payload = _mesh()
    renderer = _VertexStreamHarness(payload)
    updated_material = MaterialPayload(base_color=(0.8, 0.2, 0.1, 1.0))
    updated_transform = Transform.from_translation([4.0, 5.0, 6.0])
    snapshot = RenderObject(
        id=renderer.name,
        payload=payload,
        material=updated_material,
        transform=updated_transform,
        visibility=False,
        metadata={"step": 2},
    )

    assert renderer.update_mesh_vertex_stream(snapshot) is False
    assert renderer.ensure_object(snapshot) is True

    geometry = renderer._objects[renderer.name].geometry
    assert geometry.positions.update_full_calls == 0
    assert geometry.normals.update_full_calls == 0
    assert geometry.indices.update_full_calls == 0
    assert renderer._materials[renderer.name] == updated_material
    assert renderer.material_calls == 1
    assert renderer.last_transform == updated_transform
    assert renderer.last_visibility is False
    assert renderer._pick_metadata[renderer.name] == {"step": 2}


def test_vertex_stream_rejects_same_shape_changed_connectivity_before_mutation() -> None:
    initial = _mesh()
    changed = _mesh(
        offset=0.5,
        triangles=np.asarray([[0, 1, 3], [0, 3, 2]], dtype=np.int32),
    )
    renderer = _VertexStreamHarness(initial)
    original_positions = renderer._objects[renderer.name].geometry.positions.data.copy()

    assert (
        renderer.update_mesh_vertex_stream(
            RenderObject(id=renderer.name, payload=changed, material=renderer.material)
        )
        is False
    )

    np.testing.assert_array_equal(
        renderer._objects[renderer.name].geometry.positions.data,
        original_positions,
    )
    assert renderer.name in renderer._dirty_render_object_geometry
    assert renderer._render_object_snapshots[renderer.name][0] is initial


def test_vertex_stream_cached_incompatible_transition_skips_preparation(monkeypatch) -> None:
    initial = _mesh()
    changed = _mesh(
        offset=0.5,
        triangles=np.asarray([[0, 1, 3], [0, 3, 2]], dtype=np.int32),
    )
    renderer = _VertexStreamHarness(initial)
    snapshot = RenderObject(
        id=renderer.name,
        payload=changed,
        material=renderer.material,
    )

    assert renderer.update_mesh_vertex_stream(snapshot) is False
    assert renderer.metrics["pygfx_mesh_vertex_stream_incompatible_transition_learn_count"] == 1.0
    renderer._dirty_render_object_geometry.clear()

    def fail_preparation(*_args, **_kwargs):
        raise AssertionError("known-incompatible transition must reject before preparation")

    monkeypatch.setattr(renderer, "_prepare_geometry_buffers", fail_preparation)

    assert renderer.update_mesh_vertex_stream(snapshot) is False
    assert renderer.metrics["pygfx_mesh_vertex_stream_reject_cached_incompatible_count"] == 1.0
    assert renderer.name not in renderer._dirty_render_object_geometry


def test_vertex_stream_reloaded_topology_revision_is_verified_again() -> None:
    initial = _mesh()
    changed_triangles = np.asarray([[0, 1, 3], [0, 3, 2]], dtype=np.int32)
    first_revision = _mesh(offset=0.5, triangles=changed_triangles)
    renderer = _VertexStreamHarness(initial)

    assert (
        renderer.update_mesh_vertex_stream(
            RenderObject(
                id=renderer.name,
                payload=first_revision,
                material=renderer.material,
            )
        )
        is False
    )
    renderer._dirty_render_object_geometry.clear()

    # Equal bytes loaded into a new immutable array represent a new asset
    # revision. It must take the exact verification path once instead of
    # inheriting a negative result through an ID/hash collision.
    reloaded_revision = _mesh(
        offset=0.75,
        triangles=changed_triangles.copy(),
    )
    assert reloaded_revision.triangles is not first_revision.triangles
    assert (
        renderer.update_mesh_vertex_stream(
            RenderObject(
                id=renderer.name,
                payload=reloaded_revision,
                material=renderer.material,
            )
        )
        is False
    )

    assert (
        renderer.metrics.get(
            "pygfx_mesh_vertex_stream_reject_cached_incompatible_count",
            0.0,
        )
        == 0.0
    )
    assert renderer.metrics["pygfx_mesh_vertex_stream_incompatible_transition_learn_count"] == 2.0
    assert renderer.metrics["pygfx_mesh_vertex_stream_reject_changed_indices_count"] == 2.0


def test_vertex_stream_does_not_cache_unreadable_native_topology() -> None:
    initial = _mesh()
    changed = _mesh(
        offset=0.5,
        triangles=np.asarray([[0, 1, 3], [0, 3, 2]], dtype=np.int32),
    )
    renderer = _VertexStreamHarness(initial)
    renderer._objects[renderer.name].geometry.indices = SimpleNamespace()
    snapshot = RenderObject(
        id=renderer.name,
        payload=changed,
        material=renderer.material,
    )

    assert renderer.update_mesh_vertex_stream(snapshot) is False

    assert (
        renderer.metrics.get(
            "pygfx_mesh_vertex_stream_incompatible_transition_learn_count",
            0.0,
        )
        == 0.0
    )
    assert renderer.name not in getattr(
        renderer,
        "_vertex_stream_incompatible_transitions",
        {},
    )


def test_vertex_stream_negative_cache_is_bounded_and_removed_with_object(monkeypatch) -> None:
    monkeypatch.setattr(
        pygfx_geometry_module,
        "_VERTEX_STREAM_INCOMPATIBLE_TRANSITION_LIMIT",
        2,
    )
    renderer = _VertexStreamHarness(_mesh())
    renderer._remember_incompatible_mesh_transition(renderer.name, ((1,), (2,)))
    renderer._remember_incompatible_mesh_transition(renderer.name, ((2,), (3,)))
    renderer._remember_incompatible_mesh_transition(renderer.name, ((3,), (4,)))

    transitions = renderer._vertex_stream_incompatible_transitions[renderer.name]
    assert list(transitions) == [((2,), (3,)), ((3,), (4,))]

    renderer._name_to_handle.clear()
    assert renderer.remove_object(renderer.name) is True
    assert renderer.name not in renderer._vertex_stream_incompatible_transitions


def test_vertex_stream_array_identity_cache_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(
        pygfx_geometry_module,
        "_VERTEX_STREAM_ARRAY_TOKEN_CACHE_LIMIT",
        2,
    )
    renderer = _VertexStreamHarness(_mesh())
    arrays = [np.asarray([index], dtype=np.int32) for index in range(3)]

    tokens = [renderer._immutable_array_token(array) for array in arrays]

    assert all(token is not None for token in tokens)
    assert len(set(tokens)) == 3
    assert len(renderer._vertex_stream_array_tokens) == 2


def test_vertex_stream_updates_changed_vertex_color_buffer() -> None:
    initial_colors = np.zeros((4, 3), dtype=np.float32)
    desired_colors = np.ones((4, 3), dtype=np.float32)
    initial = _mesh(
        colors=initial_colors,
        color_source=SurfaceColorSource.VERTEX,
    )
    desired = _mesh(
        offset=0.5,
        colors=desired_colors,
        color_source=SurfaceColorSource.VERTEX,
    )
    renderer = _VertexStreamHarness(initial)

    assert (
        renderer.update_mesh_vertex_stream(
            RenderObject(id=renderer.name, payload=desired, material=renderer.material)
        )
        is True
    )

    geometry = renderer._objects[renderer.name].geometry
    assert geometry.positions.update_full_calls == 1
    assert geometry.colors.update_full_calls == 1
    assert geometry.indices.update_full_calls == 0
    np.testing.assert_array_equal(geometry.colors.data, desired_colors)


def test_vertex_stream_ignores_loader_colors_in_material_color_mode() -> None:
    initial = _mesh(colors=np.zeros((4, 3), dtype=np.float32))
    desired = _mesh(
        offset=0.5,
        colors=np.ones((4, 3), dtype=np.float32),
    )
    renderer = _VertexStreamHarness(initial)

    assert (
        renderer.update_mesh_vertex_stream(
            RenderObject(id=renderer.name, payload=desired, material=renderer.material)
        )
        is True
    )

    geometry = renderer._objects[renderer.name].geometry
    assert not hasattr(geometry, "colors")
    assert geometry.positions.update_full_calls == 1


def test_vertex_stream_accepts_identical_face_corner_uv_expansion() -> None:
    triangle_uvs = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.2, 0.3],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    initial = _mesh(triangle_uvs=triangle_uvs)
    desired = _mesh(offset=0.5, triangle_uvs=triangle_uvs)
    renderer = _VertexStreamHarness(initial)

    assert (
        renderer.update_mesh_vertex_stream(
            RenderObject(id=renderer.name, payload=desired, material=renderer.material)
        )
        is True
    )

    geometry = renderer._objects[renderer.name].geometry
    assert len(geometry.positions.data) > len(initial.vertices)
    assert geometry.positions.update_full_calls == 1
    assert geometry.indices.update_full_calls == 0
    assert geometry.texcoords.update_full_calls == 0


def test_vertex_stream_rejects_changed_face_corner_uvs_before_mutation() -> None:
    initial_uvs = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.2, 0.3],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    desired_uvs = initial_uvs.copy()
    desired_uvs[3] = [0.4, 0.6]
    initial = _mesh(triangle_uvs=initial_uvs)
    desired = _mesh(offset=0.5, triangle_uvs=desired_uvs)
    renderer = _VertexStreamHarness(initial)

    assert (
        renderer.update_mesh_vertex_stream(
            RenderObject(id=renderer.name, payload=desired, material=renderer.material)
        )
        is False
    )

    geometry = renderer._objects[renderer.name].geometry
    assert geometry.positions.update_full_calls == 0
    assert geometry.indices.update_full_calls == 0
    assert geometry.texcoords.update_full_calls == 0
    assert renderer.name in renderer._dirty_render_object_geometry


def test_vertex_stream_failure_stays_dirty_and_does_not_advance_snapshot() -> None:
    initial = _mesh()
    desired = _mesh(offset=0.5)
    renderer = _VertexStreamHarness(initial)
    renderer._objects[renderer.name].geometry.positions.fail = True

    assert (
        renderer.update_mesh_vertex_stream(
            RenderObject(id=renderer.name, payload=desired, material=renderer.material)
        )
        is False
    )

    assert renderer.name in renderer._dirty_render_object_geometry
    assert renderer._render_object_snapshots[renderer.name][0] is initial
