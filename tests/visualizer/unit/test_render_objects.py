"""Tests for renderer-neutral visualizer model objects."""

from __future__ import annotations

import numpy as np
import pytest

from visualizer.src.model import (
    MaterialState,
    NodeMarker,
    RenderObject,
    RenderObjectState,
    TrajectoryPreview,
    Transform,
    make_text_label_state,
    offset_render_state_transform,
    render_state_center,
    render_state_colors,
    render_state_points,
    render_state_triangles,
    replace_render_state_payload,
    set_render_state_colors,
    set_render_state_points,
    set_render_state_triangles,
    tint_render_state_payload,
)
from visualizer.src.types.render_payloads import MaterialPayload, MeshPayload, TextLabelPayload


def _mesh_payload(*, cache_key: str | None = None) -> MeshPayload:
    return MeshPayload(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
        cache_key=cache_key,
    )


def test_transform_normalizes_and_exposes_translation() -> None:
    transform = Transform.from_translation([1, 2, 3])

    assert transform.matrix.shape == (4, 4)
    np.testing.assert_allclose(transform.translation, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        transform.matrix[0, 0] = 2.0


def test_transform_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="4x4"):
        Transform(np.eye(3))

    with pytest.raises(ValueError, match="3 values"):
        Transform.from_translation([1.0, 2.0])


def test_render_object_wraps_state_under_stable_id() -> None:
    payload = _mesh_payload()
    material = MaterialPayload(base_color=(0.1, 0.2, 0.3, 1.0))
    matrix = np.eye(4)
    matrix[:3, 3] = [4.0, 5.0, 6.0]

    render_object = RenderObject(
        id="  node:tx_0::marker  ",
        payload=payload,
        material=material,
        transform=matrix,
        visibility=False,
        metadata={"domain": "node"},
    )

    assert render_object.id == "node:tx_0::marker"
    assert render_object.payload is payload
    assert isinstance(render_object.material, MaterialState)
    assert render_object.material_payload == material
    assert render_object.visible is False
    np.testing.assert_allclose(render_object.transform_matrix[:3, 3], [4.0, 5.0, 6.0])
    with pytest.raises(TypeError):
        render_object.metadata["domain"] = "target"  # type: ignore[index]


def test_render_object_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        RenderObject(id=" ", payload=_mesh_payload())


def test_render_object_state_tracks_transform_color_and_snapshot() -> None:
    state = RenderObjectState(
        id="node:tx_0::marker",
        payload=_mesh_payload(),
        material=MaterialPayload(base_color=(1.0, 0.0, 0.0, 1.0)),
    )

    assert not hasattr(state, "get_center")
    assert not hasattr(state, "paint_uniform_color")
    assert not hasattr(state, "vertices")
    np.testing.assert_allclose(render_state_center(state), [1.0 / 3.0, 1.0 / 3.0, 0.0])
    offset_render_state_transform(state, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(render_state_center(state), [7.0 / 3.0, 10.0 / 3.0, 4.0])

    tint_render_state_payload(state, [0.2, 0.4, 0.6])
    assert state.material.base_color == (0.2, 0.4, 0.6, 1.0)
    colors = render_state_colors(state)
    assert colors is not None
    np.testing.assert_allclose(colors[0], [0.2, 0.4, 0.6])

    state.visible = False
    snapshot = state.to_render_object()
    assert snapshot.id == "node:tx_0::marker"
    assert snapshot.payload is state.payload
    assert snapshot.visible is False
    np.testing.assert_allclose(snapshot.transform_matrix, state.world_transform.matrix)


def test_render_object_state_payload_replacement_uses_new_payload_identity() -> None:
    state = RenderObjectState(id="scene:building", payload=_mesh_payload())
    replacement = _mesh_payload()

    replace_render_state_payload(state, replacement)

    assert state.payload is replacement
    assert state.to_render_object(effective_visible=False).visible is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda state: set_render_state_points(
            state,
            np.asarray(state.payload.vertices) + 1.0,
        ),
        lambda state: set_render_state_triangles(
            state,
            np.asarray([[0, 2, 1]], dtype=np.int32),
        ),
        lambda state: set_render_state_colors(
            state,
            np.asarray([[0.2, 0.3, 0.4]] * 3, dtype=float),
        ),
        lambda state: tint_render_state_payload(state, [0.2, 0.4, 0.6]),
    ),
    ids=("vertices", "triangles", "colors", "tint"),
)
def test_mesh_payload_mutations_clear_prepared_buffer_cache_key(mutation) -> None:
    state = RenderObjectState(
        id="target:person::mesh",
        payload=_mesh_payload(cache_key="targets/person/frame_0001.ply"),
    )

    mutation(state)

    assert isinstance(state.payload, MeshPayload)
    assert state.payload.cache_key is None


def test_text_label_state_uses_the_common_render_object_contract() -> None:
    label = make_text_label_state(
        "node:tx_0::label",
        "TX1",
        [1.0, 0.25, 0.0],
        font_size=0.4,
        position=[2.0, 3.0, 4.0],
        visible=False,
    )

    assert isinstance(label.payload, TextLabelPayload)
    assert label.payload.text == "TX1"
    assert label.payload.font_size == pytest.approx(0.4)
    assert label.material.base_color == (1.0, 0.25, 0.0, 1.0)
    np.testing.assert_allclose(label.world_transform.translation, [2.0, 3.0, 4.0])
    assert label.to_render_object().visible is False


def test_render_state_array_access_is_read_only() -> None:
    payload = _mesh_payload()
    state = RenderObjectState(id="scene:building", payload=payload)

    with pytest.raises(ValueError):
        render_state_points(state)[0, 0] = 2.0
    with pytest.raises(ValueError):
        render_state_triangles(state)[0, 0] = 2

    state.replace_payload(
        MeshPayload(
            vertices=payload.vertices,
            triangles=payload.triangles,
            vertex_colors=np.ones((3, 3), dtype=float),
        )
    )
    colors = render_state_colors(state)
    assert colors is not None
    with pytest.raises(ValueError):
        colors[0, 0] = 0.5


def test_render_object_state_hashes_by_stable_id() -> None:
    first = RenderObjectState(id="node:rx_0::marker", payload=_mesh_payload())
    second = RenderObjectState(id="node:rx_0::marker", payload=_mesh_payload())

    assert first == second
    assert {first, second} == {first}


def test_trajectory_preview_freezes_points() -> None:
    preview = TrajectoryPreview(
        object_id="trajectory:tx_0",
        render_id="trajectory:tx_0::line",
        kind="TX",
        points=np.array([[0, 0, 0], [1, 1, 1]], dtype=float),
    )

    assert preview.kind == "tx"
    np.testing.assert_allclose(preview.points[1], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        preview.points[0, 0] = 10.0


def test_trajectory_preview_rejects_non_xyz_points() -> None:
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        TrajectoryPreview(
            object_id="trajectory:tx_0",
            render_id="trajectory:tx_0::line",
            kind="tx",
            points=np.array([0.0, 1.0, 2.0]),
        )


def test_node_marker_normalizes_kind_and_metadata() -> None:
    marker = NodeMarker(
        object_id="node:rx_1",
        render_id="node:rx_1::marker",
        kind="RX",
        index="1",  # type: ignore[arg-type]
        metadata={"source": "frame"},
    )

    assert marker.kind == "rx"
    assert marker.index == 1
    with pytest.raises(TypeError):
        marker.metadata["source"] = "ui"  # type: ignore[index]
