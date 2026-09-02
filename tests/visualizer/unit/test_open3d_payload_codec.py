"""Tests for immutable payload conversion at the Open3D foreign boundary."""

from __future__ import annotations

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from visualizer.src.backends.open3d_payload_codec import (
    lines_payload_to_o3d,
    mesh_payload_to_o3d,
    o3d_mesh_to_payload,
    points_payload_to_o3d,
)
from visualizer.src.types.render_payloads import (
    LineSetPayload,
    MeshPayload,
    PointCloudPayload,
)


def test_mesh_conversion_accepts_read_only_payload_and_detaches_native_buffers() -> None:
    payload = MeshPayload(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        normals=np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=np.float32),
        vertex_colors=np.asarray(
            [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.5]],
            dtype=np.float32,
        ),
        triangle_uvs=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    mesh = mesh_payload_to_o3d(payload)

    assert payload.vertices.flags.writeable is False
    assert payload.triangles.flags.writeable is False
    np.testing.assert_allclose(np.asarray(mesh.vertices), payload.vertices)
    np.testing.assert_array_equal(np.asarray(mesh.triangles), payload.triangles)
    np.testing.assert_allclose(np.asarray(mesh.vertex_normals), payload.normals)
    np.testing.assert_allclose(np.asarray(mesh.vertex_colors), payload.vertex_colors[:, :3])
    np.testing.assert_allclose(np.asarray(mesh.triangle_uvs), payload.triangle_uvs)

    np.asarray(mesh.vertices)[0, 0] = 99.0
    assert payload.vertices[0, 0] == pytest.approx(0.0)
    assert payload.vertices.flags.writeable is False


@pytest.mark.parametrize(
    ("payload", "converter", "native_attribute", "payload_attribute"),
    [
        (
            LineSetPayload(
                points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
                lines=np.asarray([[0, 1]], dtype=np.int32),
                colors=np.asarray([[0.2, 0.4, 0.6, 0.8]], dtype=np.float32),
            ),
            lines_payload_to_o3d,
            "points",
            "points",
        ),
        (
            PointCloudPayload(
                points=np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32),
                colors=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            ),
            points_payload_to_o3d,
            "points",
            "points",
        ),
    ],
)
def test_line_and_point_conversion_accept_read_only_payloads(
    payload,
    converter,
    native_attribute: str,
    payload_attribute: str,
) -> None:
    source = getattr(payload, payload_attribute)
    assert source.flags.writeable is False

    native = converter(payload)
    native_array = np.asarray(getattr(native, native_attribute))
    np.testing.assert_allclose(native_array, source)

    native_array[0, 0] = 99.0
    assert source[0, 0] == pytest.approx(0.0)
    assert source.flags.writeable is False


def test_open3d_to_payload_conversion_detaches_immutable_snapshot() -> None:
    mesh = o3d.geometry.TriangleMesh.create_box()

    payload = o3d_mesh_to_payload(mesh)
    np.asarray(mesh.vertices)[0, 0] = 99.0

    assert payload.vertices[0, 0] != pytest.approx(99.0)
    assert payload.vertices.flags.writeable is False
