"""Tests for renderer-neutral geometry payload construction."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from visualizer.src.scene.geometry_payload_factory import (
    extract_wireframe_payload,
    load_mesh_payload,
    make_box_payload,
    make_lines_payload,
    make_pointcloud_payload,
    make_sphere_payload,
    merge_mesh_payloads,
)


def test_sphere_payload_is_backend_neutral() -> None:
    payload = make_sphere_payload(radius=0.5, color=[1.0, 0.0, 0.0], resolution=6)

    assert payload.vertices.shape[1] == 3
    assert payload.triangles.shape[1] == 3
    assert payload.normals is not None
    assert payload.vertex_colors is not None
    assert payload.vertex_colors.shape == payload.vertices.shape


def test_box_payload_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_box_payload(1.0, 0.0, 1.0)


def test_lines_and_pointcloud_payloads_normalize_arrays() -> None:
    lines = make_lines_payload(
        points=[[0, 0, 0], [1, 0, 0]],
        lines=[[0, 1]],
        colors=[[0.2, 0.3, 0.4]],
    )
    points = make_pointcloud_payload(points=[[1, 2, 3]], colors=[[1, 0, 0]])

    assert lines.points.dtype.kind == "f"
    assert lines.lines.dtype == np.int32
    assert points.points.shape == (1, 3)


def test_extract_wireframe_payload_deduplicates_edges() -> None:
    mesh = make_box_payload(1.0, 1.0, 1.0)
    wireframe = extract_wireframe_payload(mesh)

    assert wireframe.points.shape == mesh.vertices.shape
    assert wireframe.lines.shape[1] == 2
    assert len(wireframe.lines) < len(mesh.triangles) * 3


def test_merge_mesh_payloads_offsets_triangles() -> None:
    first = make_box_payload(1.0, 1.0, 1.0, color=[1, 0, 0])
    second = make_box_payload(2.0, 1.0, 1.0, color=[0, 1, 0])

    merged = merge_mesh_payloads([first, second])

    assert len(merged.vertices) == len(first.vertices) + len(second.vertices)
    assert int(merged.triangles[len(first.triangles), 0]) >= len(first.vertices)
    assert merged.vertex_colors is not None


def test_load_obj_payload(tmp_path) -> None:
    obj_path = tmp_path / "triangle.obj"
    obj_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "vn 0 0 1",
                "f 1//1 2//1 3//1",
            ]
        ),
        encoding="utf-8",
    )

    payload = load_mesh_payload(obj_path)

    assert payload.vertices.shape == (3, 3)
    assert payload.triangles.tolist() == [[0, 1, 2]]
    assert payload.normals is not None
    assert payload.cache_key == str(obj_path)


def test_load_obj_payload_preserves_face_corner_uvs(tmp_path) -> None:
    obj_path = tmp_path / "textured_quad.obj"
    obj_path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 1 1 0",
                "v 0 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 1 1",
                "vt 0 1",
                "f 1/1 2/2 3/3 4/4",
            ]
        ),
        encoding="utf-8",
    )

    payload = load_mesh_payload(obj_path)

    assert payload.triangles.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert payload.triangle_uvs is not None
    assert payload.triangle_uvs.shape == (6, 2)
    np.testing.assert_allclose(payload.triangle_uvs[0], [0.0, 0.0])
    np.testing.assert_allclose(payload.triangle_uvs[-1], [0.0, 1.0])


def test_load_ascii_ply_payload(tmp_path) -> None:
    ply_path = tmp_path / "triangle.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "0 0 0 255 0 0",
                "1 0 0 0 255 0",
                "0 1 0 0 0 255",
                "3 0 1 2",
            ]
        ),
        encoding="utf-8",
    )

    payload = load_mesh_payload(ply_path)

    assert payload.vertices.shape == (3, 3)
    assert payload.triangles.tolist() == [[0, 1, 2]]
    assert payload.vertex_colors is not None
    np.testing.assert_allclose(payload.vertex_colors[0], [1.0, 0.0, 0.0])


def test_load_binary_little_endian_ply_payload(tmp_path) -> None:
    ply_path = tmp_path / "triangle_binary.ply"
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 3",
            "property double x",
            "property double y",
            "property double z",
            "property double nx",
            "property double ny",
            "property double nz",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "element face 1",
            "property list uchar uint vertex_indices",
            "end_header",
            "",
        ]
    ).encode("ascii")
    vertices = b"".join(
        struct.pack("<ddddddBBB", *values)
        for values in [
            (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 255, 0, 0),
            (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 255, 0),
            (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0, 0, 255),
        ]
    )
    face = struct.pack("<BIII", 3, 0, 1, 2)
    ply_path.write_bytes(header + vertices + face)

    payload = load_mesh_payload(ply_path)

    assert payload.vertices.shape == (3, 3)
    assert payload.triangles.tolist() == [[0, 1, 2]]
    assert payload.normals is not None
    np.testing.assert_allclose(payload.normals[0], [0.0, 0.0, 1.0])
    assert payload.vertex_colors is not None
    np.testing.assert_allclose(payload.vertex_colors[1], [0.0, 1.0, 0.0])
