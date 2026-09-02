from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.scene.surface_payloads import (
    BeamformingSurface,
    build_coverage_payloads,
    compute_vertex_normals,
)
from visualizer.src.types.render_payloads import MeshPayload


def test_compute_vertex_normals_accumulates_triangle_normals() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2]], dtype=np.int32)

    normals = compute_vertex_normals(vertices, triangles)

    np.testing.assert_allclose(normals, [[0.0, 0.0, 1.0]] * 3)


def test_build_coverage_payloads_returns_mesh_and_isolines() -> None:
    vm = SimpleNamespace(
        show_coverage=True,
        coverage_vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        coverage_triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        coverage_colors=np.asarray([[1.0, 0.0, 0.0]] * 3, dtype=np.float32),
        coverage_isoline_points=np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        coverage_isoline_lines=np.asarray([[0, 1]], dtype=np.int32),
        coverage_isoline_colors=np.asarray([[0.2, 0.2, 0.2]], dtype=np.float32),
    )

    payloads = build_coverage_payloads(vm)

    assert payloads.mesh is not None
    assert payloads.isolines is not None
    np.testing.assert_allclose(payloads.mesh.normals, [[0.0, 0.0, 1.0]] * 3)
    np.testing.assert_allclose(payloads.isolines.colors, [[0.2, 0.2, 0.2]])


def test_build_coverage_payloads_empty_when_hidden() -> None:
    payloads = build_coverage_payloads(
        SimpleNamespace(show_coverage=False, coverage_vertices=np.zeros((3, 3)))
    )

    assert payloads.mesh is None
    assert payloads.isolines is None


def test_beamforming_surface_is_frozen_and_owns_final_payload() -> None:
    mesh = MeshPayload(
        vertices=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        normals=np.asarray([[1.0, 1.0, 1.0]] * 3, dtype=np.float32),
        vertex_colors=np.asarray([[0.2, 0.4, 0.8]] * 3, dtype=np.float32),
    )
    surface = BeamformingSurface(id="beamforming:tx_0:mesh", payload=mesh)

    assert surface.payload is mesh
    assert mesh.vertices.flags.writeable is False
    with pytest.raises(FrozenInstanceError):
        surface.id = "changed"
    with pytest.raises(ValueError):
        mesh.vertices[0, 0] = 10.0
