"""Tests for geometry utility functions."""

from typing import Any, Dict

import numpy as np

from visualizer.src.types.render_payloads import MeshPayload
from visualizer.src.utils.geometry import (
    apply_transform_to_payload,
    build_rotation_matrix_from_angles,
    build_transform_matrix,
    compute_position_after_scale_rotation,
    mesh_center,
    mesh_vertices,
)

# =============================================================================
# Rotation Matrix Tests
# =============================================================================


def test_rotation_matrix_identity_for_zero_angles() -> None:
    """Verify zero rotation returns identity matrix."""
    result = build_rotation_matrix_from_angles((0.0, 0.0, 0.0))

    np.testing.assert_allclose(result, np.eye(3), atol=1e-10)


def test_rotation_matrix_identity_for_none() -> None:
    """Verify None rotation returns identity matrix."""
    result = build_rotation_matrix_from_angles(None)

    np.testing.assert_allclose(result, np.eye(3), atol=1e-10)


def test_rotation_matrix_90_degree_yaw() -> None:
    """Verify 90 degree yaw (Z-axis) rotation."""
    result = build_rotation_matrix_from_angles((90.0, 0.0, 0.0))

    # Point (1, 0, 0) should rotate to (0, 1, 0)
    point = np.array([1.0, 0.0, 0.0])
    rotated = result @ point
    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-10)


def test_rotation_matrix_90_degree_pitch() -> None:
    """Verify 90 degree pitch (Y-axis) rotation."""
    result = build_rotation_matrix_from_angles((0.0, 90.0, 0.0))

    # Point (1, 0, 0) should rotate to (0, 0, -1)
    point = np.array([1.0, 0.0, 0.0])
    rotated = result @ point
    np.testing.assert_allclose(rotated, [0.0, 0.0, -1.0], atol=1e-10)


def test_rotation_matrix_90_degree_roll() -> None:
    """Verify 90 degree roll (X-axis) rotation."""
    result = build_rotation_matrix_from_angles((0.0, 0.0, 90.0))

    # Point (0, 1, 0) should rotate to (0, 0, 1)
    point = np.array([0.0, 1.0, 0.0])
    rotated = result @ point
    np.testing.assert_allclose(rotated, [0.0, 0.0, 1.0], atol=1e-10)


def test_rotation_matrix_is_orthogonal() -> None:
    """Verify rotation matrix is orthogonal (R @ R.T = I)."""
    result = build_rotation_matrix_from_angles((30.0, 45.0, 60.0))

    identity = result @ result.T
    np.testing.assert_allclose(identity, np.eye(3), atol=1e-10)


def test_rotation_matrix_determinant_is_one() -> None:
    """Verify rotation matrix has determinant 1 (proper rotation)."""
    result = build_rotation_matrix_from_angles((30.0, 45.0, 60.0))

    det = np.linalg.det(result)
    np.testing.assert_allclose(det, 1.0, atol=1e-10)


# =============================================================================
# Transform Matrix Tests
# =============================================================================


def test_transform_matrix_identity_for_empty_state() -> None:
    """Verify empty transform state returns identity matrix."""
    result = build_transform_matrix({})

    np.testing.assert_allclose(result, np.eye(4), atol=1e-10)


def test_transform_matrix_scale_only() -> None:
    """Verify scale-only transform."""
    result = build_transform_matrix({"scale": 2.0})

    expected = np.diag([2.0, 2.0, 2.0, 1.0])
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_transform_matrix_translation_only() -> None:
    """Verify translation-only transform."""
    result = build_transform_matrix({"translate": (1.0, 2.0, 3.0)})

    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_transform_matrix_rotation_only() -> None:
    """Verify rotation-only transform."""
    result = build_transform_matrix({"rotation": (90.0, 0.0, 0.0)})

    # Point (1, 0, 0, 1) should rotate to (0, 1, 0, 1)
    point = np.array([1.0, 0.0, 0.0, 1.0])
    transformed = result @ point
    np.testing.assert_allclose(transformed[:3], [0.0, 1.0, 0.0], atol=1e-10)


def test_transform_matrix_combined() -> None:
    """Verify combined scale, rotation, and translation."""
    state = {
        "scale": 2.0,
        "rotation": (0.0, 0.0, 0.0),
        "translate": (10.0, 0.0, 0.0),
    }
    result = build_transform_matrix(state)

    # Point at origin should be scaled then translated
    point = np.array([1.0, 0.0, 0.0, 1.0])
    transformed = result @ point
    # First scale: (1, 0, 0) -> (2, 0, 0)
    # Then translate: (2, 0, 0) -> (12, 0, 0)
    np.testing.assert_allclose(transformed[:3], [12.0, 0.0, 0.0], atol=1e-10)


def test_transform_matrix_is_4x4() -> None:
    """Verify transform matrix is always 4x4."""
    result = build_transform_matrix({"scale": 1.5, "rotation": (10.0, 20.0, 30.0)})

    assert result.shape == (4, 4)


# =============================================================================
# Position After Scale Rotation Tests
# =============================================================================


def test_compute_position_identity_transform() -> None:
    """Verify identity transform returns original position."""
    center = np.array([1.0, 2.0, 3.0])
    state: Dict[str, Any] = {"scale": 1.0, "rotation": (0.0, 0.0, 0.0)}

    result = compute_position_after_scale_rotation(center, state)

    np.testing.assert_allclose(result, center, atol=1e-10)


def test_compute_position_scale_only() -> None:
    """Verify scale-only transform."""
    center = np.array([1.0, 2.0, 3.0])
    state: Dict[str, Any] = {"scale": 2.0, "rotation": (0.0, 0.0, 0.0)}

    result = compute_position_after_scale_rotation(center, state)

    np.testing.assert_allclose(result, [2.0, 4.0, 6.0], atol=1e-10)


def test_compute_position_rotation_only() -> None:
    """Verify rotation-only transform."""
    center = np.array([1.0, 0.0, 0.0])
    state: Dict[str, Any] = {"scale": 1.0, "rotation": (90.0, 0.0, 0.0)}

    result = compute_position_after_scale_rotation(center, state)

    np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-10)


def test_compute_position_scale_and_rotation() -> None:
    """Verify combined scale and rotation."""
    center = np.array([1.0, 0.0, 0.0])
    state: Dict[str, Any] = {"scale": 2.0, "rotation": (90.0, 0.0, 0.0)}

    result = compute_position_after_scale_rotation(center, state)

    # First scale: (1, 0, 0) -> (2, 0, 0)
    # Then rotate 90 deg around Z: (2, 0, 0) -> (0, 2, 0)
    np.testing.assert_allclose(result, [0.0, 2.0, 0.0], atol=1e-10)


def test_compute_position_empty_state() -> None:
    """Verify empty state returns original position."""
    center = np.array([1.0, 2.0, 3.0])
    state: Dict[str, Any] = {}

    result = compute_position_after_scale_rotation(center, state)

    np.testing.assert_allclose(result, center, atol=1e-10)


def test_apply_transform_to_payload_replaces_vertices_and_reports_centers() -> None:
    payload = MeshPayload(
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
        triangles=np.asarray([[0, 1, 1]], dtype=np.int32),
        normals=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
    )
    original_vertices = np.asarray(payload.vertices, dtype=float).copy()
    original_center = mesh_center(payload)

    updated, after_scale_rotation, final_center = apply_transform_to_payload(
        payload,
        original_vertices,
        original_center,
        {"scale": 2.0, "rotation": [90.0, 0.0, 0.0], "translate": [1.0, 0.0, 0.0]},
    )

    np.testing.assert_allclose(after_scale_rotation, [0.0, 1.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(final_center, [1.0, 1.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(updated.vertices[1], [1.0, 2.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(updated.normals[0], [0.0, 1.0, 0.0], atol=1e-10)


def test_mesh_helpers_ignore_native_like_mesh_fallbacks() -> None:
    class NativeLikeMesh:
        vertices = [[9.0, 9.0, 9.0]]

        def get_center(self):
            raise AssertionError("native get_center should not be used")

    native = NativeLikeMesh()

    assert mesh_vertices(native) is None
    np.testing.assert_allclose(mesh_center(native), [0.0, 0.0, 0.0])
