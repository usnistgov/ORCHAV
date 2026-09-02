"""Tests for aperture geometry utilities including device orientation support."""

from __future__ import annotations

import numpy as np

from visualizer.src.utils.aperture_geometry import (
    angular_reference_label_positions,
    create_angular_reference_line_payload,
    create_aperture_line_payload,
    create_aperture_mesh_payload,
    yaw_pitch_roll_to_rotation_matrix,
)


class TestRotationMatrix:
    """Tests for yaw_pitch_roll_to_rotation_matrix function."""

    def test_identity_for_zero_angles(self):
        """Zero angles should produce identity matrix."""
        R = yaw_pitch_roll_to_rotation_matrix(0.0, 0.0, 0.0)
        expected = np.eye(3)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_90_degree_yaw(self):
        """90-degree yaw should rotate X axis to Y axis."""
        R = yaw_pitch_roll_to_rotation_matrix(np.pi / 2, 0.0, 0.0)
        # Rotate unit X vector
        x = np.array([1.0, 0.0, 0.0])
        result = R @ x
        # Should be roughly [0, 1, 0] (Y axis)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-10)

    def test_90_degree_pitch(self):
        """90-degree pitch should rotate X axis to -Z axis."""
        R = yaw_pitch_roll_to_rotation_matrix(0.0, np.pi / 2, 0.0)
        # Rotate unit X vector
        x = np.array([1.0, 0.0, 0.0])
        result = R @ x
        # Should be roughly [0, 0, -1] (-Z axis)
        np.testing.assert_allclose(result, [0.0, 0.0, -1.0], atol=1e-10)

    def test_rotation_matrix_is_orthogonal(self):
        """Rotation matrix should be orthogonal (R @ R.T = I)."""
        R = yaw_pitch_roll_to_rotation_matrix(0.5, 0.3, 0.2)
        product = R @ R.T
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_rotation_matrix_determinant_is_one(self):
        """Rotation matrix should have determinant of 1."""
        R = yaw_pitch_roll_to_rotation_matrix(0.5, 0.3, 0.2)
        det = np.linalg.det(R)
        np.testing.assert_allclose(det, 1.0, atol=1e-10)


class TestApertureLinePayloadOrientation:
    """Tests for aperture line payloads with device orientation."""

    def test_create_aperture_no_orientation(self):
        """Aperture without orientation should work as before."""
        line_payload = create_aperture_line_payload(
            center=np.array([0.0, 0.0, 0.0]),
            az_min_deg=0.0,
            az_max_deg=90.0,
            el_min_deg=-10.0,
            el_max_deg=10.0,
            radius=1.0,
            color=[1.0, 0.0, 0.0],
            orientation=None,
        )
        assert line_payload is not None
        points = np.asarray(line_payload.points)
        assert len(points) > 0

    def test_create_aperture_with_zero_orientation(self):
        """Aperture with zero orientation should be same as no orientation."""
        line_payload_no_ori = create_aperture_line_payload(
            center=np.array([0.0, 0.0, 0.0]),
            az_min_deg=0.0,
            az_max_deg=90.0,
            el_min_deg=-10.0,
            el_max_deg=10.0,
            radius=1.0,
            color=[1.0, 0.0, 0.0],
            orientation=None,
        )
        line_payload_zero_ori = create_aperture_line_payload(
            center=np.array([0.0, 0.0, 0.0]),
            az_min_deg=0.0,
            az_max_deg=90.0,
            el_min_deg=-10.0,
            el_max_deg=10.0,
            radius=1.0,
            color=[1.0, 0.0, 0.0],
            orientation=(0.0, 0.0, 0.0),
        )

        points_no_ori = np.asarray(line_payload_no_ori.points)
        points_zero_ori = np.asarray(line_payload_zero_ori.points)
        np.testing.assert_allclose(points_no_ori, points_zero_ori, atol=1e-10)

    def test_create_aperture_with_yaw_rotation(self):
        """Aperture with 90-degree yaw should be rotated."""
        # Create aperture at origin pointing toward +X (az=0)
        line_payload_original = create_aperture_line_payload(
            center=np.array([0.0, 0.0, 0.0]),
            az_min_deg=-10.0,
            az_max_deg=10.0,
            el_min_deg=-10.0,
            el_max_deg=10.0,
            radius=1.0,
            color=[1.0, 0.0, 0.0],
            orientation=None,
        )
        # Same aperture but with 90-degree yaw - should point toward +Y
        line_payload_rotated = create_aperture_line_payload(
            center=np.array([0.0, 0.0, 0.0]),
            az_min_deg=-10.0,
            az_max_deg=10.0,
            el_min_deg=-10.0,
            el_max_deg=10.0,
            radius=1.0,
            color=[1.0, 0.0, 0.0],
            orientation=(np.pi / 2, 0.0, 0.0),  # 90-degree yaw
        )

        points_original = np.asarray(line_payload_original.points)
        points_rotated = np.asarray(line_payload_rotated.points)

        # Center should be unchanged
        np.testing.assert_allclose(points_original[0], points_rotated[0], atol=1e-10)

        # Other points should be different (rotated)
        # Specifically, X coordinates should become Y coordinates for 90-degree yaw
        # Check that the apertures are actually different
        assert not np.allclose(points_original[1:], points_rotated[1:], atol=1e-3)

    def test_create_aperture_preserves_center(self):
        """Aperture center should not move when orientation is applied."""
        center = np.array([10.0, 20.0, 5.0])
        line_payload = create_aperture_line_payload(
            center=center,
            az_min_deg=0.0,
            az_max_deg=90.0,
            el_min_deg=-10.0,
            el_max_deg=10.0,
            radius=1.0,
            color=[1.0, 0.0, 0.0],
            orientation=(0.5, 0.3, 0.2),
        )
        points = np.asarray(line_payload.points)
        # First point should be the center
        np.testing.assert_allclose(points[0], center, atol=1e-10)

    def test_create_aperture_empty_when_no_bounds(self):
        """Aperture with no bounds should return an empty line payload."""
        line_payload = create_aperture_line_payload(
            center=np.array([0.0, 0.0, 0.0]),
            az_min_deg=None,
            az_max_deg=None,
            el_min_deg=None,
            el_max_deg=None,
            radius=1.0,
            color=[1.0, 0.0, 0.0],
            orientation=(0.5, 0.0, 0.0),
        )
        assert line_payload is not None
        points = np.asarray(line_payload.points)
        assert len(points) == 0

    def test_create_aperture_with_one_sided_azimuth_bound(self):
        """One-sided azimuth bounds should still produce a preview."""
        line_payload = create_aperture_line_payload(
            center=np.array([0.0, 0.0, 0.0]),
            az_min_deg=-179.0,
            az_max_deg=None,
            el_min_deg=None,
            el_max_deg=None,
            radius=1.0,
            color=[1.0, 0.0, 0.0],
        )
        assert line_payload is not None
        assert len(line_payload.points) > 0
        assert len(line_payload.lines) > 0

    def test_create_aperture_with_one_sided_elevation_bound(self):
        """One-sided elevation bounds should still produce a preview."""
        line_payload = create_aperture_line_payload(
            center=np.array([0.0, 0.0, 0.0]),
            az_min_deg=None,
            az_max_deg=None,
            el_min_deg=None,
            el_max_deg=45.0,
            radius=1.0,
            color=[1.0, 0.0, 0.0],
        )
        assert line_payload is not None
        assert len(line_payload.points) > 0
        assert len(line_payload.lines) > 0

    def test_create_aperture_full_range_draws_wireframe(self):
        """A full angular range is a valid aperture preview."""
        line_payload = create_aperture_line_payload(
            center=np.array([0.0, 0.0, 0.0]),
            az_min_deg=-180.0,
            az_max_deg=180.0,
            el_min_deg=-90.0,
            el_max_deg=90.0,
            radius=1.0,
            color=[1.0, 0.0, 0.0],
        )
        assert line_payload is not None
        assert len(line_payload.points) > 0
        assert len(line_payload.lines) > 0

    def test_aperture_outline_connects_partial_sector_to_center(self):
        """A bounded aperture outline should visibly originate at the device."""
        line_payload = create_aperture_line_payload(
            center=np.array([0.0, 0.0, 0.0]),
            az_min_deg=-20.0,
            az_max_deg=20.0,
            el_min_deg=-15.0,
            el_max_deg=15.0,
            radius=1.0,
            color=[1.0, 0.0, 0.0],
        )
        assert line_payload is not None
        lines = np.asarray(line_payload.lines)
        center_ray_count = int(np.sum(np.any(lines == 0, axis=1)))
        assert center_ray_count == 4

    def test_aperture_mesh_payload_represents_accepted_patch(self):
        """The pygfx aperture mesh should contain a filled accepted angular patch."""
        payload = create_aperture_mesh_payload(
            center=np.array([0.0, 0.0, 0.0]),
            az_min_deg=-20.0,
            az_max_deg=20.0,
            el_min_deg=-15.0,
            el_max_deg=15.0,
            radius=1.0,
        )
        assert payload is not None
        assert payload.vertices.shape[1] == 3
        assert payload.triangles.shape[1] == 3
        assert len(payload.vertices) > 4
        assert len(payload.triangles) > 2

    def test_aperture_mesh_payload_fills_sector_from_device(self):
        """A filtered sector should connect its translucent shell to the device."""
        center = np.array([3.0, -2.0, 7.0])
        payload = create_aperture_mesh_payload(
            center=center,
            az_min_deg=-14.0,
            az_max_deg=180.0,
            el_min_deg=44.0,
            el_max_deg=90.0,
            radius=5.0,
        )
        assert payload is not None

        center_vertices = np.flatnonzero(
            np.all(np.isclose(payload.vertices, center, atol=1e-6), axis=1)
        )
        assert len(center_vertices) > 0
        assert set(center_vertices).issubset(set(payload.triangles.reshape(-1)))

        areas2 = np.linalg.norm(
            np.cross(
                payload.vertices[payload.triangles[:, 1]]
                - payload.vertices[payload.triangles[:, 0]],
                payload.vertices[payload.triangles[:, 2]]
                - payload.vertices[payload.triangles[:, 0]],
            ),
            axis=1,
        )
        assert np.all(areas2 > 1e-8)

    def test_aperture_outline_skips_collapsed_pole_edges(self):
        """Elevation limits at +/-90 deg should not draw collapsed pole arcs."""
        line_payload = create_aperture_line_payload(
            center=np.array([0.0, 0.0, 0.0]),
            az_min_deg=-20.0,
            az_max_deg=20.0,
            el_min_deg=0.0,
            el_max_deg=90.0,
            radius=1.0,
            color=[1.0, 0.0, 0.0],
        )
        points = np.asarray(line_payload.points)
        lines = np.asarray(line_payload.lines)
        lengths = np.linalg.norm(points[lines[:, 1]] - points[lines[:, 0]], axis=1)
        assert np.all(lengths > 1e-8)
        center_ray_count = int(np.sum(np.any(lines == 0, axis=1)))
        assert center_ray_count == 3

    def test_aperture_mesh_payload_skips_degenerate_pole_triangles(self):
        """Filled pole-cap previews should not contain zero-area triangles."""
        payload = create_aperture_mesh_payload(
            center=np.array([0.0, 0.0, 0.0]),
            az_min_deg=-20.0,
            az_max_deg=20.0,
            el_min_deg=0.0,
            el_max_deg=90.0,
            radius=1.0,
        )
        assert payload is not None
        verts = payload.vertices
        areas2 = np.linalg.norm(
            np.cross(
                verts[payload.triangles[:, 1]] - verts[payload.triangles[:, 0]],
                verts[payload.triangles[:, 2]] - verts[payload.triangles[:, 0]],
            ),
            axis=1,
        )
        assert np.all(areas2 > 1e-8)

    def test_full_range_mesh_has_no_artificial_center_walls(self):
        """A full all-angle preview is a sphere without an arbitrary seam wall."""
        center = np.array([3.0, -2.0, 7.0])
        radius = 2.5
        payload = create_aperture_mesh_payload(
            center=center,
            az_min_deg=-180.0,
            az_max_deg=180.0,
            el_min_deg=-90.0,
            el_max_deg=90.0,
            radius=radius,
        )
        assert payload is not None
        vertex_radii = np.linalg.norm(payload.vertices - center, axis=1)
        np.testing.assert_allclose(vertex_radii, radius, atol=1e-6)

    def test_wrapped_azimuth_sector_stays_narrow_and_anchored(self):
        """A range crossing 180 degrees should use the short wrapped sector."""
        payload = create_aperture_mesh_payload(
            center=np.zeros(3),
            az_min_deg=170.0,
            az_max_deg=-170.0,
            el_min_deg=-10.0,
            el_max_deg=10.0,
            radius=1.0,
        )
        assert payload is not None
        radii = np.linalg.norm(payload.vertices, axis=1)
        outer_vertices = payload.vertices[radii > 0.99]
        azimuths = np.degrees(np.arctan2(outer_vertices[:, 1], outer_vertices[:, 0]))
        assert np.all((azimuths >= 169.9) | (azimuths <= -169.9))
        assert np.any(radii < 1e-8)

    def test_reversed_elevation_bounds_return_empty_geometry(self):
        """An elevation range that accepts no angles should not draw a sector."""
        kwargs = {
            "center": np.zeros(3),
            "az_min_deg": -20.0,
            "az_max_deg": 20.0,
            "el_min_deg": 40.0,
            "el_max_deg": -40.0,
            "radius": 1.0,
        }
        line_payload = create_aperture_line_payload(
            **kwargs,
            color=[1.0, 0.0, 0.0],
        )
        assert len(line_payload.points) == 0
        assert len(line_payload.lines) == 0
        assert create_aperture_mesh_payload(**kwargs) is None

    def test_point_aperture_has_only_nonzero_outline_segments(self):
        """An exact angular value should remain a ray without degenerate faces."""
        kwargs = {
            "center": np.zeros(3),
            "az_min_deg": 0.0,
            "az_max_deg": 0.0,
            "el_min_deg": 0.0,
            "el_max_deg": 0.0,
            "radius": 1.0,
        }
        line_payload = create_aperture_line_payload(
            **kwargs,
            color=[1.0, 0.0, 0.0],
        )
        points = np.asarray(line_payload.points)
        lines = np.asarray(line_payload.lines)
        lengths = np.linalg.norm(points[lines[:, 1]] - points[lines[:, 0]], axis=1)
        assert len(lines) == 1
        assert np.all(lengths > 1e-8)

        mesh_payload = create_aperture_mesh_payload(**kwargs)
        assert mesh_payload is not None
        assert mesh_payload.triangles.shape == (0, 3)

    def test_pitch_orientation_rotates_local_elevation_to_world(self):
        """A local forward aperture should follow the device pitch convention."""
        line_payload = create_aperture_line_payload(
            center=np.array([0.0, 0.0, 0.0]),
            az_min_deg=0.0,
            az_max_deg=0.0,
            el_min_deg=0.0,
            el_max_deg=0.0,
            radius=1.0,
            color=[1.0, 0.0, 0.0],
            orientation=(0.0, np.radians(30.0), 0.0),
        )
        points = np.asarray(line_payload.points)[1:]
        radii = np.linalg.norm(points, axis=1)
        elevations = np.degrees(np.arcsin(np.clip(points[:, 2] / radii, -1.0, 1.0)))
        np.testing.assert_allclose(elevations, -30.0, atol=1e-9)

    def test_local_angular_reference_uses_device_yaw(self):
        """Local angular reference axes should rotate with device orientation."""
        line_payload = create_angular_reference_line_payload(
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0,
            orientation=(np.pi / 2, 0.0, 0.0),
            local=True,
        )
        points = np.asarray(line_payload.points)
        lines = np.asarray(line_payload.lines)
        x_axis_line = lines[-6]
        x_axis_endpoint = points[x_axis_line[1]]
        np.testing.assert_allclose(x_axis_endpoint, [0.0, 0.9, 0.0], atol=1e-10)

    def test_angular_reference_label_positions_include_major_angles(self):
        """Reference labels should expose azimuth and elevation anchors."""
        labels = angular_reference_label_positions(
            center=np.array([0.0, 0.0, 0.0]),
            radius=1.0,
            local=False,
        )
        label_text = {text for text, _position, _color in labels}
        assert {"G 0", "G 90", "G 180", "G -90", "G +El", "G -El"} == label_text
