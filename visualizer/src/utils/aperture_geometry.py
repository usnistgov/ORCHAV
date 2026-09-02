"""Aperture geometry utilities for AOA/AOD visualization."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from shared.logging import get_logger

from ..types.render_payloads import LineSetPayload, MeshPayload

logger = get_logger("orchav.utils.aperture_geometry")


def yaw_pitch_roll_to_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Convert yaw/pitch/roll (radians) to a 3x3 rotation matrix.

    Uses the ZYX Euler angle convention (yaw around Z, pitch around Y, roll around X).
    This matches Sionna RT's orientation convention.

    Args:
        yaw: Rotation around Z axis in radians
        pitch: Rotation around Y axis in radians
        roll: Rotation around X axis in radians

    Returns:
        3x3 rotation matrix
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    # ZYX rotation order: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )

    return R


def _spherical_to_cartesian(az_rad: float, el_rad: float, radius: float) -> np.ndarray:
    """Convert ORCHAV azimuth/elevation angles to a Cartesian direction."""
    cos_el = np.cos(el_rad)
    return np.array(
        [
            radius * cos_el * np.cos(az_rad),
            radius * cos_el * np.sin(az_rad),
            radius * np.sin(el_rad),
        ],
        dtype=np.float64,
    )


def _unwrap_azimuth_range(az_min_deg: float, az_max_deg: float) -> tuple[float, float]:
    """Return an azimuth range that can be sampled linearly, preserving wrap."""
    az_min = float(az_min_deg)
    az_max = float(az_max_deg)
    if abs(az_max - az_min) >= 359.9:
        return az_min, az_min + 360.0
    if az_max < az_min:
        az_max += 360.0
    return az_min, az_max


def _sample_degrees(start_deg: float, end_deg: float, step_deg: float = 5.0) -> np.ndarray:
    """Sample an inclusive angular interval at roughly ``step_deg`` spacing."""
    span = abs(float(end_deg) - float(start_deg))
    segments = max(1, int(np.ceil(span / max(float(step_deg), 1e-6))))
    return np.linspace(float(start_deg), float(end_deg), segments + 1, dtype=np.float64)


def _is_pole_elevation(el_deg: float) -> bool:
    """Return True when an elevation boundary collapses to a pole point."""
    return abs(abs(float(el_deg)) - 90.0) <= 1e-6


def _apply_orientation(
    points: np.ndarray,
    center: np.ndarray,
    orientation: Optional[Tuple[float, float, float]],
) -> np.ndarray:
    """Rotate points around ``center`` using an optional yaw/pitch/roll orientation."""
    if orientation is None:
        return points
    yaw, pitch, roll = orientation
    if abs(yaw) <= 1e-9 and abs(pitch) <= 1e-9 and abs(roll) <= 1e-9:
        return points
    rotation_matrix = yaw_pitch_roll_to_rotation_matrix(yaw, pitch, roll)
    center_arr = np.asarray(center, dtype=np.float64)
    local_points = points - center_arr
    return (rotation_matrix @ local_points.T).T + center_arr


def _create_line_payload(
    points: list[np.ndarray],
    lines: list[list[int]],
    colors: list[list[float]],
) -> LineSetPayload:
    """Build a backend-neutral line payload from prepared buffers."""
    return LineSetPayload(
        points=np.asarray(points, dtype=np.float64).reshape((-1, 3)),
        lines=np.asarray(lines, dtype=np.int32).reshape((-1, 2)),
        colors=np.asarray(colors, dtype=np.float64).reshape((-1, 3)),
    )


def create_aperture_line_payload(
    center: np.ndarray,
    az_min_deg: Optional[float],
    az_max_deg: Optional[float],
    el_min_deg: Optional[float],
    el_max_deg: Optional[float],
    radius: float,
    color: list[float],
    orientation: Optional[Tuple[float, float, float]] = None,
    density_deg: float = 15.0,
) -> LineSetPayload:
    """Create a readable spherical angular aperture preview.

    The preview intentionally avoids construction grids. In pygfx, the filled
    transparent patch shows the accepted zone; this line set only draws its
    boundary. For a full all-angle aperture it draws three great circles instead.

    When an orientation is provided, the aperture is rotated to match the device's
    local coordinate frame, so azimuth=0 points in the device's forward direction.

    Args:
        center: The origin point (3D position) for the aperture
        az_min_deg: Minimum azimuth angle in degrees (can be negative, e.g., -180 to 180)
        az_max_deg: Maximum azimuth angle in degrees (can be negative, e.g., -180 to 180)
        el_min_deg: Minimum elevation angle in degrees (-90 to 90)
        el_max_deg: Maximum elevation angle in degrees (-90 to 90)
        radius: Radius of the spherical wedge in meters
        color: RGB color as [r, g, b] with values 0-1
        orientation: Optional device orientation as (yaw, pitch, roll) in radians.
            When provided, the aperture is rotated to align with the device's
            local coordinate frame.
        density_deg: Target spacing between grid lines in degrees (default 15.0).
            Smaller values create denser wireframes.

    Returns:
        Neutral line payload, or None if no valid bounds are set.
    """
    has_any_bound = any(
        bound is not None for bound in (az_min_deg, az_max_deg, el_min_deg, el_max_deg)
    )
    if not has_any_bound:
        return _create_line_payload([], [], [])

    # Use natural angular extents for missing sides. The UI represents a
    # natural min/max as None, so one-sided filters such as [-179, Max] still
    # need a drawable aperture.
    az_min = az_min_deg if az_min_deg is not None else -180.0
    az_max = az_max_deg if az_max_deg is not None else 180.0
    el_min = el_min_deg if el_min_deg is not None else -90.0
    el_max = el_max_deg if el_max_deg is not None else 90.0
    if el_max < el_min:
        return _create_line_payload([], [], [])

    center = np.asarray(center, dtype=np.float64)
    az_min, az_max = _unwrap_azimuth_range(az_min, az_max)
    az_range = abs(az_max - az_min)
    el_range = abs(el_max - el_min)
    full_az = az_range >= 359.9
    full_el = el_range >= 179.9

    logger.debug(f"Aperture boundary: az_range={az_range:.1f}°, el_range={el_range:.1f}°")

    points: list[np.ndarray] = [center]
    lines: list[list[int]] = []
    colors: list[list[float]] = []

    def add_point(local_point: np.ndarray) -> int:
        """Append one local aperture point translated around the center."""
        points.append(center + local_point)
        return len(points) - 1

    def add_segment(start_idx: int, end_idx: int) -> None:
        """Append one non-degenerate boundary segment."""
        if start_idx == end_idx:
            return
        if np.linalg.norm(points[end_idx] - points[start_idx]) <= 1e-9:
            return
        lines.append([start_idx, end_idx])
        colors.append(color)

    def add_polyline(local_points: list[np.ndarray]):
        """Append connected boundary segments from local-space points."""
        if len(local_points) < 2:
            return
        indices = [add_point(pt) for pt in local_points]
        for left, right in zip(indices[:-1], indices[1:]):
            add_segment(left, right)

    if full_az and full_el:
        circle_angles = _sample_degrees(-180.0, 180.0, step_deg=7.5)
        add_polyline([_spherical_to_cartesian(np.radians(az), 0.0, radius) for az in circle_angles])
        add_polyline(
            [
                radius
                * np.array(
                    [np.cos(np.radians(a)), 0.0, np.sin(np.radians(a))],
                    dtype=np.float64,
                )
                for a in circle_angles
            ]
        )
        add_polyline(
            [
                radius
                * np.array(
                    [0.0, np.cos(np.radians(a)), np.sin(np.radians(a))],
                    dtype=np.float64,
                )
                for a in circle_angles
            ]
        )
    else:
        az_samples = _sample_degrees(az_min, az_max, step_deg=min(float(density_deg), 5.0))
        el_samples = _sample_degrees(el_min, el_max, step_deg=min(float(density_deg), 5.0))

        # Spherical patch boundary.
        if not full_el:
            for el in (el_min, el_max):
                # At +/-90° all azimuths map to the same pole. Drawing that
                # collapsed "arc" creates overlaid zero-length line segments
                # that look like black/dashed artifacts in pygfx.
                if _is_pole_elevation(el):
                    continue
                add_polyline(
                    [
                        _spherical_to_cartesian(np.radians(az), np.radians(el), radius)
                        for az in az_samples
                    ]
                )
        if not full_az:
            for az in (az_min, az_max):
                add_polyline(
                    [
                        _spherical_to_cartesian(np.radians(az), np.radians(el), radius)
                        for el in el_samples
                    ]
                )

        # Keep the outline visibly anchored without restoring the old dense
        # construction grid. Four corner rays describe a regular bounded
        # sector; geometrically coincident pole rays are collapsed to one.
        radial_endpoints: list[np.ndarray] = []
        for az in (() if full_az else (az_min, az_max)):
            for el in (el_min, el_max):
                endpoint = _spherical_to_cartesian(np.radians(az), np.radians(el), radius)
                if np.linalg.norm(endpoint) <= 1e-9:
                    continue
                if any(
                    np.allclose(endpoint, existing, rtol=0.0, atol=1e-8)
                    for existing in radial_endpoints
                ):
                    continue
                radial_endpoints.append(endpoint)
                add_segment(0, add_point(endpoint))

    if len(points) > 1:
        points_array = _apply_orientation(np.asarray(points), center, orientation)
        points = [point for point in points_array]

    return _create_line_payload(points, lines, colors)


def create_aperture_mesh_payload(
    center: np.ndarray,
    az_min_deg: Optional[float],
    az_max_deg: Optional[float],
    el_min_deg: Optional[float],
    el_max_deg: Optional[float],
    radius: float,
    orientation: Optional[Tuple[float, float, float]] = None,
    density_deg: float = 6.0,
) -> MeshPayload | None:
    """Create a filled spherical sector for the accepted angular aperture.

    This payload is intended for the pygfx renderer, where alpha-blended mesh
    overlays make the filtered-in angular zone much clearer than line limits.
    Partial angular ranges include radial boundary walls so the sector visibly
    originates at the selected device instead of floating at ``radius``.
    """
    has_any_bound = any(
        bound is not None for bound in (az_min_deg, az_max_deg, el_min_deg, el_max_deg)
    )
    if not has_any_bound:
        return None

    az_min = az_min_deg if az_min_deg is not None else -180.0
    az_max = az_max_deg if az_max_deg is not None else 180.0
    el_min = el_min_deg if el_min_deg is not None else -90.0
    el_max = el_max_deg if el_max_deg is not None else 90.0
    if el_max < el_min:
        return None

    center = np.asarray(center, dtype=np.float64)
    az_min, az_max = _unwrap_azimuth_range(az_min, az_max)
    az_range = abs(az_max - az_min)
    el_range = abs(el_max - el_min)
    full_az = az_range >= 359.9
    full_el = el_range >= 179.9
    az_values = _sample_degrees(az_min, az_max, step_deg=float(density_deg))
    el_values = _sample_degrees(el_min, el_max, step_deg=float(density_deg))

    local_vertices: list[np.ndarray] = []
    local_normals: list[np.ndarray] = []
    for el in el_values:
        for az in az_values:
            direction = _spherical_to_cartesian(np.radians(az), np.radians(el), 1.0)
            local_vertices.append(direction * float(radius))
            local_normals.append(direction / max(np.linalg.norm(direction), 1e-12))

    n_az = len(az_values)
    n_el = len(el_values)
    triangles: list[list[int]] = []

    def add_shell_triangle(indices: list[int]) -> None:
        """Append a non-degenerate indexed triangle on the outer shell."""
        v0, v1, v2 = (local_vertices[index] for index in indices)
        if np.linalg.norm(np.cross(v1 - v0, v2 - v0)) > 1e-9:
            triangles.append(indices)

    def add_flat_wall_triangle(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> None:
        """Append one flat-shaded radial wall triangle."""
        normal = np.cross(v1 - v0, v2 - v0)
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= 1e-9:
            return
        normal = normal / normal_length
        first = len(local_vertices)
        local_vertices.extend((v0, v1, v2))
        local_normals.extend((normal, normal, normal))
        triangles.append([first, first + 1, first + 2])

    def add_radial_wall(edge_indices: list[int], *, reverse: bool) -> None:
        """Fan one sampled angular boundary from the device center."""
        apex = np.zeros(3, dtype=np.float64)
        for current, following in zip(edge_indices[:-1], edge_indices[1:]):
            left = local_vertices[current]
            right = local_vertices[following]
            if reverse:
                add_flat_wall_triangle(apex, right, left)
            else:
                add_flat_wall_triangle(apex, left, right)

    for ei in range(n_el - 1):
        for ai in range(n_az - 1):
            p00 = ei * n_az + ai
            p10 = ei * n_az + ai + 1
            p01 = (ei + 1) * n_az + ai
            p11 = (ei + 1) * n_az + ai + 1
            for tri in ([p00, p10, p11], [p00, p11, p01]):
                add_shell_triangle(tri)

    # Close only real angular-domain boundaries. A full-range seam is an
    # implementation detail, not a filter wall, and pole arcs collapse to a
    # point. Duplicate flat-shaded wall vertices keep their lighting normals
    # independent from the shell's radial normals.
    if not full_el:
        if not _is_pole_elevation(el_min):
            add_radial_wall(list(range(n_az)), reverse=True)
        if el_range > 1e-9 and not _is_pole_elevation(el_max):
            top_start = (n_el - 1) * n_az
            add_radial_wall(
                list(range(top_start, top_start + n_az)),
                reverse=False,
            )
    if not full_az:
        add_radial_wall([ei * n_az for ei in range(n_el)], reverse=False)
        if az_range > 1e-9:
            add_radial_wall(
                [ei * n_az + n_az - 1 for ei in range(n_el)],
                reverse=True,
            )

    vertices = center + np.asarray(local_vertices, dtype=np.float64)
    normals = np.asarray(local_normals, dtype=np.float64)
    if len(vertices):
        vertices = _apply_orientation(vertices, center, orientation)
        normals = _apply_orientation(normals, np.zeros(3, dtype=np.float64), orientation)

    return MeshPayload(
        vertices=np.asarray(vertices, dtype=np.float32),
        triangles=np.asarray(triangles, dtype=np.int32).reshape((-1, 3)),
        normals=np.asarray(normals, dtype=np.float32),
    )


GLOBAL_REFERENCE_COLOR = [0.8, 0.8, 0.8]
LOCAL_REFERENCE_COLOR = [1.0, 0.85, 0.2]
AXIS_X_COLOR = [1.0, 0.2, 0.2]
AXIS_Y_COLOR = [0.2, 0.9, 0.35]
AXIS_Z_COLOR = [0.35, 0.55, 1.0]


def create_angular_reference_line_payload(
    center: np.ndarray,
    radius: float,
    *,
    orientation: Optional[Tuple[float, float, float]] = None,
    local: bool = False,
) -> LineSetPayload:
    """Create a compact angular reference overlay around a selected node.

    The overlay shows the azimuth plane, the elevation plane, and six signed
    axes. When ``local`` is true, the overlay is rotated by the node
    orientation; otherwise it stays aligned with the global/world axes.
    """
    center = np.asarray(center, dtype=np.float64)
    ring_radius = max(float(radius) * 0.75, 0.1)
    axis_radius = max(float(radius) * 0.9, 0.1)
    ref_color = LOCAL_REFERENCE_COLOR if local else GLOBAL_REFERENCE_COLOR

    points: list[np.ndarray] = [center]
    lines: list[list[int]] = []
    colors: list[list[float]] = []

    def add_point(local_point: np.ndarray) -> int:
        """Append one local reference point translated around the center."""
        points.append(center + local_point)
        return len(points) - 1

    def add_segment(start_idx: int, end_idx: int, segment_color: list[float]) -> None:
        """Append one colored reference segment."""
        lines.append([start_idx, end_idx])
        colors.append(segment_color)

    def add_polyline(local_points: list[np.ndarray], segment_color: list[float]) -> None:
        """Append connected reference segments from local-space points."""
        indices = [add_point(pt) for pt in local_points]
        for left, right in zip(indices[:-1], indices[1:]):
            add_segment(left, right, segment_color)

    def add_axis(direction: np.ndarray, segment_color: list[float]) -> None:
        """Append one signed axis ray from the overlay center."""
        add_segment(0, add_point(axis_radius * direction), segment_color)

    circle_angles = _sample_degrees(-180.0, 180.0, step_deg=15.0)
    add_polyline(
        [_spherical_to_cartesian(np.radians(az), 0.0, ring_radius) for az in circle_angles],
        ref_color,
    )
    add_polyline(
        [
            ring_radius
            * np.array([np.cos(np.radians(a)), 0.0, np.sin(np.radians(a))], dtype=np.float64)
            for a in circle_angles
        ],
        ref_color,
    )

    add_axis(np.array([1.0, 0.0, 0.0]), AXIS_X_COLOR)
    add_axis(np.array([-1.0, 0.0, 0.0]), AXIS_X_COLOR)
    add_axis(np.array([0.0, 1.0, 0.0]), AXIS_Y_COLOR)
    add_axis(np.array([0.0, -1.0, 0.0]), AXIS_Y_COLOR)
    add_axis(np.array([0.0, 0.0, 1.0]), AXIS_Z_COLOR)
    add_axis(np.array([0.0, 0.0, -1.0]), AXIS_Z_COLOR)

    orientation_to_apply = orientation if local else None
    points_array = _apply_orientation(np.asarray(points), center, orientation_to_apply)
    return _create_line_payload([point for point in points_array], lines, colors)


def angular_reference_label_positions(
    center: np.ndarray,
    radius: float,
    *,
    orientation: Optional[Tuple[float, float, float]] = None,
    local: bool = False,
) -> list[tuple[str, np.ndarray, list[float]]]:
    """Return label text, position, and color for an angular reference overlay."""
    center = np.asarray(center, dtype=np.float64)
    label_radius = max(float(radius) * 1.03, 0.1)
    prefix = "L" if local else "G"
    labels = [
        ("0", np.array([1.0, 0.0, 0.0], dtype=np.float64), AXIS_X_COLOR),
        ("90", np.array([0.0, 1.0, 0.0], dtype=np.float64), AXIS_Y_COLOR),
        ("180", np.array([-1.0, 0.0, 0.0], dtype=np.float64), AXIS_X_COLOR),
        ("-90", np.array([0.0, -1.0, 0.0], dtype=np.float64), AXIS_Y_COLOR),
        ("+El", np.array([0.0, 0.0, 1.0], dtype=np.float64), AXIS_Z_COLOR),
        ("-El", np.array([0.0, 0.0, -1.0], dtype=np.float64), AXIS_Z_COLOR),
    ]
    points = np.asarray([center + label_radius * direction for _, direction, _ in labels])
    points = _apply_orientation(points, center, orientation if local else None)
    return [
        (f"{prefix} {text}", points[index], color) for index, (text, _, color) in enumerate(labels)
    ]


# Predefined colors for AOA and AOD apertures
AOA_APERTURE_COLOR = [0.2, 0.5, 1.0]  # Blue (matches RX theme)
AOD_APERTURE_COLOR = [1.0, 0.3, 0.2]  # Red (matches TX theme)
