"""Exact saved coarse-beam cells for a selected report's range explanation.

These surfaces depict numerical bin support. They are neither localization
uncertainty nor a measured continuous power or detection-coverage field.
"""

from dataclasses import dataclass

import numpy as np

from ..sensing.selection import matches_detection
from ..sensing.volume_antenna import pattern_from_metadata
from ..utils.aperture_geometry import yaw_pitch_roll_to_rotation_matrix
from .range_geometry import build_range_locus, reference_receiver_pose


def receive_pattern_for_request(request):
    """Use the accepted request's metadata and frozen RX pose; read no arrays."""
    position, orientation = reference_receiver_pose(request)
    sensing = request.display_frame["sensing"]
    return pattern_from_metadata(
        sensing.get("rx_array", {}),
        position,
        orientation,
        sensing["config_resolved"],
        sensing.get("detection_filter_metadata", {}),
        source=repr(request.source_identity),
    )


def selected_report(request, selection):
    """Resolve one current saved report, never a nearest truth target."""
    sensing = request.display_frame["sensing"]
    if (
        selection is None
        or selection.source != request.source_identity
        or selection.acquisition_id != sensing["temporal"]["acquisition"]["acquisition_id"]
    ):
        raise ValueError("Select a report from the displayed saved acquisition")
    rows = [
        row
        for index, row in enumerate(sensing.get("detections", ()))
        if matches_detection(selection, sensing, index, row)
    ]
    if len(rows) != 1:
        raise ValueError("Select one saved detection report with a representative beam")
    return rows[0]


def selected_beam_index(request, selection, pattern):
    """Validate the recorded coarse beam index, distinct from refined angles."""
    report = selected_report(request, selection)
    index = report.get("beam_index")
    if (
        type(index) is not int
        or not pattern.scan_enabled
        or not 0 <= index < len(pattern.beam_directions)
    ):
        raise ValueError("This report has no supported saved coarse receive beam")
    return index


@dataclass(frozen=True)
class AngularCell:
    vertices: np.ndarray
    triangles: np.ndarray
    outline: np.ndarray
    beam_index: int
    label: str


def build_angular_cell(request, selection):
    """Restrict the selected monostatic range bin to its recorded coarse beam."""
    locus = build_range_locus(request, selection)
    if request.range_mode != "monostatic" or len(locus.support_m) != 2:
        raise ValueError("Angular cells require a monostatic range bin with saved boundaries")
    report = selected_report(request, selection)
    if (report.get("range_bin"), report.get("doppler_bin")) != (
        selection.cell.range_bin,
        selection.cell.doppler_bin,
    ):
        raise ValueError("The selected cell does not match the report's representative bin")
    pattern = receive_pattern_for_request(request)
    index = selected_beam_index(request, selection, pattern)
    if min(pattern.rows, pattern.columns) < 2:
        raise ValueError("A 3D angular cell requires a two-dimensional receive array")
    direction = pattern.beam_directions[index]
    y, z = direction[1:]
    dy = 0.5 / (pattern.columns * pattern.horizontal_spacing)
    dz = 0.5 / (pattern.rows * pattern.vertical_spacing)
    ys = np.linspace(max(-1, y - dy), min(1, y + dy), 9)
    zs = np.linspace(max(-1, z - dz), min(1, z + dz), 9)
    zz, yy = np.meshgrid(zs, ys, indexing="ij")
    edge = np.sqrt(np.maximum(1 - zz * zz, 0))
    yy = np.clip(yy, -edge, edge)
    xx = np.sqrt(np.maximum(1 - yy * yy - zz * zz, 0))
    directions = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    rotation = yaw_pitch_roll_to_rotation_matrix(*pattern.orientation_rad)
    center = np.asarray(pattern.position_m)
    vertices = np.concatenate(
        [directions @ rotation.T * radius + center for radius in locus.support_m]
    )
    grid = np.arange(len(directions)).reshape(zz.shape)
    a, b = grid[:-1, :-1].ravel(), grid[1:, :-1].ravel()
    surface = np.vstack((np.column_stack((a, b, a + 1)), np.column_stack((a + 1, b, b + 1))))
    boundary = np.r_[grid[0, :], grid[1:, -1], grid[-1, -2::-1], grid[-2:0:-1, 0], grid[0, 0]]
    count = len(directions)
    walls = []
    for first, second in zip(boundary[:-1], boundary[1:]):
        walls.extend(((first, second, first + count), (second, second + count, first + count)))
    triangles = np.vstack((surface, surface[:, ::-1] + count, walls)).astype(np.int32)
    outline = vertices[np.r_[boundary, boundary[::-1] + count, boundary[:1]]]
    return AngularCell(
        vertices.astype(np.float32),
        triangles,
        outline.astype(np.float32),
        index,
        f"Saved coarse beam {index} × range bin {selection.cell.range_bin}; "
        "numerical bin support, not uncertainty. Refined report angles can differ.",
    )
