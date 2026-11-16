"""RX-only scalar array response from verified native sensing metadata.

This display is independent of recorded detections. Its logarithmic radius is
an arbitrary visual scale, never a received-power contour or detectable range.
Planar isotropic elements retain their physical front/back ambiguity; the
detector's positive-local-X reporting convention does not remove the rear lobe.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from shared.extensions.sensing import canonical_sensing_hash
from shared.sensing.arrays import (
    PLANAR_ELEMENT_ORDER,
    SPATIAL_PHASE_CONVENTION,
    direction_cosine_axis,
    forward_hemisphere_angles,
    planar_element_coordinates,
)

from ..beamforming.visualization import _rotation_matrix
from ..model import RenderObject
from ..types.render_payloads import MaterialPayload, MeshPayload, SurfaceColorSource

MAX_ELEMENTS = 256
DISPLAY_FLOOR_DB = -40.0


def receive_metadata_signature(sensing):
    """Fingerprint bounded RX/config/pair evidence while normal indexing reads it.

    The digest retains no arrays, arbitrary configuration, samples or reports.
    Missing metadata cannot authorize a response at another scene epoch.
    """
    array = sensing.get("rx_array")
    config = sensing.get("config_resolved")
    processing = sensing.get("detection_filter_metadata")
    acquisition = (sensing.get("temporal") or {}).get("acquisition")
    pairs = sensing.get("pair_axis")
    if (
        any(not isinstance(value, Mapping) for value in (array, config, processing, acquisition))
        or not isinstance(pairs, list)
        or len(pairs) != 1
        or not isinstance(pairs[0], Mapping)
    ):
        return None
    evidence = [
        array.get(key)
        for key in (
            "rows",
            "columns",
            "element_count",
            "element_order",
            "element_pattern",
            "polarization",
            "local_horizontal_direction",
            "local_vertical_direction",
            "spatial_phase_convention",
            "spatial_fft_normalization",
            "vertical_spacing_wavelengths",
            "horizontal_spacing_wavelengths",
        )
    ]
    evidence.append(config.get("carrier_freq_hz"))
    evidence.extend(
        processing.get(key)
        for key in (
            "array_detection_mode",
            "beam_count",
            "angular_hemisphere",
        )
    )
    evidence.extend(
        pairs[0].get(key)
        for key in (
            "pair_index",
            "frame_pair_index",
            "pair_id",
            "tx_sensor_id",
            "rx_sensor_id",
        )
    )
    evidence.extend(acquisition.get(key) for key in ("pair_id", "tx_sensor_id", "rx_sensor_id"))
    if any(not isinstance(value, (str, int, float, bool, type(None))) for value in evidence):
        return None
    if any(isinstance(value, float) and not np.isfinite(value) for value in evidence):
        return None
    return canonical_sensing_hash(evidence)


@dataclass(frozen=True)
class ReceivePattern:
    """A fixed single-pair receiver, with spacings expressed in wavelengths."""

    rows: int
    columns: int
    vertical_spacing: float
    horizontal_spacing: float
    position_m: tuple[float, float, float]
    orientation_rad: tuple[float, float, float]
    carrier_frequency_hz: float
    scan_enabled: bool
    source: str

    @property
    def element_positions(self):
        """Centred Sionna-ordered local coordinates in wavelengths."""
        indices = np.asarray(planar_element_coordinates(self.rows, self.columns))
        return np.column_stack(
            (
                np.zeros(len(indices)),
                (indices[:, 1] - (self.columns - 1) / 2) * self.horizontal_spacing,
                ((self.rows - 1) / 2 - indices[:, 0]) * self.vertical_spacing,
            )
        )

    @property
    def beam_directions(self):
        """Actual coarse spatial-FFT bins; no angle-refinement oversampling."""
        y = direction_cosine_axis(self.columns, self.horizontal_spacing)
        z = -direction_cosine_axis(self.rows, self.vertical_spacing)
        valid, _, _ = forward_hemisphere_angles(z, y)
        zz, yy = np.meshgrid(z, y, indexing="ij")
        return np.column_stack((np.sqrt(1 - yy[valid] ** 2 - zz[valid] ** 2), yy[valid], zz[valid]))


def pattern_from_metadata(array, position, orientation, config, processing, *, source):
    """Reject incomplete or unsupported metadata instead of inventing an array."""
    rows, columns = array.get("rows"), array.get("columns")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (rows, columns)):
        raise ValueError("RX array dimensions are missing or invalid")
    if rows * columns > MAX_ELEMENTS or array.get("element_count") != rows * columns:
        raise ValueError(f"RX response supports at most {MAX_ELEMENTS} explicitly ordered elements")
    expected = {
        "element_order": PLANAR_ELEMENT_ORDER,
        "element_pattern": "iso",
        "local_horizontal_direction": "+y",
        "local_vertical_direction": "+z",
        "spatial_phase_convention": SPATIAL_PHASE_CONVENTION,
        "spatial_fft_normalization": "unitary_ortho",
    }
    if any(array.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "RX response requires recorded isotropic elements and the native planar convention"
        )
    if array.get("polarization") not in {"V", "H"}:
        raise ValueError(
            "RX response supports one recorded polarization, not dual-polarized arrays"
        )
    v, h = (
        float(array.get(name, np.nan))
        for name in ("vertical_spacing_wavelengths", "horizontal_spacing_wavelengths")
    )
    frequency = float(config.get("carrier_freq_hz", np.nan))
    if not np.all(np.isfinite([v, h, frequency])) or min(v, h, frequency) <= 0:
        raise ValueError("RX spacing and carrier must be finite and positive")
    position, orientation = np.asarray(position, float), np.asarray(orientation, float)
    if any(a.shape != (3,) or not np.all(np.isfinite(a)) for a in (position, orientation)):
        raise ValueError("RX position/orientation must be explicit finite three-vectors")
    scan = processing.get("array_detection_mode") == "beam_scan"
    result = ReceivePattern(
        rows, columns, v, h, tuple(position), tuple(orientation), frequency, scan, source
    )
    if scan and (
        processing.get("angular_hemisphere") != "rx_local_positive_x"
        or processing.get("beam_count") != len(result.beam_directions)
    ):
        raise ValueError(
            "Recorded coarse beam count or angular convention disagrees with the array"
        )
    return result


def load_receive_pattern(reader):
    """Lazily authenticate one completed trial and read only metadata/devices.

    The source trial is selected by sorted grid index, independently of outcomes.
    Missing native data is an explicit unavailable state. Hash failures propagate;
    another trial is never substituted for failed authentication.
    """
    import yaml

    from shared.extensions.sensing import assert_sensing_contract
    from shared.frames.contracts import FrameComponent, FrameReadRequest
    from shared.sensing.loader import _open_provider
    from shared.sensing.volume.inspection import fixed_sensor_orientation

    config = yaml.safe_load(reader.verify_scene().read_text())
    receivers = config.get("actors", {}).get("rx", [])
    if len(receivers) != 1 or receivers[0].get("mobility", {}).get("type") != "stationary":
        raise ValueError("Volume RX response requires one explicitly stationary receiver")
    declared = fixed_sensor_orientation(config)
    if declared is None:
        raise ValueError("Volume RX response requires a recorded fixed orientation")
    declared_rotation = _rotation_matrix(*declared)

    key = next(
        (k for k in sorted(reader.cells) if reader.native_availability(k) == "available"), None
    )
    if key is None:
        raise ValueError("No included completed native trial supplies exact RX metadata")
    source = reader.verify_native(key)
    _, manifest, provider = _open_provider(source)
    try:
        # Metadata-only projection: no CIR, RD, angle or path arrays are read.
        request = FrameReadRequest(
            components=frozenset({FrameComponent.SENSING, FrameComponent.PROVENANCE}),
            sensing_products=frozenset({"products"}),
        )
        sensing = provider.load_frame_projection(manifest.frame_ids[-1], request).frame.sensing
        if not sensing or not sensing.get("temporal", {}).get("result"):
            raise ValueError("Selected native trial has no completed sensing result")
        assert_sensing_contract({"sensing": sensing}, require_schema_version=True)
        temporal = sensing["temporal"]
        scene = provider.load_frame_projection(
            temporal["display_binding"]["scene_frame_id"],
            FrameReadRequest(
                components=frozenset({FrameComponent.DEVICES, FrameComponent.PROVENANCE})
            ),
        ).frame
        name = temporal["acquisition"]["rx_sensor_id"]
        if scene.rx_names.count(name) != 1:
            raise ValueError("Native RX identity is missing or ambiguous")
        index = scene.rx_names.index(name)
        if not np.isclose(
            scene.timestamp_s, temporal["result"]["reference_time_s"], rtol=0, atol=1e-9
        ):
            raise ValueError("Native RX pose is not bound to the result reference epoch")
        if not np.allclose(
            scene.rx_positions[index], reader.metadata["sensor_position_m"], rtol=0, atol=1e-5
        ):
            raise ValueError("Native RX position disagrees with the volume sensor")
        if not np.allclose(
            _rotation_matrix(*scene.rx_orientations[index]), declared_rotation, rtol=0, atol=1e-7
        ):
            raise ValueError("Native RX orientation disagrees with the volume scene")
        return pattern_from_metadata(
            sensing["rx_array"],
            scene.rx_positions[index],
            scene.rx_orientations[index],
            sensing["config_resolved"],
            sensing.get("detection_filter_metadata", {}),
            source=str(source),
        )
    finally:
        provider.close()


def receive_power(pattern, directions, mode, *, beam_index=None):
    """Unit-input scalar receive power from normalized complex beam weights.

    ``directions`` are unit vectors in the RX local frame. The scan envelope is
    a maximum over individual coarse-beam powers, not a coherently combined beam.
    Work is bounded to 256 directions at a time; no propagation backend is used.
    """
    directions = np.asarray(directions, float)
    if directions.ndim != 2 or directions.shape[1] != 3 or not np.all(np.isfinite(directions)):
        raise ValueError("Pattern directions must be finite unit three-vectors")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError("Pattern directions must have unit length")
    if mode not in {"broadside", "scan", "beam"} or (
        mode in {"scan", "beam"} and not pattern.scan_enabled
    ):
        raise ValueError("Requested RX response is not supported by the recorded processing")
    positions = pattern.element_positions
    beams = pattern.beam_directions if mode == "scan" else np.array([[1.0, 0, 0]])
    if mode == "beam":
        if type(beam_index) is not int or not 0 <= beam_index < len(pattern.beam_directions):
            raise ValueError("Select an available recorded coarse beam")
        beams = pattern.beam_directions[beam_index : beam_index + 1]
    weights = np.exp(2j * np.pi * positions @ beams.T) / np.sqrt(len(positions))
    power = np.empty(len(directions))
    for start in range(0, len(directions), 256):
        steering = np.exp(2j * np.pi * positions @ directions[start : start + 256].T)
        power[start : start + 256] = np.max(np.abs(weights.conj().T @ steering) ** 2, axis=0)
    return power


def response_object(pattern, mode, radius_m, opacity, *, beam_index=None):
    """A full-sphere logarithmic response surface, including physical rear lobes."""
    if not np.isfinite(radius_m) or not 0 < radius_m <= 10000 or not 0 < opacity <= 1:
        raise ValueError("Invalid RX display radius or opacity")
    azimuth = np.linspace(-np.pi, np.pi, 121)
    elevation = np.linspace(-np.pi / 2, np.pi / 2, 61)
    az, el = np.meshgrid(azimuth, elevation)
    directions = np.column_stack(
        (
            np.cos(el).ravel() * np.cos(az).ravel(),
            np.cos(el).ravel() * np.sin(az).ravel(),
            np.sin(el).ravel(),
        )
    )
    power = receive_power(pattern, directions, mode, beam_index=beam_index)
    # A unit-norm beam has a theoretical peak N for a unit-amplitude plane wave.
    # Use that exact reference, not the maximum of a possibly off-peak mesh grid.
    relative = np.clip(power / (pattern.rows * pattern.columns), 1e-4, 1)
    db = 10 * np.log10(relative)
    fraction = (db - DISPLAY_FLOOR_DB) / -DISPLAY_FLOOR_DB
    rotation = _rotation_matrix(*pattern.orientation_rad)
    vertices = (directions * (radius_m * fraction[:, None])) @ rotation.T + pattern.position_m
    indices = np.arange(az.size).reshape(az.shape)
    a, b = indices[:-1, :-1].ravel(), indices[1:, :-1].ravel()
    triangles = np.vstack((np.column_stack((a, b, a + 1)), np.column_stack((a + 1, b, b + 1))))
    colors = np.column_stack(
        (0.50 + 0.45 * fraction, 0.25 + 0.50 * fraction, np.ones(len(fraction)))
    )
    return RenderObject(
        id="sensing:volume:rx_response",
        payload=MeshPayload(
            vertices=vertices.astype(np.float32),
            triangles=triangles.astype(np.int32),
            vertex_colors=colors.astype(np.float32),
            color_source=SurfaceColorSource.VERTEX,
        ),
        material=MaterialPayload(base_color=(1, 1, 1, opacity), shader="unlit"),
        metadata={"type": "rx_array_response", "depth_write": False, "double_sided": True},
    )
