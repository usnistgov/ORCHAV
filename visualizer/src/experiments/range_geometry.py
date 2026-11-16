"""Selected-cell direct-two-leg loci at an explicitly bound reference epoch.

This NumPy-only kernel does not inspect targets, MPCs, RCS, or the current scene.
Its surfaces explain a numerical range coordinate, not a localization estimate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..sensing.selection import SensingSelection
    from ..sensing.update_request import SensingUpdateRequest

_FloatArray: TypeAlias = NDArray[np.float64]
_IndexArray: TypeAlias = NDArray[np.int32]
_AZIMUTH_STEPS = 64
_ELEVATION_STEPS = 32


def _immutable(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, slots=True)
class RangeLocus:
    """Detached immutable world geometry and its exact scientific identity.

    ``surfaces`` contains lower/upper numerical display boundaries, or one
    center surface if support is unavailable. ``equator`` is the closed
    horizontal section through the center-coordinate surface's center.
    """

    surfaces: tuple[tuple[_FloatArray, _IndexArray], ...]
    center: _FloatArray
    foci: _FloatArray
    equator: _FloatArray
    identity: tuple
    label: str
    disclosures: tuple[str, ...]
    support_m: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "surfaces",
            tuple((_immutable(v), _immutable(t)) for v, t in self.surfaces),
        )
        for name in ("center", "foci", "equator"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Range locus requires {name}")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Range locus requires an exact {name}")
    return value


def _real_array(value: Any, name: str) -> _FloatArray:
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real coordinates")
    return np.asarray(array, dtype=np.float64)


def _epoch(value: Any, name: str) -> float:
    array = _real_array(value, name)
    if array.shape != ():
        raise ValueError(f"{name} must be a finite scalar epoch")
    return float(array)


def _axis(value: Any, name: str) -> _FloatArray:
    array = _real_array(value, name)
    if array.ndim != 1 or not len(array) or np.any(np.diff(array) <= 0):
        raise ValueError(f"{name} must be a nonempty strictly increasing axis")
    return array


def _reference_binding(
    request: SensingUpdateRequest, temporal: Mapping[str, Any], reference: float
) -> tuple[int | str, float]:
    frame = request.display_frame
    binding = _mapping(temporal.get("display_binding"), "temporal.display_binding")
    bound_id = binding.get("scene_frame_id")
    if not isinstance(bound_id, (int, str)) or isinstance(bound_id, bool):
        raise ValueError("Range locus requires an explicit bound scene_frame_id")
    # Never substitute state_step or state_frame for the selected result's pose.
    if request.display_step is None or request.display_step != bound_id:
        raise ValueError("Selected display frame disagrees with temporal scene binding")
    provenance = frame.get("_sensing_display_provenance")
    if provenance is not None:
        provenance = _mapping(provenance, "display provenance")
        if provenance.get("bound_scene_frame_id") != bound_id:
            raise ValueError("Display provenance disagrees with temporal scene binding")
    epochs = []
    for key, scale in (("timestamp_s", 1.0), ("timestamp_ns", 1e-9)):
        if key in frame:
            epochs.append(_epoch(frame[key], key) * scale)
    source = frame.get("_source")
    if isinstance(source, Mapping) and "timestamp" in source:
        epochs.append(_epoch(source["timestamp"], "display source timestamp"))
    event = request.replay_event
    if event is not None:
        if event.scene_binding.bound_scene_frame_id != bound_id:
            raise ValueError("Replay scene binding disagrees with the selected display frame")
        expected_key = {
            "clock_id": temporal["clock"]["clock_id"],
            "processing_run_id": temporal["result"]["processing_run_id"],
            "acquisition_id": temporal["acquisition"]["acquisition_id"],
            "pair_id": temporal["acquisition"]["pair_id"],
            "reference_time_s": reference,
        }
        if any(getattr(event.key, name) != value for name, value in expected_key.items()):
            raise ValueError(
                "Replay occurrence or reference epoch disagrees with the selected result"
            )
        epochs.append(_epoch(event.scene_binding.bound_scene_time_s, "replay bound scene epoch"))
    if not epochs:
        raise ValueError("Range locus requires the display frame's recorded reference epoch")
    if any(epoch != reference for epoch in epochs):
        raise ValueError("Bound scene pose epoch must equal the result reference epoch")
    return bound_id, reference


def _pair_foci(
    frame: Mapping[str, Any], sensing: Mapping[str, Any], acquisition: Mapping[str, Any]
) -> tuple[_FloatArray, int]:
    pairs = sensing.get("pair_axis")
    if not isinstance(pairs, list) or len(pairs) != 1:
        raise ValueError("Range locus requires exactly one typed pair_axis row")
    pair = _mapping(pairs[0], "pair_axis row")
    if type(pair.get("pair_index")) is not int or pair["pair_index"] != 0:
        raise ValueError("Range locus requires pair_axis pair_index 0")
    for name in ("pair_id", "tx_sensor_id", "rx_sensor_id"):
        expected = _string(acquisition.get(name), f"acquisition {name}")
        if pair.get(name) != expected:
            raise ValueError(f"pair_axis {name} disagrees with the acquisition")
    index = pair.get("frame_pair_index")
    if type(index) is not int or index < 0:
        raise ValueError("pair_axis requires a nonnegative integer frame_pair_index")
    positions = []
    for name in ("tx_positions", "rx_positions"):
        array = np.asarray(frame.get(name))
        if array.ndim != 2 or array.shape[1:] != (3,) or index >= len(array):
            raise ValueError(f"Selected frame_pair_index requires a valid {name} [pair, 3] row")
        positions.append(_real_array(array[index], name))
    return np.asarray(positions, dtype=np.float64), index


def reference_receiver_pose(request):
    """Return the exact result-bound RX pose for explanatory array geometry."""
    sensing = _mapping(request.display_frame.get("sensing"), "typed sensing payload")
    temporal = _mapping(sensing.get("temporal"), "sensing temporal identity")
    reference = _epoch(temporal["result"].get("reference_time_s"), "result reference time")
    _reference_binding(request, temporal, reference)
    foci, index = _pair_foci(request.display_frame, sensing, temporal["acquisition"])
    orientations = _real_array(request.display_frame.get("rx_orientations"), "RX orientations")
    if orientations.ndim != 2 or orientations.shape[1:] != (3,) or index >= len(orientations):
        raise ValueError("The result requires an explicit RX orientation for its sensor pair")
    if not request.source_identity:
        raise ValueError("The result requires an explicit recording identity")
    return foci[1].copy(), orientations[index].copy()


def _support(
    sensing: Mapping[str, Any], axis: Mapping[str, Any], ranges: _FloatArray, index: int
) -> tuple[tuple[float, ...], str]:
    # Optional explicit edge exports are numerical cell support only. Neither
    # bandwidth nor sample spacing is an uncertainty/resolution estimate.
    exported = []
    for field in ("edges_key", "bin_edges_key"):
        if field in axis:
            key = _string(axis[field], f"range {field}")
            if key not in sensing:
                raise ValueError(f"Exported range edges {key!r} are unavailable")
            exported.append(_axis(sensing[key], "Exported range edges"))
    for key in ("range_edges_m", "range_bin_edges_m"):
        if key in sensing:
            exported.append(_axis(sensing[key], "Exported range edges"))
    if exported:
        edges = exported[0]
        if any(not np.array_equal(edges, other) for other in exported[1:]):
            raise ValueError("Exported range edge sources disagree")
        if (
            edges.shape != (len(ranges) + 1,)
            or np.any(ranges < edges[:-1])
            or np.any(ranges > edges[1:])
        ):
            raise ValueError("Exported range edges must bracket every range coordinate")
        return (float(edges[index]), float(edges[index + 1])), "Exported numerical bin boundaries"
    if len(ranges) == 1:
        return (float(ranges[0]),), "Single coordinate: support unknown; unthickened center surface"
    edges = np.concatenate(
        (
            [ranges[0] - (ranges[1] - ranges[0]) / 2],
            ranges[:-1] + np.diff(ranges) / 2,
            [ranges[-1] + (ranges[-1] - ranges[-2]) / 2],
        )
    )
    if not np.all(np.isfinite(edges)):
        raise ValueError("Range midpoint display boundaries must be finite")
    return (float(edges[index]), float(edges[index + 1])), "Midpoint-derived display boundaries"


def _semiaxes(value: float, baseline: float, mode: str) -> tuple[float, float]:
    if mode == "monostatic":
        if baseline != 0:
            raise ValueError("monostatic_range requires coincident TX/RX positions")
        if value <= 0:
            raise ValueError(
                "Monostatic radius must be positive; a zero-radius locus is degenerate"
            )
        return value, value
    if value < baseline:
        raise ValueError("Bistatic path length is shorter than the TX/RX baseline")
    if value == baseline:
        raise ValueError("Bistatic path length equals the baseline: degenerate line-segment locus")
    a = value / 2
    # Product form avoids cancellation in a*a - c*c near the baseline.
    b = np.sqrt((value - baseline) / 2) * np.sqrt(value / 2 + baseline / 2)
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        raise ValueError("Range locus semiaxes must be finite and nondegenerate")
    return a, float(b)


def _basis(delta: _FloatArray, baseline: float) -> _FloatArray:
    direction = delta / baseline if baseline else np.array([1.0, 0.0, 0.0])
    helper = np.eye(3)[np.argmin(np.abs(direction))]
    side = np.cross(direction, helper)
    side /= np.linalg.norm(side)
    return np.column_stack((direction, side, np.cross(direction, side)))


def _surface(
    center: _FloatArray, basis: _FloatArray, a: float, b: float
) -> tuple[_FloatArray, _IndexArray]:
    phi = np.arange(_AZIMUTH_STEPS) * (2 * np.pi / _AZIMUTH_STEPS)
    theta = np.arange(1, _ELEVATION_STEPS) * (np.pi / _ELEVATION_STEPS)
    rings = np.stack(
        np.broadcast_arrays(
            a * np.cos(theta)[:, None],
            b * np.sin(theta)[:, None] * np.cos(phi),
            b * np.sin(theta)[:, None] * np.sin(phi),
        ),
        axis=-1,
    ).reshape(-1, 3)
    local = np.concatenate(([[a, 0, 0]], rings, [[-a, 0, 0]]))
    vertices = local @ basis.T + center
    triangles = []
    last = len(vertices) - 1
    for j in range(_AZIMUTH_STEPS):
        following = (j + 1) % _AZIMUTH_STEPS
        triangles.append((0, 1 + j, 1 + following))
        for ring in range(_ELEVATION_STEPS - 2):
            first = 1 + ring * _AZIMUTH_STEPS
            next_ring = first + _AZIMUTH_STEPS
            triangles.extend(
                (
                    (first + j, next_ring + j, first + following),
                    (first + following, next_ring + j, next_ring + following),
                )
            )
        first = last - _AZIMUTH_STEPS
        triangles.append((first + j, last, first + following))
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Range locus world coordinates must be finite")
    return vertices, np.asarray(triangles, dtype=np.int32)


def _equator(center: _FloatArray, basis: _FloatArray, a: float, b: float) -> _FloatArray:
    # Restrict the ellipsoid to world z=center.z. Its horizontal axes are
    # parallel/perpendicular to the projected baseline. This form avoids
    # cancellation in the quadratic matrix near a degenerate horizontal locus.
    direction = basis[:, 0]
    horizontal = float(np.hypot(*direction[:2]))
    along = direction[:2] / horizontal if horizontal else np.array([1.0, 0.0])
    across = np.array([-along[1], along[0]])
    radius = b / np.hypot(horizontal * (b / a), direction[2])
    phi = np.arange(_AZIMUTH_STEPS) * (2 * np.pi / _AZIMUTH_STEPS)
    points = np.repeat(center[None, :], _AZIMUTH_STEPS, axis=0)
    points[:, :2] += np.cos(phi)[:, None] * radius * along + np.sin(phi)[:, None] * b * across
    if not np.all(np.isfinite(points)):
        raise ValueError("Horizontal range section coordinates must be finite")
    return np.concatenate((points, points[:1]))


def build_range_locus(request: SensingUpdateRequest, selection: SensingSelection) -> RangeLocus:
    """Build a bounded reference-pose locus, rejecting missing or stale bindings.

    Only the exact selected product's meter-valued range coordinates are used.
    A bistatic coordinate is total path length L (semimajor axis L/2); a
    monostatic coordinate is radius R, requiring coincident transmitter/receiver.
    """
    if selection is None or selection.cell is None:
        raise ValueError("Select an exact sensing cell before building a range locus")
    cell = selection.cell
    sensing = _mapping(request.display_frame.get("sensing"), "typed sensing payload")
    temporal = _mapping(sensing.get("temporal"), "sensing temporal identity")
    acquisition = _mapping(temporal.get("acquisition"), "temporal acquisition")
    result = _mapping(temporal.get("result"), "temporal result")
    clock = _mapping(temporal.get("clock"), "temporal clock")
    scope = (
        _string(clock.get("clock_id"), "clock_id"),
        _string(result.get("processing_run_id"), "processing_run_id"),
        _string(acquisition.get("pair_id"), "pair_id"),
    )
    if (
        not request.source_identity
        or selection.source != request.source_identity
        or cell.source != request.source_identity
    ):
        raise ValueError("Selected cell source does not match the request source identity")
    if selection.scope != scope:
        raise ValueError("Selected cell scope does not match the clock, run, and pair")
    if selection.acquisition_id != _string(acquisition.get("acquisition_id"), "acquisition_id"):
        raise ValueError("Selected cell belongs to a different acquisition")
    if cell.recipe != _string(result.get("processing_recipe_id"), "processing_recipe_id"):
        raise ValueError("Selected cell processing recipe is stale")
    if cell.range_mode != request.range_mode or request.range_mode not in {
        "monostatic",
        "bistatic",
    }:
        raise ValueError("Selected cell range mode is incompatible with the request")
    if cell.variant != request.variant:
        raise ValueError("Selected cell variant is stale")
    products = _mapping(sensing.get("products"), "typed product catalog")
    catalog = tuple(
        sorted(
            (key, _mapping(p, "product descriptor").get("product_id"))
            for key, p in products.items()
        )
    )
    if not cell.catalog_identity or cell.catalog_identity != catalog:
        raise ValueError("Selected cell product catalog is stale")
    matches = [p for p in products.values() if p.get("product_id") == cell.product_id]
    if len(matches) != 1:
        raise ValueError("Selected cell requires one exact product identity")
    product = matches[0]
    if (
        product.get("variant") != request.variant
        or str(product.get("combine", {})) != cell.combination
    ):
        raise ValueError("Selected product variant or combination does not match the cell")
    lineage = _mapping(product.get("lineage"), "product lineage")
    for name, expected in (
        ("acquisition_id", selection.acquisition_id),
        ("processing_run_id", scope[1]),
        ("processing_recipe_id", cell.recipe),
    ):
        if lineage.get(name) != expected:
            raise ValueError(f"Selected product {name} does not match the acquisition/result")
    pair = _mapping(product.get("pair"), "product pair identity")
    if product.get("pair_ids") != [scope[2]] or any(
        pair.get(name) != acquisition.get(name)
        for name in ("pair_id", "tx_sensor_id", "rx_sensor_id")
    ):
        raise ValueError("Selected product pair identity does not match the acquisition")
    reference = _epoch(result.get("reference_time_s"), "result reference_time_s")
    span = _mapping(product.get("time_span"), "product time span")
    if (
        span.get("clock_id") != scope[0]
        or _epoch(span.get("reference_time_s"), "product reference_time_s") != reference
    ):
        raise ValueError("Selected product reference epoch or clock does not match the result")
    bound_id, reference = _reference_binding(request, temporal, reference)
    foci, frame_pair_index = _pair_foci(request.display_frame, sensing, acquisition)
    axes = product.get("axes")
    if not isinstance(axes, list) or not all(isinstance(axis, Mapping) for axis in axes):
        raise ValueError("Selected product requires typed axes")
    range_axes = [axis for axis in axes if axis.get("name") == "range"]
    if len(range_axes) != 1:
        raise ValueError("Selected product requires exactly one range axis")
    axis = range_axes[0]
    semantics = "monostatic_range" if request.range_mode == "monostatic" else "bistatic_path_length"
    if (
        axis.get("coordinate_semantics") != semantics
        or axis.get("unit") != "m"
        or axis.get("coordinate_kind") != "dense"
        or axis.get("reference_frame") != "tx_target_rx_path"
    ):
        raise ValueError("Selected product has incompatible typed range-coordinate semantics")
    ranges = _axis(cell.ranges, "Selected ranges")
    key = _string(axis.get("coordinate_key"), "range coordinate key")
    if axis.get("size") != len(ranges) or not np.array_equal(
        _axis(sensing.get(key), "Product ranges"), ranges
    ):
        raise ValueError("Selected range coordinates disagree with the exact product axis")
    if np.any(ranges < 0):
        raise ValueError("Range coordinates cannot be negative")
    if type(cell.range_bin) is not int or not 0 <= cell.range_bin < len(ranges):
        raise ValueError("Selected range bin is outside the exact product axis")
    if cell.doppler:
        doppler_axes = [a for a in axes if a.get("name") == "doppler_frequency"]
        if len(doppler_axes) != 1:
            raise ValueError("Selected Doppler coordinates disagree with the exact product axis")
        doppler_coordinates = sensing.get(doppler_axes[0].get("coordinate_key"))
        if doppler_coordinates is None or not np.array_equal(doppler_coordinates, cell.doppler):
            raise ValueError("Selected Doppler coordinates disagree with the exact product axis")
    if cell.doppler_bin is not None and (
        type(cell.doppler_bin) is not int or not 0 <= cell.doppler_bin < len(cell.doppler)
    ):
        raise ValueError("Selected Doppler bin is outside the exact product axis")
    values, support_label = _support(sensing, axis, ranges, cell.range_bin)
    center = foci[0] / 2 + foci[1] / 2
    delta = foci[1] - foci[0]
    baseline = float(np.hypot.reduce(delta))
    if not np.isfinite(baseline):
        raise ValueError("TX/RX baseline must be finite")
    basis = _basis(delta, baseline)
    coordinate = float(ranges[cell.range_bin])
    a, b = _semiaxes(coordinate, baseline, request.range_mode)
    surfaces = tuple(
        _surface(center, basis, *_semiaxes(value, baseline, request.range_mode)) for value in values
    )
    identity: tuple = (
        request.source_identity,
        scope,
        selection.acquisition_id,
        cell,
        bound_id,
        reference,
        frame_pair_index,
        tuple(map(tuple, foci)),
        values,
    )
    symbol = "R" if request.range_mode == "monostatic" else "L"
    label = f"Frozen-reference {semantics} locus: {symbol}={coordinate:g} m; bin {cell.range_bin}"
    return RangeLocus(
        surfaces=surfaces,
        center=center,
        foci=foci,
        equator=_equator(center, basis, a, b),
        identity=identity,
        label=label,
        support_m=values,
        disclosures=(
            f"{support_label}; numerical display support, not uncertainty or physical resolution",
            f"Frozen TX/RX reference pose at t={reference:g} s, bound scene {bound_id}",
            "Direct-two-leg geometric locus; not a target, probability region, diffraction path, or RCS",
            "No angular, Doppler, truth, or multipath localization is inferred",
            "Horizontal center-surface section; no ground-plane intersection is inferred",
        ),
    )
