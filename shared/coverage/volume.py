"""Bounded multi-height radio coverage for exact, unsmoothed inspection.

Coverage XY origins are cell boundaries; heights are plane centres. A radio-map
cell contains a solver's area estimate, not a sensing trial or a probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .availability import (
    AVAILABILITY_PATH,
    AVAILABILITY_VERSION,
    CoverageAvailability,
    validate_availability,
)
from .hdf5 import CoverageHDF5Reader
from .schema import coverage_metric_base, resolve_coverage_tx_index

MAX_VOLUME_CELLS = 2_000_000
ALL_TRANSMITTERS = "@strongest-all"


@dataclass(frozen=True)
class CoverageVolume:
    """Immutable samples in the selected metric's units and exact grid axes."""

    values: np.ndarray
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    transmitter: str
    quantity: str
    zero_contributions: np.ndarray
    source_identity: tuple[str, int, int]
    availability: np.ndarray | None = None
    transmitter_names: tuple[str, ...] = ()

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Return validated uniform cell dimensions in world XYZ metres."""
        return tuple(float(axis[1] - axis[0]) for axis in self.axes)


def coverage_file_identity(path: str | Path) -> tuple[str, int, int]:
    """Identify a locally opened result for cache invalidation, not provenance."""
    source = Path(path).resolve()
    stat = source.stat()
    return str(source), stat.st_mtime_ns, stat.st_size


def _read_volume_grid(h5):
    """Validate size and coordinates before allocating any dense metric arrays."""
    shape = np.asarray(h5["grid/shape_xyz"], dtype=np.int64)
    if shape.shape != (3,) or np.any(shape < 2) or np.prod(shape, dtype=object) > MAX_VOLUME_CELLS:
        raise ValueError(
            "Volume requires at least two samples per axis and at most 2,000,000 cells"
        )
    stored_shape = h5["values/path_gain_linear"].shape
    if (
        len(stored_shape) != 5
        or stored_shape[0] < 1
        or stored_shape[1] != shape[2]
        or stored_shape[-2:] != tuple(shape[1::-1])
    ):
        raise ValueError("Coverage grid and stored tensor dimensions disagree")
    origin = np.asarray(h5["grid/origin"], dtype=float)
    spacing = np.asarray(h5["grid/spacing_xy"], dtype=float)
    heights = np.asarray(h5["grid/heights_m"], dtype=float)
    if (
        origin.shape != (3,)
        or spacing.shape != (2,)
        or not np.all(np.isfinite(origin))
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0)
    ):
        raise ValueError("Invalid coverage grid origin or spacing")
    if (
        heights.shape != (int(shape[2]),)
        or not np.all(np.isfinite(heights))
        or np.any(np.diff(heights) <= 0)
        or not np.allclose(np.diff(heights), heights[1] - heights[0], rtol=1e-6, atol=1e-6)
    ):
        raise ValueError(
            "3D volume requires uniformly spaced, increasing heights; use height playback for this result"
        )
    axes = (
        origin[0] + (np.arange(shape[0]) + 0.5) * spacing[0],
        origin[1] + (np.arange(shape[1]) + 0.5) * spacing[1],
        heights,
    )
    return shape, stored_shape, axes


def load_coverage_volume(path: str | Path, transmitter: str, quantity: str) -> CoverageVolume:
    """Read one bounded gain/power layer without importing Qt or the ray tracer.

    Nonuniform heights remain supported by the ordinary coverage plane viewer;
    this uniform texture path refuses to stretch them into a false regular grid.
    Zero solver gain is retained as -infinity (+infinity for path loss),
    separate from missing NaN data.
    """
    if quantity not in {"path_gain_db", "path_loss_db", "rss_dbm"}:
        raise ValueError("Coverage volume supports path gain/loss (dB) and received power (dBm)")
    if transmitter == ALL_TRANSMITTERS:
        return _load_strongest_volume(path, quantity)
    identity = coverage_file_identity(path)
    with h5py.File(path, "r") as h5:
        shape, stored_shape, axes = _read_volume_grid(h5)
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in h5["tx/names"][()]]
        if transmitter not in names or names.count(transmitter) != 1:
            raise ValueError("Choose one unambiguous recorded transmitter")
        tx_index = names.index(transmitter)
        availability = None
        if AVAILABILITY_PATH in h5:
            dataset = h5[AVAILABILITY_PATH]
            if (
                dataset.shape != stored_shape
                or dataset.attrs.get("schema_version") != AVAILABILITY_VERSION
            ):
                raise ValueError("Unsupported coverage availability schema or dimensions")
            availability = np.asarray(dataset[0, :, tx_index])
        power = float(h5["tx/powers_dbm"][tx_index])
        if quantity == "rss_dbm" and not np.isfinite(power):
            raise ValueError("Transmitter power is not recorded")
    gain = CoverageHDF5Reader(Path(path)).read_metric(f"path_gain_linear/{transmitter}")
    if gain.shape != tuple(int(v) for v in shape[::-1]):
        raise ValueError("Coverage grid and metric dimensions disagree")
    if np.any(gain < 0) or np.any(np.isinf(gain)):
        raise ValueError("Invalid recorded path gain")
    if availability is None:
        # Never infer why a historical NaN was written.
        availability = np.where(
            np.isfinite(gain) & (gain > 0),
            CoverageAvailability.POSITIVE,
            np.where(
                gain == 0, CoverageAvailability.ZERO_CONTRIBUTION, CoverageAvailability.UNKNOWN
            ),
        ).astype(np.uint8)
    availability = validate_availability(availability, gain)
    zero = availability == CoverageAvailability.ZERO_CONTRIBUTION
    with np.errstate(divide="ignore", invalid="ignore"):
        values = (10 * np.log10(gain)).astype(np.float32)
    values[zero] = -np.inf
    if quantity == "rss_dbm":
        values += power
    elif quantity == "path_loss_db":
        values = -values
    if coverage_file_identity(path) != identity:
        raise ValueError("Coverage file changed while loading; reopen the result")
    for array in (values, zero, availability, *axes):
        array.flags.writeable = False
    return CoverageVolume(values, axes, transmitter, quantity, zero, identity, availability)


def _load_strongest_volume(path: str | Path, quantity: str) -> CoverageVolume:
    """Stream strongest links without adding transmitter fields or powers.

    For path loss, select the transmitter by received power, matching the
    ordinary best-path-loss layer. Unequal transmit powers make this different
    from simply selecting the smallest loss.

    Every TX must have a usable cell before its maximum is known. Incomplete
    cells remain unavailable even if another TX has a finite estimate there.
    Reduction storage is bounded independently of TX count.
    """
    identity = coverage_file_identity(path)
    with h5py.File(path, "r") as h5:
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in h5["tx/names"][()]]
        powers = np.asarray(h5["tx/powers_dbm"], dtype=float)
    if not names or len(set(names)) != len(names) or ALL_TRANSMITTERS in names:
        raise ValueError("Combined volume requires unambiguous recorded transmitters")
    loss = quantity == "path_loss_db"
    if loss and (powers.shape != (len(names),) or not np.isfinite(powers).all()):
        raise ValueError("Combined path loss requires recorded transmitter powers")
    first = load_coverage_volume(path, names[0], quantity)
    values = first.values.copy()
    reasons = first.availability.copy()
    best_power = powers[0] - values if loss else None
    for index, name in enumerate(names[1:], start=1):
        layer = load_coverage_volume(path, name, quantity)
        if layer.source_identity != identity or any(
            not np.array_equal(a, b) for a, b in zip(first.axes, layer.axes)
        ):
            raise ValueError("Coverage file changed while combining transmitters")
        if loss:
            candidate_power = powers[index] - layer.values
            values[:] = np.where(candidate_power > best_power, layer.values, values)
            np.maximum(best_power, candidate_power, out=best_power)
            values[np.isnan(best_power)] = np.nan
        else:
            np.maximum(values, layer.values, out=values)
        invalid = (reasons == CoverageAvailability.INVALID) | (
            layer.availability == CoverageAvailability.INVALID
        )
        reasons[:] = np.where(
            np.isnan(values),
            np.where(invalid, CoverageAvailability.INVALID, CoverageAvailability.UNKNOWN),
            np.where(
                np.isinf(values),
                CoverageAvailability.ZERO_CONTRIBUTION,
                CoverageAvailability.POSITIVE,
            ),
        )
    if coverage_file_identity(path) != identity or first.source_identity != identity:
        raise ValueError("Coverage file changed while combining transmitters")
    zero = np.isinf(values)
    for array in (values, reasons, zero):
        array.flags.writeable = False
    return CoverageVolume(values, first.axes, ALL_TRANSMITTERS, quantity, zero, identity, reasons)


def coverage_threshold_mask(volume: CoverageVolume, threshold: float) -> np.ndarray:
    """Qualify finite saved cells; lower loss or higher power/gain is stronger."""
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    base, _ = coverage_metric_base(volume.quantity)
    comparison = (
        np.less_equal if base in {"path_loss_db", "best_path_loss_db"} else np.greater_equal
    )
    return (
        np.isfinite(volume.values)
        & ~volume.zero_contributions
        & comparison(volume.values, threshold)
    )


def display_samples(
    volume: CoverageVolume, low: float, high: float, *, threshold: float | None = None
) -> np.ndarray:
    """Encode exact cells for a 257-entry palette; zero means transparent.

    Finite values outside the locked colour limits saturate. Zero contributions
    and unavailable cells are transparent; their source states remain distinct.
    """
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("Colour minimum must be below maximum")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    present = np.isfinite(volume.values)
    if threshold is not None:
        present &= coverage_threshold_mask(volume, threshold)
    scaled = 1 + 255 * np.clip((volume.values - low) / (high - low), 0, 1)
    return np.where(present, scaled, 0).astype(np.float32)


def load_coverage_metric_volume(path: str | Path, metric_name: str) -> CoverageVolume:
    """Load the ordinary panel's exact logical metric across all saved heights.

    SINR, serving identity and aggregate metrics use the existing canonical
    reader and formulas, including stored transmitter powers and noise. Reads
    proceed by height to bound temporary multi-transmitter allocations. No
    metric is redefined by the volume renderer.
    """
    identity = coverage_file_identity(path)
    base, selector = coverage_metric_base(metric_name)
    from .schema import COVERAGE_PER_TX_METRICS

    with h5py.File(path, "r") as h5:
        shape, stored_shape, axes = _read_volume_grid(h5)
        names = tuple(v.decode() if isinstance(v, bytes) else str(v) for v in h5["tx/names"][()])
        if not names or len(names) != stored_shape[2] or len(set(names)) != len(names):
            raise ValueError("Coverage requires unambiguous recorded transmitters")
        tx_index = (
            resolve_coverage_tx_index(selector, names, len(names))
            if selector is not None or base in COVERAGE_PER_TX_METRICS
            else None
        )
        transmitter = names[tx_index] if tx_index is not None else ALL_TRANSMITTERS
        dataset = h5.get(AVAILABILITY_PATH)
        if dataset is not None and (
            dataset.shape != stored_shape
            or dataset.attrs.get("schema_version") != AVAILABILITY_VERSION
        ):
            raise ValueError("Unsupported coverage availability schema or dimensions")
        all_zero = np.ones(tuple(shape[::-1]), dtype=bool)
        any_invalid = np.zeros_like(all_zero)
        for index in ([tx_index] if tx_index is not None else range(len(names))):
            gain = np.asarray(h5["values/path_gain_linear"][0, :, index])
            layer = (
                np.asarray(dataset[0, :, index])
                if dataset is not None
                else np.where(
                    np.isfinite(gain) & (gain > 0),
                    CoverageAvailability.POSITIVE,
                    np.where(
                        gain == 0,
                        CoverageAvailability.ZERO_CONTRIBUTION,
                        CoverageAvailability.UNKNOWN,
                    ),
                ).astype(np.uint8)
            )
            layer = validate_availability(layer, gain)
            all_zero &= layer == CoverageAvailability.ZERO_CONTRIBUTION
            any_invalid |= layer == CoverageAvailability.INVALID
        reasons = np.where(
            all_zero,
            CoverageAvailability.ZERO_CONTRIBUTION,
            np.where(any_invalid, CoverageAvailability.INVALID, CoverageAvailability.UNKNOWN),
        ).astype(np.uint8)
    reader = CoverageHDF5Reader(path)
    values = np.empty(tuple(int(v) for v in shape[::-1]), dtype=np.float32)
    for height in range(int(shape[2])):
        layer = reader.read_metric(metric_name, height_indices=height)
        if layer.shape != (1, int(shape[1]), int(shape[0])):
            raise ValueError("Coverage grid and metric dimensions disagree")
        values[height] = layer[0]
    usable = np.isfinite(values)
    if base == "serving_tx":
        usable &= (values >= 0) & (values < len(names)) & (values == np.floor(values))
    elif base in {"path_gain_linear", "serving_path_gain_linear", "rss_w", "sinr_linear"}:
        usable &= values > 0
    zero = reasons == CoverageAvailability.ZERO_CONTRIBUTION
    reasons = np.where(usable & ~zero, CoverageAvailability.POSITIVE, reasons).astype(np.uint8)
    if coverage_file_identity(path) != identity:
        raise ValueError("Coverage file changed while loading; reopen the result")
    for array in (values, zero, reasons, *axes):
        array.flags.writeable = False
    return CoverageVolume(values, axes, transmitter, metric_name, zero, identity, reasons, names)
