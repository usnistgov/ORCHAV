"""Frame-local, NumPy-only analysis for the pygfx path experiments.

Angles are the canonical world-frame measurements, in degrees. Delays are the
canonical delay metric, in nanoseconds; provenance is reported, not guessed.
Power weighting is a relative *path-power proxy* from explicitly exported loss,
not received power, coherent combining, an RF simulation, or a physical cluster
model. Canonical path IDs have meaning only within this frame.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import ClassVar, cast

import numpy as np

from ..metrics.mpc_path_catalog import MpcPathCatalog, MpcPathScope

SPEED_OF_LIGHT_M_PER_NS = 0.299792458
_ANGLE_FIELDS = (
    "aoa_azimuth",
    "aoa_elevation",
    "aod_azimuth",
    "aod_elevation",
)


class PathWeightMode(str, Enum):
    """Numerical weight for a selected frame-local path population."""

    COUNT = "count"
    POWER = "power"


def _immutable(values: np.ndarray) -> np.ndarray:
    """Detach values onto read-only storage without freezing borrowed arrays."""
    array = np.ascontiguousarray(values)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class _ImmutableArrays:
    __slots__ = ()

    def __post_init__(self) -> None:
        assert is_dataclass(self)
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, field.name, _immutable(value))


@dataclass(frozen=True, slots=True)
class PathAnalysisData(_ImmutableArrays):
    """Detached snapshot of one pair and scope, in canonical path-ID order.

    All selected rows are retained. Zero weights identify power-ineligible rows.
    Power weights equal ``10 ** (-(loss_db - minimum_eligible_loss_db) / 10)``;
    the strongest eligible exported-loss path therefore has weight one. A common
    scale factor has no effect on shares, histograms' relative mass or quantiles.
    Provenance codes are -1 unknown/unavailable, 0 exported, and 1 estimated.
    """

    tx_id: int
    rx_id: int
    scope: MpcPathScope
    weight_mode: PathWeightMode
    path_ids: np.ndarray
    delays_ns: np.ndarray
    aoa_azimuth_deg: np.ndarray
    aoa_elevation_deg: np.ndarray
    aod_azimuth_deg: np.ndarray
    aod_elevation_deg: np.ndarray
    weights: np.ndarray
    delay_provenance_codes: np.ndarray
    path_loss_provenance_codes: np.ndarray
    disclosures: tuple[str, ...]

    @property
    def path_count(self) -> int:
        """Count before delay, angle, gate, or weight validity exclusions."""
        return int(self.path_ids.size)

    @property
    def weighted_path_count(self) -> int:
        return int(np.count_nonzero(self.weights > 0.0))

    @property
    def weight_label(self) -> str:
        return (
            "Path count"
            if self.weight_mode is PathWeightMode.COUNT
            else "Relative power (strongest path = 1)"
        )

    @property
    def power_available(self) -> bool:
        """Catalog provenance marks only finite, explicitly exported losses as zero."""
        return bool(np.any(self.path_loss_provenance_codes == 0))

    @property
    def available_angles(self) -> tuple[str, ...]:
        """Angle columns with at least one usable delay and positive-weight row."""
        eligible = np.isfinite(self.delays_ns) & (self.weights > 0)
        return tuple(
            angle
            for angle in _ANGLE_FIELDS
            if np.any(eligible & np.isfinite(self.angles_deg(angle)))
        )

    def angles_deg(self, angle: str) -> np.ndarray:
        """Return a canonical world angle column; missing values remain NaN."""
        if angle not in _ANGLE_FIELDS:
            raise ValueError(f"angle must be one of {_ANGLE_FIELDS}")
        return cast(np.ndarray, getattr(self, f"{angle}_deg"))


@dataclass(frozen=True, slots=True)
class ArrivalGateResult(_ImmutableArrays):
    """Inclusive delay-gate selection; p10/p50/p90 summarize this gate only.

    ``weight_share`` divides gate weight by all eligible weight in the original
    pair/scope, including any rows without a usable delay. A zero denominator
    yields zero share and NaN percentiles. Row IDs include zero-weight paths;
    ``weighted_path_count`` reports how many contribute to the weighted summary.
    """

    lower_ns: float
    upper_ns: float
    path_ids: np.ndarray
    row_indices: np.ndarray
    weighted_path_count: int
    total_weight: float
    weight_share: float
    percentiles_ns: np.ndarray
    percentile_levels: ClassVar[tuple[int, ...]] = (10, 50, 90)

    @property
    def path_count(self) -> int:
        return int(self.path_ids.size)

    @property
    def p10_ns(self) -> float:
        return float(self.percentiles_ns[0])

    @property
    def p50_ns(self) -> float:
        return float(self.percentiles_ns[1])

    @property
    def p90_ns(self) -> float:
        return float(self.percentiles_ns[2])


@dataclass(frozen=True, slots=True)
class DelayAngleHistogram(_ImmutableArrays):
    """Delay-by-angle weighted numerical bins, never physical path clusters.

    ``values[delay_bin, angle_bin]`` uses raw count or relative proxy weights.
    Row/path mappings contain only finite-delay, finite-angle, positive-weight
    contributors. Delay bins are left-inclusive and right-exclusive, as are
    angle bins. Angles wrap to [-180, 180), so +180 belongs to the -180 bin.
    ``min_bins_90`` is the fewest highest-mass bins covering at least 90% of
    histogram mass. All shares are of histogram mass, not the original scope.
    """

    angle: str
    values: np.ndarray
    delay_edges_ns: np.ndarray
    angle_edges_deg: np.ndarray
    path_ids: np.ndarray
    row_indices: np.ndarray
    delay_bin_indices: np.ndarray
    angle_bin_indices: np.ndarray
    source_path_count: int
    gated_path_count: int
    excluded_path_count: int
    total_weight: float
    dominant_bin_share: float
    occupied_bins: int
    min_bins_90: int
    disclosures: tuple[str, ...]

    @property
    def path_count(self) -> int:
        return int(self.path_ids.size)

    def _check_bin(self, delay_index: int, angle_index: int) -> tuple[int, int]:
        delay = _index(delay_index, "delay_index")
        angle = _index(angle_index, "angle_index")
        if not 0 <= delay < self.values.shape[0] or not 0 <= angle < self.values.shape[1]:
            raise IndexError("histogram bin index is out of range")
        return delay, angle

    def path_ids_for_bin(self, delay_index: int, angle_index: int) -> np.ndarray:
        """Canonical IDs contributing to a bin, in deterministic source order."""
        delay, angle = self._check_bin(delay_index, angle_index)
        selected = (self.delay_bin_indices == delay) & (self.angle_bin_indices == angle)
        return _immutable(self.path_ids[selected])

    def row_indices_for_bin(self, delay_index: int, angle_index: int) -> np.ndarray:
        """Rows in the source PathAnalysisData contributing to a selected bin."""
        delay, angle = self._check_bin(delay_index, angle_index)
        selected = (self.delay_bin_indices == delay) & (self.angle_bin_indices == angle)
        return _immutable(self.row_indices[selected])

    def delay_interval_ns(self, delay_index: int) -> tuple[float, float]:
        """Return the selected bin's [lower, upper) delay interval in ns."""
        delay, _ = self._check_bin(delay_index, 0)
        return float(self.delay_edges_ns[delay]), float(self.delay_edges_ns[delay + 1])

    def angle_interval_deg(self, angle_index: int) -> tuple[float, float]:
        """Return the selected bin's [lower, upper) wrapped angular interval."""
        _, angle = self._check_bin(0, angle_index)
        return float(self.angle_edges_deg[angle]), float(self.angle_edges_deg[angle + 1])


@dataclass(frozen=True, slots=True)
class GeometricProgress(_ImmutableArrays):
    """Bounded geometry-only marker positions after emission at time zero.

    Positions clamp to the TX before emission and RX after geometric arrival.
    Invalid geometry yields NaN positions/times and false ``valid``/``arrived``.
    ``geometric_arrival_ns`` is arc length / c, independent of exported delays.
    Output order preserves caller order; a deterministic prefix meets both caps.
    """

    time_ns: float
    path_ids: np.ndarray
    positions: np.ndarray
    geometric_arrival_ns: np.ndarray
    valid: np.ndarray
    arrived: np.ndarray
    active: np.ndarray
    omitted_count: int
    disclosures: tuple[str, ...]


def _index(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def select_paths(
    catalog: MpcPathCatalog,
    *,
    tx_id: int,
    rx_id: int,
    scope: MpcPathScope | str = MpcPathScope.ALL,
    weight_mode: PathWeightMode | str = PathWeightMode.COUNT,
) -> PathAnalysisData:
    """Select exactly one known TX/RX pair without changing frame or UI state."""
    tx = _index(tx_id, "tx_id")
    rx = _index(rx_id, "rx_id")
    if tx < 0 or rx < 0:
        raise ValueError("tx_id and rx_id must be known, nonnegative IDs")
    selected_scope = MpcPathScope(scope)
    mode = PathWeightMode(weight_mode)
    selected = catalog.scope_mask(selected_scope) & (catalog.tx_ids == tx) & (catalog.rx_ids == rx)
    ids: np.ndarray = np.flatnonzero(selected).astype(np.int32)
    delays = np.asarray(catalog.delays_ns[ids], dtype=np.float64)
    losses = np.asarray(catalog.path_losses_db[ids], dtype=np.float64)
    delay_codes = catalog.delay_provenance_codes[ids]
    loss_codes = catalog.path_loss_provenance_codes[ids]
    angles = {name: getattr(catalog, f"{name}_deg")[ids] for name in _ANGLE_FIELDS}
    disclosures: list[str] = []
    for label, values, provenance in (
        ("delay", delays, delay_codes),
        ("loss", losses, loss_codes),
    ):
        for explanation, mask in (
            ("missing/nonfinite", ~np.isfinite(values)),
            ("estimated", np.isfinite(values) & (provenance == 1)),
            ("unknown provenance", np.isfinite(values) & (provenance == -1)),
        ):
            count = int(np.count_nonzero(mask))
            if count:
                disclosures.append(f"{count} path(s): {label} {explanation}.")
    for name, values in angles.items():
        count = int(np.count_nonzero(~np.isfinite(values)))
        if count:
            disclosures.append(f"{count} path(s): {name.replace('_', ' ')} missing/nonfinite.")
    weights: np.ndarray = np.ones(ids.size, dtype=np.float64)
    if mode is PathWeightMode.POWER:
        eligible = np.isfinite(losses) & (loss_codes == 0)
        weights.fill(0.0)
        if np.any(eligible):
            with np.errstate(over="ignore", under="ignore"):
                weights[eligible] = np.power(
                    10.0, -(losses[eligible] - losses[eligible].min()) / 10.0
                )
        excluded = int(np.count_nonzero(~eligible))
        if excluded:
            disclosures.append(
                f"{excluded} path(s) excluded from power weights: finite loss with explicit "
                "exported provenance is required."
            )
        underflow = int(np.count_nonzero(eligible & (weights == 0.0)))
        if underflow:
            disclosures.append(f"{underflow} exported-loss proxy weight(s) underflowed to zero.")
        disclosures.append("Relative path-power proxy only; no received-power or coherence model.")
    return PathAnalysisData(
        tx_id=tx,
        rx_id=rx,
        scope=selected_scope,
        weight_mode=mode,
        path_ids=ids,
        delays_ns=delays,
        **{f"{name}_deg": values for name, values in angles.items()},
        weights=weights,
        delay_provenance_codes=delay_codes,
        path_loss_provenance_codes=loss_codes,
        disclosures=tuple(disclosures),
    )


def weighted_delay_percentiles(
    data: PathAnalysisData,
    percentiles: tuple[float, ...] = (10.0, 50.0, 90.0),
    *,
    row_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse weighted CDF, in ns; default summarizes the whole pair/scope.

    A percentile is the first observed delay whose cumulative positive weight
    reaches the requested percentage (no invented interpolated path). Unknown
    and estimated delays may contribute and are disclosed by ``select_paths``.
    Missing/nonfinite delays and zero weights are excluded. Empty results are NaN.
    """
    levels: np.ndarray = np.asarray(percentiles, dtype=np.float64)
    if levels.ndim != 1 or np.any(~np.isfinite(levels)) or np.any((levels < 0) | (levels > 100)):
        raise ValueError("percentiles must be finite values between 0 and 100")
    if row_indices is None:
        delays = data.delays_ns
        weights = data.weights
    else:
        rows = _checked_ids(row_indices, data.path_count, "row_indices")
        delays = data.delays_ns[rows]
        weights = data.weights[rows]
    valid = np.isfinite(delays) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return _immutable(np.full(levels.shape, np.nan))
    order = np.argsort(delays[valid], kind="stable")
    sorted_delays = delays[valid][order]
    cumulative: np.ndarray = np.cumsum(weights[valid][order], dtype=np.float64)
    targets = cumulative[-1] * (levels / 100.0)
    indices = np.minimum(np.searchsorted(cumulative, targets, side="left"), sorted_delays.size - 1)
    return _immutable(sorted_delays[indices])


def _gate_bounds(lower_ns: float, upper_ns: float) -> tuple[float, float]:
    lower, upper = float(lower_ns), float(upper_ns)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("gate bounds must be finite and lower_ns <= upper_ns")
    return lower, upper


def arrival_gate(data: PathAnalysisData, lower_ns: float, upper_ns: float) -> ArrivalGateResult:
    """Return all finite-delay paths inside an inclusive delay interval."""
    lower, upper = _gate_bounds(lower_ns, upper_ns)
    rows = np.flatnonzero(
        np.isfinite(data.delays_ns) & (data.delays_ns >= lower) & (data.delays_ns <= upper)
    )
    gate_weights = data.weights[rows]
    total = float(np.sum(gate_weights, dtype=np.float64))
    denominator = float(np.sum(data.weights, dtype=np.float64))
    return ArrivalGateResult(
        lower_ns=lower,
        upper_ns=upper,
        path_ids=data.path_ids[rows],
        row_indices=rows,
        weighted_path_count=int(np.count_nonzero(gate_weights > 0.0)),
        total_weight=total,
        weight_share=total / denominator if denominator > 0.0 else 0.0,
        percentiles_ns=weighted_delay_percentiles(data, row_indices=rows),
    )


def delay_angle_histogram(
    data: PathAnalysisData,
    *,
    angle: str = "aoa_azimuth",
    angular_bin_width_deg: float = 10.0,
    delay_bin_width_ns: float = 10.0,
    gate_ns: tuple[float, float] | None = None,
    max_cells: int = 262_144,
) -> DelayAngleHistogram:
    """Build bounded numerical bins and deterministic row/path membership.

    Requested resolutions are retained; over-budget grids raise ValueError
    rather than silently widening bins. The terminal angular bin may be narrower.
    A finite inclusive gate is optional and independent of histogram bin edges.
    Empty histograms use one [0, delay_bin_width_ns) delay bin.
    """
    angles = data.angles_deg(angle)
    angular_width = float(angular_bin_width_deg)
    delay_width = float(delay_bin_width_ns)
    if not np.isfinite(angular_width) or angular_width <= 0.0:
        raise ValueError("angular_bin_width_deg must be finite and positive")
    if not np.isfinite(delay_width) or delay_width <= 0.0:
        raise ValueError("delay_bin_width_ns must be finite and positive")
    budget = _index(max_cells, "max_cells")
    if budget < 1 or budget > 1_048_576:
        raise ValueError("max_cells must be between 1 and 1048576")
    angular_ratio = 360.0 / angular_width
    if not np.isfinite(angular_ratio) or angular_ratio > budget:
        raise ValueError("histogram exceeds max_cells; increase angular bin width")
    n_angle = max(1, int(np.ceil(angular_ratio)))
    gate: np.ndarray = np.ones(data.path_count, dtype=bool)
    if gate_ns is not None:
        lower, upper = _gate_bounds(*gate_ns)
        gate &= np.isfinite(data.delays_ns) & (data.delays_ns >= lower) & (data.delays_ns <= upper)
    valid = gate & np.isfinite(data.delays_ns) & np.isfinite(angles) & (data.weights > 0.0)
    rows = np.flatnonzero(valid)
    delay_values = data.delays_ns[rows]
    if rows.size:
        with np.errstate(over="ignore", invalid="ignore"):
            start = np.floor(delay_values.min() / delay_width) * delay_width
            if start > delay_values.min():
                start -= delay_width
            delay_ratio = (delay_values.max() - start) / delay_width
        if (
            not np.isfinite(start)
            or not np.isfinite(delay_ratio)
            or delay_ratio >= budget // n_angle
        ):
            raise ValueError("histogram exceeds max_cells; increase delay bin width")
        n_delay = max(1, int(np.floor(delay_ratio)) + 1)
    else:
        start, n_delay = 0.0, 1
    if n_delay * n_angle > budget:
        raise ValueError("histogram exceeds max_cells; increase bin widths")
    with np.errstate(over="ignore", invalid="ignore"):
        delay_edges = start + np.arange(n_delay + 1, dtype=np.float64) * delay_width
        # Division can round the maximum into the preceding bin while edge
        # addition lands exactly on it. Keep the terminal edge strictly above
        # every contributor so mappings honor the advertised [lower, upper).
        if rows.size and delay_edges[-1] <= delay_values.max():
            n_delay += 1
            if n_delay * n_angle > budget:
                raise ValueError("histogram exceeds max_cells; increase bin widths")
            delay_edges = np.append(delay_edges, start + n_delay * delay_width)
    angle_edges = np.minimum(-180.0 + np.arange(n_angle + 1) * min(angular_width, 360.0), 180.0)
    angle_edges[-1] = 180.0
    if np.any(~np.isfinite(delay_edges)) or np.any(np.diff(delay_edges) <= 0.0):
        raise ValueError("delay bin width cannot resolve the magnitude of the delays")
    # Avoid adding 180 before the remainder: it can round the representable
    # value immediately below +180 onto the wrap boundary.
    wrapped = np.fmod(np.asarray(angles[rows], dtype=np.float64), 360.0)
    wrapped[wrapped < -180.0] += 360.0
    wrapped[wrapped >= 180.0] -= 360.0
    # Both queries are arrays. Scoped checking can lose NumPy's imported
    # scalar/array overload aliases, so retain the known array result type.
    angle_bins: np.ndarray = (
        cast(np.ndarray, np.searchsorted(angle_edges, wrapped, side="right")) - 1
    )
    delay_bins: np.ndarray = (
        cast(np.ndarray, np.searchsorted(delay_edges, delay_values, side="right")) - 1
    )
    linear_bins = delay_bins * n_angle + angle_bins
    values = np.bincount(
        linear_bins, weights=data.weights[rows], minlength=n_delay * n_angle
    ).reshape(n_delay, n_angle)
    mass = np.sort(values[values > 0.0])[::-1]
    total = float(np.sum(mass, dtype=np.float64))
    min_bins_90 = (
        min(int(np.searchsorted(np.cumsum(mass), 0.9 * total)) + 1, int(mass.size)) if total else 0
    )
    excluded = int(np.count_nonzero(gate & ~valid))
    disclosures = list(data.disclosures)
    if excluded:
        disclosures.append(
            f"{excluded} in-scope/gate path(s) excluded from bins: nonfinite delay/angle or zero weight."
        )
    disclosures.append(
        "Bins are numerical groups, not physical clusters; shares use binned weight only."
    )
    return DelayAngleHistogram(
        angle=angle,
        values=values,
        delay_edges_ns=delay_edges,
        angle_edges_deg=angle_edges,
        path_ids=data.path_ids[rows],
        row_indices=rows,
        delay_bin_indices=delay_bins,
        angle_bin_indices=angle_bins,
        source_path_count=data.path_count,
        gated_path_count=int(np.count_nonzero(gate)),
        excluded_path_count=excluded,
        total_weight=total,
        dominant_bin_share=float(mass[0] / total) if total else 0.0,
        occupied_bins=int(mass.size),
        min_bins_90=min_bins_90,
        disclosures=tuple(disclosures),
    )


def _checked_ids(values: np.ndarray, count: int, name: str) -> np.ndarray:
    ids: np.ndarray = np.asarray(values)
    if ids.ndim != 1 or (ids.size and not np.issubdtype(ids.dtype, np.integer)):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(ids < 0) or np.any(ids >= count):
        raise ValueError(f"{name} contains out-of-range IDs")
    return ids.astype(np.intp, copy=False)


def geometric_progress(
    catalog: MpcPathCatalog,
    path_ids: np.ndarray,
    time_ns: float,
    *,
    max_paths: int = 2_000,
    max_points: int = 1_000_000,
    _geometry_cache: dict | None = None,
) -> GeometricProgress:
    """Vectorized propagation along path polylines at c, never exported delay.

    No Python path loop or dense path-by-max-depth matrix is used. Both output
    path count and temporary point storage are bounded. A point-budget overflow
    truncates the prefix before the first path that would exceed that budget.
    """
    moment = float(time_ns)
    if not np.isfinite(moment):
        raise ValueError("time_ns must be finite")
    path_limit = _index(max_paths, "max_paths")
    point_limit = _index(max_points, "max_points")
    if not 1 <= path_limit <= 100_000 or not 2 <= point_limit <= 2_000_000:
        raise ValueError("max_paths must be 1..100000 and max_points must be 2..2000000")
    supplied_ids = _checked_ids(path_ids, catalog.path_count, "path_ids")
    ids = supplied_ids[:path_limit]
    starts = catalog.path_start_indices[ids].astype(np.intp, copy=False)
    ends = catalog.path_end_indices[ids].astype(np.intp, copy=False)
    counts = ends - starts
    if np.any(starts < 0) or np.any(ends > catalog.point_count) or np.any(counts < 2):
        raise ValueError("selected paths have invalid canonical point bounds")
    keep = int(np.count_nonzero(np.cumsum(counts, dtype=np.int64) <= point_limit))
    ids, starts, counts = ids[:keep], starts[:keep], counts[:keep]
    omitted = int(supplied_ids.size - ids.size)
    disclosures: list[str] = []
    if omitted:
        disclosures.append(f"{omitted} path(s) omitted by geometric marker/point budgets.")
    positions: np.ndarray = np.full((ids.size, 3), np.nan, dtype=np.float64)
    arrivals: np.ndarray = np.full(ids.size, np.nan, dtype=np.float64)
    valid: np.ndarray = np.zeros(ids.size, dtype=bool)
    arrived: np.ndarray = np.zeros(ids.size, dtype=bool)
    active: np.ndarray = np.zeros(ids.size, dtype=bool)
    if ids.size:
        cache_key = (
            (id(catalog), tuple(int(v) for v in ids), point_limit)
            if _geometry_cache is not None
            else None
        )
        cached = (
            _geometry_cache.get("geometry")
            if _geometry_cache is not None and _geometry_cache.get("key") == cache_key
            else None
        )
        if cached is not None:
            (
                local_starts,
                local_ends,
                point_rows,
                points,
                lengths,
                cumulative,
                total_lengths,
                valid,
                arrivals,
            ) = cached
            valid, arrivals = valid.copy(), arrivals.copy()
        else:
            local_starts = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.intp)))
            local_ends = local_starts + counts
            point_rows: np.ndarray = np.repeat(np.arange(ids.size), counts)
            offsets = np.arange(int(counts.sum())) - np.repeat(local_starts, counts)
            source_indices = np.repeat(starts, counts) + offsets
            points = np.asarray(catalog.canonical_data.points[source_indices], dtype=np.float64)
            valid = np.logical_and.reduceat(np.all(np.isfinite(points), axis=1), local_starts)
            with np.errstate(over="ignore", invalid="ignore"):
                deltas = np.diff(points, axis=0)
                lengths = np.hypot.reduce(deltas, axis=1)
            lengths[local_ends[:-1] - 1] = 0.0  # Never connect different paths.
            segment_valid = np.isfinite(lengths)
            bad_segments = np.flatnonzero(~segment_valid)
            valid[point_rows[bad_segments]] = False
            lengths[~segment_valid] = 0.0
            # Segmented prefix scan keeps each path's arc lengths independent.
            # Subtracting a global cumulative origin loses short-path distances
            # after a long path. This bounded scan uses O(points) storage and only
            # O(log(max path depth)) vectorized passes, never a Python path loop.
            cumulative = np.concatenate(([0.0], lengths))
            offset = 1
            with np.errstate(over="ignore", invalid="ignore"):
                while offset < int(counts.max()):
                    same_path = point_rows[offset:] == point_rows[:-offset]
                    cumulative[offset:] += np.where(same_path, cumulative[:-offset], 0.0)
                    offset *= 2
            total_lengths = cumulative[local_ends - 1]
            valid &= np.isfinite(total_lengths)
            with np.errstate(over="ignore"):
                arrivals[valid] = total_lengths[valid] / SPEED_OF_LIGHT_M_PER_NS
            valid &= np.isfinite(arrivals)
            arrivals[~valid] = np.nan
            if _geometry_cache is not None:
                _geometry_cache.clear()
                _geometry_cache.update(
                    key=cache_key,
                    geometry=(
                        local_starts,
                        local_ends,
                        point_rows,
                        points,
                        lengths,
                        cumulative,
                        total_lengths,
                        valid,
                        arrivals,
                    ),
                )
                valid, arrivals = valid.copy(), arrivals.copy()
        # c in metres/ns is less than one, so finite nonnegative time cannot
        # overflow this multiplication.
        traveled = max(moment, 0.0) * SPEED_OF_LIGHT_M_PER_NS
        targets = np.minimum(traveled, total_lengths)
        segment = np.maximum.reduceat(
            np.where(cumulative <= targets[point_rows], np.arange(points.shape[0]), -1),
            local_starts,
        )
        segment = np.minimum(np.maximum(segment, local_starts), local_ends - 2)
        fraction: np.ndarray = np.zeros(ids.size, dtype=np.float64)
        np.divide(
            targets - cumulative[segment],
            lengths[segment],
            out=fraction,
            where=lengths[segment] > 0.0,
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        positions[valid] = points[segment[valid]] + fraction[valid, None] * (
            points[segment[valid] + 1] - points[segment[valid]]
        )
        # Endpoint assignment is exact, including repeated/zero-length segments.
        if moment <= 0.0:
            positions[valid] = points[local_starts[valid]]
        arrived = valid & (moment >= arrivals)
        positions[arrived] = points[(local_ends - 1)[arrived]]
        active = valid & (moment >= 0.0) & ~arrived
        invalid_count = int(np.count_nonzero(~valid))
        if invalid_count:
            disclosures.append(f"{invalid_count} path(s) have nonfinite/invalid geometry.")
    return GeometricProgress(
        time_ns=moment,
        path_ids=ids.astype(np.int32),
        positions=positions,
        geometric_arrival_ns=arrivals,
        valid=valid,
        arrived=arrived,
        active=active,
        omitted_count=omitted,
        disclosures=tuple(disclosures),
    )
