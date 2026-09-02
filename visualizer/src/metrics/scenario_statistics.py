"""Pure scenario statistics aggregation from selective frame projections.

The accumulator consumes only compact topology, interaction, and path-metric
arrays. It has no Qt, renderer, canonical-geometry, or storage dependency, so
a provider can stream projections through it without filling the visualizer's
playback and ViewModel caches.

Metric availability comes exclusively from ``metric_valid_bits``.  A measured
zero is therefore retained, while an unavailable NaN is omitted.  Delay-spread
calculations use the intersection of the delay and path-loss validity masks so
values belonging to different paths are never paired accidentally.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sized
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shared.frames import (
    PATH_METRIC_ARRAY_FIELDS,
    FrameProjection,
    ProjectedMPCFrame,
)
from shared.frames.contracts import (
    PATH_METRIC_ORDER,
    FrameReadRequest,
    PathMetric,
)
from shared.statistics.core.metrics import compute_delay_spread

ProgressCallback = Callable[[int, int], None]

SCENARIO_STATISTICS_SCHEMA_VERSION = 3
"""Version of the persisted result semantics produced by this accumulator."""

# These are the interaction types that have an explicit ORCHAV/Sionna
# interpretation.  Virtual paths (99) are authored data and must not be folded
# into the unknown or LoS categories.
KNOWN_MPC_TYPES = (0, 1, 2, 4, 8, 99)
UNKNOWN_MPC_TYPE = -1
"""Stable statistics-only bucket for an unrecognized first interaction code."""

MPC_TYPE_BUCKETS = (*KNOWN_MPC_TYPES, UNKNOWN_MPC_TYPE)
_EVOLUTION_REFLECTION_ORDERS = tuple(range(7))
PAIR_VISIBILITY_CATEGORIES = (
    "direct_path_present",
    "indirect_only",
    "no_path",
)

SCENARIO_STATISTICS_REQUEST = FrameReadRequest.for_metrics(
    PATH_METRIC_ORDER,
    include_interactions=True,
)
"""Smallest projection that supplies every current statistics chart."""

FloatArray = NDArray[np.floating[Any]]
Float64Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _FrameStatistics:
    """Intermediate statistics for one projected frame."""

    mpc_count: int
    reflection_order_counts: dict[int, int]
    mpc_type_counts: dict[int, int]
    path_losses: FloatArray
    delays: FloatArray
    aoa_az_values: FloatArray
    aoa_el_values: FloatArray
    aod_az_values: FloatArray
    aod_el_values: FloatArray
    joint_delays: FloatArray
    joint_path_losses: FloatArray
    joint_metrics_aligned: bool
    delay_spread: float
    pair_visibility_counts: dict[str, int]
    pair_aggregate_path_gains_db: FloatArray
    pair_rms_delay_spreads_ns: FloatArray
    strongest_single_path_loss_db: float


def _empty_float64() -> Float64Array:
    """Return a consistently typed empty metric vector."""

    return np.empty((0,), dtype=np.float64)


def _normalize_azimuth_deg(values: Float64Array) -> Float64Array:
    """Normalize azimuth values to the visualizer's ``[-180, 180)`` convention."""

    return ((values + 180.0) % 360.0) - 180.0


def _counts(values: NDArray[np.integer[Any]]) -> dict[int, int]:
    """Return a Python-int histogram for an integer vector."""

    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return {int(value): int(count) for value, count in zip(unique, counts, strict=True)}


def _reflection_order_counts(
    reflection_orders: NDArray[np.integer[Any]],
) -> dict[int, int]:
    """Count orders 0 through 5 and combine every higher order as ``6+``."""

    counts = _counts(reflection_orders)
    overflow = sum(count for order, count in counts.items() if order >= 6)
    result = {order: count for order, count in counts.items() if 0 <= order < 6}
    if overflow:
        result[6] = overflow
    return result


def _metric_values(
    frame: ProjectedMPCFrame,
    metric: PathMetric,
    valid: NDArray[np.bool_] | None = None,
) -> FloatArray:
    """Return valid values while retaining the metric's native precision.

    Statistics are published as float64 arrays during :meth:`finalize`. Keeping
    the streamed chunks in compact float32 form until then avoids retaining a
    float64 copy and another float64 concatenation target at the same time.
    """

    values = getattr(frame, PATH_METRIC_ARRAY_FIELDS[metric])
    if values is None:
        raise ValueError(f"Scenario statistics projection is missing {metric.value}")
    validity = frame.metric_is_valid(metric) if valid is None else valid
    if bool(np.all(validity)):
        return values
    return values[validity]


def _joint_delay_loss(
    frame: ProjectedMPCFrame,
    *,
    delays: FloatArray,
    path_losses: FloatArray,
    delay_valid: NDArray[np.bool_],
    path_loss_valid: NDArray[np.bool_],
) -> tuple[FloatArray, FloatArray, bool]:
    """Return co-valid delay/loss values and whether metric vectors already align.

    Complete generated frames normally use the same validity mask for both
    metrics. In that common case the joint vectors can alias the retained metric
    chunks instead of duplicating them. Asymmetric authored data still gets an
    explicitly aligned pair of vectors.
    """

    raw_delays = frame.delays_ns
    raw_path_losses = frame.path_loss_db
    if raw_delays is None or raw_path_losses is None:
        raise ValueError("Scenario statistics projection requires delay and path loss")
    if np.array_equal(delay_valid, path_loss_valid):
        return delays, path_losses, True

    joint_valid = delay_valid & path_loss_valid
    return (
        raw_delays[joint_valid],
        raw_path_losses[joint_valid],
        False,
    )


def _first_interaction_types(
    bounce_offsets: NDArray[np.integer[Any]],
    interactions: NDArray[np.integer[Any]],
) -> NDArray[np.int32]:
    """Classify paths by their first physical bounce interaction."""

    orders = np.diff(bounce_offsets)
    result = np.zeros((len(orders),), dtype=np.int32)
    bounced = orders > 0
    if not np.any(bounced):
        return result

    starts = bounce_offsets[:-1][bounced].astype(np.intp, copy=False)
    first_types = interactions[starts].astype(np.int32, copy=False)
    recognized = np.isin(first_types, np.asarray(KNOWN_MPC_TYPES[1:], dtype=np.int32))
    result[bounced] = np.where(recognized, first_types, UNKNOWN_MPC_TYPE)
    return result


def _aggregate_path_gain_db(path_losses_db: FloatArray) -> float:
    """Return the incoherent aggregate path-gain ratio in dB."""

    if not path_losses_db.size:
        return float("nan")
    losses = path_losses_db.astype(np.float64, copy=False)
    strongest_loss = float(np.min(losses))
    relative_power = np.sum(10.0 ** (-(losses - strongest_loss) / 10.0))
    return float(-strongest_loss + 10.0 * np.log10(relative_power))


def _pair_frame_metrics(
    frame: ProjectedMPCFrame,
    reflection_orders: NDArray[np.integer[Any]],
    delay_valid: NDArray[np.bool_],
    path_loss_valid: NDArray[np.bool_],
) -> tuple[dict[str, int], Float64Array, Float64Array]:
    """Compute channel statistics for every represented TX/RX pair.

    Generator frames represent the full Cartesian TX/RX topology. External
    producers may intentionally store a subset, so the categories cover only
    rows present in ``tx_rx_pairs`` and ``pair_path_offsets``. Aggregate gain
    uses every path with valid measured loss; RMS delay spread uses only paths
    whose measured delay and loss are both valid.
    """

    pair_offsets = frame.pair_path_offsets
    pairs = frame.tx_rx_pairs
    raw_delays = frame.delays_ns
    raw_losses = frame.path_loss_db
    if pair_offsets is None or pairs is None:
        raise ValueError("Scenario statistics projection requires pair topology")
    if raw_delays is None or raw_losses is None:
        raise ValueError("Scenario statistics projection requires delay and path loss")

    visibility = {category: 0 for category in PAIR_VISIBILITY_CATEGORIES}
    aggregate_gains: list[float] = []
    delay_spreads: list[float] = []
    for pair_index in range(len(pairs)):
        start = int(pair_offsets[pair_index])
        stop = int(pair_offsets[pair_index + 1])
        if start == stop:
            visibility["no_path"] += 1
        elif np.any(reflection_orders[start:stop] == 0):
            visibility["direct_path_present"] += 1
        else:
            visibility["indirect_only"] += 1

        valid_losses = path_loss_valid[start:stop]
        if np.any(valid_losses):
            pair_losses = raw_losses[start:stop][valid_losses]
            aggregate_gains.append(_aggregate_path_gain_db(pair_losses))

        joint_valid = delay_valid[start:stop] & path_loss_valid[start:stop]
        if np.any(joint_valid):
            pair_delays = raw_delays[start:stop][joint_valid].astype(np.float64, copy=False)
            pair_losses = raw_losses[start:stop][joint_valid].astype(np.float64, copy=False)
            pair_powers = 10.0 ** (-pair_losses / 10.0)
            delay_spreads.append(compute_delay_spread(pair_delays, pair_powers))

    return (
        visibility,
        np.asarray(aggregate_gains, dtype=np.float64),
        np.asarray(delay_spreads, dtype=np.float64),
    )


class ScenarioStatisticsAccumulator:
    """Incrementally aggregate scenario statistics from compact projections.

    Call :meth:`add_projection` as projections arrive, followed by
    :meth:`finalize`, or use :meth:`collect_from_projections` for the complete
    operation.  One accumulator represents one collection pass and cannot be
    appended to after finalization.

    The result keeps the dictionary keys and array-oriented payload expected by
    the statistics panel. ``unique_tx_count`` and ``unique_rx_count`` count the
    device IDs participating in the stored pair topology.
    """

    def __init__(self) -> None:
        """Initialize an empty, streamable statistics pass."""

        self._finalized = False
        self._stats: dict[str, Any] = {
            "total_mpcs": 0,
            "total_frames": 0,
            "reflection_order_dist": {},
            "mpc_type_dist": {},
            "path_loss_values": [],
            "delay_values": [],
            "aoa_az_values": [],
            "aoa_el_values": [],
            "aod_az_values": [],
            "aod_el_values": [],
            "tx_rx_pairs": set(),
            "unique_tx_count": 0,
            "unique_rx_count": 0,
            "mpc_evolution": [],
            "reflection_order_evolution_per_frame": {
                order: [] for order in _EVOLUTION_REFLECTION_ORDERS
            },
            "mpc_type_evolution_per_frame": {mpc_type: [] for mpc_type in MPC_TYPE_BUCKETS},
            "frame_indices": [],
            "delay_spread_evolution": [],
            "pair_visibility_counts": {category: 0 for category in PAIR_VISIBILITY_CATEGORIES},
            "pair_visibility_evolution": {category: [] for category in PAIR_VISIBILITY_CATEGORIES},
            "direct_path_pair_share_evolution": [],
            "pair_aggregate_path_gain_db_values": [],
            "pair_rms_delay_spread_ns_values": [],
            "strongest_single_path_loss_evolution": [],
        }
        self._tx_ids: set[int] = set()
        self._rx_ids: set[int] = set()
        self._joint_delay_chunks: list[FloatArray] = []
        self._joint_path_loss_chunks: list[FloatArray] = []
        self._joint_metrics_aligned = True
        self.stats: dict[str, Any] | None = None

    def add_projection(self, projection: FrameProjection) -> None:
        """Add one topology/interactions/metrics projection to the scenario."""

        if self._finalized:
            raise RuntimeError("Cannot add a projection after statistics were finalized")
        self._require_statistics_projection(projection)

        frame = projection.frame
        frame_stats = self._collect_frame_stats(frame)
        stats = self._stats

        stats["total_frames"] += 1
        stats["total_mpcs"] += frame_stats.mpc_count
        stats["frame_indices"].append(int(frame.frame_index))
        stats["mpc_evolution"].append(frame_stats.mpc_count)

        for order in _EVOLUTION_REFLECTION_ORDERS:
            stats["reflection_order_evolution_per_frame"][order].append(
                frame_stats.reflection_order_counts.get(order, 0)
            )
        for mpc_type in MPC_TYPE_BUCKETS:
            stats["mpc_type_evolution_per_frame"][mpc_type].append(
                frame_stats.mpc_type_counts.get(mpc_type, 0)
            )

        self._merge_counts(
            stats["reflection_order_dist"],
            frame_stats.reflection_order_counts,
        )
        self._merge_counts(stats["mpc_type_dist"], frame_stats.mpc_type_counts)

        for result_key, values in (
            ("path_loss_values", frame_stats.path_losses),
            ("delay_values", frame_stats.delays),
            ("aoa_az_values", frame_stats.aoa_az_values),
            ("aoa_el_values", frame_stats.aoa_el_values),
            ("aod_az_values", frame_stats.aod_az_values),
            ("aod_el_values", frame_stats.aod_el_values),
        ):
            if values.size:
                stats[result_key].append(values)

        if frame_stats.joint_delays.size:
            self._joint_delay_chunks.append(frame_stats.joint_delays)
            self._joint_path_loss_chunks.append(frame_stats.joint_path_losses)
        self._joint_metrics_aligned &= frame_stats.joint_metrics_aligned

        pairs = frame.tx_rx_pairs
        if pairs is None:
            raise ValueError("Scenario statistics projection is missing tx_rx_pairs")
        for tx_index, rx_index in pairs:
            tx_id = int(tx_index)
            rx_id = int(rx_index)
            stats["tx_rx_pairs"].add((tx_id, rx_id))
            self._tx_ids.add(tx_id)
            self._rx_ids.add(rx_id)

        stats["delay_spread_evolution"].append(frame_stats.delay_spread)
        pair_count = sum(frame_stats.pair_visibility_counts.values())
        direct_count = frame_stats.pair_visibility_counts["direct_path_present"]
        stats["direct_path_pair_share_evolution"].append(
            float(direct_count) / pair_count if pair_count else float("nan")
        )
        for category in PAIR_VISIBILITY_CATEGORIES:
            count = frame_stats.pair_visibility_counts[category]
            stats["pair_visibility_counts"][category] += count
            stats["pair_visibility_evolution"][category].append(count)
        if frame_stats.pair_aggregate_path_gains_db.size:
            stats["pair_aggregate_path_gain_db_values"].append(
                frame_stats.pair_aggregate_path_gains_db
            )
        if frame_stats.pair_rms_delay_spreads_ns.size:
            stats["pair_rms_delay_spread_ns_values"].append(frame_stats.pair_rms_delay_spreads_ns)
        stats["strongest_single_path_loss_evolution"].append(
            frame_stats.strongest_single_path_loss_db
        )

    def collect_from_projections(
        self,
        projections: Iterable[FrameProjection],
        *,
        total_frames: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Consume an iterable of projections and return finalized statistics.

        ``total_frames`` is needed only for progress reporting when
        ``projections`` has no length.  When supplied, it is also checked
        against the number actually consumed so a truncated scan cannot be
        mistaken for a complete result.
        """

        if self._stats["total_frames"] or self._finalized:
            raise RuntimeError("This accumulator has already started a collection pass")

        expected_total = total_frames
        if expected_total is None and isinstance(projections, Sized):
            expected_total = len(projections)
        if expected_total is not None and expected_total < 0:
            raise ValueError("total_frames must be non-negative")
        if on_progress is not None and expected_total is None:
            raise ValueError("total_frames is required for progress on a streaming iterable")

        for current, projection in enumerate(projections, start=1):
            self.add_projection(projection)
            if on_progress is not None:
                assert expected_total is not None
                on_progress(current, expected_total)

        actual_total = int(self._stats["total_frames"])
        if expected_total is not None and actual_total != expected_total:
            raise ValueError(
                f"Expected {expected_total} frame projections, received {actual_total}"
            )
        return self.finalize()

    def finalize(self) -> dict[str, Any]:
        """Finalize arrays and derived summaries for the accumulated scenario."""

        if self._finalized:
            assert self.stats is not None
            return self.stats

        stats = self._stats
        stats["unique_tx_count"] = len(self._tx_ids)
        stats["unique_rx_count"] = len(self._rx_ids)
        stats["unique_tx_rx_pairs"] = len(stats["tx_rx_pairs"])
        frame_count = int(stats["total_frames"])
        stats["avg_mpcs_per_frame"] = float(stats["total_mpcs"]) / frame_count if frame_count else 0
        stats["tx_rx_pairs"] = sorted(stats["tx_rx_pairs"])

        path_losses = self._finalize_metric(stats, "path_loss_values")
        delays = self._finalize_metric(stats, "delay_values")
        for key in (
            "aoa_az_values",
            "aoa_el_values",
            "aod_az_values",
            "aod_el_values",
        ):
            self._finalize_metric(stats, key)

        stats["path_loss_stats"] = self._summary(path_losses)
        stats["delay_stats"] = self._summary(delays)

        pair_gains = self._finalize_metric(
            stats,
            "pair_aggregate_path_gain_db_values",
        )
        pair_delay_spreads = self._finalize_metric(
            stats,
            "pair_rms_delay_spread_ns_values",
        )
        stats["pair_aggregate_path_gain_stats"] = self._percentile_summary(pair_gains)
        stats["pair_rms_delay_spread_stats"] = self._percentile_summary(pair_delay_spreads)
        pair_state_total = sum(stats["pair_visibility_counts"].values())
        stats["pair_visibility_summary"] = {
            category: {
                "count": stats["pair_visibility_counts"][category],
                "percent": (
                    100.0 * stats["pair_visibility_counts"][category] / pair_state_total
                    if pair_state_total
                    else 0.0
                ),
            }
            for category in PAIR_VISIBILITY_CATEGORIES
        }

        if self._joint_delay_chunks:
            if self._joint_metrics_aligned:
                joint_delays = delays
                joint_losses = path_losses
            else:
                joint_delays = np.concatenate(
                    self._joint_delay_chunks,
                    dtype=np.float64,
                )
                joint_losses = np.concatenate(
                    self._joint_path_loss_chunks,
                    dtype=np.float64,
                )
            # The finalized vectors own every value needed by the
            # delay-spread calculation, so the streamed references can be
            # released before its temporary workspaces are allocated.
            self._joint_delay_chunks.clear()
            self._joint_path_loss_chunks.clear()
            joint_powers = 10.0 ** (-joint_losses / 10.0)
            stats["overall_delay_spread"] = compute_delay_spread(
                joint_delays,
                joint_powers,
            )
        else:
            stats["overall_delay_spread"] = None

        if len(stats["mpc_evolution"]) > 1:
            mpc_counts = np.asarray(stats["mpc_evolution"], dtype=np.float64)
            mean_mpcs = float(np.mean(mpc_counts))
            stats["mpc_count_variation_coeff"] = (
                float(np.std(mpc_counts) / mean_mpcs) if mean_mpcs > 0.0 else 0.0
            )
        else:
            stats["mpc_count_variation_coeff"] = 0.0

        self._finalized = True
        self.stats = stats
        return stats

    @staticmethod
    def _require_statistics_projection(projection: FrameProjection) -> None:
        """Reject projections that omit a statistics input."""

        if projection.satisfies(SCENARIO_STATISTICS_REQUEST):
            return
        missing_components = SCENARIO_STATISTICS_REQUEST.components - projection.loaded_components
        missing_metrics = SCENARIO_STATISTICS_REQUEST.metrics - projection.loaded_path_metrics
        details: list[str] = []
        if missing_components:
            details.append(
                "components="
                + ",".join(sorted(component.value for component in missing_components))
            )
        if missing_metrics:
            details.append(
                "metrics=" + ",".join(sorted(metric.value for metric in missing_metrics))
            )
        raise ValueError(
            "Incomplete scenario statistics projection"
            + (": " + "; ".join(details) if details else "")
        )

    @staticmethod
    def _collect_frame_stats(frame: ProjectedMPCFrame) -> _FrameStatistics:
        """Compute statistics from one validated selective frame."""

        bounce_offsets = frame.bounce_offsets
        interactions = frame.interactions
        if bounce_offsets is None or interactions is None:
            raise ValueError("Scenario statistics projection requires path interactions")

        reflection_orders = np.diff(bounce_offsets).astype(np.int64, copy=False)
        mpc_types = _first_interaction_types(bounce_offsets, interactions)
        path_loss_valid = frame.metric_is_valid(PathMetric.PATH_LOSS_DB)
        delay_valid = frame.metric_is_valid(PathMetric.DELAY_NS)
        path_losses = _metric_values(
            frame,
            PathMetric.PATH_LOSS_DB,
            path_loss_valid,
        )
        delays = _metric_values(
            frame,
            PathMetric.DELAY_NS,
            delay_valid,
        )
        joint_delays, joint_path_losses, joint_metrics_aligned = _joint_delay_loss(
            frame,
            delays=delays,
            path_losses=path_losses,
            delay_valid=delay_valid,
            path_loss_valid=path_loss_valid,
        )

        angle_values: dict[PathMetric, FloatArray] = {}
        for metric in (
            PathMetric.AOA_AZ_DEG,
            PathMetric.AOA_EL_DEG,
            PathMetric.AOD_AZ_DEG,
            PathMetric.AOD_EL_DEG,
        ):
            values = _metric_values(frame, metric)
            angle_values[metric] = values

        if joint_delays.size:
            joint_delays_float64 = joint_delays.astype(np.float64, copy=False)
            joint_path_losses_float64 = joint_path_losses.astype(np.float64, copy=False)
            powers = 10.0 ** (-joint_path_losses_float64 / 10.0)
            delay_spread = compute_delay_spread(joint_delays_float64, powers)
        else:
            delay_spread = float("nan")

        pair_visibility, pair_gains, pair_delay_spreads = _pair_frame_metrics(
            frame,
            reflection_orders,
            delay_valid,
            path_loss_valid,
        )

        return _FrameStatistics(
            mpc_count=len(reflection_orders),
            reflection_order_counts=_reflection_order_counts(reflection_orders),
            mpc_type_counts=_counts(mpc_types),
            path_losses=path_losses,
            delays=delays,
            aoa_az_values=angle_values[PathMetric.AOA_AZ_DEG],
            aoa_el_values=angle_values[PathMetric.AOA_EL_DEG],
            aod_az_values=angle_values[PathMetric.AOD_AZ_DEG],
            aod_el_values=angle_values[PathMetric.AOD_EL_DEG],
            joint_delays=joint_delays,
            joint_path_losses=joint_path_losses,
            joint_metrics_aligned=joint_metrics_aligned,
            delay_spread=delay_spread,
            pair_visibility_counts=pair_visibility,
            pair_aggregate_path_gains_db=pair_gains,
            pair_rms_delay_spreads_ns=pair_delay_spreads,
            strongest_single_path_loss_db=(
                float(np.min(path_losses)) if path_losses.size else float("nan")
            ),
        )

    @staticmethod
    def _merge_counts(target: dict[int, int], source: dict[int, int]) -> None:
        """Add a per-frame integer histogram to an aggregate histogram."""

        for value, count in source.items():
            target[value] = target.get(value, 0) + count

    @staticmethod
    def _finalize_metric(
        stats: dict[str, Any],
        key: str,
    ) -> Float64Array:
        """Concatenate one metric's frame chunks and store the stable array form."""

        chunks = stats[key]
        values = np.concatenate(chunks, dtype=np.float64) if chunks else _empty_float64()
        if key in ("aoa_az_values", "aod_az_values"):
            values = _normalize_azimuth_deg(values)
        stats[key] = values
        return values

    @staticmethod
    def _summary(values: Float64Array) -> dict[str, float] | None:
        """Return the panel's four-number summary for a valid metric vector."""

        if not values.size:
            return None
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
        }

    @staticmethod
    def _percentile_summary(values: Float64Array) -> dict[str, float | int] | None:
        """Return robust summary values for a pair-frame metric distribution."""

        if not values.size:
            return None
        return {
            "count": int(values.size),
            "min": float(np.min(values)),
            "p10": float(np.percentile(values, 10.0)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90.0)),
            "max": float(np.max(values)),
        }


__all__ = [
    "KNOWN_MPC_TYPES",
    "MPC_TYPE_BUCKETS",
    "PAIR_VISIBILITY_CATEGORIES",
    "SCENARIO_STATISTICS_SCHEMA_VERSION",
    "SCENARIO_STATISTICS_REQUEST",
    "UNKNOWN_MPC_TYPE",
    "ScenarioStatisticsAccumulator",
]
