"""Borrowed, frame-local access to canonical MPC paths.

The catalog is the renderer-neutral data boundary for MPC inspection.  Its
constructor deliberately performs only shape inspection: bulk canonical arrays
remain borrowed, and O(paths) or O(segments) work is deferred until a caller
requests the corresponding scope or derived column.

Canonical path IDs are zero-based and valid only for the wrapped frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any, Callable, Optional, TypeVar

import numpy as np

from .mpc_canon import CanonicalStepData

_T = TypeVar("_T")
_MISSING_ID = -1
_GEOMETRY_CHUNK_SIZE = 250_000


class MpcPathScope(str, Enum):
    """Population used by one Explorer query."""

    ALL = "all"
    FILTERED = "filtered"
    RENDERED = "rendered"


class MpcPathCatalogError(ValueError):
    """Raised when canonical arrays cannot satisfy the path-catalog contract."""


@dataclass(frozen=True, slots=True)
class PairRelativeMetrics:
    """Path classifications calculated independently within each TX/RX pair.

    ``relative_power_proxy`` is the dimensionless
    ``10 ** (-path_loss_db / 10)`` quantity. It is not received power or dBm.
    ``contribution_tier`` uses 1/2/3/4 for the cumulative 50/90/99/tail sets;
    zero denotes unavailable loss.
    """

    strength_rank: np.ndarray
    path_loss_delta_db: np.ndarray
    excess_delay_ns: np.ndarray
    relative_power_proxy: np.ndarray
    cumulative_contribution: np.ndarray
    contribution_tier: np.ndarray

    def contribution_mask(self, threshold_percent: int) -> np.ndarray:
        """Return membership in the cumulative 50, 90, or 99 percent set."""
        max_tier = {50: 1, 90: 2, 99: 3}.get(int(threshold_percent))
        if max_tier is None:
            raise ValueError("threshold_percent must be one of 50, 90, or 99")
        return (self.contribution_tier > 0) & (self.contribution_tier <= max_tier)


def interaction_type_bit(interaction_type: int) -> np.uint16:
    """Return the compact category bit used by interaction-mix classifications."""
    value = int(interaction_type)
    bit_index = {
        0: 0,  # LoS (used only for paths without nonzero interactions)
        1: 1,
        2: 2,
        4: 3,
        8: 4,
        99: 5,
    }.get(value, 6)
    return np.uint16(1 << bit_index)


class MpcPathCatalog:
    """Wrap one canonical frame without copying its bulk arrays.

    Args:
        canonical_data: Canonical arrays for one accepted frame.
        filtered_path_mask: Optional pre-Top-K path mask.
        rendered_path_mask: Optional final path mask.  When omitted, the mask is
            derived lazily from ``rendered_segment_mask`` and
            ``canonical_data.segment_path_id``.
        rendered_segment_mask: Optional final segment mask.
        rendered_paths_enabled: Effective MPC-path visibility. When false, the
            rendered scope is empty without scanning segments.
        validate: Run O(paths + segments) monotonicity/range validation.  Normal
            frame presentation should leave this disabled; tests and ingestion
            boundaries may opt in.
    """

    def __init__(
        self,
        canonical_data: CanonicalStepData,
        *,
        filtered_path_mask: Optional[np.ndarray] = None,
        rendered_path_mask: Optional[np.ndarray] = None,
        rendered_segment_mask: Optional[np.ndarray] = None,
        rendered_paths_enabled: bool = True,
        validate: bool = False,
    ) -> None:
        if canonical_data is None:
            raise MpcPathCatalogError("canonical_data is required")

        self._canonical_data = canonical_data
        self._points = self._required_matrix("points", columns=3)
        self._lines = self._optional_matrix("lines", columns=2)
        self._path_count = self._infer_path_count()
        if self._path_count > np.iinfo(np.int32).max:
            raise MpcPathCatalogError("canonical path count exceeds int32 path-ID capacity")

        starts = getattr(canonical_data, "path_start_indices", None)
        if starts is None:
            if self._path_count:
                raise MpcPathCatalogError(
                    "path_start_indices are required for non-empty canonical frames"
                )
            starts = np.empty((0,), dtype=np.int32)
        self._path_start_indices = self._borrow_vector(
            starts,
            "path_start_indices",
            expected_size=self._path_count,
            integer=True,
        )

        self._filtered_path_mask = self._borrow_mask(
            filtered_path_mask,
            "filtered_path_mask",
            self._path_count,
        )
        self._rendered_path_mask = self._borrow_mask(
            rendered_path_mask,
            "rendered_path_mask",
            self._path_count,
        )
        self._rendered_paths_enabled = bool(rendered_paths_enabled)

        segment_count = self._segment_count
        self._rendered_segment_mask = self._borrow_mask(
            rendered_segment_mask,
            "rendered_segment_mask",
            segment_count,
        )

        self._cache: dict[str, Any] = {}
        self._cache_lock = RLock()

        if validate:
            self.validate()

    @property
    def canonical_data(self) -> CanonicalStepData:
        """Return the borrowed canonical frame."""
        return self._canonical_data

    @property
    def path_count(self) -> int:
        """Number of frame-local canonical paths."""
        return self._path_count

    @property
    def point_count(self) -> int:
        """Number of canonical MPC points."""
        return int(self._points.shape[0])

    @property
    def segment_count(self) -> int:
        """Number of canonical MPC segments."""
        return self._segment_count

    @property
    def cached_derived_columns(self) -> frozenset[str]:
        """Names of derived values materialized so far, for diagnostics/tests."""
        with self._cache_lock:
            return frozenset(self._cache)

    def derived_column_is_ready(self, column_name: str) -> bool:
        """Return whether an expensive optional table column is worker-warmed."""
        cache_keys = {
            "geometric_length": ("geometric_lengths_m",),
            "stretch_ratio": ("stretch_ratios",),
            "excess_delay": ("pair_relative",),
            "strength_rank": ("pair_relative",),
            "relative_path_loss": ("pair_relative",),
            "relative_power": ("pair_relative",),
            "delay_provenance": ("delay_provenance_codes",),
            "path_loss_provenance": ("path_loss_provenance_codes",),
        }.get(str(column_name), ())
        if not cache_keys:
            return True
        with self._cache_lock:
            return all(key in self._cache for key in cache_keys)

    def prewarm_columns(self, column_names: tuple[str, ...]) -> None:
        """Materialize requested optional columns on the query worker.

        This method is intentionally explicit. Merely adding a hidden optional
        column to the Qt model must not scan all paths or segments.
        """
        builders: dict[str, Callable[[], Any]] = {
            "geometric_length": lambda: self.geometric_lengths_m,
            "stretch_ratio": lambda: self.stretch_ratios,
            "excess_delay": lambda: self.pair_relative,
            "strength_rank": lambda: self.pair_relative,
            "relative_path_loss": lambda: self.pair_relative,
            "relative_power": lambda: self.pair_relative,
            "delay_provenance": lambda: self.delay_provenance_codes,
            "path_loss_provenance": lambda: self.path_loss_provenance_codes,
        }
        for name in dict.fromkeys(str(value) for value in column_names):
            builder = builders.get(name)
            if builder is not None:
                builder()

    @property
    def path_start_indices(self) -> np.ndarray:
        """Borrowed point offset for each path."""
        return self._path_start_indices

    @property
    def path_ids(self) -> np.ndarray:
        """Return a temporary canonical-ID vector.

        The catalog intentionally does not retain this vector: after a worker
        sort, the model's int32 permutation is the sole permanent row-order
        allocation.
        """
        return np.arange(self._path_count, dtype=np.int32)

    @property
    def path_end_indices(self) -> np.ndarray:
        """Exclusive point offset for each path."""

        def build() -> np.ndarray:
            ends = np.empty((self._path_count,), dtype=np.int32)
            if self._path_count:
                ends[:-1] = self._path_start_indices[1:]
                ends[-1] = self.point_count
            return ends

        return self._cached("path_end_indices", build)

    @property
    def path_point_counts(self) -> np.ndarray:
        """Point count, including TX and RX, for each path."""
        return self._cached(
            "path_point_counts",
            lambda: np.subtract(
                self.path_end_indices,
                self._path_start_indices,
                dtype=np.int32,
            ),
        )

    @property
    def tx_ids(self) -> np.ndarray:
        """TX ID per path, or -1 when unavailable."""
        return self._path_values(
            cache_key="tx_ids",
            path_field="path_tx",
            point_field="tx_id",
            missing_value=_MISSING_ID,
            missing_dtype=np.int32,
        )

    @property
    def rx_ids(self) -> np.ndarray:
        """RX ID per path, or -1 when unavailable."""
        return self._path_values(
            cache_key="rx_ids",
            path_field="path_rx",
            point_field="rx_id",
            missing_value=_MISSING_ID,
            missing_dtype=np.int32,
        )

    @property
    def interaction_counts(self) -> np.ndarray:
        """Count of nonzero interaction mechanisms per path."""
        values = getattr(self._canonical_data, "path_orders", None)
        if values is not None:
            return self._checked_path_vector(values, "path_orders")

        def build() -> np.ndarray:
            if self._path_count == 0:
                return np.empty((0,), dtype=np.int32)
            interaction_types = self._required_point_vector("itype")
            nonzero = interaction_types != 0
            return np.add.reduceat(
                nonzero,
                self._path_start_indices.astype(np.intp, copy=False),
                dtype=np.int32,
            )

        return self._cached("interaction_counts", build)

    @property
    def delays_ns(self) -> np.ndarray:
        """Whole-path delay in nanoseconds, or NaN when unavailable."""
        return self._path_values(
            cache_key="delays_ns",
            path_field="path_delays",
            point_field="delay",
            missing_value=np.nan,
            missing_dtype=np.float32,
        )

    @property
    def path_losses_db(self) -> np.ndarray:
        """Whole-path loss in dB, or NaN when unavailable."""
        return self._path_values(
            cache_key="path_losses_db",
            path_field="path_losses",
            point_field="loss",
            missing_value=np.nan,
            missing_dtype=np.float32,
        )

    @property
    def aod_azimuth_deg(self) -> np.ndarray:
        """Azimuth angle of departure per path; missing values remain NaN."""
        return self._path_angle_or_nan(
            path_field="path_aod_az",
            point_field="aod_az",
            cache_key="aod_azimuth_deg",
        )

    @property
    def aod_elevation_deg(self) -> np.ndarray:
        """Elevation angle of departure per path; missing values remain NaN."""
        return self._path_angle_or_nan(
            path_field="path_aod_el",
            point_field="aod_el",
            cache_key="aod_elevation_deg",
        )

    @property
    def aoa_azimuth_deg(self) -> np.ndarray:
        """Azimuth angle of arrival per path; missing values remain NaN."""
        return self._path_angle_or_nan(
            path_field="path_aoa_az",
            point_field="aoa_az",
            cache_key="aoa_azimuth_deg",
        )

    @property
    def aoa_elevation_deg(self) -> np.ndarray:
        """Elevation angle of arrival per path; missing values remain NaN."""
        return self._path_angle_or_nan(
            path_field="path_aoa_el",
            point_field="aoa_el",
            cache_key="aoa_elevation_deg",
        )

    @property
    def delay_provenance_codes(self) -> np.ndarray:
        """Per-path delay provenance: -1 unknown, 0 exported, 1 estimated."""
        return self._provenance_codes(
            flag_field="path_delay_is_estimated",
            values=self.delays_ns,
            cache_key="delay_provenance_codes",
        )

    @property
    def path_loss_provenance_codes(self) -> np.ndarray:
        """Per-path loss provenance: -1 unknown, 0 exported, 1 estimated."""
        return self._provenance_codes(
            flag_field="path_loss_is_estimated",
            values=self.path_losses_db,
            cache_key="path_loss_provenance_codes",
        )

    @property
    def first_material_ids(self) -> np.ndarray:
        """First interior material ID per path.

        ``0`` means a physical bounce without a material assignment (and is
        also used by canonical LoS endpoints). ``-1`` means material metadata
        is absent for the frame.
        """

        def build() -> np.ndarray:
            material_ids = getattr(self._canonical_data, "material_ids", None)
            if material_ids is None:
                return np.full((self._path_count,), _MISSING_ID, dtype=np.int16)
            values = self._borrow_vector(
                material_ids,
                "material_ids",
                expected_size=self.point_count,
                integer=True,
            )
            result = np.zeros((self._path_count,), dtype=np.int16)
            has_bounce = self.path_point_counts > 2
            if np.any(has_bounce):
                result[has_bounce] = values[
                    self._path_start_indices[has_bounce].astype(np.intp, copy=False) + 1
                ]
            return result

        return self._cached("first_material_ids", build)

    @property
    def first_material_sort_codes(self) -> np.ndarray:
        """Case-insensitive material-name ranks, with unavailable values as NaN."""

        def build() -> np.ndarray:
            material_ids = self.first_material_ids
            result = np.full((self._path_count,), np.nan, dtype=np.float64)
            available = material_ids >= 0
            if not np.any(available):
                return result
            unique_ids = np.unique(material_ids[available])
            ordered_ids = sorted(
                (int(value) for value in unique_ids),
                key=lambda value: (self.material_name(value).casefold(), value),
            )
            for rank, material_id in enumerate(ordered_ids):
                result[material_ids == material_id] = rank
            return result

        return self._cached("first_material_sort_codes", build)

    @property
    def interaction_mix_codes(self) -> np.ndarray:
        """Compact category bit mask per path, computed only when requested."""

        def build() -> np.ndarray:
            result = np.zeros((self._path_count,), dtype=np.uint16)
            if self._path_count == 0:
                return result

            segment_types = getattr(self._canonical_data, "segment_itype", None)
            segment_path_ids = getattr(self._canonical_data, "segment_path_id", None)
            if segment_types is not None and segment_path_ids is not None:
                types = self._borrow_vector(
                    segment_types,
                    "segment_itype",
                    expected_size=self._segment_count,
                    integer=True,
                )
                path_ids = self._borrow_vector(
                    segment_path_ids,
                    "segment_path_id",
                    expected_size=self._segment_count,
                    integer=True,
                )
            else:
                types = self._required_point_vector("itype")
                path_ids = self._required_point_vector("path_id", integer=True)

            for start in range(0, int(types.shape[0]), _GEOMETRY_CHUNK_SIZE):
                stop = min(start + _GEOMETRY_CHUNK_SIZE, int(types.shape[0]))
                type_chunk = types[start:stop]
                path_chunk = path_ids[start:stop].astype(np.intp, copy=False)
                codes = self._interaction_bits(type_chunk)
                nonzero = codes != 0
                if np.any(nonzero):
                    np.bitwise_or.at(result, path_chunk[nonzero], codes[nonzero])

            los_paths = self.interaction_counts == 0
            result[los_paths] |= interaction_type_bit(0)
            return result

        return self._cached("interaction_mix_codes", build)

    @property
    def mixed_interaction_mask(self) -> np.ndarray:
        """Whether a path contains more than one mechanism category."""

        def build() -> np.ndarray:
            values = self.interaction_mix_codes.astype(np.uint16, copy=False)
            # x & (x - 1) is nonzero exactly when more than one bit is set.
            return (values != 0) & ((values & (values - np.uint16(1))) != 0)

        return self._cached("mixed_interaction_mask", build)

    @property
    def geometric_lengths_m(self) -> np.ndarray:
        """Cumulative geometric arc length for every path."""
        return self._cached("geometric_lengths_m", self._build_geometric_lengths)

    @property
    def direct_distances_m(self) -> np.ndarray:
        """Straight TX-to-RX distance for every path."""

        def build() -> np.ndarray:
            result = np.empty((self._path_count,), dtype=np.float64)
            for start in range(0, self._path_count, _GEOMETRY_CHUNK_SIZE):
                stop = min(start + _GEOMETRY_CHUNK_SIZE, self._path_count)
                starts = self._path_start_indices[start:stop].astype(np.intp, copy=False)
                ends = self.path_end_indices[start:stop].astype(np.intp, copy=False) - 1
                delta = np.asarray(self._points[ends], dtype=np.float64)
                delta -= self._points[starts]
                result[start:stop] = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            return result

        return self._cached("direct_distances_m", build)

    @property
    def stretch_ratios(self) -> np.ndarray:
        """Geometric path length divided by direct TX/RX distance."""

        def build() -> np.ndarray:
            direct = self.direct_distances_m
            result = np.full((self._path_count,), np.nan, dtype=np.float64)
            valid = np.isfinite(direct) & (direct > 0.0)
            np.divide(
                self.geometric_lengths_m,
                direct,
                out=result,
                where=valid,
            )
            return result

        return self._cached("stretch_ratios", build)

    @property
    def pair_relative(self) -> PairRelativeMetrics:
        """Lazily computed classifications local to each TX/RX pair."""
        return self._cached("pair_relative", self._build_pair_relative_metrics)

    def validate(self) -> None:
        """Run full monotonicity and bounds checks on the borrowed frame."""
        starts = self._path_start_indices
        if self._path_count:
            if int(starts[0]) < 0 or int(starts[-1]) >= self.point_count:
                raise MpcPathCatalogError("path_start_indices fall outside points")
            if np.any(np.diff(starts.astype(np.int64, copy=False)) <= 0):
                raise MpcPathCatalogError("path_start_indices must be strictly increasing")
            if np.any(self.path_point_counts < 2):
                raise MpcPathCatalogError("every canonical path must contain TX and RX")

        for name in (
            "path_orders",
            "path_delays",
            "path_losses",
            "path_tx",
            "path_rx",
            "path_delay_is_estimated",
            "path_loss_is_estimated",
            "path_aoa_az",
            "path_aoa_el",
            "path_aod_az",
            "path_aod_el",
        ):
            values = getattr(self._canonical_data, name, None)
            if values is not None and np.asarray(values).reshape(-1).shape[0] != self._path_count:
                raise MpcPathCatalogError(f"{name} does not align with canonical paths")

        segment_path_ids = getattr(self._canonical_data, "segment_path_id", None)
        if segment_path_ids is not None:
            ids = self._borrow_vector(
                segment_path_ids,
                "segment_path_id",
                expected_size=self._segment_count,
                integer=True,
            )
            if ids.size and (
                np.any(ids < 0)
                or np.any(ids >= self._path_count)
                or np.any(np.diff(ids.astype(np.int64, copy=False)) < 0)
            ):
                raise MpcPathCatalogError(
                    "segment_path_id must be in range and grouped by canonical path"
                )

    def path_points(self, path_id: int) -> np.ndarray:
        """Return a borrowed ``[TX, bounces..., RX]`` point slice."""
        start, end = self._path_bounds(path_id)
        return self._points[start:end]

    def interior_bounces(self, path_id: int) -> np.ndarray:
        """Return the borrowed physical-bounce point slice for one path."""
        start, end = self._path_bounds(path_id)
        return self._points[start + 1 : end - 1]

    def interaction_sequence(self, path_id: int) -> np.ndarray:
        """Return the borrowed interaction sequence for one path."""
        start, end = self._path_bounds(path_id)
        interaction_types = self._required_point_vector("itype")
        return interaction_types[start + 1 : end - 1]

    def material_sequence(self, path_id: int) -> Optional[np.ndarray]:
        """Return borrowed interior material IDs, or ``None`` if unavailable."""
        material_ids = getattr(self._canonical_data, "material_ids", None)
        if material_ids is None:
            return None
        values = self._borrow_vector(
            material_ids,
            "material_ids",
            expected_size=self.point_count,
            integer=True,
        )
        start, end = self._path_bounds(path_id)
        return values[start + 1 : end - 1]

    def first_material_id(self, path_id: int) -> int:
        """Return one first material ID in O(1), without building a full column."""
        start, end = self._path_bounds(path_id)
        material_ids = getattr(self._canonical_data, "material_ids", None)
        if material_ids is None:
            return _MISSING_ID
        if end - start <= 2:
            return 0
        values = self._borrow_vector(
            material_ids,
            "material_ids",
            expected_size=self.point_count,
            integer=True,
        )
        return int(values[start + 1])

    def material_name(self, material_id: int) -> str:
        """Resolve a canonical material ID without constructing path strings."""
        value = int(material_id)
        if value <= 0:
            return ""
        mapping = getattr(self._canonical_data, "material_id_to_name", None) or {}
        name = str(mapping.get(value, "") or "").strip()
        return name if name.lower() != "none" else ""

    def delay_provenance(self, path_id: int) -> str:
        """Return ``exported``, ``estimated``, ``unknown``, or ``unavailable``."""
        index = self._require_path_id(path_id)
        if not np.isfinite(self.delays_ns[index]):
            return "unavailable"
        return self._provenance_label(self.delay_provenance_codes[index])

    def path_loss_provenance(self, path_id: int) -> str:
        """Return ``exported``, ``estimated``, ``unknown``, or ``unavailable``."""
        index = self._require_path_id(path_id)
        if not np.isfinite(self.path_losses_db[index]):
            return "unavailable"
        return self._provenance_label(self.path_loss_provenance_codes[index])

    def contains_interaction(self, interaction_type: int) -> np.ndarray:
        """Return a path mask for one raw interaction type."""
        value = int(interaction_type)
        cache_key = f"contains_interaction:{value}"

        def build() -> np.ndarray:
            if value in (0, 1, 2, 4, 8, 99):
                return (self.interaction_mix_codes & interaction_type_bit(value)) != 0

            result = np.zeros((self._path_count,), dtype=bool)
            if self._path_count == 0:
                return result
            interaction_types = self._required_point_vector("itype")
            point_path_ids = self._required_point_vector("path_id", integer=True)
            matching = interaction_types == value
            if np.any(matching):
                result[point_path_ids[matching].astype(np.intp, copy=False)] = True
            return result

        return self._cached(cache_key, build)

    def pure_interaction(self, interaction_type: int) -> np.ndarray:
        """Return paths whose nonzero interactions all have the requested type."""
        value = int(interaction_type)
        cache_key = f"pure_interaction:{value}"

        def build() -> np.ndarray:
            if value == 0:
                return self.interaction_counts == 0
            if value in (1, 2, 4, 8, 99):
                return self.interaction_mix_codes == interaction_type_bit(value)
            contains = self.contains_interaction(value)
            candidate_ids = np.flatnonzero(contains)
            result = np.zeros((self._path_count,), dtype=bool)
            for path_id in candidate_ids:
                sequence = self.interaction_sequence(int(path_id))
                result[path_id] = bool(sequence.size) and bool(np.all(sequence == value))
            return result

        return self._cached(cache_key, build)

    def exact_interaction_sequence(self, sequence: tuple[int, ...]) -> np.ndarray:
        """Return paths whose ordered interaction sequence matches exactly."""
        normalized = tuple(int(value) for value in sequence)
        cache_key = "exact_interaction_sequence:" + ",".join(map(str, normalized))

        def build() -> np.ndarray:
            expected_count = len(normalized)
            result = self.interaction_counts == expected_count
            if expected_count == 0:
                return result

            candidate_ids = np.flatnonzero(result).astype(np.intp, copy=False)
            if candidate_ids.size == 0:
                return result
            starts = self._path_start_indices[candidate_ids].astype(np.intp, copy=False)
            interaction_types = self._required_point_vector("itype")
            for offset, expected in enumerate(normalized, start=1):
                result[candidate_ids] &= interaction_types[starts + offset] == expected
            return result

        return self._cached(cache_key, build)

    def scope_mask(self, scope: MpcPathScope | str) -> np.ndarray:
        """Return a lazy boolean mask for the requested path population."""
        normalized = MpcPathScope(scope)
        if normalized is MpcPathScope.ALL:
            return self._cached(
                "scope_mask:all",
                lambda: np.ones((self._path_count,), dtype=bool),
            )
        if normalized is MpcPathScope.FILTERED:
            if self._filtered_path_mask is not None:
                return self._filtered_path_mask
            return self.scope_mask(MpcPathScope.ALL)

        if not self._rendered_paths_enabled:
            return self._cached(
                "scope_mask:rendered",
                lambda: np.zeros((self._path_count,), dtype=bool),
            )
        if self._rendered_path_mask is not None:
            return self._rendered_path_mask

        def build_rendered() -> np.ndarray:
            if self._rendered_segment_mask is None:
                return np.array(self.scope_mask(MpcPathScope.FILTERED), copy=True)
            result = np.zeros((self._path_count,), dtype=bool)
            segment_path_ids = getattr(self._canonical_data, "segment_path_id", None)
            if segment_path_ids is None:
                return result
            path_ids = self._borrow_vector(
                segment_path_ids,
                "segment_path_id",
                expected_size=self._segment_count,
                integer=True,
            )
            selected = path_ids[self._rendered_segment_mask]
            if selected.size:
                if np.any(selected < 0) or np.any(selected >= self._path_count):
                    raise MpcPathCatalogError("rendered segment path IDs are out of range")
                result[selected.astype(np.intp, copy=False)] = True
            return result

        return self._cached("scope_mask:rendered", build_rendered)

    def scope_path_ids(self, scope: MpcPathScope | str) -> np.ndarray:
        """Return canonical IDs in one scope as int32."""
        normalized = MpcPathScope(scope)
        if normalized is MpcPathScope.ALL:
            return self.path_ids
        mask = self.scope_mask(normalized)
        return np.flatnonzero(mask).astype(np.int32, copy=False)

    @property
    def rendered_segment_indices(self) -> Optional[np.ndarray]:
        """Map rendered packet segments to canonical segments.

        ``None`` denotes the identity mapping. A filtered mapping is built only
        when the visible Explorer requests viewport selection, and is cached
        for the accepted catalog lifetime.
        """

        def build() -> Optional[np.ndarray]:
            mask = self._rendered_segment_mask
            if mask is None or mask.size == 0 or bool(np.all(mask)):
                return None
            return np.flatnonzero(mask).astype(np.int32, copy=False)

        return self._cached("rendered_segment_indices", build)

    @property
    def rendered_segment_mapping_is_ready(self) -> bool:
        """Whether viewport-pick remapping has already been worker-prepared."""
        with self._cache_lock:
            return "rendered_segment_indices" in self._cache

    def is_filtered(self, path_id: int) -> bool:
        """Whether one canonical path survived the pre-Top-K filters."""
        index = self._require_path_id(path_id)
        if self._filtered_path_mask is None:
            return True
        return bool(self._filtered_path_mask[index])

    def is_rendered(self, path_id: int) -> bool:
        """Whether at least one segment of one path is in the final set."""
        index = self._require_path_id(path_id)
        if not self._rendered_paths_enabled:
            return False
        if self._rendered_path_mask is not None:
            return bool(self._rendered_path_mask[index])
        with self._cache_lock:
            cached_mask = self._cache.get("scope_mask:rendered")
        if cached_mask is not None:
            return bool(cached_mask[index])
        if self._rendered_segment_mask is None:
            return self.is_filtered(index)

        # Canonical points and segments are grouped by path. The number of
        # preceding segments is the number of preceding points minus one
        # terminal point for every preceding path.
        point_start, point_end = self._path_bounds(index)
        segment_start = point_start - index
        segment_end = segment_start + (point_end - point_start - 1)
        return bool(np.any(self._rendered_segment_mask[segment_start:segment_end]))

    def delay_bands(self, width_ns: float = 10.0) -> np.ndarray:
        """Return fixed-width delay-band indices, preserving NaN."""
        width = self._positive_width(width_ns, "delay band")
        return self._numeric_bands(self.delays_ns, width, f"delay_bands:{width!r}")

    def path_loss_bands(self, width_db: float = 10.0) -> np.ndarray:
        """Return fixed-width path-loss-band indices, preserving NaN."""
        width = self._positive_width(width_db, "path-loss band")
        return self._numeric_bands(
            self.path_losses_db,
            width,
            f"path_loss_bands:{width!r}",
        )

    def column(self, name: str) -> np.ndarray:
        """Return a borrowed or lazy path-level numeric column by stable name."""
        normalized = str(name).strip().lower()
        columns: dict[str, Callable[[], np.ndarray]] = {
            "path_id": lambda: self.path_ids,
            "tx": lambda: self.tx_ids,
            "rx": lambda: self.rx_ids,
            "interactions": lambda: self.interaction_counts,
            "interaction_count": lambda: self.interaction_counts,
            "interaction_mix": lambda: self.interaction_mix_codes,
            "first_material": lambda: self.first_material_ids,
            "first_material_sort": lambda: self.first_material_sort_codes,
            "delay_ns": lambda: self.delays_ns,
            "delay": lambda: self.delays_ns,
            "path_loss_db": lambda: self.path_losses_db,
            "path_loss": lambda: self.path_losses_db,
            "delay_provenance": lambda: self.delay_provenance_codes,
            "path_loss_provenance": lambda: self.path_loss_provenance_codes,
            "aod_azimuth_deg": lambda: self.aod_azimuth_deg,
            "aod_elevation_deg": lambda: self.aod_elevation_deg,
            "aoa_azimuth_deg": lambda: self.aoa_azimuth_deg,
            "aoa_elevation_deg": lambda: self.aoa_elevation_deg,
            "geometric_length_m": lambda: self.geometric_lengths_m,
            "geometric_length": lambda: self.geometric_lengths_m,
            "stretch_ratio": lambda: self.stretch_ratios,
            "strength_rank": lambda: self.pair_relative.strength_rank,
            "path_loss_delta_db": lambda: self.pair_relative.path_loss_delta_db,
            "relative_path_loss": lambda: self.pair_relative.path_loss_delta_db,
            "excess_delay_ns": lambda: self.pair_relative.excess_delay_ns,
            "relative_power_proxy": lambda: self.pair_relative.relative_power_proxy,
            "cumulative_contribution": lambda: self.pair_relative.cumulative_contribution,
            "contribution_tier": lambda: self.pair_relative.contribution_tier,
        }
        try:
            return columns[normalized]()
        except KeyError as exc:
            raise KeyError(f"unknown MPC path column: {name!r}") from exc

    @property
    def _segment_count(self) -> int:
        segment_path_ids = getattr(self._canonical_data, "segment_path_id", None)
        if segment_path_ids is not None:
            return int(np.asarray(segment_path_ids).reshape(-1).shape[0])
        if self._lines is not None:
            return int(self._lines.shape[0])
        return 0

    def _infer_path_count(self) -> int:
        starts = getattr(self._canonical_data, "path_start_indices", None)
        if starts is not None:
            return int(np.asarray(starts).reshape(-1).shape[0])
        for name in (
            "path_orders",
            "path_delays",
            "path_losses",
            "path_tx",
            "path_rx",
        ):
            values = getattr(self._canonical_data, name, None)
            if values is not None:
                return int(np.asarray(values).reshape(-1).shape[0])
        return 0

    def _required_matrix(self, name: str, *, columns: int) -> np.ndarray:
        values = getattr(self._canonical_data, name, None)
        if values is None:
            raise MpcPathCatalogError(f"{name} is required")
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != columns:
            raise MpcPathCatalogError(f"{name} must have shape [N,{columns}]")
        return array

    def _optional_matrix(self, name: str, *, columns: int) -> Optional[np.ndarray]:
        values = getattr(self._canonical_data, name, None)
        if values is None:
            return None
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != columns:
            raise MpcPathCatalogError(f"{name} must have shape [N,{columns}]")
        return array

    @staticmethod
    def _borrow_vector(
        values: np.ndarray,
        name: str,
        *,
        expected_size: Optional[int] = None,
        integer: bool = False,
    ) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim != 1:
            raise MpcPathCatalogError(f"{name} must be one-dimensional")
        if expected_size is not None and array.shape[0] != expected_size:
            raise MpcPathCatalogError(
                f"{name} has {array.shape[0]} values; expected {expected_size}"
            )
        if integer and array.dtype.kind not in "iu":
            raise MpcPathCatalogError(f"{name} must use an integer dtype")
        return array

    @classmethod
    def _borrow_mask(
        cls,
        values: Optional[np.ndarray],
        name: str,
        expected_size: int,
    ) -> Optional[np.ndarray]:
        if values is None:
            return None
        array = cls._borrow_vector(values, name, expected_size=expected_size)
        if array.dtype != np.bool_:
            raise MpcPathCatalogError(f"{name} must use bool dtype")
        return array

    def _required_point_vector(self, name: str, *, integer: bool = False) -> np.ndarray:
        values = getattr(self._canonical_data, name, None)
        if values is None:
            raise MpcPathCatalogError(f"{name} is required")
        return self._borrow_vector(
            values,
            name,
            expected_size=self.point_count,
            integer=integer,
        )

    def _checked_path_vector(self, values: np.ndarray, name: str) -> np.ndarray:
        return self._borrow_vector(
            values,
            name,
            expected_size=self._path_count,
        )

    def _path_values(
        self,
        *,
        cache_key: str,
        path_field: str,
        point_field: str,
        missing_value: int | float,
        missing_dtype: np.dtype[Any] | type[Any],
    ) -> np.ndarray:
        path_values = getattr(self._canonical_data, path_field, None)
        if path_values is not None:
            return self._checked_path_vector(path_values, path_field)

        def build() -> np.ndarray:
            point_values = getattr(self._canonical_data, point_field, None)
            if point_values is None:
                return np.full(
                    (self._path_count,),
                    missing_value,
                    dtype=missing_dtype,
                )
            values = self._borrow_vector(
                point_values,
                point_field,
                expected_size=self.point_count,
            )
            return values[self._path_start_indices.astype(np.intp, copy=False)]

        return self._cached(cache_key, build)

    def _path_angle_or_nan(
        self,
        *,
        path_field: str,
        point_field: str,
        cache_key: str,
    ) -> np.ndarray:
        """Return a path angle, deriving it only when the Explorer asks.

        Canonical rendering already keeps each whole-path angle broadcast over
        its points. Keeping a second four-column path copy on every presented
        frame would charge Explorer-only memory while the Explorer is closed.
        """
        values = self._path_values(
            cache_key=cache_key,
            path_field=path_field,
            point_field=point_field,
            missing_value=np.nan,
            missing_dtype=np.float32,
        )
        return np.asarray(values, dtype=np.float32)

    def _provenance_codes(
        self,
        *,
        flag_field: str,
        values: np.ndarray,
        cache_key: str,
    ) -> np.ndarray:
        def build() -> np.ndarray:
            flags = getattr(self._canonical_data, flag_field, None)
            if flags is None:
                result = np.full((self._path_count,), -1, dtype=np.int8)
            else:
                borrowed = self._borrow_vector(
                    flags,
                    flag_field,
                    expected_size=self._path_count,
                )
                if borrowed.dtype != np.bool_:
                    raise MpcPathCatalogError(f"{flag_field} must use bool dtype")
                result = borrowed.astype(np.int8, copy=True)
            result[~np.isfinite(values)] = -1
            return result

        return self._cached(cache_key, build)

    @staticmethod
    def _provenance_label(code: int) -> str:
        return {-1: "unknown", 0: "exported", 1: "estimated"}.get(int(code), "unknown")

    def _path_bounds(self, path_id: int) -> tuple[int, int]:
        index = self._require_path_id(path_id)
        start = int(self._path_start_indices[index])
        end = (
            int(self._path_start_indices[index + 1])
            if index + 1 < self._path_count
            else self.point_count
        )
        if start < 0 or end > self.point_count or end - start < 2:
            raise MpcPathCatalogError(f"canonical bounds are invalid for path {index}")
        return start, end

    def _require_path_id(self, path_id: int) -> int:
        index = int(path_id)
        if index < 0 or index >= self._path_count:
            raise IndexError(f"canonical path ID {index} is out of range")
        return index

    def _cached(self, key: str, builder: Callable[[], _T]) -> _T:
        # Do not hold the catalog-wide lock during an O(paths) or O(segments)
        # NumPy builder. A viewport selection must never wait behind an
        # unrelated optional-column derivation. Concurrent callers may perform
        # duplicate pure computations; only the first completed value is kept.
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]
        value = builder()
        with self._cache_lock:
            if key not in self._cache:
                self._cache[key] = value
            return self._cache[key]

    @staticmethod
    def _interaction_bits(interaction_types: np.ndarray) -> np.ndarray:
        values = np.asarray(interaction_types)
        result = np.zeros(values.shape, dtype=np.uint16)
        known_nonzero = np.zeros(values.shape, dtype=bool)
        for interaction_type in (1, 2, 4, 8, 99):
            matching = values == interaction_type
            result[matching] = interaction_type_bit(interaction_type)
            known_nonzero |= matching
        unknown = (values != 0) & ~known_nonzero
        result[unknown] = interaction_type_bit(-1)
        return result

    def _build_geometric_lengths(self) -> np.ndarray:
        result = np.zeros((self._path_count,), dtype=np.float64)
        if self._path_count == 0 or self._segment_count == 0:
            return result
        if self._lines is None:
            raise MpcPathCatalogError("lines are required for geometric path lengths")

        segment_path_ids = getattr(self._canonical_data, "segment_path_id", None)
        point_path_ids = getattr(self._canonical_data, "path_id", None)
        if segment_path_ids is None and point_path_ids is None:
            raise MpcPathCatalogError(
                "segment_path_id or point path_id is required for path lengths"
            )
        if segment_path_ids is not None:
            all_segment_path_ids = self._borrow_vector(
                segment_path_ids,
                "segment_path_id",
                expected_size=self._segment_count,
                integer=True,
            )
        else:
            all_segment_path_ids = None
            all_point_path_ids = self._borrow_vector(
                point_path_ids,
                "path_id",
                expected_size=self.point_count,
                integer=True,
            )

        for start in range(0, self._segment_count, _GEOMETRY_CHUNK_SIZE):
            stop = min(start + _GEOMETRY_CHUNK_SIZE, self._segment_count)
            line_chunk = self._lines[start:stop]
            line_starts = line_chunk[:, 0].astype(np.intp, copy=False)
            line_ends = line_chunk[:, 1].astype(np.intp, copy=False)
            delta = np.asarray(self._points[line_ends], dtype=np.float64)
            delta -= self._points[line_starts]
            lengths = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            if all_segment_path_ids is not None:
                path_ids = all_segment_path_ids[start:stop].astype(np.intp, copy=False)
            else:
                path_ids = all_point_path_ids[line_starts].astype(np.intp, copy=False)

            if path_ids.size:
                run_starts = np.empty((path_ids.size,), dtype=bool)
                run_starts[0] = True
                run_starts[1:] = path_ids[1:] != path_ids[:-1]
                offsets = np.flatnonzero(run_starts)
                run_ids = path_ids[offsets]
                run_sums = np.add.reduceat(lengths, offsets)
                # Canonical segments are grouped by path. Full validation
                # verifies this once at ingestion/test boundaries.
                result[run_ids] += run_sums
        return result

    def _build_pair_relative_metrics(self) -> PairRelativeMetrics:
        count = self._path_count
        ranks = np.zeros((count,), dtype=np.int32)
        loss_delta = np.full((count,), np.nan, dtype=np.float32)
        excess_delay = np.full((count,), np.nan, dtype=np.float32)
        power_proxy = np.full((count,), np.nan, dtype=np.float64)
        cumulative = np.full((count,), np.nan, dtype=np.float64)
        contribution_tier = np.zeros((count,), dtype=np.uint8)
        if count == 0:
            return PairRelativeMetrics(
                ranks,
                loss_delta,
                excess_delay,
                power_proxy,
                cumulative,
                contribution_tier,
            )

        tx_ids = np.asarray(self.tx_ids)
        rx_ids = np.asarray(self.rx_ids)
        path_ids = self.path_ids
        path_losses = np.asarray(self.path_losses_db, dtype=np.float64)
        finite_loss = np.isfinite(path_losses)
        loss_key = np.array(path_losses, copy=True)
        loss_key[~finite_loss] = 0.0
        pair_order = np.lexsort((path_ids, loss_key, ~finite_loss, rx_ids, tx_ids))
        ordered_tx = tx_ids[pair_order]
        ordered_rx = rx_ids[pair_order]
        group_starts = np.empty((count,), dtype=bool)
        group_starts[0] = True
        group_starts[1:] = (ordered_tx[1:] != ordered_tx[:-1]) | (ordered_rx[1:] != ordered_rx[:-1])
        boundaries = np.flatnonzero(group_starts)

        group_index = np.cumsum(group_starts, dtype=np.int32) - 1
        positions = np.arange(count, dtype=np.int32)
        group_start_for_row = np.maximum.accumulate(np.where(group_starts, positions, np.int32(0)))

        ordered_losses = path_losses[pair_order]
        ordered_finite_loss = np.isfinite(ordered_losses)
        ordered_ranks = positions - group_start_for_row + np.int32(1)
        ordered_ranks[~ordered_finite_loss] = 0
        ranks[pair_order] = ordered_ranks

        strongest_by_group = ordered_losses[boundaries]
        strongest_for_row = strongest_by_group[group_index]
        ordered_loss_delta = ordered_losses - strongest_for_row
        ordered_loss_delta[~ordered_finite_loss] = np.nan
        loss_delta[pair_order] = ordered_loss_delta.astype(np.float32, copy=False)
        power_proxy[pair_order[ordered_finite_loss]] = np.power(
            10.0,
            -ordered_losses[ordered_finite_loss] / 10.0,
        )

        # Use strongest-relative values only as a stable temporary for
        # cumulative fractions. The stored proxy remains 10 ** (-loss_db / 10).
        normalized_proxy = np.zeros((count,), dtype=np.float64)
        normalized_proxy[ordered_finite_loss] = np.power(
            10.0,
            -ordered_loss_delta[ordered_finite_loss] / 10.0,
        )
        proxy_sum_by_group = np.add.reduceat(normalized_proxy, boundaries)
        cumulative_global = np.cumsum(normalized_proxy)
        cumulative_offsets = np.zeros(boundaries.shape, dtype=np.float64)
        if boundaries.size > 1:
            cumulative_offsets[1:] = cumulative_global[boundaries[1:] - 1]
        cumulative_within_group = cumulative_global - cumulative_offsets[group_index]
        denominator = proxy_sum_by_group[group_index]
        ordered_cumulative = np.full((count,), np.nan, dtype=np.float64)
        valid_contribution = ordered_finite_loss & (denominator > 0.0)
        np.divide(
            cumulative_within_group,
            denominator,
            out=ordered_cumulative,
            where=valid_contribution,
        )
        cumulative[pair_order] = ordered_cumulative

        preceding = np.full((count,), np.nan, dtype=np.float64)
        preceding[valid_contribution] = (
            ordered_cumulative[valid_contribution]
            - normalized_proxy[valid_contribution] / denominator[valid_contribution]
        )
        ordered_tiers = np.zeros((count,), dtype=np.uint8)
        ordered_tiers[valid_contribution] = 4
        ordered_tiers[valid_contribution & (preceding < 0.99)] = 3
        ordered_tiers[valid_contribution & (preceding < 0.90)] = 2
        ordered_tiers[valid_contribution & (preceding < 0.50)] = 1
        contribution_tier[pair_order] = ordered_tiers

        # Delay minima use the same contiguous pair grouping; no Python loop is
        # required even when every path belongs to a different TX/RX pair.
        delays = np.asarray(self.delays_ns, dtype=np.float64)
        ordered_delays = delays[pair_order]
        delay_for_min = np.array(ordered_delays, copy=True)
        delay_for_min[~np.isfinite(delay_for_min)] = np.inf
        earliest_by_group = np.minimum.reduceat(delay_for_min, boundaries)
        earliest_for_row = earliest_by_group[group_index]
        ordered_excess_delay = ordered_delays - earliest_for_row
        ordered_excess_delay[~np.isfinite(ordered_excess_delay)] = np.nan
        excess_delay[pair_order] = ordered_excess_delay.astype(np.float32, copy=False)

        return PairRelativeMetrics(
            ranks,
            loss_delta,
            excess_delay,
            power_proxy,
            cumulative,
            contribution_tier,
        )

    def _numeric_bands(
        self,
        values: np.ndarray,
        width: float,
        cache_key: str,
    ) -> np.ndarray:
        def build() -> np.ndarray:
            source = np.asarray(values, dtype=np.float64)
            result = np.full(source.shape, np.nan, dtype=np.float64)
            finite = np.isfinite(source)
            result[finite] = np.floor(source[finite] / width)
            return result

        return self._cached(cache_key, build)

    @staticmethod
    def _positive_width(value: float, label: str) -> float:
        width = float(value)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError(f"{label} width must be finite and positive")
        return width


__all__ = [
    "MpcPathCatalog",
    "MpcPathCatalogError",
    "MpcPathScope",
    "PairRelativeMetrics",
    "interaction_type_bit",
]
