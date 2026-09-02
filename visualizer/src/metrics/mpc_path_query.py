"""Vectorized filtering and deterministic compound sorting for MPC paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Iterable, Optional

import numpy as np

from .mpc_path_catalog import MpcPathCatalog, MpcPathScope


class SortDirection(str, Enum):
    """Direction for one compound-sort clause."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


class MpcSortField(str, Enum):
    """Numeric classifications supported by the NumPy sort engine."""

    PATH_ID = "path_id"
    TX = "tx"
    RX = "rx"
    PATH_LOSS = "path_loss"
    DELAY = "delay"
    INTERACTIONS = "interactions"
    INTERACTION_MIX = "interaction_mix"
    FIRST_MATERIAL = "first_material"
    DELAY_BAND = "delay_band"
    PATH_LOSS_BAND = "path_loss_band"
    GEOMETRIC_LENGTH = "geometric_length"
    STRETCH_RATIO = "stretch_ratio"
    EXCESS_DELAY = "excess_delay"
    STRENGTH_RANK = "strength_rank"
    RELATIVE_PATH_LOSS = "relative_path_loss"
    RELATIVE_POWER = "relative_power"
    AOD_AZIMUTH = "aod_azimuth"
    AOD_ELEVATION = "aod_elevation"
    AOA_AZIMUTH = "aoa_azimuth"
    AOA_ELEVATION = "aoa_elevation"


class MpcGrouping(str, Enum):
    """Flat-table grouping modes.

    Grouping keys are prepended to the user-visible compound sort, keeping
    equal groups contiguous without introducing a tree model.
    """

    NONE = "none"
    TX_RX = "tx_rx"
    RX_TX = "rx_tx"
    INTERACTIONS = "interactions"
    INTERACTION_MIX = "interaction_mix"
    FIRST_MATERIAL = "first_material"
    DELAY_BAND = "delay_band"
    PATH_LOSS_BAND = "path_loss_band"


class MpcSortPreset(str, Enum):
    """Built-in MPC Explorer classifications."""

    TX_RX_STRONGEST = "tx_rx_strongest"
    TX_RX_EARLIEST = "tx_rx_earliest"
    STRONGEST_OVERALL = "strongest_overall"
    EARLIEST_OVERALL = "earliest_overall"
    INTERACTIONS_STRONGEST = "interactions_strongest"
    INTERACTION_MIX_STRONGEST = "interaction_mix_strongest"
    FIRST_MATERIAL_STRONGEST = "first_material_strongest"
    DELAY_BAND_STRONGEST = "delay_band_strongest"
    LOSS_BAND_EARLIEST = "loss_band_earliest"


@dataclass(frozen=True, slots=True)
class MpcSortClause:
    """One field and direction in a compound ordering."""

    field: MpcSortField
    direction: SortDirection = SortDirection.ASCENDING

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", MpcSortField(self.field))
        object.__setattr__(self, "direction", SortDirection(self.direction))


@dataclass(frozen=True, slots=True)
class MpcSortSpec:
    """Two to four user-visible sort clauses.

    Canonical path ID is always appended internally as the final ascending
    tie-breaker and therefore does not consume a clause.
    """

    clauses: tuple[MpcSortClause, ...]

    def __post_init__(self) -> None:
        clauses = tuple(
            clause if isinstance(clause, MpcSortClause) else MpcSortClause(*clause)
            for clause in self.clauses
        )
        if not 2 <= len(clauses) <= 4:
            raise ValueError("compound MPC sorts require between 2 and 4 clauses")
        fields = [clause.field for clause in clauses]
        if len(set(fields)) != len(fields):
            raise ValueError("compound MPC sort fields must be unique")
        object.__setattr__(self, "clauses", clauses)


DEFAULT_SORT_SPEC = MpcSortSpec(
    (
        MpcSortClause(MpcSortField.TX),
        MpcSortClause(MpcSortField.RX),
        MpcSortClause(MpcSortField.PATH_LOSS),
        MpcSortClause(MpcSortField.DELAY),
    )
)


@dataclass(frozen=True, slots=True)
class MpcQuerySpec:
    """Scope, filters, grouping, and ordering for one Explorer request.

    ``contains_interactions`` uses AND semantics: every listed mechanism must
    occur somewhere in the path.  Exact interaction order is represented by
    ``exact_interaction_sequence`` rather than a precomputed string column.
    Optional column and picking prewarms run only inside
    :meth:`MpcPathQueryEngine.execute`.
    """

    scope: MpcPathScope = MpcPathScope.ALL
    grouping: MpcGrouping = MpcGrouping.TX_RX
    sort: MpcSortSpec = DEFAULT_SORT_SPEC
    tx_ids: tuple[int, ...] = ()
    rx_ids: tuple[int, ...] = ()
    path_loss_min_db: Optional[float] = None
    path_loss_max_db: Optional[float] = None
    delay_min_ns: Optional[float] = None
    delay_max_ns: Optional[float] = None
    interaction_count_min: Optional[int] = None
    interaction_count_max: Optional[int] = None
    contains_interactions: tuple[int, ...] = ()
    pure_interaction: Optional[int] = None
    mixed_only: bool = False
    exact_interaction_sequence: Optional[tuple[int, ...]] = None
    first_material_ids: tuple[int, ...] = ()
    delay_band_width_ns: float = 10.0
    path_loss_band_width_db: float = 10.0
    include_status: bool = True
    prewarm_columns: tuple[str, ...] = ()
    include_pick_mapping: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", MpcPathScope(self.scope))
        object.__setattr__(self, "grouping", MpcGrouping(self.grouping))
        if not isinstance(self.sort, MpcSortSpec):
            object.__setattr__(self, "sort", MpcSortSpec(tuple(self.sort)))

        for name in ("tx_ids", "rx_ids", "contains_interactions", "first_material_ids"):
            values = tuple(dict.fromkeys(int(value) for value in getattr(self, name)))
            object.__setattr__(self, name, values)
        prewarm_columns = tuple(
            dict.fromkeys(
                normalized
                for value in self.prewarm_columns
                if (normalized := str(value).strip().lower())
            )
        )
        object.__setattr__(self, "prewarm_columns", prewarm_columns)
        object.__setattr__(self, "include_pick_mapping", bool(self.include_pick_mapping))
        if self.exact_interaction_sequence is not None:
            object.__setattr__(
                self,
                "exact_interaction_sequence",
                tuple(int(value) for value in self.exact_interaction_sequence),
            )
        if self.pure_interaction is not None:
            object.__setattr__(self, "pure_interaction", int(self.pure_interaction))
        if self.pure_interaction is not None and self.mixed_only:
            raise ValueError("pure_interaction and mixed_only are mutually exclusive")

        self._validate_range(
            self.path_loss_min_db,
            self.path_loss_max_db,
            "path-loss",
        )
        self._validate_range(self.delay_min_ns, self.delay_max_ns, "delay")
        self._validate_range(
            self.interaction_count_min,
            self.interaction_count_max,
            "interaction-count",
        )
        self._validate_positive(self.delay_band_width_ns, "delay_band_width_ns")
        self._validate_positive(
            self.path_loss_band_width_db,
            "path_loss_band_width_db",
        )

    @staticmethod
    def _validate_range(
        minimum: Optional[float],
        maximum: Optional[float],
        label: str,
    ) -> None:
        for bound_name, bound in (("minimum", minimum), ("maximum", maximum)):
            if bound is not None and not np.isfinite(float(bound)):
                raise ValueError(f"{label} {bound_name} must be finite")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"{label} minimum cannot exceed maximum")

    @staticmethod
    def _validate_positive(value: float, label: str) -> None:
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{label} must be finite and positive")


@dataclass(frozen=True, slots=True)
class MpcQueryResult:
    """Immutable result delivered from a query worker to the Qt model."""

    generation: int
    path_ids: np.ndarray
    total_path_count: int
    scope_path_count: int
    elapsed_ms: float
    spec: MpcQuerySpec = field(repr=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.path_ids)
        if values.ndim != 1 or values.dtype != np.int32 or not values.flags.c_contiguous:
            raise ValueError("query path_ids must be a C-contiguous int32 vector")
        object.__setattr__(self, "path_ids", values)
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "total_path_count", int(self.total_path_count))
        object.__setattr__(self, "scope_path_count", int(self.scope_path_count))
        object.__setattr__(self, "elapsed_ms", float(self.elapsed_ms))

    @property
    def matching_path_count(self) -> int:
        """Number of rows matching all scope and filter clauses."""
        return int(self.path_ids.shape[0])


def sort_spec_for_preset(preset: MpcSortPreset | str) -> MpcSortSpec:
    """Return the visible compound ordering for a built-in preset."""
    normalized = MpcSortPreset(preset)
    clauses = {
        MpcSortPreset.TX_RX_STRONGEST: (
            MpcSortClause(MpcSortField.TX),
            MpcSortClause(MpcSortField.RX),
            MpcSortClause(MpcSortField.PATH_LOSS),
            MpcSortClause(MpcSortField.DELAY),
        ),
        MpcSortPreset.TX_RX_EARLIEST: (
            MpcSortClause(MpcSortField.TX),
            MpcSortClause(MpcSortField.RX),
            MpcSortClause(MpcSortField.DELAY),
            MpcSortClause(MpcSortField.PATH_LOSS),
        ),
        MpcSortPreset.STRONGEST_OVERALL: (
            MpcSortClause(MpcSortField.PATH_LOSS),
            MpcSortClause(MpcSortField.DELAY),
        ),
        MpcSortPreset.EARLIEST_OVERALL: (
            MpcSortClause(MpcSortField.DELAY),
            MpcSortClause(MpcSortField.PATH_LOSS),
        ),
        MpcSortPreset.INTERACTIONS_STRONGEST: (
            MpcSortClause(MpcSortField.INTERACTIONS),
            MpcSortClause(MpcSortField.PATH_LOSS),
        ),
        MpcSortPreset.INTERACTION_MIX_STRONGEST: (
            MpcSortClause(MpcSortField.INTERACTION_MIX),
            MpcSortClause(MpcSortField.PATH_LOSS),
        ),
        MpcSortPreset.FIRST_MATERIAL_STRONGEST: (
            MpcSortClause(MpcSortField.FIRST_MATERIAL),
            MpcSortClause(MpcSortField.PATH_LOSS),
        ),
        MpcSortPreset.DELAY_BAND_STRONGEST: (
            MpcSortClause(MpcSortField.DELAY_BAND),
            MpcSortClause(MpcSortField.PATH_LOSS),
        ),
        MpcSortPreset.LOSS_BAND_EARLIEST: (
            MpcSortClause(MpcSortField.PATH_LOSS_BAND),
            MpcSortClause(MpcSortField.DELAY),
        ),
    }[normalized]
    return MpcSortSpec(clauses)


def grouping_for_preset(preset: MpcSortPreset | str) -> MpcGrouping:
    """Return the grouping naturally associated with a built-in preset."""
    normalized = MpcSortPreset(preset)
    return {
        MpcSortPreset.TX_RX_STRONGEST: MpcGrouping.TX_RX,
        MpcSortPreset.TX_RX_EARLIEST: MpcGrouping.TX_RX,
        MpcSortPreset.STRONGEST_OVERALL: MpcGrouping.NONE,
        MpcSortPreset.EARLIEST_OVERALL: MpcGrouping.NONE,
        MpcSortPreset.INTERACTIONS_STRONGEST: MpcGrouping.INTERACTIONS,
        MpcSortPreset.INTERACTION_MIX_STRONGEST: MpcGrouping.INTERACTION_MIX,
        MpcSortPreset.FIRST_MATERIAL_STRONGEST: MpcGrouping.FIRST_MATERIAL,
        MpcSortPreset.DELAY_BAND_STRONGEST: MpcGrouping.DELAY_BAND,
        MpcSortPreset.LOSS_BAND_EARLIEST: MpcGrouping.PATH_LOSS_BAND,
    }[normalized]


def query_spec_for_preset(
    preset: MpcSortPreset | str,
    **overrides: object,
) -> MpcQuerySpec:
    """Build a query spec with a preset's sort and natural grouping."""
    normalized = MpcSortPreset(preset)
    values = {
        "sort": sort_spec_for_preset(normalized),
        "grouping": grouping_for_preset(normalized),
        **overrides,
    }
    return MpcQuerySpec(**values)


class MpcPathQueryEngine:
    """Execute one NumPy path query without Qt or renderer dependencies."""

    def __init__(self, catalog: MpcPathCatalog) -> None:
        self._catalog = catalog

    @property
    def catalog(self) -> MpcPathCatalog:
        """Catalog queried by this engine."""
        return self._catalog

    def execute(self, spec: MpcQuerySpec, *, generation: int = 0) -> MpcQueryResult:
        """Filter and sort, returning an int32 canonical-path permutation."""
        if not isinstance(spec, MpcQuerySpec):
            raise TypeError("spec must be an MpcQuerySpec")
        started = perf_counter()
        path_ids, scope_count = self._filtered_path_ids(spec)
        sorted_ids = self._sort_path_ids(path_ids, spec)
        if spec.include_status:
            # Status is a default table column. Materialize the derived mask on
            # the query worker so the first GUI data() call remains O(1). The
            # filtered mask is already borrowed (or implicitly all paths).
            self._catalog.scope_mask(MpcPathScope.RENDERED)
        if spec.prewarm_columns:
            self._catalog.prewarm_columns(spec.prewarm_columns)
        if spec.include_pick_mapping:
            # Viewport picking needs the packet-segment to canonical-segment
            # mapping. Keep its O(segments) construction off the GUI thread.
            self._catalog.rendered_segment_indices
        return MpcQueryResult(
            generation=int(generation),
            path_ids=sorted_ids,
            total_path_count=self._catalog.path_count,
            scope_path_count=scope_count,
            elapsed_ms=(perf_counter() - started) * 1_000.0,
            spec=spec,
        )

    def _filtered_path_ids(self, spec: MpcQuerySpec) -> tuple[np.ndarray, int]:
        count = self._catalog.path_count
        mask: Optional[np.ndarray] = None
        if spec.scope is not MpcPathScope.ALL:
            scope_mask = self._catalog.scope_mask(spec.scope)
            scope_count = int(np.count_nonzero(scope_mask))
            mask = np.array(scope_mask, copy=True)
        else:
            scope_count = count

        def combine(condition: np.ndarray) -> None:
            nonlocal mask
            values = np.asarray(condition, dtype=bool)
            if values.shape != (count,):
                raise ValueError("query condition does not align with catalog paths")
            if mask is None:
                mask = np.array(values, copy=True)
            else:
                np.logical_and(mask, values, out=mask)

        if spec.tx_ids:
            combine(np.isin(self._catalog.tx_ids, spec.tx_ids))
        if spec.rx_ids:
            combine(np.isin(self._catalog.rx_ids, spec.rx_ids))

        if spec.path_loss_min_db is not None:
            combine(self._catalog.path_losses_db >= float(spec.path_loss_min_db))
        if spec.path_loss_max_db is not None:
            combine(self._catalog.path_losses_db <= float(spec.path_loss_max_db))

        if spec.delay_min_ns is not None:
            combine(self._catalog.delays_ns >= float(spec.delay_min_ns))
        if spec.delay_max_ns is not None:
            combine(self._catalog.delays_ns <= float(spec.delay_max_ns))

        if spec.interaction_count_min is not None:
            combine(self._catalog.interaction_counts >= int(spec.interaction_count_min))
        if spec.interaction_count_max is not None:
            combine(self._catalog.interaction_counts <= int(spec.interaction_count_max))

        for interaction_type in spec.contains_interactions:
            combine(self._catalog.contains_interaction(interaction_type))
        if spec.pure_interaction is not None:
            combine(self._catalog.pure_interaction(spec.pure_interaction))
        if spec.mixed_only:
            combine(self._catalog.mixed_interaction_mask)
        if spec.exact_interaction_sequence is not None:
            combine(
                self._catalog.exact_interaction_sequence(
                    spec.exact_interaction_sequence,
                )
            )
        if spec.first_material_ids:
            combine(np.isin(self._catalog.first_material_ids, spec.first_material_ids))

        if mask is None:
            return np.arange(count, dtype=np.int32), scope_count
        return np.flatnonzero(mask).astype(np.int32, copy=False), scope_count

    def _sort_path_ids(self, path_ids: np.ndarray, spec: MpcQuerySpec) -> np.ndarray:
        if path_ids.size <= 1:
            return np.ascontiguousarray(path_ids, dtype=np.int32)

        clauses = self._effective_clauses(spec)
        # np.lexsort considers the final key primary. Path ID is placed first
        # so it remains the least-significant deterministic tie-breaker.
        keys: list[np.ndarray] = [path_ids]
        for clause in reversed(clauses):
            values = self._field_values(clause.field, spec)
            selected = np.asarray(values)[path_ids]
            value_key, missing_key = self._normalized_key(
                selected,
                clause.field,
                clause.direction,
            )
            keys.append(value_key)
            if missing_key is not None:
                # Missing flag follows the value in the tuple and is therefore
                # more significant, keeping unavailable values last for both
                # ascending and descending sorts.
                keys.append(missing_key)

        permutation = np.lexsort(tuple(keys))
        result = path_ids[permutation]
        return np.ascontiguousarray(result, dtype=np.int32)

    @staticmethod
    def _effective_clauses(spec: MpcQuerySpec) -> tuple[MpcSortClause, ...]:
        grouping_fields = {
            MpcGrouping.NONE: (),
            MpcGrouping.TX_RX: (MpcSortField.TX, MpcSortField.RX),
            MpcGrouping.RX_TX: (MpcSortField.RX, MpcSortField.TX),
            MpcGrouping.INTERACTIONS: (MpcSortField.INTERACTIONS,),
            MpcGrouping.INTERACTION_MIX: (MpcSortField.INTERACTION_MIX,),
            MpcGrouping.FIRST_MATERIAL: (MpcSortField.FIRST_MATERIAL,),
            MpcGrouping.DELAY_BAND: (MpcSortField.DELAY_BAND,),
            MpcGrouping.PATH_LOSS_BAND: (MpcSortField.PATH_LOSS_BAND,),
        }[spec.grouping]
        requested_clauses = {clause.field: clause for clause in spec.sort.clauses}
        grouped = tuple(
            requested_clauses.get(sort_field, MpcSortClause(sort_field))
            for sort_field in grouping_fields
        )
        remaining = tuple(
            clause for clause in spec.sort.clauses if clause.field not in grouping_fields
        )
        return (*grouped, *remaining)

    def _field_values(self, sort_field: MpcSortField, spec: MpcQuerySpec) -> np.ndarray:
        catalog = self._catalog
        return {
            MpcSortField.PATH_ID: lambda: catalog.path_ids,
            MpcSortField.TX: lambda: catalog.tx_ids,
            MpcSortField.RX: lambda: catalog.rx_ids,
            MpcSortField.PATH_LOSS: lambda: catalog.path_losses_db,
            MpcSortField.DELAY: lambda: catalog.delays_ns,
            MpcSortField.INTERACTIONS: lambda: catalog.interaction_counts,
            MpcSortField.INTERACTION_MIX: lambda: catalog.interaction_mix_codes,
            MpcSortField.FIRST_MATERIAL: lambda: catalog.first_material_sort_codes,
            MpcSortField.DELAY_BAND: lambda: catalog.delay_bands(spec.delay_band_width_ns),
            MpcSortField.PATH_LOSS_BAND: lambda: catalog.path_loss_bands(
                spec.path_loss_band_width_db
            ),
            MpcSortField.GEOMETRIC_LENGTH: lambda: catalog.geometric_lengths_m,
            MpcSortField.STRETCH_RATIO: lambda: catalog.stretch_ratios,
            MpcSortField.EXCESS_DELAY: lambda: catalog.pair_relative.excess_delay_ns,
            MpcSortField.STRENGTH_RANK: lambda: catalog.pair_relative.strength_rank,
            MpcSortField.RELATIVE_PATH_LOSS: lambda: (catalog.pair_relative.path_loss_delta_db),
            MpcSortField.RELATIVE_POWER: lambda: (catalog.pair_relative.relative_power_proxy),
            MpcSortField.AOD_AZIMUTH: lambda: catalog.aod_azimuth_deg,
            MpcSortField.AOD_ELEVATION: lambda: catalog.aod_elevation_deg,
            MpcSortField.AOA_AZIMUTH: lambda: catalog.aoa_azimuth_deg,
            MpcSortField.AOA_ELEVATION: lambda: catalog.aoa_elevation_deg,
        }[sort_field]()

    @staticmethod
    def _normalized_key(
        selected: np.ndarray,
        sort_field: MpcSortField,
        direction: SortDirection,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        values = np.asarray(selected)
        if values.dtype.kind in "fc":
            numeric = np.array(values, copy=True)
            missing = ~np.isfinite(numeric)
            numeric[missing] = 0.0
            if direction is SortDirection.DESCENDING:
                np.negative(numeric, out=numeric)
            return numeric, missing

        if values.dtype.kind not in "iub":
            raise TypeError(f"{sort_field.value} is not a numeric sort column")

        missing: Optional[np.ndarray] = None
        if sort_field in (MpcSortField.TX, MpcSortField.RX, MpcSortField.STRENGTH_RANK):
            missing = values < 0 if sort_field is not MpcSortField.STRENGTH_RANK else values == 0

        if direction is SortDirection.ASCENDING:
            numeric = np.array(values, copy=True)
            if missing is not None and np.any(missing):
                numeric[missing] = 0
            return numeric, missing

        if values.dtype.itemsize <= 1:
            signed_dtype = np.int16
        elif values.dtype.itemsize <= 2:
            signed_dtype = np.int32
        else:
            signed_dtype = np.int64
        numeric = -values.astype(signed_dtype, copy=False)
        if missing is not None and np.any(missing):
            numeric[missing] = 0
        return numeric, missing


def group_key_for_path(
    catalog: MpcPathCatalog,
    path_id: int,
    spec: MpcQuerySpec,
) -> tuple[int | float | None, ...]:
    """Return a compact group key for flat-table separator roles."""
    index = int(path_id)
    grouping = spec.grouping
    if grouping is MpcGrouping.NONE:
        return ()
    if grouping is MpcGrouping.TX_RX:
        return (int(catalog.tx_ids[index]), int(catalog.rx_ids[index]))
    if grouping is MpcGrouping.RX_TX:
        return (int(catalog.rx_ids[index]), int(catalog.tx_ids[index]))
    if grouping is MpcGrouping.INTERACTIONS:
        return (int(catalog.interaction_counts[index]),)
    if grouping is MpcGrouping.INTERACTION_MIX:
        return (int(catalog.interaction_mix_codes[index]),)
    if grouping is MpcGrouping.FIRST_MATERIAL:
        return (int(catalog.first_material_ids[index]),)
    if grouping is MpcGrouping.DELAY_BAND:
        value = float(catalog.delay_bands(spec.delay_band_width_ns)[index])
        return (value if np.isfinite(value) else None,)
    value = float(catalog.path_loss_bands(spec.path_loss_band_width_db)[index])
    return (value if np.isfinite(value) else None,)


def replace_sort_primary(
    spec: MpcSortSpec,
    field: MpcSortField | str,
    direction: SortDirection | str,
) -> MpcSortSpec:
    """Move one field to the front while retaining up to three other clauses."""
    primary = MpcSortClause(MpcSortField(field), SortDirection(direction))
    clauses = [primary]
    clauses.extend(clause for clause in spec.clauses if clause.field is not primary.field)
    if len(clauses) < 2:
        fallback = (
            MpcSortField.DELAY
            if primary.field is MpcSortField.PATH_LOSS
            else MpcSortField.PATH_LOSS
        )
        clauses.append(MpcSortClause(fallback))
    return MpcSortSpec(tuple(clauses[:4]))


def sort_labels(spec: MpcSortSpec) -> tuple[str, ...]:
    """Return compact labels suitable for visible compound-sort chips."""
    arrow = {
        SortDirection.ASCENDING: "ASC",
        SortDirection.DESCENDING: "DESC",
    }
    return tuple(
        f"{clause.field.value.replace('_', ' ').title()} {arrow[clause.direction]}"
        for clause in spec.clauses
    )


def normalize_sort_clauses(
    clauses: Iterable[MpcSortClause],
) -> MpcSortSpec:
    """Convenience boundary for callers assembling custom clause sequences."""
    return MpcSortSpec(tuple(clauses))


__all__ = [
    "DEFAULT_SORT_SPEC",
    "MpcGrouping",
    "MpcPathQueryEngine",
    "MpcQueryResult",
    "MpcQuerySpec",
    "MpcSortClause",
    "MpcSortField",
    "MpcSortPreset",
    "MpcSortSpec",
    "SortDirection",
    "group_key_for_path",
    "grouping_for_preset",
    "normalize_sort_clauses",
    "query_spec_for_preset",
    "replace_sort_primary",
    "sort_labels",
    "sort_spec_for_preset",
]
