"""Sensing product adapter and 3D overlay builder for visualizer frames."""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import CancelledError, Future, TimeoutError
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from threading import Event, RLock
from typing import TYPE_CHECKING, Any, Deque, Optional
from weakref import WeakKeyDictionary

import numpy as np

from shared.extensions.sensing import (
    UnsupportedSensingSchemaError,
    assert_sensing_contract,
    assert_supported_sensing_schema,
)
from shared.frames import (
    FrameComponent,
    FrameProjection,
    FrameReadRequest,
    StandardMPCFrame,
    project_standard_mpc_frame,
)
from shared.frames.providers import Hdf5Provider
from shared.logging import get_logger
from shared.sensing.products import select_product_by_contract
from shared.sensing.replay import (
    ReplayEvent,
    ReplaySceneContext,
    build_replay_event,
    rebind_replay_event,
)
from shared.sensing.signal_processing import (
    RangeDopplerResult,
    RangeProfileResult,
    apply_detection_velocity_metadata,
)

from ..io.packed_frame_payload import visualizer_frame_provider
from ..model import RenderObject
from ..sensing.cir_provider import (
    CIRComputationResult,
    CIRProvider,
    StoredCIRProvider,
)
from ..sensing.display_style import color_rgba, detection_color, identity_color
from ..sensing.path_context import AnalyticalPathContext, analytical_source_record
from ..sensing.recording_cache import RecordingCache
from ..sensing.stage_inspector import (
    DETECTION_STAGE_KINDS,
    DetectionStageBundle,
    DetectionStageProduct,
    InspectionDisposition,
)
from ..sensing.stage_inspector import (
    load_detection_stage_bundle as inspect_detection_stage_bundle,
)
from ..sensing.volume_antenna import receive_metadata_signature
from ..types.render_payloads import LineSetPayload, MaterialPayload, PointCloudPayload
from ..utils.aperture_geometry import yaw_pitch_roll_to_rotation_matrix
from .base import BaseService
from .sensing_overlay_sync import SensingOverlaySync

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer
    from ..renderers.protocol import RendererProtocol

logger = get_logger("orchav.sensing.service")

_SHARED_RESULT_INDEX_LOCK = RLock()
_SHARED_RESULT_INDEXES: WeakKeyDictionary = WeakKeyDictionary()

_DETECTION_HISTORY_REQUEST = FrameReadRequest(
    components=frozenset({FrameComponent.SENSING}),
    sensing_products=frozenset({"detections"}),
)

_RANGE_PROFILE_REQUEST_NAMES = frozenset(
    {"range_profile", "range_profile_raw", "range_profile_by_pair"}
)
_RANGE_DOPPLER_REQUEST_NAMES = frozenset(
    {
        "range_doppler_map",
        "range_doppler_map_raw",
        "range_doppler",
        "range_doppler_by_pair",
        "range_doppler_power_by_pair",
    }
)


@dataclass(frozen=True)
class _SensingProductContract:
    """Exact descriptor query used by persisted visualizer consumers."""

    kind: str
    axis_names: tuple[str, ...]
    axis_units: tuple[str, ...]
    representations: tuple[str, ...]


_RANGE_PROFILE_CONTRACT = _SensingProductContract(
    kind="range_profile",
    axis_names=("pair", "range"),
    axis_units=("1", "m"),
    representations=("power_linear",),
)
_RANGE_DOPPLER_CONTRACT = _SensingProductContract(
    kind="range_doppler",
    axis_names=("pair", "doppler_frequency", "range"),
    axis_units=("1", "Hz", "m"),
    representations=("complex_voltage", "power_linear"),
)


_DESCRIPTOR_PROBE_REQUEST = FrameReadRequest(
    components=frozenset({FrameComponent.SENSING}),
    sensing_products=frozenset({"products"}),
)

_SELECTED_PRODUCT_CACHE_CAPACITY = 2
_DETECTION_STAGE_BUNDLE_CACHE_CAPACITY = 2
_VALIDATED_REPLAY_CACHE_CAPACITY = 8
# Compact detection history needs to cover ordinary recordings (the poster
# example has 192 results); 64 entries caused full-history playback to evict
# and reread its entire history at every tick. Accumulated render snapshots
# stay smaller; current detections get a separate cache so a complete replay
# does not evict every view before the next loop reaches it.
_OVERLAY_HISTORY_CACHE_CAPACITY = 512
_SCENE_OVERLAY_CACHE_CAPACITY = 64
_CURRENT_DETECTION_CACHE_CAPACITY = 512
_SENSING_RESOLVE_MAX_ATTEMPTS = 2


class _SensingInventoryChanged(RuntimeError):
    """Signal that one replay read crossed provider generations."""


def _copy_sensing_snapshot(value: Any, *, freeze_arrays: bool = False) -> Any:
    """Detach containers while sharing only immutable, owned numeric buffers.

    Read-only ndarray flags alone are insufficient: an owner can re-enable
    writes, and sharing the ndarray itself exposes mutable shape metadata.
    Private snapshots therefore own bytes; each consumer gets a separate
    ndarray header backed directly by those bytes.
    """
    # Most entries are ordinary JSON scalars/containers. Avoid repeated NumPy
    # and abstract-Mapping checks for every acquisition sample and descriptor.
    kind = type(value)
    if kind in (str, int, float, bool, complex, bytes, type(None)):
        return value
    if kind is dict:
        return {
            key: _copy_sensing_snapshot(item, freeze_arrays=freeze_arrays)
            for key, item in value.items()
        }
    if kind is list:
        return [_copy_sensing_snapshot(item, freeze_arrays=freeze_arrays) for item in value]
    if kind is tuple:
        return tuple(_copy_sensing_snapshot(item, freeze_arrays=freeze_arrays) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return deepcopy(value)
        owner = value
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        buffer = owner.base
        if (
            freeze_arrays
            or not isinstance(buffer, bytes)
            or not value.flags.c_contiguous
            or value.nbytes != len(buffer)
        ):
            buffer = value.tobytes(order="C")
        return np.frombuffer(buffer, dtype=value.dtype).reshape(value.shape)
    if isinstance(value, np.void):
        # Structured scalars can expose writable fields and nested objects.
        return deepcopy(value)
    if isinstance(value, Mapping):
        return {
            key: _copy_sensing_snapshot(item, freeze_arrays=freeze_arrays)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_copy_sensing_snapshot(item, freeze_arrays=freeze_arrays) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_sensing_snapshot(item, freeze_arrays=freeze_arrays) for item in value)
    if value is None or isinstance(value, (str, bytes, int, float, complex, bool, np.generic)):
        return value
    return deepcopy(value)


class _ValidatedSensingProductFrame(dict[str, Any]):
    """Service-private projection, detached from both provider and consumers."""

    __slots__ = ("cache_token",)

    def __init__(self, frame: dict[str, Any], *, require_complete: bool = False) -> None:
        super().__init__(_copy_sensing_snapshot(frame, freeze_arrays=True))
        _assert_sensing_frame(self, require_described_products=require_complete)
        # An object token cannot alias a later projection after LRU eviction.
        self.cache_token = object()


@dataclass(frozen=True, slots=True)
class _ValidatedReplayIdentity:
    event: ReplayEvent
    manifest_id: str
    product_token: object | None = None


class _SensingSelectionMetadataFrame(dict[str, Any]):
    """Bounded internal metadata record used only for replay selection."""

    __slots__ = ("declared_products",)

    def __init__(
        self,
        sensing: dict[str, Any],
        *,
        declared_products: frozenset[str],
    ) -> None:
        super().__init__({"sensing": sensing})
        self.declared_products = declared_products


@dataclass(frozen=True, slots=True)
class SensingReplayTelemetrySnapshot:
    """Detached counters for one service-owned sensing replay data path."""

    metadata_probes: int = 0
    metadata_cache_hits: int = 0
    selected_dense_loads: int = 0
    selected_dense_cache_hits: int = 0
    replay_builds: int = 0
    replay_cache_hits: int = 0
    replay_rebindings: int = 0
    stage_loads: int = 0
    stage_cache_hits: int = 0
    index_builds: int = 0
    index_memory_hits: int = 0
    index_disk_hits: int = 0

    def to_manifest(self) -> dict[str, int]:
        """Return a stable JSON-safe counter mapping for benchmark evidence."""
        return {
            "metadata_probes": self.metadata_probes,
            "metadata_cache_hits": self.metadata_cache_hits,
            "selected_dense_loads": self.selected_dense_loads,
            "selected_dense_cache_hits": self.selected_dense_cache_hits,
            "replay_builds": self.replay_builds,
            "replay_cache_hits": self.replay_cache_hits,
            "replay_rebindings": self.replay_rebindings,
            "stage_loads": self.stage_loads,
            "stage_cache_hits": self.stage_cache_hits,
            "index_builds": self.index_builds,
            "index_memory_hits": self.index_memory_hits,
            "index_disk_hits": self.index_disk_hits,
        }


@dataclass(frozen=True, slots=True)
class SensingReplayCacheState:
    """Detached sizes and limits for the service-owned replay caches."""

    metadata_entries: int
    selected_product_entries: int
    selected_product_capacity: int
    detection_stage_entries: int
    detection_stage_capacity: int
    validated_event_entries: int = 0
    validated_event_capacity: int = _VALIDATED_REPLAY_CACHE_CAPACITY


class SensingReplayTelemetry:
    """Opt-in, process-local counters for replay selection and stage loading.

    A service created without an explicit collector owns a disabled instance,
    keeping normal application behavior observationally and computationally
    unchanged. Benchmark and diagnostic callers may inject an enabled collector
    and obtain atomic detached snapshots without parsing logs.
    """

    _COUNTER_NAMES = tuple(SensingReplayTelemetrySnapshot().to_manifest())

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = bool(enabled)
        self._counts = {name: 0 for name in self._COUNTER_NAMES}
        self._lock = RLock()

    def record(self, counter: str, count: int = 1) -> None:
        """Increment one declared counter when collection is enabled."""
        if not self._enabled:
            return
        if counter not in self._counts:
            raise KeyError(f"Unknown sensing replay telemetry counter: {counter}")
        increment = int(count)
        if increment < 0:
            raise ValueError("telemetry increments must be non-negative")
        with self._lock:
            self._counts[counter] += increment

    def set_enabled(self, enabled: bool) -> None:
        """Pause collection without discarding already observed counters."""
        with self._lock:
            self._enabled = bool(enabled)

    def snapshot(self) -> SensingReplayTelemetrySnapshot:
        """Return an immutable snapshot without resetting counters."""
        with self._lock:
            return SensingReplayTelemetrySnapshot(**self._counts)

    def reset(self) -> None:
        """Reset every counter while retaining the collector's enabled state."""
        if not self._enabled:
            return
        with self._lock:
            for name in self._counts:
                self._counts[name] = 0


def _needs_descriptor_bound_load(product_names: tuple[str, ...]) -> bool:
    """Return whether requested products are discovered through typed descriptors."""
    return bool(product_names)


def _descriptor_matches_contract(
    descriptor: Mapping[str, Any],
    contract: _SensingProductContract,
) -> bool:
    """Return whether descriptor semantics exactly match a supported contract."""
    axes = descriptor.get("axes")
    if not isinstance(axes, list):
        return False
    axis_names = tuple(axis.get("name") if isinstance(axis, Mapping) else None for axis in axes)
    axis_units = tuple(axis.get("unit") if isinstance(axis, Mapping) else None for axis in axes)
    return (
        descriptor.get("kind") == contract.kind
        and descriptor.get("variant") in {"primary", "raw"}
        and axis_names == contract.axis_names
        and axis_units == contract.axis_units
        and descriptor.get("representation") in contract.representations
    )


def _descriptor_bound_read_request(
    sensing: Mapping[str, Any],
    product_names: tuple[str, ...],
) -> FrameReadRequest:
    """Bind a second projection to exact descriptor data and coordinate keys."""
    available = {
        d.get("kind") for d in sensing.get("products", {}).values() if isinstance(d, Mapping)
    }
    # Input inspection prefers complete complex evidence when it exists. Avoid
    # materializing a duplicate frequency cube alongside retained delay samples.
    names = set(product_names)
    if "delay_domain_slow_time" in names:
        if "delay_domain_slow_time" in available:
            names.difference_update({"baseband_frequency_observation", "delay_domain_power"})
        elif "baseband_frequency_observation" in available:
            names.discard("delay_domain_power")
    if "baseband_frequency_observation" in names and "baseband_frequency_observation" in available:
        names.discard("baseband_frequency_power")
    product_names = tuple(name for name in product_names if name in names)
    descriptors = sensing.get("products") or {}
    exact_keys = {name for name in product_names if name in descriptors}
    contracts: set[_SensingProductContract] = set()
    requested: set[str] = set()
    for name in product_names:
        if name in exact_keys:
            requested.add(name)
        elif name in _RANGE_PROFILE_REQUEST_NAMES:
            contracts.add(_RANGE_PROFILE_CONTRACT)
        elif name in _RANGE_DOPPLER_REQUEST_NAMES:
            contracts.add(_RANGE_DOPPLER_CONTRACT)
        else:
            requested.add(name)

    products = sensing.get("products")
    if isinstance(products, Mapping):
        for data_key, descriptor in products.items():
            if (
                not isinstance(data_key, str)
                or not isinstance(descriptor, Mapping)
                or not (
                    data_key in exact_keys
                    or descriptor.get("kind") in product_names
                    or any(
                        _descriptor_matches_contract(descriptor, contract) for contract in contracts
                    )
                )
            ):
                continue
            if descriptor.get("kind") in {
                "delay_domain_slow_time",
                "baseband_frequency_observation",
            }:
                from ..sensing.processing_inspection import preflight_input

                preflight_input(descriptor)
            requested.add(data_key)
            axes = descriptor.get("axes")
            if not isinstance(axes, list):
                continue
            for axis in axes:
                coordinate_key = axis.get("coordinate_key") if isinstance(axis, Mapping) else None
                if isinstance(coordinate_key, str):
                    requested.add(coordinate_key)

    # Preserve a selective sensing projection when no matching descriptor exists.
    if not requested:
        requested.add("products")
    return FrameReadRequest(
        components=frozenset({FrameComponent.SENSING}),
        sensing_products=frozenset(requested),
    )


def _require_numeric_coordinate(
    coordinate: np.ndarray | list[Any],
    *,
    axis_name: str,
    size: int,
) -> np.ndarray:
    """Return one exact persisted real coordinate axis or fail closed."""
    values = np.asarray(coordinate)
    if values.ndim != 1 or values.shape != (size,) or not np.issubdtype(values.dtype, np.number):
        raise ValueError(
            f"Typed sensing axis {axis_name!r} must be a one-dimensional numeric array "
            f"with shape {(size,)}, got {values.shape}"
        )
    if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"Typed sensing axis {axis_name!r} must contain finite real values")
    values = values.astype(np.float64, copy=False)
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        raise ValueError(
            f"Typed sensing axis {axis_name!r} must be strictly increasing in persisted order"
        )
    return values


def _require_single_pair_product(
    sensing: Mapping[str, Any],
    *,
    contract: _SensingProductContract,
    variant: str,
) -> tuple[
    str,
    np.ndarray,
    tuple[np.ndarray, ...],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    """Resolve one exact typed product without key, rank, or pair inference."""
    data_key, data, coordinates, descriptor = select_product_by_contract(
        sensing,
        kind=contract.kind,
        variant=variant,
        axis_names=contract.axis_names,
        representations=contract.representations,
    )

    raw_axes = descriptor.get("axes")
    if not isinstance(raw_axes, list):
        raise ValueError(f"Typed sensing product {data_key!r} has no ordered axis descriptors")
    declared_units = tuple(
        axis.get("unit") if isinstance(axis, Mapping) else None for axis in raw_axes
    )
    if declared_units != contract.axis_units:
        raise ValueError(
            f"Typed sensing product {data_key!r} must declare axis units "
            f"{contract.axis_units!r}, got {declared_units!r}"
        )
    if data.ndim != len(contract.axis_names) or len(coordinates) != len(contract.axis_names):
        raise ValueError(
            f"Typed sensing product {data_key!r} rank/coordinate count disagrees with its axes"
        )

    pair_coordinates = coordinates[0]
    pair_ids = descriptor.get("pair_ids")
    if (
        not isinstance(pair_coordinates, list)
        or len(pair_coordinates) != 1
        or data.shape[0] != 1
        or not isinstance(pair_ids, list)
        or len(pair_ids) != 1
    ):
        raise ValueError(
            f"Typed sensing product {data_key!r} requires exactly one explicit pair; "
            "multi-pair visualization is unsupported"
        )
    pair = pair_coordinates[0]
    if not isinstance(pair, Mapping) or pair.get("pair_index") != 0:
        raise ValueError(
            f"Typed sensing product {data_key!r} requires pair_axis[0] with pair_index 0"
        )

    temporal = sensing.get("temporal")
    acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
    if not isinstance(acquisition, Mapping):
        raise ValueError(f"Typed sensing product {data_key!r} has no acquisition pair identity")
    for field in ("pair_id", "tx_sensor_id", "rx_sensor_id"):
        if pair.get(field) != acquisition.get(field):
            raise ValueError(
                f"Typed sensing product {data_key!r} pair_axis {field} disagrees with "
                "temporal acquisition identity"
            )
    if pair_ids[0] != pair.get("pair_id"):
        raise ValueError(
            f"Typed sensing product {data_key!r} descriptor pair_ids disagrees with pair_axis"
        )

    numeric_coordinates = tuple(
        _require_numeric_coordinate(
            coordinate,
            axis_name=axis_name,
            size=int(data.shape[axis_index]),
        )
        for axis_index, (coordinate, axis_name) in enumerate(
            zip(coordinates[1:], contract.axis_names[1:], strict=True),
            start=1,
        )
    )
    return data_key, np.asarray(data[0]), numeric_coordinates, descriptor, pair


def _contract_variants(
    sensing: Mapping[str, Any],
    contract: _SensingProductContract,
) -> frozenset[str]:
    """Return loaded, fully valid descriptor variants for one exact contract."""
    products = sensing.get("products")
    if not isinstance(products, Mapping):
        return frozenset()
    candidates: set[str] = set()
    for data_key, descriptor in products.items():
        if not isinstance(data_key, str) or not isinstance(descriptor, Mapping):
            continue
        variant = descriptor.get("variant")
        if (
            _descriptor_matches_contract(descriptor, contract)
            and isinstance(variant, str)
            and data_key in sensing
        ):
            candidates.add(variant)

    for variant in sorted(candidates):
        _require_single_pair_product(
            sensing,
            contract=contract,
            variant=variant,
        )
    return frozenset(candidates)


def range_profile_product_variants(sensing: Mapping[str, Any]) -> frozenset[str]:
    """Return exact loaded range-profile descriptor variants."""
    return _contract_variants(sensing, _RANGE_PROFILE_CONTRACT)


def range_doppler_product_variants(sensing: Mapping[str, Any]) -> frozenset[str]:
    """Return exact loaded range-Doppler descriptor variants."""
    return _contract_variants(sensing, _RANGE_DOPPLER_CONTRACT)


def _explicit_product_variant(
    requested: str | None,
    available: frozenset[str],
) -> str:
    """Resolve UI aliases to one explicit descriptor variant without fallback."""
    choice = str(requested or "auto").strip().lower()
    if choice == "clean":
        choice = "primary"
    if choice == "auto":
        if "primary" in available:
            return "primary"
        if "raw" in available:
            return "raw"
        raise KeyError("No loaded primary or raw typed sensing product is available")
    if choice not in {"primary", "raw"}:
        raise ValueError(f"Unsupported typed sensing product variant {choice!r}")
    if choice not in available:
        raise KeyError(f"No loaded typed sensing product has explicit variant {choice!r}")
    return choice


def _assert_sensing_frame(
    frame: dict[str, Any], *, require_described_products: bool = False
) -> None:
    """Validate the optional sensing extension before visualizer consumption."""
    sensing = frame.get("sensing")
    assert_supported_sensing_schema(sensing)
    assert_sensing_contract(
        frame,
        require_schema_version=True,
        require_described_products=require_described_products,
    )


@dataclass(frozen=True)
class SensingSceneLayer:
    """A renderer-ready sensing overlay layer."""

    name: str
    payload: PointCloudPayload | LineSetPayload
    material: MaterialPayload | None = None
    metadata: Mapping[str, Any] | None = None

    def to_render_object(self) -> RenderObject:
        """Return the complete renderer-neutral snapshot for this layer."""
        metadata = dict(self.metadata or {})
        metadata["type"] = "sensing_overlay"
        return RenderObject(
            id=self.name,
            payload=self.payload,
            material=self.material,
            metadata=metadata,
        )


@dataclass(frozen=True)
class SensingDisplaySelection:
    """One result selected through its explicit schema-v4 display binding."""

    display_frame: dict[str, Any]
    product_frame: dict[str, Any]
    storage_frame_id: int | str
    bound_scene_frame_id: int | str
    current_scene_frame_id: int | str
    reference_time_s: float
    current_scene_time_s: float | None
    scene_result_time_offset_s: float | None
    stale_age_s: float | None
    pinned: bool
    replay_event: ReplayEvent | None = None
    replay_reason: str | None = None
    _inventory_token: tuple[Any, ...] | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )
    _inventory_storage_ids: tuple[int | str, ...] = dataclass_field(
        default=(),
        repr=False,
        compare=False,
    )
    _replay_identity: _ValidatedReplayIdentity | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def stale_for_current_scene(self) -> bool:
        """Return whether explicit pinning shows a product bound elsewhere."""
        return self.bound_scene_frame_id != self.current_scene_frame_id

    @property
    def replay_event_reason(self) -> str | None:
        """Return the actionable reason canonical replay is unavailable."""
        return self.replay_reason


class SensingService(BaseService):
    """Read-only sensing product adapter for the visualizer.

    The visualizer primarily consumes generator-produced sensing payloads
    (CIR, range profile, range–Doppler maps, detections) rather than
    reprocessing raw data interactively. The service keeps a short CIR
    history only to support viewer-side micro-Doppler fallbacks.
    """

    def __init__(
        self,
        visualizer: OrchavVisualizer,
        *,
        cir_provider: Optional[CIRProvider] = None,
        history_size: int = 128,
        replay_telemetry: SensingReplayTelemetry | None = None,
    ) -> None:
        """Initialize stored sensing access and bounded CIR history."""
        super().__init__()
        self.visualizer = visualizer
        # Default to stored sensing data provider; callers can override explicitly.
        self.cir_provider: CIRProvider = cir_provider or StoredCIRProvider()
        self.cir_history: Deque[CIRComputationResult] = deque(maxlen=history_size)
        self._scene_overlay_sync = SensingOverlaySync()
        self.path_context = AnalyticalPathContext()
        self._analytical_source_token: tuple | None = None
        self._analytical_sources: dict[int, dict[str, Any]] = {}
        # Metadata is indexed once per immutable provider inventory.  Product
        # arrays are loaded only for the selected event, avoiding a broad
        # range-Doppler read on every 500 ms dashboard refresh.
        self._sensing_index_token: tuple[Any, ...] | None = None
        self._sensing_cache_lock = RLock()
        self._storage_snapshot: tuple[int, ...] | None = None
        self._storage_snapshot_provider: Any = None
        self._storage_candidates: tuple[int | str, ...] = ()
        self._sensing_metadata_index: dict[tuple[type[Any], int | str], dict[str, Any] | None] = {}
        self._sensing_result_index: tuple[tuple[int | str, dict[str, Any]], ...] | None = None
        self._scene_result_index_rows = None
        self._scene_result_index: dict[int | str, tuple[tuple[int | str, dict[str, Any]], ...]] = {}
        self._dashboard_scene_metadata_rows = None
        self._dashboard_scene_metadata: dict[tuple, dict[str, Any] | None] = {}
        self._sensing_selected_product_cache: OrderedDict[
            tuple[type[Any], int | str, tuple[str, ...], bool], dict[str, Any]
        ] = OrderedDict()
        self._detection_stage_bundle_cache: OrderedDict[str, DetectionStageBundle] = OrderedDict()
        self._validated_replay_cache: OrderedDict[tuple[Any, ...], _ValidatedReplayIdentity] = (
            OrderedDict()
        )
        # Bound, validated detection-only records; no CIR/angular/MPC arrays.
        self._overlay_history_cache: OrderedDict[tuple[type[Any], int | str], tuple] = OrderedDict()
        self._scene_overlay_cache: OrderedDict[
            tuple, tuple[dict, list[SensingSceneLayer]] | None
        ] = OrderedDict()
        self._current_detection_cache: OrderedDict[tuple, tuple[dict, list[SensingSceneLayer]]] = (
            OrderedDict()
        )
        self._replay_telemetry = replay_telemetry or SensingReplayTelemetry(enabled=False)
        self._reader_cancelled: Event | None = None
        self._reader_progress: Callable[[str, int, int], None] | None = None
        self.recording_cache: RecordingCache | None = None
        self.prepared_scene = None

    def configure_background_reader(
        self, cancelled: Event, progress: Callable[[str, int, int], None]
    ) -> None:
        """Install worker-only cancellation/progress; callbacks must not access Qt."""
        self._reader_cancelled = cancelled
        self._reader_progress = progress

    def report_read_progress(self, stage: str, completed: int, total: int) -> None:
        if self._reader_cancelled is not None and self._reader_cancelled.is_set():
            raise CancelledError("Sensing loading cancelled")
        if self._reader_progress is not None:
            self._reader_progress(stage, completed, total)

    def replay_telemetry_snapshot(self) -> SensingReplayTelemetrySnapshot:
        """Return detached replay/load counters for this service instance."""
        return self._replay_telemetry.snapshot()

    def set_replay_telemetry_enabled(self, enabled: bool) -> None:
        """Collect reader counters only while diagnostics explicitly requests them."""
        self._replay_telemetry.set_enabled(enabled)

    def reset_replay_telemetry(self) -> None:
        """Reset opt-in replay/load counters without changing data caches."""
        self._replay_telemetry.reset()

    def replay_cache_state(self) -> SensingReplayCacheState:
        """Return a thread-safe snapshot of replay-cache occupancy and limits."""
        with self._sensing_cache_lock:
            return SensingReplayCacheState(
                metadata_entries=len(self._sensing_metadata_index),
                selected_product_entries=len(self._sensing_selected_product_cache),
                selected_product_capacity=_SELECTED_PRODUCT_CACHE_CAPACITY,
                detection_stage_entries=len(self._detection_stage_bundle_cache),
                detection_stage_capacity=_DETECTION_STAGE_BUNDLE_CACHE_CAPACITY,
                validated_event_entries=len(self._validated_replay_cache),
            )

    @staticmethod
    def _metadata_only_sensing_frame(
        frame: dict[str, Any],
    ) -> _SensingSelectionMetadataFrame | None:
        """Build the bounded metadata record needed by replay selection.

        The source contract is validated before projection.  Only scalar event
        identity plus the descriptor fields used to bind a second selective
        read are copied.  Arbitrary config/debug mappings, acquisition sample
        lists, detections, and numeric arrays are represented only by their
        top-level declared names and therefore cannot be pinned by the
        inventory-sized metadata index. A bounded RX/pair digest preserves
        array identity without retaining its configuration mappings.
        """
        _assert_sensing_frame(frame)
        sensing = frame.get("sensing")
        if not isinstance(sensing, Mapping) or not sensing:
            return None

        temporal = sensing.get("temporal")
        clock = temporal.get("clock") if isinstance(temporal, Mapping) else None
        acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
        result = temporal.get("result") if isinstance(temporal, Mapping) else None
        display_binding = temporal.get("display_binding") if isinstance(temporal, Mapping) else None
        projected_temporal: dict[str, Any] = {
            "clock": {"clock_id": clock.get("clock_id") if isinstance(clock, Mapping) else None},
            "acquisition": {
                "pair_id": acquisition.get("pair_id") if isinstance(acquisition, Mapping) else None,
                "acquisition_id": (
                    acquisition.get("acquisition_id") if isinstance(acquisition, Mapping) else None
                ),
            },
            "result": {
                "reference_time_s": (
                    result.get("reference_time_s") if isinstance(result, Mapping) else None
                ),
                "processing_run_id": (
                    result.get("processing_run_id") if isinstance(result, Mapping) else None
                ),
            },
            "display_binding": {
                "scene_frame_id": (
                    display_binding.get("scene_frame_id")
                    if isinstance(display_binding, Mapping)
                    else None
                ),
            },
        }

        projected_products: dict[str, dict[str, Any]] = {}
        products = sensing.get("products")
        if isinstance(products, Mapping):
            for data_key, descriptor in products.items():
                if not isinstance(data_key, str) or not isinstance(descriptor, Mapping):
                    continue
                axes = descriptor.get("axes")
                projected_axes = []
                if isinstance(axes, list):
                    for axis in axes:
                        if not isinstance(axis, Mapping):
                            continue
                        projected_axes.append(
                            {name: axis.get(name) for name in ("name", "unit", "coordinate_key")}
                        )
                projected_products[data_key] = {
                    "kind": descriptor.get("kind"),
                    "variant": descriptor.get("variant"),
                    "representation": descriptor.get("representation"),
                    "shape": descriptor.get("shape"),
                    "dtype": descriptor.get("dtype"),
                    "axes": projected_axes,
                }

        projected_sensing = {
            "schema_version": sensing.get("schema_version"),
            "temporal": projected_temporal,
            "products": projected_products,
            "_analytical_path_source": analytical_source_record(sensing),
            "_receive_array_signature": receive_metadata_signature(sensing),
        }
        declared_products = frozenset(
            str(name)
            for name, value in sensing.items()
            if isinstance(name, str)
            and value is not None
            and (
                (isinstance(value, np.ndarray) and value.size > 0)
                or (isinstance(value, (Mapping, list, tuple, str, bytes)) and len(value) > 0)
                or not isinstance(value, (np.ndarray, Mapping, list, tuple, str, bytes))
            )
        )
        return _SensingSelectionMetadataFrame(
            projected_sensing,
            declared_products=declared_products,
        )

    def _selected_product_cache_get(
        self,
        key: tuple[type[Any], int | str, tuple[str, ...], bool],
        *,
        index_token: tuple[Any, ...] | None,
    ) -> dict[str, Any] | None:
        """Return and promote one selected dense projection."""
        with self._sensing_cache_lock:
            if index_token is not None and index_token != self._sensing_index_token:
                raise _SensingInventoryChanged
            cached = self._sensing_selected_product_cache.get(key)
            if cached is None and not key[3]:
                # A validated owner can supply a narrower view without another
                # read. Exact descriptor keys keep primary/raw demand distinct.
                for owner_key, owner in reversed(self._sensing_selected_product_cache.items()):
                    descriptors = (owner.get("sensing") or {}).get("products") or {}
                    if (
                        owner_key[:2] == key[:2]
                        and any(name in descriptors for name in key[2])
                        and self._has_all_sensing_products(owner, key[2])
                    ):
                        cached = owner
                        key = owner_key
                        break
            if cached is not None:
                self._sensing_selected_product_cache.move_to_end(key)
            return cached

    def _selected_product_cache_put(
        self,
        key: tuple[type[Any], int | str, tuple[str, ...], bool],
        frame: _ValidatedSensingProductFrame,
        *,
        index_token: tuple[Any, ...] | None,
    ) -> None:
        """Store an owned projection after validation and metadata agreement."""
        with self._sensing_cache_lock:
            if index_token is not None and index_token != self._sensing_index_token:
                raise _SensingInventoryChanged
            self._sensing_selected_product_cache[key] = frame
            self._sensing_selected_product_cache.move_to_end(key)
            while len(self._sensing_selected_product_cache) > _SELECTED_PRODUCT_CACHE_CAPACITY:
                self._sensing_selected_product_cache.popitem(last=False)

    def _detection_stage_cache_get(
        self,
        key: str,
        *,
        index_token: tuple[Any, ...] | None,
    ) -> DetectionStageBundle | None:
        """Return and promote one detector-stage bundle."""
        with self._sensing_cache_lock:
            if index_token is not None and index_token != self._sensing_index_token:
                raise _SensingInventoryChanged
            cached = self._detection_stage_bundle_cache.get(key)
            if cached is not None:
                self._detection_stage_bundle_cache.move_to_end(key)
            return cached

    def _detection_stage_cache_put(
        self,
        key: str,
        bundle: DetectionStageBundle,
        *,
        index_token: tuple[Any, ...] | None,
    ) -> None:
        """Store one detector-stage bundle and evict the least-recently used."""
        with self._sensing_cache_lock:
            if index_token is not None and index_token != self._sensing_index_token:
                return
            self._detection_stage_bundle_cache[key] = bundle
            self._detection_stage_bundle_cache.move_to_end(key)
            while len(self._detection_stage_bundle_cache) > _DETECTION_STAGE_BUNDLE_CACHE_CAPACITY:
                self._detection_stage_bundle_cache.popitem(last=False)

    @staticmethod
    def _display_binding_record(frame: dict[str, Any]) -> Mapping[str, Any] | None:
        """Return a validated result display binding, if the frame has one."""
        if not isinstance(frame, (_SensingSelectionMetadataFrame, _ValidatedSensingProductFrame)):
            _assert_sensing_frame(frame)
        sensing = frame.get("sensing")
        if not isinstance(sensing, Mapping) or not sensing:
            return None
        temporal = sensing.get("temporal")
        if not isinstance(temporal, Mapping) or not isinstance(temporal.get("result"), Mapping):
            return None
        binding = temporal.get("display_binding")
        return binding if isinstance(binding, Mapping) else None

    @staticmethod
    def _has_any_sensing_product(
        frame: dict[str, Any],
        product_names: tuple[str, ...],
    ) -> bool:
        """Return whether an exact requested typed contract or JSON product is loaded."""
        sensing = frame.get("sensing")
        if not isinstance(sensing, Mapping):
            return False
        if not product_names:
            return True
        for name in product_names:
            if name in _RANGE_PROFILE_REQUEST_NAMES:
                if range_profile_product_variants(sensing):
                    return True
            elif name in _RANGE_DOPPLER_REQUEST_NAMES:
                if range_doppler_product_variants(sensing):
                    return True
            else:
                value = sensing.get(name)
                if value is None:
                    continue
                try:
                    if np.size(value) > 0:
                        return True
                except (TypeError, ValueError):
                    continue
        return False

    @staticmethod
    def _has_all_sensing_products(frame: dict[str, Any], names: tuple[str, ...]) -> bool:
        """Check every exact requested array while retaining legacy kind requests."""
        sensing = frame.get("sensing") or {}
        descriptors = sensing.get("products") or {}
        return all(
            (
                isinstance(sensing.get(name), np.ndarray) and sensing[name].size > 0
                if name in descriptors
                else SensingService._has_any_sensing_product(frame, (name,))
            )
            for name in names
        )

    @staticmethod
    def _has_complete_sensing_payload(frame: dict[str, Any]) -> bool:
        """Return whether every product declared by the frame is materialized."""
        try:
            assert_sensing_contract(
                frame,
                require_schema_version=True,
                require_described_products=True,
            )
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _declares_any_sensing_product(
        frame: dict[str, Any],
        product_names: tuple[str, ...],
    ) -> bool:
        """Return whether metadata declares a requested typed/JSON product.

        Unlike :meth:`_has_any_sensing_product`, this predicate intentionally
        does not require dense values to be loaded.  It is used only for the
        first, metadata-only event-index pass; the selected event is loaded and
        validated against its exact descriptor before it is returned.
        """
        sensing = frame.get("sensing")
        if not isinstance(sensing, Mapping):
            return False
        if not product_names:
            return True
        products = sensing.get("products")
        descriptors = products.values() if isinstance(products, Mapping) else ()
        for name in product_names:
            if name in _RANGE_PROFILE_REQUEST_NAMES and any(
                isinstance(descriptor, Mapping)
                and _descriptor_matches_contract(descriptor, _RANGE_PROFILE_CONTRACT)
                for descriptor in descriptors
            ):
                return True
            if name in _RANGE_DOPPLER_REQUEST_NAMES and any(
                isinstance(descriptor, Mapping)
                and _descriptor_matches_contract(descriptor, _RANGE_DOPPLER_CONTRACT)
                for descriptor in descriptors
            ):
                return True
            if (
                isinstance(frame, _SensingSelectionMetadataFrame)
                and name in frame.declared_products
            ):
                return True
            value = sensing.get(name)
            if value is None:
                continue
            try:
                if np.size(value) > 0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _current_metadata_index_token(
        self,
        storage_ids: tuple[int | str, ...],
    ) -> tuple[Any, ...]:
        """Return the provider generation and exact storage inventory token."""
        provider = visualizer_frame_provider(self.visualizer)
        frame_source = getattr(self.visualizer, "frame_source", None)
        provider_info = None
        try:
            provider_info = getattr(provider, "info", None) if provider is not None else None
        except (OSError, RuntimeError, TypeError, ValueError):
            provider_info = None
        live_provider = getattr(frame_source, "provider", None)
        return (
            id(provider),
            getattr(provider_info, "generation_id", None),
            getattr(provider_info, "frame_set_id", None),
            getattr(live_provider, "_generation_epoch", None),
            id(getattr(provider, "frame_ids_snapshot", None)),
            # Candidates contain only int/str (never bool), so the immutable
            # tuple already distinguishes typed IDs. Do not copy 24k IDs for
            # every generation guard during an ordinary display update.
            storage_ids,
        )

    def _install_metadata_index_token(self, token: tuple[Any, ...]) -> None:
        """Install one inventory token and invalidate generation-bound caches."""
        with self._sensing_cache_lock:
            if token != self._sensing_index_token:
                self._sensing_index_token = token
                self._sensing_metadata_index.clear()
                self._sensing_result_index = None
                self._scene_result_index_rows = None
                self._scene_result_index.clear()
                self._dashboard_scene_metadata_rows = None
                self._dashboard_scene_metadata.clear()
                self._sensing_selected_product_cache.clear()
                self._detection_stage_bundle_cache.clear()
                self._validated_replay_cache.clear()
                self._overlay_history_cache.clear()
                self._scene_overlay_cache.clear()
                self._current_detection_cache.clear()

    def _refresh_metadata_index_token(
        self,
        storage_ids: tuple[int | str, ...],
    ) -> tuple[Any, ...]:
        """Pin and return the current immutable-provider inventory token."""
        token = self._current_metadata_index_token(storage_ids)
        with self._sensing_cache_lock:
            self._install_metadata_index_token(token)
            # Reuse the canonical tuple object. Rebuilding an equal inventory
            # tuple here made every cached-row guard compare the entire frame
            # inventory, turning a warm metadata scan into quadratic work.
            assert self._sensing_index_token is not None
            return self._sensing_index_token

    def _indexed_sensing_results(
        self, storage_ids: tuple[int | str, ...], expected_token: tuple[Any, ...]
    ) -> tuple[tuple[int | str, dict[str, Any]], ...]:
        """Share one validated index across readers of the same local provider."""
        with self._sensing_cache_lock:
            if self._sensing_index_token is not expected_token:
                raise _SensingInventoryChanged
            if self._sensing_result_index is not None:
                self._replay_telemetry.record("index_memory_hits")
                return self._sensing_result_index
        provider = visualizer_frame_provider(self.visualizer)
        cache = RecordingCache.for_provider(provider)
        self.recording_cache = cache
        if not isinstance(provider, Hdf5Provider) or cache is None:
            return self._build_sensing_result_index(storage_ids, expected_token, cache)

        identity = (expected_token, cache.signature)
        while True:
            with _SHARED_RESULT_INDEX_LOCK:
                entry = _SHARED_RESULT_INDEXES.get(provider)
                if entry is None or entry[0] != identity:
                    future = Future()
                    _SHARED_RESULT_INDEXES[provider] = (identity, future)
                    owner = True
                else:
                    future = entry[1]
                    owner = False
            if owner:
                try:
                    rows = self._build_sensing_result_index(storage_ids, expected_token, cache)
                    if not cache.unchanged():
                        raise _SensingInventoryChanged
                except BaseException as exc:
                    future.set_exception(exc)
                    with _SHARED_RESULT_INDEX_LOCK:
                        if _SHARED_RESULT_INDEXES.get(provider) == (identity, future):
                            del _SHARED_RESULT_INDEXES[provider]
                    raise
                future.set_result(rows)
                return rows
            self.report_read_progress("Waiting for result index", 0, len(storage_ids))
            try:
                rows = future.result(timeout=0.1)
            except TimeoutError:
                if future.done():
                    raise
                continue
            except CancelledError:
                continue
            self._require_metadata_index_token(storage_ids, expected_token)
            if not cache.unchanged():
                raise _SensingInventoryChanged
            rows = deepcopy(rows)
            with self._sensing_cache_lock:
                if self._sensing_index_token is not expected_token:
                    raise _SensingInventoryChanged
                self._sensing_result_index = rows
                self._sensing_metadata_index.update({(type(i), i): f for i, f in rows})
            self._replay_telemetry.record("index_memory_hits")
            self.report_read_progress("Shared result index", len(storage_ids), len(storage_ids))
            return rows

    def _build_sensing_result_index(self, storage_ids, expected_token, cache):
        """Load or build one service-owned index from the authoritative source."""
        self.report_read_progress("Indexing results", 0, len(storage_ids))
        cached = (
            cache.load("index", ("input-preflight-v3-rx-array",)) if cache is not None else None
        )
        if cached is not None:
            try:
                allowed = set(storage_ids)
                restored = []
                for row in cached:
                    storage_id, sensing, declared = row
                    if storage_id not in allowed or not isinstance(declared, list):
                        raise ValueError("Cached index belongs to a different inventory")
                    assert_supported_sensing_schema(sensing)
                    frame = _SensingSelectionMetadataFrame(
                        sensing, declared_products=frozenset(declared)
                    )
                    if self._display_binding_record(frame) is None:
                        raise ValueError("Cached result has no binding")
                    reference_time = sensing["temporal"]["result"]["reference_time_s"]
                    if reference_time is not None and not math.isfinite(float(reference_time)):
                        raise ValueError("Cached result has an invalid physical epoch")
                    restored.append((storage_id, frame))
                self._require_metadata_index_token(storage_ids, expected_token)
                with self._sensing_cache_lock:
                    self._sensing_result_index = tuple(restored)
                    self._sensing_metadata_index.update({(type(i), i): f for i, f in restored})
                self.report_read_progress("Cached result index", len(storage_ids), len(storage_ids))
                self._replay_telemetry.record("index_disk_hits")
                return self._sensing_result_index
            except (KeyError, TypeError, ValueError):
                logger.debug("Discarding invalid sensing display index cache", exc_info=True)
        self._replay_telemetry.record("index_builds")
        rows = []
        for index, (storage_id, metadata) in enumerate(
            self._iter_sensing_metadata_candidates(storage_ids, expected_token)
        ):
            self.report_read_progress("Indexing results", index + 1, len(storage_ids))
            if metadata is not None and self._display_binding_record(metadata) is not None:
                rows.append((storage_id, metadata))
        self._require_metadata_index_token(storage_ids, expected_token)
        with self._sensing_cache_lock:
            if self._sensing_index_token is not expected_token:
                raise _SensingInventoryChanged
            self._sensing_result_index = tuple(rows)
        if cache is not None:
            saved = cache.save(
                "index",
                [(i, f["sensing"], sorted(f.declared_products)) for i, f in rows],
                ("input-preflight-v3-rx-array",),
            )
            if cache._building and not saved:
                raise OSError("Could not save the sensing result index")
        return self._sensing_result_index

    def _iter_sensing_metadata_candidates(
        self, storage_ids: tuple[int | str, ...], expected_token: tuple[Any, ...]
    ) -> Iterator[tuple[int | str, dict[str, Any] | None]]:
        """Batch cold descriptor reads; keep cache, validation, and fallback semantics."""
        with self._sensing_cache_lock:
            missing = tuple(
                value
                for value in storage_ids
                if (type(value), value) not in self._sensing_metadata_index
            )
        cache = getattr(self.visualizer, "cache_service", None)
        if cache is not None:
            # Complete cached frames already provide the validation input. Do
            # not prefetch those rows merely because the derived index is cold.
            missing = tuple(
                value for value in missing if not ((cache.get_frame(value) or {}).get("sensing"))
            )
        missing_set = {(type(value), value) for value in missing}
        provider = visualizer_frame_provider(self.visualizer)
        batch_loader = getattr(provider, "iter_frame_projections", None)
        batch = None
        if missing and callable(batch_loader):
            try:
                batch = iter(batch_loader(missing, _DESCRIPTOR_PROBE_REQUEST))
            except UnsupportedSensingSchemaError:
                raise
            except (OSError, KeyError, TypeError, RuntimeError):
                logger.debug("Provider cannot batch sensing metadata", exc_info=True)
        try:
            for storage_id in storage_ids:
                self._require_installed_metadata_index_token(expected_token)
                projection = None
                if batch is not None and (type(storage_id), storage_id) in missing_set:
                    try:
                        projection = next(batch)
                        if projection.frame.frame_index != storage_id:
                            raise RuntimeError(
                                "Sensing metadata batch changed requested frame order"
                            )
                    except UnsupportedSensingSchemaError:
                        raise
                    except (OSError, KeyError, TypeError, RuntimeError, StopIteration):
                        logger.debug(
                            "Batch metadata read failed; retrying individual rows", exc_info=True
                        )
                        close = getattr(batch, "close", None)
                        if callable(close):
                            close()
                        batch = None
                        projection = None
                yield storage_id, self._load_sensing_metadata_candidate(
                    storage_id,
                    storage_ids=storage_ids,
                    expected_index_token=expected_token,
                    prefetched_projection=projection,
                )
        finally:
            close = getattr(batch, "close", None)
            if callable(close):
                close()

    def _require_metadata_index_token(
        self,
        storage_ids: tuple[int | str, ...],
        expected: tuple[Any, ...],
    ) -> None:
        """Fail one attempt when provider generation changes during a read."""
        observed = self._current_metadata_index_token(storage_ids)
        if observed != expected:
            self._install_metadata_index_token(observed)
            raise _SensingInventoryChanged
        with self._sensing_cache_lock:
            if self._sensing_index_token != expected:
                raise _SensingInventoryChanged

    def _require_installed_metadata_index_token(self, expected: tuple[Any, ...]) -> None:
        """Reject an attempt already invalidated by another service thread."""
        with self._sensing_cache_lock:
            if self._sensing_index_token != expected:
                raise _SensingInventoryChanged

    def _load_sensing_metadata_candidate(
        self,
        storage_frame_id: int | str,
        *,
        storage_ids: tuple[int | str, ...] = (),
        expected_index_token: tuple[Any, ...] | None = None,
        prefetched_projection: FrameProjection | None = None,
    ) -> dict[str, Any] | None:
        """Load and cache one metadata-only sensing event projection."""
        if expected_index_token is not None:
            self._require_installed_metadata_index_token(expected_index_token)
        identity = (type(storage_frame_id), storage_frame_id)
        with self._sensing_cache_lock:
            start_index_token = self._sensing_index_token
            cache_hit = identity in self._sensing_metadata_index
            indexed = self._sensing_metadata_index.get(identity)
        if cache_hit:
            if expected_index_token is not None:
                self._require_installed_metadata_index_token(expected_index_token)
            self._replay_telemetry.record("metadata_cache_hits")
            return indexed
        self._replay_telemetry.record("metadata_probes")

        metadata_frame: dict[str, Any] | None = None
        cache = getattr(self.visualizer, "cache_service", None)
        if cache is not None:
            cached = cache.get_frame(storage_frame_id)
            if cached is not None:
                metadata_frame = self._metadata_only_sensing_frame(cached)
                if expected_index_token is not None:
                    self._require_installed_metadata_index_token(expected_index_token)

        provider = visualizer_frame_provider(self.visualizer)
        projection_loaded = False
        if metadata_frame is None and provider is not None:
            try:
                projection = (
                    prefetched_projection
                    if prefetched_projection is not None
                    else provider.load_frame_projection(
                        storage_frame_id,
                        _DESCRIPTOR_PROBE_REQUEST,
                    )
                )
            except UnsupportedSensingSchemaError:
                raise
            except (OSError, KeyError, TypeError, RuntimeError):
                projection = None
            if projection is not None:
                # An empty successful projection establishes that this row has
                # no sensing event. Loading the complete frame cannot add one.
                # A scene-only cache entry does not provide that guarantee.
                projection_loaded = True
                metadata_frame = self._metadata_only_sensing_frame(
                    {"sensing": dict(projection.frame.sensing or {})}
                )
                if expected_index_token is not None:
                    self._require_installed_metadata_index_token(expected_index_token)

        frame_source = getattr(self.visualizer, "frame_source", None)
        if (
            metadata_frame is None
            and not projection_loaded
            and frame_source is not None
            and hasattr(frame_source, "load_frame")
        ):
            try:
                frame = frame_source.load_frame(storage_frame_id)
            except UnsupportedSensingSchemaError:
                raise
            except (OSError, KeyError, RuntimeError):
                frame = None
            if frame is not None:
                if not isinstance(frame, StandardMPCFrame):
                    raise TypeError("frame sources must return StandardMPCFrame")
                projection = project_standard_mpc_frame(frame, _DESCRIPTOR_PROBE_REQUEST)
                metadata_frame = self._metadata_only_sensing_frame(
                    {"sensing": dict(projection.frame.sensing or {})}
                )
                if expected_index_token is not None:
                    self._require_installed_metadata_index_token(expected_index_token)

        with self._sensing_cache_lock:
            if (
                expected_index_token == self._sensing_index_token
                if expected_index_token is not None
                else start_index_token == self._sensing_index_token
            ):
                self._sensing_metadata_index[identity] = metadata_frame
        return metadata_frame

    @staticmethod
    def _selection_key(
        frame: dict[str, Any],
        storage_frame_id: int | str,
    ) -> tuple[float, str, str, str]:
        """Return a cache-order-independent key for otherwise equivalent results."""
        sensing = frame.get("sensing") or {}
        temporal = sensing.get("temporal") or {}
        result = temporal.get("result") or {}
        acquisition = temporal.get("acquisition") or {}
        return (
            float(result.get("reference_time_s", float("-inf"))),
            str(result.get("processing_run_id", "")),
            str(acquisition.get("acquisition_id", "")),
            f"{type(storage_frame_id).__name__}:{storage_frame_id!r}",
        )

    @staticmethod
    def _compose_bound_display_frame(
        scene_frame: dict[str, Any],
        product_frame: dict[str, Any],
        *,
        bound_scene_frame: Mapping[str, Any] | None = None,
        storage_frame_id: int | str,
        bound_scene_frame_id: int | str,
        current_scene_frame_id: int | str,
        pinned: bool,
    ) -> dict[str, Any]:
        """Attach selected products to the bound scene's geometry and device pose."""
        current_scene_time_s = SensingService._scene_time_s(scene_frame)
        sensing = product_frame.get("sensing") or {}
        temporal = sensing.get("temporal") or {}
        result = temporal.get("result") or {}
        reference_time_s = result.get("reference_time_s")
        scene_result_time_offset_s = (
            current_scene_time_s - float(reference_time_s)
            if current_scene_time_s is not None
            and isinstance(reference_time_s, (int, float))
            and not isinstance(reference_time_s, bool)
            and math.isfinite(float(reference_time_s))
            else None
        )
        stale_age_s = (
            max(scene_result_time_offset_s, 0.0)
            if pinned
            and bound_scene_frame_id != current_scene_frame_id
            and scene_result_time_offset_s is not None
            and scene_result_time_offset_s >= 0.0
            else None
        )
        display_frame = dict(bound_scene_frame or scene_frame)
        display_frame["sensing"] = dict(sensing)
        display_frame["_sensing_display_provenance"] = {
            "storage_frame_id": storage_frame_id,
            "bound_scene_frame_id": bound_scene_frame_id,
            "current_scene_frame_id": current_scene_frame_id,
            "current_scene_time_s": current_scene_time_s,
            "scene_result_time_offset_s": scene_result_time_offset_s,
            "stale_age_s": stale_age_s,
            "pinned": bool(pinned),
            "stale_for_current_scene": bound_scene_frame_id != current_scene_frame_id,
        }
        return display_frame

    @staticmethod
    def _scene_time_s(frame: Mapping[str, Any]) -> float | None:
        """Return an explicitly unit-tagged physical scene timestamp, if present."""
        candidates = (("timestamp_s", 1.0), ("timestamp_ns", 1e-9))
        for key, scale in candidates:
            value = frame.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                time_s = float(value) * scale
                if math.isfinite(time_s):
                    return time_s
        source = frame.get("_source")
        if isinstance(source, Mapping):
            value = source.get("timestamp")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                time_s = float(value)
                if math.isfinite(time_s):
                    return time_s
        return None

    def _selection_replay_event(
        self,
        replay_frame: dict[str, Any],
        *,
        storage_frame_id: int | str,
        bound_scene_frame_id: int | str,
        current_scene_frame_id: int | str,
        bound_scene_frame: Mapping[str, Any],
        current_scene_frame: Mapping[str, Any],
        reference_time_s: float,
        scene_result_time_offset_s: float | None,
        stale_age_s: float | None,
        pinned: bool,
    ) -> tuple[ReplayEvent | None, str | None]:
        """Build and cross-check canonical replay for one resolved selection.

        Selection remains useful to existing visual consumers when strict replay
        cannot be formed.  The returned reason makes that fail-closed state
        explicit instead of silently presenting a partial event as canonical.
        """
        bound_scene_time_s = self._scene_time_s(bound_scene_frame)
        current_scene_time_s = self._scene_time_s(current_scene_frame)
        missing_times = [
            label
            for label, value in (
                ("bound scene time", bound_scene_time_s),
                ("current scene time", current_scene_time_s),
            )
            if value is None
        ]
        if missing_times:
            return None, "Canonical replay unavailable: missing " + " and ".join(missing_times)

        try:
            self._replay_telemetry.record("replay_builds")
            event = build_replay_event(
                replay_frame,
                ReplaySceneContext(
                    storage_frame_id=storage_frame_id,
                    bound_scene_frame_id=bound_scene_frame_id,
                    current_scene_frame_id=current_scene_frame_id,
                    bound_scene_time_s=bound_scene_time_s,
                    current_scene_time_s=current_scene_time_s,
                    pinned=pinned,
                ),
                require_product_data=False,
            )
        except (TypeError, ValueError) as exc:
            return None, f"Canonical replay unavailable: {exc}"

        binding = event.scene_binding
        identity_mismatches: list[str] = []
        if not math.isclose(
            event.key.reference_time_s,
            reference_time_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            identity_mismatches.append("reference time")
        for label, actual, expected in (
            ("storage frame", binding.storage_frame_id, storage_frame_id),
            ("bound scene", binding.bound_scene_frame_id, bound_scene_frame_id),
            ("current scene", binding.current_scene_frame_id, current_scene_frame_id),
            ("pin state", binding.pinned, pinned),
        ):
            if actual != expected:
                identity_mismatches.append(label)
        if scene_result_time_offset_s is None or not math.isclose(
            binding.current_scene_result_offset_s,
            scene_result_time_offset_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            identity_mismatches.append("current scene/result offset")
        if binding.stale:
            if stale_age_s is None or not math.isclose(
                binding.stale_age_s,
                stale_age_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                identity_mismatches.append("stale age")
        if identity_mismatches:
            return (
                None,
                "Canonical replay unavailable: replay/selection mismatch in "
                + ", ".join(identity_mismatches),
            )

        scene_check = next(
            (check for check in event.preflight.checks if check.check_id == "scene_alignment"),
            None,
        )
        expected_scene_status = "warning" if binding.stale else "pass"
        if scene_check is None or scene_check.status != expected_scene_status:
            return None, "Canonical replay unavailable: scene preflight disagrees with selection"
        if event.preflight.disposition == "blocked":
            failed = [
                f"{check.message} Action: {check.action}"
                for check in event.preflight.checks
                if check.status == "fail"
            ]
            return None, "Canonical replay blocked: " + "; ".join(failed)
        return event, None

    @staticmethod
    def _validated_replay_key(
        product_frame: dict[str, Any],
        replay_frame: dict[str, Any],
        *,
        storage_frame_id: int | str,
        bound_scene_frame_id: int | str,
        current_scene_frame_id: int | str,
        bound_scene_frame: Mapping[str, Any],
        current_scene_frame: Mapping[str, Any],
        pinned: bool,
    ) -> tuple[Any, ...] | None:
        """Bind private scientific data to the exact mutable display context.

        Only service-owned projections are eligible. External/current frames
        and custom loaders keep the full validation boundary on every call.
        Scene poses are small and mutable, so compare their relevant values,
        never their object IDs. Invalid context falls through to the ordinary
        canonical builder, which supplies the actionable failure reason.
        """
        if not isinstance(product_frame, _ValidatedSensingProductFrame):
            return None
        bound_time = SensingService._scene_time_s(bound_scene_frame)
        current_time = SensingService._scene_time_s(current_scene_frame)
        if bound_time is None or current_time is None:
            return None
        try:
            context = ReplaySceneContext(
                storage_frame_id=storage_frame_id,
                bound_scene_frame_id=bound_scene_frame_id,
                current_scene_frame_id=current_scene_frame_id,
                bound_scene_time_s=bound_time,
                current_scene_time_s=current_time,
                pinned=pinned,
            )
            frame_index = replay_frame.get("frame_index")
            if frame_index is not None and (
                not isinstance(frame_index, (int, str)) or isinstance(frame_index, bool)
            ):
                return None
            pair_index = product_frame["sensing"]["pair_axis"][0]["frame_pair_index"]
            poses = []
            for name in ("tx_positions", "tx_orientations", "rx_positions", "rx_orientations"):
                values = np.asarray(replay_frame.get(name), dtype=np.float64)
                if values.ndim != 2 or values.shape[1:] != (3,):
                    return None
                row = values[pair_index]
                if not bool(np.all(np.isfinite(row))):
                    return None
                # Canonical JSON preserves signed zero, unlike float equality.
                poses.append(tuple(float(value).hex() for value in row))
        except (IndexError, KeyError, TypeError, ValueError):
            return None
        return (
            product_frame.cache_token,
            context,
            (type(frame_index), frame_index),
            tuple(poses),
            bound_time.hex(),
            current_time.hex(),
        )

    def analytical_path_source(
        self, scene_frame: dict[str, Any], step: int
    ) -> tuple[dict[str, Any] | None, tuple]:
        """Find explicit analytical acquisition evidence for a scene's RT source.

        The complete result metadata is validated by the shared index. No
        scenario-name, empty-path, frame-stride, or YAML-only inference is used.
        Warm lookups use a small source map, not a scan of scene frames.
        """
        provider = visualizer_frame_provider(self.visualizer)
        info = getattr(provider, "info", None)
        token = (
            id(provider),
            getattr(info, "generation_id", None),
            getattr(info, "frame_set_id", None),
            getattr(self.visualizer, "_mpc_presented_source_epoch", None),
        )
        source = (scene_frame.get("_source") or {}).get("source_rt_frame_idx")
        if isinstance(source, bool) or not isinstance(source, int) or source > step:
            return None, token
        if self.prepared_scene is not None and self.prepared_scene.matches_provider(provider):
            record = self.prepared_scene.analytical_sources.get(source)
            if record is not None and not source <= step < record["stop_scene_id"]:
                record = None
            return record, token
        if token != self._analytical_source_token:
            self._analytical_sources.clear()
            storage_ids = self.candidate_storage_frame_ids(step)
            expected = self._refresh_metadata_index_token(storage_ids)
            try:
                rows = self._indexed_sensing_results(storage_ids, expected)
                for _, metadata in rows:
                    record = (metadata.get("sensing") or {}).get("_analytical_path_source")
                    if record is not None:
                        self._analytical_sources[record["source_scene_id"]] = record
                self._require_metadata_index_token(storage_ids, expected)
            except _SensingInventoryChanged:
                self._analytical_sources.clear()
                return None, token
            self._analytical_source_token = token
        record = self._analytical_sources.get(source)
        if record is not None and not source <= step < record["stop_scene_id"]:
            record = None
        return record, token

    def candidate_storage_frame_ids(
        self,
        current_storage_frame_id: int | str,
    ) -> tuple[int | str, ...]:
        """Return deterministic storage candidates without assigning them time semantics."""
        provider = visualizer_frame_provider(self.visualizer)
        source = getattr(self.visualizer, "frame_source", None)
        snapshot = getattr(provider, "frame_ids_snapshot", None)
        # An immutable provider advertises the complete recording inventory.
        # Cached payloads are not additional rows of that recording. Keep the
        # general union below for live sources and independent source mappings.
        from ..io.frame_sources import FileSource

        if snapshot is not None and (
            source is provider
            or not callable(getattr(source, "list_frames", None))
            or (isinstance(source, FileSource) and source.provider is provider)
        ):
            with self._sensing_cache_lock:
                if (
                    snapshot is not self._storage_snapshot
                    or provider is not self._storage_snapshot_provider
                ):
                    self._storage_snapshot = snapshot
                    self._storage_snapshot_provider = provider
                    self._storage_candidates = tuple(
                        sorted(snapshot, key=lambda value: f"{type(value).__name__}:{value!r}")
                    )
                # Navigation probes may use 0 even when this recording starts
                # later. An absent ID must not change the authoritative inventory
                # and invalidate its index on every subsequent real-frame query.
                return self._storage_candidates
        candidates: dict[tuple[type[Any], int | str], int | str] = {}

        def remember(value: Any) -> None:
            if isinstance(value, (int, str)) and not isinstance(value, bool):
                candidates[(type(value), value)] = value

        remember(current_storage_frame_id)
        cache = getattr(self.visualizer, "cache_service", None)
        frame_cache = getattr(cache, "frame_cache", None)
        cache_keys = getattr(frame_cache, "keys", None)
        if callable(cache_keys):
            try:
                for value in cache_keys():
                    remember(value)
            except (OSError, TypeError, ValueError, RuntimeError):
                pass

        for owner in (
            visualizer_frame_provider(self.visualizer),
            getattr(self.visualizer, "frame_source", None),
        ):
            list_frames = getattr(owner, "list_frames", None)
            if not callable(list_frames):
                continue
            try:
                for value in list_frames():
                    remember(value)
            except (OSError, TypeError, ValueError, RuntimeError):
                continue

        return tuple(
            sorted(
                candidates.values(),
                key=lambda value: f"{type(value).__name__}:{value!r}",
            )
        )

    def dashboard_result_rows(self) -> tuple[tuple[int | str, dict[str, Any]], ...]:
        """Return recorded result occurrences for a background dashboard reader."""
        ids = self.candidate_storage_frame_ids(0)
        token = self._refresh_metadata_index_token(ids)
        return self._indexed_sensing_results(ids, token)

    def cached_dashboard_result_rows(self) -> tuple[tuple[int | str, dict[str, Any]], ...] | None:
        """Borrow the ready immutable index without loading or refreshing it.

        Cursor-only consumers must fall back to the worker when the provider
        generation or frame inventory changes. The token checks those identities
        without enumerating the recording again.
        """
        with self._sensing_cache_lock:
            token = self._sensing_index_token
        if token is None or self._current_metadata_index_token(token[-1]) != token:
            return None
        with self._sensing_cache_lock:
            return self._sensing_result_index if self._sensing_index_token == token else None

    def _exact_scene_result_rows(
        self,
        rows: tuple[tuple[int | str, dict[str, Any]], ...],
        scene_id: int | str,
    ) -> tuple[tuple[int | str, dict[str, Any]], ...]:
        """Group an owned result index once, preserving every exact candidate.

        The existing inventory token retires these references with the source
        index. Grouping does not select a winner: product eligibility, duplicate
        bindings and run/pair tie-breaking remain the caller's responsibility.
        Integer and string scene IDs retain their ordinary equality semantics.
        """
        with self._sensing_cache_lock:
            if self._scene_result_index_rows is not rows:
                grouped: dict[int | str, list[tuple[int | str, dict[str, Any]]]] = {}
                for storage_id, frame in rows:
                    bound = frame["sensing"]["temporal"]["display_binding"].get("scene_frame_id")
                    if isinstance(bound, (int, str)) and not isinstance(bound, bool):
                        grouped.setdefault(bound, []).append((storage_id, frame))
                self._scene_result_index = {
                    bound: tuple(candidates) for bound, candidates in grouped.items()
                }
                self._scene_result_index_rows = rows
            return self._scene_result_index.get(scene_id, ())

    def cached_dashboard_result_metadata(
        self, scene_id: int | str, context: tuple[str, str, str]
    ) -> dict[str, Any] | None:
        """Borrow one already indexed result's metadata without reading anything.

        Missing, stale or ambiguous bindings return no evidence. The derived
        lookup never loads metadata or populates the underlying result index.
        """
        with self._sensing_cache_lock:
            indexed_token = self._sensing_index_token
        if indexed_token is None:
            return None
        # Reuse the indexed inventory tuple; the token also checks its provider
        # generation and immutable frame-snapshot identity without enumerating it.
        current_token = self._current_metadata_index_token(indexed_token[-1])
        with self._sensing_cache_lock:
            if current_token != self._sensing_index_token:
                return None
            rows = self._sensing_result_index
            if rows is None:
                self._dashboard_scene_metadata_rows = None
                self._dashboard_scene_metadata.clear()
                return None
            if self._dashboard_scene_metadata_rows is not rows:
                indexed = {}
                for _storage_id, frame in rows:
                    sensing = frame["sensing"]
                    temporal = sensing["temporal"]
                    key = (
                        temporal["display_binding"]["scene_frame_id"],
                        temporal["clock"]["clock_id"],
                        temporal["result"]["processing_run_id"],
                        temporal["acquisition"]["pair_id"],
                    )
                    indexed[key] = None if key in indexed else sensing
                self._dashboard_scene_metadata = indexed
                self._dashboard_scene_metadata_rows = rows
            return self._dashboard_scene_metadata.get((scene_id, *context))

    def sensing_playback_frame_ids(self) -> tuple[int, ...]:
        """Return saved scene bindings, including results with zero detections.

        Storage rows and RT solve cadence are not scene addresses. Use the
        explicit binding and intersect with the recording inventory instead.
        This metadata-only query is suitable for a background reader.
        """
        ids = self.candidate_storage_frame_ids(0)
        token = self._refresh_metadata_index_token(ids)
        provider = visualizer_frame_provider(self.visualizer)
        snapshot = getattr(provider, "frame_ids_snapshot", None)
        available = set(
            snapshot if snapshot is not None else provider.list_frames() if provider else ids
        )
        bound = {
            frame["sensing"]["temporal"]["display_binding"]["scene_frame_id"]
            for _, frame in self._indexed_sensing_results(ids, token)
        }
        self._require_metadata_index_token(ids, token)
        return tuple(
            sorted(
                step
                for step in bound
                if isinstance(step, int) and not isinstance(step, bool) and step in available
            )
        )

    @staticmethod
    def _eligible_dashboard_result(frame, scene_id, scene_time_s, *, hold):
        """Apply result binding/time eligibility without changing the shared index."""
        temporal = (frame.get("sensing") or {}).get("temporal") or {}
        result = temporal.get("result") or {}
        binding = temporal.get("display_binding") or {}
        bound = binding.get("scene_frame_id")
        epoch = result.get("reference_time_s")
        if (
            not isinstance(bound, (int, str))
            or isinstance(bound, bool)
            or not isinstance(epoch, (int, float))
            or isinstance(epoch, bool)
            or not math.isfinite(epoch)
        ):
            return False
        return epoch <= scene_time_s if hold else bound == scene_id

    def dashboard_exact_result_id(
        self,
        scene_id: int | str,
        *,
        pin_freshest: bool = False,
        scene_time_s: float | None = None,
        product_names: tuple[str, ...] = ("range_profile", "range_doppler"),
    ) -> int | str | None:
        """Select a cached cursor by the ordinary exact-binding/tie-break rule.

        Never perform cold I/O here. Changed inventories fall back to the
        background reader. Inspect all runs/pairs before choosing; the caller
        may reuse history only if it contains that selected storage row.
        During playback, also require the physical-time winner to have this
        exact scene binding. A held result needs an ordinary replay request.
        Product eligibility matches the requesting reader; the default retains
        the range/Doppler requirement used by history cursors.
        """
        selected = self.dashboard_result_cache_hint(
            scene_id,
            pin_freshest=pin_freshest,
            scene_time_s=scene_time_s,
            product_names=product_names,
        )
        return selected[0] if selected is not None and selected[1] == scene_id else None

    def dashboard_result_cache_hint(
        self,
        scene_id: int | str,
        *,
        pin_freshest: bool = False,
        scene_time_s: float | None = None,
        product_names: tuple[str, ...] = ("range_profile", "range_doppler"),
    ) -> tuple[int | str, int | str] | None:
        """Locate a ready-index winner's storage and original scene binding.

        This performs no cold I/O and permits a causal held result when requested.
        A prepared product is only a hint: callers must resolve the ordinary
        replay and compare its occurrence identity before attaching cached arrays.
        """
        if pin_freshest and (scene_time_s is None or not math.isfinite(scene_time_s)):
            return None
        token = self._current_metadata_index_token(self.candidate_storage_frame_ids(0))
        with self._sensing_cache_lock:
            if token != self._sensing_index_token or self._sensing_result_index is None:
                return None
            rows = self._sensing_result_index
        if not pin_freshest:
            rows = self._exact_scene_result_rows(rows, scene_id)
        eligible = [
            (storage_id, frame)
            for storage_id, frame in rows
            if self._eligible_dashboard_result(frame, scene_id, scene_time_s, hold=pin_freshest)
            and self._declares_any_sensing_product(frame, product_names)
        ]
        if not eligible:
            return None
        storage_id, selected = max(eligible, key=lambda row: self._selection_key(row[1], row[0]))
        binding = selected["sensing"]["temporal"]["display_binding"]
        return storage_id, binding["scene_frame_id"]

    def dashboard_load_row(
        self, storage_id: int | str, products: tuple[str, ...]
    ) -> dict[str, Any] | None:
        """Select only named products and their authored coordinate arrays."""
        return self._load_sensing_candidate(storage_id, products)

    def dashboard_history_products(
        self, storage_id: int | str, variant: str
    ) -> tuple[dict[str, Any] | None, Any, Any]:
        """Read and validate a history row once before extracting its two plots.

        The loader validates the complete selective sensing contract. Neither
        extraction mutates it, so repeating that validation for each plot would
        recheck the same acquisition sample IDs and product hashes twice more.
        This guarantee is local to this call; mutable caller frames are never
        marked as permanently validated or trusted across later edits.
        """
        kinds = self.dashboard_product_kinds(storage_id)
        names = tuple(
            kind for kind in ("range_doppler", "range_profile", "micro_doppler") if kind in kinds
        )
        frame = self.dashboard_load_row(storage_id, names or ("config_resolved",))
        if frame is None:
            return None, None, None
        rp, rd = self._dashboard_products_from_validated(frame["sensing"], variant)
        return frame, rp, rd

    def dashboard_spectral_products(
        self, storage_id: int | str, variant: str, families: Iterable[str]
    ) -> tuple[dict[str, Any] | None, Any, Any]:
        """Read only the authored variants needed by visible overview families."""
        from ..sensing.figure_requirements import product_data_names, spectral_requirements

        descriptors = self.dashboard_product_descriptors(storage_id)
        requirements = spectral_requirements(families, descriptors, variant)
        names = ("config_resolved", *product_data_names(requirements, descriptors))
        frame = self.dashboard_load_row(storage_id, names)
        if frame is None:
            return None, None, None
        rp, rd = self._dashboard_products_from_validated(frame["sensing"], variant)
        return frame, rp, rd

    def dashboard_products(self, frame: dict[str, Any], variant: str) -> tuple[Any, Any]:
        """Validate a caller-owned frame once for both saved dashboard products."""
        _assert_sensing_frame(frame)
        return self._dashboard_products_from_validated(frame["sensing"], variant)

    def _dashboard_products_from_validated(
        self, sensing: dict[str, Any], preference: str
    ) -> tuple[Any, Any]:
        """Extract saved products immediately after their owning call validates."""
        products = []
        for variants, extract in (
            (range_profile_product_variants(sensing), self._range_profile_from_validated),
            (range_doppler_product_variants(sensing), self._range_doppler_from_validated),
        ):
            requested = "primary" if preference == "clean" else preference
            if requested == "auto":
                requested = next((v for v in ("primary", "raw") if v in variants), None)
            products.append(extract(sensing, requested) if requested in variants else None)
        return products[0], products[1]

    def dashboard_scene_frame(self, scene_id: int | str) -> dict[str, Any] | None:
        """Read device pose and physical time without geometry or path arrays."""
        return self._load_scene_pose_frame(scene_id)

    def dashboard_product_kinds(self, storage_id: int | str) -> frozenset[str]:
        """Report capabilities from descriptors, without loading their arrays."""
        products = self.dashboard_product_descriptors(storage_id)
        return frozenset(d["kind"] for d in products.values() if d.get("kind"))

    def dashboard_product_descriptors(self, storage_id: int | str) -> dict[str, Any]:
        """Return detached descriptor metadata without loading numeric products."""
        metadata = self._load_sensing_metadata_candidate(storage_id) or {}
        products = (metadata.get("sensing") or {}).get("products") or {}
        return deepcopy(products)

    def dashboard_inventory_token(self) -> tuple | None:
        """Return the indexed inventory identity, including live source growth."""
        with self._sensing_cache_lock:
            return self._sensing_index_token

    @classmethod
    def dashboard_world_detections(cls, frame: dict[str, Any]) -> list[dict[str, Any]]:
        """Project saved detection/track states using the same pose as 3D overlays."""
        pose = cls._resolve_radar_pose(frame)
        if pose is None:
            return []
        origin, rotation = pose
        sensing = frame.get("sensing") or {}
        result = []
        for det in cls._prepare_detections(
            sensing.get("detections") or [], range_mode=cls._overlay_range_mode(sensing)
        ):
            item = dict(det)
            for key, track in (("world_position", False), ("world_track_position", True)):
                local = cls._local_cartesian_position(det, prefer_track_state=track)
                if local is not None:
                    item[key] = cls._sensor_to_world(origin, rotation, *local)
            result.append(item)
        return result

    def _load_sensing_candidate(
        self,
        storage_frame_id: int | str,
        product_names: tuple[str, ...],
        *,
        load_all_sensing_products: bool = False,
        storage_ids: tuple[int | str, ...] = (),
        expected_index_token: tuple[Any, ...] | None = None,
        own_snapshot: bool = False,
    ) -> dict[str, Any] | None:
        """Load requested products, or the complete selected sensing payload.

        Indexed selections detach provider data before validating the owned
        snapshot. Validate once at that boundary, rather than checking the
        provider projection and then the identical copy a second time.
        Caller-owned cache frames still follow their normal validation path.
        """
        if expected_index_token is not None:
            self._require_metadata_index_token(storage_ids, expected_index_token)
        needs_descriptor_load = not load_all_sensing_products and _needs_descriptor_bound_load(
            product_names
        )
        descriptor_probe: dict[str, Any] | None = None
        with self._sensing_cache_lock:
            indexed = self._sensing_metadata_index.get((type(storage_frame_id), storage_frame_id))
        indexed_sensing = indexed.get("sensing") if isinstance(indexed, Mapping) else None
        if isinstance(indexed_sensing, Mapping):
            descriptor_probe = dict(indexed_sensing)
        cache = getattr(self.visualizer, "cache_service", None)
        if cache is not None:
            cached = cache.get_frame(storage_frame_id)
            if cached is not None:
                _assert_sensing_frame(cached)
                if expected_index_token is not None:
                    self._require_metadata_index_token(storage_ids, expected_index_token)
                cached_has_requested = self._has_all_sensing_products(cached, product_names)
                if cached_has_requested and (
                    not load_all_sensing_products or self._has_complete_sensing_payload(cached)
                ):
                    return cached
                sensing = cached.get("sensing")
                if isinstance(sensing, Mapping):
                    descriptor_probe = dict(sensing)

        provider = visualizer_frame_provider(self.visualizer)
        if provider is not None:
            if needs_descriptor_load and descriptor_probe is None:
                try:
                    projection = provider.load_frame_projection(
                        storage_frame_id,
                        _DESCRIPTOR_PROBE_REQUEST,
                    )
                except UnsupportedSensingSchemaError:
                    raise
                except (OSError, KeyError, TypeError, RuntimeError):
                    projection = None
                if projection is not None:
                    probe_frame = {"sensing": dict(projection.frame.sensing or {})}
                    _assert_sensing_frame(probe_frame)
                    if expected_index_token is not None:
                        self._require_metadata_index_token(
                            storage_ids,
                            expected_index_token,
                        )
                    descriptor_probe = probe_frame["sensing"]
            request = (
                FrameReadRequest(components=frozenset({FrameComponent.SENSING}))
                if load_all_sensing_products
                else _descriptor_bound_read_request(
                    descriptor_probe or {},
                    product_names,
                )
            )
            try:
                projection = provider.load_frame_projection(storage_frame_id, request)
            except UnsupportedSensingSchemaError:
                raise
            except (OSError, KeyError, TypeError, RuntimeError):
                projection = None
            if projection is not None:
                projected = {"sensing": dict(projection.frame.sensing or {})}
                if own_snapshot:
                    projected = _ValidatedSensingProductFrame(
                        projected, require_complete=load_all_sensing_products
                    )
                else:
                    _assert_sensing_frame(projected)
                if expected_index_token is not None:
                    self._require_metadata_index_token(storage_ids, expected_index_token)
                return projected

        frame_source = getattr(self.visualizer, "frame_source", None)
        if frame_source is not None and hasattr(frame_source, "load_frame"):
            try:
                frame = frame_source.load_frame(storage_frame_id)
            except UnsupportedSensingSchemaError:
                raise
            except (OSError, KeyError, RuntimeError):
                return None
            if not isinstance(frame, StandardMPCFrame):
                raise TypeError("frame sources must return StandardMPCFrame")
            if expected_index_token is not None:
                self._require_metadata_index_token(storage_ids, expected_index_token)
            if needs_descriptor_load:
                probe_projection = project_standard_mpc_frame(
                    frame,
                    _DESCRIPTOR_PROBE_REQUEST,
                )
                probe_frame = {"sensing": dict(probe_projection.frame.sensing or {})}
                _assert_sensing_frame(probe_frame)
                descriptor_probe = probe_frame["sensing"]
            request = (
                FrameReadRequest(components=frozenset({FrameComponent.SENSING}))
                if load_all_sensing_products
                else _descriptor_bound_read_request(
                    descriptor_probe or {},
                    product_names,
                )
            )
            projection = project_standard_mpc_frame(frame, request)
            projected = {"sensing": dict(projection.frame.sensing or {})}
            if own_snapshot:
                projected = _ValidatedSensingProductFrame(
                    projected, require_complete=load_all_sensing_products
                )
            else:
                _assert_sensing_frame(projected)
            if expected_index_token is not None:
                self._require_metadata_index_token(storage_ids, expected_index_token)
            return projected
        return None

    def resolve_display_sensing_frame(
        self,
        scene_frame: dict[str, Any],
        *,
        scene_frame_id: int | str,
        storage_frame_id: int | str,
        product_names: Iterable[str],
        load_all_sensing_products: bool = False,
        pin_freshest: bool = False,
        candidate_storage_ids: Iterable[int | str] | None = None,
        load_candidate: Callable[[int | str], dict[str, Any] | None] | None = None,
        load_scene_frame: Callable[[int | str], dict[str, Any] | None] | None = None,
        product_selector: Callable[[Mapping[str, Any]], tuple[str, ...]] | None = None,
    ) -> SensingDisplaySelection | None:
        """Resolve products by explicit display binding, never by storage proximity.

        Normal automatic display accepts only results whose bound scene ID equals
        ``scene_frame_id``. ``pin_freshest`` is an explicit view override: it may
        retain a result bound to another scene, and the returned provenance marks
        that state as stale. Candidate iteration order never affects selection.
        ``load_all_sensing_products`` retains the metadata-only scan but materializes
        every stored product for the one selected acquisition.
        ``product_selector`` resolves exact descriptor keys after that winner
        is known, so view/variant demand does not change the scene binding rule.
        """
        names = tuple(str(name) for name in product_names)
        storage_ids = tuple(
            candidate_storage_ids
            if candidate_storage_ids is not None
            else self.candidate_storage_frame_ids(storage_frame_id)
        )
        if load_candidate is not None:
            return self._resolve_display_sensing_frame_once(
                scene_frame,
                scene_frame_id=scene_frame_id,
                storage_frame_id=storage_frame_id,
                names=names,
                load_all_sensing_products=load_all_sensing_products,
                pin_freshest=pin_freshest,
                storage_ids=storage_ids,
                load_candidate=load_candidate,
                load_scene_frame=load_scene_frame,
                expected_index_token=None,
                product_selector=product_selector,
            )

        for _attempt in range(_SENSING_RESOLVE_MAX_ATTEMPTS):
            expected_index_token = self._refresh_metadata_index_token(storage_ids)
            try:
                selection = self._resolve_display_sensing_frame_once(
                    scene_frame,
                    scene_frame_id=scene_frame_id,
                    storage_frame_id=storage_frame_id,
                    names=names,
                    load_all_sensing_products=load_all_sensing_products,
                    pin_freshest=pin_freshest,
                    storage_ids=storage_ids,
                    load_candidate=None,
                    load_scene_frame=load_scene_frame,
                    expected_index_token=expected_index_token,
                    product_selector=product_selector,
                )
                self._require_metadata_index_token(storage_ids, expected_index_token)
            except _SensingInventoryChanged:
                continue
            return selection

        logger.warning(
            "Sensing replay selection aborted after %d provider generation changes",
            _SENSING_RESOLVE_MAX_ATTEMPTS,
        )
        return None

    def _resolve_display_sensing_frame_once(
        self,
        scene_frame: dict[str, Any],
        *,
        scene_frame_id: int | str,
        storage_frame_id: int | str,
        names: tuple[str, ...],
        load_all_sensing_products: bool,
        pin_freshest: bool,
        storage_ids: tuple[int | str, ...],
        load_candidate: Callable[[int | str], dict[str, Any] | None] | None,
        load_scene_frame: Callable[[int | str], dict[str, Any] | None] | None,
        expected_index_token: tuple[Any, ...] | None,
        product_selector: Callable[[Mapping[str, Any]], tuple[str, ...]] | None = None,
    ) -> SensingDisplaySelection | None:
        """Resolve one attempt against one pinned provider generation."""
        use_metadata_index = expected_index_token is not None
        loader = (
            load_candidate
            if load_candidate is not None
            else lambda candidate_id: self._load_sensing_metadata_candidate(
                candidate_id,
                storage_ids=storage_ids,
                expected_index_token=expected_index_token,
            )
        )

        candidates: list[tuple[dict[str, Any], int | str, int | str]] = []
        seen: set[tuple[type[Any], int | str]] = set()

        def consider(frame: dict[str, Any] | None, candidate_id: int | str) -> bool:
            identity = (type(candidate_id), candidate_id)
            if identity in seen:
                return True
            if frame is None:
                return False
            binding = self._display_binding_record(frame)
            has_requested_product = (
                self._declares_any_sensing_product(frame, names)
                if use_metadata_index
                else self._has_any_sensing_product(frame, names)
            )
            if binding is None or not has_requested_product:
                return False
            bound_scene_id = binding.get("scene_frame_id")
            if not isinstance(bound_scene_id, (int, str)) or isinstance(bound_scene_id, bool):
                return False
            seen.add(identity)
            candidates.append((frame, candidate_id, bound_scene_id))
            return True

        consider(scene_frame, storage_frame_id)
        indexed_rows = (
            self._indexed_sensing_results(storage_ids, expected_index_token)
            if expected_index_token is not None
            else ((candidate_id, loader(candidate_id)) for candidate_id in storage_ids)
        )
        if expected_index_token is not None and not pin_freshest:
            indexed_rows = self._exact_scene_result_rows(indexed_rows, scene_frame_id)
        for candidate_id, candidate_metadata in indexed_rows:
            identity = (type(candidate_id), candidate_id)
            if identity in seen:
                continue
            consider(candidate_metadata, candidate_id)
        if expected_index_token is not None:
            self._require_metadata_index_token(storage_ids, expected_index_token)

        if pin_freshest:
            # ``pin_freshest`` is a hold-last display policy, not permission to
            # leak a future result into an earlier scene.  Physical scene time
            # is the only valid ordering coordinate.  Without it, pinning
            # degrades to the same exact-binding rule as automatic display.
            current_scene_time_s = self._scene_time_s(scene_frame)
            if current_scene_time_s is None:
                eligible = [candidate for candidate in candidates if candidate[2] == scene_frame_id]
            else:
                eligible = [
                    candidate
                    for candidate in candidates
                    if float(
                        ((candidate[0].get("sensing") or {}).get("temporal") or {})
                        .get("result", {})
                        .get("reference_time_s", float("inf"))
                    )
                    <= current_scene_time_s
                ]
        else:
            eligible = [candidate for candidate in candidates if candidate[2] == scene_frame_id]
        if not eligible:
            return None
        product_frame, selected_storage_id, bound_scene_id = max(
            eligible,
            key=lambda candidate: self._selection_key(candidate[0], candidate[1]),
        )
        if product_selector is not None:
            names = tuple(product_selector(product_frame.get("sensing") or {}))
        has_products = (
            self._has_all_sensing_products
            if product_selector is not None
            else self._has_any_sensing_product
        )
        selected_payload_incomplete = not has_products(product_frame, names) or (
            load_all_sensing_products and not self._has_complete_sensing_payload(product_frame)
        )
        if use_metadata_index and selected_payload_incomplete:
            assert expected_index_token is not None
            product_cache_key = (
                type(selected_storage_id),
                selected_storage_id,
                names,
                load_all_sensing_products,
            )
            self._require_metadata_index_token(storage_ids, expected_index_token)
            loaded_product_frame = self._selected_product_cache_get(
                product_cache_key,
                index_token=expected_index_token,
            )
            product_cache_miss = loaded_product_frame is None
            if loaded_product_frame is None:
                self._replay_telemetry.record("selected_dense_loads")
                loaded_product_frame = self._load_sensing_candidate(
                    selected_storage_id,
                    names,
                    load_all_sensing_products=load_all_sensing_products,
                    storage_ids=storage_ids,
                    expected_index_token=expected_index_token,
                    own_snapshot=True,
                )
            else:
                self._replay_telemetry.record("selected_dense_cache_hits")
                self._require_metadata_index_token(storage_ids, expected_index_token)
            if loaded_product_frame is None:
                return None
            if product_cache_miss and not isinstance(
                loaded_product_frame, _ValidatedSensingProductFrame
            ):
                loaded_product_frame = _ValidatedSensingProductFrame(
                    loaded_product_frame, require_complete=load_all_sensing_products
                )
            loaded_binding = self._display_binding_record(loaded_product_frame)
            if (
                loaded_binding is None
                or loaded_binding.get("scene_frame_id") != bound_scene_id
                or not has_products(loaded_product_frame, names)
                or self._selection_key(loaded_product_frame, selected_storage_id)
                != self._selection_key(product_frame, selected_storage_id)
            ):
                raise ValueError(
                    "Indexed sensing metadata disagrees with the selected product projection"
                )
            if product_cache_miss:
                assert isinstance(loaded_product_frame, _ValidatedSensingProductFrame)
                self._selected_product_cache_put(
                    product_cache_key,
                    loaded_product_frame,
                    index_token=expected_index_token,
                )
            product_frame = loaded_product_frame
        sensing = product_frame.get("sensing") or {}
        temporal = sensing.get("temporal") or {}
        result = temporal.get("result") or {}
        reference_time_s = float(result["reference_time_s"])
        bound_scene_frame: Mapping[str, Any] | None = None
        if bound_scene_id != scene_frame_id:
            scene_loader = load_scene_frame or self._load_scene_pose_frame
            bound_scene_frame = scene_loader(bound_scene_id)
            if bound_scene_frame is None:
                return None
        display_frame = self._compose_bound_display_frame(
            scene_frame,
            product_frame,
            bound_scene_frame=bound_scene_frame,
            storage_frame_id=selected_storage_id,
            bound_scene_frame_id=bound_scene_id,
            current_scene_frame_id=scene_frame_id,
            pinned=pin_freshest,
        )
        provenance = display_frame.get("_sensing_display_provenance") or {}
        selected_bound_scene_frame = bound_scene_frame or scene_frame
        replay_context = dict(
            storage_frame_id=selected_storage_id,
            bound_scene_frame_id=bound_scene_id,
            current_scene_frame_id=scene_frame_id,
            bound_scene_frame=selected_bound_scene_frame,
            current_scene_frame=scene_frame,
            pinned=bool(pin_freshest),
        )
        replay_cache_key = self._validated_replay_key(
            product_frame, display_frame, **replay_context
        )
        replay_identity = None
        if expected_index_token is not None and replay_cache_key is not None:
            with self._sensing_cache_lock:
                self._require_installed_metadata_index_token(expected_index_token)
                replay_identity = self._validated_replay_cache.get(replay_cache_key)
                if replay_identity is not None:
                    self._validated_replay_cache.move_to_end(replay_cache_key)
                else:
                    for previous_key, previous in reversed(self._validated_replay_cache.items()):
                        if (
                            previous_key[0] is not replay_cache_key[0]
                            or previous_key[2:5] != replay_cache_key[2:5]
                        ):
                            continue
                        try:
                            rebound = rebind_replay_event(previous.event, replay_cache_key[1])
                        except ValueError:
                            # A changed bound-scene time requires a full build.
                            continue
                        replay_identity = _ValidatedReplayIdentity(
                            rebound, previous.manifest_id, previous.product_token
                        )
                        self._replay_telemetry.record("replay_rebindings")
                        break
                    if replay_identity is not None:
                        self._validated_replay_cache[replay_cache_key] = replay_identity
                        while len(self._validated_replay_cache) > _VALIDATED_REPLAY_CACHE_CAPACITY:
                            self._validated_replay_cache.popitem(last=False)
        if replay_identity is not None:
            self._replay_telemetry.record("replay_cache_hits")
            replay_event, replay_reason = replay_identity.event, None
        else:
            replay_event, replay_reason = self._selection_replay_event(
                display_frame,
                **replay_context,
                reference_time_s=reference_time_s,
                scene_result_time_offset_s=provenance.get("scene_result_time_offset_s"),
                stale_age_s=provenance.get("stale_age_s"),
            )
            if replay_event is not None:
                replay_identity = _ValidatedReplayIdentity(
                    replay_event,
                    replay_event.manifest_id,
                    (
                        product_frame.cache_token
                        if isinstance(product_frame, _ValidatedSensingProductFrame)
                        else None
                    ),
                )
                if expected_index_token is not None and replay_cache_key is not None:
                    with self._sensing_cache_lock:
                        self._require_installed_metadata_index_token(expected_index_token)
                        self._validated_replay_cache[replay_cache_key] = replay_identity
                        self._validated_replay_cache.move_to_end(replay_cache_key)
                        while len(self._validated_replay_cache) > _VALIDATED_REPLAY_CACHE_CAPACITY:
                            self._validated_replay_cache.popitem(last=False)
        replay_manifest_id = replay_identity.manifest_id if replay_identity is not None else None
        provenance.update(
            {
                "replay_disposition": (
                    replay_event.preflight.disposition if replay_event is not None else None
                ),
                "replay_event_id": replay_manifest_id,
                "replay_event_identity": (
                    {
                        **replay_event.key.to_manifest(),
                        "manifest_id": replay_manifest_id,
                        "scene_binding": replay_event.scene_binding.to_manifest(),
                    }
                    if replay_event is not None
                    else None
                ),
                "replay_reason": replay_reason,
            }
        )
        display_frame["_sensing_display_provenance"] = provenance
        if isinstance(product_frame, _ValidatedSensingProductFrame):
            product_frame = _copy_sensing_snapshot(product_frame)
            display_frame["sensing"] = dict(product_frame.get("sensing") or {})
        return SensingDisplaySelection(
            display_frame=display_frame,
            product_frame=product_frame,
            storage_frame_id=selected_storage_id,
            bound_scene_frame_id=bound_scene_id,
            current_scene_frame_id=scene_frame_id,
            reference_time_s=reference_time_s,
            current_scene_time_s=self._scene_time_s(scene_frame),
            scene_result_time_offset_s=(display_frame.get("_sensing_display_provenance") or {}).get(
                "scene_result_time_offset_s"
            ),
            stale_age_s=(display_frame.get("_sensing_display_provenance") or {}).get("stale_age_s"),
            pinned=bool(pin_freshest),
            replay_event=replay_event,
            replay_reason=replay_reason,
            _inventory_token=expected_index_token,
            _inventory_storage_ids=storage_ids if expected_index_token is not None else (),
            _replay_identity=replay_identity,
        )

    @staticmethod
    def _detection_stage_read_request(
        sensing: Mapping[str, Any],
    ) -> tuple[FrameReadRequest, frozenset[str]]:
        """Return an exact request for every descriptor-bound Phase-4 stage."""
        products = sensing.get("products")
        requested = {"detections", "products"}
        required: set[str] = set()
        if isinstance(products, Mapping):
            for data_key, descriptor in products.items():
                if (
                    not isinstance(data_key, str)
                    or not isinstance(descriptor, Mapping)
                    or descriptor.get("kind") not in DETECTION_STAGE_KINDS
                ):
                    continue
                requested.add(data_key)
                required.add(data_key)
                axes = descriptor.get("axes")
                if not isinstance(axes, list):
                    continue
                for axis in axes:
                    if not isinstance(axis, Mapping):
                        continue
                    coordinate_key = axis.get("coordinate_key")
                    if isinstance(coordinate_key, str):
                        requested.add(coordinate_key)
                        required.add(coordinate_key)
                    alternate_keys = axis.get("alternate_coordinate_keys")
                    if isinstance(alternate_keys, list):
                        for alternate_key in alternate_keys:
                            if isinstance(alternate_key, str):
                                requested.add(alternate_key)
                                required.add(alternate_key)
        return (
            FrameReadRequest(
                components=frozenset({FrameComponent.SENSING}),
                sensing_products=frozenset(requested),
            ),
            frozenset(required),
        )

    @staticmethod
    def _explicit_stage_bundle(
        disposition: InspectionDisposition,
        reason: str,
        *,
        products: tuple[DetectionStageProduct, ...] = (),
    ) -> DetectionStageBundle:
        """Return an explicit fail-closed detector-stage model."""
        return DetectionStageBundle(
            disposition=disposition,
            reason=reason,
            event=None,
            products=products,
        )

    def _load_detection_stage_projection(
        self,
        storage_frame_id: int | str,
        request: FrameReadRequest,
        required_keys: frozenset[str],
    ) -> dict[str, Any] | None:
        """Load detector stages for one selected storage event without scanning."""
        cache = getattr(self.visualizer, "cache_service", None)
        if cache is not None:
            cached = cache.get_frame(storage_frame_id)
            cached_sensing = cached.get("sensing") if isinstance(cached, Mapping) else None
            if isinstance(cached_sensing, Mapping) and required_keys.issubset(cached_sensing):
                projected = {"sensing": dict(cached_sensing)}
                _assert_sensing_frame(projected)
                return projected

        provider = visualizer_frame_provider(self.visualizer)
        if provider is not None:
            try:
                projection = provider.load_frame_projection(storage_frame_id, request)
            except UnsupportedSensingSchemaError:
                raise
            except (OSError, KeyError, RuntimeError):
                projection = None
            if projection is not None:
                projected = {"sensing": dict(projection.frame.sensing or {})}
                _assert_sensing_frame(projected)
                return projected

        frame_source = getattr(self.visualizer, "frame_source", None)
        if frame_source is not None and hasattr(frame_source, "load_frame"):
            try:
                frame = frame_source.load_frame(storage_frame_id)
            except UnsupportedSensingSchemaError:
                raise
            except (OSError, KeyError, RuntimeError):
                return None
            if not isinstance(frame, StandardMPCFrame):
                raise TypeError("frame sources must return StandardMPCFrame")
            projection = project_standard_mpc_frame(frame, request)
            projected = {"sensing": dict(projection.frame.sensing or {})}
            _assert_sensing_frame(projected)
            return projected
        return None

    @staticmethod
    def _cross_check_stage_bundle(
        bundle: DetectionStageBundle,
        replay_event: ReplayEvent,
    ) -> DetectionStageBundle:
        """Reject a detector-stage bundle bound to a different replay event."""
        if bundle.disposition is InspectionDisposition.INVALID or (
            bundle.disposition is InspectionDisposition.UNAVAILABLE and bundle.event is None
        ):
            return bundle
        stage_event = bundle.event
        if stage_event is None:
            return SensingService._explicit_stage_bundle(
                InspectionDisposition.INVALID,
                "detector-stage bundle has no event identity",
                products=bundle.products,
            )
        expected = replay_event.key
        identity_matches = (
            stage_event.clock_id == expected.clock_id
            and stage_event.acquisition_id == expected.acquisition_id
            and stage_event.processing_run_id == expected.processing_run_id
            and stage_event.processing_recipe_id == replay_event.recipe.processing_recipe_id
            and stage_event.pair_id == expected.pair_id
            and math.isclose(
                stage_event.reference_time_s,
                expected.reference_time_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        replay_product_ids = {product.product_id for product in replay_event.products}
        stage_product_ids = {
            product.product_id for product in bundle.products if product.product_id is not None
        }
        if not identity_matches or not stage_product_ids.issubset(replay_product_ids):
            return SensingService._explicit_stage_bundle(
                InspectionDisposition.INVALID,
                "detector-stage bundle event identity does not match ReplayEvent",
                products=bundle.products,
            )
        return bundle

    def load_detection_stage_bundle(
        self,
        selection: SensingDisplaySelection,
    ) -> DetectionStageBundle:
        """Load all persisted detector stages for exactly one selected replay event.

        Descriptor discovery uses the already-selected product metadata.  At
        most one selective projection is issued for its storage row, and the
        immutable result is cached by canonical replay manifest identity.
        """
        if not isinstance(selection, SensingDisplaySelection):
            raise TypeError("selection must be a SensingDisplaySelection")
        replay_event = selection.replay_event
        if replay_event is None:
            return self._explicit_stage_bundle(
                InspectionDisposition.INVALID,
                selection.replay_reason or "canonical_replay_event_unavailable",
            )
        binding = replay_event.scene_binding
        if any(
            type(actual) is not type(expected) or actual != expected
            for actual, expected in (
                (selection.storage_frame_id, binding.storage_frame_id),
                (selection.bound_scene_frame_id, binding.bound_scene_frame_id),
                (selection.current_scene_frame_id, binding.current_scene_frame_id),
                (selection.pinned, binding.pinned),
            )
        ) or (
            selection.reference_time_s != replay_event.key.reference_time_s
            or selection.current_scene_time_s != binding.current_scene_time_s
            or selection.scene_result_time_offset_s != binding.current_scene_result_offset_s
            or selection.stale_age_s != (binding.stale_age_s if binding.stale else None)
        ):
            return self._explicit_stage_bundle(
                InspectionDisposition.INVALID,
                "selection_binding_does_not_match_replay_event",
            )
        index_token = selection._inventory_token
        storage_ids = selection._inventory_storage_ids
        if index_token is not None:
            try:
                self._require_metadata_index_token(storage_ids, index_token)
            except _SensingInventoryChanged:
                return self._explicit_stage_bundle(
                    InspectionDisposition.UNAVAILABLE,
                    "sensing_inventory_changed_since_selection",
                )
        replay_identity = selection._replay_identity
        cache_key = (
            replay_identity.manifest_id
            if replay_identity is not None and replay_identity.event is replay_event
            else replay_event.manifest_id
        )
        trusted_snapshot = (
            index_token is not None
            and replay_identity is not None
            and replay_identity.event is replay_event
            and replay_identity.product_token is not None
        )
        try:
            cached = (
                self._detection_stage_cache_get(cache_key, index_token=index_token)
                if trusted_snapshot
                else None
            )
        except _SensingInventoryChanged:
            return self._explicit_stage_bundle(
                InspectionDisposition.UNAVAILABLE,
                "sensing_inventory_changed_since_selection",
            )
        if cached is not None:
            if index_token is not None:
                try:
                    self._require_metadata_index_token(storage_ids, index_token)
                except _SensingInventoryChanged:
                    return self._explicit_stage_bundle(
                        InspectionDisposition.UNAVAILABLE,
                        "sensing_inventory_changed_since_selection",
                    )
            self._replay_telemetry.record("stage_cache_hits")
            return cached
        self._replay_telemetry.record("stage_loads")

        sensing = selection.product_frame.get("sensing")
        snapshot_missing = False
        if trusted_snapshot:
            # Public selections deliberately expose detached, editable metadata.
            # Never let edits there change a bundle cached for immutable science.
            with self._sensing_cache_lock:
                snapshot = next(
                    (
                        frame
                        for frame in self._sensing_selected_product_cache.values()
                        if isinstance(frame, _ValidatedSensingProductFrame)
                        and frame.cache_token is replay_identity.product_token
                    ),
                    None,
                )
            snapshot_missing = snapshot is None
            sensing = (
                snapshot.get("sensing")
                if snapshot is not None
                else {
                    "products": {
                        item.data_key: item.to_manifest() for item in replay_event.products
                    }
                }
            )
        if not isinstance(sensing, Mapping):
            bundle = self._explicit_stage_bundle(
                InspectionDisposition.INVALID,
                "selected product frame has no sensing mapping",
            )
        else:
            request, required_keys = self._detection_stage_read_request(sensing)
            if not required_keys and not snapshot_missing:
                bundle = inspect_detection_stage_bundle(sensing)
            elif required_keys.issubset(sensing) and not snapshot_missing:
                bundle = inspect_detection_stage_bundle(sensing)
            else:
                try:
                    projected = self._load_detection_stage_projection(
                        selection.storage_frame_id,
                        request,
                        required_keys,
                    )
                except (UnsupportedSensingSchemaError, TypeError, ValueError) as exc:
                    bundle = self._explicit_stage_bundle(
                        InspectionDisposition.INVALID,
                        str(exc) or type(exc).__name__,
                    )
                else:
                    if projected is None:
                        bundle = self._explicit_stage_bundle(
                            InspectionDisposition.UNAVAILABLE,
                            "detector_stage_projection_unavailable",
                        )
                    else:
                        projected_sensing = projected.get("sensing")
                        bundle = (
                            inspect_detection_stage_bundle(projected_sensing)
                            if isinstance(projected_sensing, Mapping)
                            else self._explicit_stage_bundle(
                                InspectionDisposition.INVALID,
                                "detector-stage projection has no sensing mapping",
                            )
                        )
        if index_token is not None:
            try:
                self._require_metadata_index_token(storage_ids, index_token)
            except _SensingInventoryChanged:
                return self._explicit_stage_bundle(
                    InspectionDisposition.UNAVAILABLE,
                    "sensing_inventory_changed_since_selection",
                )
        bundle = self._cross_check_stage_bundle(bundle, replay_event)
        if trusted_snapshot and bundle.disposition is InspectionDisposition.AVAILABLE:
            self._detection_stage_cache_put(
                cache_key,
                bundle,
                index_token=index_token,
            )
        return bundle

    # CIR helpers
    def compute_cir(
        self,
        raw_frame: dict[str, Any],
        *,
        sampling_frequency: float,
        normalize_delays: bool = False,
        store_in_history: bool = True,
    ) -> Optional[CIRComputationResult]:
        """Compute or load CIR samples and optionally append them to history."""
        _assert_sensing_frame(raw_frame)
        result = self.cir_provider.compute(
            raw_frame, sampling_frequency=sampling_frequency, normalize_delays=normalize_delays
        )
        if result is None:
            logger.debug("SensingService: CIR provider returned no result")
            return None
        if store_in_history:
            sample_time_s = result.sample_time_s
            if sample_time_s is None or not math.isfinite(float(sample_time_s)):
                logger.debug("CIR history skipped because physical sample time is unavailable")
            elif self.cir_history:
                previous_time_s = self.cir_history[-1].sample_time_s
                if previous_time_s is None or sample_time_s < previous_time_s:
                    self.cir_history.clear()
                    self.cir_history.append(result)
                elif sample_time_s == previous_time_s:
                    # Re-render/replay of the same physical sample replaces, not duplicates.
                    self.cir_history[-1] = result
                else:
                    self.cir_history.append(result)
            else:
                self.cir_history.append(result)
        return result

    # Range profile
    def compute_range_profile(
        self,
        raw_frame: dict[str, Any],
        *,
        sampling_frequency: float,
        fft_size: int = 256,
        window: str | None = "hann",
        store_in_history: bool = False,
        variant: str = "auto",
    ) -> Optional[RangeProfileResult]:
        """Return one exact typed range-profile variant and persisted coordinates."""
        _assert_sensing_frame(raw_frame)
        return self._range_profile_from_validated(raw_frame.get("sensing") or {}, variant)

    def _range_profile_from_validated(
        self, sensing_block: dict[str, Any], variant: str
    ) -> RangeProfileResult:
        """Extract the profile without mutating its just-validated payload."""
        try:
            explicit_variant = _explicit_product_variant(
                variant,
                range_profile_product_variants(sensing_block),
            )
            data_key, profile, coordinates, descriptor, pair = _require_single_pair_product(
                sensing_block,
                contract=_RANGE_PROFILE_CONTRACT,
                variant=explicit_variant,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Typed range-profile product unavailable for requested variant {variant!r}: "
                f"{exc}"
            ) from exc
        if np.iscomplexobj(profile) or not np.all(np.isfinite(profile)) or np.any(profile < 0):
            raise ValueError("Typed range-profile power_linear values must be real and nonnegative")
        range_axis = coordinates[0]
        range_mode = self._range_mode_from_descriptor(descriptor)
        # Native recordings keep response calibration on the sensing block;
        # replay products may carry more specific processing annotations.
        response_metadata = dict(sensing_block.get("range_response") or {})
        response_metadata.update(descriptor.get("processing") or {})
        return RangeProfileResult(
            range_axis_m=range_axis,
            power_linear=profile,
            fft_spectrum=None,
            range_axis_label=self._range_label_for_mode(range_mode),
            metadata={
                "variant": descriptor.get("variant"),
                "source": "stored_typed_product",
                "data_key": data_key,
                "product_id": descriptor.get("product_id"),
                "representation": descriptor.get("representation"),
                "pair_id": pair.get("pair_id"),
                "range_mode": range_mode,
                "temporal_support": dict(descriptor.get("temporal_support") or {}),
                "time_span": dict(descriptor.get("time_span") or {}),
                "range_sample_spacing_m": response_metadata.get("range_sample_spacing_m"),
                "range_resolution_scale_m": response_metadata.get("range_resolution_scale_m"),
                "range_response_model": response_metadata.get("model"),
                "frequency_window": response_metadata.get("frequency_window"),
            },
        )

    # Range–Doppler
    def compute_range_doppler(
        self,
        *,
        frame: Optional[dict[str, Any]] = None,
        sampling_frequency: float,
        prf_hz: float,
        fft_size_range: int = 256,
        fft_size_doppler: int = 64,
        window_range: str | None = "hann",
        window_doppler: str | None = "hann",
        min_history: Optional[int] = None,
        variant: str = "auto",
    ) -> Optional[RangeDopplerResult]:
        """Return one exact typed range-Doppler variant and persisted axes."""
        if frame is None:
            raise RuntimeError("Schema-v4 range-Doppler visualization requires a sensing frame")
        _assert_sensing_frame(frame)
        return self._range_doppler_from_validated(frame.get("sensing") or {}, variant)

    def _range_doppler_from_validated(
        self, sensing_block: dict[str, Any], variant: str
    ) -> RangeDopplerResult:
        """Extract RD power without mutating its just-validated payload."""
        try:
            explicit_variant = _explicit_product_variant(
                variant,
                range_doppler_product_variants(sensing_block),
            )
            data_key, rd_values, coordinates, descriptor, pair = _require_single_pair_product(
                sensing_block,
                contract=_RANGE_DOPPLER_CONTRACT,
                variant=explicit_variant,
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Typed range-Doppler product unavailable for requested variant {variant!r}: "
                f"{exc}"
            ) from exc

        representation = descriptor.get("representation")
        if representation == "complex_voltage":
            if not np.iscomplexobj(rd_values) or not np.all(np.isfinite(rd_values)):
                raise ValueError(
                    "Typed range-Doppler complex_voltage values must be finite and complex"
                )
            cube: np.ndarray | None = rd_values
            power_linear = np.abs(rd_values) ** 2
        elif representation == "power_linear":
            if (
                np.iscomplexobj(rd_values)
                or not np.all(np.isfinite(rd_values))
                or np.any(rd_values < 0)
            ):
                raise ValueError(
                    "Typed range-Doppler power_linear values must be finite, real, and "
                    "nonnegative"
                )
            cube = None
            power_linear = rd_values
        else:  # Defensive: contract selection restricts representations.
            raise ValueError(f"Unsupported typed range-Doppler representation {representation!r}")

        doppler_axis, range_axis = coordinates
        range_mode = self._range_mode_from_descriptor(descriptor)
        response_metadata = dict(sensing_block.get("range_response") or {})
        response_metadata.update(descriptor.get("processing") or {})
        cfg = sensing_block.get("config_resolved") or {}
        recipe = cfg.get("processing_recipe") or {}
        result = RangeDopplerResult(
            range_axis_m=range_axis,
            doppler_axis_hz=doppler_axis,
            power_linear=power_linear,
            cube=cube,
            range_axis_label=self._range_label_for_mode(range_mode),
            doppler_axis_label="Doppler (Hz)",
            metadata={
                "variant": descriptor.get("variant"),
                "source": "stored_typed_product",
                "data_key": data_key,
                "product_id": descriptor.get("product_id"),
                "representation": descriptor.get("representation"),
                "pair_id": pair.get("pair_id"),
                "range_mode": range_mode,
                "range_response_model": response_metadata.get("model"),
                "frequency_window": response_metadata.get("frequency_window"),
                "doppler_taper": response_metadata.get(
                    "doppler_taper", cfg.get("window_doppler", recipe.get("window_doppler"))
                ),
            },
        )
        apply_detection_velocity_metadata(result.metadata, range_mode=range_mode)
        return result

    # Helpers
    @staticmethod
    def _range_mode_from_descriptor(descriptor: Mapping[str, Any]) -> str:
        """Resolve range labeling only from the bound range-axis semantics."""
        axes = descriptor.get("axes")
        if not isinstance(axes, list):
            raise ValueError("Typed sensing product has no ordered axes")
        range_axes = [
            axis for axis in axes if isinstance(axis, Mapping) and axis.get("name") == "range"
        ]
        if len(range_axes) != 1:
            raise ValueError("Typed sensing product requires exactly one range axis")
        semantics = range_axes[0].get("coordinate_semantics")
        if semantics == "monostatic_range":
            return "monostatic"
        if semantics == "bistatic_path_length":
            return "bistatic"
        raise ValueError(
            "Typed sensing range axis requires coordinate_semantics "
            "'monostatic_range' or 'bistatic_path_length'"
        )

    @staticmethod
    def _range_label_for_mode(mode: str) -> str:
        """Return the dashboard range-axis label for a range-mode value."""
        return "Range (m)" if mode == "monostatic" else "Path Length (m)"

    # 3D detection overlay

    DETECTION_GEOMETRY_NAME = "sensing_detections"
    EVIDENCE_GEOMETRY_NAME = "sensing_evidence"
    TRACK_POINT_PREFIX = "sensing_track_point_"
    TRACK_TRAIL_PREFIX = "sensing_track_trail_"
    TRACK_ARROW_PREFIX = "sensing_track_arrow_"
    _RING_SAMPLES = 64
    _MIN_ARROW_SPEED_M_S = 0.05
    _ARROW_LOOKAHEAD_S = 0.75

    @staticmethod
    def _replay_layer_metadata(raw_frame: Mapping[str, Any]) -> dict[str, Any]:
        """Copy canonical selected-event identity into renderer metadata."""
        provenance = raw_frame.get("_sensing_display_provenance")
        if not isinstance(provenance, Mapping):
            return {}
        identity = provenance.get("replay_event_identity")
        if identity is None:
            return {}
        if not isinstance(identity, Mapping):
            raise ValueError("Sensing display replay_event_identity must be a mapping")
        scene_binding = identity.get("scene_binding")
        required = (
            "manifest_id",
            "clock_id",
            "acquisition_id",
            "processing_run_id",
            "pair_id",
            "reference_time_s",
        )
        if any(key not in identity for key in required) or not isinstance(scene_binding, Mapping):
            raise ValueError("Sensing display replay_event_identity is incomplete")
        if provenance.get("replay_event_id") != identity.get("manifest_id"):
            raise ValueError("Sensing display replay identity contradicts replay_event_id")
        return {
            "replay_event_identity": {
                **{key: identity[key] for key in required},
                "scene_binding": dict(scene_binding),
            }
        }

    def sync_scene_overlays(
        self,
        renderer: RendererProtocol,
        layers: list[SensingSceneLayer],
    ) -> bool:
        """Synchronize complete sensing-layer snapshots without losing failures."""
        return self._scene_overlay_sync.sync(
            renderer,
            (layer.to_render_object() for layer in layers),
        )

    def clear_scene_overlays(self, renderer: RendererProtocol) -> bool:
        """Remove every sensing overlay, retaining any failed removal for retry."""
        return self._scene_overlay_sync.clear(renderer)

    def resolve_scene_overlay_layers(
        self,
        scene_frame: dict[str, Any],
        *,
        step: int,
        mode: str,
        history_frames: int,
        use_full_history: bool,
        show_velocity_arrows: bool,
        pin_freshest: bool,
    ) -> tuple[dict, list[SensingSceneLayer]] | None:
        """Resolve an overlay without mixing generations during a live refresh."""
        if self.prepared_scene is not None:
            prepared = self.prepared_scene.resolve(
                self,
                scene_frame,
                step=step,
                mode=mode,
                history_frames=history_frames,
                use_full_history=use_full_history,
                show_velocity_arrows=show_velocity_arrows,
                pin_freshest=pin_freshest,
            )
            if prepared is not NotImplemented:
                return prepared
        for _attempt in range(_SENSING_RESOLVE_MAX_ATTEMPTS):
            try:
                return self._resolve_scene_overlay_layers_once(
                    scene_frame,
                    step=step,
                    mode=mode,
                    history_frames=history_frames,
                    use_full_history=use_full_history,
                    show_velocity_arrows=show_velocity_arrows,
                    pin_freshest=pin_freshest,
                )
            except _SensingInventoryChanged:
                continue
        logger.warning("Sensing overlays deferred because the provider generation kept changing")
        return None

    def _resolve_scene_overlay_layers_once(
        self,
        scene_frame: dict[str, Any],
        *,
        step: int,
        mode: str,
        history_frames: int,
        use_full_history: bool,
        show_velocity_arrows: bool,
        pin_freshest: bool,
    ) -> tuple[dict, list[SensingSceneLayer]] | None:
        """Prepare bounded overlay snapshots once per scene and display choice.

        Only provider-generation-bound views are retained. Mutable, cache-only
        callers continue through validation on every call. Cached values contain
        annotation detections and immutable render payloads, never channel or
        angular arrays. Device poses and scene time participate in the key so a
        moved receiver or a changed hold-last selection cannot reuse stale geometry.
        """
        storage_ids = self.candidate_storage_frame_ids(step)
        token = self._refresh_metadata_index_token(storage_ids)
        cacheable = visualizer_frame_provider(self.visualizer) is not None
        pose_key = tuple(
            (name, np.asarray(scene_frame[name]).shape, np.asarray(scene_frame[name]).tobytes())
            for name in ("rx_positions", "rx_orientations", "tx_positions", "tx_orientations")
            if scene_frame.get(name) is not None
        )
        key = (
            type(step),
            step,
            self._scene_time_s(scene_frame),
            pose_key,
            mode,
            history_frames,
            use_full_history,
            show_velocity_arrows,
            pin_freshest,
        )
        if cacheable:
            with self._sensing_cache_lock:
                if key in self._current_detection_cache:
                    self._current_detection_cache.move_to_end(key)
                    return self._current_detection_cache[key]
                if key in self._scene_overlay_cache:
                    self._scene_overlay_cache.move_to_end(key)
                    return self._scene_overlay_cache[key]
        selection = self.resolve_display_sensing_frame(
            scene_frame,
            scene_frame_id=step,
            storage_frame_id=step,
            product_names=("detections",),
            # Trails/history remain meaningful between sensing epochs. Resolve
            # them at the latest prior result, retaining its bound sensor pose
            # and stale provenance; never advance or extrapolate the estimate.
            pin_freshest=pin_freshest or mode in {"tracks", "evidence", "full"},
            candidate_storage_ids=storage_ids,
        )
        prepared = None
        if selection is not None:
            frame = selection.display_frame
            # Selection already validates the loaded result, including on a
            # selected-product cache hit. Avoid revalidating every descriptor
            # and acquisition sample again just to project the same detections.
            layers = self._build_scene_overlays_from_validated(
                frame,
                step=step,
                mode=mode,
                history_frames=history_frames,
                use_full_history=use_full_history,
                show_velocity_arrows=show_velocity_arrows,
            )
            annotations = {
                "sensing": {
                    "detections": deepcopy((frame.get("sensing") or {}).get("detections") or []),
                    "temporal": self._annotation_temporal(frame["sensing"]),
                },
                "stale": selection.stale_for_current_scene,
                "result_time_s": selection.reference_time_s,
                "scene_time_s": selection.current_scene_time_s,
            }
            prepared = (annotations, layers)
        if cacheable:
            self._require_metadata_index_token(storage_ids, token)
            with self._sensing_cache_lock:
                if self._sensing_index_token is token:
                    if mode == "detections" and prepared is not None:
                        self._current_detection_cache[key] = prepared
                        self._current_detection_cache.move_to_end(key)
                        while (
                            len(self._current_detection_cache) > _CURRENT_DETECTION_CACHE_CAPACITY
                        ):
                            self._current_detection_cache.popitem(last=False)
                        return prepared
                    # Empty intermediate scene rows must not evict prepared
                    # detections. They share the smaller general view cache.
                    self._scene_overlay_cache[key] = prepared
                    self._scene_overlay_cache.move_to_end(key)
                    while len(self._scene_overlay_cache) > _SCENE_OVERLAY_CACHE_CAPACITY:
                        self._scene_overlay_cache.popitem(last=False)
        return prepared

    @staticmethod
    def _annotation_temporal(sensing: dict) -> dict:
        """Detach only the scalar identities used by marker picking/highlighting."""
        temporal = sensing["temporal"]
        return {
            "clock": {"clock_id": temporal["clock"]["clock_id"]},
            "acquisition": {
                field: temporal["acquisition"][field] for field in ("acquisition_id", "pair_id")
            },
            "result": {"processing_run_id": temporal["result"]["processing_run_id"]},
        }

    def build_scene_overlays(
        self,
        raw_frame: dict[str, Any],
        *,
        step: int,
        mode: str = "detections",
        history_frames: int = 12,
        use_full_history: bool = False,
        show_velocity_arrows: bool = False,
    ) -> list[SensingSceneLayer]:
        """Build renderer-ready sensing overlays for the current frame."""
        _assert_sensing_frame(raw_frame)
        return self._build_scene_overlays_from_validated(
            raw_frame,
            step=step,
            mode=mode,
            history_frames=history_frames,
            use_full_history=use_full_history,
            show_velocity_arrows=show_velocity_arrows,
        )

    def _build_scene_overlays_from_validated(
        self,
        raw_frame: dict[str, Any],
        *,
        step: int,
        mode: str,
        history_frames: int,
        use_full_history: bool,
        show_velocity_arrows: bool,
        history: list[tuple[int | str, dict[str, Any]]] | None = None,
    ) -> list[SensingSceneLayer]:
        """Project a result just validated by the public builder or resolver."""
        sensing = raw_frame.get("sensing") or {}
        detections = sensing.get("detections") or []

        radar_pose = self._resolve_radar_pose(raw_frame)
        if radar_pose is None:
            return []
        radar_xyz, rotation_matrix = radar_pose

        range_mode = self._overlay_range_mode(sensing)

        prepared = self._prepare_detections(detections, range_mode=range_mode)
        normalized_mode = str(mode or "detections").strip().lower()
        if normalized_mode not in {"detections", "tracks", "evidence", "full"}:
            normalized_mode = "detections"

        has_xy = any(self._has_local_cartesian(det) for det in prepared)
        has_history_geometry = has_xy or any(
            self._local_cartesian_position(det, prefer_track_state=True) is not None
            for det in prepared
        )
        layers: list[SensingSceneLayer] = []

        current_layer = (
            raw_frame.get("_sensing_scene_current")
            if raw_frame.get("_sensing_scene_prepared")
            else self._build_current_detection_layer(
                prepared, radar_xyz, rotation_matrix, has_xy=has_xy, range_mode=range_mode
            )
        )
        if current_layer is not None and normalized_mode != "tracks":
            layers.append(current_layer)

        history_len = max(1, int(history_frames))
        history = (
            history
            if history is not None
            else (
                self._history_frames(
                    raw_frame,
                    step=step,
                    history_frames=history_len,
                    use_full_history=bool(use_full_history),
                )
                if normalized_mode != "detections" and (has_history_geometry or not detections)
                else []
            )
        )
        if normalized_mode in {"tracks", "full"}:
            layers.extend(
                self._build_track_layers(
                    raw_frame,
                    history=history,
                    current_detections=prepared,
                    show_velocity_arrows=bool(show_velocity_arrows),
                )
            )

        if normalized_mode in {"evidence", "full"} and has_xy:
            evidence_layer = self._build_evidence_layer(
                raw_frame,
                history=history,
                current_detections=prepared,
            )
            if evidence_layer is not None:
                layers.append(evidence_layer)

        layers.extend(
            raw_frame["_sensing_scene_truth"]
            if raw_frame.get("_sensing_scene_prepared")
            else self._build_truth_layers(sensing)
        )
        replay_metadata = self._replay_layer_metadata(raw_frame)
        if replay_metadata:
            layers = [
                SensingSceneLayer(
                    name=layer.name,
                    payload=layer.payload,
                    material=layer.material,
                    metadata={**(layer.metadata or {}), **replay_metadata},
                )
                for layer in layers
            ]
        return layers

    @staticmethod
    def _build_truth_layers(sensing: dict) -> list[SensingSceneLayer]:
        """Show authored world positions only at their matching result epoch."""
        temporal = sensing.get("temporal") or {}
        if (temporal.get("truth") or {}).get("reference_time_s") != (
            temporal.get("result") or {}
        ).get("reference_time_s"):
            return []
        layers = []
        for target in (sensing.get("truth") or {}).get("targets", []):
            if not target.get("valid") or target.get("lifecycle") != "active":
                continue
            position = np.asarray(target.get("position_m"), dtype=np.float32)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                continue
            tid = target["target_id"]
            layers.append(
                SensingSceneLayer(
                    name=f"sensing_truth_{tid}",
                    payload=PointCloudPayload(points=position[None, :]),
                    material=MaterialPayload(
                        base_color=color_rgba(identity_color(tid, truth=True)), point_size=16.0
                    ),
                    metadata={"target_id": tid, "sensing_role": "truth"},
                )
            )
        return layers

    @staticmethod
    def _bound_frame_pair_index(raw_frame: Mapping[str, Any]) -> int:
        """Return the one typed pair's exact source-frame pose row."""
        sensing = raw_frame.get("sensing")
        pair_axis = sensing.get("pair_axis") if isinstance(sensing, Mapping) else None
        if not isinstance(pair_axis, list) or len(pair_axis) != 1:
            raise ValueError("Sensing overlays require exactly one typed pair_axis row")
        pair = pair_axis[0]
        if not isinstance(pair, Mapping) or pair.get("pair_index") != 0:
            raise ValueError("Sensing overlays require typed pair_axis pair_index 0")
        frame_pair_index = pair.get("frame_pair_index")
        if (
            not isinstance(frame_pair_index, int)
            or isinstance(frame_pair_index, bool)
            or frame_pair_index < 0
        ):
            raise ValueError("Sensing pair_axis requires a nonnegative frame_pair_index")

        temporal = sensing.get("temporal") if isinstance(sensing, Mapping) else None
        acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
        if not isinstance(acquisition, Mapping):
            raise ValueError("Sensing pair_axis has no temporal acquisition identity")
        for field in ("pair_id", "tx_sensor_id", "rx_sensor_id"):
            if pair.get(field) != acquisition.get(field):
                raise ValueError(
                    f"Sensing pair_axis {field} disagrees with temporal acquisition identity"
                )
        return frame_pair_index

    @classmethod
    def _resolve_radar_pose(cls, raw_frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the typed sensing pair's complete RX world pose.

        Frame orientations use Sionna's intrinsic yaw/pitch/roll radians.  A
        full rotation matrix is retained so pitch and roll affect points,
        trails, and velocity vectors exactly like yaw.
        """
        frame_pair_index = cls._bound_frame_pair_index(raw_frame)
        rx_pos = raw_frame.get("rx_positions")
        if rx_pos is None:
            raise ValueError("Sensing overlay pair pose requires rx_positions")
        rx_arr = np.asarray(rx_pos, dtype=np.float64)
        if rx_arr.ndim != 2 or rx_arr.shape[1:] != (3,):
            raise ValueError("Sensing overlay rx_positions must have shape [frame_pair, 3]")
        if frame_pair_index >= rx_arr.shape[0]:
            raise ValueError("Sensing pair_axis frame_pair_index is outside the rx_positions array")
        radar_xyz = rx_arr[frame_pair_index]
        if not np.all(np.isfinite(radar_xyz)):
            raise ValueError("Sensing overlay RX position must contain finite coordinates")

        rx_ori = raw_frame.get("rx_orientations")
        if rx_ori is None:
            raise ValueError("Sensing overlay pair pose requires rx_orientations")
        ori = np.asarray(rx_ori, dtype=np.float64)
        if ori.ndim != 2 or ori.shape[1:] != (3,):
            raise ValueError("Sensing overlay rx_orientations must have shape [frame_pair, 3]")
        if frame_pair_index >= ori.shape[0]:
            raise ValueError(
                "Sensing pair_axis frame_pair_index is outside the rx_orientations array"
            )
        orientation = ori[frame_pair_index]
        if not np.all(np.isfinite(orientation)):
            raise ValueError("Sensing overlay RX orientation must contain finite coordinates")
        rotation_matrix = yaw_pitch_roll_to_rotation_matrix(*orientation)
        return radar_xyz, rotation_matrix

    @staticmethod
    def _overlay_range_mode(sensing: Mapping[str, Any]) -> str:
        """Return the exact range semantics governing 3D projection."""
        config = sensing.get("config_resolved")
        range_mode = config.get("range_mode") if isinstance(config, Mapping) else None
        if range_mode not in {"monostatic", "bistatic"}:
            raise ValueError("Sensing overlays require config_resolved.range_mode")
        return str(range_mode)

    @staticmethod
    def _prepare_detections(
        detections: list[dict[str, Any]],
        *,
        range_mode: str = "monostatic",
    ) -> list[dict[str, Any]]:
        """Normalize monostatic Cartesian fields when angle data is available.

        A bistatic path length plus one receiver-local angle is not a radial
        distance from the receiver.  Synthesizing Cartesian coordinates with
        the monostatic formula would therefore be scientifically wrong; only
        explicitly persisted Cartesian estimates are admitted in that mode.
        """
        prepared: list[dict[str, Any]] = []
        for det in detections:
            item = dict(det)
            if range_mode == "monostatic" and not np.isfinite(item.get("x_m", float("nan"))):
                az = item.get("aoa_az_deg")
                r = item.get("range_m")
                if az is not None and r is not None and np.isfinite(az) and np.isfinite(r):
                    az_rad = np.radians(float(az))
                    el = item.get("aoa_el_deg")
                    if el is not None and np.isfinite(el):
                        el_rad = np.radians(float(el))
                        horiz = float(r) * np.cos(el_rad)
                        item["x_m"] = horiz * np.cos(az_rad)
                        item["y_m"] = horiz * np.sin(az_rad)
                        item["z_m"] = float(r) * np.sin(el_rad)
                    else:
                        item["x_m"] = float(r) * np.cos(az_rad)
                        item["y_m"] = float(r) * np.sin(az_rad)
            prepared.append(item)
        return prepared

    @staticmethod
    def _has_local_cartesian(det: dict[str, Any]) -> bool:
        """Return whether a detection has finite local X/Y coordinates."""
        return np.isfinite(det.get("x_m", float("nan"))) and np.isfinite(
            det.get("y_m", float("nan"))
        )

    @staticmethod
    def _local_cartesian_position(
        det: dict[str, Any], *, prefer_track_state: bool = False
    ) -> tuple[float, float, float] | None:
        """Return local x/y/z coordinates for raw detections or tracker state."""
        x_key = "x_m"
        y_key = "y_m"
        z_key = "z_m"
        if (
            prefer_track_state
            and np.isfinite(det.get("track_x_m", float("nan")))
            and np.isfinite(det.get("track_y_m", float("nan")))
        ):
            x_key = "track_x_m"
            y_key = "track_y_m"
            z_key = "track_z_m"

        x = det.get(x_key, float("nan"))
        y = det.get(y_key, float("nan"))
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        z = det.get(z_key, det.get("z_m", 0.0))
        if z is None or not np.isfinite(z):
            z = 0.0
        return float(x), float(y), float(z)

    @staticmethod
    def _sensor_to_world(
        radar_xyz: np.ndarray,
        rotation_matrix: np.ndarray,
        lx: float,
        ly: float,
        lz: float,
    ) -> np.ndarray:
        """Rotate a local sensor-space point into visualizer world coordinates."""
        local = np.asarray([lx, ly, lz], dtype=np.float64)
        world = (
            np.asarray(radar_xyz, dtype=np.float64)
            + np.asarray(rotation_matrix, dtype=np.float64) @ local
        )
        return world.astype(np.float32, copy=False)

    def _build_current_detection_layer(
        self,
        detections: list[dict[str, Any]],
        radar_xyz: np.ndarray,
        rotation_matrix: np.ndarray,
        *,
        has_xy: bool,
        range_mode: str,
    ) -> SensingSceneLayer | None:
        """Build the current-frame detection layer as points or range rings."""
        if not detections:
            return None
        if has_xy:
            overlay = self._detection_points(detections, radar_xyz, rotation_matrix)
            if overlay is None:
                return None
            positions, colors, _kind = overlay
            return SensingSceneLayer(
                name=self.DETECTION_GEOMETRY_NAME,
                payload=PointCloudPayload(points=positions, colors=colors),
                material=MaterialPayload(
                    base_color=(1.0, 1.0, 1.0, 1.0),
                    point_size=10.0,
                ),
                metadata={
                    "detection_indices": tuple(
                        i for i, d in enumerate(detections) if self._has_local_cartesian(d)
                    )
                },
            )
        if range_mode != "monostatic":
            raise ValueError(
                "Bistatic range-only detections have no unique point or circular-ring overlay"
            )
        overlay = self._detection_rings(detections, radar_xyz)
        if overlay is None:
            return None
        positions, colors, _kind = overlay
        ring_size = self._RING_SAMPLES
        lines = []
        for ring_start in range(0, len(positions), ring_size):
            for j in range(ring_size):
                a = ring_start + j
                b = ring_start + (j + 1) % ring_size
                if b < len(positions):
                    lines.append([a, b])
        payload = LineSetPayload(
            points=positions,
            lines=np.asarray(lines, dtype=np.int32) if lines else np.empty((0, 2), dtype=np.int32),
            colors=colors,
        )
        return SensingSceneLayer(
            name=self.DETECTION_GEOMETRY_NAME,
            payload=payload,
            material=MaterialPayload(
                base_color=(1.0, 1.0, 1.0, 0.95),
                line_width=2.25,
            ),
            metadata={
                "detection_indices": tuple(
                    i
                    for i, d in enumerate(detections)
                    if d.get("range_m") is not None
                    and np.isfinite(d["range_m"])
                    and d["range_m"] > 0
                ),
                # The backend expands each indexed ring edge to two vertices.
                "pick_vertex_span": 2 * self._RING_SAMPLES,
            },
        )

    @staticmethod
    def _result_reference_time_s(frame: dict[str, Any]) -> float | None:
        """Return the validated physical result epoch for one sensing frame."""
        binding = SensingService._display_binding_record(frame)
        if binding is None:
            return None
        sensing = frame.get("sensing") or {}
        temporal = sensing.get("temporal") or {}
        result = temporal.get("result") or {}
        return float(result["reference_time_s"])

    def _load_scene_pose_frame(self, scene_frame_id: int | str) -> dict[str, Any] | None:
        """Load device pose for an explicitly named scene without consuming sensing."""
        cache = getattr(self.visualizer, "cache_service", None)
        if cache is not None:
            cached = cache.get_frame(scene_frame_id)
            if cached is not None and (
                cached.get("rx_positions") is not None or cached.get("tx_positions") is not None
            ):
                return cached

        request = FrameReadRequest(
            components=frozenset({FrameComponent.DEVICES, FrameComponent.PROVENANCE})
        )
        provider = visualizer_frame_provider(self.visualizer)
        if provider is not None:
            try:
                projection = provider.load_frame_projection(scene_frame_id, request)
            except (OSError, KeyError, TypeError, ValueError, RuntimeError):
                projection = None
            if projection is not None:
                frame = projection.frame
                return {
                    "tx_positions": np.asarray(frame.tx_positions),
                    "rx_positions": np.asarray(frame.rx_positions),
                    "tx_orientations": np.asarray(frame.tx_orientations),
                    "rx_orientations": np.asarray(frame.rx_orientations),
                    "timestamp_s": frame.timestamp_s,
                }

        frame_source = getattr(self.visualizer, "frame_source", None)
        if frame_source is not None and hasattr(frame_source, "load_frame"):
            try:
                frame = frame_source.load_frame(scene_frame_id)
            except (OSError, KeyError, TypeError, ValueError, RuntimeError):
                return None
            if not isinstance(frame, StandardMPCFrame):
                raise TypeError("frame sources must return StandardMPCFrame")
            return {
                "tx_positions": np.asarray(frame.tx_positions),
                "rx_positions": np.asarray(frame.rx_positions),
                "tx_orientations": np.asarray(frame.tx_orientations),
                "rx_orientations": np.asarray(frame.rx_orientations),
                "timestamp_s": frame.timestamp_s,
            }
        return None

    def _history_frames(
        self,
        current_frame: dict[str, Any],
        *,
        step: int,
        history_frames: int,
        use_full_history: bool = False,
    ) -> list[tuple[int | str, dict[str, Any]]]:
        """Collect bound detection history ordered by physical result time."""
        _assert_sensing_frame(current_frame)
        temporal = (current_frame.get("sensing") or {}).get("temporal") or {}
        current_reference = (temporal.get("result") or {}).get("reference_time_s")
        if current_reference is None:
            return []
        provenance = current_frame.get("_sensing_display_provenance") or {}
        current_storage_id = provenance.get("storage_frame_id", step)

        records: dict[
            tuple[str, str], tuple[float, tuple[float, str, str, str], int | str, dict]
        ] = {}

        def add_candidate(product_frame: dict[str, Any], storage_id: int | str) -> None:
            # The current frame and _load_history_frame results are validated
            # at their read boundaries, before entering this private builder.
            sensing = product_frame.get("sensing") or {}
            if not sensing.get("detections"):
                return
            temporal = sensing.get("temporal") or {}
            result = temporal.get("result") or {}
            acquisition = temporal.get("acquisition") or {}
            reference_time = float(result["reference_time_s"])
            if reference_time > current_reference:
                return
            binding = temporal.get("display_binding") or {}
            bound_scene_id = binding.get("scene_frame_id")
            if not isinstance(bound_scene_id, (int, str)) or isinstance(bound_scene_id, bool):
                return

            if product_frame is current_frame:
                bound_frame = current_frame
            else:
                scene_pose = self._load_scene_pose_frame(bound_scene_id)
                if scene_pose is None:
                    return
                bound_frame = self._compose_bound_display_frame(
                    scene_pose,
                    product_frame,
                    storage_frame_id=storage_id,
                    bound_scene_frame_id=bound_scene_id,
                    current_scene_frame_id=bound_scene_id,
                    pinned=False,
                )
            result_identity = (
                str(result.get("processing_run_id", "")),
                str(acquisition.get("acquisition_id", "")),
            )
            record = (
                reference_time,
                self._selection_key(product_frame, storage_id),
                bound_scene_id,
                bound_frame,
            )
            previous = records.get(result_identity)
            if previous is None or record[1] > previous[1]:
                records[result_identity] = record
            if product_frame is not current_frame:
                # Pose loads above are small; detach metadata/detections and
                # discard dense arrays if the main frame cache supplied a
                # complete sensing payload rather than a selective read.
                compact = dict(bound_frame)
                compact["sensing"] = deepcopy(
                    {
                        key: value
                        for key, value in bound_frame["sensing"].items()
                        if not isinstance(value, np.ndarray)
                    }
                )
                record = (*record[:3], compact)
                cache_key = (type(storage_id), storage_id)
                with self._sensing_cache_lock:
                    if self._sensing_index_token is not expected_token:
                        raise _SensingInventoryChanged
                    self._overlay_history_cache[cache_key] = (result_identity, record)
                    self._overlay_history_cache.move_to_end(cache_key)
                    while len(self._overlay_history_cache) > _OVERLAY_HISTORY_CACHE_CAPACITY:
                        self._overlay_history_cache.popitem(last=False)

        add_candidate(current_frame, current_storage_id)
        storage_ids = self.candidate_storage_frame_ids(current_storage_id)
        expected_token = self._refresh_metadata_index_token(storage_ids)
        eligible = []
        for storage_id, metadata in self._indexed_sensing_results(storage_ids, expected_token):
            if storage_id == current_storage_id:
                continue
            if metadata is None or not self._declares_any_sensing_product(
                metadata, ("detections",)
            ):
                continue
            binding = self._display_binding_record(metadata)
            if binding is None:
                continue
            reference = self._result_reference_time_s(metadata)
            if reference is None or reference > current_reference:
                continue
            eligible.append((self._selection_key(metadata, storage_id), storage_id))

        # History counts sensing results, not scene rows. Select by physical
        # result time before loading any detections/poses; a 12-result trail
        # must not reread thousands of empty/intermediate scene frames.
        for _selection_key, storage_id in sorted(eligible, reverse=True):
            if not use_full_history and len(records) >= max(1, int(history_frames)):
                break
            cache_key = (type(storage_id), storage_id)
            with self._sensing_cache_lock:
                cached = self._overlay_history_cache.get(cache_key)
                if cached is not None:
                    self._overlay_history_cache.move_to_end(cache_key)
            if cached is not None:
                identity, record = cached
                previous = records.get(identity)
                if previous is None or record[1] > previous[1]:
                    records[identity] = record
                continue
            # A preceding displayed result is already a validated projection
            # from this generation. Reuse it instead of loading its detection
            # metadata from HDF5 again just to extend the trail by one point.
            candidate = self._selected_product_cache_get(
                (type(storage_id), storage_id, ("detections",), False),
                index_token=expected_token,
            )
            if candidate is None:
                candidate = self._load_history_frame(storage_id)
            if candidate is not None:
                add_candidate(candidate, storage_id)
        self._require_metadata_index_token(storage_ids, expected_token)

        ordered = sorted(records.values(), key=lambda record: (record[0], record[1]))
        if not use_full_history:
            ordered = ordered[-max(1, int(history_frames)) :]
        return [(record[2], record[3]) for record in ordered]

    def _load_history_frame(self, step: int) -> dict[str, Any] | None:
        """Load only the detection history needed for one earlier frame."""
        cache = getattr(self.visualizer, "cache_service", None)
        if cache is not None:
            cached = cache.get_frame(step)
            if cached is not None:
                _assert_sensing_frame(cached)
                # Scene playback/preload can cache geometry without sensing.
                # Such a cache hit is not evidence that this result has no
                # detections: obtain its selective detection projection below.
                if "detections" in (cached.get("sensing") or {}):
                    return cached

        provider = visualizer_frame_provider(self.visualizer)
        if provider is not None:
            try:
                projection = provider.load_frame_projection(
                    step,
                    _DETECTION_HISTORY_REQUEST,
                )
            except UnsupportedSensingSchemaError:
                raise
            except (OSError, KeyError, ValueError, RuntimeError):
                return None
            projected = {"sensing": dict(projection.frame.sensing or {})}
            _assert_sensing_frame(projected)
            return projected

        frame_source = getattr(self.visualizer, "frame_source", None)
        if frame_source is not None and hasattr(frame_source, "load_frame"):
            try:
                frame = frame_source.load_frame(step)
            except UnsupportedSensingSchemaError:
                raise
            except (OSError, KeyError, ValueError, RuntimeError):
                return None
            if frame is None:
                return None
            if not isinstance(frame, StandardMPCFrame):
                raise TypeError("frame sources must return StandardMPCFrame")
            projection = project_standard_mpc_frame(
                frame,
                _DETECTION_HISTORY_REQUEST,
            )
            projected = {"sensing": dict(projection.frame.sensing or {})}
            _assert_sensing_frame(projected)
            return projected
        return None

    def _build_evidence_layer(
        self,
        raw_frame: dict[str, Any],
        *,
        history: list[tuple[int | str, dict[str, Any]]],
        current_detections: list[dict[str, Any]],
    ) -> SensingSceneLayer | None:
        """Build fading detection evidence points across recent history."""
        pts: list[np.ndarray] = []
        cols: list[list[float]] = []
        total = max(1, len(history) - 1)
        for idx, (_frame_step, frame) in enumerate(history):
            if frame.get("_sensing_scene_prepared"):
                points, colors = frame["_sensing_scene_evidence"]
                age_strength = 0.25 + 0.75 * (idx / total if total > 0 else 1.0)
                faded = np.array(colors, copy=True)
                faded[:, 3] = np.minimum(1.0, faded[:, 3] * age_strength * 0.7)
                pts.extend(points)
                cols.extend(faded)
                continue
            frame_pose = self._resolve_radar_pose(frame)
            if frame_pose is None:
                continue
            frame_radar_xyz, frame_rotation_matrix = frame_pose
            frame_sensing = frame.get("sensing") or {}
            detections = (
                current_detections
                if frame is raw_frame
                else self._prepare_detections(
                    frame_sensing.get("detections") or [],
                    range_mode=self._overlay_range_mode(frame_sensing),
                )
            )
            age_strength = 0.25 + 0.75 * (idx / total if total > 0 else 1.0)
            for det in detections:
                if not self._has_local_cartesian(det):
                    continue
                world = self._sensor_to_world(
                    frame_radar_xyz,
                    frame_rotation_matrix,
                    float(det["x_m"]),
                    float(det["y_m"]),
                    float(det.get("z_m", 0.0) if np.isfinite(det.get("z_m", 0.0)) else 0.0),
                )
                base = self._color_for_detection(det)
                pts.append(world)
                cols.append([base[0], base[1], base[2], min(1.0, base[3] * age_strength * 0.7)])
        if not pts:
            return None
        return SensingSceneLayer(
            name=self.EVIDENCE_GEOMETRY_NAME,
            payload=PointCloudPayload(
                points=np.asarray(pts, dtype=np.float32),
                colors=np.asarray(cols, dtype=np.float32),
            ),
            material=MaterialPayload(
                base_color=(1.0, 1.0, 1.0, 0.9),
                point_size=6.0,
            ),
        )

    def _build_track_layers(
        self,
        raw_frame: dict[str, Any],
        *,
        history: list[tuple[int | str, dict[str, Any]]],
        current_detections: list[dict[str, Any]],
        show_velocity_arrows: bool,
    ) -> list[SensingSceneLayer]:
        """Build per-track point, trail, and optional velocity-arrow layers."""
        track_histories: dict[int, list[dict[str, Any]]] = {}
        for frame_step, frame in history:
            if frame.get("_sensing_scene_prepared"):
                for tid, sample in frame["_sensing_scene_tracks"].items():
                    track_histories.setdefault(int(tid), []).append(sample)
                continue
            frame_pose = self._resolve_radar_pose(frame)
            if frame_pose is None:
                continue
            frame_radar_xyz, frame_rotation_matrix = frame_pose
            frame_sensing = frame.get("sensing") or {}
            detections = (
                current_detections
                if frame is raw_frame
                else self._prepare_detections(
                    frame_sensing.get("detections") or [],
                    range_mode=self._overlay_range_mode(frame_sensing),
                )
            )
            per_track: dict[int, list[tuple[np.ndarray, dict[str, Any]]]] = {}
            for det in detections:
                tid = det.get("track_id")
                local_pos = self._local_cartesian_position(det, prefer_track_state=True)
                if tid is None or local_pos is None:
                    continue
                lx, ly, lz = local_pos
                world = self._sensor_to_world(
                    frame_radar_xyz,
                    frame_rotation_matrix,
                    lx,
                    ly,
                    lz,
                )
                per_track.setdefault(int(tid), []).append((world, det))
            for tid, entries in per_track.items():
                worlds = np.asarray([item[0] for item in entries], dtype=np.float32)
                centroid = worlds.mean(axis=0)
                strongest_det = max(
                    entries,
                    key=lambda item: (
                        float(item[1].get("snr_db", -1e9))
                        if np.isfinite(item[1].get("snr_db", float("nan")))
                        else -1e9
                    ),
                )[1]
                track_histories.setdefault(tid, []).append(
                    {
                        "step": frame_step,
                        "world": centroid,
                        "confirmed": bool(strongest_det.get("track_confirmed", True)),
                        "snr_db": float(strongest_det.get("snr_db", 0.0)),
                        "velocity_world": self._track_velocity_world(
                            strongest_det, frame_rotation_matrix
                        ),
                    }
                )

        layers: list[SensingSceneLayer] = []
        for tid in sorted(track_histories):
            history = track_histories[tid]
            if not history:
                continue
            latest = history[-1]
            confirmed = bool(latest.get("confirmed", True))
            rgba = self._track_rgba(tid, confirmed)
            current = any(det.get("track_id") == tid for det in current_detections)
            alpha = (1.0 if confirmed else 0.7) if current else 0.3
            metadata = {"track_id": tid, "current": current, "confirmed": confirmed}
            point_material = MaterialPayload(
                base_color=(float(rgba[0]), float(rgba[1]), float(rgba[2]), alpha),
                point_size=10.0 if confirmed else 8.0,
            )
            point_payload = PointCloudPayload(
                points=np.asarray([latest["world"]], dtype=np.float32),
                colors=None,
            )
            layers.append(
                SensingSceneLayer(
                    name=f"{self.TRACK_POINT_PREFIX}{tid}",
                    payload=point_payload,
                    material=point_material,
                    metadata=metadata,
                )
            )

            if len(history) >= 2:
                trail_points = np.asarray([sample["world"] for sample in history], dtype=np.float32)
                line_idx = self._polyline_indices(len(trail_points))
                if len(line_idx) > 0:
                    trail_material = MaterialPayload(
                        base_color=(
                            float(rgba[0]),
                            float(rgba[1]),
                            float(rgba[2]),
                            0.55 if current else 0.25,
                        ),
                        line_width=3.5 if confirmed else 2.25,
                    )
                    layers.append(
                        SensingSceneLayer(
                            name=f"{self.TRACK_TRAIL_PREFIX}{tid}",
                            payload=LineSetPayload(
                                points=trail_points,
                                lines=line_idx,
                                colors=None,
                                line_strip=True,
                            ),
                            material=trail_material,
                            metadata=metadata,
                        )
                    )

            if show_velocity_arrows and current:
                arrow_payload = (
                    latest["arrow"]
                    if "arrow" in latest
                    else self._build_velocity_arrow_payload(
                        latest["world"], latest.get("velocity_world")
                    )
                )
                if arrow_payload is not None:
                    arrow_points, arrow_lines = arrow_payload
                    arrow_material = MaterialPayload(
                        base_color=(
                            float(rgba[0]),
                            float(rgba[1]),
                            float(rgba[2]),
                            0.95 if confirmed else 0.65,
                        ),
                        line_width=3.0 if confirmed else 2.0,
                    )
                    layers.append(
                        SensingSceneLayer(
                            name=f"{self.TRACK_ARROW_PREFIX}{tid}",
                            payload=LineSetPayload(
                                points=arrow_points,
                                lines=arrow_lines,
                                colors=None,
                            ),
                            material=arrow_material,
                            metadata=metadata,
                        )
                    )

        return layers

    def _track_rgba(self, tid: int, confirmed: bool) -> np.ndarray:
        """Return the color assigned to a confirmed or tentative track."""
        return np.asarray(
            color_rgba(identity_color(tid), 1.0 if confirmed else 0.7), dtype=np.float32
        )

    @staticmethod
    def _polyline_indices(num_points: int) -> np.ndarray:
        """Return line indices connecting consecutive trail points."""
        if num_points < 2:
            return np.empty((0, 2), dtype=np.int32)
        starts = np.arange(0, num_points - 1, dtype=np.int32)
        ends = np.arange(1, num_points, dtype=np.int32)
        return np.column_stack([starts, ends]).astype(np.int32, copy=False)

    def _track_velocity_world(
        self,
        det: dict[str, Any],
        rotation_matrix: np.ndarray,
    ) -> np.ndarray | None:
        """Rotate tracker velocity into world coordinates for arrow overlays."""
        vx = det.get("track_vx_m_s")
        vy = det.get("track_vy_m_s")
        vz = det.get("track_vz_m_s", 0.0)
        if vx is None or vy is None:
            return None
        if not (np.isfinite(vx) and np.isfinite(vy)):
            return None
        if vz is None or not np.isfinite(vz):
            vz = 0.0
        world = (
            np.asarray(rotation_matrix, dtype=np.float64)
            @ np.asarray([float(vx), float(vy), float(vz)], dtype=np.float64)
        ).astype(np.float32, copy=False)
        if float(np.linalg.norm(world)) < self._MIN_ARROW_SPEED_M_S:
            return None
        return world

    def _build_velocity_arrow_payload(
        self,
        origin: np.ndarray,
        velocity_world: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Build shaft and arrowhead line geometry from a world velocity vector."""
        if velocity_world is None:
            return None
        speed = float(np.linalg.norm(velocity_world))
        if speed < self._MIN_ARROW_SPEED_M_S:
            return None

        direction = velocity_world / speed
        shaft_len = float(np.clip(speed * self._ARROW_LOOKAHEAD_S, 0.35, 1.5))
        tip = np.asarray(origin, dtype=np.float32) + direction * shaft_len

        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        side = np.cross(direction, up)
        side_norm = float(np.linalg.norm(side))
        if side_norm < 1e-4:
            side = np.cross(direction, np.array([0.0, 1.0, 0.0], dtype=np.float32))
            side_norm = float(np.linalg.norm(side))
        if side_norm < 1e-4:
            return None
        side /= side_norm

        head_len = min(0.35, shaft_len * 0.35)
        head_width = head_len * 0.6
        wing_a = tip - direction * head_len + side * head_width
        wing_b = tip - direction * head_len - side * head_width

        points = np.asarray(
            [origin, tip, tip, wing_a, tip, wing_b],
            dtype=np.float32,
        )
        lines = np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
        return points, lines

    def get_detection_overlay(
        self, raw_frame: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, str] | None:
        """Build 3D points (or range rings) for the current frame's detections.

        Returns ``(positions_Nx3, colors_Nx4, geometry_kind)`` where
        *geometry_kind* is ``"points"`` or ``"lines"``, or ``None`` if no
        detections are available.
        """
        _assert_sensing_frame(raw_frame)
        sensing = raw_frame.get("sensing") or {}
        detections = sensing.get("detections")
        if not detections:
            return None

        radar_pose = self._resolve_radar_pose(raw_frame)
        if radar_pose is None:
            return None
        radar_xyz, rotation_matrix = radar_pose
        range_mode = self._overlay_range_mode(sensing)
        detections = self._prepare_detections(list(detections), range_mode=range_mode)

        has_xy = any(
            np.isfinite(d.get("x_m", float("nan"))) and np.isfinite(d.get("y_m", float("nan")))
            for d in detections
        )

        if has_xy:
            return self._detection_points(detections, radar_xyz, rotation_matrix)
        if range_mode != "monostatic":
            raise ValueError(
                "Bistatic range-only detections have no unique point or circular-ring overlay"
            )
        return self._detection_rings(detections, radar_xyz)

    def _detection_points(
        self,
        detections: list[dict[str, Any]],
        radar_xyz: np.ndarray,
        rotation_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str] | None:
        """Build point cloud from detections with x_m/y_m."""
        pts: list[list[float]] = []
        cols: list[list[float]] = []
        for det in detections:
            lx = det.get("x_m", float("nan"))
            ly = det.get("y_m", float("nan"))
            if not (np.isfinite(lx) and np.isfinite(ly)):
                continue
            lz = det.get("z_m", 0.0)
            if not np.isfinite(lz):
                lz = 0.0
            pts.append(
                self._sensor_to_world(
                    radar_xyz,
                    rotation_matrix,
                    float(lx),
                    float(ly),
                    float(lz),
                ).tolist()
            )
            cols.append(self._color_for_detection(det))
        if not pts:
            return None
        return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.float32), "points"

    def _detection_rings(
        self,
        detections: list[dict[str, Any]],
        radar_xyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str] | None:
        """Build range rings for detections without azimuth."""
        all_pts: list[list[float]] = []
        all_cols: list[list[float]] = []
        angles = np.linspace(0, 2 * np.pi, self._RING_SAMPLES, endpoint=False)
        for det in detections:
            r = det.get("range_m")
            if r is None or not np.isfinite(r) or r <= 0:
                continue
            color = self._color_for_detection(det)
            cx, cy, cz = float(radar_xyz[0]), float(radar_xyz[1]), float(radar_xyz[2])
            for a in angles:
                all_pts.append([cx + r * np.cos(a), cy + r * np.sin(a), cz])
                all_cols.append(color)
        if not all_pts:
            return None
        return np.array(all_pts, dtype=np.float32), np.array(all_cols, dtype=np.float32), "lines"

    def _color_for_detection(self, det: dict[str, Any]) -> list[float]:
        """Return RGBA color for a detection."""
        return list(
            color_rgba(detection_color(det), 1.0 if det.get("track_confirmed", True) else 0.7)
        )
