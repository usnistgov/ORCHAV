"""Strict, renderer-independent inspection of persisted detector stages.

The sensing producer can publish ten typed products describing the
two-dimensional CFAR, SNR, NMS, and clustering decisions.  This module turns
those optional products into an immutable view model.  It deliberately
selects products by their scientific descriptor contract rather than their
storage key, and it never recomputes detector values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from shared.sensing.products import (
    select_product_by_contract,
    validate_product_descriptors,
)

_STAGE_AXES = ("pair", "doppler_frequency", "range")


@dataclass(frozen=True, slots=True)
class _StageContract:
    kind: str
    representation: str
    producer_data_key: str


_STAGE_CONTRACTS = (
    _StageContract(
        "range_doppler_detection_input_power",
        "power_linear",
        "range_doppler_detection_input_power_by_pair",
    ),
    _StageContract(
        "range_doppler_cfar_noise_power",
        "power_linear",
        "range_doppler_cfar_noise_power_by_pair",
    ),
    _StageContract(
        "range_doppler_cfar_threshold_power",
        "power_linear",
        "range_doppler_cfar_threshold_power_by_pair",
    ),
    _StageContract(
        "range_doppler_cfar_training_cell_count",
        "integer_count",
        "range_doppler_cfar_training_cell_count_by_pair",
    ),
    _StageContract(
        "range_doppler_cfar_eligible_mask",
        "boolean_mask",
        "range_doppler_cfar_eligible_mask_by_pair",
    ),
    _StageContract(
        "range_doppler_cfar_mask",
        "boolean_mask",
        "range_doppler_cfar_mask_by_pair",
    ),
    _StageContract(
        "range_doppler_nms_mask",
        "boolean_mask",
        "range_doppler_nms_mask_by_pair",
    ),
    _StageContract(
        "range_doppler_snr_db",
        "log_power_ratio_db",
        "range_doppler_snr_db_by_pair",
    ),
    _StageContract(
        "range_doppler_snr_valid_mask",
        "boolean_mask",
        "range_doppler_snr_valid_mask_by_pair",
    ),
    _StageContract(
        "range_doppler_cluster_labels",
        "integer_label",
        "range_doppler_cluster_labels_by_pair",
    ),
)

DETECTION_STAGE_KINDS = tuple(contract.kind for contract in _STAGE_CONTRACTS)


class InspectionDisposition(str, Enum):
    """Whether an inspection model may expose persisted stage values."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"


class DetectionStageContractError(ValueError):
    """Raised when loaded detector-stage products contradict their contract."""


@dataclass(frozen=True, slots=True)
class DetectionStageEvent:
    """Persisted occurrence identity shared by all products in a bundle."""

    clock_id: str
    acquisition_id: str
    processing_run_id: str
    processing_recipe_id: str
    pair_id: str
    reference_time_s: float


@dataclass(frozen=True, slots=True)
class ProductTimeSpan:
    """Exact temporal support copied from a typed product descriptor."""

    clock_id: str
    start_time_s: float
    end_time_s: float
    reference_time_s: float
    reference_time_policy: str


@dataclass(frozen=True, slots=True)
class LineageNode:
    """One resolved or external node in a product derivation trace."""

    product_id: str
    kind: str | None
    source_product_ids: tuple[str, ...]
    descriptor_available: bool
    data_loaded: bool


@dataclass(frozen=True, slots=True)
class LineageTrace:
    """Deterministic root-first trace of one typed detector-stage product."""

    root_product_id: str
    nodes: tuple[LineageNode, ...]
    unresolved_source_product_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DetectionStageProduct:
    """One optional typed stage and its immutable numeric payload."""

    kind: str
    representation: str
    disposition: InspectionDisposition
    reason: str | None
    data_key: str | None = None
    product_id: str | None = None
    evidence_role: str | None = None
    values: np.ndarray | None = None
    lineage_trace: LineageTrace | None = None

    @property
    def available(self) -> bool:
        """Return whether exact persisted values are available for inspection."""

        return self.disposition is InspectionDisposition.AVAILABLE


@dataclass(frozen=True, slots=True)
class CellSelection:
    """Stable pair/bin selection independent of tables and playback order."""

    pair_id: str
    doppler_bin: int
    range_bin: int
    cluster_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ValueError("CellSelection.pair_id must be a non-empty string")
        for field_name in ("doppler_bin", "range_bin"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"CellSelection.{field_name} must be a nonnegative integer")
        if self.cluster_id is not None and (
            not isinstance(self.cluster_id, int)
            or isinstance(self.cluster_id, bool)
            or self.cluster_id < 0
        ):
            raise ValueError("CellSelection.cluster_id must be a nonnegative integer or None")


@dataclass(frozen=True, slots=True)
class DetectionLink:
    """Stable link from a detector cluster to its published peak detection."""

    acquisition_id: str
    processing_run_id: str
    pair_id: str
    cluster_id: int
    doppler_bin: int
    range_bin: int

    @property
    def stable_key(self) -> tuple[str, str, str, int, int, int]:
        """Return the occurrence-scoped detection identity."""

        return (
            self.acquisition_id,
            self.processing_run_id,
            self.pair_id,
            self.cluster_id,
            self.doppler_bin,
            self.range_bin,
        )


@dataclass(frozen=True, slots=True)
class CellInspection:
    """Exact persisted decisions and values for one RD cell."""

    disposition: InspectionDisposition
    reason: str | None
    selection: CellSelection
    doppler_hz: float | None = None
    range_m: float | None = None
    detection_input_power: float | None = None
    cfar_noise_power: float | None = None
    cfar_threshold_power: float | None = None
    cfar_training_cell_count: int | None = None
    cfar_eligible: bool | None = None
    cfar_passed: bool | None = None
    nms_retained: bool | None = None
    snr_db: float | None = None
    snr_valid: bool | None = None
    cluster_id: int | None = None
    detection_link: DetectionLink | None = None
    eligibility_disposition: str = "unavailable"
    cfar_disposition: str = "unavailable"
    nms_disposition: str = "unavailable"
    cluster_disposition: str = "unavailable"
    detection_disposition: str = "unavailable"


@dataclass(frozen=True, slots=True)
class DetectionStageBundle:
    """Immutable, event-bound collection of recorded detector stages."""

    disposition: InspectionDisposition
    reason: str | None
    event: DetectionStageEvent | None
    products: tuple[DetectionStageProduct, ...]
    time_span: ProductTimeSpan | None = None
    doppler_axis_hz: np.ndarray | None = None
    range_axis_m: np.ndarray | None = None
    detection_links: tuple[DetectionLink, ...] = ()

    @property
    def available(self) -> bool:
        """Return whether at least one valid stage payload is available."""

        return self.disposition is InspectionDisposition.AVAILABLE

    def product(self, kind: str) -> DetectionStageProduct:
        """Return the explicitly represented stage for ``kind``."""

        for product in self.products:
            if product.kind == kind:
                return product
        raise KeyError(f"Unknown detector-stage kind {kind!r}")

    def trace_lineage(self, kind: str) -> LineageTrace | None:
        """Return the persisted derivation trace for an available stage."""

        return self.product(kind).lineage_trace

    def inspect(self, selection: CellSelection) -> CellInspection:
        """Inspect one exact pair/doppler/range cell without recomputation."""

        return inspect_detection_cell(self, selection)


def _immutable_array(value: Any) -> np.ndarray:
    """Return an ndarray backed by immutable bytes rather than caller memory."""

    array = np.ascontiguousarray(np.asarray(value))
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    assert not immutable.flags.writeable
    return immutable


def _required_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DetectionStageContractError(f"{path} must be a mapping")
    return value


def _required_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise DetectionStageContractError(f"{path} must be a non-empty string")
    return value


def _event_identity(sensing: Mapping[str, Any]) -> DetectionStageEvent:
    temporal = _required_mapping(sensing.get("temporal"), "sensing.temporal")
    clock = _required_mapping(temporal.get("clock"), "sensing.temporal.clock")
    acquisition = _required_mapping(temporal.get("acquisition"), "sensing.temporal.acquisition")
    result = _required_mapping(temporal.get("result"), "sensing.temporal.result")
    reference_time = result.get("reference_time_s")
    if (
        not isinstance(reference_time, (int, float, np.integer, np.floating))
        or isinstance(reference_time, (bool, np.bool_))
        or not np.isfinite(reference_time)
    ):
        raise DetectionStageContractError("sensing.temporal.result.reference_time_s must be finite")
    return DetectionStageEvent(
        clock_id=_required_string(clock.get("clock_id"), "temporal.clock.clock_id"),
        acquisition_id=_required_string(
            acquisition.get("acquisition_id"), "temporal.acquisition.acquisition_id"
        ),
        processing_run_id=_required_string(
            result.get("processing_run_id"), "temporal.result.processing_run_id"
        ),
        processing_recipe_id=_required_string(
            result.get("processing_recipe_id"), "temporal.result.processing_recipe_id"
        ),
        pair_id=_required_string(acquisition.get("pair_id"), "temporal.acquisition.pair_id"),
        reference_time_s=float(reference_time),
    )


def _time_span(raw: Any, path: str) -> ProductTimeSpan:
    span = _required_mapping(raw, path)
    numeric: dict[str, float] = {}
    for field_name in ("start_time_s", "end_time_s", "reference_time_s"):
        value = span.get(field_name)
        if (
            not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)
        ):
            raise DetectionStageContractError(f"{path}.{field_name} must be finite")
        numeric[field_name] = float(value)
    return ProductTimeSpan(
        clock_id=_required_string(span.get("clock_id"), f"{path}.clock_id"),
        start_time_s=numeric["start_time_s"],
        end_time_s=numeric["end_time_s"],
        reference_time_s=numeric["reference_time_s"],
        reference_time_policy=_required_string(
            span.get("reference_time_policy"), f"{path}.reference_time_policy"
        ),
    )


def _descriptor_index(
    sensing: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, tuple[str, Mapping[str, Any]]]]:
    products = sensing.get("products")
    if not isinstance(products, Mapping):
        return {}, {}
    by_id: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for data_key, descriptor in products.items():
        if not isinstance(data_key, str) or not isinstance(descriptor, Mapping):
            continue
        product_id = descriptor.get("product_id")
        if isinstance(product_id, str):
            if product_id in by_id:
                raise DetectionStageContractError(
                    f"duplicate typed product identity {product_id!r}"
                )
            by_id[product_id] = (data_key, descriptor)
    return products, by_id


def _build_lineage_trace(
    *,
    root_product_id: str,
    descriptor_by_id: Mapping[str, tuple[str, Mapping[str, Any]]],
    sensing: Mapping[str, Any],
) -> LineageTrace:
    nodes: list[LineageNode] = []
    unresolved: list[str] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(product_id: str) -> None:
        if product_id in visiting:
            raise DetectionStageContractError("detector-stage product lineage contains a cycle")
        if product_id in visited:
            return
        visited.add(product_id)
        resolved = descriptor_by_id.get(product_id)
        if resolved is None:
            unresolved.append(product_id)
            nodes.append(
                LineageNode(
                    product_id=product_id,
                    kind=None,
                    source_product_ids=(),
                    descriptor_available=False,
                    data_loaded=False,
                )
            )
            return
        data_key, descriptor = resolved
        lineage = _required_mapping(
            descriptor.get("lineage"), f"sensing.products[{data_key!r}].lineage"
        )
        raw_sources = lineage.get("source_product_ids")
        if not isinstance(raw_sources, list) or any(
            not isinstance(source, str) for source in raw_sources
        ):
            raise DetectionStageContractError(
                f"sensing.products[{data_key!r}].lineage.source_product_ids must be a list"
            )
        sources = tuple(raw_sources)
        nodes.append(
            LineageNode(
                product_id=product_id,
                kind=(str(descriptor["kind"]) if isinstance(descriptor.get("kind"), str) else None),
                source_product_ids=sources,
                descriptor_available=True,
                data_loaded=data_key in sensing,
            )
        )
        visiting.add(product_id)
        for source in sources:
            visit(source)
        visiting.remove(product_id)

    visit(root_product_id)
    return LineageTrace(
        root_product_id=root_product_id,
        nodes=tuple(nodes),
        unresolved_source_product_ids=tuple(dict.fromkeys(unresolved)),
    )


def _unavailable_product(contract: _StageContract, reason: str) -> DetectionStageProduct:
    return DetectionStageProduct(
        kind=contract.kind,
        representation=contract.representation,
        disposition=InspectionDisposition.UNAVAILABLE,
        reason=reason,
    )


def _invalid_products(reason: str) -> tuple[DetectionStageProduct, ...]:
    return tuple(
        DetectionStageProduct(
            kind=contract.kind,
            representation=contract.representation,
            disposition=InspectionDisposition.INVALID,
            reason=reason,
        )
        for contract in _STAGE_CONTRACTS
    )


def _loaded_stage_product(
    sensing: Mapping[str, Any],
    *,
    contract: _StageContract,
    products: Mapping[str, Any],
    descriptor_by_id: Mapping[str, tuple[str, Mapping[str, Any]]],
) -> tuple[
    DetectionStageProduct,
    tuple[np.ndarray | list[Any], ...] | None,
    Mapping[str, Any] | None,
]:
    candidates = [
        (data_key, descriptor)
        for data_key, descriptor in products.items()
        if isinstance(data_key, str)
        and isinstance(descriptor, Mapping)
        and descriptor.get("kind") == contract.kind
    ]
    if not candidates:
        if contract.producer_data_key in sensing:
            raise DetectionStageContractError(
                f"numeric detector stage {contract.kind!r} is present without a descriptor"
            )
        return _unavailable_product(contract, "descriptor_absent"), None, None
    if len(candidates) != 1:
        raise DetectionStageContractError(
            f"detector stage {contract.kind!r} has {len(candidates)} descriptors"
        )
    data_key, descriptor = candidates[0]
    axes = descriptor.get("axes")
    if not isinstance(axes, list):
        raise DetectionStageContractError(f"detector stage {contract.kind!r} axes must be a list")
    axis_names = tuple(axis.get("name") if isinstance(axis, Mapping) else None for axis in axes)
    if descriptor.get("variant") != "primary":
        raise DetectionStageContractError(
            f"detector stage {contract.kind!r} must use variant 'primary'"
        )
    if axis_names != _STAGE_AXES:
        raise DetectionStageContractError(
            f"detector stage {contract.kind!r} must use axes {_STAGE_AXES!r}"
        )
    if descriptor.get("representation") != contract.representation:
        raise DetectionStageContractError(
            f"detector stage {contract.kind!r} must use representation "
            f"{contract.representation!r}"
        )
    expected_axis_metadata = (
        ("pair", "1", "catalog"),
        ("doppler_frequency", "Hz", "dense"),
        ("range", "m", "dense"),
    )
    for index, (name, unit, coordinate_kind) in enumerate(expected_axis_metadata):
        axis = axes[index]
        if (
            not isinstance(axis, Mapping)
            or axis.get("name") != name
            or axis.get("unit") != unit
            or axis.get("coordinate_kind") != coordinate_kind
        ):
            raise DetectionStageContractError(
                f"detector stage {contract.kind!r} axis {index} metadata is contradictory"
            )
    product_id = _required_string(
        descriptor.get("product_id"), f"sensing.products[{data_key!r}].product_id"
    )
    lineage_trace = _build_lineage_trace(
        root_product_id=product_id,
        descriptor_by_id=descriptor_by_id,
        sensing=sensing,
    )
    if data_key not in sensing:
        return (
            DetectionStageProduct(
                kind=contract.kind,
                representation=contract.representation,
                disposition=InspectionDisposition.UNAVAILABLE,
                reason="data_not_loaded",
                data_key=data_key,
                product_id=product_id,
                evidence_role=(
                    str(descriptor["evidence_role"])
                    if isinstance(descriptor.get("evidence_role"), str)
                    else None
                ),
                lineage_trace=lineage_trace,
            ),
            None,
            descriptor,
        )
    try:
        selected_key, values, coordinates, selected_descriptor = select_product_by_contract(
            sensing,
            kind=contract.kind,
            variant="primary",
            axis_names=_STAGE_AXES,
            representations=(contract.representation,),
        )
    except (KeyError, ValueError) as exc:
        raise DetectionStageContractError(str(exc)) from exc
    if selected_key != data_key or selected_descriptor != descriptor:
        raise DetectionStageContractError(
            f"detector stage {contract.kind!r} did not resolve to its unique descriptor"
        )
    return (
        DetectionStageProduct(
            kind=contract.kind,
            representation=contract.representation,
            disposition=InspectionDisposition.AVAILABLE,
            reason=None,
            data_key=data_key,
            product_id=product_id,
            evidence_role=str(descriptor["evidence_role"]),
            values=_immutable_array(values),
            lineage_trace=lineage_trace,
        ),
        coordinates,
        descriptor,
    )


def _validate_stage_identity(
    *, descriptor: Mapping[str, Any], event: DetectionStageEvent, data_key: str
) -> ProductTimeSpan:
    if descriptor.get("temporal_support") != {"type": "full_acquisition"}:
        raise DetectionStageContractError(
            f"sensing.products[{data_key!r}].temporal_support must be full_acquisition"
        )
    lineage = _required_mapping(
        descriptor.get("lineage"), f"sensing.products[{data_key!r}].lineage"
    )
    expected_lineage = {
        "acquisition_id": event.acquisition_id,
        "processing_run_id": event.processing_run_id,
        "processing_recipe_id": event.processing_recipe_id,
    }
    for field_name, expected in expected_lineage.items():
        if lineage.get(field_name) != expected:
            raise DetectionStageContractError(
                f"sensing.products[{data_key!r}].lineage.{field_name} "
                "contradicts the selected event"
            )
    if descriptor.get("pair_ids") != [event.pair_id]:
        raise DetectionStageContractError(
            f"sensing.products[{data_key!r}].pair_ids contradicts the selected event"
        )
    pair = _required_mapping(descriptor.get("pair"), f"sensing.products[{data_key!r}].pair")
    if pair.get("pair_id") != event.pair_id:
        raise DetectionStageContractError(
            f"sensing.products[{data_key!r}].pair.pair_id contradicts the selected event"
        )
    span = _time_span(descriptor.get("time_span"), f"sensing.products[{data_key!r}].time_span")
    if span.clock_id != event.clock_id or span.reference_time_s != event.reference_time_s:
        raise DetectionStageContractError(
            f"sensing.products[{data_key!r}].time_span contradicts the selected event"
        )
    return span


def _coordinate_signature(descriptor: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    axes = descriptor.get("axes")
    if not isinstance(axes, list):
        raise DetectionStageContractError("detector-stage descriptor axes must be a list")
    return tuple(
        (
            axis.get("name"),
            axis.get("size"),
            axis.get("unit"),
            axis.get("coordinate_kind"),
            axis.get("coordinate_sha256"),
        )
        for axis in axes
        if isinstance(axis, Mapping)
    )


def _exact_dense_axis(value: Any, *, name: str) -> np.ndarray:
    axis = np.asarray(value)
    if axis.ndim != 1 or not np.issubdtype(axis.dtype, np.number):
        raise DetectionStageContractError(f"{name} coordinates must be a numeric vector")
    if not np.all(np.isfinite(axis)):
        raise DetectionStageContractError(f"{name} coordinates must be finite")
    return _immutable_array(axis)


def _validate_pair_coordinate(value: Any, event: DetectionStageEvent) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], Mapping)
        or value[0].get("pair_id") != event.pair_id
    ):
        raise DetectionStageContractError(
            "detector-stage pair coordinates must contain the selected event pair exactly once"
        )


def _exact_int(value: Any, *, path: str, minimum: int = 0) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise DetectionStageContractError(f"{path} must be an exact integer")
    parsed = int(value)
    if parsed < minimum:
        raise DetectionStageContractError(f"{path} must be at least {minimum}")
    return parsed


def _detection_links(
    sensing: Mapping[str, Any],
    *,
    event: DetectionStageEvent,
    shape: tuple[int, int, int],
) -> tuple[DetectionLink, ...]:
    raw_detections = sensing.get("detections", [])
    if not isinstance(raw_detections, list):
        raise DetectionStageContractError("sensing.detections must be a list when present")
    links: list[DetectionLink] = []
    cluster_ids: set[int] = set()
    for index, raw_detection in enumerate(raw_detections):
        if not isinstance(raw_detection, Mapping):
            raise DetectionStageContractError(f"sensing.detections[{index}] must be a mapping")
        cluster_id = _exact_int(
            raw_detection.get("cluster_id"), path=f"sensing.detections[{index}].cluster_id"
        )
        doppler_bin = _exact_int(
            raw_detection.get("doppler_bin"),
            path=f"sensing.detections[{index}].doppler_bin",
        )
        range_bin = _exact_int(
            raw_detection.get("range_bin"), path=f"sensing.detections[{index}].range_bin"
        )
        if cluster_id in cluster_ids:
            raise DetectionStageContractError(
                f"sensing.detections has duplicate cluster_id {cluster_id}"
            )
        if doppler_bin >= shape[1] or range_bin >= shape[2]:
            raise DetectionStageContractError(
                f"sensing.detections[{index}] peak bin is outside detector-stage coordinates"
            )
        # Angle-aware modes can collapse multiple three-dimensional clusters
        # onto one RD cell.  Their two-dimensional label product therefore
        # cannot prove a one-to-one detection join; the occurrence-scoped
        # cluster/bin tuple below remains the stable link.
        cluster_ids.add(cluster_id)
        links.append(
            DetectionLink(
                acquisition_id=event.acquisition_id,
                processing_run_id=event.processing_run_id,
                pair_id=event.pair_id,
                cluster_id=cluster_id,
                doppler_bin=doppler_bin,
                range_bin=range_bin,
            )
        )
    return tuple(sorted(links, key=lambda link: link.stable_key))


def _validate_decision_relations(products: Mapping[str, DetectionStageProduct]) -> None:
    def array(kind: str) -> np.ndarray | None:
        product = products[kind]
        return product.values if product.available else None

    eligible = array("range_doppler_cfar_eligible_mask")
    cfar = array("range_doppler_cfar_mask")
    nms = array("range_doppler_nms_mask")
    snr_valid = array("range_doppler_snr_valid_mask")
    noise = array("range_doppler_cfar_noise_power")
    labels = array("range_doppler_cluster_labels")
    if cfar is not None and eligible is not None and np.any(cfar & ~eligible):
        raise DetectionStageContractError("CFAR-positive cells must be eligible")
    if snr_valid is not None and eligible is not None and np.any(snr_valid & ~eligible):
        raise DetectionStageContractError("SNR-valid cells must be CFAR-eligible")
    if snr_valid is not None and noise is not None and np.any(snr_valid & (noise <= 0.0)):
        raise DetectionStageContractError("SNR-valid cells must have positive noise power")
    if labels is not None:
        if np.any(labels < -1):
            raise DetectionStageContractError("cluster labels may only use -1 as background")
        if nms is not None and np.any((labels >= 0) & ~nms):
            raise DetectionStageContractError("clustered cells must be retained by NMS")


def build_detection_stage_bundle(sensing: Mapping[str, Any]) -> DetectionStageBundle:
    """Build a strict detector-stage bundle, raising on contradictions.

    Missing optional descriptors or selectively unloaded values are represented
    explicitly.  A malformed loaded descriptor, payload, coordinate, lineage,
    or time span raises :class:`DetectionStageContractError`.
    """

    if not isinstance(sensing, Mapping):
        raise DetectionStageContractError("sensing must be a mapping")
    event = _event_identity(sensing)
    products, descriptor_by_id = _descriptor_index(sensing)
    if not products:
        orphaned = [
            contract.producer_data_key
            for contract in _STAGE_CONTRACTS
            if contract.producer_data_key in sensing
        ]
        if orphaned:
            raise DetectionStageContractError(
                "detector-stage arrays are present without typed descriptors: "
                + ", ".join(orphaned)
            )
        return DetectionStageBundle(
            disposition=InspectionDisposition.UNAVAILABLE,
            reason="no_typed_product_descriptors",
            event=event,
            products=tuple(
                _unavailable_product(contract, "descriptor_absent") for contract in _STAGE_CONTRACTS
            ),
        )
    descriptor_errors = validate_product_descriptors(sensing, require_described_data=False)
    if descriptor_errors:
        raise DetectionStageContractError(
            "invalid typed sensing products: " + "; ".join(descriptor_errors)
        )

    stage_products: list[DetectionStageProduct] = []
    coordinate_sets: list[tuple[np.ndarray | list[Any], ...]] = []
    time_spans: list[ProductTimeSpan] = []
    signatures: list[tuple[tuple[Any, ...], ...]] = []
    for contract in _STAGE_CONTRACTS:
        product, coordinates, descriptor = _loaded_stage_product(
            sensing,
            contract=contract,
            products=products,
            descriptor_by_id=descriptor_by_id,
        )
        stage_products.append(product)
        if descriptor is not None:
            if product.data_key is None:
                raise DetectionStageContractError(
                    f"detector stage {contract.kind!r} lost its descriptor key"
                )
            span = _validate_stage_identity(
                descriptor=descriptor,
                event=event,
                data_key=product.data_key,
            )
            time_spans.append(span)
            signatures.append(_coordinate_signature(descriptor))
        if product.available:
            assert coordinates is not None and descriptor is not None
            coordinate_sets.append(coordinates)

    available = [product for product in stage_products if product.available]
    if any(span != time_spans[0] for span in time_spans[1:]):
        raise DetectionStageContractError("detector stages do not share one exact time span")
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise DetectionStageContractError("detector stages do not share exact coordinates")
    if not available:
        return DetectionStageBundle(
            disposition=InspectionDisposition.UNAVAILABLE,
            reason=(
                "detector_stages_not_recorded"
                if all(p.reason == "descriptor_absent" for p in stage_products)
                else "detector_stage_values_not_loaded"
            ),
            event=event,
            products=tuple(stage_products),
            time_span=time_spans[0] if time_spans else None,
        )

    reference_coordinates = coordinate_sets[0]
    if len(reference_coordinates) != 3:
        raise DetectionStageContractError("detector stages require three exact coordinates")
    _validate_pair_coordinate(reference_coordinates[0], event)
    doppler_axis = _exact_dense_axis(reference_coordinates[1], name="doppler_frequency")
    range_axis = _exact_dense_axis(reference_coordinates[2], name="range")
    for coordinates in coordinate_sets[1:]:
        _validate_pair_coordinate(coordinates[0], event)
        if not np.array_equal(np.asarray(coordinates[1]), doppler_axis):
            raise DetectionStageContractError("detector stages have different Doppler coordinates")
        if not np.array_equal(np.asarray(coordinates[2]), range_axis):
            raise DetectionStageContractError("detector stages have different range coordinates")

    expected_shape = (1, int(doppler_axis.size), int(range_axis.size))
    by_kind = {product.kind: product for product in stage_products}
    for product in available:
        assert product.values is not None
        if product.values.shape != expected_shape:
            raise DetectionStageContractError(
                f"detector stage {product.kind!r} has shape {product.values.shape}, "
                f"expected {expected_shape}"
            )
    _validate_decision_relations(by_kind)
    links = _detection_links(
        sensing,
        event=event,
        shape=expected_shape,
    )
    return DetectionStageBundle(
        disposition=InspectionDisposition.AVAILABLE,
        reason=None,
        event=event,
        products=tuple(stage_products),
        time_span=time_spans[0],
        doppler_axis_hz=doppler_axis,
        range_axis_m=range_axis,
        detection_links=links,
    )


def load_detection_stage_bundle(sensing: Mapping[str, Any]) -> DetectionStageBundle:
    """Load a bundle and convert contract failures into a fail-closed model."""

    try:
        return build_detection_stage_bundle(sensing)
    except (DetectionStageContractError, TypeError, ValueError) as exc:
        reason = str(exc) or type(exc).__name__
        return DetectionStageBundle(
            disposition=InspectionDisposition.INVALID,
            reason=reason,
            event=None,
            products=_invalid_products(reason),
        )


def _cell_value(bundle: DetectionStageBundle, kind: str, selection: CellSelection) -> Any:
    product = bundle.product(kind)
    if not product.available or product.values is None:
        return None
    return product.values[0, selection.doppler_bin, selection.range_bin].item()


def inspect_detection_cell(
    bundle: DetectionStageBundle, selection: CellSelection
) -> CellInspection:
    """Return exact persisted values and decisions for ``selection``."""

    if bundle.disposition is not InspectionDisposition.AVAILABLE:
        return CellInspection(
            disposition=bundle.disposition,
            reason=bundle.reason,
            selection=selection,
        )
    assert bundle.event is not None
    assert bundle.doppler_axis_hz is not None and bundle.range_axis_m is not None
    if selection.pair_id != bundle.event.pair_id:
        raise ValueError("CellSelection.pair_id does not match the detector-stage event")
    if selection.doppler_bin >= bundle.doppler_axis_hz.size:
        raise IndexError("CellSelection.doppler_bin is outside the persisted Doppler axis")
    if selection.range_bin >= bundle.range_axis_m.size:
        raise IndexError("CellSelection.range_bin is outside the persisted range axis")

    input_power = _cell_value(bundle, "range_doppler_detection_input_power", selection)
    noise_power = _cell_value(bundle, "range_doppler_cfar_noise_power", selection)
    threshold_power = _cell_value(bundle, "range_doppler_cfar_threshold_power", selection)
    training_count = _cell_value(bundle, "range_doppler_cfar_training_cell_count", selection)
    eligible = _cell_value(bundle, "range_doppler_cfar_eligible_mask", selection)
    cfar_passed = _cell_value(bundle, "range_doppler_cfar_mask", selection)
    nms_retained = _cell_value(bundle, "range_doppler_nms_mask", selection)
    snr_valid = _cell_value(bundle, "range_doppler_snr_valid_mask", selection)
    raw_snr = _cell_value(bundle, "range_doppler_snr_db", selection)
    raw_cluster_id = _cell_value(bundle, "range_doppler_cluster_labels", selection)

    eligible_bool = bool(eligible) if eligible is not None else None
    cfar_bool = bool(cfar_passed) if cfar_passed is not None else None
    nms_bool = bool(nms_retained) if nms_retained is not None else None
    snr_valid_bool = bool(snr_valid) if snr_valid is not None else None
    # The persisted SNR fill value has no scientific meaning when its explicit
    # denominator-validity mask is false or unavailable.
    snr_db = float(raw_snr) if raw_snr is not None and snr_valid_bool is True else None
    cluster_id = (
        int(raw_cluster_id) if raw_cluster_id is not None and int(raw_cluster_id) >= 0 else None
    )

    if eligible_bool is None:
        eligibility_disposition = "unavailable"
    else:
        eligibility_disposition = "eligible" if eligible_bool else "ineligible"
    if cfar_bool is None:
        cfar_disposition = "unavailable"
    elif eligible_bool is False:
        cfar_disposition = "ineligible"
    else:
        cfar_disposition = "passed" if cfar_bool else "rejected"
    if nms_bool is None:
        nms_disposition = "unavailable"
    elif nms_bool:
        nms_disposition = "retained"
    elif cfar_bool is False:
        nms_disposition = "not_candidate"
    elif cfar_bool is True:
        nms_disposition = "suppressed"
    else:
        nms_disposition = "not_retained"
    if raw_cluster_id is None:
        cluster_disposition = "unavailable"
    else:
        cluster_disposition = "clustered" if cluster_id is not None else "background"

    requested_cluster_id = selection.cluster_id
    linked_cluster_id = requested_cluster_id if requested_cluster_id is not None else cluster_id
    link = next(
        (item for item in bundle.detection_links if item.cluster_id == linked_cluster_id),
        None,
    )
    if requested_cluster_id is not None:
        if link is None:
            raise KeyError(
                f"CellSelection.cluster_id {requested_cluster_id} is not published in this event"
            )
        if link.doppler_bin != selection.doppler_bin or link.range_bin != selection.range_bin:
            raise ValueError("CellSelection cluster/bin identity is contradictory")
    if link is None and linked_cluster_id is None:
        link = next(
            (
                item
                for item in bundle.detection_links
                if item.doppler_bin == selection.doppler_bin
                and item.range_bin == selection.range_bin
            ),
            None,
        )
    if link is not None:
        detection_disposition = "linked"
    elif cluster_id is not None:
        detection_disposition = "cluster_not_published"
    else:
        detection_disposition = "background"

    return CellInspection(
        disposition=InspectionDisposition.AVAILABLE,
        reason=None,
        selection=selection,
        doppler_hz=float(bundle.doppler_axis_hz[selection.doppler_bin]),
        range_m=float(bundle.range_axis_m[selection.range_bin]),
        detection_input_power=(float(input_power) if input_power is not None else None),
        cfar_noise_power=(float(noise_power) if noise_power is not None else None),
        cfar_threshold_power=(float(threshold_power) if threshold_power is not None else None),
        cfar_training_cell_count=(int(training_count) if training_count is not None else None),
        cfar_eligible=eligible_bool,
        cfar_passed=cfar_bool,
        nms_retained=nms_bool,
        snr_db=snr_db,
        snr_valid=snr_valid_bool,
        cluster_id=cluster_id,
        detection_link=link,
        eligibility_disposition=eligibility_disposition,
        cfar_disposition=cfar_disposition,
        nms_disposition=nms_disposition,
        cluster_disposition=cluster_disposition,
        detection_disposition=detection_disposition,
    )


__all__ = [
    "CellInspection",
    "CellSelection",
    "DETECTION_STAGE_KINDS",
    "DetectionLink",
    "DetectionStageBundle",
    "DetectionStageContractError",
    "DetectionStageEvent",
    "DetectionStageProduct",
    "InspectionDisposition",
    "LineageNode",
    "LineageTrace",
    "ProductTimeSpan",
    "build_detection_stage_bundle",
    "inspect_detection_cell",
    "load_detection_stage_bundle",
]
