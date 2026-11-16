"""Typed sensing-product descriptors and immutable derivation lineage.

The generic packed-HDF5 container remains responsible only for storing arrays
and JSON.  This module defines the logical scientific contract carried inside
the optional sensing extension: what each array represents, its ordered axes,
its physical units, and the acquisition/recipe that produced it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from shared.frames.json_codec import dumps_frame_json

SENSING_PRODUCT_CONTRACT_VERSION = 2

TEMPORAL_SUPPORT_FULL_ACQUISITION = "full_acquisition"
TEMPORAL_SUPPORT_LAST_VALID_SAMPLE = "last_valid_acquisition_sample"
ACQUISITION_SAMPLE_REFERENCE_POLICY = "acquisition_sample_exact"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COORDINATE_KINDS = frozenset({"dense", "catalog"})
_VALUE_REPRESENTATIONS = frozenset(
    {
        "boolean_mask",
        "complex_voltage",
        "integer_count",
        "integer_label",
        "log_power_ratio_db",
        "power_linear",
    }
)
_VALUE_REPRESENTATION_UNITS = {
    "boolean_mask": "1",
    "complex_voltage": "1",
    "integer_count": "1",
    "integer_label": "1",
    "log_power_ratio_db": "dB",
    "power_linear": "1",
}
_SCALE_SEMANTICS = frozenset({"relative_unitless"})
_PRODUCT_DESCRIPTOR_FIELDS = frozenset(
    {
        "axes",
        "combine",
        "contract_version",
        "data_key",
        "data_sha256",
        "dtype",
        "evidence_role",
        "kind",
        "lineage",
        "pair",
        "pair_ids",
        "processing",
        "product_id",
        "representation",
        "scale_semantics",
        "shape",
        "temporal_support",
        "time_span",
        "unit",
        "variant",
    }
)


def _representation_contract_errors(
    *,
    representation: Any,
    unit: Any,
    dtype: Any,
    values: np.ndarray | None = None,
) -> list[str]:
    """Return value-representation violations for one numeric product.

    Product contract v2 intentionally admits only the units and storage
    categories exercised by the native sensing pipeline. An absolute voltage
    or power unit therefore requires a future reviewed contract revision
    instead of being accepted through an arbitrary string.
    """

    if representation not in _VALUE_REPRESENTATIONS:
        return []

    errors: list[str] = []
    expected_unit = _VALUE_REPRESENTATION_UNITS[representation]
    if unit != expected_unit:
        errors.append(f"{representation} requires unit {expected_unit!r}")

    try:
        resolved_dtype = np.dtype(dtype)
    except (TypeError, ValueError):
        errors.append(f"{representation} requires a valid NumPy dtype")
        return errors

    if representation == "boolean_mask":
        valid_dtype = np.issubdtype(resolved_dtype, np.bool_)
        dtype_requirement = "boolean dtype"
    elif representation in {"integer_count", "integer_label"}:
        valid_dtype = np.issubdtype(resolved_dtype, np.integer)
        dtype_requirement = "signed or unsigned integer dtype"
    elif representation == "complex_voltage":
        valid_dtype = np.issubdtype(resolved_dtype, np.complexfloating)
        dtype_requirement = "complex floating dtype"
    else:
        valid_dtype = np.issubdtype(resolved_dtype, np.floating)
        dtype_requirement = "real floating dtype"
    if not valid_dtype:
        errors.append(f"{representation} requires {dtype_requirement}")
        return errors

    if values is None:
        return errors
    array = np.asarray(values)
    if representation == "integer_count" and np.any(array < 0):
        errors.append("integer_count values must be nonnegative")
    elif representation == "power_linear":
        if not np.all(np.isfinite(array)):
            errors.append("power_linear values must be finite")
        if np.any(array < 0):
            errors.append("power_linear values must be nonnegative")
    elif representation == "log_power_ratio_db" and not np.all(np.isfinite(array)):
        errors.append("log_power_ratio_db values must be finite")
    return errors


def _content_hash(value: Any, *, domain: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"orchav.sensing.{domain}.v1\0".encode("ascii"))
    digest.update(dumps_frame_json(value).encode("utf-8"))
    return digest.hexdigest()


def _canonical_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    """Return the contract's endian-stable numeric dtype."""

    return dtype if dtype.itemsize <= 1 or dtype.byteorder == "|" else dtype.newbyteorder("<")


def _array_hash(value: Any, *, domain: str) -> str:
    """Hash exact numeric shape, precision, and little-endian C-order bytes."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError("Typed sensing product arrays cannot use object dtype")
    canonical_dtype = _canonical_dtype(array.dtype)
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(f"orchav.sensing.{domain}.array.v1\0".encode("ascii"))
    digest.update(
        dumps_frame_json({"shape": list(canonical.shape), "dtype": canonical_dtype.str}).encode()
    )
    digest.update(canonical.view(np.uint8).tobytes())
    return digest.hexdigest()


def _coordinate_hash(value: Any, coordinate_kind: str) -> str:
    return (
        _content_hash(value, domain="identity_axis")
        if coordinate_kind == "catalog"
        else _array_hash(value, domain="coordinate")
    )


def _semantic_product_record(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Remove container-local bindings before computing scientific identity."""

    semantic = deepcopy(
        {key: value for key, value in descriptor.items() if key not in {"product_id", "data_key"}}
    )
    for axis in semantic.get("axes", []):
        if not isinstance(axis, dict):
            continue
        axis.pop("coordinate_key", None)
        axis.pop("alternate_coordinate_keys", None)
        alternate_hashes = axis.pop("alternate_coordinate_sha256", None)
        if isinstance(alternate_hashes, Mapping):
            axis["alternate_coordinate_hashes"] = sorted(
                str(value) for value in alternate_hashes.values()
            )
    return semantic


def dense_axis(
    name: str,
    *,
    size: int,
    unit: str,
    coordinate_key: str,
    alternate_coordinate_keys: Sequence[str] = (),
    coordinate_semantics: str | None = None,
    reference_frame: str = "none",
) -> dict[str, Any]:
    """Describe one ordered array dimension with persisted coordinates."""

    return {
        "name": str(name),
        "size": int(size),
        "unit": str(unit),
        "coordinate_kind": "dense",
        "coordinate_key": str(coordinate_key),
        "alternate_coordinate_keys": [str(key) for key in alternate_coordinate_keys],
        "coordinate_semantics": str(coordinate_semantics or f"explicit_{name}_coordinates"),
        "reference_frame": str(reference_frame),
    }


def catalog_axis(
    name: str,
    *,
    size: int,
    catalog_key: str,
    coordinate_semantics: str | None = None,
    reference_frame: str = "none",
) -> dict[str, Any]:
    """Describe one ordered array dimension through a JSON catalog."""

    return {
        "name": str(name),
        "size": int(size),
        "unit": "1",
        "coordinate_kind": "catalog",
        "coordinate_key": str(catalog_key),
        "alternate_coordinate_keys": [],
        "coordinate_semantics": str(coordinate_semantics or f"ordered_{name}_identity"),
        "reference_frame": str(reference_frame),
    }


def build_product_descriptor(
    sensing: Mapping[str, Any],
    *,
    data_key: str,
    kind: str,
    variant: str = "primary",
    axes: Sequence[Mapping[str, Any]],
    representation: str,
    unit: str,
    scale_semantics: str,
    pair_ids: Sequence[str],
    evidence_role: str = "observation_derived",
    source_product_ids: Sequence[str] = (),
    temporal_support: str = TEMPORAL_SUPPORT_FULL_ACQUISITION,
    combine: Mapping[str, Any] | None = None,
    processing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a content-addressed descriptor for an existing numeric product.

    The product identifier hashes exact numeric content, scientific meaning,
    and occurrence lineage while excluding container-local data/coordinate
    binding keys.
    """

    if data_key not in sensing:
        raise ValueError(f"Cannot describe absent sensing product {data_key!r}")
    values = np.asarray(sensing[data_key])
    representation_errors = _representation_contract_errors(
        representation=representation,
        unit=unit,
        dtype=values.dtype,
        values=values,
    )
    if representation not in _VALUE_REPRESENTATIONS:
        representation_errors.insert(
            0,
            f"representation must be one of {sorted(_VALUE_REPRESENTATIONS)!r}",
        )
    if representation_errors:
        raise ValueError(
            f"Invalid sensing product {data_key!r} value representation: "
            + "; ".join(representation_errors)
        )
    if scale_semantics not in _SCALE_SEMANTICS:
        raise ValueError(
            "Typed sensing product scale_semantics must be exactly "
            f"one of {sorted(_SCALE_SEMANTICS)!r}"
        )
    temporal = sensing.get("temporal")
    clock = temporal.get("clock") if isinstance(temporal, Mapping) else None
    acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
    result = temporal.get("result") if isinstance(temporal, Mapping) else None
    if not isinstance(acquisition, Mapping):
        raise ValueError("Typed sensing products require temporal acquisition lineage")
    acquisition_id = acquisition.get("acquisition_id")
    if not isinstance(acquisition_id, str) or not acquisition_id:
        raise ValueError("Typed sensing products require a non-empty acquisition_id")
    clock_id = clock.get("clock_id") if isinstance(clock, Mapping) else None
    if not isinstance(clock_id, str) or not clock_id:
        raise ValueError("Typed sensing products require a non-empty clock_id")
    pair_id = acquisition.get("pair_id")
    tx_sensor_id = acquisition.get("tx_sensor_id")
    rx_sensor_id = acquisition.get("rx_sensor_id")
    if not all(isinstance(value, str) and value for value in (pair_id, tx_sensor_id, rx_sensor_id)):
        raise ValueError("Typed sensing products require exact acquisition pair identity")
    normalized_pair_ids = [str(value) for value in pair_ids]
    if normalized_pair_ids != [pair_id]:
        raise ValueError("Typed sensing product pair_ids must match temporal acquisition")

    if len(axes) != values.ndim:
        raise ValueError(
            f"Product {data_key!r} has rank {values.ndim}, but {len(axes)} axes were declared"
        )
    normalized_axes = [dict(axis) for axis in axes]
    for index, (axis, size) in enumerate(zip(normalized_axes, values.shape)):
        if int(axis.get("size", -1)) != int(size):
            raise ValueError(
                f"Product {data_key!r} axis {index} declares size {axis.get('size')!r}, "
                f"but the data size is {size}"
            )
        coordinate_kind = axis.get("coordinate_kind")
        coordinate_key = axis.get("coordinate_key")
        if coordinate_kind not in _COORDINATE_KINDS or not isinstance(coordinate_key, str):
            raise ValueError(f"Product {data_key!r} axis {index} has no typed coordinates")
        if coordinate_key not in sensing:
            raise ValueError(
                f"Product {data_key!r} axis {index} coordinates {coordinate_key!r} are absent"
            )
        axis["coordinate_sha256"] = _coordinate_hash(sensing[coordinate_key], str(coordinate_kind))
        alternate_hashes: dict[str, str] = {}
        for alternate_key in axis.get("alternate_coordinate_keys", []):
            if alternate_key not in sensing:
                raise ValueError(
                    f"Product {data_key!r} alternate coordinates {alternate_key!r} are absent"
                )
            alternate_hashes[str(alternate_key)] = _array_hash(
                sensing[alternate_key], domain="alternate_coordinate"
            )
        axis["alternate_coordinate_sha256"] = alternate_hashes

    resolved_temporal_support: dict[str, Any]
    if temporal_support == TEMPORAL_SUPPORT_FULL_ACQUISITION:
        resolved_temporal_support = {"type": TEMPORAL_SUPPORT_FULL_ACQUISITION}
        start_time_s = acquisition.get("start_time_s")
        end_time_s = acquisition.get("end_time_s")
        reference_time_s = (
            result.get("reference_time_s")
            if isinstance(result, Mapping)
            else acquisition.get("default_reference_time_s")
        )
        reference_time_policy = (
            result.get("reference_time_policy")
            if isinstance(result, Mapping)
            else acquisition.get("default_reference_policy")
        )
    elif temporal_support == TEMPORAL_SUPPORT_LAST_VALID_SAMPLE:
        samples = acquisition.get("samples")
        if not isinstance(samples, list):
            raise ValueError(
                "Last-valid-sample sensing products require ordered temporal acquisition slots"
            )
        valid_samples = [
            sample
            for sample in samples
            if isinstance(sample, Mapping) and sample.get("valid") is True
        ]
        if not valid_samples:
            raise ValueError(
                "Last-valid-sample sensing products require at least one valid acquisition slot"
            )
        selected_sample = valid_samples[-1]
        selected_index = selected_sample.get("sample_index")
        selected_time_s = selected_sample.get("sample_time_s")
        if (
            not isinstance(selected_index, int)
            or isinstance(selected_index, bool)
            or not _finite_number(selected_time_s)
        ):
            raise ValueError(
                "Last-valid-sample sensing products require an indexed finite sample epoch"
            )
        resolved_temporal_support = {
            "type": "acquisition_sample",
            "selection_policy": TEMPORAL_SUPPORT_LAST_VALID_SAMPLE,
            "sample_index": int(selected_index),
        }
        start_time_s = selected_time_s
        end_time_s = selected_time_s
        reference_time_s = selected_time_s
        reference_time_policy = ACQUISITION_SAMPLE_REFERENCE_POLICY
    else:
        raise ValueError(
            "Typed sensing product temporal_support must be "
            f"{TEMPORAL_SUPPORT_FULL_ACQUISITION!r} or "
            f"{TEMPORAL_SUPPORT_LAST_VALID_SAMPLE!r}"
        )
    if not all(_finite_number(value) for value in (start_time_s, end_time_s, reference_time_s)):
        raise ValueError("Typed sensing products require finite temporal support epochs")
    if not isinstance(reference_time_policy, str) or not reference_time_policy:
        raise ValueError("Typed sensing products require a reference_time_policy")
    descriptor: dict[str, Any] = {
        "contract_version": SENSING_PRODUCT_CONTRACT_VERSION,
        "data_key": str(data_key),
        "kind": str(kind),
        "variant": str(variant),
        "shape": [int(size) for size in values.shape],
        "dtype": _canonical_dtype(values.dtype).str,
        "data_sha256": _array_hash(values, domain="product_data"),
        "axes": normalized_axes,
        "representation": str(representation),
        "unit": str(unit),
        "scale_semantics": str(scale_semantics),
        "evidence_role": str(evidence_role),
        "pair_ids": normalized_pair_ids,
        "pair": {
            "pair_id": pair_id,
            "tx_sensor_id": tx_sensor_id,
            "rx_sensor_id": rx_sensor_id,
        },
        "temporal_support": resolved_temporal_support,
        "time_span": {
            "clock_id": clock_id,
            "start_time_s": float(start_time_s),
            "end_time_s": float(end_time_s),
            "reference_time_s": float(reference_time_s),
            "reference_time_policy": reference_time_policy,
        },
        "lineage": {
            "acquisition_id": acquisition_id,
            "source_product_ids": [str(product_id) for product_id in source_product_ids],
        },
    }
    if isinstance(result, Mapping):
        descriptor["lineage"].update(
            {
                "processing_run_id": result.get("processing_run_id"),
                "processing_recipe_id": result.get("processing_recipe_id"),
            }
        )
    if combine is not None:
        descriptor["combine"] = dict(combine)
    if processing is not None:
        descriptor["processing"] = dict(processing)
    descriptor["product_id"] = _content_hash(_semantic_product_record(descriptor), domain="product")
    return descriptor


def attach_product_descriptors(
    sensing: dict[str, Any],
    descriptors: Sequence[Mapping[str, Any]],
) -> None:
    """Install a validated descriptor mapping keyed by numeric ``data_key``."""

    products: dict[str, Any] = {}
    for raw_descriptor in descriptors:
        descriptor = dict(raw_descriptor)
        data_key = descriptor.get("data_key")
        if not isinstance(data_key, str) or not data_key:
            raise ValueError("Every sensing product descriptor requires a data_key")
        if data_key in products:
            raise ValueError(f"Duplicate sensing product descriptor for {data_key!r}")
        products[data_key] = descriptor
    sensing["product_contract_version"] = SENSING_PRODUCT_CONTRACT_VERSION
    sensing["products"] = products
    errors = validate_product_descriptors(sensing, require_described_data=True)
    if errors:
        raise ValueError("Invalid sensing product contract: " + "; ".join(errors))


def add_product_spec(
    sensing: dict[str, Any],
    *,
    data_key: str,
    kind: str,
    variant: str = "primary",
    axes: Sequence[Mapping[str, Any]],
    representation: str,
    unit: str,
    scale_semantics: str = "relative_unitless",
    evidence_role: str = "observation_derived",
    source_data_keys: Sequence[str] = (),
    source_product_ids: Sequence[str] = (),
    temporal_support: str = TEMPORAL_SUPPORT_FULL_ACQUISITION,
    combine: Mapping[str, Any] | None = None,
    processing: Mapping[str, Any] | None = None,
) -> None:
    """Queue one producer-authored descriptor until temporal IDs are bound."""

    specs = sensing.setdefault("_product_specs", [])
    if not isinstance(specs, list):
        raise ValueError("sensing._product_specs must be an internal list")
    specs.append(
        {
            "data_key": data_key,
            "kind": kind,
            "variant": variant,
            "axes": [dict(axis) for axis in axes],
            "representation": representation,
            "unit": unit,
            "scale_semantics": scale_semantics,
            "evidence_role": evidence_role,
            "source_data_keys": list(source_data_keys),
            "source_product_ids": list(source_product_ids),
            "temporal_support": temporal_support,
            "combine": dict(combine) if combine is not None else None,
            "processing": dict(processing) if processing is not None else None,
        }
    )


def materialize_product_specs(sensing: dict[str, Any]) -> None:
    """Bind queued product specs to pair/time IDs and content hashes.

    This is called after the temporal contract has assigned acquisition and
    processing occurrence identities.  It also creates the categorical pair
    coordinates and exact slow-time coordinate array used by producer-authored
    axis declarations.
    """

    temporal = sensing.get("temporal")
    acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
    if not isinstance(acquisition, Mapping):
        raise ValueError("Typed sensing products require temporal acquisition metadata")
    pair_id = acquisition.get("pair_id")
    tx_sensor_id = acquisition.get("tx_sensor_id")
    rx_sensor_id = acquisition.get("rx_sensor_id")
    if not all(isinstance(value, str) and value for value in (pair_id, tx_sensor_id, rx_sensor_id)):
        raise ValueError("Typed sensing products require exact pair and sensor identities")
    sensing["pair_axis"] = [
        {
            "pair_index": 0,
            "frame_pair_index": 0,
            "pair_id": pair_id,
            "tx_sensor_id": tx_sensor_id,
            "rx_sensor_id": rx_sensor_id,
            "tx_device_index": 0,
            "rx_device_index": 0,
        }
    ]
    raw_specs = sensing.pop("_product_specs", [])
    if not isinstance(raw_specs, list):
        raise ValueError("sensing._product_specs must be an internal list")
    existing_products = sensing.get("products")

    def _uses_slow_time_coordinates(spec: Mapping[str, Any]) -> bool:
        axes = spec.get("axes")
        if not isinstance(axes, list):
            return False
        return any(
            isinstance(axis, Mapping)
            and (
                axis.get("coordinate_key") == "slow_time_axis_s"
                or "slow_time_axis_s" in axis.get("alternate_coordinate_keys", ())
            )
            for axis in axes
        )

    described_specs = [spec for spec in raw_specs if isinstance(spec, Mapping)]
    if isinstance(existing_products, Mapping):
        described_specs.extend(
            descriptor
            for descriptor in existing_products.values()
            if isinstance(descriptor, Mapping)
        )
    samples = acquisition.get("samples")
    if isinstance(samples, list) and any(
        _uses_slow_time_coordinates(spec) for spec in described_specs
    ):
        sensing["slow_time_axis_s"] = np.asarray(
            [sample.get("sample_time_s") for sample in samples], dtype=np.float64
        )
    else:
        sensing.pop("slow_time_axis_s", None)

    described_replacements = {
        raw_spec.get("data_key") for raw_spec in raw_specs if isinstance(raw_spec, Mapping)
    }
    descriptors: list[dict[str, Any]] = []
    descriptor_by_key: dict[str, dict[str, Any]] = {}
    if isinstance(existing_products, Mapping):
        for data_key, raw_descriptor in existing_products.items():
            if (
                isinstance(data_key, str)
                and data_key in sensing
                and data_key not in described_replacements
                and isinstance(raw_descriptor, Mapping)
            ):
                descriptor = dict(raw_descriptor)
                descriptor_by_key[data_key] = descriptor
                descriptors.append(descriptor)
    active_specs: dict[str, Mapping[str, Any]] = {}
    active_spec_order: list[str] = []
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, Mapping):
            raise ValueError("Every queued sensing product spec must be a mapping")
        data_key = raw_spec.get("data_key")
        if not isinstance(data_key, str) or data_key not in sensing:
            # Result stripping and optional raw-payload policies may remove the
            # numeric value before temporal binding.  No stale descriptor is
            # allowed to survive for such an absent value.
            continue
        if data_key in active_specs:
            raise ValueError(f"Duplicate queued sensing product spec for {data_key!r}")
        source_keys = raw_spec.get("source_data_keys", [])
        if not isinstance(source_keys, list) or any(
            not isinstance(key, str) or not key for key in source_keys
        ):
            raise ValueError(
                f"Product spec {data_key!r} source_data_keys must be a list of " "non-empty strings"
            )
        active_specs[data_key] = raw_spec
        active_spec_order.append(data_key)

    missing_by_product: dict[str, list[str]] = {}
    for data_key in active_spec_order:
        source_keys = active_specs[data_key].get("source_data_keys", [])
        assert isinstance(source_keys, list)
        missing = [
            key for key in source_keys if key not in descriptor_by_key and key not in active_specs
        ]
        if missing:
            missing_by_product[data_key] = missing
    if missing_by_product:
        details = "; ".join(
            f"{data_key!r} -> {missing!r}" for data_key, missing in missing_by_product.items()
        )
        raise ValueError(
            "Queued sensing product specs reference missing source_data_keys: " + details
        )

    pending = list(active_spec_order)
    while pending:
        progressed = False
        for data_key in tuple(pending):
            raw_spec = active_specs[data_key]
            source_keys = raw_spec.get("source_data_keys", [])
            assert isinstance(source_keys, list)
            if any(key not in descriptor_by_key for key in source_keys):
                continue
            external_source_ids = raw_spec.get("source_product_ids", [])
            if not isinstance(external_source_ids, list):
                raise ValueError(f"Product spec {data_key!r} source_product_ids must be a list")
            descriptor = build_product_descriptor(
                sensing,
                data_key=data_key,
                kind=str(raw_spec.get("kind", "")),
                variant=str(raw_spec.get("variant", "")),
                axes=raw_spec.get("axes", []),
                representation=str(raw_spec.get("representation", "")),
                unit=str(raw_spec.get("unit", "")),
                scale_semantics=str(raw_spec.get("scale_semantics", "")),
                pair_ids=(pair_id,),
                evidence_role=str(raw_spec.get("evidence_role", "")),
                source_product_ids=tuple(
                    descriptor_by_key[key]["product_id"] for key in source_keys
                )
                + tuple(str(product_id) for product_id in external_source_ids),
                temporal_support=str(raw_spec.get("temporal_support", "")),
                combine=(
                    raw_spec.get("combine")
                    if isinstance(raw_spec.get("combine"), Mapping)
                    else None
                ),
                processing=(
                    raw_spec.get("processing")
                    if isinstance(raw_spec.get("processing"), Mapping)
                    else None
                ),
            )
            descriptor_by_key[data_key] = descriptor
            descriptors.append(descriptor)
            pending.remove(data_key)
            progressed = True
        if progressed:
            continue

        pending_set = set(pending)
        dependencies = {
            data_key: [
                key
                for key in active_specs[data_key].get("source_data_keys", [])
                if key in pending_set
            ]
            for data_key in pending
        }
        cycle: list[str] = []
        visiting: list[str] = []
        visited: set[str] = set()

        def find_cycle(data_key: str) -> bool:
            if data_key in visiting:
                start = visiting.index(data_key)
                cycle.extend((*visiting[start:], data_key))
                return True
            if data_key in visited:
                return False
            visiting.append(data_key)
            for source_key in dependencies[data_key]:
                if find_cycle(source_key):
                    return True
            visiting.pop()
            visited.add(data_key)
            return False

        for data_key in pending:
            if find_cycle(data_key):
                break
        cycle_text = " -> ".join(cycle) if cycle else ", ".join(pending)
        raise ValueError(
            "Queued sensing product source_data_keys contain a dependency cycle: " + cycle_text
        )
    attach_product_descriptors(sensing, descriptors)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def validate_product_descriptors(
    sensing: Mapping[str, Any],
    *,
    require_described_data: bool = False,
) -> list[str]:
    """Return typed-product violations without guessing absent projections.

    A selective HDF5 read may omit numeric data while retaining the complete
    descriptor table.  Such descriptors remain valid.  When a numeric product
    is present, however, its rank, shape, dtype, axes, and coordinates must
    exactly match the descriptor.
    """

    errors: list[str] = []
    version = sensing.get("product_contract_version")
    products = sensing.get("products")
    if version != SENSING_PRODUCT_CONTRACT_VERSION:
        return [
            "sensing.product_contract_version must equal " f"{SENSING_PRODUCT_CONTRACT_VERSION}"
        ]
    if not isinstance(products, Mapping):
        return ["sensing.products must be a mapping keyed by data_key"]

    seen_product_ids: set[str] = set()
    for mapping_key, raw_descriptor in products.items():
        path = f"sensing.products[{mapping_key!r}]"
        if not isinstance(mapping_key, str) or not mapping_key:
            errors.append("sensing.products keys must be non-empty strings")
            continue
        if not isinstance(raw_descriptor, Mapping):
            errors.append(f"{path} must be a mapping")
            continue
        descriptor = dict(raw_descriptor)
        unexpected_fields = sorted(
            set(descriptor) - _PRODUCT_DESCRIPTOR_FIELDS,
            key=repr,
        )
        if unexpected_fields:
            errors.append(f"{path} contains unexpected fields {unexpected_fields!r}")
        if descriptor.get("contract_version") != SENSING_PRODUCT_CONTRACT_VERSION:
            errors.append(f"{path}.contract_version must equal {SENSING_PRODUCT_CONTRACT_VERSION}")
        if descriptor.get("data_key") != mapping_key:
            errors.append(f"{path}.data_key must equal its mapping key")
        for field in (
            "kind",
            "variant",
            "representation",
            "unit",
            "evidence_role",
        ):
            if not isinstance(descriptor.get(field), str) or not descriptor[field]:
                errors.append(f"{path}.{field} must be a non-empty string")
        if descriptor.get("scale_semantics") not in _SCALE_SEMANTICS:
            errors.append(
                f"{path}.scale_semantics must be exactly one of " f"{sorted(_SCALE_SEMANTICS)!r}"
            )
        if descriptor.get("representation") not in _VALUE_REPRESENTATIONS:
            errors.append(
                f"{path}.representation must be one of {sorted(_VALUE_REPRESENTATIONS)!r}"
            )

        product_id = descriptor.get("product_id")
        if not isinstance(product_id, str) or _SHA256_RE.fullmatch(product_id) is None:
            errors.append(f"{path}.product_id must be a lowercase SHA-256 digest")
        else:
            expected = _content_hash(_semantic_product_record(descriptor), domain="product")
            if product_id != expected:
                errors.append(f"{path}.product_id does not match its semantic descriptor")
            if product_id in seen_product_ids:
                errors.append(f"{path}.product_id duplicates another descriptor")
            seen_product_ids.add(product_id)

        shape = descriptor.get("shape")
        if not isinstance(shape, list) or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape
        ):
            errors.append(f"{path}.shape must be a list of nonnegative integers")
            shape = []
        axes = descriptor.get("axes")
        if not isinstance(axes, list) or len(axes) != len(shape):
            errors.append(f"{path}.axes must contain exactly one entry per data dimension")
            axes = []
        elif not axes or not isinstance(axes[0], Mapping) or axes[0].get("name") != "pair":
            errors.append(f"{path}.axes[0] must be the explicit pair axis")
        elif not shape or shape[0] != 1:
            errors.append(f"{path} supports exactly one explicit pair in schema-v4")

        axis_names: set[str] = set()
        for axis_index, raw_axis in enumerate(axes):
            axis_path = f"{path}.axes[{axis_index}]"
            if not isinstance(raw_axis, Mapping):
                errors.append(f"{axis_path} must be a mapping")
                continue
            axis = raw_axis
            name = axis.get("name")
            if not isinstance(name, str) or not name:
                errors.append(f"{axis_path}.name must be a non-empty string")
            elif name in axis_names:
                errors.append(f"{axis_path}.name duplicates an earlier axis")
            else:
                axis_names.add(name)
            if axis.get("size") != shape[axis_index]:
                errors.append(f"{axis_path}.size must equal shape[{axis_index}]")
            if not isinstance(axis.get("unit"), str) or not axis["unit"]:
                errors.append(f"{axis_path}.unit must be a non-empty string")
            if (
                not isinstance(axis.get("coordinate_semantics"), str)
                or not axis["coordinate_semantics"]
            ):
                errors.append(f"{axis_path}.coordinate_semantics must be a non-empty string")
            if not isinstance(axis.get("reference_frame"), str) or not axis["reference_frame"]:
                errors.append(f"{axis_path}.reference_frame must be a non-empty string")
            coordinate_kind = axis.get("coordinate_kind")
            if coordinate_kind not in _COORDINATE_KINDS:
                errors.append(f"{axis_path}.coordinate_kind must be dense or catalog")
                continue
            coordinate_key = axis.get("coordinate_key")
            if not isinstance(coordinate_key, str) or not coordinate_key:
                errors.append(f"{axis_path}.coordinate_key must be a non-empty string")
                continue
            coordinates = sensing.get(coordinate_key)
            if coordinates is not None:
                coordinate_size = (
                    len(coordinates)
                    if coordinate_kind == "catalog" and isinstance(coordinates, list)
                    else np.asarray(coordinates).size
                )
                if coordinate_size != shape[axis_index]:
                    errors.append(
                        f"{axis_path}.coordinate_key {coordinate_key!r} has size "
                        f"{coordinate_size}, expected {shape[axis_index]}"
                    )
                coordinate_hash = axis.get("coordinate_sha256")
                if (
                    not isinstance(coordinate_hash, str)
                    or _SHA256_RE.fullmatch(coordinate_hash) is None
                ):
                    errors.append(f"{axis_path}.coordinate_sha256 must be a SHA-256 digest")
                elif coordinate_hash != _coordinate_hash(coordinates, str(coordinate_kind)):
                    errors.append(f"{axis_path}.coordinate_sha256 does not match coordinates")
            elif require_described_data or mapping_key in sensing:
                errors.append(
                    f"{axis_path}.coordinate_key {coordinate_key!r} is absent from the payload"
                )

            alternate = axis.get("alternate_coordinate_keys")
            if not isinstance(alternate, list) or any(
                not isinstance(key, str) or not key for key in alternate
            ):
                errors.append(f"{axis_path}.alternate_coordinate_keys must be a string list")
            else:
                alternate_hashes = axis.get("alternate_coordinate_sha256")
                if not isinstance(alternate_hashes, Mapping):
                    errors.append(f"{axis_path}.alternate_coordinate_sha256 must be a mapping")
                    alternate_hashes = {}
                for alternate_key in alternate:
                    alternate_coordinates = sensing.get(alternate_key)
                    if (
                        alternate_coordinates is not None
                        and np.asarray(alternate_coordinates).size != shape[axis_index]
                    ):
                        errors.append(
                            f"{axis_path} alternate coordinates {alternate_key!r} have "
                            "the wrong size"
                        )
                    elif alternate_coordinates is not None and alternate_hashes.get(
                        alternate_key
                    ) != _array_hash(alternate_coordinates, domain="alternate_coordinate"):
                        errors.append(
                            f"{axis_path} alternate-coordinate hash for {alternate_key!r} "
                            "does not match"
                        )

        values = sensing.get(mapping_key)
        if values is None:
            for error in _representation_contract_errors(
                representation=descriptor.get("representation"),
                unit=descriptor.get("unit"),
                dtype=descriptor.get("dtype"),
            ):
                errors.append(f"{path} {error}")
            if require_described_data:
                errors.append(f"{path} describes data absent from the payload")
        else:
            array = np.asarray(values)
            for error in _representation_contract_errors(
                representation=descriptor.get("representation"),
                unit=descriptor.get("unit"),
                dtype=array.dtype,
                values=array,
            ):
                errors.append(f"{path} {error}")
            if list(array.shape) != shape:
                errors.append(f"{path}.shape does not match the numeric payload")
            if descriptor.get("dtype") != _canonical_dtype(array.dtype).str:
                errors.append(f"{path}.dtype does not match the numeric payload")
            data_hash = descriptor.get("data_sha256")
            if not isinstance(data_hash, str) or _SHA256_RE.fullmatch(data_hash) is None:
                errors.append(f"{path}.data_sha256 must be a SHA-256 digest")
            elif data_hash != _array_hash(array, domain="product_data"):
                errors.append(f"{path}.data_sha256 does not match the numeric payload")

        pair_ids = descriptor.get("pair_ids")
        if (
            not isinstance(pair_ids, list)
            or not pair_ids
            or any(
                not isinstance(pair_id, str) or _SHA256_RE.fullmatch(pair_id) is None
                for pair_id in pair_ids
            )
        ):
            errors.append(f"{path}.pair_ids must contain canonical pair SHA-256 identities")
        acquisition = (
            sensing.get("temporal", {}).get("acquisition")
            if isinstance(sensing.get("temporal"), Mapping)
            else None
        )
        pair = descriptor.get("pair")
        if not isinstance(pair, Mapping):
            errors.append(f"{path}.pair must be a mapping")
        elif isinstance(acquisition, Mapping):
            for field in ("pair_id", "tx_sensor_id", "rx_sensor_id"):
                if pair.get(field) != acquisition.get(field):
                    errors.append(f"{path}.pair.{field} must match temporal acquisition")
        pair_axis = sensing.get("pair_axis")
        if isinstance(pair_axis, list) and isinstance(pair_ids, list):
            axis_pair_ids = [
                entry.get("pair_id") if isinstance(entry, Mapping) else None for entry in pair_axis
            ]
            if axis_pair_ids != pair_ids:
                errors.append(f"{path}.pair_ids must match ordered sensing.pair_axis")

        temporal = sensing.get("temporal")
        clock = temporal.get("clock") if isinstance(temporal, Mapping) else None
        result = temporal.get("result") if isinstance(temporal, Mapping) else None
        temporal_support = descriptor.get("temporal_support")
        expected_span: dict[str, Any] | None = None
        if not isinstance(temporal_support, Mapping):
            errors.append(f"{path}.temporal_support must be a mapping")
        else:
            support_type = temporal_support.get("type")
            if support_type == TEMPORAL_SUPPORT_FULL_ACQUISITION:
                if set(temporal_support) != {"type"}:
                    errors.append(f"{path}.temporal_support full_acquisition has unknown fields")
                if isinstance(acquisition, Mapping):
                    expected_span = {
                        "start_time_s": acquisition.get("start_time_s"),
                        "end_time_s": acquisition.get("end_time_s"),
                        "reference_time_s": (
                            result.get("reference_time_s")
                            if isinstance(result, Mapping)
                            else acquisition.get("default_reference_time_s")
                        ),
                        "reference_time_policy": (
                            result.get("reference_time_policy")
                            if isinstance(result, Mapping)
                            else acquisition.get("default_reference_policy")
                        ),
                    }
            elif support_type == "acquisition_sample":
                if set(temporal_support) != {"type", "selection_policy", "sample_index"}:
                    errors.append(
                        f"{path}.temporal_support acquisition_sample has unknown or missing fields"
                    )
                if temporal_support.get("selection_policy") != TEMPORAL_SUPPORT_LAST_VALID_SAMPLE:
                    errors.append(
                        f"{path}.temporal_support.selection_policy must equal "
                        f"{TEMPORAL_SUPPORT_LAST_VALID_SAMPLE!r}"
                    )
                selected_index = temporal_support.get("sample_index")
                if not isinstance(selected_index, int) or isinstance(selected_index, bool):
                    errors.append(f"{path}.temporal_support.sample_index must be an integer")
                samples = acquisition.get("samples") if isinstance(acquisition, Mapping) else None
                if not isinstance(samples, list):
                    errors.append(
                        f"{path}.temporal_support requires ordered temporal acquisition slots"
                    )
                elif isinstance(selected_index, int) and not isinstance(selected_index, bool):
                    valid_samples = [
                        sample
                        for sample in samples
                        if isinstance(sample, Mapping) and sample.get("valid") is True
                    ]
                    if not valid_samples:
                        errors.append(
                            f"{path}.temporal_support requires at least one valid acquisition slot"
                        )
                    else:
                        selected_sample = next(
                            (
                                sample
                                for sample in samples
                                if isinstance(sample, Mapping)
                                and sample.get("sample_index") == selected_index
                            ),
                            None,
                        )
                        last_valid_sample = valid_samples[-1]
                        if not isinstance(selected_sample, Mapping):
                            errors.append(
                                f"{path}.temporal_support.sample_index does not name an "
                                "acquisition slot"
                            )
                        elif selected_sample.get("valid") is not True:
                            errors.append(
                                f"{path}.temporal_support.sample_index must name a valid "
                                "acquisition slot"
                            )
                        if selected_index != last_valid_sample.get("sample_index"):
                            errors.append(
                                f"{path}.temporal_support.sample_index must name the last valid "
                                "acquisition slot"
                            )
                        selected_time_s = (
                            selected_sample.get("sample_time_s")
                            if isinstance(selected_sample, Mapping)
                            else None
                        )
                        if _finite_number(selected_time_s):
                            expected_span = {
                                "start_time_s": selected_time_s,
                                "end_time_s": selected_time_s,
                                "reference_time_s": selected_time_s,
                                "reference_time_policy": ACQUISITION_SAMPLE_REFERENCE_POLICY,
                            }
                        else:
                            errors.append(
                                f"{path}.temporal_support selected sample time must be finite"
                            )
            else:
                errors.append(
                    f"{path}.temporal_support.type must be "
                    f"{TEMPORAL_SUPPORT_FULL_ACQUISITION!r} or 'acquisition_sample'"
                )

        time_span = descriptor.get("time_span")
        if not isinstance(time_span, Mapping):
            errors.append(f"{path}.time_span must be a mapping")
        else:
            if not isinstance(time_span.get("clock_id"), str) or not time_span["clock_id"]:
                errors.append(f"{path}.time_span.clock_id must be non-empty")
            if (
                not isinstance(time_span.get("reference_time_policy"), str)
                or not time_span["reference_time_policy"]
            ):
                errors.append(f"{path}.time_span.reference_time_policy must be non-empty")
            for field in ("start_time_s", "end_time_s", "reference_time_s"):
                if not _finite_number(time_span.get(field)):
                    errors.append(f"{path}.time_span.{field} must be finite")
            if all(
                _finite_number(time_span.get(field))
                for field in ("start_time_s", "end_time_s", "reference_time_s")
            ):
                if float(time_span["start_time_s"]) > float(time_span["end_time_s"]):
                    errors.append(f"{path}.time_span start must not exceed end")
                if not (
                    float(time_span["start_time_s"])
                    <= float(time_span["reference_time_s"])
                    <= float(time_span["end_time_s"])
                ):
                    errors.append(f"{path}.time_span reference must lie within its support")
            if isinstance(clock, Mapping) and time_span.get("clock_id") != clock.get("clock_id"):
                errors.append(f"{path}.time_span.clock_id must match temporal clock")
            if expected_span is not None:
                for field, expected_value in expected_span.items():
                    if time_span.get(field) != expected_value:
                        support_label = (
                            temporal_support.get("type")
                            if isinstance(temporal_support, Mapping)
                            else "declared support"
                        )
                        errors.append(
                            f"{path}.time_span.{field} must match {support_label} temporal support"
                        )
            if (
                isinstance(temporal_support, Mapping)
                and temporal_support.get("type") == TEMPORAL_SUPPORT_FULL_ACQUISITION
                and isinstance(result, Mapping)
            ):
                if time_span.get("reference_time_s") != result.get("reference_time_s"):
                    errors.append(f"{path}.time_span.reference_time_s must match temporal result")
                if time_span.get("reference_time_policy") != result.get("reference_time_policy"):
                    errors.append(
                        f"{path}.time_span.reference_time_policy must match temporal result"
                    )

        lineage = descriptor.get("lineage")
        if not isinstance(lineage, Mapping):
            errors.append(f"{path}.lineage must be a mapping")
        else:
            if not isinstance(lineage.get("acquisition_id"), str) or not lineage["acquisition_id"]:
                errors.append(f"{path}.lineage.acquisition_id must be non-empty")
            parents = lineage.get("source_product_ids")
            if not isinstance(parents, list) or any(
                not isinstance(parent, str) or _SHA256_RE.fullmatch(parent) is None
                for parent in parents
            ):
                errors.append(f"{path}.lineage.source_product_ids must be SHA-256 digests")
            elif product_id in parents:
                errors.append(f"{path} cannot list itself as a source product")
            temporal = sensing.get("temporal")
            acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
            result = temporal.get("result") if isinstance(temporal, Mapping) else None
            if isinstance(acquisition, Mapping) and lineage.get(
                "acquisition_id"
            ) != acquisition.get("acquisition_id"):
                errors.append(f"{path}.lineage.acquisition_id must match temporal acquisition")
            if isinstance(result, Mapping):
                for field in ("processing_run_id", "processing_recipe_id"):
                    if lineage.get(field) != result.get(field):
                        errors.append(f"{path}.lineage.{field} must match temporal result")

    product_graph = {
        descriptor.get("product_id"): tuple(
            descriptor.get("lineage", {}).get("source_product_ids", ())
        )
        for descriptor in products.values()
        if isinstance(descriptor, Mapping)
        and isinstance(descriptor.get("product_id"), str)
        and isinstance(descriptor.get("lineage"), Mapping)
        and isinstance(descriptor.get("lineage", {}).get("source_product_ids"), list)
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(product: str) -> bool:
        if product in visiting:
            return True
        if product in visited:
            return False
        visiting.add(product)
        cycle = any(parent in product_graph and visit(parent) for parent in product_graph[product])
        visiting.remove(product)
        visited.add(product)
        return cycle

    if any(visit(product) for product in product_graph if product not in visited):
        errors.append("sensing.products source_product_ids contain a lineage cycle")

    return errors


def product_with_coordinates(
    sensing: Mapping[str, Any],
    data_key: str,
) -> tuple[np.ndarray, tuple[np.ndarray | list[Any], ...], Mapping[str, Any]]:
    """Return a product and its declared coordinates without shape heuristics."""

    products = sensing.get("products")
    descriptor = products.get(data_key) if isinstance(products, Mapping) else None
    if not isinstance(descriptor, Mapping):
        raise KeyError(f"No typed sensing product descriptor for {data_key!r}")
    if data_key not in sensing:
        raise KeyError(f"Typed sensing product {data_key!r} is not loaded")
    coordinates: list[np.ndarray | list[Any]] = []
    for axis in descriptor.get("axes", []):
        coordinate_key = axis.get("coordinate_key") if isinstance(axis, Mapping) else None
        if not isinstance(coordinate_key, str) or coordinate_key not in sensing:
            raise KeyError(
                f"Typed sensing product {data_key!r} is missing coordinates {coordinate_key!r}"
            )
        raw = sensing[coordinate_key]
        coordinates.append(raw if isinstance(raw, list) else np.asarray(raw))
    return np.asarray(sensing[data_key]), tuple(coordinates), descriptor


def select_product_by_contract(
    sensing: Mapping[str, Any],
    *,
    kind: str,
    variant: str = "primary",
    axis_names: Sequence[str],
    representations: Sequence[str],
) -> tuple[
    str,
    np.ndarray,
    tuple[np.ndarray | list[Any], ...],
    Mapping[str, Any],
]:
    """Select exactly one loaded product by scientific contract.

    Storage keys are deliberately excluded from the selection criteria.  A
    consumer must state the product kind, variant, ordered axes, and accepted
    scientific representations.  Missing and ambiguous matches fail visibly
    instead of falling back to dtype, shape, or key-name heuristics.
    """

    expected_axes = tuple(str(name) for name in axis_names)
    accepted_representations = frozenset(str(value) for value in representations)
    if not kind or not variant or not expected_axes or not accepted_representations:
        raise ValueError(
            "Typed sensing-product selection requires kind, variant, axes, "
            "and at least one representation"
        )

    products = sensing.get("products")
    if not isinstance(products, Mapping):
        raise KeyError("Sensing payload has no typed product descriptor mapping")

    matches: list[str] = []
    for data_key, raw_descriptor in products.items():
        if not isinstance(data_key, str) or not isinstance(raw_descriptor, Mapping):
            continue
        descriptor_axes = raw_descriptor.get("axes")
        names = (
            tuple(
                axis.get("name") if isinstance(axis, Mapping) else None for axis in descriptor_axes
            )
            if isinstance(descriptor_axes, list)
            else ()
        )
        if (
            raw_descriptor.get("kind") == kind
            and raw_descriptor.get("variant") == variant
            and names == expected_axes
            and raw_descriptor.get("representation") in accepted_representations
            and data_key in sensing
        ):
            matches.append(data_key)

    contract = (
        f"kind={kind!r}, variant={variant!r}, axes={expected_axes!r}, "
        f"representations={sorted(accepted_representations)!r}"
    )
    if not matches:
        raise KeyError(f"No loaded sensing product matches {contract}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous sensing products match {contract}: {sorted(matches)!r}")
    data_key = matches[0]
    values, coordinates, descriptor = product_with_coordinates(sensing, data_key)
    return data_key, values, coordinates, descriptor


__all__ = [
    "ACQUISITION_SAMPLE_REFERENCE_POLICY",
    "SENSING_PRODUCT_CONTRACT_VERSION",
    "TEMPORAL_SUPPORT_FULL_ACQUISITION",
    "TEMPORAL_SUPPORT_LAST_VALID_SAMPLE",
    "add_product_spec",
    "attach_product_descriptors",
    "build_product_descriptor",
    "catalog_axis",
    "dense_axis",
    "materialize_product_specs",
    "product_with_coordinates",
    "select_product_by_contract",
    "validate_product_descriptors",
]
