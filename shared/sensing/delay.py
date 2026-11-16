"""Strict schema-v4 selection of persisted delay-domain sensing products.

Persisted consumers use this module instead of compatibility arrays such as
``cir`` and ``slow_time_matrix``.  The selector keeps producer-authored time
and range coordinates intact and applies only the combine policy declared by
the typed product descriptor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from shared.sensing.products import select_product_by_contract

_CIR_KIND = "delay_domain_cir"
_SLOW_TIME_KIND = "delay_domain_slow_time"
_COMPLEX_VOLTAGE = "complex_voltage"


@dataclass(frozen=True)
class DelayDomainSamples:
    """One validated delay-domain product reduced to ``[time, range]``."""

    data_key: str
    kind: str
    samples: np.ndarray
    sample_times_s: np.ndarray
    range_axis_m: np.ndarray
    pair_id: str
    rx_element_identities: tuple[tuple[int, int, int], ...]
    combine_policy: str


def _is_finite_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _descriptors_for_kind(sensing: Mapping[str, Any], kind: str) -> list[Mapping[str, Any]]:
    products = sensing.get("products")
    if not isinstance(products, Mapping):
        return []
    return [
        descriptor
        for descriptor in products.values()
        if isinstance(descriptor, Mapping) and descriptor.get("kind") == kind
    ]


def _select_delay_product(
    sensing: Mapping[str, Any],
) -> (
    tuple[
        str,
        str,
        np.ndarray,
        tuple[np.ndarray | list[Any], ...],
        Mapping[str, Any],
    ]
    | None
):
    """Prefer an authored slow-time product, then a single-epoch CIR product."""

    slow_descriptors = _descriptors_for_kind(sensing, _SLOW_TIME_KIND)
    if slow_descriptors:
        try:
            data_key, values, coordinates, descriptor = select_product_by_contract(
                sensing,
                kind=_SLOW_TIME_KIND,
                variant="primary",
                axis_names=("pair", "rx_element", "slow_time", "range"),
                representations=(_COMPLEX_VOLTAGE,),
            )
        except KeyError as exc:
            raise ValueError(
                "Persisted delay_domain_slow_time exists but does not satisfy the "
                "primary pair/rx_element/slow_time/range complex_voltage contract"
            ) from exc
        return data_key, _SLOW_TIME_KIND, values, coordinates, descriptor

    cir_descriptors = _descriptors_for_kind(sensing, _CIR_KIND)
    if cir_descriptors:
        try:
            data_key, values, coordinates, descriptor = select_product_by_contract(
                sensing,
                kind=_CIR_KIND,
                variant="primary",
                axis_names=("pair", "rx_element", "range"),
                representations=(_COMPLEX_VOLTAGE,),
            )
        except KeyError as exc:
            raise ValueError(
                "Persisted delay_domain_cir exists but does not satisfy the primary "
                "pair/rx_element/range complex_voltage contract"
            ) from exc
        return data_key, _CIR_KIND, values, coordinates, descriptor

    return None


def _validate_axis_contract(
    descriptor: Mapping[str, Any],
    *,
    kind: str,
) -> None:
    expected = (
        (
            ("pair", "1", "catalog"),
            ("rx_element", "1", "catalog"),
            ("slow_time", "s", "dense"),
            ("range", "m", "dense"),
        )
        if kind == _SLOW_TIME_KIND
        else (
            ("pair", "1", "catalog"),
            ("rx_element", "1", "catalog"),
            ("range", "m", "dense"),
        )
    )
    axes = descriptor.get("axes")
    if not isinstance(axes, list) or len(axes) != len(expected):
        raise ValueError(f"Typed {kind} has an invalid ordered axis contract")
    observed = tuple(
        (
            (
                axis.get("name"),
                axis.get("unit"),
                axis.get("coordinate_kind"),
            )
            if isinstance(axis, Mapping)
            else (None, None, None)
        )
        for axis in axes
    )
    if observed != expected:
        raise ValueError(f"Typed {kind} axes must be exactly {expected!r}; received {observed!r}")


def _validate_pair_identity(
    sensing: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    pair_coordinates: np.ndarray | list[Any],
) -> str:
    if not isinstance(pair_coordinates, list) or len(pair_coordinates) != 1:
        raise ValueError("Delay-domain products require exactly one explicit pair axis row")
    pair_row = pair_coordinates[0]
    if not isinstance(pair_row, Mapping):
        raise ValueError("Delay-domain pair coordinates must be identity mappings")
    if pair_row.get("pair_index") != 0 or pair_row.get("frame_pair_index") != 0:
        raise ValueError("Delay-domain pair coordinates require exact pair index zero")

    pair_id = pair_row.get("pair_id")
    tx_sensor_id = pair_row.get("tx_sensor_id")
    rx_sensor_id = pair_row.get("rx_sensor_id")
    if not all(isinstance(value, str) and value for value in (pair_id, tx_sensor_id, rx_sensor_id)):
        raise ValueError("Delay-domain pair coordinates lack exact sensor identities")
    if descriptor.get("pair_ids") != [pair_id]:
        raise ValueError("Delay-domain descriptor pair_ids do not match its pair axis")
    descriptor_pair = descriptor.get("pair")
    expected_pair = {
        "pair_id": pair_id,
        "tx_sensor_id": tx_sensor_id,
        "rx_sensor_id": rx_sensor_id,
    }
    if not isinstance(descriptor_pair, Mapping) or any(
        descriptor_pair.get(key) != value for key, value in expected_pair.items()
    ):
        raise ValueError("Delay-domain descriptor pair identity is inconsistent")

    temporal = sensing.get("temporal")
    acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
    if not isinstance(acquisition, Mapping) or any(
        acquisition.get(key) != value for key, value in expected_pair.items()
    ):
        raise ValueError("Delay-domain pair identity does not match temporal acquisition")
    return pair_id


def _validate_element_identities(
    element_coordinates: np.ndarray | list[Any], element_count: int
) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(element_coordinates, list) or len(element_coordinates) != element_count:
        raise ValueError(
            "Delay-domain rx_element coordinates must contain one row per array element"
        )
    identities: list[tuple[int, int, int]] = []
    for expected_index, coordinate in enumerate(element_coordinates):
        if not isinstance(coordinate, Mapping):
            raise ValueError("Delay-domain rx_element coordinates must be mappings")
        element_index = coordinate.get("element_index")
        row = coordinate.get("row")
        column = coordinate.get("column")
        if (
            element_index != expected_index
            or not isinstance(row, (int, np.integer))
            or isinstance(row, (bool, np.bool_))
            or not isinstance(column, (int, np.integer))
            or isinstance(column, (bool, np.bool_))
            or int(row) < 0
            or int(column) < 0
        ):
            raise ValueError(
                "Delay-domain rx_element coordinates require ordered nonnegative "
                "element_index/row/column identities"
            )
        identities.append((expected_index, int(row), int(column)))
    if len({(row, column) for _, row, column in identities}) != element_count:
        raise ValueError("Delay-domain rx_element row/column identities must be unique")
    return tuple(identities)


def _validate_dense_coordinates(
    coordinates: np.ndarray | list[Any],
    *,
    expected_size: int,
    label: str,
) -> np.ndarray:
    values = np.asarray(coordinates)
    if (
        values.ndim != 1
        or values.size != expected_size
        or values.dtype.kind not in "fiu"
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            f"Delay-domain {label} coordinates must be a finite real vector of "
            f"length {expected_size}"
        )
    if values.size > 1 and not np.all(np.diff(values) > 0):
        raise ValueError(f"Delay-domain {label} coordinates must be strictly increasing")
    return np.array(values, copy=True)


def _validate_slow_time_membership(sensing: Mapping[str, Any], sample_times_s: np.ndarray) -> None:
    temporal = sensing.get("temporal")
    acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
    samples = acquisition.get("samples") if isinstance(acquisition, Mapping) else None
    if not isinstance(samples, list) or len(samples) != len(sample_times_s):
        raise ValueError(
            "Delay-domain slow-time coordinates do not match ordered temporal membership"
        )
    authored_times: list[float] = []
    for sample in samples:
        if not isinstance(sample, Mapping) or sample.get("valid") is not True:
            raise ValueError(
                "A persisted delay-domain slow-time row must have valid temporal membership"
            )
        sample_time_s = sample.get("sample_time_s")
        if not _is_finite_real(sample_time_s):
            raise ValueError("Delay-domain temporal membership has a non-finite sample time")
        authored_times.append(float(sample_time_s))
    if not np.array_equal(np.asarray(authored_times, dtype=np.float64), sample_times_s):
        raise ValueError(
            "Delay-domain slow-time coordinates differ from temporal acquisition samples"
        )


def _combine_rx_elements(
    values_by_element: np.ndarray,
    descriptor: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    combine = descriptor.get("combine")
    if not isinstance(combine, Mapping):
        raise ValueError(
            "Persisted delay-domain products require an explicit named rx_element combine"
        )
    policy = combine.get("policy")
    normalization = combine.get("normalization")
    input_axis = combine.get("input_axis")
    element_count = int(values_by_element.shape[0])
    if element_count < 1:
        raise ValueError("Delay-domain combine requires at least one rx_element")
    if policy == "siso_identity":
        if normalization != "identity" or input_axis not in (None, "rx_element"):
            raise ValueError("siso_identity requires identity normalization on rx_element")
        if element_count != 1:
            raise ValueError("siso_identity requires exactly one rx_element")
        return np.array(values_by_element[0], copy=True), str(policy)
    if policy == "rx_broadside_unitary_voltage_sum":
        if input_axis != "rx_element" or normalization != "one_over_sqrt_element_count":
            raise ValueError(
                "rx_broadside_unitary_voltage_sum requires the declared unitary "
                "rx_element normalization"
            )
        return (
            np.sum(values_by_element, axis=0) / np.sqrt(float(element_count)),
            str(policy),
        )
    raise ValueError(f"Unsupported delay-domain combine policy {policy!r}")


def extract_delay_domain_samples(
    sensing: Mapping[str, Any],
) -> DelayDomainSamples | None:
    """Select and validate one canonical persisted delay-domain product.

    A primary slow-time product is preferred because it carries every exact
    authored sample coordinate.  If none exists, a primary CIR product is
    bound to its descriptor reference epoch.  Legacy compatibility arrays are
    never consulted.
    """

    temporal = sensing.get("temporal")
    if not isinstance(temporal, Mapping) or not isinstance(temporal.get("result"), Mapping):
        return None
    selected = _select_delay_product(sensing)
    if selected is None:
        return None
    data_key, kind, values, coordinates, descriptor = selected
    _validate_axis_contract(descriptor, kind=kind)
    expected_rank = 4 if kind == _SLOW_TIME_KIND else 3
    if values.ndim != expected_rank or values.shape[0] != 1:
        raise ValueError(
            f"Typed {kind} must use exactly one pair and rank {expected_rank}; "
            f"received shape {values.shape}"
        )
    if values.dtype.kind != "c" or not np.all(np.isfinite(values)):
        raise ValueError(f"Typed {kind} values must be finite complex voltage samples")

    pair_id = _validate_pair_identity(sensing, descriptor, coordinates[0])
    element_identities = _validate_element_identities(coordinates[1], int(values.shape[1]))
    range_axis_m = _validate_dense_coordinates(
        coordinates[-1], expected_size=int(values.shape[-1]), label="range"
    )

    combined, combine_policy = _combine_rx_elements(values[0], descriptor)
    if kind == _SLOW_TIME_KIND:
        sample_times_s = _validate_dense_coordinates(
            coordinates[2], expected_size=int(values.shape[2]), label="slow-time"
        )
        _validate_slow_time_membership(sensing, sample_times_s)
        samples = combined
    else:
        time_span = descriptor.get("time_span")
        reference_time_s = (
            time_span.get("reference_time_s") if isinstance(time_span, Mapping) else None
        )
        if not _is_finite_real(reference_time_s):
            raise ValueError("Typed delay_domain_cir has no finite descriptor reference epoch")
        result = temporal["result"]
        if result.get("reference_time_s") != reference_time_s:
            raise ValueError("Typed delay_domain_cir reference epoch differs from temporal result")
        sample_times_s = np.asarray([float(reference_time_s)], dtype=np.float64)
        samples = combined[np.newaxis, :]

    if samples.ndim != 2 or samples.shape != (len(sample_times_s), len(range_axis_m)):
        raise ValueError("Combined delay-domain samples must use exact [time, range] axes")
    return DelayDomainSamples(
        data_key=data_key,
        kind=kind,
        samples=np.array(samples, copy=True),
        sample_times_s=sample_times_s,
        range_axis_m=range_axis_m,
        pair_id=pair_id,
        rx_element_identities=element_identities,
        combine_policy=combine_policy,
    )


def require_compatible_delay_sequence(
    reference: DelayDomainSamples,
    current: DelayDomainSamples,
) -> None:
    """Reject changes to pair, element, combine, or exact range identity."""

    if reference.pair_id != current.pair_id:
        raise ValueError("Delay-domain sequence mixes sensor pair identities")
    if reference.rx_element_identities != current.rx_element_identities:
        raise ValueError("Delay-domain sequence changes rx_element identities")
    if reference.combine_policy != current.combine_policy:
        raise ValueError("Delay-domain sequence changes rx_element combine policy")
    if not np.array_equal(reference.range_axis_m, current.range_axis_m):
        raise ValueError("Delay-domain sequence changes exact range coordinates")


def append_delay_domain_samples(
    product: DelayDomainSamples,
    samples: list[np.ndarray],
    sample_times_s: list[float],
) -> int:
    """Append new exact epochs and de-duplicate identical overlapping windows."""

    if len(samples) != len(sample_times_s):
        raise ValueError("Delay-domain samples and timestamps lost alignment")
    appended = 0
    for sample, raw_time_s in zip(product.samples, product.sample_times_s, strict=True):
        time_s = float(raw_time_s)
        duplicate_indices = [
            index
            for index, existing_time_s in enumerate(sample_times_s)
            if existing_time_s == time_s
        ]
        if duplicate_indices:
            existing = np.asarray(samples[duplicate_indices[0]])
            if existing.shape != sample.shape or not np.array_equal(existing, sample):
                raise ValueError(
                    f"Delay-domain sample epoch {time_s} has conflicting persisted values"
                )
            continue
        if sample_times_s and time_s <= sample_times_s[-1]:
            raise ValueError(
                "Delay-domain sample epochs must arrive in increasing coordinate order"
            )
        samples.append(np.array(sample, copy=True))
        sample_times_s.append(time_s)
        appended += 1
    return appended


__all__ = [
    "DelayDomainSamples",
    "append_delay_domain_samples",
    "extract_delay_domain_samples",
    "require_compatible_delay_sequence",
]
