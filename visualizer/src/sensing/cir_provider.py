"""Read stored CIR data from visual frame payloads."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

from shared.extensions.sensing import (
    assert_sensing_contract,
    assert_supported_sensing_schema,
)
from shared.sensing.delay import DelayDomainSamples, extract_delay_domain_samples
from shared.sensing.signal_processing import C0


@dataclass
class CIRComputationResult:
    """Container for computed CIR/CFR data."""

    cir: np.ndarray  # complex64/complex128 [P]
    delays_s: np.ndarray  # float64/float32 [P]
    pair_indices: Optional[np.ndarray] = None  # int32 [P]
    metadata: Optional[dict[str, Any]] = None
    sample_time_s: Optional[float] = None


@dataclass(frozen=True)
class TimestampedFastTimeBatch:
    """Fast-time rows aligned one-to-one with persisted acquisition times."""

    samples: tuple[np.ndarray, ...]
    sample_times_s: tuple[float, ...]
    acquisition_id: str | None
    strategy: str
    expanded_acquisition: bool = False
    alignment_error: str | None = None
    range_axis_m: np.ndarray | None = None
    range_coordinate_semantics: str | None = None
    data_key: str | None = None
    product_id: str | None = None
    pair_id: str | None = None
    rx_element_identities: tuple[tuple[int, int, int], ...] = ()
    combine_policy: str | None = None


def _acquisition_membership(sensing_block: Mapping[str, Any]) -> tuple[list[Any], str | None]:
    temporal = sensing_block.get("temporal")
    acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
    if not isinstance(acquisition, Mapping):
        return [], None
    samples = acquisition.get("samples")
    return (
        list(samples) if isinstance(samples, (list, tuple)) else [],
        (
            str(acquisition.get("acquisition_id"))
            if acquisition.get("acquisition_id") is not None
            else None
        ),
    )


def _aligned_complete_times(
    membership: list[Any],
    row_count: int,
    *,
    require_phase_coherent: bool = False,
) -> tuple[float, ...] | None:
    """Return row-aligned times only for complete, ordered membership."""
    if row_count < 1 or len(membership) != row_count:
        return None
    times: list[float] = []
    for expected_index, sample in enumerate(membership):
        if not isinstance(sample, Mapping):
            return None
        value = sample.get("sample_time_s")
        if (
            sample.get("valid") is not True
            or sample.get("sample_index") != expected_index
            or (require_phase_coherent and sample.get("phase_coherent") is not True)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            return None
        sample_time_s = float(value)
        if not math.isfinite(sample_time_s) or (times and sample_time_s <= times[-1]):
            return None
        times.append(sample_time_s)
    return tuple(times)


def _product_batch_fields(
    sensing_block: Mapping[str, Any],
    product: DelayDomainSamples,
) -> dict[str, Any]:
    """Return exact descriptor and coordinate provenance for one shared selection."""
    products = sensing_block.get("products")
    descriptor = products.get(product.data_key) if isinstance(products, Mapping) else None
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"Typed sensing product {product.data_key!r} lost its descriptor")
    axes = descriptor.get("axes")
    range_axis = axes[-1] if isinstance(axes, list) and axes else None
    range_semantics = (
        range_axis.get("coordinate_semantics") if isinstance(range_axis, Mapping) else None
    )
    if not isinstance(range_semantics, str) or not range_semantics:
        raise ValueError(
            f"Typed sensing product {product.data_key!r} has no range coordinate semantics"
        )
    product_id = descriptor.get("product_id")
    if not isinstance(product_id, str) or not product_id:
        raise ValueError(f"Typed sensing product {product.data_key!r} has no product_id")
    return {
        "range_axis_m": product.range_axis_m,
        "range_coordinate_semantics": range_semantics,
        "data_key": product.data_key,
        "product_id": product_id,
        "pair_id": product.pair_id,
        "rx_element_identities": product.rx_element_identities,
        "combine_policy": product.combine_policy,
    }


def _requires_complete_acquisition_expansion(
    *,
    strategy: str,
    resolved: Mapping[str, Any] | None,
    row_count: int,
) -> bool:
    """Return whether all aligned rows must be exposed for acquisition merging.

    Sequential STFT frames persist overlapping sliding windows, so their
    service-level history should still advance by one latest row per frame.
    Coherent, pre-grouped measurement, and derived products can retain a
    complete acquisition under a different current processing strategy; those
    rows must be expanded from the persisted acquisition contract instead of
    being classified by the result strategy alone.
    """
    if row_count <= 1:
        return False
    if strategy == "coherent_cpi_fft":
        return True
    if not isinstance(resolved, Mapping):
        return False

    processing = resolved.get("processing")
    processing_mode = processing.get("mode") if isinstance(processing, Mapping) else None
    source_processing = resolved.get("source_processing")
    source_mode = source_processing.get("mode") if isinstance(source_processing, Mapping) else None
    if processing_mode == "sequential_stft" or source_mode == "sequential_stft":
        return False

    acquisition = resolved.get("acquisition")
    if not isinstance(acquisition, Mapping):
        return processing_mode == "measurement_import" or source_mode == "measurement_import"
    if acquisition.get("sample_source") == "coherent_one_solve":
        return True
    if processing_mode == "measurement_import" or source_mode == "measurement_import":
        return True

    hop = acquisition.get("cpi_hop_samples")
    return isinstance(hop, int) and not isinstance(hop, bool) and hop > 1


def extract_timestamped_fast_time_samples(
    sensing_block: Mapping[str, Any],
) -> TimestampedFastTimeBatch:
    """Extract typed CIR rows with authoritative product and temporal coordinates."""
    resolved = sensing_block.get("config_resolved")
    processing = resolved.get("processing") if isinstance(resolved, Mapping) else None
    resolved_strategy = processing.get("strategy") if isinstance(processing, Mapping) else None
    if not isinstance(resolved_strategy, str) or not resolved_strategy.strip():
        raise ValueError("Sensing CIR data requires config_resolved.processing.strategy")
    strategy = resolved_strategy.strip()
    strategy_mirror = sensing_block.get("processing_strategy")
    if strategy_mirror is not None and (
        not isinstance(strategy_mirror, str) or strategy_mirror.strip() != strategy
    ):
        raise ValueError(
            "Sensing processing_strategy disagrees with config_resolved.processing.strategy"
        )
    membership, acquisition_id = _acquisition_membership(sensing_block)
    product = extract_delay_domain_samples(sensing_block)
    if product is None and (
        sensing_block.get("slow_time_matrix") is not None or sensing_block.get("cir") is not None
    ):
        raise ValueError(
            "Visualizer fast-time consumption requires schema-v4 typed CIR/slow-time "
            "products; legacy cir and slow_time_matrix arrays are unsupported"
        )
    if product is None:
        return TimestampedFastTimeBatch(
            samples=(),
            sample_times_s=(),
            acquisition_id=acquisition_id,
            strategy=strategy,
            alignment_error="no typed delay-domain sensing product is available",
        )

    fields = _product_batch_fields(sensing_block, product)
    product_times = tuple(float(value) for value in product.sample_times_s)
    if product.kind == "delay_domain_cir":
        return TimestampedFastTimeBatch(
            samples=(np.asarray(product.samples[0], dtype=np.complex128),),
            sample_times_s=product_times,
            acquisition_id=acquisition_id,
            strategy=strategy,
            **fields,
        )

    expand_acquisition = _requires_complete_acquisition_expansion(
        strategy=strategy,
        resolved=resolved if isinstance(resolved, Mapping) else None,
        row_count=product.samples.shape[0],
    )
    acquisition = resolved.get("acquisition") if isinstance(resolved, Mapping) else None
    require_phase_coherent = strategy == "coherent_cpi_fft" or (
        isinstance(acquisition, Mapping)
        and acquisition.get("sample_source") == "coherent_one_solve"
    )
    aligned_times = _aligned_complete_times(
        membership,
        product.samples.shape[0],
        require_phase_coherent=require_phase_coherent,
    )
    if aligned_times is None or aligned_times != product_times:
        return TimestampedFastTimeBatch(
            samples=(),
            sample_times_s=(),
            acquisition_id=acquisition_id,
            strategy=strategy,
            alignment_error=(
                f"typed product {product.data_key!r} coordinates do not align exactly "
                "with complete acquisition membership"
            ),
            **fields,
        )
    if expand_acquisition:
        return TimestampedFastTimeBatch(
            samples=tuple(np.asarray(row, dtype=np.complex128) for row in product.samples),
            sample_times_s=aligned_times,
            acquisition_id=acquisition_id,
            strategy=strategy,
            expanded_acquisition=True,
            **fields,
        )

    # Sequential windows overlap; only the latest exact slow-time row is new.
    return TimestampedFastTimeBatch(
        samples=(np.asarray(product.samples[-1], dtype=np.complex128),),
        sample_times_s=(aligned_times[-1],),
        acquisition_id=acquisition_id,
        strategy=strategy,
        **fields,
    )


class CIRProvider(Protocol):
    """Protocol for on-demand CIR computation from raw frame data."""

    def compute(
        self,
        raw_frame: dict[str, Any],
        *,
        sampling_frequency: float,
        normalize_delays: bool = False,
    ) -> Optional[CIRComputationResult]:
        """Return CIR samples for *raw_frame* or ``None`` when unsupported."""


class StoredCIRProvider:
    """
    CIR provider that consumes exact schema-v4 typed fast-time products.

    Product selection is descriptor-based. Pair/element identities, range
    coordinates, representation, and variant must all be explicit.
    """

    def compute(
        self,
        raw_frame: dict[str, Any],
        *,
        sampling_frequency: float,
        normalize_delays: bool = False,
    ) -> Optional[CIRComputationResult]:
        """Use stored fast-time sensing samples from a frame's ``sensing`` block."""
        assert_supported_sensing_schema(raw_frame.get("sensing"))
        assert_sensing_contract(raw_frame, require_schema_version=True)
        sensing_block = raw_frame.get("sensing") or {}
        batch = extract_timestamped_fast_time_samples(sensing_block)
        if not batch.samples:
            frame_idx = (raw_frame.get("config") or {}).get("frame_idx", "unknown")
            raise RuntimeError(
                f"Timestamp-aligned sensing CIR data is missing on frame {frame_idx}: "
                f"{batch.alignment_error}. Regenerate frames with sensing enabled."
            )
        sample_time_s = batch.sample_times_s[-1]
        samples = np.asarray(batch.samples[-1], dtype=np.complex128)
        if samples.size == 0:
            raise RuntimeError(
                "Stored typed sensing CIR product is empty; cannot compute sensing outputs."
            )
        range_axis_m = batch.range_axis_m
        if range_axis_m is None or range_axis_m.shape != samples.shape:
            raise RuntimeError(
                "Stored typed sensing CIR product has no exact row-aligned range coordinates"
            )
        if batch.range_coordinate_semantics == "monostatic_range":
            delays = 2.0 * range_axis_m / C0
        elif batch.range_coordinate_semantics == "bistatic_path_length":
            delays = range_axis_m / C0
        else:
            raise RuntimeError(
                "Stored typed sensing CIR product requires range coordinate semantics "
                "'monostatic_range' or 'bistatic_path_length'"
            )
        if normalize_delays:
            delays = delays - np.min(delays)

        return CIRComputationResult(
            cir=samples,
            delays_s=delays,
            metadata={
                "source": "stored_frame",
                "data_key": batch.data_key,
                "product_id": batch.product_id,
                "pair_id": batch.pair_id,
                "rx_element_identities": batch.rx_element_identities,
                "combine_policy": batch.combine_policy,
                "range_axis_m": range_axis_m,
                "range_coordinate_semantics": batch.range_coordinate_semantics,
                "representation": "complex_voltage",
                "variant": "primary",
                "sample_time_s": sample_time_s,
                "acquisition_id": batch.acquisition_id,
                "processing_strategy": batch.strategy,
            },
            sample_time_s=sample_time_s,
        )
