"""Exact, selectively loaded spectral histories independent of result detail rows.

Saved STFT columns, sums of saved RD power and saved range profiles retain their
own physical time grids. No transform, interpolation or detector is rerun here.
One caller-provided budget bounds the retained families together; loading and
plotting working buffers are separate from that retention allowance.
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from copy import deepcopy
from dataclasses import dataclass, replace
from threading import Event

import numpy as np

from shared.cache_sizing import estimate_retained_bytes

from .recorded_products import saved_micro_doppler

SPECTRAL_PROTOCOL = "exact-saved-spectral-v1"
SPECTRAL_FAMILIES = ("saved-stft", "summed-rd", "range-profile")
_KINDS = dict(zip(SPECTRAL_FAMILIES, ("micro_doppler", "range_doppler", "range_profile")))


class _UnsampledHistory(ValueError):
    """A valid static acquisition has no physical spectral-history footprint."""


@dataclass(frozen=True)
class SpectralCoverage:
    """Coverage of authored result rows, distinct from numeric column count."""

    available_results: int
    loaded_results: int
    available_result_span: tuple[float, float] | None
    state: str
    byte_limit: int
    reason: str = ""


@dataclass(frozen=True)
class SpectralHistory:
    """One compatible saved product family with exact column provenance."""

    family: str
    power: np.ndarray
    times_s: np.ndarray
    widths_s: np.ndarray
    axis: np.ndarray
    variant: str
    recipe: dict
    # Each column names the stored occurrence and its acquisition. Equal saved
    # STFT columns may be authored in overlapping blocks; one occurrence owns it.
    sources: tuple[tuple[int | str, str], ...]
    coverage: SpectralCoverage
    available_variants: tuple[str, ...] = ()

    @property
    def retained_bytes(self) -> int:
        # Empty capability/status objects are fixed control bookkeeping, not
        # retained signal data. Nonempty families include their metadata owner.
        return estimate_retained_bytes(self) if self.power.size else 0

    @property
    def loaded_time_span(self) -> tuple[float, float] | None:
        return (float(self.times_s[0]), float(self.times_s[-1])) if self.times_s.size else None

    def coverage_text(self) -> str:
        coverage = self.coverage
        count = f"{coverage.loaded_results:,}/{coverage.available_results:,} saved results"
        span = self.loaded_time_span
        if span is not None:
            count += f" · {span[0]:.4g}–{span[1]:.4g} s"
        if coverage.state == "limited":
            count += " · memory limit reached"
        elif coverage.state == "error":
            count += " · unavailable"
        return count


@dataclass(frozen=True)
class MotionOverview:
    """The requested families for exactly one clock, run, pair and variant."""

    context: tuple[str, str, str]
    variant: str
    families: dict[str, SpectralHistory]
    byte_limit: int

    @property
    def retained_bytes(self) -> int:
        return sum(item.retained_bytes for item in self.families.values())


def spectral_cache_key(context, variant, family, byte_limit=None) -> tuple:
    """Complete scientific data is independent of sibling admission allowances."""
    return (
        (*context, variant, family, "complete")
        if byte_limit is None
        else (*context, variant, family, "limited", int(byte_limit))
    )


def _owned(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def _empty(family, variant, count, span, limit, state="unavailable", reason=""):
    return SpectralHistory(
        family,
        _owned(np.empty((0, 0))),
        _owned([]),
        _owned([]),
        _owned([]),
        variant,
        {},
        (),
        SpectralCoverage(count, 0, span, state, limit, reason),
    )


def failed_motion_overview(context, variant, families, byte_limit, error) -> MotionOverview:
    """Represent a failed GUI worker without claiming saved-data coverage."""
    reason = "Spectral history unavailable: " + (str(error) or type(error).__name__)
    return MotionOverview(
        tuple(context),
        variant,
        {
            family: _empty(family, variant, 0, None, byte_limit, "error", reason)
            for family in families
        },
        byte_limit,
    )


def _eligible_rows(service, context, family, variant):
    from .figure_requirements import product_requirements

    # Last stored occurrence owns a repeated acquisition, matching Run Summary.
    unique = {}
    for storage, frame in service.dashboard_result_rows():
        sensing = frame["sensing"]
        temporal = sensing["temporal"]
        identity = (
            temporal["clock"]["clock_id"],
            temporal["result"]["processing_run_id"],
            temporal["acquisition"]["pair_id"],
        )
        time = temporal["result"]["reference_time_s"]
        binding = temporal["display_binding"]["scene_frame_id"]
        if (
            identity == context
            and type(time) in (int, float)
            and np.isfinite(time)
            and type(binding) in (int, str)
            and product_requirements((_KINDS[family],), sensing.get("products", {}), variant)
        ):
            unique[temporal["acquisition"]["acquisition_id"]] = (storage, frame)
    return sorted(
        unique.values(), key=lambda row: row[1]["sensing"]["temporal"]["result"]["reference_time_s"]
    )


def _postprocessor_policy(sensing, descriptor):
    """Compare replay settings independently of each noisy observation's identity."""
    from shared.extensions.sensing import canonical_processing_recipe_id, canonical_sensing_hash

    processing = descriptor.get("processing", {})
    if sensing["config_resolved"].get("processing", {}).get("output_selection") is not None:
        from shared.sensing.window_selection import output_hop_samples

        output_hop_samples(sensing)
        recipe = sensing["config_resolved"]["processing_recipe"]
        recipe_id = canonical_processing_recipe_id(recipe)
        lineage = descriptor.get("lineage", {})
        source_id = processing.get("source_product_id")
        if (
            lineage.get("processing_recipe_id") != recipe_id
            or source_id not in lineage.get("source_product_ids", ())
            or source_id != sensing.get("postprocessing", {}).get("source_product_id")
        ):
            raise ValueError("Selected replay product does not match its source observation")
        policy = deepcopy(processing)
        policy.pop("source_product_id")
        noise = policy.get("noise")
        pulses = noise.get("pulses") if isinstance(noise, dict) else None
        if isinstance(pulses, list):
            # These two fields identify each acquired noise occurrence. Keep
            # every pulse's noise distribution, variance, seed and algorithm.
            for pulse in pulses:
                if isinstance(pulse, dict):
                    for key in ("observation_sha256", "realization_index"):
                        pulse.pop(key, None)
        return canonical_sensing_hash({"recipe": recipe, "producer": policy})
    if processing.get("tool") != "shared.sensing_postprocessor":
        return None
    recipe = sensing["config_resolved"].get("processing_recipe", {})
    recipe_id = processing.get("recipe_id")
    if (
        recipe.get("tool") != processing["tool"]
        or recipe.get("input_kind") != processing.get("input_kind")
        or canonical_processing_recipe_id(recipe) != recipe_id
        or descriptor.get("lineage", {}).get("processing_recipe_id") != recipe_id
    ):
        raise ValueError("Saved postprocessor product and declared processing recipe disagree")
    policy = deepcopy(recipe)
    inherited = policy.get("inherited_numeric_model")
    response = inherited.get("range_response_processing") if isinstance(inherited, dict) else None
    noise = response.get("noise") if isinstance(response, dict) else None
    # These are per-observation evidence, not noise settings. Keep generator,
    # seed, variance and every other declared transform/processing value exact.
    if isinstance(noise, dict):
        for name in ("observation_sha256", "realization_index"):
            noise.pop(name, None)
    return canonical_sensing_hash(
        {
            "recipe": policy,
            "producer": {key: value for key, value in processing.items() if key != "recipe_id"},
        }
    )


def _block(service, storage, variant, family):
    frame, rp, rd = service.dashboard_spectral_products(storage, variant, frozenset({family}))
    if frame is None:
        raise ValueError("An indexed saved spectral result could not be loaded")
    sensing = frame["sensing"]
    temporal = sensing["temporal"]
    source = (storage, temporal["acquisition"]["acquisition_id"])
    policy = None
    if family == "saved-stft":
        micro = saved_micro_doppler(sensing, variant)
        if micro is None:
            raise ValueError("The indexed saved STFT product is unavailable")
        data = micro["power"], micro["times_s"], np.full(len(micro["times_s"]), micro["hop_s"])
        axis, recipe = micro["doppler_hz"], micro["recipe"]
        resolved = micro["variant"]
    else:
        product = rd if family == "summed-rd" else rp
        if product is None:
            raise ValueError("The requested saved spectrum variant is unavailable")
        resolved = product.metadata["variant"]
        descriptor = next(
            d
            for d in sensing["products"].values()
            if d["kind"] == _KINDS[family] and d["variant"] == resolved
        )
        recipe = descriptor.get("processing", {})
        policy = _postprocessor_policy(sensing, descriptor)
        cfg = sensing["config_resolved"]["acquisition"]
        if cfg["sample_period_s"] is None:
            raise _UnsampledHistory("This recording has no sampled acquisition clock")
        from shared.sensing.window_selection import output_hop_samples

        width = float(cfg["sample_period_s"]) * output_hop_samples(sensing)
        data = (
            (
                np.sum(rd.power_linear, axis=-1)
                if rd is not None and family == "summed-rd"
                else rp.power_linear
            )[None, :],
            np.array([temporal["result"]["reference_time_s"]]),
            np.array([width]),
        )
        axis = rd.doppler_axis_hz if family == "summed-rd" else rp.range_axis_m
    return data, axis, recipe, resolved, source, policy


def _validate(item, family, limit):
    if item.family != family or item.retained_bytes > limit:
        raise ValueError("Spectral cache exceeds its requested retention allowance")
    if (
        item.power.shape != (item.times_s.size, item.axis.size)
        or item.widths_s.shape != item.times_s.shape
    ):
        raise ValueError("Invalid saved spectral matrix shape")
    if len(item.sources) != item.times_s.size:
        raise ValueError("Saved spectral column provenance is incomplete")
    if any(not np.isfinite(a).all() for a in (item.power, item.times_s, item.widths_s, item.axis)):
        raise ValueError("Nonfinite saved spectral values")
    if (
        np.any(item.power < 0)
        or np.any(item.widths_s <= 0)
        or np.any(np.diff(item.times_s) <= 0)
        or np.any(np.diff(item.axis) <= 0)
    ):
        raise ValueError("Invalid saved spectral coordinates or power")
    if not 0 <= item.coverage.loaded_results <= item.coverage.available_results:
        raise ValueError("Invalid spectral coverage counts")
    if (
        type(item.coverage.loaded_results) is not int
        or type(item.coverage.available_results) is not int
        or item.coverage.state not in {"complete", "limited", "unavailable"}
    ):
        raise ValueError("Invalid spectral coverage contract")
    if item.coverage.loaded_results and not item.times_s.size:
        raise ValueError("Loaded spectral results have no numeric columns")
    if item.coverage.state == "unavailable" and item.times_s.size:
        raise ValueError("Unavailable spectral data contains numeric columns")
    if (
        item.coverage.state == "complete"
        and item.coverage.loaded_results != item.coverage.available_results
    ):
        raise ValueError("Incomplete spectral data advertised as complete")


def _load_family(service, context, variant, family, cancelled, limit, progress):
    from .component_cache import load_component, save_component
    from .figure_requirements import product_requirements

    rows = _eligible_rows(service, context, family, variant)
    count = len(rows)
    span = (
        tuple(
            float(row[1]["sensing"]["temporal"]["result"]["reference_time_s"])
            for row in (rows[0], rows[-1])
        )
        if rows
        else None
    )
    cache = service.recording_cache
    available_variants = tuple(
        sorted(
            {
                descriptor["variant"]
                for _, frame in rows
                for descriptor in frame["sensing"].get("products", {}).values()
                if descriptor.get("kind") == _KINDS[family]
            }
        )
    )
    expected_variant, expected_recipe = None, None
    if rows:
        descriptors = rows[0][1]["sensing"]["products"]
        requirement = product_requirements((_KINDS[family],), descriptors, variant)[0]
        descriptor = next(
            d
            for d in descriptors.values()
            if d["kind"] == requirement.kind and d["variant"] == requirement.variant
        )
        if "processing" not in descriptor:
            # The shared selection index deliberately keeps only compact
            # descriptor fields. Recover one authored metadata contract; this
            # projection requests no signal arrays, even on a warm cache hit.
            authored = service.dashboard_load_row(rows[0][0], ("config_resolved",))
            if authored is None:
                raise ValueError("The saved spectral processing contract is unavailable")
            descriptor = next(
                (
                    d
                    for d in authored["sensing"]["products"].values()
                    if d["kind"] == requirement.kind and d["variant"] == requirement.variant
                ),
                None,
            )
            if descriptor is None:
                raise ValueError("The saved spectral product differs from its selection index")
        expected_variant = requirement.variant
        processing = descriptor.get("processing", {})
        expected_recipe = (
            processing.get("recipe", processing) if family == "saved-stft" else processing
        )

    def read_cached(key):
        cached = (
            load_component(
                cache,
                "motion-overview",
                key,
                protocol=SPECTRAL_PROTOCOL,
                byte_limit=limit,
                cancelled=cancelled,
            )
            if cache is not None
            else None
        )
        if cached is None:
            return None
        metadata, arrays = cached
        try:
            details = dict(metadata["coverage"])
            if details["available_result_span"] is not None:
                details["available_result_span"] = tuple(details["available_result_span"])
            coverage = replace(SpectralCoverage(**details), byte_limit=limit)
            item = SpectralHistory(
                family,
                **arrays,
                variant=metadata["variant"],
                recipe=metadata["recipe"],
                sources=tuple(tuple(v) for v in metadata["sources"]),
                coverage=coverage,
                available_variants=tuple(metadata["available_variants"]),
            )
            _validate(item, family, limit)
            if coverage.available_results != count or coverage.available_result_span != span:
                raise ValueError("Spectral cache coverage differs from the source inventory")
            if item.available_variants != available_variants or (
                item.times_s.size
                and (item.variant != expected_variant or item.recipe != expected_recipe)
            ):
                raise ValueError(
                    "Spectral cache recipe or variant differs from the authored product"
                )
            if key[-1] == "complete" and coverage.state not in {"complete", "unavailable"}:
                raise ValueError("A limited artifact cannot certify complete coverage")
            return item
        except (KeyError, TypeError, ValueError):
            return None

    for key in (
        spectral_cache_key(context, variant, family),
        spectral_cache_key(context, variant, family, limit),
    ):
        cached = read_cached(key)
        if cached is not None:
            return cached
    columns = {}
    axis = None
    recipe = {}
    policy = None
    resolved = variant
    loaded = 0
    retained_size = 2048
    state, reason = (
        ("complete", "") if count else ("unavailable", "No saved product in this recording")
    )
    item = _empty(family, variant, count, span, limit, state, reason)
    for index, (storage, _metadata) in enumerate(rows):
        if cancelled.is_set():
            raise CancelledError()
        progress(f"Loading {family}", index, count)
        if cancelled.is_set():
            raise CancelledError()
        try:
            (
                (power, times, widths),
                block_axis,
                block_recipe,
                block_variant,
                source,
                block_policy,
            ) = _block(service, storage, variant, family)
        except _UnsampledHistory as exc:
            state, reason = "unavailable", str(exc)
            columns.clear()
            loaded = 0
            break
        if axis is not None and (
            not np.array_equal(axis, block_axis)
            or (
                recipe != block_recipe
                and (policy is None or block_policy is None or policy != block_policy)
            )
            or resolved != block_variant
        ):
            raise ValueError(
                "Saved spectral grids, variants or processing recipes differ within this recording"
            )
        if axis is None:
            axis, recipe, resolved = _owned(block_axis), block_recipe, block_variant
            policy = block_policy
            retained_size += estimate_retained_bytes(axis, recipe)
        additions = {}
        for time, width, column in zip(times, widths, power, strict=True):
            previous = columns.get(float(time)) or additions.get(float(time))
            if previous is not None:
                if family != "saved-stft":
                    raise ValueError("Distinct saved results share the same spectral epoch")
                if not np.array_equal(previous[0], column) or previous[1] != float(width):
                    raise ValueError(
                        "Conflicting saved spectral columns at the same physical epoch"
                    )
                continue
            additions[float(time)] = (_owned(column), float(width), source)
        # Account before retaining a block. The final matrix has the same
        # numeric bytes; this conservative owner estimate also includes lists.
        addition_size = estimate_retained_bytes(additions)
        if retained_size + addition_size > limit:
            state, reason = (
                "limited",
                "Increase --sensing-overview-cache-mb to retain more spectral history",
            )
            break
        columns.update(additions)
        retained_size += addition_size
        loaded += 1
    if columns:
        times = sorted(columns)
        item = SpectralHistory(
            family,
            _owned(np.stack([columns[t][0] for t in times])),
            _owned(times),
            _owned([columns[t][1] for t in times]),
            axis,
            resolved,
            recipe,
            tuple(columns[t][2] for t in times),
            SpectralCoverage(count, loaded, span, state, limit, reason),
        )
    else:
        item = _empty(family, variant, count, span, limit, state, reason)
    item = replace(
        item,
        available_variants=available_variants,
    )
    # Empty status objects have small fixed bookkeeping overhead. They carry no
    # signal allocation when the requested budget is too small for one column.
    if columns:
        _validate(item, family, limit)
    if cancelled.is_set():
        raise CancelledError()
    if cache is not None:
        from dataclasses import asdict

        key = spectral_cache_key(
            context, variant, family, limit if item.coverage.state == "limited" else None
        )
        saved = save_component(
            cache,
            "motion-overview",
            key,
            {
                "variant": item.variant,
                "recipe": item.recipe,
                "sources": item.sources,
                "available_variants": item.available_variants,
                "coverage": asdict(item.coverage),
            },
            {name: getattr(item, name) for name in ("power", "times_s", "widths_s", "axis")},
            protocol=SPECTRAL_PROTOCOL,
            cancelled=cancelled,
        )
        if cache._building and (not saved or read_cached(key) is None):
            raise OSError("Could not save the spectral overview")
    progress(f"{family} ready", count, count)
    return item


def load_motion_overview(
    service, context, variant, families, cancelled: Event, *, byte_limit=64 * 1024**2, progress=None
):
    """Load only requested spectral families under a shared retention budget."""
    if not set(families).issubset(SPECTRAL_FAMILIES):
        raise ValueError("Unknown spectral overview family")
    progress = progress or (lambda *_args: None)
    results = {}
    remaining = byte_limit
    for family in SPECTRAL_FAMILIES:
        if family not in families:
            continue
        if cancelled.is_set():
            raise CancelledError()
        try:
            item = _load_family(service, context, variant, family, cancelled, remaining, progress)
        except (KeyError, TypeError, ValueError) as exc:
            if service.recording_cache is not None and service.recording_cache._building:
                raise
            item = _empty(family, variant, 0, None, remaining, "error", str(exc))
        results[family] = item
        remaining = max(0, remaining - item.retained_bytes)
    return MotionOverview(tuple(context), variant, results, byte_limit)
