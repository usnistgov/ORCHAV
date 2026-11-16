"""Pure product demand for visible sensing figures and recording overviews.

Descriptors decide which saved evidence supplies a view. This module performs
no reads or signal processing; readers still validate each selected product.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

SPECTRAL_FAMILIES = frozenset({"saved-stft", "summed-rd", "range-profile"})
_FAMILY_KINDS = {
    "saved-stft": "micro_doppler",
    "summed-rd": "range_doppler",
    "range-profile": "range_profile",
}


@dataclass(frozen=True, order=True)
class ProductRequirement:
    """One authored product variant; data keys are resolved from descriptors."""

    kind: str
    variant: str


@dataclass(frozen=True)
class FigureRequirements:
    """Union of current products, compact spectra and detailed result history."""

    selected_products: tuple[ProductRequirement, ...]
    spectral_families: frozenset[str]
    rich_history: bool
    product_names: tuple[str, ...]


def product_requirements(
    kinds: Iterable[str],
    descriptors: Mapping[str, Mapping[str, Any]] | None,
    variant: str = "auto",
    *,
    both_variants: bool = False,
) -> tuple[ProductRequirement, ...]:
    """Resolve only authored variants, preserving fixed-STFT fallback policy."""
    preferred = "primary" if variant == "clean" else variant
    if preferred not in {"auto", "primary", "raw"}:
        raise ValueError(f"Unknown sensing product variant: {variant}")
    selected = set()
    for kind in kinds:
        available = (
            {d.get("variant") for d in descriptors.values() if d.get("kind") == kind}
            if descriptors is not None
            else {"primary", "raw"}
        )
        if both_variants:
            variants = tuple(v for v in ("primary", "raw") if v in available)
        elif descriptors is None:
            variants = (preferred,)
        elif preferred == "auto" or kind == "micro_doppler" and preferred not in available:
            variants = next(((v,) for v in ("primary", "raw") if v in available), ())
        else:
            variants = (preferred,) if preferred in available else ()
        selected.update(ProductRequirement(kind, v) for v in variants)
    return tuple(sorted(selected))


def product_data_names(
    requirements: Iterable[ProductRequirement],
    descriptors: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[str, ...]:
    """Return exact descriptor keys, or eligibility kinds before selection."""
    required = frozenset(requirements)
    if descriptors is None:
        return tuple(sorted({r.kind for r in required}))
    return tuple(
        sorted(
            key
            for key, descriptor in descriptors.items()
            if ProductRequirement(descriptor.get("kind"), descriptor.get("variant")) in required
        )
    )


def spectral_requirements(
    families: Iterable[str],
    descriptors: Mapping[str, Mapping[str, Any]],
    variant: str = "auto",
) -> tuple[ProductRequirement, ...]:
    """Select the saved products supplying the requested overview families."""
    families = frozenset(families)
    if not families <= SPECTRAL_FAMILIES:
        raise ValueError(f"Unknown spectral families: {sorted(families - SPECTRAL_FAMILIES)}")
    return product_requirements((_FAMILY_KINDS[f] for f in families), descriptors, variant)


def _input_kind(view: str, kinds: set[str]) -> str | None:
    candidates = (
        ("baseband_frequency_observation", "baseband_frequency_power")
        if view == "processing-frequency"
        else ("delay_domain_slow_time", "baseband_frequency_observation", "delay_domain_power")
    )
    return next((kind for kind in candidates if kind in kinds), None)


def figure_requirements(
    view: str,
    *,
    variant: str = "auto",
    micro_mode: str = "auto",
    descriptors: Mapping[str, Mapping[str, Any]] | None = None,
    visible_products: Iterable[str] = (),
    range_time_visible: bool = True,
    rich_history_visible: bool = False,
) -> FigureRequirements:
    """Describe actual figure demand without reading hidden maps or variants.

    Inputs preserve complex evidence when available: delay samples, otherwise
    frequency samples, otherwise saved delay power. Filtering deliberately
    compares both saved RD variants; reconstruction adds complex inputs only
    when the user explicitly requests it. Motion Auto remains summed RD power.
    """
    kinds = {d.get("kind") for d in descriptors.values()} if descriptors is not None else set()
    selected: set[ProductRequirement] = set()
    current = set(visible_products)
    families = set()
    rich = rich_history_visible or view in {"current", "analysis", "tracking", "evaluation"}
    if view in {"current", "analysis"}:
        current.update(("range_profile", "range_doppler"))
    elif view == "analysis-inspection":
        current.update(("range_profile", "range_doppler"))
    elif view == "angular":
        current.update(("range_doppler", "range_doppler_angle"))
    elif view in {"processing-inputs", "processing-frequency"}:
        if descriptors is None:
            current.update(
                ("baseband_frequency_observation", "baseband_frequency_power")
                if view == "processing-frequency"
                else (
                    "delay_domain_slow_time",
                    "baseband_frequency_observation",
                    "delay_domain_power",
                )
            )
        elif (kind := _input_kind(view, kinds)) is not None:
            selected.update(product_requirements((kind,), descriptors, "primary"))
    elif view in {"processing-clutter", "processing-clutter-rebuild"}:
        selected.update(
            product_requirements(("range_doppler",), descriptors, variant, both_variants=True)
        )
        if view == "processing-clutter-rebuild":
            source = next(
                (
                    k
                    for k in ("delay_domain_slow_time", "baseband_frequency_observation")
                    if k in kinds
                ),
                None,
            )
            if source is not None:
                selected.update(product_requirements((source,), descriptors, "primary"))
    elif view == "micro-doppler":
        if micro_mode == "saved-stft":
            families.add("saved-stft")
            current.add("micro_doppler")
        elif micro_mode in {"auto", "stored", "collapsed"}:
            families.add("summed-rd")
            current.add("range_doppler")
        else:
            current.add("range_doppler")
        if range_time_visible:
            families.add("range-profile")
    selected.update(product_requirements(current, descriptors, variant))
    requirements = tuple(sorted(selected))
    names = ("config_resolved", *product_data_names(requirements, descriptors))
    return FigureRequirements(requirements, frozenset(families), rich, names)


def supported_saved_views(descriptors: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """List finite saved-view scopes; explicit signal reconstruction is excluded."""
    kinds = {d.get("kind") for d in descriptors.values()}
    views = ["summary", "tracking"]
    if kinds & {"range_profile", "range_doppler"}:
        views.append("analysis")
    if "range_doppler_detection_input_power" in kinds:
        views.append("diagnostics")
    if "range_doppler" in kinds:
        views.append("processing-clutter")
    if "range_doppler_angle" in kinds:
        views.append("angular")
    for view in ("processing-inputs", "processing-frequency"):
        if _input_kind(view, kinds) is not None:
            views.append(view)
    if kinds & {"range_doppler", "micro_doppler"}:
        views.append("micro-doppler")
    return tuple(views)


def preparable_saved_views(descriptors: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    """Saved views suitable for warming without reading optional complex inputs.

    An available complex-input view remains deferred until explicitly selected.
    Its presence does not silently switch the view to a magnitude-only product.
    """
    return tuple(
        view
        for view in supported_saved_views(descriptors)
        if view not in {"processing-inputs", "processing-frequency"}
        or not any(
            product.kind in {"delay_domain_slow_time", "baseband_frequency_observation"}
            for product in figure_requirements(view, descriptors=descriptors).selected_products
        )
    )
