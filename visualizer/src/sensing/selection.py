"""Saved estimate selection shared by the dashboard and scene overlays.

Track IDs are scoped to a clock, processor run and sensor pair. Untracked
detections are scoped to one acquisition and their stored row; neither a nearby
truth target nor an MPC path is inferred from screen proximity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InspectionCell:
    """Exact coordinate selection, scoped to a source occurrence and recipe.

    Coordinates may link different receiver representations, not their signal
    values. A report's representative bin is never replaced by its centroid.
    """

    source: tuple
    recipe: str
    range_bin: int
    doppler_bin: int | None
    ranges: tuple[float, ...]
    doppler: tuple[float, ...]
    range_mode: str
    variant: str
    product_id: str
    combination: str
    origin: str = "cell"
    catalog_identity: tuple = ()


def catalog_identity(sensing: dict) -> tuple:
    """Content-addressed descriptor IDs detect in-place scientific replacements."""
    return tuple(
        sorted((key, p.get("product_id")) for key, p in sensing.get("products", {}).items())
    )


@dataclass(frozen=True)
class SensingSelection:
    """Saved report, track, or cell identity within one clock/run/pair scope."""

    scope: tuple[str, str, str]
    acquisition_id: str
    track_id: int | None = None
    detection_index: int | None = None
    cell: InspectionCell | None = None
    source: tuple = ()
    detection_collection: str = "detections"


def inspection_cell(
    request, r: int, d: int | None, *, origin="cell", model=None
) -> InspectionCell | None:
    """Bind loaded axes only; never read products or infer a bin from truth."""
    sensing = request.display_frame["sensing"]
    rd = request.rd_result
    if rd is None and request.detection_stage_bundle is not None:
        rd = request.detection_stage_bundle
    ranges = model.ranges if model is not None else (rd.range_axis_m if rd is not None else None)
    doppler = (
        model.doppler if model is not None else (rd.doppler_axis_hz if rd is not None else None)
    )
    if (
        ranges is None
        or not 0 <= r < len(ranges)
        or (d is not None and doppler is not None and not 0 <= d < len(doppler))
    ):
        return None
    products = sensing.get("products", {})
    kind = (
        "delay_domain_slow_time"
        if model is not None and model.mode == "inputs"
        else "range_doppler"
    )
    descriptor = next(
        (
            p
            for p in products.values()
            if p.get("kind") == kind
            and p.get("variant")
            == ("primary" if kind == "delay_domain_slow_time" else request.variant)
        ),
        {},
    )
    return InspectionCell(
        request.source_identity,
        sensing["temporal"]["result"]["processing_recipe_id"],
        r,
        d,
        tuple(np.asarray(ranges, dtype=float)),
        tuple(np.asarray(doppler, dtype=float)) if doppler is not None else (),
        request.range_mode,
        request.variant,
        descriptor.get("product_id", "not recorded"),
        str(descriptor.get("combine", {})),
        origin,
        catalog_identity(sensing),
    )


def selection_scope(sensing: dict) -> tuple[str, str, str]:
    """Return the clock, processing-run, and sensor-pair IDs that scope track IDs."""
    temporal = sensing["temporal"]
    return (
        temporal["clock"]["clock_id"],
        temporal["result"]["processing_run_id"],
        temporal["acquisition"]["pair_id"],
    )


def select_estimate(
    sensing: dict, *, detection_index: int | None = None, track_id: int | None = None
) -> SensingSelection:
    """Bind a clicked saved row/track to the recording's authored identity."""
    if detection_index is not None:
        track_id = (sensing.get("detections") or [])[detection_index].get("track_id")
    return SensingSelection(
        selection_scope(sensing),
        sensing["temporal"]["acquisition"]["acquisition_id"],
        track_id,
        detection_index,
        detection_collection=sensing.get("_display_detection_collection", "detections"),
    )


def matches_detection(
    selection: SensingSelection | None, sensing: dict, index: int, detection: dict[str, Any]
) -> bool:
    """Match a saved row within its acquisition, or a track across its scoped run."""
    if (
        selection is None
        or selection.scope != selection_scope(sensing)
        or selection.detection_collection
        != sensing.get("_display_detection_collection", "detections")
    ):
        return False
    if selection.detection_index is not None:
        return (
            selection.acquisition_id == sensing["temporal"]["acquisition"]["acquisition_id"]
            and selection.detection_index == index
        )
    return selection.track_id is not None and detection.get("track_id") == selection.track_id
