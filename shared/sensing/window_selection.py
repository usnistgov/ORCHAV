"""Select existing complete acquisitions without changing their source lineage.

The cadence describes derived publication. It never changes the acquired pulse
stream, reconstructs missing observations, or relabels an upstream acquisition.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from shared.extensions.sensing import canonical_sensing_hash
from shared.frames.contracts import FrameReadRequest
from shared.frames.manifest import load_frame_manifest
from shared.frames.providers import Hdf5Provider

logger = logging.getLogger(__name__)


def _identity(sensing: Mapping[str, Any], frame_set_id: str) -> str:
    temporal = sensing["temporal"]
    acquisition = deepcopy(temporal["acquisition"])
    for sample in acquisition["samples"]:
        if isinstance(sample.get("source"), dict):
            sample["source"].setdefault("source_frame_set_id", frame_set_id)
    return canonical_sensing_hash(
        {
            "clock": temporal["clock"],
            "acquisition": acquisition,
            "config_resolved": sensing["config_resolved"],
            "result": temporal.get("result"),
            "display_binding": temporal.get("display_binding"),
            "products": {
                key: value["product_id"] for key, value in sensing.get("products", {}).items()
            },
        }
    )


@dataclass(frozen=True)
class SavedWindowSelection:
    frame_ids: tuple[int, ...]
    identities: tuple[str, ...]
    recipe: Mapping[str, Any]

    def bind(self, frame: dict[str, Any], index: int) -> None:
        """Bind a verified source window to its new local output scene ID."""
        if (
            frame["_source_frame_set_id"] != self.recipe["source_frame_set_id"]
            or frame["_source_generation_id"] != self.recipe["source_generation_id"]
            or frame["_frame_index"] != self.frame_ids[index]
            or _identity(frame["_typed_sensing"], self.recipe["source_frame_set_id"])
            != self.identities[index]
        ):
            raise ValueError("Selected source acquisition changed after preflight")
        frame["_output_selection"] = dict(self.recipe)
        frame["_frame_index"] = index


def select_saved_windows(frames_dir: Path, hop_samples: int) -> SavedWindowSelection:
    """Scan metadata and select exact existing windows on a declared cadence.

    A single uniformly sampled clock/pair/acquisition policy is required. The
    first complete window is the cadence anchor; each requested subsequent
    window must exist. Partial acquisitions are never presented as results.
    """
    if type(hop_samples) is not int or hop_samples < 1:
        raise ValueError("result hop samples must be a positive integer")
    frames_dir = Path(frames_dir)
    manifest = load_frame_manifest(frames_dir)
    provider = Hdf5Provider(frames_dir.parent, frames_subdir=frames_dir.name)
    request = FrameReadRequest(sensing_products=frozenset({"config_resolved"}))
    selected: list[int] = []
    identities: list[str] = []
    context = None
    anchor = None
    previous_tick = None
    last_tick = 0
    complete_count = 0
    previous_selected = None
    scope = None
    try:
        for index, projection in enumerate(
            provider.iter_frame_projections(manifest.frame_ids, request)
        ):
            sensing = projection.frame.sensing or {}
            temporal = sensing.get("temporal", {})
            acquisition = temporal.get("acquisition", {})
            if acquisition.get("completeness") != "complete":
                continue
            plan = sensing["config_resolved"]["acquisition"]
            if plan.get("sample_period_s") is None:
                raise ValueError("Window selection requires a sampled acquisition clock")
            period = float(plan["sample_period_s"])
            length = int(plan["cpi_length_samples"])
            source_hop = int(plan["cpi_hop_samples"])
            current_context = canonical_sensing_hash(
                {
                    "clock": temporal["clock"],
                    "pair": acquisition["pair_id"],
                    "plan": plan,
                    "acquisition_config_id": acquisition["acquisition_config_id"],
                }
            )
            if context is None:
                context = current_context
                scope = (
                    temporal["clock"]["clock_id"],
                    acquisition["pair_id"],
                    acquisition["acquisition_config_id"],
                )
                if (
                    not np.isfinite(period)
                    or period <= 0
                    or not source_hop <= hop_samples <= length
                    or hop_samples % source_hop
                ):
                    raise ValueError(
                        "Result hop must be a multiple of the source hop, no larger than its window"
                    )
            elif context != current_context:
                raise ValueError("Window selection requires one clock, pair and acquisition policy")
            samples = acquisition["samples"]
            times = np.asarray([sample["sample_time_s"] for sample in samples], dtype=float)
            if (
                len(samples) != length
                or not all(sample.get("valid", False) for sample in samples)
                or not np.all(np.isfinite(times))
                or not np.allclose(np.diff(times), period, rtol=1e-7, atol=1e-12)
            ):
                raise ValueError(
                    "Window selection requires complete uniformly sampled acquisitions"
                )
            if anchor is None:
                anchor = float(times[0])
            offset = (float(times[0]) - anchor) / period
            tick = round(offset)
            if not np.isclose(offset, tick, rtol=0, atol=1e-6) or (
                previous_tick is not None and tick <= previous_tick
            ):
                raise ValueError("Source windows must have unique increasing sample epochs")
            previous_tick = last_tick = tick
            complete_count += 1
            if tick % hop_samples == 0:
                if tick != len(selected) * hop_samples:
                    raise ValueError("A requested complete window is missing from the source")
                current_samples = {
                    round((float(sample["sample_time_s"]) - anchor) / period): sample
                    for sample in samples
                }
                if previous_selected is not None:
                    for sample_tick in current_samples.keys() & previous_selected.keys():
                        # Local sample indices differ between overlapping windows;
                        # their acquired pulse identity and time must not differ.
                        old = {
                            key: value
                            for key, value in previous_selected[sample_tick].items()
                            if key != "sample_index"
                        }
                        new = {
                            key: value
                            for key, value in current_samples[sample_tick].items()
                            if key != "sample_index"
                        }
                        if old != new:
                            raise ValueError(
                                "Overlapping selected windows disagree about acquired pulse identity"
                            )
                previous_selected = current_samples
                selected.append(int(projection.frame.frame_index))
                identities.append(_identity(sensing, manifest.frame_set_id))
            if index % 512 == 0:
                logger.info(
                    "Selecting saved windows: %d/%d source rows, %d selected",
                    index + 1,
                    len(manifest.frame_ids),
                    len(selected),
                )
    finally:
        provider.close()
    if not selected:
        raise ValueError("No complete saved acquisitions are available for selection")
    if len(selected) != last_tick // hop_samples + 1:
        raise ValueError("A requested complete window is missing from the source")
    recipe = {
        "kind": "existing_complete_acquisitions",
        "source_generation_id": manifest.generation_id,
        "source_frame_set_id": manifest.frame_set_id,
        "source_hop_samples": source_hop,
        "hop_samples": hop_samples,
        "sample_period_s": period,
        "window_samples": length,
        "anchor_sample_time_s": anchor,
        "source_complete_count": complete_count,
        "selected_count": len(selected),
        "source_frame_ids_sha256": canonical_sensing_hash(list(selected)),
        "clock_id": scope[0],
        "pair_id": scope[1],
        "acquisition_config_id": scope[2],
    }
    return SavedWindowSelection(tuple(selected), tuple(identities), recipe)


def output_hop_samples(sensing: Mapping[str, Any]) -> int:
    """Resolve a validated derived publication cadence, or the native cadence."""
    from shared.extensions.sensing import canonical_processing_recipe_id

    config = sensing["config_resolved"]
    plan = config["acquisition"]
    native_hop = plan["cpi_hop_samples"]
    selection = config["processing"].get("output_selection")
    if selection is None:
        return int(native_hop)
    recipe = config.get("processing_recipe", {})
    temporal = sensing["temporal"]
    acquisition = temporal["acquisition"]
    if (
        config["processing"]["mode"] != "offline_reprocess"
        or recipe.get("tool") != "shared.sensing_postprocessor"
        or recipe.get("output_selection") != selection
        or canonical_processing_recipe_id(recipe) != temporal["result"]["processing_recipe_id"]
        or selection.get("kind") != "existing_complete_acquisitions"
    ):
        raise ValueError("Derived output cadence is not bound to its processing recipe")
    hop, count = selection.get("hop_samples"), selection.get("selected_count")
    if (
        type(hop) is not int
        or type(count) is not int
        or count < 1
        or not native_hop <= hop <= plan["cpi_length_samples"]
        or hop % native_hop
        or selection.get("source_hop_samples") != native_hop
        or selection.get("window_samples") != plan["cpi_length_samples"]
        or selection.get("sample_period_s") != plan["sample_period_s"]
        or selection.get("clock_id") != temporal["clock"]["clock_id"]
        or selection.get("pair_id") != acquisition["pair_id"]
        or selection.get("acquisition_config_id") != acquisition["acquisition_config_id"]
        or type(selection.get("source_complete_count")) is not int
        or selection["source_complete_count"] < count
    ):
        raise ValueError("Derived output cadence disagrees with its acquired window")
    period = float(selection["sample_period_s"])
    anchor = float(selection["anchor_sample_time_s"])
    samples = acquisition["samples"]
    times = np.asarray([sample["sample_time_s"] for sample in samples], dtype=float)
    if (
        len(samples) != plan["cpi_length_samples"]
        or acquisition.get("completeness") != "complete"
        or not all(sample.get("valid") is True for sample in samples)
        or not np.all(np.isfinite(times))
        or not np.allclose(np.diff(times), period, rtol=1e-7, atol=1e-12)
    ):
        raise ValueError("Derived output cadence requires complete uniform window membership")
    start = float(times[0])
    if not np.isfinite(period) or period <= 0 or not np.isfinite(anchor) or not np.isfinite(start):
        raise ValueError("Derived output cadence requires finite sampled epochs")
    offset = (start - anchor) / period
    index = round(offset / hop)
    if not 0 <= index < count or not np.isclose(offset, index * hop, rtol=0, atol=1e-6):
        raise ValueError("Acquired window is outside its declared output schedule")
    return hop
