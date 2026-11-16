"""Navigation through saved result bindings, independent of playback stride."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResultLocation:
    """A recorded result's physical epoch and explicit scene/storage addresses."""

    time_s: float
    scene_id: int | str
    storage_id: int | str
    context: tuple[str, str, str]
    acquisition_id: str


def result_locations(rows: tuple, context: tuple | None = None) -> tuple[ResultLocation, ...]:
    """Build compact navigation for one clock, run and pair, without dense reads."""
    winners = {}
    for storage_id, frame in rows:
        temporal = frame["sensing"]["temporal"]
        time_s = temporal["result"]["reference_time_s"]
        scene_id = temporal["display_binding"]["scene_frame_id"]
        if not isinstance(time_s, (float, int)) or not math.isfinite(time_s) or scene_id is None:
            continue
        scope = (
            temporal["clock"]["clock_id"],
            temporal["result"]["processing_run_id"],
            temporal["acquisition"]["pair_id"],
        )
        key = (
            time_s,
            scope[1],
            temporal["acquisition"]["acquisition_id"],
            f"{type(storage_id).__name__}:{storage_id!r}",
        )
        if scene_id not in winners or key > winners[scene_id][0]:
            winners[scene_id] = (
                key,
                ResultLocation(
                    float(time_s),
                    scene_id,
                    storage_id,
                    scope,
                    temporal["acquisition"]["acquisition_id"],
                ),
            )
    locations = [item[1] for item in winners.values()]
    if not locations:
        return ()
    if context is None:
        context = max(locations, key=lambda r: (r.time_s, r.context)).context
    return tuple(
        sorted(
            (r for r in locations if r.context == context),
            key=lambda r: (r.time_s, str(r.storage_id)),
        )
    )


def adjacent_result(
    locations: tuple[ResultLocation, ...], scene_id: Any, scene_order: list, direction: int
) -> ResultLocation | None:
    """Find the nearest bound scene before/after the displayed scene in playback order."""
    order = {step: index for index, step in enumerate(scene_order)}
    current = order.get(scene_id)
    if current is None:
        return None
    candidates = [
        r
        for r in locations
        if r.scene_id in order and (order[r.scene_id] - current) * direction > 0
    ]
    return min(candidates, key=lambda r: abs(order[r.scene_id] - current), default=None)


def nearest_result(locations: tuple[ResultLocation, ...], time_s: float) -> ResultLocation | None:
    """Resolve a spectrogram click to the nearest saved result, never a guessed frame."""
    if not locations or not locations[0].time_s <= time_s <= locations[-1].time_s:
        return None
    index = bisect_left([r.time_s for r in locations], time_s)
    return min(locations[max(0, index - 1) : index + 1], key=lambda r: abs(r.time_s - time_s))
