"""Resolve saved selection anchors independently of visible renderer overlays."""

import numpy as np

from ..services.sensing_service import SensingService
from .selection import matches_detection, selection_scope


def selected_world_position(request, selection, *, track=None):
    """Use the displayed result's pose and saved estimates; never infer a truth match."""
    if request is None:
        return None
    sensing = request.display_frame["sensing"]
    if selection is not None and (
        selection.source != request.source_identity
        or selection.scope != selection_scope(sensing)
        or selection.detection_collection
        != sensing.get("_display_detection_collection", "detections")
    ):
        return None
    try:
        if track is not None and (selection is None or selection.detection_index is None):
            position = track.get("position_m")
            if position is None:
                return None
            position = np.asarray(position, dtype=float)
            if not np.all(np.isfinite(position)):
                return None
            if track.get("position_frame") == "world":
                return position.copy() if position.shape == (3,) else None
            if track.get("position_frame") != "receiver local" or position.shape not in {
                (2,),
                (3,),
            }:
                return None
            rows = [{f"track_{axis}_m": value for axis, value in zip("xyz", position)}]
            key = "world_track_position"
        else:
            rows = [
                row
                for index, row in enumerate(sensing.get("detections", ()))
                if matches_detection(selection, sensing, index, row)
            ]
            key = (
                "world_position"
                if selection is not None and selection.detection_index is not None
                else "world_track_position"
            )
        if not rows:
            return None
        projected = SensingService.dashboard_world_detections(
            {**request.display_frame, "sensing": {**sensing, "detections": rows}}
        )
        for row in projected:
            position = row.get(key)
            if (
                position is not None
                and np.asarray(position).shape == (3,)
                and np.all(np.isfinite(position))
            ):
                return np.array(position, copy=True)
    except (KeyError, TypeError, ValueError):
        # A missing pose or a range-only output has no unique camera anchor.
        return None
    return None
