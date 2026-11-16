"""Optional decision records made by trackers, and saved-data inspection.

Detection indices refer to this update's input rows, before output confirmation
filtering. These records explain actual assignments; readers never rerun a
tracker or reconstruct assignments from output proximity.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np


def _state(track):
    mean = track.x.tolist() if hasattr(track, "x") else [track.range_m, track.velocity_m_s]
    return {
        "track_id": int(track.track_id),
        "mean": mean,
        "covariance": track.P.tolist() if hasattr(track, "P") else None,
        "confirmed": bool(track.confirmed),
        "misses": int(track.misses),
        "hits": int(track.hits) if hasattr(track, "hits") else sum(track.hit_history),
        "hit_history": list(track.hit_history) if hasattr(track, "hit_history") else None,
    }


class TrackingDecisionRecorder:
    """Created only when explicitly enabled; does not affect numerical state."""

    def __init__(self, tracks, detections, time_s, labels, *, prediction_kind="model"):
        self.predictions = [_state(t) for t in tracks.values()]
        self.inputs = [
            {
                k: deepcopy(d[k])
                for k in (
                    "range_m",
                    "velocity_m_s",
                    "x_m",
                    "y_m",
                    "z_m",
                    "aoa_az_deg",
                    "range_bin",
                    "doppler_bin",
                    "cluster_id",
                    "detection_id",
                )
                if k in d
            }
            for d in detections
        ]
        self.time_s, self.labels = float(time_s), labels
        self.prediction_kind = prediction_kind
        self.candidates = []

    def candidate(self, track_id, detection_index, cost, *, probability=None):
        self.candidates.append(
            {
                "track_id": int(track_id),
                "detection_index": int(detection_index),
                "admissible": cost is not None,
                "cost": None if cost is None else float(cost),
                "probability": None if probability is None else float(probability),
            }
        )

    def matrix(self, track_ids, costs):
        for row, tid in enumerate(track_ids):
            for col, value in enumerate(costs[row]):
                self.candidate(tid, col, value if value < 1e8 else None)

    def finish(self, tracks, assignments, outputs, *, missed_probabilities=None):
        previous = {p["track_id"] for p in self.predictions}
        current = set(tracks)
        emitted = {int(d["track_id"]) for d in outputs}
        states = [_state(t) for t in tracks.values()]
        for state in states:
            tid = state["track_id"]
            state["emitted"] = tid in emitted
            state["status"] = (
                "coasting"
                if state["misses"]
                else ("confirmed" if state["confirmed"] else "tentative")
            )
        return {
            "schema_version": 1,
            "update_time_s": self.time_s,
            "state_labels": self.labels,
            "prediction_kind": self.prediction_kind,
            "inputs": self.inputs,
            "predictions": self.predictions,
            "candidates": self.candidates,
            "assignments": [
                {"detection_index": int(di), "track_id": int(tid)}
                for di, tid in assignments.items()
            ],
            "birth_track_ids": sorted(current - previous),
            "retired_track_ids": sorted(previous - current),
            "tracks": states,
            "missed_probabilities": {
                str(k): float(v) for k, v in (missed_probabilities or {}).items()
            },
        }


def tracking_decisions(sensing):
    """Read the optional versioned record; absent legacy evidence stays absent."""
    metadata = sensing.get("detection_filter_metadata", {}).get("tracker", {})
    record = metadata.get("decisions")
    if record is None:
        return None, metadata
    if not isinstance(record, Mapping) or record.get("schema_version") != 1:
        raise ValueError("Unsupported tracking decision record")
    if not np.isfinite(record["update_time_s"]):
        raise ValueError("Invalid tracking decision time")
    return record, metadata
