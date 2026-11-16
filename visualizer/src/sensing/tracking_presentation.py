"""Read-only track states and evidence for the sensing browser and inspector.

Detector output rows are not a complete tracker population. Prefer explicitly
saved internal states, and keep source identity and observation status attached
to every row. Nothing here predicts a state or associates a measurement.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from shared.sensing.tracking_inspection import tracking_decisions


@dataclass(frozen=True)
class TrackingSnapshot:
    """One exact result epoch; missing states are distinct from an empty population."""

    source_key: tuple
    states: tuple[dict, ...] = ()
    associations: tuple[dict, ...] = ()
    state_labels: tuple[str, ...] = ()
    note: str = "No saved track states"
    complete_states: bool = False


def _observation_status(predicted_only):
    return (
        "coasting"
        if predicted_only is True
        else ("observed" if predicted_only is False else "observation unknown")
    )


def _native_evidence(sensing, track_id):
    temporal = sensing.get("temporal", {})
    acquisition = temporal.get("acquisition", {})
    result = temporal.get("result", {})
    for index, row in enumerate(sensing.get("detections", ())):
        if row.get("track_id") == track_id:
            return {
                "acquisition_id": acquisition.get("acquisition_id"),
                "processing_recipe_id": result.get("processing_recipe_id"),
                "time_s": result.get("reference_time_s"),
                "detection_collection": sensing.get("_display_detection_collection", "detections"),
                "detection_index": index,
            }
    return None


def _native_state(raw, labels, source_key, sensing, *, complete):
    state = dict(raw)
    values = dict(zip(labels, raw.get("mean", ()), strict=True))
    misses = raw.get("misses")
    predicted = (
        bool(misses) if complete and misses is not None else (False if not complete else None)
    )
    state.update(
        source_kind="native",
        source_key=source_key,
        state_labels=tuple(labels),
        predicted_only=predicted,
        status=_observation_status(predicted),
        position_m=None,
        position_frame=None,
        state_frame="range / closing velocity" if "range_m" in values else None,
        estimated_range_m=values.get("range_m"),
        radial_velocity_m_s=values.get("closing_velocity_m_s"),
        evidence=None if predicted else _native_evidence(sensing, raw["track_id"]),
    )
    if "x_m" in values and "y_m" in values:
        state["position_m"] = [values[k] for k in ("x_m", "y_m", "z_m") if k in values]
        state["position_frame"] = "receiver local"
        state["state_frame"] = "receiver local"
    return state


def _output_states(sensing, source_key):
    """Read emitted estimates only; never invent hidden/coasting states."""
    states = {}
    for row in sensing.get("detections", ()):
        tid = row.get("track_id")
        if tid is None or tid in states:
            continue
        if "track_x_m" in row and "track_y_m" in row:
            labels = tuple(
                k
                for k in ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")
                if f"track_{k}" in row
            )
            mean = [row[f"track_{k}"] for k in labels]
        else:
            fields = (("range_m", "track_range_m"), ("closing_velocity_m_s", "track_velocity_m_s"))
            labels = tuple(label for label, key in fields if key in row)
            mean = [row[key] for _, key in fields if key in row]
        state = _native_state(
            {
                "track_id": tid,
                "mean": mean,
                "confirmed": row.get("track_confirmed"),
                "covariance": None,
            },
            labels,
            source_key,
            sensing,
            complete=False,
        )
        if row.get("world_track_position") is not None:
            state["world_position_m"] = row["world_track_position"]
        states[tid] = state
    return tuple(states.values())


def tracking_snapshot(
    sensing: dict[str, Any] | None,
    *,
    saved_tracking=None,
    time_s: float | None = None,
    include_decisions: bool = False,
) -> TrackingSnapshot:
    """Adapt native or verified world states without file reads unless requested.

    A separate world bundle must contain this exact epoch. Native candidates use
    tracker-input indices, which must never be treated as emitted-output indices.
    """
    sensing = sensing or {}
    if saved_tracking is not None:
        reader, sensor_id = saved_tracking.reader, saved_tracking.sensor_id
        key = ("saved_world_track", reader.inspection_id, sensor_id)
        if time_s is None or not len(reader.times):
            return TrackingSnapshot(key, note="No tracking epoch at this time")
        index = reader.epoch_index(time_s)
        if abs(reader.times[index] - time_s) > 1e-9:
            return TrackingSnapshot(key, note="No tracking epoch at this time")
        sensor = np.asarray(reader["sources"][sensor_id]["sensor"]["position_m"])
        states = []
        labels = ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")
        for raw in reader.epoch(index)["tracks"]:
            delta = np.subtract(raw["position_m"], sensor)
            distance = float(np.linalg.norm(delta))
            predicted = raw.get("predicted_only")
            evidence = None
            if predicted is False:
                evidence = next(
                    (
                        s["native_source"]
                        for s in raw.get("sources", ())
                        if s.get("sensor_id") == sensor_id
                        and isinstance(s.get("native_source"), dict)
                    ),
                    None,
                )
            states.append(
                {
                    **raw,
                    "source_kind": "saved_world_track",
                    "source_key": key,
                    "state_labels": labels,
                    "mean": [*raw["position_m"], *raw["velocity_m_s"]],
                    "position_frame": "world",
                    "state_frame": "world",
                    "status": _observation_status(predicted),
                    "estimated_range_m": distance,
                    "radial_velocity_m_s": (
                        -float(np.dot(raw["velocity_m_s"], delta / distance)) if distance else None
                    ),
                    "evidence": evidence,
                }
            )
        associations = ()
        if include_decisions:
            decisions = reader.decisions(index).get("decisions")
            if isinstance(decisions, list):
                associations = tuple(dict(d) for d in decisions)
        return TrackingSnapshot(
            key,
            tuple(states),
            associations,
            labels,
            "Saved 3D tracks",
            True,
        )

    temporal = sensing.get("temporal", {})
    key = (
        "native",
        temporal.get("clock", {}).get("clock_id"),
        temporal.get("result", {}).get("processing_run_id"),
        temporal.get("acquisition", {}).get("pair_id"),
    )
    decisions, _metadata = tracking_decisions(sensing)
    if decisions is None:
        return TrackingSnapshot(
            key,
            _output_states(sensing, key),
            note="Published track states",
        )
    labels = tuple(decisions["state_labels"])
    states = tuple(
        _native_state(raw, labels, key, sensing, complete=True) for raw in decisions["tracks"]
    )
    assignments = {
        (row["track_id"], row["detection_index"]) for row in decisions.get("assignments", ())
    }
    associations = tuple(
        {
            **row,
            "outcome": (
                "associated"
                if (row["track_id"], row["detection_index"]) in assignments
                else "candidate" if row["admissible"] else "gate rejection"
            ),
        }
        for row in decisions.get("candidates", ())
    )
    return TrackingSnapshot(
        key,
        states,
        associations,
        labels,
        "Saved tracker states",
        True,
    )


def state_uncertainty(state: dict) -> tuple[tuple[str, float], ...]:
    """Marginal standard deviations from a saved covariance, never inferred noise."""
    covariance = state.get("covariance")
    labels = tuple(state.get("state_labels", ()))
    if covariance is None:
        return ()
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (len(labels), len(labels)) or not np.isfinite(matrix).all():
        return ()
    diagonal = matrix.diagonal()
    if (diagonal < 0).any():
        return ()
    return tuple(zip(labels, np.sqrt(diagonal).tolist(), strict=True))


def covariance_ellipse(state: dict, plane: str = "xy", sigma: float = 1.0) -> np.ndarray | None:
    """A marginal model-covariance ellipse in the state's labelled Cartesian frame.

    ``sigma`` scales principal standard deviations; this is not an accuracy or
    calibrated confidence guarantee. Radial states have no Cartesian ellipse.
    """
    axes = {"xy": ("x_m", "y_m"), "xz": ("x_m", "z_m"), "yz": ("y_m", "z_m")}[plane]
    labels = tuple(state.get("state_labels", ()))
    if state.get("covariance") is None or not all(axis in labels for axis in axes):
        return None
    covariance = np.asarray(state["covariance"], dtype=float)
    mean = np.asarray(state["mean"], dtype=float)
    if covariance.shape != (len(labels), len(labels)) or mean.shape != (len(labels),):
        return None
    indices = [labels.index(axis) for axis in axes]
    matrix = covariance[np.ix_(indices, indices)]
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if (eigenvalues < -1e-9).any():
        return None
    angles = np.linspace(0, 2 * np.pi, 65)
    circle = np.stack((np.cos(angles), np.sin(angles)))
    return (
        mean[indices]
        + (eigenvectors @ (np.sqrt(np.maximum(eigenvalues, 0))[:, None] * circle)).T * sigma
    )
