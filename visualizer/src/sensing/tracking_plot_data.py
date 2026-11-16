"""Reusable plot arrays built from saved estimates and time-aligned truth."""

from dataclasses import dataclass, field

import numpy as np

from .display_style import detection_color


@dataclass(frozen=True)
class TimedValues:
    """Recorded samples whose prefix can be selected without rebuilding history."""

    times: np.ndarray
    values: np.ndarray

    def through(self, time_s):
        stop = np.searchsorted(self.times, time_s + 1e-12, side="right")
        return self.times[:stop], self.values[:stop]

    def projected(self, plane="xy"):
        """Select a Cartesian plane without transforming the recorded coordinates."""
        indices = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
        return TimedValues(self.times, self.values[:, indices])


def _series(rows, width):
    times = np.asarray([row[0] for row in rows], dtype=float)
    values = np.asarray([row[1] for row in rows], dtype=float).reshape(-1, width)
    times.setflags(write=False)
    values.setflags(write=False)
    return TimedValues(times, values)


@dataclass(frozen=True)
class TrackingPlotData:
    """Plot coordinates only; no detector or tracker is run during preparation."""

    tracks: dict[str, TimedValues]
    truths: dict[str, TimedValues]
    ranges: dict[str, TimedValues]
    detections: TimedValues
    detection_colors: tuple[str, ...]
    estimates: dict[str, TimedValues] = field(default_factory=dict)
    statuses: dict[str, TimedValues] = field(default_factory=dict)
    spatial_dimensions: int = 0


def prepare_tracking_plot_data(history):
    """Group saved points once, preserving the existing per-epoch track mean."""
    tracks, truths, ranges, detections = {}, {}, {}, []
    estimates, statuses = {}, {}
    has_saved_z = False
    colors = []
    for record in sorted(history.records, key=lambda row: row.time_s):
        per_track = {}
        for detection in record.detections:
            tid = detection.get("track_id")
            point = detection.get("world_track_position")
            if tid is not None and point is not None:
                per_track.setdefault(str(tid), []).append(point)
                has_saved_z |= (
                    detection.get("track_z_m") is not None
                    or detection.get("source_kind") == "saved_world_track"
                    or (
                        detection.get("track_x_m") is None
                        and any(detection.get(key) is not None for key in ("z_m", "aoa_el_deg"))
                    )
                )
            detections.append(
                (record.time_s, (detection.get("range_m"), detection.get("velocity_m_s")))
            )
            colors.append(detection_color(detection))
        states = getattr(record, "tracking_states", ())
        if not states:
            # Older overview caches still contain emitted filtered state fields.
            from .tracking_presentation import tracking_snapshot

            states = tracking_snapshot({"detections": record.detections}).states
        for state in states:
            tid = str(state["track_id"])
            has_saved_z |= "z_m" in state.get("state_labels", ()) or (
                state.get("position_frame") == "world"
                and state.get("position_m") is not None
                and len(state["position_m"]) == 3
            )
            if state.get("world_position_m") is not None:
                per_track[tid] = [state["world_position_m"]]
            elif state.get("position_frame") == "world" and state.get("position_m") is not None:
                per_track[tid] = [state["position_m"]]
            r, v = state.get("estimated_range_m"), state.get("radial_velocity_m_s")
            if r is not None or v is not None:
                estimates.setdefault(tid, []).append((record.time_s, (r, v)))
            statuses.setdefault(tid, []).append(
                (
                    record.time_s,
                    (
                        state.get("predicted_only"),
                        state.get("confirmed"),
                    ),
                )
            )
        for tid, points in per_track.items():
            tracks.setdefault(tid, []).append((record.time_s, np.mean(points, axis=0)))
        for tid, point in record.truth_positions:
            truths.setdefault(tid, []).append((record.time_s, point))
        if record.truth is not None:
            gt = record.truth
            for tid, r, v in zip(
                gt["target_ids"], gt["target_ranges"], gt["target_speeds"], strict=True
            ):
                ranges.setdefault(tid, []).append((record.time_s, (r, v)))
    return TrackingPlotData(
        tracks={tid: _series(rows, 3) for tid, rows in tracks.items()},
        truths={tid: _series(rows, 3) for tid, rows in truths.items()},
        ranges={tid: _series(rows, 2) for tid, rows in ranges.items()},
        detections=_series(detections, 2),
        detection_colors=tuple(colors),
        estimates={tid: _series(rows, 2) for tid, rows in estimates.items()},
        statuses={tid: _series(rows, 2) for tid, rows in statuses.items()},
        spatial_dimensions=3 if has_saved_z and tracks else 2 if tracks else 0,
    )
