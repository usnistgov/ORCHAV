"""Standard summaries of recorded receiver-local Cartesian track states."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from shared.sensing.plotting.style import PALETTE, sensing_figure_style


def _colocated_truth_pair(sensing: Mapping[str, Any], scene: Mapping[str, Any] | None) -> bool:
    """Require pair-matched device positions at the exact recorded truth epoch.

    A monostatic range convention alone does not prove physical co-location.
    A coherent output frame can describe a later scene than its solve-start
    truth; that frame is insufficient evidence for this optional overlay.
    """
    if scene is None:
        return False
    temporal = sensing.get("temporal", {})
    truth_time = temporal.get("truth", {}).get("reference_time_s")
    scene_time = scene.get("_frame_timestamp_s", scene.get("scene_time_s"))
    if truth_time is None or scene_time != truth_time:
        return False
    acquisition = temporal.get("acquisition", {})
    pairs = sensing.get("pair_axis")
    if not isinstance(pairs, list) or len(pairs) != 1:
        return False
    pair = pairs[0]
    if any(
        pair.get(key) != acquisition.get(key) for key in ("pair_id", "tx_sensor_id", "rx_sensor_id")
    ):
        return False
    positions = []
    for device in ("tx", "rx"):
        native_key = f"{device}_positions_snapshot"
        native = native_key in scene
        array = np.asarray(scene.get(native_key if native else f"{device}_positions"), dtype=float)
        index = pair.get(f"{device}_device_index" if native else "frame_pair_index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or array.ndim != 2
            or array.shape[1:] != (3,)
            or not 0 <= index < len(array)
            or not np.isfinite(array[index]).all()
        ):
            return False
        positions.append(array[index])
    return bool(np.allclose(positions[0], positions[1], rtol=0, atol=1e-9))


def cartesian_tracking_event(
    sensing: Mapping[str, Any], *, scene_context: Mapping[str, Any] | None = None
) -> dict | None:
    """Keep published positions and same-epoch receiver-local truth separately.

    Track coordinates must be explicit Cartesian state fields. A radial track
    ID attached to a localized detection does not supply a Cartesian state.
    Truth uses the recorded receiver-local angular reference and monostatic
    center range only with same-epoch device co-location evidence; world
    positions are never mixed with receiver-local estimates.
    """
    result = sensing.get("temporal", {}).get("result")
    if not isinstance(result, Mapping):
        return None
    time_s = float(result["reference_time_s"])
    if not np.isfinite(time_s):
        raise ValueError("Cartesian tracking summaries require finite result times")
    detections, tracks = [], []
    for detection in sensing.get("detections") or []:
        raw = np.asarray([detection.get(f"{axis}_m", np.nan) for axis in "xyz"], dtype=float)
        if np.isfinite(raw[:2]).all():
            detections.append(raw.tolist())
        state = np.asarray(
            [detection.get(f"track_{axis}_m", np.nan) for axis in "xyz"], dtype=float
        )
        if detection.get("track_id") is not None and np.isfinite(state[:2]).all():
            tracks.append(
                {
                    "track_id": int(detection["track_id"]),
                    "position_m": state.tolist(),
                    "confirmed": bool(detection.get("track_confirmed", False)),
                }
            )
    truth = []
    if sensing.get("config_resolved", {}).get(
        "range_mode"
    ) == "monostatic" and _colocated_truth_pair(sensing, scene_context):
        from shared.sensing.angular import angular_event

        angular = angular_event(sensing)
        for target in (angular or {}).get("targets", []):
            azimuth, elevation = np.radians([target["azimuth_deg"], target["elevation_deg"]])
            distance = float(target["range_m"])
            position = distance * np.asarray(
                [
                    np.cos(elevation) * np.cos(azimuth),
                    np.cos(elevation) * np.sin(azimuth),
                    np.sin(elevation),
                ]
            )
            if np.isfinite(position).all():
                truth.append({"target_id": target["target_id"], "position_m": position.tolist()})
    temporal = sensing["temporal"]
    return {
        "time_s": time_s,
        "pair_id": temporal["acquisition"]["pair_id"],
        "acquisition_id": temporal["acquisition"]["acquisition_id"],
        "scene_frame_id": temporal.get("display_binding", {}).get("scene_frame_id"),
        "coordinate_frame": "rx_array_local",
        "tracks": tracks,
        "detections": detections,
        "truth": truth,
    }


def _position_histories(events: Sequence[Mapping[str, Any]]):
    """Align identities to every result epoch, retaining missing-state gaps."""
    if not events:
        raise ValueError("No Cartesian tracking observations are available")
    times = np.asarray([event["time_s"] for event in events], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Cartesian tracking result times must be finite and increase strictly")
    if any(
        event["pair_id"] != events[0]["pair_id"] or event["coordinate_frame"] != "rx_array_local"
        for event in events
    ):
        raise ValueError("Cartesian tracking history requires one receiver-local sensor pair")
    tracks, truth = {}, {}
    for index, event in enumerate(events):
        for source, key, destination in (
            ("tracks", "track_id", tracks),
            ("truth", "target_id", truth),
        ):
            seen = set()
            for entry in event[source]:
                identity = entry[key]
                if identity in seen:
                    raise ValueError(f"Cartesian tracking event repeats {key}")
                seen.add(identity)
                record = destination.setdefault(
                    identity,
                    {
                        "positions": np.full((len(events), 3), np.nan),
                        "confirmed": np.zeros(len(events), dtype=bool),
                    },
                )
                position = np.asarray(entry["position_m"], dtype=float)
                if position.shape != (3,):
                    raise ValueError("Cartesian history positions require three coordinates")
                record["positions"][index] = position
                record["confirmed"][index] = entry.get("confirmed", True)
    return times, tracks, truth


@sensing_figure_style
def create_cartesian_tracking_history_figure(
    events: Sequence[Mapping[str, Any]], output_path: Path
) -> Path:
    """Plot saved X/Y/Z states and reports against physical result timestamps."""
    from shared.sensing.plotting.viz import SUMMARY_FIGURE_DPI, plt

    times, tracks, truth = _position_histories(events)
    if not any(np.isfinite(track["positions"]).all(axis=1).any() for track in tracks.values()):
        raise ValueError("No recorded three-dimensional Cartesian track states are available")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for dimension, axis in enumerate(axes):
        reports = [
            (event["time_s"], p[dimension])
            for event in events
            for p in event["detections"]
            if np.isfinite(p[dimension])
        ]
        if reports:
            axis.scatter(
                *np.asarray(reports).T,
                color="#999999",
                s=15,
                alpha=0.55,
                label="Reported detections",
            )
        for index, (identity, series) in enumerate(sorted(truth.items())):
            axis.plot(
                times,
                series["positions"][:, dimension],
                linestyle=":",
                linewidth=1.7,
                color=PALETTE[index % len(PALETTE)],
                label=f"GT {identity}",
            )
        for index, (identity, series) in enumerate(sorted(tracks.items())):
            values = series["positions"][:, dimension]
            color = PALETTE[(index + 2) % len(PALETTE)]
            axis.plot(times, values, linestyle="--", linewidth=1, alpha=0.6, color=color)
            for sample in range(len(times) - 1):
                if series["confirmed"][sample]:
                    axis.plot(
                        times[sample : sample + 2],
                        values[sample : sample + 2],
                        color=color,
                        linewidth=1.8,
                    )
            axis.scatter(times, values, color=color, s=14, label=f"Track {identity}", zorder=4)
        axis.set_ylabel(f"Receiver-local {'XYZ'[dimension]} (m)")
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8, ncol=2)
    axes[0].set_title("Cartesian tracking · recorded positions")
    axes[-1].set_xlabel("Result reference time (s)")
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=SUMMARY_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_cartesian_tracking_figures(
    events: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    figures: set[str],
    output_format: str = "png",
) -> list[Path]:
    """Dispatch receiver-local histories through the shared spatial renderers."""
    from shared.sensing.plotting.viz import create_xy_tracking_figure, create_xyz_tracking_figure

    times, tracks, truth = _position_histories(events)

    def positions(series):
        return [
            {
                "frame": index,
                "reference_time_s": float(time),
                **{f"{axis}_m": float(position[dimension]) for dimension, axis in enumerate("xyz")},
            }
            for index, (time, position) in enumerate(zip(times, series["positions"], strict=True))
        ]

    track_history = {identity: positions(series) for identity, series in tracks.items()}
    gt_history = {identity: positions(series) for identity, series in truth.items()}
    reports = [
        {f"{axis}_m": float(position[index]) for index, axis in enumerate("xyz")}
        for event in events
        for position in event["detections"]
    ]
    output_dir = Path(output_dir)
    generated = []
    if "tracking_cartesian" in figures:
        generated.append(
            create_cartesian_tracking_history_figure(
                events, output_dir / f"tracking_cartesian.{output_format}"
            )
        )
    if "tracking_xy" in figures:
        path = output_dir / f"tracking_xy.{output_format}"
        create_xy_tracking_figure(
            track_history,
            path,
            gt_tracks=gt_history,
            raw_detections=reports,
            coordinate_frame="receiver_local",
            show_detections=True,
        )
        generated.append(path)
    if "tracking_xyz" in figures:
        if not any(np.isfinite(track["positions"]).all(axis=1).any() for track in tracks.values()):
            raise ValueError("No recorded three-dimensional Cartesian track states are available")
        path = output_dir / f"tracking_xyz.{output_format}"
        create_xyz_tracking_figure(
            track_history,
            gt_tracks=gt_history,
            output_path=path,
            coordinate_frame="receiver_local",
            raw_detections=reports,
            connect_tracks=True,
        )
        generated.append(path)
    return generated
