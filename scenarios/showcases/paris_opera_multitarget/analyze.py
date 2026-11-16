"""Evaluate saved radar reports in receiver-local 3D coordinates.

Hungarian assignment uses target-centre truth only here, after detection and
tracking. Extended cars receive a 3 m gate, other meshes a 1.8 m gate, with a
common 0.35 m/s bulk-velocity gate. All unmatched reports are counted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from shared.frames.contracts import FrameComponent, FrameReadRequest
from shared.frames.providers import Hdf5Provider
from shared.sensing.angular import angular_event

COLORS = {
    "Car_Outbound": "#4999ce",
    "Car_Inbound": "#e8ad48",
    "Scooter": "#25b9a1",
    "Pedestrian": "#e9746d",
    "Drone": "#a891e8",
}


def local_point(target):
    a, e = np.deg2rad([target["azimuth_deg"], target["elevation_deg"]])
    return target["range_m"] * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def review(scenario, output, reports_path=None):
    overrides = (
        {}
        if reports_path is None
        else {row["frame"]: row for row in json.loads(reports_path.read_text())["epochs"]}
    )
    provider = Hdf5Provider(scenario)
    request = FrameReadRequest(
        components=frozenset({FrameComponent.SENSING, FrameComponent.PROVENANCE}),
        sensing_products=frozenset({"products"}),
    )
    rows = []
    for step in provider.list_frames():
        if (step + 1) % 128:
            continue
        sensing = provider.load_frame_projection(step, request).frame.sensing
        if not sensing:
            continue
        event = angular_event(sensing)
        if event is None:
            continue
        if reports_path is not None:
            derived = overrides[int(step)]
            fingerprint = hashlib.sha256(
                json.dumps(sensing["detections"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if derived["input_sha256"] != fingerprint:
                raise ValueError("Derived reports belong to a different detector input")
            if abs(derived["time_s"] - event["time_s"]) > 1.0e-9:
                raise ValueError("Derived report time differs from the saved CPI reference")
            event["detections"] = derived["detections"]
        targets, detections = event["targets"], event["detections"]
        cost = np.full((len(targets), len(detections)), 1.0e6)
        for i, target in enumerate(targets):
            gate = 3.0 if target["target_id"].startswith("Car") else 1.8
            point = local_point(target)
            target["local_position_m"] = point.tolist()
            for j, detection in enumerate(detections):
                estimate = np.array([detection[k] for k in ("x_m", "y_m", "z_m")])
                distance = np.linalg.norm(estimate - point)
                dv = abs(detection["velocity_m_s"] - target["velocity_m_s"])
                if distance <= gate and dv <= 0.35:
                    cost[i, j] = (distance / gate) ** 2 + (dv / 0.35) ** 2
        assigned = []
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] >= 1.0e6:
                continue
            target, detection = targets[i], detections[j]
            estimate = np.array([detection[k] for k in ("x_m", "y_m", "z_m")])
            assigned.append(
                dict(
                    target=target["target_id"],
                    detection_index=int(j),
                    track_id=detection.get("track_id"),
                    confirmed=detection.get("track_confirmed", False),
                    position_error_m=float(np.linalg.norm(estimate - target["local_position_m"])),
                    velocity_error_m_s=float(detection["velocity_m_s"] - target["velocity_m_s"]),
                )
            )
        rows.append(
            dict(
                frame=int(step),
                time_s=event["time_s"],
                targets=targets,
                detections=detections,
                matches=assigned,
                false_reports=len(detections) - len(assigned),
            )
        )
    provider.close()
    if not rows:
        raise ValueError("No complete coherent intervals")
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(
        epochs=len(rows),
        observed_duration_s=rows[-1]["time_s"] + 0.064,
        reports_source=str(reports_path) if reports_path else "source HDF5 reports",
        gates=dict(car_position_m=3.0, other_position_m=1.8, velocity_m_s=0.35),
        targets={},
        total_unmatched_reports=sum(row["false_reports"] for row in rows),
    )
    for name in COLORS:
        matches = [m for row in rows for m in row["matches"] if m["target"] == name]
        ids = [m["track_id"] for m in matches]
        summary["targets"][name] = dict(
            detected_epochs=len(matches),
            total_epochs=len(rows),
            confirmed_epochs=sum(m["confirmed"] for m in matches),
            track_ids=sorted(set(ids)),
            adjacent_id_changes=sum(a != b for a, b in zip(ids, ids[1:])),
            median_position_error_m=(
                float(np.median([m["position_error_m"] for m in matches])) if matches else None
            ),
            max_position_error_m=max((m["position_error_m"] for m in matches), default=None),
        )
    summary["matched_reports"] = sum(len(row["matches"]) for row in rows)
    summary["target_opportunities"] = len(rows) * len(COLORS)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "epochs.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return rows, summary


def plot(rows, output):
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "figure.facecolor": "#0c1727",
            "axes.facecolor": "#111f32",
            "text.color": "#e7edf4",
            "axes.labelcolor": "#c5d1e0",
            "xtick.color": "#aabbd0",
            "ytick.color": "#aabbd0",
            "axes.edgecolor": "#425168",
            "grid.color": "#425168",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    times = np.array([r["time_s"] for r in rows])
    for name, color in COLORS.items():
        truth = [next(t for t in row["targets"] if t["target_id"] == name) for row in rows]
        for ax, field in zip(axes.flat[:3], ["range_m", "velocity_m_s", "elevation_deg"]):
            ax.plot(times, [t[field] for t in truth], color=color, alpha=0.65, lw=1.4, ls="--")
        for row in rows:
            match = next((m for m in row["matches"] if m["target"] == name), None)
            if match:
                d = row["detections"][match["detection_index"]]
                for ax, field in zip(axes.flat[:3], ["range_m", "velocity_m_s", "aoa_el_deg"]):
                    ax.scatter(row["time_s"], d[field], s=10, color=color)
        matched = [
            (row["time_s"], m["position_error_m"])
            for row in rows
            for m in row["matches"]
            if m["target"] == name
        ]
        if matched:
            axes[1, 1].plot(*np.array(matched).T, color=color, lw=1.6, label=name.replace("_", " "))
    labelled = False
    for row in rows:
        matched_indices = {m["detection_index"] for m in row["matches"]}
        for index, detection in enumerate(row["detections"]):
            if index in matched_indices:
                continue
            for ax, field in zip(axes.flat[:3], ["range_m", "velocity_m_s", "aoa_el_deg"]):
                ax.scatter(
                    row["time_s"], detection[field], marker="x", color="#f2f6fa", s=55, zorder=8
                )
            point = np.array([detection[k] for k in ("x_m", "y_m", "z_m")])
            error = min(np.linalg.norm(point - local_point(target)) for target in row["targets"])
            axes[1, 1].scatter(
                row["time_s"],
                error,
                marker="x",
                color="#f2f6fa",
                s=55,
                label=None if labelled else "Unmatched report",
                zorder=8,
            )
            labelled = True
    for ax, label in zip(
        axes.flat,
        [
            "Slant range (m)",
            "Closing velocity (m/s)",
            "Receiver-local elevation (°)",
            "Scatter-point to target-centre error (m)",
        ],
    ):
        ax.set(xlabel="Simulation time (s)", ylabel=label)
        ax.grid(alpha=0.35)
    axes[1, 1].legend(fontsize=9, facecolor="#111f32", edgecolor="#425168")
    fig.suptitle(
        "PARIS OPÉRA  /  FIVE MESH TARGETS\nSaved radar estimates · dashed lines: target-centre reference",
        fontsize=17,
    )
    fig.savefig(output / "sensing_review.png", dpi=180)
    fig.savefig(output / "sensing_review.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reports-json", type=Path)
    args = parser.parse_args()
    output = args.output or args.scenario / "review"
    rows, summary = review(args.scenario, output, args.reports_json)
    plot(rows, output)
