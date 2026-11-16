"""Receiver-local angular truth and diagnostics for native sensing results.

Angles refer to the target center at the result's truth epoch, not the strongest
scattering point. Ground truth is used only in these evaluation functions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from shared.sensing.evaluation import associate_detections_to_gt


def receiver_local_angles(position, receiver_position, orientation_rad) -> tuple[float, float]:
    """Return azimuth/elevation in degrees using world-from-local Rz Ry Rx.

    Orientation is yaw, pitch, roll in radians. Positive azimuth points toward
    local +Y; positive elevation points toward local +Z. Coincident positions
    and the azimuth singularity at the poles have no finite azimuth.
    """
    values = [np.asarray(v, dtype=float) for v in (position, receiver_position, orientation_rad)]
    if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in values):
        raise ValueError("Angular truth requires finite three-dimensional positions/orientation")
    yaw, pitch, roll = values[2]
    cy, sy, cp, sp, cr, sr = (
        np.cos(yaw),
        np.sin(yaw),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(roll),
        np.sin(roll),
    )
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    x, y, z = rotation.T @ (values[0] - values[1])
    horizontal = np.hypot(x, y)
    if horizontal <= 1e-12:
        raise ValueError("Azimuth is undefined for a coincident or polar target")
    return float(np.degrees(np.arctan2(y, x))), float(np.degrees(np.arctan2(z, horizontal)))


def angular_truth_fields(frame: Mapping[str, Any], position, receiver_index: int = 0) -> dict:
    """Use only the explicitly supplied truth-epoch runtime sensor snapshot."""
    positions = frame.get("rx_positions_snapshot")
    orientations = frame.get("rx_orientations_snapshot")
    if positions is None or orientations is None:
        return {}
    try:
        azimuth, elevation = receiver_local_angles(
            position, positions[receiver_index], orientations[receiver_index]
        )
    except (ValueError, IndexError):
        return {}
    return {
        "receiver_azimuth_deg": azimuth,
        "receiver_elevation_deg": elevation,
        "angular_reference_frame": "rx_array_local",
    }


def angular_event(sensing: Mapping[str, Any]) -> dict | None:
    """Extract a compact event without inventing angles for older recordings."""
    temporal = sensing.get("temporal", {})
    truth_time = temporal.get("truth", {}).get("reference_time_s")
    result_time = temporal.get("result", {}).get("reference_time_s")
    if truth_time is None or result_time is None or truth_time != result_time:
        return None
    array = sensing.get("rx_array", {})
    if max(int(array.get("columns", 1)), int(array.get("rows", 1))) <= 1:
        return None
    pair_id = temporal.get("acquisition", {}).get("pair_id")
    cfg = sensing["config_resolved"]
    range_key = (
        "monostatic_equivalent_range_m"
        if cfg["range_mode"] == "monostatic"
        else "bistatic_path_length_m"
    )
    targets = []
    for target in sensing.get("truth", {}).get("targets", []):
        if not target.get("valid") or target.get("lifecycle") != "active":
            continue
        observations = [o for o in target["pair_observables"] if o["pair_id"] == pair_id]
        if len(observations) != 1:
            continue
        observable = observations[0]
        if observable.get("angular_reference_frame") != "rx_array_local":
            continue
        targets.append(
            {
                "target_id": target["target_id"],
                "azimuth_deg": observable["receiver_azimuth_deg"],
                "elevation_deg": observable["receiver_elevation_deg"],
                "range_m": observable[range_key],
                # Detection velocity is lambda*f_D/2 in both range modes.
                "velocity_m_s": observable["monostatic_equivalent_velocity_m_s"],
                "velocity_valid": observable["velocity_valid"],
            }
        )
    if not targets:
        return None
    return {
        "time_s": float(result_time),
        "targets": targets,
        "detections": [dict(d) for d in sensing.get("detections", []) if "aoa_az_deg" in d],
        "azimuth_resolved": int(array.get("columns", 1)) > 1,
        "elevation_resolved": int(array.get("rows", 1)) > 1,
        "range_gate_m": float(cfg["eval_association_range_gate_m"]),
        "velocity_gate_m_s": float(cfg["eval_association_velocity_gate_m_s"]),
    }


def angular_residuals(events: Sequence[Mapping[str, Any]]) -> dict:
    """Match in RD only, then measure wrapped angular error independently."""
    rows = []
    unmatched = 0
    missed = 0
    for event in events:
        truth = [t for t in event["targets"] if t["velocity_valid"]]
        ds = event["detections"]
        association = associate_detections_to_gt(
            np.array([t["range_m"] for t in truth]),
            np.array([t["velocity_m_s"] for t in truth]),
            np.array([d["range_m"] for d in ds]),
            np.array([d["velocity_m_s"] for d in ds]),
            range_gate=event["range_gate_m"],
            velocity_gate=event["velocity_gate_m_s"],
        )
        unmatched += association.false_positives
        missed += association.false_negatives
        for i, j in association.gt_to_det.items():
            if j is None:
                continue
            t, d = truth[i], ds[j]
            rows.append(
                {
                    "time_s": event["time_s"],
                    "target_id": t["target_id"],
                    "range_m": t["range_m"],
                    "azimuth_error_deg": (d["aoa_az_deg"] - t["azimuth_deg"] + 180) % 360 - 180,
                    "elevation_error_deg": (
                        float(d["aoa_el_deg"] - t["elevation_deg"]) if "aoa_el_deg" in d else None
                    ),
                }
            )
    return {
        "matches": rows,
        "unmatched_reports": unmatched,
        "missed_targets": missed,
        "association": "range_velocity_only; angles do not affect matching",
    }


def create_angular_summary(events: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    """Write all-estimate/GT angular timelines, error plots and numeric evidence."""
    import json

    import matplotlib.pyplot as plt

    if not events:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = sorted({t["target_id"] for e in events for t in e["targets"]})
    colors = {name: f"C{i % 10}" for i, name in enumerate(names)}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for ax, field, key, label, resolved in zip(
        axes,
        ("azimuth_deg", "elevation_deg"),
        ("aoa_az_deg", "aoa_el_deg"),
        ("Azimuth", "Elevation"),
        ("azimuth_resolved", "elevation_resolved"),
        strict=True,
    ):
        for name in names:
            xy = [
                (e["time_s"], t[field])
                for e in events
                for t in e["targets"]
                if t["target_id"] == name
            ]
            # Break at the wrap boundary rather than drawing an artificial 360° jump.
            x, y = np.array(xy).T
            y = y.copy()
            y[1:][abs(np.diff(y)) > 180] = np.nan
            ax.plot(x, y, color=colors[name], label=f"{name} center GT")
        estimates = [(e["time_s"], d[key]) for e in events for d in e["detections"] if key in d]
        if estimates:
            x, y = np.array(estimates).T
            measured = all(e[resolved] for e in events)
            ax.scatter(
                x,
                y,
                marker="x",
                s=22,
                color="black",
                label="All estimates" if measured else "Reported assumption (axis unresolved)",
            )
        else:
            ax.text(0.5, 0.5, "No angular estimates saved", transform=ax.transAxes, ha="center")
        ax.set(ylabel=f"{label} (degrees, RX local)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncols=2)
    axes[-1].set_xlabel("Sensing result reference time (s)")
    fig.suptitle("Angular estimates versus target-center ground truth")
    timeline = output_dir / "angular_tracking.png"
    fig.savefig(timeline, dpi=160)
    plt.close(fig)

    evidence = angular_residuals(events)
    evidence["events"] = list(events)
    numeric = output_dir / "angular_summary.json"
    numeric.write_text(json.dumps(evidence, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, key, label, resolved in zip(
        axes,
        ("azimuth_error_deg", "elevation_error_deg"),
        ("Azimuth", "Elevation"),
        ("azimuth_resolved", "elevation_resolved"),
        strict=True,
    ):
        for name in names:
            rows = [r for r in evidence["matches"] if r["target_id"] == name and r[key] is not None]
            ax.scatter(
                [r["range_m"] for r in rows],
                [r[key] for r in rows],
                s=16,
                color=colors[name],
                label=name,
            )
        ax.axhline(0, color="gray", linestyle="--")
        suffix = "" if all(e[resolved] for e in events) else " (reported assumption)"
        ax.set(xlabel="GT range (m)", ylabel=f"{label} error (degrees)", title=label + suffix)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"RD-matched angular errors: {len(evidence['matches'])} matches; {evidence['unmatched_reports']} unmatched reports\nAngles do not affect matching; extended targets need not return from their centers",
        fontsize=10,
    )
    errors = output_dir / "angular_errors.png"
    fig.savefig(errors, dpi=160)
    plt.close(fig)
    return [timeline, errors, numeric]
