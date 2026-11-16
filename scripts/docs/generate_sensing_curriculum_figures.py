#!/usr/bin/env python3
"""Plot recorded tutorial products without reprocessing detection or tracking.

Run from an installed ORCHAV checkout. Each invocation writes a PNG and a JSON
receipt to a new or existing output directory, refusing to replace either file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shlex
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.sensing.loader import load_frames_sensing, load_saved_results
from shared.sensing.plotting.limits import range_doppler_zoom_limits


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_db(power: np.ndarray) -> np.ndarray:
    peak = float(np.max(power))
    if peak <= 0:
        raise ValueError("Tutorial illustration requires a nonzero saved return")
    return 10 * np.log10(np.maximum(power / peak, 1e-8))


def draw(scenario: Path, output: Path, name: str) -> dict:
    """Read producer-authored axes, powers and estimates; never rerun a processor."""
    image_path = output / f"{name}.png"
    receipt_path = output / f"{name}.json"
    if image_path.exists() or receipt_path.exists():
        raise FileExistsError(f"Choose new outputs for {name}")
    frames = load_frames_sensing(scenario / "frames")
    results = load_saved_results(scenario)
    if not results:
        raise ValueError("No complete typed sensing results")
    last = results[-1]
    sensing = frames[-1]["_typed_sensing"]
    times = [r["temporal"]["result"]["reference_time_s"] for r in results]
    has_tracks = any(d.get("track_id") is not None for r in results for d in r["detections"])
    angular = frames[-1].get("_angular_product")
    show_elevation = (
        not has_tracks and angular is not None and len(angular.elevation_direction_cosine) > 1
    )
    fig, axes = plt.subplots(
        1,
        3 if show_elevation else 2,
        figsize=(15, 4) if show_elevation else (11, 4),
        layout="constrained",
    )
    if has_tracks:
        for ax, key, track_key, truth_key, label in zip(
            axes,
            ("range_m", "velocity_m_s"),
            ("track_range_m", "track_velocity_m_s"),
            ("monostatic_equivalent_range_m", "monostatic_equivalent_velocity_m_s"),
            ("Range (m)", "Closing velocity (m/s)"),
        ):
            # Each target and saved track has its own series; never reuse the
            # first target's truth for a multi-object illustration.
            target_ids = sorted(
                {
                    target["target_id"]
                    for frame in frames
                    for target in frame["_typed_sensing"]["truth"]["targets"]
                }
            )
            for target_id in target_ids:
                values = [
                    next(
                        target["pair_observables"][0][truth_key]
                        for target in frame["_typed_sensing"]["truth"]["targets"]
                        if target["target_id"] == target_id
                    )
                    for frame in frames
                ]
                # Truth IDs and tracker IDs have independent orderings. Neutral
                # reference lines avoid implying a color-based association.
                ax.plot(
                    times,
                    values,
                    "+--",
                    color="0.35",
                    label=f"{target_id}: reference",
                    alpha=0.8,
                )
            track_ids = sorted(
                {
                    d["track_id"]
                    for result in results
                    for d in result["detections"]
                    if d.get("track_id") is not None
                }
            )
            for track_id in track_ids:
                samples = [
                    (times[i], d)
                    for i, result in enumerate(results)
                    for d in result["detections"]
                    if d.get("track_id") == track_id
                ]
                color = f"C{track_ids.index(track_id) % 10}"
                ax.scatter(
                    [time for time, d in samples],
                    [d[key] for time, d in samples],
                    marker="x",
                    color=color,
                    label=f"Track {track_id}: detections",
                )
                for confirmed in (False, True):
                    subset = [(time, d) for time, d in samples if d["track_confirmed"] == confirmed]
                    if subset:
                        ax.scatter(
                            [time for time, d in subset],
                            [d[track_key] for time, d in subset],
                            marker="D",
                            facecolors=color if confirmed else "none",
                            edgecolors=color,
                            label=f"Track {track_id}: {'confirmed' if confirmed else 'tentative'}",
                        )
            ax.set(xlabel="Result reference time (s)", ylabel=label)
            ax.legend(fontsize=8)
    else:
        r = np.asarray(last["range_axis"])
        velocity = (
            np.asarray(last["doppler_axis"]) * 299792458 / float(sensing["carrier_freq_hz"]) / 2
        )
        power = np.asarray(last["rd_map"])
        automatic = range_doppler_zoom_limits(
            relative_db(power),
            r,
            velocity,
            extra_points=[(d["range_m"], d["velocity_m_s"]) for d in last["detections"]],
        )
        if automatic is None:
            automatic = ((float(r[0]), float(r[-1])), (float(velocity[0]), float(velocity[-1])))
        range_limits = list(sensing["config_resolved"].get("display_range_xlim") or automatic[0])
        velocity_limits = sensing["config_resolved"].get("display_velocity_ylim") or automatic[1]
        # Keep every published report visible when the tutorial's target detail
        # bounds would otherwise hide weak detections outside the main peak.
        padding = float(r[1] - r[0]) * 3 if len(r) > 1 else 0.1
        for detection in last["detections"]:
            range_limits[0] = min(range_limits[0], detection["range_m"] - padding)
            range_limits[1] = max(range_limits[1], detection["range_m"] + padding)
        mesh = axes[0].pcolormesh(
            r, velocity, relative_db(power), shading="nearest", vmin=-50, vmax=0, cmap="viridis"
        )
        axes[0].set(
            xlim=range_limits,
            ylim=velocity_limits,
            xlabel="Monostatic range (m)",
            ylabel="Closing velocity (m/s)",
            title=f"Saved range–Doppler · {len(last['detections'])} reports",
        )
        for d in last["detections"]:
            axes[0].plot(d["range_m"], d["velocity_m_s"], "x", color="white")
        fig.colorbar(mesh, ax=axes[0], label="dB relative to this product's peak")
        if angular is None:
            profile = sensing.get("range_profile_by_pair")
            if profile is not None:
                profile_power = np.asarray(profile)[0]
                profile_title = "Saved range profile: last acquisition sample"
            else:
                profile_power = power.sum(axis=0)
                profile_title = "Range power (sum over saved Doppler bins)"
            axes[1].plot(r, relative_db(profile_power), linewidth=1)
            axes[1].set(
                xlim=range_limits,
                ylim=(-60, 3),
                xlabel="Monostatic range (m)",
                ylabel="Relative power (dB)",
                title=profile_title,
            )
        else:
            cube = np.asarray(angular.power_linear)
            uy = np.asarray(angular.azimuth_direction_cosine)
            uz = np.asarray(angular.elevation_direction_cosine)
            if len(uz) == 1:
                physical = np.abs(uy) < 1
                azimuth = np.rad2deg(np.arcsin(uy[physical]))
                projection = cube.max(axis=(0, 2))[physical]
                mesh = axes[1].pcolormesh(
                    r, azimuth, relative_db(projection), shading="nearest", vmin=-40, vmax=0
                )
                axes[1].set(
                    xlim=(3.5, 6),
                    xlabel="Range (m)",
                    ylabel="Broadside azimuth projection (deg)",
                    title="Maximum over Doppler and elevation",
                )
            else:
                projection = cube.max(axis=(2, 3))
                physical = uz[:, None] ** 2 + uy[None, :] ** 2 < 1
                values = np.ma.masked_where(~physical, relative_db(projection))
                mesh = axes[1].pcolormesh(uy, uz, values, shading="nearest", vmin=-40, vmax=0)
                axes[1].set(
                    xlabel="Horizontal direction cosine",
                    ylabel="Vertical direction cosine",
                    title="Maximum over range and Doppler",
                    xlim=(-1, 1),
                    ylim=(-1, 1),
                    aspect="equal",
                )
            fig.colorbar(mesh, ax=axes[1], label="dB relative to this projection's peak")
        if show_elevation:
            truth = [
                f["_typed_sensing"]["truth"]["targets"][0]["pair_observables"][0][
                    "receiver_elevation_deg"
                ]
                for f in frames
            ]
            axes[2].plot(times, truth, "+--", color="0.4", label="Cube-center bearing")
            for i, result in enumerate(results):
                for detection in result["detections"]:
                    axes[2].plot(
                        times[i],
                        detection["aoa_el_deg"],
                        "o",
                        color="tab:blue",
                        label="Saved beam estimate" if i == 0 else None,
                    )
            axes[2].set(
                xlabel="Result reference time (s)",
                ylabel="Elevation (deg)",
                title="Elevation over four acquisitions",
            )
            axes[2].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle(f"{name.replace('_', ' ').title()} · saved observations")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    source_files = [scenario / "scenario.yaml", *sorted((scenario / "frames").glob("*"))]
    source_files = [p for p in source_files if p.is_file()]
    root = Path(__file__).resolve().parents[2]
    tutorial = next((root / "scenarios/sensing").glob(f"*/{name}/scenario.yaml"))
    receipt = {
        "license_spdx": "GPL-3.0-only",
        "provenance": "ORCHAV project-generated tutorial illustration",
        "tutorial_scenario": tutorial.relative_to(root).as_posix(),
        "tutorial_scenario_sha256": sha256(tutorial),
        "image": image_path.name,
        "sha256": sha256(image_path),
        "kind": "plot_of_saved_native_products_no_reprocessing",
        "scenario_name": name,
        "command": shlex.join(
            [
                "python",
                "-m",
                "scripts.docs.generate_sensing_curriculum_figures",
                "--scenario",
                scenario.as_posix(),
                "--name",
                name,
                "--output-dir",
                output.as_posix(),
            ]
        ),
        "reproduction": {
            "working_directory": "repository root",
            "commands": [
                f"python -m generator {tutorial.parent.relative_to(root).as_posix()}",
                f"python -m scripts.docs.generate_sensing_curriculum_figures --scenario {tutorial.parent.relative_to(root).as_posix()} --name {name} --output-dir analysis/tutorial-figures",
            ],
            "scope": "Fresh acquisition from the authored configuration; original execution and input identities are recorded separately. Use a fresh figure output directory.",
        },
        "tooling": {
            n: importlib.metadata.version(n)
            for n in ("orchav", "sionna-rt", "mitsuba", "numpy", "matplotlib")
        },
        "source_hashes": {p.relative_to(scenario).as_posix(): sha256(p) for p in source_files},
        "plotter_sha256": sha256(Path(__file__)),
        "source_module_hashes": {
            p.relative_to(root).as_posix(): sha256(p)
            for p in [
                *sorted((root / "generator/core/sensing").rglob("*.py")),
                root / "shared/sensing/loader.py",
                root / "shared/sensing/tracker.py",
            ]
        },
        "frame_set_id": last["_source_frame_set_id"],
        "generation_id": last["_source_generation_id"],
        "results": [
            {
                "storage_frame": r["_frame_index"],
                "temporal": r["temporal"],
                "detection_count": len(r["detections"]),
            }
            for r in results
        ],
        "processing_recipe": sensing["config_resolved"]["processing_recipe"],
        "limits": "One generated run; figures show saved outputs, not general detection or localization performance.",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    draw(args.scenario, args.output_dir, args.name)


if __name__ == "__main__":
    main()
