"""Build a portable, synchronized demo from native captures and saved RF products."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from analyze import plot, review

from shared.frames.contracts import FrameComponent, FrameReadRequest
from shared.frames.providers import Hdf5Provider
from visualizer.src.sensing.display_style import identity_color


def encode(captures, output, encoder):
    metadata = json.loads((captures / "capture.json").read_text())
    sequence = metadata.get("sequence", [])
    if len(sequence) < 2:
        raise ValueError("Capture at least two movie frames with capture.py --movie-frames")
    subprocess.run(
        [
            str(encoder),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "25",
            "-i",
            str(captures / "sequence/%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output / "scene.mp4"),
        ],
        check=True,
    )
    return (sequence[1]["time_s"] - sequence[0]["time_s"]) * 25.0


def plots(scenario, rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "#0d192a",
            "axes.facecolor": "#0d192a",
            "text.color": "#e7edf4",
            "axes.labelcolor": "#bacbdd",
            "xtick.color": "#9fb3c9",
            "ytick.color": "#9fb3c9",
            "axes.edgecolor": "#3a4f67",
        }
    )
    directory = output / "plots"
    directory.mkdir(exist_ok=True)
    provider = Hdf5Provider(scenario)
    request = FrameReadRequest(
        components=frozenset({FrameComponent.SENSING}),
        sensing_products=frozenset(
            {
                "range_doppler_power_by_pair",
                "range_doppler_by_pair",
                "range_doppler_angle_by_pair",
                "range_axis_m",
                "doppler_axis_hz",
                "azimuth_broadside_angle_deg",
            }
        ),
    )
    spectra = []
    reference = dict(range_doppler=0.0, angular_scan=0.0)
    for row in rows:
        s = provider.load_frame_projection(row["frame"], request).frame.sensing
        if "range_doppler_power_by_pair" in s:
            rd = np.asarray(s["range_doppler_power_by_pair"], dtype=float)[0]
        else:
            rd_values = np.asarray(s["range_doppler_by_pair"])[0]
            rd = abs(rd_values) ** 2 if np.iscomplexobj(rd_values) else rd_values.astype(float)
        angle_values = np.asarray(s["range_doppler_angle_by_pair"])[0]
        cube = abs(angle_values) ** 2 if np.iscomplexobj(angle_values) else angle_values
        ra = np.max(cube, axis=(0, 2))
        reference["range_doppler"] = max(reference["range_doppler"], float(rd.max()))
        reference["angular_scan"] = max(reference["angular_scan"], float(ra.max()))
        spectra.append(
            (
                rd,
                ra.copy(),
                np.array(s["range_axis_m"]),
                np.array(s["doppler_axis_hz"]) * 299792458 / 28.0e9 / 2,
                np.array(s["azimuth_broadside_angle_deg"]),
            )
        )
        if len(spectra) % 10 == 0:
            print(f"Read radar products {len(spectra)}/{len(rows)}", flush=True)
    provider.close()
    for index, (row, (rd, ra, ranges, velocities, angles)) in enumerate(zip(rows, spectra)):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.2), layout="constrained")
        for ax, values, vertical, title, label, scale in (
            (
                axes[0],
                rd,
                velocities,
                "RANGE–DOPPLER",
                "Closing velocity (m/s)",
                reference["range_doppler"],
            ),
            (
                axes[1],
                ra,
                angles,
                "DIGITAL ANGULAR SCAN",
                "Horizontal array angle (°)",
                reference["angular_scan"],
            ),
        ):
            db = 10 * np.log10(np.maximum(values, scale * 1.0e-12) / scale)
            im = ax.pcolormesh(
                ranges,
                vertical,
                db,
                cmap="magma",
                vmin=-55,
                vmax=0,
                shading="auto",
                rasterized=True,
            )
            ax.set(xlim=(5, 65), xlabel="Slant range (m)", ylabel=label)
            ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=10)
        axes[0].set_ylim(-3.5, 3.5)
        axes[1].set_ylim(-75, 75)
        for d in row["detections"]:
            identity = int(d.get("track_id", 0))
            color = identity_color(identity)
            a, e = np.deg2rad([d["aoa_az_deg"], d["aoa_el_deg"]])
            angle = np.rad2deg(np.arcsin(np.sin(a) * np.cos(e)))
            for ax, y in ((axes[0], d["velocity_m_s"]), (axes[1], angle)):
                ax.scatter(
                    d["range_m"],
                    y,
                    s=70,
                    marker="o",
                    facecolors="none",
                    edgecolors=color,
                    linewidths=1.5,
                )
                ax.annotate(
                    f"T{identity}",
                    (d["range_m"], y),
                    xytext=(5, 6),
                    textcoords="offset points",
                    color="#e9f2fb",
                    fontsize=8,
                )
        cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.84, pad=0.035, aspect=45)
        cb.set_label("Relative power (dB) · fixed recording-wide peak per product", fontsize=9)
        fig.savefig(directory / f"{index:04d}.png", dpi=140)
        if index == len(rows) // 2:
            fig.savefig(output / "radar_snapshot.pdf")
        plt.close(fig)
        if (index + 1) % 10 == 0:
            print(f"Radar plots {index + 1}/{len(rows)}", flush=True)
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", type=Path)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--reports-json", type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = review(args.scenario, output / "evidence", args.reports_json)
    if args.reports_json:
        shutil.copy2(args.reports_json, output / "evidence/derived_reports.json")
        review(args.scenario, output / "evidence/source_acquisition")
    plot(rows, output / "evidence")
    speed = encode(args.captures, output, args.encoder)
    reference = plots(args.scenario, rows, output)
    for name in (
        "hero",
        "facade",
        "city",
        "street",
        "Car_Inbound",
        "Car_Outbound",
        "Scooter",
        "Pedestrian",
        "Drone",
    ):
        source = args.captures / f"{name}.png"
        if source.exists():
            shutil.copy2(source, output / source.name)
    data = dict(
        summary=summary,
        epochs=rows,
        playback_speed=speed,
        power_reference_linear=reference,
        acquisition=dict(
            carrier_ghz=28,
            bandwidth_mhz=500,
            array_rows=16,
            array_columns=16,
            pulse_rate_hz=2000,
            samples_per_cpi=128,
        ),
    )
    (output / "data.js").write_text(
        "window.DEMO = " + json.dumps(data, separators=(",", ":")) + ";\n"
    )
    shutil.copy2(HERE / "scripts/demo.html", output / "index.html")
    shutil.copy2(HERE / "ATTRIBUTION.md", output / "ATTRIBUTION.md")
    (output / "START_HERE.txt").write_text(
        "Open index.html in a browser. All assets are local; no service or network is needed.\n"
        "paris_opera.mp4 combines the scene and radar views in one 1080p movie.\n"
        "scene.mp4 is the native ORCHAV capture. evidence/ contains simulated detection and tracking results.\n"
        "The presentation uses the normal visualizer's saved track overlays and shared Card label style.\n"
        "RF model: mesh surfaces, single-bounce specular/diffuse paths, illustrative materials, no added receiver noise.\n"
        "Walking articulation is frozen within each coherent radar solve; this recording shows bulk Doppler.\n"
    )
    print(f"Portable presentation: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
