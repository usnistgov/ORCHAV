"""Plot saved cube-radar observations, detections and tracks with provenance.

This tool reads producer-authored products; it does not run a detector or tracker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.sensing.loader import load_saved_results
from shared.sensing.plotting.style import sensing_figure_style

ROOT = Path(__file__).resolve().parents[3]
FAMILY = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_db(power: np.ndarray, reference: float | None = None) -> np.ndarray:
    peak = float(np.max(power)) if reference is None else reference
    if peak <= 0:
        raise ValueError("A nonzero saved observation is required")
    return 10 * np.log10(np.maximum(power / peak, 1e-15))


def write_receipt(
    path: Path, *, sources: dict[str, Path], scenarios: list[str], details: dict, command: str
) -> None:
    source_files = [FAMILY / name / "scenario.yaml" for name in scenarios]
    source_files += [
        Path(__file__),
        ROOT / "shared/sensing/plotting/style.py",
        ROOT / "shared/sensing/loader.py",
    ]
    recordings = {}
    for label, source in sources.items():
        frames = source if (source / "frames_manifest.json").is_file() else source / "frames"
        recordings[label] = {
            "source_hashes": {p.name: sha256(p) for p in sorted(frames.iterdir()) if p.is_file()},
            "execution_scenario_sha256": (
                sha256(source / "scenario.yaml") if (source / "scenario.yaml").is_file() else None
            ),
        }
    receipt = {
        "license_spdx": "GPL-3.0-only",
        "provenance": "ORCHAV project-generated plots of saved sensing products",
        "kind": "plot_of_saved_native_products_no_reprocessing",
        "image": path.name,
        "tutorial_scenario": source_files[0].relative_to(ROOT).as_posix(),
        "tutorial_scenario_sha256": sha256(source_files[0]),
        "sha256": sha256(path),
        "source_files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files},
        "recordings": recordings,
        "command": command,
        "limits": "One controlled specular acquisition; no claim of general detection or tracking performance.",
        **details,
    }
    path.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


def _result_summary(results: list[dict]) -> dict:
    return {
        "result_count": len(results),
        "result_times_s": [r["temporal"]["result"]["reference_time_s"] for r in results],
        "detection_counts": [len(r["detections"]) for r in results],
        "reports": [
            [
                {
                    k: d[k]
                    for k in ("range_m", "velocity_m_s", "track_id", "track_confirmed")
                    if k in d
                }
                for d in r["detections"]
            ]
            for r in results
        ],
        "frame_set_id": results[0]["_source_frame_set_id"],
        "generation_id": results[0]["_source_generation_id"],
        "processing_recipe": results[0]["_typed_sensing"]["config_resolved"]["processing_recipe"],
    }


def _velocity(result: dict) -> np.ndarray:
    return (
        np.asarray(result["doppler_axis"])
        * 299792458
        / float(result["_typed_sensing"]["carrier_freq_hz"])
        / 2
    )


def _rd_panel(ax, result: dict, *, title: str, limits=(1, 10), floor=-50):
    mesh = ax.pcolormesh(
        result["range_axis"],
        _velocity(result),
        relative_db(result["rd_map"]),
        shading="nearest",
        vmin=floor,
        vmax=0,
    )
    for index, detection in enumerate(result["detections"]):
        ax.plot(
            detection["range_m"],
            detection["velocity_m_s"],
            "x",
            color="white",
            markersize=7,
            label="Published detection" if index == 0 else None,
        )
    ax.set(
        xlim=limits,
        ylim=(-1.5, 1.5),
        xlabel="Monostatic range (m)",
        ylabel="Closing velocity (m/s)",
        title=title,
    )
    return mesh


@sensing_figure_style
def draw(experiment: str, source: Path, output: Path) -> None:
    image_path = output / f"cube_radar_{experiment}.png"
    if image_path.exists() or image_path.with_suffix(".json").exists():
        raise FileExistsError(f"Choose fresh outputs: {image_path}")
    output.mkdir(parents=True, exist_ok=True)
    sources = {experiment: source}
    details = {}
    if experiment in {"stationary", "moving"}:
        results = load_saved_results(source)
        if not results:
            raise ValueError(f"No complete results in {source}")
        last = results[-1]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
        mesh = _rd_panel(axes[0], last, title=f"Saved map · {len(last['detections'])} detection")
        fig.colorbar(mesh, ax=axes[0], label="dB relative to map peak")
        if experiment == "stationary":
            profile = np.asarray(last["_typed_sensing"]["range_profile_by_pair"])[0]
            axes[1].plot(
                last["range_axis"], relative_db(profile), label="Saved last-sample range profile"
            )
            axes[1].axvline(
                4.5, color="0.35", linestyle="--", label="Front face: geometric reference"
            )
            axes[1].axvline(5, color="0.6", linestyle=":", label="Cube center: geometric reference")
            axes[1].set(
                xlim=(3, 7),
                ylim=(-60, 3),
                xlabel="Monostatic range (m)",
                ylabel="dB relative to profile peak",
                title="Range measurement",
            )
            axes[1].legend(fontsize=8)
        else:
            times = [r["temporal"]["result"]["reference_time_s"] for r in results]
            axes[1].plot(
                times,
                [r["detections"][0]["range_m"] for r in results],
                "o-",
                label="Published front-face range",
            )
            axes[1].plot(
                times,
                4.5 - np.asarray(times),
                "--",
                color="0.35",
                label="Front face: geometric reference",
            )
            axes[1].set(
                xlabel="Saved result time / solve start (s)",
                ylabel="Monostatic range (m)",
                title="16 coherent observations",
            )
            axes[1].legend(fontsize=8)
        scenarios = [experiment]
        details = _result_summary(results)
    elif experiment == "filtering":
        names = ("cfar_only", "bounded", "outside_range")
        variants = {name: load_saved_results(source / name) for name in names}
        maps = [variants[name][0]["rd_map"] for name in names]
        if not all(np.array_equal(maps[0], values) for values in maps[1:]):
            raise ValueError("Filtering comparison changed the observed range-Doppler map")
        fig, axes = plt.subplots(
            1, 3, figsize=(14, 4.2), sharex=True, sharey=True, layout="constrained"
        )
        titles = {"cfar_only": "CFAR only", "bounded": "Bounded", "outside_range": "Outside range"}
        for ax, name in zip(axes, names):
            result = variants[name][0]
            mesh = _rd_panel(
                ax,
                result,
                title=f"{titles[name]} · {len(result['detections'])} report{'s' if len(result['detections']) != 1 else ''}",
                limits=(0, 38.4),
                floor=-130,
            )
        fig.colorbar(mesh, ax=axes, label="dB relative to common map peak")
        fig.suptitle("Same observation · different report filters")
        sources = {name: source / name for name in names}
        scenarios = ["stationary"]
        details = {
            "maps_identical": True,
            "variants": {name: _result_summary(rows) for name, rows in variants.items()},
        }
    else:
        detections = load_saved_results(source / "detections")
        tracks = load_saved_results(source / "tracks")
        if len(detections) != len(tracks) or not all(
            np.array_equal(a["rd_map"], b["rd_map"]) for a, b in zip(detections, tracks)
        ):
            raise ValueError("Tracking comparison changed the observed range-Doppler maps")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
        track_ids = sorted({d["track_id"] for r in tracks for d in r["detections"]})
        times = [r["temporal"]["result"]["reference_time_s"] for r in tracks]
        for ax, key, label in zip(
            axes, ("range_m", "velocity_m_s"), ("Monostatic range (m)", "Closing velocity (m/s)")
        ):
            for i, result in enumerate(detections):
                ax.scatter(
                    [times[i]] * len(result["detections"]),
                    [d[key] for d in result["detections"]],
                    marker="x",
                    color="0.55",
                    label="Per-CPI detections" if i == 0 else None,
                )
            for track_id in track_ids:
                points = [
                    (times[i], d[key])
                    for i, result in enumerate(tracks)
                    for d in result["detections"]
                    if d["track_id"] == track_id
                ]
                ax.plot(*zip(*points), "o-", fillstyle="none", label=f"Confirmed track {track_id}")
            ax.set(xlabel="Saved result time / solve start (s)", ylabel=label)
            ax.legend(fontsize=8)
        fig.suptitle("Same detections · temporal association and confirmation")
        sources = {name: source / name for name in ("detections", "tracks")}
        scenarios = ["moving" if experiment == "tracking" else "two_moving"]
        details = {
            "maps_identical": True,
            "detections": _result_summary(detections),
            "tracks": _result_summary(tracks),
        }
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
    default_source = (
        f"scenarios/sensing/cube_radar/{scenarios[0]}"
        if experiment in {"stationary", "moving"}
        else f"analysis/cube-radar-{'two-moving' if experiment == 'two_moving' else experiment}"
    )
    write_receipt(
        image_path,
        sources=sources,
        scenarios=scenarios,
        details=details,
        command=f"python -m scenarios.sensing.cube_radar.figures {experiment} --source {default_source} --output analysis/cube-radar-figures",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment", choices=("stationary", "moving", "filtering", "tracking", "two_moving")
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(args.experiment, args.source, args.output)


if __name__ == "__main__":
    main()
