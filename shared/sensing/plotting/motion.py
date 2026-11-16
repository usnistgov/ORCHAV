"""Presentation of motion-analysis exports; all numerical work lives upstream."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def motion_figure(comparison: dict):
    """Build the same figure for the native dialog and PNG/PDF exports."""
    from matplotlib.figure import Figure

    records = comparison["results"]
    display = comparison["definition"].get("display", {})
    normalization = display.get("normalization", "absolute")
    if normalization not in {"absolute", "per_record_peak"}:
        raise ValueError("Normalization must be absolute or per_record_peak")
    modes = [mode for mode in ("retained", "removed") if mode + "_psd" in records[0]]
    limits = display.get("color_limits_db", [-185, -115])
    if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError("Color limits must be finite and increasing")
    doppler_limits = display.get("doppler_limits_hz")
    fig = Figure(
        figsize=(max(6.4, len(modes) * 5), max(3.2, 2.7 * len(records))), layout="constrained"
    )
    axes = fig.subplots(len(records), len(modes), squeeze=False, sharex=False, sharey=True)
    for row, result in enumerate(records):
        peak = max(float(np.max(result[mode + "_psd"])) for mode in modes)
        reference = peak if normalization == "per_record_peak" and peak > 0 else 1.0
        for col, mode in enumerate(modes):
            ax = axes[row, col]
            power_db = 10 * np.log10(np.maximum(result[mode + "_psd"] / reference, 1e-300))
            image = ax.pcolormesh(
                result["time_s"],
                result["doppler_hz"],
                power_db,
                shading="nearest",
                rasterized=True,
                cmap="magma",
                vmin=limits[0],
                vmax=limits[1],
            )
            ax.set_title(f"{result['label']} · mean {mode}")
            ax.set_xlabel("Window center (s)")
            ax.set_ylabel("Doppler (Hz)")
            if doppler_limits is not None:
                ax.set_ylim(*doppler_limits)
    unit = (
        "Unit-input channel PSD (dB / Hz)"
        if normalization == "absolute"
        else "PSD relative to each recording's common peak (dB)"
    )
    fig.colorbar(image, ax=axes.ravel().tolist(), label=unit, shrink=0.9)
    recipe = records[0]["metadata"]["recipe"]
    fig.suptitle(
        f"Motion signatures · {recipe['range_min_m']:g}–{recipe['range_max_m']:g} m · {recipe['window']}{recipe['window_samples']}, hop {recipe['hop_samples']} · no delay taper"
    )
    return fig


def render_motion_comparison(comparison: dict, destination: Path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = motion_figure(comparison)
    FigureCanvasAgg(fig)
    fig.savefig(Path(destination) / "motion_comparison.png", dpi=160)
    fig.savefig(
        Path(destination) / "motion_comparison.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    if any("root_displacement_m" in r for r in comparison["results"]):
        root = root_motion_figure(comparison)
        FigureCanvasAgg(root)
        root.savefig(Path(destination) / "root_motion.png", dpi=160)
        root.savefig(
            Path(destination) / "root_motion.pdf", metadata={"CreationDate": None, "ModDate": None}
        )
    return fig


def root_motion_figure(comparison: dict):
    """Inspect exported source-root timing without inferring RF contributions."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7, 3.6), layout="constrained")
    axis = figure.subplots()
    for result in comparison["results"]:
        if "root_displacement_m" in result:
            axis.plot(result["root_time_s"], result["root_displacement_m"], label=result["label"])
    axis.set(
        xlabel="Authored source time (s)",
        ylabel="Root displacement from initial pose (m)",
        title="Source-motion reference",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    return figure
