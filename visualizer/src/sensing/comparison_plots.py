"""Native interactive views of already calculated sensing comparisons."""

import numpy as np
import pyqtgraph as pg

from ..widgets.inline_plot import SERIES_COLORS, InlinePlotGrid


def volume_comparison(canvas, result):
    canvas.clear()
    left, right = canvas.plots
    conditions = result["conditions"]
    x = np.arange(len(conditions))
    for shift, field, label, color in zip(
        (-0.27, 0, 0.27),
        ("matched_locations", "extra_reports", "empty_outputs"),
        ("Matched locations", "Extra reports", "Empty outputs"),
        SERIES_COLORS[:3],
        strict=True,
    ):
        bar = pg.BarGraphItem(
            x=x + shift,
            height=[c["summary"][field] for c in conditions],
            width=0.25,
            brush=color,
            pen=color,
        )
        left.addItem(bar)
        left._external_legend.addItem(bar, label)
    left.getAxis("bottom").setTicks([list(enumerate([c["source"]["label"] for c in conditions]))])
    for index, condition in enumerate(conditions):
        errors = np.asarray(condition["matched_errors_m"])
        if errors.size:
            curve = right.curve(
                errors,
                np.arange(1, len(errors) + 1) / len(errors),
                name=condition["source"]["label"],
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
            )
            curve.setData(errors, np.arange(1, len(errors) + 1) / len(errors), stepMode="right")
    gate = result["definition"]["position_gate_m"]
    left.configure(title=f"Saved reports · {gate:g} m gate")
    right.setYRange(0, 1.03, padding=0)
    right.setXRange(0, gate, padding=0)
    right.set_empty_message(
        "" if any(c["matched_errors_m"] for c in conditions) else "No matched positions"
    )


def motion_comparison(comparison):
    records = comparison["results"]
    display = comparison["definition"].get("display", {})
    normalization = display.get("normalization", "absolute")
    if normalization not in ("absolute", "per_record_peak"):
        raise ValueError("Normalization must be absolute or per_record_peak")
    limits = display.get("color_limits_db", [-185, -115])
    if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError("Color limits must be finite and increasing")
    modes = [mode for mode in ("retained", "removed") if mode + "_psd" in records[0]]
    canvas = InlinePlotGrid(
        [
            (f"{r['label']} · mean {mode}", "Window center (s)", "Doppler (Hz)")
            for r in records
            for mode in modes
        ],
        columns=len(modes),
    )
    unit = (
        "Unit-input channel PSD (dB / Hz)"
        if normalization == "absolute"
        else "PSD relative to recording peak (dB)"
    )
    for row, result in enumerate(records):
        peak = max(float(np.max(result[mode + "_psd"])) for mode in modes)
        reference = peak if normalization == "per_record_peak" and peak > 0 else 1
        for col, mode in enumerate(modes):
            plot = canvas.plots[row * len(modes) + col]
            power_db = 10 * np.log10(np.maximum(result[mode + "_psd"] / reference, 1e-300))
            plot.heatmap(
                result["time_s"],
                result["doppler_hz"],
                power_db,
                limits=limits,
                colormap="magma",
                label=unit,
            )
            if display.get("doppler_limits_hz") is not None:
                plot.setYRange(*display["doppler_limits_hz"], padding=0)
    return canvas


def root_motion(comparison):
    canvas = InlinePlotGrid(
        [
            (
                "Source-motion reference",
                "Authored source time (s)",
                "Root displacement from initial pose (m)",
            )
        ]
    )
    for index, result in enumerate(comparison["results"]):
        if "root_displacement_m" in result:
            canvas.plots[0].curve(
                result["root_time_s"],
                result["root_displacement_m"],
                name=result["label"],
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
            )
    return canvas
