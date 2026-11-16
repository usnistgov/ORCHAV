"""Plot composition and portable exports for the shared coverage calculation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .comparison import json_value


def draw_comparison(figure, result, *, row_index=0, threshold_dbm=-75.0):
    """Draw reusable threshold curves and all-source summary from calculated data."""
    figure.clear()
    axes = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.15]})
    rows = result["rows"]
    row = rows[min(max(0, row_index), len(rows) - 1)]
    thresholds = np.asarray(row["thresholds_dbm"])
    for subset, lo, hi in zip(
        row["subsets"], row["fraction_lower"], row["fraction_upper"], strict=True
    ):
        (line,) = axes[0].plot(thresholds, 100 * np.asarray(lo), label="+".join(subset))
        axes[0].fill_between(
            thresholds,
            100 * np.asarray(lo),
            100 * np.asarray(hi),
            color=line.get_color(),
            alpha=0.15,
        )
    axes[0].axvline(threshold_dbm, color=".4", ls=":", lw=0.8)
    axes[0].set(
        xlabel="Received-power criterion (dBm)",
        ylabel="Eligible support passing (%)",
        title=row["label"],
        ylim=(-1, 101),
    )
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.2)
    selected = np.flatnonzero(np.isclose(thresholds, threshold_dbm, rtol=0, atol=1e-12))
    if not len(selected):
        raise ValueError("The displayed criterion must be included in the analysis thresholds")
    index = int(selected[0])
    matrix = np.asarray([r["fraction_lower"] for r in rows])[:, :, index] * 100
    axes[1].imshow(matrix, aspect="auto", vmin=0, vmax=100, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        axes[1].text(
            j,
            i,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if value > 60 else "black",
        )
    axes[1].set_xticks(
        range(len(row["subsets"])),
        ["+".join(s) for s in row["subsets"]],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    axes[1].set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=8)
    bounded = any(
        np.any(
            np.asarray(item["fraction_upper"])[:, index]
            > np.asarray(item["fraction_lower"])[:, index]
        )
        for item in rows
    )
    prefix = "Lower-bound fraction" if bounded else "Fraction"
    axes[1].set(title=f"{prefix} at {thresholds[index]:g} dBm (%)")
    figure.tight_layout()


def export_comparison(result, output, *, row_index=0, threshold_dbm=-75.0):
    """Export PNG/PDF, strict JSON recipes/provenance and exact numeric arrays."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(12, 5))
    FigureCanvasAgg(figure)
    draw_comparison(figure, result, row_index=row_index, threshold_dbm=threshold_dbm)
    for suffix in ("png", "pdf"):
        figure.savefig(
            output.with_suffix("." + suffix),
            dpi=180,
            **({"metadata": {"CreationDate": None, "ModDate": None}} if suffix == "pdf" else {}),
        )
    metadata = {
        **result,
        "display": {
            "row_index": row_index,
            "threshold_dbm": threshold_dbm,
            "normalization": "absolute fractions, fixed 0–100 percent; shading bounds unknown/invalid observations",
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(json_value(metadata), indent=2, allow_nan=False) + "\n"
    )
    arrays = {"thresholds_dbm": np.asarray(result["definition"]["thresholds_dbm"])}
    for i, row in enumerate(result["rows"]):
        for key in ("fraction_lower", "fraction_upper", "count_lower", "count_upper"):
            if key in row:
                arrays[f"row{i}_{key}"] = np.asarray(row[key])
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    return [str(output.with_suffix("." + suffix)) for suffix in ("png", "pdf", "json", "npz")]
