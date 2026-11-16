"""Plot composition for link comparisons; calculations live in link_comparison."""

from __future__ import annotations

import numpy as np


def comparison_figure(result):
    """Render selected pair powers and differences with explicit physical units."""
    from matplotlib.figure import Figure

    count = 2 if result["differences"] else 1
    fig = Figure(figsize=(10, 3.3 * count), layout="constrained")
    axes = np.atleast_1d(fig.subplots(count, 1, sharex=True))
    styles = ("-", "--", ":", "-.")
    for source, trace in enumerate(result["traces"]):
        for tx, name in enumerate(result["transmitters"]):
            values = np.where(np.isfinite(trace["values"][:, tx]), trace["values"][:, tx], np.nan)
            axes[0].plot(
                trace["x"],
                values,
                color=f"C{tx%10}",
                linestyle=styles[source % 4],
                label=f"{trace['label']} · {name}",
            )
    axes[0].set(ylabel="Incoherent received power (dBm)", title=f"Receiver: {result['receiver']}")
    axes[0].legend(fontsize=8, ncol=2)
    for source, diff in enumerate(result["differences"]):
        for tx, name in enumerate(result["transmitters"]):
            axes[1].plot(
                result["traces"][0]["x"],
                diff["delta_db"][:, tx],
                label=f"{diff['label']} · {name}",
                color=f"C{tx%10}",
                linestyle=styles[source % len(styles)],
            )
        axes[1].set_ylabel("Paired power difference (dB)")
        axes[1].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Time (s)" if result["axis"] == "time" else "Sampled traveled distance (m)")
    for ax in axes:
        ax.grid(alpha=0.2)
    return fig
