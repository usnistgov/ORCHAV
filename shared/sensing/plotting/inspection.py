"""Figures of recorded processing decisions, shared by summaries and the GUI.

No detector, tracker, truth matching or ray tracing runs in this module.
Matplotlib is imported only when a figure is explicitly requested.
"""

from __future__ import annotations

import numpy as np

# Display floor for finite dB values; this is not a detector noise threshold.
_DISPLAY_POWER_FLOOR = 1e-30
_DETECTION_RANGE_PADDING_BINS = 4  # Retain neighboring cells around saved candidates.


def detection_stages_figure(
    bundle,
    *,
    doppler_bin=None,
    show_slice=True,
    vertical_axis=None,
    vertical_label="Doppler (Hz)",
    range_label="Range (m)",
    range_limits=None,
    vertical_limits=None,
    power_limits_db=None,
):
    """Compare saved stages with optional axes and power limits from an interactive view.

    Arrays retain their producer order, Doppler rows by range columns. A supplied
    vertical axis changes the displayed units, never the selected cell indices.
    Hide the slice when the interactive view has explicitly cleared its cell.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 7), layout="constrained")
    axes = fig.subplots(2, 3)
    if not bundle.available:
        fig.text(
            0.1,
            0.5,
            "Detector stages not recorded. Enable stage retention or reprocess explicitly.",
        )
        return fig
    r = bundle.range_axis_m
    d = bundle.doppler_axis_hz if vertical_axis is None else np.asarray(vertical_axis)
    if d.shape != bundle.doppler_axis_hz.shape or not np.all(np.isfinite(d)):
        raise ValueError("The display axis must match the saved Doppler coordinates")

    def values(suffix):
        product = bundle.product("range_doppler_" + suffix)
        return np.asarray(product.values[0]) if product.available else None

    power = values("detection_input_power")
    threshold = values("cfar_threshold_power")
    eligible = values("cfar_eligible_mask")
    cfar, nms, clusters = (values(n) for n in ("cfar_mask", "nms_mask", "cluster_labels"))
    if doppler_bin is None:
        doppler_bin = (
            int(np.unravel_index(np.argmax(power), power.shape)[0])
            if power is not None
            else len(d) // 2
        )
    doppler_bin = int(np.clip(doppler_bin, 0, len(d) - 1))

    def missing(ax):
        ax.text(0.5, 0.5, "Not recorded", transform=ax.transAxes, ha="center")

    # A shared range zoom follows recorded candidates; no values are changed.
    limits = (r[0], r[-1])
    if cfar is not None and np.any(cfar):
        columns = np.where(cfar)[1]
        limits = (
            r[max(0, int(columns.min()) - _DETECTION_RANGE_PADDING_BINS)],
            r[min(len(r) - 1, int(columns.max()) + _DETECTION_RANGE_PADDING_BINS)],
        )
    if range_limits is not None:
        limits = range_limits
    ax = axes[0, 0]
    ax.set_title("1 · Recorded detector input")
    if power is None:
        missing(ax)
    else:
        db = 10 * np.log10(np.maximum(power, _DISPLAY_POWER_FLOOR))
        low, high = power_limits_db if power_limits_db is not None else (db.max() - 60, db.max())
        image = ax.pcolormesh(
            r,
            d,
            db,
            shading="nearest",
            cmap="viridis",
            vmax=high,
            vmin=low,
            rasterized=True,
        )
        fig.colorbar(image, ax=ax, label="Power (dB)")
        if show_slice:
            ax.axhline(d[doppler_bin], color="#eab83c", lw=0.8)
    ax = axes[0, 1]
    ax.set_title(f"2 · CFAR slice · Doppler bin {doppler_bin}")
    if not show_slice:
        ax.set_title("2 · CFAR slice")
        ax.text(
            0.5, 0.5, "Select an RD cell to inspect its slice", transform=ax.transAxes, ha="center"
        )
    elif power is None or threshold is None or eligible is None:
        missing(ax)
    else:
        ax.plot(
            r,
            10 * np.log10(np.maximum(power[doppler_bin], _DISPLAY_POWER_FLOOR)),
            label="Detector input",
        )
        curve = np.where(eligible[doppler_bin], threshold[doppler_bin], np.nan)
        ax.plot(
            r,
            10 * np.log10(np.maximum(curve, _DISPLAY_POWER_FLOOR)),
            label="Saved CFAR threshold",
        )
        if cfar is not None:
            keep = cfar[doppler_bin].astype(bool)
            ax.scatter(
                r[keep],
                10 * np.log10(np.maximum(power[doppler_bin, keep], _DISPLAY_POWER_FLOOR)),
                s=16,
                label="CFAR passes",
            )
        ax.legend(fontsize=7)
    if power_limits_db is not None:
        ax.set_ylim(*power_limits_db)
    elif power is not None:
        peak_db = 10 * np.log10(max(float(np.max(power[doppler_bin])), _DISPLAY_POWER_FLOOR))
        ax.set_ylim(peak_db - 70, peak_db + 5)
    ax.set_ylabel("Power (dB)")
    for ax, data, title in (
        (axes[0, 2], cfar, "3 · CFAR-positive cells"),
        (axes[1, 0], nms, "4 · NMS survivors"),
    ):
        ax.set_title(title + (f" ({np.count_nonzero(data)})" if data is not None else ""))
        if data is None:
            missing(ax)
        else:
            passed = np.asarray(data, dtype=bool)
            colors = np.zeros((*passed.shape, 4))
            colors[passed] = (244 / 255, 67 / 255, 54 / 255, 1)
            ax.pcolormesh(
                r,
                d,
                np.ma.masked_where(~passed, np.ones_like(data)),
                shading="nearest",
                cmap=ListedColormap(["#00e640"]),
                edgecolors=colors.reshape(-1, 4),
                linewidth=0.6,
                rasterized=True,
            )
    ax = axes[1, 1]
    ids = np.unique(clusters[clusters >= 0]) if clusters is not None else []
    ax.set_title(
        f"5 · Cluster membership ({len(ids)})" if clusters is not None else "5 · Cluster membership"
    )
    if clusters is None:
        missing(ax)
    else:
        for index, group in enumerate(ids):
            rows, cols = np.where(clusters == group)
            ax.scatter(r[cols], d[rows], s=25, color=f"C{index%10}", label=f"Cluster {group}")
        if len(ids) <= 8 and len(ids):
            ax.legend(fontsize=7)
    ax = axes[1, 2]
    ax.set_title(f"6 · Published linked reports ({len(bundle.detection_links)})")
    for link in bundle.detection_links:
        group_index = list(ids).index(link.cluster_id) if link.cluster_id in ids else 0
        color = f"C{group_index%10}"
        xy = (r[link.range_bin], d[link.doppler_bin])
        ax.scatter(*xy, marker="x", color=color, s=45)
        ax.annotate(
            f"C{link.cluster_id}", xy, xytext=(4, 4), textcoords="offset points", fontsize=7
        )
        axes[1, 1].scatter(*xy, marker="x", color="black", s=55)

    for ax in axes.flat:
        ax.set_xlabel(range_label)
        ax.set_xlim(*limits)
        if ax is not axes[0, 1]:
            ax.set_ylabel(vertical_label)
            ax.set_ylim(*(vertical_limits if vertical_limits is not None else (d[0], d[-1])))
    event = bundle.event
    fig.suptitle(
        f"Detection stages · {event.reference_time_s:g} s",
        fontsize=10,
    )
    fig.supxlabel(
        "Saved pair grid · × published representatives · no detector rerun",
        fontsize=8,
    )
    return fig


def tracking_decisions_figure(record, metadata):
    """Plot saved associations, gates, and track states without rerunning tracking.

    ``metadata`` supplies the recorded tracker type and physical gate sizes.
    A missing decision record produces a figure explaining its unavailability.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Ellipse, Rectangle

    fig = Figure(figsize=(11, 5), layout="constrained")
    if record is None:
        fig.text(
            0.1,
            0.5,
            "Association decisions not recorded. Enable tracking diagnostics in a new run.",
        )
        return fig
    ax, table = fig.subplots(1, 2)
    labels = record["state_labels"]
    radial = labels[0] == "range_m"
    keys = ("range_m", "velocity_m_s") if radial else ("x_m", "y_m")
    inputs = record["inputs"]
    predictions = {p["track_id"]: p for p in record["predictions"]}
    assignments = {a["detection_index"]: a["track_id"] for a in record["assignments"]}
    for di, det in enumerate(inputs):
        xy = [det[k] for k in keys]
        ax.scatter(*xy, marker="x", color="black", zorder=5, gid="orchav-neutral")
        ax.annotate(f"D{di}", xy, xytext=(4, 4), textcoords="offset points", fontsize=8)
    for tid, prediction in predictions.items():
        xy = prediction["mean"][:2]
        color = f"C{tid%10}"
        ax.scatter(*xy, marker="+", s=90, color=color)
        if radial:
            rg, vg = metadata["association_range_gate_m"], metadata["association_velocity_gate_m_s"]
            ax.add_patch(
                Rectangle(
                    (xy[0] - rg, xy[1] - vg), 2 * rg, 2 * vg, fill=False, ls="--", color=color
                )
            )
        else:
            ax.add_patch(
                Circle(xy, metadata["association_xy_gate_m"], fill=False, ls="--", color=color)
            )
        covariance = prediction["covariance"]
        if covariance is not None:
            vals, vecs = np.linalg.eigh(np.asarray(covariance)[:2, :2])
            angle = np.rad2deg(np.arctan2(vecs[1, -1], vecs[0, -1]))
            ax.add_patch(
                Ellipse(
                    xy,
                    2 * np.sqrt(max(vals[-1], 0)),
                    2 * np.sqrt(max(vals[0], 0)),
                    angle=angle,
                    fill=False,
                    color=color,
                    alpha=0.5,
                )
            )
        ax.annotate(
            f"T{tid} prior",
            xy,
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    for di, tid in assignments.items():
        if tid in predictions:
            a, b = predictions[tid]["mean"][:2], [inputs[di][k] for k in keys]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=f"C{tid%10}", lw=1)
    for state in record["tracks"]:
        ax.scatter(
            *state["mean"][:2], marker="o", facecolors="none", edgecolors=f"C{state['track_id']%10}"
        )
    ax.autoscale()
    ax.margins(0.2)
    ax.set_xlabel("Range (m)" if radial else "Local X (m)")
    ax.set_ylabel("Closing velocity (m/s)" if radial else "Local Y (m)")
    ax.set_title(
        "+ prior · × input · ○ posterior\nDashed: physical gate; ellipse: prior 1σ, not gate",
        fontsize=10,
    )
    table.axis("off")
    rows = [
        [
            str(s["track_id"]),
            s["status"],
            str(s["hits"]),
            str(s["misses"]),
            "yes" if s["emitted"] else "no",
        ]
        for s in record["tracks"]
    ]
    if rows:
        t = table.table(
            cellText=rows[:16],
            colLabels=["Track", "State", "Hits", "Misses", "Output"],
            loc="center",
        )
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        t.scale(1, 1.5)
    if len(rows) > 16:
        table.text(
            0.5,
            0.05,
            f"First 16 of {len(rows)} active tracks",
            ha="center",
            transform=table.transAxes,
        )
    admitted = sum(c["admissible"] for c in record["candidates"])
    table.set_title(
        f"Candidates: {len(record['candidates'])}; admitted: {admitted}\n"
        f"Births: {record['birth_track_ids']}; retired: {record['retired_track_ids']}",
        fontsize=10,
    )
    kind = metadata.get("tracker_type", "m_of_n")
    note = (
        "Last-observation comparison; no motion prediction."
        if record["prediction_kind"] == "last_observation"
        else "Saved prediction and update; no tracker rerun."
    )
    if kind == "jpdaf":
        note += " Lines are reporting assignments; state updates use probability weights."
    if kind.startswith("cartesian"):
        note += " XY projection; radial/3D gates also apply."
    else:
        note += " Additional azimuth/Mahalanobis gates may apply."
    fig.suptitle(f"{kind} · {record['update_time_s']:g} s\n{note}", fontsize=9)
    return fig
