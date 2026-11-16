"""Ground-truth evaluation and standalone figures for saved fusion results.

All matching in this module is for scoring only. Estimation has already finished.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from shared.sensing.fusion.core import gated_assignment
from shared.sensing.fusion.result import FusionResult


def evaluate_fusion(result: dict, position_gate_m: float = 0.75) -> dict:
    """Score confirmed outputs, including labelled coasting, by world 3-D distance."""
    result = FusionResult(result)
    if not np.isfinite(position_gate_m) or position_gate_m <= 0:
        raise ValueError("Evaluation gate must be positive and finite")
    evaluation = {"position_gate_m": position_gate_m, "modes": {}}
    for mode, history in result["histories"].items():
        matches, missed, unmatched, switches = [], 0, 0, 0
        last_ids = {}
        unavailable_epochs = 0
        target_present_epochs = 0
        all_targets_matched_epochs = 0
        opportunities = {}
        first_truth = {}
        for epoch, truth in zip(history, result["truth_for_evaluation"], strict=True):
            if epoch["time_s"] != truth["time_s"]:
                raise ValueError("Fusion/truth evaluation epochs differ")
            if not truth.get("available", True):
                unavailable_epochs += 1
                continue
            targets = truth["targets"]
            for target in targets:
                name = target["target_id"]
                opportunities[name] = opportunities.get(name, 0) + 1
                first_truth.setdefault(name, epoch["time_s"])
            tracks = [t for t in epoch["tracks"] if t["confirmed"]]
            costs = np.array(
                [
                    [np.linalg.norm(np.subtract(t["position_m"], g["position_m"])) for t in tracks]
                    for g in targets
                ]
            ).reshape(len(targets), len(tracks))
            pairs = gated_assignment(costs, position_gate_m)
            if targets:
                target_present_epochs += 1
                all_targets_matched_epochs += int(len(pairs) == len(targets))
            missed += len(targets) - len(pairs)
            unmatched += len(tracks) - len(pairs)
            for i, j in pairs:
                gt, track = targets[i], tracks[j]
                name = gt["target_id"]
                previous = last_ids.get(name)
                switches += int(previous is not None and previous != track["track_id"])
                last_ids[name] = track["track_id"]
                matches.append(
                    {
                        "time_s": epoch["time_s"],
                        "target_id": name,
                        "track_id": track["track_id"],
                        "position_error_m": np.subtract(
                            track["position_m"], gt["position_m"]
                        ).tolist(),
                        "velocity_error_m_s": np.subtract(
                            track["velocity_m_s"], gt["velocity_m_s"]
                        ).tolist(),
                        "predicted_only": track["predicted_only"],
                    }
                )

        def rmse(rows, field):
            return (
                float(np.sqrt(np.mean(np.sum(np.square([r[field] for r in rows]), axis=1))))
                if rows
                else None
            )

        evaluation["modes"][mode] = {
            "truth_unavailable_epochs": unavailable_epochs,
            "matched_outputs": len(matches),
            "target_epoch_opportunities": len(matches) + missed,
            "coverage": len(matches) / (len(matches) + missed) if len(matches) + missed else None,
            # Extra confirmed outputs are reported separately: this measures
            # simultaneous target coverage, not the absence of false tracks.
            "target_present_epochs": target_present_epochs,
            "all_targets_matched_epochs": all_targets_matched_epochs,
            "all_targets_coverage": (
                all_targets_matched_epochs / target_present_epochs
                if target_present_epochs
                else None
            ),
            "duration_s": float(history[-1]["time_s"] - history[0]["time_s"]) if history else 0.0,
            "missed_target_epochs": missed,
            "unmatched_confirmed_outputs": unmatched,
            "evaluation_id_switches": switches,
            "position_rmse_m": rmse(matches, "position_error_m"),
            "velocity_vector_rmse_m_s": rmse(matches, "velocity_error_m_s"),
            "targets": {
                name: {
                    "opportunities": opportunities[name],
                    "matched_outputs": sum(r["target_id"] == name for r in matches),
                    "coverage": sum(r["target_id"] == name for r in matches) / opportunities[name],
                    "first_confirmed_match_delay_s": next(
                        (
                            r["time_s"] - first_truth[name]
                            for r in matches
                            if r["target_id"] == name
                        ),
                        None,
                    ),
                    "position_rmse_m": rmse(
                        [r for r in matches if r["target_id"] == name], "position_error_m"
                    ),
                    "velocity_vector_rmse_m_s": rmse(
                        [r for r in matches if r["target_id"] == name], "velocity_error_m_s"
                    ),
                }
                for name in sorted(opportunities)
            },
            "matches": matches,
        }
    return evaluation


def _xyz_histories(epochs, collection, identity):
    """Keep saved world positions and explicit gaps for the standard summary plot."""
    rows = [
        {
            entry[identity]: entry["position_m"]
            for entry in epoch[collection]
            if epoch.get("available", True) and entry.get("confirmed", True)
        }
        for epoch in epochs
    ]
    return {
        name: [
            dict(zip(("x_m", "y_m", "z_m"), row.get(name, (np.nan,) * 3), strict=True))
            for row in rows
        ]
        for name in sorted({name for row in rows for name in row})
    }


def generate_fusion_tracking_summaries(result, figures: Path) -> None:
    """Render every comparison mode using the sensing summary's 3D track plot.

    Show all confirmed estimates, including coasting and unmatched outputs.
    Truth supplies reference curves only; it never selects which tracks appear.
    """
    from shared.sensing.plotting.viz import create_xyz_tracking_figure

    truth = _xyz_histories(result["truth_for_evaluation"], "targets", "target_id")
    for mode, epochs in result["histories"].items():
        names = list(result["sources"]) if mode == "fused" else [mode]
        reports = [
            dict(zip(("x_m", "y_m", "z_m"), report["world_position_m"], strict=True))
            for name in names
            for epoch in result["observations"][name]
            for report in epoch["detections"]
            if report.get("world_position_m") is not None
        ]
        create_xyz_tracking_figure(
            _xyz_histories(epochs, "tracks", "track_id"),
            gt_tracks=truth,
            raw_detections=reports,
            coordinate_frame="world",
            connect_tracks=True,
            output_path=figures / f"tracking_xyz_{quote(mode, safe='')}.png",
        )


def _plot_track_availability(result, evaluation, output: Path) -> None:
    """Show gaps and coasting explicitly instead of connecting across missed epochs."""
    import matplotlib.pyplot as plt

    targets = sorted(
        {g["target_id"] for epoch in result["truth_for_evaluation"] for g in epoch["targets"]}
    )
    if not targets:
        return
    modes = list(evaluation["modes"])
    fig, axes = plt.subplots(
        len(targets), 1, figsize=(10, 2.2 * len(targets) + 1), sharex=True, squeeze=False
    )
    styles = (
        ("Matched estimate", "o", "#277fba", "#277fba"),
        ("Matched prediction only", "o", "none", "#277fba"),
        ("Missed target", "x", "#c46a4b", "#c46a4b"),
    )
    for ax, target in zip(axes[:, 0], targets, strict=True):
        times = [
            epoch["time_s"]
            for epoch in result["truth_for_evaluation"]
            if epoch.get("available", True)
            and any(g["target_id"] == target for g in epoch["targets"])
        ]
        for row, mode in enumerate(modes):
            matches = {
                match["time_s"]: match
                for match in evaluation["modes"][mode]["matches"]
                if match["target_id"] == target
            }
            status = [
                2 if time not in matches else int(matches[time]["predicted_only"]) for time in times
            ]
            for code, (label, marker, fill, edge) in enumerate(styles):
                selected = [
                    time for time, state in zip(times, status, strict=True) if state == code
                ]
                kwargs = (
                    {"color": edge} if marker == "x" else {"facecolors": fill, "edgecolors": edge}
                )
                ax.scatter(
                    selected,
                    [row] * len(selected),
                    s=18,
                    marker=marker,
                    label=label if row == 0 else None,
                    linewidths=0.9,
                    **kwargs,
                )
        ax.set_yticks(range(len(modes)), modes)
        ax.set_ylim(len(modes) - 0.5, -0.5)
        ax.set_title(target)
        ax.grid(axis="x", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False
    )
    axes[-1, 0].set_xlabel("Common simulation time (s)")
    fig.suptitle(f"Confirmed track coverage — {evaluation['position_gate_m']:g} m evaluation gate")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def generate_fusion_figures(result: dict, output: Path) -> dict:
    """Show all tracks and raw reports; never prune outputs using truth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = FusionResult(result)
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    generate_fusion_tracking_summaries(result, figures)
    evaluation = evaluate_fusion(
        result, result.get("evaluation_config", {}).get("position_gate_m", 0.75)
    )
    (output / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, allow_nan=False) + "\n"
    )
    _plot_track_availability(result, evaluation, figures / "track_availability.png")
    modes = list(result["histories"])
    targets = sorted({g["target_id"] for e in result["truth_for_evaluation"] for g in e["targets"]})
    palette = plt.get_cmap("tab10")
    colors = {mode: palette(i) for i, mode in enumerate(modes)}
    fig, axes = plt.subplots(
        1, len(modes), figsize=(5 * len(modes), 5), sharex=True, sharey=True, squeeze=False
    )
    for ax, mode in zip(axes[0], modes):
        for name in targets:
            points = np.array(
                [
                    g["position_m"]
                    for e in result["truth_for_evaluation"]
                    for g in e["targets"]
                    if g["target_id"] == name
                ]
            )
            ax.plot(points[:, 0], points[:, 1], "k--", linewidth=1.4, label=f"GT {name}")
            ax.scatter(*points[0, :2], c="k", marker="s", s=22)
        names = list(result["sources"]) if mode == "fused" else [mode]
        for n in names:
            p = result["sources"][n]["sensor"]["position_m"]
            ax.scatter(p[0], p[1], marker="*", s=110, color=colors[n], label=f"{n} radar")
            raw = np.array(
                [
                    d["world_position_m"]
                    for e in result["observations"][n]
                    for d in e["detections"]
                    if d.get("world_position_m") is not None
                ]
            )
            if raw.size:
                ax.scatter(
                    raw[:, 0],
                    raw[:, 1],
                    s=10,
                    marker="+",
                    alpha=0.35,
                    color=colors[n],
                    label=f"{n} detections",
                )
        history = result["histories"][mode]
        for tid in sorted({t["track_id"] for e in history for t in e["tracks"]}):
            seq = [(e["time_s"], t) for e in history for t in e["tracks"] if t["track_id"] == tid]
            ps = np.array([t["position_m"] for _, t in seq])
            ax.plot(ps[:, 0], ps[:, 1], lw=1, alpha=0.35, color=palette(tid % 10))
            confirmed = np.array([t["position_m"] for _, t in seq if t["confirmed"]])
            if confirmed.size:
                ax.plot(
                    confirmed[:, 0],
                    confirmed[:, 1],
                    ".-",
                    ms=3,
                    color=palette(tid % 10),
                    label=f"Track {tid}",
                )
        ax.set_title(f"{mode}: all reports / tracks")
        ax.set_xlabel("World X (m)")
        ax.grid(alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=7, loc="lower left")
    axes[0, 0].set_ylabel("World Y (m)")
    fig.suptitle("Two synchronized monostatic sensors — measurement-level fusion")
    fig.tight_layout()
    fig.savefig(figures / "tracking_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(
        2, max(1, len(targets)), figsize=(6 * max(1, len(targets)), 6), sharex=True, squeeze=False
    )
    if not targets:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No target GT available", ha="center", transform=ax.transAxes)
    for column, name in enumerate(targets):
        for mode, data in evaluation["modes"].items():
            rows = [r for r in data["matches"] if r["target_id"] == name]
            for row, field in enumerate(("position_error_m", "velocity_error_m_s")):
                axes[row, column].plot(
                    [r["time_s"] for r in rows],
                    [np.linalg.norm(r[field]) for r in rows],
                    label=mode,
                    color=colors[mode],
                )
        axes[0, column].set_title(name)
        axes[0, column].legend()
        axes[1, column].set_xlabel("Common simulation time (s)")
        for ax in axes[:, column]:
            ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("3-D position error (m)")
    axes[1, 0].set_ylabel("Velocity-vector error (m/s)")
    fig.suptitle(
        f"Confirmed estimates versus GT (matching gate: {evaluation['position_gate_m']:g} m)"
    )
    fig.tight_layout()
    fig.savefig(figures / "errors_vs_time.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(
        2, max(1, len(targets)), figsize=(6 * max(1, len(targets)), 6), sharex=True, squeeze=False
    )
    if not targets:
        for ax in axes.flat:
            ax.text(0.5, 0.5, "No target GT available", ha="center", transform=ax.transAxes)
    for column, name in enumerate(targets):
        gt_rows = [
            (e["time_s"], g)
            for e in result["truth_for_evaluation"]
            for g in e["targets"]
            if g["target_id"] == name
        ]
        for component in range(2):
            ax = axes[component, column]
            ax.plot(
                [t for t, _ in gt_rows],
                [g["velocity_m_s"][component] for _, g in gt_rows],
                "k--",
                label="GT",
            )
            for mode in modes:
                lookup = {
                    (e["time_s"], t["track_id"]): t
                    for e in result["histories"][mode]
                    for t in e["tracks"]
                }
                rows = [r for r in evaluation["modes"][mode]["matches"] if r["target_id"] == name]
                ax.plot(
                    [r["time_s"] for r in rows],
                    [lookup[(r["time_s"], r["track_id"])]["velocity_m_s"][component] for r in rows],
                    label=mode,
                    color=colors[mode],
                )
            ax.grid(alpha=0.25)
        axes[0, column].set_title(name)
        axes[0, column].legend()
        axes[1, column].set_xlabel("Common simulation time (s)")
    axes[0, 0].set_ylabel("World Vx (m/s)")
    axes[1, 0].set_ylabel("World Vy (m/s)")
    fig.suptitle("Complementary radial measurements constrain world velocity")
    fig.tight_layout()
    fig.savefig(figures / "velocity_components.png", dpi=160)
    plt.close(fig)

    history = result["histories"]["fused"]
    ids = sorted({t["track_id"] for e in history for t in e["tracks"]})
    fig, ax = plt.subplots(figsize=(11, max(2.5, 0.5 * len(ids) + 1)))
    markers = ["s", "o", "^", "v"]
    for epoch in history:
        for track in epoch["tracks"]:
            y = ids.index(track["track_id"])
            if track["predicted_only"]:
                ax.scatter(epoch["time_s"], y, marker="x", c="gray", s=18)
            for k, n in enumerate(sorted(result["sources"])):
                if any(s["sensor_id"] == n for s in track["sources"]):
                    ax.scatter(
                        epoch["time_s"],
                        y + (k - 0.5) * 0.18,
                        marker=markers[k % 4],
                        color=colors[n],
                        s=15,
                        alpha=1 if track["confirmed"] else 0.35,
                    )
    for k, n in enumerate(sorted(result["sources"])):
        ax.scatter([], [], marker=markers[k % 4], color=colors[n], label=n)
    ax.scatter([], [], marker="x", color="gray", label="Prediction only")
    ax.set_yticks(range(len(ids)), [f"Track {tid}" for tid in ids])
    ax.set_xlabel("Common simulation time (s)")
    ax.set_title("Sensor contributions to every fused track (pale = tentative)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "sensor_contributions.png", dpi=160)
    plt.close(fig)
    return {
        mode: {k: v for k, v in data.items() if k != "matches"}
        for mode, data in evaluation["modes"].items()
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render summaries from a saved fusion result.")
    parser.add_argument("result", type=Path, help="Fusion bundle directory or fusion_results.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate_fusion_figures(FusionResult(args.result), args.output_dir)
