"""Paired wall/no-wall evidence, retaining misses and extra detector reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shared.sensing.volume.reader import SensingVolume


def counts(rows):
    measured = [r for r in rows if r["status"] == "measured"]
    errors = [r["matched_error_m"] for r in measured if r["hit"]]
    return {
        "measured": len(measured),
        "tp": sum(r["tp"] for r in measured),
        "fp": sum(r["fp"] for r in measured),
        "fn": sum(r["fn"] for r in measured),
        "no_reports": sum(r["tp"] + r["fp"] == 0 for r in measured),
        "reports_but_no_spatial_match": sum(r["fn"] == 1 and r["fp"] > 0 for r in measured),
        "median_error_m": float(np.median(errors)) if errors else None,
        "p90_error_m": float(np.percentile(errors, 90)) if errors else None,
    }


def write_bookmarks(root, readers):
    """Save source-bound inspection views of actual measured cells, without RT."""
    from shared.sensing.volume.inspection import context_snapshot, save_bookmark

    directory = root / "report/bookmarks"
    directory.mkdir(exist_ok=True)
    inventory = []
    for condition, reader in readers.items():
        examples = {}
        for name, position in (
            ("before_wall", [-2, 0, 2]),
            ("behind_wall", [3, 0, 2]),
            ("above_wall", [3, 0, 5.5]),
            ("wall_edge", [3, 3, 2]),
        ):
            key = next(
                (k for k, row in reader.cells.items() if row["position_m"] == position), None
            )
            if key is not None and reader.cells[key]["status"] == "measured":
                examples[name] = key
        for name, predicate in (
            ("match_with_extras", lambda row: row["hit"] and row["fp"] > 0),
            ("reports_without_match", lambda row: not row["hit"] and row["fp"] > 0),
        ):
            key = next(
                (
                    k
                    for k, row in sorted(reader.cells.items())
                    if row["status"] == "measured" and predicate(row)
                ),
                None,
            )
            if key is not None:
                examples[name] = key
        for name, key in examples.items():
            row = reader.cells[key]
            x, y, z = row["position_m"]
            view = dict(
                quantity="details",
                mode="slice",
                x=x,
                y=y,
                altitude=z,
                opacity=0.85,
                uniform_colour=True,
                readable_sensor=True,
                visible=True,
                inspect=True,
                cell_edges=True,
                unavailable_edges=True,
                inspection={"orientation": "xy", "vectors": True},
            )
            path = directory / f"{condition}_{name}.json"
            save_bookmark(
                path,
                context_snapshot(reader, key[0], key, view),
                title=f"{condition}: {name.replace('_', ' ')}",
                description=f"Recorded position {row['position_m']} m; TP {row['tp']}, FP {row['fp']}, FN {row['fn']} at the saved gate.",
            )
            inventory.append(str(path.relative_to(root / "report")))
    return inventory


def summarize(wall, control):
    """Pair by physical sample, requiring equal frame, grid, motion and gate."""
    if (
        wall.metadata["world_frame"] != control.metadata["world_frame"]
        or wall.headings != control.headings
    ):
        raise ValueError("Paired volumes must declare identical world frame and target motion")
    if (
        any(not np.array_equal(wall.axes[n], control.axes[n]) for n in ("x", "y", "z"))
        or wall.metadata["position_gate_m"] != control.metadata["position_gate_m"]
    ):
        raise ValueError("Paired volumes must use identical sample centres and matching gate")
    if wall.cells.keys() != control.cells.keys():
        raise ValueError("Paired sample inventories differ")
    if wall.metadata["sensor_position_m"] != control.metadata["sensor_position_m"]:
        raise ValueError("Paired volumes must use the same sensor position")
    pairs = [
        (w, control.cells[k])
        for k, w in wall.cells.items()
        if w["status"] == "measured" and control.cells[k]["status"] == "measured"
    ]
    groups = {}
    for name, predicate in (
        ("before_wall", lambda w: w["position_m"][0] < -0.1),
        ("all_nine_reference_rays_blocked", lambda w: w["blocked_target_probes"] == 9),
        ("some_reference_rays_blocked", lambda w: 0 < w["blocked_target_probes"] < 9),
        ("all_nine_reference_rays_clear", lambda w: w["blocked_target_probes"] == 0),
    ):
        selected = [(w, c) for w, c in pairs if predicate(w)]
        groups[name] = {
            "wall": counts([w for w, _ in selected]),
            "open": counts([c for _, c in selected]),
        }
    return {
        "manifest_sha256": {"wall": wall.identity, "open": control.identity},
        "position_gate_m": wall.metadata["position_gate_m"],
        "counts": {"wall": counts(wall.cells.values()), "open": counts(control.cells.values())},
        "excluded_per_condition": sum(
            r["status"] == "excluded_geometry" for r in wall.cells.values()
        ),
        "pending_per_condition": {
            name: sum(r["status"] == "pending" for r in reader.cells.values())
            for name, reader in (("wall", wall), ("open", control))
        },
        "paired_measured_positions": len(pairs),
        "paired_outcomes": {
            "hit_in_both": sum(w["hit"] and c["hit"] for w, c in pairs),
            "hit_only_without_wall": sum(not w["hit"] and c["hit"] for w, c in pairs),
            "hit_only_with_wall": sum(w["hit"] and not c["hit"] for w, c in pairs),
            "miss_in_both": sum(not w["hit"] and not c["hit"] for w, c in pairs),
        },
        "geometric_groups_using_wall_reference": groups,
        "interpretation": "One noiseless CPI per centre and condition. Geometry labels do not modify detector output. Fractions are conditional spatial outcomes, not detection probabilities. Nine rays probe the centre and corners of a finite cube; near-edge partial visibility is retained.",
    }


def write_report(root):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch, Rectangle

    root = Path(root)
    readers = {name: SensingVolume(root / name) for name in ("open", "wall")}
    for reader in readers.values():
        reader.verify_scene()
    wall = readers["wall"]
    spacing = float(np.diff(wall.axes["x"])[0])
    result = summarize(wall, readers["open"])
    out = root / "report"
    out.mkdir(exist_ok=True)
    result["bookmarks"] = write_bookmarks(root, readers)
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    cmap = ListedColormap(["#f2731a", "#19ccef"])
    cmap.set_bad("#dddddd")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], 2)
    iz = int(np.argmin(abs(wall.axes["z"] - 2)))
    iy = int(np.argmin(abs(wall.axes["y"])))
    reference = wall.geometry_grid(0)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for column, (condition, reader) in enumerate(readers.items()):
        grid = reader.outcome_grid(0)
        for row, coordinate in enumerate(("y", "z")):
            ax = axes[row, column]
            values = grid[iz] if row == 0 else grid[:, iy, :]
            blocked = reference[iz] if row == 0 else reference[:, iy, :]
            ax.pcolormesh(
                wall.axes["x"],
                wall.axes[coordinate],
                values,
                shading="nearest",
                cmap=cmap,
                norm=norm,
            )
            if values.size <= 32:
                xx, yy = np.meshgrid(wall.axes["x"], wall.axes[coordinate])
                ax.scatter(xx, yy, s=12, color="black", marker="+", zorder=5)
            ax.contour(
                wall.axes["x"],
                wall.axes[coordinate],
                blocked,
                levels=[0.5],
                colors=["#643cb0"],
                linewidths=1,
                linestyles="--",
            )
            if condition == "wall":
                ax.add_patch(
                    Rectangle((-0.1, -2 if row == 0 else 0), 0.2, 4, facecolor="#555555", zorder=4)
                )
            ax.plot(-6, 0 if row == 0 else 2, "*", color="#a77b00", markersize=13)
            ax.set(
                xlabel="World X (m)",
                ylabel=f"World {coordinate.upper()} (m)",
                aspect="equal",
                xlim=(-6.5, 5.5),
                ylim=(-4.5, 4.5) if row == 0 else (-0.3, 7.0),
                title=(
                    f"{'Wall present' if condition == 'wall' else 'No-wall control'} · "
                    + (
                        f"top view at Z = {wall.axes['z'][iz]:g} m"
                        if row == 0
                        else f"side view at Y = {wall.axes['y'][iy]:g} m"
                    )
                ),
            )
    axes[0, 0].legend(
        handles=[
            Patch(color="#19ccef", label="Hit"),
            Patch(color="#f2731a", label="Miss"),
            Patch(color="#ddd", label="Excluded / pending"),
        ],
        loc="upper left",
        fontsize=8,
    )
    fig.suptitle(
        "Same cube, motion, radar and detector · only the wall changes\nPurple dashed contour: blocked centre rays for the wall geometry, independent of detection",
        fontsize=13,
    )
    fig.savefig(out / "paired_sections.png", dpi=170)
    plt.close(fig)

    rows = "".join(
        f"<tr><td>{name}</td><td>{c['measured']}</td><td>{c['tp']}</td><td>{c['fn']}</td><td>{c['no_reports']}</td><td>{c['reports_but_no_spatial_match']}</td><td>{c['fp']}</td></tr>"
        for name, c in result["counts"].items()
    )
    groups = "".join(
        f"<tr><td>{name.replace('_', ' ')}</td><td>{d['open']['tp']}/{d['open']['measured']}</td><td>{d['wall']['tp']}/{d['wall']['measured']}</td></tr>"
        for name, d in result["geometric_groups_using_wall_reference"].items()
    )
    (out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Cube and wall detection volume</title>"
        "<style>body{font:17px system-ui;max-width:1150px;margin:3em auto;padding:0 1em;line-height:1.5}img{max-width:100%}td,th{text-align:left;padding:.5em 1em;border-bottom:1px solid #ccc}table{border-collapse:collapse}</style>"
        "<h1>A cube, a wall, and its measured detection shadow</h1>"
        "<p>The cube moves along +X at 1 m/s. The radar is at (−6, 0, 2) m; the wall at X = 0 is 4 m wide, 4 m high and 0.2 m thick. The receive array is fixed at 8 × 8 elements. The two experiments change only the wall. There is no ground plane, thermal noise, diffraction or transmission in this controlled model.</p>"
        "<p>Each position starts an independent acquisition: one RT solve at the starting geometry, followed by 64 coherent samples at 1 kHz using analytical path Doppler phase evolution. The animation advances 0.063 m, but propagation geometry and visibility remain fixed within that acquisition. The detector estimates motion from the signal. This model does not capture visibility changes during a CPI.</p>"
        f"<p>Each displayed {spacing:g} m block represents one simulation at its centre. A hit has a saved report within {result['position_gate_m']:g} m of the cube centre. The {result['excluded_per_condition']} centres that would place the finite cube in the wall are excluded from both arms.</p>"
        "<table><tr><th>Condition</th><th>Measured</th><th>Hits</th><th>Misses</th><th>No reports</th><th>Reports outside gate</th><th>Extra reports</th></tr>"
        + rows
        + "</table>"
        "<h2>Read the volume</h2><p>Open the wall result, choose <b>View → 3D volume</b> and <b>Show → Missed cells</b>, and rotate around the wall. Orange is a measured miss. Switch to <b>Direct path blockage (geometry)</b> to compare the purple geometric shadow. These are independently computed quantities. The <b>wall/open condition selector</b> keeps the camera and sampled position while opening the saved control. Changing views does not rerun RT or the detector.</p>"
        "<p>Cyan shows spatial hits. <b>Show → All measured positions</b> shows both disjoint regions together; each measured cell belongs to exactly one. Surfaces at different depths can overlap or hide the interior. Use <b>View → Slice</b> to compare one height. Maximum projection gives hits priority and can conceal misses.</p>"
        "<p>Use <b>Open view bookmark…</b> to load a measured example from <code>report/bookmarks/</code>. These views select before-wall, behind-wall, over-wall and edge positions where available, plus actual trials with extra or unmatched reports. <b>Open trial in visualizer</b> opens the selected cell's native acquisition. Bookmarks are bound to the collected manifest; regenerate the report after collecting additional trials.</p>"
        "<img src='paired_sections.png' alt='Paired top and side outcome sections with an independent geometric shadow contour'>"
        "<h2>Geometry versus recorded outcomes</h2><p>The following groups use the wall reference in both conditions. The before-wall row overlaps the clear-ray group. Near an edge, a cube can be partly visible even when its centre ray is blocked.</p>"
        "<table><tr><th>Reference group</th><th>Hits without wall</th><th>Hits with wall</th></tr>"
        + groups
        + "</table>"
        "<h2>Interpretation limits</h2><p>"
        + result["interpretation"]
        + " A spatial hit does not imply an extra-report-free result or a confirmed track. This example isolates obstruction; it is not a general claim that real walls prevent every radar detection.</p>"
        "<p><a href='summary.json'>Machine-readable counts and paired outcomes</a></p>"
    )
    return result
