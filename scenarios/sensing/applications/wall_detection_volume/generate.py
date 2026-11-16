"""Acquire a paired cube/wall experiment with independent native CPIs per cell."""

from __future__ import annotations

import argparse
import hashlib
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from generator.core.configuration.loader import resolve_sensing_document

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scenarios.sensing.volume_cli import prepare_volume
from scenarios.sensing.volume_storage import (
    MODES,
    configure_trial_summary,
    confirm_storage,
    retention_spec,
)
from scenarios.sensing.volume_tools import collect, run_trials, write_json
from shared.scenarios import validate_scenario_data

DIRECTORY = Path(__file__).resolve().parent
SENSOR = np.array([-6.0, 0.0, 2.0])
WALL_MIN = np.array([-0.1, -2.0, 0.0])
WALL_MAX = np.array([0.1, 2.0, 4.0])
CUBE_HALF = 0.25
FRAME = {
    "id": "orchav:cube-wall-volume:v1",
    "origin": "Ground below the wall centre",
    "axes": ["x", "y", "z"],
    "handedness": "right",
    "length_unit": "m",
}

GRID_PRESETS = {
    # Keep a tiny smoke grid; the tutorial resolves both wall edges and its top.
    "smoke": {"spacing_m": 3.0, "bounds_m": [(-3, 3), (0, 3), (3, 6)]},
    "tutorial": {"spacing_m": 1.0, "bounds_m": [(-3, 3), (-3, 3), (1, 6)]},
    "standard": {"spacing_m": 0.5, "bounds_m": [(-3, 5), (-4, 4), (0.5, 6.5)]},
    "inspection": {"spacing_m": 0.5, "bounds_m": [(-2, 4), (-3, 3), (0.5, 5.5)]},
    "dense": {"spacing_m": 0.25, "bounds_m": [(-2, 4), (-3, 3), (0.5, 5.5)]},
}


def grid_axes(preset):
    """Exact regular sample centres; density never implies interpolated trials."""
    grid = GRID_PRESETS[preset]
    spacing = grid["spacing_m"]
    return {
        axis: np.linspace(low, high, round((high - low) / spacing) + 1).tolist()
        for axis, (low, high) in zip("xyz", grid["bounds_m"], strict=True)
    }


def intersects_wall(position, sensor=SENSOR):
    """Segment/AABB intersection, including tangency, between sensor and point."""
    direction = np.asarray(position, dtype=float) - sensor
    low, high = 0.0, 1.0
    for origin, delta, lower, upper in zip(sensor, direction, WALL_MIN, WALL_MAX, strict=True):
        if abs(delta) < 1e-12:
            if not lower <= origin <= upper:
                return False
        else:
            a, b = sorted(((lower - origin) / delta, (upper - origin) / delta))
            low, high = max(low, a), min(high, b)
            if low > high:
                return False
    return True


def geometry(position, wall_present, sensor=SENSOR):
    """Keep centre-path blockage and nine finite-target probes distinct."""
    p = np.asarray(position, dtype=float)
    offsets = np.array(
        [
            [x, y, z]
            for x in (-CUBE_HALF, CUBE_HALF)
            for y in (-CUBE_HALF, CUBE_HALF)
            for z in (-CUBE_HALF, CUBE_HALF)
        ]
    )
    blocked = [wall_present and intersects_wall(v, sensor) for v in np.vstack((p, p + offsets))]
    return {
        "direct_path_blocked": bool(blocked[0]),
        "blocked_target_probes": int(sum(blocked)),
        "target_probe_count": len(blocked),
    }


def excluded(position):
    """Exclude a cube intersecting the wall from both arms of the paired test."""
    p = np.asarray(position, dtype=float)
    # Include the cube's small +X translation throughout this CPI.
    return bool(
        np.all(p + CUBE_HALF + [0.063, 0, 0] >= WALL_MIN) and np.all(p - CUBE_HALF <= WALL_MAX)
    )


def prepare(root, grid="standard", retention="full"):
    """Freeze identical sample centres and detector settings for both arms."""
    if root.exists():
        raise FileExistsError(f"Preserve existing experiment: {root}")
    root.mkdir(parents=True)
    base = resolve_sensing_document(yaml.safe_load((DIRECTORY / "scenario.yaml").read_text()))
    axes = grid_axes(grid)
    for condition in ("wall", "open"):
        case = root / condition
        case.mkdir()
        cfg = deepcopy(base)
        cfg["scene"]["id"] = str(DIRECTORY / f"scene/{condition}.xml")
        cfg["actors"]["targets"][0]["asset"]["path"] = str(
            DIRECTORY.parents[3] / "libraries/targets/cube"
        )
        trials = []
        for iz, z in enumerate(axes["z"]):
            for iy, y in enumerate(axes["y"]):
                for ix, x in enumerate(axes["x"]):
                    pos = [x, y, z]
                    name = f"h0_z{iz}_y{iy}_x{ix}"
                    trial = {
                        "id": name,
                        "heading": 0,
                        "index_zyx": [iz, iy, ix],
                        "position_m": pos,
                        "velocity_m_s": [1.0, 0.0, 0.0],
                        "status": "excluded_geometry" if excluded(pos) else "pending",
                        **geometry(pos, condition == "wall"),
                    }
                    if trial["status"] == "pending":
                        native = case / "trials" / name / "native"
                        native.mkdir(parents=True)
                        cfg["actors"]["targets"][0]["mobility"] = {
                            "type": "linear",
                            "start_m": pos,
                            "end_m": [x + 0.063, y, z],
                        }
                        trial["native_summary"] = configure_trial_summary(cfg, trials)
                        validate_scenario_data(cfg)
                        path = native / "scenario.yaml"
                        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                        trial.update(
                            native=str(native.relative_to(case)),
                            configuration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        )
                    else:
                        trial["exclusion_reason"] = (
                            "Cube intersects the wall; the paired control excludes the same centre."
                        )
                    trials.append(trial)
        scene = deepcopy(cfg)
        scene["actors"]["targets"] = []
        scene["sensing"]["enabled"] = False
        scene["raytracing"]["enabled"] = False
        scene["timeline"] = {"steps": 1, "sample_rate_hz": 1.0}
        (case / "scenario.yaml").write_text(yaml.safe_dump(scene, sort_keys=False))
        write_json(
            case / "volume_recipe.json",
            {
                "schema_version": 1,
                "kind": "conditional_detection_outcomes",
                "world_frame": FRAME,
                "axes_m": axes,
                "grid_preset": grid,
                "grid_spacing_m": GRID_PRESETS[grid]["spacing_m"],
                "retention": retention_spec(trials, retention),
                "summary_trial_ids": [t["id"] for t in trials if t.get("native_summary")],
                "estimated_native_mb_per_trial": 1.5,
                "headings": [{"name": "Along +X, 1 m/s", "velocity_m_s": [1.0, 0.0, 0.0]}],
                "sensor_position_m": SENSOR.tolist(),
                "sensor_id": "Radar",
                "cpi_samples": 64,
                "scene_scenario": "scenario.yaml",
                "position_gate_m": 0.75,
                "target_label": "cube",
                "motion_label": "Target motion",
                "condition": condition,
                "comparison_volume": "../open" if condition == "wall" else "../wall",
                "comparison_label": (
                    "Open no-wall control" if condition == "wall" else "Open wall experiment"
                ),
                "geometry_description": "Sensor-to-cube-centre segment intersects the finite wall; this geometric label does not determine detection.",
                "wall_bounds_m": [WALL_MIN.tolist(), WALL_MAX.tolist()],
                "trials": trials,
                "assumptions": [
                    "One fixed-seed noiseless CPI per centre; outcomes are not detection probabilities.",
                    "A 0.5 m metal cube translates at 1 m/s along +X; yaw and RX array remain fixed.",
                    "The wall is 0.2 m thick, 4 m wide and 4 m high. No ground plane, diffraction or transmission is modeled.",
                    "The two conditions differ only by wall geometry. No artificial geometric mask changes detector outputs.",
                    "Centre-ray blockage is distinct from partial visibility of a finite cube and from detector outcomes.",
                    f"Each {GRID_PRESETS[grid]['spacing_m']:g} m displayed block represents one sampled centre, not exhaustive testing inside the block.",
                ],
            },
        )
        print(
            f"Prepared {condition}: {sum(t['status']=='pending' for t in trials)} acquisitions",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--collect", action="store_true")
    action.add_argument("--report", action="store_true")
    parser.add_argument("--condition", choices=("wall", "open"))
    parser.add_argument(
        "--grid",
        choices=tuple(GRID_PRESETS),
        default="standard",
        help="Grid preset used by --prepare; dense uses 0.25 m samples near the wall.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        help="Launch disjoint worker processes on these CUDA GPU indices.",
    )
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument(
        "--retention",
        choices=MODES,
        help="With --prepare: full recordings (default) or compact volume results.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Accept the displayed storage/retention plan for batch acquisition.",
    )
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--capture-3d", action="store_true", help="With --report, export actual pygfx volume views."
    )
    args = parser.parse_args()
    if args.capture_3d and not args.report:
        parser.error("--capture-3d requires --report")
    if not 0 <= args.worker < args.workers or (args.limit is not None and args.limit < 1):
        parser.error("Choose 0 <= worker < workers and a positive limit")
    if args.jobs_per_gpu < 1 or (
        args.gpus and (min(args.gpus) < 0 or len(set(args.gpus)) != len(args.gpus))
    ):
        parser.error("Choose positive jobs-per-gpu and distinct nonnegative GPU indices")
    if args.gpus and (
        args.prepare or args.collect or args.report or args.worker or args.workers != 1
    ):
        parser.error("--gpus is for acquisition and manages --worker/--workers itself")
    if args.grid != "standard" and not args.prepare:
        parser.error("--grid applies only to --prepare; acquisition reads the frozen recipe")
    if args.retention is not None and not args.prepare:
        parser.error(
            "--retention applies only to --prepare; use a new directory to change retention"
        )
    root = args.root.resolve()
    if args.prepare:
        prepare_volume(
            root,
            prepare,
            args.grid,
            args.retention or "full",
            recipe_roots=[root / "wall", root / "open"],
        )
    elif args.report:
        from scenarios.sensing.applications.wall_detection_volume.report import write_report

        write_report(root)
        if args.capture_3d:
            from visualizer.capture_sensing import capture_volume_report

            capture_volume_report(root, ("wall", "open"))
    elif args.gpus:
        from scenarios.sensing.applications.wall_detection_volume.parallel import run_parallel

        confirm_storage(
            [root / c for c in ((args.condition,) if args.condition else ("wall", "open"))],
            yes=args.yes,
            worker=None,
            workers=len(args.gpus) * args.jobs_per_gpu,
            limit=args.limit,
        )
        run_parallel(root, args.gpus, args.jobs_per_gpu, args.condition, args.limit)
    else:
        if not args.collect:
            confirm_storage(
                [root / c for c in ((args.condition,) if args.condition else ("wall", "open"))],
                yes=args.yes,
                worker=args.worker,
                workers=args.workers,
                limit=args.limit,
            )
        conditions = (args.condition,) if args.condition else ("wall", "open")
        for index, condition in enumerate(conditions, 1):
            if args.collect:
                collect(root / condition)
            else:
                print(f"\nCondition {index}/{len(conditions)}: {condition}", flush=True)
                run_trials(root / condition, args.worker, args.workers, args.limit)


if __name__ == "__main__":
    main()
