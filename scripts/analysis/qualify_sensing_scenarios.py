"""Inventory authored sensing configurations using the normal runtime resolver.

No acquisition is launched. Generated trial YAMLs are deliberately excluded:
only tracked authored inputs enter the inventory. Numerical/live evidence is
recorded separately, never inferred from successful configuration parsing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def inventory() -> list[dict]:
    from generator import build_simulation_config
    from generator.core.sensing.acquisition import resolve_sensing_execution_config
    from shared.scenarios import load_scenario_configuration
    from shared.scenarios.paths import resolve_actor_resource

    paths = subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "ls-files", "scenarios"], cwd=ROOT, text=True
    ).splitlines()
    rows = []
    for name in paths:
        path = ROOT / name
        if path.suffix not in {".yaml", ".yml"} or not path.is_file():
            continue
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict) or not data.get("sensing", {}).get("enabled"):
            continue
        if name.startswith("scenarios/sensing/comparisons/"):
            scope = "internal_comparison"
        elif name.startswith("scenarios/sensing/"):
            scope = "public_sensing"
        elif name.startswith("scenarios/papers/"):
            scope = "paper"
        else:
            scope = "other_internal"
        row = dict(
            path=name,
            scope=scope,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            authored_sensing=data["sensing"],
            assets=[],
            evidence="configuration only",
            path_delay_extrema_s=None,
        )
        try:
            config = load_scenario_configuration(path)
            sim = build_simulation_config(config)
            s = asdict(sim.sensing)
            try:
                execution = resolve_sensing_execution_config(
                    sim,
                    frame_dt_s=sim.duration / max(sim.num_steps - 1, 1),
                    tx_device_count=len(data.get("actors", {}).get("tx", [])),
                    rx_device_count=len(data.get("actors", {}).get("rx", [])),
                )
                row["execution"] = asdict(execution)
                s["processing_mode"] = execution.processing.mode
            except Exception as exc:
                row["execution_error"] = f"{type(exc).__name__}: {exc}"
            rt = data.get("raytracing", {})
            rx = rt.get("antenna", {}).get("rx", {})
            elements = rx.get("num_rows", 1) * rx.get("num_cols", 1)
            row.update(
                resolved=s,
                carrier_hz=sim.carrier_frequency_hz,
                cir_samples=sim.cir_time_steps,
                slow_time_hz=sim.cir_sampling_frequency_hz,
                scene_steps=sim.num_steps,
                scene_duration_s=sim.duration,
                rx_array=rx,
                rx_elements=elements,
                ray_budget=rt.get("quality", {}),
                scene=data.get("scene"),
                actors=data.get("actors"),
                estimated_nonoverlapping_cpis=(
                    sim.num_steps // max(sim.cir_time_steps, 1)
                    if s["processing_mode"] == "coherent_cpi"
                    else None
                ),
                coherent_cpi_duration_s=(
                    sim.cir_time_steps / sim.cir_sampling_frequency_hz
                    if s["processing_mode"] == "coherent_cpi"
                    else None
                ),
                parse_status="pass",
            )
            n = s["range_response"]["num_frequency_bins"]
            factor = 2 if s["range_mode"] == "monostatic" else 1
            row["range_response_period_m"] = (
                299792458 * n / (factor * s["bandwidth"]) if n else None
            )
            row["cfr_complex128_bytes_per_cpi"] = (
                elements * sim.cir_time_steps * n * 16 if n else None
            )
            row["runtime_evidence_s"] = None
            asset_paths = []
            if config.scene_xml is not None:
                asset_paths.append(config.scene_xml)
                if config.scene_xml.exists():
                    try:
                        asset_paths.extend(
                            config.scene_xml.parent / tag.attrib["value"]
                            for tag in ET.parse(config.scene_xml).iter("string")
                            if tag.attrib.get("name") == "filename"
                        )
                    except ET.ParseError:
                        pass
            for target in data.get("actors", {}).get("targets", []):
                asset = target.get("asset", {})
                if "path" in asset:
                    asset_paths.append(
                        resolve_actor_resource(
                            asset["path"], scenario_root=path.parent, project_root=ROOT
                        )
                    )
                motion = target.get("mobility", {})
                for key in ("graph_path", "path"):
                    if isinstance(motion.get(key), str):
                        asset_paths.append(
                            resolve_actor_resource(
                                motion[key], scenario_root=path.parent, project_root=ROOT
                            )
                        )
            row["assets"] = [dict(path=str(p), exists=p.exists()) for p in asset_paths]
            row["all_resolved_assets_exist"] = all(p.exists() for p in asset_paths)
            frames = config.frames_dir
            row["local_frames_directory"] = str(frames) if frames.exists() else None
            row["local_frame_bytes"] = (
                sum(p.stat().st_size for p in frames.glob("*.h5")) if frames.exists() else 0
            )
        except Exception as exc:
            row.update(parse_status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = inventory()
    (args.output / "inventory.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")
    counts = Counter(
        (
            r["scope"],
            r.get("resolved", {}).get("processing_mode", "error"),
            r.get("resolved", {}).get("range_response", {}).get("model", "error"),
        )
        for r in rows
    )
    summary = [dict(scope=k[0], mode=k[1], model=k[2], count=v) for k, v in sorted(counts.items())]
    (args.output / "inventory_counts.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Authored sensing inventory",
        "",
        "Runtime-resolved configuration; numerical qualification is recorded separately.",
        "",
        "| Scenario | Scope | Acquisition | Range model | Array | Clutter | Assets |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        s = r.get("resolved", {})
        lines.append(
            f'| `{r["path"]}` | {r["scope"]} | {s.get("processing_mode","ERROR")} | {s.get("range_response",{}).get("model","—")} | {r.get("rx_elements","—")} / {s.get("array_detection_mode","—")} | {s.get("coherent_clutter_mode","—")} | {r.get("all_resolved_assets_exist",False)} |'
        )
    (args.output / "INVENTORY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    print("Errors:", [(r["path"], r["error"]) for r in rows if r["parse_status"] == "error"])


if __name__ == "__main__":
    main()
