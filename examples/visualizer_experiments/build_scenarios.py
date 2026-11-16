"""Build tiny local scene inputs; scientific frame data comes from ``generator``.

Run this from the repository root, then run ``python -m generator`` on any
created directory. No downloaded geometry, native RCS, or frame fabrication
is used here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

SCENARIO_NAMES = (
    "arrival_courtyard",
    "angular_corridor",
    "bistatic_range_locus",
    "monostatic_range_locus",
)


def _wall_y(y: float) -> list[tuple[float, float, float]]:
    return [(-12, y, -4), (12, y, -4), (12, y, 8), (-12, y, 8)]


def _wall_x(x: float) -> list[tuple[float, float, float]]:
    return [(x, -3, -4), (x, 3, -4), (x, 3, 8), (x, -3, 8)]


def _actor(name: str, position: list[float]) -> dict:
    return {"name": name, "mobility": {"type": "stationary", "position_m": position}}


def scenario_inputs(name: str) -> tuple[dict, dict[str, list[tuple[float, float, float]]]]:
    """Return schema-v2 configuration and authored rectangular mesh corners."""
    if name not in SCENARIO_NAMES:
        raise ValueError(f"Unknown visual experiment: {name}")
    sensing = name in {"bistatic_range_locus", "monostatic_range_locus"}
    monostatic = name == "monostatic_range_locus"
    tx = [-4.0, 0.0, 1.5]
    receivers = [_actor("main_rx", [4.0, 0.0, 1.5])]
    if name == "arrival_courtyard":
        walls = {"near_wall": _wall_y(3), "far_wall": _wall_y(-6)}
        receivers.append(_actor("offset_rx", [4.0, 2.0, 1.5]))
    elif name == "angular_corridor":
        walls = {
            "north_wall": _wall_y(3),
            "south_wall": _wall_y(-3),
            "end_wall": _wall_x(10),
        }
    else:
        walls = {"reflecting_wall": _wall_y(8)}
        tx = [0.0 if monostatic else -3.0, 0.0, 1.5]
        receivers = [_actor("main_rx", [0.0 if monostatic else 3.0, 0.0, 1.5])]
    config = {
        "schema_version": 2,
        "scene": {"source": "local", "id": "scene.xml"},
        "timeline": {"steps": 8 if sensing else 1, "sample_rate_hz": 1000.0},
        "raytracing": {
            "enabled": True,
            "carrier_frequency_hz": 3.5e9,
            "bandwidth_hz": 300e6,
            "antenna": {
                "tx": {"pattern": "iso", "polarization": "V", "num_rows": 1, "num_cols": 1},
                "rx": {"pattern": "iso", "polarization": "V", "num_rows": 1, "num_cols": 1},
            },
            "quality": {
                "preset": "low",
                "custom": {
                    "max_depth": 1,
                    "samples_per_src": 10000,
                    "max_num_paths_per_src": 1000,
                    "los": not sensing,
                    "specular_reflection": True,
                    "diffuse_reflection": False,
                    "refraction": False,
                    "seed": 42,
                },
            },
        },
        "actors": {"tx": [_actor("main_tx", tx)], "rx": receivers},
        "data": {"mode": "files", "files": {"directory": "frames", "chunk_size": 8}},
        "generator_summary": {"enabled": False},
    }
    if sensing:
        config["raytracing"].update({"cir_time_steps": 8, "cir_sampling_frequency_hz": 1000.0})
        config["sensing"] = {
            "enabled": True,
            "processing_mode": "coherent_cpi",
            "range_mode": "monostatic" if monostatic else "bistatic",
            "bandwidth": 300e6,
            "prf_hz": 1000.0,
            "fft_size_range": 128,
            "fft_size_doppler": 8,
            "window_range": "boxcar",
            "window_doppler": "boxcar",
            "range_response": {
                "model": "cfr_ifft",
                "num_frequency_bins": 128,
                "frequency_window": "boxcar",
            },
            "raw_payload_policy": "always",
            "noise_enabled": False,
            "clutter_removal_enabled": False,
            "output_range_doppler": True,
            "output_range_profile": True,
            "output_detections": True,
            "persist_detection_stage_artifacts": True,
        }
    return config, walls


def _write_ply(path: Path, corners: list[tuple[float, float, float]]) -> None:
    header = (
        "ply\nformat ascii 1.0\ncomment Procedural visual-experiment rectangle\n"
        "element vertex 4\nproperty float x\nproperty float y\nproperty float z\n"
        "element face 2\nproperty list uchar int vertex_indices\nend_header\n"
    )
    vertices = "".join(f"{x:g} {y:g} {z:g}\n" for x, y, z in corners)
    path.write_text(header + vertices + "3 0 1 2\n3 0 2 3\n", encoding="utf-8")


def build_scenarios(output_dir: Path) -> list[Path]:
    """Write only portable YAML/XML/PLY inputs, leaving generation to the CLI."""
    directories = []
    for name in SCENARIO_NAMES:
        config, walls = scenario_inputs(name)
        directory = output_dir / name
        mesh_dir = directory / "meshes"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        (directory / "scenario.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        shapes = []
        for wall_name, corners in walls.items():
            _write_ply(mesh_dir / f"{wall_name}.ply", corners)
            shapes.append(
                f'  <shape type="ply" id="mesh-{wall_name}">\n'
                f'    <string name="filename" value="meshes/{wall_name}.ply"/>\n'
                '    <boolean name="face_normals" value="true"/>\n'
                '    <ref id="reflector-metal" name="bsdf"/>\n'
                "  </shape>\n"
            )
        xml = (
            '<scene version="3.0.0">\n'
            '  <bsdf type="itu-radio-material" id="reflector-metal">\n'
            '    <string name="type" value="metal"/>\n'
            '    <float name="thickness" value="0.01"/>\n'
            "  </bsdf>\n" + "".join(shapes) + "</scene>\n"
        )
        (directory / "scene.xml").write_text(xml, encoding="utf-8")
        directories.append(directory)
    return directories


def main() -> None:
    """Create standalone input bundles in the explicitly selected directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for directory in build_scenarios(args.output.expanduser().resolve()):
        sys.stdout.write(f"{directory}\n")


if __name__ == "__main__":
    main()
