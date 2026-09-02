#!/usr/bin/env python3
"""Verify that a committed ORCHAV archive contains every public scenario."""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_SCENARIO_DIRS = frozenset(
    {
        "scenarios/generator/coverage/multi_tx",
        "scenarios/generator/coverage/single_tx",
        "scenarios/generator/mobility_and_orientation/actor_mobility",
        "scenarios/generator/mobility_and_orientation/actor_orientation",
        "scenarios/generator/propagation_and_materials/refraction_and_diffraction",
        "scenarios/generator/propagation_and_materials/scene_diffuse_scattering",
        "scenarios/generator/propagation_and_materials/scene_diffuse_scattering/explicit_concrete",
        "scenarios/generator/propagation_and_materials/scene_diffuse_scattering/itu_preset",
        "scenarios/generator/propagation_and_materials/scene_diffuse_scattering/itu_preset_concrete_override",
        "scenarios/generator/propagation_and_materials/specular_reflection",
        "scenarios/generator/targets/mesh_targets",
        "scenarios/generator/targets/target_diffuse_scattering",
        "scenarios/generator/targets/target_orientation",
        "scenarios/getting_started/hello_world",
        "scenarios/getting_started/hello_world_scripted",
        "scenarios/visualizer/beamforming/circular_rx",
        "scenarios/visualizer/beamforming/multi_device",
        "scenarios/visualizer/data_modes/hdf5_files",
        "scenarios/visualizer/data_modes/live_grpc",
        "scenarios/visualizer/data_modes/remote_hdf5",
        "scenarios/visualizer/data_modes/remote_hdf5/generation",
        "scenarios/visualizer/metrics_evolution",
        "scenarios/visualizer/mpc_inspection",
        "scenarios/visualizer/multi_device_trajectory",
        "scenarios/visualizer/notebook_mode",
        "scenarios/visualizer/statistics",
        "scenarios/visualizer/synthetic_mpc_benchmark",
    }
)

COVERAGE_REQUIRED_PATHS = frozenset(
    {
        "scenarios/generator/coverage/multi_tx/README.md",
        "scenarios/generator/coverage/multi_tx/scenario.yaml",
        "scenarios/generator/coverage/single_tx/README.md",
        "scenarios/generator/coverage/single_tx/scenario.yaml",
    }
)


def required_scenario_archive_paths(
    scenario_directories: Iterable[str] = EXPECTED_SCENARIO_DIRS,
) -> frozenset[str]:
    """Return the canonical YAML and README paths required in an archive."""
    return frozenset(
        path
        for directory in scenario_directories
        for path in (f"{directory}/scenario.yaml", f"{directory}/README.md")
    )


def missing_scenario_archive_paths(members: Iterable[str]) -> list[str]:
    """Return required public scenario files absent from archive members."""
    available = {member.rstrip("/") for member in members}
    required = required_scenario_archive_paths()
    return sorted(required - available)


def git_archive_members(repo_root: Path, treeish: str) -> frozenset[str]:
    """Return member names from a real committed-tree Git archive."""
    result = subprocess.run(
        ["git", "archive", "--format=tar", treeish],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git archive failed for {treeish!r}: {detail}")
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
        return frozenset(member.name.rstrip("/") for member in archive.getmembers())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-ish", default="HEAD")
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    members = git_archive_members(args.repo_root.resolve(), args.tree_ish)
    missing = missing_scenario_archive_paths(members)
    if missing:
        print("Committed archive is missing required public scenario files:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 1
    if not COVERAGE_REQUIRED_PATHS.issubset(members):
        raise AssertionError("coverage archive invariant diverged from scenario inventory")
    print(
        "Committed archive contains "
        f"{len(required_scenario_archive_paths())} required scenario files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
