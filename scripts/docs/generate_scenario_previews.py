#!/usr/bin/env python3
"""Regenerate checked-in 2D preview images for curated ORCHAV scenarios.

The previews are documentation assets, not simulation outputs. The script
captures the scenario actors, renders the generator's 2D scene summary, and
embeds the resulting image in each scenario README.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "docs" / "assets" / "scenarios"

# These scenarios have checked-in preview assets. Every default entry must also
# exist in a release checkout, but not every released scenario needs a preview.
CURATED_SCENARIOS = [
    "scenarios/getting_started/hello_world",
    "scenarios/getting_started/hello_world_scripted",
    "scenarios/generator/mobility_and_orientation/actor_mobility",
    "scenarios/generator/mobility_and_orientation/actor_orientation",
    "scenarios/generator/coverage/single_tx",
    "scenarios/generator/coverage/multi_tx",
    "scenarios/generator/propagation_and_materials/specular_reflection",
    "scenarios/generator/propagation_and_materials/scene_diffuse_scattering",
    "scenarios/generator/propagation_and_materials/refraction_and_diffraction",
    "scenarios/generator/targets/mesh_targets",
    "scenarios/generator/targets/target_diffuse_scattering",
    "scenarios/visualizer/mpc_inspection",
    "scenarios/visualizer/data_modes/live_grpc",
    "scenarios/visualizer/data_modes/hdf5_files",
    "scenarios/visualizer/data_modes/remote_hdf5",
    "scenarios/visualizer/statistics",
    "scenarios/visualizer/synthetic_mpc_benchmark",
    "scenarios/visualizer/multi_device_trajectory",
]


@dataclass
class CapturedScenario:
    tx_configs: list[Any]
    rx_configs: list[Any]
    target_configs: list[Any]
    simulation_config: Any
    scenario_context: Any


@dataclass
class PreviewTargetManager:
    config: Any

    @property
    def meshes(self) -> list[str]:
        return []


def _slug(scenario_dir: Path) -> str:
    return "_".join(scenario_dir.relative_to(REPO_ROOT).parts)


def _load_module(script_path: Path):
    module_name = f"_orchav_preview_{_slug(script_path.parent)}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _capture_from_script(scenario_dir: Path) -> CapturedScenario | None:
    script_path = scenario_dir / "generate.py"
    if not script_path.exists():
        return None
    if "perform_pipeline" not in script_path.read_text(encoding="utf-8"):
        return None

    module = _load_module(script_path)
    captured: dict[str, CapturedScenario] = {}

    def fake_perform_pipeline(
        tx_configs=None,
        rx_configs=None,
        target_configs=None,
        simulation_config=None,
        scenario_configuration=None,
        *args,
        actors=None,
        groups=None,
        **kwargs,
    ):
        del args, kwargs
        if simulation_config is None or scenario_configuration is None:
            raise ValueError("preview capture requires scenario and simulation configs")
        if actors is not None:
            if tx_configs is not None or rx_configs is not None or target_configs is not None:
                raise ValueError("actor specs cannot be mixed with explicit runtime configs")
            captured["scenario"] = _capture_prepared_actors(
                scenario_configuration,
                simulation_config,
                actors=actors,
                groups=groups,
            )
        elif tx_configs is None and rx_configs is None:
            prepared = _capture_prepared_actors(
                scenario_configuration,
                simulation_config,
            )
            captured["scenario"] = (
                prepared
                if target_configs is None
                else _capture_explicit_runtime(
                    prepared.tx_configs,
                    prepared.rx_configs,
                    target_configs,
                    simulation_config,
                    prepared.scenario_context,
                )
            )
        elif (tx_configs is None) != (rx_configs is None):
            raise ValueError("explicit runtime capture requires both TX and RX lists")
        else:
            captured["scenario"] = _capture_explicit_runtime(
                tx_configs,
                rx_configs,
                target_configs,
                simulation_config,
                scenario_configuration,
            )
        return "preview-only"

    module.perform_pipeline = fake_perform_pipeline
    module.main()
    return captured.get("scenario")


def _capture_prepared_actors(
    scenario_configuration: Any,
    simulation_config: Any,
    *,
    actors: Any | None = None,
    groups: Any | None = None,
) -> CapturedScenario:
    """Build preview runtime adapters from immutable actor specifications."""
    from generator.core.scenario_actors.runtime import prepare_actor_runtime

    effective_scenario = scenario_configuration
    if actors is not None:
        effective_scenario = replace(
            scenario_configuration,
            actors=actors,
            groups=(
                tuple(groups)
                if groups is not None
                else tuple(getattr(scenario_configuration, "groups", ()) or ())
            ),
        )
    runtime = prepare_actor_runtime(effective_scenario)
    return CapturedScenario(
        tx_configs=list(runtime.transmitters),
        rx_configs=list(runtime.receivers),
        target_configs=list(runtime.targets),
        simulation_config=simulation_config,
        scenario_context=effective_scenario,
    )


def _capture_explicit_runtime(
    tx_configs: Any,
    rx_configs: Any,
    target_configs: Any,
    simulation_config: Any,
    scenario_configuration: Any,
) -> CapturedScenario:
    """Capture low-level runtime configs supplied by a streaming extension."""
    transmitters = list(tx_configs or [])
    receivers = list(rx_configs or [])
    targets = list(target_configs or [])
    return CapturedScenario(
        transmitters,
        receivers,
        targets,
        simulation_config,
        scenario_configuration,
    )


def _capture_from_yaml(scenario_dir: Path) -> CapturedScenario:
    from generator import build_simulation_config
    from shared.scenarios import load_scenario_configuration

    scenario = load_scenario_configuration(scenario_dir, project_root=REPO_ROOT)
    simulation_config = build_simulation_config(scenario)
    return _capture_prepared_actors(scenario, simulation_config)


def _capture_from_module_constants(scenario_dir: Path) -> CapturedScenario | None:
    script_path = scenario_dir / "generate.py"
    if not script_path.exists():
        return None
    module = _load_module(script_path)
    tx_pos = getattr(module, "TX_POS", None)
    rx_pos = getattr(module, "RX_POS", None)
    if tx_pos is None or rx_pos is None:
        return None

    from generator import build_simulation_config
    from shared.scenarios import (
        ActorsSpec,
        RxActorSpec,
        StationaryMobilitySpec,
        TxActorSpec,
        load_scenario_configuration,
    )

    scenario = load_scenario_configuration(scenario_dir, project_root=REPO_ROOT)
    simulation_config = build_simulation_config(scenario)
    tx_arr = np.asarray(tx_pos, dtype=np.float64).reshape(-1, 3)
    rx_arr = np.asarray(rx_pos, dtype=np.float64).reshape(-1, 3)
    actors = ActorsSpec(
        tx=tuple(
            TxActorSpec(
                name=f"TX{i + 1}",
                mobility=StationaryMobilitySpec(position_m=tuple(float(value) for value in pos)),
            )
            for i, pos in enumerate(tx_arr)
        ),
        rx=tuple(
            RxActorSpec(
                name=f"RX{i + 1}",
                mobility=StationaryMobilitySpec(position_m=tuple(float(value) for value in pos)),
            )
            for i, pos in enumerate(rx_arr)
        ),
    )
    return _capture_prepared_actors(
        scenario,
        simulation_config,
        actors=actors,
    )


def _capture_scenario(scenario_dir: Path) -> CapturedScenario:
    captured = _capture_from_script(scenario_dir)
    if captured is not None:
        return captured
    captured = _capture_from_yaml(scenario_dir)
    if captured.tx_configs or captured.rx_configs or captured.target_configs:
        return captured
    fallback = _capture_from_module_constants(scenario_dir)
    if fallback is not None:
        return fallback
    return captured


def _render_preview(
    scenario_dir: Path,
    captured: CapturedScenario,
    *,
    filename_suffix: str = "summary2d",
) -> Path:
    from generator.core.scenario_actors.state import ActorStateManager
    from generator.figures.generator_summary_fig import create_2d_scene_summary_figures

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ASSET_DIR / f"{_slug(scenario_dir)}_{filename_suffix}.png"
    target_managers = [PreviewTargetManager(config=cfg) for cfg in captured.target_configs]
    viz_config = {}
    if captured.scenario_context is not None:
        summary_config = getattr(captured.scenario_context, "generator_summary", None) or {}
        viz_config = summary_config.get("visualization", {}) or {}
    actor_state_manager = ActorStateManager(
        tx_configs=captured.tx_configs,
        rx_configs=captured.rx_configs,
        target_managers=target_managers,
        steps=captured.simulation_config.num_steps,
        duration=captured.simulation_config.duration,
        motion_mode="cached",
        mesh_update_interval_s=getattr(captured.simulation_config, "mesh_update_interval_s", None),
    )
    create_2d_scene_summary_figures(
        captured.tx_configs,
        captured.rx_configs,
        target_managers,
        captured.simulation_config,
        actor_state_manager,
        output_path=output_path,
        scenario_context=captured.scenario_context,
        rendering_mode="auto",
        resolution=0.25,
        show_material_legend=bool(viz_config.get("scene2d_material_legend", False)),
        actor_label_mode=str(viz_config.get("actor_label_mode", "role")),
    )
    return output_path


def _relative_link(readme_path: Path, image_path: Path) -> str:
    return Path(os.path.relpath(image_path, readme_path.parent)).as_posix()


def _upsert_preview_section(scenario_dir: Path, image_path: Path) -> None:
    readme_path = scenario_dir / "README.md"
    if not readme_path.exists():
        return
    rel = _relative_link(readme_path, image_path)
    section = "## Scene Layout\n\n" f"![2D scene layout]({rel})\n"
    text = readme_path.read_text(encoding="utf-8")
    heading = re.search(r"(?im)^## (scene layout|preview)\s*$", text)
    if heading:
        before = text[: heading.start()]
        rest = text[heading.end() :]
        next_heading = re.search(r"(?m)^## ", rest)
        body = rest[: next_heading.start()] if next_heading else rest
        after = rest[next_heading.start() :] if next_heading else ""
        preserved = re.sub(
            r"^\s*!\[[^\]]*\]\([^)]+\)\s*\n?",
            "",
            body,
            count=1,
        ).lstrip("\n")
        text = before.rstrip() + "\n\n" + section
        if preserved:
            text += "\n" + preserved.rstrip() + "\n"
        if after:
            text += "\n" + after.lstrip()
        elif not preserved:
            text += "\n"
    else:
        marker = "\n## "
        if marker in text:
            before, after = text.split(marker, 1)
            text = before.rstrip() + "\n\n" + section + "\n## " + after
        else:
            text = text.rstrip() + "\n\n" + section + "\n"
    readme_path.write_text(text, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate curated 2D preview images for scenario README files.",
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        help=("Scenario directories to render. Defaults to the curated " "scenario list."),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the curated scenario set without rendering.",
    )
    return parser.parse_args()


def _scenario_list(args: argparse.Namespace) -> list[str]:
    if args.scenarios:
        return [str(Path(s).as_posix()).rstrip("/") for s in args.scenarios]
    return CURATED_SCENARIOS


def main() -> None:
    args = _parse_args()
    if args.list:
        for rel in CURATED_SCENARIOS:
            print(rel)
        return

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    generated: list[Path] = []
    for rel in _scenario_list(args):
        scenario_dir = REPO_ROOT / rel
        if not (scenario_dir / "scenario.yaml").exists():
            raise FileNotFoundError(f"Missing scenario.yaml: {scenario_dir}")
        print(f"[preview] {rel}")
        captured = _capture_scenario(scenario_dir)
        if not (captured.tx_configs or captured.rx_configs or captured.target_configs):
            raise RuntimeError(f"No previewable actors for {rel}")
        image_path = _render_preview(scenario_dir, captured)
        _upsert_preview_section(scenario_dir, image_path)
        generated.append(image_path)

    print(f"Generated {len(generated)} preview image(s) in {ASSET_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
