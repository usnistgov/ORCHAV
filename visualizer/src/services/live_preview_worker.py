"""Subprocess worker for interactive live-preview frame recomputation.

The worker keeps generator/Sionna runtime objects out of the Qt process during
drag editing. It accepts JSON-line commands over stdio, reuses a warm simulation
when the scenario and step window permit it, writes preview frames as pickles,
and reports only small protocol messages back to the UI-side service.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from shared.frames import StandardMPCFrame
from shared.source_identity import loaded_source_identity

from .live_preview_payloads import (
    build_live_overrides,
    decode_orientation_array,
    decode_position_array,
)


class PreviewWorkerState:
    """Keep one Sionna runtime context warm across preview solve requests."""

    def __init__(self) -> None:
        """Initialize an empty runtime cache for one worker process."""
        self.simulation: Any = None
        self.scenario_root: Optional[Path] = None
        self.project_root: Optional[Path] = None
        self.max_step = -1

    def ensure_simulation(self, request: dict[str, Any]) -> Any:
        """Return a simulation compatible with the request's scenario and step."""
        scenario_root = Path(str(request["scenario_root"]))
        project_root_raw = request.get("project_root")
        project_root = Path(project_root_raw) if project_root_raw else None
        step = int(request.get("step", 0) or 0)
        if (
            self.simulation is not None
            and self.scenario_root == scenario_root
            and self.project_root == project_root
            and step <= self.max_step
        ):
            self._apply_solver_settings(request)
            return self.simulation

        self.simulation = _load_simulation(request)
        self.scenario_root = scenario_root
        self.project_root = project_root
        sim_config = getattr(self.simulation, "simulation_config", None)
        try:
            configured_steps = int(getattr(sim_config, "num_steps", step + 1) or step + 1)
        except (TypeError, ValueError):
            configured_steps = step + 1
        self.max_step = max(step, configured_steps - 1, 0)
        self._apply_solver_settings(request)
        return self.simulation

    def _apply_solver_settings(self, request: dict[str, Any]) -> None:
        """Update the cached simulation with the request's preview settings."""
        if self.simulation is None:
            return
        settings = dict(request.get("solver_settings") or {})
        if settings:
            current = getattr(self.simulation, "settings", None)
            if isinstance(current, dict):
                current.update(settings)
            else:
                self.simulation.settings = settings


def _load_simulation(request: dict[str, Any]) -> Any:
    """Build an on-demand generator runtime for one preview scenario."""
    from generator.core.configuration import build_simulation_config
    from generator.core.runtime import build_on_demand_objects
    from generator.core.scenario_actors.runtime import prepare_actor_runtime
    from shared.scenarios import load_scenario_configuration

    scenario_root = Path(str(request["scenario_root"]))
    project_root_raw = request.get("project_root")
    project_root = Path(project_root_raw) if project_root_raw else None
    scenario_cfg = load_scenario_configuration(scenario_root, project_root=project_root)
    simulation_config = build_simulation_config(scenario_cfg)
    step = int(request.get("step", 0) or 0)
    simulation_config.num_steps = max(
        int(getattr(simulation_config, "num_steps", 1) or 1),
        step + 1,
    )

    actor_runtime = prepare_actor_runtime(scenario_cfg)
    tx_configs = list(actor_runtime.transmitters)
    rx_configs = list(actor_runtime.receivers)
    target_configs = list(actor_runtime.targets)
    if not tx_configs or not rx_configs:
        raise RuntimeError("Preview requires at least one TX and one RX in scenario YAML")

    return build_on_demand_objects(
        tx_configs,
        rx_configs,
        target_configs,
        simulation_config,
        scenario_configuration=scenario_cfg,
        motion_mode="step",
        raytracing_settings=dict(request.get("solver_settings") or {}),
    )


def _solve_preview_frame(request: dict[str, Any], simulation: Any) -> StandardMPCFrame:
    """Compute one canonical frame with live-preview provenance."""
    from generator.core.propagation.raytracing import compute_ray_tracing_step
    from generator.io.frames.builder import process_frame_data

    step = int(request.get("step", 0) or 0)
    sequence = int(request.get("sequence", 0) or 0)
    tx_positions = decode_position_array(request.get("tx_positions"), key="tx_positions")
    rx_positions = decode_position_array(request.get("rx_positions"), key="rx_positions")
    target_positions = decode_position_array(
        request.get("target_positions"),
        key="target_positions",
    )
    target_orientations = decode_orientation_array(
        request.get("target_orientations"),
        key="target_orientations",
    )
    overrides = build_live_overrides(
        simulation,
        tx_positions,
        rx_positions,
        target_positions,
        target_orientations,
    )
    raw = compute_ray_tracing_step(simulation, step, live_overrides=overrides)
    if raw is None:
        raise RuntimeError("Sionna preview solve returned no frame")

    settings = dict(request.get("solver_settings") or {})
    timestamp_s = float(time.time())
    provenance = {
        "provider": "live_preview",
        "preview": True,
        "preview_quality": str(request.get("quality", "release")),
        "frame_idx": int(step),
        "sequence_id": int(sequence),
        "timestamp": timestamp_s,
        "solver_settings": settings,
        "seed": int(settings.get("seed", 42)),
        "tx_positions": tx_positions.tolist(),
        "rx_positions": rx_positions.tolist(),
        "target_positions": target_positions.tolist(),
        "target_orientations": target_orientations.tolist(),
        "scenario_root": str(request.get("scenario_root") or ""),
    }
    simulation_config = raw.get("simulation_config")
    return process_frame_data(
        step,
        raw["tx_list"],
        raw["rx_list"],
        raw["paths"],
        raw.get("target_objects", []),
        raw.get("target_managers", []),
        simulation_config,
        material_mapping=raw.get("material_mapping"),
        bandwidth_hz=getattr(simulation_config, "bandwidth_hz", None),
        provenance=provenance,
        timestamp_s=timestamp_s,
    )


def _write_frame(frame: StandardMPCFrame, output_path: Path) -> None:
    """Persist a preview frame for the UI process to load by path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(frame, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _emit(message: dict[str, Any]) -> None:
    """Emit one JSON-line protocol message to stdout."""
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _handle_init(state: PreviewWorkerState, request: dict[str, Any]) -> None:
    """Warm the runtime and notify the UI process that solves may start."""
    state.ensure_simulation(request)
    _emit(
        {
            "type": "ready",
            "source_identity": loaded_source_identity("visualizer").to_dict(),
        }
    )


def _handle_solve(state: PreviewWorkerState, request: dict[str, Any]) -> None:
    """Run one preview solve and report the output pickle path."""
    sequence = int(request.get("sequence", 0) or 0)
    output_path = Path(str(request["output_path"]))
    simulation = state.ensure_simulation(request)
    frame = _solve_preview_frame(request, simulation)
    _write_frame(frame, output_path)
    _emit(
        {
            "type": "result",
            "status": "ok",
            "sequence": sequence,
            "quality": str(request.get("quality", "release")),
            "output_path": str(output_path),
        }
    )


def serve_stdio() -> int:
    """Serve JSON-line init/solve/shutdown commands until stdin closes."""
    state = PreviewWorkerState()
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        try:
            message = json.loads(text)
            command = str(message.get("command", ""))
            if command == "shutdown":
                _emit({"type": "bye"})
                return 0
            request = dict(message.get("request") or {})
            if command == "init":
                _handle_init(state, request)
            elif command == "solve":
                _handle_solve(state, request)
            else:
                _emit({"type": "error", "error": f"Unknown command: {command}"})
        except Exception:
            traceback.print_exc(file=sys.stderr)
            _emit({"type": "error", "error": traceback.format_exc(limit=5)})
    return 0


def run_one_shot(request_path: Path, output_path: Path) -> int:
    """Run a single preview request from disk for diagnostics and tests."""
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        state = PreviewWorkerState()
        simulation = state.ensure_simulation(request)
        frame = _solve_preview_frame(request, simulation)
        _write_frame(frame, output_path)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse worker CLI arguments and select stdio or one-shot mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdio", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if args.stdio:
        return serve_stdio()
    if args.request is None or args.output is None:
        parser.error("--request and --output are required unless --stdio is used")
    return run_one_shot(args.request, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
