"""Notebook-integrated generator for ORCHAV.

Provides helpers to run Sionna RT ray tracing from notebook cells and reload
the results into the visualizer without restarting the kernel.

Two usage levels:

* **Integrated** — call ``viz.regenerate(tx_positions=..., rx_positions=...)``
  on a ``PygfxNotebookViz`` instance.
* **Standalone** — call ``generate_frames("scenarios/hello_world", ...)`` in
  one cell, then create a fresh viz object to inspect the output.

Both write standard HDF5 frames via ``perform_offline_pipeline`` so the full
visualizer stack (caching, canonical data, view models) works unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from shared.logging import get_logger

logger = get_logger("orchav.notebook_generator")


class NotebookGenerator:
    """Runs the Sionna RT pipeline from notebook cells.

    Holds resolved paths for a scenario and builds
    ``TransmitterConfig``/``ReceiverConfig`` from simple position lists so
    users don't have to import the full generator config API.

    Args:
        scene_xml_path: Absolute path to the Mitsuba XML scene file.
        project_root: Project root directory.
        scenario_root: Scenario output directory (frames written here).
        scenario_configuration: The full ``ScenarioConfiguration`` from
            ``shared.scenarios``.
    """

    def __init__(
        self,
        scene_xml_path: str,
        project_root: Path,
        scenario_root: Path,
        scenario_configuration: Any,
    ) -> None:
        """Store resolved scenario paths for later notebook-triggered generation."""
        self._scene_xml_path = scene_xml_path
        self._project_root = project_root
        self._scenario_root = scenario_root
        self._scenario_configuration = scenario_configuration

    def generate(
        self,
        tx_positions: Sequence[Sequence[float]],
        rx_positions: Sequence[Sequence[float]],
        *,
        quality: str = "low",
        steps: int = 1,
        duration: float | None = None,
        target_configs: list | None = None,
        carrier_frequency_hz: float | None = None,
        bandwidth_hz: float | None = None,
    ) -> list[int]:
        """Run ray tracing and write HDF5 frames to the scenario directory.

        Args:
            tx_positions: TX positions as ``[[x, y, z], ...]``.
            rx_positions: RX positions as ``[[x, y, z], ...]``.
            quality: Ray tracing quality preset
                (``"ultra-low"``, ``"low"``, ``"medium"``, ``"high"``).
            steps: Number of simulation steps.
            duration: Simulation duration in seconds.  Defaults to
                ``steps`` (i.e. 1 s per step).
            target_configs: Optional list of ``TargetConfig`` objects.
            carrier_frequency_hz: Override carrier frequency (Hz).
            bandwidth_hz: Override channel bandwidth (Hz).

        Returns:
            List of generated frame indices (e.g. ``[0]`` for a single step).
        """
        from generator.core.configuration import (
            ReceiverConfig,
            SimulationConfig,
            TransmitterConfig,
            build_simulation_config,
        )
        from generator.core.mobility import StationaryMobility
        from generator.core.orientation import FixedOrientationSpec
        from generator.core.pipeline import perform_offline_pipeline

        if duration is None:
            duration = float(steps)

        tx_configs = [
            TransmitterConfig(
                name=f"TX{i}",
                mobility=StationaryMobility(start_pos=tuple(pos)),
                orientation=FixedOrientationSpec(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0),
            )
            for i, pos in enumerate(tx_positions)
        ]

        rx_configs = [
            ReceiverConfig(
                name=f"RX{i}",
                mobility=StationaryMobility(start_pos=tuple(pos)),
                orientation=FixedOrientationSpec(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0),
            )
            for i, pos in enumerate(rx_positions)
        ]

        overrides: dict[str, Any] = {
            "num_steps": steps,
            "duration": duration,
            "quality": quality,
        }
        if carrier_frequency_hz is not None:
            overrides["carrier_frequency_hz"] = carrier_frequency_hz
        if bandwidth_hz is not None:
            overrides["bandwidth_hz"] = bandwidth_hz

        sim_config: SimulationConfig = build_simulation_config(
            self._scenario_configuration, overrides=overrides
        )

        logger.info(
            "Running pipeline: %d TX, %d RX, %d steps, quality=%s",
            len(tx_configs),
            len(rx_configs),
            steps,
            quality,
        )

        perform_offline_pipeline(
            tx_configs,
            rx_configs,
            target_configs or [],
            sim_config,
            self._scenario_configuration,
        )

        frame_indices = list(range(steps))
        logger.info("Generated %d frames in %s", len(frame_indices), self._scenario_root)
        return frame_indices


def generate_frames(
    scenario_path: str | Path,
    *,
    tx_positions: Sequence[Sequence[float]],
    rx_positions: Sequence[Sequence[float]],
    project_root: str | Path | None = None,
    quality: str = "low",
    steps: int = 1,
    duration: float | None = None,
    target_configs: list | None = None,
    carrier_frequency_hz: float | None = None,
    bandwidth_hz: float | None = None,
) -> list[int]:
    """Standalone helper: load a scenario, run ray tracing, write frames.

    This is a convenience function for the two-cell workflow where you
    generate in one cell and create a new ``PygfxNotebookViz`` in the next.

    Args:
        scenario_path: Path to scenario directory or YAML file.
        tx_positions: TX positions as ``[[x, y, z], ...]``.
        rx_positions: RX positions as ``[[x, y, z], ...]``.
        project_root: Explicit project root (auto-detected if ``None``).
        quality: Ray tracing quality preset.
        steps: Number of simulation steps.
        duration: Simulation duration in seconds.
        target_configs: Optional list of ``TargetConfig`` objects.
        carrier_frequency_hz: Override carrier frequency (Hz).
        bandwidth_hz: Override channel bandwidth (Hz).

    Returns:
        List of generated frame indices.
    """
    from shared.scenarios import load_scenario_configuration

    if project_root is not None:
        proj = Path(project_root).resolve()
    else:
        from shared.scenarios.paths import find_project_root

        proj = find_project_root(Path.cwd())

    scenario_abs = Path(scenario_path)
    if not scenario_abs.is_absolute():
        scenario_abs = proj / scenario_path

    scenario_cfg = load_scenario_configuration(scenario_abs, project_root=proj)

    if scenario_cfg.scene_xml is None:
        raise RuntimeError(
            f"No scene XML resolved for scenario at {scenario_abs}. "
            "Check the scene.source/scene.id in scenario.yaml."
        )

    gen = NotebookGenerator(
        scene_xml_path=str(scenario_cfg.scene_xml),
        project_root=proj,
        scenario_root=scenario_cfg.root,
        scenario_configuration=scenario_cfg,
    )

    return gen.generate(
        tx_positions,
        rx_positions,
        quality=quality,
        steps=steps,
        duration=duration,
        target_configs=target_configs,
        carrier_frequency_hz=carrier_frequency_hz,
        bandwidth_hz=bandwidth_hz,
    )
