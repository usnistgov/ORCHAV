#!/usr/bin/env python3
"""Command-line entry point for running generator scenarios.

This file backs the ``orchav-generator`` console script and ``python -m
generator``.  It is deliberately thin: it parses CLI arguments, lists curated
scenario entry points, and hands YAML scenarios to the core pipeline.  Python
``generate.py`` scenarios remain script-owned and are listed as direct commands
rather than being imported here.
"""

import argparse
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple, Sequence

from shared.source_identity import (
    EXPECTED_SOURCE_IDENTITY_ENV,
    SourceIdentity,
    SourceIdentityError,
    loaded_source_identity,
)

_SCRIPT_DIRS = [
    ("Getting Started", "scenarios/getting_started"),
    ("Generator Examples", "scenarios/generator"),
    ("Visualizer Examples", "scenarios/visualizer"),
    ("Paper Reproductions", "scenarios/papers"),
]
# The CLI catalog is curated user navigation, not a raw filesystem listing.
# Omit underscore-prefixed or explicitly excluded support directories.
_NON_CATALOG_PATH_PARTS = {"private"}
_AUTHORING_SNAPSHOT_PREFIX = ".orchav-aq-"
_CANONICAL_FRAMES_DIRECTORY = "frames"
_GENERATOR_DATA_MODES = ("files", "live_grpc")


class ScenarioEntry(NamedTuple):
    """Catalog entry for a curated scenario."""

    path: Path
    kind: str
    command: str


def _project_root() -> Path:
    """Return the repository root used for scenario discovery."""
    return Path(__file__).resolve().parent.parent


def _display_path(path: str | Path) -> str:
    """Return a shell/documentation-friendly path on every platform."""
    return Path(path).as_posix()


def _verified_source_identity() -> SourceIdentity:
    """Return this generator's identity after checking its parent contract."""
    actual = loaded_source_identity("generator")
    raw_expected = os.environ.get(EXPECTED_SOURCE_IDENTITY_ENV)
    if raw_expected is None:
        return actual

    try:
        expected_payload = json.loads(raw_expected)
    except json.JSONDecodeError as exc:
        raise SourceIdentityError(
            f"{EXPECTED_SOURCE_IDENTITY_ENV} is not valid JSON: {exc.msg}"
        ) from exc
    try:
        expected = SourceIdentity.from_mapping(expected_payload)
    except SourceIdentityError as exc:
        raise SourceIdentityError(f"{EXPECTED_SOURCE_IDENTITY_ENV} is invalid: {exc}") from exc
    if not actual.matches(expected):
        raise SourceIdentityError(
            "generator source identity does not match the launching process: "
            f"expected {expected.to_dict()!r}, got {actual.to_dict()!r}"
        )
    return actual


def _is_catalog_path(path: Path, project_root: Path) -> bool:
    """Return whether a discovered scenario path belongs in the CLI catalog."""
    try:
        relative = path.relative_to(project_root)
    except ValueError:
        return False
    return all(
        not part.startswith("_") and part not in _NON_CATALOG_PATH_PARTS for part in relative.parts
    )


def _scenario_data_mode(scenario_yaml: Path) -> str | None:
    """Read the declared data mode needed to classify a catalog entry.

    Catalog discovery intentionally reads only the small ``data.mode`` contract.
    Invalid scenario files remain visible in the catalog so users can run the
    validator and receive its complete diagnostic.
    """
    import yaml

    try:
        payload = yaml.safe_load(scenario_yaml.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return None
    if not isinstance(payload, Mapping):
        return None
    data = payload.get("data")
    if not isinstance(data, Mapping):
        return None
    mode = data.get("mode")
    return mode.strip() if isinstance(mode, str) else None


def discover_scenarios() -> list[tuple[str, list[ScenarioEntry]]]:
    """Discover curated frame-producer scenario entry points.

    A directory with ``generate.py`` is script-owned even when it also carries
    shared settings in ``scenario.yaml``. Consumer-only ``remote_hdf5`` YAML
    configurations are omitted because they are opened by the Visualizer, not
    run by the Generator.
    """
    project_root = _project_root()
    results: list[tuple[str, list[ScenarioEntry]]] = []
    for label, rel_dir in _SCRIPT_DIRS:
        d = project_root / rel_dir
        if not d.exists():
            continue

        yaml_dirs = {
            path.parent
            for path in d.glob("**/scenario.yaml")
            if _is_catalog_path(path, project_root)
        }
        script_dirs = {
            path.parent for path in d.glob("**/generate.py") if _is_catalog_path(path, project_root)
        }

        entries: list[ScenarioEntry] = []
        for scenario_dir in sorted(yaml_dirs | script_dirs):
            rel_path = scenario_dir.relative_to(project_root)
            display_path = _display_path(rel_path)
            has_script = scenario_dir in script_dirs
            if has_script:
                entries.append(
                    ScenarioEntry(
                        path=rel_path,
                        kind="Python-scripted",
                        command=f"python {display_path}/generate.py",
                    )
                )
                continue

            if _scenario_data_mode(scenario_dir / "scenario.yaml") == "remote_hdf5":
                continue

            entries.append(
                ScenarioEntry(
                    path=rel_path,
                    kind="YAML",
                    command=f"orchav-generator {display_path}/",
                )
            )

        if entries:
            results.append((label, entries))
    return results


def _run_from_yaml(
    scenario_path: Path,
    geometry_only: bool = False,
    progress_format: str = "text",
    *,
    data_mode: str | None = None,
    grpc_port: int | None = None,
    grpc_bind_host: str | None = None,
    _authoring_snapshot_yaml: Path | None = None,
) -> Any:
    """Run a canonical scenario, optionally from a trusted Builder snapshot.

    The private snapshot supplies configuration bytes only. ``scenario_path``
    remains authoritative for path resolution and fixed output publication.
    """
    source_identity = _verified_source_identity()

    from shared.scenarios import load_scenario_configuration
    from shared.scenarios.paths import find_project_root

    resolved = scenario_path.resolve()
    scenario_root = resolved if resolved.is_dir() else resolved.parent
    snapshot_yaml = (
        _validate_authoring_snapshot_path(scenario_root, _authoring_snapshot_yaml)
        if _authoring_snapshot_yaml is not None
        else None
    )
    # Establish the project-root policy before loading YAML so relative paths
    # inside scenario files resolve the same way as they do in scripted runs.
    find_project_root(scenario_root)

    # Import the heavy generator facade only for execution.  Listing scenarios
    # and printing --help should not initialize Sionna/Mitsuba.
    from generator import build_simulation_config, perform_pipeline
    from generator.core.pipeline.progress import JsonlProgressReporter

    scenario = (
        load_scenario_configuration(scenario_root, yaml_path=snapshot_yaml)
        if snapshot_yaml is not None
        else load_scenario_configuration(resolved)
    )
    if data_mode is not None:
        if data_mode not in _GENERATOR_DATA_MODES:
            raise ValueError(f"Unsupported generator data mode: {data_mode}")
        scenario.data_mode = data_mode
    effective_data_mode = str(getattr(scenario, "data_mode", "files"))
    if effective_data_mode == "remote_hdf5":
        raise ValueError(
            "remote_hdf5 is a visualizer playback mode; generate with data mode "
            "files, then run generator.io.grpc.file_server"
        )
    if (grpc_port is not None or grpc_bind_host is not None) and effective_data_mode != "live_grpc":
        raise ValueError(
            "--grpc-port and --bind-host require the effective data mode to be live_grpc"
        )
    if snapshot_yaml is not None:
        _require_authoring_snapshot_output_contract(scenario, scenario_root)
    sim_config = build_simulation_config(scenario)

    reporter = None
    if progress_format == "jsonl":
        first_step = int(getattr(sim_config, "start_step", 0) or 0)
        num_steps = int(getattr(sim_config, "num_steps", 0) or 0)
        reporter = JsonlProgressReporter(
            first_step=first_step,
            total_steps=max(0, num_steps - first_step),
            source_identity=source_identity.to_dict(),
        )
        reporter.run_started()

    try:
        result = perform_pipeline(
            simulation_config=sim_config,
            scenario_configuration=scenario,
            geometry_only=geometry_only,
            grpc_port=grpc_port,
            grpc_bind_host=grpc_bind_host,
            on_step_complete=(reporter.step_completed if reporter is not None else None),
            show_progress=reporter is None,
        )
        if reporter is not None:
            rt_enabled = bool((getattr(scenario, "raytracing", {}) or {}).get("enabled", False))
            if rt_enabled and not geometry_only and reporter.completed_steps < reporter.total_steps:
                raise RuntimeError(
                    "generator stopped before all output steps completed "
                    f"({reporter.completed_steps}/{reporter.total_steps})"
                )
            reporter.run_completed(str(result) if result is not None else None)
    except Exception as exc:
        if reporter is not None:
            reporter.run_failed(str(exc), error_type=type(exc).__name__)
        raise

    if result and not hasattr(result, "shutdown"):
        logging.getLogger(__name__).info("Output: %s", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the small CLI surface for the generator package."""
    parser = argparse.ArgumentParser(
        prog="orchav-generator",
        description="Run an ORCHAV YAML scenario. With no scenario, list curated producer entry points.",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        help=(
            "Scenario directory or scenario.yaml file to run. Use generate.py "
            "directly for Python-scripted scenarios."
        ),
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help=(
            "Use minimum-budget ray tracing for file output to check the "
            "environment scene, actors, and ordinary frame/topology output."
        ),
    )
    parser.add_argument(
        "--progress-format",
        choices=("text", "jsonl"),
        default="text",
        help=(
            "Progress output format for scenario runs. 'text' keeps the interactive "
            "stderr display; 'jsonl' writes versioned events to stdout and logs to stderr."
        ),
    )
    parser.add_argument(
        "--data-mode",
        choices=_GENERATOR_DATA_MODES,
        help=(
            "Override the scenario data mode for this run. 'files' writes the "
            "canonical frame set; 'live_grpc' starts the on-demand Live Generator server."
        ),
    )
    parser.add_argument(
        "--grpc-port",
        type=_parse_grpc_port,
        help="Override the Live Generator port declared by the scenario.",
    )
    parser.add_argument(
        "--bind-host",
        dest="grpc_bind_host",
        help=(
            "Override the Live Generator listener address. This does not change "
            "the endpoint advertised to Visualizer clients."
        ),
    )
    parser.add_argument(
        "--_authoring-snapshot-yaml",
        dest="_authoring_snapshot_yaml",
        help=argparse.SUPPRESS,
    )
    return parser


def _parse_grpc_port(value: str) -> int:
    """Return a valid TCP port for an explicit CLI live-stream override."""
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gRPC port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("gRPC port must be between 1 and 65535")
    return port


def _validate_scenario_argument(parser: argparse.ArgumentParser, scenario_arg: str) -> Path:
    """Return a validated scenario path or stop with a CLI-friendly error."""
    scenario_path = Path(scenario_arg).expanduser()

    if not scenario_path.exists():
        parser.error(
            f"scenario path does not exist: {_display_path(scenario_path)}\n"
            "Run `orchav-generator` with no arguments to see included scenarios."
        )

    if scenario_path.is_dir():
        if not (scenario_path / "scenario.yaml").is_file():
            parser.error(
                "scenario directory does not contain scenario.yaml: "
                f"{_display_path(scenario_path)}\n"
                "Pass a scenario directory with scenario.yaml or pass that canonical file."
            )
    elif not scenario_path.is_file() or scenario_path.name != "scenario.yaml":
        parser.error(
            "normal generation accepts only a scenario directory or its canonical "
            f"scenario.yaml: {_display_path(scenario_path)}"
        )

    return scenario_path


def _validate_authoring_snapshot_path(scenario_root: Path, snapshot_path: Path) -> Path:
    """Return a private direct-child snapshot without changing scenario identity."""

    root = Path(scenario_root).expanduser().resolve()
    entry = Path(os.path.abspath(Path(snapshot_path).expanduser()))
    is_junction = getattr(entry, "is_junction", None)
    try:
        indirect = entry.is_symlink() or (callable(is_junction) and is_junction())
        real_file = entry.is_file()
        snapshot = entry.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"could not inspect private authoring snapshot: {exc}") from exc
    if (
        indirect
        or not real_file
        or snapshot.parent != root
        or not entry.name.startswith(_AUTHORING_SNAPSHOT_PREFIX)
        or entry.suffix.lower() != ".yaml"
    ):
        raise ValueError(
            "authoring snapshot must be a real private .orchav-aq-*.yaml file "
            "directly below the authoritative scenario root"
        )
    return snapshot


def _require_authoring_snapshot_output_contract(scenario: Any, scenario_root: Path) -> None:
    """Require snapshot configuration to retain the authoritative fixed output root."""

    root = Path(scenario_root).resolve()
    configured_root = Path(getattr(scenario, "root", "")).resolve()
    configured_frames = Path(getattr(scenario, "frames_dir", "")).resolve()
    declared_frames = getattr(scenario, "frames_directory", None)
    if (
        configured_root != root
        or declared_frames != _CANONICAL_FRAMES_DIRECTORY
        or configured_frames != root / _CANONICAL_FRAMES_DIRECTORY
    ):
        raise ValueError(
            "private authoring snapshots may supply configuration only; outputs remain "
            "fixed at the authoritative scenario root"
        )


def format_scenario_catalog() -> str:
    """Return the human-readable catalog printed when no scenario is given."""
    sections = discover_scenarios()
    if not sections:
        return "\n".join(
            [
                "ORCHAV Generator - Available Scenarios",
                "=" * 60,
                "",
                "No curated scenario entry points found.",
                "Expected locations: scenarios/<category>/**/scenario.yaml or generate.py",
            ]
        )

    lines = [
        "ORCHAV Generator - Available Scenarios",
        "=" * 60,
        "",
        "YAML scenarios run through orchav-generator. Python-scripted scenarios",
        "run directly through their generate.py driver.",
        "",
        "Examples:",
        "  orchav-generator scenarios/getting_started/hello_world/",
        "  python scenarios/getting_started/hello_world_scripted/generate.py",
        "",
    ]

    for label, entries in sections:
        lines.append(f"{label}:")
        for entry in entries:
            lines.append(f"  [{entry.kind}] {_display_path(entry.path)}/")
            lines.append(f"    {entry.command}")
        lines.append("")

    return "\n".join(lines).rstrip()


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.scenario:
        scenario_path = _validate_scenario_argument(parser, args.scenario)
        transport_overrides = {}
        if args.data_mode is not None:
            transport_overrides["data_mode"] = args.data_mode
        if args.grpc_port is not None:
            transport_overrides["grpc_port"] = args.grpc_port
        if args.grpc_bind_host is not None:
            transport_overrides["grpc_bind_host"] = args.grpc_bind_host
        snapshot_yaml = None
        if args._authoring_snapshot_yaml:
            if EXPECTED_SOURCE_IDENTITY_ENV not in os.environ:
                parser.error(
                    "the private authoring snapshot option is reserved for Scenario Builder"
                )
            scenario_root = scenario_path if scenario_path.is_dir() else scenario_path.parent
            try:
                snapshot_yaml = _validate_authoring_snapshot_path(
                    scenario_root,
                    Path(args._authoring_snapshot_yaml),
                )
            except ValueError as exc:
                parser.error(str(exc))
        if snapshot_yaml is not None:
            _run_from_yaml(
                scenario_path,
                geometry_only=args.geometry_only,
                progress_format=args.progress_format,
                _authoring_snapshot_yaml=snapshot_yaml,
                **transport_overrides,
            )
        else:
            _run_from_yaml(
                scenario_path,
                geometry_only=args.geometry_only,
                progress_format=args.progress_format,
                **transport_overrides,
            )
        return

    if args.geometry_only:
        parser.error("--geometry-only requires a scenario path")
    if args.progress_format != "text":
        parser.error("--progress-format requires a scenario path")
    if args._authoring_snapshot_yaml:
        parser.error("the private authoring snapshot option requires a scenario path")
    if args.data_mode is not None or args.grpc_port is not None or args.grpc_bind_host is not None:
        parser.error("data-mode and live gRPC overrides require a scenario path")

    print(format_scenario_catalog())  # noqa: T201 - CLI catalog output


if __name__ == "__main__":
    main()
