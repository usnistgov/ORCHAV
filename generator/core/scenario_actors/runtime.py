"""Expose immutable prepared actor poses to Sionna scene construction.

The pose kernel owns mobility and orientation sampling.  This module supplies
names, initial positions, mesh settings, and the small protocols consumed by
the live-scene services.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from generator.core.target import TargetConfig
from generator.core.target.metadata import load_target_asset_metadata
from generator.core.target.representation import NativeTargetConfig
from shared.scenarios.actors import NativeMeshGeometrySpec, TargetActorSpec
from shared.scenarios.paths import resolve_actor_resource

from ._adapters import value
from .preparation import prepare_scenario
from .quaternion import Quaternion
from .types import PreparedActorPose, PreparedMobility, PreparedOrientation, PreparedScenario

if TYPE_CHECKING:
    from .animated_asset import PreparedAnimatedAsset

_MESH_SUFFIXES = frozenset({".obj", ".ply", ".stl", ".glb", ".gltf"})


class PreparedMobilityAdapter:
    """Read-only mobility protocol backed by exact kernel output."""

    def __init__(self, mobility: PreparedMobility, timeline_steps: int, duration_s: float):
        self._mobility = mobility
        self._steps = int(timeline_steps)
        self._duration_s = float(duration_s)

    @property
    def start_pos(self) -> tuple[float, float, float]:
        """Return the first prepared position used for scene construction."""

        return self._mobility.positions_m[0]

    @property
    def has_physical_motion(self) -> bool:
        """Report whether the prepared samples represent physical motion."""

        return self._mobility.has_physical_velocity

    def prepare(
        self,
        steps: int,
        duration: float,
        initial_position: object | None = None,
    ) -> None:
        """Validate that consumers use the timeline already prepared for actors."""

        del initial_position
        if int(steps) != self._steps or abs(float(duration) - self._duration_s) > 1e-9:
            raise ValueError(
                "prepared actor mobility cannot be resampled with a different timeline"
            )

    def prepared_positions(self) -> list[tuple[float, float, float]]:
        """Return an ordinary list for Sionna-facing cache code."""

        return list(self._mobility.positions_m)

    def prepared_velocities(self) -> list[tuple[float, float, float]]:
        """Return velocities computed by the same canonical pose kernel."""

        return list(self._mobility.velocities_mps)


class PreparedOrientationAdapter:
    """Read-only degree-valued orientation protocol backed by quaternions."""

    def __init__(
        self,
        orientation: PreparedOrientation,
        timeline_steps: int,
        duration_s: float,
    ) -> None:
        self._orientation = orientation
        self._steps = int(timeline_steps)
        self._duration_s = float(duration_s)

    def prepare(
        self,
        steps: int,
        duration: float,
        context: object | None = None,
    ) -> None:
        """Validate the prepared timeline; cross-actor context is already resolved."""

        del context
        if int(steps) != self._steps or abs(float(duration) - self._duration_s) > 1e-9:
            raise ValueError(
                "prepared actor orientation cannot be resampled with a different timeline"
            )

    def orientations(self) -> list[tuple[float, float, float]]:
        """Return yaw/pitch/roll degrees at the Sionna engine boundary."""

        return list(self._orientation.euler_deg)

    @property
    def asset_alignment_applied(self) -> bool:
        """Report whether canonical preparation already composed asset alignment."""

        return self._orientation.asset_alignment_applied


@dataclass(slots=True)
class RadioActorRuntime:
    """Minimal live-scene configuration for a TX or RX actor."""

    name: str
    role: str
    mobility: PreparedMobilityAdapter
    orientation: PreparedOrientationAdapter
    power_dbm: float | None = None

    @property
    def initial_position(self) -> tuple[float, float, float]:
        return self.mobility.start_pos


@dataclass(frozen=True, slots=True)
class ActorRuntimeSet:
    """Role-grouped scene adapters plus their immutable prepared source."""

    prepared: PreparedScenario
    transmitters: tuple[RadioActorRuntime, ...]
    receivers: tuple[RadioActorRuntime, ...]
    targets: tuple[TargetConfig | NativeTargetConfig, ...]


def prepare_actor_runtime(
    scenario_configuration: object,
    *,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> ActorRuntimeSet:
    """Prepare a validated scenario once and build Sionna scene adapters.

    Target asset front-axis metadata is composed in quaternion space before
    Euler conversion.  The same ``PreparedScenario`` therefore supplies the
    builder preview and every generated frame without target-only correction
    math in either consumer.
    """

    alignments = target_asset_alignments(scenario_configuration)
    root = Path(value(scenario_configuration, "root", default=Path.cwd()))
    # Scoped to this preparation: later runs must reauthenticate source files.
    animations: dict[str, PreparedAnimatedAsset] = {}
    prepared = prepare_scenario(
        scenario_configuration,
        base_dir=root,
        asset_alignments=alignments,
        _animated_assets=animations,
        on_progress=on_progress,
    )
    actors = value(scenario_configuration, "actors")
    specs_by_name = {
        str(value(actor, "name")): actor
        for section in ("tx", "rx", "targets")
        for actor in tuple(value(actors, section, default=()) or ())
    }
    timeline = value(scenario_configuration, "timeline")
    timeline_steps = int(value(timeline, "steps"))
    duration_s = float(value(timeline, "duration_s"))

    transmitters = tuple(
        _radio_runtime(
            actor,
            specs_by_name[actor.name],
            steps=timeline_steps,
            duration_s=duration_s,
        )
        for actor in prepared.actors_for_role("tx")
    )
    receivers = tuple(
        _radio_runtime(
            actor,
            specs_by_name[actor.name],
            steps=timeline_steps,
            duration_s=duration_s,
        )
        for actor in prepared.actors_for_role("rx")
    )
    targets = tuple(
        _target_runtime(
            actor,
            specs_by_name[actor.name],
            scenario_configuration=scenario_configuration,
            animation=animations.get(actor.name),
            on_progress=on_progress,
        )
        for actor in prepared.actors_for_role("target")
    )
    return ActorRuntimeSet(prepared, transmitters, receivers, targets)


def _radio_runtime(
    actor: PreparedActorPose,
    spec: object,
    *,
    steps: int,
    duration_s: float,
) -> RadioActorRuntime:
    power = value(spec, "power_dbm", default=None)
    return RadioActorRuntime(
        name=actor.name,
        role=actor.role,
        mobility=PreparedMobilityAdapter(actor.mobility, steps, duration_s),
        orientation=PreparedOrientationAdapter(
            actor.orientation,
            steps,
            duration_s,
        ),
        power_dbm=None if power is None else float(power),
    )


def _target_runtime(
    actor: PreparedActorPose,
    spec: object,
    *,
    scenario_configuration: object,
    animation: PreparedAnimatedAsset | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> TargetConfig | NativeTargetConfig:
    if str(value(spec, "representation", default="mesh_material")) != "mesh_material":
        return _native_target_runtime(actor, spec, scenario_configuration=scenario_configuration)
    asset = value(spec, "asset")
    if str(value(asset, "source")) == "animated_asset":
        from .animated_asset import materialize_animated_meshes

        if animation is None:
            raise ValueError("animated runtime requires its prepared animation binding")
        root = Path(value(scenario_configuration, "root", default=Path.cwd())).resolve()
        resolved_mesh_directory = materialize_animated_meshes(
            animation, root, on_progress=on_progress
        )
        mesh_directory = resolved_mesh_directory.relative_to(root).as_posix()
        mesh_pattern = "sample_*.ply"
    else:
        mesh_directory, resolved_mesh_directory, mesh_pattern = _target_mesh_location(
            asset,
            scenario_configuration=scenario_configuration,
            target_name=actor.name,
        )
    # The animated lane is already resolved at exact physical timeline times.
    # Ordinary mesh sequences retain their existing start/stride/cadence rules.
    timeline = value(scenario_configuration, "timeline")
    mobility = PreparedMobilityAdapter(
        actor.mobility,
        int(value(timeline, "steps")),
        float(value(timeline, "duration_s")),
    )
    orientation = PreparedOrientationAdapter(
        actor.orientation,
        int(value(timeline, "steps")),
        float(value(timeline, "duration_s")),
    )
    return TargetConfig(
        name=actor.name,
        mobility=mobility,
        mesh_pattern=mesh_pattern,
        mesh_directory=mesh_directory,
        resolved_mesh_directory=resolved_mesh_directory,
        scale=float(value(asset, "scale", default=1.0)),
        orientation=orientation,
        material_type=str(value(asset, "material_type", default="glass")),
        switch_meshes=animation is not None or bool(value(asset, "switch_meshes", default=False)),
        use_ply_position=False,
        mesh_start_index=int(value(asset, "start_index", default=0)),
        mesh_frame_stride=int(value(asset, "frame_stride", default=1)),
        mesh_end_behavior=str(value(asset, "mesh_end_behavior", default="loop")),
        _initial_position=actor.positions_m[0],
        animated_asset=animation,
        asset_front_yaw_offset_deg=0.0 if animation is not None else None,
    )


def _native_target_runtime(
    actor: PreparedActorPose,
    spec: object,
    *,
    scenario_configuration: object,
) -> NativeTargetConfig:
    """Bind native geometry to the same canonical rigid-body pose timeline."""
    native_spec = (
        spec
        if isinstance(spec, TargetActorSpec)
        else TargetActorSpec.model_validate(spec, from_attributes=True)
    )
    if native_spec.representation == "mesh_material":
        raise ValueError("native runtime requires a native target representation")
    timeline = value(scenario_configuration, "timeline")
    steps, duration_s = int(value(timeline, "steps")), float(value(timeline, "duration_s"))
    geometry = native_spec.geometry
    if geometry is None:
        raise ValueError(f"native target {actor.name!r} requires geometry")
    resolved_mesh_path = None
    if isinstance(geometry, NativeMeshGeometrySpec):
        root = Path(value(scenario_configuration, "root", default=Path.cwd())).resolve()
        resolved_mesh_path = resolve_actor_resource(
            geometry.path,
            scenario_root=root,
            project_root=Path(value(scenario_configuration, "project_root", default=root)),
        )
        if not resolved_mesh_path.is_file():
            raise ValueError(
                f"native target {actor.name!r} geometry.path={geometry.path!r} "
                f"resolved to missing file {resolved_mesh_path}"
            )
    return NativeTargetConfig(
        name=actor.name,
        representation=native_spec.representation,
        geometry=geometry,
        constant_rcs=native_spec.constant_rcs,
        tr38901=native_spec.tr38901,
        resolved_mesh_path=resolved_mesh_path,
        mobility=PreparedMobilityAdapter(actor.mobility, steps, duration_s),
        orientation=PreparedOrientationAdapter(actor.orientation, steps, duration_s),
        _initial_position=actor.positions_m[0],
        _initial_velocity=actor.mobility.velocities_mps[0],
    )


def target_asset_alignments(
    scenario_configuration: object,
) -> dict[str, Quaternion]:
    """Resolve catalog front-axis corrections keyed by target actor name."""

    actors = value(scenario_configuration, "actors", default=None)
    if actors is None:
        return {}
    alignments: dict[str, Quaternion] = {}
    for actor in tuple(value(actors, "targets", default=()) or ()):
        if str(value(actor, "representation", default="mesh_material")) != "mesh_material":
            continue
        asset = value(actor, "asset")
        if str(value(asset, "source")) == "animated_asset":
            continue
        asset_directory = _target_asset_directory(
            asset,
            scenario_configuration=scenario_configuration,
        )
        metadata = load_target_asset_metadata(asset_directory)
        if abs(float(metadata.front_yaw_offset_deg)) > 1e-12:
            alignments[str(value(actor, "name"))] = Quaternion.from_euler_deg(
                yaw_deg=float(metadata.front_yaw_offset_deg)
            )
    return alignments


def _target_asset_directory(
    asset: object,
    *,
    scenario_configuration: object,
) -> Path:
    source = str(value(asset, "source"))
    resolved = _target_asset_path(
        asset,
        scenario_configuration=scenario_configuration,
    )
    return resolved.parent if source == "file" else resolved


def _target_asset_path(
    asset: object,
    *,
    scenario_configuration: object,
) -> Path:
    """Resolve one target asset without changing its authored metadata path."""

    scenario_root = Path(value(scenario_configuration, "root", default=Path.cwd())).resolve()
    targets_dir = Path(
        value(
            scenario_configuration,
            "targets_dir",
            default=scenario_root / "libraries" / "targets",
        )
    ).resolve()
    project_root = Path(
        value(
            scenario_configuration,
            "project_root",
            default=targets_dir.parent.parent,
        )
    ).resolve()
    source = str(value(asset, "source"))
    if source == "catalog":
        try:
            return resolve_actor_resource(
                str(value(asset, "id")),
                scenario_root=targets_dir,
                project_root=project_root,
                confine_to=targets_dir,
            )
        except ValueError as exc:
            raise ValueError("target catalog asset resolves outside the catalog root") from exc

    return resolve_actor_resource(
        str(value(asset, "path")),
        scenario_root=scenario_root,
        project_root=project_root,
    )


def _target_mesh_location(
    asset: object,
    *,
    scenario_configuration: object,
    target_name: str,
) -> tuple[str, Path, str]:
    source = str(value(asset, "source"))
    resolved_asset = _target_asset_path(
        asset,
        scenario_configuration=scenario_configuration,
    )
    if source == "catalog":
        asset_id = str(PurePosixPath(str(value(asset, "id"))))
        authored = f"libraries/targets/{asset_id}"
        try:
            pattern = _catalog_mesh_pattern(resolved_asset)
        except ValueError as exc:
            raise ValueError(
                f"target {target_name!r} field actors.targets[].asset.id={asset_id!r} "
                f"resolved to {resolved_asset}: {exc}"
            ) from exc
        return authored, resolved_asset, pattern

    configured = PurePosixPath(str(value(asset, "path")).replace("\\", "/"))
    if source == "file":
        if not resolved_asset.is_file():
            raise ValueError(
                f"target {target_name!r} field actors.targets[].asset.path="
                f"{str(configured)!r} resolved to missing file {resolved_asset}"
            )
        if resolved_asset.suffix.lower() not in _MESH_SUFFIXES:
            raise ValueError(
                f"target {target_name!r} field actors.targets[].asset.path="
                f"{str(configured)!r} resolved to unsupported mesh {resolved_asset}"
            )
        parent = str(configured.parent)
        return ("." if parent == "." else parent), resolved_asset.parent, configured.name

    pattern = str(value(asset, "pattern", default="*.ply"))
    if not resolved_asset.is_dir():
        raise ValueError(
            f"target {target_name!r} field actors.targets[].asset.path="
            f"{str(configured)!r} resolved to missing directory {resolved_asset}"
        )
    try:
        has_matching_mesh = any(
            path.is_file() and path.suffix.lower() in _MESH_SUFFIXES
            for path in resolved_asset.glob(pattern)
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"target {target_name!r} field actors.targets[].asset.pattern={pattern!r} "
            f"could not be checked in {resolved_asset}: {exc}"
        ) from exc
    if not has_matching_mesh:
        raise ValueError(
            f"target {target_name!r} field actors.targets[].asset.pattern={pattern!r} "
            f"matches no supported mesh files in resolved directory {resolved_asset} "
            f"(authored path {str(configured)!r})"
        )
    return str(configured), resolved_asset, pattern


def _catalog_mesh_pattern(directory: Path) -> str:
    manifest_path = directory / "mesh_library.json"
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"target catalog manifest is unreadable: {manifest_path}") from exc
        mesh_format = str(payload.get("mesh_format", "")).strip().lower().lstrip(".")
        if mesh_format:
            suffix = f".{mesh_format}"
            if suffix not in _MESH_SUFFIXES:
                raise ValueError(
                    f"target catalog manifest declares unsupported mesh format: {mesh_format}"
                )
            try:
                has_matching_mesh = any(
                    path.is_file() and path.suffix.lower() == suffix for path in directory.iterdir()
                )
            except OSError as exc:
                raise ValueError(f"target catalog asset is unavailable: {directory}") from exc
            if has_matching_mesh:
                return f"*{suffix}"
            raise ValueError(
                f"target catalog manifest has no matching {suffix} meshes: {directory}"
            )

    try:
        meshes = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _MESH_SUFFIXES
        )
    except OSError as exc:
        raise ValueError(f"target catalog asset is unavailable: {directory}") from exc
    if not meshes:
        raise ValueError(f"target catalog asset has no supported mesh: {directory}")
    suffixes = {path.suffix.lower() for path in meshes}
    if len(suffixes) == 1:
        return f"*{next(iter(suffixes))}"
    if len(meshes) == 1:
        return meshes[0].name
    raise ValueError(
        f"target catalog asset mixes mesh formats and needs a catalog pattern: {directory}"
    )
