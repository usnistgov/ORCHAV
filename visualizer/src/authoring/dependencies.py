"""Semantic dependency collection for portable Scenario Builder copies."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from xml.etree import ElementTree

from shared.scenarios.paths import find_asset_project_root, resolve_actor_resource
from shared.scenarios.yaml import validate_scenario_data

from ..materials.texture_policy import TEXTURE_MAP_KEYS
from .domain import AuthoringScenario, ScenarioDependency


class ScenarioDependencyCollector:
    """Collect only inputs whose relative meaning changes with scenario root."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else find_asset_project_root(Path(__file__).resolve())
        )
        self.library_root = (self.project_root / "libraries").resolve()

    def collect(
        self,
        mapping: Mapping[str, Any],
        source_directory: Path | str,
    ) -> tuple[ScenarioDependency, ...]:
        """Return deterministic dependencies from one validated mapping."""

        validate_scenario_data(dict(mapping))
        return self._collect(mapping, source_directory)

    def refresh_scene(
        self,
        scenario: AuthoringScenario,
        scene_mapping: Mapping[str, Any] | None,
        scenario_directory: Path,
    ) -> AuthoringScenario:
        """Re-read scene sidecars on compilation while retaining actor ownership.

        Unsaved portable copies still refer to their original scene directory.
        Use that root until save promotes their dependencies to the destination.
        Required XML, mesh and texture references are collected even if missing;
        optional sidecars only belong to the closure while they exist.
        """
        root = Path(scenario_directory).resolve()
        anchor = next(
            (
                dependency
                for dependency in scenario.dependencies
                if dependency.origin_path in {"scene.id", "scene.osm"} and not dependency.external
            ),
            None,
        )
        if anchor is not None and anchor.problem:
            # These parse errors are produced only after the metadata anchor
            # passes path-safety validation. Re-read that original root so a
            # repaired file can recover without reopening the copied document.
            metadata_parse_error = anchor.problem.startswith(
                ("OSM metadata is unreadable at ", "OSM metadata must contain an object: ")
            )
            if not metadata_parse_error:
                return scenario
        if anchor is not None:
            root = anchor.source_path
            for _part in PurePosixPath(anchor.relative_path).parts:
                root = root.parent
        retained = tuple(
            dependency
            for dependency in scenario.dependencies
            if not dependency.origin_path.startswith("scene.")
        )
        current = self._collect({"scene": scene_mapping}, root)
        return replace(scenario, dependencies=retained + current)

    def _collect(
        self,
        mapping: Mapping[str, Any],
        source_directory: Path | str,
    ) -> tuple[ScenarioDependency, ...]:
        root = Path(source_directory).resolve()
        dependencies: dict[tuple[str, str], ScenarioDependency] = {}

        def add(raw: Any, kind: str, origin: str) -> ScenarioDependency | None:
            if raw is None:
                return None
            text = str(raw)
            path = Path(text)
            source = resolve_actor_resource(
                text, scenario_root=root, project_root=self.project_root
            )
            # Project references retain their authored meaning after Save As.
            # Only scenario-owned relative resources belong to the copy closure.
            if text.startswith("${PROJECT_ROOT}") or source.is_relative_to(self.library_root):
                return None
            if path.is_absolute() or PureWindowsPath(text).is_absolute():
                dependency = ScenarioDependency(
                    source,
                    source.name,
                    kind,
                    origin,
                    external=True,
                )
                dependencies[(kind, origin)] = dependency
                return dependency
            normalized = str(PurePosixPath(text.replace("\\", "/")))
            has_traversal = ".." in PurePosixPath(text.replace("\\", "/")).parts
            dependency = ScenarioDependency(
                source,
                normalized,
                kind,
                origin,
                problem=(
                    f"Dependency at {origin} contains parent traversal: {text}"
                    if has_traversal
                    else ""
                ),
            )
            if not has_traversal and not source.is_relative_to(root):
                dependency = ScenarioDependency(
                    source,
                    normalized,
                    kind,
                    origin,
                    problem=(
                        f"Dependency at {origin} resolves outside the source scenario: " f"{source}"
                    ),
                )
            dependencies[(normalized, origin)] = dependency
            return dependency

        scene = mapping.get("scene")
        if isinstance(scene, Mapping):
            source = scene.get("source")
            if source == "local":
                dependency = add(scene.get("id"), "scene_xml", "scene.id")
                if dependency is not None and not dependency.external and not dependency.problem:
                    self._collect_xml_dependencies(
                        dependency,
                        root,
                        dependencies,
                    )
                    graph = dependency.source_path.with_name("street_network.graphml")
                    if graph.is_file():
                        add(
                            str(PurePosixPath(dependency.relative_path).with_name(graph.name)),
                            "network_graph",
                            "scene.street_graph",
                        )
            elif source == "osm":
                metadata = add(
                    "osm_metadata.json",
                    "osm_metadata",
                    "scene.osm",
                )
                if metadata is not None and not metadata.problem:
                    self._collect_osm_artifacts(metadata, root, dependencies)

        marker = (mapping.get("visualizer") or {}).get("scenario_builder") or {}
        reference = marker.get("map_reference")
        if isinstance(reference, str):
            add(reference, "builder_map", "visualizer.scenario_builder.map_reference")
            add(
                str(PurePosixPath(reference).with_name("map.png")),
                "builder_map",
                "visualizer.scenario_builder.map_texture",
            )

        actors = mapping.get("actors")
        if isinstance(actors, Mapping):
            for section in ("tx", "rx", "targets"):
                entries = actors.get(section)
                if not isinstance(entries, list):
                    continue
                for index, entry in enumerate(entries):
                    if not isinstance(entry, Mapping):
                        continue
                    base = f"actors.{section}.{index}"
                    self._collect_mobility(entry.get("mobility"), base, add)
                    if section == "targets":
                        self._collect_target(entry.get("asset"), base, root, add, dependencies)
                        geometry = entry.get("geometry")
                        if isinstance(geometry, Mapping) and geometry.get("type") == "mesh":
                            add(geometry.get("path"), "target_file", f"{base}.geometry.path")

        groups = mapping.get("groups")
        if isinstance(groups, list):
            for index, entry in enumerate(groups):
                if isinstance(entry, Mapping):
                    self._collect_mobility(entry.get("mobility"), f"groups.{index}", add)

        defaults = mapping.get("view_defaults") or {}
        add(
            defaults.get("appearance_recipe"),
            "appearance_recipe",
            "view_defaults.appearance_recipe",
        )
        for index, profile in enumerate(defaults.get("visual_profiles") or ()):
            origin = f"view_defaults.visual_profiles.{index}.surface"
            dependency = add(profile.get("surface"), "target_surface", origin)
            if (
                dependency is None
                or dependency.external
                or dependency.problem
                or not dependency.source_path.is_file()
            ):
                continue
            try:
                surface = json.loads(dependency.source_path.read_text(encoding="utf-8"))
                resources = {"texture": surface.get("texture"), "uv_map": surface.get("uv_map")}
                resources.update(
                    {
                        key: value
                        for key, value in (surface.get("material") or {}).items()
                        if key in TEXTURE_MAP_KEYS
                    }
                )
                for key, value in resources.items():
                    if value:
                        add(
                            str(PurePosixPath(dependency.relative_path).parent / value),
                            "target_surface_resource",
                            f"{origin}.{key}",
                        )
            except (OSError, ValueError, TypeError):
                # Material preparation reports the authoritative sidecar error.
                pass

        return tuple(
            sorted(
                dependencies.values(),
                key=lambda dependency: (
                    dependency.external,
                    dependency.relative_path,
                    dependency.origin_path,
                ),
            )
        )

    @staticmethod
    def _collect_mobility(mobility: Any, base: str, add: Any) -> None:
        if not isinstance(mobility, Mapping):
            return
        mobility_type = mobility.get("type")
        if mobility_type == "network_route":
            add(
                mobility.get("graph_path") or "street_network.graphml",
                "network_graph",
                f"{base}.mobility.graph_path",
            )
        elif mobility_type == "mesh_sequence":
            add(
                mobility.get("positions_path"),
                "position_sequence",
                f"{base}.mobility.positions_path",
            )

    def _collect_target(
        self,
        asset: Any,
        base: str,
        root: Path,
        add: Any,
        dependencies: dict[tuple[str, str], ScenarioDependency],
    ) -> None:
        if not isinstance(asset, Mapping) or asset.get("source", "catalog") == "catalog":
            return
        raw = asset.get("path")
        origin = f"{base}.asset.path"
        dependency = add(raw, f"target_{asset.get('source')}", origin)
        if (
            dependency is not None
            and not dependency.external
            and not dependency.problem
            and asset.get("source") == "animated_asset"
            and dependency.source_path.is_file()
        ):
            # The manifest authenticates this exact closure. Copy resources next
            # to it so relative references and content hashes survive Save As.
            try:
                manifest = json.loads(dependency.source_path.read_text(encoding="utf-8"))
                for resource in manifest.get("resources", []):
                    relative = PurePosixPath(dependency.relative_path).parent / resource["path"]
                    add(str(relative), "animated_resource", origin)
                if any(
                    str(value).startswith("adapter_provenance:")
                    for value in manifest.get("provenance", {}).get("source_ids", [])
                ):
                    relative = (
                        PurePosixPath(dependency.relative_path).parent / "adapter_provenance.json"
                    )
                    add(str(relative), "animated_resource", origin)
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                # Asset preparation supplies the authoritative manifest error.
                pass
        if dependency is None or dependency.external or asset.get("source") != "directory":
            return
        directory = dependency.source_path
        if not directory.is_dir():
            return
        pattern = str(asset.get("pattern") or "*.ply")
        try:
            matches = tuple(path for path in directory.glob(pattern) if path.is_file())
        except (OSError, ValueError):
            return
        metadata = tuple(
            path
            for suffix in ("*.json", "*.yaml", "*.yml")
            for path in directory.glob(suffix)
            if path.is_file()
        )
        dependencies.pop((dependency.relative_path, dependency.origin_path), None)
        for path in (*matches, *metadata):
            relative = str(
                PurePosixPath(dependency.relative_path) / path.relative_to(directory).as_posix()
            )
            dependencies[(relative, origin)] = ScenarioDependency(
                path,
                relative,
                "target_directory_file",
                origin,
            )

    def _collect_xml_dependencies(
        self,
        dependency: ScenarioDependency,
        root: Path,
        dependencies: dict[tuple[str, str], ScenarioDependency],
    ) -> None:
        pending = [dependency]
        visited: set[Path] = set()
        while pending:
            current = pending.pop()
            source = current.source_path
            if source in visited or not source.is_file():
                continue
            visited.add(source)
            try:
                xml_root = ElementTree.parse(source).getroot()
            except (ElementTree.ParseError, OSError) as exc:
                dependencies[(current.relative_path, current.origin_path)] = ScenarioDependency(
                    current.source_path,
                    current.relative_path,
                    current.kind,
                    current.origin_path,
                    external=current.external,
                    problem=f"Scene XML dependency is unreadable at {source}: {exc}",
                )
                continue
            self._collect_scene_appearance_metadata(source, root, dependencies)
            self._collect_visual_surfaces(source, root, dependencies)
            for element in xml_root.iter():
                raw: str | None = None
                if element.tag == "include":
                    raw = element.get("filename") or element.get("value")
                elif element.tag == "string" and element.get("name") in {
                    "filename",
                    "bitmap",
                }:
                    raw = element.get("value")
                if not raw:
                    continue
                referenced = Path(raw)
                absolute = referenced.is_absolute() or PureWindowsPath(raw).is_absolute()
                resolved = resolve_actor_resource(
                    raw, scenario_root=source.parent, project_root=self.project_root
                )
                if raw.startswith("${PROJECT_ROOT}") or resolved.is_relative_to(self.library_root):
                    continue
                origin = f"{current.origin_path}:{raw}"
                if absolute:
                    relative = resolved.name
                    external = True
                    problem = ""
                else:
                    normalized = PurePosixPath(raw.replace("\\", "/"))
                    has_traversal = ".." in normalized.parts
                    try:
                        relative = (source.parent / Path(normalized)).relative_to(root).as_posix()
                    except ValueError:
                        relative = str(normalized)
                    external = False
                    if has_traversal:
                        problem = f"Scene dependency at {origin} contains parent traversal: {raw}"
                    elif not resolved.is_relative_to(root):
                        problem = (
                            f"Scene dependency at {origin} escapes the source scenario: "
                            f"{resolved}"
                        )
                    else:
                        problem = ""
                nested = ScenarioDependency(
                    resolved,
                    relative,
                    "scene_include" if resolved.suffix.lower() == ".xml" else "scene_asset",
                    origin,
                    external=external,
                    problem=problem,
                )
                dependencies[(relative, nested.origin_path)] = nested
                if resolved.suffix.lower() == ".xml" and not external and not problem:
                    pending.append(nested)

    @staticmethod
    def _collect_scene_appearance_metadata(
        scene_xml: Path,
        root: Path,
        dependencies: dict[tuple[str, str], ScenarioDependency],
    ) -> None:
        """Preserve classification precedence and the scene's geographic defaults.

        Building metadata overrides OSM face classification, but OSM metadata
        still identifies geographic material defaults in the ordinary loader.
        Keeping both existing sidecars preserves those independent decisions.
        """
        for name in ("building_visuals.json", "osm_metadata.json"):
            metadata = scene_xml.with_name(name)
            if not metadata.is_file():
                continue
            relative = metadata.relative_to(root).as_posix()
            origin = f"scene.appearance:{relative}"
            problem = (
                f"Scene appearance metadata escapes the source scenario: {metadata}"
                if not metadata.resolve().is_relative_to(root)
                else ""
            )
            dependencies[(relative, origin)] = ScenarioDependency(
                metadata, relative, "scene_appearance_metadata", origin, problem=problem
            )

    def _collect_visual_surfaces(
        self,
        scene_xml: Path,
        root: Path,
        dependencies: dict[tuple[str, str], ScenarioDependency],
    ) -> None:
        """Keep authored mappings and maps together when copying a local scene.

        Native viewing can resolve maps outside a scenario, but a portable copy
        cannot preserve those relative references. Report them through the same
        dependency problem path used for escaping XML assets, without changing
        the source files or silently dropping the authored appearance.
        """
        sidecar = scene_xml.with_name("visual_surfaces.json")
        if not sidecar.exists():
            return
        relative = sidecar.relative_to(root).as_posix()
        origin = f"scene.visual_surfaces:{relative}"

        def record(problem: str = "") -> None:
            dependencies[(relative, origin)] = ScenarioDependency(
                sidecar, relative, "visual_surfaces", origin, problem=problem
            )

        if not sidecar.resolve().is_relative_to(root):
            record(f"Visual surface sidecar escapes the source scenario: {sidecar}")
            return
        record()
        try:
            document = json.loads(sidecar.read_text(encoding="utf-8"))
            if (
                not isinstance(document, Mapping)
                or type(document.get("schema_version")) is not int
                or document["schema_version"] != 1
                or not isinstance(document.get("meshes"), Mapping)
            ):
                raise ValueError("expected schema_version 1 and a meshes object")
            for mesh_name, mesh in document["meshes"].items():
                if not isinstance(mesh, Mapping) or not isinstance(mesh.get("surfaces"), list):
                    raise ValueError(f"mesh {mesh_name!r} requires a surfaces list")
                for index, surface in enumerate(mesh["surfaces"]):
                    material = surface.get("material") if isinstance(surface, Mapping) else None
                    if not isinstance(material, Mapping):
                        raise ValueError(f"mesh {mesh_name!r} surface {index} requires a material")
                    for map_key in TEXTURE_MAP_KEYS:
                        raw = material.get(map_key)
                        if raw is None:
                            continue
                        if not isinstance(raw, str) or not raw.strip():
                            raise ValueError(
                                f"{mesh_name!r} surface {index} {map_key} is not a path"
                            )
                        normalized = PurePosixPath(raw.replace("\\", "/"))
                        path = Path(normalized)
                        absolute = path.is_absolute() or PureWindowsPath(raw).is_absolute()
                        resolved = path.resolve() if absolute else (sidecar.parent / path).resolve()
                        map_origin = f"{origin}:{mesh_name}:{index}:{map_key}"
                        map_relative = (
                            resolved.name
                            if absolute
                            else (sidecar.parent / path).relative_to(root).as_posix()
                        )
                        if absolute:
                            problem = f"Visual texture at {map_origin} must be relative: {raw}"
                        elif ".." in normalized.parts:
                            problem = (
                                f"Visual texture at {map_origin} contains parent traversal: {raw}"
                            )
                        elif not resolved.is_relative_to(root):
                            problem = f"Visual texture at {map_origin} escapes the source scenario: {resolved}"
                        else:
                            problem = ""
                        dependencies[(map_relative, map_origin)] = ScenarioDependency(
                            resolved,
                            map_relative,
                            "visual_surface_texture",
                            map_origin,
                            external=absolute,
                            problem=problem,
                        )
        except (OSError, UnicodeError, ValueError) as exc:
            record(f"Visual surface sidecar is unreadable at {sidecar}: {exc}")

    def _collect_osm_artifacts(
        self,
        metadata: ScenarioDependency | None,
        root: Path,
        dependencies: dict[tuple[str, str], ScenarioDependency],
    ) -> None:
        if metadata is None or not metadata.source_path.is_file():
            return
        try:
            payload = json.loads(metadata.source_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            dependencies[(metadata.relative_path, metadata.origin_path)] = ScenarioDependency(
                metadata.source_path,
                metadata.relative_path,
                metadata.kind,
                metadata.origin_path,
                problem=f"OSM metadata is unreadable at {metadata.source_path}: {exc}",
            )
            return
        if not isinstance(payload, Mapping):
            dependencies[(metadata.relative_path, metadata.origin_path)] = ScenarioDependency(
                metadata.source_path,
                metadata.relative_path,
                metadata.kind,
                metadata.origin_path,
                problem=f"OSM metadata must contain an object: {metadata.source_path}",
            )
            return
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return
        declared: list[str] = []
        for key in ("scene_xml", "street_network_graphml"):
            value = artifacts.get(key)
            if isinstance(value, str):
                declared.append(value)
        meshes = artifacts.get("mesh_files")
        if isinstance(meshes, list):
            declared.extend(str(value) for value in meshes if isinstance(value, str))
        for raw in declared:
            normalized = PurePosixPath(raw.replace("\\", "/"))
            candidate = Path(raw)
            absolute = candidate.is_absolute() or PureWindowsPath(raw).is_absolute()
            source = candidate.resolve() if absolute else (root / Path(normalized)).resolve()
            if source.is_relative_to(self.library_root):
                continue
            origin = f"scene.osm.artifacts.{raw}"
            if absolute:
                relative = source.name
                external = True
                problem = ""
            else:
                relative = str(normalized)
                external = False
                if ".." in normalized.parts:
                    problem = f"OSM artifact at {origin} contains parent traversal: {raw}"
                elif not source.is_relative_to(root):
                    problem = f"OSM artifact at {origin} escapes the source scenario: {source}"
                else:
                    problem = ""
            dependency = ScenarioDependency(
                source,
                relative,
                "osm_artifact",
                origin,
                external=external,
                problem=problem,
            )
            dependencies[(relative, f"scene.osm.artifacts.{relative}")] = dependency
            if raw == artifacts.get("scene_xml") and not external and not problem:
                self._collect_xml_dependencies(dependency, root, dependencies)


__all__ = ["ScenarioDependencyCollector"]
