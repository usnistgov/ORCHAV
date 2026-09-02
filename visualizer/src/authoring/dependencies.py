"""Semantic dependency collection for portable Scenario Builder copies."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from xml.etree import ElementTree

from shared.scenarios.paths import find_project_root
from shared.scenarios.yaml import validate_scenario_data

from .domain import ScenarioDependency


class ScenarioDependencyCollector:
    """Collect only inputs whose relative meaning changes with scenario root."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else find_project_root(Path(__file__).resolve())
        )
        self.library_root = (self.project_root / "libraries").resolve()

    def collect(
        self,
        mapping: Mapping[str, Any],
        source_directory: Path | str,
    ) -> tuple[ScenarioDependency, ...]:
        """Return deterministic dependencies from one validated mapping."""

        validate_scenario_data(dict(mapping))
        root = Path(source_directory).resolve()
        dependencies: dict[tuple[str, str], ScenarioDependency] = {}

        def add(raw: Any, kind: str, origin: str) -> ScenarioDependency | None:
            if raw is None:
                return None
            text = str(raw)
            path = Path(text)
            if path.is_absolute() or PureWindowsPath(text).is_absolute():
                source = path.resolve()
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
            source = (root / Path(normalized)).resolve()
            project_source = (self.project_root / Path(normalized)).resolve()
            if project_source.is_relative_to(self.library_root) and project_source.exists():
                return None
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
            elif source == "osm":
                metadata = add(
                    "osm_metadata.json",
                    "osm_metadata",
                    "scene.osm",
                )
                if metadata is not None and not metadata.problem:
                    self._collect_osm_artifacts(metadata, root, dependencies)

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

        groups = mapping.get("groups")
        if isinstance(groups, list):
            for index, entry in enumerate(groups):
                if isinstance(entry, Mapping):
                    self._collect_mobility(entry.get("mobility"), f"groups.{index}", add)

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
                resolved = (
                    referenced.resolve() if absolute else (source.parent / referenced).resolve()
                )
                if resolved.is_relative_to(self.library_root):
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
            dependencies[(relative, f"scene.osm.artifacts.{relative}")] = ScenarioDependency(
                source,
                relative,
                "osm_artifact",
                origin,
                external=external,
                problem=problem,
            )


__all__ = ["ScenarioDependencyCollector"]
