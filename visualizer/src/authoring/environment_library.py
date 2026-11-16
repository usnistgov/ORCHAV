"""User-owned environment snapshots, independent of actors and sensing settings."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .compiler import canonical_scenario_mapping
from .dependencies import ScenarioDependencyCollector
from .domain import ScenarioDependency


def library_root() -> Path:
    """Keep saved environments outside the disposable download cache."""
    override = os.environ.get("ORCHAV_ENVIRONMENT_LIBRARY")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return base / "orchav" / "environments"


@dataclass(frozen=True)
class EnvironmentEntry:
    root: Path
    metadata: dict

    @property
    def name(self):
        return self.metadata["name"]

    def mapping(self):
        return {
            "schema_version": 2,
            "timeline": {"steps": 1, "duration_s": 1.0},
            "scene": self.metadata["scene"],
            "visualizer": {"scenario_builder": self.metadata.get("builder", {})},
        }


class EnvironmentLibrary:
    def __init__(self, root: Path | None = None, *, project_root: Path):
        self.root = Path(root) if root is not None else library_root()
        self.project_root = project_root

    def entries(self) -> tuple[EnvironmentEntry, ...]:
        entries = []
        for path in self.root.glob("*/environment.json"):
            if path.parent.name.startswith("."):
                continue
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if (
                    isinstance(metadata, dict)
                    and metadata.get("version") == 1
                    and isinstance(metadata.get("name"), str)
                    and isinstance(metadata.get("saved_utc"), str)
                    and isinstance(metadata.get("scene"), dict)
                ):
                    entries.append(EnvironmentEntry(path.parent, metadata))
            except (OSError, ValueError, TypeError):
                continue
        return tuple(sorted(entries, key=lambda e: e.metadata["saved_utc"], reverse=True))

    def save(
        self, scenario, source_root: Path, name: str, *, resolved_scene: Path | None = None
    ) -> EnvironmentEntry:
        """Copy the environment's inputs atomically; never copy actors or frames."""
        name = name.strip()
        if not name or len(name) > 120:
            raise ValueError("Use an environment name between 1 and 120 characters.")
        mapping = canonical_scenario_mapping(scenario)
        if not mapping.get("scene"):
            raise ValueError("Choose an environment first.")
        builder = {}
        reference = mapping.get("visualizer", {}).get("scenario_builder", {}).get("map_reference")
        if reference:
            builder["map_reference"] = reference
        source_root = Path(source_root)
        if resolved_scene is not None:
            resolved_scene = Path(resolved_scene)
            identifier_path = Path(mapping["scene"]["id"])
            if identifier_path.is_absolute():
                mapping["scene"]["id"] = identifier_path.name
                source_root = resolved_scene.parent
            elif mapping["scene"]["source"] in {"local", "osm"}:
                source_root = resolved_scene.parents[len(identifier_path.parts) - 1]
        envelope = {
            "schema_version": 2,
            "timeline": {"steps": 1, "duration_s": 1.0},
            "scene": mapping["scene"],
            "visualizer": {"scenario_builder": builder},
        }
        assets = ScenarioDependencyCollector(self.project_root).collect(envelope, source_root)
        for asset in assets:
            if asset.problem or asset.external or not asset.available:
                raise ValueError(
                    asset.problem
                    or f"Environment input must be local and available: {asset.relative_path}"
                )
        self.root.mkdir(parents=True, exist_ok=True)
        identifier = uuid4().hex
        destination = self.root / identifier
        with tempfile.TemporaryDirectory(prefix=".saving-", dir=self.root) as scratch:
            staged = Path(scratch) / "entry"
            staged.mkdir()
            for asset in assets:
                target = (staged / asset.relative_path).resolve()
                if not target.is_relative_to(staged.resolve()):
                    raise ValueError("Environment input escapes the saved environment.")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(asset.source_path, target)
            metadata = {
                "version": 1,
                "name": name,
                "saved_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "scene": mapping["scene"],
                "builder": builder,
                "files": [d.relative_path for d in assets],
            }
            if resolved_scene is not None:
                osm_path = resolved_scene.parent / "osm_metadata.json"
                if osm_path.is_file():
                    osm = json.loads(osm_path.read_text(encoding="utf-8"))
                    metadata["bbox"] = osm.get("bbox")
                    metadata["center"] = osm.get("center_latlon")
            (staged / "environment.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            entry = EnvironmentEntry(staged, metadata)
            self.dependencies(entry)
            os.replace(staged, destination)
        return EnvironmentEntry(destination, metadata)

    def dependencies(self, entry: EnvironmentEntry) -> tuple[ScenarioDependency, ...]:
        """Resolve a selected snapshot, including graphs independent of actors."""
        assets = list(
            ScenarioDependencyCollector(self.project_root).collect(entry.mapping(), entry.root)
        )
        known = {a.relative_path for a in assets}
        for relative in entry.metadata.get("files", []):
            path = (entry.root / relative).resolve()
            if not path.is_relative_to(entry.root.resolve()):
                raise ValueError("Environment input escapes its directory.")
            if relative not in known:
                assets.append(
                    ScenarioDependency(path, relative, "environment_input", "scene.environment")
                )
        for asset in assets:
            if asset.problem or asset.external or not asset.available:
                raise ValueError(
                    asset.problem or f"Missing environment input: {asset.relative_path}"
                )
        return tuple(assets)
