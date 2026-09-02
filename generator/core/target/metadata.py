"""Optional metadata loaded from target mesh library directories.

Target asset directories may include ``target_metadata.json`` to describe local
asset conventions that are not obvious from geometry alone. Today this is used
for front-axis alignment: a mesh library can declare the yaw offset needed to
make scenario-facing orientation values point the intended side of the asset
forward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from shared.logging import get_logger

logger = get_logger(__name__)

TARGET_METADATA_FILENAME = "target_metadata.json"
TARGET_METADATA_CACHE_SIZE = 512


@dataclass(frozen=True)
class TargetAssetMetadata:
    """Metadata that describes target-library conventions.

    The metadata file is optional. Missing files or fields intentionally fall
    back to zeros so custom target libraries do not need any companion data.
    """

    front_yaw_offset_deg: float = 0.0
    front_axis: str = "+X"


def load_target_asset_metadata(
    mesh_directory: str | Path,
    *,
    project_root: str | Path | None = None,
) -> TargetAssetMetadata:
    """Load optional target metadata for a mesh directory.

    The companion file is optional. If it is not present, or if the field is
    omitted, callers receive default values. Invalid metadata is ignored with a
    warning because the geometry itself can still be usable.
    """

    metadata_path = _find_metadata_path(
        str(mesh_directory), str(project_root) if project_root else ""
    )
    if metadata_path is None:
        return TargetAssetMetadata()

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable target metadata %s: %s", metadata_path, exc)
        return TargetAssetMetadata()

    return _parse_target_asset_metadata(payload, metadata_path)


@lru_cache(maxsize=TARGET_METADATA_CACHE_SIZE)
def _find_metadata_path(mesh_directory: str, project_root: str = "") -> Path | None:
    for directory in _candidate_mesh_directories(mesh_directory, project_root):
        metadata_path = directory / TARGET_METADATA_FILENAME
        if metadata_path.is_file():
            return metadata_path
    return None


def _candidate_mesh_directories(mesh_directory: str, project_root: str = "") -> Iterable[Path]:
    """Yield plausible metadata lookup directories for relative mesh paths."""
    directory = Path(mesh_directory)
    candidates: list[Path] = []

    if directory.is_absolute():
        candidates.append(directory)
    else:
        if project_root:
            candidates.append(Path(project_root) / directory)
        candidates.append(Path.cwd() / directory)
        try:
            from shared.scenarios.paths import find_project_root

            candidates.append(find_project_root(Path.cwd()) / directory)
        except (ImportError, OSError, RuntimeError, ValueError):
            pass

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        yield candidate


def _parse_target_asset_metadata(
    payload: dict[str, Any],
    metadata_path: Path,
) -> TargetAssetMetadata:
    """Parse the supported target metadata fields from JSON payloads."""
    front = payload.get("front", {})
    if front is None:
        front = {}
    if not isinstance(front, dict):
        logger.warning("Ignoring invalid front metadata in %s", metadata_path)
        front = {}

    yaw_offset = front.get("yaw_offset_deg", payload.get("front_yaw_offset_deg", 0.0))
    front_axis = str(front.get("axis", payload.get("front_axis", "+X")))

    try:
        yaw_offset = float(yaw_offset)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid front yaw offset in %s: %r",
            metadata_path,
            yaw_offset,
        )
        yaw_offset = 0.0

    return TargetAssetMetadata(
        front_yaw_offset_deg=yaw_offset,
        front_axis=front_axis,
    )
