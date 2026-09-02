"""Lexical validation for scenario-declared HDF5 read locations.

``data.files.directory`` selects an existing frame set and
``data.files.pattern`` optionally filters its filenames. Normal generation
does not use either value to choose an output; it always publishes to the
scenario's fixed ``frames`` child.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

DEFAULT_FRAMES_DIRECTORY = "frames"
DEFAULT_FRAMES_PATTERN = "mpc_frames_*.h5"


def resolve_scenario_frames_dir(scenario_path: str | Path) -> Path:
    """Resolve the configured frame directory for a scenario path.

    A scenario YAML is authoritative when one is present, including a custom
    ``data.files.directory``.  Some read-only tools also accept lightweight
    fixture directories without YAML; those retain the shared ``frames``
    default instead of each tool inventing its own fallback.
    """

    source = Path(scenario_path).expanduser()
    is_yaml_path = source.suffix.lower() in {".yaml", ".yml"}
    scenario_yaml = source if is_yaml_path or source.is_file() else source / "scenario.yaml"
    if is_yaml_path or scenario_yaml.is_file():
        # Local import avoids a module cycle: the scenario loader imports the
        # lexical validators from this module while constructing its model.
        from shared.scenarios.loader import load_scenario_configuration

        return Path(load_scenario_configuration(source).frames_dir)

    from shared.scenarios.paths import normalize_path

    return normalize_path(DEFAULT_FRAMES_DIRECTORY, base=source)


def validate_frames_directory(value: str) -> str:
    """Return a normalized directory declaration after lexical safety checks.

    Reader profiles may use absolute locations. This function rejects
    ambiguous glob and parent-traversal syntax before path resolution removes
    that information.
    """

    if not isinstance(value, str):
        raise ValueError("data.files.directory must be a string path")
    directory = value.strip()
    if not directory:
        raise ValueError("data.files.directory must be a non-empty path")

    if ":" in directory:
        windows_path = PureWindowsPath(directory)
        is_plain_windows_absolute = (
            len(directory) >= 3
            and directory[0].isalpha()
            and directory[1] == ":"
            and directory.count(":") == 1
            and windows_path.is_absolute()
        )
        if not is_plain_windows_absolute:
            raise ValueError(
                "data.files.directory must not use a drive-relative path or URI prefix"
            )

    portable_parts = directory.replace("\\", "/").split("/")
    if ".." in portable_parts:
        raise ValueError("data.files.directory must not contain '..' traversal")
    if any(character in directory for character in "*?[]"):
        raise ValueError("data.files.directory must not contain glob characters")
    if not any(part not in {"", "."} for part in portable_parts):
        raise ValueError("data.files.directory must not select the scenario root")
    for path_type in (PurePosixPath, PureWindowsPath):
        path = path_type(directory)
        if path.anchor and len(path.parts) == 1:
            raise ValueError("data.files.directory must not select a filesystem root")

    return directory


def validate_frames_pattern(value: str) -> str:
    """Return a basename-only read pattern.

    HDF5 v2 writers use manifest-owned chunk names.  Keeping this optional
    filter basename-only prevents it from being mistaken for a writable path.
    """

    if not isinstance(value, str):
        raise ValueError("data.files.pattern must be a string")
    pattern = value.strip()
    if not pattern:
        raise ValueError("data.files.pattern must be a non-empty filename pattern")
    if "/" in pattern or "\\" in pattern:
        raise ValueError(
            "data.files.pattern must be a filename pattern without directories; "
            "set data.files.directory separately"
        )
    if pattern in {".", ".."}:
        raise ValueError("data.files.pattern must name frame files")
    if ":" in pattern:
        raise ValueError("data.files.pattern must not contain a drive or URI prefix")

    return pattern
