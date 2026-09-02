"""Shared path resolution policy for scenario-driven tools.

Scenario YAML may be loaded from the repository, a source export, tests, or a
user-selected folder. ``PathPolicy`` defines how relative paths, project-root
markers, and ``${PROJECT_ROOT}`` references are interpreted so generator,
visualizer, and CLI code do not depend on the current working directory.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from shared.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PathPolicy:
    """
    Immutable path policy that defines how paths should be resolved.

    Attributes:
        config_path: Path to the configuration file (scenario.yaml, app.toml, etc.)
        project_root_override: Optional override for project root detection
    """

    config_path: Path
    project_root_override: Optional[Path] = None

    @property
    def config_dir(self) -> Path:
        """Directory containing the configuration file."""
        return self.config_path.parent.resolve()

    @property
    def project_root(self) -> Path:
        """Detected or overridden project root directory."""
        if self.project_root_override:
            return self.project_root_override.resolve()
        return find_project_root(self.config_dir)

    def resolve_path(self, path_like: Union[str, Path], *, base: Optional[Path] = None) -> Path:
        """
        Resolve a path using this policy's rules.

        Args:
            path_like: Path string or Path object to resolve
            base: Base directory for relative paths (defaults to config_dir)

        Returns:
            Absolute, resolved Path object
        """
        if base is None:
            base = self.config_dir

        # Expand variables first
        if isinstance(path_like, str):
            path_str = expand_vars(path_like, self.project_root)
        else:
            path_str = str(path_like)

        # Convert to Path and resolve
        path = Path(path_str)
        if path.is_absolute():
            return path.resolve()
        else:
            return (base / path).resolve()


def find_project_root(start: Path) -> Path:
    """Select the project root from ancestors of ``start``.

    The nearest ancestor containing both ``libraries`` and ``scenarios`` is
    the structural project boundary. This keeps project-relative resources
    attached to the scenario tree even when a higher ancestor contains more
    generic marker names. If no structural boundary exists, the candidate
    containing the most ORCHAV project markers wins and a VCS marker breaks
    ties. When no candidate qualifies, the resolved starting directory is
    returned with a warning.

    Args:
        start: Starting directory to search from

    Returns:
        Path to project root directory

    Raises:
        RuntimeError: If project root cannot be determined
    """
    start = start.resolve()

    project_markers = ["README.md", "config", "libraries", "scenarios", "visualizer", "shared"]
    ancestors = [start] + list(start.parents)

    for parent in ancestors:
        if (parent / "libraries").is_dir() and (parent / "scenarios").is_dir():
            logger.debug("Found project root via structural boundary: %s", parent)
            return parent.resolve()

    candidates: list[tuple[Path, int, bool]] = []
    for parent in ancestors:
        marker_count = sum(1 for marker in project_markers if (parent / marker).exists())
        has_git = (parent / ".git").exists()
        if marker_count >= 2:
            candidates.append((parent, marker_count, has_git))

    if candidates:
        best_parent, best_score, best_has_git = max(
            candidates,
            key=lambda item: (item[1], item[2]),
        )
        logger.debug(
            "Found project root via strongest marker match: %s (markers=%d, vcs=%s)",
            best_parent,
            best_score,
            best_has_git,
        )
        return best_parent.resolve()

    logger.warning("Could not determine project root from %s, using start directory", start)
    return start.resolve()


def find_project_root_from_script(script_path: Path) -> Path:
    """
    Find project root starting from a script's location.

    Args:
        script_path: Path to the script file

    Returns:
        Path to project root directory
    """
    return find_project_root(script_path.parent)


def create_path_policy(
    config_path: Path, project_root_override: Optional[Path] = None
) -> PathPolicy:
    """
    Create a path policy for configuration-based tools (visualizer, scenario loader).

    Args:
        config_path: Path to configuration file
        project_root_override: Optional override for project root

    Returns:
        PathPolicy instance
    """
    return PathPolicy(config_path, project_root_override)


def create_script_path_policy(
    script_path: Path, project_root_override: Optional[Path] = None
) -> PathPolicy:
    """
    Create a path policy for script-based tools (generator, tests).

    Args:
        script_path: Path to the script file
        project_root_override: Optional override for project root

    Returns:
        PathPolicy instance
    """
    # The synthetic filename supplies config_dir semantics without requiring a
    # physical configuration file.
    synthetic_config = script_path.parent / "script_config.yaml"
    return PathPolicy(synthetic_config, project_root_override)


def expand_vars(s: str, project_root: Path) -> str:
    """
    Expand variables in a string, particularly ${PROJECT_ROOT}.

    Args:
        s: String containing variables to expand
        project_root: Project root path to substitute

    Returns:
        String with variables expanded
    """
    return s.replace("${PROJECT_ROOT}", str(project_root))


def normalize_path(
    path_like: Union[str, Path],
    *,
    base: Path,
    project_root: Optional[Path] = None,
) -> Path:
    """
    Normalize a path relative to a base directory.

    Args:
        path_like: Path string or Path object
        base: Base directory for relative paths.
        project_root: Root substituted for ``${PROJECT_ROOT}``. The marker is
            left unchanged when no project root is supplied.

    Returns:
        Absolute, resolved Path object
    """
    raw = str(path_like)
    if project_root is not None:
        raw = expand_vars(raw, Path(project_root).resolve())
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    else:
        return (base / path).resolve()


def resolve_actor_resource(
    path_like: Union[str, Path],
    *,
    scenario_root: Path,
    project_root: Optional[Path] = None,
    confine_to: Optional[Path] = None,
) -> Path:
    """Resolve a scenario actor's external resource to an absolute path.

    Ordinary relative paths belong to the scenario. Paths beginning with
    ``libraries`` belong to the ORCHAV project, and absolute paths retain their
    authored meaning. ``confine_to`` applies the same resolution while
    rejecting catalog references that escape their declared root.
    """

    scenario_root = Path(scenario_root).resolve()
    resolved_project_root = (
        find_project_root(scenario_root) if project_root is None else Path(project_root).resolve()
    )
    raw = expand_vars(str(path_like), resolved_project_root)
    candidate = Path(raw)

    if candidate.is_absolute():
        resolved = candidate.resolve()
    elif confine_to is not None:
        resolved = (Path(confine_to).resolve() / candidate).resolve()
    elif candidate.parts and candidate.parts[0].casefold() == "libraries":
        resolved = (resolved_project_root / candidate).resolve()
    else:
        resolved = (scenario_root / candidate).resolve()

    if confine_to is not None:
        confinement_root = Path(confine_to).resolve()
        try:
            resolved.relative_to(confinement_root)
        except ValueError as exc:
            raise ValueError(
                f"actor resource resolves outside the allowed root {confinement_root}: "
                f"{path_like!s}"
            ) from exc
    return resolved
