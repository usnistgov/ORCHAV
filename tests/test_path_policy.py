import logging
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_path_policy_module():
    if "shared" not in sys.modules:
        shared_pkg = types.ModuleType("shared")
        shared_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["shared"] = shared_pkg
    if "shared.logging" not in sys.modules:
        logging_mod = types.ModuleType("shared.logging")
        logging_mod.get_logger = logging.getLogger
        sys.modules["shared.logging"] = logging_mod

    root = Path(__file__).resolve().parents[1]
    module_path = root / "shared" / "scenarios" / "paths.py"
    spec = spec_from_file_location("_test_path_policy_module", module_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_find_project_root_prefers_repository_root_over_scenarios_dir(tmp_path: Path) -> None:
    find_project_root = _load_path_policy_module().find_project_root
    repo_root = tmp_path / "repo"
    scenarios_root = repo_root / "scenarios"
    nested = scenarios_root / "generator" / "tutorials" / "demo"
    nested.mkdir(parents=True)

    # Real project-root markers.
    (repo_root / ".git").mkdir()
    (repo_root / "config").mkdir()
    (repo_root / "libraries").mkdir()
    (repo_root / "visualizer").mkdir()
    (repo_root / "shared").mkdir()
    (repo_root / "README.md").write_text("root", encoding="utf-8")

    # Markers that can make ``scenarios`` look like a root if selection is naive.
    (scenarios_root / "README.md").write_text("scenarios", encoding="utf-8")
    (scenarios_root / "visualizer").mkdir()

    resolved = find_project_root(nested)
    assert resolved == repo_root.resolve()


def test_find_project_root_without_git_uses_strongest_marker_match(tmp_path: Path) -> None:
    find_project_root = _load_path_policy_module().find_project_root
    repo_root = tmp_path / "repo"
    scenarios_root = repo_root / "scenarios"
    nested = scenarios_root / "generator"
    nested.mkdir(parents=True)

    # No .git on purpose (e.g. source export), but root still has core markers.
    (repo_root / "config").mkdir()
    (repo_root / "libraries").mkdir()
    (repo_root / "visualizer").mkdir()
    (repo_root / "shared").mkdir()
    (repo_root / "README.md").write_text("root", encoding="utf-8")
    (scenarios_root / "README.md").write_text("scenarios", encoding="utf-8")
    (scenarios_root / "visualizer").mkdir()

    resolved = find_project_root(nested)
    assert resolved == repo_root.resolve()


def test_find_project_root_prefers_nested_project_over_unrelated_outer_git(
    tmp_path: Path,
) -> None:
    find_project_root = _load_path_policy_module().find_project_root
    outer_repo = tmp_path / "unrelated-repo"
    project_root = outer_repo / "exports" / "orchav"
    scenario_root = project_root / "scenarios" / "custom" / "demo"
    scenario_root.mkdir(parents=True)

    # The outer repository has enough generic markers to be a candidate, but
    # it does not own the standalone ORCHAV project nested within it.
    (outer_repo / ".git").mkdir()
    (outer_repo / "README.md").write_text("unrelated", encoding="utf-8")
    (outer_repo / "config").mkdir()

    # A source export has no VCS metadata but carries the stronger project
    # structure that must own project-relative scenario resources.
    (project_root / "README.md").write_text("ORCHAV", encoding="utf-8")
    for marker in ("config", "libraries", "visualizer", "shared"):
        (project_root / marker).mkdir()

    assert find_project_root(scenario_root) == project_root.resolve()


def test_find_project_root_prefers_nearest_structural_project_boundary(
    tmp_path: Path,
) -> None:
    find_project_root = _load_path_policy_module().find_project_root
    outer_root = tmp_path / "workspace"
    project_root = outer_root / "project"
    scenario_root = project_root / "scenarios" / "example"
    scenario_root.mkdir(parents=True)
    (project_root / "libraries").mkdir()

    # Generic names in a higher-level workspace do not own the nested
    # project's scenario and library trees.
    (outer_root / ".git").mkdir()
    (outer_root / "README.md").write_text("workspace", encoding="utf-8")
    for marker in ("config", "visualizer", "shared"):
        (outer_root / marker).mkdir()

    assert find_project_root(scenario_root) == project_root.resolve()


def test_actor_library_path_uses_nested_standalone_project_root(tmp_path: Path) -> None:
    resolve_actor_resource = _load_path_policy_module().resolve_actor_resource
    outer_repo = tmp_path / "unrelated-repo"
    project_root = outer_repo / "exports" / "orchav"
    scenario_root = project_root / "scenarios" / "custom" / "demo"
    asset_path = project_root / "libraries" / "targets" / "person" / "person.ply"
    scenario_root.mkdir(parents=True)
    asset_path.parent.mkdir(parents=True)
    asset_path.write_text("ply", encoding="utf-8")

    (outer_repo / ".git").mkdir()
    (outer_repo / "README.md").write_text("unrelated", encoding="utf-8")
    (outer_repo / "config").mkdir()
    (project_root / "README.md").write_text("ORCHAV", encoding="utf-8")
    for marker in ("config", "visualizer", "shared"):
        (project_root / marker).mkdir()

    resolved = resolve_actor_resource(
        "libraries/targets/person/person.ply",
        scenario_root=scenario_root,
    )

    assert resolved == asset_path.resolve()
    assert resolved.is_file()
