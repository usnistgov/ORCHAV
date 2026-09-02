"""Regression tests for the yaml_path parameter in load_scenario_configuration.

Ensures:
1. yaml_path=None falls back to scenario_path/scenario.yaml (existing behavior).
2. yaml_path= explicitly loads a different YAML file.
3. The scenario root is still derived from scenario_path, not yaml_path.
"""

import textwrap
from pathlib import Path

import pytest

from shared.scenarios import load_scenario_configuration


@pytest.fixture()
def scenario_dir(tmp_path: Path) -> Path:
    """Create a temp scenario directory with two YAML files."""
    scenario = tmp_path / "test_scenario"
    scenario.mkdir()

    # Default scenario.yaml
    (scenario / "scenario.yaml").write_text(textwrap.dedent("""\
        schema_version: 2
        timeline:
          steps: 10
          duration_s: 1.0
        scene:
          source: sionna
          id: etoile
        data:
          mode: files
          files:
            format: hdf5
            directory: frames_default
            pattern: frame_*.h5
        raytracing:
          enabled: true
        """))

    # Alternate YAML for coherent variant
    (scenario / "scenario_coherent.yaml").write_text(textwrap.dedent("""\
        schema_version: 2
        timeline:
          steps: 20
          duration_s: 2.0
        scene:
          source: sionna
          id: etoile
        data:
          mode: files
          files:
            format: hdf5
            directory: frames_coherent
            pattern: frame_*.h5
        raytracing:
          enabled: true
        """))

    return scenario


class TestYamlPathParameter:
    def test_default_loads_scenario_yaml(self, scenario_dir: Path):
        cfg = load_scenario_configuration(scenario_dir)
        assert "frames_default" in str(cfg.frames_dir)

    def test_explicit_yaml_path_loads_alternate(self, scenario_dir: Path):
        alt_yaml = scenario_dir / "scenario_coherent.yaml"
        cfg = load_scenario_configuration(scenario_dir, yaml_path=alt_yaml)
        assert "frames_coherent" in str(cfg.frames_dir)

    def test_root_derived_from_scenario_path(self, scenario_dir: Path):
        alt_yaml = scenario_dir / "scenario_coherent.yaml"
        cfg = load_scenario_configuration(scenario_dir, yaml_path=alt_yaml)
        assert cfg.root == scenario_dir.resolve()
        assert cfg.project_root.is_absolute()

    def test_project_root_override_is_exposed(self, scenario_dir: Path):
        project_root = scenario_dir.parent

        cfg = load_scenario_configuration(scenario_dir, project_root=project_root)

        assert cfg.project_root == project_root.resolve()

    def test_yaml_path_none_is_same_as_default(self, scenario_dir: Path):
        cfg_default = load_scenario_configuration(scenario_dir)
        cfg_explicit = load_scenario_configuration(scenario_dir, yaml_path=None)
        assert cfg_default.frames_directory == cfg_explicit.frames_directory
        assert cfg_default.frames_pattern == cfg_explicit.frames_pattern

    def test_missing_yaml_path_raises(self, scenario_dir: Path):
        bad_path = scenario_dir / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            load_scenario_configuration(scenario_dir, yaml_path=bad_path)


def test_loader_materializes_file_defaults_when_data_is_omitted(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "default-data"
    scenario_dir.mkdir()
    (scenario_dir / "scenario.yaml").write_text(
        textwrap.dedent("""\
            schema_version: 2
            timeline: {steps: 1, duration_s: 0.0}
            scene: {source: sionna, id: etoile}
            """),
        encoding="utf-8",
    )

    config = load_scenario_configuration(scenario_dir, project_root=tmp_path)

    assert config.data_mode == "files"
    assert config.frames_format == "h5"
    assert config.frames_directory == "frames"
    assert config.frames_dir == (scenario_dir / "frames").resolve()
    assert config.frames_pattern == "mpc_frames_*.h5"
    assert config.chunk_size == 100
    assert config.compression == "lzf"


def test_loader_materializes_file_defaults_around_authored_overrides(
    tmp_path: Path,
) -> None:
    scenario_dir = tmp_path / "selected-data"
    scenario_dir.mkdir()
    (scenario_dir / "scenario.yaml").write_text(
        textwrap.dedent("""\
            schema_version: 2
            timeline: {steps: 1, duration_s: 0.0}
            scene: {source: sionna, id: etoile}
            data:
              files:
                directory: selected-frames
                chunk_size: 7
            """),
        encoding="utf-8",
    )

    config = load_scenario_configuration(scenario_dir, project_root=tmp_path)

    assert config.data_mode == "files"
    assert config.frames_format == "h5"
    assert config.frames_directory == "selected-frames"
    assert config.frames_pattern == "mpc_frames_*.h5"
    assert config.chunk_size == 7
    assert config.compression == "lzf"
