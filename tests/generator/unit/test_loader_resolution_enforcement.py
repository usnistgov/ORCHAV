import textwrap
from pathlib import Path

import pytest

from shared.scenarios import load_scenario_configuration


def write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "scenario.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def test_coverage_resolution_2d_ok(tmp_path: Path):
    yaml = write_yaml(
        tmp_path,
        """
        schema_version: 2
        timeline:
          steps: 1
          duration_s: 0.0
        scene:
          id: box/box.xml
          source: library
        data:
          files:
            format: hdf5
            directory: frames
        coverage:
          enabled: true
          grid:
            resolution_m: [1.0, 2.0]
            heights_m: [1.0]
        """,
    )
    scx = load_scenario_configuration(yaml, project_root=Path.cwd())
    assert tuple(scx.coverage_cfg["resolution_m"]) == (1.0, 2.0)


def test_coverage_resolution_3d_rejected(tmp_path: Path):
    yaml = write_yaml(
        tmp_path,
        """
        schema_version: 2
        timeline:
          steps: 1
          duration_s: 0.0
        scene:
          id: box/box.xml
          source: library
        data:
          files:
            format: hdf5
            directory: frames
        coverage:
          enabled: true
          grid:
            resolution_m: [1.0, 2.0, 3.0]
        """,
    )
    with pytest.raises(ValueError):
        load_scenario_configuration(yaml, project_root=Path.cwd())
