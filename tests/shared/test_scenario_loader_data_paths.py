import textwrap
from pathlib import Path

import pytest

from shared.scenarios.frame_paths import (
    resolve_scenario_frames_dir,
    validate_frames_directory,
)
from shared.scenarios.parsers import parse_data_config


def test_parse_data_config_uses_explicit_directory_for_frames_dir(tmp_path: Path) -> None:
    scenario_data = {
        "data": {
            "mode": "files",
            "files": {
                "format": "hdf5",
                "directory": "frames_snapshot",
                "pattern": "selected_frames_*.h5",
            },
        }
    }

    data_cfg = parse_data_config(scenario_data, tmp_path)

    assert data_cfg["frames_directory"] == "frames_snapshot"
    assert data_cfg["frames_dir"] == (tmp_path / "frames_snapshot").resolve()
    assert data_cfg["frames_pattern"] == "selected_frames_*.h5"


def test_parse_data_config_defaults_to_frames_directory(tmp_path: Path) -> None:
    data_cfg = parse_data_config({}, tmp_path)

    assert data_cfg["mode"] == "files"
    assert data_cfg["frames_format"] == "h5"
    assert data_cfg["frames_directory"] == "frames"
    assert data_cfg["frames_dir"] == (tmp_path / "frames").resolve()
    assert data_cfg["frames_pattern"] == "mpc_frames_*.h5"
    assert data_cfg["chunk_size"] == 100
    assert data_cfg["compression"] == "lzf"


def test_parse_data_config_pattern_never_changes_writable_directory(tmp_path: Path) -> None:
    scenario_data = {
        "data": {
            "mode": "files",
            "files": {
                "format": "hdf5",
                "directory": "managed-output",
                "pattern": "legacy_frame_*.h5",
            },
        }
    }

    data_cfg = parse_data_config(scenario_data, tmp_path)

    assert data_cfg["frames_dir"] == (tmp_path / "managed-output").resolve()
    assert data_cfg["frames_pattern"] == "legacy_frame_*.h5"


@pytest.mark.parametrize(
    "pattern",
    [
        "frames/mpc_frames_*.h5",
        r"frames\mpc_frames_*.h5",
        "../mpc_frames_*.h5",
        "C:mpc_frames_*.h5",
    ],
)
def test_parse_data_config_rejects_path_bearing_read_pattern(
    tmp_path: Path,
    pattern: str,
) -> None:
    scenario_data = {"data": {"files": {"pattern": pattern}}}

    with pytest.raises(ValueError, match=r"data\.files\.pattern"):
        parse_data_config(scenario_data, tmp_path)


@pytest.mark.parametrize(
    "directory",
    [
        "../frames",
        r"..\frames",
        "frames/*",
        ".",
        "/",
        "C:\\",
        "C:frames",
        "file:frames",
    ],
)
def test_parse_data_config_rejects_ambiguous_directory_syntax(
    tmp_path: Path,
    directory: str,
) -> None:
    scenario_data = {"data": {"files": {"directory": directory}}}

    with pytest.raises(ValueError, match=r"data\.files\.directory"):
        parse_data_config(scenario_data, tmp_path)


@pytest.mark.parametrize("directory", [r"C:\frames", "D:/generated/frames"])
def test_windows_absolute_directory_declarations_remain_supported(directory: str) -> None:
    assert validate_frames_directory(directory) == directory


def test_parse_data_config_supports_explicit_absolute_directory(tmp_path: Path) -> None:
    external = (tmp_path / "external" / "frames").resolve()
    scenario_data = {"data": {"files": {"directory": str(external)}}}

    data_cfg = parse_data_config(scenario_data, tmp_path / "scenario")

    assert data_cfg["frames_directory"] == str(external)
    assert data_cfg["frames_dir"] == external


def test_parse_data_config_preserves_selectable_file_overrides(tmp_path: Path) -> None:
    scenario_data = {
        "data": {
            "mode": "files",
            "files": {
                "format": "hdf5",
                "directory": "selected",
                "pattern": "selected_*.h5",
                "chunk_size": 7,
                "compression": "gzip-4",
            },
        }
    }

    data_cfg = parse_data_config(scenario_data, tmp_path)

    assert data_cfg["mode"] == "files"
    assert data_cfg["frames_format"] == "h5"
    assert data_cfg["frames_dir"] == (tmp_path / "selected").resolve()
    assert data_cfg["frames_pattern"] == "selected_*.h5"
    assert data_cfg["chunk_size"] == 7
    assert data_cfg["compression"] == "gzip-4"


def test_parse_data_config_uses_interactive_lzf_compression_by_default(
    tmp_path: Path,
) -> None:
    data_cfg = parse_data_config({}, tmp_path)

    assert data_cfg["compression"] == "lzf"
    assert "write_canonical_view" not in data_cfg


@pytest.mark.parametrize("use_yaml_path", [False, True])
def test_resolve_scenario_frames_dir_uses_authored_directory(
    tmp_path: Path,
    use_yaml_path: bool,
) -> None:
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    scenario_yaml = scenario_dir / "scenario.yaml"
    scenario_yaml.write_text(
        textwrap.dedent("""\
            schema_version: 2
            timeline: {steps: 1, duration_s: 0.0}
            scene: {source: sionna, id: etoile}
            data:
              files:
                directory: generated/packed
            """),
        encoding="utf-8",
    )

    source = scenario_yaml if use_yaml_path else scenario_dir

    assert resolve_scenario_frames_dir(source) == (scenario_dir / "generated" / "packed").resolve()


def test_resolve_scenario_frames_dir_uses_shared_default_without_yaml(
    tmp_path: Path,
) -> None:
    scenario_dir = tmp_path / "lightweight-fixture"
    scenario_dir.mkdir()

    assert resolve_scenario_frames_dir(scenario_dir) == (scenario_dir / "frames").resolve()
