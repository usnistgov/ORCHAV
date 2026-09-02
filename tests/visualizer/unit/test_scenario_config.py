from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from visualizer.src.io.scenario_config import (
    AppConfig,
    Scenario,
    load_app_config,
    load_scenario,
    resolve_scene_meshes,
)


class TestScenarioConfig:
    """Unit tests for scenario and application config parsing."""

    @pytest.fixture
    def mock_app_config_toml(self):
        return """
        [paths]
        scenes = "custom/scenes"
        targets = "custom/targets"
        scenarios = "custom/scenarios"
        output = "custom/output"

        [live_grpc]
        sionna = "grpc://localhost:50051"
        """

    @patch("visualizer.src.io.scenario_config.find_project_root")
    @patch("builtins.open", new_callable=mock_open)
    @patch("visualizer.src.io.scenario_config.tomllib.load")
    def test_load_app_config(self, mock_toml_load, mock_file, mock_find_root, mock_app_config_toml):
        """Test loading application configuration."""
        mock_find_root.return_value = Path("/mock/root")
        mock_toml_load.return_value = {
            "paths": {
                "scenes": "custom/scenes",
                "targets": "custom/targets",
                "scenarios": "custom/scenarios",
                "output": "custom/output",
            },
            "live_grpc": {"sionna": "grpc://localhost:50051"},
        }

        with patch("pathlib.Path.exists", return_value=True):
            config = load_app_config(Path("/mock/config.toml"))

        assert isinstance(config, AppConfig)
        assert config.scenes == Path("/mock/root/custom/scenes").resolve()
        assert config.targets == Path("/mock/root/custom/targets").resolve()
        assert config.live_grpc["sionna"] == "grpc://localhost:50051"

    @patch("shared.scenarios.paths.find_project_root")
    def test_load_app_config_defaults(self, mock_find_root):
        """Test loading default application configuration."""
        mock_find_root.return_value = Path("/mock/root")

        with patch("pathlib.Path.exists", return_value=False):
            config = load_app_config(None)

        assert isinstance(config, AppConfig)
        assert config.scenes == Path("/mock/root/libraries/scenes").resolve()

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_load_scenario(self, mock_yaml_load, mock_file):
        """Test loading scenario configuration."""
        mock_yaml_load.return_value = {
            "schema_version": 2,
            "timeline": {"steps": 1, "duration_s": 0.0},
            "scene": {"id": "test_scene", "source": "library"},
            "data": {"mode": "files"},
            "view_defaults": {},
        }

        app_config = AppConfig.get_defaults()
        with patch("pathlib.Path.exists", return_value=True):
            scenario = load_scenario(Path("/mock/scenario.yaml"), app_config)

        assert isinstance(scenario, Scenario)
        assert scenario.scene_spec["id"] == "test_scene"
        assert scenario.scene_id == "test_scene"
        assert scenario.scene_source == "library"
        assert scenario.data_mode == "files"
        assert scenario.data_spec["files"] == {
            "format": "h5",
            "directory": "frames",
            "pattern": "mpc_frames_*.h5",
            "chunk_size": 100,
            "compression": "lzf",
        }
        assert scenario.debug_level == "WARNING"
        assert scenario.visualizer_cfg == {}
        mock_file.assert_called_once_with(Path("/mock/scenario.yaml"), encoding="utf-8")

    def test_load_scenario_expands_project_root_read_paths(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        project_root = tmp_path / "project"
        scenario_root = project_root / "scenarios" / "reader"
        scene_xml = project_root / "assets" / "scène-東京.xml"
        frames_dir = project_root / "recordings" / "frames"
        scenario_root.mkdir(parents=True)
        scene_xml.parent.mkdir()
        scene_xml.write_text("<scene />", encoding="utf-8")
        frames_dir.mkdir(parents=True)
        (scenario_root / "scenario.yaml").write_text(
            """schema_version: 2
timeline: {steps: 1, duration_s: 0.0}
scene:
  source: "  local  "
  id: ${PROJECT_ROOT}/assets/scène-東京.xml
data:
  files:
    directory: ${PROJECT_ROOT}/recordings/frames
""",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "visualizer.src.io.scenario_config.find_scenario_project_root",
            lambda _path: project_root,
        )
        app_config = AppConfig(
            scenes=project_root / "libraries" / "scenes",
            targets=project_root / "libraries" / "targets",
            scenarios=project_root / "scenarios",
            output=project_root / "output",
            live_grpc={},
        )

        scenario = load_scenario(scenario_root, app_config)

        assert scenario.scene_spec["id"] == str(scene_xml.resolve())
        assert scenario.scene_spec["source"] == "local"
        assert scenario.scene_source == "local"
        assert scenario.frames_dir == frames_dir.resolve()

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_load_scenario_materializes_omitted_data_defaults(
        self,
        mock_yaml_load,
        mock_file,
    ):
        mock_yaml_load.return_value = {
            "schema_version": 2,
            "timeline": {"steps": 1, "duration_s": 0.0},
            "scene": {"id": "test_scene", "source": "library"},
        }

        app_config = AppConfig.get_defaults()
        with patch("pathlib.Path.exists", return_value=True):
            scenario = load_scenario(Path("/mock/scenario.yaml"), app_config)

        assert scenario.data_mode == "files"
        assert scenario.data_spec == {
            "mode": "files",
            "files": {
                "format": "h5",
                "directory": "frames",
                "pattern": "mpc_frames_*.h5",
                "chunk_size": 100,
                "compression": "lzf",
            },
        }
        assert scenario.frames_dir == (Path("/mock") / "frames").resolve()

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_load_scenario_preserves_file_overrides_and_fills_other_defaults(
        self,
        mock_yaml_load,
        mock_file,
    ):
        mock_yaml_load.return_value = {
            "schema_version": 2,
            "timeline": {"steps": 1, "duration_s": 0.0},
            "scene": {"id": "test_scene", "source": "library"},
            "data": {
                "files": {
                    "directory": "selected-frames",
                    "compression": "gzip-4",
                }
            },
        }

        app_config = AppConfig.get_defaults()
        with patch("pathlib.Path.exists", return_value=True):
            scenario = load_scenario(Path("/mock/scenario.yaml"), app_config)

        assert scenario.data_mode == "files"
        assert scenario.data_spec["files"] == {
            "format": "h5",
            "directory": "selected-frames",
            "pattern": "mpc_frames_*.h5",
            "chunk_size": 100,
            "compression": "gzip-4",
        }
        assert scenario.frames_dir == (Path("/mock") / "selected-frames").resolve()

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_load_scenario_uses_normalized_validated_file_settings(
        self,
        mock_yaml_load,
        mock_file,
    ):
        mock_yaml_load.return_value = {
            "schema_version": 2,
            "timeline": {"steps": 1, "duration_s": 0.0},
            "scene": {"id": "test_scene", "source": "library"},
            "data": {
                "files": {
                    "format": "hdf5",
                    "directory": " selected-frames ",
                    "pattern": " mpc_frames_*.h5 ",
                }
            },
        }

        app_config = AppConfig.get_defaults()
        with patch("pathlib.Path.exists", return_value=True):
            scenario = load_scenario(Path("/mock/scenario.yaml"), app_config)

        assert scenario.data_spec["files"]["format"] == "h5"
        assert scenario.data_spec["files"]["directory"] == "selected-frames"
        assert scenario.data_spec["files"]["pattern"] == "mpc_frames_*.h5"
        assert scenario.frames_dir == (Path("/mock") / "selected-frames").resolve()

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_load_scenario_preserves_actor_contract(self, mock_yaml_load, mock_file):
        """Visualizer loading retains typed actors, groups, and timeline data."""
        mock_yaml_load.return_value = {
            "schema_version": 2,
            "timeline": {"steps": 4, "duration_s": 3.0},
            "scene": {"id": "test_scene", "source": "library"},
            "actors": {
                "tx": [
                    {
                        "name": "MainTransmitter",
                        "mobility": {"type": "stationary", "position_m": [0, 0, 10]},
                    }
                ],
                "rx": [
                    {
                        "name": "WalkingReceiver",
                        "mobility": {"type": "group_member", "group": "Receivers"},
                    },
                    {
                        "name": "StaticReceiver",
                        "mobility": {"type": "group_member", "group": "Receivers"},
                    },
                ],
                "targets": [
                    {
                        "name": "Car",
                        "mobility": {"type": "stationary", "position_m": [2, 1, 0]},
                        "asset": {"source": "catalog", "id": "simple_car"},
                    }
                ],
            },
            "groups": [
                {
                    "name": "Receivers",
                    "mobility": {"type": "stationary", "position_m": [5, 0, 1.5]},
                }
            ],
            "data": {"mode": "files"},
            "view_defaults": {},
        }

        app_config = AppConfig.get_defaults()
        with patch("pathlib.Path.exists", return_value=True):
            scenario = load_scenario(Path("/mock/scenario.yaml"), app_config)

        assert scenario.actors.tx[0].name == "MainTransmitter"
        assert scenario.actors.rx[0].name == "WalkingReceiver"
        assert scenario.actors.targets[0].name == "Car"
        assert scenario.groups[0].name == "Receivers"
        assert scenario.timeline.steps == 4
        assert scenario.timeline.duration_s == 3.0

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_load_scenario_reads_visualizer_configuration(self, mock_yaml_load, mock_file):
        """The canonical `visualizer` mapping populates visualizer settings."""
        mock_yaml_load.return_value = {
            "schema_version": 2,
            "timeline": {"steps": 1, "duration_s": 0.0},
            "scene": {"id": "test_scene"},
            "visualizer": {"panels": {"statistics": {"enabled": False}}},
            "data": {"mode": "files"},
        }
        app_config = AppConfig.get_defaults()

        with patch("pathlib.Path.exists", return_value=True):
            scenario = load_scenario(Path("/mock/scenario.yaml"), app_config)

        assert scenario.visualizer_cfg["panels"]["statistics"]["enabled"] is False

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_load_scenario_live_grpc_endpoints_merge(self, mock_yaml_load, mock_file):
        """Scenario live gRPC endpoint overrides merge on top of app defaults."""
        mock_yaml_load.return_value = {
            "schema_version": 2,
            "timeline": {"steps": 1, "duration_s": 0.0},
            "scene": {"id": "test_scene"},
            "live_grpc_endpoints": {"aux": "grpc://aux-override:50061"},
            "data": {
                "mode": "live_grpc",
                "live_grpc_endpoints": {"http": "http://override:8080"},
                "live_grpc": {"endpoint": "grpc://generator:50051"},
            },
        }
        app_config = AppConfig.get_defaults()
        app_config.live_grpc = {
            "sionna": "grpc://default:50051",
            "aux": "grpc://default:50061",
        }

        with patch("pathlib.Path.exists", return_value=True):
            scenario = load_scenario(Path("/mock/scenario.yaml"), app_config)

        assert scenario.live_grpc_endpoints["sionna"] == "grpc://generator:50051"
        assert scenario.live_grpc_endpoints["aux"] == "grpc://aux-override:50061"
        assert scenario.live_grpc_endpoints["http"] == "http://override:8080"

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_runtime_live_override_preserves_host_and_replaces_port(
        self,
        mock_yaml_load,
        mock_file,
    ):
        mock_yaml_load.return_value = {
            "schema_version": 2,
            "timeline": {"steps": 1, "duration_s": 0.0},
            "scene": {"id": "test_scene"},
            "data": {"mode": "files"},
        }
        app_config = AppConfig.get_defaults()
        app_config.live_grpc = {"sionna": "grpc://generator-host:50051"}

        with patch("pathlib.Path.exists", return_value=True):
            scenario = load_scenario(
                Path("/mock/scenario.yaml"),
                app_config,
                data_mode_override="live_grpc",
                grpc_port_override=50052,
            )

        assert scenario.data_mode == "live_grpc"
        assert scenario.data_spec["mode"] == "live_grpc"
        assert scenario.data_spec["live_grpc"]["endpoint"] == ("grpc://generator-host:50052")
        assert scenario.live_grpc_endpoints["sionna"] == "grpc://generator-host:50052"

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_runtime_live_override_preserves_ipv6_host(
        self,
        mock_yaml_load,
        mock_file,
    ):
        mock_yaml_load.return_value = {
            "schema_version": 2,
            "timeline": {"steps": 1, "duration_s": 0.0},
            "scene": {"id": "test_scene"},
            "data": {
                "mode": "live_grpc",
                "live_grpc": {"endpoint": "grpc://[::1]:50051"},
            },
        }

        with patch("pathlib.Path.exists", return_value=True):
            scenario = load_scenario(
                Path("/mock/scenario.yaml"),
                AppConfig.get_defaults(),
                grpc_port_override=50052,
            )

        assert scenario.data_spec["live_grpc"]["endpoint"] == "grpc://[::1]:50052"
        assert scenario.live_grpc_endpoints["sionna"] == "grpc://[::1]:50052"

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_runtime_remote_override_preserves_host_and_replaces_port(
        self,
        mock_yaml_load,
        mock_file,
    ):
        mock_yaml_load.return_value = {
            "schema_version": 2,
            "timeline": {"steps": 1, "duration_s": 0.0},
            "scene": {"id": "test_scene"},
            "data": {
                "mode": "remote_hdf5",
                "remote_hdf5": {"server": "frame-host:50052", "cache_size": 12},
            },
        }

        with patch("pathlib.Path.exists", return_value=True):
            scenario = load_scenario(
                Path("/mock/scenario.yaml"),
                AppConfig.get_defaults(),
                grpc_port_override=50053,
            )

        assert scenario.data_mode == "remote_hdf5"
        assert scenario.data_spec["remote_hdf5"] == {
            "server": "frame-host:50053",
            "cache_size": 12,
        }

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_runtime_remote_override_preserves_ipv6_host(
        self,
        mock_yaml_load,
        mock_file,
    ):
        mock_yaml_load.return_value = {
            "schema_version": 2,
            "timeline": {"steps": 1, "duration_s": 0.0},
            "scene": {"id": "test_scene"},
            "data": {
                "mode": "remote_hdf5",
                "remote_hdf5": {"server": "[::1]:50052"},
            },
        }

        with patch("pathlib.Path.exists", return_value=True):
            scenario = load_scenario(
                Path("/mock/scenario.yaml"),
                AppConfig.get_defaults(),
                grpc_port_override=50053,
            )

        assert scenario.data_spec["remote_hdf5"]["server"] == "[::1]:50053"

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_runtime_grpc_port_rejects_effective_files_mode(
        self,
        mock_yaml_load,
        mock_file,
    ):
        mock_yaml_load.return_value = {
            "schema_version": 2,
            "timeline": {"steps": 1, "duration_s": 0.0},
            "scene": {"id": "test_scene"},
            "data": {"mode": "files"},
        }

        with (
            patch("pathlib.Path.exists", return_value=True),
            pytest.raises(ValueError, match="requires live_grpc or remote_hdf5"),
        ):
            load_scenario(
                Path("/mock/scenario.yaml"),
                AppConfig.get_defaults(),
                grpc_port_override=50052,
            )

    @patch("builtins.open", new_callable=mock_open)
    @patch("yaml.safe_load")
    def test_load_scenario_rejects_unsupported_schema(self, mock_yaml_load, mock_file):
        mock_yaml_load.return_value = {
            "schema_version": 1,
            "timeline": {"steps": 1, "duration_s": 0.0},
        }

        with (
            patch("pathlib.Path.exists", return_value=True),
            pytest.raises(ValueError, match="Unsupported scenario schema version"),
        ):
            load_scenario(Path("/mock/scenario.yaml"), AppConfig.get_defaults())

    def test_resolve_scene_meshes(self):
        """Test resolving scene meshes."""
        app_config = AppConfig.get_defaults()
        app_config.scenes = Path("/mock/scenes")

        scene_spec = {"source": "library", "id": "test_scene"}

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.rglob", return_value=[Path("mesh1.obj")]),
            patch("pathlib.Path.glob", return_value=[Path("mesh2.ply")]),
        ):
            meshes = resolve_scene_meshes(scene_spec, app_config)

        assert len(meshes) == 2
        assert Path("mesh1.obj") in meshes
        assert Path("mesh2.ply") in meshes
