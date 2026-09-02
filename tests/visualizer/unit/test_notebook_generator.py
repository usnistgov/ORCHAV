"""Tests for the notebook generator integration (reload / regenerate)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_scenario():
    """Minimal Scenario-like object for testing."""
    s = MagicMock()
    s.root = Path("/tmp/fake_scenario")
    s.data_spec = {"files": {"pattern": "frames/frame_*.h5", "format": "h5"}}
    s.data_mode = "files"
    return s


@pytest.fixture()
def mock_scenario_cfg():
    """Minimal ScenarioConfiguration-like object."""
    cfg = MagicMock()
    cfg.root = Path("/tmp/fake_scenario")
    cfg.scene_xml = Path("/tmp/fake_scenario/scene.xml")
    cfg.frames_dir = Path("/tmp/fake_scenario/frames")
    cfg.frames_pattern = "frames/frame_*.h5"
    cfg.frames_format = "h5"
    return cfg


class TestNotebookGenerator:
    """Test NotebookGenerator.generate() with mocked pipeline."""

    def test_generate_calls_pipeline(self, mock_scenario_cfg):
        from visualizer.src.notebook.generator import NotebookGenerator

        gen = NotebookGenerator(
            scene_xml_path="/tmp/scene.xml",
            project_root=Path("/tmp/project"),
            scenario_root=Path("/tmp/fake_scenario"),
            scenario_configuration=mock_scenario_cfg,
        )

        with patch("visualizer.src.notebook.generator.NotebookGenerator.generate") as mock_gen:
            mock_gen.return_value = [0, 1, 2]
            result = gen.generate(
                tx_positions=[[0, 0, 25]],
                rx_positions=[[10, 0, 1.5]],
                steps=3,
            )
            assert result == [0, 1, 2]

    @patch("generator.core.pipeline.perform_offline_pipeline")
    @patch("generator.core.configuration.build_simulation_config")
    def test_generate_builds_configs(self, mock_build_sim, mock_pipeline, mock_scenario_cfg):
        from visualizer.src.notebook.generator import NotebookGenerator

        mock_build_sim.return_value = MagicMock()

        gen = NotebookGenerator(
            scene_xml_path="/tmp/scene.xml",
            project_root=Path("/tmp/project"),
            scenario_root=Path("/tmp/fake_scenario"),
            scenario_configuration=mock_scenario_cfg,
        )

        result = gen.generate(
            tx_positions=[[0, 0, 25], [5, 5, 20]],
            rx_positions=[[10, 0, 1.5]],
            quality="ultra-low",
            steps=2,
        )

        assert result == [0, 1]
        mock_pipeline.assert_called_once()
        call_args = mock_pipeline.call_args
        tx_configs = call_args[0][0]
        rx_configs = call_args[0][1]
        assert len(tx_configs) == 2
        assert len(rx_configs) == 1
        assert tx_configs[0].name == "TX0"
        assert rx_configs[0].name == "RX0"

    @patch("generator.core.pipeline.perform_offline_pipeline")
    @patch("generator.core.configuration.build_simulation_config")
    def test_generate_passes_overrides(self, mock_build_sim, mock_pipeline, mock_scenario_cfg):
        from visualizer.src.notebook.generator import NotebookGenerator

        mock_build_sim.return_value = MagicMock()

        gen = NotebookGenerator(
            scene_xml_path="/tmp/scene.xml",
            project_root=Path("/tmp/project"),
            scenario_root=Path("/tmp/fake_scenario"),
            scenario_configuration=mock_scenario_cfg,
        )

        gen.generate(
            tx_positions=[[0, 0, 25]],
            rx_positions=[[10, 0, 1.5]],
            quality="high",
            steps=5,
            carrier_frequency_hz=60e9,
            bandwidth_hz=1e9,
        )

        overrides = mock_build_sim.call_args[1]["overrides"]
        assert overrides["quality"] == "high"
        assert overrides["num_steps"] == 5
        assert overrides["carrier_frequency_hz"] == 60e9
        assert overrides["bandwidth_hz"] == 1e9


class TestGenerateFrames:
    """Test the standalone generate_frames() function."""

    @patch("visualizer.src.notebook.generator.NotebookGenerator.generate")
    @patch("shared.scenarios.load_scenario_configuration")
    def test_standalone_function(self, mock_load_scenario, mock_generate):
        from visualizer.src.notebook.generator import generate_frames

        mock_cfg = MagicMock()
        mock_cfg.root = Path("/tmp/fake_scenario")
        mock_cfg.scene_xml = Path("/tmp/fake_scenario/scene.xml")
        mock_load_scenario.return_value = mock_cfg
        mock_generate.return_value = [0]

        result = generate_frames(
            "/tmp/fake_scenario",
            tx_positions=[[0, 0, 25]],
            rx_positions=[[10, 0, 1.5]],
            project_root="/tmp/project",
        )

        assert result == [0]
        mock_load_scenario.assert_called_once()
        mock_generate.assert_called_once()

    @patch("shared.scenarios.load_scenario_configuration")
    def test_raises_on_missing_scene_xml(self, mock_load_scenario):
        from visualizer.src.notebook.generator import generate_frames

        mock_cfg = MagicMock()
        mock_cfg.root = Path("/tmp/fake_scenario")
        mock_cfg.scene_xml = None
        mock_load_scenario.return_value = mock_cfg

        with pytest.raises(RuntimeError, match="No scene XML"):
            generate_frames(
                "/tmp/fake_scenario",
                tx_positions=[[0, 0, 25]],
                rx_positions=[[10, 0, 1.5]],
                project_root="/tmp/project",
            )


class TestReload:
    """Test that reload() resets the frame source and clears caches."""

    def test_reload_recreates_file_source_and_clears_notebook_caches(self, mock_scenario):
        from visualizer.src.notebook.pygfx import PygfxNotebookViz

        mock_frame_source = MagicMock()
        mock_frame_source.list_frames.return_value = [0, 1, 2]

        mock_mpc_core = MagicMock()
        old_source = MagicMock()

        with patch("visualizer.src.io.frame_sources.FileSource", return_value=mock_frame_source):
            obj = object.__new__(PygfxNotebookViz)
            obj._scenario = mock_scenario
            obj._frame_source = old_source
            obj._mpc_core = mock_mpc_core
            obj.mpc_view_cache = {0: object()}
            obj.last_app_state = object()

            obj.reload()

            assert obj._frame_source == mock_frame_source
            old_source.close.assert_called_once()
            mock_frame_source.open.assert_called_once()
            mock_mpc_core.invalidate_step.assert_called_once_with(step=None)
            assert obj.mpc_view_cache == {}
            assert obj.last_app_state is None

    def test_close_is_idempotent_and_invalidates_frame_caches(self, mock_scenario):
        from visualizer.src.notebook.pygfx import PygfxNotebookViz

        source = MagicMock()
        obj = object.__new__(PygfxNotebookViz)
        obj._scenario = mock_scenario
        obj._frame_source = source
        obj._mpc_core = MagicMock()
        obj.mpc_view_cache = {0: object()}
        obj.last_app_state = object()

        with obj as entered:
            assert entered is obj

        obj.close()

        source.close.assert_called_once()
        assert obj._frame_source is None
        assert obj.mpc_view_cache == {}
        assert obj.last_app_state is None
        assert obj._mpc_core.invalidate_step.call_count == 2

    def test_reload_closes_failed_replacement(self, mock_scenario):
        from visualizer.src.notebook.pygfx import PygfxNotebookViz

        old_source = MagicMock()
        replacement = MagicMock()
        replacement.open.side_effect = OSError("cannot open")
        obj = object.__new__(PygfxNotebookViz)
        obj._scenario = mock_scenario
        obj._frame_source = old_source
        obj._mpc_core = MagicMock()

        with patch("visualizer.src.io.frame_sources.FileSource", return_value=replacement):
            with pytest.raises(OSError, match="cannot open"):
                obj.reload()

        old_source.close.assert_called_once()
        replacement.close.assert_called_once()
        assert obj._frame_source is None


class TestRegenerate:
    """Test that regenerate() calls generate + reload in sequence."""

    def test_regenerate_calls_close_generate_then_reload(self, mock_scenario):
        from visualizer.src.notebook.pygfx import PygfxNotebookViz

        obj = object.__new__(PygfxNotebookViz)
        obj._scenario = mock_scenario
        obj._scene_xml_path = "/tmp/scene.xml"
        obj._project_root = Path("/tmp/project")
        obj._generator = MagicMock()
        events = []
        obj.close = MagicMock(side_effect=lambda: events.append("close"))
        obj._generator.generate.side_effect = lambda *args, **kwargs: (
            events.append("generate") or [0, 1]
        )

        with patch.object(
            obj, "reload", side_effect=lambda: events.append("reload")
        ) as mock_reload:
            result = obj.regenerate(
                tx_positions=[[0, 0, 25]],
                rx_positions=[[10, 0, 1.5]],
                steps=2,
            )

        assert result == [0, 1]
        assert events == ["close", "generate", "reload"]
        obj.close.assert_called_once()
        obj._generator.generate.assert_called_once()
        mock_reload.assert_called_once()

    def test_regenerate_reopens_rolled_back_frames_after_failure(self, mock_scenario):
        from visualizer.src.notebook.pygfx import PygfxNotebookViz

        original = RuntimeError("generation failed")
        obj = object.__new__(PygfxNotebookViz)
        obj._scenario = mock_scenario
        obj._scene_xml_path = "/tmp/scene.xml"
        obj._project_root = Path("/tmp/project")
        obj._generator = MagicMock()
        obj._generator.generate.side_effect = original
        obj.close = MagicMock()
        obj.reload = MagicMock()

        with pytest.raises(RuntimeError, match="generation failed") as raised:
            obj.regenerate(
                tx_positions=[[0, 0, 25]],
                rx_positions=[[10, 0, 1.5]],
            )

        assert raised.value is original
        obj.close.assert_called_once()
        obj.reload.assert_called_once()

    def test_regenerate_recovery_does_not_mask_generation_failure(self, mock_scenario):
        from visualizer.src.notebook.pygfx import PygfxNotebookViz

        original = RuntimeError("generation failed")
        obj = object.__new__(PygfxNotebookViz)
        obj._scenario = mock_scenario
        obj._scene_xml_path = "/tmp/scene.xml"
        obj._project_root = Path("/tmp/project")
        obj._generator = MagicMock()
        obj._generator.generate.side_effect = original
        obj.close = MagicMock()
        obj.reload = MagicMock(side_effect=OSError("reopen failed"))

        with pytest.raises(RuntimeError, match="generation failed") as raised:
            obj.regenerate(
                tx_positions=[[0, 0, 25]],
                rx_positions=[[10, 0, 1.5]],
            )

        assert raised.value is original
        assert any("reopen failed" in note for note in raised.value.__notes__)

    def test_regenerate_raises_without_scene_xml(self, mock_scenario):
        from visualizer.src.notebook.pygfx import PygfxNotebookViz

        obj = object.__new__(PygfxNotebookViz)
        obj._scenario = mock_scenario
        obj._scene_xml_path = None
        obj._generator = None

        with pytest.raises(RuntimeError, match="no scene XML"):
            obj.regenerate(
                tx_positions=[[0, 0, 25]],
                rx_positions=[[10, 0, 1.5]],
            )
