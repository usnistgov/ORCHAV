"""Tests for the pygfx notebook and standalone headless interfaces."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from visualizer.src.notebook.pygfx import (
    PYGFX_AVAILABLE,
    PygfxNotebookViz,
    _normalize_pygfx_ibl_intensity,
)

HELLO_WORLD = "scenarios/getting_started/hello_world"


def test_normalize_pygfx_ibl_intensity_accepts_both_scales():
    assert _normalize_pygfx_ibl_intensity(0) == pytest.approx(0.0)
    assert _normalize_pygfx_ibl_intensity(1.0) == pytest.approx(1.0)
    assert _normalize_pygfx_ibl_intensity(2.67) == pytest.approx(2.67)
    assert _normalize_pygfx_ibl_intensity(30000.0) == pytest.approx(1.0)
    assert _normalize_pygfx_ibl_intensity(80000.0) == pytest.approx(80000.0 / 30000.0)


def test_notebook_readmes_use_current_pygfx_import_path():
    current_import = "from visualizer.src.notebook.pygfx import PygfxNotebookViz"
    stale_import = "from visualizer.src.notebook_pygfx import PygfxNotebookViz"

    for doc_path in (
        Path("examples/notebook/README.md"),
        Path("scenarios/visualizer/notebook_mode/README.md"),
    ):
        text = doc_path.read_text(encoding="utf-8")
        assert stale_import not in text
        assert current_import in text


def test_notebook_ipynb_is_cleared_and_references_current_fixture():
    notebook_path = Path("examples/notebook/notebook_mode.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "from visualizer.src.notebook.pygfx import PygfxNotebookViz" in code
    assert "from visualizer.src.notebook_pygfx import PygfxNotebookViz" not in code
    assert "scenarios/visualizer/notebook_mode" in code
    assert "find_repo_root(Path.cwd())" in code
    assert "project_root=REPO_ROOT" in code
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []


@pytest.mark.headless
@pytest.mark.parametrize("draw_raises", (False, True), ids=("success", "draw-error"))
def test_offscreen_render_always_closes_canvas(draw_raises):
    """Native canvas resources are released on success and render failure."""
    canvas = MagicMock()
    renderer = MagicMock()
    if draw_raises:
        canvas.draw.side_effect = RuntimeError("render failed")
    else:
        canvas.draw.return_value = np.zeros((24, 32, 4), dtype=np.uint8)

    viz = MagicMock()
    viz._build_pygfx_scene.return_value = (MagicMock(), MagicMock())

    with (
        patch("rendercanvas.offscreen.RenderCanvas", return_value=canvas),
        patch(
            "visualizer.src.renderers.pygfx.canvas.create_wgpu_renderer",
            return_value=renderer,
        ),
    ):
        if draw_raises:
            with pytest.raises(RuntimeError, match="render failed"):
                PygfxNotebookViz.render(viz, display=False, width=32, height=24)
        else:
            image = PygfxNotebookViz.render(viz, display=False, width=32, height=24)
            assert image.shape == (24, 32, 3)

    canvas.close.assert_called_once_with()


@pytest.mark.headless
@pytest.mark.skipif(not PYGFX_AVAILABLE, reason="pygfx not installed")
class TestPygfxNotebookViz:
    @pytest.fixture(scope="class")
    def viz(self):
        """Shared PygfxNotebookViz instance (expensive to create)."""
        viz = PygfxNotebookViz(HELLO_WORLD)
        if viz.num_frames < 1:
            viz.close()
            pytest.skip("Notebook frame fixtures are not included in this checkout")
        try:
            yield viz
        finally:
            viz.close()

    def test_repr(self, viz):
        r = repr(viz)
        assert "PygfxNotebookViz" in r
        assert "meshes=" in r
        assert "frames=" in r

    def test_frames_property(self, viz):
        frames = viz.frames
        assert isinstance(frames, list)
        assert len(frames) >= 1
        assert all(isinstance(f, int) for f in frames)

    def test_num_frames(self, viz):
        assert viz.num_frames == len(viz.frames)

    @pytest.mark.pygfx_runtime
    def test_render_returns_numpy(self, viz):
        img = viz.render(frame=0, display=False, width=320, height=240)
        assert isinstance(img, np.ndarray)
        assert img.shape == (240, 320, 3)
        assert img.dtype == np.uint8

    @pytest.mark.pygfx_runtime
    def test_render_custom_resolution(self, viz):
        img = viz.render(frame=0, display=False, width=640, height=480)
        assert img.shape == (480, 640, 3)

    @pytest.mark.pygfx_runtime
    def test_render_color_mode_mpc_type(self, viz):
        img = viz.render(
            frame=0,
            display=False,
            width=320,
            height=240,
            color_mode="mpc_type",
        )
        assert img.shape == (240, 320, 3)

    @pytest.mark.pygfx_runtime
    def test_render_without_mpc_paths(self, viz):
        img = viz.render(
            frame=0,
            display=False,
            width=320,
            height=240,
            show_mpc_paths=False,
        )
        assert img.shape == (240, 320, 3)

    @pytest.mark.pygfx_runtime
    def test_render_without_mpc_bounce_points(self, viz):
        img = viz.render(
            frame=0,
            display=False,
            width=320,
            height=240,
            show_mpc_bounce_points=False,
        )
        assert img.shape == (240, 320, 3)

    @pytest.mark.pygfx_runtime
    def test_render_camera_params(self, viz):
        img = viz.render(
            frame=0,
            display=False,
            width=320,
            height=240,
            azimuth=90,
            elevation=60,
            distance=200,
        )
        assert img.shape == (240, 320, 3)

    @pytest.mark.pygfx_runtime
    def test_render_has_content(self, viz):
        """Verify the rendered image is not all-black (scene is visible)."""
        img = viz.render(frame=0, display=False, width=320, height=240)
        non_black = (img.sum(axis=2) > 0).sum()
        total = img.shape[0] * img.shape[1]
        assert non_black > total * 0.05, "Image appears mostly black"

    @pytest.mark.pygfx_runtime
    def test_render_scene_only(self, viz):
        """Scene-only render (no MPCs) should still have content."""
        img = viz.render(
            frame=0,
            display=False,
            width=320,
            height=240,
            mpc_layer_enabled=False,
        )
        non_black = (img.sum(axis=2) > 0).sum()
        total = img.shape[0] * img.shape[1]
        assert non_black > total * 0.05


@pytest.mark.headless
@pytest.mark.skipif(not PYGFX_AVAILABLE, reason="pygfx not installed")
class TestPygfxHeadlessCLI:
    def test_headless_parser_accepts_ibl_name(self):
        """Test the headless CLI exposes IBL environment selection."""
        from visualizer.headless import _build_parser

        args = _build_parser().parse_args(
            [
                HELLO_WORLD,
                "--ibl-name",
                "neutral_white",
            ]
        )
        assert args.ibl_name == "neutral_white"
        assert not hasattr(args, "renderer")

    @pytest.mark.parametrize("renderer", ("pygfx", "open3d"))
    def test_headless_parser_rejects_renderer_selection(self, renderer: str):
        """Standalone headless rendering has no backend selector."""
        from visualizer.headless import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args([HELLO_WORLD, "--renderer", renderer])

    def test_headless_parser_exposes_semantic_mpc_visibility_flags(self):
        from visualizer.headless import _build_parser

        args = _build_parser().parse_args(
            [
                HELLO_WORLD,
                "--disable-mpc-layer",
                "--hide-mpc-paths",
                "--hide-mpc-bounce-points",
            ]
        )

        assert args.mpc_layer_enabled is False
        assert args.show_mpc_paths is False
        assert args.show_mpc_bounce_points is False

    def test_render_scenario_closes_frame_source(self):
        from visualizer.headless import render_scenario

        expected = np.zeros((24, 32, 3), dtype=np.uint8)
        fake_viz = MagicMock()
        fake_viz.__enter__.return_value = fake_viz
        fake_viz.render.return_value = expected

        with patch(
            "visualizer.src.notebook.pygfx.PygfxNotebookViz",
            return_value=fake_viz,
        ):
            actual = render_scenario(HELLO_WORLD, width=32, height=24)

        assert actual is expected
        fake_viz.__exit__.assert_called_once_with(None, None, None)

    @pytest.mark.pygfx_runtime
    def test_render_scenario_pygfx(self, tmp_path):
        """Test that render_scenario() uses the supported pygfx path."""
        from visualizer.headless import render_scenario

        out = tmp_path / "test.png"
        img = render_scenario(
            HELLO_WORLD,
            frame=0,
            output=str(out),
            width=320,
            height=240,
            ibl_name="neutral_white",
        )
        assert img.shape == (240, 320, 3)
        assert out.exists()


@pytest.mark.headless
class TestPygfxAvailableGuard:
    def test_pygfx_available_is_bool(self):
        assert isinstance(PYGFX_AVAILABLE, bool)
