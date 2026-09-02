"""Pygfx headless rendering API for scripted and CLI use.

``render_scenario()`` loads a scenario, renders one frame through pygfx and
wgpu, and returns a NumPy image array. It can also save the image to a file.
From a cloned repository, install the default runtime with
``python -m pip install -e .``.

No display server, no Qt, no Jupyter required.

Python API::

    from visualizer.headless import render_scenario

    # Returns (H, W, 3) uint8 numpy array
    img = render_scenario("scenarios/my_scenario", frame=3, azimuth=45)

    # Save to file and return array
    render_scenario("scenarios/my_scenario", frame=3, output="frame3.png")

CLI::

    python -m visualizer.headless scenarios/my_scenario --frame 3 -o frame3.png
    python -m visualizer.headless scenarios/my_scenario --frames 0,1,2 -o frames/
    python -m visualizer.headless scenarios/my_scenario --frame 0 --base64

See ``docs/visualizer/headless_rendering.md`` for full documentation.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from shared.logging import get_logger

from .src.materials.texture_policy import apply_texture_launch_policy

logger = get_logger("orchav.headless")


def render_scenario(
    scenario_path: str | Path,
    frame: int = 0,
    *,
    output: str | Path | None = None,
    color_mode: str = "reflection_order",
    selected_tx: int | str = "all",
    selected_rx: int | str = "all",
    azimuth: float | None = None,
    elevation: float | None = None,
    distance: float | None = None,
    center: list[float] | None = None,
    width: int = 1280,
    height: int = 900,
    mpc_layer_enabled: bool = True,
    show_mpc_paths: bool = True,
    show_mpc_bounce_points: bool = True,
    line_width: float = 2.0,
    point_size: float = 5.0,
    node_radius: float = 1.0,
    ibl_name: str = "neutral_outdoor",
    ibl_intensity: float = 1.0,
    tx_positions: Sequence[Sequence[float]] | None = None,
    rx_positions: Sequence[Sequence[float]] | None = None,
    project_root: str | Path | None = None,
) -> np.ndarray:
    """Render a scenario frame to an image array, optionally saving to file.

    Args:
        scenario_path: Path to scenario directory or YAML file.
        frame: Frame index to render.
        output: If given, save the rendered image to this path (PNG/JPG).
        color_mode: MPC coloring — ``"reflection_order"``, ``"mpc_type"``,
            ``"delay"``, or ``"path_loss"``.
        selected_tx: TX filter (``"all"`` or 0-based index).
        selected_rx: RX filter (``"all"`` or 0-based index).
        azimuth: Camera azimuth in degrees (0 = +X axis).
        elevation: Camera elevation in degrees (0 = horizon, 90 = top-down).
        distance: Camera distance from scene center.
        center: Camera look-at point ``[x, y, z]``.
        width: Image width in pixels.
        height: Image height in pixels.
        mpc_layer_enabled: Enable the MPC layer master switch.
        show_mpc_paths: Show MPC path lines while the MPC layer is enabled.
        show_mpc_bounce_points: Show physical MPC interaction points while the layer is enabled.
        line_width: MPC path line width.
        point_size: Bounce point size.
        node_radius: TX/RX marker radius.
        ibl_name: Image-based lighting environment name from ``libraries/ibl``.
        ibl_intensity: Image-based lighting intensity multiplier.
        tx_positions: Override TX marker positions (visual only).
        rx_positions: Override RX marker positions (visual only).
        project_root: Explicit project root. Auto-detected when ``None``.

    Returns:
        Rendered image as ``(H, W, 3)`` uint8 numpy array.
    """
    from visualizer.src.notebook.pygfx import PygfxNotebookViz

    with PygfxNotebookViz(scenario_path, project_root=project_root) as viz:
        img = viz.render(
            frame=frame,
            display=False,
            color_mode=color_mode,
            selected_tx=selected_tx,
            selected_rx=selected_rx,
            azimuth=azimuth,
            elevation=elevation,
            distance=distance,
            center=center,
            width=width,
            height=height,
            mpc_layer_enabled=mpc_layer_enabled,
            show_mpc_paths=show_mpc_paths,
            show_mpc_bounce_points=show_mpc_bounce_points,
            line_width=line_width,
            point_size=point_size,
            node_radius=node_radius,
            ibl_name=ibl_name,
            ibl_intensity=ibl_intensity,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
        )
    if output is not None:
        _save_image(img, Path(output))
    return img


def _save_image(img: np.ndarray, path: Path) -> None:
    """Save a numpy image array to a file."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(str(path))
    logger.info("Saved %dx%d image to %s", img.shape[1], img.shape[0], path)


def _encode_base64_png(img: np.ndarray) -> str:
    """Encode a numpy image array as a base64 PNG string."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _build_parser() -> argparse.ArgumentParser:
    """Build the headless renderer command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m visualizer.headless",
        description="Render scenario frames to images with pygfx (no display needed).",
    )
    parser.add_argument(
        "scenario",
        help="Path to scenario directory or YAML file.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Single frame index to render (default: 0).",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default=None,
        help="Comma-separated frame indices (e.g. 0,1,2).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file (single frame) or directory (multiple frames).",
    )
    parser.add_argument(
        "--base64",
        action="store_true",
        help="Print base64-encoded PNG to stdout.",
    )
    texture_group = parser.add_mutually_exclusive_group()
    texture_group.add_argument(
        "--enable-textures",
        action="store_true",
        help="Enable albedo/detail texture maps for the pygfx renderer.",
    )
    texture_group.add_argument(
        "--disable-textures",
        action="store_true",
        help="Disable all albedo/detail texture maps (default).",
    )

    # Camera
    camera = parser.add_argument_group("camera")
    camera.add_argument("--azimuth", type=float, default=None)
    camera.add_argument("--elevation", type=float, default=None)
    camera.add_argument("--distance", type=float, default=None)
    camera.add_argument(
        "--center",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
    )

    # Rendering
    render = parser.add_argument_group("rendering")
    render.add_argument("--width", type=int, default=1280)
    render.add_argument("--height", type=int, default=900)
    render.add_argument(
        "--color-mode",
        type=str,
        default="reflection_order",
        choices=["reflection_order", "mpc_type", "delay", "path_loss"],
    )
    render.add_argument(
        "--disable-mpc-layer",
        dest="mpc_layer_enabled",
        action="store_false",
        help="Hide the complete MPC layer.",
    )
    render.add_argument(
        "--hide-mpc-paths",
        dest="show_mpc_paths",
        action="store_false",
        help="Hide MPC path segments.",
    )
    render.add_argument(
        "--hide-mpc-bounce-points",
        dest="show_mpc_bounce_points",
        action="store_false",
        help="Hide physical MPC interaction points.",
    )
    render.add_argument("--line-width", type=float, default=2.0)
    render.add_argument("--point-size", type=float, default=5.0)
    render.add_argument("--node-radius", type=float, default=1.0)

    # Lighting
    light = parser.add_argument_group("lighting")
    light.add_argument(
        "--ibl-name",
        type=str,
        default="neutral_outdoor",
        help="IBL environment name from libraries/ibl (default: neutral_outdoor).",
    )
    light.add_argument("--ibl-intensity", type=float, default=1.0)

    return parser


def _main(argv: list[str] | None = None) -> None:
    """Run the headless rendering CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # When outputting base64 to stdout, suppress all non-error output to
    # avoid corrupting the data stream.
    if args.base64:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    apply_texture_launch_policy(
        enable_textures=args.enable_textures,
        disable_textures=args.disable_textures,
    )

    if args.frames is not None:
        frame_indices = [int(x.strip()) for x in args.frames.split(",")]
    elif args.frame is not None:
        frame_indices = [args.frame]
    else:
        frame_indices = [0]

    multi = len(frame_indices) > 1

    render_kwargs = dict(
        scenario_path=args.scenario,
        color_mode=args.color_mode,
        azimuth=args.azimuth,
        elevation=args.elevation,
        distance=args.distance,
        center=args.center,
        width=args.width,
        height=args.height,
        mpc_layer_enabled=args.mpc_layer_enabled,
        show_mpc_paths=args.show_mpc_paths,
        show_mpc_bounce_points=args.show_mpc_bounce_points,
        line_width=args.line_width,
        point_size=args.point_size,
        node_radius=args.node_radius,
        ibl_name=args.ibl_name,
        ibl_intensity=args.ibl_intensity,
    )

    for frame_idx in frame_indices:
        out_path: str | None = None
        if args.output:
            out = Path(args.output)
            if multi:
                out.mkdir(parents=True, exist_ok=True)
                out_path = str(out / f"frame_{frame_idx:04d}.png")
            else:
                out_path = args.output

        img = render_scenario(frame=frame_idx, output=out_path, **render_kwargs)

        if args.base64:
            sys.stdout.write(_encode_base64_png(img))
            if multi:
                sys.stdout.write("\n")
            sys.stdout.flush()

    if not args.base64 and not args.output:
        logger.info(
            "Rendered %d frame(s). Use -o to save or --base64 to print.",
            len(frame_indices),
        )


if __name__ == "__main__":
    _main()
