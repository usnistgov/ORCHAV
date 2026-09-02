"""Capture a deterministic production-pygfx MPC selection overlay.

This is a visual-regression aid, not a renderer benchmark. It constructs the
normal pygfx renderer, adds a muted synthetic MPC population through the
renderer-neutral geometry API, then presents one
``MpcPathInspectionSnapshot`` through the production selection overlay.
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def _logical_capture_resolution_scale(runtime_stats: Mapping[str, Any]) -> float:
    """Return the post-readback scale for a logical-pixel regression artifact."""
    for key in ("canvas_pixel_ratio", "renderer_pixel_ratio"):
        try:
            pixel_ratio = float(runtime_stats.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(pixel_ratio) and pixel_ratio > 0.0:
            return 1.0 / pixel_ratio
    return 1.0


def _background_paths() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic dense-scene context for the selected path."""
    rng = np.random.default_rng(20260724)
    points: list[np.ndarray] = []
    lines: list[tuple[int, int]] = []
    colors: list[tuple[float, float, float, float]] = []
    for path_index in range(34):
        tx = np.asarray((-6.0, -1.5, 0.0), dtype=np.float32)
        rx = np.asarray((6.0, 1.5, 0.0), dtype=np.float32)
        first = np.asarray(
            (
                rng.uniform(-4.8, -1.0),
                rng.uniform(-3.8, 3.8),
                rng.uniform(0.2, 3.4),
            ),
            dtype=np.float32,
        )
        second = np.asarray(
            (
                rng.uniform(1.0, 4.8),
                rng.uniform(-3.8, 3.8),
                rng.uniform(0.2, 3.4),
            ),
            dtype=np.float32,
        )
        path = np.vstack((tx, first, second, rx))
        offset = len(points)
        points.extend(path)
        for segment in range(3):
            lines.append((offset + segment, offset + segment + 1))
            alpha = 0.14 + 0.12 * ((path_index % 5) / 4.0)
            colors.append((0.18, 0.38, 0.58, alpha))
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(lines, dtype=np.int32),
        np.asarray(colors, dtype=np.float32),
    )


def capture_mpc_explorer_visual(output: Path, *, width: int = 1200, height: int = 760) -> Path:
    """Render the selected-path presentation at the requested logical pixel size.

    The renderer captures its DPR-native physical framebuffer. This regression
    helper applies a post-readback scale so the saved comparison artifact has
    stable dimensions across displays.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("ORCHAV_PYGFX_PRESENT_METHOD", "bitmap")
    os.environ.setdefault("ORCHAV_PYGFX_CANVAS_SCHEDULER", "manual")

    from PySide6.QtWidgets import QApplication

    from visualizer.src.renderers.mpc_path_inspection import MpcPathInspectionSnapshot
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer
    from visualizer.src.types.camera_state import CameraState
    from visualizer.src.types.render_payloads import LineSetPayload, MaterialPayload

    app = QApplication.instance() or QApplication([])
    visualizer = SimpleNamespace(
        animation_running=False,
        _cli_driven_frame_run=True,
        close=lambda: None,
    )
    renderer = PygfxRenderer(visualizer)
    try:
        renderer.initialize_visualizer(
            "MPC Explorer visual regression",
            width=width,
            height=height,
            install_default_interactions=False,
        )
        renderer.set_background_color((0.012, 0.025, 0.055))

        points, lines, colors = _background_paths()
        context = LineSetPayload(points=points, lines=lines, colors=colors)
        context_material = MaterialPayload(
            base_color=(0.35, 0.55, 0.75, 0.24),
            shader="unlit",
            line_width=1.25,
        )
        if not renderer.ensure_named_geometry(
            "mpc_explorer_visual::context",
            context,
            material=context_material,
        ):
            raise RuntimeError("could not create visual-regression MPC context")

        selected_points = np.asarray(
            (
                (-6.0, -1.5, 0.0),
                (-3.4, 2.6, 1.6),
                (-0.2, -0.4, 3.1),
                (3.1, 2.9, 1.8),
                (6.0, 1.5, 0.0),
            ),
            dtype=np.float32,
        )
        snapshot = MpcPathInspectionSnapshot(
            frame_token=("mpc-explorer-visual", 0),
            canonical_path_id=17,
            points=selected_points,
            bounce_interaction_types=np.asarray((1, 8, 2), dtype=np.int32),
            bounce_colors=np.asarray(
                (
                    (1.0, 0.42, 0.12, 1.0),
                    (0.72, 0.30, 1.0, 1.0),
                    (0.18, 0.82, 1.0, 1.0),
                ),
                dtype=np.float32,
            ),
        )
        if not renderer.set_mpc_path_inspection(snapshot):
            raise RuntimeError("production pygfx selection overlay rejected the snapshot")
        if not renderer.update_mpc_path_flow(0.58):
            raise RuntimeError("production pygfx selection pulse could not be positioned")

        camera = CameraState(
            eye=(13.5, -18.0, 11.5),
            lookat=(0.0, 0.6, 1.2),
            up=(0.0, 0.0, 1.0),
            fov_deg=42.0,
        )
        if not renderer.set_camera_state(camera):
            raise RuntimeError("could not set the visual-regression camera")

        app.processEvents()
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        resolution_scale = _logical_capture_resolution_scale(renderer.get_runtime_stats())
        if not renderer.export_screenshot(
            str(output),
            resolution_scale=resolution_scale,
        ):
            raise RuntimeError("pygfx screenshot export returned no visible image")
        return output
    finally:
        renderer.close()
        app.processEvents()


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mpc_explorer/mpc_selected_path.png"),
        help="PNG path (default: artifacts/mpc_explorer/mpc_selected_path.png)",
    )
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=760)
    args = parser.parse_args()
    output = capture_mpc_explorer_visual(
        args.output,
        width=max(320, args.width),
        height=max(240, args.height),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
