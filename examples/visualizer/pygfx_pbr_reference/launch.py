"""Open the generated reference in the actual ORCHAV desktop application."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=HERE.parents[2])
    parser.add_argument("--capture-dir", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    os.environ.setdefault("ORCHAV_ENABLE_TEXTURES", "1")
    os.environ.setdefault("ORCHAV_PYGFX_DEFER_DEFAULT_IBL", "1")
    from generate import generate

    # Publication uses the public canonical frame writer. This is a controlled
    # appearance fixture with empty MPC paths, never an RF simulation result.
    generate(HERE)
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    from visualizer.src.types.camera_state import CameraState
    from visualizer.visualizer import OrchavVisualizer

    app = QApplication([])
    if args.capture_dir:

        def unexpected_dialog(*values, **_kwargs):
            raise RuntimeError(f"Unexpected dialog during capture: {values[1:]}")

        QMessageBox.critical = unexpected_dialog
        QMessageBox.warning = unexpected_dialog
    viz = OrchavVisualizer(renderer_type="pygfx", viewport_mode="embedded")
    viz.resize(1600, 1100)
    viz.show()
    app.processEvents()
    viz._deferred_init()
    outcome = viz.open_scenario(str(HERE / "scenario.yaml"))
    if not getattr(outcome, "frame_source_ready", False):
        raise RuntimeError(f"Reference scenario frame did not open: {outcome}")

    def configure():
        renderer = viz.renderer
        renderer.set_camera_state(CameraState((0, -19, 20), (0, 0, 0), (0, 0, 1), 48))
        if not renderer.set_ibl(str(HERE / "assets" / "reference")):
            raise RuntimeError("Generated reference HDR failed to load")
        renderer.set_ibl_intensity(10500)
        viz.set_state(show_labels=True)
        renderer.request_redraw()
        if args.capture_dir:
            QTimer.singleShot(2500, capture)

    def capture():
        import pygfx
        import wgpu
        from PIL import Image

        output = args.capture_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        viz.grab().save(str(output / "application.png"))
        renderer = viz.renderer
        for scale in (1.0, 2.0):
            pixels = renderer.export_screenshot_to_array(scale, include_hud=True)
            Image.fromarray(pixels).save(output / f"viewport_{int(scale)}x.png")
        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "scope": "Actual ORCHAV Qt application, native scenario and canonical empty frame; controlled visual QA only",
                    "repo_root": str(args.repo_root.resolve()),
                    "renderer_module": sys.modules[type(renderer).__module__].__file__,
                    "camera": renderer.get_camera_state().to_dict(),
                    "pygfx": pygfx.__version__,
                    "wgpu": wgpu.__version__,
                    "frame_source_ready": bool(outcome.frame_source_ready),
                    "environment": "Explicit generated reference.hdr with linear radiance0.18,1,4; not neutral_outdoor LFS asset",
                    "scenario": str(HERE / "scenario.yaml"),
                },
                indent=2,
            )
        )
        app.quit()

    QTimer.singleShot(1200, configure)
    app.exec()


if __name__ == "__main__":
    main()
