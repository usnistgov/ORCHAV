"""Reproduce sensing-volume and saved-fusion figures with the actual pygfx visualizer.

Run ``python -m visualizer.capture_sensing --help`` for the source and output options.
Only saved evidence is read; no acquisitions or sensing estimates are regenerated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def capture_volume_report(root, conditions):
    """Run isolated GUI captures after a scenario's numerical report is saved."""
    for condition in conditions:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "visualizer.capture_sensing",
                "--volume",
                str(root / condition),
                "--output-dir",
                str(root / "report/3d" / condition),
            ],
            check=True,
        )
    page = root / "report/index.html"
    if page.is_file():
        import html

        figures = "".join(
            f'<h3>{html.escape(condition.title())}</h3><img src="3d/{condition}/cutaway_3d.png" '
            f'alt="{html.escape(condition)}: actual Visualizer cutaway">'
            for condition in conditions
        )
        page.write_text(
            page.read_text() + "<h2>3D Visualizer views</h2>"
            "<p>Front cells are hidden to expose the volume interior. These are saved-result displays.</p>"
            + figures
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--volume", type=Path)
    source.add_argument("--fusion", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=-1, help="Fusion epoch; -1 selects the last.")
    parser.add_argument("--camera-eye", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-lookat", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--include-layout",
        action="store_true",
        help="Also capture the native flight trajectory; requires fusion's source recordings.",
    )
    args = parser.parse_args()
    if (args.camera_eye is None) != (args.camera_lookat is None):
        parser.error("Supply both --camera-eye and --camera-lookat for a custom view")
    if args.include_layout and args.fusion is None:
        parser.error("--include-layout requires --fusion")
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from visualizer.src.app.lifecycle import shutdown_visualizer
    from visualizer.src.types.camera_state import CameraState
    from visualizer.visualizer import OrchavVisualizer, ProgressReporter

    app = QApplication.instance() or QApplication([])

    def wait_until(predicate, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.02)
        raise RuntimeError("Timed out loading the saved sensing view")

    def settle():
        deadline = time.monotonic() + 1.2
        wait_until(lambda: time.monotonic() >= deadline, timeout=5)

    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    viz = OrchavVisualizer(
        progress=ProgressReporter(enabled=False), renderer_type="pygfx", viewport_mode="detached"
    )
    viz.resize(1280, 1000)
    viz.show()
    app.processEvents()
    viz._deferred_init()
    viz._cli_driven_frame_run = True
    viz.set_background_update_enabled(False)
    viz.use_preload_mode = False
    viz.cancel_startup_preload()
    viz.cancel_scheduled_update()
    captures = []

    def capture(name):
        if args.camera_eye is not None:
            viz.renderer.set_camera_state(
                CameraState(
                    eye=tuple(args.camera_eye),
                    lookat=tuple(args.camera_lookat),
                    up=(0, 0, 1),
                    fov_deg=45,
                )
            )
        settle()
        if not viz.renderer.export_screenshot(str(directory / name), include_hud=True):
            raise RuntimeError(f"Could not export {name}")
        captures.append(name)

    try:
        if args.volume:
            opened = viz.open_scenario(str(args.volume.resolve()), autorun_initial_frame=False)
            if not opened.succeeded:
                raise RuntimeError(str(opened))
            wait_until(lambda: viz.sensing_volume_window.evaluation_controls.ready)
            window = viz.sensing_volume_window
        else:
            from visualizer.src.sensing.fusion_window import open_fusion_window

            window = open_fusion_window(viz, args.fusion.resolve())
            wait_until(lambda: window.reader is not None)
            index = len(window.reader.times) - 1 if args.epoch == -1 else args.epoch
            if not 0 <= index < len(window.reader.times):
                raise ValueError("Fusion epoch is outside the saved recording")
            window.timeline.setValue(index)
            window.visibility["truth"].setChecked(True)
        # The detached renderer owns its top-level Qt window. Resizing through
        # Qt keeps the canvas and camera aspect ratio in sync without backend internals.
        for widget in app.topLevelWidgets():
            if widget is not viz and widget.isVisible():
                widget.resize(1200, 800)
        settle()
        from visualizer.src.types.label_style import LabelCalloutStyle

        viz.ui_controller.handle_label_callout_changed(LabelCalloutStyle(mode="callout"))
        if args.volume:
            viz.set_state(node_label_mode="sensing")
            viz.node_service.recreate_tx_rx_labels(viz.label_font_size)
            window.cell_edges.setChecked(True)
            window.quantity.setCurrentIndex(window.quantity.findData("outcomes"))
            window.mode.setCurrentIndex(window.mode.findData("both"))
            window._prepare_camera()
            viz.renderer.set_camera_state(
                CameraState(
                    eye=(10, -13, 10),
                    lookat=(0, 0, 2.7),
                    up=(0, 0, 1),
                    fov_deg=38,
                )
            )
            capture("outcomes_3d.png")
            window.cutaway.axis.setCurrentIndex(window.cutaway.axis.findData("y"))
            window.cutaway.side.setCurrentIndex(window.cutaway.side.findData(False))
            window.cutaway.boundary.setCurrentIndex(window.cutaway.boundary.count() // 2)
            window.cutaway.active.setChecked(True)
            capture("cutaway_3d.png")
            if window.reader.is_fusion:
                window.quantity.setCurrentIndex(window.quantity.findData("contributions"))
                capture("contributions_3d.png")
            identity = window.reader.identity
        else:
            import numpy as np

            viz.ui_controller.handle_mpc_layer_toggled(False)
            # Fusion supplies both radar labels, including the sensor outside
            # the native recording. Hide its duplicate TX/RX scene labels.
            viz.ui_controller.handle_labels_toggled(False)
            viz.renderer.set_ground_grid_visible(True)
            settle()
            points = [s["sensor"]["position_m"] for s in window.reader["sources"].values()]
            points += [
                t["position_m"]
                for epoch in window.reader["histories"]["fused"]
                for t in epoch["tracks"]
            ]
            points = np.asarray(points)
            centre = (points.min(axis=0) + points.max(axis=0)) / 2
            radius = max(float(np.ptp(points, axis=0).max()), 1)
            tools = viz.camera_controller.presentation()
            tools.prepare()
            tools.free_camera()
            viz.renderer.set_camera_state(
                CameraState(
                    eye=tuple(centre + np.array([0.65, -0.8, 0.6]) * radius),
                    lookat=tuple(centre),
                    up=(0, 0, 1),
                    fov_deg=45,
                )
            )
            for name in window.reader["sources"]:
                window.visibility[name].setChecked(False)
            for mode in window.reader["histories"]:
                for key, control in window.visibility.items():
                    if key == "fused" or key.startswith("tracks:"):
                        control.setChecked(
                            key == ("fused" if mode == "fused" else f"tracks:{mode}")
                        )
                capture("fusion_3d.png" if mode == "fused" else f"{mode.lower()}_3d.png")
            if args.include_layout:
                # An ordinary scene view introduces the flight before its estimates.
                for key, control in window.visibility.items():
                    if key == "fused" or key.startswith("tracks:"):
                        control.setChecked(False)
                viz.ui_controller.handle_target_trajectory_toggled(True)
                coordinator = viz.ui_controller.trajectory_load_coordinator
                wait_until(lambda: not coordinator.is_loading and coordinator.snapshot is not None)
                capture("layout_3d.png")
                viz.ui_controller.handle_target_trajectory_toggled(False)
                window.visibility["fused"].setChecked(True)
            settle()
            window.comparison.grab().save(str(directory / "comparison.png"))
            captures.append("comparison.png")
            identity = window.reader.inspection_id
        (directory / "capture.json").write_text(
            json.dumps(
                dict(
                    source=str((args.volume or args.fusion).resolve()),
                    identity=identity,
                    renderer="pygfx",
                    captures=captures,
                    camera=viz.renderer.get_camera_state().to_dict(),
                ),
                indent=2,
            )
        )
        print(f"Visualizer figures: {directory}", flush=True)  # noqa: T201 - CLI output
    finally:
        volume = getattr(viz, "sensing_volume_window", None)
        if volume is not None and not volume.closed:
            volume.close()
        fusion = getattr(viz, "fusion_window", None)
        if fusion is not None and not fusion.closed:
            fusion.close()
        shutdown_visualizer(viz, persist_state=False)
        viz.close()
        app.processEvents()


if __name__ == "__main__":
    main()
    # Resources and files are closed above. PySide/pyqtgraph can otherwise
    # double-delete graphics objects during interpreter shutdown on headless hosts.
    import pyqtgraph as pg

    pg.exit()
