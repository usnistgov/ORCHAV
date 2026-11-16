"""Export stills and a camera sequence from the normal ORCHAV scene.

All target materials, tracks and labels come from the scenario and native UI.
The helper only selects camera poses, steps saved frames and exports images.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

CAMERAS = {
    "hero": ((-6, -129, 16), (-38, -55, 9), 60),
    "city": ((-5, -142, 42), (-36, -59, 10), 56),
    "facade": ((-15, -112, 22), (-36, -45, 14), 58),
    "street": ((-14, -98, 6), (-36, -45, 11), 65),
}


def select_capture_adapter(adapters, index):
    """Select a zero-based NVIDIA/Vulkan adapter for the optional capture workflow."""
    matching = [
        adapter
        for adapter in adapters
        if str(adapter.info.get("vendor", "")).casefold() == "nvidia"
        and str(adapter.info.get("backend_type", "")).casefold() == "vulkan"
    ]
    if not matching:
        raise RuntimeError(
            "No NVIDIA Vulkan adapter was detected for Paris capture. "
            "Check the GPU driver, Vulkan support, and active graphical session."
        )
    if not 0 <= index < len(matching):
        raise ValueError(
            f"--gpu {index} is unavailable: found {len(matching)} NVIDIA Vulkan "
            f"adapter(s). Select an index from 0 to {len(matching) - 1}."
        )
    return matching[index]


def capture(args):
    import pygfx
    import wgpu
    from PySide6.QtWidgets import QApplication

    from visualizer.src.app.lifecycle import shutdown_visualizer
    from visualizer.src.sensing.runtime import _annotation_adapter, state_value, update_state
    from visualizer.src.types.camera_state import CameraState
    from visualizer.visualizer import OrchavVisualizer, ProgressReporter

    adapter = select_capture_adapter(wgpu.gpu.enumerate_adapters_sync(), args.gpu)
    pygfx.renderers.wgpu.select_adapter(adapter)
    app = QApplication([])
    viz = OrchavVisualizer(
        progress=ProgressReporter(enabled=False), renderer_type="pygfx", viewport_mode="embedded"
    )
    viz.resize(args.width + 300, args.height + 100)
    viz.show()
    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(
        renderer="ORCHAV pygfx",
        scenario=str(args.scenario.resolve()),
        overlays="native saved tracks; shared Card labels",
        views={},
        sequence=[],
    )

    def events(seconds=0.2):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            app.processEvents()
            time.sleep(0.01)

    def step(frame):
        viz.force_update_next_frame = True
        if not viz.update_frame(frame):
            raise RuntimeError(f"Cannot load saved scene frame {frame}")
        events(0.06)
        service = viz.extension_services.get("sensing")
        waited = False
        deadline = time.monotonic() + 120
        while getattr(service, "preparation_pending", False):
            if time.monotonic() > deadline:
                raise RuntimeError("Native sensing preparation did not finish")
            events(0.1)
            waited = True
        if waited:
            viz.force_update_next_frame = True
            if not viz.update_frame(frame):
                raise RuntimeError(f"Cannot refresh scene frame {frame}")
            events(0.1)
        if state_value(viz.app_state, "show_detections"):
            displayed = _annotation_adapter(viz.renderer)._selection_frame
            provenance = displayed.get("_sensing_display_provenance", {})
            scene_time = displayed.get("scene_time_s", provenance.get("current_scene_time_s"))
            result_time = displayed.get("result_time_s")
            if result_time is None:
                result_time = (
                    displayed.get("sensing", {})
                    .get("temporal", {})
                    .get("result", {})
                    .get("reference_time_s")
                )
            if scene_time is None or not np.isclose(scene_time, frame / 2000):
                raise RuntimeError(f"Sensing overlay is not bound to scene frame {frame}")
            if result_time is None or not np.isclose(result_time, (frame // 128) * 0.064):
                raise RuntimeError(f"Sensing overlay uses the wrong CPI at scene frame {frame}")
            return dict(
                scene_frame_id=frame,
                scene_time_s=scene_time,
                result_time_s=result_time,
                track_ids=[d.get("track_id") for d in displayed["sensing"]["detections"]],
            )

    def save(path):
        if not viz.renderer.export_screenshot(str(path)):
            raise RuntimeError(f"Cannot export {path}")

    try:
        events()
        viz._deferred_init(pending_camera=None)
        viz._cli_driven_frame_run = True
        viz.set_background_update_enabled(False)
        viz.use_preload_mode = False
        viz.cancel_startup_preload()
        viz.cancel_scheduled_update()
        outcome = viz.open_scenario(str(args.scenario), autorun_initial_frame=False)
        if not outcome.succeeded:
            raise RuntimeError(str(outcome))
        source = viz.frame_source
        expected = {"Car_Inbound", "Car_Outbound", "Scooter", "Drone", "Pedestrian"}
        loaded = {entry["target_name"] for entry in viz.target_entries}
        if loaded != expected:
            raise RuntimeError(f"Scenario actor meshes are incomplete: {loaded}")
        r = viz.renderer
        r._canvas.setFixedSize(args.width, args.height)
        r.resize(args.width, args.height)
        events(1)
        # Only deterministic image capture disables mouse gestures. Interactive
        # launches use python -m visualizer and retain all normal controls.
        camera_tools = getattr(r, "camera_tools", None)
        if camera_tools is not None:
            camera_tools.stop()
            camera_tools.smooth = False
        r._controller.enabled = False
        r.cancel_camera_gestures()
        r._camera.zoom = 1.0
        step(args.frame)
        events(1)
        states = {t["name"]: t for t in source.load_frame(args.frame).targets_metadata}
        cameras = dict(CAMERAS)
        for name, offset in (
            ("Car_Inbound", (6, -7, 3)),
            ("Car_Outbound", (-5, 7, 2.5)),
            ("Scooter", (4, -5, 2)),
            ("Drone", (4, -7, 2)),
        ):
            point = np.array(states[name]["current_position"])
            if name.startswith("Car_"):
                entry = next(t for t in viz.target_entries if t["target_name"] == name)
                rotation = np.asarray(r._objects[entry["geometry_name"]].world.matrix)[:3, :3]
                offset = rotation @ np.array(offset)
            cameras[name] = (tuple(point + offset), tuple(point), 40)
        root = states["Pedestrian"]["animated_asset"]["state"]["sample"]["root"][
            "resolved_affine_4x4"
        ]
        point = np.asarray(root)[:3, 3] + [0, 0, 0.85]
        cameras["Pedestrian"] = (tuple(point + [3, -4, 1]), tuple(point), 35)
        for name, (eye, lookat, fov) in cameras.items():
            update_state(viz, show_detections=name in CAMERAS)
            step(args.frame)
            camera = CameraState(eye, lookat, (0, 0, 1), fov)
            r.set_camera_state(camera)
            r.refocus_shadows()
            r.set_shadow_focus_radius(75 if name in CAMERAS else 7)
            events(0.3)
            save(args.output / f"{name}.png")
            record["views"][name] = camera.to_dict()
            print("Captured", name, flush=True)
        if args.movie_frames:
            update_state(viz, show_detections=True)
            r.set_shadow_focus_radius(75)
            r.set_camera_state(CameraState(*CAMERAS["hero"][:2], (0, 0, 1), CAMERAS["hero"][2]))
            r.refocus_shadows()
            directory = args.output / "sequence"
            directory.mkdir(exist_ok=True)
            steps = [f for f in source.list_frames() if f % args.movie_step == 0]
            if args.limit:
                steps = steps[: args.limit]
            for i, frame in enumerate(steps):
                overlay = step(frame)
                fraction = i / max(len(steps) - 1, 1)
                eye = np.array(CAMERAS["hero"][0]) + fraction * np.array([-1.5, 2, 0.5])
                camera = CameraState(tuple(eye), CAMERAS["hero"][1], (0, 0, 1), 60)
                r.set_camera_state(camera)
                r._camera.zoom = 1.0
                events(0.06)
                save(directory / f"{i:04d}.png")
                record["sequence"].append(
                    dict(
                        index=i,
                        scene_frame=frame,
                        time_s=frame / 2000,
                        camera=r.get_camera_state().to_dict(),
                        zoom=float(r._camera.zoom),
                        sensing_overlay=overlay,
                    )
                )
                if i % 10 == 0:
                    (args.output / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
                    print(f"Movie {i+1}/{len(steps)}", flush=True)
        record["frame"] = args.frame
        (args.output / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    finally:
        shutdown_visualizer(viz, persist_state=False)
        viz.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Zero-based index among NVIDIA Vulkan adapters (default: 0)",
    )
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--movie-frames", action="store_true")
    parser.add_argument("--movie-step", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="orchav-native-capture-") as settings:
        os.environ.update(XDG_CONFIG_HOME=settings, ORCHAV_IBL_CACHE_DIR=settings + "/ibl")
        capture(args)
