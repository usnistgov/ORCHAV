"""Matched native figure exports using ORCHAV's controlled PBR reference scene.

The before image calls the actual main-branch capture implementation, supplied
with --baseline-root. Both captures share identical scene, camera, materials,
lighting and fixed logical annotation layout. Only export resolution handling
changes. Generated figures stay outside the checkout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root.resolve()))
    os.environ["ORCHAV_ENABLE_TEXTURES"] = "1"
    os.environ.pop("ORCHAV_DISABLE_TEXTURES", None)

    import pygfx as gfx
    import wgpu
    from PIL import Image
    from rendercanvas.offscreen import RenderCanvas

    from visualizer.src.renderers.pygfx.callouts import sync_callout
    from visualizer.src.renderers.pygfx.capture import PygfxCaptureMixin
    from visualizer.src.renderers.pygfx.labels import PygfxLabelMixin

    reference = runpy.run_path(str(Path(__file__).with_name("pygfx_pbr_reference.py")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene, camera, manager = reference["build_scene"](args.output_dir / "inputs")
    size = reference["SIZE"]

    class Owner(PygfxCaptureMixin, PygfxLabelMixin):
        pass

    owner = Owner()
    owner._gfx = gfx
    owner._initialized = True
    owner._canvas = RenderCanvas(size=size, pixel_ratio=1)
    owner._renderer = gfx.WgpuRenderer(owner._canvas, pixel_scale=1, ppaa="fxaa")
    owner._scene, owner._camera = scene, camera
    owner._minimap_enabled = False
    owner._objects = {}
    # Use real ORCHAV callout helpers at the two native actor positions.
    for x, name, background in ((-7, "Tx", (1, 1, 1, 0.95)), (7, "Rx", (0.7, 0.82, 1, 0.95))):
        for child in tuple(scene.children):
            if isinstance(child, gfx.Text) and np.allclose(child.local.position, (x, 4.2, 3.2)):
                scene.remove(child)
        origin = (x, 4.2, 2.6)
        label = gfx.Text(
            text=name,
            font_size=20,
            screen_space=True,
            material=gfx.TextMaterial(
                color="black", aa=True, alpha_mode="blend", depth_test=False, depth_write=False
            ),
            render_order=31000,
        )
        label.local.position = origin
        scene.add(label)
        owner._objects[name] = label
        owner._sync_label_background(name, background, connector_origin=origin)
        sync_callout(
            owner,
            name,
            dict(
                base_position=origin,
                offset_x=32,
                offset_y=42,
                marker_size=18,
                on_top=True,
                leader=True,
            ),
            {"label_background_rgba": background},
        )

    baseline_path = args.baseline_root / "visualizer/src/renderers/pygfx/capture.py"
    spec = importlib.util.spec_from_file_location("baseline_figure_capture", baseline_path)
    baseline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline_module)
    original = baseline_module.PygfxCaptureMixin
    try:
        native = owner.export_screenshot_to_array()
        before_state = camera.get_state()
        before_view_size = camera._view_size
        before_canvas = owner._canvas.get_physical_size()
        started = perf_counter()
        before = original.export_screenshot_to_array(owner, args.scale)
        before_seconds = perf_counter() - started
        started = perf_counter()
        after = owner.export_screenshot_to_array(args.scale)
        after_seconds = perf_counter() - started
        for key, value in before_state.items():
            np.testing.assert_equal(camera.get_state()[key], value)
        assert camera._view_size == before_view_size
        assert owner._canvas.get_physical_size() == before_canvas
        assert before.shape == after.shape
        Image.fromarray(native).save(args.output_dir / "native_viewport.png")
        Image.fromarray(before).save(args.output_dir / "before_bilinear_export.png")
        Image.fromarray(after).save(args.output_dir / "after_native_export.png")
        # Logical image bounds, preserved as metadata for a reproducible crop.
        crop = (300, 130, 640, 260)
        physical_crop = tuple(round(value * args.scale) for value in crop)
        Image.fromarray(before).crop(physical_crop).save(args.output_dir / "before_detail_crop.png")
        Image.fromarray(after).crop(physical_crop).save(args.output_dir / "after_detail_crop.png")
        delta = np.abs(before.astype(np.float32) - after.astype(np.float32))
        manifest = dict(
            scope="Actual ORCHAV capture code and native helpers in controlled backend scene; not a full GUI screenshot",
            baseline_capture_source=str(baseline_path.resolve()),
            baseline_commit="bc3680ded0d6123c2d808aceab0f76baa5fbcb80",
            figure_source=str(Path(__file__).resolve()),
            logical_size=size,
            output_size=[after.shape[1], after.shape[0]],
            resolution_scale=args.scale,
            crop_logical=crop,
            crop_physical=physical_crop,
            before_method="Unmodified main-branch capture.py: current frame then Pillow bilinear resize",
            after_method="Current camera rendered natively; automatic padded tiles above 4096-pixel edge budget",
            before_seconds=before_seconds,
            after_seconds=after_seconds,
            pixel_difference_mean=float(delta.mean()),
            changed_channel_values=int(np.count_nonzero(delta)),
            camera_state_restored=True,
            canvas_size_unchanged=True,
            camera_position=reference["CAMERA_POSITION"],
            camera_target=reference["CAMERA_TARGET"],
            background_srgb=[0.2, 0.2, 0.2],
            ppaa="fxaa",
            ibl_intensity=0.35,
            adapter=dict(owner._renderer.device.adapter.info),
            pygfx=gfx.__version__,
            wgpu=wgpu.__version__,
            software_rendering=True,
        )
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2))
    finally:
        owner._canvas.close()
        del manager


if __name__ == "__main__":
    main()
