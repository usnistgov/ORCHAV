"""Controlled native volume inspection: production renderer and production panels.

Synthetic data only. This harness never runs ray tracing or detection.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import imageio.v3 as iio
import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QHBoxLayout, QScrollArea, QWidget

from visualizer.src.coverage.volume_window import CoverageVolumeWindow
from visualizer.src.model import RenderObject, Transform
from visualizer.src.renderers.pygfx import PygfxRenderer
from visualizer.src.sensing.volume_window import SensingVolumeWindow
from visualizer.src.types.camera_state import CameraState
from visualizer.src.types.render_payloads import MaterialPayload, MeshPayload


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    axes = {"x": np.arange(10, 22, 2), "y": np.arange(-18, -3, 3), "z": np.arange(7, 23, 4)}
    z, y, x = np.indices((4, 5, 6))
    db = np.asarray(-110 + 4 * x + 3 * y + 5 * z, np.float32)
    db[0, 2, 2], db[3, 2, 2] = -10, -20
    db[0, 0, 0], db[0, 0, 1] = -np.inf, np.nan
    gains = np.power(10.0, db / 10).astype(np.float32)
    path = root / "coverage.h5"
    with h5py.File(path, "w") as h:
        h.attrs["coverage_schema_version"] = 2
        h.attrs["coverage_storage_layout"] = "canonical_derived_v2"
        h.attrs["fixture"] = "controlled synthetic, no propagation solver"
        h.create_dataset("values/path_gain_linear", data=gains[None, :, None])
        h.create_dataset("grid/shape_xyz", data=[6, 5, 4])
        h.create_dataset("grid/origin", data=[9, -19.5, 7])
        h.create_dataset("grid/spacing_xy", data=[2, 3])
        h.create_dataset("grid/heights_m", data=axes["z"])
        h.create_dataset("tx/names", data=[b"FixtureTX"])
        h.create_dataset("tx/powers_dbm", data=[20])
        codes = np.where(np.isnan(db), 0, np.where(np.isneginf(db), 2, 1)).astype(np.uint8)
        ds = h.create_dataset("availability/path_gain", data=codes[None, :, None])
        ds.attrs["schema_version"] = 1
    scene = root / "scenario.yaml"
    scene.write_text(
        "schema_version: 2\nscene: {source: local, id: scene.xml}\ntimeline: {steps: 1, duration_s: 0.0}\nraytracing: {enabled: false}\nactors: {tx: [{name: FixtureTX, mobility: {type: stationary, position_m: [7, -12, 4]}}]}\n"
    )
    (root / "scene.xml").write_text(
        '<scene version="3.0.0"><shape type="cube"><transform name="to_world"><scale x="8" y="9" z="0.25"/><translate x="15" y="-12" z="0"/></transform><bsdf type="diffuse"><rgb name="reflectance" value="0.5 0.5 0.5"/></bsdf></shape></scene>'
    )
    trials = []
    for iz, iy, ix in np.ndindex((4, 5, 6)):
        pos = [float(axes[n][i]) for n, i in zip("xyz", (ix, iy, iz), strict=True)]
        status = (
            "excluded_geometry"
            if (iz, iy, ix) == (0, 0, 1)
            else "pending" if (iz, iy, ix) == (0, 0, 0) else "measured"
        )
        hit = (ix + iy + iz) % 3 != 0
        row = dict(
            id=f"fixture-z{iz}-y{iy}-x{ix}",
            index_zyx=[iz, iy, ix],
            heading=0,
            position_m=pos,
            velocity_m_s=[1, 0, 0],
            status=status,
        )
        if status == "measured":
            report = dict(
                detection_id="controlled-report",
                position_m=pos,
                range_m=float(np.linalg.norm(np.subtract(pos, [7, -12, 4]))),
                closing_velocity_m_s=0.0,
            )
            delta = np.subtract(pos, [7, -12, 4])
            row.update(
                hit=hit,
                tp=int(hit),
                fp=0,
                fn=int(not hit),
                matched_error_m=0 if hit else None,
                radial_closing_velocity_m_s=-float(delta[0] / np.linalg.norm(delta)),
                native=f"not-retained/{row['id']}",
                native_evidence="not_included",
                source_hashes={
                    "controlled_fixture": hashlib.sha256(row["id"].encode()).hexdigest()
                },
                reports=[report] if hit else [],
            )
        trials.append(row)
    manifest = dict(
        schema_version=1,
        kind="conditional_detection_outcomes",
        fixture="Controlled synthetic fixture; no RT or detector was run",
        axes_m={n: a.tolist() for n, a in axes.items()},
        world_frame=dict(
            id="controlled-translated",
            origin="authored fixture",
            axes=list("xyz"),
            handedness="right",
            length_unit="m",
        ),
        headings=[dict(name="+X controlled translation", velocity_m_s=[1, 0, 0])],
        sensor_position_m=[7, -12, 4],
        sensor_id="FixtureTX/RX",
        position_gate_m=1,
        scene_scenario="scenario.yaml",
        scene_configuration_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
        trials=trials,
    )
    (root / "sensing_volume.json").write_text(json.dumps(manifest, indent=2))
    return path


class NativeFixtureHost(QWidget):
    def __init__(self, root, kind):
        super().__init__()
        self.root = root
        self.setWindowTitle("ORCHAV · controlled native volume inspection (synthetic)")
        self.resize(1640, 1100)
        self.app_state = SimpleNamespace(show_coverage=False, label_callout=None)
        self.ui_controller = SimpleNamespace(
            handle_coverage_toggled=lambda value: None,
            handle_label_callout_changed=lambda value: None,
        )
        self.current_scenario_path = root / "scenario.yaml"
        self.coverage_data = dict(
            coverage_file=str(root / "coverage.h5"),
            tx_names=["FixtureTX"],
            tx_positions=[[7, -12, 4]],
        )
        layout = QHBoxLayout(self)
        viewport = QWidget()
        viewport.setMinimumWidth(940)
        layout.addWidget(viewport, 3)
        self.renderer = PygfxRenderer(self)
        self.renderer.initialize_authoring_viewport(viewport, width=1000, height=1000)
        self.renderer._deferred_default_ibl_name = None
        self.renderer._controller.register_events(self.renderer._renderer)
        self.add_context()
        self.panel = (
            CoverageVolumeWindow(self)
            if kind == "coverage"
            else SensingVolumeWindow(self, root / "sensing_volume.json")
        )
        self.panel.setWindowFlags(Qt.Widget)
        layout.addWidget(self.panel, 2)
        self.panel.show()

    def open_scenario(self, path, **kwargs):
        # This harness implements only its declared scene, already rendered.
        return SimpleNamespace(
            succeeded=Path(path).resolve() == self.current_scenario_path.resolve()
        )

    def add_context(self):
        gfx = self.renderer._gfx
        for name, size, centre, color in [
            ("floor", (17, 19, 0.5), (15, -12, 0), (0.4, 0.4, 0.4, 1)),
            ("FixtureTX::pole", (0.13, 0.13, 4), (7, -12, 2), (0.01, 0.01, 0.01, 1)),
        ]:
            geom = gfx.box_geometry(*size)
            obj = RenderObject(
                id=name,
                payload=MeshPayload(
                    vertices=geom.positions.data, triangles=geom.indices.data.reshape(-1, 3)
                ),
                material=MaterialPayload(base_color=color),
                transform=Transform.from_translation(centre),
            )
            self.renderer.ensure_object(obj)


def wait(app, predicate, seconds=25):
    until = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > until:
            raise RuntimeError("Timed out waiting for native fixture")
        app.processEvents()
        time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("coverage", "sensing"), default="coverage")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Capture old controls against a pristine main checkout on PYTHONPATH",
    )
    args = parser.parse_args()
    prepare(args.output)
    source_hashes = {
        name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
        for name in ("coverage.h5", "sensing_volume.json")
    }
    app = QApplication.instance() or QApplication([])
    host = NativeFixtureHost(args.output, args.kind)
    host.show()
    p, r = host.panel, host.renderer
    wait(
        app, lambda: p.model is not None if args.kind == "coverage" else p.evaluation_controls.ready
    )
    if args.kind == "coverage":
        p.high.setValue(-50)
        p.mode.setCurrentIndex(p.mode.findData("mip"))
        centre = np.array([14, -12, 13])
    else:
        p.mode.setCurrentIndex(p.mode.findData("both"))
        p.uniform_colour.setChecked(True)
        centre = np.array([14, -12, 13])
    r.set_camera_state(CameraState((14, -44, 40), tuple(centre), (0, 0, 1), 60))
    for _ in range(20):
        r._canvas.force_draw()
        app.processEvents()
    if not args.capture:
        app.exec()
        return
    if args.baseline:
        # The pristine panel has no direct picker. Select the same source cell
        # with its existing controls for an honest like-for-like UI comparison.
        if args.kind == "sensing":
            from PySide6.QtCore import QSignalBlocker

            for widget, index in ((p.x, 2), (p.y, 0), (p.altitude, 3)):
                with QSignalBlocker(widget):
                    widget.setCurrentIndex(index)
            p.inspect.setChecked(True)
            p.inspection_controls.slice_plot.setChecked(True)
            p._draw()
        else:
            p.coordinate.setCurrentIndex(3)
        for _ in range(20):
            r._canvas.force_draw()
            app.processEvents()
        iio.imwrite(args.output / f"{args.kind}_baseline_native.png", r._renderer.snapshot())
        scroll = p.findChild(QScrollArea)
        if args.kind == "sensing":
            scroll.verticalScrollBar().setValue(
                p.cell_note.mapTo(scroll.widget(), QPoint(0, 0)).y()
            )
            app.processEvents()
        host.grab().save(str(args.output / f"{args.kind}_baseline_ui.png"))
        (args.output / f"{args.kind}_baseline.json").write_text(
            json.dumps(
                dict(
                    baseline=True,
                    synthetic=True,
                    camera=r.get_camera_state().to_dict(),
                    manual_source_index_xyz=[2, 0, 3] if args.kind == "sensing" else None,
                ),
                indent=2,
            )
        )
        p.close()
        r.close()
        host.close()
        app.processEvents()
        return
    # Search actual visible native pick buffer; no synthetic selection indices.
    w, h = r._canvas.get_logical_size()
    selected = None
    for yy in sorted(np.linspace(h * 0.35, h * 0.65, 12), key=lambda value: abs(value - h / 2)):
        for xx in sorted(np.linspace(w * 0.35, w * 0.65, 12), key=lambda value: abs(value - w / 2)):
            info = r._renderer.get_pick_info((xx, yy))
            obj = info.get("world_object")
            if obj is not None and hasattr(obj, "_orchav_volume_payload"):
                selected = xx, yy, info
                break
        if selected:
            break
    if selected is None:
        raise RuntimeError("No visible native volume hit")
    x, y, info = selected
    import pygfx as gfx

    previous = p.selected_indices if args.kind == "coverage" else p._cell_key()
    camera_before_drag = r.get_camera_state()
    # Native root/controller dispatch must reject a camera drag, then accept a click.
    for kind, xx, buttons in (
        ("pointer_down", x, [1]),
        ("pointer_move", x + 20, [1]),
        ("pointer_up", x + 20, []),
    ):
        event = gfx.PointerEvent(
            kind,
            x=xx,
            y=y,
            button=1,
            buttons=buttons,
            modifiers=[],
            pick_info=info,
            target=info["world_object"],
            root=r._renderer,
        )
        r._renderer.dispatch_event(event)
    assert (p.selected_indices if args.kind == "coverage" else p._cell_key()) == previous
    r._controller.enabled = False
    r.set_camera_state(camera_before_drag)
    r._controller.enabled = True
    for _ in range(10):
        r._canvas.force_draw()
        app.processEvents()
    info = r._renderer.get_pick_info((x, y))
    if not hasattr(info.get("world_object"), "_orchav_volume_payload"):
        raise RuntimeError("The drag moved the test point off the volume; choose a larger fixture")
    for kind, buttons in (("pointer_down", [1]), ("pointer_up", [])):
        event = gfx.PointerEvent(
            kind,
            x=x,
            y=y,
            button=1,
            buttons=buttons,
            modifiers=[],
            pick_info=info,
            target=info["world_object"],
            root=r._renderer,
        )
        r._renderer.dispatch_event(event)
    assert (
        p.selected_indices if args.kind == "coverage" else tuple(reversed(p._cell_key()[1:]))
    ) == tuple(info["index"])
    for _ in range(20):
        r._canvas.force_draw()
        app.processEvents()
    iio.imwrite(args.output / f"{args.kind}_native.png", r._renderer.snapshot())
    scroll = p.findChild(QScrollArea)
    target = p.cell_note
    app.processEvents()
    scroll.verticalScrollBar().setValue(target.mapTo(scroll.widget(), QPoint(0, 0)).y())
    app.processEvents()
    host.grab().save(str(args.output / f"{args.kind}_selected_ui.png"))
    slice_view = p.slice_view if args.kind == "coverage" else p.inspection_controls.slice_view
    slice_view.grab().save(str(args.output / f"{args.kind}_linked_slices.png"))
    evidence = dict(
        kind=args.kind,
        index_xyz=info["index"],
        selected_text=p.cell_note.text(),
        renderer="production PygfxRenderer",
        panel=type(p).__name__,
        synthetic=True,
        native_root_drag_rejected=True,
        native_root_click_selected=True,
        camera=r.get_camera_state().to_dict(),
    )
    (args.output / f"{args.kind}_selection.json").write_text(json.dumps(evidence, indent=2))
    # Exercise unavailable source cells through the same linked-slice selection flow.
    select_slice = p._select_slice if args.kind == "coverage" else p.workflow.select_slice
    select_slice("xz", 12, 7)
    select_slice("xy", 12, -18)
    for _ in range(10):
        r._canvas.force_draw()
        app.processEvents()
    scroll.verticalScrollBar().setValue(target.mapTo(scroll.widget(), QPoint(0, 0)).y())
    app.processEvents()
    host.grab().save(str(args.output / f"{args.kind}_unavailable_ui.png"))
    evidence["unavailable_cell_text"] = p.cell_note.text()
    evidence["source_bytes_unchanged"] = all(
        hashlib.sha256((args.output / name).read_bytes()).hexdigest() == digest
        for name, digest in source_hashes.items()
    )
    assert evidence["source_bytes_unchanged"]
    (args.output / f"{args.kind}_selection.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
    p.close()
    r.close()
    host.close()
    app.processEvents()


if __name__ == "__main__":
    main()
