"""Exact grid placement, texture reuse and real Qt deferred cleanup."""

from dataclasses import replace
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.sensing.volume_presentation import volume_template
from visualizer.src.types.render_payloads import VolumePayload


def test_zero_radial_control_explains_wall_volume_and_updates_for_heading(qtbot):
    from visualizer.src.sensing.volume_evidence import geometry_objects
    from visualizer.src.sensing.volume_inspection_controls import InspectionControls

    # The wall example samples in front of X=-6; its +X motion never has zero
    # direct radial velocity there. Changing heading can bring the plane inside.
    axes = {
        "x": np.arange(-3, 5.1, 0.5),
        "y": np.arange(-4, 4.1, 0.5),
        "z": np.arange(0.5, 6.6, 0.5),
    }
    reader = SimpleNamespace(
        axes=axes,
        metadata={"sensor_position_m": [-6, 0, 2]},
        headings=[{"velocity_m_s": v} for v in ([1, 0, 0], [0, 1, 0], [0, 0, 0])],
    )
    controls = InspectionControls()
    qtbot.addWidget(controls)
    controls.restore({"zero": True})
    for heading in (0, 1, 2, 1):
        controls.set_geometry_context(
            reader.metadata["sensor_position_m"],
            reader.headings[heading]["velocity_m_s"],
            axes,
            forward_available=False,
        )
        assert controls.zero.isChecked()  # Preserve the preference across headings.
        assert controls.zero.isEnabled() == (heading == 1)
        assert controls.motion.isEnabled() == (heading != 2)
        assert not controls.forward.isEnabled()
        note = controls.geometry_note.text()
        assert controls.zero.toolTip() in note
        if heading == 0:
            assert "World X = -6 m" in note
            assert "-3.25 to 5.25 m" in note
            assert "nothing to draw" in note
        elif heading == 2:
            assert "zero everywhere; no unique plane" in note
        else:
            assert "World Y = 0 m" in note
            assert "clipped" in note
        objects = geometry_objects(reader, heading, None, show_zero=True)
        assert bool(objects) == (heading == 1)
        if objects:
            np.testing.assert_allclose(objects[0].payload.vertices[:, 1], 0)


def test_zero_radial_oblique_plane_context(qtbot):
    from visualizer.src.sensing.volume_inspection_controls import InspectionControls

    controls = InspectionControls()
    qtbot.addWidget(controls)
    axes = {n: np.array([-1.0, 0, 1]) for n in ("x", "y", "z")}
    controls.set_geometry_context([0, 0, 0], [3, 4, 0], axes, forward_available=True)
    assert controls.zero.isEnabled() and controls.forward.isEnabled()
    assert "normal (0.6, 0.8, 0)" in controls.geometry_note.text()
    controls.set_geometry_context([9, 9, 9], [3, 4, 0], axes, forward_available=True)
    assert not controls.zero.isEnabled()
    assert "nothing to draw" in controls.geometry_note.text()


def test_volume_centres_and_unknown_palette_are_explicit():
    reader = SimpleNamespace(
        identity="test",
        is_fusion=False,
        native_availability=lambda key: "Native evidence not included",
        axes={"x": np.array([10, 20]), "y": np.array([20, 40]), "z": np.array([30, 60])},
        outcome_grid=lambda heading: np.array([[[1, 0], [np.nan, np.nan]]] * 2),
    )
    payload, transform = volume_template(reader, 0)
    np.testing.assert_array_equal(payload.values[0], [[2, 1], [0, 0]])
    np.testing.assert_allclose(transform.matrix @ [1, 1, 1, 1], [20, 40, 60, 1])
    assert payload.palette[0, 3] == 0 and payload.palette[1, 3] == 1
    changed = replace(payload, opacity=0.2, mode="mip")
    assert changed.values is payload.values and changed.palette is payload.palette
    with pytest.raises(ValueError):
        payload.values[0, 0, 0] = 999


@pytest.mark.parametrize("region,expected", [("hit", [1, 0, 0]), ("miss", [0, 1, 0])])
def test_ray_marched_regions_never_fill_unknown_cells(region, expected):
    reader = SimpleNamespace(
        identity="test",
        is_fusion=False,
        native_availability=lambda key: "Native evidence not included",
        axes={name: np.array([10, 20]) for name in ("x", "y", "z")},
        outcome_grid=lambda heading: np.array([[[1, 0, np.nan]]]),
    )
    payload, _ = volume_template(reader, 0, region)
    assert payload.mode == "iso" and payload.threshold == 0.5
    np.testing.assert_array_equal(payload.values[0, 0], expected)


def test_geometric_shadow_uses_geometry_instead_of_detected_outcomes():
    reader = SimpleNamespace(
        identity="test",
        is_fusion=False,
        native_availability=lambda key: "Native evidence not included",
        axes={name: np.array([10, 20]) for name in ("x", "y", "z")},
        outcome_grid=lambda heading: np.array([[[1, 1, 0]]]),
        geometry_grid=lambda heading: np.array([[[1, 0, np.nan]]]),
    )
    payload, _ = volume_template(reader, 0, "blocked")
    np.testing.assert_array_equal(payload.values[0, 0], [1, 0, 0])


def test_native_volume_reuses_texture_and_has_discrete_clamped_palette():
    import pygfx

    from visualizer.src.renderers.pygfx.volume import ensure_volume

    handles = count()
    redraws = []
    renderer = SimpleNamespace(
        _initialized=True,
        _scene=pygfx.Scene(),
        _gfx=pygfx,
        _objects={},
        _kinds={},
        _name_to_handle={},
        _handle_to_name={},
        _reverse_objects={},
        _geometry_upload_center={},
        _allocate_handle=lambda: next(handles),
        request_redraw=lambda: redraws.append(True),
    )
    original = np.ones((2, 2, 2), dtype=np.float32)
    payload = VolumePayload(
        values=original, palette=np.array([[0, 0, 0, 0], [1, 1, 1, 1]]), limits=(0, 1)
    )
    original[:] = 0
    assert ensure_volume(renderer, "volume", payload)
    native = renderer._objects["volume"]
    texture = native.geometry.grid
    np.testing.assert_array_equal(texture.data, 1)
    assert ensure_volume(renderer, "volume", replace(payload, mode="mip", opacity=0.4))
    assert native.geometry.grid is texture
    assert isinstance(native.material, pygfx.VolumeMipMaterial)
    assert native.material.map.wrap_s == "clamp"
    assert native.material.interpolation == "nearest"

    # Fully opaque regions need a depth-writing pass; switching opacity or
    # lighting must leave the original samples and GPU texture unchanged.
    from visualizer.src.renderers.pygfx.cell_volume import CellVolumeMaterial

    assert ensure_volume(
        renderer, "volume", replace(payload, mode="iso", opacity=1, lighting=False)
    )
    assert isinstance(native.material, CellVolumeMaterial)
    assert not native.material.cell_lighting
    np.testing.assert_array_equal(native.material.map.texture.data, payload.palette)
    assert native.material.alpha_mode == "dither"
    assert native.material.depth_write
    assert native.geometry.grid is texture
    assert ensure_volume(
        renderer, "volume", replace(payload, mode="iso", opacity=0.6, lighting=False)
    )
    assert native.material.alpha_mode == "weighted_blend"
    assert not native.material.depth_write
    assert native.geometry.grid is texture
    assert ensure_volume(renderer, "volume", replace(payload, mode="iso", opacity=1, lighting=True))
    assert isinstance(native.material, CellVolumeMaterial)
    assert native.material.cell_lighting
    assert native.material.pick_write
    np.testing.assert_array_equal(native.material.map.texture.data, payload.palette)
    assert native.material.alpha_mode == "dither"
    assert native.geometry.grid is texture

    # Opacity-only changes must repaint an idle canvas without replacing data.
    for mode in ("slice", "mip", "iso"):
        assert ensure_volume(renderer, "volume", replace(payload, mode=mode))
        before = len(redraws)
        assert ensure_volume(renderer, "volume", replace(payload, mode=mode, opacity=0.2))
        assert len(redraws) == before + 1
        assert native.material.opacity == pytest.approx(0.2)
        assert native.material.alpha_mode == "weighted_blend"
        assert not native.material.depth_write
        assert native.geometry.grid is texture
    assert ensure_volume(renderer, "volume", replace(payload, mode="iso", threshold=0.7))
    assert native.geometry.grid is texture
    assert isinstance(native.material, pygfx.VolumeIsoMaterial)
    assert native.material.threshold == pytest.approx(0.7)
    assert native.material.step_size == 1
    assert native.material.shininess == 0
    assert native.material.interpolation == "nearest"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"values": np.ones((2, 2))},
        {"values": np.full((2, 2, 2), np.nan)},
        {"plane": (0, 0, 0, 1)},
        {"mode": "smooth"},
        {"limits": (1, 1)},
        {"threshold": float("nan")},
    ],
)
def test_invalid_volume_payload_is_rejected(kwargs):
    fields = dict(
        values=np.ones((2, 2, 2)), palette=np.array([[0, 0, 0, 0], [1, 1, 1, 1]]), limits=(0, 1)
    )
    with pytest.raises(ValueError):
        VolumePayload(**(fields | kwargs))


def test_volume_window_retires_layers_on_actual_deferred_deletion(qtbot, monkeypatch, tmp_path):
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QWidget
    from shiboken6 import isValid

    from visualizer.src.sensing.volume_window import SensingVolumeWindow

    objects = {}

    def ensure(obj):
        objects[obj.id] = obj
        return True

    def remove(name):
        objects.pop(name, None)
        return True

    scene = tmp_path / "scenario.yaml"
    scene.write_text("{}")
    reader = SimpleNamespace(
        identity="test",
        is_fusion=False,
        native_availability=lambda key: "Native evidence not included",
        axes={"x": np.array([10, 20]), "y": np.array([20, 40]), "z": np.array([30, 60])},
        headings=[{"name": "Test flight", "velocity_m_s": [1, 0, 0]}],
        cells={},
        scene_path=scene,
        metadata={"position_gate_m": 5, "sensor_position_m": [0, 0, 0]},
        outcome_grid=lambda heading: np.array(
            [[[1, 0], [np.nan, 1]], [[0, 1], [0, np.nan]]], dtype=np.float32
        ),
    )
    monkeypatch.setattr(SensingVolumeWindow, "_load", staticmethod(lambda path: reader))
    monkeypatch.setattr(SensingVolumeWindow, "_load_evaluation", lambda self: None)
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.current_scenario_path = scene
    parent.open_scenario = lambda *args, **kwargs: SimpleNamespace(succeeded=True)
    parent.renderer = SimpleNamespace(
        capabilities=RendererCapabilities(),
        ensure_object=ensure,
        remove_object=remove,
        request_redraw=lambda: None,
        set_analysis_legend=lambda legend: None,
        set_overview_camera=lambda *args: None,
    )
    for _ in range(3):
        window = SensingVolumeWindow(parent, "unused")
        window.show()
        qtbot.waitUntil(lambda: window.reader is reader)
        assert "sensing:outcome_volume" in objects
        window.mode.setCurrentIndex(window.mode.findData("both"))
        hit = objects["sensing:outcome_volume"].payload
        miss = objects["sensing:outcome_volume:misses"].payload
        assert hit.mode == miss.mode == "iso"
        np.testing.assert_array_equal(
            hit.values + miss.values, np.isfinite(reader.outcome_grid(0)).astype(np.float32)
        )
        assert not np.any(hit.values * miss.values)
        window.visible.setChecked(False)
        assert not objects["sensing:outcome_volume"].visible
        assert not objects["sensing:outcome_volume:misses"].visible
        window.visible.setChecked(True)
        window.opacity.setValue(0.3)
        assert objects["sensing:outcome_volume"].payload.values is hit.values
        assert objects["sensing:outcome_volume:misses"].payload.values is miss.values
        assert objects["sensing:outcome_volume:misses"].payload.opacity == 0.3
        for display, render_mode in (("hit", "iso"), ("miss", "iso"), ("mip", "mip")):
            window.mode.setCurrentIndex(window.mode.findData(display))
            assert objects["sensing:outcome_volume"].payload.mode == render_mode
            assert "sensing:outcome_volume:misses" not in objects
            assert "Match: position error" in window.display_note.text()
            assert not window.altitude.isEnabled()
            assert window.inspect_section.isHidden()
        window.inspect.setChecked(True)
        assert window.altitude.isEnabled()
        assert not window.altitude.isHidden()
        assert window.altitude_label.text() == "Z"
        window.inspect.setChecked(False)
        window.mode.setCurrentIndex(window.mode.findData("slice"))
        assert not window.slice_coordinate.isHidden()
        assert window.slice_coordinate.property("axis") == "z"
        window.inspect.setChecked(True)
        assert "sensing:volume:tested_label" in objects or not reader.cells
        window.visible.setChecked(False)
        assert not objects["sensing:outcome_volume"].visible
        window.mode.setCurrentIndex(window.mode.findData("both"))
        window.close()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert not isValid(window)
        assert not objects


def test_paired_open_preserves_view_and_position(qtbot, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QWidget

    from visualizer.src.sensing import volume_window as module

    scene = tmp_path / "wall/scenario.yaml"
    scene.parent.mkdir()
    scene.write_text("{}")
    control = tmp_path / "open"
    control.mkdir()
    (control / "sensing_volume.json").write_text("{}")
    reader = SimpleNamespace(
        identity="test",
        is_fusion=False,
        native_availability=lambda key: "Native evidence not included",
        path=scene.parent / "sensing_volume.json",
        scene_path=scene,
        axes={name: np.array([0.0, 0.5]) for name in ("x", "y", "z")},
        headings=[{"name": "+X", "velocity_m_s": [1, 0, 0]}],
        cells={},
        metadata={
            "position_gate_m": 0.75,
            "sensor_position_m": [-6, 0, 2],
            "motion_label": "Target motion",
            "geometry_description": "Centre ray",
            "comparison_volume": "../open",
            "comparison_label": "Open no-wall control",
        },
        outcome_grid=lambda h: np.zeros((2, 2, 2)),
        geometry_grid=lambda h: np.ones((2, 2, 2)),
    )
    monkeypatch.setattr(module.SensingVolumeWindow, "_load", staticmethod(lambda _: reader))
    monkeypatch.setattr(module.SensingVolumeWindow, "_load_evaluation", lambda self: None)
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.current_scenario_path = scene
    parent.open_scenario = lambda *a, **k: SimpleNamespace(succeeded=True)
    camera = object()
    parent.renderer = SimpleNamespace(
        capabilities=RendererCapabilities(),
        ensure_object=lambda o: True,
        remove_object=lambda n: True,
        request_redraw=lambda: None,
        set_analysis_legend=lambda legend: None,
        set_overview_camera=lambda *a: None,
        get_camera_state=lambda: camera,
    )
    opened = []
    monkeypatch.setattr(module, "open_sensing_volume", lambda *a, **kw: opened.append((a, kw)))
    window = module.SensingVolumeWindow(parent, scene.parent)
    window.show()
    qtbot.waitUntil(lambda: window.reader is reader)
    assert window.mode.currentData() == "both"
    assert not window.heading.isEnabled()
    assert window.conditions.count() == 2
    window.inspect.setChecked(True)
    window.x.setCurrentIndex(0)
    window.opacity.setValue(0.35)
    window.cell_edges.setChecked(True)
    window.unavailable_edges.setChecked(True)
    window.quantity.setCurrentIndex(window.quantity.findData("geometry"))
    window.conditions.setCurrentIndex(
        window.conditions.findData(str(control / "sensing_volume.json"))
    )
    args, kwargs = opened[0]
    assert Path(args[1]).resolve() == control / "sensing_volume.json"
    state = kwargs["view_state"]
    assert state["camera"] is camera and state["mode"] == "both" and state["quantity"] == "geometry"
    assert state["x"] == 0 and state["opacity"] == 0.35 and state["inspect"]
    assert state["cell_edges"] and state["unavailable_edges"]
    window.close()


def test_live_gates_recolor_counts_and_keep_no_report_errors_separate(qtbot, monkeypatch, tmp_path):
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QWidget
    from shiboken6 import isValid

    from shared.sensing.volume.evaluation import VolumeEvaluation
    from visualizer.src.sensing import volume_window as module

    scene = tmp_path / "scenario.yaml"
    scene.write_text("{}")
    trials = {
        (0, 0, 0, 0): dict(
            id="a",
            status="measured",
            position_m=[10, 0, 0],
            radial_closing_velocity_m_s=-1,
            tp=1,
            fp=0,
            fn=0,
            hit=True,
        ),
        (0, 0, 0, 1): dict(
            id="b",
            status="measured",
            position_m=[11, 0, 0],
            radial_closing_velocity_m_s=-1,
            tp=0,
            fp=0,
            fn=1,
            hit=False,
        ),
    }
    reader = SimpleNamespace(
        identity="test",
        is_fusion=False,
        native_availability=lambda key: "Native evidence not included",
        path=tmp_path / "sensing_volume.json",
        scene_path=scene,
        axes={"x": np.array([10, 11]), "y": np.array([0, 1]), "z": np.array([0, 1])},
        shape=(2, 2, 2),
        headings=[{"name": "+X", "velocity_m_s": [1, 0, 0]}],
        cells=trials,
        metadata={"position_gate_m": 0.75, "sensor_position_m": [0, 0, 0]},
        outcome_grid=lambda h: np.array(
            [[[1, 0], [np.nan, np.nan]], [[np.nan, np.nan], [np.nan, np.nan]]]
        ),
    )
    evaluation = VolumeEvaluation(
        reader,
        {
            "a": [
                dict(
                    detection_id="a",
                    position_m=[10.5, 0, 0],
                    range_m=10.5,
                    closing_velocity_m_s=-1.2,
                )
            ],
            "b": [],
        },
    )
    monkeypatch.setattr(module.SensingVolumeWindow, "_load", staticmethod(lambda _: reader))
    monkeypatch.setattr(module, "load_evaluation", lambda *a, **kw: evaluation)
    objects = {}

    def ensure(obj):
        objects[obj.id] = obj
        return True

    def remove(name):
        objects.pop(name, None)
        return True

    parent = QWidget()
    qtbot.addWidget(parent)
    parent.current_scenario_path = scene
    parent.open_scenario = lambda *a, **k: SimpleNamespace(succeeded=True)
    parent.renderer = SimpleNamespace(
        capabilities=RendererCapabilities(),
        ensure_object=ensure,
        remove_object=remove,
        request_redraw=lambda: None,
        set_analysis_legend=lambda legend: None,
        set_overview_camera=lambda *a: None,
    )
    window = module.SensingVolumeWindow(parent, scene.parent)
    window.show()
    qtbot.waitUntil(lambda: window.reader is reader)
    controls = window.evaluation_controls
    assert not controls.expand.isChecked()
    assert window.quantity.findData("reports") >= 0
    qtbot.waitUntil(lambda: controls.ready)
    assert "TP 1 / FP 0 / FN 1" in window.status.text()
    controls.position_gate.setValue(0.4)
    assert "TP 0 / FP 1 / FN 2" in window.status.text()
    assert objects["sensing:outcome_volume:misses"].payload.values[0, 0, 0] == 1
    controls.position_gate.setValue(0.5)
    assert "TP 1 / FP 0 / FN 1" in window.status.text()
    controls.basis.setCurrentIndex(controls.basis.findData("range_velocity"))
    controls.velocity_gate.setValue(0.1)
    assert "TP 0 / FP 1 / FN 2" in window.status.text()
    window.quantity.setCurrentIndex(window.quantity.findData("reports"))
    assert controls.expand.isHidden() and not controls.error_settings.isHidden()
    assert "TP 0 / FP 1 / FN 2" in window.status.text()
    values = objects["sensing:outcome_volume"].payload.values
    assert values[0, 0, 0] > 3 and values[0, 0, 1] == 1
    assert (values > 0).sum() == 2
    window.mode.setCurrentIndex(window.mode.findData("both"))
    assert objects["sensing:outcome_volume"].payload.mode == "iso"
    assert "sensing:outcome_volume:misses" not in objects
    window.quantity.setCurrentIndex(window.quantity.findData("outcomes"))
    assert controls.expand.isHidden() and controls.error_settings.isHidden()
    assert not window.evaluation_section.is_expanded()
    assert window.tabs.tabText(window.tabs.currentIndex()) == "View"
    assert "TP 0 / FP 1 / FN 2" in window.status.text()
    controls.basis.setCurrentIndex(controls.basis.findData("position"))
    assert "TP 1 / FP 0 / FN 1" in window.status.text()
    assert "sensing:outcome_volume:misses" in objects
    before = (
        window.evaluation_snapshot["tp"],
        window.evaluation_snapshot["fp"],
        window.evaluation_snapshot["fn"],
    )
    baseline_values = objects["sensing:outcome_volume"].payload.values.copy()
    assert window.cell_edges.isChecked()
    assert window.unavailable_edges.isEnabled()
    window.unavailable_edges.setChecked(True)
    assert window.unavailable_edges.isEnabled()
    edge_ids = {key for key in objects if "cell_edges" in key}
    assert len(edge_ids) == 2
    assert any("available" in key for key in edge_ids)
    np.testing.assert_array_equal(objects["sensing:outcome_volume"].payload.values, baseline_values)
    assert before == (
        window.evaluation_snapshot["tp"],
        window.evaluation_snapshot["fp"],
        window.evaluation_snapshot["fn"],
    )
    # Filters hide fills, while support edges keep the measured trial visible.
    window.inspection_controls.category_filters[3].setChecked(False)
    assert objects["sensing:outcome_volume:misses"].payload.values[0, 0, 1] == 0
    assert edge_ids <= objects.keys()
    window.inspection_controls.category_filters[3].setChecked(True)
    window.visible.setChecked(False)
    assert all(not objects[key].visible for key in edge_ids if key in objects)
    window.visible.setChecked(True)
    # A pending report read retires old support, even if the user toggles edges.
    window.evaluation = None
    window.quantity.setCurrentIndex(window.quantity.findData("details"))
    assert not objects
    window.cell_edges.setChecked(False)
    assert not objects
    window.evaluation = evaluation
    window.quantity.setCurrentIndex(window.quantity.findData("outcomes"))
    window.cell_edges.setChecked(False)
    assert not any("cell_edges" in key for key in objects)
    assert not window.unavailable_edges.isEnabled()
    window.cutaway.active.setChecked(True)
    window.cutaway.axis.setCurrentIndex(window.cutaway.axis.findData("x"))
    window.cutaway.boundary.setCurrentIndex(0)
    assert np.all(objects["sensing:outcome_volume:misses"].payload.values[:, :, 1] == 0)
    assert before == (
        window.evaluation_snapshot["tp"],
        window.evaluation_snapshot["fp"],
        window.evaluation_snapshot["fn"],
    )
    window.cutaway.mode.setCurrentIndex(window.cutaway.mode.findData("box"))
    assert objects["sensing:outcome_volume:misses"].payload.values[1, 0, 0] == 0
    assert before == (
        window.evaluation_snapshot["tp"],
        window.evaluation_snapshot["fp"],
        window.evaluation_snapshot["fn"],
    )
    window.mode.setCurrentIndex(window.mode.findData("slice"))
    assert window.cutaway.isHidden()
    assert objects["sensing:outcome_volume"].payload.values[0, 0, 1] == 1
    window.close()
    assert window.evaluation_cancel.is_set() and not objects
    # Retire the owned dialog while its application parent is still alive,
    # matching the real main-window lifecycle (qtbot only keeps weakrefs).
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not isValid(window)


@pytest.mark.parametrize("axis,index,keep_lower", [("x", 0, True), ("y", 1, False), ("z", 0, True)])
def test_cutaway_preserves_source_values_and_hides_whole_cells(axis, index, keep_lower):
    from visualizer.src.sensing.volume_cutaway import cut_volume

    original = VolumePayload(
        values=np.ones((2, 2, 2)), palette=np.array([[0, 0, 0, 0], [1, 1, 1, 1]]), limits=(0, 1)
    )
    cropped = cut_volume(original, (axis, index, keep_lower))
    assert cropped.values.sum() == 4
    assert original.values.sum() == 8
    assert cropped.palette is original.palette
    assert cut_volume(original, None) is original
    slices = [slice(None)] * 3
    slices[{"x": 2, "y": 1, "z": 0}[axis]] = index
    assert np.all(cropped.values[tuple(slices)] == 1)


def test_closing_during_report_loading_cancels_and_ignores_late_completion(
    qtbot, monkeypatch, tmp_path
):
    from threading import Event

    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QWidget
    from shiboken6 import isValid

    from visualizer.src.sensing import volume_window as module

    scene = tmp_path / "scenario.yaml"
    scene.write_text("{}")
    reader = SimpleNamespace(
        identity="test",
        is_fusion=False,
        native_availability=lambda key: "Native evidence not included",
        scene_path=scene,
        axes={n: np.array([0.0, 1.0]) for n in ("x", "y", "z")},
        headings=[{"name": "Test", "velocity_m_s": [1, 0, 0]}],
        cells={},
        metadata={"position_gate_m": 1.0, "sensor_position_m": [0, 0, 3]},
        outcome_grid=lambda h: np.zeros((2, 2, 2)),
    )
    started = Event()

    def loading(reader, *, cancel, progress):
        started.set()
        if not cancel.wait(5):
            raise RuntimeError("Cancellation was not signalled")
        raise InterruptedError("Cancelled")

    monkeypatch.setattr(module.SensingVolumeWindow, "_load", staticmethod(lambda _: reader))
    monkeypatch.setattr(module, "load_evaluation", loading)
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.current_scenario_path = scene
    parent.open_scenario = lambda *a, **kw: SimpleNamespace(succeeded=True)
    legends = []
    parent.renderer = SimpleNamespace(
        capabilities=RendererCapabilities(),
        ensure_object=lambda o: True,
        remove_object=lambda n: True,
        request_redraw=lambda: None,
        set_overview_camera=lambda *a: None,
        set_analysis_legend=legends.append,
    )
    window = module.SensingVolumeWindow(parent, scene)
    window.show()
    qtbot.waitUntil(started.is_set)
    window.quantity.setCurrentIndex(window.quantity.findData("reports"))
    assert "Loading saved reports" in window.status.text()
    future = window.evaluation_pending
    window.close()
    qtbot.waitUntil(future.done)
    assert isinstance(future.exception(), InterruptedError)
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not isValid(window) and legends[-1] is None
