"""Street-coordinate fidelity, lazy scene ownership, and bounded label layout."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from visualizer.src.model import RenderObject, Transform
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.renderers.pygfx.street_labels import (
    select_street_labels,
    sync_street_label,
    update_street_labels,
)
from visualizer.src.scene.street_names import load_street_names, street_graph_path
from visualizer.src.services.street_name_service import StreetNameService
from visualizer.src.types.render_payloads import MeshPayload, TextLabelPayload


def graph(tmp_path, *, coordinates=True):
    path = tmp_path / "street_network.graphml"
    path.write_text("""<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
      <key id="x" for="node" attr.name="x_local"/>
      <key id="y" for="node" attr.name="y_local"/>
      <key id="n" for="edge" attr.name="name"/>
      <key id="g" for="edge" attr.name="geometry_local"/>
      <key id="b" for="edge" attr.name="bridge_z_offset"/>
      <graph edgedefault="directed">
        <node id="a"><data key="x">0</data><data key="y">0</data></node>
        <node id="b"><data key="x">20</data><data key="y">20</data></node>
        <edge source="a" target="b"><data key="n">Rue de l’Opéra</data>
          <data key="g">[[0,0],[20,0],[20,20]]</data><data key="b">4</data></edge>
        <edge source="b" target="a"><data key="n">Rue de l’Opéra</data>
          <data key="g">[[20,20],[20,0],[0,0]]</data><data key="b">4</data></edge>
        <edge source="a" target="b"><data key="n">['Rue A', 'Rue B']</data></edge>
        <edge source="a" target="b"/>
      </graph></graphml>""")
    if not coordinates:
        path.write_text(
            path.read_text()
            .replace('attr.name="x_local"', 'attr.name="lon"')
            .replace('attr.name="y_local"', 'attr.name="lat"')
            .replace('attr.name="geometry_local"', 'attr.name="unknown"')
        )
    return path


def test_curve_midpoint_duplicate_directions_bridge_and_list_names(tmp_path):
    names = load_street_names(graph(tmp_path), {})
    assert len(names) == 2
    curved = next(n for n in names if "Opéra" in n.name)
    assert curved.position == (20.0, 0.0, 4.17)
    assert curved.length_m == 40
    assert next(n for n in names if n.name == "Rue A / Rue B").position == (10, 10, 0.17)


def test_geographic_coordinates_are_not_scene_metres(tmp_path):
    assert load_street_names(graph(tmp_path, coordinates=False), {}) == ()


def test_variant_uses_explicit_graph_artifact(tmp_path):
    scene = tmp_path / "variant" / "scene.xml"
    scene.parent.mkdir()
    scene.with_name("osm_metadata.json").write_text(
        json.dumps({"artifacts": {"street_network_graphml": "../street_network.graphml"}})
    )
    assert street_graph_path(scene)[0] == graph(tmp_path)


def test_service_is_off_lazy_cached_and_cleared_on_scene_change(tmp_path, monkeypatch):
    graph(tmp_path)
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(screen_space_labels=True),
        ensure_object=Mock(return_value=True),
        remove_object=Mock(return_value=True),
    )
    service = StreetNameService(SimpleNamespace(renderer=renderer, mesh_entries=[]))
    service.set_scene(tmp_path / "scene.xml")
    assert not service.enabled and service.labels is None and service.available
    assert service.set_enabled(True)
    assert len(service.labels) == 2
    monkeypatch.setattr(
        "visualizer.src.services.street_name_service.load_street_names",
        Mock(side_effect=AssertionError("graph reparsed")),
    )
    assert service.set_enabled(False)
    assert not renderer.ensure_object.call_args.args[0].visible
    assert service.set_enabled(True)
    service.set_scene(tmp_path / "missing" / "scene.xml")
    assert renderer.remove_object.call_count == 2
    assert not service.available and not service.enabled and service.labels is None


def test_bad_graph_does_not_break_scene(tmp_path):
    (tmp_path / "street_network.graphml").write_text("broken XML")
    service = StreetNameService(
        SimpleNamespace(
            renderer=SimpleNamespace(capabilities=RendererCapabilities(screen_space_labels=True)),
            mesh_entries=[],
        )
    )
    service.set_scene(tmp_path / "scene.xml")
    assert not service.set_enabled(True)
    assert not service.enabled and not service.available
    assert "Could not read" in service.error


@pytest.mark.parametrize("mode", ["plain", "box", "card"])
def test_styles_reuse_source_and_remain_nonpickable_across_visibility_and_rebuild(
    tmp_path, monkeypatch, mode
):
    graph(tmp_path)
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(screen_space_labels=True, label_callouts=True),
        ensure_object=Mock(return_value=True),
        remove_object=Mock(return_value=True),
    )
    service = StreetNameService(SimpleNamespace(renderer=renderer, mesh_entries=[]))
    service.set_scene(tmp_path / "scene.xml")
    assert service.style_mode == "card"
    assert service.set_style(mode)
    assert service.labels is None
    renderer.ensure_object.assert_not_called()
    assert service.set_enabled(True)
    sources = service.labels
    label = renderer.ensure_object.call_args.args[0]
    assert not label.metadata["pickable"] and label.metadata["depth_test"]
    assert "label_callout" not in sources[-1].metadata
    assert "label_background_rgba" not in sources[-1].metadata
    if mode == "plain":
        assert "label_callout" not in label.metadata
        assert "label_background_rgba" not in label.metadata
    else:
        assert "label_callout" not in label.metadata
        assert "label_connector_origin" not in label.metadata
        assert label.metadata["label_card"] is (mode == "card")
    np.testing.assert_array_equal(label.transform_matrix, sources[-1].transform_matrix)
    with monkeypatch.context() as patch:
        patch.setattr(
            "visualizer.src.services.street_name_service.load_street_names",
            Mock(side_effect=AssertionError("graph reparsed")),
        )
        assert service.set_enabled(False)
        assert service.set_enabled(True)
        assert service.labels is sources
        assert renderer.ensure_object.call_args.args[0].metadata == label.metadata
    service.set_scene(tmp_path / "scene.xml")
    assert service.style_mode == mode and service.labels is None
    assert service.set_enabled(True)
    assert renderer.ensure_object.call_args.args[0].metadata == label.metadata


def test_style_switch_does_not_reparse_and_plain_removes_card_metadata(tmp_path, monkeypatch):
    graph(tmp_path)
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(screen_space_labels=True, label_callouts=True),
        ensure_object=Mock(return_value=True),
        remove_object=Mock(return_value=True),
    )
    service = StreetNameService(SimpleNamespace(renderer=renderer, mesh_entries=[]))
    service.set_scene(tmp_path / "scene.xml")
    service.set_enabled(True)
    monkeypatch.setattr(
        "visualizer.src.services.street_name_service.load_street_names",
        Mock(side_effect=AssertionError("graph reparsed")),
    )
    assert service.set_style("box")
    assert not renderer.ensure_object.call_args.args[0].metadata["label_card"]
    assert service.set_style("plain")
    assert "label_callout" not in renderer.ensure_object.call_args.args[0].metadata
    before = renderer.ensure_object.call_count
    assert service.set_style("plain")
    assert renderer.ensure_object.call_count == before
    service.set_enabled(False)
    before = renderer.ensure_object.call_count
    service.set_style("card")
    assert renderer.ensure_object.call_count == before
    with pytest.raises(ValueError, match="Unknown street-name style"):
        service.set_style("invalid")
    assert service.style_mode == "card"


def test_street_labels_follow_ground_and_bridge_surface_with_only_two_centimetre_lift(tmp_path):
    graph(tmp_path)
    ground = MeshPayload(
        vertices=np.array([[0, 0, 2], [20, 0, 2], [20, 20, 2]]), triangles=np.array([[0, 1, 2]])
    )
    roads = MeshPayload(
        vertices=np.array(
            [[10, 10, 2.3], [11, 10, 2.3], [10, 11, 2.3], [20, 0, 6.3], [21, 0, 6.3], [20, 1, 6.3]]
        ),
        triangles=np.array([[0, 1, 2], [3, 4, 5]]),
    )
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(screen_space_labels=True),
        ensure_object=Mock(return_value=True),
        remove_object=Mock(),
    )
    service = StreetNameService(
        SimpleNamespace(
            renderer=renderer,
            mesh_entries=[
                {"rel_path": "ground.ply", "mesh": SimpleNamespace(payload=ground)},
                {"rel_path": "streets.ply", "mesh": SimpleNamespace(payload=roads)},
            ],
        )
    )
    service.set_scene(tmp_path / "scene.xml")
    service.set_enabled(True)
    elevations = {label.payload.text: label.transform_matrix[2, 3] for label in service.labels}
    assert elevations["Rue de l’Opéra"] == pytest.approx(6.32)
    assert elevations["Rue A / Rue B"] == pytest.approx(2.32)


def test_screen_selection_preserves_anchors_and_limits_clutter():
    positions = np.array([[0, 0, 0], [0.01, 0, 0], [0.7, 0, 0], [-0.7, 0, 0], [2, 0, 0], [0, 0, 2]])
    original = positions.copy()
    selected = select_street_labels(
        positions, np.tile([100, 20], (6, 1)), ["A", "B", "A", "C", "D", "E"], np.eye(4), [800, 600]
    )
    assert selected.tolist() == [True, False, False, True, False, False]
    np.testing.assert_array_equal(positions, original)


def test_native_layout_cache_toggle_camera_and_removed_objects(monkeypatch):
    gfx = pytest.importorskip("pygfx")
    camera = gfx.PerspectiveCamera(45)
    camera.local.position = (0, 0, 30)
    native = gfx.Text(
        text="Rue Auber", font_size=14, screen_space=True, material=gfx.TextMaterial()
    )
    obj = RenderObject(
        "street:0",
        TextLabelPayload("Rue Auber"),
        transform=Transform.from_translation([0, 0, 0]),
        metadata={"street_label": "Rue Auber"},
    )
    renderer = SimpleNamespace(
        _objects={obj.id: native},
        _camera=camera,
        _canvas=SimpleNamespace(get_logical_size=lambda: (800, 600)),
    )
    sync_street_label(renderer, obj, native)
    from visualizer.src.renderers.pygfx import street_labels

    select = Mock(wraps=street_labels.select_street_labels)
    monkeypatch.setattr(street_labels, "select_street_labels", select)
    update_street_labels(renderer)
    assert native.visible
    update_street_labels(renderer)
    assert select.call_count == 1
    # An incidental native transform must neither recull an unchanged view nor
    # replace the immutable road anchor used by the next camera selection.
    native.world.position = (1000, 2000, 0)
    update_street_labels(renderer)
    assert select.call_count == 1
    camera.local.position = (0, 0, 40)
    update_street_labels(renderer)
    assert select.call_count == 2
    np.testing.assert_array_equal(select.call_args.args[0], [[0, 0, 0]])
    sync_street_label(renderer, replace(obj, visibility=False), native)
    assert not renderer._street_label_objects
    # Loaded labels remain cached natively, but hiding the feature must not
    # read the camera/viewport or build a per-frame decluttering key.
    canvas = renderer._canvas
    renderer._canvas = SimpleNamespace(
        get_logical_size=Mock(side_effect=AssertionError("hidden street projection"))
    )
    update_street_labels(renderer)
    assert not native.visible
    renderer._canvas = canvas
    sync_street_label(renderer, obj, native)
    update_street_labels(renderer)
    assert native.visible
    renderer._objects.clear()
    update_street_labels(renderer)
    assert not renderer._street_label_objects


def test_native_street_style_roundtrip_reuses_objects_keeps_road_anchors_and_picking(
    tmp_path, monkeypatch
):
    gfx = pytest.importorskip("pygfx")
    from visualizer.src.renderers.pygfx import street_labels
    from visualizer.src.renderers.pygfx.geometry import PygfxGeometryMixin
    from visualizer.src.renderers.pygfx.labels import PygfxLabelMixin

    graph(tmp_path)
    camera = gfx.PerspectiveCamera(45)
    camera.local.position = (10, 10, 80)
    renderer = SimpleNamespace(
        _gfx=gfx,
        _objects={},
        _camera=camera,
        _canvas=SimpleNamespace(get_logical_size=lambda: (800, 600)),
        capabilities=RendererCapabilities(screen_space_labels=True, label_callouts=True),
        remove_object=Mock(),
    )

    def ensure(obj):
        if obj.id not in renderer._objects:
            renderer._objects[obj.id] = gfx.Text(
                text=obj.payload.text, font_size=14, screen_space=True, material=gfx.TextMaterial()
            )
        native = renderer._objects[obj.id]
        native.world.position = obj.transform_matrix[:3, 3]
        native.visible = obj.visible
        PygfxGeometryMixin._sync_render_object_metadata(renderer, obj)
        return True

    renderer.ensure_object = ensure
    service = StreetNameService(SimpleNamespace(renderer=renderer, mesh_entries=[]))
    service.set_scene(tmp_path / "scene.xml")
    service.set_enabled(True)
    source = service.labels[0]
    native = renderer._objects[source.id]
    select = Mock(wraps=street_labels.select_street_labels)
    monkeypatch.setattr(street_labels, "select_street_labels", select)
    for mode in ("card", "box", "plain", "box", "card", "plain"):
        service.set_style(mode)
        PygfxLabelMixin._update_label_backgrounds(renderer)
        assert renderer._objects[source.id] is native
        assert native.visible and not native.material.pick_write
        assert native.material.depth_test
        before = native.world.position.copy()
        count = select.call_count
        for _ in range(3):
            PygfxLabelMixin._update_label_backgrounds(renderer)
        assert select.call_count == count
        np.testing.assert_array_equal(native.world.position, before)
        np.testing.assert_array_equal(native.world.position, source.transform_matrix[:3, 3])
        assert getattr(native, "_orchav_callout_group", None) is None
        if mode == "plain":
            assert not native.children
            np.testing.assert_array_equal(native.world.position, source.transform_matrix[:3, 3])
        else:
            assert len(native.children) == 1
            plate = native.children[0]
            assert len(plate.children) == 1
            assert plate._orchav_shape_key[0] is (mode == "card")
            for child in (plate, *plate.children):
                material = getattr(child, "material", None)
                if material is not None:
                    assert not material.pick_write and material.depth_test
        service.set_enabled(False)
        PygfxLabelMixin._update_label_backgrounds(renderer)
        assert not native.visible
        service.set_enabled(True)
        PygfxLabelMixin._update_label_backgrounds(renderer)
        assert native.visible


@pytest.fixture
def native_street_renderer():
    gfx = pytest.importorskip("pygfx")
    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer

    renderer = PygfxRenderer(SimpleNamespace(app_state=SimpleNamespace(label_screen_space=True)))
    renderer._gfx = gfx
    renderer._initialized = True
    renderer._scene = gfx.Scene()
    renderer._camera = gfx.PerspectiveCamera(45)
    renderer._camera.local.position = (10, 10, 80)
    renderer._canvas = SimpleNamespace(get_logical_size=lambda: (800, 600))
    renderer.request_redraw = lambda: None
    return renderer


def test_hidden_street_nodes_and_cards_are_dormant_then_reused(native_street_renderer):
    renderer = native_street_renderer
    obj = RenderObject(
        "street:0",
        TextLabelPayload("Rue Auber"),
        metadata={
            "street_label": "Rue Auber",
            "label_background_rgba": (0.03, 0.06, 0.1, 0.9),
            "label_card": True,
        },
    )
    assert renderer.ensure_object(obj)
    renderer._update_label_backgrounds()
    native = renderer._objects[obj.id]
    plate = renderer._label_backgrounds[obj.id][1]
    for _ in range(3):
        assert renderer.ensure_object(replace(obj, visibility=False))
        assert native.parent is None
        assert native._orchav_street_parent is renderer._scene
        assert native._orchav_street_background is plate
        assert not renderer._street_label_objects
        assert not renderer._label_backgrounds
        assert tuple(renderer._scene.children) == ()
        # Camera motion and the shared native/export layout entry cannot wake
        # a disabled layer or scan its retained backplates.
        renderer._camera.local.position = (10, 10, 100)
        renderer._canvas.get_logical_size = Mock(side_effect=AssertionError("dormant layout"))
        renderer._update_label_backgrounds()
    renderer._canvas.get_logical_size = lambda: (800, 600)
    assert renderer.ensure_object(obj)
    renderer._update_label_backgrounds()
    assert renderer._objects[obj.id] is native
    assert renderer._label_backgrounds[obj.id][1] is plate
    assert native.parent is renderer._scene
    assert tuple(renderer._scene.children) == (native,)
    assert tuple(native.children) == (plate,)
    assert native._orchav_street_parent is None
    assert native._orchav_street_background is None


def test_dormant_street_style_changes_and_source_clear(native_street_renderer, tmp_path):
    renderer = native_street_renderer
    graph(tmp_path)
    service = StreetNameService(SimpleNamespace(renderer=renderer, mesh_entries=[]))
    service.set_scene(tmp_path / "scene.xml")
    assert service.set_enabled(True)
    source = service.labels[0]
    native = renderer._objects[source.id]
    old_plate = renderer._label_backgrounds[source.id][1]
    assert service.set_enabled(False)
    assert service.set_style("plain")
    assert service.set_enabled(True)
    assert renderer._objects[source.id] is native
    assert not native.children
    assert old_plate.parent is None
    assert not renderer._label_backgrounds
    assert service.set_enabled(False)
    assert service.set_style("box")
    assert service.set_enabled(True)
    renderer._update_label_backgrounds()
    assert len(native.children) == 1
    assert not native.children[0]._orchav_shape_key[0]
    assert service.set_enabled(False)
    # Reconcile a changed hidden payload as well as the service's lazy style
    # choice. Existing cards must be removed without attaching hidden glyphs.
    assert renderer.ensure_object(replace(source, visibility=False))
    assert native.parent is None and not native.children
    assert not renderer._label_backgrounds
    assert service.set_style("card")
    assert service.set_enabled(True)
    assert service.set_enabled(False)
    assert native._orchav_street_background is not None
    service.set_scene(tmp_path / "missing" / "scene.xml")
    assert not renderer._objects
    assert not renderer._street_label_objects
    assert not renderer._label_backgrounds
    assert not renderer._scene.children
    assert native._orchav_street_parent is None
    assert native._orchav_street_background is None
    assert renderer.remove_object(source.id)
