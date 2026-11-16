"""Callout anchors, reversible native objects, UI and saved-state compatibility."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from visualizer.src.model import RenderObject, Transform
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.renderers.pygfx.geometry import PygfxGeometryMixin
from visualizer.src.renderers.pygfx.labels import PygfxLabelMixin
from visualizer.src.services.label_callout_service import label_with_callout
from visualizer.src.state import AppState, create_initial_state
from visualizer.src.types.label_style import LabelCalloutStyle
from visualizer.src.types.render_payloads import MaterialPayload, TextLabelPayload


def label_at(position=(1.0, 2.0, 3.0), kind="rx"):
    return RenderObject(
        id=f"node:{kind}_0::label",
        payload=TextLabelPayload("Rx1"),
        material=MaterialPayload(base_color=(0, 0.4, 1, 1)),
        transform=Transform.from_translation(np.asarray(position) + [0, 0, 1]),
        metadata={"layout_anchor": position, "layout_offset": (0, 0, 1)},
    )


def owner(style=None):
    return SimpleNamespace(
        renderer=SimpleNamespace(capabilities=RendererCapabilities(label_callouts=True)),
        app_state=create_initial_state(label_callout=style),
    )


def test_callout_payload_reused_without_reusing_pose_or_style_metadata():
    viz = owner(LabelCalloutStyle(mode="callout"))
    first = label_with_callout(label_at(), viz)
    moved = label_with_callout(label_at((8, 9, 10)), viz)
    assert moved.payload is first.payload
    assert moved.metadata["label_connector_origin"] == [8, 9, 10]
    viz.app_state = replace(
        viz.app_state, label_callout=LabelCalloutStyle(mode="callout", opacity=0.4)
    )
    changed = label_with_callout(label_at(), viz)
    assert changed.payload is first.payload
    assert changed.metadata["label_background_rgba"][-1] == 0.4
    assert first.metadata["label_background_rgba"][-1] != 0.4


def test_label_text_content_skips_repeats_and_retries_partial_failure():
    calls = []
    native = SimpleNamespace(
        set_text=lambda text: calls.append((text, False)),
        set_markdown=lambda text: calls.append((text, True)),
    )
    apply = PygfxLabelMixin._set_label_text
    apply(native, "Tx1", True)
    apply(native, "Tx1", True)
    apply(native, "Tx1", False)
    assert calls == [("**Tx1**", True), ("Tx1", False)]

    def fail(text):
        calls.append((text, False))
        raise RuntimeError("partial native text change")

    native.set_text = fail
    with pytest.raises(RuntimeError):
        apply(native, "Tx2", False)
    native.set_text = lambda text: calls.append((text, False))
    apply(native, "Tx1", False)
    assert calls[-1] == ("Tx1", False)
    assert len(calls) == 4


def test_style_is_a_detached_snapshot_and_plain_or_unsupported_is_unchanged():
    source = label_at()
    viz = owner()
    assert label_with_callout(source, viz) is source
    viz.app_state = replace(viz.app_state, label_callout=LabelCalloutStyle(mode="callout"))
    result = label_with_callout(source, viz)
    assert result is not source
    assert "label_callout" not in source.metadata
    np.testing.assert_array_equal(result.transform_matrix, source.transform_matrix)
    assert result.metadata["label_connector_origin"] == [1, 2, 3]
    assert result.metadata["label_background_rgba"][:3] == (0.7, 0.82, 1.0)
    tx = label_with_callout(label_at(kind="tx"), viz)
    assert tx.metadata["label_background_rgba"][:3] == (1, 1, 1)
    assert tx.metadata["label_bold"]
    viz.renderer.capabilities = RendererCapabilities()
    assert label_with_callout(source, viz) is source


def test_style_roundtrip_old_sessions_and_invalid_display_settings():
    style = LabelCalloutStyle(mode="callout", offset_x=-60, offset_y=80, on_top=False)
    state = create_initial_state(label_callout=style)
    assert AppState.from_dict(state.to_dict()).label_callout == style
    old = state.to_dict()
    old.pop("label_callout")
    assert AppState.from_dict(old).label_callout == LabelCalloutStyle()
    malformed = {
        "mode": "unknown",
        "offset_x": float("nan"),
        "offset_y": "bad",
        "marker_size": 10000,
        "opacity": -2,
        "on_top": "false",
    }
    checked = LabelCalloutStyle.from_dict(malformed)
    assert checked.mode == "plain" and checked.offset_x == 32 and checked.offset_y == 42
    assert checked.marker_size == 60 and checked.opacity == 0.1 and checked.on_top


def test_per_node_pixel_sizes_leave_target_and_locator_defaults_intact():
    style = LabelCalloutStyle.from_dict(
        {"mode": "card", "marker_size": 24, "tx_marker_size": 34, "rx_marker_size": 12}
    )
    viz = owner(style)
    for identifier, expected in (
        ("node:tx_0::label", 34),
        ("node:rx_0::label", 12),
        ("target:car::label", 24),
        ("locator:transmitter_label", 24),
    ):
        source = replace(label_at(), id=identifier)
        result = label_with_callout(source, viz)
        assert result.metadata["label_callout"]["marker_size"] == expected
    assert AppState.from_dict(viz.app_state.to_dict()).label_callout == style
    checked = LabelCalloutStyle.from_dict(
        {"tx_marker_size": float("nan"), "rx_marker_size": 10000, "marker_size": 26}
    )
    assert checked.node_marker_size("tx") == 26
    assert checked.node_marker_size("rx") == 60


def test_native_marker_follows_actor_not_card_and_switching_back_removes_objects():
    gfx = pytest.importorskip("pygfx")
    source = label_at()
    viz = owner(LabelCalloutStyle(mode="callout"))
    native = gfx.Text(text="Rx1", font_size=18, screen_space=True, material=gfx.TextMaterial())
    camera = gfx.PerspectiveCamera(45)
    camera.local.position = (0, 0, 30)
    renderer = SimpleNamespace(
        _gfx=gfx,
        _objects={source.id: native},
        _camera=camera,
        _canvas=SimpleNamespace(get_logical_size=lambda: (800, 600)),
    )

    def sync(label):
        decorated = label_with_callout(label, viz)
        PygfxGeometryMixin._sync_render_object_metadata(renderer, decorated)
        PygfxLabelMixin._update_label_backgrounds(renderer)
        return native._orchav_callout_group

    group = sync(source)
    np.testing.assert_allclose(group.children[2].geometry.positions.data[0], [1, 2, 3])
    assert not np.allclose(native.world.position, source.transform_matrix[:3, 3])
    first_position = native.world.position.copy()
    assert sync(source) is group
    np.testing.assert_allclose(native.world.position, first_position)  # No cumulative offset.
    camera.local.position = (0, 0, 60)
    sync(source)
    assert group.children[2].material.size == 18
    np.testing.assert_allclose(group.world.matrix, np.eye(4), atol=1e-6)
    moved = label_at((5, 7, 9))
    assert sync(moved) is group
    np.testing.assert_allclose(group.children[2].geometry.positions.data[0], [5, 7, 9])
    assert len(native.children) == 2  # One backplate and one annotation group.
    viz.app_state = replace(viz.app_state, label_callout=LabelCalloutStyle())
    sync(moved)
    assert not native.children
    np.testing.assert_allclose(native.world.position, moved.transform_matrix[:3, 3])
    assert native.material.depth_test
    assert native.render_order == 0


def test_controls_change_and_restore_style_without_duplicate_signals(qtbot):
    from visualizer.src.panels.label_style_controls import LabelStyleControls

    viz = owner()
    calls = []

    def changed(style):
        calls.append(style)
        viz.app_state = replace(viz.app_state, label_callout=style)

    viz.ui_controller = SimpleNamespace(handle_label_callout_changed=changed)
    controls = LabelStyleControls(viz)
    qtbot.addWidget(controls)
    controls.mode.setCurrentIndex(controls.mode.findData("callout"))
    assert len(calls) == 1 and calls[0].mode == "callout"
    controls.spins["offset_x"].setValue(100)
    assert calls[-1].offset_x == 100
    before = len(calls)
    controls.sync_from_state()
    assert len(calls) == before
    viz.renderer.capabilities = RendererCapabilities()
    controls.sync_from_state()
    assert not controls.mode.isEnabled()


def test_card_style_controls_and_session_roundtrip(qtbot):
    from visualizer.src.panels.label_style_controls import LabelStyleControls

    viz = owner()
    viz.ui_controller = SimpleNamespace(
        handle_label_callout_changed=lambda style: setattr(
            viz, "app_state", replace(viz.app_state, label_callout=style)
        )
    )
    controls = LabelStyleControls(viz)
    qtbot.addWidget(controls)
    controls.mode.setCurrentIndex(controls.mode.findData("card"))
    controls.background.setCurrentIndex(controls.background.findData("ink"))
    style = viz.app_state.label_callout
    assert style.mode == "card" and style.background == "ink"
    assert AppState.from_dict(viz.app_state.to_dict()).label_callout == style
    decorated = label_with_callout(label_at(), viz)
    assert min(decorated.material_payload.base_color[:3]) > 0.9
    assert max(decorated.metadata["label_background_rgba"][:3]) < 0.3
    viz.app_state = replace(viz.app_state, label_callout=LabelCalloutStyle(mode="card"))
    # The existing "White Tx / tinted Rx and targets" choice keeps its meaning.
    tx = label_with_callout(label_at(kind="tx"), viz)
    assert tx.metadata["label_background_rgba"][:3] == (1, 1, 1)


def test_nearby_cards_separate_without_moving_markers_and_without_drift():
    gfx = pytest.importorskip("pygfx")
    viz = owner(LabelCalloutStyle(mode="callout"))
    camera = gfx.PerspectiveCamera(45)
    camera.local.position = (0, 0, 30)
    renderer = SimpleNamespace(
        _gfx=gfx,
        _objects={},
        _camera=camera,
        _canvas=SimpleNamespace(get_logical_size=lambda: (800, 600)),
    )
    for index in range(5):
        source = replace(label_at((index * 0.05, 0, 0)), id=f"node:rx_{index}::label")
        native = gfx.Text(
            text=f"Rx{index + 1}", font_size=18, screen_space=True, material=gfx.TextMaterial()
        )
        renderer._objects[source.id] = native
        PygfxGeometryMixin._sync_render_object_metadata(renderer, label_with_callout(source, viz))
    PygfxLabelMixin._update_label_backgrounds(renderer)
    rects = [native._orchav_callout_rect.copy() for native in renderer._objects.values()]
    for index, rect in enumerate(rects):
        for other in rects[index + 1 :]:
            assert np.any(rect[1] <= other[0]) or np.any(other[1] <= rect[0])
    positions = [native.world.position.copy() for native in renderer._objects.values()]
    PygfxLabelMixin._update_label_backgrounds(renderer)
    for index, native in enumerate(renderer._objects.values()):
        np.testing.assert_allclose(native.world.position, positions[index])
        np.testing.assert_allclose(
            native._orchav_callout_group.children[2].geometry.positions.data[0],
            [index * 0.05, 0, 0],
        )


def test_moving_target_service_uses_same_callouts_and_retains_source_state():
    from visualizer.src.model import make_text_label_state
    from visualizer.src.services.entity_render_service import EntityRenderService
    from visualizer.src.services.target_render_sync import TargetRenderSync

    viz = owner(LabelCalloutStyle(mode="callout"))
    captured = []
    viz.renderer.ensure_object = lambda obj: captured.append(obj) or True
    sync = TargetRenderSync(viz, EntityRenderService(viz))
    label = make_text_label_state("target:car::label", "Car", [0.2, 0.5, 0.7])
    for position in ([1, 2, 3], [20, 3, 4]):
        assert sync.sync_target_label(
            geometry_name=label.id,
            label=label,
            visible=True,
            anchor_position=position,
            offset=[0, 0, 2],
        )
        result = captured[-1]
        assert result.metadata["label_connector_origin"] == position
        assert result.metadata["label_callout"]["mode"] == "callout"
        assert "label_callout" not in label.metadata
    assert sync.sync_target_label(geometry_name=label.id, label=label, visible=False)
    assert not captured[-1].visible


def test_volume_locators_replace_old_stems_only_for_enabled_callouts():
    from visualizer.src.services.label_callout_service import objects_with_callouts
    from visualizer.src.services.site_locator_service import site_locator_objects

    source = site_locator_objects(
        [1, 2, 3], "Radar", [np.arange(4)] * 3, ("point", "stem", "label"), readable=True
    )
    viz = owner(LabelCalloutStyle(mode="callout"))
    decorated = objects_with_callouts(source, viz)
    assert [obj.visible for obj in decorated] == [False, False, True]
    assert decorated[-1].metadata["label_bold"]
    assert all(obj.visible for obj in source)
    viz.app_state = replace(viz.app_state, label_callout=LabelCalloutStyle())
    assert objects_with_callouts(source, viz) == source


@pytest.fixture
def native_callout():
    gfx = pytest.importorskip("pygfx")
    source = label_at()
    native = gfx.Text(text="Rx1", font_size=18, screen_space=True, material=gfx.TextMaterial())
    camera = gfx.PerspectiveCamera(45)
    camera.local.position = (0, 0, 30)
    renderer = SimpleNamespace(
        _gfx=gfx,
        _objects={source.id: native},
        _camera=camera,
        _canvas=SimpleNamespace(get_logical_size=lambda: (800, 600)),
    )
    viz = owner(LabelCalloutStyle(mode="callout"))
    PygfxGeometryMixin._sync_render_object_metadata(renderer, label_with_callout(source, viz))
    return renderer, native, source, viz


def test_card_padding_solid_border_leader_and_reversible_shape(native_callout):
    r, native, source, viz = native_callout
    viz.app_state = replace(
        viz.app_state, label_callout=LabelCalloutStyle(mode="card", background="ink")
    )
    PygfxGeometryMixin._sync_render_object_metadata(r, label_with_callout(source, viz))
    PygfxLabelMixin._update_label_backgrounds(r)
    plate = r._label_backgrounds[source.id][1]
    group = native._orchav_callout_group
    np.testing.assert_allclose(group.children[2].geometry.positions.data[0], [1, 2, 3])
    assert group.children[2].material.marker == "circle"
    assert group.children[0].geometry.positions.draw_range == (0, 2)
    assert len(plate.geometry.positions.data) > 4
    assert plate.children[0].material.thickness == 3
    np.testing.assert_allclose(plate.children[0].material.color.rgba, [0, 0, 0, 1])

    def project(points):
        matrix = r._camera.projection_matrix @ r._camera.world.inverse_matrix
        clip = np.c_[points, np.ones(len(points))] @ matrix.T
        return (clip[:, :2] / clip[:, 3, None] + 1) * [400, 300]

    vertices = np.c_[plate.geometry.positions.data, np.ones(len(plate.geometry.positions.data))]
    card_points = (vertices @ plate.world.matrix.T)[:, :3]
    padding = (np.ptp(project(card_points), axis=0) - np.ptp(native._aabb[:, :2], axis=0)) / 2
    np.testing.assert_allclose(padding, [12, 8], atol=1e-4)
    start, end = project(group.children[0].geometry.positions.data[:2])
    bounds = native._orchav_callout_rect
    # The guide ends on the card boundary, never at the text center.
    assert np.any(
        np.isclose(end, bounds[0] + 2, atol=1e-4) | np.isclose(end, bounds[1] - 2, atol=1e-4)
    )
    assert np.linalg.norm(end - start) > 0
    geometry = plate.geometry
    PygfxLabelMixin._update_label_backgrounds(r)
    assert plate.geometry is geometry

    viz.app_state = replace(viz.app_state, label_callout=LabelCalloutStyle(mode="callout"))
    PygfxGeometryMixin._sync_render_object_metadata(r, label_with_callout(source, viz))
    PygfxLabelMixin._update_label_backgrounds(r)
    assert len(plate.geometry.positions.data) == 4
    assert group.children[2].material.marker == "square"
    assert group.children[0].geometry.positions.draw_range[1] > 2


def test_layout_reuse_and_camera_text_membership_invalidation(native_callout, monkeypatch):
    from unittest.mock import Mock

    import visualizer.src.renderers.pygfx.labels as labels

    r, native, source, viz = native_callout
    place = Mock(wraps=labels.position_callout)
    monkeypatch.setattr(labels, "position_callout", place)

    def draw():
        PygfxLabelMixin._update_label_backgrounds(r)

    draw()
    draw()
    assert place.call_count == 1
    r._camera.fov = 60
    draw()
    r._canvas.get_logical_size = lambda: (1000, 700)
    draw()
    native.font_size = 24
    draw()
    renamed = replace(source, payload=replace(source.payload, text="A longer name"))
    PygfxGeometryMixin._sync_render_object_metadata(r, label_with_callout(renamed, viz))
    draw()
    r._label_anchor_key_by_name = {source.id: (0, 0, 0)}
    r._label_anchor_groups = {(0, 0, 0): {source.id, "node:rx_1::label"}}
    draw()
    assert place.call_count == 6
    native.visible = False
    draw()
    moved = replace(label_at((10, 11, 12)), visibility=False)
    PygfxGeometryMixin._sync_render_object_metadata(r, label_with_callout(moved, viz))
    draw()
    assert place.call_count == 6  # Hidden poses are retained without layout work.
    native.visible = True
    draw()
    assert place.call_count == 7
    np.testing.assert_allclose(
        native._orchav_callout_group.children[2].geometry.positions.data[0], [10, 11, 12]
    )


def test_partial_layout_failure_is_not_cached_when_inputs_revert(native_callout, monkeypatch):
    from unittest.mock import Mock

    import visualizer.src.renderers.pygfx.labels as labels

    r, native, _, _ = native_callout

    def draw():
        PygfxLabelMixin._update_label_backgrounds(r)

    draw()
    first = native.world.position.copy()
    original = labels.reserve_callout_rect
    r._camera.local.position = (0, 0, 60)
    failing = Mock(side_effect=RuntimeError("layout interrupted"))
    monkeypatch.setattr(labels, "reserve_callout_rect", failing)
    with pytest.raises(RuntimeError):
        draw()
    r._camera.local.position = (0, 0, 30)
    repaired = Mock(wraps=original)
    monkeypatch.setattr(labels, "reserve_callout_rect", repaired)
    draw()
    assert repaired.call_count == 1
    np.testing.assert_allclose(native.world.position, first)


def test_callout_buffers_share_capacity_shrink_and_skip_unchanged_uploads(
    native_callout, monkeypatch
):
    from unittest.mock import Mock

    from visualizer.src.renderers.pygfx.callouts import update_callout

    r, native, source, viz = native_callout
    PygfxLabelMixin._update_label_backgrounds(r)
    group = native._orchav_callout_group
    marker = group.children[2].geometry.positions
    upload = Mock(wraps=marker.update_full)
    monkeypatch.setattr(marker, "update_full", upload)
    # Long guide, then a shorter guide within the same allocation.
    long_style = LabelCalloutStyle(mode="callout", offset_x=200, offset_y=100)
    viz.app_state = replace(viz.app_state, label_callout=long_style)
    PygfxGeometryMixin._sync_render_object_metadata(r, label_with_callout(source, viz))
    PygfxLabelMixin._update_label_backgrounds(r)
    guide = group.children[0].geometry
    long_count = guide.positions.draw_range[1]
    viz.app_state = replace(viz.app_state, label_callout=LabelCalloutStyle(mode="callout"))
    PygfxGeometryMixin._sync_render_object_metadata(r, label_with_callout(source, viz))
    PygfxLabelMixin._update_label_backgrounds(r)
    assert guide is group.children[0].geometry is group.children[1].geometry
    assert guide.positions.draw_range[1] < long_count
    upload.assert_not_called()  # The recorded actor position did not move.
    guide_upload = Mock(wraps=guide.positions.update_full)
    monkeypatch.setattr(guide.positions, "update_full", guide_upload)
    update_callout(
        r,
        native,
        r._label_backgrounds[source.id][1],
        r._camera.projection_matrix,
        np.array([800, 600]),
    )
    guide_upload.assert_not_called()


@pytest.mark.parametrize("child", [0, 2])
def test_callout_retries_failed_upload_with_identical_positions(native_callout, monkeypatch, child):
    r, native, source, viz = native_callout

    def draw():
        PygfxLabelMixin._update_label_backgrounds(r)

    draw()
    buffer = native._orchav_callout_group.children[child].geometry.positions
    original = buffer.update_full
    calls = []

    def fail_once():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("upload interrupted")
        original()

    monkeypatch.setattr(buffer, "update_full", fail_once)
    moved = label_at((1.1, 2, 3))
    PygfxGeometryMixin._sync_render_object_metadata(r, label_with_callout(moved, viz))
    with pytest.raises(RuntimeError):
        draw()
    draw()
    assert len(calls) == 2
