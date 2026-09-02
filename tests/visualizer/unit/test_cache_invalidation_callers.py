from __future__ import annotations

import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import Mock

from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.io import frame_sources
from visualizer.src.materials.appearance import MaterialDisplayMode
from visualizer.src.services.cache_service import CacheInvalidationScope
from visualizer.src.services.material_modes import MaterialModeService
from visualizer.src.services.scene_edit_service import SceneEditService


def test_live_scene_object_update_invalidates_frame_and_static_geometry(monkeypatch):
    class FakeLiveGrpcSource:
        def __init__(self) -> None:
            self.update_object_via_xml = Mock(return_value=True)
            self.clear_buffer = Mock()

    monkeypatch.setattr(frame_sources, "LiveGrpcSource", FakeLiveGrpcSource)

    root = ET.Element("scene")
    shape = ET.SubElement(root, "shape")
    transform = ET.SubElement(shape, "transform", {"name": "to_world"})
    ET.SubElement(transform, "translate", {"x": "0", "y": "0", "z": "0"})

    frame_source = FakeLiveGrpcSource()
    viz = SimpleNamespace(
        frame_source=frame_source,
        xml_root=root,
        xml_path=None,
        cache_service=SimpleNamespace(invalidate=Mock()),
        animation_step=0,
        total_animation_steps=1,
        force_update_next_frame=False,
        update_frame=Mock(),
    )

    service = SceneEditService(viz)

    assert service.push_scene_updates_online("unit") is True

    frame_source.clear_buffer.assert_called_once_with()
    scopes = viz.cache_service.invalidate.call_args.args[0]
    assert set(scopes) == {
        CacheInvalidationScope.FRAME_DATA,
        CacheInvalidationScope.STATIC_SCENE_GEOMETRY,
    }
    assert viz.cache_service.invalidate.call_args.kwargs == {"reason": "OBJECT_UPDATE"}
    assert viz.force_update_next_frame is True
    viz.update_frame.assert_called_once_with(0)


def test_target_node_update_preserves_updated_target_geometry_cache(monkeypatch):
    monkeypatch.setattr(
        "visualizer.src.services.scene_edit_service.QTimer.singleShot",
        lambda _delay_ms, _callback: None,
    )

    frame_source = SimpleNamespace(
        update_node_properties=Mock(return_value=True),
        clear_buffer=Mock(),
    )
    loader = SimpleNamespace(
        validate_node_editing_context=Mock(return_value=True),
        warn_unsupported_node=Mock(),
        warn_missing_identifier=Mock(),
        inform_no_changes=Mock(),
        show_node_update_error=Mock(),
    )
    viz = SimpleNamespace(
        scenario_loader_service=loader,
        frame_source=frame_source,
        cache_service=SimpleNamespace(invalidate=Mock()),
        target_scale_overrides={},
        target_service=SimpleNamespace(apply_target_scale_from_metadata=Mock()),
        _set_status_message=Mock(),
        animation_step=0,
        total_animation_steps=1,
        force_update_next_frame=False,
        ui_controller=SimpleNamespace(populate_controls=Mock()),
    )

    service = SceneEditService(viz)
    entry = {
        "entry_type": "target",
        "target_name": "Pedestrian",
        "name": "Pedestrian",
        "supports_position": True,
        "supports_orientation": True,
        "supports_scale": True,
    }

    assert service.edit_node_properties(entry, {"scale": 1.25}) is True

    viz.target_service.apply_target_scale_from_metadata.assert_called_once_with(entry, 1.25)
    viz.cache_service.invalidate.assert_called_once_with(
        CacheInvalidationScope.FRAME_DATA,
        reason="NODE_UPDATE Pedestrian",
    )
    assert viz.force_update_next_frame is True
    assert viz.target_scale_overrides["Pedestrian"] == 1.25
    viz._set_status_message.assert_called_once_with(
        "Node update sent for Pedestrian; waiting for refreshed MPC data…",
        5000,
    )


def test_mpc_material_filter_does_not_change_persistent_object_material_mode() -> None:
    modes = MaterialModeService()
    viz = SimpleNamespace(
        mpc_allowed_materials={"brick"},
        _material_filter_dirty=False,
        schedule_update=Mock(),
    )
    controller = SimpleNamespace(
        visualizer=viz,
        material_mode_service=modes,
        _invalidate_cache=Mock(),
    )

    UIController.handle_mpc_material_filter_changed(controller, "brick", False)

    assert viz.mpc_allowed_materials == set()
    assert modes.get_mode("brick") is MaterialDisplayMode.NORMAL
    controller._invalidate_cache.assert_called_once_with(
        CacheInvalidationScope.FILTERS,
        reason="mpc_material_filter",
    )
    viz.schedule_update.assert_called_once_with()


def test_mpc_material_filter_full_selection_normalizes_to_no_filter() -> None:
    viz = SimpleNamespace(
        mpc_allowed_materials={"brick"},
        _mpc_material_filter_choices={"brick", "glass"},
        _material_filter_dirty=False,
        schedule_update=Mock(),
    )
    controller = SimpleNamespace(
        visualizer=viz,
        _invalidate_cache=Mock(),
    )

    UIController.handle_mpc_material_filter_changed(controller, "glass", True)

    assert viz.mpc_allowed_materials is None
    controller._invalidate_cache.assert_called_once_with(
        CacheInvalidationScope.FILTERS,
        reason="mpc_material_filter",
    )
    viz.schedule_update.assert_called_once_with()
