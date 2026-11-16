from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

from visualizer.src.controllers.ui_controller import UIController


def _make_controller(viz) -> UIController:
    controller = UIController.__new__(UIController)
    controller.visualizer = viz
    return controller


def _make_visualizer():
    viz = SimpleNamespace()
    viz.app_state = SimpleNamespace(show_labels=True, show_target_labels=True)
    viz.vis_initialized = True
    viz.renderer = SimpleNamespace()
    viz.node_service = Mock()
    viz.schedule_update = Mock()

    def _set_state(**changes):
        for key, value in changes.items():
            setattr(viz.app_state, key, value)

    viz.set_state = Mock(side_effect=_set_state)
    return viz


def test_labels_toggle_updates_declarative_labels_immediately():
    viz = _make_visualizer()
    controller = _make_controller(viz)

    controller.handle_labels_toggled(False)

    viz.node_service.update_label_visibility.assert_called_once_with()
    viz.schedule_update.assert_called_once()
    assert viz.app_state.show_labels is False


def test_target_labels_toggle_updates_declarative_labels_immediately():
    viz = _make_visualizer()
    controller = _make_controller(viz)

    controller.handle_target_labels_toggled(False)

    viz.node_service.update_target_label_visibility.assert_called_once_with()
    viz.schedule_update.assert_called_once()
    assert viz.app_state.show_target_labels is False


def test_callout_change_refreshes_open_fusion_sensor_labels():
    from visualizer.src.types.label_style import LabelCalloutStyle

    viz = _make_visualizer()
    viz.fusion_window = SimpleNamespace(closed=False, _draw_world=Mock())
    controller = _make_controller(viz)
    style = LabelCalloutStyle(mode="callout")
    controller.handle_label_callout_changed(style)
    assert viz.app_state.label_callout == style
    viz.fusion_window._draw_world.assert_called_once_with()
    viz.fusion_window.closed = True
    controller.handle_label_callout_changed(style)
    assert viz.fusion_window._draw_world.call_count == 1


def test_scene_label_style_updates_volume_without_duplicate_style_controls():
    from visualizer.src.types.label_style import LabelCalloutStyle

    viz = _make_visualizer()
    viz.sensing_volume_window = SimpleNamespace(closed=False, _draw=Mock())
    controller = _make_controller(viz)
    style = LabelCalloutStyle(mode="callout")
    controller.handle_label_callout_changed(style)
    assert viz.app_state.label_callout == style
    viz.node_service.apply_label_offsets.assert_called_once_with()
    viz.sensing_volume_window._draw.assert_called_once_with()


def test_building_label_toggle_publishes_all_labels_in_one_batch():
    viz = _make_visualizer()
    entries = [{"name": "A"}, {"name": "B"}]
    batch_depth = 0
    batch_events = []

    @contextmanager
    def _tracked_batch():
        nonlocal batch_depth
        batch_depth += 1
        batch_events.append("enter")
        try:
            yield
        finally:
            batch_events.append("exit")
            batch_depth -= 1

    def _set_visibility(*_args, **_kwargs):
        assert batch_depth == 1

    viz.renderer = SimpleNamespace(batch_updates=_tracked_batch)
    viz.mesh_entries = entries
    viz.vis = object()
    viz.object_appearance_service = SimpleNamespace(
        set_building_label_visibility=Mock(side_effect=_set_visibility)
    )
    controller = _make_controller(viz)

    controller.handle_building_labels_toggled(False)

    assert batch_events == ["enter", "exit"]
    assert viz.object_appearance_service.set_building_label_visibility.call_count == 2
    for entry in entries:
        viz.object_appearance_service.set_building_label_visibility.assert_any_call(
            entry,
            False,
            update_renderer=False,
        )
    viz.schedule_update.assert_called_once()


def test_authored_false_labels_sync_actual_startup_controls(qtbot):
    from PySide6.QtWidgets import QWidget

    from visualizer.src.app.scenario_presentation import apply_presentation_defaults
    from visualizer.src.panels.nodes_panel import NodesSelectionPanel
    from visualizer.src.state import create_initial_state, update_state

    viz = QWidget()
    qtbot.addWidget(viz)
    viz.app_state = create_initial_state(show_labels=False, show_target_labels=False)
    viz.set_state = lambda **kw: setattr(viz, "app_state", update_state(viz.app_state, **kw))
    panel = NodesSelectionPanel(viz)
    labels = panel._create_display_group()
    qtbot.addWidget(labels)
    viz.ui_manager = SimpleNamespace(panels={"nodes": panel})
    apply_presentation_defaults(viz, {})
    assert not panel.widgets["labels_cb"].isChecked()
    assert not panel.widgets["target_labels_cb"].isChecked()
    # Synchronization is signal blocked; subsequent ordinary user toggles work.
    panel.widgets["labels_cb"].toggled.connect(lambda value: viz.set_state(show_labels=value))
    panel.widgets["target_labels_cb"].toggled.connect(
        lambda value: viz.set_state(show_target_labels=value)
    )
    panel.widgets["labels_cb"].click()
    panel.widgets["target_labels_cb"].click()
    assert viz.app_state.show_labels and viz.app_state.show_target_labels
