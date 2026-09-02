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
