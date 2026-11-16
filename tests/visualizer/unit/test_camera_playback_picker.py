"""Camera actor choices stay usable while recorded positions advance."""

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QPersistentModelIndex, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QVBoxLayout, QWidget

from visualizer.src.controllers import camera_controller as module
from visualizer.src.services.camera_scene_query_service import CameraSceneQueryService


@pytest.fixture
def picker(qtbot):
    host = QWidget()
    combo = QComboBox(host)
    QVBoxLayout(host).addWidget(combo)
    qtbot.addWidget(host)
    viz = SimpleNamespace(
        target_focus_dropdown=combo,
        current_view_model=SimpleNamespace(
            target_metadata=[{"name": "alpha"}, {"name": "beta"}],
            target_positions=[[0, 0, 0], [10, 0, 0]],
            tx_positions=[],
            rx_positions=[],
        ),
        target_entries=[
            {"target_name": "alpha", "stable_target_id": "alpha"},
            {"target_name": "beta", "stable_target_id": "beta"},
        ],
        app_state=SimpleNamespace(
            tx_labels=(),
            rx_labels=(),
            tx_device_names=(),
            rx_device_names=(),
            node_label_mode="role",
        ),
        animation_running=True,
    )
    controller = module.CameraController(viz, CameraSceneQueryService(viz))
    combo.currentTextChanged.connect(lambda _: controller.remember_focus_selection())
    controller.update_target_focus_dropdown()
    combo.setCurrentIndex(1)
    host.show()
    yield (controller, combo, viz)
    combo.hidePopup()


def test_popup_highlight_and_identity_survive_frame_refresh(picker):
    controller, combo, viz = picker
    combo.showPopup()
    index = combo.model().index(2, 0)
    persistent = QPersistentModelIndex(index)
    combo.view().setCurrentIndex(index)
    resets = []
    combo.model().modelReset.connect(lambda: resets.append(True))
    for tick in range(5):
        viz.current_view_model.target_positions = [[tick, 0, 0], [10 + tick, 0, 0]]
        controller.update_target_focus_dropdown()
    assert persistent.isValid()
    assert combo.view().currentIndex().row() == 2
    assert not resets
    QTest.keyClick(combo.view(), Qt.Key_Return)
    assert combo.currentData()["stable_target_id"] == "beta"
    assert controller._preferred_focus_data == combo.currentData()
    assert viz.animation_running
    assert controller.scene_query.get_focus_position() == [14.0, 0.0, 0.0]


def test_inventory_changes_preserve_identity_then_fall_back(picker):
    controller, combo, viz = picker
    combo.setCurrentIndex(2)
    viz.current_view_model.target_metadata.reverse()
    viz.current_view_model.target_positions.reverse()
    controller.update_target_focus_dropdown()
    assert combo.currentData()["stable_target_id"] == "beta"
    assert combo.currentIndex() == 1
    viz.current_view_model.target_metadata = [{"name": "alpha"}]
    viz.current_view_model.target_positions = [[0, 0, 0]]
    controller.update_target_focus_dropdown()
    assert combo.currentData() == {"type": "auto"}


def test_live_contents_are_repaired_and_signal_block_is_preserved(picker):
    controller, combo, _ = picker
    combo.setItemText(1, "external edit")
    controller.update_target_focus_dropdown()
    assert combo.itemText(1) == "Target 1 - alpha"
    combo.setItemData(1, {"type": "tx", "index": 99})
    combo.blockSignals(True)
    controller.update_target_focus_dropdown()
    assert combo.itemData(1)["stable_target_id"] == "alpha"
    assert combo.signalsBlocked()
    combo.blockSignals(False)


def test_failed_choice_construction_does_not_change_popup_or_signals(picker, monkeypatch):
    controller, combo, _ = picker
    index = QPersistentModelIndex(combo.model().index(1, 0))
    original = [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]

    def unavailable(_):
        raise ValueError("temporary metadata failure")

    monkeypatch.setattr(controller.scene_query, "target_focus_selection", unavailable)
    controller.update_target_focus_dropdown()
    assert index.isValid()
    assert [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())] == original
    assert not combo.signalsBlocked()


@pytest.mark.parametrize("mode", ["overview", "follow", "pov"])
def test_playback_picker_routes_new_actor_to_active_camera_mode(picker, monkeypatch, mode):
    from visualizer.src.controllers.ui_controller import UIController

    controller, combo, viz = picker
    viz.app_state.camera_mode = mode
    viz.camera_controller = controller
    ui = UIController.__new__(UIController)
    ui.visualizer = viz
    calls = []
    monkeypatch.setattr(controller, "presentation", lambda: None)
    monkeypatch.setattr(
        controller,
        "update_follow_camera_focus",
        lambda: calls.append(("follow", controller.scene_query.get_focus_position())),
    )
    monkeypatch.setattr(
        controller,
        "set_pov_camera",
        lambda: calls.append(("pov", controller.scene_query.get_focus_position())),
    )
    combo.currentTextChanged.disconnect()
    combo.currentTextChanged.connect(ui.handle_target_focus_changed)
    combo.showPopup()
    combo.view().setCurrentIndex(combo.model().index(2, 0))
    for tick in range(3):
        viz.current_view_model.target_positions = [[tick, 0, 0], [10 + tick, 0, 0]]
        controller.update_target_focus_dropdown()
    QTest.keyClick(combo.view(), Qt.Key_Return)
    assert combo.currentData()["stable_target_id"] == "beta"
    assert controller._preferred_focus_data == combo.currentData()
    assert calls == ([] if mode == "overview" else [(mode, [12.0, 0.0, 0.0])])
    assert viz.animation_running
