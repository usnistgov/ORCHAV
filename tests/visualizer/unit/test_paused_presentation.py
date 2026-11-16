"""Playback keeps data overlays enabled while stopping paused-only presentation effects."""

from types import SimpleNamespace

from PySide6.QtWidgets import QCheckBox

from visualizer.src.sensing.scene_explanation import SceneExplanationControls
from visualizer.src.sensing.uncertainty_controls import TrackingUncertaintyControls
from visualizer.src.services.paused_presentation import frames_advancing, playback_changed


def test_playback_preserves_uncertainty_and_disables_cinematic_effects(qtbot):
    uncertainty = TrackingUncertaintyControls()
    reveal = SceneExplanationControls(QCheckBox())
    qtbot.addWidget(uncertainty)
    qtbot.addWidget(reveal)
    uncertainty.enabled.setChecked(True)
    reveal._play()
    calls = []
    viz = SimpleNamespace(
        animation_running=False,
        ui_manager=SimpleNamespace(
            panels={
                "sensing": SimpleNamespace(
                    uncertainty_controls=uncertainty,
                    _sensing_window=SimpleNamespace(scene_explanation=reveal),
                )
            }
        ),
        extension_services={"experiments": SimpleNamespace(set_presentation_playing=calls.append)},
    )
    playback_changed(viz, True)
    assert frames_advancing(viz)
    assert uncertainty.enabled.isChecked()
    assert reveal.phase.currentData() == "off"
    assert uncertainty.enabled.isEnabled()
    assert not reveal.play.isEnabled() and not reveal.phase.isEnabled()
    assert not reveal.timer.isActive()
    playback_changed(viz, False)
    assert not frames_advancing(viz)
    assert uncertainty.enabled.isEnabled()
    assert reveal.play.isEnabled() and reveal.phase.isEnabled()
    assert uncertainty.enabled.isChecked()
    reveal._play()
    assert reveal.timer.isActive() and reveal.phase.currentData() == "range"
    reveal.stop()
    assert calls == [True, False]


def test_inactive_boundary_does_not_create_services_or_timers():
    viz = SimpleNamespace()
    playback_changed(viz, True)
    assert vars(viz) == {"_presentation_playback_active": True}
    playback_changed(viz, False)
    assert vars(viz) == {"_presentation_playback_active": False}


def test_uncertainty_toggle_embeds_in_existing_flags_without_hidden_parent(qtbot):
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    from visualizer.src.widgets.wrapping_layout import WrappingLayout

    parent = QWidget()
    qtbot.addWidget(parent)
    layout = QVBoxLayout(parent)
    flags = WrappingLayout()
    layout.addLayout(flags)
    controls = TrackingUncertaintyControls()
    layout.addWidget(controls)
    controls.embed_toggle(flags)
    parent.show()
    assert controls.enabled.isVisible()
    assert not controls.isVisible()
    controls.enabled.setChecked(True)
    assert controls.isVisible()
    assert controls.scale.isVisible() and controls.scope.isVisible()
    controls.set_playing(True)
    assert controls.enabled.isChecked()
    assert controls.note.isVisible()
    controls.set_playing(False)
    assert controls.isVisible()
    assert controls.enabled.isVisible()


def test_playback_boundary_ignores_retired_qt_controls(qtbot):
    from PySide6.QtCore import QTimer
    from shiboken6 import delete

    controls = TrackingUncertaintyControls()
    reveal = SceneExplanationControls(QCheckBox())
    timer = QTimer()
    viz = SimpleNamespace(
        animation_controller=SimpleNamespace(_live_playback_timer=timer),
        ui_manager=SimpleNamespace(
            panels={
                "sensing": SimpleNamespace(
                    uncertainty_controls=controls,
                    _sensing_window=SimpleNamespace(scene_explanation=reveal),
                )
            }
        ),
    )
    delete(controls)
    delete(reveal)
    delete(timer)
    playback_changed(viz, True)
    playback_changed(viz, False)
    assert not frames_advancing(viz)
