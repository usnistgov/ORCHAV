"""Focused Qt tests for the complete orientation editor surface."""

from __future__ import annotations

from uuid import UUID

import pytest
from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QSpinBox

from shared.scenarios.actors import (
    MAX_RANDOM_SEED,
    AlignMotionOrientationSpec,
    FixedOrientationSpec,
    KeyframesOrientationSpec,
    OrientationKeyframeSpec,
    RandomOrientationSpec,
    SpinOrientationSpec,
)
from visualizer.src.authoring.orientation_editor import OrientationEditor
from visualizer.src.authoring.orientation_models import (
    OrientationKind,
    actor_look_at_orientation,
    point_look_at_orientation,
)

_LOOK_AT_TARGET = UUID("10000000-0000-0000-0000-000000000001")


@pytest.mark.parametrize(
    ("orientation", "page_name"),
    (
        (
            FixedOrientationSpec(
                yaw_deg=12.5,
                pitch_deg=-30.25,
                roll_deg=179.0,
            ),
            "orientationFixedPage",
        ),
        (
            KeyframesOrientationSpec(
                keyframes=(
                    OrientationKeyframeSpec(
                        time_s=0.25,
                        yaw_deg=0.0,
                        pitch_deg=1.0,
                        roll_deg=2.0,
                    ),
                    OrientationKeyframeSpec(
                        time_s=2.5,
                        yaw_deg=30.0,
                        pitch_deg=-15.0,
                        roll_deg=5.0,
                    ),
                    OrientationKeyframeSpec(
                        time_s=4.75,
                        yaw_deg=90.0,
                        pitch_deg=0.0,
                        roll_deg=-10.0,
                    ),
                )
            ),
            "orientationKeyframesPage",
        ),
        (
            AlignMotionOrientationSpec(
                allow_pitch=False,
                smoothing_time_s=0.75,
                yaw_offset_deg=10.0,
                pitch_offset_deg=-5.0,
                roll_offset_deg=2.0,
                max_yaw_rate_deg_s=45.0,
                max_pitch_rate_deg_s=20.0,
            ),
            "orientationAlignMotionPage",
        ),
        (
            actor_look_at_orientation(
                _LOOK_AT_TARGET,
                allow_pitch=False,
                smoothing_time_s=0.5,
                max_yaw_rate_deg_s=60.0,
                max_pitch_rate_deg_s=30.0,
                yaw_offset_deg=1.0,
                pitch_offset_deg=2.0,
                roll_offset_deg=3.0,
                yaw_limits_deg=(-70.0, 80.0),
                pitch_limits_deg=(-35.0, 40.0),
            ),
            "orientationLookAtPage",
        ),
        (
            SpinOrientationSpec(
                axis="pitch",
                rate_deg_s=-135.0,
                yaw_deg=10.0,
                pitch_deg=-5.0,
                roll_deg=2.5,
            ),
            "orientationSpinPage",
        ),
        (
            RandomOrientationSpec(
                seed=MAX_RANDOM_SEED,
                yaw_range_deg=(-120.0, 130.0),
                pitch_range_deg=(-45.0, 50.0),
                roll_range_deg=(-15.0, 20.0),
                update_interval_s=0.125,
            ),
            "orientationRandomPage",
        ),
    ),
    ids=("fixed", "keyframes", "align-motion", "look-at", "spin", "random"),
)
def test_all_orientation_pages_round_trip_every_field_without_type_signal(
    qapp,
    orientation,
    page_name,
) -> None:
    editor = OrientationEditor()
    emitted: list[str] = []
    editor.orientation_type_changed.connect(emitted.append)

    editor.set_orientation(orientation)

    assert editor.type_combo.currentData() == orientation.type
    assert editor.page_stack.currentWidget().objectName() == page_name
    assert editor.get_orientation() == orientation
    assert emitted == []
    editor.close()


def test_type_signal_switches_pages_without_applying_conversion(qapp) -> None:
    editor = OrientationEditor()
    emitted: list[str] = []
    editor.orientation_type_changed.connect(emitted.append)
    original = FixedOrientationSpec(yaw_deg=8.0, pitch_deg=9.0, roll_deg=10.0)
    editor.set_orientation(original)

    editor.set_orientation(
        SpinOrientationSpec(
            axis="roll",
            rate_deg_s=45.0,
            yaw_deg=3.0,
            pitch_deg=-4.0,
            roll_deg=5.0,
        ),
        preserve_type_signal=True,
    )
    assert emitted == ["spin"]

    editor.type_combo.setCurrentIndex(editor.type_combo.findData("fixed"))
    assert emitted == ["spin", "fixed"]
    assert editor.orientation() == original
    editor.close()


def test_actor_look_at_choices_use_uuid_identity_and_current_labels(qapp) -> None:
    editor = OrientationEditor()
    first_id = UUID("20000000-0000-0000-0000-000000000001")
    second_id = UUID("20000000-0000-0000-0000-000000000002")
    missing_id = UUID("20000000-0000-0000-0000-000000000003")
    editor.set_look_at_choices(((first_id, "Alpha"), (second_id, "Bravo")))

    editor.set_orientation(actor_look_at_orientation(second_id))
    assert editor.look_at_target_mode_combo.currentData() == "actor"
    assert editor.look_at_combo.currentData() == second_id
    assert editor.look_at_combo.currentText() == "Bravo"
    assert editor.orientation() == actor_look_at_orientation(second_id)

    editor.set_look_at_choices(((second_id, "Bravo renamed"), (first_id, "Alpha")))
    assert editor.look_at_combo.currentData() == second_id
    assert editor.look_at_combo.currentText() == "Bravo renamed"

    editor.set_orientation(actor_look_at_orientation(missing_id))
    assert editor.look_at_combo.currentData() == missing_id
    assert str(missing_id) in editor.look_at_combo.currentText()
    assert editor.orientation() == actor_look_at_orientation(missing_id)

    editor.set_look_at_choices(((first_id, "Alpha"),), preserve_selection=False)
    assert editor.look_at_combo.currentIndex() == -1
    assert editor.look_at_combo.findData(missing_id) == -1
    editor.close()


def test_point_look_at_round_trips_without_an_actor_choice(qapp) -> None:
    editor = OrientationEditor()
    orientation = point_look_at_orientation(
        (10.0, -20.0, 30.0),
        allow_pitch=False,
        smoothing_time_s=0.25,
        max_yaw_rate_deg_s=50.0,
        max_pitch_rate_deg_s=25.0,
        yaw_offset_deg=4.0,
        pitch_offset_deg=5.0,
        roll_offset_deg=6.0,
        yaw_limits_deg=(-90.0, 90.0),
        pitch_limits_deg=(-30.0, 30.0),
    )

    editor.set_orientation(orientation)

    assert editor.look_at_target_mode_combo.currentData() == "point"
    assert editor.look_at_target_stack.currentIndex() == 1
    assert editor.orientation() == orientation
    editor.close()


def test_actor_look_at_mode_requires_a_selected_actor(qapp) -> None:
    editor = OrientationEditor()
    editor.type_combo.setCurrentIndex(editor.type_combo.findData("look_at"))
    editor.look_at_target_mode_combo.setCurrentIndex(
        editor.look_at_target_mode_combo.findData("actor")
    )

    with pytest.raises(ValueError, match="requires a target actor"):
        editor.orientation()
    editor.close()


def test_keyframes_include_time_and_support_add_remove_and_reorder(qapp) -> None:
    editor = OrientationEditor()
    editor.set_orientation(
        KeyframesOrientationSpec(
            keyframes=(
                OrientationKeyframeSpec(
                    time_s=0.0,
                    yaw_deg=10.0,
                    pitch_deg=1.0,
                    roll_deg=-1.0,
                ),
                OrientationKeyframeSpec(
                    time_s=1.0,
                    yaw_deg=20.0,
                    pitch_deg=2.0,
                    roll_deg=-2.0,
                ),
                OrientationKeyframeSpec(
                    time_s=3.0,
                    yaw_deg=30.0,
                    pitch_deg=3.0,
                    roll_deg=-3.0,
                ),
            )
        )
    )
    apply_requests: list[bool] = []
    editor.apply_requested.connect(lambda: apply_requests.append(True))

    editor.keyframes_table.selectRow(1)
    editor.keyframe_up_button.click()
    moved = editor.orientation()
    assert isinstance(moved, KeyframesOrientationSpec)
    assert tuple(keyframe.time_s for keyframe in moved.keyframes) == (0.0, 1.0, 3.0)
    assert tuple(keyframe.yaw_deg for keyframe in moved.keyframes) == (20.0, 10.0, 30.0)

    editor.keyframe_down_button.click()
    restored = editor.orientation()
    assert isinstance(restored, KeyframesOrientationSpec)
    assert tuple(keyframe.yaw_deg for keyframe in restored.keyframes) == (
        10.0,
        20.0,
        30.0,
    )

    editor.keyframe_add_button.click()
    added = editor.orientation()
    assert isinstance(added, KeyframesOrientationSpec)
    assert tuple(keyframe.time_s for keyframe in added.keyframes) == (
        0.0,
        1.0,
        2.0,
        3.0,
    )
    assert added.keyframes[2].yaw_deg == 20.0

    editor.keyframe_remove_button.click()
    assert editor.orientation() == restored

    editor.apply_button.click()
    assert apply_requests == [True]
    assert editor.keyframes_table.cellWidget(0, 0).objectName() == "orientationKeyframe0TimeSpin"
    assert editor.keyframes_table.cellWidget(0, 1).objectName() == "orientationKeyframe0YawSpin"
    editor.close()


def test_keyframes_require_two_rows_before_returning(qapp) -> None:
    editor = OrientationEditor()
    editor.type_combo.setCurrentIndex(editor.type_combo.findData("keyframes"))
    editor._set_keyframe_rows((OrientationKeyframeSpec(time_s=0.0),))

    with pytest.raises(ValueError, match="at least two rows"):
        editor.orientation()
    editor.close()


def test_advanced_sections_start_collapsed_and_can_expand(qapp) -> None:
    editor = OrientationEditor()

    for section in (
        editor.align_advanced_section,
        editor.look_at_advanced_section,
        editor.random_advanced_section,
    ):
        assert not section.expanded
        assert section.content.isHidden()
        section.toggle.click()
        assert section.expanded
        assert not section.content.isHidden()
    editor.close()


def test_numeric_controls_are_named_adaptive_and_respect_read_only(qapp) -> None:
    editor = OrientationEditor()

    assert editor.objectName() == "orientationEditor"
    assert editor.type_combo.objectName() == "orientationTypeCombo"
    assert {editor.type_combo.itemData(index) for index in range(editor.type_combo.count())} == {
        kind.value for kind in OrientationKind
    }
    assert editor.align_motion_explanation.wordWrap()

    numeric_controls = (
        *editor.findChildren(QDoubleSpinBox),
        *editor.findChildren(QSpinBox),
    )
    assert numeric_controls
    for spin in numeric_controls:
        assert spin.stepType() == QAbstractSpinBox.StepType.AdaptiveDecimalStepType
        assert spin.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.UpDownArrows
        assert spin.objectName()

    editor.set_orientation(
        KeyframesOrientationSpec(
            keyframes=(
                OrientationKeyframeSpec(time_s=0.0),
                OrientationKeyframeSpec(time_s=1.0),
            )
        )
    )
    keyframe_spin = editor.keyframes_table.cellWidget(0, 0)
    assert isinstance(keyframe_spin, QDoubleSpinBox)

    editor.set_read_only(True)
    assert not editor.type_combo.isEnabled()
    assert not keyframe_spin.isEnabled()
    assert not editor.keyframe_add_button.isEnabled()
    assert not editor.apply_button.isEnabled()

    editor.set_read_only(False)
    assert editor.type_combo.isEnabled()
    assert keyframe_spin.isEnabled()
    assert editor.keyframe_add_button.isEnabled()
    assert editor.apply_button.isEnabled()
    editor.close()


def test_optional_controls_preserve_none_and_explicit_values(qapp) -> None:
    editor = OrientationEditor()

    editor.set_orientation(AlignMotionOrientationSpec())
    assert editor.orientation() == AlignMotionOrientationSpec()
    assert not editor.align_max_yaw_rate_enabled_check.isChecked()
    assert not editor.align_max_pitch_rate_enabled_check.isChecked()

    editor.set_orientation(RandomOrientationSpec(seed=9))
    random = editor.orientation()
    assert isinstance(random, RandomOrientationSpec)
    assert random.update_interval_s is None

    editor.random_update_interval_enabled_check.setChecked(True)
    editor.random_update_interval_spin.setValue(0.75)
    random = editor.orientation()
    assert isinstance(random, RandomOrientationSpec)
    assert random.update_interval_s == pytest.approx(0.75)
    editor.close()


def test_random_seed_editor_matches_the_lossless_shared_contract(qapp) -> None:
    editor = OrientationEditor()
    editor.set_orientation(RandomOrientationSpec(seed=MAX_RANDOM_SEED))

    assert editor.random_seed_spin.minimum() == 0
    assert editor.random_seed_spin.maximum() == MAX_RANDOM_SEED
    assert editor.orientation().seed == MAX_RANDOM_SEED
    editor.close()
