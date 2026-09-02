from __future__ import annotations

from visualizer.src.scene.target_runtime import target_label_visible, target_runtime_visible
from visualizer.src.scene.visibility_policy import (
    effective_entry_label_visibility,
    effective_entry_visibility,
)


def _hide_target(node_type: str, index: int) -> bool:
    return node_type == "target" and index == 2


def test_target_runtime_visible_combines_user_frame_and_pov_state() -> None:
    entry = {"visible": True, "_frame_visible": True}

    assert target_runtime_visible(entry, 1, _hide_target)
    assert not target_runtime_visible(entry, 2, _hide_target)
    assert not target_runtime_visible({"visible": False, "_frame_visible": True}, 1, _hide_target)
    assert not target_runtime_visible({"visible": True, "_frame_visible": False}, 1, _hide_target)


def test_target_runtime_visible_can_use_explicit_frame_visibility() -> None:
    entry = {"visible": True, "_frame_visible": False}

    assert target_runtime_visible(entry, 1, _hide_target, frame_visible=True)
    assert not target_runtime_visible(entry, 1, _hide_target, frame_visible=False)


def test_target_label_visible_combines_global_entry_and_pov_state() -> None:
    entry = {"show_label": True}

    assert target_label_visible(entry, 1, _hide_target, show_target_labels=True)
    assert not target_label_visible(entry, 2, _hide_target, show_target_labels=True)
    assert not target_label_visible(entry, 1, _hide_target, show_target_labels=False)
    assert not target_label_visible({"show_label": False}, 1, _hide_target)


def test_target_label_visible_can_require_runtime_visibility() -> None:
    entry = {"show_label": True}

    assert target_label_visible(entry, 1, _hide_target, runtime_visible=True)
    assert not target_label_visible(entry, 1, _hide_target, runtime_visible=False)


def test_target_label_visibility_inherits_parent_semantic_and_frame_state() -> None:
    entry = {"visible": False, "_frame_visible": True, "show_label": True}
    assert not target_label_visible(entry, 1, _hide_target)

    entry["visible"] = True
    entry["_frame_visible"] = False
    assert not target_label_visible(entry, 1, _hide_target)

    entry["_frame_visible"] = True
    assert target_label_visible(entry, 1, _hide_target)


def test_shared_entry_visibility_policy_covers_scene_and_target_parents() -> None:
    scene_entry = {"entry_type": "mesh", "visible": False, "show_label": True}
    assert not effective_entry_visibility(scene_entry)
    assert not effective_entry_label_visibility(scene_entry)

    target_entry = {
        "entry_type": "target",
        "visible": True,
        "_frame_visible": False,
        "show_label": True,
    }
    assert not effective_entry_visibility(
        target_entry,
        target_index=1,
        is_hidden_for_pov=_hide_target,
    )
    assert not effective_entry_label_visibility(
        target_entry,
        labels_enabled=True,
        target_index=1,
        is_hidden_for_pov=_hide_target,
    )


def test_shared_label_policy_combines_global_local_and_parent_visibility() -> None:
    entry = {"entry_type": "mesh", "visible": True, "show_label": True}
    assert effective_entry_label_visibility(entry, labels_enabled=True)
    assert not effective_entry_label_visibility(entry, labels_enabled=False)

    entry["show_label"] = False
    assert not effective_entry_label_visibility(entry, labels_enabled=True)
