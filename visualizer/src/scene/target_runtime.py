"""Renderer-neutral target runtime visibility helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

TargetHiddenPredicate = Callable[[str, int], bool]


def _is_target_hidden_for_pov(
    target_index: Any,
    is_hidden_for_pov: TargetHiddenPredicate | None,
) -> bool:
    """Return whether the target index is hidden by the current POV camera."""
    if target_index is None or is_hidden_for_pov is None:
        return False
    try:
        index = int(target_index)
    except (TypeError, ValueError):
        return False
    return bool(is_hidden_for_pov("target", index))


def target_runtime_visible(
    entry: Mapping[str, Any],
    target_index: Any,
    is_hidden_for_pov: TargetHiddenPredicate | None = None,
    *,
    frame_visible: bool | None = None,
) -> bool:
    """Return whether a target mesh should be visible for the current frame."""
    active_frame_visible = (
        bool(frame_visible)
        if frame_visible is not None
        else bool(entry.get("_frame_visible", True))
    )
    return bool(
        entry.get("visible", True)
        and active_frame_visible
        and not _is_target_hidden_for_pov(target_index, is_hidden_for_pov)
    )


def target_label_visible(
    entry: Mapping[str, Any],
    target_index: Any,
    is_hidden_for_pov: TargetHiddenPredicate | None = None,
    *,
    show_target_labels: bool = True,
    runtime_visible: bool | None = None,
) -> bool:
    """Return whether a target label should be visible for current UI state.

    When a caller does not already have the final target visibility, derive it
    from the same semantic, frame-presence, and POV policy as the target mesh.
    This keeps global/per-entry label controls subordinate to their parent
    target instead of allowing a hidden target to leave an orphan label.
    """
    parent_visible = (
        target_runtime_visible(entry, target_index, is_hidden_for_pov)
        if runtime_visible is None
        else bool(runtime_visible)
    )
    return bool(
        parent_visible
        and show_target_labels
        and entry.get("show_label", True)
        and not _is_target_hidden_for_pov(target_index, is_hidden_for_pov)
    )
