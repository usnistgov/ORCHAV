"""Application-owned visibility policy for persistent scene entries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .target_runtime import TargetHiddenPredicate, target_label_visible, target_runtime_visible


def effective_entry_visibility(
    entry: Mapping[str, Any],
    *,
    state_visible: bool = True,
    entry_type: str | None = None,
    target_index: Any = None,
    is_hidden_for_pov: TargetHiddenPredicate | None = None,
    frame_visible: bool | None = None,
) -> bool:
    """Return final renderer visibility without mutating semantic state."""
    kind = str(entry_type or entry.get("entry_type", "mesh")).lower()
    if kind == "target":
        return bool(
            state_visible
            and target_runtime_visible(
                entry,
                target_index,
                is_hidden_for_pov,
                frame_visible=frame_visible,
            )
        )
    return bool(state_visible and entry.get("visible", True))


def effective_entry_label_visibility(
    entry: Mapping[str, Any],
    *,
    labels_enabled: bool = True,
    state_visible: bool = True,
    entry_type: str | None = None,
    target_index: Any = None,
    is_hidden_for_pov: TargetHiddenPredicate | None = None,
    frame_visible: bool | None = None,
) -> bool:
    """Return final label visibility including global, local, and parent policy."""
    kind = str(entry_type or entry.get("entry_type", "mesh")).lower()
    if kind == "target":
        return bool(
            state_visible
            and target_label_visible(
                entry,
                target_index,
                is_hidden_for_pov,
                show_target_labels=labels_enabled,
                runtime_visible=effective_entry_visibility(
                    entry,
                    state_visible=state_visible,
                    entry_type="target",
                    target_index=target_index,
                    is_hidden_for_pov=is_hidden_for_pov,
                    frame_visible=frame_visible,
                ),
            )
        )
    return bool(
        labels_enabled
        and entry.get("show_label", True)
        and effective_entry_visibility(
            entry,
            state_visible=state_visible,
            entry_type=entry_type,
            target_index=target_index,
            is_hidden_for_pov=is_hidden_for_pov,
            frame_visible=frame_visible,
        )
    )
