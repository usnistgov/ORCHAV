"""Decorate label snapshots without modifying actor state or backend objects."""

from dataclasses import replace
from functools import lru_cache

import numpy as np

from ..renderers.protocol import renderer_capabilities
from ..types.label_style import LabelCalloutStyle
from ..types.render_payloads import TextLabelPayload


def active_callout_style(visualizer) -> LabelCalloutStyle | None:
    """Return the screen-space style only when the renderer can display it."""
    style = LabelCalloutStyle.from_dict(
        getattr(getattr(visualizer, "app_state", None), "label_callout", None)
    )
    if (
        style.mode in ("callout", "card")
        and renderer_capabilities(getattr(visualizer, "renderer", None)).label_callouts
    ):
        return style
    return None


@lru_cache(maxsize=256)
def _callout_text_payload(payload: TextLabelPayload) -> TextLabelPayload:
    """Reuse immutable text; positions and display metadata stay per snapshot."""
    return replace(payload, screen_space=True, outline_thickness=0)


def label_with_callout(label, visualizer):
    """Decorate a label with the active renderer-supported style."""
    return label_with_callout_style(label, active_callout_style(visualizer))


def label_with_callout_style(label, style: LabelCalloutStyle | None):
    """Apply the shared style to a positioned actor/locator text snapshot.

    Plain style returns the original snapshot. This makes switching back fully
    reversible and keeps the original text colour, size and world offset.
    """
    if not isinstance(label.payload, TextLabelPayload):
        return label
    if style is None or style.mode == "plain":
        return label
    anchor = label.metadata.get("layout_anchor", label.metadata.get("label_connector_origin"))
    if anchor is None:
        return label
    anchor = np.asarray(anchor, dtype=float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        return label
    is_tx = (
        label.metadata.get("locator_role") == "radio"
        or label.id.startswith("node:tx_")
        or "transmitter_label" in label.id
    )
    if label.material_payload is None:
        return label
    color = np.asarray(label.material_payload.base_color[:3], dtype=float)
    white = style.background == "white" or (style.background == "paper" and is_tx)
    # A light tint preserves identity while keeping black text readable.
    rgb = np.ones(3) if white else 0.7 + 0.3 * np.clip(color, 0, 1)
    text_color = (0, 0, 0, 1)
    if style.mode == "card":
        # Subtle node tints and navy text remain legible against busy geometry.
        rgb = np.ones(3) if white else 0.90 + 0.10 * np.clip(color, 0, 1)
        text_color = (0.125, 0.231, 0.420, 1)
    if style.background == "ink":
        rgb = np.array([0.075, 0.125, 0.215])
        text_color = (0.97, 0.98, 1, 1)
    options = style.to_dict()
    for kind in ("tx", "rx"):
        if label.id.startswith(f"node:{kind}_") or label.metadata.get("node_kind") == kind:
            options["marker_size"] = style.node_marker_size(kind)
            break
    options["base_position"] = label.transform_matrix[:3, 3].tolist()
    if style.mode == "card":
        options["accent_color"] = (*np.clip(color, 0, 1), 1)
    return replace(
        label,
        payload=_callout_text_payload(label.payload),
        material=replace(label.material_payload, base_color=text_color),
        metadata={
            **label.metadata,
            "label_background_rgba": (*rgb, style.opacity),
            "label_connector_origin": anchor.tolist(),
            "label_callout": options,
            "label_bold": is_tx or style.mode == "card",
            "label_anchor_offset": 10.0,
            "depth_test": not style.on_top,
            "depth_write": False,
            "alpha_mode": "blend",
            "render_order": 31000 if style.on_top else 0,
        },
    )


def objects_with_callouts(objects, visualizer):
    """Replace legacy locator stems/points only when a native callout owns them."""
    decorated = [label_with_callout(obj, visualizer) for obj in objects]
    replaced = {
        name
        for obj in decorated
        if "label_callout" in obj.metadata
        for name in obj.metadata.get("locator_companions", ())
    }
    return [replace(obj, visibility=False) if obj.id in replaced else obj for obj in decorated]
