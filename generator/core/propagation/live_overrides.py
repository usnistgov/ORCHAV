"""Normalize live actor overrides before propagation scene assignment.

Live overrides are request-time changes from interactive clients. They are
applied to the mutable per-frame actor-state lists before those lists are
written onto Sionna objects. This keeps the prepared actor-state cache unchanged
while allowing a single frame request to use adjusted TX/RX/target state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from shared.logging import get_logger

from ..mobility.base import Position3
from ..orientation.base import Orientation3, orientation_to_tuple
from ..utils import point_to_tuple, to_float

logger = get_logger(__name__)

LiveActorCategory: TypeAlias = Literal["tx", "rx", "target"]


@dataclass(frozen=True)
class LiveActorOverride:
    """One normalized live-state override keyed by category and actor name.

    Position and orientation apply to TX, RX, and targets. Scale is retained
    only for targets because TX/RX scene assignment has no scale field.
    """

    name: str
    category: LiveActorCategory
    position: Position3 | None = None
    orientation: Orientation3 | None = None
    scale: float | None = None


LiveOverrideMap: TypeAlias = dict[LiveActorCategory, dict[str, LiveActorOverride]]


def _empty_override_map() -> LiveOverrideMap:
    return {"tx": {}, "rx": {}, "target": {}}


# visualizer.proto NodeType wire values. Keep these as local named constants so
# propagation can parse protobuf-like payloads without importing generated gRPC
# modules at package import time.
_NODE_TYPE_TX = 1
_NODE_TYPE_RX = 2
_NODE_TYPE_TARGET = 3

TYPE_TO_CATEGORY: dict[int, LiveActorCategory] = {
    _NODE_TYPE_TX: "tx",
    _NODE_TYPE_RX: "rx",
    _NODE_TYPE_TARGET: "target",
}


def category_from_value(value: Any) -> LiveActorCategory | None:
    """Resolve a visualizer/string category value to a propagation category."""
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("tx", "transmitter"):
            return "tx"
        if lowered in ("rx", "receiver"):
            return "rx"
        if lowered in ("target", "object"):
            return "target"
        return None
    try:
        return TYPE_TO_CATEGORY.get(int(value))
    except (TypeError, ValueError):
        return None


def _clean_name(value: Any) -> str | None:
    if value is None:
        return None
    name = str(value).strip()
    return name or None


def _position_from_values(values: Any, name: str | None) -> Position3 | None:
    if isinstance(values, Mapping) and all(key in values for key in ("x", "y", "z")):
        values = (values.get("x"), values.get("y"), values.get("z"))
    try:
        return point_to_tuple(values, error_type=ValueError, attribute_converter=to_float)
    except (TypeError, ValueError, AttributeError) as exc:
        logger.warning("Override %s has invalid position %s: %s", name, values, exc)
        return None


def _position_from_mapping(override: Mapping[str, Any], name: str | None) -> Position3 | None:
    position_values = override.get("position")
    if position_values is not None:
        return _position_from_values(position_values, name)
    if all(key in override for key in ("x", "y", "z")):
        return _position_from_values(
            (override.get("x"), override.get("y"), override.get("z")),
            name,
        )
    return None


def _has_message_field(value: Any, field_name: str) -> bool:
    """Return whether a protobuf-like or plain object exposes a populated field."""
    has_field = getattr(value, "HasField", None)
    if callable(has_field):
        try:
            return bool(has_field(field_name))
        except (TypeError, ValueError):
            return getattr(value, field_name, None) is not None
    return getattr(value, field_name, None) is not None


def _position_from_object(override: Any, name: str | None) -> Position3 | None:
    if _has_message_field(override, "position"):
        pos = getattr(override, "position", None)
        if pos is not None:
            return _position_from_values(
                (getattr(pos, "x", None), getattr(pos, "y", None), getattr(pos, "z", None)),
                name,
            )
    if all(hasattr(override, attr) for attr in ("x", "y", "z")):
        return _position_from_values((override.x, override.y, override.z), name)
    return None


def _orientation_from_values(values: Any, name: str | None) -> Orientation3 | None:
    if values is None:
        return None
    try:
        return orientation_to_tuple(values)
    except (TypeError, ValueError, AttributeError) as exc:
        logger.warning("Override %s has invalid orientation %s: %s", name, values, exc)
        return None


def _scale_from_value(value: Any, name: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Override %s has invalid scale %s", name, value)
        return None


def normalize_live_overrides(live_overrides: Sequence[Any] | None) -> LiveOverrideMap:
    """Normalize override payloads by category and lower-cased actor name."""
    normalized = _empty_override_map()
    if not live_overrides:
        return normalized

    for override in live_overrides:
        try:
            if isinstance(override, Mapping):
                name = _clean_name(override.get("name"))
                category = category_from_value(override.get("type") or override.get("category"))
                position = _position_from_mapping(override, name)
                orientation = _orientation_from_values(override.get("orientation"), name)
                scale = _scale_from_value(override.get("scale"), name)
            else:
                name = _clean_name(getattr(override, "name", None))
                raw_category = getattr(override, "category", None) or getattr(
                    override, "type", None
                )
                category = category_from_value(raw_category)
                position = _position_from_object(override, name)
                orientation = _orientation_from_values(
                    getattr(override, "orientation", None),
                    name,
                )
                scale = _scale_from_value(getattr(override, "scale", None), name)

            if category not in normalized or not name:
                logger.warning("Ignoring override with invalid category/name: %s", override)
                continue

            if scale is not None and category != "target":
                logger.debug("Ignoring non-target scale override for %s", name)
                scale = None

            normalized[category][name.lower()] = LiveActorOverride(
                name=name,
                category=category,
                position=position,
                orientation=orientation,
                scale=scale,
            )
        except (TypeError, ValueError, AttributeError, KeyError) as exc:
            logger.warning("Failed to process override %s: %s", override, exc)

    return normalized


def apply_live_overrides(
    live_overrides: Sequence[Any] | None,
    tx_configs: list[Any],
    rx_configs: list[Any],
    target_configs: list[Any],
    tx_positions: list[Any],
    rx_positions: list[Any],
    target_positions: list[Any],
    tx_orientations: list[Any],
    rx_orientations: list[Any],
    target_orientations: list[Any],
) -> LiveOverrideMap:
    """Apply live overrides to mutable per-frame actor-state lists.

    This function does not touch Sionna scene objects. It edits the plain Python
    position/orientation lists that ``compute_ray_tracing_step`` later passes to
    ``apply_transceiver_state_to_scene`` and ``apply_target_state_to_scene``.
    """
    override_map = normalize_live_overrides(live_overrides)
    applied = _empty_override_map()

    def _apply(
        category: LiveActorCategory,
        configs: list[Any],
        positions: list[Any],
        orientations: list[Any],
    ) -> None:
        overrides_for_category = override_map.get(category, {})
        if not overrides_for_category:
            return

        unmatched_keys = set(overrides_for_category.keys())
        limit = min(len(configs), len(positions))
        for idx in range(limit):
            cfg = configs[idx]
            cfg_name = str(getattr(cfg, "name", f"{category}_{idx + 1}"))
            possible_keys = [cfg_name.lower(), f"{category}_{idx + 1}", f"{category}{idx + 1}"]
            override_data = None
            matched_key = None
            for key in possible_keys:
                key_l = key.lower()
                if key_l in overrides_for_category:
                    override_data = overrides_for_category[key_l]
                    matched_key = key_l
                    break

            if override_data is None:
                continue

            if matched_key is not None:
                unmatched_keys.discard(matched_key)
            unmatched_keys.discard(cfg_name.lower())

            if override_data.position is not None:
                positions[idx] = override_data.position
                logger.debug(
                    "Applied %s %s position override -> %s",
                    category.upper(),
                    cfg_name,
                    override_data.position,
                )

            if (
                override_data.orientation is not None
                and orientations is not None
                and idx < len(orientations)
            ):
                orientations[idx] = override_data.orientation

            applied[category][cfg_name] = LiveActorOverride(
                name=cfg_name,
                category=category,
                position=override_data.position,
                orientation=override_data.orientation,
                scale=override_data.scale,
            )

        for override_name in unmatched_keys:
            original = overrides_for_category[override_name].name
            logger.warning("Override references unknown %s '%s'", category.upper(), original)

    override_targets: tuple[
        tuple[LiveActorCategory, list[Any], list[Any], list[Any]],
        ...,
    ] = (
        ("tx", tx_configs, tx_positions, tx_orientations),
        ("rx", rx_configs, rx_positions, rx_orientations),
        ("target", target_configs, target_positions, target_orientations),
    )
    for category, cfg_list, pos_list, ori_list in override_targets:
        _apply(category, cfg_list, pos_list, ori_list)

    return applied
