"""Recognize sensing evidence independently of whether live processing is enabled."""

from collections.abc import Mapping


def has_sensing_label_context(visualizer):
    """Volume overviews disable processing but still represent sensing scenarios."""
    sensing = getattr(getattr(visualizer, "scenario_config", None), "sensing", None)
    if isinstance(sensing, Mapping) and sensing.get("enabled", False):
        return True
    if getattr(visualizer, "_opening_sensing_volume_scene", False) is True:
        return True
    volume = getattr(visualizer, "sensing_volume_window", None)
    if (
        volume is not None
        and getattr(volume, "closed", True) is False
        and (
            getattr(volume, "reader", None) is not None
            or getattr(volume, "opening_scene", False) is True
        )
    ):
        return True
    fusion = getattr(visualizer, "fusion_window", None)
    return (
        fusion is not None
        and getattr(fusion, "closed", True) is False
        and (
            getattr(fusion, "reader", None) is not None
            or getattr(fusion, "volume_context", None) is not None
        )
    )
