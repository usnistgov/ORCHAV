"""Serializable, renderer-neutral settings for optional bloom."""

import math

DEFAULT_BLOOM = dict(
    enabled=False,
    bloom_strength=0.035,
    max_mip_levels=6,
    filter_radius=0.005,
    use_karis_average=False,
)


def validate_bloom(settings: dict) -> dict:
    """Validate a complete bloom recipe before changing renderer state."""
    if not isinstance(settings, dict) or set(settings) != set(DEFAULT_BLOOM):
        raise ValueError("Invalid bloom settings")
    for key in ("enabled", "use_karis_average"):
        if not isinstance(settings[key], bool):
            raise ValueError(f"Invalid bloom flag: {key}")
    for key, lo, hi in (("bloom_strength", 0, 3), ("filter_radius", 0, 0.05)):
        value = settings[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not lo <= value <= hi
        ):
            raise ValueError(f"Invalid bloom value: {key}")
    if type(settings["max_mip_levels"]) is not int or not 1 <= settings["max_mip_levels"] <= 10:
        raise ValueError("Bloom mip levels must be between 1 and 10")
    return dict(settings)
