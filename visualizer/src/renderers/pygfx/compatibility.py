"""Runtime checks for features tied to pygfx's backend shader contracts."""

from functools import lru_cache

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

# Match the runtime dependency pin: the ISO adapters specialize 0.17 WGSL.
VOLUME_PYGFX_REQUIREMENT = ">=0.17,<0.18"
_VOLUME_VERSIONS = SpecifierSet(VOLUME_PYGFX_REQUIREMENT)


@lru_cache(maxsize=8)
def volume_unavailable_reason(version: str) -> str:
    """Return an actionable reason, or an empty string for a supported runtime."""
    try:
        supported = Version(version) in _VOLUME_VERSIONS
    except InvalidVersion:
        supported = False
    if supported:
        return ""
    return (
        f"3D volumes require pygfx {VOLUME_PYGFX_REQUIREMENT}; "
        f"this Python environment has {version or 'an unknown version'}. "
        "Activate a supported environment or install this checkout's dependencies "
        "with: python -m pip install -e ."
    )
