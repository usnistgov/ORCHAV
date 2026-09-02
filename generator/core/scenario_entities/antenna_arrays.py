"""Sionna RT antenna array construction helpers for radio devices.

TX/RX device configs can carry antenna settings that must become Sionna
``PlanarArray`` objects before scene creation. The helper lives in
``scenario_entities`` because it sits on the same config-to-runtime boundary as
device construction, while targets do not use antenna arrays.
"""

from sionna.rt import PlanarArray

from ..configuration import AntennaConfig
from ..configuration.defaults import (
    DEFAULT_ANTENNA_NUM_COLS,
    DEFAULT_ANTENNA_NUM_ROWS,
    DEFAULT_ANTENNA_PATTERN,
    DEFAULT_ANTENNA_POLARIZATION,
)


def create_planar_array(config: AntennaConfig | None = None) -> PlanarArray:
    """Create a Sionna ``PlanarArray`` from an ``AntennaConfig``.

    Args:
        config: Antenna array configuration. If ``None``, returns a default
            1x1 isotropic vertically-polarized array.

    Returns:
        Configured ``PlanarArray`` instance.
    """
    if config is None:
        return PlanarArray(
            num_rows=DEFAULT_ANTENNA_NUM_ROWS,
            num_cols=DEFAULT_ANTENNA_NUM_COLS,
            pattern=DEFAULT_ANTENNA_PATTERN,
            polarization=DEFAULT_ANTENNA_POLARIZATION,
        )
    return PlanarArray(
        num_rows=config.num_rows,
        num_cols=config.num_cols,
        pattern=config.pattern,
        polarization=config.polarization,
        vertical_spacing=config.vertical_spacing,
        horizontal_spacing=config.horizontal_spacing,
    )
