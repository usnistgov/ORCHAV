"""Coverage metric conversion helpers.

Sionna RT's ``RadioMapSolver`` returns per-transmitter path gain in linear power
units. The coverage solver stores that raw layer and derives user-facing layers
such as path loss, RSS, SINR, and serving transmitter from it. These helpers
centralize the unit conversions so dB conventions and invalid-value handling
stay consistent across stored and displayed metrics.
"""

from typing import Any

import numpy as np

# Coverage metrics are power quantities, so dB conversion uses 10*log10(.).
DB_POWER_SCALE = 10.0


def _db_from_linear(values: np.ndarray) -> np.ndarray:
    """Convert linear power values to dB and map non-finite results to NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = DB_POWER_SCALE * np.log10(values)
    return np.where(np.isfinite(result), result, np.nan).astype(np.float32)


def _path_loss_db(path_gain: np.ndarray) -> np.ndarray:
    """Convert linear path gain to positive path loss in dB."""
    return (-_db_from_linear(path_gain)).astype(np.float32)


def _tx_power_dbm(tx: Any) -> float:
    """Return transmitter power in dBm, preferring ``power_dbm`` over ``power``."""
    for attr in ("power_dbm", "power"):
        value = getattr(tx, attr, None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0
