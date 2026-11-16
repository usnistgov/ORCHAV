"""Optional coverage availability evidence, independent of logarithmic RF layers.

A zero solver estimate is not proof of no physical signal. Legacy NaNs stay
unknown: the old producer discarded the distinction between zero and invalid.
"""

from enum import IntEnum

import numpy as np


class CoverageAvailability(IntEnum):
    UNKNOWN = 0
    POSITIVE = 1
    ZERO_CONTRIBUTION = 2
    INVALID = 3


AVAILABILITY_PATH = "availability/path_gain"
AVAILABILITY_VERSION = 1


def classify_solver_gain(raw_gain: np.ndarray) -> np.ndarray:
    """Classify before the producer masks nonpositive estimates for log metrics."""
    gain = np.asarray(raw_gain)
    result = np.full(gain.shape, CoverageAvailability.INVALID, dtype=np.uint8)
    result[np.isfinite(gain) & (gain > 0)] = CoverageAvailability.POSITIVE
    result[gain == 0] = CoverageAvailability.ZERO_CONTRIBUTION
    return result


def validate_availability(codes: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Reject inconsistent evidence rather than reinterpreting stored gains."""
    codes = np.asarray(codes)
    if (
        codes.dtype.kind not in "iu"
        or codes.shape != gain.shape
        or not np.isin(codes, list(CoverageAvailability)).all()
    ):
        raise ValueError("Invalid coverage availability codes or dimensions")
    positive = np.isfinite(gain) & (gain > 0)
    if not np.array_equal(codes == CoverageAvailability.POSITIVE, positive):
        raise ValueError("Coverage availability disagrees with positive gain")
    if np.any((codes == CoverageAvailability.ZERO_CONTRIBUTION) & ~(np.isnan(gain) | (gain == 0))):
        raise ValueError("Zero-contribution availability disagrees with gain")
    return codes.astype(np.uint8, copy=False)
