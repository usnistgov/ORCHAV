"""Display-bound validation preserves acquisition-derived defaults."""

import numpy as np
import pytest

from generator.core.configuration import SensingConfig
from generator.core.exceptions import ConfigurationError
from shared.sensing.recording import add_display_products


@pytest.mark.parametrize("bounds", [{"range_min_m": 150.0}, {"range_max_m": 150.0}])
def test_one_sided_display_bound_uses_acquired_span(bounds):
    config = SensingConfig(
        enabled=True,
        recording_profile="maps",
        recording_display=bounds,
        bandwidth=100e6,
        fft_size_range=128,
        fft_size_doppler=4,
    )
    block = {}
    ranges = np.array([0.0, 80.0, 160.0, 240.0])
    add_display_products(
        block,
        config,
        samples=np.ones((4, 4), dtype=complex),
        times_s=np.arange(4) / config.prf_hz,
        range_axis_m=ranges,
    )
    recipe = next(s for s in block["_product_specs"] if s["kind"] == "micro_doppler")["processing"][
        "recipe"
    ]
    assert recipe["range_min_m"] == bounds.get("range_min_m", ranges[0])
    assert recipe["range_max_m"] == bounds.get("range_max_m", ranges[-1])
    assert config.recording_display == bounds


@pytest.mark.parametrize(
    "bounds",
    [
        {"range_min_m": 200.0, "range_max_m": 100.0},
        {"range_min_m": np.inf},
        {"range_max_m": np.nan},
    ],
)
def test_invalid_explicit_display_bounds_still_fail_preflight(bounds):
    with pytest.raises(ConfigurationError, match="Range limits"):
        SensingConfig(recording_profile="maps", recording_display=bounds)
