import numpy as np
import pytest

from shared.sensing.range_response import (
    FrequencyNoise,
    RangeResponseConfig,
    add_frequency_noise,
    paths_to_frequency,
)


@pytest.mark.parametrize("shape", [(1, 64, 256), (2, 129, 512)])
@pytest.mark.parametrize("variance", [0.0, 0.003, 100.0])
def test_exact_native_stream_and_input_preserved(shape, variance):
    source = np.ones(shape, dtype=np.complex128) * (1 + 2j)
    expected = source.copy()
    rng = np.random.default_rng(np.random.SeedSequence([17, 3]))
    if variance:
        flat = expected.reshape(-1)
        for first in range(0, flat.size, 65536):
            count = min(65536, flat.size - first)
            draws = rng.standard_normal((count, 2))
            flat[first : first + count] += np.sqrt(variance / 2) * (draws[:, 0] + 1j * draws[:, 1])
    actual = add_frequency_noise(source, FrequencyNoise(variance, 17), realization_index=3)
    assert actual.tobytes() == expected.tobytes()
    assert np.all(source == 1 + 2j)
    assert not np.shares_memory(actual, source)


def test_empty_channel_injection():
    x = np.zeros((1, 64, 256), complex)
    assert np.any(add_frequency_noise(x, FrequencyNoise(1, 7)))
    assert not np.any(x)


def test_native_projection_calls_identical_injector():
    a = np.ones((1, 1, 8), complex)
    t = np.arange(8) / 1000
    clean, _ = paths_to_frequency(
        a,
        np.array([1e-7]),
        np.array([True]),
        t,
        bandwidth=1e8,
        config=RangeResponseConfig("cfr_ifft", 64),
    )
    noisy, _ = paths_to_frequency(
        a,
        np.array([1e-7]),
        np.array([True]),
        t,
        bandwidth=1e8,
        config=RangeResponseConfig("cfr_ifft", 64, "boxcar", FrequencyNoise(2, 9)),
        realization_index=4,
    )
    assert (
        noisy.tobytes()
        == add_frequency_noise(clean, FrequencyNoise(2, 9), realization_index=4).tobytes()
    )
