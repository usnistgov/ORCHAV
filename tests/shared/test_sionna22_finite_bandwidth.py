"""Optional full-Sionna 2.2 comparison; ordinary ORCHAV installs need no PHY.

ORCHAV_RUN_SIONNA22=1 python -m pytest --no-cov tests/shared/test_sionna22_finite_bandwidth.py
Add ORCHAV_RUN_SIONNA22_LIVE=1 for CUDA RT controls. An isolated PHY installation
can be selected through PYTHONPATH; the normal RT 2.0.x environment is untouched.
"""

import os

import numpy as np
import pytest

from scripts.analysis.validate_sionna22_finite_bandwidth import (
    C0,
    CARRIER,
    SAMPLES,
    WINDOWS,
    PathCase,
    deterministic_cases,
    project_case,
    qualify_live,
    qualify_transform,
    reference_api,
    replay,
    sinc_comparison,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("ORCHAV_RUN_SIONNA22") != "1",
    reason="Optional independent comparison requires full Sionna PHY 2.2; set ORCHAV_RUN_SIONNA22=1",
)


@pytest.fixture(scope="module")
def api():
    if os.environ.get("ORCHAV_RUN_SIONNA22_LIVE") == "1":
        import mitsuba as mi

        mi.set_variant("cuda_ad_mono_polarized")
    return reference_api()


@pytest.mark.parametrize("case", deterministic_cases(), ids=lambda c: c.name)
@pytest.mark.parametrize("windows", WINDOWS, ids=lambda w: "_".join(w))
@pytest.mark.parametrize("size", [128, 512])
def test_complete_complex_cfr_delay_doppler(case, windows, size, api):
    observation, _, _ = project_case(case, api)
    result = qualify_transform(observation, *windows, size, api)
    if windows == ("boxcar", "boxcar") and size == 128:
        assert result["sionna_to_orchav_complex_scale"] == np.sqrt(SAMPLES)
        assert result["orchav_total_power"] / result["sionna_total_power"] == pytest.approx(SAMPLES)
    if case.name == "moving_off_grid":
        assert result["peak_velocities_m_s"][0] == pytest.approx(
            1.8, abs=C0 / (2 * CARRIER) * 1000 / 64 / 2
        )
    if case.name.startswith("stationary"):
        assert result["peak_velocities_m_s"] == [0]


def test_invalid_padding_has_identical_validity_for_both_projectors(api):
    original = deterministic_cases()[2]
    extended = PathCase(
        "invalid_padding",
        np.concatenate([original.coefficients, np.full_like(original.coefficients, 1e10j)], axis=1),
        np.r_[original.delays, -1],
        np.array([True, False]),
        np.r_[original.dopplers, 0],
    )
    actual, reference, _ = project_case(extended, api)
    expected, _, _ = project_case(original, api)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(reference, expected, atol=1e-14, rtol=1e-14)


def test_sinc_is_a_distinct_nonperiodic_model_with_expected_limit(api):
    results = sinc_comparison(api)
    rows = results["convergence_fixed_bandwidth"]
    assert (
        rows[2]["mainlobe_max_complex_difference"]
        < rows[1]["mainlobe_max_complex_difference"]
        < rows[0]["mainlobe_max_complex_difference"]
    )
    assert results["periodic_magnitude_at_zero"] > 0.8
    assert results["sinc_magnitude_at_zero"] < 0.003


def test_known_receive_element_phases_survive_both_transforms(api):
    case = next(c for c in deterministic_cases() if c.name == "four_receive_elements")
    observation, _, _ = project_case(case, api)
    rd = replay(observation, "hann", "hann", 512)["element_rd"]
    expected = np.exp(0.7j * np.arange(4))[:, None, None] * rd[0]
    np.testing.assert_allclose(rd, expected, atol=4e-15, rtol=4e-15)


@pytest.mark.skipif(
    os.environ.get("ORCHAV_RUN_SIONNA22_LIVE") != "1",
    reason="Set ORCHAV_RUN_SIONNA22_LIVE=1 with an exposed CUDA driver for actual RT 2.2 paths",
)
@pytest.mark.parametrize("kind", ["constant_rcs", "specular_mesh"])
def test_live_common_paths_preserve_cfr_and_delay_doppler(kind, api, tmp_path):
    result = qualify_live(kind, api, tmp_path)
    assert result["variant"] == "cuda_ad_mono_polarized"
    assert result["valid_paths"] > 0
