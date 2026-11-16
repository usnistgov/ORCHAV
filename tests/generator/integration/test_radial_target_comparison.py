"""Opt-in RT 2.2 controls for the radial response investigation."""

import os

import numpy as np
import pytest

from scripts.analysis.validate_range_doppler_targets import acquire, analyze

pytestmark = [
    pytest.mark.integration,
    pytest.mark.optional_runtime,
    pytest.mark.skipif(
        os.environ.get("ORCHAV_RUN_LIVE_RCS") != "1",
        reason="Set ORCHAV_RUN_LIVE_RCS=1 with Sionna RT 2.2 and a usable polarized backend",
    ),
]


@pytest.mark.parametrize("cpi", range(3))
def test_native_constant_rcs_matches_point_shape_power_and_floor_projection(cpi):
    record, _ = analyze(acquire("constant_radial", cpi))
    assert record["path_count"] == 1
    point = record["single_point_reference"]
    assert point["complex_shape_relative_error"] < 1e-4
    assert point["radar_equation_relative_error"] < 3e-5
    assert record["cfr_native_backend_relative_error"] < 2e-5
    assert record["replay_generator_relative_error"] < 1e-14
    assert record["direct_generator_relative_error"] < 3e-14
    assert record["windows"]["boxcar/boxcar"]["zero_at_target_relative_db"] == pytest.approx(
        -31.3322764, abs=0.01
    )
    assert record["windows"]["boxcar/hann"]["zero_at_target_relative_db"] < -85
    assert record["windows"]["hann/boxcar"]["far_range_peak_relative_db"] < -55
    assert record["point_bin_floor"]["occupied_bins"] == 1
    assert record["point_bin_floor"]["projection_relative_error"] == 0
    assert record["point_bin_floor"]["profile_relative_error"] == 0


def test_original_off_axis_cube_needs_diffuse_paths_and_face_normal_control_returns():
    for cpi in range(3):
        no_diffuse = acquire("cube_specular", cpi)
        assert len(no_diffuse["tau"]) == 0
    aligned = acquire("cube_specular_aligned")
    assert len(aligned["tau"]) == 1
    center = np.linalg.norm(aligned["position"] - np.array([0, 0, 0.5]))
    assert aligned["tau"][0] * 299792458 / 2 == pytest.approx(center - 0.5, abs=2e-6)
    diffuse, _ = analyze(acquire("cube_diffuse"))
    assert diffuse["path_count"] > 1
    assert diffuse["point_bin_floor"]["occupied_bins"] == 4
    assert diffuse["point_bin_floor"]["projection_relative_error"] == 0
