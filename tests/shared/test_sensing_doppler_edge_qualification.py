"""Independent checks of an isolated, non-production circular-Doppler candidate."""

import numpy as np
import pytest

from scripts.analysis.qualify_doppler_edges import circular_doppler_cfar, circular_reference
from shared.sensing.detection.cfar import nms_peaks_2d


@pytest.mark.parametrize("shape", [(16, 24), (32, 40), (64, 64)])
@pytest.mark.parametrize("kind", ["positive", "sparse", "dynamic"])
def test_circular_candidate_matches_direct_sums_and_never_wraps_range(shape, kind):
    rng = np.random.default_rng(12)
    power = rng.exponential(size=shape)
    if kind == "sparse":
        power = np.where(power > 2, power, 0)
    if kind == "dynamic":
        power *= 1e-200
        power[0, 15] = 1
    expected = circular_reference(power)
    actual = circular_doppler_cfar(
        power, guard_cells=(1, 2), reference_cells=(4, 6), threshold_scale=7.0
    )
    np.testing.assert_array_equal(actual.eligible, expected.eligible)
    np.testing.assert_array_equal(actual.training_cell_count, expected.training_cell_count)
    np.testing.assert_array_equal(actual.detections, expected.detections)
    np.testing.assert_allclose(actual.noise, expected.noise, rtol=2e-12, atol=0)
    assert not actual.eligible[:, :8].any()
    assert not actual.eligible[:, -8:].any()
    assert actual.eligible[0, 8:-8].all() and actual.eligible[-1, 8:-8].all()


def test_cfar_only_wrapping_does_not_solve_nms_seam_or_velocity_aliasing():
    power = np.zeros((64, 40))
    power[0, 20] = 100
    power[-1, 20] = 99
    cf = circular_doppler_cfar(
        power, guard_cells=(1, 2), reference_cells=(4, 6), threshold_scale=7.0
    )
    assert nms_peaks_2d(cf.detections, power).sum() == 2
    with pytest.raises(ValueError, match="repeat a physical Doppler cell"):
        circular_doppler_cfar(
            power[:8], guard_cells=(1, 2), reference_cells=(4, 6), threshold_scale=7.0
        )
