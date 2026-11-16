"""Independent path identity and physical-coordinate checks through coherent processing."""

from types import SimpleNamespace

import numpy as np
import pytest

from generator.core.configuration import AntennaConfig, SensingConfig
from generator.core.sensing.acquisition import resolve_sensing_execution_values
from generator.core.sensing.processor import create_sensing_processor
from shared.sensing.range_response import RangeResponseConfig
from shared.sensing.signal_processing import apply_aoa_filter


def _processor(model, *, elements=1, **overrides):
    config = SensingConfig(
        enabled=True,
        processing_mode="coherent_cpi",
        bandwidth=100e6,
        prf_hz=1000,
        fft_size_range=64,
        fft_size_doppler=32,
        window_range="boxcar",
        window_doppler="hann",
        range_mode="monostatic",
        raw_payload_policy="always",
        range_response=RangeResponseConfig(model, 64 if model == "cfr_ifft" else None),
        **overrides,
    )
    execution = resolve_sensing_execution_values(
        config,
        cir_time_steps=32,
        cir_sampling_frequency_hz=1000,
        frame_dt_s=0.001,
        rx_elements=elements,
    )
    return create_sensing_processor(config, 28e9, execution)


@pytest.mark.parametrize("model", ["point_bin_floor", "cfr_ifft"])
@pytest.mark.parametrize("elements", [1, 4])
@pytest.mark.parametrize("bad_index", [0, 1, 2])
@pytest.mark.parametrize("bad_field", ["gain", "delay"])
@pytest.mark.parametrize("angle", ["az", "el"])
def test_bad_path_removal_preserves_original_angle_selection(
    model, elements, bad_index, bad_field, angle
):
    processor = _processor(
        model,
        elements=elements,
        output_detections=False,
        aoa_filter_enabled=True,
        **{f"aoa_{angle}_min_deg": -10.0, f"aoa_{angle}_max_deg": 10.0},
    )
    tone = np.exp(2j * np.pi * 125 * np.arange(32) / 1000)
    gains = np.array([1.0, 2.0, 3.0])[:, None] * tone
    tau = np.array([10.25, 20.25, 30.25]) / 100e6
    if bad_field == "gain":
        gains[bad_index] = np.nan
    else:
        tau[bad_index] = np.nan
    angles = np.array([0.0, 90.0, 0.0])
    paths = SimpleNamespace(
        phi_r=np.deg2rad(angles if angle == "az" else np.zeros(3))[None, None],
        theta_r=np.deg2rad(90 - (angles if angle == "el" else np.zeros(3)))[None, None],
    )
    simulation = SimpleNamespace(
        simulation_config=SimpleNamespace(
            rx_antenna=AntennaConfig(num_rows=1, num_cols=elements),
        )
    )
    a = np.broadcast_to(gains, (elements, 3, 32)).copy()
    result = processor.process(
        simulation,
        0,
        paths,
        cir_stack=(a[None, :, None, None], tau[None, None]),
        source_time_s=0.0,
    )
    selected = (np.arange(3) != bad_index) & (angles == 0)
    if model == "point_bin_floor":
        expected = np.zeros((32, 64), dtype=complex)
        for index in np.flatnonzero(selected):
            expected[:, int(tau[index] * 100e6)] = gains[index]
        actual = result["slow_time_by_pair_rx_element"][0]
    else:
        frequencies = (np.arange(64) - 32) * 100e6 / 64
        expected = sum(
            (
                gains[index, :, None] * np.exp(-2j * np.pi * frequencies * tau[index])[None, :]
                for index in np.flatnonzero(selected)
            ),
            start=np.zeros((32, 64), dtype=complex),
        )
        actual = result["cfr_by_pair_rx_element"][0]
    np.testing.assert_allclose(
        actual, np.broadcast_to(expected, actual.shape), atol=2e-13, rtol=2e-13
    )


@pytest.mark.parametrize("rows,uy,uz", [(1, -0.5, 0.0), (4, 0.5, 0.5), (4, -0.5, -0.5)])
def test_point_bin_array_report_has_physical_spherical_angles(rows, uy, uz):
    elements = rows * 4
    processor = _processor("point_bin_floor", elements=elements, output_detections=True)
    steering = np.array(
        [
            np.exp(1j * np.pi * (col * uy - row * uz)) + 0.05
            for col in range(4)
            for row in range(rows)
        ]
    )
    a = steering[:, None, None] * np.exp(2j * np.pi * 123 * np.arange(32) / 1000)[None, None]
    simulation = SimpleNamespace(
        simulation_config=SimpleNamespace(
            rx_antenna=AntennaConfig(num_rows=rows, num_cols=4),
        )
    )
    result = processor.process(
        simulation,
        0,
        None,
        cir_stack=(a[None, :, None, None], np.array([[[20.3 / 100e6]]])),
        source_time_s=0.0,
    )
    assert result["detections"]
    peak = max(result["detections"], key=lambda d: d["power_linear"])
    assert peak["aoa_az_deg"] == pytest.approx(
        np.rad2deg(np.arctan2(uy, np.sqrt(1 - uy**2 - uz**2))), abs=1e-5
    )
    assert peak.get("aoa_el_deg", 0.0) == pytest.approx(np.rad2deg(np.arcsin(uz)), abs=1e-5)
    measured = [peak["x_m"], peak["y_m"]]
    direction = [np.sqrt(1 - uy**2 - uz**2), uy]
    if rows > 1:
        measured.append(peak["z_m"])
        direction.append(uz)
    else:
        assert np.isnan(peak["z_m"])  # ULA does not observe elevation.
    np.testing.assert_allclose(measured, peak["range_m"] * np.array(direction), atol=1e-5)


@pytest.mark.parametrize("phi,theta", [([0.0, 0.0], [1.0]), ([0.0], [1.0, 1.0])])
def test_ambiguous_angle_axes_are_rejected_instead_of_truncated(phi, theta):
    with pytest.raises(ValueError, match="angle/path axes"):
        apply_aoa_filter(
            np.ones((1, 2)),
            np.array([1e-7]),
            SimpleNamespace(phi_r=phi, theta_r=theta),
            SimpleNamespace(aoa_filter_enabled=True),
            0,
        )
