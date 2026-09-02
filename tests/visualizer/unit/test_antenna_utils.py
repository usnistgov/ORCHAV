"""Tests for visualizer.src.utils.antenna_utils."""

from __future__ import annotations

import numpy as np
import pytest

from visualizer.src.utils.antenna_utils import (
    beamforming_defaults_from_scenario_config,
    get_element_positions,
    infer_array_dimensions,
    normalize_visual_element_pattern,
    spacing_m_to_wavelengths,
    spacing_wavelengths_to_m,
    wavelength_m_from_frequency_ghz,
)


class TestGetElementPositions:
    def test_single_element(self):
        pos = get_element_positions(1, 1, 0.005, 0.005)
        assert pos.shape == (1, 3)
        np.testing.assert_allclose(pos[0], [0.0, 0.0, 0.0])

    def test_2x2_shape(self):
        pos = get_element_positions(2, 2, 0.01, 0.01)
        assert pos.shape == (4, 3)

    def test_x_always_zero(self):
        pos = get_element_positions(4, 8, 0.005, 0.005)
        np.testing.assert_allclose(pos[:, 0], 0.0)

    def test_centred_layout(self):
        pos = get_element_positions(1, 3, 0.01, 0.01)
        # 3 columns centred: y = [-0.01, 0, 0.01]
        np.testing.assert_allclose(pos[:, 1], [-0.01, 0.0, 0.01])

    def test_spacing_applied(self):
        pos = get_element_positions(2, 1, 0.005, 0.02)
        # 2 rows, 1 col → z values differ by v_spacing
        assert abs(pos[1, 2] - pos[0, 2] - 0.02) < 1e-12

    def test_dtype(self):
        pos = get_element_positions(2, 2, 0.005, 0.005)
        assert pos.dtype == np.float64


class TestInferArrayDimensions:
    @pytest.mark.parametrize(
        "n, expected",
        [(32, (2, 16)), (64, (8, 8)), (128, (8, 16)), (256, (16, 16))],
    )
    def test_known_configs(self, n: int, expected: tuple[int, int]):
        assert infer_array_dimensions(n) == expected

    def test_square(self):
        rows, cols = infer_array_dimensions(36)
        assert rows * cols == 36
        assert rows == 6 and cols == 6

    def test_prime(self):
        rows, cols = infer_array_dimensions(7)
        assert rows * cols == 7  # 1 x 7

    def test_one(self):
        assert infer_array_dimensions(1) == (1, 1)


class TestSpacingConversions:
    def test_wavelength_from_frequency_ghz(self):
        assert wavelength_m_from_frequency_ghz(60.0) == pytest.approx(299_792_458.0 / 60e9)

    def test_spacing_m_to_wavelengths(self):
        wavelength = 299_792_458.0 / 28e9

        assert spacing_m_to_wavelengths(0.75 * wavelength, 28.0) == pytest.approx(0.75)

    def test_spacing_wavelengths_to_m(self):
        wavelength = 299_792_458.0 / 28e9

        assert spacing_wavelengths_to_m(0.25, 28.0) == pytest.approx(0.25 * wavelength)

    def test_invalid_spacing_uses_half_wavelength(self):
        wavelength = 299_792_458.0 / 28e9

        assert spacing_wavelengths_to_m(-1.0, 28.0) == pytest.approx(0.5 * wavelength)
        assert spacing_m_to_wavelengths(-1.0, 28.0) == pytest.approx(0.5)


class TestBeamformingDefaultsFromScenario:
    def test_missing_config_uses_safe_defaults(self):
        defaults = beamforming_defaults_from_scenario_config(None)

        assert defaults["standalone_antenna_rows"] == 1
        assert defaults["standalone_antenna_cols"] == 1
        assert defaults["standalone_carrier_frequency_ghz"] == 28.0
        assert defaults["standalone_horizontal_spacing_m"] == pytest.approx(
            299_792_458.0 / 28e9 * 0.5
        )
        assert defaults["beamforming_tx_element_pattern"] == "isotropic"
        assert defaults["beamforming_rx_element_pattern"] == "isotropic"

    def test_yaml_spacing_lambda_converts_to_meters(self):
        scenario = {
            "raytracing": {
                "carrier_frequency_hz": 60e9,
                "antenna": {
                    "tx": {
                        "pattern": "tr38901",
                        "num_rows": 4,
                        "num_cols": 8,
                        "horizontal_spacing": 0.25,
                        "vertical_spacing": 0.75,
                    },
                    "rx": {"pattern": "dipole"},
                },
            }
        }

        defaults = beamforming_defaults_from_scenario_config(scenario)
        wavelength = 299_792_458.0 / 60e9

        assert defaults["standalone_antenna_rows"] == 4
        assert defaults["standalone_antenna_cols"] == 8
        assert defaults["standalone_horizontal_spacing_m"] == pytest.approx(0.25 * wavelength)
        assert defaults["standalone_vertical_spacing_m"] == pytest.approx(0.75 * wavelength)
        assert defaults["standalone_carrier_frequency_ghz"] == 60.0
        assert defaults["beamforming_tx_element_pattern"] == "tr38901"
        assert defaults["beamforming_rx_element_pattern"] == "dipole"

    def test_unknown_pattern_falls_back_to_isotropic_with_status(self):
        defaults = beamforming_defaults_from_scenario_config(
            {"raytracing": {"antenna": {"tx": {"pattern": "custom_panel"}}}}
        )

        assert defaults["beamforming_tx_element_pattern"] == "isotropic"
        assert "custom_panel" in defaults["beamforming_pattern_status"]

    def test_hw_dipole_maps_to_dipole_visual_model(self):
        pattern, status = normalize_visual_element_pattern("hw_dipole")

        assert pattern == "dipole"
        assert "dipole visual model" in status
