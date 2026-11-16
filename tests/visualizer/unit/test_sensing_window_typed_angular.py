"""Visualizer coverage for exact typed angular products."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload

_APP = QApplication.instance() or QApplication([])

try:
    from visualizer.src.sensing.sensing_window import SensingWindow

    HAS_SENSING_WINDOW = True
except ImportError:
    HAS_SENSING_WINDOW = False


@pytest.mark.skipif(not HAS_SENSING_WINDOW, reason="SensingWindow dependencies unavailable")
def test_detection_table_exposes_saved_angles_without_inventing_missing_coordinates():
    from PySide6.QtCore import Qt

    window = SensingWindow()
    try:
        detections = [
            {"range_m": 4.65, "aoa_az_deg": 11.00446, "aoa_el_deg": 10.80692},
            {"range_m": 4.50, "aoa_az_deg": -3.0, "aoa_el_deg": None},
        ]
        window._update_detection_table({"detections": detections})
        model = window.detection_table_model
        assert model.columnCount() == 6
        assert model.headerData(4, Qt.Orientation.Horizontal) == "Azimuth (deg)"
        assert model.headerData(5, Qt.Orientation.Horizontal) == "Elevation (deg)"
        assert model.data(model.index(0, 5)) == "10.8"
        assert model.data(model.index(0, 5), Qt.ItemDataRole.UserRole) == 10.80692
        assert model.data(model.index(1, 5)) == "—"
        window.detection_table_proxy.sort(4, Qt.SortOrder.AscendingOrder)
        assert window.detection_table_proxy.data(window.detection_table_proxy.index(0, 4)) == "-3.0"
        assert detections[0]["aoa_el_deg"] == 10.80692
        # A new acquisition replaces columns as well as rows: no stale angles.
        window._update_detection_table({"detections": [{"range_m": 5.0}]})
        assert model.columnCount() == 4
        window._update_detection_table({"detections": [{"aoa_az_deg": 0.0, "aoa_el_deg": np.nan}]})
        assert model.columnCount() == 5
        assert model.data(model.index(0, 4)) == "0.0"
        projected = [{"aoa_az_deg": 12.0, "aoa_el_deg": 0.0}]
        window._update_detection_table(
            {"detections": projected, "config_resolved": {"antenna_rows": 1}}
        )
        assert model.columnCount() == 5
        assert projected[0]["aoa_el_deg"] == 0.0
        window._update_detection_table(
            {"detections": projected, "config_resolved": {"antenna_rows": 4}}
        )
        assert model.columnCount() == 6
        assert model.data(model.index(0, 5)) == "0.0"
    finally:
        window.close()


def _angular_payload(*, azimuth_degrees: np.ndarray | None = None) -> dict:
    azimuth_axis = np.asarray(
        [-30.0, 0.0, 30.0] if azimuth_degrees is None else azimuth_degrees,
        dtype=np.float64,
    )
    return build_visualizer_sensing_payload(
        {
            "config_resolved": {
                "range_mode": "monostatic",
                "carrier_freq_hz": 28e9,
            },
            "range_axis_m": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "doppler_axis_hz": np.array([-100.0, 100.0], dtype=np.float64),
            "elevation_direction_cosine": np.array([-0.5, 0.5], dtype=np.float64),
            "azimuth_direction_cosine": np.array([-0.5, 0.0, 0.5], dtype=np.float64),
            "elevation_broadside_angle_deg": np.array([-25.0, 25.0], dtype=np.float64),
            "azimuth_broadside_angle_deg": azimuth_axis,
            "range_doppler_angle_by_pair": np.arange(36, dtype=np.float32).reshape(1, 2, 3, 2, 3),
        }
    )


@pytest.mark.skipif(not HAS_SENSING_WINDOW, reason="SensingWindow dependencies unavailable")
def test_visualizer_renders_typed_angular_product() -> None:
    window = SensingWindow()
    try:
        window._update_angular_products(_angular_payload(), "primary", "", 28e9)

        assert window._angular_status == "typed primary"
        assert window._angular_shape == (2, 3, 2, 3)
        assert window.range_azimuth_image.isVisible()
        assert window.doppler_azimuth_image.isVisible()
        assert window.angular_spectrum_image.isVisible()
        assert window.angular_spectrum_plot.getViewBox().state["yInverted"]
        np.testing.assert_allclose(
            window.angular_spectrum_image.z,
            10
            * np.log10(np.max(_angular_payload()["range_doppler_angle_by_pair"][0], axis=(2, 3))),
        )
        assert not window.range_azimuth_image.boundingRect().isEmpty()
        assert not window.doppler_azimuth_image.boundingRect().isEmpty()
    finally:
        window.close()


@pytest.mark.skipif(not HAS_SENSING_WINDOW, reason="SensingWindow dependencies unavailable")
def test_visualizer_disables_absent_typed_angular_product_visibly() -> None:
    window = SensingWindow()
    try:
        window._update_angular_products(
            build_visualizer_sensing_payload(),
            "primary",
            "",
            28e9,
        )

        assert window._angular_shape is None
        assert window._angular_status == "no typed primary product"
        title = window.range_azimuth_plot.getPlotItem().titleLabel.text
        assert "unavailable" in title.lower()
        assert not window.angular_spectrum_image.isVisible()
    finally:
        window.close()


@pytest.mark.skipif(not HAS_SENSING_WINDOW, reason="SensingWindow dependencies unavailable")
def test_visualizer_renders_nonuniform_angle_axis_with_coordinate_mesh() -> None:
    window = SensingWindow()
    try:
        window._update_angular_products(
            _angular_payload(azimuth_degrees=np.array([-40.0, -10.0, 30.0])),
            "primary",
            "",
            28e9,
        )

        assert window._angular_shape == (2, 3, 2, 3)
        assert window._angular_status == "typed primary"
        assert window.range_azimuth_image.isVisible()
        np.testing.assert_allclose(
            window._axis_edges(
                np.array([-40.0, -10.0, 30.0]),
                axis_name="azimuth",
            ),
            [-55.0, -25.0, 10.0, 50.0],
        )
    finally:
        window.close()


@pytest.mark.skipif(not HAS_SENSING_WINDOW, reason="SensingWindow dependencies unavailable")
def test_compact_array_display_and_elevated_detection_share_projection_coordinates():
    payload = _angular_payload()
    payload["azimuth_direction_cosine"] = np.array([-2.0, 0.0, 0.5])
    payload["azimuth_broadside_angle_deg"] = np.array([np.nan, 0.0, 30.0])
    payload["detections"] = [
        {
            "range_m": 1.0,
            "velocity_m_s": 0.1,
            "doppler_hz": 20.0,
            "aoa_az_deg": 35.264389682754654,
            "aoa_el_deg": 30.0,
        }
    ]
    window = SensingWindow()
    try:
        window._update_angular_products(payload, "primary", "", 28e9)
        assert window._angular_status == "typed primary"
        assert window.range_azimuth_image.isVisible()
        assert not window.range_azimuth_image.boundingRect().isEmpty()
        x, y = window.range_azimuth_det_scatter.getData()
        np.testing.assert_allclose(x, [1.0])
        np.testing.assert_allclose(y, [30.0], atol=1e-6)
        x, y = window.angular_spectrum_det_scatter.getData()
        np.testing.assert_allclose(x, [35.264389682754654])
        np.testing.assert_allclose(y, [60.0])
        assert np.isnan(window.angular_spectrum_image.z[:, 0]).all()
        assert payload["detections"][0]["aoa_az_deg"] == 35.264389682754654
    finally:
        window.close()


def test_angular_spectrum_uses_joint_spherical_coordinates_without_mutating_power():
    from types import SimpleNamespace

    from visualizer.src.sensing.angular_spectrum import angular_spectrum_mesh

    power = np.arange(3 * 3 * 2 * 4, dtype=float).reshape(3, 3, 2, 4)
    before = power.copy()
    product = SimpleNamespace(
        elevation_direction_cosine=np.array([-0.6, 0, 0.6]),
        azimuth_direction_cosine=np.array([-0.6, 0, 0.6]),
        power_linear=power,
    )
    azimuth, zenith, values = angular_spectrum_mesh(product)
    assert azimuth.shape == zenith.shape == (4, 4)
    assert np.isfinite(azimuth).all() and np.isfinite(zenith).all()
    assert azimuth[1, 2] == pytest.approx(np.degrees(np.arctan2(0.3, np.sqrt(0.82))))
    assert zenith[1, 2] == pytest.approx(np.degrees(np.arccos(-0.3)))
    np.testing.assert_array_equal(values, power.max(axis=(2, 3)))
    np.testing.assert_array_equal(power, before)


def test_angular_spectrum_clears_when_elevation_is_unobservable():
    window = SensingWindow()
    try:
        payload = _angular_payload()
        window._update_angular_products(payload, "primary", "", 28e9)
        assert window.angular_spectrum_image.isVisible()
        payload["config_resolved"]["antenna_rows"] = 1
        window._update_angular_products(payload, "primary", "", 28e9)
        assert window.range_azimuth_image.isVisible()
        assert not window.angular_spectrum_image.isVisible()
        assert (
            "elevation not resolved" in window.angular_spectrum_plot.getPlotItem().titleLabel.text
        )
        assert len(window.angular_spectrum_det_scatter.points()) == 0
    finally:
        window.close()
