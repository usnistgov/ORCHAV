"""Prepared angular displays retain scientific coordinates and Qt interactions."""

from dataclasses import fields, replace

import numpy as np
import pytest

from shared.sensing.angle import select_range_doppler_angle_product
from tests.visualizer.unit.test_sensing_dashboard_recording import _frame, _request
from tests.visualizer.unit.test_sensing_panel_fallback import _typed_dashboard_payload
from tests.visualizer.unit.test_sensing_window_typed_angular import _angular_payload
from visualizer.src.sensing.angular_spectrum import angular_spectrum_mesh
from visualizer.src.sensing.prepared_angle import prepare_angular
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.sensing.update_request import SensingUpdateRequest

pytestmark = pytest.mark.usefixtures("sensing_figure_preferences")


@pytest.mark.parametrize("nonphysical", [False, True])
def test_prepared_projections_preserve_exact_saved_coordinates_and_mask(nonphysical):
    payload = _angular_payload(azimuth_degrees=np.array([-40.0, -10.0, 30.0]))
    if nonphysical:
        payload["azimuth_direction_cosine"] = np.array([-2.0, 0.0, 0.5])
        payload["azimuth_broadside_angle_deg"] = np.array([np.nan, -10.0, 30.0])
    original = payload["range_doppler_angle_by_pair"].copy()
    angular = select_range_doppler_angle_product(payload)
    prepared = prepare_angular(payload)
    assert prepared.variant == angular.variant
    assert prepared.product_id == angular.product_id
    assert prepared.shape == angular.power_linear.shape
    visible = np.isfinite(angular.azimuth_broadside_angle_deg)
    power = angular.forward_power_linear[:, visible]
    np.testing.assert_array_equal(prepared.range_azimuth_power, power.max(axis=(0, 2)))
    np.testing.assert_array_equal(prepared.doppler_azimuth_power, power.max(axis=(0, 3)))
    np.testing.assert_array_equal(
        prepared.azimuth_axis_deg, angular.azimuth_broadside_angle_deg[visible]
    )
    for actual, expected in zip(
        (prepared.spectrum_azimuth_deg, prepared.spectrum_zenith_deg, prepared.spectrum_power),
        angular_spectrum_mesh(angular),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    for field in fields(prepared):
        value = getattr(prepared, field.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.setflags(write=True)
            assert not np.shares_memory(value, payload["range_doppler_angle_by_pair"])
    np.testing.assert_array_equal(payload["range_doppler_angle_by_pair"], original)


def test_preparation_does_not_invent_missing_variants_or_accept_invalid_power():
    payload = _angular_payload()
    assert prepare_angular(payload, "raw") is None
    payload["range_doppler_angle_by_pair"][0, 0, 0, 0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        prepare_angular(payload)


def test_single_row_product_has_broadside_projections_without_elevation_spectrum():
    payload = _typed_dashboard_payload(scene_frame_id=0, sample_time_s=0.0)
    prepared = prepare_angular(payload)
    assert prepared.shape == (1, 3, 4, 8)
    assert prepared.range_azimuth_power.shape == (3, 8)
    assert prepared.doppler_azimuth_power.shape == (3, 4)
    assert prepared.spectrum_power is None
    assert prepared.spectrum_azimuth_deg is None
    assert prepared.spectrum_zenith_deg is None


@pytest.mark.parametrize("antenna_rows", [1, 2])
@pytest.mark.parametrize("doppler_hz", [False, True])
def test_prepared_and_raw_plots_match_without_cube_on_gui(
    qtbot, monkeypatch, antenna_rows, doppler_hz
):
    import visualizer.src.sensing.sensing_window as window_module

    payload = _angular_payload(azimuth_degrees=np.array([-40.0, -10.0, 30.0]))
    payload["config_resolved"]["antenna_rows"] = antenna_rows
    payload["detections"] = [
        dict(range_m=1.0, velocity_m_s=0.1, doppler_hz=20, aoa_az_deg=30, aoa_el_deg=15)
    ]
    prepared = prepare_angular(payload)
    assert (prepared.spectrum_power is None) == (antenna_rows == 1)
    window = SensingWindow()
    qtbot.addWidget(window)
    window._doppler_hz_cb.setChecked(doppler_hz)
    window._update_angular_products(payload, "primary", "", 28e9)
    images = (window.range_azimuth_image, window.doppler_azimuth_image)
    if antenna_rows > 1:
        images += (window.angular_spectrum_image,)
    original = [(image.x.copy(), image.y.copy(), image.z.copy()) for image in images]
    overlay = tuple(a.copy() for a in window.range_azimuth_det_scatter.getData())
    del payload["range_doppler_angle_by_pair"]
    frame = {"sensing": payload}
    window._last_request = SensingUpdateRequest(
        display_frame=frame, state_frame=frame, state_step=0, prepared_angular=prepared
    )
    monkeypatch.setattr(
        window_module,
        "prepare_angular",
        lambda *a, **k: pytest.fail("prepared GUI update recomputed angular projections"),
    )
    window._update_angular_products(payload, "primary", "", 28e9)
    for image, expected in zip(images, original, strict=True):
        for actual, values in zip((image.x, image.y, image.z), expected, strict=True):
            np.testing.assert_array_equal(actual, values)
    for actual, expected in zip(window.range_azimuth_det_scatter.getData(), overlay, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert window._angular_shape == prepared.shape
    assert window.angular_spectrum_image.isVisible() == (antenna_rows > 1)
    assert window.range_azimuth_det_scatter.points()[0].data() == 0
    window.close()


def test_prepared_snapshot_bypasses_missing_cube_pending_state(qtbot):
    payload = _angular_payload()
    prepared = prepare_angular(payload)
    del payload["range_doppler_angle_by_pair"]
    frame = {**_frame(), "sensing": payload}
    request = replace(
        _request(),
        display_frame=frame,
        state_frame=frame,
        prepared_angular=prepared,
        selective_products=True,
        available_product_kinds=frozenset({"range_doppler_angle"}),
    )
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window._initial_view_selected = True
    window._set_view_mode("angular")
    window.update_from_request(request)
    assert window._angular_status == "typed primary"
    assert window._angular_shape == prepared.shape
    assert window.range_azimuth_image.isVisible()
    window._set_view_mode("analysis")
    window._set_view_mode("angular")
    assert window._angular_status == "typed primary"
    window.close()


def test_mesh_reuses_immutable_display_buffer_and_exact_nonuniform_grid(qtbot, monkeypatch):
    payload = _angular_payload(azimuth_degrees=np.array([-40.0, -10.0, 30.0]))
    prepared = prepare_angular(payload)
    window = SensingWindow()
    qtbot.addWidget(window)
    frame = {"sensing": payload}
    window._last_request = SensingUpdateRequest(
        display_frame=frame, state_frame=frame, state_step=0, prepared_angular=prepared
    )
    window._update_angular_products(payload, "primary", "", 28e9)
    image = window.range_azimuth_image
    x, y, db = image.x, image.y, image.z
    np.testing.assert_array_equal(
        db, (10 * np.log10(np.maximum(prepared.range_azimuth_power, 1e-30))).T
    )
    np.testing.assert_array_equal(
        y[0], window._axis_edges(prepared.azimuth_axis_deg, axis_name="azimuth")
    )
    with monkeypatch.context() as patch:
        patch.setattr(image, "setData", lambda *a, **k: pytest.fail("Unchanged mesh rebuilt"))
        window._update_angular_products(payload, "primary", "", 28e9)
    assert image.x is x and image.y is y and image.z is db
    # A different immutable result reuses geometry, never its power values.
    changed = replace(
        prepared,
        range_azimuth_power=np.frombuffer(
            (prepared.range_azimuth_power * 2).tobytes(), dtype=prepared.range_azimuth_power.dtype
        ).reshape(prepared.range_azimuth_power.shape),
    )
    window._last_request = replace(window._last_request, prepared_angular=changed)
    window._update_angular_products(payload, "primary", "", 28e9)
    assert image.x is x and image.y is y
    np.testing.assert_array_equal(
        image.z, (10 * np.log10(np.maximum(changed.range_azimuth_power, 1e-30))).T
    )
    window.reset_recording()
    assert not window._heatmap_buffers


def test_mutable_display_inputs_are_never_reused_by_identity(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    power = np.ones((2, 3))
    before = window._heatmap_db(window.rd_image, power)
    power[:] = 2
    after = window._heatmap_db(window.rd_image, power)
    assert not np.shares_memory(before, after)
    np.testing.assert_allclose(after, 10 * np.log10(2))


def test_spectrum_refreshes_changed_coordinates_even_with_shared_power(qtbot):
    prepared = prepare_angular(_angular_payload())
    window = SensingWindow()
    qtbot.addWidget(window)
    window._update_angular_spectrum(prepared, {}, "")
    before = window.angular_spectrum_image.z
    changed = replace(prepared, spectrum_azimuth_deg=prepared.spectrum_azimuth_deg + 1.0)
    window._update_angular_spectrum(changed, {}, "")
    assert window.angular_spectrum_image.z is before
    np.testing.assert_array_equal(window.angular_spectrum_image.x, changed.spectrum_azimuth_deg)
    np.testing.assert_array_equal(window.angular_spectrum_image.y, prepared.spectrum_zenith_deg)
