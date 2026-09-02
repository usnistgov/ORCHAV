"""Tests for UIController MPC range-filter panel updates."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import QObject

from visualizer.src.controllers.ui_controller import UIController
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.services.cache_service import CacheInvalidationScope
from visualizer.src.utils.antenna_utils import spacing_wavelengths_to_m


def _make_controller(viz) -> UIController:
    controller = UIController.__new__(UIController)
    controller.visualizer = viz
    return controller


class _FakeVisualizer(SimpleNamespace):
    def set_state(self, **updates):
        for key, value in updates.items():
            setattr(self.app_state, key, value)


class _FakeDropdown:
    def __init__(self, index: int, data: object) -> None:
        self._index = index
        self._data = data

    def currentIndex(self) -> int:
        return self._index

    def itemData(self, _index: int) -> object:
        return self._data


class _FakeSpin(QObject):
    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = value

    def value(self) -> float:
        return self._value

    def setValue(self, value: float) -> None:
        self._value = value


def _make_canon(**angle_fields):
    defaults = {
        "aoa_az": None,
        "aoa_el": None,
        "aod_az": None,
        "aod_el": None,
    }
    defaults.update(angle_fields)
    return SimpleNamespace(
        delay_min=0.0,
        delay_max=10.0,
        loss_min=70.0,
        loss_max=120.0,
        aoa_az_min=-180.0,
        aoa_az_max=180.0,
        aoa_el_min=-90.0,
        aoa_el_max=90.0,
        aod_az_min=-180.0,
        aod_az_max=180.0,
        aod_el_min=-90.0,
        aod_el_max=90.0,
        **defaults,
    )


def test_range_filter_bounds_forward_canonical_ranges():
    panel = SimpleNamespace(update_range_filter_bounds=Mock())
    controller = _make_controller(
        SimpleNamespace(ui_manager=SimpleNamespace(panels={"mpc": panel}))
    )
    canon = _make_canon()

    controller.update_range_filter_bounds_from_canonical(canon)

    assert panel.update_range_filter_bounds.call_args.kwargs == {
        "delay_min": 0.0,
        "delay_max": 10.0,
        "loss_min": 70.0,
        "loss_max": 120.0,
        "aoa_az_min": -180.0,
        "aoa_az_max": 180.0,
        "aoa_el_min": -90.0,
        "aoa_el_max": 90.0,
        "aod_az_min": -180.0,
        "aod_az_max": 180.0,
        "aod_el_min": -90.0,
        "aod_el_max": 90.0,
    }


def test_trajectory_speed_range_does_not_span_disconnected_tracks():
    trajectory_data = {
        "tx_positions": {
            0: [(0, 0.0, 0.0, 0.0), (1, 1.0, 0.0, 0.0)],
            1: [(2, 100.0, 0.0, 0.0), (3, 101.0, 0.0, 0.0)],
        }
    }

    assert UIController._compute_global_scalar_range("speed", trajectory_data) == (1.0, 1.0)


def test_trajectory_angular_speed_range_does_not_span_disconnected_tracks():
    trajectory_data = {
        "target_positions": {
            "a": [(0, 0.0, 0.0, 0.0), (1, 1.0, 0.0, 0.0), (2, 1.0, 1.0, 0.0)],
            "b": [
                (3, 100.0, 0.0, 0.0),
                (4, 100.0, -1.0, 0.0),
                (5, 99.0, -1.0, 0.0),
            ],
        }
    }

    assert UIController._compute_global_scalar_range(
        "angular_speed", trajectory_data
    ) == pytest.approx((np.pi / 2, np.pi / 2))


def test_aperture_toggle_off_forces_service_update_to_clear_geometry():
    aperture_service = SimpleNamespace(update_apertures=Mock())
    viz = _FakeVisualizer(
        app_state=SimpleNamespace(show_aoa_aperture=True, show_aod_aperture=False),
        aperture_service=aperture_service,
    )
    controller = _make_controller(viz)

    controller.handle_aoa_aperture_toggled(False)

    assert viz.app_state.show_aoa_aperture is False
    aperture_service.update_apertures.assert_called_once_with()


def test_aod_aperture_toggle_off_forces_service_update_to_clear_geometry():
    aperture_service = SimpleNamespace(update_apertures=Mock())
    viz = _FakeVisualizer(
        app_state=SimpleNamespace(show_aoa_aperture=False, show_aod_aperture=True),
        aperture_service=aperture_service,
    )
    controller = _make_controller(viz)

    controller.handle_aod_aperture_toggled(False)

    assert viz.app_state.show_aod_aperture is False
    aperture_service.update_apertures.assert_called_once_with()


def test_interaction_marker_toggle_without_renderer_schedules_update():
    viz = _FakeVisualizer(
        app_state=SimpleNamespace(show_mpc_type_markers=False),
        renderer=None,
        schedule_update=Mock(),
    )
    controller = _make_controller(viz)

    controller.handle_mpc_interaction_markers_toggled(True)

    assert viz.app_state.show_mpc_type_markers is True
    viz.schedule_update.assert_called_once_with()


def test_interaction_marker_toggle_refreshes_supported_renderer():
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(mpc_type_markers=True),
        refresh_mpc_point_markers=Mock(),
    )
    viz = _FakeVisualizer(
        app_state=SimpleNamespace(show_mpc_type_markers=False),
        renderer=renderer,
        schedule_update=Mock(),
    )
    controller = _make_controller(viz)

    controller.handle_mpc_interaction_markers_toggled(True)

    assert viz.app_state.show_mpc_type_markers is True
    renderer.refresh_mpc_point_markers.assert_called_once_with()
    viz.schedule_update.assert_not_called()


def test_interaction_marker_toggle_rejects_unsupported_renderer():
    renderer = SimpleNamespace(capabilities=RendererCapabilities())
    viz = _FakeVisualizer(
        app_state=SimpleNamespace(show_mpc_type_markers=True),
        renderer=renderer,
        schedule_update=Mock(),
    )
    controller = _make_controller(viz)

    controller.handle_mpc_interaction_markers_toggled(True)

    assert viz.app_state.show_mpc_type_markers is False
    viz.schedule_update.assert_called_once_with()


def test_global_angular_reference_toggle_forces_service_update():
    aperture_service = SimpleNamespace(update_apertures=Mock())
    viz = _FakeVisualizer(
        app_state=SimpleNamespace(
            show_aoa_aperture=False,
            show_aod_aperture=False,
            show_global_angular_reference=False,
            show_local_angular_reference=False,
        ),
        aperture_service=aperture_service,
    )
    controller = _make_controller(viz)

    controller.handle_global_angular_reference_toggled(True)

    assert viz.app_state.show_global_angular_reference is True
    aperture_service.update_apertures.assert_called_once_with()


def test_local_angular_reference_toggle_forces_service_update():
    aperture_service = SimpleNamespace(update_apertures=Mock())
    viz = _FakeVisualizer(
        app_state=SimpleNamespace(
            show_aoa_aperture=False,
            show_aod_aperture=False,
            show_global_angular_reference=False,
            show_local_angular_reference=False,
        ),
        aperture_service=aperture_service,
    )
    controller = _make_controller(viz)

    controller.handle_local_angular_reference_toggled(True)

    assert viz.app_state.show_local_angular_reference is True
    aperture_service.update_apertures.assert_called_once_with()


def test_missing_orientation_frame_is_synchronized_once_by_pipeline():
    process_frame = Mock()
    create_orientation_frames = Mock()
    viz = SimpleNamespace(
        animation_step=7,
        force_update_next_frame=False,
        cache_service=SimpleNamespace(get_frame=Mock(return_value=None)),
        node_service=SimpleNamespace(create_orientation_frames=create_orientation_frames),
        _process_frame_step=process_frame,
    )
    controller = _make_controller(viz)

    controller._ensure_orientation_frames()

    assert viz.force_update_next_frame is True
    process_frame.assert_called_once_with(7)
    create_orientation_frames.assert_not_called()


@pytest.mark.parametrize(
    ("handler_name", "value", "state_name", "expected"),
    [
        ("handle_tx_orientation_toggled", True, "show_tx_orientation", True),
        ("handle_rx_orientation_toggled", True, "show_rx_orientation", True),
        (
            "handle_target_orientation_toggled",
            True,
            "show_target_orientation",
            True,
        ),
        ("handle_orientation_scale_changed", 4.5, "orientation_scale", 4.5),
    ],
)
def test_orientation_ui_change_does_not_request_a_second_renderer_update(
    handler_name,
    value,
    state_name,
    expected,
):
    viz = SimpleNamespace(
        vis_initialized=True,
        renderer=SimpleNamespace(update_renderer=Mock()),
        show_tx_orientation=False,
        show_rx_orientation=False,
        show_target_orientation=False,
        orientation_scale=3.0,
    )
    controller = _make_controller(viz)
    controller._ensure_orientation_frames = Mock()

    getattr(controller, handler_name)(value)

    assert getattr(viz, state_name) == expected
    controller._ensure_orientation_frames.assert_called_once_with()
    viz.renderer.update_renderer.assert_not_called()


def test_tx_selection_refreshes_mpc_panel_from_ui_manager():
    panel = SimpleNamespace(refresh_aperture_preview_state=Mock())
    viz = _FakeVisualizer(
        app_state=SimpleNamespace(
            selected_tx="all",
            selected_rx="all",
            beamforming_tx_node="auto",
            beamforming_rx_node="auto",
            show_aoa_aperture=False,
            show_aod_aperture=False,
        ),
        tx_dropdown=_FakeDropdown(index=1, data=0),
        cache_service=SimpleNamespace(invalidate=Mock()),
        mpc_view_cache=SimpleNamespace(clear=Mock()),
        vis_initialized=False,
        schedule_update=Mock(),
        aperture_service=None,
        beamforming_ui_controller=SimpleNamespace(
            clear_result_metadata=Mock(),
            apply_selector_state=Mock(),
        ),
        ui_manager=SimpleNamespace(panels={"mpc": panel}),
    )
    controller = _make_controller(viz)

    controller.handle_tx_selection_changed("TX1")

    assert viz.app_state.selected_tx == 0
    assert viz.app_state.beamforming_tx_node == "tx_1"
    viz.cache_service.invalidate.assert_called_once_with(
        CacheInvalidationScope.FILTERS,
        reason="tx_selection",
    )
    viz.mpc_view_cache.clear.assert_not_called()
    viz.beamforming_ui_controller.clear_result_metadata.assert_called_once_with()
    viz.beamforming_ui_controller.apply_selector_state.assert_called_once_with()
    panel.refresh_aperture_preview_state.assert_called_once_with()


def test_rx_selection_refreshes_mpc_panel_from_ui_manager():
    panel = SimpleNamespace(refresh_aperture_preview_state=Mock())
    viz = _FakeVisualizer(
        app_state=SimpleNamespace(
            selected_tx="all",
            selected_rx="all",
            beamforming_tx_node="auto",
            beamforming_rx_node="auto",
            show_aoa_aperture=False,
            show_aod_aperture=False,
        ),
        rx_dropdown=_FakeDropdown(index=1, data=0),
        cache_service=SimpleNamespace(invalidate=Mock()),
        mpc_view_cache=SimpleNamespace(clear=Mock()),
        vis_initialized=False,
        schedule_update=Mock(),
        aperture_service=None,
        beamforming_ui_controller=SimpleNamespace(
            clear_result_metadata=Mock(),
            apply_selector_state=Mock(),
        ),
        ui_manager=SimpleNamespace(panels={"mpc": panel}),
    )
    controller = _make_controller(viz)

    controller.handle_rx_selection_changed("RX1")

    assert viz.app_state.selected_rx == 0
    assert viz.app_state.beamforming_rx_node == "rx_1"
    viz.cache_service.invalidate.assert_called_once_with(
        CacheInvalidationScope.FILTERS,
        reason="rx_selection",
    )
    viz.mpc_view_cache.clear.assert_not_called()
    viz.beamforming_ui_controller.clear_result_metadata.assert_called_once_with()
    viz.beamforming_ui_controller.apply_selector_state.assert_called_once_with()
    panel.refresh_aperture_preview_state.assert_called_once_with()


def test_tx_selection_all_resets_beamforming_tx_to_auto():
    viz = _FakeVisualizer(
        app_state=SimpleNamespace(
            selected_tx=0,
            selected_rx="all",
            beamforming_tx_node="tx_1",
            beamforming_rx_node="auto",
            show_aoa_aperture=False,
            show_aod_aperture=False,
        ),
        tx_dropdown=_FakeDropdown(index=0, data=None),
        mpc_view_cache=SimpleNamespace(clear=Mock()),
        vis_initialized=False,
        schedule_update=Mock(),
        aperture_service=None,
        beamforming_ui_controller=SimpleNamespace(
            clear_result_metadata=Mock(),
            apply_selector_state=Mock(),
        ),
        ui_manager=SimpleNamespace(panels={}),
    )
    controller = _make_controller(viz)

    controller.handle_tx_selection_changed("All TX")

    assert viz.app_state.selected_tx == "all"
    assert viz.app_state.beamforming_tx_node == "auto"
    viz.beamforming_ui_controller.clear_result_metadata.assert_called_once_with()


def test_rx_selection_all_resets_beamforming_rx_to_auto():
    viz = _FakeVisualizer(
        app_state=SimpleNamespace(
            selected_tx="all",
            selected_rx=1,
            beamforming_tx_node="auto",
            beamforming_rx_node="rx_2",
            show_aoa_aperture=False,
            show_aod_aperture=False,
        ),
        rx_dropdown=_FakeDropdown(index=0, data=None),
        mpc_view_cache=SimpleNamespace(clear=Mock()),
        vis_initialized=False,
        schedule_update=Mock(),
        aperture_service=None,
        beamforming_ui_controller=SimpleNamespace(
            clear_result_metadata=Mock(),
            apply_selector_state=Mock(),
        ),
        ui_manager=SimpleNamespace(panels={}),
    )
    controller = _make_controller(viz)

    controller.handle_rx_selection_changed("All RX")

    assert viz.app_state.selected_rx == "all"
    assert viz.app_state.beamforming_rx_node == "auto"
    viz.beamforming_ui_controller.clear_result_metadata.assert_called_once_with()


def test_standalone_spacing_spinboxes_store_wavelengths_but_state_uses_meters():
    viz = _FakeVisualizer(
        app_state=SimpleNamespace(
            standalone_carrier_frequency_ghz=28.0,
            standalone_horizontal_spacing_m=spacing_wavelengths_to_m(0.5, 28.0),
            standalone_vertical_spacing_m=spacing_wavelengths_to_m(0.5, 28.0),
            show_beamforming=False,
        ),
        standalone_h_spacing=_FakeSpin(0.75),
        standalone_v_spacing=_FakeSpin(0.25),
        mpc_view_cache=SimpleNamespace(clear=Mock()),
        schedule_update=Mock(),
    )
    controller = _make_controller(viz)

    controller.handle_standalone_spacing_changed(0.75)

    assert viz.app_state.standalone_horizontal_spacing_m == pytest.approx(
        spacing_wavelengths_to_m(0.75, 28.0)
    )
    assert viz.app_state.standalone_vertical_spacing_m == pytest.approx(
        spacing_wavelengths_to_m(0.25, 28.0)
    )
    viz.mpc_view_cache.clear.assert_called_once_with()
    viz.schedule_update.assert_not_called()


def test_standalone_frequency_change_preserves_displayed_wavelength_spacing():
    viz = _FakeVisualizer(
        app_state=SimpleNamespace(
            standalone_carrier_frequency_ghz=28.0,
            standalone_horizontal_spacing_m=spacing_wavelengths_to_m(0.5, 28.0),
            standalone_vertical_spacing_m=spacing_wavelengths_to_m(0.75, 28.0),
        ),
        standalone_h_spacing=_FakeSpin(0.5),
        standalone_v_spacing=_FakeSpin(0.75),
        mpc_view_cache=SimpleNamespace(clear=Mock()),
        schedule_update=Mock(),
    )
    controller = _make_controller(viz)

    controller.handle_standalone_frequency_changed(60.0)

    assert viz.app_state.standalone_carrier_frequency_ghz == 60.0
    assert viz.app_state.standalone_horizontal_spacing_m == pytest.approx(
        spacing_wavelengths_to_m(0.5, 60.0)
    )
    assert viz.app_state.standalone_vertical_spacing_m == pytest.approx(
        spacing_wavelengths_to_m(0.75, 60.0)
    )
    assert viz.standalone_h_spacing.value() == pytest.approx(0.5)
    assert viz.standalone_v_spacing.value() == pytest.approx(0.75)
    assert viz.standalone_h_spacing.signalsBlocked() is False
    assert viz.standalone_v_spacing.signalsBlocked() is False
