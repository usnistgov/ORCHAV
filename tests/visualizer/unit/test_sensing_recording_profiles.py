"""Production storage profiles expose only their retained numerical evidence."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from generator.core.configuration import SensingConfig
from tests.generator.unit.test_sensing_cfr_lifecycle import native_cfr
from tests.visualizer.unit.test_sensing_dashboard_recording import _history, _request
from visualizer.src.sensing.dashboard_data import RecordingHistory, load_dashboard_request
from visualizer.src.sensing.processing_inspection import inspect_processing
from visualizer.src.sensing.processing_panel import ProcessingPanel
from visualizer.src.sensing.recorded_products import product_kinds, saved_micro_doppler
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.services.sensing_service import SensingService, _descriptor_bound_read_request

pytestmark = pytest.mark.usefixtures("sensing_figure_preferences")


def request_for(sensing, *, source="recording"):
    frame = {"sensing": sensing}
    service = SensingService(SimpleNamespace())
    rp, rd = service.dashboard_products(frame, "primary")
    return replace(
        _request(),
        display_frame=frame,
        state_frame=frame,
        rd_result=rd,
        rp_result=rp,
        source_identity=(source,),
        available_product_kinds=product_kinds(sensing),
    )


def _authored_display(**display):
    with patch(
        "tests.generator.unit.test_sensing_cfr_lifecycle.SensingConfig",
        lambda **kwargs: SensingConfig(**kwargs, recording_display=display),
    ):
        return native_cfr(recording="maps", clutter="none")


def test_single_mean_stft_policy_survives_source_switch(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    for mean, variant in (("removed", "primary"), ("retained", "raw"), ("removed", "primary")):
        sensing = _authored_display(mean=mean)
        request = request_for(sensing, source=mean)
        window.update_from_request(request)
        window._set_view_mode("micro-doppler")
        window._micro_mode_box.setCurrentIndex(window._micro_mode_box.findData("saved-stft"))
        window.update_from_request(request)
        assert window.product_variant() == variant
        assert window._variant_box.isHidden()
        assert f"mean {mean}" in window._product_note.text()
        for other in ("primary", "raw"):
            item = window._variant_box.model().item(window._variant_box.findData(other))
            assert item.isEnabled() == (other == variant)
            if other != variant:
                assert "not saved" in item.toolTip()
        expected = saved_micro_doppler(sensing, variant)
        np.testing.assert_allclose(
            window.collapsed_doppler_image.image,
            10 * np.log10(np.maximum(expected["power"], 1e-30)),
        )


def test_empty_saved_stft_gate_explains_why_and_clears_previous_source(qtbot):
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    populated = request_for(_authored_display(mean="retained"), source="populated")
    window.update_from_request(populated)
    window._set_view_mode("micro-doppler")
    window._micro_mode_box.setCurrentIndex(window._micro_mode_box.findData("saved-stft"))
    assert window.collapsed_doppler_image.image is not None
    empty = _authored_display(mean="retained", time_start_s=1.0)
    assert empty["display_status"]["micro_doppler"] == "outside saved display interval"
    window.update_from_request(request_for(empty, source="empty"))
    item = window._micro_mode_box.model().item(window._micro_mode_box.findData("saved-stft"))
    assert not item.isEnabled()
    assert "outside saved display interval" in item.toolTip()
    assert "outside saved display interval" in window.collapsed_doppler_plot.toolTip()
    assert window.collapsed_doppler_image.image is None


def test_saved_stft_history_remains_available_outside_current_time_gate(qtbot):
    populated = _authored_display(mean="retained")
    micro = saved_micro_doppler(populated)
    record = replace(
        _history().records[0],
        spectrum=None,
        micro_power=micro["power"],
        micro_times_s=micro["times_s"],
        micro_doppler_hz=micro["doppler_hz"],
        micro_hop_s=micro["hop_s"],
        micro_recipe=micro["recipe"],
        sensing_metadata={"products": populated["products"]},
    )
    empty = _authored_display(mean="retained", time_start_s=1.0)
    request = replace(request_for(empty), recording_history=RecordingHistory((record,)))
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window.update_from_request(request)
    window._set_view_mode("micro-doppler")
    window._micro_mode_box.setCurrentIndex(window._micro_mode_box.findData("saved-stft"))
    window.update_from_request(request)
    item = window._micro_mode_box.model().item(window._micro_mode_box.findData("saved-stft"))
    assert item.isEnabled()
    assert window.product_variant() == "raw"
    np.testing.assert_allclose(
        window.collapsed_doppler_image.image,
        10 * np.log10(np.maximum(micro["power"], 1e-30)),
    )


def test_empty_stft_gate_loads_without_requesting_absent_rd_variant():
    sensing = _authored_display(mean="retained", time_start_s=1.0)
    frame = {"sensing": sensing}
    service = SensingService(SimpleNamespace())
    service.dashboard_scene_frame = lambda _step: frame
    service.dashboard_product_kinds = lambda _step: product_kinds(sensing)
    service.resolve_display_sensing_frame = lambda *_args, **_kwargs: SimpleNamespace(
        display_frame=frame,
        storage_frame_id=0,
        bound_scene_frame_id=0,
        stale_for_current_scene=False,
        replay_event=None,
    )
    request = load_dashboard_request(service, 0, "micro-doppler", "raw", False, "saved-stft")
    assert request is not None and request.rd_result is None
    assert request.display_frame["sensing"]["display_status"]["micro_doppler"]


@pytest.mark.parametrize("mode", ["inputs", "frequency"])
def test_maps_input_power_keeps_magnitude_without_inventing_phase(mode, qtbot):
    sensing = native_cfr(recording="maps", elements=4)
    model = inspect_processing(sensing, mode)
    assert model.samples is None
    assert model.sample_power is not None
    panel = ProcessingPanel()
    qtbot.addWidget(panel)
    panel.set_model(model, mode)
    panel.show()
    assert panel.before_curve.yData is not None
    assert not panel.phase.isEnabled()
    assert not panel.element.isEnabled()
    assert panel.phase_curve.yData is None or panel.phase_curve.yData.size == 0
    assert "not retained" in panel.phase.getPlotItem().titleLabel.text
    index = panel.range_bin.value()
    expected = (
        model.sample_power[0, index] if mode == "frequency" else model.sample_power[0, :, index]
    )
    np.testing.assert_allclose(panel.before_curve.yData, np.sqrt(expected))
    panel.release()


def test_canonical_frequency_reconstructs_input_on_reader_without_source_mutation():
    sensing = native_cfr(recording="reprocessing", elements=4)
    before = sensing["cfr_by_pair_rx_element"].copy()
    descriptors = tuple(sensing["products"])
    assert "slow_time_by_pair_rx_element" not in sensing
    model = inspect_processing(sensing, "inputs")
    assert model.reconstructed, model.notice
    reference = native_cfr(elements=4)
    np.testing.assert_array_equal(model.samples, reference["slow_time_by_pair_rx_element"][0])
    np.testing.assert_array_equal(sensing["cfr_by_pair_rx_element"], before)
    assert tuple(sensing["products"]) == descriptors


@pytest.mark.parametrize("recording_managed", [True, False])
def test_saved_stft_is_distinct_from_rd_history_and_requires_no_fft(
    qtbot, monkeypatch, recording_managed
):
    sensing = native_cfr(recording="maps")
    micro = saved_micro_doppler(sensing)
    raw = saved_micro_doppler(sensing, "raw")
    assert micro is not None and raw is not None
    assert micro["hop_s"] == pytest.approx(0.016)
    request = replace(request_for(sensing), recording_managed=recording_managed)
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window.update_from_request(request)
    window._set_view_mode("micro-doppler")
    monkeypatch.setattr(
        np.fft, "fft", lambda *_a, **_kw: pytest.fail("Display must not recompute STFT")
    )
    window._micro_mode_box.setCurrentIndex(window._micro_mode_box.findData("saved-stft"))
    window.update_from_request(request)
    np.testing.assert_allclose(
        window.collapsed_doppler_image.image, 10 * np.log10(np.maximum(micro["power"], 1e-30))
    )
    assert "saved STFT" in window.collapsed_doppler_plot.getPlotItem().titleLabel.text
    assert window._micro_mode_used == "saved-stft"
    assert window._variant_box.model().item(window._variant_box.findData("raw")).isEnabled()


def test_results_only_disables_missing_views_and_source_change_clears_images(qtbot):
    full = native_cfr(recording="maps")
    minimal = native_cfr(recording="results")
    window = SensingWindow()
    qtbot.addWidget(window)
    window.show()
    window.update_from_request(request_for(full, source="maps"))
    window._set_view_mode("micro-doppler")
    window.update_from_request(
        replace(request_for(full, source="maps"), recording_history=_history())
    )
    assert window.collapsed_doppler_image.image is not None
    window.update_from_request(request_for(minimal, source="results"))
    motion = window.workspace_box.model().item(window.workspace_box.findData("micro-doppler"))
    assert not motion.isEnabled() and "recorded" in motion.toolTip()
    assert window.collapsed_doppler_image.image is None
    window._set_view_mode("analysis")
    window.update_from_request(request_for(minimal, source="results"))
    for index in range(1, window.workspace_tabs.count()):
        assert not window.workspace_tabs.isTabEnabled(index)
        assert window.workspace_tabs.tabToolTip(index)
    assert not window.rd_plot.isEnabled()
    assert not window.rp_plot.isEnabled()
    assert window.rd_plot.isHidden() and window.rp_plot.isHidden()
    assert window.gt_range_plot.isVisible() and window.gt_speed_plot.isVisible()
    assert window.current_product_notice.isVisible()
    assert window.current_product_notice.text() == (
        "Range–Doppler and range profiles were not recorded."
    )
    assert window.rd_image.image is None
    assert "map was not recorded" in window.rd_plot.getPlotItem().titleLabel.text
    assert not window.rd_colorbar.isVisible()
    window.update_from_request(request_for(full, source="maps"))
    assert window.rd_plot.isVisible() and window.rp_plot.isVisible()
    assert not window.current_product_notice.isVisible()
    assert window.rd_colorbar.isVisible()
    assert "unavailable" not in window.rd_plot.getPlotItem().titleLabel.text
    assert product_kinds(full, ()) == frozenset()


def test_selective_maps_read_never_requests_complex_observations():
    maps = native_cfr(recording="maps")
    request = _descriptor_bound_read_request(
        maps, ("delay_domain_power", "delay_domain_slow_time", "baseband_frequency_observation")
    )
    assert "delay_power_by_pair" in request.sensing_products
    assert "cfr_by_pair_rx_element" not in request.sensing_products
    assert "slow_time_by_pair_rx_element" not in request.sensing_products


def test_results_history_requests_only_metadata(monkeypatch):
    sensing = native_cfr(recording="results")
    service = SensingService(SimpleNamespace())
    monkeypatch.setattr(service, "dashboard_product_kinds", lambda _: frozenset())
    requested = []
    monkeypatch.setattr(
        service,
        "dashboard_load_row",
        lambda _id, products: requested.append(products) or {"sensing": sensing},
    )
    frame, rp, rd = service.dashboard_history_products(0, "primary")
    assert frame["sensing"] is sensing and rp is None and rd is None
    assert requested == [("config_resolved",)]
