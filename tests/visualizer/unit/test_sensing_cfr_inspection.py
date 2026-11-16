"""T23: lazy CFR selection, honest absence and actual Qt magnitude/phase view."""

import numpy as np
import pytest

from tests.generator.unit.test_sensing_cfr_lifecycle import native_cfr
from visualizer.src.sensing.processing_inspection import ProcessingInspection, inspect_processing
from visualizer.src.sensing.processing_panel import ProcessingPanel
from visualizer.src.services.sensing_service import _descriptor_bound_read_request


def test_T23_qt_frequency_selection_and_absence(qtbot):
    sensing = native_cfr(elements=4)
    model = inspect_processing(sensing, "frequency")
    assert model.samples.shape == (4, 16, 64)
    panel = ProcessingPanel()
    qtbot.addWidget(panel)
    panel.set_model(model, "frequency")
    panel.show()
    panel.element.setCurrentIndex(2)
    panel.range_bin.setValue(3)
    qtbot.wait(20)
    np.testing.assert_array_equal(panel.before_curve.xData, model.frequencies / 1e6)
    np.testing.assert_allclose(panel.before_curve.yData, abs(model.samples[2, 3]))
    panel.set_model(inspect_processing(native_cfr(policy="none"), "frequency"), "frequency")
    assert "unavailable" in panel.notice.text()
    assert panel.before_curve.xData is None or len(panel.before_curve.xData) == 0
    panel.release()


def test_T23_phase_mask_and_legacy_recording(qtbot):
    from dataclasses import replace

    model = inspect_processing(native_cfr(), "frequency")
    samples = model.samples.copy()
    samples[..., 0] = 1e-25
    model = replace(model, samples=samples)
    panel = ProcessingPanel()
    qtbot.addWidget(panel)
    panel.set_model(model, "frequency")
    assert np.isnan(panel.phase_curve.yData[0])
    assert inspect_processing({"products": {}}, "frequency").samples is None
    panel.release()


@pytest.mark.parametrize("retained", ["power", "complex"])
@pytest.mark.parametrize("amplitude", [9.025e-5, 9.025e-15, 0.0])
def test_stationary_magnitude_remains_readable_across_input_views(qtbot, retained, amplitude):
    panel = ProcessingPanel()
    qtbot.addWidget(panel)
    panel.resize(900, 650)
    panel.show()
    try:
        for mode in ("inputs", "frequency", "inputs"):
            values = np.full((1, 8, 4), amplitude)
            if mode == "frequency" and amplitude:
                # A flat CFR can differ only by floating-point rounding.
                values[:, :, 1::2] = np.nextafter(amplitude, np.inf)
            model = ProcessingInspection(
                mode,
                samples=values.astype(complex) if retained == "complex" else None,
                sample_power=values**2 if retained == "power" else None,
                times=np.arange(8) / 1000,
                ranges=np.arange(4.0) if mode == "inputs" else None,
                frequencies=np.arange(4.0) * 1e6 if mode == "frequency" else None,
                elements=("Saved receiver",),
            )
            panel.set_model(model, mode)
            qtbot.wait(10)
            expected = values[0, :, 0] if mode == "inputs" else values[0, 0]
            np.testing.assert_allclose(panel.before_curve.yData, expected, rtol=1e-15, atol=0)
            low, high = panel.slice.viewRange()[1]
            assert low == 0
            if amplitude:
                assert amplitude < high < 1.2 * amplitude
                axis = panel.slice.getAxis("left")
                spacing, ticks = axis.tickValues(low, high, axis.height())[0]
                labels = axis.tickStrings(ticks, axis.autoSIPrefixScale, spacing)
                assert len(labels) >= 2 and len(set(labels)) == len(labels)
            else:
                assert np.isfinite(high) and high > 0
                assert not panel.slice.isHidden()
            assert panel.phase.isEnabled() == (retained == "complex")
    finally:
        panel.release()


def test_T23_frequency_loading_is_explicit():
    sensing = native_cfr()
    ordinary = _descriptor_bound_read_request(sensing, ("range_doppler",))
    frequency = _descriptor_bound_read_request(sensing, ("baseband_frequency_observation",))
    assert "cfr_by_pair_rx_element" not in ordinary.sensing_products
    assert "cfr_by_pair_rx_element" in frequency.sensing_products
    assert "frequency_axis_hz" in frequency.sensing_products


@pytest.mark.parametrize("mode", ["inputs", "clutter"])
def test_canonical_frequency_reconstruction_checks_padded_output_before_allocation(
    monkeypatch, mode
):
    from shared.sensing import cfr_replay as sensing_cfr_replay

    sensing = native_cfr(recording="reprocessing", elements=4)
    for key in tuple(sensing["products"]):
        descriptor = sensing["products"][key]
        if descriptor["kind"] == "range_doppler" and descriptor["variant"] == "raw":
            del sensing["products"][key]
            del sensing[key]
    sensing["config_resolved"]["fft_size_range"] = 2**27

    def unexpected_transform(*_args, **_kwargs):
        pytest.fail("Oversized output must be refused before a reconstruction is allocated")

    monkeypatch.setattr(sensing_cfr_replay, "reconstruct_delay_observation", unexpected_transform)
    monkeypatch.setattr(sensing_cfr_replay, "rebuild_frequency_observation", unexpected_transform)
    model = inspect_processing(sensing, mode, reconstruct=True)
    assert "Projected reconstruction exceeds" in model.notice
    assert model.samples is None and model.before is None


def test_small_zero_padded_canonical_input_remains_inspectable():
    sensing = native_cfr(recording="reprocessing", elements=4, fft=128)
    assert sensing["cfr_by_pair_rx_element"].shape == (1, 4, 16, 64)
    model = inspect_processing(sensing, "inputs")
    assert model.reconstructed, model.notice
    assert model.samples.shape == (4, 16, 128)
    assert np.isfinite(model.samples).all()


def test_paris_array_delay_reconstruction_fits_bounded_worker():
    from visualizer.src.sensing.processing_inspection import (
        MAX_WORK_BYTES,
        _preflight_frequency_reconstruction,
    )

    # Metadata-only check: the real Paris array must stay inspectable without
    # allocating its 128 MiB observation during this regression test.
    work = _preflight_frequency_reconstruction(
        {"config_resolved": {"fft_size_range": 256}},
        {"shape": [1, 256, 128, 256], "dtype": "complex128"},
    )
    assert work <= MAX_WORK_BYTES
