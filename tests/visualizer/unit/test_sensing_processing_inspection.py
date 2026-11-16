"""Signal provenance, progressive controls and exact-bin 3D inspection."""

from copy import deepcopy
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent

from visualizer.src.sensing import processing_inspection as module
from visualizer.src.sensing.processing_panel import ProcessingPanel
from visualizer.src.sensing.sensing_window import SensingWindow
from visualizer.src.sensing.spectrum_surface import SpectrumSurface


def signal_recording(*, raw=False):
    n, r, elements = 32, 8, 2
    times = np.arange(n) / n
    samples = np.broadcast_to(
        10 + np.exp(2j * np.pi * 3 * times)[None, :, None], (elements, n, r)
    ).copy()
    combined = samples.sum(axis=0) / np.sqrt(elements)
    before = np.fft.fftshift(np.fft.fft(combined, axis=0), axes=0) / np.sqrt(n)
    after = np.fft.fftshift(np.fft.fft(combined - combined.mean(axis=0), axis=0), axes=0) / np.sqrt(
        n
    )
    recipe = {
        "range_processing": "delay_domain_cir_bins",
        "range_fft_applied": False,
        "fft_axis": "slow_time",
        "fft_normalization": "unitary_1_over_sqrt_n",
        "window_normalization": "energy_preserving_rms",
        "slow_time_mean_removal_operator": "finite_cpi_mean_projection",
        "slow_time_mean_removal_applied": True,
        "delay_taper": "rect",
        "doppler_taper": "rect",
    }
    common = {
        "variant": "primary",
        "pair_ids": ["pair"],
        "time_span": {"start": 0, "end": 1},
        "combine": {
            "policy": "rx_broadside_unitary_voltage_sum",
            "input_axis": "rx_element",
            "normalization": "one_over_sqrt_element_count",
        },
        "representation": "complex_voltage",
        "scale_semantics": "relative_unitless",
        "unit": "1",
        "lineage": {
            "acquisition_id": "acq",
            "processing_run_id": "run",
            "processing_recipe_id": "recipe",
            "source_product_ids": ["input"],
        },
    }

    def axes(pairs):
        return [{"name": name, "coordinate_key": key} for name, key in pairs]

    rd_axes = axes([("pair", "pairs"), ("doppler_frequency", "frequencies"), ("range", "ranges")])
    primary = {
        **deepcopy(common),
        "product_id": "after",
        "kind": "range_doppler",
        "axes": rd_axes,
        "processing": recipe,
    }
    input_descriptor = {
        **deepcopy(common),
        "product_id": "input",
        "kind": "delay_domain_slow_time",
        "axes": axes(
            [
                ("pair", "pairs"),
                ("rx_element", "elements"),
                ("slow_time", "times"),
                ("range", "ranges"),
            ]
        ),
    }
    sensing = {
        "products": {"rd": primary, "samples": input_descriptor},
        "rd": after[None],
        "samples": samples[None],
        "pairs": [{"pair_id": "pair"}],
        "elements": [{"element_index": i} for i in range(elements)],
        "ranges": np.arange(r, dtype=float),
        "frequencies": np.arange(-n / 2, n / 2),
        "times": times,
        "config_resolved": {
            "sampling_frequency": 299792458 / 2,
            "prf_hz": n,
            "range_mode": "monostatic",
            "carrier_freq_hz": 3.5e9,
        },
    }
    if raw:
        sensing["products"]["raw"] = {
            **deepcopy(primary),
            "product_id": "before",
            "variant": "raw",
            "processing": {**recipe, "slow_time_mean_removal_applied": False},
        }
        sensing["raw"] = before[None]
    return sensing


def test_saved_comparison_does_not_recompute_and_uses_same_power_reference(monkeypatch):
    sensing = signal_recording(raw=True)
    monkeypatch.setattr(
        module,
        "compute_range_doppler",
        lambda **kwargs: pytest.fail("Saved comparison must not run an FFT"),
    )
    model = module.inspect_processing(sensing, "clutter")
    assert model.before is not None and not model.reconstructed
    np.testing.assert_allclose(model.before[16], 6400)
    np.testing.assert_allclose(model.after[16], 0, atol=1e-24)
    np.testing.assert_allclose(model.before[19], 64)
    np.testing.assert_allclose(model.after[19], 64)
    assert not model.before.flags.writeable


def test_missing_spectrum_reconstruction_is_explicit_and_verified():
    sensing = signal_recording()
    saved = sensing["samples"].copy()
    model = module.inspect_processing(sensing, "clutter")
    assert model.before is None and model.can_reconstruct
    model = module.inspect_processing(sensing, "clutter", reconstruct=True)
    assert model.before is not None and model.reconstructed
    np.testing.assert_allclose(model.before[16], 6400)
    assert "Verified against saved complex primary" in model.details
    np.testing.assert_array_equal(saved, sensing["samples"])
    sensing["truth"] = {"range": -999}
    altered_truth = module.inspect_processing(sensing, "clutter", reconstruct=True)
    np.testing.assert_array_equal(model.before, altered_truth.before)


def test_window_weighted_mean_reconstruction_verifies_saved_result():
    from shared.sensing.signal_processing import (
        _window,
        _window_energy_correction,
        remove_window_weighted_mean,
    )

    sensing = signal_recording()
    recipe = sensing["products"]["rd"]["processing"]
    recipe.update(doppler_taper="hann", slow_time_mean_removal_policy="window_weighted_mean")
    samples = sensing["samples"][0].sum(axis=0) / np.sqrt(2)
    window = _window("hann", len(samples))
    centered = remove_window_weighted_mean(samples, window, axis=0)
    sensing["rd"] = (
        np.fft.fftshift(
            np.fft.fft(centered * window[:, None] * _window_energy_correction(window), axis=0),
            axes=0,
        )
        / np.sqrt(len(samples))
    )[None]
    model = module.inspect_processing(sensing, "clutter", reconstruct=True)
    assert model.reconstructed, model.notice
    sensing["rd"] *= 2
    failed = module.inspect_processing(sensing, "clutter", reconstruct=True)
    assert failed.before is None
    assert "does not reproduce" in failed.notice


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda s: s["products"]["raw"].update(combine={"policy": "other"}), "different combine"),
        (
            lambda s: s["products"]["raw"]["lineage"].update(acquisition_id="other"),
            "different acquisition",
        ),
        (
            lambda s: s["products"]["raw"]["lineage"].update(source_product_ids=["other"]),
            "different source",
        ),
        (
            lambda s: s["products"]["raw"]["processing"].update(doppler_taper="hann"),
            "matching before/after recipes",
        ),
        (
            lambda s: s["products"]["raw"]["processing"].update(extra_filter="unrelated"),
            "matching before/after recipes",
        ),
    ],
)
def test_incompatible_saved_spectra_are_not_compared(mutation, reason):
    sensing = signal_recording(raw=True)
    mutation(sensing)
    model = module.inspect_processing(sensing, "clutter")
    assert model.before is None and reason in model.notice


def test_input_values_and_missing_or_unsupported_recipes():
    sensing = signal_recording()
    model = module.inspect_processing(sensing, "inputs")
    np.testing.assert_array_equal(model.samples, sensing["samples"][0])
    np.testing.assert_array_equal(model.times, sensing["times"])
    assert len(model.elements) == 2 and not model.samples.flags.writeable
    sensing["times"][12] += 0.001
    assert "contiguous" in module.inspect_processing(sensing, "clutter", reconstruct=True).notice
    sensing["products"]["rd"]["processing"]["doppler_taper"] = "unknown"
    assert "unsupported" in module.inspect_processing(sensing, "clutter", reconstruct=True).notice
    sensing.pop("samples")
    assert module.inspect_processing(sensing, "inputs").samples is None


def test_invalid_acquisition_samples_cannot_be_reconstructed():
    sensing = signal_recording()
    sensing["temporal"] = {"acquisition": {"samples": [{"valid": False}]}}
    model = module.inspect_processing(sensing, "clutter", reconstruct=True)
    assert model.before is None and "valid acquisition samples" in model.notice


@pytest.mark.parametrize(
    "view,needs_inputs",
    (
        ("analysis", False),
        ("processing-clutter", False),
        ("processing-inputs", True),
        ("processing-clutter-rebuild", True),
    ),
)
def test_deeper_requests_select_inputs_only_when_needed(tmp_path, monkeypatch, view, needs_inputs):
    """Inspect actual HDF5 reads after metadata selection and demand extension."""
    from shared.frames.providers import Hdf5Provider
    from tests.visualizer.unit.test_sensing_prepared_cache import recording
    from visualizer.src.sensing.dashboard_data import load_dashboard_request
    from visualizer.src.services.sensing_service import SensingService

    root, payloads = recording.__wrapped__(tmp_path, monkeypatch)
    provider = Hdf5Provider(root)
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    reads = set()
    original = h5py.Dataset.__getitem__

    def read(dataset, key):
        if dataset.name.startswith("/sensing/fixed/") and dataset.name.endswith("/data"):
            reads.add(dataset.name.split("/")[3])
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    try:
        request = load_dashboard_request(service, 0, view, "primary", False, "auto")
        assert request is not None
        kinds = {
            descriptor["kind"]
            for key, descriptor in payloads[0]["products"].items()
            if key in reads
        }
        assert kinds
        complex_inputs = kinds & {"delay_domain_slow_time", "baseband_frequency_observation"}
        assert complex_inputs == ({"delay_domain_slow_time"} if needs_inputs else set())
        if view == "processing-inputs":
            assert kinds == {"delay_domain_slow_time"}
            assert request.processing_inspection.samples is not None
        else:
            assert "range_doppler" in kinds
    finally:
        provider.close()


def test_progressive_views_and_reconstruction_request_identity(qtbot):
    window = SensingWindow()
    window.show()
    assert window.processing_panel.isHidden()
    assert window.stage_surface.canvas is None
    assert window.processing_panel.surface.canvas is None
    window._set_view_mode("diagnostics")
    assert window._support_box.currentData() == "cfar"
    window._support_box.setCurrentIndex(window._support_box.findData("clutter"))
    assert window.request_view_mode() == "processing-clutter"
    assert window.detection_stage_inspector.isHidden()
    assert not window.processing_panel.isHidden()
    window._request_spectrum_reconstruction()
    assert window.request_view_mode() == "processing-clutter-rebuild"
    window.reset_recording(preserve_options=True)
    assert window.request_view_mode() == "processing-clutter-rebuild"
    window.reset_recording()
    assert window.request_view_mode() == "processing-clutter"
    window.close()


def test_comparison_input_and_optional_surface_share_immutable_data(qtbot):
    panel = ProcessingPanel()
    panel.resize(900, 650)
    panel.show()
    sensing = signal_recording(raw=True)
    model = module.inspect_processing(sensing, "clutter")
    panel.set_model(model, "clutter")
    assert panel.surface.canvas is None and not panel.maps.isHidden()
    np.testing.assert_array_equal(panel.before_image.getLevels(), panel.after_image.getLevels())
    panel.range_bin.setValue(3)
    panel.surface_toggle.setChecked(True)
    qtbot.wait(20)
    assert panel.surface.canvas is not None and panel.maps.isHidden()
    np.testing.assert_array_equal(
        panel.before_curve.yData, 10 * np.log10(np.maximum(model.before[:, 3], 1e-30))
    )
    panel.set_model(module.inspect_processing(sensing, "inputs"), "inputs")
    assert panel.surface.isHidden() and not panel.phase.isHidden()
    panel.element.setCurrentIndex(1)
    np.testing.assert_allclose(panel.before_curve.yData, np.abs(sensing["samples"][0, 1, :, 3]))
    panel.reset()
    assert panel.model is None and panel.surface.last_slices is None
    panel.release()
    panel.close()
    panel.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_3d_neighborhood_preserves_exact_bins_and_releases_canvas(qtbot):
    surface = SpectrumSurface()
    assert surface.canvas is None
    ranges = np.arange(200.0)
    doppler = np.arange(-64.0, 64.0)
    power = np.ones((128, 200))
    power[65, 100] = 100
    threshold = np.full_like(power, 10)
    eligible = np.ones_like(power, dtype=bool)
    eligible[62:65] = False
    surface.set_spectra(
        ranges,
        doppler,
        power,
        threshold,
        selected=(65, 100),
        labels=("Input", "Threshold"),
        threshold=True,
        eligible=eligible,
    )
    ds, rs = surface.last_slices
    assert (ds.start, ds.stop, rs.start, rs.stop) == (41, 90, 68, 133)
    assert surface._sources[0][65, 100] == 100
    assert "Grey floor" in surface.toolTip()
    assert surface.caption.isHidden()
    surface.clear("Source unavailable")
    assert not surface.axes.collections
    surface.release()
    surface.close()
    surface.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_full_surface_keeps_distant_reports_and_selection_zoom_is_explicit(qtbot):
    surface = SpectrumSurface()
    qtbot.addWidget(surface)
    p = np.ones((64, 100))
    p[5, 5], p[60, 95] = 20, 40
    surface.scope.setCurrentText("Whole grid")
    surface.set_spectra(
        np.arange(100.0),
        np.arange(64.0),
        p,
        p * 0.5,
        selected=(5, 5),
        labels=("Input", "Threshold"),
        threshold=True,
        eligible=np.ones(p.shape, bool),
        detections=((5, 5), (60, 95)),
    )
    assert surface.last_report_cells == ((5, 5), (60, 95))
    assert surface.axes.get_xlim() == (0, 99)
    surface.scope.setCurrentText("Selected neighborhood")
    assert surface.last_report_cells == ((5, 5),)
    assert "1/2 published peak cells" in surface.toolTip()
    assert surface.caption.isHidden()
    surface.clear()
    assert surface.last_report_cells == ()
    surface.release()


def test_surface_report_region_handles_outside_selection_and_cached_redraw(qtbot):
    surface = SpectrumSurface()
    qtbot.addWidget(surface)
    surface.scope.setCurrentText("All detections region")
    p = np.ones((100, 100))
    kwargs = dict(
        selected=(99, 99), labels=("Input", "Threshold"), detections=((3, 3),), cache_key="same"
    )
    for _ in range(2):
        surface.set_spectra(np.arange(100.0), np.arange(100.0), p, p, **kwargs)
        assert surface.last_slices == (slice(0, 7), slice(0, 7))
        assert surface.last_report_cells == ((3, 3),)
    surface.release()


def test_disabled_clutter_does_not_present_duplicate_spectra():
    sensing = signal_recording(raw=True)
    sensing["products"]["rd"]["processing"]["slow_time_mean_removal_applied"] = False
    sensing["rd"] = sensing["raw"].copy()
    before = sensing["rd"].copy()
    model = module.inspect_processing(sensing, "clutter", reconstruct=True)
    assert model.before is None and model.after is None
    assert not model.can_reconstruct
    assert model.notice == "Clutter suppression off"
    np.testing.assert_array_equal(sensing["rd"], before)
