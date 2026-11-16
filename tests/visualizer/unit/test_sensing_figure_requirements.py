"""Actual HDF5 reads follow visible figure families and authored variants."""

from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.providers import Hdf5Provider
from tests.generator.unit.test_sensing_cfr_lifecycle import native_cfr
from tests.visualizer.fixtures.semantic_mpc import build_standard_mpc_frame
from visualizer.src.sensing.figure_requirements import (
    ProductRequirement,
    figure_requirements,
    preparable_saved_views,
    product_requirements,
    supported_saved_views,
)
from visualizer.src.sensing.processing_inspection import inspect_processing
from visualizer.src.services.sensing_service import SensingService


@pytest.fixture(params=("maps", "reprocessing"))
def product_recording(request, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    payload = native_cfr(recording=request.param, profile=True, elements=2)
    writer = FrameSetWriter.create_new(tmp_path / "frames", compression=None)
    writer.append(replace(build_standard_mpc_frame(frame_idx=0), sensing=payload, timestamp_s=0.0))
    writer.append(replace(build_standard_mpc_frame(frame_idx=1), timestamp_s=1.0))
    writer.finalize()
    provider = Hdf5Provider(tmp_path)
    service = SensingService(SimpleNamespace(frame_source=SimpleNamespace(provider=provider)))
    yield service, provider, payload
    provider.close()


def _reads(monkeypatch):
    calls = []
    original = h5py.Dataset.__getitem__

    def read(dataset, key):
        value = original(dataset, key)
        if dataset.name.startswith("/sensing/fixed/") and dataset.name.endswith("/data"):
            calls.append((dataset.name.split("/")[3], np.asarray(value).nbytes))
        return value

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    return calls


def _numeric_keys(payload, calls):
    return {key for key, _ in calls if key in payload["products"]}


def _select(service, view, *, step=0, variant="primary", **kwargs):
    return service.resolve_display_sensing_frame(
        service.dashboard_scene_frame(step),
        scene_frame_id=step,
        storage_frame_id=step,
        product_names=("config_resolved",),
        pin_freshest=True,
        product_selector=lambda sensing: figure_requirements(
            view, variant=variant, descriptors=sensing.get("products", {}), **kwargs
        ).product_names,
    )


@pytest.mark.parametrize(
    "families", ({"saved-stft"}, {"summed-rd"}, {"range-profile"}, {"saved-stft", "range-profile"})
)
def test_spectral_families_read_only_their_authored_products(
    product_recording, monkeypatch, families
):
    service, _, payload = product_recording
    calls = _reads(monkeypatch)
    frame, rp, rd = service.dashboard_spectral_products(0, "primary", families)
    family_kind = {
        "saved-stft": "micro_doppler",
        "summed-rd": "range_doppler",
        "range-profile": "range_profile",
    }
    required = product_requirements(
        (family_kind[f] for f in families), payload["products"], "primary"
    )
    expected = {
        key
        for key, d in payload["products"].items()
        if ProductRequirement(d["kind"], d["variant"]) in required
    }
    assert _numeric_keys(payload, calls) == expected
    for key in expected:
        np.testing.assert_array_equal(frame["sensing"][key], payload[key])
    assert (rd is not None) == ("summed-rd" in families)
    assert (rp is not None) == ("range-profile" in families)


@pytest.mark.parametrize("view", ("analysis", "angular", "diagnostics", "summary", "micro-doppler"))
def test_non_input_views_never_read_complex_observations(product_recording, monkeypatch, view):
    service, _, payload = product_recording
    calls = _reads(monkeypatch)
    selection = _select(service, view)
    assert selection is not None
    expected = figure_requirements(view, descriptors=payload["products"], variant="primary")
    assert _numeric_keys(payload, calls) == set(expected.product_names) - {"config_resolved"}
    assert not any(
        payload["products"][key]["kind"]
        in {"delay_domain_slow_time", "baseband_frequency_observation"}
        for key in _numeric_keys(payload, calls)
    )


@pytest.mark.parametrize(
    "view,mode", (("processing-inputs", "inputs"), ("processing-frequency", "frequency"))
)
def test_inputs_preserve_the_existing_complex_or_power_route(
    product_recording, monkeypatch, view, mode
):
    service, _, payload = product_recording
    before = inspect_processing(payload, mode)
    calls = _reads(monkeypatch)
    selection = _select(service, view)
    after = inspect_processing(selection.display_frame["sensing"], mode)
    expected = figure_requirements(view, descriptors=payload["products"], variant="primary")
    assert len(expected.selected_products) == 1
    assert _numeric_keys(payload, calls) == set(expected.product_names) - {"config_resolved"}
    assert (after.samples is None) == (before.samples is None)
    assert after.notice == before.notice
    np.testing.assert_array_equal(
        after.samples if after.samples is not None else after.sample_power,
        before.samples if before.samples is not None else before.sample_power,
    )


def test_validated_superset_reuse_preserves_variant_and_held_binding(
    product_recording, monkeypatch
):
    service, _, payload = product_recording
    first = _select(service, "analysis")
    calls = _reads(monkeypatch)
    narrowed = _select(service, "micro-doppler", step=1)
    assert not _numeric_keys(payload, calls)
    assert narrowed.storage_frame_id == first.storage_frame_id == 0
    assert narrowed.bound_scene_frame_id == 0
    assert narrowed.stale_for_current_scene
    assert narrowed.display_frame["_sensing_display_provenance"]["current_scene_frame_id"] == 1
    raw_keys = {
        k
        for k, d in payload["products"].items()
        if d["kind"] == "range_doppler" and d["variant"] == "raw"
    }
    if raw_keys:
        calls.clear()
        raw = _select(service, "micro-doppler", variant="raw")
        assert _numeric_keys(payload, calls) == raw_keys
        for key in raw_keys:
            np.testing.assert_array_equal(raw.display_frame["sensing"][key], payload[key])


def test_filtering_deliberately_requests_both_saved_variants(product_recording, monkeypatch):
    service, _, payload = product_recording
    calls = _reads(monkeypatch)
    selection = _select(service, "processing-clutter")
    expected = {k for k, d in payload["products"].items() if d["kind"] == "range_doppler"}
    assert _numeric_keys(payload, calls) == expected
    assert (
        inspect_processing(selection.display_frame["sensing"], "clutter").notice
        == inspect_processing(payload, "clutter").notice
    )


def test_saved_stft_variant_fallback_does_not_broaden_rd_variant():
    descriptors = {
        "stft": {"kind": "micro_doppler", "variant": "raw"},
        "rd": {"kind": "range_doppler", "variant": "primary"},
    }
    req = figure_requirements(
        "micro-doppler",
        descriptors=descriptors,
        variant="primary",
        micro_mode="saved-stft",
        range_time_visible=False,
    )
    assert req.selected_products == (ProductRequirement("micro_doppler", "raw"),)
    assert req.product_names == ("config_resolved", "stft")
    assert req.spectral_families == {"saved-stft"}


def test_visible_pin_union_and_range_time_independence():
    req = figure_requirements(
        "micro-doppler",
        micro_mode="saved-stft",
        visible_products=("range_doppler_angle",),
        range_time_visible=False,
    )
    assert {p.kind for p in req.selected_products} == {"micro_doppler", "range_doppler_angle"}
    assert req.spectral_families == {"saved-stft"}
    assert not req.rich_history
    gated = figure_requirements("micro-doppler", micro_mode="gated")
    assert {p.kind for p in gated.selected_products} == {"range_doppler"}
    assert gated.spectral_families == {"range-profile"}


def test_complex_inputs_are_supported_but_not_implicitly_prepared(product_recording):
    _, _, payload = product_recording
    available = supported_saved_views(payload["products"])
    prepared = preparable_saved_views(payload["products"])
    assert "processing-inputs" in available and "processing-frequency" in available
    for view in ("processing-inputs", "processing-frequency"):
        complex_input = any(
            p.kind in {"delay_domain_slow_time", "baseband_frequency_observation"}
            for p in figure_requirements(view, descriptors=payload["products"]).selected_products
        )
        assert (view in prepared) is (not complex_input)
    assert "processing-clutter-rebuild" not in available
