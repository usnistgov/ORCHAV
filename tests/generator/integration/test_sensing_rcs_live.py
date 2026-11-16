"""Bounded native RT 2.2 solves on the polarized LLVM CPU backend."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from generator.core.exceptions import ConfigurationError
from generator.core.propagation.native_rcs import compute_native_paths
from generator.core.target.rcs import NativeTargetManager
from generator.core.target.representation import NativeTargetConfig
from shared.scenarios.actors import ConstantRCSSpec, NativeBoxGeometrySpec, TR38901RCSSpec

pytestmark = [pytest.mark.integration, pytest.mark.optional_runtime]

_C = 299_792_458.0
_FREQUENCY = 3.5e9


@pytest.fixture(scope="module")
def engine():
    mi = pytest.importorskip("mitsuba")
    requested_variant = os.environ.get("ORCHAV_NATIVE_RCS_VARIANT", "llvm_ad_mono_polarized")
    if requested_variant not in {"llvm_ad_mono_polarized", "cuda_ad_mono_polarized"}:
        pytest.fail("ORCHAV_NATIVE_RCS_VARIANT must select a supported polarized backend")
    if mi.variant() is None:
        mi.set_variant(requested_variant)
    if mi.variant() != requested_variant:
        pytest.fail(f"native sensing tests requested {requested_variant}, got {mi.variant()}")
    dr = pytest.importorskip("drjit")
    rt = pytest.importorskip("sionna.rt")
    rcs = pytest.importorskip("sionna.rt.rcs")
    if not hasattr(rcs, "ConstantRCSSensingTarget"):
        pytest.skip("Sionna RT 2.2 native sensing targets are unavailable")
    previous_thread_count = dr.thread_count()
    dr.set_thread_count(2)
    try:
        yield rt
    finally:
        dr.set_thread_count(previous_thread_count)


def _simulation(engine, *, velocity=(0, 0, 0), rx_position=(0, 0, 1.5), tr38901=None):
    scene = engine.load_scene()
    scene.frequency = _FREQUENCY
    scene.tx_array = engine.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = engine.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.add(engine.Transmitter(name="tx", position=[0, 0, 1.5]))
    scene.add(engine.Receiver(name="rx", position=rx_position))
    config = NativeTargetConfig(
        name="native",
        representation="constant_rcs" if tr38901 is None else "tr38901",
        geometry=NativeBoxGeometrySpec(),
        constant_rcs=ConstantRCSSpec(sigma_m2=1) if tr38901 is None else None,
        tr38901=tr38901,
        _initial_position=(40, 0, 1.5),
        _initial_velocity=velocity,
    )
    manager = NativeTargetManager(config, scene)
    manager.create_target()
    return SimpleNamespace(
        scene=scene,
        tx_list=list(scene.transmitters.values()),
        rx_list=list(scene.receivers.values()),
        target_managers=[manager],
        settings={
            "rcs": {
                "channel": "target_only",
                "max_depth": 1,
                "samples_per_sp": 32,
                "buffer_size_per_sp": 32,
                "los": True,
                "specular_reflection": False,
                "refraction": False,
                "seed": 42,
            }
        },
        path_solver=engine.PathSolver(),
        rcs_solver=None,
    )


def _background_kwargs(simulation):
    return {
        "scene": simulation.scene,
        "max_depth": 1,
        "samples_per_src": 32,
        "max_num_paths_per_src": 32,
        "synthetic_array": True,
        "los": True,
        "specular_reflection": False,
        "refraction": False,
        "seed": 42,
    }


def _solve(simulation, channel="target_only"):
    simulation.settings["rcs"]["channel"] = channel
    return compute_native_paths(simulation, _background_kwargs(simulation))


def _valid_channel(paths):
    a, delay = paths.cir(normalize_delays=False, out_type="numpy")
    valid = np.asarray(paths.valid).reshape(-1)
    return a.reshape(-1)[valid], delay.reshape(-1)[valid]


@pytest.mark.parametrize("velocity_mps", [0.0, -3.0, 3.0])
def test_constant_monostatic_matches_radar_equation_delay_and_doppler(engine, velocity_mps):
    simulation = _simulation(engine, velocity=(velocity_mps, 0, 0))
    paths, metadata = _solve(simulation)
    a, delay = _valid_channel(paths)
    assert len(a) == 1
    expected_power = (_C / _FREQUENCY) ** 2 / ((4 * np.pi) ** 3 * 40**4)
    assert abs(a[0]) ** 2 == pytest.approx(expected_power, rel=3e-5)
    assert delay[0] == pytest.approx(80 / _C, rel=2e-6)
    doppler = np.asarray(paths.doppler).reshape(-1)[0]
    assert doppler == pytest.approx(-2 * velocity_mps * _FREQUENCY / _C, abs=1e-4)
    interactions = np.asarray(paths.interactions)
    assert np.count_nonzero(interactions == 16) == 1
    assert metadata["target_catalog_names"] == ["native"]
    assert metadata["delay_convention"] == "absolute"


@pytest.mark.parametrize(
    "object_type,model_type",
    [
        ("uav-small-size", 1),
        ("human", 1),
        ("human", 2),
        ("uav-large-size", 2),
        ("vehicle-single-sp", 2),
        ("vehicle-multi-sp", 2),
        ("agv-single-sp", 2),
        ("agv-multi-sp", 2),
    ],
)
def test_all_native_presets_produce_finite_physical_paths(engine, object_type, model_type):
    model = TR38901RCSSpec(object_type=object_type, model_type=model_type)
    simulation = _simulation(engine, tr38901=model)
    paths, _ = _solve(simulation)
    a, delay = _valid_channel(paths)
    assert len(a) == model.num_scattering_points
    assert np.all(np.isfinite(a)) and np.all(np.abs(a) > 0)
    assert np.all(np.isfinite(delay)) and np.all(delay > 0)
    assert np.count_nonzero(np.asarray(paths.interactions) == 16) == len(a)


def test_combined_channel_contains_each_background_and_target_path_once(engine):
    simulation = _simulation(engine, rx_position=(0, 5, 1.5))
    target, _ = _solve(simulation, "target_only")
    background, _ = _solve(simulation, "background_only")
    combined, metadata = _solve(simulation, "combined")
    target_a, target_delay = _valid_channel(target)
    background_a, background_delay = _valid_channel(background)
    combined_a, combined_delay = _valid_channel(combined)
    assert len(target_a) == len(background_a) == 1
    assert len(combined_a) == 2
    order = np.argsort(combined_delay)
    np.testing.assert_allclose(
        combined_delay[order], [background_delay[0], target_delay[0]], rtol=2e-6
    )
    np.testing.assert_allclose(combined_a[order], [background_a[0], target_a[0]], rtol=2e-6)
    assert np.count_nonzero(np.asarray(background.interactions) == 16) == 0
    assert np.count_nonzero(np.asarray(combined.interactions) == 16) == 1
    assert metadata["channel"] == "combined"
    assert len(simulation.scene.sensing_targets) == 1
    assert len(simulation.scene.objects) == 1


def test_absorbing_proxy_occludes_background_without_ordinary_scattering(engine):
    simulation = _simulation(engine, rx_position=(80, 0, 1.5))
    background, _ = _solve(simulation, "background_only")
    a, delay = _valid_channel(background)
    assert len(a) == len(delay) == 0
    assert simulation.rcs_solver is None


def test_target_solver_empty_return_is_preserved(engine):
    simulation = _simulation(engine)
    simulation.settings["rcs"]["los"] = False
    paths, metadata = _solve(simulation)
    a, delay = _valid_channel(paths)
    assert len(a) == len(delay) == 0
    assert metadata["channel"] == "target_only"


def test_native_solver_rejects_unsupported_topology_before_solving(engine):
    simulation = _simulation(engine)
    kwargs = _background_kwargs(simulation)
    kwargs["synthetic_array"] = False
    with pytest.raises(ConfigurationError, match="synthetic_array"):
        compute_native_paths(simulation, kwargs)
    simulation.scene.add(engine.Transmitter(name="tx2", position=[0, 1, 1.5]))
    simulation.tx_list = list(simulation.scene.transmitters.values())
    with pytest.raises(ConfigurationError, match="one TX/RX pair"):
        compute_native_paths(simulation, _background_kwargs(simulation))
    assert simulation.rcs_solver is None
