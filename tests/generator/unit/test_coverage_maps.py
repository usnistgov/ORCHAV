import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

import generator.core.coverage as coverage_pkg
from generator.core.configuration import CoverageConfig, SimulationConfig
from generator.core.coverage import bounds as coverage_bounds
from generator.core.coverage import compute_coverage_map, solver
from generator.core.coverage import metrics as coverage_metrics
from generator.core.coverage.quality import CoverageQuality
from shared.coverage.schema import COVERAGE_NO_SERVING_TX


def test_coverage_public_surface_has_no_legacy_grid_helpers():
    assert coverage_pkg.CoverageBoundsError is coverage_bounds.CoverageBoundsError
    assert coverage_pkg.__all__ == ["CoverageBoundsError", "compute_coverage_map"]
    assert not hasattr(coverage_pkg, "CoverageQuality")
    assert not hasattr(coverage_pkg, "generate_coverage_grid")
    assert not hasattr(solver, "generate_points_simple")


def test_coverage_metric_conversions_preserve_current_numeric_behavior():
    values = np.asarray([1.0, 10.0, 0.0, -1.0, np.nan], dtype=np.float32)

    db_values = coverage_metrics._db_from_linear(values)

    np.testing.assert_allclose(db_values[:2], [0.0, 10.0], rtol=1e-6)
    assert db_values.dtype == np.float32
    assert np.isnan(db_values[2:]).all()

    path_loss = coverage_metrics._path_loss_db(np.asarray([1.0, 0.01], dtype=np.float32))
    np.testing.assert_allclose(path_loss, [-0.0, 20.0], atol=1e-6)
    assert path_loss.dtype == np.float32


def test_coverage_metric_tx_power_prefers_dbm_then_power_then_zero():
    assert coverage_metrics._tx_power_dbm(SimpleNamespace(power_dbm="12.5", power=1.0)) == 12.5
    assert coverage_metrics._tx_power_dbm(SimpleNamespace(power="7.0")) == 7.0
    assert coverage_metrics._tx_power_dbm(SimpleNamespace(power_dbm=object(), power="bad")) == 0.0


@pytest.mark.parametrize(
    "canonical_shape",
    [
        (1, 3, 1),
        (1, 1, 4),
        (1, 1, 1),
        (2, 3, 1),
        (2, 1, 4),
    ],
)
def test_radio_map_normalization_preserves_singleton_spatial_axes(canonical_shape):
    canonical = np.arange(np.prod(canonical_shape), dtype=np.float32).reshape(canonical_shape)

    normalized = solver._normalise_radio_map_tensor(
        canonical,
        canonical_shape[0],
        name="path_gain",
    )

    np.testing.assert_array_equal(normalized, canonical)


@pytest.mark.parametrize("shape", [(2, 4), (1, 2, 4, 3), (3, 2, 4)])
def test_radio_map_normalization_rejects_non_sionna_axis_contract(shape):
    with pytest.raises(ValueError, match=r"expected \(2, ny, nx\)"):
        solver._normalise_radio_map_tensor(
            np.ones(shape, dtype=np.float32),
            2,
            name="path_gain",
        )


def test_compute_coverage_map_uses_single_solver_entrypoint(monkeypatch):
    sionna = types.ModuleType("sionna")
    sionna_rt = types.ModuleType("sionna.rt")
    sionna_rt.RadioMapSolver = object
    sionna.rt = sionna_rt
    monkeypatch.setitem(sys.modules, "sionna", sionna)
    monkeypatch.setitem(sys.modules, "sionna.rt", sionna_rt)

    captured = {}

    def fake_solver_runner(
        scene,
        tx_list,
        rx_list,
        target_objects,
        coverage_config,
        simulation_config,
        *,
        bbox,
        quality_settings=None,
    ):
        captured.update(
            {
                "scene": scene,
                "tx_list": tx_list,
                "rx_list": rx_list,
                "target_objects": target_objects,
                "coverage_config": coverage_config,
                "simulation_config": simulation_config,
                "bbox": bbox,
                "quality_settings": quality_settings,
            }
        )
        return {"ok": True}

    monkeypatch.setattr(solver, "_run_radio_map_solver", fake_solver_runner)

    scene = object()
    simulation_config = SimulationConfig(
        scene_name="etoile",
        quality="low",
        bandwidth_hz=123.0,
        temperature_k=456.0,
    )
    coverage_config = CoverageConfig(
        enabled=True,
        bbox=((-4.0, 6.0), (-2.0, 8.0), (0.0, 2.0)),
        resolution=(1.0, 1.0),
        heights=[1.5],
    )

    result = compute_coverage_map(
        scene,
        tx_list=[],
        rx_list=[],
        target_objects=[],
        coverage_config=coverage_config,
        simulation_config=simulation_config,
    )

    assert result == {"ok": True}
    assert captured["scene"] is scene
    assert captured["coverage_config"] is coverage_config
    assert captured["simulation_config"] is simulation_config
    assert captured["bbox"] == coverage_config.bbox
    assert captured["quality_settings"]["samples_per_tx"] == int(
        SimulationConfig.QUALITY_PRESETS["medium"]["samples_per_src"]
    )
    assert captured["quality_settings"]["max_depth"] == int(
        SimulationConfig.QUALITY_PRESETS["medium"]["max_depth"]
    )


def test_auto_bbox_uses_scene_mesh_bounds():
    class FakePoint:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class FakeMesh:
        def __init__(self, lower, upper):
            self._lower = FakePoint(*lower)
            self._upper = FakePoint(*upper)

        def bbox(self):
            return SimpleNamespace(min=self._lower, max=self._upper)

    scene = SimpleNamespace(
        objects={"building": SimpleNamespace(mi_mesh=FakeMesh((0.0, 5.0, 0.0), (10.0, 20.0, 3.0)))}
    )
    coverage_config = CoverageConfig(enabled=True, resolution=(1.0, 1.0))

    bbox = coverage_bounds.resolve_coverage_bbox(coverage_config, scene=scene)

    assert bbox == ((-5.0, 15.0), (0.0, 25.0), (-1.0, 4.0))


def test_auto_bbox_ignores_non_finite_scene_mesh_bounds():
    class FakePoint:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class FakeMesh:
        def __init__(self, lower, upper):
            self._lower = FakePoint(*lower)
            self._upper = FakePoint(*upper)

        def bbox(self):
            return SimpleNamespace(min=self._lower, max=self._upper)

    scene = SimpleNamespace(
        objects={
            "bad": SimpleNamespace(mi_mesh=FakeMesh((np.nan, 0.0, 0.0), (10.0, 10.0, 2.0))),
            "good": SimpleNamespace(mi_mesh=FakeMesh((100.0, 200.0, 0.0), (120.0, 230.0, 5.0))),
        }
    )
    coverage_config = CoverageConfig(enabled=True, resolution=(10.0, 10.0))

    bbox = coverage_bounds.resolve_coverage_bbox(coverage_config, scene=scene)

    assert bbox == ((90.0, 130.0), (190.0, 240.0), (-1.0, 6.0))


def test_auto_bbox_requires_scene_bounds():
    coverage_config = CoverageConfig(enabled=True, resolution=(1.0, 1.0))

    with pytest.raises(ValueError, match="requires scene geometry bounds"):
        coverage_bounds.resolve_coverage_bbox(
            coverage_config,
            scene=SimpleNamespace(objects={}),
            scenario_context=None,
        )


def test_auto_bbox_uses_shared_cached_xml_geometry(monkeypatch):
    coverage_config = CoverageConfig(enabled=True, resolution=(1.0, 1.0))
    scenario_context = SimpleNamespace(scene_xml="scene.xml")
    geometry = [
        {
            "mesh": SimpleNamespace(
                vertices=np.asarray(
                    [
                        [0.0, 5.0, 0.0],
                        [10.0, 20.0, 3.0],
                    ],
                    dtype=float,
                )
            )
        }
    ]
    calls = []

    def fake_get_scene_geometry(*, scenario_context=None, scene_xml=None):
        calls.append((scenario_context, scene_xml))
        return geometry

    monkeypatch.setattr(coverage_bounds, "get_scene_geometry", fake_get_scene_geometry)

    bbox = coverage_bounds.resolve_coverage_bbox(
        coverage_config,
        scene=SimpleNamespace(objects={}),
        scenario_context=scenario_context,
    )

    assert calls == [(scenario_context, None)]
    assert bbox == ((-5.0, 15.0), (0.0, 25.0), (-1.0, 4.0))


def test_coverage_quality_accepts_samples_per_src_custom_override():
    simulation_config = SimulationConfig(quality="medium")
    scenario_context = SimpleNamespace(
        coverage_cfg={
            "quality": {
                "custom": {
                    "samples_per_src": 12345,
                    "max_depth": 4,
                    "diffraction": True,
                }
            }
        }
    )

    quality = CoverageQuality.from_context(simulation_config, scenario_context=scenario_context)

    assert quality.to_radio_map_args()["samples_per_tx"] == 12345
    assert quality.to_radio_map_args()["max_depth"] == 4
    assert quality.to_radio_map_args()["diffraction"] is True


def test_coverage_quality_merges_solver_preset_with_quality_custom():
    simulation_config = SimulationConfig(quality="low")
    scenario_context = SimpleNamespace(
        coverage_cfg={
            "solver": {"preset": "medium"},
            "quality": {
                "custom": {
                    "samples_per_src": 2468,
                    "diffraction": True,
                }
            },
        }
    )

    args = CoverageQuality.from_context(
        simulation_config, scenario_context=scenario_context
    ).to_radio_map_args()

    assert args["max_depth"] == int(SimulationConfig.QUALITY_PRESETS["medium"]["max_depth"])
    assert args["samples_per_tx"] == 2468
    assert args["diffraction"] is True


def test_solver_settings_accept_raytracing_sample_name():
    settings = solver._apply_solver_settings_overrides(
        {
            "samples_per_tx": 100,
            "max_depth": 2,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": True,
            "refraction": True,
            "diffraction": False,
        },
        {
            "samples_per_src": 321,
            "max_depth": 5,
            "diffraction": True,
        },
    )

    assert settings["samples_per_tx"] == 321
    assert settings["max_depth"] == 5
    assert settings["diffraction"] is True


def test_serving_tx_uses_negative_one_for_no_signal_cells():
    path_gain = np.array(
        [
            [
                [
                    [[1.0, np.nan], [0.2, np.nan]],
                    [[0.5, np.nan], [0.4, np.nan]],
                ]
            ]
        ],
        dtype=np.float32,
    )

    derived = solver._derive_coverage_layers(
        path_gain,
        tx_power_dbm=np.array([0.0, 0.0], dtype=np.float32),
        noise_power_w=1e-12,
        derived_requested=["serving_tx", "best_path_loss_db"],
        metric_name="best_path_loss_db",
    )

    np.testing.assert_array_equal(
        derived["serving_tx"],
        np.array([[[[0, COVERAGE_NO_SERVING_TX], [1, COVERAGE_NO_SERVING_TX]]]], dtype=np.int16),
    )
    assert np.isnan(derived["best_path_loss_db"][0, 0, 0, 1])


def test_sum_power_preserves_cells_without_finite_coverage():
    path_gain = np.array(
        [[[[[np.nan, 1.0]], [[np.nan, np.nan]]]]],
        dtype=np.float32,
    )

    derived = solver._derive_coverage_layers(
        path_gain,
        tx_power_dbm=np.array([0.0, 0.0], dtype=np.float32),
        noise_power_w=1e-12,
        derived_requested=["sum_rss_dbm"],
        metric_name="sum_rss_dbm",
    )

    summed = derived["sum_rss_dbm"]
    assert summed.shape == (1, 1, 1, 2)
    assert summed.dtype == np.float32
    assert np.isnan(summed[0, 0, 0, 0])
    # The coverage layer is float32, so watts-to-dBm conversion can retain
    # micro-dB roundoff around the physically exact 0 dBm result.
    assert summed[0, 0, 0, 1] == pytest.approx(0.0, abs=1e-5)


def test_generator_sinr_retains_weak_interference_at_high_dynamic_range():
    path_gain = np.asarray(
        [[[[[1.0]], [[1.0e-20]], [[2.0e-20]]]]],
        dtype=np.float32,
    )

    derived = solver._derive_coverage_layers(
        path_gain,
        tx_power_dbm=np.zeros(3, dtype=np.float32),
        noise_power_w=1.0e-30,
        derived_requested=["sinr_linear"],
        metric_name="sinr_linear",
    )

    assert derived["sinr_linear"].item() == pytest.approx(1.0 / 3.0e-20, rel=1.0e-6)


def test_radio_map_solver_runner_reports_height_progress(monkeypatch, capsys):
    class FakePoint3f:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class FakePoint2f:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class FakeRadioMap:
        path_gain = np.ones((1, 2, 2), dtype=np.float32)

    class FakeRadioMapSolver:
        def __call__(
            self,
            scene,
            center,
            orientation,
            size,
            cell_size,
            samples_per_tx,
            max_depth,
            los,
            specular_reflection,
            diffuse_reflection,
            refraction,
            diffraction=False,
        ):
            return FakeRadioMap()

    mitsuba = types.ModuleType("mitsuba")
    mitsuba.Point3f = FakePoint3f
    mitsuba.Point2f = FakePoint2f
    mitsuba.TensorXf = list
    sionna = types.ModuleType("sionna")
    sionna_rt = types.ModuleType("sionna.rt")
    sionna_rt.PlanarArray = object
    sionna_rt.RadioMapSolver = FakeRadioMapSolver
    sionna_rt.load_scene = lambda *args, **kwargs: None
    sionna_rt_scene_object = types.ModuleType("sionna.rt.scene_object")
    sionna_rt_scene_object.SceneObject = object
    sionna.rt = sionna_rt
    monkeypatch.setitem(sys.modules, "mitsuba", mitsuba)
    monkeypatch.setitem(sys.modules, "sionna", sionna)
    monkeypatch.setitem(sys.modules, "sionna.rt", sionna_rt)
    monkeypatch.setitem(sys.modules, "sionna.rt.scene_object", sionna_rt_scene_object)

    scene = SimpleNamespace(tx_array=object(), rx_array=object())
    coverage_config = CoverageConfig(
        enabled=True,
        bbox=((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)),
        resolution=(1.0, 1.0),
        heights=[0.5, 1.5],
    )

    result = solver._run_radio_map_solver(
        scene,
        tx_list=[],
        rx_list=[],
        target_objects=[],
        coverage_config=coverage_config,
        simulation_config=SimulationConfig(quality="low"),
        bbox=coverage_config.bbox,
        quality_settings={
            "samples_per_tx": 10,
            "max_depth": 1,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": False,
            "refraction": False,
            "diffraction": False,
        },
    )

    stderr = capsys.readouterr().err
    assert result is not None
    assert "Coverage: solving 2 height slice(s)" in stderr
    assert "Coverage: solving height 1/2 (0.50m)" in stderr
    assert "Coverage: solving height 2/2 (1.50m)" in stderr
    assert "Coverage [" in stderr
    assert "1/2" in stderr
    assert "2/2" in stderr
