"""Native scenario to retained acquisition, identities and independent CPU replay."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from generator.core.configuration.loader import resolve_sensing_document

pytestmark = pytest.mark.optional_runtime


@pytest.fixture
def ray_runtime():
    mi = pytest.importorskip("mitsuba")
    requested_variant = os.environ.get("ORCHAV_NATIVE_RCS_VARIANT", "llvm_ad_mono_polarized")
    if requested_variant not in {"llvm_ad_mono_polarized", "cuda_ad_mono_polarized"}:
        pytest.fail("ORCHAV_NATIVE_RCS_VARIANT must select a supported polarized backend")
    mi.set_variant(requested_variant)
    import drjit as dr

    original_threads = dr.thread_count()
    dr.set_thread_count(2)
    yield
    dr.set_thread_count(original_threads)


@pytest.fixture
def native_runtime(ray_runtime):
    pytest.importorskip("sionna.rt.rcs")


@pytest.mark.parametrize("preset", [None, "human", "vehicle-multi-sp"])
def test_native_yaml_generation_retention_and_cpu_replay(tmp_path, native_runtime, preset):
    from generator.core.configuration import load_simulation_config
    from generator.core.pipeline.dispatch import perform_pipeline
    from shared.frames.packed_hdf5 import PackedHDF5Reader
    from shared.frames.sensing_identity import sensing_bounce_identity
    from shared.scenarios.loader import load_scenario_configuration
    from shared.sensing.plotting.diagram import build_sensing_diagram_model

    root = Path(__file__).resolve().parents[3]
    example = root / "examples/native_rcs"
    config = resolve_sensing_document(yaml.safe_load((example / "scenario.yaml").read_text()))
    target = config["actors"]["targets"][0]
    if preset is not None:
        target["representation"] = "tr38901"
        target.pop("constant_rcs")
        target["geometry"] = {"type": "box"}
        target["tr38901"] = {"object_type": preset, "model_type": 2}
    (tmp_path / "scene.xml").write_text((example / "scene.xml").read_text())
    (tmp_path / "scenario.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    scenario = load_scenario_configuration(tmp_path, project_root=root)
    simulation = load_simulation_config(scenario)
    perform_pipeline(
        simulation_config=simulation, scenario_configuration=scenario, show_progress=False
    )
    reader = PackedHDF5Reader(scenario.frames_dir)
    try:
        assert reader.frame_ids == list(range(32))
        start = reader.load_standard_frame(0)
        completed = reader.load_standard_frame(15)
        metadata = completed.targets_metadata[0]
        expected_count = 5 if preset == "vehicle-multi-sp" else 1
        assert len(metadata["scattering_point_ids"]) == expected_count
        assert completed.sensing["schema_version"] == 4
        assert "detections" in completed.sensing
        diagram = build_sensing_diagram_model(completed.sensing)
        assert diagram.evidence_status == "BOUNDED"
        assert diagram.range_processing == "finite_bandwidth_delay_response"
        assert completed.provenance["source_rt_frame_idx"] == 0
        native = completed.provenance["sionna_sensing"]
        assert native["channel"] == "combined"
        assert native["engine_versions"]["sionna-rt"] == "2.2.0"
        np.testing.assert_allclose(
            native["target_state_at_solve"][0]["current_position"],
            start.target_positions_m[0],
            atol=1e-5,
        )
        # Output-time marker state and source-solve state are distinct and explicit.
        assert (
            metadata["current_position"][0]
            < native["target_state_at_solve"][0]["current_position"][0]
        )
        assert (
            metadata["scattering_points_world_m"]
            != start.targets_metadata[0]["scattering_points_world_m"]
        )
        for bounce in np.flatnonzero(completed.interactions == 16):
            target_id, point_id = sensing_bounce_identity(completed, int(bounce))
            assert target_id == 0
            assert point_id in metadata["scattering_point_ids"]
    finally:
        reader.close()

    # A fresh process verifies replay owns no live Sionna/Dr.Jit/Mitsuba objects.
    code = """
import json,sys,numpy as np
from shared.sensing.loader import load_frames_sensing
from shared.sensing.postprocessor import SensingPostProcessor,PostProcessingConfig
frames=load_frames_sensing(sys.argv[1])
errors=[]
for frame in frames:
    sensing=frame['_typed_sensing']
    replay=SensingPostProcessor(PostProcessingConfig(skip_detection=False)).process_from_frequency_observation(sensing)
    errors.append(float(np.max(np.abs(replay['rd_cube']-sensing['range_doppler_by_pair'][0]))))
assert not any(name in sys.modules for name in ('sionna.rt','mitsuba','drjit'))
print(json.dumps({'count':len(frames),'errors':errors}))
"""
    env = dict(os.environ, PYTHONPATH=str(root), LOKY_MAX_CPU_COUNT="2")
    result = subprocess.run(
        [sys.executable, "-c", code, str(scenario.frames_dir)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    report = json.loads(result.stdout)
    assert report["count"] == 2
    assert report["errors"] == [0.0, 0.0]


def test_existing_mesh_lane_runs_on_candidate_engine(tmp_path, ray_runtime):
    import mitsuba as mi

    from generator.core.configuration import load_simulation_config
    from generator.core.pipeline.dispatch import perform_pipeline
    from shared.frames.packed_hdf5 import PackedHDF5Reader
    from shared.scenarios.loader import load_scenario_configuration

    root = Path(__file__).resolve().parents[3]
    example = root / "examples/native_rcs"
    config = resolve_sensing_document(yaml.safe_load((example / "scenario.yaml").read_text()))
    target = config["actors"]["targets"][0]
    for key in ("representation", "constant_rcs", "geometry"):
        target.pop(key)
    target["asset"] = {"source": "file", "path": "cube.ply", "material_type": "metal"}
    target["mobility"]["start_m"] = [5, 0, 1]
    target["mobility"]["end_m"] = [4.907, 0, 1]
    config["raytracing"].pop("rcs")
    config["raytracing"]["quality"]["custom"]["specular_reflection"] = True
    config["raytracing"]["quality"]["custom"]["samples_per_src"] = 1000
    config["actors"]["rx"][0]["mobility"]["position_m"] = [1, 0, 1]
    config["sensing"]["range_mode"] = "bistatic"
    mi.load_dict({"type": "cube"}).write_ply(str(tmp_path / "cube.ply"))
    (tmp_path / "scene.xml").write_text((example / "scene.xml").read_text())
    (tmp_path / "scenario.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    scenario = load_scenario_configuration(tmp_path, project_root=root)
    perform_pipeline(
        simulation_config=load_simulation_config(scenario),
        scenario_configuration=scenario,
        show_progress=False,
    )
    reader = PackedHDF5Reader(scenario.frames_dir)
    try:
        frame = reader.load_standard_frame(15)
        assert frame.num_paths > 0
        assert 1 in frame.interactions  # A real material reflection, not only LOS.
        assert 16 not in frame.interactions
        assert "sionna_sensing" not in frame.provenance
        assert frame.targets_metadata[0]["material_type"] == "metal"
        assert frame.sensing["schema_version"] == 4
    finally:
        reader.close()
