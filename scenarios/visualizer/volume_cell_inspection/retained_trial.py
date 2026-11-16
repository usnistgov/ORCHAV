"""One deterministic synthetic retained trial for real Open trial verification.

Uses canonical v4 contracts and packed-v2 storage. No solver or detector runs;
all reports and truth are explicitly authored QA values.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

from shared.extensions.sensing import assert_sensing_contract, canonical_sensing_pair_id
from shared.frames import FrameSetWriter, standard_mpc_frame_from_pair_data
from shared.frames.providers import Hdf5Provider
from shared.sensing.fusion.experiment import fingerprints
from shared.sensing.loader import load_frames_sensing
from shared.sensing.targets import build_target_catalog, build_target_state, build_target_truth


def prepare_retained_trial(root: Path, trial_id="fixture-z0-y0-x2") -> dict:
    """Publish one matched synthetic recording and bind its exact hashes."""
    root = Path(root).resolve()
    path = root / "sensing_volume.json"
    manifest = json.loads(path.read_text())
    trial = next(row for row in manifest["trials"] if row["id"] == trial_id)
    position = np.asarray(trial["position_m"], dtype=float)
    sensor = np.asarray(manifest["sensor_position_m"], dtype=float)
    velocity = np.asarray(trial["velocity_m_s"], dtype=float)
    local = position - sensor
    distance = float(np.linalg.norm(local))
    closing = -float(np.dot(local, velocity) / distance)
    target_id = "controlled-volume-target"
    pair_id = canonical_sensing_pair_id("FixtureTX", "FixtureRX")
    report = {
        "detection_id": "controlled-report",
        "target_id": target_id,
        "position_m": position.tolist(),
        "range_m": distance,
        "closing_velocity_m_s": closing,
        "velocity_m_s": closing,
        "x_m": float(local[0]),
        "y_m": float(local[1]),
        "z_m": float(local[2]),
        "snr_db": 30.0,
        "source": "authored_synthetic_fixture",
    }
    # Reuse the repository's exact typed test fixture builder rather than
    # inventing an ad hoc sensing schema. A normal repository checkout contains it.
    repository = Path(__file__).resolve().parents[3]
    fixture_path = repository / "tests/visualizer/fixtures/sensing_v3.py"
    spec = importlib.util.spec_from_file_location("orchav_typed_fixture", fixture_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    range_axis = np.linspace(0.0, 2 * distance, 17)
    doppler_center = 2 * closing * 28.0e9 / 299792458.0
    report["doppler_hz"] = doppler_center
    doppler_axis = doppler_center + np.linspace(-200.0, 200.0, 9)
    profile = np.full((1, 17), 0.001, dtype=np.float32)
    profile[0, 8] = 1.0
    rd_power = np.full((1, 9, 17), 0.001, dtype=np.float32)
    rd_power[0, 4, 8] = 1.0
    sensing = module.build_visualizer_sensing_payload(
        {
            "range_axis_m": range_axis,
            "doppler_axis_hz": doppler_axis,
            "range_profile_by_pair": profile,
            "range_doppler_power_by_pair": rd_power,
            "detections": [report],
        },
        scene_frame_id=0,
        sample_time_s=0.0,
    )
    sensing["temporal"]["acquisition"].update(
        pair_id=pair_id, tx_sensor_id="FixtureTX", rx_sensor_id="FixtureRX"
    )
    sensing["temporal"]["truth"] = {
        "reference_time_s": 0.0,
        "scene_frame_id": 0,
        "policy": "authored_exact_synthetic_truth",
        "source": "controlled_visual_fixture",
    }
    catalog = build_target_catalog(
        [
            {
                "target_id": target_id,
                "display_name": "Controlled synthetic target",
                "identity_source": "authored fixture",
            }
        ]
    )
    state_row = {
        "target_id": target_id,
        "lifecycle": "active",
        "valid": True,
        "position_m": position.tolist(),
        "orientation_rad": [0, 0, 0],
        "velocity_m_s": velocity.tolist(),
        "invalid_reason": None,
    }
    sensing["target_catalog"] = catalog
    sensing["target_state"] = build_target_state(
        catalog, scene_frame_id=0, reference_time_s=0.0, targets=[state_row]
    )
    observable = {
        "pair_id": pair_id,
        "bistatic_path_length_m": 2 * distance,
        "monostatic_equivalent_range_m": distance,
        "path_closing_rate_m_s": 2 * closing,
        "monostatic_equivalent_velocity_m_s": closing,
        "bistatic_bisector_velocity_m_s": closing,
        "undefined_reason": None,
        "velocity_valid": True,
        "velocity_derivation_source": "analytic_velocity_projection",
        "velocity_reference_dt_s": None,
    }
    sensing["truth"] = build_target_truth(
        catalog, targets=[{**state_row, "pair_observables": [observable]}]
    )
    sensing["fixture"] = (
        "Synthetic report and truth authored for UI verification; no detector or propagation solver"
    )
    module.materialize_visualizer_product_specs(sensing)
    assert_sensing_contract(
        {"sensing": sensing}, require_schema_version=True, require_described_products=True
    )
    native = root / "native" / trial_id
    native.mkdir(parents=True, exist_ok=True)
    assets = native / "assets"
    assets.mkdir(exist_ok=True)
    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ]
    )
    with (assets / "controlled_target.ply").open("w") as stream:
        stream.write(
            "ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\nelement face 12\nproperty list uchar int vertex_indices\nend_header\n"
        )
        np.savetxt(stream, vertices, fmt="%.6g")
        np.savetxt(stream, np.column_stack((np.full(12, 3), faces)), fmt="%d")
    config = {
        "schema_version": 2,
        "scene": {"source": "local", "id": "../../scene.xml"},
        "data": {"mode": "files", "files": {"format": "hdf5", "directory": "frames"}},
        "raytracing": {"enabled": False},
        "sensing": {
            "enabled": True,
            "range_mode": "monostatic",
            "processing_mode": "analytic_path_doppler_oracle",
            "output_range_profile": True,
            "output_range_doppler": True,
            "output_detections": True,
        },
        "timeline": {"steps": 1, "duration_s": 0.0},
        "actors": {
            role: [
                {"name": name, "mobility": {"type": "stationary", "position_m": sensor.tolist()}}
            ]
            for role, name in (("tx", "FixtureTX"), ("rx", "FixtureRX"))
        },
        "generator_summary": {"enabled": False},
    }
    (native / "scenario.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    frame = standard_mpc_frame_from_pair_data(
        frame_index=0,
        timestamp_s=0.0,
        tx_rx_pairs=[[0, 0]],
        tx_positions=[sensor],
        rx_positions=[sensor],
        tx_names=["FixtureTX"],
        rx_names=["FixtureRX"],
        vertices_by_pair=[np.empty((0, 0, 3))],
        interactions_by_pair=[np.empty((0, 0), np.uint8)],
        target_positions_m=[position],
        targets_metadata=[
            {
                "name": target_id,
                "target_id": target_id,
                "current_position": position.tolist(),
                "mesh_file": "controlled_target.ply",
                "mesh_directory": "assets",
                "scale": 1.0,
                "orientation": [0, 0, 0],
                "material_type": "concrete",
                "use_ply_position": False,
            }
        ],
        sensing=sensing,
        provenance={
            "kind": "controlled_visual_fixture",
            "simulation": False,
            "trial_id": trial_id,
            "source": "authored synthetic report and truth",
        },
    )
    with FrameSetWriter.for_scenario(native) as writer:
        writer.append(frame)
        writer.finalize(
            provenance={
                "kind": "controlled_visual_fixture",
                "simulation": False,
                "trial_id": trial_id,
            }
        )
    provider = Hdf5Provider(native)
    try:
        restored = provider.load_frame(0)
        assert restored.sensing["detections"][0]["detection_id"] == report["detection_id"]
        np.testing.assert_array_equal(restored.rx_positions, [sensor])
        np.testing.assert_array_equal(
            restored.sensing["truth"]["targets"][0]["position_m"], position
        )
    finally:
        provider.close()
    loaded = load_frames_sensing(native)
    assert len(loaded) == 1
    trial.update(
        native=str(native.relative_to(root)),
        native_evidence="included",
        source_hashes=fingerprints(native / "frames"),
        configuration_sha256=hashlib.sha256((native / "scenario.yaml").read_bytes()).hexdigest(),
        reports=[
            {
                key: report[key]
                for key in ("detection_id", "position_m", "range_m", "closing_velocity_m_s")
            }
        ],
        hit=True,
        tp=1,
        fp=0,
        fn=0,
        matched_error_m=0.0,
        radial_closing_velocity_m_s=closing,
    )
    path.write_text(json.dumps(manifest, indent=2))
    return {
        "trial_id": trial_id,
        "position_m": position.tolist(),
        "native": str(native),
        "target_id": target_id,
        "report_id": report["detection_id"],
        "pair_id": pair_id,
        "source_hashes": trial["source_hashes"],
        "configuration_sha256": trial["configuration_sha256"],
        "provider_readback": True,
        "sensing_loader_records": len(loaded),
        "synthetic": True,
    }
