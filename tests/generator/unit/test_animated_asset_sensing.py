"""Animation identities survive native sensing and unchanged packed-HDF5 v2."""

from copy import deepcopy
from types import SimpleNamespace

import h5py
import numpy as np

from generator.core.pipeline.schedule import FixedIntervalScheduler
from generator.core.pipeline.sensing import SensingPipelineState
from generator.core.scenario_actors.animated_asset import prepare_animated_asset
from generator.core.scenario_actors.types import Timeline
from shared.animated_target_workflow import _write_synthetic_asset
from shared.frames.manifest import manifest_from_chunks, write_frame_manifest_atomic
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed_hdf5 import PackedHDF5Reader
from shared.frames.packed_hdf5_writer import PackedMPCChunkWriter
from shared.sensing.targets import validate_target_state
from tests.generator.unit.test_pipeline_sensing_temporal_identity import (
    ANALYTIC_ORACLE_MODE,
    GENERATION_ID,
    _execution,
    _multi_target_frame,
    _oracle_sensing,
)


def test_native_animation_catalog_and_pose_root_state_roundtrip(tmp_path):
    _write_synthetic_asset(tmp_path / "asset")
    asset = {
        "source": "animated_asset",
        "path": "asset/animated_target.json",
        "playback": {
            "scenario_clock_id": "scenario:walk",
            "motion_clock_id": "motion:walk",
            "playback_rate": 1.0,
            "playback_offset_s": 0.0,
        },
    }
    prepared = prepare_animated_asset(
        asset, Timeline(3, 1.0), target_name="Walker", base_dir=tmp_path
    )
    state = SensingPipelineState.from_config(
        SimpleNamespace(
            sensing={"enabled": True, "range_mode": "monostatic"},
            actors=SimpleNamespace(targets=(SimpleNamespace(name="Walker", asset=asset),)),
        ),
        frame_dt_s=0.5,
        gt_dt_s=0.5,
        first_step=0,
        source_generation_id=GENERATION_ID,
        execution_config=_execution(ANALYTIC_ORACLE_MODE, frame_dt_s=0.5),
    )
    scheduler = FixedIntervalScheduler(first_step=0, last_step=1, interval_length=1)
    frames = []
    for step in range(2):
        frame = _multi_target_frame(
            step,
            (("Walker", 2.0, 1.2),),
            scene_time_s=0.5 * step,
            sensing=_oracle_sensing(pair_index=0),
        )
        frame["target_positions_snapshot"] = np.array([prepared.centers_m[step]])
        frame["target_velocities_snapshot"] = np.array([prepared.root_velocities_mps[step]])
        frame["targets_metadata"][0]["animated_asset"] = {
            "catalog": prepared.catalog_metadata(),
            "state": prepared.sample_metadata(step),
        }
        state.process_frame(
            frame, step_idx=step, is_rt_step=True, schedule_event=scheduler.event_for_step(step)
        )
        frames.append(frame)
    first_catalog = frames[0]["sensing"]["target_catalog"]
    assert frames[1]["sensing"]["target_catalog"] == first_catalog
    assert first_catalog["targets"][0]["animated_asset"]["license"]["redistribution"] == "allowed"
    for step, frame in enumerate(frames):
        dynamic = frame["sensing"]["target_state"]["targets"][0]["animated_asset"]
        assert dynamic == prepared.sample_metadata(step)
        assert dynamic["sample"]["scenario_time_s"] == step * 0.5
    bad_state = deepcopy(frames[0]["sensing"]["target_state"])
    bad_state["targets"][0]["animated_asset"]["binding_id"] = "a" * 64
    assert any(
        "binding_id must match" in error
        for error in validate_target_state(first_catalog, bad_state)
    )

    for mutate in (
        lambda row: row.pop("animated_asset"),
        lambda row: row["animated_asset"]["sample"].update(sample_id=""),
        lambda row: row["animated_asset"]["sample"].pop("root"),
        lambda row: row["animated_asset"]["sample"].pop("pose_sample_id"),
        lambda row: row["animated_asset"]["sample"].update(motion_time_s=100.0),
        lambda row: row["animated_asset"]["sample"].pop("validity"),
        lambda row: row["animated_asset"]["sample"]["root"].update(resolved_affine_sha256="a" * 64),
    ):
        candidate = deepcopy(frames[0]["sensing"]["target_state"])
        mutate(candidate["targets"][0])
        assert validate_target_state(first_catalog, candidate)

    output = tmp_path / "frames"
    output.mkdir()
    writer = PackedMPCChunkWriter(output, generation_id=GENERATION_ID, compression=None)
    for step, frame in enumerate(frames):
        writer.append(
            standard_mpc_frame_from_pair_data(
                frame_index=step,
                timestamp_s=step * 0.5,
                tx_rx_pairs=np.array([[0, 0]], dtype=np.int32),
                tx_positions=np.array([[0.0, 0.0, 0.0]]),
                rx_positions=np.array([[0.0, 0.0, 0.0]]),
                vertices_by_pair=[np.zeros((1, 0, 3), dtype=np.float32)],
                interactions_by_pair=[np.zeros((1, 0), dtype=np.int32)],
                path_lengths_by_pair=[np.zeros((1,), dtype=np.int64)],
                target_positions_m=frame["target_positions_snapshot"],
                targets_metadata=tuple(frame["targets_metadata"]),
                sensing=frame["sensing"],
            )
        )
    chunk = writer.finalize_to_range_name()
    write_frame_manifest_atomic(
        output,
        manifest_from_chunks(
            generation_id=GENERATION_ID,
            frame_set_id="animated-asset-test",
            chunks=[chunk],
            compression={"algorithm": "none"},
            segmentation={"policy": "test"},
            provenance={"fixture": "authored-synthetic"},
            created_utc="2026-09-14T00:00:00+00:00",
        ),
    )
    with h5py.File(output / chunk.file, "r") as handle:
        assert int(handle.attrs["schema_version"]) == 2
    reader = PackedHDF5Reader(output)
    try:
        for step in range(2):
            loaded = reader.load_standard_frame(step)
            assert loaded.sensing["target_catalog"] == first_catalog
            assert loaded.sensing["target_state"] == frames[step]["sensing"]["target_state"]
            assert loaded.targets_metadata[0]["animated_asset"][
                "state"
            ] == prepared.sample_metadata(step)
    finally:
        reader.close()
