"""Real Scenario Builder-to-generator HDF5 v2 process-boundary gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shared.frames.packed_hdf5 import PackedHDF5Reader
from shared.frames.providers import Hdf5Provider
from shared.source_identity import loaded_source_identity
from visualizer.src.authoring.generation import GenerationJob, GenerationState
from visualizer.src.io.packed_frame_payload import (
    projection_to_visual_frame,
    visual_frame_read_request,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.optional_runtime
def test_builder_runs_loaded_generator_and_publishes_visualizable_v2(
    tmp_path: Path,
) -> None:
    """Exercise the real child, writer, manifest, reader, and MPC geometry."""

    source_yaml = PROJECT_ROOT / "scenarios" / "getting_started" / "hello_world" / "scenario.yaml"
    scenario_yaml = tmp_path / "scenario.yaml"
    scenario_yaml.write_text(source_yaml.read_text(encoding="utf-8"), encoding="utf-8")

    result = (
        GenerationJob(
            scenario_yaml,
            launched_revision=0,
            job_id="real-v2-boundary",
        )
        .start()
        .wait(timeout=180)
    )

    diagnostics = "\n".join((*result.stdout_log, *result.stderr_log))
    assert (
        result.state is GenerationState.SUCCEEDED
    ), f"{result.error_message or 'generation failed'}\n{diagnostics}"

    frames_dir = tmp_path / "frames"
    assert (frames_dir / "frames_manifest.json").is_file()
    assert not (frames_dir / "frames_index.json").exists()
    assert not (frames_dir / "run_manifest.json").exists()

    reader = PackedHDF5Reader(frames_dir)
    try:
        assert reader.manifest.provenance["git_sha"] == loaded_source_identity("generator").git_sha
        reader.validate_all_chunks()
        assert reader.frame_ids == [0]
        frame = reader.load_standard_frame(0)
    finally:
        reader.close()

    assert frame.tx_positions.shape == (1, 3)
    assert np.all(np.isfinite(frame.tx_positions))
    assert frame.rx_positions.shape == (1, 3)
    assert np.all(np.isfinite(frame.rx_positions))
    assert frame.tx_rx_pairs.shape == (1, 2)
    assert int(frame.pair_path_offsets[-1]) > 0
    assert frame.bounce_xyz_m.ndim == 2
    assert frame.bounce_xyz_m.shape[1] == 3

    provider = Hdf5Provider(tmp_path)
    try:
        projection = provider.load_frame_projection(0, visual_frame_read_request())
        visual_frame = projection_to_visual_frame(projection)
    finally:
        provider.close()
    assert visual_frame["num_tx"] == 1
    assert visual_frame["num_rx"] == 1
    path_starts = visual_frame["canonical_data"].path_start_indices
    assert path_starts is not None
    assert path_starts.size > 0
