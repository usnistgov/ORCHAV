"""Independent wall geometry, paired comparisons and native acquisition bounds."""

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from generator.core.configuration.loader import resolve_sensing_document
from scenarios.sensing.applications.wall_detection_volume.generate import (
    CUBE_HALF,
    DIRECTORY,
    SENSOR,
    WALL_MAX,
    WALL_MIN,
    excluded,
    geometry,
    grid_axes,
    intersects_wall,
)
from shared.scenarios import validate_scenario_data


def test_dense_grid_preserves_wall_geometry_and_independent_acquisition_count():
    axes = grid_axes("dense")
    assert [len(axes[n]) for n in "xyz"] == [25, 25, 21]
    for values in axes.values():
        np.testing.assert_allclose(np.diff(values), 0.25)
    positions = [[x, y, z] for z in axes["z"] for y in axes["y"] for x in axes["x"]]
    assert sum(not excluded(p) for p in positions) == 12213
    assert [-2, 0, 2] in positions and [3, 0, 2] in positions and [3, 0, 5.5] in positions
    standard = grid_axes("standard")
    for name, start, stop in (("x", -3, 5), ("y", -4, 4), ("z", 0.5, 6.5)):
        np.testing.assert_array_equal(standard[name], np.arange(start, stop + 0.25, 0.5))
    inspection = grid_axes("inspection")
    assert [len(inspection[n]) for n in "xyz"] == [13, 13, 11]
    assert (
        sum(
            not excluded([x, y, z])
            for z in inspection["z"]
            for y in inspection["y"]
            for x in inspection["x"]
        )
        == 1787
    )


@pytest.mark.parametrize(
    "position,blocked",
    [
        ([-2, 0, 2], False),
        ([3, 0, 2], True),
        ([3, 4, 2], False),
        ([3, 0, 6], False),
        ([3, 3, 2], True),
    ],
)
def test_direct_path_geometry_is_independent_of_detector(position, blocked):
    assert intersects_wall(position) is blocked
    assert geometry(position, True)["direct_path_blocked"] is blocked
    assert geometry(position, False)["blocked_target_probes"] == 0


def test_cube_partial_visibility_and_solid_exclusions():
    assert geometry([3, 0, 2], True)["blocked_target_probes"] == 9
    assert 0 < geometry([3, 3, 2], True)["blocked_target_probes"] < 9
    assert excluded([0, 0, 2])
    assert not excluded([-0.5, 0, 2])
    assert not excluded([0, 3, 2])
    assert not excluded([0, 0, 4.5])


def test_authored_wall_mesh_and_acquisition_match_geometry():
    lines = (DIRECTORY / "scene/wall.ply").read_text().splitlines()
    start = lines.index("end_header") + 1
    vertices = np.array([[float(v) for v in row.split()] for row in lines[start : start + 8]])
    np.testing.assert_allclose(vertices.min(axis=0), WALL_MIN)
    np.testing.assert_allclose(vertices.max(axis=0), WALL_MAX)
    cfg = resolve_sensing_document(yaml.safe_load((DIRECTORY / "scenario.yaml").read_text()))
    validate_scenario_data(cfg)
    assert cfg["actors"]["targets"][0]["asset"]["scale"] == 2 * CUBE_HALF
    assert cfg["actors"]["rx"][0]["mobility"]["position_m"] == SENSOR.tolist()
    assert cfg["raytracing"]["antenna"]["rx"]["num_cols"] == 8
    assert cfg["sensing"]["coherent_clutter_mode"] == "windowed_mean"


def test_incomplete_native_cpi_cannot_be_mistaken_for_a_miss(tmp_path, monkeypatch):
    from scenarios.sensing.volume_tools import run_trials

    monkeypatch.setitem(
        sys.modules,
        "generator.__main__",
        SimpleNamespace(
            _run_from_yaml=lambda *a, **k: pytest.fail(
                "Existing incomplete data must not be overwritten"
            )
        ),
    )
    native = tmp_path / "native"
    (native / "frames").mkdir(parents=True)
    (native / "scenario.yaml").write_text("{}")
    (native / "frames/frames_manifest.json").write_text(json.dumps({"frame_ids": list(range(63))}))
    (tmp_path / "volume_recipe.json").write_text(
        json.dumps(
            {
                "cpi_samples": 64,
                "sensor_id": "Radar",
                "trials": [
                    {
                        "id": "a",
                        "native": "native",
                        "status": "pending",
                        "configuration_sha256": hashlib.sha256(b"{}").hexdigest(),
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="Incomplete native acquisition"):
        run_trials(tmp_path, 0, 1, None)
    assert not (tmp_path / "outcome.json").exists()


def test_paired_comparison_separates_no_reports_from_localization_misses():
    from scenarios.sensing.applications.wall_detection_volume.report import counts

    result = counts(
        [
            {"status": "measured", "hit": False, "tp": 0, "fp": 0, "fn": 1},
            {"status": "measured", "hit": False, "tp": 0, "fp": 1, "fn": 1},
            {"status": "measured", "hit": True, "tp": 1, "fp": 1, "fn": 0, "matched_error_m": 0.2},
            {"status": "pending"},
        ]
    )
    assert result["measured"] == 3 and result["tp"] == 1 and result["fn"] == 2
    assert result["no_reports"] == 1 and result["reports_but_no_spatial_match"] == 1
    assert result["fp"] == 2
