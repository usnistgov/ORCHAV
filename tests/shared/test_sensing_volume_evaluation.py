"""Live gates classify saved reports; they cannot change detector evidence."""

import json
import threading
from copy import deepcopy

import numpy as np
import pytest

from shared.sensing.evaluation import associate_detections_to_gt
from shared.sensing.volume.evaluation import VolumeEvaluation, load_evaluation
from shared.sensing.volume.reader import SensingVolume


def example(tmp_path):
    rows = {}
    trials = []
    for ix, reports in enumerate(
        [
            [
                {
                    "detection_id": "a",
                    "position_m": [10.5, 0, 0],
                    "range_m": 10.5,
                    "closing_velocity_m_s": -1.1,
                },
                {
                    "detection_id": "b",
                    "position_m": [0, 10, 0],
                    "range_m": 10,
                    "closing_velocity_m_s": -1,
                },
            ],
            [],
        ]
    ):
        name = str(ix)
        trials.append(
            dict(
                id=name,
                index_zyx=[0, 0, ix],
                heading=0,
                position_m=[10 + ix, 0, 0],
                velocity_m_s=[1, 0, 0],
                status="measured",
                hit=ix == 0,
                tp=int(ix == 0),
                fn=int(ix != 0),
                fp=1 if ix == 0 else 0,
                matched_error_m=0.5 if ix == 0 else None,
                radial_closing_velocity_m_s=-1,
                native=f"native_{ix}",
                configuration_sha256="declared",
                source_hashes={"f.h5": "declared"},
                reports=reports,
            )
        )
        rows[name] = reports
    data = dict(
        schema_version=1,
        kind="conditional_detection_outcomes",
        axes_m={"x": [10, 11], "y": [0, 1], "z": [0, 1]},
        world_frame=dict(
            id="test", origin="test", axes=["x", "y", "z"], handedness="right", length_unit="m"
        ),
        headings=[dict(name="+X", velocity_m_s=[1, 0, 0])],
        sensor_position_m=[0, 0, 0],
        position_gate_m=1,
        trials=trials,
    )
    path = tmp_path / "sensing_volume.json"
    path.write_text(json.dumps(data))
    return SensingVolume(path), rows


def test_gate_modes_disagree_for_bad_angle_and_counts_conserve_reports(tmp_path):
    reader, rows = example(tmp_path)
    before = deepcopy(rows)
    evaluation = VolumeEvaluation(reader, rows)
    for gate, expected_tp in [(0.49, 0), (0.5, 1), (1, 1)]:
        result = evaluation.evaluate(0, position_gate_m=gate)
        assert result["tp"] == expected_tp
        assert result["tp"] + result["fp"] == 2
        assert result["tp"] + result["fn"] == 2
        assert result["no_reports"] == 1
        assert np.isnan(result["outcomes"][1]).all()
    radial = evaluation.evaluate(
        0, mode="range_velocity", range_gate_m=0.01, velocity_gate_m_s=0.01
    )
    assert radial["tp"] == 1
    assert evaluation.report_ids[radial["selected_report"][0, 0, 0]] == "b"
    assert radial["nearest_error_m"][0, 0, 0] == 0.5
    assert rows == before


def test_radial_gates_match_shared_dashboard_assignment(tmp_path):
    reader, rows = example(tmp_path)
    rng = np.random.default_rng(20260923)
    reports = rows["0"]
    for _ in range(30):
        for report in reports:
            report["range_m"] = float(rng.uniform(9, 11))
            report["closing_velocity_m_s"] = float(rng.uniform(-2, 0))
        evaluation = VolumeEvaluation(reader, rows)
        expected = associate_detections_to_gt(
            [10],
            [-1],
            [r["range_m"] for r in reports],
            [r["closing_velocity_m_s"] for r in reports],
            range_gate=0.5,
            velocity_gate=0.3,
        )
        actual = evaluation.evaluate(
            0, mode="range_velocity", range_gate_m=0.5, velocity_gate_m_s=0.3
        )
        assert actual["tp"] == expected.true_positives
        assert actual["selected_report"][0, 0, 0] == (
            expected.gt_to_det[0] if expected.gt_to_det[0] is not None else -1
        )


def test_absent_radial_component_does_not_become_zero(tmp_path):
    reader, rows = example(tmp_path)
    for report in rows["0"]:
        report["closing_velocity_m_s"] = None
    evaluation = VolumeEvaluation(reader, rows)
    assert evaluation.evaluate(0)["tp"] == 1
    assert evaluation.evaluate(0, mode="range_velocity")["tp"] == 0


def test_cache_is_bound_to_manifest_and_corruption_rebuilds(tmp_path, monkeypatch):
    reader, _ = example(tmp_path)
    import shared.sensing.fusion.experiment as source

    monkeypatch.setattr(
        source,
        "load_sensor",
        lambda *a: pytest.fail("Embedded/cache reports need no native reread"),
    )
    first = load_evaluation(reader)
    assert first.evaluate(0)["tp"] == 1
    cache = tmp_path / "evaluation_reports.json"
    original = cache.read_bytes()
    payload = json.loads(original)
    payload["reports"]["0"][0]["position_m"] = [900, 900, 900]
    cache.write_text(json.dumps(payload))
    assert load_evaluation(reader).evaluate(0)["tp"] == 1
    assert cache.read_bytes() == original
    for malformed in ("[]", "null", "{truncated"):
        cache.write_text(malformed)
        assert load_evaluation(reader).evaluate(0)["tp"] == 1
        assert cache.read_bytes() == original
    reader.identity = "new manifest"
    load_evaluation(reader)
    assert json.loads(cache.read_text())["volume_sha256"] == "new manifest"


def test_cancellation_publishes_no_partial_cache(tmp_path):
    reader, _ = example(tmp_path)
    cancel = threading.Event()
    with pytest.raises(InterruptedError):
        load_evaluation(reader, cancel=cancel, progress=lambda done, total: cancel.set())
    assert not (tmp_path / "evaluation_reports.json").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_duplicate_ids_and_missing_trials_are_rejected(tmp_path):
    reader, rows = example(tmp_path)
    with pytest.raises(ValueError, match="inventory"):
        VolumeEvaluation(reader, {})
    rows["0"][1]["detection_id"] = "a"
    with pytest.raises(ValueError, match="unique"):
        VolumeEvaluation(reader, rows)


def test_cache_values_must_reproduce_recorded_baseline(tmp_path):
    reader, rows = example(tmp_path)
    rows["0"][0]["position_m"] = [900, 900, 900]
    with pytest.raises(ValueError, match="recorded matching outcome"):
        VolumeEvaluation(reader, rows)


def test_saved_match_identity_must_agree_even_when_counts_and_error_match(tmp_path):
    reader, rows = example(tmp_path)
    reader.cells[(0, 0, 0, 0)]["matched_detection_id"] = "b"
    with pytest.raises(ValueError, match="matched report ID"):
        VolumeEvaluation(reader, rows)
    reader.cells[(0, 0, 0, 0)]["matched_detection_id"] = "a"
    assert VolumeEvaluation(reader, rows).evaluate(0)["tp"] == 1


def test_read_only_result_still_allows_in_memory_evaluation(tmp_path, monkeypatch):
    from pathlib import Path

    reader, _ = example(tmp_path)
    original = Path.write_text

    def readonly(path, *args, **kwargs):
        if path.suffix == ".partial":
            raise PermissionError("read-only result bundle")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", readonly)
    result = load_evaluation(reader)
    assert result.evaluate(0)["tp"] == 1
    assert not (tmp_path / "evaluation_reports.json").exists()


def test_installed_scene_geometry_is_verified_without_importing_rt(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    import shared.sensing.volume.reader as module

    reader, _ = example(tmp_path)
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text("scene: {source: sionna, id: etoile}\n")
    package = tmp_path / "sionna"
    asset = package / "rt/scenes/etoile/meshes/arc.ply"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"original geometry")
    reader.metadata.update(
        scene_scenario="scenario.yaml",
        scene_configuration_sha256=hashlib.sha256(scenario.read_bytes()).hexdigest(),
        scene_asset_hashes={"meshes/arc.ply": hashlib.sha256(asset.read_bytes()).hexdigest()},
    )
    monkeypatch.setattr(
        module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[str(package)]),
    )
    assert reader.verify_scene() == scenario
    asset.write_bytes(b"changed geometry")
    with pytest.raises(ValueError, match="Installed scene asset changed"):
        reader.verify_scene()
