"""Saved-report comparisons preserve denominators, identities and empty states."""

import json

import pytest

from shared.sensing.volume.comparison import (
    compare_volumes,
    export_comparison,
    load_definition,
    save_definition,
    source_definition,
)
from tests.shared.test_sensing_volume_inspection import write_fixture


def definition(tmp_path, count=2):
    sources = [
        source_definition(write_fixture(tmp_path / f"source{i}").path, label=f"Option {i}")
        for i in range(count)
    ]
    return dict(schema_version=1, sources=sources, support="common_measured", position_gate_m=1)


def mutate_source(source, update):
    path = source["path"]
    from pathlib import Path

    data = json.loads(Path(path).read_text())
    update(data)
    Path(path).write_text(json.dumps(data))
    return source_definition(path, label=source["label"], heading=source["heading"])


@pytest.mark.parametrize("count", [1, 3])
def test_counts_errors_and_gate_change_are_exact_without_changing_reports(tmp_path, count):
    recipe = definition(tmp_path, count)
    result = compare_volumes(recipe)
    for condition in result["conditions"]:
        s = condition["summary"]
        assert [
            s[k]
            for k in (
                "denominator",
                "matched_locations",
                "extra_reports",
                "empty_outputs",
                "out_of_gate_outputs",
            )
        ] == [4, 2, 3, 1, 1]
        assert s["matched_rmse_m"] == 0
        assert s["unknown_or_excluded_locations"] == 20
        assert condition["matched_errors_m"] == [0, 0]
    wide = compare_volumes(recipe | {"position_gate_m": 100})
    for before, after in zip(result["conditions"], wide["conditions"], strict=True):
        assert before["report_snapshot_sha256"] == after["report_snapshot_sha256"]
        assert after["summary"]["matched_locations"] == 3
        assert after["summary"]["extra_reports"] == 2


def test_intersection_has_explicit_denominator_and_per_source_is_separate(tmp_path):
    recipe = definition(tmp_path)

    def remove_trial(data):
        data["trials"] = data["trials"][1:]

    recipe["sources"][1] = mutate_source(recipe["sources"][1], remove_trial)
    common = compare_volumes(recipe)["conditions"]
    assert [c["summary"]["denominator"] for c in common] == [3, 3]
    assert common[0]["summary"]["measured_outside_support"] == 1
    separate = compare_volumes(recipe | {"support": "per_source"})["conditions"]
    assert [c["summary"]["denominator"] for c in separate] == [4, 3]


def test_empty_support_and_missing_position_are_not_fabricated_matches(tmp_path):
    recipe = definition(tmp_path, 1)

    def remove_positions(data):
        for trial in data["trials"]:
            if trial["status"] != "measured":
                continue
            for report in trial["reports"]:
                report["position_m"] = None
            trial.update(tp=0, fn=1, hit=False, fp=len(trial["reports"]), matched_error_m=None)

    recipe["sources"][0] = mutate_source(recipe["sources"][0], remove_positions)
    s = compare_volumes(recipe)["conditions"][0]["summary"]
    assert s["reports_without_position"] == 3 and s["matched_rmse_m"] is None

    def remove_measured(data):
        data["trials"] = [t for t in data["trials"] if t["status"] != "measured"]

    recipe["sources"][0] = mutate_source(recipe["sources"][0], remove_measured)
    s = compare_volumes(recipe)["conditions"][0]["summary"]
    assert s["denominator"] == 0 and s["matched_fraction"] is None


@pytest.mark.parametrize("gate", [0, -1, float("nan"), True])
def test_invalid_gates_rejected(tmp_path, gate):
    with pytest.raises(ValueError, match="gate"):
        compare_volumes(definition(tmp_path) | {"position_gate_m": gate})


@pytest.mark.parametrize("existing_empty_directory", [False, True])
def test_saved_definition_exports_and_changed_identity(tmp_path, existing_empty_directory):
    recipe = definition(tmp_path)
    path = tmp_path / "portable" / "recipe.json"
    save_definition(path, recipe)
    assert load_definition(path) == recipe
    assert json.loads(path.read_text())["sources"][0]["path"].startswith("../")
    result = compare_volumes(load_definition(path))
    if existing_empty_directory:
        (tmp_path / "export").mkdir()
    export_comparison(result, tmp_path / "export")
    assert json.loads((tmp_path / "export/comparison.json").read_text()) == result
    assert load_definition(tmp_path / "export/definition.json") == recipe
    for name in ("summary.csv", "comparison.png", "comparison.pdf", "comparison.svg"):
        assert (tmp_path / "export" / name).stat().st_size > 100
    recipe["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity changed"):
        compare_volumes(recipe)


@pytest.mark.parametrize("existing_name", ["comparison.json", "unrelated.txt"])
def test_export_refuses_nonempty_directory_without_changing_contents(tmp_path, existing_name):
    result = compare_volumes(definition(tmp_path))
    output = tmp_path / "export"
    output.mkdir()
    (output / existing_name).write_bytes(b"existing user results\n")
    (output / "nested").mkdir()
    (output / "nested" / "recording.bin").write_bytes(b"\x00\x01\x02")
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}

    with pytest.raises(FileExistsError, match="choose a new or empty directory"):
        export_comparison(result, output)

    after = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert after == before


def test_different_world_rejected(tmp_path):
    recipe = definition(tmp_path)
    recipe["sources"][1] = mutate_source(
        recipe["sources"][1], lambda d: d["world_frame"].update(id="other")
    )
    with pytest.raises(ValueError, match="world frame"):
        compare_volumes(recipe)


def test_different_selected_motion_rejected(tmp_path):
    recipe = definition(tmp_path)

    def change_motion(data):
        data["headings"][0]["velocity_m_s"] = [-1, 0, 0]
        for trial in data["trials"]:
            trial["velocity_m_s"] = [-1, 0, 0]
            if trial["status"] == "measured":
                trial["radial_closing_velocity_m_s"] *= -1

    recipe["sources"][1] = mutate_source(recipe["sources"][1], change_motion)
    with pytest.raises(ValueError, match="target velocity"):
        compare_volumes(recipe)


def test_alternate_grids_require_explicit_per_source_support(tmp_path):
    recipe = definition(tmp_path)
    recipe["sources"][1] = mutate_source(
        recipe["sources"][1], lambda d: d["axes_m"].update(y=[20, 23, 26])
    )
    with pytest.raises(ValueError, match="identical grid"):
        compare_volumes(recipe)
    result = compare_volumes(recipe | {"support": "per_source"})
    assert [c["summary"]["denominator"] for c in result["conditions"]] == [4, 4]
    assert [c["summary"]["authored_grid_locations"] for c in result["conditions"]] == [24, 36]


def test_ambiguous_or_absent_heading_and_duplicate_labels_rejected(tmp_path):
    recipe = definition(tmp_path)
    recipe["sources"][1]["label"] = recipe["sources"][0]["label"]
    with pytest.raises(ValueError, match="unique"):
        compare_volumes(recipe)
    recipe["sources"][1]["label"] = "Second"
    recipe["sources"][1]["heading"] = "Not recorded"
    with pytest.raises(ValueError, match="exactly one"):
        compare_volumes(recipe)


def test_cpu_import_does_not_load_rf_or_gui(tmp_path):
    import subprocess
    import sys

    recipe = definition(tmp_path, 1)
    path = tmp_path / "recipe.json"
    save_definition(path, recipe)
    program = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sionna','mitsuba','drjit','torch','PySide6','visualizer','h5py','yaml'}:
            raise RuntimeError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, Block())
from shared.sensing.volume.comparison import load_definition, compare_volumes
assert compare_volumes(load_definition(sys.argv[1]))['conditions'][0]['summary']['matched_locations'] == 2
"""
    subprocess.run([sys.executable, "-c", program, str(path)], check=True)
