"""Authenticate included sensing figures using their producer's receipt contract.

Current PNG and authored-input bytes are verified directly. Source recordings
remain external evidence: recorded hashes identify them without asserting that
an unshipped recording or a historical plotter equals the current source tree.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path, PureWindowsPath

import pytest
import yaml

from scripts.docs.refresh_sensing_authoring_receipts import (
    METRIC_DEFAULT_BINDINGS,
    RECORDING_BINDINGS,
    _bound_metric_default_bytes,
    _bound_retention_profile_bytes,
    authenticated_authoring_inputs,
    bound_authoring_bytes,
)

ROOT = Path(__file__).resolve().parents[2]
FOLDERS = ("sensing_curriculum", "sensing_curation", "sensing_summary", "native_rcs")
SUMMARY_KINDS = {
    "standard_sensing_summary_from_saved_products",
    "standard_sensing_summary_from_native_products",
    "standard_native_sensing_summary",
}
DOPPLER_RECEIPTS = {
    "cube_radar_summary_motion",
    "cube_radar_summary_motion_coherent",
    "cube_radar_summary_motion_matched",
    "cube_radar_summary_motion_sequential",
    "marching_person_coherent_doppler",
    "marching_person_matched_doppler",
    "marching_person_sequential_doppler",
    "micro_motion_coherent_cube_doppler",
    "micro_motion_stft_cube_doppler",
}
RECEIPTS = sorted(
    path for folder in FOLDERS for path in (ROOT / "docs/assets" / folder).glob("*.json")
)
REBOUND_SUMMARIES = tuple(
    path.stem
    for path in RECEIPTS
    if path.parent.name == "sensing_summary"
    and "authored_configuration_binding" in json.loads(path.read_text())
)
FIVE_POSE_TOPOLOGIES = sorted(
    path
    for path in (ROOT / "docs/assets/scenarios").glob("scenarios_sensing_*_topology.json")
    if json.loads(path.read_text()).get("topology_options", {}).get("pose_samples") == 5
)


def _digest(path: Path) -> str:
    return hashlib.sha256(_authored_bytes(path)).hexdigest()


def _authored_bytes(path: Path) -> bytes:
    if path.suffix not in {".yaml", ".yml", ".py"}:
        return path.read_bytes()
    try:
        return bound_authoring_bytes(path, ROOT)
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc


def _document_digest(document: dict) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _authored_hashes(path: Path) -> set[str]:
    return {
        hashlib.sha256(source).hexdigest() for source in authenticated_authoring_inputs(path, ROOT)
    }


def test_storage_only_bindings_preserve_the_complete_non_recording_document() -> None:
    inventory = ROOT / "docs/assets/sensing_recording/authoring_bindings.json"
    for name, row in json.loads(inventory.read_text())["scenarios"].items():
        retained = _bound_retention_profile_bytes((ROOT / name).read_bytes(), name, ROOT)
        current_bytes = _bound_metric_default_bytes(retained, name, ROOT)
        assert hashlib.sha256(current_bytes).hexdigest() == row["current_authored_sha256"]
        current = yaml.safe_load(current_bytes)
        previous_bytes = bound_authoring_bytes(ROOT / name, ROOT)
        previous = yaml.safe_load(previous_bytes)
        assert hashlib.sha256(previous_bytes).hexdigest() == row["previous_authored_sha256"]
        assert "profile" in current["sensing"].pop("recording")
        assert "profile" not in previous["sensing"].pop("recording")
        assert current == previous


def _retention_binding_fixture(tmp_path: Path, previous: bytes, current: bytes):
    name = "scenarios/example/scenario.yaml"
    normalized = yaml.safe_load(previous)
    normalized["sensing"]["recording"].pop("profile")
    row = {
        "current_authored_sha256": hashlib.sha256(current).hexdigest(),
        "previous_authored_sha256": hashlib.sha256(previous).hexdigest(),
        "current_recording_block": current.decode(),
        "previous_recording_block": previous.decode(),
        "non_profile_sha256": _document_digest(normalized),
    }
    inventory = {
        "schema": "orchav.recording-authoring-bindings.v1",
        "scenarios": {},
        "retention_profiles": {name: row},
    }
    authored = tmp_path / name
    authored.parent.mkdir(parents=True)
    authored.write_bytes(current)
    binding_path = tmp_path / RECORDING_BINDINGS
    binding_path.parent.mkdir(parents=True)
    binding_path.write_text(json.dumps(inventory))
    return authored, binding_path, inventory, row


def test_retention_profile_chain_preserves_all_previously_authenticated_inputs():
    inventory = json.loads((ROOT / RECORDING_BINDINGS).read_text())
    assert inventory["retention_profiles"]
    for name, row in inventory["retention_profiles"].items():
        stages = authenticated_authoring_inputs(ROOT / name, ROOT)
        assert len(stages) == 4
        assert stages[0] == (ROOT / name).read_bytes()
        assert hashlib.sha256(stages[1]).hexdigest() == row["previous_authored_sha256"]
        assert (
            hashlib.sha256(stages[-1]).hexdigest()
            == inventory["scenarios"][name]["previous_authored_sha256"]
        )


def test_marching_hop_change_has_no_legacy_equivalence_binding():
    name = "scenarios/sensing/targets_motion/marching_person/sequential/scenario.yaml"
    recording = json.loads((ROOT / RECORDING_BINDINGS).read_text())
    metric = json.loads((ROOT / METRIC_DEFAULT_BINDINGS).read_text())
    assert name not in recording["scenarios"]
    assert name not in recording["retention_profiles"]
    assert name not in metric["scenarios"]
    current = (ROOT / name).read_bytes()
    assert yaml.safe_load(current)["sensing"]["processing"]["doppler"]["hop_samples"] == 64
    assert authenticated_authoring_inputs(ROOT / name, ROOT) == (current,)


def test_retention_profile_binding_authenticates_only_profile_removal_from_comparison(tmp_path):
    previous = b"sensing:\n  recording:\n    profile: reprocessing\n    detection_stages: false\n"
    current = previous.replace(b"reprocessing", b"maps")
    authored, _, _, _ = _retention_binding_fixture(tmp_path, previous, current)
    assert authenticated_authoring_inputs(authored, tmp_path) == (current, previous)


@pytest.mark.parametrize(
    ("before", "after"),
    [("maps", "reprocessing"), ("reprocessing", "detections"), ("reprocessing", "reprocessing")],
)
def test_retention_profile_binding_rejects_other_transitions(tmp_path, before, after):
    previous = f"sensing:\n  recording:\n    profile: {before}\n".encode()
    current = f"sensing:\n  recording:\n    profile: {after}\n".encode()
    authored, _, _, _ = _retention_binding_fixture(tmp_path, previous, current)
    with pytest.raises(ValueError, match="Unsupported retention profile transition"):
        bound_authoring_bytes(authored, tmp_path)


@pytest.mark.parametrize(
    "change",
    (
        "unbound_bytes",
        "previous_hash",
        "anchor",
        "empty_block",
        "duplicate_block",
        "other_setting",
        "forged_hashes",
    ),
)
def test_retention_profile_binding_rejects_mismatches(tmp_path, change):
    previous = b"sensing:\n  enabled: true\n  recording:\n    profile: reprocessing\n"
    current = previous.replace(b"reprocessing", b"maps")
    authored, path, inventory, row = _retention_binding_fixture(tmp_path, previous, current)
    if change == "unbound_bytes":
        authored.write_bytes(current + b"# changed\n")
    elif change == "previous_hash":
        row["previous_authored_sha256"] = "0" * 64
    elif change == "anchor":
        row["non_profile_sha256"] = "0" * 64
    elif change == "empty_block":
        row["current_recording_block"] = ""
    elif change == "duplicate_block":
        authored.write_bytes(current * 2)
        row["current_authored_sha256"] = hashlib.sha256(authored.read_bytes()).hexdigest()
    else:
        changed = current.replace(b"enabled: true", b"enabled: false")
        authored.write_bytes(changed)
        row["current_authored_sha256"] = hashlib.sha256(changed).hexdigest()
        row["current_recording_block"] = changed.decode()
        if change == "forged_hashes":
            changed_previous = previous.replace(b"enabled: true", b"enabled: false")
            row["previous_recording_block"] = changed_previous.decode()
            row["previous_authored_sha256"] = hashlib.sha256(changed_previous).hexdigest()
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError):
        bound_authoring_bytes(authored, tmp_path)


def _metric_binding_fixture(tmp_path: Path, previous: bytes, current: bytes):
    name = "scenarios/example/scenario.yaml"
    previous_document = yaml.safe_load(previous)
    rt = previous_document.setdefault("raytracing", {})
    rt.setdefault("export_path_metrics", False)
    row = {
        "current_authored_sha256": hashlib.sha256(current).hexdigest(),
        "previous_authored_sha256": hashlib.sha256(previous).hexdigest(),
        "current_block": current.decode(),
        "previous_block": previous.decode(),
        "resolved_document_sha256": _document_digest(previous_document),
    }
    inventory = {
        "schema": "orchav.metric-default-authoring-bindings.v1",
        "scenarios": {name: row},
    }
    authored = tmp_path / name
    authored.parent.mkdir(parents=True)
    authored.write_bytes(current)
    binding_path = tmp_path / METRIC_DEFAULT_BINDINGS
    binding_path.parent.mkdir(parents=True)
    binding_path.write_text(json.dumps(inventory))
    return authored, binding_path, inventory, row


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        (
            b"raytracing:\n  enabled: true\n  export_path_metrics: true\n",
            b"raytracing:\n  enabled: true\n",
        ),
        (
            b"raytracing:\n  enabled: true\n",
            b"raytracing:\n  enabled: true\n  export_path_metrics: false\n",
        ),
        (
            b"schema_version: 2\n",
            b"schema_version: 2\nraytracing:\n  export_path_metrics: false\n",
        ),
    ],
)
def test_metric_default_binding_reconstructs_only_equivalent_spellings(tmp_path, previous, current):
    authored, _, _, _ = _metric_binding_fixture(tmp_path, previous, current)
    assert bound_authoring_bytes(authored, tmp_path) == previous


@pytest.mark.parametrize(
    ("previous", "current", "message"),
    [
        (
            b"raytracing:\n  export_path_metrics: false\n",
            b"raytracing: {}\n",
            "Unsupported metric default spelling",
        ),
        (
            b"raytracing:\n  export_path_metrics: true\n",
            b"raytracing:\n  export_path_metrics: false\n",
            "Unsupported metric default spelling",
        ),
        (
            b"raytracing: {}\n",
            b"raytracing: {}\n",
            "Unsupported metric default spelling",
        ),
        (
            b"raytracing:\n  export_path_metrics: 'true'\n",
            b"raytracing: {}\n",
            "requires a boolean",
        ),
        (
            b"raytracing: {}\n",
            b"raytracing:\n  export_path_metrics: 0\n",
            "requires a boolean",
        ),
        (
            b"raytracing:\n  enabled: true\n  export_path_metrics: true\n",
            b"raytracing:\n  enabled: false\n",
            "Scientific configuration changed",
        ),
    ],
)
def test_metric_default_binding_rejects_different_effective_inputs(
    tmp_path, previous, current, message
):
    authored, _, _, _ = _metric_binding_fixture(tmp_path, previous, current)
    with pytest.raises(ValueError, match=message):
        bound_authoring_bytes(authored, tmp_path)


def test_metric_default_binding_rejects_scientific_edits_with_updated_byte_hashes(tmp_path):
    previous = b"raytracing:\n  carrier_frequency_hz: 28000000000\n  export_path_metrics: true\n"
    current = b"raytracing:\n  carrier_frequency_hz: 28000000000\n"
    authored, inventory_path, inventory, row = _metric_binding_fixture(tmp_path, previous, current)
    changed = current.replace(b"28000000000", b"30000000000")
    authored.write_bytes(changed)
    with pytest.raises(ValueError, match="Current metric-default input changed"):
        bound_authoring_bytes(authored, tmp_path)

    changed_previous = previous.replace(b"28000000000", b"30000000000")
    row.update(
        current_authored_sha256=hashlib.sha256(changed).hexdigest(),
        previous_authored_sha256=hashlib.sha256(changed_previous).hexdigest(),
        current_block=changed.decode(),
        previous_block=changed_previous.decode(),
    )
    inventory_path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="Scientific configuration changed"):
        bound_authoring_bytes(authored, tmp_path)


def test_metric_default_bindings_authenticate_every_declared_input():
    inventory = json.loads((ROOT / METRIC_DEFAULT_BINDINGS).read_text())
    assert inventory["scenarios"]
    for name, row in inventory["scenarios"].items():
        retained = _bound_retention_profile_bytes((ROOT / name).read_bytes(), name, ROOT)
        previous = _bound_metric_default_bytes(retained, name, ROOT)
        assert hashlib.sha256(previous).hexdigest() == row["previous_authored_sha256"]
        bound_authoring_bytes(ROOT / name, ROOT)


def test_storage_binding_rejects_scientific_edits_with_updated_byte_hash(
    tmp_path,
) -> None:
    name = "scenarios/sensing/cube_radar/stationary/scenario.yaml"
    inventory_name = "docs/assets/sensing_recording/authoring_bindings.json"
    inventory = json.loads((ROOT / inventory_name).read_text())
    inventory["scenarios"] = {name: inventory["scenarios"][name]}
    document = yaml.safe_load((ROOT / name).read_text())
    document["sensing"]["acquisition"]["bandwidth_hz"] *= 2
    authored = tmp_path / name
    authored.parent.mkdir(parents=True)
    authored.write_text(yaml.safe_dump(document, sort_keys=False))
    row = inventory["scenarios"][name]
    row["current_authored_sha256"] = hashlib.sha256(authored.read_bytes()).hexdigest()
    forged_previous = authored.read_bytes().replace(
        row["current_recording_block"].encode(),
        row["previous_recording_block"].encode(),
    )
    row["previous_authored_sha256"] = hashlib.sha256(forged_previous).hexdigest()
    destination = tmp_path / inventory_name
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="Scientific configuration changed"):
        bound_authoring_bytes(authored, tmp_path)


def _pointer(document: dict, path: str):
    assert path.startswith("/"), path
    value = document
    for token in path[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _validate_authored_configuration_binding(receipt: dict) -> None:
    binding = receipt["authored_configuration_binding"]
    assert binding["schema_version"] == 1 and binding["kind"] == "summary_only_rebind"
    assert binding["acquisition_repeated"] is False and binding["figure_rerendered"] is False
    _hash(binding["original_receipt_sha256"])
    original = binding["original_authored_configuration_fields"]
    updated = binding["updated_authored_binding_fields"]
    assert set(original) == set(updated) and len(updated) == len(set(updated))
    configurations = binding["configurations"]
    assert configurations
    by_path = {row["scenario"]: row for row in configurations}
    assert len(by_path) == len(configurations)
    for scenario, row in by_path.items():
        authored = ROOT / scenario
        assert row["current_authored_sha256"] == _digest(authored)
        document = yaml.safe_load(_authored_bytes(authored))
        assert row["excluded_top_level_keys"] == ["generator_summary"]
        assert row["current_generator_summary"] == document.get("generator_summary")
        document.pop("generator_summary", None)
        assert (
            _document_digest(document) == row["unchanged_non_summary_document_sha256"]
        ), "Scientific configuration changed"
    expected = {"/tutorial_scenario_sha256": receipt["tutorial_scenario"]}
    expected.update(
        {
            "/authored_scenarios/" + name.replace("~", "~0").replace("/", "~1"): name
            for name in receipt.get("authored_scenarios", {})
        }
    )
    assert set(updated) == set(expected)
    assert set(by_path) == set(expected.values())
    for field, scenario in expected.items():
        row = by_path[scenario]
        assert original[field] == row["previous_authored_sha256"]
        assert _pointer(receipt, field) == row["current_authored_sha256"]


def _validate_topology_binding(path: Path, receipt: dict) -> None:
    assert _digest(path.with_suffix(".png")) == receipt["output_sha256"]
    assert receipt["topology_options"]["pose_samples"] == 5
    # A locally available authored scene needs no external-path binding.
    if "configuration_binding" not in receipt:
        scenario = receipt["scenario"]
        assert receipt["source_files"][scenario] in _authored_hashes(ROOT / scenario)
        document = yaml.safe_load(_authored_bytes(ROOT / scenario))
        assert document["generator_summary"]["visualization"]["topology_pose_samples"] == 5
        _hashes(receipt["source_files"])
        assert all(
            not Path(name).is_absolute() and not PureWindowsPath(name).is_absolute()
            for name in receipt["source_files"]
        )
        return
    binding = receipt["configuration_binding"]
    assert binding["schema_version"] == 1
    _hash(binding["original_receipt_sha256"])
    rendered_hash = receipt["rendered_scenario_sha256"]
    assert binding["rendered_configuration_sha256"] == rendered_hash
    assert receipt["source_files"]["scenario.yaml"] == rendered_hash
    assert receipt["source_files"][receipt["rendered_scenario"]] == rendered_hash
    _hashes(receipt["source_files"])
    assert all(
        not Path(name).is_absolute() and not PureWindowsPath(name).is_absolute()
        for name in receipt["source_files"]
    )
    if binding["kind"] == "scripted_topology_input":
        assert receipt["scenario"] == receipt["rendered_scenario"]
        helper = binding["authored_preparation_source"]
        assert _digest(ROOT / helper) == binding["authored_preparation_source_sha256"]
        assert receipt["authored_preparation_source"] == helper
        assert receipt["source_files"][helper] == binding["authored_preparation_source_sha256"]
        assert binding["preparation_case"] == "human"
        _hash(binding["preparation_parity_proof_sha256"])
        assert "authored_scenario" not in receipt
        return
    assert binding["kind"] == "topology_authored_to_bound_input"
    assert receipt["scenario"] == receipt["authored_scenario"]
    scenario = receipt["authored_scenario"]
    authored_hash = _digest(ROOT / scenario)
    assert receipt["authored_scenario_sha256"] == authored_hash
    assert binding["authored_configuration_sha256"] == authored_hash
    assert receipt["source_files"][scenario] == authored_hash
    document = yaml.safe_load(_authored_bytes(ROOT / scenario))
    assert document["generator_summary"]["visualization"]["topology_pose_samples"] == 5
    adaptations = binding["adaptations"]
    assert {row["field"] for row in adaptations} == set(binding["excluded_fields"])
    for row in adaptations:
        field = row["field"]
        assert (
            re.fullmatch(r"/actors/targets/\d+/(asset/path|mobility/graph_path)|/scene/id", field)
            or (field == "/scene/source" and row["kind"] == "scene_path_resolution")
            or (
                field
                in {
                    "/generator_summary/create",
                    "/generator_summary/visualization/scene2d_mode",
                }
                and row["kind"] == "unused_summary_selection"
            )
        )
        parent_path, _, token = field.rpartition("/")
        parent = _pointer(document, parent_path)
        assert parent.get(token) == row["before"]
        parent.pop(token, None)
    assert _document_digest(document) == binding["unchanged_configuration_projection_sha256"]


def _hash(value: object) -> None:
    assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), value


def _hashes(values: dict) -> None:
    assert isinstance(values, dict) and values
    for name, value in values.items():
        assert name
        _hash(value)


def _producer(values: dict, expected: str) -> None:
    _hashes(values)
    assert expected in values, f"Missing producer identity: {expected}"
    # Receipt keys identify the historical producer. Its current implementation
    # can move without rewriting that recorded identity or its source hash.
    current_path = {
        "shared/sensing_viz.py": "shared/sensing/plotting/viz.py",
        "shared/sensing_fusion_figures.py": "shared/sensing/fusion/figures.py",
        "shared/sensing_postprocessor.py": "shared/sensing/postprocessor.py",
        "shared/sensing_doppler_summary.py": "shared/sensing/plotting/doppler_summary.py",
    }.get(expected, expected)
    assert (ROOT / current_path).is_file()


def _windows(windows: list[dict]) -> None:
    assert windows
    for window in windows:
        assert window["product_id"]
        _hash(window["processing_recipe_id"])
        times = window["sample_times_s"]
        assert times and all(a < b for a, b in zip(times, times[1:]))


def _comparison_sources(sources: list[dict], *, compact: bool = False) -> None:
    assert len(sources) == 2
    for source in sources:
        _recording_source(source, compact=compact)


def _recording_source(source: dict, *, compact: bool = False) -> None:
    _hashes(source["source_hashes"])
    assert source["calibration"]["processing"]
    if "windows" in source:
        _windows(source["windows"])
    elif not compact:
        assert source["window_count"] > 0
        assert source["first_window"]["product_id"]
        assert source["last_window"]["product_id"]
        assert source["processing_recipe_ids"]


def _validate_receipt(path: Path, receipt: dict) -> None:
    image = path.with_suffix(".png")
    assert receipt["image"] == image.name
    assert image.is_file()
    assert receipt["sha256"] == _digest(image), "Image identity mismatch"
    authored = ROOT / receipt["tutorial_scenario"]
    assert authored.is_file()
    authenticated = _authored_hashes(authored)
    assert receipt["tutorial_scenario_sha256"] in authenticated, "Authored input mismatch"
    for name, value in receipt.get("authored_scenarios", {}).items():
        assert value in _authored_hashes(ROOT / name)
    if "authored_experiment" in receipt:
        assert receipt["authored_experiment_sha256"] == _digest(
            ROOT / receipt["authored_experiment"]
        )
    if "authored_configuration_binding" in receipt:
        _validate_authored_configuration_binding(receipt)

    kind = receipt.get("kind")
    if kind in {
        "standard_sensing_tracking_summary_from_fusion",
        "saved_fusion_track_availability",
    }:
        _hash(receipt["fusion_results_sha256"])
        assert receipt["run_id"] and receipt["mode"]
        assert receipt["coordinate_frame"] == "world"
        assert receipt["source_unchanged"] is True
        if kind == "standard_sensing_tracking_summary_from_fusion":
            _producer(receipt["plotting_source_sha256"], "shared/sensing_viz.py")
        else:
            _hash(receipt["evaluation_sha256"])
            assert receipt["position_gate_m"] > 0
        _producer(receipt["plotting_source_sha256"], "shared/sensing_fusion_figures.py")
        assert "shared.sensing_fusion_figures" in receipt["reproduction_command"]
        return
    if kind in SUMMARY_KINDS:
        assert receipt["source_identity"]["generation_id"]
        assert receipt["source_identity"]["frame_set_id"]
        _hash(receipt["manifest_sha256"])
        _hashes(receipt["source_files_sha256"])
        assert receipt["source_files_sha256"]["frames_manifest.json"] == receipt["manifest_sha256"]
        assert receipt["result_count"] == len(receipt["results"]) > 0
        assert receipt["figures"] and receipt["outputs"]
        times = [row["reference_time_s"] for row in receipt["results"]]
        assert all(a < b for a, b in zip(times, times[1:]))
        for row in receipt["results"]:
            assert row["acquisition_id"]
            _hash(row["processing_recipe_id"])
        producer = receipt.get("plotting_source_sha256", receipt.get("source_module_sha256"))
        _producer(producer, "generator/figures/summary/sensing.py")
        return
    if kind == "plot_of_saved_native_products_no_reprocessing":
        assert receipt["license_spdx"] == "GPL-3.0-only"
        if path.stem.startswith("cube_radar_"):
            _producer(receipt["source_files"], "scenarios/sensing/cube_radar/figures.py")
            assert receipt["recordings"] and receipt["command"]
            for recording in receipt["recordings"].values():
                _hashes(recording["source_hashes"])
                assert "frames_manifest.json" in recording["source_hashes"]
                _hash(recording["execution_scenario_sha256"])
        else:
            _hashes(receipt["source_hashes"])
            _hash(receipt["source_hashes"]["scenario.yaml"])
            _hashes(receipt["source_module_hashes"])
            assert receipt["frame_set_id"] and receipt["generation_id"]
            assert receipt["processing_recipe"] and receipt["results"]
            _hash(receipt["plotter_sha256"])
            if "command" in receipt:
                assert "generate_sensing_curriculum_figures" in receipt["command"]
            else:
                _hash(receipt["historical_command_sha256"])
                _hash(receipt["source_receipt_sha256"])
                assert (
                    receipt["historical_command_producer"]
                    == "scripts/docs/generate_sensing_curriculum_figures.py"
                )
                assert "not a runnable reproduction command" in receipt["historical_command_scope"]
        return
    if kind == "visualizer_gui_capture":
        assert receipt["image_sha256"] == receipt["sha256"]
        assert receipt["renderer"] == "pygfx"
        assert receipt["pygfx_version"] and receipt["wgpu_version"]
        if path.parent.name == "native_rcs":
            _hashes(receipt["source_sha256"])
            _producer(
                receipt["visualizer_source_sha256"],
                "visualizer/src/scene/native_target_geometry.py",
            )
            assert receipt["native_target"] and receipt["rendered_point_count"] > 0
        else:
            _hash(receipt["source_sha256"])
            _producer(
                receipt["implementation_files_sha256"],
                "visualizer/src/sensing/fusion_window.py",
            )
            assert receipt["native_sources"] and receipt["source_unchanged"]
            assert "python -m visualizer" in receipt["capture_command"]
        _hash(receipt["capture_script_sha256"])
        return
    if kind == "standard_range_profile_from_analytical_response":
        assert (
            receipt["input_origin"]
            == "Exact single-path frequency response; no ray tracing or recorded acquisition"
        )
        assert receipt["model_recipe"]["model"] == "cfr_ifft"
        _hashes(receipt["input_files_sha256"])
        _producer(receipt["plotting_source_sha256"], "shared/sensing_viz.py")
        assert receipt["reported_detections"] == []
        assert "--summary-figures" in receipt["reproduction_command"]
        return
    if kind == "plot_of_saved_paired_outcomes":
        assert path.stem == "wall_volume"
        _hashes(receipt["source_hashes"])
        _hash(receipt["launcher_sha256"])
        _hash(receipt["plotter_sha256"])
        assert receipt["summary"]["manifest_sha256"]
        assert receipt["reproduction_commands"]
        return
    if kind == "standard_sensing_postprocessor_recipe_figures":
        assert receipt["source_identity"] and receipt["source_unchanged_after_processing"]
        _hashes(receipt["source_files_sha256"])
        _hashes(receipt["result_files_sha256"])
        _producer(receipt["code_sha256"], "shared/sensing_postprocessor.py")
        _hash(receipt["recipe_sha256"])
        assert receipt["recipe"] and receipt["result_identity"]
        return
    if kind == "standard_saved_doppler_history":
        assert receipt["axis"] == "velocity" and receipt["time_reference"] == "midpoint"
        assert receipt["common_power_reference"] > 0
        assert receipt["acquisition_repeated"] is False
        assert receipt["signal_reprocessing"] is False
        assert receipt["renderer"] == "shared.sensing.plotting.doppler_summary.plot_doppler_history"
        assert "-m shared.sensing.plotting.doppler_summary " in receipt["command"]
        _hash(receipt["plotter_sha256"])
        _hash(receipt["comparison_receipt_sha256"])
        source = receipt["recording"]
        _recording_source(source)
        assert source["frame_set_id"] and source["generation_id"]
        _hash(source["scenario_sha256"])
        assert receipt["spectrum_count"] == len(source["windows"]) > 0
        assert "comparison_receipt" not in receipt
        return
    assert kind is None and path.stem in DOPPLER_RECEIPTS, "Unrecognized receipt contract"
    assert receipt["axis"] == "velocity" and receipt["time_reference"] == "midpoint"
    assert receipt["common_power_reference"] > 0
    _hash(receipt["plotter_sha256"])
    if path.stem == "cube_radar_summary_motion":
        _comparison_sources(receipt["recordings"])
    elif path.stem.startswith("micro_motion_"):
        _producer(receipt["plotting_source_sha256"], "shared/sensing_doppler_summary.py")
        _comparison_sources(receipt["comparison_receipt"]["recordings"])
        assert receipt["acquisition_run_receipt"]["returncode"] == 0
    else:
        modules = ("shared.sensing_doppler_summary", "shared.sensing.plotting.doppler_summary")
        renderers = {f"{module}.plot_doppler_history" for module in modules}
        if receipt.get("layout") == "combined":
            renderers.add("shared.sensing.plotting.doppler_summary.plot_doppler_comparison")
        assert receipt["renderer"] in renderers
        assert any(f"-m {module} " in receipt["command"] for module in modules)
        _comparison_sources(receipt["comparison_sources"], compact=path.stem.startswith("cube_"))
        _hash(receipt["comparison_receipt_sha256"])
        if path.stem.startswith("cube_"):
            _windows(receipt["windows"])
            assert receipt["spectrum_count"] == len(receipt["windows"])


@pytest.mark.parametrize("path", RECEIPTS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_sensing_figure_receipt_authenticates_current_assets(path: Path) -> None:
    _validate_receipt(path, json.loads(path.read_text()))


def test_sensing_figure_receipts_cover_images() -> None:
    assert RECEIPTS
    for folder in FOLDERS:
        directory = ROOT / "docs/assets" / folder
        assert {p.stem for p in directory.glob("*.png")} == {
            p.stem for p in directory.glob("*.json")
        }


@pytest.mark.parametrize(
    "mode",
    [
        mode
        for mode in ("coherent", "sequential", "matched")
        if (ROOT / "docs/assets/sensing_summary" / f"marching_person_{mode}_doppler.json").is_file()
    ],
)
def test_marching_figures_preserve_derived_selection_and_source_acquisition(mode):
    path = ROOT / "docs/assets/sensing_summary" / f"marching_person_{mode}_doppler.json"
    receipt = json.loads(path.read_text())
    assert "authored_configuration_binding" not in receipt
    assert "summary_configuration_equivalence" not in receipt
    sources = receipt["comparison_sources"]
    lineage = receipt["source_lineage"]["sequential"]
    selection = lineage["output_selection"]
    assert lineage["kind"] == "derived_existing_complete_acquisitions"
    assert lineage["source_acquisition"]["cpi_hop_samples"] == 1
    assert selection["source_hop_samples"] == 1
    assert selection["hop_samples"] == 64
    assert selection["window_samples"] == 256
    assert selection["selected_count"] == len(sources[1]["windows"]) == 125
    assert selection["source_complete_count"] == 7937
    assert lineage["acquired_pulses"] == 8192
    assert selection["source_frame_set_id"] == lineage["source_frame_set_id"]
    assert selection["source_generation_id"] == lineage["source_generation_id"]
    assert lineage["output_frame_set_id"] == sources[1]["frame_set_id"]
    assert lineage["source_storage_indices"] == list(range(255, 8192, 64))
    assert lineage["source_bound_scene_indices"] == list(range(127, 8064, 64))
    _hash(lineage["verification_receipt_sha256"])
    _hash(lineage["source_snapshot_receipt_sha256"])
    assert "comparison_receipt" not in receipt  # No unshipped sibling-path reference.
    if mode == "matched":
        assert receipt["renderer"].endswith(".plot_doppler_comparison")
        assert receipt["matching_window_indices"] == list(range(0, 125, 4))
        assert receipt["spectrum_count"] == len(sources[0]["windows"]) == 32
        for first, index in zip(sources[0]["windows"], receipt["matching_window_indices"]):
            # The producer's midpoint comparison matches all acquired pulse epochs,
            # rather than selecting the nearest result timestamp.
            assert first["sample_times_s"] == pytest.approx(
                sources[1]["windows"][index]["sample_times_s"], abs=4e-15, rel=0
            )
    else:
        assert receipt["renderer"].endswith(".plot_doppler_history")
        assert receipt["matching_window_indices"] is None
        assert receipt["spectrum_count"] == (32 if mode == "coherent" else 125)


@pytest.mark.parametrize("change", ["source_hash", "window_recipe", "count", "reprocessed"])
def test_saved_doppler_history_rejects_corrupted_provenance(change):
    path = ROOT / "docs/assets/sensing_summary/marching_person_rigid_doppler.json"
    receipt = json.loads(path.read_text())
    _validate_receipt(path, receipt)
    assert receipt["spectrum_count"] == 6
    if change == "source_hash":
        receipt["recording"]["source_hashes"]["frames_manifest.json"] = "invalid"
    elif change == "window_recipe":
        receipt["recording"]["windows"][0]["processing_recipe_id"] = "invalid"
    elif change == "count":
        receipt["spectrum_count"] += 1
    else:
        receipt["signal_reprocessing"] = True
    with pytest.raises(AssertionError):
        _validate_receipt(path, receipt)


@pytest.mark.parametrize("field", ["sha256", "tutorial_scenario_sha256", "kind"])
def test_sensing_figure_receipt_rejects_corrupted_identity(field: str) -> None:
    path = ROOT / "docs/assets/sensing_curriculum/cube_radar_summary_stationary_range.json"
    receipt = copy.deepcopy(json.loads(path.read_text()))
    receipt[field] = "invalid" if field == "kind" else "0" * 64
    with pytest.raises(AssertionError):
        _validate_receipt(path, receipt)


@pytest.mark.parametrize("missing", ["frame_set_id", "manifest_binding", "producer"])
def test_sensing_summary_receipt_requires_origin_identity(missing: str) -> None:
    path = ROOT / "docs/assets/sensing_curriculum/cube_radar_summary_stationary_range.json"
    receipt = json.loads(path.read_text())
    if missing == "frame_set_id":
        receipt["source_identity"]["frame_set_id"] = ""
    elif missing == "manifest_binding":
        receipt["source_files_sha256"]["frames_manifest.json"] = "0" * 64
    else:
        del receipt["plotting_source_sha256"]["generator/figures/summary/sensing.py"]
    with pytest.raises(AssertionError):
        _validate_receipt(path, receipt)


@pytest.mark.parametrize("name", REBOUND_SUMMARIES)
def test_summary_only_rebinding_authenticates_current_and_secondary_inputs(
    name: str,
) -> None:
    path = ROOT / "docs/assets/sensing_summary" / f"{name}.json"
    _validate_authored_configuration_binding(json.loads(path.read_text()))


@pytest.fixture
def summary_rebind_receipt(tmp_path, monkeypatch):
    """An authenticated two-input fixture independent of regenerated figures."""
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    document = {
        "timeline": {"steps": 8192},
        "scene": {"id": "scene.xml"},
        "raytracing": {"carrier_frequency_hz": 28e9, "quality": {"custom": {"seed": 1}}},
        "sensing": {
            "acquisition": {"bandwidth_hz": 500e6},
            "processing": {"clutter": {"history_window": 4}, "doppler": {"hop_samples": 1}},
            "detection": {"cfar": {"threshold_scale": 5.0}},
            "tracking": {"association": {"range_gate_m": 1.0}},
        },
        "actors": {
            "targets": [
                {"mobility": {"traversal": {"speed_mps": 1.0}}},
                {"asset": {"path": "animation.json"}},
            ]
        },
    }
    scenarios = ("scenarios/first/scenario.yaml", "scenarios/second/scenario.yaml")
    receipt = {
        "tutorial_scenario": scenarios[0],
        "authored_scenarios": {},
        "authored_configuration_binding": {
            "schema_version": 1,
            "kind": "summary_only_rebind",
            "acquisition_repeated": False,
            "figure_rerendered": False,
            "original_receipt_sha256": hashlib.sha256(b"synthetic original receipt").hexdigest(),
            "original_authored_configuration_fields": {},
            "updated_authored_binding_fields": {},
            "configurations": [],
        },
    }
    binding = receipt["authored_configuration_binding"]
    for scenario in scenarios:
        previous = yaml.safe_dump({**document, "generator_summary": {"create": False}}).encode()
        current = yaml.safe_dump({**document, "generator_summary": {"create": True}}).encode()
        authored = tmp_path / scenario
        authored.parent.mkdir(parents=True)
        authored.write_bytes(current)
        old_hash, new_hash = (hashlib.sha256(value).hexdigest() for value in (previous, current))
        receipt["authored_scenarios"][scenario] = new_hash
        fields = ["/authored_scenarios/" + scenario.replace("/", "~1")]
        if scenario == scenarios[0]:
            receipt["tutorial_scenario_sha256"] = new_hash
            fields.append("/tutorial_scenario_sha256")
        for field in fields:
            binding["original_authored_configuration_fields"][field] = old_hash
            binding["updated_authored_binding_fields"][field] = new_hash
        binding["configurations"].append(
            {
                "scenario": scenario,
                "previous_authored_sha256": old_hash,
                "current_authored_sha256": new_hash,
                "excluded_top_level_keys": ["generator_summary"],
                "current_generator_summary": {"create": True},
                "unchanged_non_summary_document_sha256": _document_digest(document),
            }
        )
    _validate_authored_configuration_binding(receipt)
    return receipt


def test_summary_rebinding_accepts_only_summary_changes(summary_rebind_receipt):
    _validate_authored_configuration_binding(summary_rebind_receipt)


@pytest.mark.parametrize(
    "field,replacement",
    [
        (("timeline", "steps"), 2049),
        (("scene", "id"), "different_scene.xml"),
        (("raytracing", "carrier_frequency_hz"), 30e9),
        (("raytracing", "quality", "custom", "seed"), 7),
        (("sensing", "acquisition", "bandwidth_hz"), 100e6),
        (("sensing", "processing", "clutter", "history_window"), 8),
        (("sensing", "processing", "doppler", "hop_samples"), 64),
        (("sensing", "detection", "cfar", "threshold_scale"), 6.0),
        (("sensing", "tracking", "association", "range_gate_m"), 2.0),
        (("actors", "targets", 0, "mobility", "traversal", "speed_mps"), 4.0),
        (("actors", "targets", 1, "asset", "path"), "different_animation.json"),
    ],
)
def test_summary_rebinding_rejects_scientific_change_despite_updated_yaml_hash(
    tmp_path: Path, summary_rebind_receipt, field: tuple, replacement
) -> None:
    receipt = summary_rebind_receipt
    scenario = receipt["tutorial_scenario"]
    # A newly added scientific field is just as invalid as changing an existing
    # field in a summary rebind; fixture identity is checked before mutation.
    document = yaml.safe_load(_authored_bytes(ROOT / scenario))
    original_document = copy.deepcopy(document)
    parent = document
    for i, token in enumerate(field[:-1]):
        if isinstance(parent, list):
            while len(parent) <= token:
                parent.append({})
            parent = parent[token]
        else:
            parent = parent.setdefault(token, [] if isinstance(field[i + 1], int) else {})
    parent[field[-1]] = replacement
    assert document != original_document
    authored = tmp_path / scenario
    authored.write_text(yaml.safe_dump(document, sort_keys=False))
    # Updating byte hashes alone must not certify different scientific inputs.
    current_hash = _digest(authored)
    receipt["tutorial_scenario_sha256"] = current_hash
    receipt["authored_configuration_binding"]["configurations"][0][
        "current_authored_sha256"
    ] = current_hash
    with pytest.raises(AssertionError, match="Scientific configuration changed"):
        _validate_authored_configuration_binding(receipt)


def test_summary_rebinding_rejects_corrupted_secondary_authored_hash(
    summary_rebind_receipt,
) -> None:
    receipt = summary_rebind_receipt
    secondary = next(
        path for path in receipt["authored_scenarios"] if path != receipt["tutorial_scenario"]
    )
    receipt["authored_scenarios"][secondary] = "0" * 64
    with pytest.raises(AssertionError):
        _validate_authored_configuration_binding(receipt)


@pytest.mark.parametrize("path", FIVE_POSE_TOPOLOGIES, ids=lambda path: path.stem)
def test_five_pose_topology_keeps_bound_input_and_portable_authored_identity(
    path: Path,
) -> None:
    _validate_topology_binding(path, json.loads(path.read_text()))


def test_topology_binding_cannot_exclude_scientific_processing(tmp_path, monkeypatch) -> None:
    # The negative contract needs no optional rendered body or shipped figure.
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    scenario = "scenario.yaml"
    document = {
        "generator_summary": {"visualization": {"topology_pose_samples": 5}},
        "sensing": {"acquisition": {"bandwidth_hz": 500e6}},
    }
    (tmp_path / scenario).write_text(yaml.safe_dump(document))
    path = tmp_path / "topology.json"
    path.write_text("{}")
    path.with_suffix(".png").write_bytes(b"synthetic image identity fixture")
    authored_hash = _digest(tmp_path / scenario)
    receipt = {
        "scenario": scenario,
        "topology_options": {"pose_samples": 5},
        "output_sha256": _digest(path.with_suffix(".png")),
        "authored_scenario": scenario,
        "authored_scenario_sha256": authored_hash,
        "rendered_scenario": scenario,
        "rendered_scenario_sha256": authored_hash,
        "source_files": {scenario: authored_hash},
        "configuration_binding": {
            "schema_version": 1,
            "kind": "topology_authored_to_bound_input",
            "original_receipt_sha256": _digest(path),
            "rendered_configuration_sha256": authored_hash,
            "authored_configuration_sha256": authored_hash,
            "excluded_fields": [],
            "adaptations": [],
            "unchanged_configuration_projection_sha256": _document_digest(document),
        },
    }
    _validate_topology_binding(path, receipt)
    receipt["configuration_binding"]["excluded_fields"].append("/sensing/acquisition/bandwidth_hz")
    receipt["configuration_binding"]["adaptations"].append(
        {
            "field": "/sensing/acquisition/bandwidth_hz",
            "before": 500e6,
            "after": 100e6,
            "kind": "path_resolution",
        }
    )
    with pytest.raises(AssertionError):
        _validate_topology_binding(path, receipt)
