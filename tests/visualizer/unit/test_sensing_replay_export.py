"""Deterministic renderer-neutral sensing replay export tests."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from PIL import Image

from shared.extensions.sensing import canonical_sensing_hash
from shared.sensing.replay import (
    FrozenMap,
    ReplayCredibilityPreflight,
    ReplayEvent,
    ReplayEventKey,
    ReplayEvidenceReference,
    ReplayPairPose,
    ReplayPreflightCheck,
    ReplayProductReference,
    ReplayPulse,
    ReplayRecipeSnapshot,
    ReplaySceneBinding,
    compare_replay_events,
    freeze_replay_json,
)
from tests.shared.test_sensing_v4_hdf5 import _typed_sensing
from tests.visualizer.unit.test_sensing_stage_inspector import _rematerialize_products
from visualizer.src.sensing import replay_export as replay_export_module
from visualizer.src.sensing.replay_export import (
    ReplayExportBlockedError,
    ReplayExportError,
    export_replay_session,
)
from visualizer.src.sensing.stage_inspector import (
    CellSelection,
    DetectionStageBundle,
    InspectionDisposition,
    build_detection_stage_bundle,
)

CLOCK_ID = "5d6ac341-f653-4b6d-a856-f4c4efdd86fa"
ACQUISITION_ID = "34c16263-7c2f-4d32-b23f-53c0ebf082f2"
PAIR_ID = canonical_sensing_hash({"rx": "replay-rx", "tx": "replay-tx"})
ROOT_SOURCE_PRODUCT_ID = canonical_sensing_hash({"source": "fixture-root"})


def _uuid(label: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"orchav-replay-export-test:{label}"))


def _hash(label: str) -> str:
    return canonical_sensing_hash({"label": label})


def _frozen(payload: dict) -> FrozenMap:
    frozen = freeze_replay_json(payload, path="test")
    assert isinstance(frozen, FrozenMap)
    return frozen


def _pulse(sample_index: int, sample_time_s: float) -> ReplayPulse:
    source_reference_id = _hash(f"pulse:{sample_index}:{sample_time_s}")
    return ReplayPulse(
        sample_index=sample_index,
        sample_time_s=sample_time_s,
        valid=True,
        phase_coherent=True,
        source_generation_id=CLOCK_ID,
        source_scene_frame_id=f"scene-{sample_index}",
        sample_offset=sample_index,
        source_frame_set_id="fixture-frame-set",
        solve_id=f"solve-{sample_index // 2}",
        source_reference_id=source_reference_id,
        record=_frozen(
            {
                "sample_index": sample_index,
                "sample_time_s": sample_time_s,
                "source_reference_id": source_reference_id,
            }
        ),
    )


def _product(
    *,
    layer: str,
    kind: str,
    data_key: str,
    recipe_threshold: float,
) -> ReplayProductReference:
    return ReplayProductReference(
        product_id=_hash(f"{layer}:{kind}:{data_key}:{recipe_threshold}"),
        data_key=data_key,
        kind=kind,
        variant="primary",
        representation="power_linear",
        unit="1",
        scale_semantics="linear_power",
        axes=(_frozen({"name": "range", "size": 8, "unit": "m"}),),
        lineage=_frozen(
            {
                "acquisition_id": ACQUISITION_ID,
                "source_product_ids": [],
            }
        ),
        time_span=_frozen({"end_time_s": 0.003, "start_time_s": 0.0}),
        evidence_layer=layer,  # type: ignore[arg-type]
    )


def _evidence(
    *,
    layer: str,
    kind: str,
    identity: str,
    ordinal: int,
    recipe_threshold: float,
) -> ReplayEvidenceReference:
    payload_hash = _hash(f"{layer}:{kind}:{identity}:{ordinal}:{recipe_threshold}")
    return ReplayEvidenceReference(
        reference_id=_hash(f"reference:{payload_hash}"),
        kind=kind,  # type: ignore[arg-type]
        layer=layer,  # type: ignore[arg-type]
        identity=identity,
        ordinal=ordinal,
        payload_hash=payload_hash,
    )


def _preflight(disposition: str = "ready") -> ReplayCredibilityPreflight:
    status = "fail" if disposition == "blocked" else "pass"
    return ReplayCredibilityPreflight(
        disposition=disposition,  # type: ignore[arg-type]
        checks=(
            ReplayPreflightCheck(
                check_id="fixture_check",
                status=status,
                message="Deterministic export fixture preflight.",
                action="No action required." if status == "pass" else "Inspect fixture.",
                metrics=_frozen({"fixture": True}),
            ),
        ),
    )


def _event(
    *,
    run: str = "run-a",
    recipe_threshold: float = 3.0,
    disposition: str = "ready",
) -> ReplayEvent:
    processing_run_id = _uuid(run)
    recipe_parameters = {"cfar_threshold": recipe_threshold, "window": "hann"}
    recipe_id = canonical_sensing_hash(recipe_parameters)
    pulses = (_pulse(1, 0.001), _pulse(0, 0.0), _pulse(2, 0.002))
    products = (
        _product(
            layer="truth",
            kind="target_truth",
            data_key="truth_by_target",
            recipe_threshold=recipe_threshold,
        ),
        _product(
            layer="estimate",
            kind="range_profile",
            data_key="z_range_profile",
            recipe_threshold=recipe_threshold,
        ),
        _product(
            layer="oracle",
            kind="oracle_micro_doppler",
            data_key="oracle_micro_doppler",
            recipe_threshold=recipe_threshold,
        ),
        _product(
            layer="estimate",
            kind="range_doppler",
            data_key="a_range_doppler",
            recipe_threshold=recipe_threshold,
        ),
    )
    evidence = (
        _evidence(
            layer="truth",
            kind="truth",
            identity="walker-truth",
            ordinal=0,
            recipe_threshold=0.0,
        ),
        _evidence(
            layer="oracle",
            kind="detection",
            identity="oracle-detection",
            ordinal=0,
            recipe_threshold=recipe_threshold,
        ),
        _evidence(
            layer="estimate",
            kind="track",
            identity="track-2",
            ordinal=1,
            recipe_threshold=recipe_threshold,
        ),
        _evidence(
            layer="estimate",
            kind="detection",
            identity="detection-1",
            ordinal=0,
            recipe_threshold=recipe_threshold,
        ),
    )
    return ReplayEvent(
        key=ReplayEventKey(
            clock_id=CLOCK_ID,
            acquisition_id=ACQUISITION_ID,
            processing_run_id=processing_run_id,
            pair_id=PAIR_ID,
            reference_time_s=0.001,
        ),
        scene_binding=ReplaySceneBinding(
            storage_frame_id="storage-result",
            bound_scene_frame_id="scene-1",
            current_scene_frame_id="scene-1",
            bound_scene_time_s=0.001,
            current_scene_time_s=0.001,
            current_scene_result_offset_s=0.0,
            stale_age_s=0.0,
            pinned=False,
            state="fresh",
            fresh=True,
            stale=False,
        ),
        pair_pose=ReplayPairPose(
            pair_id=PAIR_ID,
            pair_index=0,
            frame_pair_index=0,
            tx_sensor_id="replay-tx",
            rx_sensor_id="replay-rx",
            tx_device_index=0,
            rx_device_index=0,
            tx_position_m=(1.0, 2.0, 3.0),
            tx_orientation_yaw_pitch_roll_rad=(0.1, -0.2, 0.3),
            rx_position_m=(4.0, 5.0, 6.0),
            rx_orientation_yaw_pitch_roll_rad=(0.4, 0.2, -0.1),
        ),
        pulses=pulses,
        source_samples_id=_hash("shared-source-samples"),
        recipe=ReplayRecipeSnapshot(
            processing_run_id=processing_run_id,
            processing_recipe_id=recipe_id,
            canonical_hash=recipe_id,
            parameters=_frozen(recipe_parameters),
            processing_plan=_frozen({"strategy": "coherent_cpi_fft"}),
        ),
        products=products,
        root_source_product_ids=(ROOT_SOURCE_PRODUCT_ID,),
        evidence=evidence,
        preflight=_preflight(disposition),
    )


def _stage_bundle_for_event(
    event: ReplayEvent,
) -> tuple[DetectionStageBundle, CellSelection]:
    """Retarget the exact Phase-4 fixture to one canonical replay occurrence."""

    sensing = _typed_sensing()
    temporal = sensing["temporal"]
    assert isinstance(temporal, dict)
    clock = temporal["clock"]
    acquisition = temporal["acquisition"]
    result = temporal["result"]
    assert isinstance(clock, dict)
    assert isinstance(acquisition, dict)
    assert isinstance(result, dict)
    clock["clock_id"] = event.key.clock_id
    acquisition["acquisition_id"] = event.key.acquisition_id
    acquisition["pair_id"] = event.key.pair_id
    acquisition["tx_sensor_id"] = event.pair_pose.tx_sensor_id
    acquisition["rx_sensor_id"] = event.pair_pose.rx_sensor_id
    samples = acquisition["samples"]
    assert isinstance(samples, list)
    pulses = sorted(event.pulses, key=lambda pulse: pulse.sample_index)
    assert len(samples) == len(pulses)
    for sample, pulse in zip(samples, pulses, strict=True):
        sample["sample_index"] = pulse.sample_index
        sample["sample_time_s"] = pulse.sample_time_s
        sample["valid"] = pulse.valid
        sample["phase_coherent"] = pulse.phase_coherent
        source = sample["source"]
        source["source_generation_id"] = pulse.source_generation_id
        source["scene_frame_id"] = pulse.source_scene_frame_id
        source["sample_offset"] = pulse.sample_offset
    acquisition["start_time_s"] = pulses[0].sample_time_s
    acquisition["end_time_s"] = pulses[-1].sample_time_s
    acquisition["default_reference_time_s"] = event.key.reference_time_s
    result["processing_run_id"] = event.key.processing_run_id
    result["processing_recipe_id"] = event.recipe.processing_recipe_id
    result["reference_time_s"] = event.key.reference_time_s
    _rematerialize_products(sensing)
    sensing["detections"] = [
        {
            "cluster_id": 0,
            "doppler_bin": 1,
            "range_bin": 2,
            "range_m": 2.0,
            "doppler_hz": 0.0,
            "snr_db": 3.0102999566,
        }
    ]
    bundle = build_detection_stage_bundle(sensing)
    assert bundle.event is not None
    selection = CellSelection(
        pair_id=bundle.event.pair_id,
        doppler_bin=1,
        range_bin=2,
        cluster_id=0,
    )
    return bundle, selection


def _directory_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir(), key=lambda candidate: candidate.name)
    }


def test_replay_export_is_byte_deterministic_and_stably_ordered(tmp_path: Path) -> None:
    event = _event()
    first = export_replay_session(event, tmp_path / "first")
    second = export_replay_session(event, tmp_path / "second")

    first_bytes = _directory_bytes(first.output_directory)
    second_bytes = _directory_bytes(second.output_directory)
    assert first_bytes == second_bytes
    assert first.artifacts == second.artifacts
    assert set(first_bytes) == {
        "evidence_estimate.csv",
        "evidence_oracle.csv",
        "evidence_truth.csv",
        "export_manifest.json",
        "products_estimate.csv",
        "products_oracle.csv",
        "products_truth.csv",
        "pulse_timeline.csv",
        "replay_manifest.json",
        "replay_summary.png",
    }
    assert first_bytes["replay_manifest.json"] == (event.to_json() + "\n").encode("utf-8")

    pulse_rows = list(
        csv.DictReader(io.StringIO(first_bytes["pulse_timeline.csv"].decode("utf-8")))
    )
    assert [row["sample_index"] for row in pulse_rows] == ["0", "1", "2"]
    assert [row["sample_time_s"] for row in pulse_rows] == ["0", "0.001", "0.002"]

    product_rows = list(
        csv.DictReader(io.StringIO(first_bytes["products_estimate.csv"].decode("utf-8")))
    )
    assert [row["kind"] for row in product_rows] == ["range_doppler", "range_profile"]

    export_manifest = json.loads(first_bytes["export_manifest.json"])
    assert "exported_at" not in export_manifest
    assert "timestamp" not in export_manifest
    assert export_manifest["event_manifest_id"] == event.manifest_id
    assert export_manifest["event_view_manifest_id"] == event.view_manifest_id
    assert first.event_view_manifest_id == event.view_manifest_id
    assert export_manifest["blocked_override_used"] is False
    assert str(tmp_path).encode("utf-8") not in b"".join(first_bytes.values())

    with Image.open(io.BytesIO(first_bytes["replay_summary.png"])) as image:
        assert image.info["Software"] == "ORCHAV deterministic replay exporter"
        assert image.info["Title"] == "ORCHAV sensing replay summary"
        assert "Creation Time" not in image.info


def test_composite_replay_and_stage_export_is_byte_deterministic_and_hashed(
    tmp_path: Path,
) -> None:
    event = _event()
    bundle, selection = _stage_bundle_for_event(event)
    selected_kinds = (
        "range_doppler_snr_db",
        "range_doppler_cfar_mask",
    )

    first = export_replay_session(
        event,
        tmp_path / "composite-first",
        stage_bundle=bundle,
        stage_selection=selection,
        stage_kinds=selected_kinds,
    )
    second = export_replay_session(
        event,
        tmp_path / "composite-second",
        stage_bundle=bundle,
        stage_selection=selection,
        stage_kinds=tuple(reversed(selected_kinds)),
    )

    first_bytes = _directory_bytes(first.output_directory)
    second_bytes = _directory_bytes(second.output_directory)
    assert first_bytes == second_bytes
    assert first.artifacts == second.artifacts
    stage_names = {
        "detection_stages_manifest.json",
        "detection_stages_detections.csv",
        "detection_stages_cell.csv",
        "detection_stages_diagnostic.png",
    }
    assert stage_names <= set(first_bytes)

    export_manifest = json.loads(first_bytes["export_manifest.json"])
    assert export_manifest["replay_export_version"] == 2
    assert export_manifest["detection_stages"] == {
        "bundle_disposition": "available",
        "included": True,
        "selected_cell": {
            "cluster_id": 0,
            "doppler_bin": 1,
            "pair_id": event.key.pair_id,
            "range_bin": 2,
        },
        "selected_stage_kinds": [
            "range_doppler_cfar_mask",
            "range_doppler_snr_db",
        ],
    }
    artifact_by_name = {artifact["name"]: artifact for artifact in export_manifest["artifacts"]}
    assert stage_names <= set(artifact_by_name)
    for name in stage_names:
        payload = first_bytes[name]
        assert artifact_by_name[name]["byte_size"] == len(payload)
        assert artifact_by_name[name]["sha256"] == hashlib.sha256(payload).hexdigest()

    stage_manifest = json.loads(first_bytes["detection_stages_manifest.json"])
    assert stage_manifest["event"]["acquisition_id"] == event.key.acquisition_id
    assert stage_manifest["event"]["processing_run_id"] == event.key.processing_run_id
    assert stage_manifest["selected_cell"]["selection"] == {
        "cluster_id": 0,
        "doppler_bin": 1,
        "pair_id": event.key.pair_id,
        "range_bin": 2,
    }
    with Image.open(io.BytesIO(first_bytes["detection_stages_diagnostic.png"])) as image:
        assert image.info["Software"] == "ORCHAV deterministic sensing stage export"


def test_composite_export_rejects_invalid_or_mismatched_stage_inputs(
    tmp_path: Path,
) -> None:
    event = _event()
    bundle, selection = _stage_bundle_for_event(event)

    mismatched_event = replace(
        bundle.event,
        processing_run_id=_uuid("different-stage-run"),
    )
    mismatched_bundle = replace(bundle, event=mismatched_event)
    mismatch_destination = tmp_path / "mismatched-stage"
    with pytest.raises(ReplayExportError, match="does not match replay event: run"):
        export_replay_session(
            event,
            mismatch_destination,
            stage_bundle=mismatched_bundle,
            stage_selection=selection,
        )
    assert not mismatch_destination.exists()

    invalid_bundle = replace(
        bundle,
        disposition=InspectionDisposition.INVALID,
        reason="contradictory fixture metadata",
    )
    invalid_destination = tmp_path / "invalid-stage"
    with pytest.raises(ReplayExportError, match="bundle is invalid"):
        export_replay_session(
            event,
            invalid_destination,
            stage_bundle=invalid_bundle,
        )
    assert not invalid_destination.exists()

    dangling_destination = tmp_path / "dangling-selection"
    with pytest.raises(ReplayExportError, match="requires stage_bundle"):
        export_replay_session(
            event,
            dangling_destination,
            stage_selection=selection,
        )
    assert not dangling_destination.exists()


def test_composite_export_write_failure_leaves_no_published_or_staging_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    event = _event()
    bundle, selection = _stage_bundle_for_event(event)
    destination = tmp_path / "atomic-failure"
    real_write = replay_export_module._write_fsynced

    def fail_after_replay_files(path: Path, payload: bytes) -> None:
        if path.name == "detection_stages_cell.csv":
            raise OSError("injected detector-stage write failure")
        real_write(path, payload)

    monkeypatch.setattr(replay_export_module, "_write_fsynced", fail_after_replay_files)

    with pytest.raises(OSError, match="injected detector-stage write failure"):
        export_replay_session(
            event,
            destination,
            stage_bundle=bundle,
            stage_selection=selection,
        )

    assert not destination.exists()
    assert list(tmp_path.glob(".atomic-failure.tmp-*")) == []


def test_replay_export_keeps_truth_oracle_and_estimate_layers_separate(
    tmp_path: Path,
) -> None:
    result = export_replay_session(_event(), tmp_path / "layers")
    payloads = _directory_bytes(result.output_directory)

    estimate_evidence = payloads["evidence_estimate.csv"].decode("utf-8")
    truth_evidence = payloads["evidence_truth.csv"].decode("utf-8")
    oracle_evidence = payloads["evidence_oracle.csv"].decode("utf-8")
    assert "detection-1" in estimate_evidence
    assert "track-2" in estimate_evidence
    assert "walker-truth" not in estimate_evidence
    assert "oracle-detection" not in estimate_evidence
    assert "walker-truth" in truth_evidence
    assert "detection-1" not in truth_evidence
    assert "oracle-detection" in oracle_evidence
    assert "walker-truth" not in oracle_evidence

    assert "truth_by_target" in payloads["products_truth.csv"].decode("utf-8")
    assert "truth_by_target" not in payloads["products_estimate.csv"].decode("utf-8")
    assert "oracle_micro_doppler" in payloads["products_oracle.csv"].decode("utf-8")


def test_replay_export_refuses_blocked_preflight_without_explicit_override(
    tmp_path: Path,
) -> None:
    event = _event(disposition="blocked")
    refused = tmp_path / "refused"

    with pytest.raises(ReplayExportBlockedError, match="allow_blocked=True"):
        export_replay_session(event, refused)

    assert refused.exists() is False
    allowed = export_replay_session(event, tmp_path / "allowed", allow_blocked=True)
    manifest = json.loads((allowed.output_directory / "export_manifest.json").read_bytes())
    assert allowed.blocked_override_used is True
    assert manifest["blocked_override_used"] is True
    assert manifest["preflight_disposition"] == "blocked"


def test_replay_ab_export_preserves_same_source_validation_and_exact_diff(
    tmp_path: Path,
) -> None:
    left = _event(run="run-left", recipe_threshold=3.0)
    right = _event(run="run-right", recipe_threshold=8.0)
    comparison = compare_replay_events(left, right)

    result = export_replay_session(
        left,
        tmp_path / "ab",
        comparison=comparison,
    )

    assert result.comparison_id == comparison.comparison_id
    assert (result.output_directory / "ab_comparison.json").read_bytes() == (
        comparison.to_json() + "\n"
    ).encode("utf-8")
    export_manifest = json.loads((result.output_directory / "export_manifest.json").read_bytes())
    assert export_manifest["comparison_id"] == comparison.comparison_id

    invalid = replace(comparison, source_samples_id=_hash("different-source-samples"))
    invalid_destination = tmp_path / "invalid-ab"
    with pytest.raises(ReplayExportError, match="exactly the event's source samples"):
        export_replay_session(left, invalid_destination, comparison=invalid)
    assert invalid_destination.exists() is False

    invalid_roots = replace(
        comparison,
        root_source_product_ids=(_hash("different-root-source-product"),),
    )
    invalid_roots_destination = tmp_path / "invalid-ab-roots"
    with pytest.raises(ReplayExportError, match="exactly the event's root source products"):
        export_replay_session(
            left,
            invalid_roots_destination,
            comparison=invalid_roots,
        )
    assert invalid_roots_destination.exists() is False


def test_replay_export_rejects_absolute_private_paths_without_publishing(
    tmp_path: Path,
) -> None:
    event = _event()
    unsafe_recipe = replace(
        event.recipe,
        parameters=_frozen({"fixture_path": "/private/internal/scenario.yaml"}),
    )
    unsafe_event = replace(event, recipe=unsafe_recipe)
    destination = tmp_path / "unsafe"

    with pytest.raises(ReplayExportError, match="forbids absolute path"):
        export_replay_session(unsafe_event, destination)

    assert destination.exists() is False


@pytest.mark.parametrize("destination_kind", ["directory", "file"])
def test_replay_export_preserves_a_destination_created_during_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    destination_kind: str,
) -> None:
    destination = tmp_path / "racing-output"
    real_write = replay_export_module._write_fsynced

    def create_competing_destination(path: Path, payload: bytes) -> None:
        real_write(path, payload)
        if path.name == "export_manifest.json":
            if destination_kind == "directory":
                # POSIX replace would silently consume an empty competing directory.
                destination.mkdir()
            else:
                destination.write_bytes(b"another publisher's data")

    monkeypatch.setattr(replay_export_module, "_write_fsynced", create_competing_destination)
    with pytest.raises((replay_export_module.ReplayExportError, OSError)):
        export_replay_session(_event(), destination)

    if destination_kind == "directory":
        assert destination.is_dir()
        assert list(destination.iterdir()) == []
    else:
        assert destination.read_bytes() == b"another publisher's data"
    assert list(tmp_path.glob(".racing-output.tmp-*")) == []


def test_replay_export_refuses_a_dangling_destination_link(tmp_path: Path) -> None:
    destination = tmp_path / "dangling-output"
    try:
        destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(FileExistsError, match="already exists"):
        export_replay_session(_event(), destination)
    assert destination.is_symlink()
    assert list(tmp_path.glob(".dangling-output.tmp-*")) == []
