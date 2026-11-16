"""Contract and analytic-reference tests for live sensing CIR interchange."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from shared.sensing.interchange import (
    SensingInterchangeError,
    adapt_slow_time_to_positive_receding,
    build_live_cir_interchange,
    build_single_path_stage_evidence,
    canonical_json_bytes,
    complex_pairs_to_array,
    load_live_cir_interchange,
    load_single_path_stage_evidence,
    validate_live_cir_interchange,
    validate_single_path_stage_evidence,
    write_live_cir_interchange,
    write_single_path_stage_evidence,
)


def _record(doppler_hz: float, *, case_name: str = "controlled") -> dict:
    sample_rate_hz = 2_000.0
    sample_count = 64
    sample_times_s = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
    initial = 0.5 - 0.25j
    samples = initial * np.exp(1j * 2.0 * np.pi * doppler_hz * sample_times_s)
    return build_live_cir_interchange(
        case_name=case_name,
        carrier_frequency_hz=28.0e9,
        sample_times_s=sample_times_s,
        cir_by_path=samples.reshape(1, -1),
        delays_s=[20.0e-9],
        doppler_hz=[doppler_hz],
        path_roles=["target"],
        path_ids=["sionna-path:7"],
        target_ids=["controlled-reflector"],
        angles_rad=[
            {
                "departure_azimuth": 0.1,
                "departure_zenith": 1.2,
                "arrival_azimuth": -0.1,
                "arrival_zenith": 1.1,
            }
        ],
        geometry={
            "tx_position_m": [0.0, 0.0, 1.0],
            "rx_position_m": [0.0, 0.0, 1.0],
            "tx_velocity_m_s": [0.0, 0.0, 0.0],
            "rx_velocity_m_s": [0.0, 0.0, 0.0],
            "target_reference_position_m": [0.0, 0.0, 0.0],
            "target_velocity_m_s": [0.0, 0.0, 0.5],
        },
        provenance={
            "producer": "test",
            "source_kind": "analytic_fixture",
            "source_commit": "0" * 40,
        },
    )


def _copy(record: dict) -> dict:
    return json.loads(json.dumps(record))


def _rehash(record: dict) -> dict:
    record.pop("content_sha256", None)
    record.pop("interchange_id", None)
    digest = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
    record["content_sha256"] = digest
    record["interchange_id"] = f"sha256:{digest}"
    return record


def _rehash_stage(record: dict) -> dict:
    record.pop("content_sha256", None)
    record.pop("stage_evidence_id", None)
    digest = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
    record["content_sha256"] = digest
    record["stage_evidence_id"] = f"sha256:{digest}"
    return record


def test_live_cir_interchange_round_trips_complex_samples_and_identity(tmp_path: Path) -> None:
    record = _record(93.75, case_name="approach")
    validate_live_cir_interchange(record)

    assert record["scope"] == {
        "topology": "siso",
        "capture_boundary": "live_sionna_paths_cir_before_freezing",
        "ordinary_frame_replay_supported": False,
        "case_name": "approach",
    }
    assert record["conventions"]["orchav_doppler_sign"] == "positive_approaching"
    assert record["conventions"]["positive_receding_peer_doppler_sign"] == "positive_receding"
    assert record["paths"][0]["doppler"] == {
        "orchav_hz": 93.75,
        "positive_receding_peer_hz": -93.75,
    }
    assert record["paths"][0]["interaction_role"] == "target"
    assert record["paths"][0]["target_id"] == "controlled-reflector"
    assert record["interchange_id"] == "sha256:" + record["content_sha256"]
    assert len(record["content_sha256"]) == 64

    path = write_live_cir_interchange(tmp_path / "fixture.json", record)
    loaded = load_live_cir_interchange(path)
    assert loaded == record
    np.testing.assert_array_equal(
        complex_pairs_to_array(loaded["paths"][0]["coefficient_samples_ri"]),
        complex_pairs_to_array(record["paths"][0]["coefficient_samples_ri"]),
    )
    assert not complex_pairs_to_array(record["paths"][0]["coefficient_samples_ri"]).flags.writeable


def test_interchange_publication_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "fixture.json"
    write_live_cir_interchange(destination, _record(0.0))
    original = destination.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_live_cir_interchange(destination, _record(93.75))

    assert destination.read_bytes() == original
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize(
    ("case_name", "orchav_hz", "expected_peer_hz"),
    (("static", 0.0, 0.0), ("approach", 93.75, -93.75), ("recede", -93.75, 93.75)),
)
def test_single_path_stages_cover_static_approach_and_recede(
    case_name: str, orchav_hz: float, expected_peer_hz: float
) -> None:
    evidence = build_single_path_stage_evidence(_record(orchav_hz, case_name=case_name))

    assert evidence["passed"] is True
    assert evidence["expected"]["orchav_doppler_hz"] == orchav_hz
    assert evidence["expected"]["positive_receding_peer_doppler_hz"] == expected_peer_hz
    assert evidence["orchav_convention_stages"]["raw_peak_hz"] == pytest.approx(orchav_hz)
    assert evidence["positive_receding_peer_reference_stages"]["raw_peak_hz"] == pytest.approx(
        expected_peer_hz
    )
    assert evidence["checks"]["mean_removed_energy_preserved_by_adapter"] is True
    assert evidence["scope"] == {
        "pre_detection_only": True,
        "waveform_link_budget_parity_claimed": False,
        "detector_parity_claimed": False,
        "matlab_execution_claimed": False,
    }
    assert evidence["stage_evidence_id"] == "sha256:" + evidence["content_sha256"]
    validate_single_path_stage_evidence(evidence)
    if case_name == "static":
        assert evidence["checks"]["static_raw_dc_is_visible"] is True
        assert evidence["checks"]["static_common_mean_removal_is_zero"] is True
    else:
        assert evidence["checks"]["mean_removed_peak_matches_orchav"] is True
        assert evidence["checks"]["mean_removed_peak_matches_positive_receding_peer"] is True


def test_sign_adapter_preserves_first_sample_magnitude_and_reverses_phase_slope() -> None:
    times = np.arange(64) / 2_000.0
    original = (0.3 + 0.7j) * np.exp(1j * 2.0 * np.pi * 125.0 * times)
    adapted = adapt_slow_time_to_positive_receding(original)

    assert adapted[0] == pytest.approx(original[0])
    np.testing.assert_allclose(np.abs(adapted), np.abs(original), rtol=1.0e-15, atol=0.0)
    original_slope = np.polyfit(times, np.unwrap(np.angle(original)), 1)[0]
    adapted_slope = np.polyfit(times, np.unwrap(np.angle(adapted)), 1)[0]
    assert adapted_slope == pytest.approx(-original_slope, rel=1.0e-13)
    assert not adapted.flags.writeable


def test_tampered_interchange_is_rejected() -> None:
    record = _copy(_record(93.75))
    record["paths"][0]["coefficient_samples_ri"][3][0] += 1.0

    with pytest.raises(SensingInterchangeError, match="content hash mismatch"):
        validate_live_cir_interchange(record)


def test_doppler_adapter_cannot_be_removed_or_applied_twice_silently() -> None:
    record = _copy(_record(93.75))
    record["paths"][0]["doppler"]["positive_receding_peer_hz"] = 93.75
    _rehash(record)

    with pytest.raises(SensingInterchangeError, match="signs are not exact opposites"):
        validate_live_cir_interchange(record)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("missing_target_id", "target paths require"),
        ("non_target_has_target_id", "only target paths"),
        ("missing_loss_disclosure", "lossy_fields"),
        ("nonuniform_clock", "uniformly sampled"),
        ("nan_sample", "finite path"),
    ),
)
def test_builder_fails_closed_for_incomplete_live_provenance(change: str, message: str) -> None:
    sample_rate = 2_000.0
    times = np.arange(64, dtype=np.float64) / sample_rate
    samples = np.ones((1, 64), dtype=np.complex128)
    roles = ["target"]
    targets: list[str | None] = ["reflector"]
    losses = [
        "full_mimo_tensor",
        "polarization_components",
        "interaction_vertices",
        "material_interaction_chain",
    ]
    if change == "missing_target_id":
        targets = [None]
    elif change == "non_target_has_target_id":
        roles = ["background"]
    elif change == "missing_loss_disclosure":
        losses.pop()
    elif change == "nonuniform_clock":
        times[10] += 1.0e-5
    elif change == "nan_sample":
        samples[0, 2] = np.nan + 0j

    with pytest.raises(SensingInterchangeError, match=message):
        build_live_cir_interchange(
            case_name="bad",
            carrier_frequency_hz=28.0e9,
            sample_times_s=times,
            cir_by_path=samples,
            delays_s=[1.0e-8],
            doppler_hz=[0.0],
            path_roles=roles,
            path_ids=["path:0"],
            target_ids=targets,
            geometry={
                "tx_position_m": [0.0, 0.0, 0.0],
                "rx_position_m": [0.0, 0.0, 0.0],
                "tx_velocity_m_s": [0.0, 0.0, 0.0],
                "rx_velocity_m_s": [0.0, 0.0, 0.0],
                "target_reference_position_m": [0.0, 0.0, 1.0],
                "target_velocity_m_s": [0.0, 0.0, 0.0],
            },
            provenance={"producer": "test"},
            lossy_fields=losses,
        )


def test_loader_rejects_non_json_and_nonfinite_payloads(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("not-json", encoding="utf-8")
    with pytest.raises(SensingInterchangeError, match="could not load"):
        load_live_cir_interchange(broken)

    with pytest.raises(SensingInterchangeError, match="finite"):
        complex_pairs_to_array([[1.0, 0.0], [float("inf"), 0.0]])

    malformed = _copy(_record(93.75))
    malformed["paths"][0]["coefficient_samples_ri"][0][0] = "not-a-number"
    _rehash(malformed)
    malformed_path = tmp_path / "malformed-samples.json"
    malformed_path.write_bytes(canonical_json_bytes(malformed))
    with pytest.raises(SensingInterchangeError, match="finite numeric pairs"):
        load_live_cir_interchange(malformed_path)


def test_stage_evidence_round_trip_authenticates_and_refuses_overwrite(tmp_path: Path) -> None:
    evidence = build_single_path_stage_evidence(_record(93.75))
    path = write_single_path_stage_evidence(tmp_path / "stages.json", evidence)

    assert load_single_path_stage_evidence(path) == evidence
    with pytest.raises(FileExistsError, match="refusing to overwrite stage evidence"):
        write_single_path_stage_evidence(path, evidence)


def test_stage_evidence_rejects_tampering(tmp_path: Path) -> None:
    evidence = build_single_path_stage_evidence(_record(93.75))
    evidence["expected"]["orchav_doppler_hz"] = -93.75
    path = tmp_path / "tampered-stages.json"
    path.write_bytes(canonical_json_bytes(evidence))

    with pytest.raises(SensingInterchangeError, match="content hash mismatch"):
        load_single_path_stage_evidence(path)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("matlab_claim", "scope exceeds"),
        ("short_interchange_id", "authenticated interchange"),
        ("boolean_schema", "unsupported"),
        ("missing_check", "check verdict"),
        ("malformed_stage", "finite numeric pairs"),
        ("negative_bin_spacing", "expectations are inconsistent"),
    ),
)
def test_stage_loader_rejects_self_hashed_semantically_invalid_records(
    tmp_path: Path, change: str, message: str
) -> None:
    evidence = _copy(build_single_path_stage_evidence(_record(93.75)))
    if change == "matlab_claim":
        evidence["scope"]["matlab_execution_claimed"] = True
    elif change == "short_interchange_id":
        evidence["interchange_id"] = "sha256:1234"
    elif change == "boolean_schema":
        evidence["schema_version"] = True
    elif change == "missing_check":
        evidence["checks"].pop("raw_peak_matches_orchav")
    elif change == "malformed_stage":
        evidence["orchav_convention_stages"]["raw_doppler_ri"][0][0] = "bad"
    elif change == "negative_bin_spacing":
        evidence["expected"]["bin_spacing_hz"] *= -1.0
    _rehash_stage(evidence)
    path = tmp_path / f"invalid-{change}.json"
    path.write_bytes(canonical_json_bytes(evidence))

    with pytest.raises(SensingInterchangeError, match=message):
        load_single_path_stage_evidence(path)


def test_stage_evidence_publication_cleans_partial_on_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = build_single_path_stage_evidence(_record(93.75))
    path = tmp_path / "stages.json"

    def fail_link(_source: object, _destination: object) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr("shared.sensing.interchange.os.link", fail_link)
    with pytest.raises(OSError, match="injected publication failure"):
        write_single_path_stage_evidence(path, evidence)

    assert not path.exists()
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob(".*.partial"))


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("blank_case", "case_name"),
        ("negative_carrier", "carrier_frequency"),
        ("nonnumeric_carrier", "clock scalar fields"),
        ("changed_phase_law", "complete v1 contract"),
        ("changed_delay_unit", "complete v1 contract"),
        ("malformed_angles", "angles do not match"),
        ("boolean_delay", "path delay"),
        ("string_carrier", "clock scalar fields"),
        ("boolean_time", "clock.sample_times_s"),
        ("boolean_geometry", "geometry.tx_position_m"),
        ("string_delay", "path delay"),
        ("string_doppler", "Doppler values"),
        ("string_angle", "angles must be JSON numbers"),
        ("extra_top_level", "top-level fields"),
    ),
)
def test_loader_rejects_self_hashed_semantically_invalid_v1_records(
    tmp_path: Path, change: str, message: str
) -> None:
    record = _copy(_record(93.75))
    if change == "blank_case":
        record["scope"]["case_name"] = "  "
    elif change == "negative_carrier":
        record["clock"]["carrier_frequency_hz"] = -28.0e9
    elif change == "nonnumeric_carrier":
        record["clock"]["carrier_frequency_hz"] = "not-a-number"
    elif change == "changed_phase_law":
        record["conventions"]["complex_slow_time_phase"] = "exp(-j*2*pi*f*t)"
    elif change == "changed_delay_unit":
        record["conventions"]["delay_unit"] = "ns"
    elif change == "malformed_angles":
        record["paths"][0]["angles_rad"].pop("arrival_zenith")
    elif change == "boolean_delay":
        record["paths"][0]["delay_s"] = True
    elif change == "string_carrier":
        record["clock"]["carrier_frequency_hz"] = "28000000000"
    elif change == "boolean_time":
        record["clock"]["sample_times_s"][0] = False
    elif change == "boolean_geometry":
        record["geometry"]["tx_position_m"][0] = False
    elif change == "string_delay":
        record["paths"][0]["delay_s"] = "2e-8"
    elif change == "string_doppler":
        record["paths"][0]["doppler"]["orchav_hz"] = "93.75"
    elif change == "string_angle":
        record["paths"][0]["angles_rad"]["arrival_zenith"] = "1.1"
    elif change == "extra_top_level":
        record["unversioned_extension"] = 1
    _rehash(record)
    path = tmp_path / f"{change}.json"
    path.write_bytes(canonical_json_bytes(record))

    with pytest.raises(SensingInterchangeError, match=message):
        load_live_cir_interchange(path)
