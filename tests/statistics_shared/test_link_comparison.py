"""Physical reductions, identity matching and portable link analysis."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from shared.statistics.link_comparison import (
    COMPARISON_SCHEMA,
    EMPTY,
    INVALID,
    MISSING,
    POSITIVE,
    analyze_comparison,
    export_comparison,
    export_recording,
    load_definition,
    load_link_recording,
    paired_difference,
    reduce_link_frames,
    site_selection,
    support_weights,
)


def frame(index=0, *, timestamp=0.0, losses=(10.0, 20.0), valid=(True, True)):
    return SimpleNamespace(
        frame_index=index,
        tx_names=("Roof", "South"),
        rx_names=("Cyclist", "Peer"),
        path_loss_db=np.asarray(losses),
        metric_is_valid=lambda _: np.asarray(valid),
        tx_rx_pairs=np.asarray([[0, 0], [1, 0], [1, 1]]),
        pair_path_offsets=np.asarray([0, 2, 2, 2]),
        tx_positions=np.zeros((2, 3)),
        rx_positions=np.ones((2, 3)),
        timestamp_s=timestamp,
    )


def recording():
    r = reduce_link_frames([frame(), frame(1, timestamp=2.0)], [0, 1], {"Roof": 0.0, "South": 10.0})
    r.metadata.update(
        carrier_frequency_hz=1e9,
        receiver_model={"pattern": "iso"},
        transmitter_model={"pattern": "iso"},
        scene={"id": "fixture"},
    )
    return r


def test_linear_incoherent_sum_and_distinct_empty_missing_invalid():
    r = recording()
    assert r.arrays["gain_linear"][0, 0, 0] == pytest.approx(0.11)
    assert r.arrays["rss_dbm"][0, 0, 0] == pytest.approx(10 * np.log10(0.11))
    assert r.arrays["status"][0].tolist() == [[POSITIVE, MISSING], [EMPTY, EMPTY]]
    assert np.isneginf(r.arrays["rss_dbm"][0, 1, 0])
    f = frame(valid=(True, False))
    bad = reduce_link_frames([f], [0], {"Roof": 0.0, "South": 10.0})
    assert bad.arrays["status"][0, 0, 0] == INVALID
    assert np.isnan(bad.arrays["rss_dbm"][0, 0, 0])


def test_unknown_clock_needs_exact_binding_and_mismatch_fails():
    f = frame(timestamp=None)
    with pytest.raises(ValueError, match="binding"):
        reduce_link_frames([f], [0], {"Roof": 0.0, "South": 10.0})
    r = reduce_link_frames([f], [0], {"Roof": 0.0, "South": 10.0}, time_binding={0: 4.0})
    assert r.arrays["time_s"][0] == 4
    assert np.isnan(r.arrays["recorded_time_s"][0])
    with pytest.raises(ValueError, match="every selected"):
        reduce_link_frames([f], [0], {"Roof": 0.0, "South": 10.0}, time_binding={1: 4.0})
    with pytest.raises(ValueError, match="disagree"):
        reduce_link_frames([frame()], [0], {"Roof": 0.0, "South": 10.0}, time_binding={0: 4.0})


def test_missing_power_identity_duplicate_pair_and_state_identity_fail():
    with pytest.raises(ValueError, match="power"):
        reduce_link_frames([frame()], [0], {"Roof": 0.0})
    f = frame()
    f.tx_rx_pairs[1] = [0, 0]
    with pytest.raises(ValueError, match="Duplicate"):
        reduce_link_frames([f], [0], {"Roof": 0.0, "South": 10.0})
    with pytest.raises(ValueError, match="identity"):
        reduce_link_frames([frame(2)], [0], {"Roof": 0.0, "South": 10.0})


def test_named_matching_and_physical_compatibility():
    r = recording()
    arrays = {k: v.copy() for k, v in r.arrays.items()}
    arrays["rss_dbm"][:, 0, 0] += 3.0
    other = replace(r, arrays=arrays)
    diff, mask = paired_difference(r, other, "Roof", "Cyclist")
    np.testing.assert_allclose(diff, 3.0)
    assert mask.all()
    empty, _ = paired_difference(r, other, "South", "Cyclist")
    assert np.isnan(empty).all()
    bad = replace(other, metadata={**r.metadata, "carrier_frequency_hz": 2e9})
    with pytest.raises(ValueError, match="carrier"):
        paired_difference(r, bad, "Roof", "Cyclist")
    arrays["rx_positions_m"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="positions"):
        paired_difference(r, other, "Roof", "Cyclist")


def test_site_identity_and_observed_margin_never_label_unknown_winner():
    winner, margin = site_selection(
        np.array([[-90, -80], [-np.inf, -80], [-np.inf, -np.inf], [np.nan, -80]])
    )
    assert winner.tolist() == [1, 1, -1, -2]
    assert margin[0] == 10 and np.isposinf(margin[1])
    assert np.isnan(margin[2:]).all()


def test_irregular_time_support_has_trapezoidal_not_sample_weights():
    np.testing.assert_allclose(support_weights([0, 1, 4]), [0.125, 0.5, 0.375])
    with pytest.raises(ValueError, match="Stationary"):
        support_weights([0, 0])


def test_portable_definition_and_export(tmp_path):
    r = recording()
    export_recording(r, tmp_path / "obs")
    definition = dict(
        schema=COMPARISON_SCHEMA,
        sources=[dict(label="Example", path="obs.npz")],
        receiver="Cyclist",
        transmitters=["South", "Roof"],
        axis="time",
        paired=False,
    )
    (tmp_path / "definition.json").write_text(json.dumps(definition))
    d, records = load_definition(tmp_path / "definition.json")
    result = analyze_comparison(d, records)
    assert result["traces"][0]["winner"].tolist() == [1, 1]
    export_comparison(d, records, tmp_path / "export")
    replay, loaded = load_definition(tmp_path / "export/definition.json")
    np.testing.assert_equal(
        analyze_comparison(replay, loaded)["traces"][0]["values"], result["traces"][0]["values"]
    )
    assert (tmp_path / "export/comparison.pdf").is_file()
    with (tmp_path / "obs.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="identity"):
        load_link_recording(tmp_path / "obs.npz")


def test_unrelated_native_provider_fixture(tmp_path):
    from shared.frames.providers import Hdf5Provider
    from tests.shared.test_hdf5_format_handler_v2 import _write_minimal_packed_frame_set

    _write_minimal_packed_frame_set(tmp_path, frame_id=7)
    with Hdf5Provider(tmp_path) as provider:
        native = provider.load_frame(7)
        powers = {n: 5.0 for n in native.tx_names}
    r = load_link_recording(tmp_path, tx_power_dbm=powers, time_binding={7: 0.0})
    assert r.arrays["frame_ids"].tolist() == [7]
    assert r.tx_names == tuple(native.tx_names)
    assert len(r.metadata["source_files"]) >= 2


def test_compact_shape_status_and_power_consistency():
    from shared.statistics.link_comparison import validate_recording

    record = recording()
    validate_recording(record)
    arrays = {k: v.copy() for k, v in record.arrays.items()}
    arrays["status"][0, 0, 0] = EMPTY
    with pytest.raises(ValueError, match="Status"):
        validate_recording(replace(record, arrays=arrays))
    arrays = {k: v.copy() for k, v in record.arrays.items()}
    arrays["rss_dbm"][0, 0, 0] += 2
    with pytest.raises(ValueError, match="Stored dBm"):
        validate_recording(replace(record, arrays=arrays))
    arrays["rss_dbm"] = arrays["rss_dbm"][:1]
    with pytest.raises(ValueError, match="shape"):
        validate_recording(replace(record, arrays=arrays))
