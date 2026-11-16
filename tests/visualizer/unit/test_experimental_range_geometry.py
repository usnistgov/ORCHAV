"""Scientific and binding checks for the NumPy-only selected range locus."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload
from visualizer.src.experiments.range_geometry import build_range_locus
from visualizer.src.sensing.selection import inspection_cell, select_estimate
from visualizer.src.sensing.update_request import SensingUpdateRequest


def _request(
    *,
    mode="bistatic",
    ranges=(12.0, 20.0, 34.0),
    index=1,
    tx=(-3.0, 0.0, 0.0),
    rx=(3.0, 0.0, 0.0),
    pair_index=0,
):
    sensing = build_visualizer_sensing_payload(
        {
            "range_axis_m": np.asarray(ranges, dtype=float),
            "doppler_axis_hz": np.array([-1.0, 0.0, 1.0]),
            "range_doppler_power_by_pair": np.zeros((1, 3, len(ranges))),
            "config_resolved": {"range_mode": mode},
        },
        scene_frame_id=7,
        sample_time_s=1.25,
    )
    sensing["pair_axis"][0]["frame_pair_index"] = pair_index
    tx_positions = np.full((pair_index + 1, 3), -999.0)
    rx_positions = np.full((pair_index + 1, 3), 999.0)
    tx_positions[pair_index] = tx
    rx_positions[pair_index] = rx
    frame = {
        "sensing": sensing,
        "tx_positions": tx_positions,
        "rx_positions": rx_positions,
        "timestamp_s": 1.25,
    }
    request = SensingUpdateRequest(
        display_frame=frame,
        state_frame={"tx_positions": [[1000, 0, 0]], "rx_positions": [[2000, 0, 0]]},
        state_step=900,
        display_step=7,
        source_identity=("recording", 5),
        variant="primary",
        range_mode=mode,
        rd_result=SimpleNamespace(
            range_axis_m=sensing["range_axis_m"], doppler_axis_hz=sensing["doppler_axis_hz"]
        ),
    )
    cell = inspection_cell(request, index, 1)
    selection = replace(select_estimate(sensing), source=request.source_identity, cell=cell)
    return request, selection


def _descriptor(request):
    return next(iter(request.display_frame["sensing"]["products"].values()))


def _focal_sum(points, foci):
    return np.linalg.norm(points - foci[0], axis=1) + np.linalg.norm(points - foci[1], axis=1)


def test_bistatic_mesh_vertices_obey_both_numerical_boundaries_and_center_section():
    request, selection = _request()
    locus = build_range_locus(request, selection)
    assert len(locus.surfaces) == 2
    for (vertices, triangles), length in zip(locus.surfaces, (16.0, 27.0), strict=True):
        np.testing.assert_allclose(_focal_sum(vertices, locus.foci), length, atol=2e-13)
        assert vertices.dtype == np.float64
        assert triangles.dtype == np.int32
        assert vertices.shape == (1986, 3)
        assert triangles.shape == (3968, 3)
        assert triangles.min() == 0
        assert triangles.max() == len(vertices) - 1
        # No duplicated pole rings and no zero-area triangles.
        p = vertices[triangles]
        normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        assert np.all(np.linalg.norm(normals, axis=1) > 0)
        assert np.all(np.sum(normals * (np.mean(p, axis=1) - locus.center), axis=1) > 0)
        edges = np.sort(
            np.concatenate((triangles[:, :2], triangles[:, 1:], triangles[:, ::2])), axis=1
        )
        assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
    np.testing.assert_allclose(_focal_sum(locus.equator, locus.foci), 20.0, atol=2e-13)
    np.testing.assert_array_equal(locus.equator[0], locus.equator[-1])
    np.testing.assert_array_equal(locus.equator[:, 2], np.full(65, locus.center[2]))


@pytest.mark.parametrize("direction", [(0, 0, 1), (0, 1, 0), (1, 2, 3), (-3, 2, -1)])
def test_arbitrary_three_dimensional_rotations_and_translations(direction):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    center = np.array([101.25, -83.5, 7.0])
    request, selection = _request(tx=center - 3 * direction, rx=center + 3 * direction)
    locus = build_range_locus(request, selection)
    np.testing.assert_allclose(locus.center, center)
    for (vertices, _), length in zip(locus.surfaces, (16, 27), strict=True):
        np.testing.assert_allclose(_focal_sum(vertices, locus.foci), length, atol=5e-13)
        projected = (vertices - center) @ direction
        np.testing.assert_allclose([projected.min(), projected.max()], [-length / 2, length / 2])
    np.testing.assert_allclose(_focal_sum(locus.equator, locus.foci), 20, atol=5e-13)
    np.testing.assert_allclose(locus.equator[:, 2], center[2])


def test_monostatic_range_and_coincident_bistatic_length_differ_by_factor_two():
    mono = build_range_locus(
        *_request(mode="monostatic", ranges=(20,), index=0, tx=(2, 3, 4), rx=(2, 3, 4))
    )
    bi = build_range_locus(*_request(ranges=(20,), index=0, tx=(2, 3, 4), rx=(2, 3, 4)))
    assert len(mono.surfaces) == len(bi.surfaces) == 1
    np.testing.assert_allclose(np.linalg.norm(mono.surfaces[0][0] - mono.center, axis=1), 20)
    np.testing.assert_allclose(np.linalg.norm(bi.surfaces[0][0] - bi.center, axis=1), 10)
    assert "R=20 m" in mono.label
    assert "L=20 m" in bi.label
    assert "support unknown" in bi.disclosures[0]
    assert "unthickened" in bi.disclosures[0]


def test_monostatic_convention_rejects_noncoincident_positions():
    with pytest.raises(ValueError, match="coincident TX/RX"):
        build_range_locus(*_request(mode="monostatic"))


@pytest.mark.parametrize("length, message", [(5.0, "shorter"), (6.0, "degenerate")])
def test_impossible_and_degenerate_bistatic_loci_fail_closed(length, message):
    with pytest.raises(ValueError, match=message):
        build_range_locus(*_request(ranges=(length,), index=0))


def test_invalid_lower_display_boundary_is_not_clipped_or_silently_omitted():
    with pytest.raises(ValueError, match="shorter"):
        build_range_locus(*_request(ranges=(7, 12), index=0))


def test_degenerate_lower_boundary_is_not_rendered_as_a_zero_area_mesh():
    with pytest.raises(ValueError, match="degenerate"):
        build_range_locus(*_request(ranges=(8, 12), index=0))


def test_zero_radius_monostatic_surface_is_explicitly_degenerate():
    with pytest.raises(ValueError, match="degenerate"):
        build_range_locus(
            *_request(mode="monostatic", ranges=(0,), index=0, tx=(0, 0, 0), rx=(0, 0, 0))
        )


def test_exact_pair_pose_row_wins_over_first_pair_and_current_scene():
    request, selection = _request(tx=(1, 2, 3), rx=(4, 6, 3), pair_index=2)
    locus = build_range_locus(request, selection)
    np.testing.assert_array_equal(locus.foci, [[1, 2, 3], [4, 6, 3]])
    request.state_frame.clear()
    assert build_range_locus(request, selection).identity == locus.identity


@pytest.mark.parametrize("key", ["tx_positions", "rx_positions"])
@pytest.mark.parametrize("value", [None, [], [1, 2, 3], [[np.nan, 0, 0]], [[np.inf, 0, 0]]])
def test_missing_or_invalid_bound_poses_reject_instead_of_using_current_scene(key, value):
    request, selection = _request()
    request.display_frame[key] = value
    with pytest.raises(ValueError, match=key):
        build_range_locus(request, selection)


@pytest.mark.parametrize("index", [-1, True, 0.5, 2])
def test_invalid_frame_pair_index_is_not_fallback_to_zero(index):
    request, selection = _request()
    request.display_frame["sensing"]["pair_axis"][0]["frame_pair_index"] = index
    with pytest.raises(ValueError, match="frame_pair_index"):
        build_range_locus(request, selection)


@pytest.mark.parametrize("field", ["pair_id", "tx_sensor_id", "rx_sensor_id"])
def test_pair_axis_cannot_borrow_another_acquisitions_pose(field):
    request, selection = _request()
    request.display_frame["sensing"]["pair_axis"][0][field] = "other"
    with pytest.raises(ValueError, match=field):
        build_range_locus(request, selection)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"source": ("other",)}, "source"),
        ({"scope": ("other", "run", "pair")}, "scope"),
        ({"acquisition_id": "other"}, "acquisition"),
        ({"cell": None}, "exact sensing cell"),
    ],
)
def test_selection_identity_is_exact(changes, message):
    request, selection = _request()
    with pytest.raises(ValueError, match=message):
        build_range_locus(request, replace(selection, **changes))


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"source": ("other",)}, "source"),
        ({"recipe": "other"}, "recipe"),
        ({"product_id": "other"}, "product identity"),
        ({"catalog_identity": ()}, "catalog"),
        ({"range_mode": "monostatic"}, "range mode"),
        ({"variant": "residual"}, "variant"),
        ({"combination": "other"}, "combination"),
        ({"ranges": (12.0, 20.0, 35.0)}, "range coordinates"),
        ({"doppler": (-2.0, 0.0, 1.0)}, "Doppler coordinates"),
        ({"range_bin": -1}, "range bin"),
        ({"range_bin": True}, "range bin"),
        ({"doppler_bin": 9}, "Doppler bin"),
    ],
)
def test_cell_identity_and_exact_coordinates_are_not_reinterpreted(changes, message):
    request, selection = _request()
    with pytest.raises(ValueError, match=message):
        build_range_locus(request, replace(selection, cell=replace(selection.cell, **changes)))


@pytest.mark.parametrize("field", ["acquisition_id", "processing_run_id", "processing_recipe_id"])
def test_product_lineage_must_match_selected_occurrence(field):
    request, selection = _request()
    _descriptor(request)["lineage"][field] = "other"
    with pytest.raises(ValueError, match=field):
        build_range_locus(request, selection)


def test_replaced_product_catalog_is_stale_even_if_selected_product_is_unchanged():
    request, selection = _request()
    request.display_frame["sensing"]["products"]["new"] = {"product_id": "new"}
    with pytest.raises(ValueError, match="catalog"):
        build_range_locus(request, selection)


@pytest.mark.parametrize("value", [None, float("nan"), 1.0, True])
def test_bound_pose_epoch_must_be_recorded_finite_and_exact(value):
    request, selection = _request()
    request.display_frame["timestamp_s"] = value
    with pytest.raises(ValueError, match="epoch|coordinates"):
        build_range_locus(request, selection)


def test_missing_pose_epoch_rejects_even_when_current_scene_has_the_right_time():
    request, selection = _request()
    del request.display_frame["timestamp_s"]
    request.state_frame["timestamp_s"] = 1.25
    with pytest.raises(ValueError, match="recorded reference epoch"):
        build_range_locus(request, selection)


@pytest.mark.parametrize("display_step", [None, 8])
def test_scene_binding_cannot_fall_back_to_current_frame(display_step):
    request, selection = _request()
    request = replace(request, display_step=display_step, state_step=7)
    with pytest.raises(ValueError, match="temporal scene binding"):
        build_range_locus(request, selection)


def test_stale_bound_scene_provenance_rejects():
    request, selection = _request()
    request.display_frame["_sensing_display_provenance"] = {"bound_scene_frame_id": 8}
    with pytest.raises(ValueError, match="provenance"):
        build_range_locus(request, selection)


def test_pinned_reference_frame_ignores_different_current_scene():
    request, selection = _request()
    request = replace(request, display_from_cache=True)
    request.display_frame["_sensing_display_provenance"] = {
        "bound_scene_frame_id": 7,
        "current_scene_frame_id": 900,
        "stale_for_current_scene": True,
    }
    assert "bound scene 7" in build_range_locus(request, selection).disclosures[1]


@pytest.mark.parametrize("ranges", [(20, 20), (20, 12), (12, np.nan), (12, np.inf), (-3, 20)])
def test_range_axis_must_be_strictly_increasing_finite_and_nonnegative(ranges):
    request, selection = _request()
    selection = replace(selection, cell=replace(selection.cell, ranges=ranges))
    with pytest.raises(ValueError, match="ranges|range coordinates"):
        build_range_locus(request, selection)


@pytest.mark.parametrize("key", ["range_edges_m", "range_bin_edges_m", "custom_edges"])
def test_exported_support_takes_precedence_over_nonuniform_midpoints(key):
    request, selection = _request()
    request.display_frame["sensing"][key] = np.array([9, 17, 24, 40])
    if key == "custom_edges":
        _descriptor(request)["axes"][-1]["edges_key"] = key
    locus = build_range_locus(request, selection)
    for (vertices, _), length in zip(locus.surfaces, (17, 24), strict=True):
        np.testing.assert_allclose(_focal_sum(vertices, locus.foci), length)
    assert "Exported numerical bin boundaries" in locus.disclosures[0]


@pytest.mark.parametrize(
    "edges", [[10, 15, 25], [10, 15, 15, 40], [10, 21, 25, 40], [10, 15, np.inf, 40]]
)
def test_invalid_exported_edges_reject_instead_of_midpoint_fallback(edges):
    request, selection = _request()
    request.display_frame["sensing"]["range_edges_m"] = edges
    with pytest.raises(ValueError, match="range edges"):
        build_range_locus(request, selection)


def test_single_coordinate_can_use_explicit_exported_support():
    request, selection = _request(ranges=(20,), index=0)
    request.display_frame["sensing"]["range_bin_edges_m"] = [18, 22]
    assert len(build_range_locus(request, selection).surfaces) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("coordinate_semantics", "monostatic_range"),
        ("coordinate_semantics", "range"),
        ("unit", "ns"),
        ("coordinate_kind", "catalog"),
        ("reference_frame", "rx_array_local"),
    ],
)
def test_wrong_range_semantics_are_not_guessed_from_request_mode(field, value):
    request, selection = _request()
    _descriptor(request)["axes"][-1][field] = value
    with pytest.raises(ValueError, match="semantics"):
        build_range_locus(request, selection)


def test_frozen_result_arrays_are_detached_and_cannot_be_made_writable():
    request, selection = _request()
    locus = build_range_locus(request, selection)
    arrays = [locus.center, locus.foci, locus.equator]
    arrays.extend(array for pair in locus.surfaces for array in pair)
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        locus.label = "other"
    request.display_frame["tx_positions"][:] = 123
    np.testing.assert_array_equal(locus.foci[0], [-3, 0, 0])


def test_identity_tracks_exact_acquisition_support_and_reference_pose():
    request, selection = _request()
    locus = build_range_locus(request, selection)
    assert locus.identity == build_range_locus(request, selection).identity
    request.display_frame["sensing"]["range_edges_m"] = [9, 17, 24, 40]
    assert locus.identity != build_range_locus(request, selection).identity
    del request.display_frame["sensing"]["range_edges_m"]
    request.display_frame["tx_positions"][0, 1] = 1
    assert locus.identity != build_range_locus(request, selection).identity


def test_no_truth_target_doppler_angular_or_rcs_inference():
    request, selection = _request()
    expected = build_range_locus(request, selection)
    request.display_frame["sensing"]["truth"] = {"targets": [{"position_m": [999, 999, 999]}]}
    request.display_frame["sensing"]["detections"] = [{"range_m": 999, "x_m": 123}]
    actual = build_range_locus(request, selection)
    for (before, _), (after, _) in zip(expected.surfaces, actual.surfaces, strict=True):
        np.testing.assert_array_equal(before, after)
    disclosures = " ".join(actual.disclosures)
    for word in (
        "Frozen",
        "Direct-two-leg",
        "not a target",
        "probability",
        "diffraction",
        "RCS",
        "not uncertainty",
        "physical resolution",
        "Midpoint-derived",
    ):
        assert word in disclosures


def test_unselected_pose_rows_are_not_used_or_required_to_be_finite():
    request, selection = _request(pair_index=1)
    request.display_frame["tx_positions"][0] = np.nan
    request.display_frame["rx_positions"][0] = np.inf
    np.testing.assert_array_equal(
        build_range_locus(request, selection).foci, [[-3, 0, 0], [3, 0, 0]]
    )


def test_nearly_degenerate_but_valid_ellipse_preserves_focal_sum():
    length = np.nextafter(6.0, np.inf)
    locus = build_range_locus(*_request(ranges=(length,), index=0))
    np.testing.assert_allclose(_focal_sum(locus.surfaces[0][0], locus.foci), length, atol=2e-14)
    np.testing.assert_allclose(_focal_sum(locus.equator, locus.foci), length, atol=2e-14)
    assert np.all(np.isfinite(locus.equator))


def test_tiny_nonzero_baseline_is_not_treated_as_coincident():
    with pytest.raises(ValueError, match="coincident"):
        build_range_locus(*_request(mode="monostatic", tx=(0, 0, 0), rx=(1e-200, 0, 0)))


def test_conflicting_exported_edges_cannot_silently_override_each_other():
    request, selection = _request()
    request.display_frame["sensing"]["range_edges_m"] = [10, 15, 25, 40]
    request.display_frame["sensing"]["range_bin_edges_m"] = [10, 16, 25, 40]
    with pytest.raises(ValueError, match="edge sources disagree"):
        build_range_locus(request, selection)


def test_declared_but_missing_edges_cannot_fall_back_to_midpoints():
    request, selection = _request()
    _descriptor(request)["axes"][-1]["bin_edges_key"] = "missing"
    with pytest.raises(ValueError, match="edges.*unavailable"):
        build_range_locus(request, selection)


@pytest.mark.parametrize("coordinate_index, lengths", [(0, (8, 16)), (2, (27, 41))])
def test_endpoint_bins_use_local_nonuniform_display_spacing(coordinate_index, lengths):
    locus = build_range_locus(*_request(index=coordinate_index))
    for (vertices, _), length in zip(locus.surfaces, lengths, strict=True):
        np.testing.assert_allclose(_focal_sum(vertices, locus.foci), length)


def _replay(request):
    temporal = request.display_frame["sensing"]["temporal"]
    return SimpleNamespace(
        scene_binding=SimpleNamespace(bound_scene_frame_id=7, bound_scene_time_s=1.25),
        key=SimpleNamespace(
            clock_id=temporal["clock"]["clock_id"],
            acquisition_id=temporal["acquisition"]["acquisition_id"],
            pair_id=temporal["acquisition"]["pair_id"],
            processing_run_id=temporal["result"]["processing_run_id"],
            reference_time_s=1.25,
        ),
    )


def test_exact_replay_can_supply_bound_reference_epoch():
    request, selection = _request()
    request = replace(request, replay_event=_replay(request))
    del request.display_frame["timestamp_s"]
    assert len(build_range_locus(request, selection).surfaces) == 2


@pytest.mark.parametrize(
    "field", ["clock_id", "acquisition_id", "pair_id", "processing_run_id", "reference_time_s"]
)
def test_replay_epoch_from_another_occurrence_cannot_validate_pose(field):
    request, selection = _request()
    event = _replay(request)
    setattr(event.key, field, "other")
    request = replace(request, replay_event=event)
    with pytest.raises(ValueError, match="Replay occurrence"):
        build_range_locus(request, selection)


def test_different_timestamp_representations_cannot_hide_a_stale_pose():
    request, selection = _request()
    request.display_frame["_source"] = {"timestamp": 2.5}
    with pytest.raises(ValueError, match="pose epoch"):
        build_range_locus(request, selection)


def test_source_timestamp_and_nanosecond_epoch_are_supported_without_guessing_units():
    request, selection = _request()
    del request.display_frame["timestamp_s"]
    request.display_frame["timestamp_ns"] = 1_250_000_000
    request.display_frame["_source"] = {"timestamp": 1.25}
    assert len(build_range_locus(request, selection).surfaces) == 2
