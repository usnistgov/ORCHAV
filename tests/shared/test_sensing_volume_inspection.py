"""Independent exact references for compact inspection; no Qt or ray tracer."""

import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from shared.sensing.volume.evaluation import load_evaluation
from shared.sensing.volume.inspection import (
    PairedVolume,
    category_counts,
    cell_support,
    context_snapshot,
    detail_grid,
    extras_grid,
    fixed_sensor_orientation,
    focus_mask,
    load_bookmark,
    report_details,
    save_bookmark,
    sensor_forward,
    slice_data,
    validate_context,
    zero_radial_polygon,
)
from shared.sensing.volume.reader import SensingVolume


def write_fixture(directory, condition="wall"):
    directory.mkdir(parents=True, exist_ok=True)
    scene = directory / "scenario.yaml"
    scene.write_text("actors: {rx: []}\n")
    trials = []
    for i in range(6):
        position = [10 + i * 2, 20, 30]
        row = dict(
            id=f"t{i}",
            index_zyx=[0, 0, i],
            heading=0,
            position_m=position,
            velocity_m_s=[1, 0, 0],
            status="pending",
        )
        if i < 4:
            locations = (
                [position]
                if i == 0
                else (
                    [position, [0, 0, -5], [0, 0, -5]] if i == 1 else [[0, 0, -5]] if i == 2 else []
                )
            )
            reports = [
                dict(
                    detection_id=str(j),
                    position_m=p,
                    range_m=float(np.linalg.norm(p)),
                    closing_velocity_m_s=0.0,
                )
                for j, p in enumerate(locations)
            ]
            row.update(
                status="measured",
                tp=int(i < 2),
                fp=len(reports) - int(i < 2),
                fn=int(i >= 2),
                hit=i < 2,
                matched_error_m=0 if i < 2 else None,
                radial_closing_velocity_m_s=-position[0] / np.linalg.norm(position),
                native=f"native/{i}",
                source_hashes={"f.h5": "provenance"},
                configuration_sha256="provenance",
                reports=reports,
                native_evidence="not_included",
                direct_path_blocked=i == 2,
                blocked_target_probes=9 if i == 2 else 0,
                target_probe_count=9,
            )
        elif i == 5:
            row["status"] = "excluded_geometry"
        trials.append(row)
    data = dict(
        schema_version=1,
        kind="conditional_detection_outcomes",
        axes_m={"x": list(range(10, 22, 2)), "y": [20, 23], "z": [30, 34]},
        world_frame=dict(
            id="translated",
            origin="declared",
            axes=["x", "y", "z"],
            handedness="right",
            length_unit="m",
        ),
        headings=[dict(name="+X", velocity_m_s=[1, 0, 0])],
        sensor_position_m=[0, 0, 0],
        position_gate_m=1,
        scene_scenario="scenario.yaml",
        scene_configuration_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(),
        trials=trials,
        condition=condition,
        comparison_volume="../" + ("open" if condition == "wall" else "wall"),
        wall_bounds_m=[[-0.1, -2, 0], [0.1, 2, 4]],
        target_label="cube",
        cpi_samples=64,
        inspection_target={"name": "cube"},
    )
    (directory / "sensing_volume.json").write_text(json.dumps(data))
    return SensingVolume(directory)


def test_categories_partition_measured_and_conserve_every_report(tmp_path):
    reader = write_fixture(tmp_path / "wall")
    model = load_evaluation(reader)
    before = deepcopy(model.rows)
    result = model.evaluate(0)
    values = detail_grid(result)
    np.testing.assert_equal(values[0, 0], [0, 1, 2, 3, np.nan, np.nan])
    assert list(category_counts(values).values()) == [1, 1, 1, 1]
    np.testing.assert_equal(extras_grid(result)[0, 0], [0, 2, 1, 0, np.nan, np.nan])
    measured = np.isfinite(result["outcomes"])
    np.testing.assert_array_equal(
        (result["outcomes"] + extras_grid(result))[measured], result["report_counts"][measured]
    )
    assert result["tp"] + result["fp"] == 5 and result["tp"] + result["fn"] == 4
    broad = model.evaluate(0, position_gate_m=100)
    np.testing.assert_equal(result["nearest_error_m"], broad["nearest_error_m"])
    np.testing.assert_equal(result["report_counts"], broad["report_counts"])
    assert model.rows == before


def test_matched_report_can_be_outside_its_sample_cell(tmp_path):
    reader = write_fixture(tmp_path / "wall")
    trial = reader.cells[(0, 0, 0, 0)]
    trial["reports"][0]["position_m"] = [11.1, 20, 30]
    trial["matched_error_m"] = 1.1
    reader.metadata["position_gate_m"] = 1.2
    from shared.sensing.volume.evaluation import VolumeEvaluation

    model = VolumeEvaluation(
        reader, {t["id"]: t["reports"] for t in reader.cells.values() if t["status"] == "measured"}
    )
    rows = report_details(model, (0, 0, 0, 0), model.evaluate(0, position_gate_m=1.2))
    assert rows[0]["matched"] and not rows[0]["inside_sample_cell"]
    centre, spacing, lower, upper = cell_support(reader, (0, 0, 0, 0))
    np.testing.assert_array_equal(centre, [10, 20, 30])
    np.testing.assert_array_equal(spacing, [2, 3, 4])
    np.testing.assert_array_equal(lower, [9, 18.5, 28])
    np.testing.assert_array_equal(upper, [11, 21.5, 32])


@pytest.mark.parametrize(
    "predicate,indices",
    [
        ("all", [0, 1, 2, 3]),
        ("unmatched", [2]),
        ("extras", [1]),
        ("count", [1, 2]),
        ("error", [2]),
        ("blocked", [2]),
        ("all_blocked", [2]),
        ("clear", [3]),
    ],
)
def test_focus_has_exact_membership(tmp_path, predicate, indices):
    reader = write_fixture(tmp_path / "wall")
    snapshot = load_evaluation(reader).evaluate(0)
    mask = focus_mask(reader, 0, snapshot, predicate, threshold=10)
    assert np.flatnonzero(mask).tolist() == indices
    assert snapshot["tp"] == 2 and snapshot["fp"] == 3 and snapshot["fn"] == 2


@pytest.mark.parametrize(
    "orientation,index,axis,h,v,extent",
    [
        ("xy", 1, 0, "x", "y", (9, 21, 18.5, 24.5)),
        ("xz", 1, 1, "x", "z", (9, 21, 28, 36)),
        ("yz", 2, 2, "y", "z", (18.5, 24.5, 28, 36)),
    ],
)
def test_translated_asymmetric_slices(tmp_path, orientation, index, axis, h, v, extent):
    reader = write_fixture(tmp_path / "wall")
    values = np.arange(24).reshape(2, 2, 6)
    plane, actual_h, actual_v, _, actual_extent = slice_data(
        values, reader.axes, orientation, index
    )
    np.testing.assert_equal(plane, np.take(values, index, axis=axis))
    assert (actual_h, actual_v, actual_extent) == (h, v, extent)


def test_real_report_positions_and_missing_components_are_distinct(tmp_path):
    reader = write_fixture(tmp_path / "wall")
    model = load_evaluation(reader)
    result = model.evaluate(0)
    rows = report_details(model, (0, 0, 0, 1), result)
    assert [r["matched"] for r in rows] == [True, False, False]
    assert rows[1]["position_m"] == [0, 0, -5]
    assert rows[1]["error_m"] == pytest.approx(np.sqrt(12**2 + 20**2 + 35**2))
    assert len({r["scoped_id"] for r in rows}) == 3
    assert report_details(model, (0, 0, 0, 3), result) == []
    assert reader.native_availability((0, 0, 0, 3)).startswith("Native evidence not included")
    with pytest.raises(ValueError, match="not included"):
        reader.verify_native((0, 0, 0, 3))


def test_pair_uses_common_support_and_rejects_changed_contract(tmp_path):
    a = write_fixture(tmp_path / "wall")
    b = write_fixture(tmp_path / "open", "open")
    pair = PairedVolume(a, b)
    am, bm = load_evaluation(a), load_evaluation(b)
    result = pair.evaluate(am, bm, 0)
    assert result["common"] == 4 and list(result["counts"].values()) == [2, 0, 0, 2]

    # Independent B criterion result demonstrates A-only/B-only algebra.
    def altered(heading, **kwargs):
        r = bm.evaluate(heading, **kwargs)
        r["outcomes"] = r["outcomes"].copy()
        r["outcomes"][0, 0, 1] = 0
        r["outcomes"][0, 0, 2] = 1
        return r

    from types import SimpleNamespace

    changed = pair.evaluate(am, SimpleNamespace(evaluate=altered), 0)
    np.testing.assert_equal(changed["values"][0, 0, :4], [0, 1, 2, 3])
    a.metadata["sensor_position_m"] = [1, 0, 0]
    with pytest.raises(ValueError, match="sensor_position"):
        PairedVolume(a, b)
    a.metadata["sensor_position_m"] = [0, 0, 0]
    b.axes["x"] = b.axes["x"] + 0.01
    with pytest.raises(ValueError, match="exact grid"):
        PairedVolume(a, b)


def test_bookmark_relocation_and_stale_identity(tmp_path):
    import shutil

    source = tmp_path / "package"
    reader = write_fixture(source / "wall")
    state = context_snapshot(reader, 0, (0, 0, 0, 1), {"quantity": "details"}, selected_report="1")
    path = source / "view.json"
    save_bookmark(path, state, title="Extra reports")
    assert json.loads(path.read_text())["volume"] == "wall/sensing_volume.json"
    moved = tmp_path / "relocated"
    shutil.copytree(source, moved)
    loaded = load_bookmark(moved / "view.json")
    other = SensingVolume(loaded["volume"])
    assert validate_context(other, loaded) == (0, 0, 0, 1)
    loaded["volume_sha256"] = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        validate_context(other, loaded)


def test_fixed_receiver_orientation_uses_only_versioned_defaults():
    receiver = dict(name="RadarRX", mobility=dict(type="stationary", position_m=[-6, 0, 2]))
    config = dict(schema_version=2, actors=dict(rx=[receiver]))
    np.testing.assert_array_equal(fixed_sensor_orientation(config), [0, 0, 0])
    receiver["orientation"] = dict(type="fixed", yaw_deg=90, pitch_deg=30, roll_deg=15)
    np.testing.assert_allclose(fixed_sensor_orientation(config), np.deg2rad([90, 30, 15]))
    receiver["orientation"] = dict(type="align_motion")
    assert fixed_sensor_orientation(config) is None
    receiver.pop("orientation")
    config.pop("schema_version")
    assert fixed_sensor_orientation(config) is None


def test_zero_radial_plane_and_explicit_orientation(tmp_path):
    axes = {n: np.array([-4.0, 0.0, 4.0]) for n in ("x", "y", "z")}
    sensor = np.array([1.0, 2.0, -1.0])
    velocity = np.array([2.0, -1.0, 3.0])
    polygon = zero_radial_polygon(sensor, velocity, axes)
    assert len(polygon) >= 3
    np.testing.assert_allclose((polygon - sensor) @ velocity, 0, atol=1e-8)
    assert (abs(polygon) <= 6).all()
    assert zero_radial_polygon(sensor, [0, 0, 0], axes).shape == (0, 3)
    reader = write_fixture(tmp_path / "wall")
    assert sensor_forward(reader) is None
    reader.scene_path.write_text(
        "actors:\n  rx:\n    - orientation: {type: fixed, yaw_deg: 90, pitch_deg: 30}\n"
    )
    np.testing.assert_allclose(sensor_forward(reader), [0, np.sqrt(3) / 2, -0.5], atol=1e-12)


def test_portable_native_rebinds_catalog_and_manifest_without_numeric_changes(tmp_path):
    import h5py
    import yaml

    from scripts.package_volume_example import portable_native
    from shared.sensing.targets import build_target_catalog, validate_target_catalog

    source = tmp_path / "source"
    (source / "frames").mkdir(parents=True)
    assets = tmp_path / "assets"
    assets.mkdir()
    old_scene = "/mnt/private/wall.xml"
    old_cube = "/mnt/private/cube"
    cfg = {"scene": {"id": old_scene}, "actors": {"targets": [{"asset": {"path": old_cube}}]}}
    (source / "scenario.yaml").write_text(yaml.safe_dump(cfg))
    catalog = build_target_catalog(
        [
            {
                "target_id": "Cube",
                "display_name": "Cube",
                "identity_source": "scenario_actor_name",
                "asset": {"path": old_cube},
            }
        ]
    )
    (source / "frames/frames_manifest.json").write_text(json.dumps({"scene_id": old_scene}))
    with h5py.File(source / "frames/empty.h5", "w") as f:
        f.create_dataset("static/sensing_config_json", data="null", dtype=h5py.string_dtype())
    with h5py.File(source / "frames/final.h5", "w") as f:
        f.create_dataset(
            "static/sensing_config_json",
            data=json.dumps({"target_catalog": catalog}),
            dtype=h5py.string_dtype(),
        )
        f.create_dataset(
            "state",
            data=json.dumps({"catalog_revision": catalog["catalog_revision"]}),
            dtype=h5py.string_dtype(),
        )
        f.create_dataset("measurements", data=np.arange(10))
    before = hashlib.sha256((source / "frames/final.h5").read_bytes()).hexdigest()
    destination = tmp_path / "copy"
    changed = portable_native(source, destination, assets, "wall")
    assert "frames_manifest.json:asset_references" in changed
    assert "/mnt/" not in (destination / "frames/frames_manifest.json").read_text()
    with h5py.File(destination / "frames/final.h5") as f:
        copied = json.loads(f["static/sensing_config_json"][()].decode())["target_catalog"]
        assert not validate_target_catalog(copied)
        assert json.loads(f["state"][()].decode())["catalog_revision"] == copied["catalog_revision"]
        np.testing.assert_array_equal(f["measurements"][()], np.arange(10))
    assert hashlib.sha256((source / "frames/final.h5").read_bytes()).hexdigest() == before


@pytest.mark.parametrize("name", ["cell_edges", "unavailable_edges"])
def test_bookmark_cell_edge_flags_are_optional_booleans(tmp_path, name):
    reader = write_fixture(tmp_path / "wall")
    state = context_snapshot(reader, 0, (0, 0, 0, 1), {"quantity": "details"})
    assert validate_context(reader, state) == (0, 0, 0, 1)
    state["view"][name] = True
    assert validate_context(reader, state) == (0, 0, 0, 1)
    state["view"][name] = "true"
    with pytest.raises(ValueError, match=name):
        validate_context(reader, state)
