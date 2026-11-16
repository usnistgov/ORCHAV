"""Native sensing schema validation, independent of Mitsuba and Sionna."""

import copy

import pytest
from pydantic import ValidationError

from shared.scenarios.actors import (
    ConstantRCSSpec,
    NativeBoxGeometrySpec,
    NativeMeshGeometrySpec,
    TargetActorSpec,
    TR38901RCSSpec,
)


def _native_actor():
    return {
        "name": "target",
        "mobility": {"type": "stationary", "position_m": [0, 0, 1]},
        "representation": "constant_rcs",
        "geometry": {"type": "box", "length_m": 1, "width_m": 2, "height_m": 3},
        "constant_rcs": {"sigma_m2": 0.25},
    }


def test_existing_asset_defaults_to_mesh_material():
    spec = TargetActorSpec.model_validate(
        {
            "name": "mesh",
            "mobility": {"type": "stationary", "position_m": [0, 0, 0]},
            "asset": {"source": "file", "path": "target.ply"},
        }
    )
    assert spec.representation == "mesh_material"
    assert spec.asset.material_type == "glass"
    assert spec.geometry is None
    assert TargetActorSpec.model_validate(spec.model_dump()) == spec


def test_native_round_trip_preserves_pose_and_no_extra_asset():
    data = _native_actor()
    data["orientation"] = {"type": "fixed", "yaw_deg": 27, "pitch_deg": -12}
    spec = TargetActorSpec.model_validate(data)
    assert TargetActorSpec.model_validate_json(spec.model_dump_json()) == spec
    assert spec.asset is None
    assert spec.constant_rcs.sigma_m2 == 0.25
    assert spec.constant_rcs.xpr_db is None
    assert spec.orientation.yaw_deg == 27


@pytest.mark.parametrize(
    "changes",
    [
        {"asset": {"source": "file", "path": "target.ply"}},
        {"geometry": None},
        {"constant_rcs": None},
        {"tr38901": {"object_type": "human"}},
        {"representation": "mesh_material"},
        {"representation": "unknown"},
        {"geometry": {"type": "animated_asset", "path": "manifest.json"}},
        {"geometry": {"type": "box", "switch_meshes": True}},
        {"geometry": {"type": "box", "material_type": "metal"}},
    ],
)
def test_native_rejects_missing_or_conflicting_ownership(changes):
    data = _native_actor()
    data.update(copy.deepcopy(changes))
    with pytest.raises(ValidationError):
        TargetActorSpec.model_validate(data)


@pytest.mark.parametrize("sigma", [-1, float("nan"), float("inf"), True])
def test_constant_rcs_rejects_invalid_linear_sigma(sigma):
    with pytest.raises(ValidationError):
        ConstantRCSSpec(sigma_m2=sigma)


def test_zero_sigma_and_finite_xpr_are_supported():
    assert ConstantRCSSpec(sigma_m2=0, xpr_db=-3).sigma_m2 == 0
    with pytest.raises(ValidationError):
        ConstantRCSSpec(sigma_m2=1, sigma_dbsm=0)
    with pytest.raises(ValidationError):
        ConstantRCSSpec(sigma_m2=1, xpr_db=float("nan"))


@pytest.mark.parametrize("dimension", [0, -1, float("nan"), float("inf"), True])
def test_native_geometry_dimensions_are_positive_finite(dimension):
    with pytest.raises(ValidationError):
        NativeBoxGeometrySpec(length_m=dimension)


@pytest.mark.parametrize("path", ["", "frames/*.ply", "target.glb", "target[1].obj"])
def test_mesh_geometry_is_one_supported_static_file(path):
    with pytest.raises(ValidationError):
        NativeMeshGeometrySpec(path=path)


@pytest.mark.parametrize(
    "object_type,model_type,count",
    [
        ("uav-small-size", 1, 1),
        ("human", 1, 1),
        ("human", 2, 1),
        ("uav-large-size", 2, 1),
        ("vehicle-single-sp", 2, 1),
        ("vehicle-multi-sp", 2, 5),
        ("agv-single-sp", 2, 1),
        ("agv-multi-sp", 2, 5),
    ],
)
def test_presets_have_valid_models_and_point_counts(object_type, model_type, count):
    spec = TR38901RCSSpec(object_type=object_type, model_type=model_type)
    assert spec.num_scattering_points == count
    assert not spec.random_sigma_s and not spec.random_xpr and not spec.random_phases


@pytest.mark.parametrize(
    "parameters",
    [
        {"object_type": "vehicle"},
        {"object_type": "agv"},
        {"object_type": "uav-small-size", "model_type": 2},
        {"object_type": "uav-large-size", "model_type": 1},
        {"object_type": "vehicle-multi-sp", "model_type": 1},
        {"object_type": "human", "model_type": True},
        {"object_type": "human", "model_type": 1.0},
        {"object_type": "human", "seed": 12},
        {"object_type": "human", "random_phases": "yes"},
        {"object_type": "human", "num_scattering_points": 6},
    ],
)
def test_invalid_presets_and_unsupported_overrides_are_rejected(parameters):
    with pytest.raises(ValidationError):
        TR38901RCSSpec.model_validate(parameters)
