"""Shared-schema and authoring capability-manifest parity tests."""

from typing import Any

from pydantic import TypeAdapter

from shared.scenarios.actors import MobilitySpec, OrientationSpec
from visualizer.src.authoring.mobility_control_rig import (
    MOBILITY_CONTROL_RIG_ADAPTERS,
)
from visualizer.src.authoring.model_capabilities import (
    MOBILITY_CAPABILITIES,
    ORIENTATION_CAPABILITIES,
    SUPPORTED_AUTHORING_MOBILITY_TYPES,
    SUPPORTED_AUTHORING_ORIENTATION_TYPES,
    AssetRequirement,
    AuthoringSupport,
    ContextRequirement,
    EditAffordance,
    PreviewPolicy,
    RandomnessPolicy,
    RuntimeDependency,
    mobility_capability,
    orientation_capability,
)


def _schema_discriminators(annotation: Any) -> frozenset[str]:
    discriminator = TypeAdapter(annotation).json_schema()["discriminator"]
    return frozenset(discriminator["mapping"])


def test_capability_manifests_exactly_match_shared_actor_schema() -> None:
    expected_mobility = {
        "stationary",
        "linear",
        "waypoint",
        "sampled",
        "circular",
        "survey",
        "grid_scan",
        "oscillating",
        "pendulum",
        "figure8",
        "spiral",
        "random_sampling",
        "gauss_markov",
        "random_waypoint",
        "manhattan_grid",
        "network_route",
        "mesh_sequence",
        "group_member",
    }
    expected_orientation = {
        "fixed",
        "keyframes",
        "align_motion",
        "look_at",
        "spin",
        "random",
    }

    assert set(MOBILITY_CAPABILITIES) == expected_mobility
    assert set(ORIENTATION_CAPABILITIES) == expected_orientation
    assert set(MOBILITY_CAPABILITIES) == _schema_discriminators(MobilitySpec)
    assert set(ORIENTATION_CAPABILITIES) == _schema_discriminators(OrientationSpec)

    for manifest in (MOBILITY_CAPABILITIES, ORIENTATION_CAPABILITIES):
        for yaml_type, capability in manifest.items():
            assert capability.yaml_type == yaml_type
            if capability.editable:
                assert capability.edit_affordances


def test_sampled_mobility_is_schema_known_but_deliberately_read_only() -> None:
    schema_mobility_types = _schema_discriminators(MobilitySpec)

    assert SUPPORTED_AUTHORING_MOBILITY_TYPES == schema_mobility_types - {"sampled"}
    assert SUPPORTED_AUTHORING_ORIENTATION_TYPES == _schema_discriminators(OrientationSpec)
    assert {
        yaml_type
        for yaml_type, capability in MOBILITY_CAPABILITIES.items()
        if not capability.editable
    } == {"sampled"}
    assert all(capability.editable for capability in ORIENTATION_CAPABILITIES.values())

    sampled = mobility_capability("sampled")
    assert sampled.authoring_support is AuthoringSupport.READ_ONLY_IMPORT
    assert sampled.edit_affordances == frozenset()
    assert sampled.direct_viewport_affordances == frozenset()
    assert not sampled.supports_whole_path_translation


def test_randomness_contract_matches_model_seed_inputs() -> None:
    for yaml_type in ("random_sampling", "gauss_markov", "random_waypoint", "manhattan_grid"):
        assert mobility_capability(yaml_type).randomness is RandomnessPolicy.REQUIRED_SEED

    assert mobility_capability("network_route").randomness is RandomnessPolicy.OPTIONAL_SEED
    assert mobility_capability("group_member").randomness is RandomnessPolicy.INHERITED
    assert orientation_capability("random").randomness is RandomnessPolicy.REQUIRED_SEED
    for yaml_type in ("align_motion", "look_at"):
        assert orientation_capability(yaml_type).randomness is RandomnessPolicy.INHERITED


def test_resource_models_declare_runtime_and_asset_requirements() -> None:
    network_route = mobility_capability("network_route")
    assert network_route.preview is PreviewPolicy.AFTER_ASSET_RESOLUTION
    assert network_route.required_dependencies == {RuntimeDependency.NETWORKX}
    assert network_route.required_assets == {AssetRequirement.CACHED_NETWORK_GRAPH}

    mesh_sequence = mobility_capability("mesh_sequence")
    assert mesh_sequence.preview is PreviewPolicy.AFTER_ASSET_RESOLUTION
    assert mesh_sequence.required_assets == {AssetRequirement.POSITION_SEQUENCE}


def test_viewport_handles_and_whole_path_translation_are_declared_separately() -> None:
    viewport_models = {
        yaml_type
        for yaml_type, capability in MOBILITY_CAPABILITIES.items()
        if capability.direct_viewport_affordances
    }
    assert viewport_models == {"stationary", "linear", "waypoint", "circular"}
    assert mobility_capability("stationary").direct_viewport_affordances == {
        EditAffordance.POSITION_HANDLE
    }
    assert mobility_capability("linear").direct_viewport_affordances == {
        EditAffordance.POSITION_HANDLE,
        EditAffordance.ENDPOINT_HANDLE,
    }
    assert mobility_capability("waypoint").direct_viewport_affordances == {
        EditAffordance.WAYPOINT_HANDLES
    }
    assert mobility_capability("circular").direct_viewport_affordances == {
        EditAffordance.CENTER_HANDLE,
        EditAffordance.RADIUS_HANDLE,
        EditAffordance.START_ANGLE_HANDLE,
    }

    without_whole_path_translation = {
        yaml_type
        for yaml_type, capability in MOBILITY_CAPABILITIES.items()
        if not capability.supports_whole_path_translation
    }
    assert without_whole_path_translation == {
        "network_route",
        "mesh_sequence",
        "sampled",
    }
    assert all(
        not capability.direct_viewport_affordances
        and not capability.supports_whole_path_translation
        for yaml_type, capability in ORIENTATION_CAPABILITIES.items()
        if yaml_type != "fixed"
    )
    assert orientation_capability("fixed").direct_viewport_affordances == {
        EditAffordance.ROTATION_GIZMO
    }


def test_control_rig_registry_matches_direct_viewport_capabilities() -> None:
    registered_discriminators = {
        adapter.mobility_type.model_fields["type"].default
        for adapter in MOBILITY_CONTROL_RIG_ADAPTERS.values()
    }
    declared_discriminators = {
        yaml_type
        for yaml_type, capability in MOBILITY_CAPABILITIES.items()
        if capability.direct_viewport_affordances
    }

    assert registered_discriminators == declared_discriminators


def test_relational_models_declare_reference_requirements() -> None:
    group_member = mobility_capability("group_member")
    assert group_member.required_context == {ContextRequirement.GROUP}
    assert group_member.preview is PreviewPolicy.AFTER_REFERENCE_RESOLUTION

    align_motion = orientation_capability("align_motion")
    assert align_motion.required_context == {ContextRequirement.OWNER_MOBILITY}
    assert align_motion.preview is PreviewPolicy.AFTER_REFERENCE_RESOLUTION

    look_at = orientation_capability("look_at")
    assert look_at.required_context == {ContextRequirement.OWNER_MOBILITY}
    assert look_at.optional_context == {ContextRequirement.TARGET_ACTOR}
    assert EditAffordance.ACTOR_REFERENCE_PICKER in look_at.edit_affordances
    assert EditAffordance.POINT_TARGET_EDITOR in look_at.edit_affordances
    assert look_at.preview is PreviewPolicy.AFTER_REFERENCE_RESOLUTION
