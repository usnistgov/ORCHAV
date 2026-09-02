"""Typed authoring metadata for mobility and orientation schema models.

The shared actor schema defines the accepted YAML discriminators. This module
describes the editor, preview, dependency, asset, and relationship requirements
for each discriminator without importing Qt or renderer objects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from pydantic import TypeAdapter

from shared.scenarios.actors import MobilitySpec, OrientationSpec


class ModelFamily(str, Enum):
    """Interaction family shared by mobility and orientation models."""

    FIXED = "fixed"
    CONTROL_POINT_PATH = "control_point_path"
    PARAMETRIC = "parametric"
    SPATIAL_SAMPLING = "spatial_sampling"
    STOCHASTIC = "stochastic"
    RELATIONAL = "relational"
    MAP_AWARE = "map_aware"
    RESOURCE_BACKED = "resource_backed"
    MOTION_DERIVED = "motion_derived"


class AuthoringSupport(str, Enum):
    """Whether the builder may edit and serialize a model."""

    EDITABLE = "editable"
    READ_ONLY_IMPORT = "read_only_import"


class RandomnessPolicy(str, Enum):
    """How a model obtains any random seed it consumes."""

    NONE = "none"
    REQUIRED_SEED = "required_seed"
    OPTIONAL_SEED = "optional_seed"
    INHERITED = "inherited"


class PreviewPolicy(str, Enum):
    """Preparation required before a canonical preview is available."""

    GENERATOR_PREPARED = "generator_prepared"
    AFTER_REFERENCE_RESOLUTION = "after_reference_resolution"
    AFTER_ASSET_RESOLUTION = "after_asset_resolution"


class EditAffordance(str, Enum):
    """Reusable controls from which form and viewport editors are composed."""

    POSITION_HANDLE = "position_handle"
    ENDPOINT_HANDLE = "endpoint_handle"
    WAYPOINT_HANDLES = "waypoint_handles"
    CENTER_HANDLE = "center_handle"
    RADIUS_HANDLE = "radius_handle"
    START_ANGLE_HANDLE = "start_angle_handle"
    BOUNDS_BOX = "bounds_box"
    GRID_PARAMETERS = "grid_parameters"
    PARAMETER_FORM = "parameter_form"
    ACTOR_REFERENCE_PICKER = "actor_reference_picker"
    POINT_TARGET_EDITOR = "point_target_editor"
    GROUP_REFERENCE_PICKER = "group_reference_picker"
    LOCAL_OFFSET_HANDLE = "local_offset_handle"
    MAP_ROUTE_PARAMETERS = "map_route_parameters"
    RESOURCE_PATH_PICKER = "resource_path_picker"
    ANGLE_TRIPLET = "angle_triplet"
    ANGLE_KEYFRAMES = "angle_keyframes"
    ORIENTATION_OFFSETS = "orientation_offsets"
    ROTATION_GIZMO = "rotation_gizmo"
    ANGULAR_SWEEP = "angular_sweep"
    ANGLE_BOUNDS = "angle_bounds"


class RuntimeDependency(str, Enum):
    """Optional Python packages required by a model evaluator."""

    NETWORKX = "networkx"


class AssetRequirement(str, Enum):
    """Files outside the scenario mapping consumed during preparation."""

    CACHED_NETWORK_GRAPH = "cached_network_graph"
    POSITION_SEQUENCE = "position_sequence"


class ContextRequirement(str, Enum):
    """Scenario relationships or resolution context needed by a model."""

    OWNER_MOBILITY = "owner_mobility"
    TARGET_ACTOR = "target_actor"
    GROUP = "group"


@dataclass(frozen=True, slots=True)
class ModelCapability:
    """Immutable metadata for one schema model discriminator."""

    yaml_type: str
    family: ModelFamily
    authoring_support: AuthoringSupport
    randomness: RandomnessPolicy
    preview: PreviewPolicy
    edit_affordances: frozenset[EditAffordance]
    direct_viewport_affordances: frozenset[EditAffordance] = frozenset()
    supports_whole_path_translation: bool = False
    required_dependencies: frozenset[RuntimeDependency] = frozenset()
    optional_dependencies: frozenset[RuntimeDependency] = frozenset()
    required_assets: frozenset[AssetRequirement] = frozenset()
    optional_assets: frozenset[AssetRequirement] = frozenset()
    required_context: frozenset[ContextRequirement] = frozenset()
    optional_context: frozenset[ContextRequirement] = frozenset()
    editor_variants: tuple[str, ...] = ()

    @property
    def editable(self) -> bool:
        """Return whether the authoring workspace owns this model."""

        return self.authoring_support is AuthoringSupport.EDITABLE


def _capability(
    yaml_type: str,
    family: ModelFamily,
    edit_affordances: tuple[EditAffordance, ...],
    *,
    authoring_support: AuthoringSupport = AuthoringSupport.EDITABLE,
    randomness: RandomnessPolicy = RandomnessPolicy.NONE,
    preview: PreviewPolicy = PreviewPolicy.GENERATOR_PREPARED,
    required_dependencies: tuple[RuntimeDependency, ...] = (),
    optional_dependencies: tuple[RuntimeDependency, ...] = (),
    required_assets: tuple[AssetRequirement, ...] = (),
    optional_assets: tuple[AssetRequirement, ...] = (),
    required_context: tuple[ContextRequirement, ...] = (),
    optional_context: tuple[ContextRequirement, ...] = (),
    editor_variants: tuple[str, ...] = (),
    direct_viewport_affordances: tuple[EditAffordance, ...] = (),
    supports_whole_path_translation: bool = False,
) -> ModelCapability:
    """Build one immutable manifest entry from readable tuple declarations."""

    return ModelCapability(
        yaml_type=yaml_type,
        family=family,
        authoring_support=authoring_support,
        randomness=randomness,
        preview=preview,
        edit_affordances=frozenset(edit_affordances),
        direct_viewport_affordances=frozenset(direct_viewport_affordances),
        supports_whole_path_translation=supports_whole_path_translation,
        required_dependencies=frozenset(required_dependencies),
        optional_dependencies=frozenset(optional_dependencies),
        required_assets=frozenset(required_assets),
        optional_assets=frozenset(optional_assets),
        required_context=frozenset(required_context),
        optional_context=frozenset(optional_context),
        editor_variants=editor_variants,
    )


MOBILITY_CAPABILITIES: Mapping[str, ModelCapability] = MappingProxyType(
    {
        "stationary": _capability(
            "stationary",
            ModelFamily.FIXED,
            (EditAffordance.POSITION_HANDLE,),
            authoring_support=AuthoringSupport.EDITABLE,
            direct_viewport_affordances=(EditAffordance.POSITION_HANDLE,),
            supports_whole_path_translation=True,
        ),
        "linear": _capability(
            "linear",
            ModelFamily.CONTROL_POINT_PATH,
            (EditAffordance.POSITION_HANDLE, EditAffordance.ENDPOINT_HANDLE),
            authoring_support=AuthoringSupport.EDITABLE,
            direct_viewport_affordances=(
                EditAffordance.POSITION_HANDLE,
                EditAffordance.ENDPOINT_HANDLE,
            ),
            supports_whole_path_translation=True,
        ),
        "waypoint": _capability(
            "waypoint",
            ModelFamily.CONTROL_POINT_PATH,
            (EditAffordance.WAYPOINT_HANDLES,),
            authoring_support=AuthoringSupport.EDITABLE,
            direct_viewport_affordances=(EditAffordance.WAYPOINT_HANDLES,),
            supports_whole_path_translation=True,
        ),
        "circular": _capability(
            "circular",
            ModelFamily.PARAMETRIC,
            (
                EditAffordance.CENTER_HANDLE,
                EditAffordance.RADIUS_HANDLE,
                EditAffordance.START_ANGLE_HANDLE,
                EditAffordance.PARAMETER_FORM,
            ),
            authoring_support=AuthoringSupport.EDITABLE,
            direct_viewport_affordances=(
                EditAffordance.CENTER_HANDLE,
                EditAffordance.RADIUS_HANDLE,
                EditAffordance.START_ANGLE_HANDLE,
            ),
            supports_whole_path_translation=True,
        ),
        "survey": _capability(
            "survey",
            ModelFamily.CONTROL_POINT_PATH,
            (
                EditAffordance.BOUNDS_BOX,
                EditAffordance.GRID_PARAMETERS,
                EditAffordance.PARAMETER_FORM,
            ),
            supports_whole_path_translation=True,
        ),
        "grid_scan": _capability(
            "grid_scan",
            ModelFamily.SPATIAL_SAMPLING,
            (
                EditAffordance.BOUNDS_BOX,
                EditAffordance.GRID_PARAMETERS,
                EditAffordance.PARAMETER_FORM,
            ),
            editor_variants=("snake", "raster"),
            supports_whole_path_translation=True,
        ),
        "oscillating": _capability(
            "oscillating",
            ModelFamily.PARAMETRIC,
            (EditAffordance.CENTER_HANDLE, EditAffordance.PARAMETER_FORM),
            supports_whole_path_translation=True,
        ),
        "pendulum": _capability(
            "pendulum",
            ModelFamily.PARAMETRIC,
            (EditAffordance.POSITION_HANDLE, EditAffordance.PARAMETER_FORM),
            editor_variants=("xy", "xz", "yz"),
            supports_whole_path_translation=True,
        ),
        "figure8": _capability(
            "figure8",
            ModelFamily.PARAMETRIC,
            (EditAffordance.CENTER_HANDLE, EditAffordance.PARAMETER_FORM),
            editor_variants=("xy", "xz", "yz"),
            supports_whole_path_translation=True,
        ),
        "spiral": _capability(
            "spiral",
            ModelFamily.PARAMETRIC,
            (
                EditAffordance.CENTER_HANDLE,
                EditAffordance.RADIUS_HANDLE,
                EditAffordance.PARAMETER_FORM,
            ),
            supports_whole_path_translation=True,
        ),
        "random_sampling": _capability(
            "random_sampling",
            ModelFamily.SPATIAL_SAMPLING,
            (EditAffordance.BOUNDS_BOX, EditAffordance.PARAMETER_FORM),
            randomness=RandomnessPolicy.REQUIRED_SEED,
            editor_variants=("uniform", "poisson_disk"),
            supports_whole_path_translation=True,
        ),
        "sampled": _capability(
            "sampled",
            ModelFamily.SPATIAL_SAMPLING,
            (),
            authoring_support=AuthoringSupport.READ_ONLY_IMPORT,
        ),
        "gauss_markov": _capability(
            "gauss_markov",
            ModelFamily.STOCHASTIC,
            (
                EditAffordance.POSITION_HANDLE,
                EditAffordance.BOUNDS_BOX,
                EditAffordance.PARAMETER_FORM,
            ),
            randomness=RandomnessPolicy.REQUIRED_SEED,
            supports_whole_path_translation=True,
        ),
        "random_waypoint": _capability(
            "random_waypoint",
            ModelFamily.STOCHASTIC,
            (
                EditAffordance.POSITION_HANDLE,
                EditAffordance.BOUNDS_BOX,
                EditAffordance.PARAMETER_FORM,
            ),
            randomness=RandomnessPolicy.REQUIRED_SEED,
            supports_whole_path_translation=True,
        ),
        "manhattan_grid": _capability(
            "manhattan_grid",
            ModelFamily.STOCHASTIC,
            (
                EditAffordance.POSITION_HANDLE,
                EditAffordance.GRID_PARAMETERS,
                EditAffordance.PARAMETER_FORM,
            ),
            randomness=RandomnessPolicy.REQUIRED_SEED,
            supports_whole_path_translation=True,
        ),
        "network_route": _capability(
            "network_route",
            ModelFamily.MAP_AWARE,
            (EditAffordance.MAP_ROUTE_PARAMETERS, EditAffordance.PARAMETER_FORM),
            randomness=RandomnessPolicy.OPTIONAL_SEED,
            preview=PreviewPolicy.AFTER_ASSET_RESOLUTION,
            required_dependencies=(RuntimeDependency.NETWORKX,),
            required_assets=(AssetRequirement.CACHED_NETWORK_GRAPH,),
            editor_variants=("shortest_path", "random_walk"),
        ),
        "mesh_sequence": _capability(
            "mesh_sequence",
            ModelFamily.RESOURCE_BACKED,
            (EditAffordance.RESOURCE_PATH_PICKER, EditAffordance.PARAMETER_FORM),
            preview=PreviewPolicy.AFTER_ASSET_RESOLUTION,
            required_assets=(AssetRequirement.POSITION_SEQUENCE,),
            editor_variants=("linear", "step"),
        ),
        "group_member": _capability(
            "group_member",
            ModelFamily.RELATIONAL,
            (
                EditAffordance.GROUP_REFERENCE_PICKER,
                EditAffordance.LOCAL_OFFSET_HANDLE,
            ),
            randomness=RandomnessPolicy.INHERITED,
            preview=PreviewPolicy.AFTER_REFERENCE_RESOLUTION,
            required_context=(ContextRequirement.GROUP,),
            supports_whole_path_translation=True,
        ),
    }
)


ORIENTATION_CAPABILITIES: Mapping[str, ModelCapability] = MappingProxyType(
    {
        "fixed": _capability(
            "fixed",
            ModelFamily.FIXED,
            (
                EditAffordance.ANGLE_TRIPLET,
                EditAffordance.ROTATION_GIZMO,
            ),
            authoring_support=AuthoringSupport.EDITABLE,
            direct_viewport_affordances=(EditAffordance.ROTATION_GIZMO,),
        ),
        "keyframes": _capability(
            "keyframes",
            ModelFamily.CONTROL_POINT_PATH,
            (EditAffordance.ANGLE_KEYFRAMES,),
            authoring_support=AuthoringSupport.EDITABLE,
        ),
        "align_motion": _capability(
            "align_motion",
            ModelFamily.MOTION_DERIVED,
            (EditAffordance.ORIENTATION_OFFSETS,),
            authoring_support=AuthoringSupport.EDITABLE,
            randomness=RandomnessPolicy.INHERITED,
            preview=PreviewPolicy.AFTER_REFERENCE_RESOLUTION,
            required_context=(ContextRequirement.OWNER_MOBILITY,),
        ),
        "look_at": _capability(
            "look_at",
            ModelFamily.RELATIONAL,
            (
                EditAffordance.ACTOR_REFERENCE_PICKER,
                EditAffordance.POINT_TARGET_EDITOR,
                EditAffordance.ORIENTATION_OFFSETS,
            ),
            authoring_support=AuthoringSupport.EDITABLE,
            randomness=RandomnessPolicy.INHERITED,
            preview=PreviewPolicy.AFTER_REFERENCE_RESOLUTION,
            required_context=(ContextRequirement.OWNER_MOBILITY,),
            optional_context=(ContextRequirement.TARGET_ACTOR,),
        ),
        "spin": _capability(
            "spin",
            ModelFamily.PARAMETRIC,
            (EditAffordance.ANGULAR_SWEEP,),
            authoring_support=AuthoringSupport.EDITABLE,
        ),
        "random": _capability(
            "random",
            ModelFamily.STOCHASTIC,
            (EditAffordance.ANGLE_BOUNDS, EditAffordance.PARAMETER_FORM),
            randomness=RandomnessPolicy.REQUIRED_SEED,
        ),
    }
)


def _schema_discriminators(annotation: Any) -> frozenset[str]:
    """Return discriminator values exposed by one shared schema union."""

    discriminator = TypeAdapter(annotation).json_schema().get("discriminator")
    mapping = discriminator.get("mapping") if isinstance(discriminator, dict) else None
    if not isinstance(mapping, dict) or not all(isinstance(key, str) for key in mapping):
        raise RuntimeError("shared actor schema union has no string discriminator mapping")
    return frozenset(mapping)


def _require_complete_manifest(
    label: str,
    manifest: Mapping[str, ModelCapability],
    schema_types: frozenset[str],
) -> None:
    manifest_types = frozenset(manifest)
    if manifest_types != schema_types:
        missing = sorted(schema_types - manifest_types)
        unknown = sorted(manifest_types - schema_types)
        raise RuntimeError(
            f"{label} capability manifest does not match the shared actor schema "
            f"(missing={missing}, unknown={unknown})"
        )


_require_complete_manifest(
    "mobility",
    MOBILITY_CAPABILITIES,
    _schema_discriminators(MobilitySpec),
)
_require_complete_manifest(
    "orientation",
    ORIENTATION_CAPABILITIES,
    _schema_discriminators(OrientationSpec),
)


SUPPORTED_AUTHORING_MOBILITY_TYPES = frozenset(
    yaml_type for yaml_type, capability in MOBILITY_CAPABILITIES.items() if capability.editable
)
SUPPORTED_AUTHORING_ORIENTATION_TYPES = frozenset(
    yaml_type for yaml_type, capability in ORIENTATION_CAPABILITIES.items() if capability.editable
)


def mobility_capability(yaml_type: str) -> ModelCapability:
    """Return capability metadata for one mobility schema discriminator."""

    return MOBILITY_CAPABILITIES[yaml_type]


def orientation_capability(yaml_type: str) -> ModelCapability:
    """Return capability metadata for one orientation schema discriminator."""

    return ORIENTATION_CAPABILITIES[yaml_type]
