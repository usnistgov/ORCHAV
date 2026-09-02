"""YAML loading and schema validation for scenario configuration.

This module is the first trust boundary for ``scenario.yaml``. It parses YAML,
checks the schema version, validates the data against
``shared.scenarios.model.ScenarioModel``, and turns Pydantic validation errors
into messages that point to likely typo fixes.
"""

import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from pydantic import ValidationError

from shared.logging import get_logger
from shared.scenarios.actors import SCENARIO_SCHEMA_VERSION
from shared.scenarios.model import ScenarioModel

from .extensions import (
    registered_extension_keys,
    strip_registered_extensions,
    strip_registered_scene_source_sections,
    validate_registered_extensions,
    validate_registered_scene_source,
)

logger = get_logger(__name__)


def load_scenario_yaml(scenario_path: Path) -> Dict[str, Any]:
    """Load scenario configuration from YAML file.

    Args:
        scenario_path: Path to scenario folder or .yaml file.

    Returns:
        Dictionary containing scenario configuration.

    Raises:
        FileNotFoundError: If scenario file not found.
        yaml.YAMLError: If YAML parsing fails.
    """
    if scenario_path.is_dir():
        yaml_path = scenario_path / "scenario.yaml"
    else:
        yaml_path = scenario_path

    if not yaml_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Scenario file must contain a YAML mapping: {yaml_path}")

    _require_supported_schema_version(data)

    return data


def _require_supported_schema_version(data: Dict[str, Any]) -> None:
    """Reject missing or unsupported scenario schema versions."""
    actual = data.get("schema_version")
    if actual != SCENARIO_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported scenario schema version: "
            f"{actual!r}; expected {SCENARIO_SCHEMA_VERSION}."
        )


def _get_valid_keys_for_loc(loc: tuple) -> Optional[Set[str]]:
    """Return the set of valid field names at a given Pydantic error location.

    Walks the ScenarioModel hierarchy following *loc* (skipping integer indices)
    and returns the field names of the model at that position.

    Args:
        loc: Error location tuple from a Pydantic ValidationError, e.g.
             ``("raytracing", "quality", "typo_key")``.

    Returns:
        Set of valid field names for the parent model, or ``None`` if the
        location cannot be resolved.
    """
    from shared.scenarios import model as sm

    model_cls = sm.ScenarioModel
    # Walk up to the *parent* of the bad key (everything except the last element).
    for part in loc[:-1]:
        if isinstance(part, int):
            # List index — skip, the element type stays the same.
            continue
        field_info = model_cls.model_fields.get(str(part))
        if field_info is None:
            return None
        annotation = field_info.annotation
        # Unwrap Optional / Union generics to find the concrete model.
        inner = _unwrap_annotation(annotation)
        if inner is not None and _is_pydantic_model(inner):
            model_cls = inner
        else:
            return None
    return set(model_cls.model_fields.keys())


def _unwrap_annotation(annotation: Any) -> Any:
    """Unwrap ``Optional[X]`` / ``Union[X, None]`` to return ``X``."""
    import typing

    origin = getattr(annotation, "__origin__", None)
    if origin is type(None):
        return None
    if origin is typing.Union:
        args = [a for a in annotation.__args__ if a is not type(None)]
        return args[0] if len(args) == 1 else None
    return annotation


def _is_pydantic_model(cls: Any) -> bool:
    """Check whether *cls* is a Pydantic BaseModel subclass."""
    from pydantic import BaseModel

    try:
        return isinstance(cls, type) and issubclass(cls, BaseModel)
    except TypeError:
        return False


def suggest_typo_fix(
    unknown_key: str,
    valid_keys: Set[str],
    *,
    n: int = 3,
    cutoff: float = 0.6,
) -> List[str]:
    """Suggest closest valid key names for an unknown key.

    Uses ``difflib.get_close_matches`` to find similar strings.

    Args:
        unknown_key: The unknown key that was found in the YAML.
        valid_keys: Set of valid key names at the same level.
        n: Maximum number of suggestions to return.
        cutoff: Minimum similarity ratio (0-1) for a match.

    Returns:
        List of close matches sorted by similarity (best first).
    """
    return difflib.get_close_matches(unknown_key, sorted(valid_keys), n=n, cutoff=cutoff)


def validate_scenario_data(data: Dict[str, Any]) -> ScenarioModel:
    """Validate scenario data against the Pydantic ScenarioModel.

    Args:
        data: Raw scenario dictionary loaded from YAML.

    Raises:
        ValueError: If validation fails (unknown keys, invalid values, etc.).

    Returns:
        The immutable validated scenario model.
    """
    if not isinstance(data, dict):
        raise ValueError("Scenario configuration must be a mapping")
    _require_supported_schema_version(data)

    try:
        validate_registered_extensions(data)
        scene_data = data.get("scene", {}) or {}
        if isinstance(scene_data, dict):
            validate_registered_scene_source(scene_data)
        stripped_data = strip_registered_scene_source_sections(strip_registered_extensions(data))
        return ScenarioModel.model_validate(stripped_data)
    except ValidationError as e:
        details = []
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"]) or "<root>"
            msg = error["msg"]
            type_ = error["type"]

            if type_ == "extra_forbidden":
                unknown_key = str(error["loc"][-1])
                valid_keys = _get_valid_keys_for_loc(error["loc"])
                if len(error["loc"]) == 1:
                    valid_keys = (valid_keys or set()) | registered_extension_keys()
                if valid_keys:
                    suggestions = suggest_typo_fix(unknown_key, valid_keys)
                    if suggestions:
                        hint = ", ".join(f"'{s}'" for s in suggestions)
                        details.append(f"  - Unknown key '{loc}'. Did you mean: {hint}?")
                    else:
                        details.append(f"  - Unknown key '{loc}' (no close matches)")
                else:
                    details.append(f"  - Unknown key: '{loc}'")
            else:
                details.append(f"  - '{loc}': {msg}")

        summary = "\n".join(details)
        raise ValueError(f"Scenario configuration validation failed:\n{summary}") from e
