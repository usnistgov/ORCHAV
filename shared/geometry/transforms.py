"""Validated transform subset for ORCHAV's lightweight mesh loaders.

Sionna RT and Mitsuba evaluate the complete scene transform grammar.  ORCHAV's
lightweight Open3D and renderer-neutral loaders intentionally support a much
smaller mesh-shape subset, represented as uniform scale, canonical rotations,
and translation.  This module rejects forms that cannot be represented exactly
so previews, summary overlays, and fallback bounds never use approximate mesh
placement silently.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, NoReturn

SUPPORTED_LIGHTWEIGHT_TRANSFORM = (
    "optional uniform scale, then optional X/Y/Z rotations in that order, "
    "then optional translation"
)


class UnsupportedLightweightTransformError(Exception):
    """Signal a mesh transform outside the lightweight geometry contract.

    The direct ``Exception`` base keeps this correctness error outside optional
    geometry fallbacks that catch ``ValueError`` or ``RuntimeError``. An
    unsupported placement must reach the caller instead of degrading to
    missing or approximate geometry.
    """


def _shape_context(shape: ET.Element, source_xml: str | Path, shape_index: int) -> str:
    """Return stable source and mesh identity for a validation error."""

    source = Path(source_xml).expanduser().resolve(strict=False)
    shape_id = shape.get("id") or "<unnamed>"
    shape_type = shape.get("type") or "<unknown>"
    filename = shape.find("string[@name='filename']")
    mesh_path = filename.get("value") if filename is not None else None
    return (
        f"{source}: shape[{shape_index}] id={shape_id!r} "
        f"type={shape_type!r} mesh={mesh_path or '<unspecified>'!r}"
    )


def _unsupported(
    shape: ET.Element,
    source_xml: str | Path,
    shape_index: int,
    operation_index: int,
    operation: str,
    reason: str,
) -> NoReturn:
    """Raise one contextual unsupported-transform error."""

    context = _shape_context(shape, source_xml, shape_index)
    raise UnsupportedLightweightTransformError(
        f"Unsupported lightweight mesh transform in {context}; "
        f"operation[{operation_index}] <{operation}> {reason}. "
        f"Supported form: {SUPPORTED_LIGHTWEIGHT_TRANSFORM}."
    )


def _finite_float(
    value: str | None,
    *,
    shape: ET.Element,
    source_xml: str | Path,
    shape_index: int,
    operation_index: int,
    operation: str,
    field: str,
) -> float:
    """Parse one required finite numeric transform value."""

    try:
        parsed = float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        parsed = float("nan")
    if not math.isfinite(parsed):
        _unsupported(
            shape,
            source_xml,
            shape_index,
            operation_index,
            operation,
            f"requires a finite numeric {field}, got {value!r}",
        )
    return parsed


def _vector_values(value: str | None) -> list[str]:
    """Split a Mitsuba vector written with whitespace and/or commas."""

    return [part for part in re.split(r"[\s,]+", (value or "").strip()) if part]


def parse_lightweight_shape_transform(
    shape: ET.Element,
    *,
    source_xml: str | Path,
    shape_index: int,
) -> dict[str, Any]:
    """Return an exact normalized transform for one lightweight mesh shape.

    Only direct ``ply`` and ``obj`` shape callers use this function. Sensor,
    emitter, texture, and other Mitsuba transforms remain the responsibility of
    Mitsuba and are outside this lightweight geometry contract.
    """

    state: dict[str, Any] = {
        "scale": 1.0,
        "rotation": [0.0, 0.0, 0.0],
        "translate": [0.0, 0.0, 0.0],
    }
    transforms = shape.findall("transform[@name='to_world']")
    if not transforms:
        return state
    if len(transforms) > 1:
        _unsupported(
            shape,
            source_xml,
            shape_index,
            1,
            "transform",
            "repeats the to_world transform",
        )
    transform = transforms[0]

    seen: set[str] = set()
    previous_rank = -1
    # Normalized rotation state is [yaw around Z, pitch around Y, roll around X].
    rotation_index = {"x": 2, "y": 1, "z": 0}

    for operation_index, child in enumerate(transform):
        operation = str(child.tag).rsplit("}", 1)[-1]
        if operation == "scale":
            key = "scale"
            rank = 0
            unexpected = set(child.attrib) - {"value"}
            if unexpected or "value" not in child.attrib:
                _unsupported(
                    shape,
                    source_xml,
                    shape_index,
                    operation_index,
                    operation,
                    "must use exactly one scalar value attribute",
                )
            scale = _finite_float(
                child.get("value"),
                shape=shape,
                source_xml=source_xml,
                shape_index=shape_index,
                operation_index=operation_index,
                operation=operation,
                field="scale",
            )
            if scale <= 0.0:
                _unsupported(
                    shape,
                    source_xml,
                    shape_index,
                    operation_index,
                    operation,
                    f"requires a positive uniform scale, got {scale!r}",
                )
            state["scale"] = scale
        elif operation == "rotate":
            unexpected = set(child.attrib) - {"x", "y", "z", "angle"}
            if unexpected:
                _unsupported(
                    shape,
                    source_xml,
                    shape_index,
                    operation_index,
                    operation,
                    f"has unsupported attributes {sorted(unexpected)!r}",
                )
            axis_values = {
                axis: _finite_float(
                    child.get(axis, "0"),
                    shape=shape,
                    source_xml=source_xml,
                    shape_index=shape_index,
                    operation_index=operation_index,
                    operation=operation,
                    field=f"{axis}-axis component",
                )
                for axis in ("x", "y", "z")
            }
            active_axes = [axis for axis, value in axis_values.items() if value == 1.0]
            if len(active_axes) != 1 or any(
                value not in (0.0, 1.0) for value in axis_values.values()
            ):
                _unsupported(
                    shape,
                    source_xml,
                    shape_index,
                    operation_index,
                    operation,
                    "must use exactly one positive canonical axis",
                )
            axis = active_axes[0]
            key = f"rotate-{axis}"
            rank = {"x": 1, "y": 2, "z": 3}[axis]
            angle = _finite_float(
                child.get("angle"),
                shape=shape,
                source_xml=source_xml,
                shape_index=shape_index,
                operation_index=operation_index,
                operation=operation,
                field="angle",
            )
            state["rotation"][rotation_index[axis]] = angle
        elif operation == "translate":
            key = "translate"
            rank = 4
            unexpected = set(child.attrib) - {"x", "y", "z", "value"}
            has_vector = "value" in child.attrib
            has_axes = any(axis in child.attrib for axis in ("x", "y", "z"))
            if unexpected or has_vector == has_axes:
                _unsupported(
                    shape,
                    source_xml,
                    shape_index,
                    operation_index,
                    operation,
                    "must use either a three-value vector or X/Y/Z attributes",
                )
            if has_vector:
                parts = _vector_values(child.get("value"))
                if len(parts) != 3:
                    _unsupported(
                        shape,
                        source_xml,
                        shape_index,
                        operation_index,
                        operation,
                        f"requires three vector components, got {child.get('value')!r}",
                    )
                values = parts
            else:
                values = [child.get(axis, "0") for axis in ("x", "y", "z")]
            state["translate"] = [
                _finite_float(
                    value,
                    shape=shape,
                    source_xml=source_xml,
                    shape_index=shape_index,
                    operation_index=operation_index,
                    operation=operation,
                    field=f"translation {axis}",
                )
                for axis, value in zip(("x", "y", "z"), values, strict=True)
            ]
        else:
            _unsupported(
                shape,
                source_xml,
                shape_index,
                operation_index,
                operation,
                "is not implemented",
            )

        if key in seen:
            _unsupported(
                shape,
                source_xml,
                shape_index,
                operation_index,
                operation,
                "is repeated",
            )
        if rank <= previous_rank:
            _unsupported(
                shape,
                source_xml,
                shape_index,
                operation_index,
                operation,
                "is out of the supported order",
            )
        seen.add(key)
        previous_rank = rank

    return state
