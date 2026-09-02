"""Normalize source material labels for canonical visual metadata."""

from __future__ import annotations


def _normalize_material_value(value: object) -> str:
    """Normalize raw material metadata values into stable strings."""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    if value is None:
        return ""
    text = str(value).strip()
    if text == "None":
        return ""
    return text
