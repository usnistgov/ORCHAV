"""Scene material-ID and XML material editing service."""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.logging import get_logger

from ..materials.catalog import normalize_material_type_name
from ..scene.io import MaterialHandler
from .base import BaseService
from .material_properties import sync_entry_pbr_properties_from_catalog

logger = get_logger("orchav.material_entry_editor")


class MaterialEntryEditService(BaseService):
    """Change scene-entry material IDs, colors, XML refs, and catalog PBR defaults."""

    def update_material_color(
        self,
        entry: dict[str, Any],
        new_color: list[float],
        mesh_entries: list[dict[str, Any]],
        target_entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Update material color in XML and all entries with the same material ID."""
        material_id = entry.get("material_id", "")
        updated_entries: list[dict[str, Any]] = []

        def _update_entry(target_entry: dict[str, Any]) -> None:
            target_entry["color"] = new_color
            updated_entries.append(target_entry)

        _update_entry(entry)

        for other_entry in mesh_entries:
            if other_entry is not entry and other_entry.get("material_id") == material_id:
                _update_entry(other_entry)
        for target_entry in target_entries:
            if target_entry is not entry and target_entry.get("material_id") == material_id:
                _update_entry(target_entry)

        for updated_entry in updated_entries:
            xml_bsdf = updated_entry.get("xml_bsdf")
            if xml_bsdf is not None:
                try:
                    MaterialHandler.update_material_color(xml_bsdf, new_color)
                except (ValueError, AttributeError) as exc:
                    logger.error("Failed to update XML for material %s: %s", material_id, exc)

        logger.info(
            "Updated color for material '%s': %d objects affected",
            material_id,
            len(updated_entries),
        )
        return updated_entries

    def change_material_id(
        self,
        entry: dict[str, Any],
        new_id: str,
        xml_root: Any,
        mesh_entries: list[dict[str, Any]],
        target_entries: list[dict[str, Any]],
        mpc_core: Any | None = None,
    ) -> tuple[str, list[float], Any | None]:
        """Change a scene entry's material ID and return ID, color, and BSDF."""
        old_id = entry.get("material_id", "")
        if not new_id:
            return old_id, entry.get("color", [0.7, 0.7, 0.7]), entry.get("xml_bsdf")

        logger.debug("Changing material ID: '%s' -> '%s'", old_id, new_id)

        new_bsdf = None
        new_color = [0.7, 0.7, 0.7]
        matching_entries = [
            candidate
            for candidate in mesh_entries + target_entries
            if candidate is not entry and candidate.get("material_id") == new_id
        ]
        mpc_color = self._color_from_mpc_core(mpc_core, new_id)

        if xml_root is not None:
            new_bsdf = self._find_bsdf(xml_root, new_id)
            if new_bsdf is not None:
                new_color = self._color_from_bsdf(
                    new_bsdf,
                    new_id,
                    mesh_entries,
                    target_entries,
                    mpc_core,
                    default_color=new_color,
                )
            else:
                logger.warning("Could not find BSDF with ID '%s' in XML root", new_id)
                available_ids = [bsdf.get("id", "N/A") for bsdf in xml_root.findall("bsdf")]
                logger.warning("   Available BSDF IDs in XML: %s", available_ids)
                new_color, new_bsdf = self._fallback_material_from_entries(
                    new_id,
                    entry,
                    mesh_entries,
                    target_entries,
                    default_color=new_color,
                )
                if new_bsdf is None and mpc_color is not None:
                    new_color = mpc_color
        elif matching_entries:
            new_color, new_bsdf = self._fallback_material_from_entries(
                new_id,
                entry,
                mesh_entries,
                target_entries,
                default_color=new_color,
            )
        elif mpc_color is not None:
            new_color = mpc_color

        if new_bsdf is None and not matching_entries and mpc_color is None:
            logger.warning(
                "Rejected unknown material ID '%s'; entry '%s' remains unchanged",
                new_id,
                entry.get("name", "Unknown"),
            )
            return old_id, entry.get("color", [0.7, 0.7, 0.7]), entry.get("xml_bsdf")

        actual_material_id = new_id
        if new_bsdf is not None:
            actual_material_id = new_bsdf.get("id", new_id)
            if actual_material_id != new_id:
                logger.warning(
                    "Using actual BSDF ID '%s' instead of requested '%s'",
                    actual_material_id,
                    new_id,
                )

        self._update_entry_lists(
            entry,
            actual_material_id,
            new_color,
            new_bsdf,
            mesh_entries,
            target_entries,
        )
        self._sync_material_type(entry, actual_material_id, mesh_entries, target_entries)
        self._update_shape_reference(entry, new_id, actual_material_id, new_bsdf)

        logger.info(
            "Material ID changed from %s to %s for %s (color: %s)",
            old_id,
            actual_material_id,
            entry.get("name", "Unknown"),
            new_color,
        )

        return actual_material_id, new_color, new_bsdf

    def _find_bsdf(self, xml_root: Any, new_id: str) -> Any | None:
        """Find an XML BSDF by exact, normalized, or case-insensitive ID."""
        for bsdf in xml_root.findall("bsdf"):
            bsdf_id = bsdf.get("id", "")
            if bsdf_id == new_id:
                logger.debug("Found BSDF with exact ID match: '%s'", bsdf_id)
                return bsdf

        logger.warning("Exact match not found, trying normalized matching...")
        normalized_new_id = new_id.replace("-", "_").replace("_", "-")
        for bsdf in xml_root.findall("bsdf"):
            bsdf_id = bsdf.get("id", "")
            if (
                bsdf_id == normalized_new_id
                or bsdf_id.replace("-", "_") == new_id.replace("-", "_")
                or bsdf_id.lower() == new_id.lower()
            ):
                logger.debug(
                    "Found BSDF with normalized match: '%s' (searched for '%s')",
                    bsdf_id,
                    new_id,
                )
                return bsdf
        return None

    def _color_from_bsdf(
        self,
        bsdf: Any,
        material_id: str,
        mesh_entries: list[dict[str, Any]],
        target_entries: list[dict[str, Any]],
        mpc_core: Any | None,
        *,
        default_color: list[float],
    ) -> list[float]:
        """Resolve display color from a BSDF, MPC colors, or matching entries."""
        rgb_el = bsdf.find(".//rgb[@name='reflectance']")
        if rgb_el is None:
            rgb_el = bsdf.find(".//rgb[@name='color']")
        if rgb_el is not None:
            color_str = rgb_el.get("value", "")
            if color_str:
                try:
                    color = [float(value) for value in color_str.split()]
                    logger.debug("Found RGB color: %s", color)
                    return color
                except (ValueError, AttributeError) as exc:
                    logger.warning("Failed to parse RGB color: %s", exc)

        bsdf_type = bsdf.get("type", "")
        logger.warning("No RGB found in BSDF type '%s'", bsdf_type)
        if bsdf_type == "itu-radio-material":
            mpc_color = self._color_from_mpc_core(mpc_core, material_id)
            if mpc_color is not None:
                return mpc_color

        for other_entry in mesh_entries + target_entries:
            if other_entry.get("material_id") == material_id and "color" in other_entry:
                logger.debug("Found color from other entry: %s", other_entry["color"])
                return other_entry["color"]
        return default_color

    def _fallback_material_from_entries(
        self,
        new_id: str,
        entry: dict[str, Any],
        mesh_entries: list[dict[str, Any]],
        target_entries: list[dict[str, Any]],
        *,
        default_color: list[float],
    ) -> tuple[list[float], Any | None]:
        """Find color and BSDF reference from another entry with the requested material."""
        logger.debug("Material '%s' not in XML - trying to find it in other entries", new_id)
        new_color = default_color
        new_bsdf = None
        for other_entry in mesh_entries + target_entries:
            if other_entry is not entry and other_entry.get("material_id") == new_id:
                if "color" in other_entry:
                    new_color = other_entry["color"]
                    logger.debug("Found color from other entry: %s", new_color)
                if other_entry.get("xml_bsdf") is not None:
                    new_bsdf = other_entry["xml_bsdf"]
                    logger.debug("Found BSDF reference from other entry")
                if new_color != default_color and new_bsdf is not None:
                    break
        return new_color, new_bsdf

    @staticmethod
    def _color_from_mpc_core(mpc_core: Any | None, material_id: str) -> list[float] | None:
        """Resolve a material color from MPCCore material color metadata."""
        if mpc_core is None:
            return None
        material_colors = mpc_core._get_material_colors()
        if material_colors and material_id in material_colors:
            color_array = material_colors[material_id]
            if isinstance(color_array, np.ndarray):
                return color_array.tolist()
            return list(color_array)

        normalized_id = material_id.replace("-", "_")
        for mat_id, color_array in (material_colors or {}).items():
            if mat_id.replace("-", "_") == normalized_id:
                return (
                    color_array.tolist()
                    if isinstance(color_array, np.ndarray)
                    else list(color_array)
                )
        return None

    def _update_entry_lists(
        self,
        entry: dict[str, Any],
        actual_material_id: str,
        new_color: list[float],
        new_bsdf: Any | None,
        mesh_entries: list[dict[str, Any]],
        target_entries: list[dict[str, Any]],
    ) -> None:
        """Update the edited entry and its authoritative list entry."""
        entry_name = entry.get("name", "")
        entry_type = entry.get("entry_type", "mesh")
        candidates = mesh_entries if entry_type == "mesh" else target_entries
        updated_in_list = False

        for candidate in candidates:
            if candidate is not None and candidate.get("name") == entry_name:
                self._set_entry_material(candidate, actual_material_id, new_color, new_bsdf)
                updated_in_list = True
                logger.debug(
                    "Updated entry in %s_entries: '%s' -> material '%s'",
                    entry_type,
                    entry_name,
                    actual_material_id,
                )
                break

        self._set_entry_material(entry, actual_material_id, new_color, new_bsdf)

        if not updated_in_list:
            logger.warning("Could not find entry '%s' in lists to update", entry_name)

    @staticmethod
    def _set_entry_material(
        entry: dict[str, Any],
        material_id: str,
        color: list[float],
        bsdf: Any | None,
    ) -> None:
        """Set material ID, color, and BSDF reference on one entry dict."""
        entry["material_id"] = material_id
        entry["color"] = color
        entry["xml_bsdf"] = bsdf if bsdf is not None else None

    def _sync_material_type(
        self,
        entry: dict[str, Any],
        actual_material_id: str,
        mesh_entries: list[dict[str, Any]],
        target_entries: list[dict[str, Any]],
    ) -> None:
        """Sync material_type and catalog-backed PBR defaults after an ID change."""
        new_material_type = normalize_material_type_name(actual_material_id, default="")
        if not new_material_type:
            return

        entry_name = entry.get("name", "")
        entry_type = entry.get("entry_type", "mesh")
        entry["material_type"] = new_material_type
        sync_entry_pbr_properties_from_catalog(entry, new_material_type)
        candidates = mesh_entries if entry_type == "mesh" else target_entries
        for candidate in candidates:
            if candidate is not None and candidate.get("name") == entry_name:
                candidate["material_type"] = new_material_type
                sync_entry_pbr_properties_from_catalog(candidate, new_material_type)
                break

    @staticmethod
    def _update_shape_reference(
        entry: dict[str, Any],
        requested_id: str,
        actual_material_id: str,
        new_bsdf: Any | None,
    ) -> None:
        """Update the XML shape BSDF reference when the target BSDF exists."""
        xml_shape = entry.get("xml_shape")
        if xml_shape is not None and new_bsdf is not None:
            ref = xml_shape.find("ref[@name='bsdf']")
            if ref is not None:
                ref.set("id", actual_material_id)
        elif xml_shape is not None and new_bsdf is None:
            logger.warning(
                "Material '%s' not in XML - shape reference not updated. "
                "Visual appearance will change but XML save may fail.",
                requested_id,
            )
