"""Preset service for managing MPC filter presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from shared.logging import get_logger

from ..state import MPC_ORDER_VALUES, MPC_TYPE_VALUES

logger = get_logger("orchav.preset_service")


class PresetService:
    """Manages material and filter presets for quick workflow switching."""

    RESET_PRESET_NAME = "All MPCs (Reset)"

    def __init__(self):
        """Initialize preset service with built-in and user presets."""
        self.preset_dir = Path.home() / ".orchav" / "presets"
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        self.built_in_presets = self._create_built_in_presets()
        logger.info(f"PresetService initialized: {len(self.built_in_presets)} built-in presets")

    def _create_built_in_presets(self) -> dict:
        """Create built-in filter presets for common use cases.

        MPC Type values:
            0: LoS (Line of Sight) - direct path
            1: Specular - mirror-like reflection
            2: Diffuse - rough surface scattering
            4: Refraction - through materials
            8: Diffraction - edge bending
            99: Virtual - reconstructed path

        Returns:
            Dict mapping preset names to preset configurations (ordered)
        """
        # Insertion order keeps Reset first in every preset menu.
        presets = {}

        presets[self.RESET_PRESET_NAME] = {
            "description": "Show all MPCs (no filtering) - use to reset filters",
            "mpc_allowed_orders": list(MPC_ORDER_VALUES),
            "mpc_allowed_types": list(MPC_TYPE_VALUES),
            "materials": {},  # All materials enabled
            # Range filters reset to None (no filtering)
            "delay_filter_min_ns": None,
            "delay_filter_max_ns": None,
            "power_filter_min_db": None,
            "power_filter_max_db": None,
            "aoa_az_filter_min_deg": None,
            "aoa_az_filter_max_deg": None,
            "aoa_el_filter_min_deg": None,
            "aoa_el_filter_max_deg": None,
            "aod_az_filter_min_deg": None,
            "aod_az_filter_max_deg": None,
            "aod_el_filter_min_deg": None,
            "aod_el_filter_max_deg": None,
        }

        presets["LoS Only"] = {
            "description": "Show only line-of-sight (direct) MPCs",
            "mpc_allowed_orders": [0],
            "mpc_allowed_types": [0],  # LoS type only
            "materials": {},
        }

        presets["Specular Only"] = {
            "description": "Show only specular (mirror-like) MPCs",
            "mpc_allowed_orders": list(range(1, 7)),  # 1-6 bounces (not LoS)
            "mpc_allowed_types": [1],  # Specular only
            "materials": {},
        }

        presets["Diffuse Only"] = {
            "description": "Show only diffuse (rough surface) MPCs",
            "mpc_allowed_orders": list(range(1, 7)),  # 1-6 bounces
            "mpc_allowed_types": [2],  # Diffuse only
            "materials": {},
        }

        presets["Single Bounce"] = {
            "description": "Show LoS and single-bounce MPCs only",
            "mpc_allowed_orders": [0, 1],
            "mpc_allowed_types": list(MPC_TYPE_VALUES),
            "materials": {},
        }

        presets["Low Order (≤2)"] = {
            "description": "Show MPCs with up to 2 reflections (typically stronger)",
            "mpc_allowed_orders": [0, 1, 2],
            "mpc_allowed_types": list(MPC_TYPE_VALUES),
            "materials": {},
        }

        presets["Multi-Bounce Only"] = {
            "description": "Show MPCs with 2+ reflections only (excludes LoS and single bounce)",
            "mpc_allowed_orders": [2, 3, 4, 5, 6],
            "mpc_allowed_types": list(MPC_TYPE_VALUES),
            "materials": {},
        }

        presets["No Diffraction"] = {
            "description": (
                "Hide diffraction MPCs (keep LoS, specular, diffuse, refraction, virtual)"
            ),
            "mpc_allowed_orders": list(MPC_ORDER_VALUES),
            "mpc_allowed_types": [value for value in MPC_TYPE_VALUES if value != 8],
            "materials": {},
        }

        return presets

    def save_preset(self, name: str, filter_state: dict) -> bool:
        """Save current filter state as a user preset.

        Args:
            name: Preset name (will be used as filename)
            filter_state: Dict containing filter configuration

        Returns:
            True if save succeeded, False otherwise
        """
        try:
            # Sanitize filename (remove invalid characters)
            safe_name = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_"))
            if not safe_name:
                logger.error(f"Invalid preset name: {name}")
                return False

            preset_path = self.preset_dir / f"{safe_name}.json"

            preset_data = {
                "name": name,
                "description": filter_state.get("description", "User-defined preset"),
                **filter_state,
            }

            preset_path.write_text(json.dumps(preset_data, indent=2))
            logger.info(f"Saved preset '{name}' to {preset_path}")
            return True

        except (OSError, TypeError, ValueError) as exc:
            logger.error(f"Failed to save preset '{name}': {exc}")
            return False

    def load_preset(self, name: str) -> Optional[dict]:
        """Load preset by name.

        Args:
            name: Preset name (built-in or user preset)

        Returns:
            Preset configuration dict, or None if not found
        """
        if name in self.built_in_presets:
            logger.info(f"Loading built-in preset: {name}")
            return self.built_in_presets[name]

        preset_path = self.preset_dir / f"{name}.json"
        if preset_path.exists():
            try:
                preset_data = json.loads(preset_path.read_text())
                logger.info(f"Loading user preset: {name}")
                return preset_data
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                logger.error(f"Failed to load preset '{name}': {exc}")
                return None

        logger.warning(f"Preset not found: {name}")
        return None

    def delete_preset(self, name: str) -> bool:
        """Delete a user preset.

        Args:
            name: Preset name (built-in presets cannot be deleted)

        Returns:
            True if deleted, False if not found or is built-in
        """
        # Cannot delete built-in presets
        if name in self.built_in_presets:
            logger.warning(f"Cannot delete built-in preset: {name}")
            return False

        preset_path = self.preset_dir / f"{name}.json"
        if preset_path.exists():
            try:
                preset_path.unlink()
                logger.info(f"Deleted preset: {name}")
                return True
            except OSError as exc:
                logger.error(f"Failed to delete preset '{name}': {exc}")
                return False

        logger.warning(f"Preset not found: {name}")
        return False

    def list_presets(self) -> list[str]:
        """List all available presets (built-in and user).

        Returns:
            List of preset names
        """
        presets = list(self.built_in_presets.keys())

        try:
            user_presets = [p.stem for p in self.preset_dir.glob("*.json")]
            presets.extend(user_presets)
        except OSError as exc:
            logger.error(f"Failed to list user presets: {exc}")

        return presets

    def is_built_in(self, name: str) -> bool:
        """Check if a preset is built-in.

        Args:
            name: Preset name

        Returns:
            True if preset is built-in
        """
        return name in self.built_in_presets

    def get_preset_description(self, name: str) -> Optional[str]:
        """Get description for a preset.

        Args:
            name: Preset name

        Returns:
            Preset description, or None if not found
        """
        preset = self.load_preset(name)
        if preset:
            return preset.get("description", "No description")
        return None
