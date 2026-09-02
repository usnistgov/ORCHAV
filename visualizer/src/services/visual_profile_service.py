"""Visual material profile service.

Resolves PBR preset assignments for objects based on matching rules.
Targets named ``human_walking`` can automatically render as ``Skin``
instead of their electromagnetic ITU material appearance.

Resolution order (highest priority first):

1. Per-material-type UI slider overrides (``MaterialPBRService.overrides``)
2. Scenario YAML ``visual_profiles`` rules
3. Built-in default rules (below)
4. ``ITU_TO_PBR`` fallback (unchanged)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..materials.appearance import VisualMaterialBinding, VisualMaterialSource
from ..materials.presets import BUILTIN_MATERIAL_PRESETS

if TYPE_CHECKING:
    from ...visualizer import OrchavVisualizer

logger = logging.getLogger(__name__)

# PBR property keys copied from a resolved preset.
_PBR_KEYS = tuple(
    dict.fromkeys(
        key
        for preset in BUILTIN_MATERIAL_PRESETS.values()
        for key in preset
        if key != "description"
    )
)


@dataclass
class _CompiledRule:
    """Internal compiled form of a visual-profile rule."""

    name_re: re.Pattern[str] | None
    itu: str | None
    entry_type: str | None  # "target" or "mesh"
    preset: str
    color_override: list[float] | None
    source: str  # "builtin" or "scenario"


# Built-in defaults — always active, lowest priority.
# Patterns use fullmatch (anchored), so "walker.*" matches "Walker_01" but
# not "SlowWalker".  Use ".*walker.*" to match substrings.
BUILTIN_RULES: list[dict[str, Any]] = [
    # Human / pedestrian targets
    {"match": {"name": r"human.*", "type": "target"}, "preset": "Skin"},
    {"match": {"name": r"body.*", "type": "target"}, "preset": "Skin"},
    {"match": {"name": r"person.*", "type": "target"}, "preset": "Skin"},
    {"match": {"name": r"pedestrian.*", "type": "target"}, "preset": "Skin"},
    {"match": {"name": r".*walker.*", "type": "target"}, "preset": "Skin"},
    {"match": {"name": r".*jogger.*", "type": "target"}, "preset": "Skin"},
    {"match": {"name": r".*runner.*", "type": "target"}, "preset": "Skin"},
    {"match": {"name": r".*marcher.*", "type": "target"}, "preset": "Skin"},
    # Vehicle targets
    {"match": {"name": r"vehicle.*", "type": "target"}, "preset": "Glossy"},
    {"match": {"name": r"car.*", "type": "target"}, "preset": "Glossy"},
]


def _compile_rule(rule: dict[str, Any], source: str) -> _CompiledRule | None:
    """Compile a rule dict into a ``_CompiledRule``, or *None* on error."""
    preset = rule.get("preset")
    if not preset or not isinstance(preset, str):
        logger.warning("visual_profile rule missing 'preset': %s", rule)
        return None

    match = rule.get("match", {})
    if not isinstance(match, dict):
        logger.warning("visual_profile rule 'match' is not a dict: %s", rule)
        return None

    name_pattern = match.get("name")
    name_re = None
    if name_pattern:
        try:
            name_re = re.compile(str(name_pattern), re.IGNORECASE)
        except re.error as exc:
            logger.warning("visual_profile rule bad regex %r: %s", name_pattern, exc)
            return None

    itu = match.get("itu")
    if itu is not None:
        itu = str(itu).lower()

    entry_type = match.get("type")
    if entry_type is not None:
        entry_type = str(entry_type).lower()
        if entry_type not in ("target", "mesh"):
            logger.warning("visual_profile rule bad type %r (expected target/mesh)", entry_type)
            return None

    color = rule.get("color")
    if color is not None:
        try:
            color = [float(c) for c in color[:3]]
        except (TypeError, ValueError):
            color = None

    return _CompiledRule(
        name_re=name_re,
        itu=itu,
        entry_type=entry_type,
        preset=preset,
        color_override=color,
        source=source,
    )


class VisualProfileService:
    """Resolve visual material presets for objects based on matching rules."""

    def __init__(self, visualizer: OrchavVisualizer) -> None:
        """Initialize compiled default and scenario visual-profile rules."""
        self.visualizer = visualizer
        self._scenario_rules: list[dict[str, Any]] = []
        self._compiled: list[_CompiledRule] = []
        self._rebuild_compiled()

    # Public API

    def load_scenario_rules(self, rules: list[dict[str, Any]]) -> None:
        """Load rules from a scenario YAML ``visual_profiles`` list."""
        self._scenario_rules = list(rules) if rules else []
        self._rebuild_compiled()
        logger.info(
            "Loaded %d scenario visual-profile rules (%d total with builtins)",
            len(self._scenario_rules),
            len(self._compiled),
        )

    def resolve(
        self,
        name: str,
        material_type: str,
        entry_type: str,
    ) -> dict[str, Any] | None:
        """Return PBR properties if a profile rule matches, else *None*.

        Args:
            name: Object/target name (e.g. ``"human_walking"``).
            material_type: ITU material type (e.g. ``"concrete"``).
            entry_type: ``"target"`` or ``"mesh"``.

        Returns:
            Dict with ``color``, ``roughness``, ``metallic``, ``reflectance``,
            ``alpha`` keys, or *None* if no rule matched.
        """
        matched = self._first_matching_profile(name, material_type, entry_type)
        if matched is None:
            return None
        rule, props = matched
        if rule.color_override is not None:
            props["color"] = list(rule.color_override)
        logger.debug(
            "visual_profile: %r matched rule (preset=%s, source=%s)",
            name,
            rule.preset,
            rule.source,
        )
        return props

    def resolve_binding(
        self,
        name: str,
        material_type: str,
        entry_type: str,
    ) -> VisualMaterialBinding | None:
        """Return the first matching profile as an EM-independent binding."""
        matched = self._first_matching_profile(name, material_type, entry_type)
        if matched is None:
            return None
        rule, _props = matched
        overrides: dict[str, Any] = {}
        if rule.color_override is not None:
            overrides["color"] = list(rule.color_override)
        return VisualMaterialBinding(
            source=VisualMaterialSource.PROFILE,
            material_type=material_type or "default",
            preset=rule.preset,
            overrides=overrides,
        )

    def get_active_rules(self) -> list[dict[str, Any]]:
        """Return the current scenario rules (for session persistence)."""
        return list(self._scenario_rules)

    def clear_scenario_rules(self) -> None:
        """Remove all scenario rules, keeping only builtins."""
        self._scenario_rules = []
        self._rebuild_compiled()

    # Internals

    def _first_matching_profile(
        self,
        name: str,
        material_type: str,
        entry_type: str,
    ) -> tuple[_CompiledRule, dict[str, Any]] | None:
        """Return the first matching rule whose preset can be resolved."""
        name_lower = (name or "").lower()
        mat_lower = (material_type or "").lower()
        etype_lower = (entry_type or "").lower()
        for rule in self._compiled:
            if not self._matches(rule, name_lower, mat_lower, etype_lower):
                continue
            props = self._preset_to_pbr(rule.preset)
            if props is not None:
                return rule, props
            logger.warning(
                "visual_profile: preset %r not found in PRESETS, skipping",
                rule.preset,
            )
        return None

    def _rebuild_compiled(self) -> None:
        """Recompile the rule list (scenario first, then builtins)."""
        compiled: list[_CompiledRule] = []
        for rule in self._scenario_rules:
            c = _compile_rule(rule, source="scenario")
            if c is not None:
                compiled.append(c)
        for rule in BUILTIN_RULES:
            c = _compile_rule(rule, source="builtin")
            if c is not None:
                compiled.append(c)
        self._compiled = compiled

    @staticmethod
    def _matches(
        rule: _CompiledRule,
        name_lower: str,
        mat_lower: str,
        etype_lower: str,
    ) -> bool:
        """Check if a compiled rule matches the given object attributes (AND logic)."""
        if rule.entry_type is not None and rule.entry_type != etype_lower:
            return False
        if rule.itu is not None and rule.itu != mat_lower:
            return False
        if rule.name_re is not None:
            if not rule.name_re.fullmatch(name_lower):
                return False
        return True

    @staticmethod
    def _preset_to_pbr(preset_name: str) -> dict[str, Any] | None:
        """Look up a PBR preset by name and return a copy of its properties."""
        preset = BUILTIN_MATERIAL_PRESETS.get(preset_name)
        if preset is None:
            return None
        return {k: preset[k] for k in _PBR_KEYS if k in preset}
